# Copyright 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AgentLoop-backed rollout dispatcher for async-RL.

This dispatcher owns the train prompt DataLoader and submits batches to a
TransferQueue-backed ``AgentLoopManager``. It is deliberately separate from the
rollout engine control plane so the Uni-Agent path can share async-RL trainer,
relay, router, tracker, and ExperienceBuffer infrastructure without preserving
the old direct prompt-to-rollout scheduling implementation.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.rl_dataset import collate_fn

from laminar.global_request_router.router import RouteUnavailableError
from laminar.global_request_router.runtime import kv_below_threshold
from laminar.metrics import MetricSnapshot, MetricSource, cumulative_counter_specs
from laminar.train.relay import EMPTY_VERSION, RelayClient
from laminar.utils.concurrency import log_exception_callback

logger = logging.getLogger(__name__)
# TODO: tune these budgets from measured startup and planned training durations.
_VERSION_WAIT_TIMEOUT_S = 600.0
_PLANNED_PAUSE_WAIT_TIMEOUT_S = 3600.0
_VERSION_POLL_INTERVAL_S = 0.1
_DISPATCHER_CHECKPOINT_SCHEMA_VERSION = 2
_DISPATCHER_COUNTER_KEYS = (
    "dispatcher/prompt_batches_dispatched",
    "dispatcher/prompts_dispatched",
    "dispatcher/version_wait_duration_s",
    "dispatcher/llm_server_pressure_skip_count",
)


@dataclass(frozen=True)
class RolloutDispatcherConfig:
    """Typed config surface for the AgentLoop rollout dispatcher."""

    schedule_interval_s: float
    state_batch_size: int
    # Dispatch fan-out per prompt; same knob as actor_rollout_ref.rollout.n.
    num_samples_per_prompt: int
    # Same knobs as ``rollout_router.dynamic_overload_admission``. ``enforce``
    # pauses new batches when every replica's KV is at/above the threshold.
    pressure_admission_mode: str = "off"
    kv_cache_usage_threshold: float = 0.80


class RolloutDispatcher:
    """Dispatch train prompt batches through an AgentLoopManager.

    ``data_config`` is the Hydra ``data`` subtree, not the full recipe:
    ``RLHFDataset`` / sampler already consume that section. It is passed into
    this Ray actor so the colocated StatefulDataLoader is built in-process,
    keeping workers, sampler state, and checkpointed ``prompt_index`` here.

    ``tokenizer`` is the driver-built ``hf_tokenizer`` object. It pickles
    cheaply, so the actor can reuse it instead of a second ``copy_to_local`` /
    load, and instead of ``HFModelConfig`` (rollout servers must initialize
    that on their own nodes).
    """

    def __init__(
        self,
        config: RolloutDispatcherConfig,
        data_config: Any,
        tokenizer: Any,
        agent_loop_manager: Any,
        relay_client: RelayClient,
        rollout_router: Any,
        llm_server_state_tracker: Any | None = None,
    ):
        if data_config is None:
            raise ValueError("RolloutDispatcher requires data_config.")
        if tokenizer is None:
            raise ValueError("RolloutDispatcher requires tokenizer.")

        self.config = config

        # Train half of mainline PPO dataloader construction. Chat-template
        # tokenization stays in AgentLoop; the loader only yields collated
        # raw_prompt rows.
        dataset = create_rl_dataset(
            data_config.train_files,
            data_config,
            tokenizer,
            None,
            is_train=True,
            max_samples=int(data_config.train_max_samples),
        )
        self.prompt_dataloader = StatefulDataLoader(
            dataset=dataset,
            batch_size=int(config.state_batch_size),
            num_workers=int(data_config.dataloader_num_workers),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=create_rl_sampler(data_config, dataset),
        )
        assert len(self.prompt_dataloader) >= 1, "Train prompt dataloader is empty!"
        # Bind workers at actor construct so the first dispatch is not a cold iter().
        self._prompt_iterator = iter(self.prompt_dataloader)
        self._prompt_index = 0

        # Lifecycle
        self._run_task: asyncio.Task | None = None

        # Dispatch
        self.agent_loop_manager = agent_loop_manager
        self.relay_client = relay_client
        self.rollout_router = rollout_router
        self.llm_server_state_tracker = llm_server_state_tracker

        # Metrics
        self._metrics: dict[str, int | float] = {
            "dispatcher/prompt_batches_dispatched": 0,
            "dispatcher/prompts_dispatched": 0,
            "dispatcher/latest_submitted_prompt_count": 0,
            "dispatcher/latest_submitted_trajectory_count": 0,
            "dispatcher/latest_target_weight_version": -1,
            "dispatcher/trainer_version": -1,
            "dispatcher/version_wait_duration_s": 0.0,
            "dispatcher/llm_server_pressure_skip_count": 0,
        }
        self._metric_source = MetricSource(cumulative_counter_specs("rollout", _DISPATCHER_COUNTER_KEYS))

    # --- Lifecycle ---

    async def start(self) -> None:
        if self._run_task is not None and not self._run_task.done():
            return
        self._run_task = asyncio.create_task(self._loop(), name="rollout_dispatcher.loop")
        self._run_task.add_done_callback(log_exception_callback)

    async def wait_for_stop(self) -> None:
        """Wait for the dispatch loop and surface its first fatal error."""
        await self._run_task

    # --- Dispatch ---

    async def _loop(self) -> None:
        while True:
            if await self._llm_server_pressure_open():
                await self._dispatch_next_batch()
            else:
                self._metrics["dispatcher/llm_server_pressure_skip_count"] += 1
                logger.info("No dispatching to AgentLoopManager: LLM server replicas are at KV pressure")
            await asyncio.sleep(self.config.schedule_interval_s)

    async def _llm_server_pressure_open(self) -> bool:
        """Pause new batches when every replica is at/above the KV threshold.

        Snapshot lag is allowed to slightly overshoot; this is not a precise
        in-flight session cap.
        """
        mode = str(getattr(self.config, "pressure_admission_mode", "off")).lower()
        if mode != "enforce":
            return True
        tracker = getattr(self, "llm_server_state_tracker", None)
        if tracker is None:
            return True
        threshold = float(getattr(self.config, "kv_cache_usage_threshold", 0.80))
        check = getattr(tracker, "get_routing_snapshot", None)
        if not callable(check):
            raise RuntimeError("LLMServerStateTracker is missing get_routing_snapshot")
        remote_check = getattr(check, "remote", None)
        snapshot = remote_check() if callable(remote_check) else check()
        if inspect.isawaitable(snapshot):
            snapshot = await snapshot
        if not isinstance(snapshot, Mapping):
            raise TypeError("LLMServerStateTracker routing snapshot must be a mapping")
        if not snapshot:
            return True
        # Cluster admission belongs here; Router admission uses the same
        # per-replica KV predicate only as a sticky-placement preference.
        return any(kv_below_threshold(state.get("kv_cache_usage"), threshold) for state in snapshot.values())

    async def _dispatch_next_batch(self) -> None:
        batch = self._next_prompt_batch()

        # Resolve a routable version before the framework creates pinned sessions.
        wait_started_at = asyncio.get_running_loop().time()
        trainer_version, target_weight_version = await self._wait_for_available_version()
        version_wait_duration_s = asyncio.get_running_loop().time() - wait_started_at
        if type(target_weight_version) is not int or target_weight_version < trainer_version:
            raise RuntimeError(
                "Router returned an invalid available weight version: "
                f"trainer_version={trainer_version}, available={target_weight_version!r}"
            )
        tu.assign_non_tensor_data(batch, "global_steps", target_weight_version)

        # The training adapter waits for session admission, while rollout results
        # continue asynchronously through TQ. Do not queue another batch before it returns.
        await asyncio.to_thread(self.agent_loop_manager.submit_sessions, batch)

        # Metrics
        self._metrics["dispatcher/prompt_batches_dispatched"] += 1
        self._metrics["dispatcher/prompts_dispatched"] += len(batch)
        self._metrics["dispatcher/latest_target_weight_version"] = target_weight_version
        self._metrics["dispatcher/latest_submitted_prompt_count"] = len(batch)
        self._metrics["dispatcher/latest_submitted_trajectory_count"] = len(batch) * int(
            self.config.num_samples_per_prompt
        )
        self._metrics["dispatcher/trainer_version"] = trainer_version
        self._metrics["dispatcher/version_wait_duration_s"] += version_wait_duration_s

        # Logs
        logger.info(
            "Dispatching 1 AgentLoop batch / %s prompts at target_weight_version=%s trainer_version=%s",
            len(batch),
            target_weight_version,
            trainer_version,
        )

    async def _wait_for_available_version(self) -> tuple[int, int]:
        """Keep one pre-admission wait in this dispatcher, outside Router."""
        started = asyncio.get_running_loop().time()
        async with asyncio.timeout_at(started + _VERSION_WAIT_TIMEOUT_S) as budget:
            trainer_version = await self._wait_for_trainer_version()
            planned_pause = False
            while True:
                try:
                    version = await self.rollout_router.get_routable_version.remote(weight_version=trainer_version)
                    return trainer_version, version
                except RouteUnavailableError as error:
                    if getattr(error, "planned_pause", False) and not planned_pause:
                        planned_pause = True
                        budget.reschedule(started + _PLANNED_PAUSE_WAIT_TIMEOUT_S)
                    await asyncio.sleep(_VERSION_POLL_INTERVAL_S)

    def _next_prompt_batch(self) -> TensorDict:
        try:
            batch_dict = next(self._prompt_iterator)
        except StopIteration:
            self._prompt_iterator = iter(self.prompt_dataloader)
            batch_dict = next(self._prompt_iterator)

        # Mainline collate_fn yields a dict of stacked tensors / object arrays.
        # Convert here, matching v1 `_fetch_one_gen_batch`, then overlay a fresh
        # uid so store keys `{uid}_{session}_{index}` do not collide across epochs.
        batch_dict = dict(batch_dict)
        if "raw_prompt" not in batch_dict:
            raise ValueError("AgentLoop rollout requires raw_prompt in prompt batches")
        prompt_count = len(batch_dict["raw_prompt"])
        prompts_per_epoch = len(self.prompt_dataloader) * int(self.config.state_batch_size)
        epoch_index = self._prompt_index // prompts_per_epoch
        batch_dict["uid"] = [f"{epoch_index}-{uuid.uuid4().hex}" for _ in range(prompt_count)]
        self._prompt_index += prompt_count
        return tu.get_tensordict(batch_dict)

    async def _wait_for_trainer_version(self) -> int:
        """Wait until trainer has published a target weight version."""
        while True:
            trainer_version = int(await self.relay_client.get_trainer_version())
            if trainer_version > EMPTY_VERSION:
                return trainer_version
            logger.info(
                "No dispatching to AgentLoopManager: trainer has no target version for trainer_version=%s",
                trainer_version,
            )
            await asyncio.sleep(self.config.schedule_interval_s)

    # --- Metrics ---

    async def get_metrics(self) -> MetricSnapshot:
        metrics = dict(self._metrics)
        get_efficiency = getattr(self.agent_loop_manager, "get_efficiency_metrics", None)
        if get_efficiency is not None:
            metrics.update({f"efficiency/{key}": value for key, value in (await get_efficiency()).items()})
        return self._metric_source.snapshot(metrics)

    # --- Checkpoint ---

    async def save_checkpoint(self, path: str) -> None:
        """Persist the dispatcher-owned prompt cursor without stopping dispatch."""
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "schema_version": _DISPATCHER_CHECKPOINT_SCHEMA_VERSION,
            "prompt_dataloader": self.prompt_dataloader.state_dict(),
            "prompt_index": int(self._prompt_index),
            "state_batch_size": int(self.config.state_batch_size),
            "dataloader_length": len(self.prompt_dataloader),
        }
        await asyncio.to_thread(torch.save, state, checkpoint_path)

    def load_checkpoint(self, path: str) -> None:
        """Restore the prompt cursor before the dispatcher loop starts."""
        if self._run_task is not None:
            raise RuntimeError("cannot restore RolloutDispatcher after start")

        # Load from path
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise RuntimeError(f"missing rollout dispatcher checkpoint: {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if not isinstance(state, dict):
            raise RuntimeError(f"invalid rollout dispatcher checkpoint payload: {checkpoint_path}")
        if state.get("schema_version") != _DISPATCHER_CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError(f"unsupported rollout dispatcher checkpoint schema: {state.get('schema_version')!r}")

        # Validation
        expected = {
            "state_batch_size": int(self.config.state_batch_size),
            "dataloader_length": len(self.prompt_dataloader),
        }
        actual = {key: state.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(
                f"rollout dispatcher checkpoint is incompatible: expected={expected!r} actual={actual!r}"
            )

        required_state = {"prompt_dataloader", "prompt_index"}
        missing = required_state - set(state)
        if missing:
            raise RuntimeError(f"rollout dispatcher checkpoint is missing state: {sorted(missing)}")

        # Load into runtime state
        self.prompt_dataloader.load_state_dict(state["prompt_dataloader"])
        # Re-bind after restore so the iterator sees the restored sampler, not
        # the pre-checkpoint one from __init__.
        self._prompt_iterator = iter(self.prompt_dataloader)
        self._prompt_index = int(state["prompt_index"])
