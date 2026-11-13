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
"""``AsyncRLTrainer`` — driver-side orchestrator for async-RL training.

Responsibilities:

* Initialize the FSDP or Megatron actor worker group and connect its publisher
  ranks to the Relay chain head.
* In ``fit()``, loop over training steps:
    1. Sample a :class:`SampledBatch` from :class:`ExperienceBuffer`.
    2. Materialize its trajectories and compute advantages from the immutable
       prompt-group context returned by the buffer.
    3. Compute read-only old log-probs while the previous Relay publish can
       finish in parallel; Megatron first fences export collectives.
    4. Fence the previous publish before ``actor_wg.update_actor(td)`` mutates
       actor weights and takes a PPO step.
    5. Periodically ``publish_actor_weights_to_relay(version=global_steps)`` to
       publish new weights through the relay servers.
    6. Per-step metrics route through the centralized :class:`Tracker` actor.

Critic and reference-policy branches are TODOs; the rest of the orchestration
is self-contained on the driver event loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, cast

import numpy as np
import ray
import torch
from omegaconf import DictConfig
from tensordict import NonTensorData, TensorDict
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.rollout_corr_helper import (
    compute_rollout_correction_and_rejection_mask,
)
from verl.trainer.ppo.utils import Role
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import (
    find_latest_ckpt_path,
    get_checkpoint_tracker_filename,
)
from verl.utils.metric import reduce_metrics
from verl.utils.profiler.performance import simple_timer
from verl.utils.py_functional import rename_dict
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

from laminar.metrics import Aggregation, MetricSpec, export_flat_metrics, reduce_metric_snapshots
from laminar.train.batch_metrics import sampled_metadata_metrics, tensor_stats, train_batch_metrics
from laminar.train.experience_buffer.client import ExperienceBufferClient, materialize_batch
from laminar.train.experience_buffer.server import SampledBatch
from laminar.train.experience_buffer.tags import trajectory_key_facts
from laminar.train.relay import RelayClient

if TYPE_CHECKING:
    from ray.actor import ActorHandle
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager

    from laminar.utils.tensor import TensorMeta

logger = logging.getLogger(__name__)


def _select_async_actor_worker_cls(actor_strategy: str) -> type:
    strategy = str(actor_strategy).lower()
    if strategy in {"fsdp", "fsdp2"}:
        from laminar.workers.fsdp_workers import AsyncActorWorker

        return AsyncActorWorker
    if strategy == "megatron":
        from laminar.workers.megatron_workers import AsyncMegatronActorWorker

        return AsyncMegatronActorWorker
    raise ValueError(f"async_rl actor strategy {actor_strategy!r} is not supported; expected fsdp, fsdp2, or megatron")


def _append_metric_values(metrics: dict[str, list[Any]], key: str, value: Any) -> None:
    values = value if isinstance(value, list | tuple) else [value]
    metrics.setdefault(key, []).extend(values)


def _extract_actor_update_metrics(output: Any) -> dict[str, Any]:
    """Normalize actor worker-group outputs into reduced, prefixed metrics."""
    if output is None:
        return {}
    raw_metrics = tu.get(output, "metrics", default=None)
    if not raw_metrics:
        return {}
    update_keys = (
        "update/optimizer_updates_attempted",
        "update/optimizer_updates_completed",
        "update/optimizer_updates_skipped",
    )
    validity_key = "update/optimizer_updates_valid"
    specs = {
        key: MetricSpec(key, aggregation=Aggregation.MAX, owner="actor", population="optimizer_updates", window="step")
        for key in update_keys
    }
    specs[validity_key] = MetricSpec(validity_key, aggregation=Aggregation.MIN, owner="actor", window="step")
    reduced = reduce_metric_snapshots(specs, [raw_metrics], legacy_reducer=reduce_metrics)
    # These are synchronous optimizer updates, replicated across model/DP
    # workers. A rank disagreement is unknown, never an averaged update count.
    for key in update_keys:
        values = raw_metrics.get(key, [])
        values = values if isinstance(values, list | tuple) else [values]
        if values and min(values) != max(values):
            reduced[validity_key] = 0.0
    if reduced.get(validity_key, 0.0) != 1.0:
        reduced.pop("update/optimizer_updates_completed", None)
        reduced.pop("update/optimizer_updates_skipped", None)
    actor_metrics = rename_dict(reduced, "actor/")
    if "actor/mfu" in actor_metrics:
        actor_metrics["perf/mfu/actor"] = actor_metrics.pop("actor/mfu")
    return actor_metrics


def _rollout_correction_enabled(config: DictConfig) -> bool:
    rollout_corr_config = config.algorithm.rollout_correction
    return rollout_corr_config.rollout_is is not None or rollout_corr_config.rollout_rs is not None


def _extract_reward_group_metrics(batch: SampledBatch) -> dict[str, float]:
    """Summarize reward diversity within complete uid prompt groups."""
    if not batch.group_context:
        return {}

    reward_groups = {
        uid: [float(member["reward_score"]) for member in members] for uid, members in batch.group_context.items()
    }
    group_sizes = [len(values) for values in reward_groups.values()]
    reward_spans = [max(values) - min(values) for values in reward_groups.values()]
    heterogeneous_count = sum(1 for span in reward_spans if span > 1e-12)
    metrics: dict[str, float] = {
        "train_batch/reward_group/count": float(len(reward_groups)),
        "train_batch/reward_group/heterogeneous_count": float(heterogeneous_count),
        "train_batch/reward_group/homogeneous_count": float(len(reward_groups) - heterogeneous_count),
        "train_batch/reward_group/singleton_count": float(sum(1 for size in group_sizes if size == 1)),
    }
    if group_sizes:
        metrics.update(tensor_stats("train_batch/reward_group/size", torch.tensor(group_sizes, dtype=torch.float32)))
    if reward_spans:
        metrics.update(
            tensor_stats("train_batch/reward_group/reward_span", torch.tensor(reward_spans, dtype=torch.float32))
        )
    return metrics


def _extract_train_batch_metrics(td: TensorDict) -> dict[str, float | int]:
    """Summarize sampled data before the actor update mutates the policy."""
    return train_batch_metrics(td)


def _add_train_throughput_metrics(
    metrics: dict[str, Any],
    timing_raw: dict[str, float],
    *,
    trainer_gpus: int,
    total_gpus: int,
    rollout_gpus: int | None = None,
) -> None:
    """Add trainer-side step and actor-update attention-token throughput."""
    attention_tokens = metrics.get("train_batch/num_attention_tokens")
    if attention_tokens is None:
        return

    attention_tokens = float(attention_tokens)
    metrics["resource/gpu_count/trainer"] = float(trainer_gpus)
    metrics["resource/gpu_count/total"] = float(total_gpus)
    rollout_gpus = total_gpus - trainer_gpus if rollout_gpus is None else rollout_gpus
    metrics["resource/gpu_count/rollout"] = float(rollout_gpus)
    metrics["resource/gpu_count/shared"] = float(max(0, trainer_gpus + rollout_gpus - total_gpus))
    metrics["perf/total_num_tokens"] = attention_tokens
    step_time = timing_raw.get("step")
    if step_time is not None:
        step_throughput = attention_tokens / step_time if step_time > 0 else 0.0
        metrics["train/throughput/attention_token/step"] = step_throughput
        metrics["train/throughput/attention_token_per_train_gpu/step"] = (
            step_throughput / trainer_gpus if trainer_gpus > 0 else 0.0
        )
        metrics["perf/time_per_step"] = step_time
        if total_gpus > 0 and step_time > 0:
            metrics["perf/throughput"] = attention_tokens / (step_time * total_gpus)

    update_time = timing_raw.get("update_actor")
    if update_time is not None:
        update_throughput = attention_tokens / update_time if update_time > 0 else 0.0
        metrics["train/throughput/attention_token/update_actor"] = update_throughput
        metrics["train/throughput/attention_token_per_train_gpu/update_actor"] = (
            update_throughput / trainer_gpus if trainer_gpus > 0 else 0.0
        )


@dataclass(frozen=True)
class AsyncRLTrainerConfig:
    """Typed view of async trainer options read from the existing Hydra tree."""

    total_training_steps: int | None = None
    weight_publish_interval: int = 1
    max_actor_ckpt_to_keep: int | None = None
    balance_batch: bool = True
    training_strategy: str = "separate_async"

    def __post_init__(self) -> None:
        if self.weight_publish_interval <= 0:
            raise ValueError(f"trainer.weight_publish_interval must be positive, got {self.weight_publish_interval}")

    @classmethod
    def from_config(cls, config: DictConfig) -> AsyncRLTrainerConfig:
        trainer_cfg = config.trainer
        return cls(
            total_training_steps=(
                None if trainer_cfg.total_training_steps is None else int(trainer_cfg.total_training_steps)
            ),
            weight_publish_interval=int(trainer_cfg.weight_publish_interval),
            max_actor_ckpt_to_keep=(
                None if trainer_cfg.max_actor_ckpt_to_keep is None else int(trainer_cfg.max_actor_ckpt_to_keep)
            ),
            balance_batch=bool(trainer_cfg.balance_batch),
            training_strategy=str(config.get("training_strategy", "separate_async")),
        )


class AsyncRLTrainer:
    """Driver-side actor trainer: sample ExperienceBuffer, PPO-update, publish via Relay.

    Rollout (LLM servers, dispatcher, multi-version manager) is owned by ``main_task``.
    """

    def __init__(
        self,
        config: DictConfig,
        resource_pool_manager: ResourcePoolManager,
        experience_buffer: ActorHandle,
        relay_client: RelayClient,
        tracker: ActorHandle,
        trainer_config: AsyncRLTrainerConfig,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
    ):
        self.config = config
        if _rollout_correction_enabled(config) and not bool(config.actor_rollout_ref.rollout.calculate_log_probs):
            raise ValueError("decoupled rollout correction requires actor_rollout_ref.rollout.calculate_log_probs=True")
        if bool(config.algorithm.use_kl_in_reward) or bool(config.actor_rollout_ref.actor.use_kl_loss):
            raise ValueError("async_rl does not support ref-policy KL in reward or actor loss")
        self.trainer_config = trainer_config
        self.resource_pool_manager = resource_pool_manager
        self.experience_buffer = experience_buffer
        self.experience_buffer_client = ExperienceBufferClient(experience_buffer)
        self.relay_client = relay_client
        self.rollout_dtype = str(config.actor_rollout_ref.rollout.dtype)
        # Centralized metric aggregator (Ray actor). The driver routes per-step
        # training metrics through it for ordered Tracking and JSONL writes.
        self.tracker = tracker
        self.ray_worker_group_cls = ray_worker_group_cls

        self.global_steps = 0
        self._pending_relay_publish: tuple[int, asyncio.Task[dict[str, float]]] | None = None
        self.rollout_dispatcher: ActorHandle | None = None
        self._prepared = False
        self._pending_checkpoint_metrics: dict[str, Any] | None = None
        self._colocate_rollout_manager: ActorHandle | None = None
        self._rollout_weight_version = 0

    def connect_colocate_rollout(self, *, multi_version_rollout_manager: ActorHandle) -> None:
        """Connect the existing rollout lifecycle owner for full-pool rotation."""
        self._colocate_rollout_manager = multi_version_rollout_manager

    @property
    def colocate(self) -> bool:
        return self.trainer_config.training_strategy == "colocate_async"

    async def _resume_colocate_rollout(self) -> None:
        # Publish/checkpoint operations may load actor tensors back to CUDA.
        # Fence them before explicitly releasing model, optimizer and gradients.
        await asyncio.to_thread(self.actor_wg.to, "cpu", model=True, optimizer=True, grad=True)
        await cast(Any, self._colocate_rollout_manager.resume_rollout.remote(self._rollout_weight_version))

    def connect_checkpoint_sources(self, *, rollout_dispatcher: ActorHandle) -> None:
        """Connect the prompt-dataloader owner created after the trainer actor."""
        if rollout_dispatcher is None:
            raise ValueError("AsyncRLTrainer requires rollout_dispatcher for checkpointing")
        self.rollout_dispatcher = rollout_dispatcher

    @property
    def total_training_steps(self) -> int | None:
        return self.trainer_config.total_training_steps

    # TODO(async_rl): add critic support.

    # TODO(async_rl): restore the `use_reference_policy` property when the
    # ref-policy branch is wired back in. Drive it from
    # ``algorithm.use_kl_in_reward`` OR ``actor_rollout_ref.actor.use_kl_loss``.
    # When True it should: (1) switch ``AsyncActorWorker`` role to ``actor_ref``
    # so the parent ``ActorRolloutRefWorker`` builds the ref TrainingWorker,
    # (2) compute ``ref_log_prob`` before trainer-side advantage estimation,
    # (3) feed ``ref_log_prob`` into the actor mini-batch + KL-in-loss branch
    # of the actor PPO loss, (4) optionally apply KL-in-reward before advantage.

    @property
    def train_batch_size(self) -> int:
        # We multiply by ``n`` because the dataset side reports prompt-bsz, but the
        # buffer hands us per-trajectory keys.
        rollout_cfg = self.config.actor_rollout_ref.rollout
        return int(self.config.data.train_batch_size) * int(rollout_cfg.n)

    def actor_mini_batch_size(self) -> int:
        # Keep the public actor.ppo_mini_batch_size semantics aligned with
        # sync PPO: it is configured in prompt units. The async trainer samples
        # trajectory records from the buffer, so derive the worker mini-batch at
        # the call boundary.
        rollout_cfg = self.config.actor_rollout_ref.rollout
        return int(self.config.actor_rollout_ref.actor.ppo_mini_batch_size) * int(rollout_cfg.n)

    def actor_num_mini_batches(self) -> int:
        """Return the number of sequential optimizer updates in one training step."""
        train_batch_size = self.train_batch_size
        mini_batch_size = self.actor_mini_batch_size()
        if train_batch_size % mini_batch_size != 0:
            raise ValueError(
                "async actor trajectory batch must be divisible by its trajectory mini-batch: "
                f"train_batch_size={train_batch_size}, mini_batch_size={mini_batch_size}"
            )
        return train_batch_size // mini_batch_size

    def _step_should_publish_weights(
        self,
        step: int,
        total_steps: int | None,
        *,
        checkpoint_due: bool,
    ) -> bool:
        if checkpoint_due:
            return True
        if total_steps is not None and step >= total_steps:
            return False
        return step % self.trainer_config.weight_publish_interval == 0

    # ------------------------------------------------------------ Workers --

    def init_workers(self, llm_server_controllers: Sequence[ActorHandle]) -> None:
        """Create actor workers and connect them to rollout-side relay state."""
        self._init_actor_workers()
        actor_state_meta_dict = self._connect_relay()
        self._init_relay_rollout_stages(
            llm_server_controllers=llm_server_controllers,
            state_meta_dict=actor_state_meta_dict,
        )

    def _init_actor_workers(self) -> None:
        """Create the actor worker group on ``trainer_pool`` and initialize models."""
        actor_cfg = self.config.actor_rollout_ref.actor
        actor_strategy = str(actor_cfg.strategy)
        if actor_strategy == "megatron":
            router_replay_mode = str(actor_cfg.megatron.router_replay.mode)
            if router_replay_mode == "R2":
                raise ValueError("async_rl classic path supports Megatron router replay mode disabled or R3, not R2")
            if router_replay_mode == "R3" and not bool(
                self.config.actor_rollout_ref.rollout.enable_rollout_routing_replay
            ):
                raise ValueError("Megatron R3 requires rollout.enable_rollout_routing_replay=True")
        logger.info(
            f"Initializing workers; resource pools: {list(self.resource_pool_manager.resource_pool_dict.keys())}"
        )

        # Build cls→pool mapping. We only spawn actor today.
        # TODO(async_rl): add critic support.
        actor_pool = self.resource_pool_manager.get_resource_pool(Role.Actor)
        actor_worker_cls = _select_async_actor_worker_cls(actor_strategy)
        actor_cls = RayClassWithInitArgs(
            cls=ray.remote(actor_worker_cls),
            config=self.config.actor_rollout_ref,
            role="actor",
        )
        actor_class_dict = {"actor": actor_cls}

        worker_dict_cls = create_colocated_worker_cls(class_dict=actor_class_dict)
        wg_kwargs = {}
        if self.config.trainer.ray_wait_register_center_timeout is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        wg_kwargs["device_name"] = self.config.trainer.device
        actor_worker_env = dict(self.config.trainer.worker_env)
        if actor_worker_env:
            wg_kwargs["worker_env"] = actor_worker_env

        self._wg_dict = self.ray_worker_group_cls(
            resource_pool=actor_pool, ray_cls_with_init=worker_dict_cls, **wg_kwargs
        )
        spawn_wg = self._wg_dict.spawn(prefix_set=actor_class_dict.keys())
        self.actor_wg = spawn_wg["actor"]
        logger.info("%s group spawned for strategy=%s; calling init_model()", actor_worker_cls.__name__, actor_strategy)
        self.actor_wg.init_model()

        # TODO(async_rl): when ref-policy support is added, spawn the actor with
        # role="actor_ref" so the parent builds the ref TrainingWorker
        # alongside the actor on the same pool.
        # TODO(async_rl): add critic support.

    def _connect_relay(self) -> dict[str, TensorMeta]:
        """Connect actor ranks to Relay ingress and initialize actor-derived metadata."""
        # 1. Probe the Relay chain-head endpoint from every actor rank.
        self.relay_client.connect_actor_workers(self.actor_wg)
        logger.info("Actor workers connected to Relay master")

        # 2. Allocate the full shm tensors every actor publisher writes into.
        actor_state_meta_dict = self.actor_wg.get_actor_state_meta_dict(dtype=self.rollout_dtype)
        if isinstance(actor_state_meta_dict, list):
            publisher_results = [item for item in actor_state_meta_dict if item is not None]
            if len(publisher_results) != 1:
                raise RuntimeError(
                    "get_actor_state_meta_dict expected exactly one Relay actor publisher result, "
                    f"got {len(publisher_results)}"
                )
            actor_state_meta_dict = publisher_results[0]
        ray.get(
            self.relay_client.init_full_state_dict_in_shm(
                full_state_meta_dict=actor_state_meta_dict,
            )
        )
        logger.info("Relay actor metadata initialized.")
        return actor_state_meta_dict

    def _init_relay_rollout_stages(
        self,
        *,
        llm_server_controllers: Sequence[ActorHandle],
        state_meta_dict: dict[str, TensorMeta],
    ) -> None:
        """Initialize rollout-side Relay stages at the LLM server controller locations."""
        ray.get(
            self.relay_client.init_rollout_state_dict_in_shm(
                rollout_servers=llm_server_controllers,
                state_meta_dict=state_meta_dict,
            )
        )
        logger.info("Relay rollout stages initialized.")

    # ----------------------------------------------------- Weight publish --

    async def publish_actor_weights_to_relay(self, version: int) -> dict[str, float]:
        """Publish one trainer version through the RelayClient-owned fence."""
        return await self.relay_client.publish(
            self.actor_wg,
            version=version,
            dtype=self.rollout_dtype,
        )

    def _schedule_relay_publish(self, *, version: int) -> None:
        pending = self._pending_relay_publish
        if pending is not None:
            raise RuntimeError(
                f"Cannot schedule overlapping Relay publishes: pending_version={pending[0]}, next_version={version}"
            )
        self._pending_relay_publish = (
            version,
            asyncio.create_task(
                self.publish_actor_weights_to_relay(version=version),
                name=f"relay-publish-v{version}",
            ),
        )

    async def _wait_for_pending_relay_publish(self, metrics: dict[str, Any]) -> float:
        pending = self._pending_relay_publish
        if pending is None:
            return 0.0

        version, task = pending
        wait_start_time = time.perf_counter()
        try:
            publish_metrics = await task
        except Exception:
            logger.exception("Relay publish failed for version=%s; terminating the training job", version)
            raise
        finally:
            if self._pending_relay_publish is pending:
                self._pending_relay_publish = None

        wait_time_s = time.perf_counter() - wait_start_time
        metrics.update(publish_metrics)
        self._rollout_weight_version = version
        return wait_time_s

    async def _await_sample_or_relay_failure(
        self,
        sample_ref: Any,
        metrics: dict[str, Any],
    ) -> tuple[SampledBatch, float]:
        """Await a train batch while surfacing a failed publish immediately."""
        pending = self._pending_relay_publish
        if pending is None:
            return cast(SampledBatch, await sample_ref), 0.0
        _, publish_task = pending

        sample_task = asyncio.ensure_future(sample_ref)
        done, _ = await asyncio.wait(
            {sample_task, publish_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        relay_wait_time_s = 0.0
        if publish_task in done:
            relay_wait_time_s = await self._wait_for_pending_relay_publish(metrics)
        return cast(SampledBatch, await sample_task), relay_wait_time_s

    # ----------------------------------------------------------- Checkpoint --

    def _validate_checkpoint_config(self) -> None:
        self._validate_checkpoint_storage_config()
        actor_cfg = self.config.actor_rollout_ref.actor
        save_contents = set(actor_cfg.checkpoint.save_contents)
        load_contents = set(actor_cfg.checkpoint.load_contents)
        required = {"model", "optimizer", "extra"}
        if not required.issubset(save_contents) or not required.issubset(load_contents):
            raise ValueError(
                "trainer checkpoint requires actor checkpoint save/load contents "
                f"{sorted(required)}; save={sorted(save_contents)} load={sorted(load_contents)}"
            )
        if bool(actor_cfg.checkpoint.async_save):
            raise ValueError("async-RL trainer checkpoint requires actor checkpoint.async_save=False")
        if str(actor_cfg.strategy) == "megatron" and not bool(actor_cfg.optim.use_checkpoint_opt_param_scheduler):
            raise ValueError("Megatron trainer checkpoint requires actor.optim.use_checkpoint_opt_param_scheduler=True")

    def _validate_checkpoint_storage_config(self) -> None:
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError(
                "async-RL trainer checkpoint does not support actor-only default_hdfs_dir; "
                "use a durable or HDFS-FUSE trainer.default_local_dir for the complete "
                "actor/data/latest checkpoint"
            )

    def _checkpoint_root(self) -> Path:
        root = Path(str(self.config.trainer.default_local_dir))
        return root if root.is_absolute() else Path.cwd() / root

    def _prune_complete_local_checkpoints(
        self,
        checkpoint_root: Path,
        *,
        current_step: int,
    ) -> tuple[int, int]:
        """Best-effort retention after commit, always preserving the marker target."""
        max_to_keep = self.trainer_config.max_actor_ckpt_to_keep
        if max_to_keep is None or max_to_keep <= 0:
            return 0, 0

        complete_checkpoints: list[tuple[int, Path]] = []
        try:
            checkpoint_paths = list(checkpoint_root.iterdir())
        except OSError:
            logger.warning(
                "Committed checkpoint step=%s but could not scan %s for retention",
                current_step,
                checkpoint_root,
                exc_info=True,
            )
            return 0, 1
        for path in checkpoint_paths:
            if not path.is_dir() or not path.name.startswith("global_step_"):
                continue
            try:
                step = int(path.name.removeprefix("global_step_"))
            except ValueError:
                continue
            if path.name != f"global_step_{step}" or step < 0:
                continue
            if (path / "actor").is_dir() and (path / "data.pt").is_file():
                complete_checkpoints.append((step, path))

        keep_paths = {checkpoint_root / f"global_step_{current_step}"}
        other_checkpoints = sorted(
            ((step, path) for step, path in complete_checkpoints if step != current_step),
            reverse=True,
        )
        keep_paths.update(path for _, path in other_checkpoints[: max_to_keep - 1])
        pruned_count = 0
        failed_count = 0
        for _, path in complete_checkpoints:
            if path in keep_paths:
                continue
            try:
                shutil.rmtree(path)
                pruned_count += 1
            except OSError:
                failed_count += 1
                logger.warning(
                    "Committed checkpoint step=%s but could not prune old checkpoint %s",
                    current_step,
                    path,
                    exc_info=True,
                )
        return pruned_count, failed_count

    async def _save_checkpoint(self) -> dict[str, Any]:
        """Save the actor train state and prompt cursor using the verl layout."""
        if self.rollout_dispatcher is None:
            raise RuntimeError("AsyncRLTrainer.connect_checkpoint_sources must run before checkpointing")
        self._validate_checkpoint_storage_config()
        checkpoint_root = self._checkpoint_root()
        relay_version = int(await self.relay_client.get_loadable_version())
        if relay_version != self.global_steps:
            raise RuntimeError(
                "trainer checkpoint requires an exact committed Relay version: "
                f"step={self.global_steps} loadable_version={relay_version}"
            )

        checkpoint_dir = checkpoint_root / f"global_step_{self.global_steps}"
        actor_dir = checkpoint_dir / "actor"
        data_path = checkpoint_dir / "data.pt"
        save_started = time.perf_counter()
        dispatcher_ref = self.rollout_dispatcher.save_checkpoint.remote(str(data_path))  # type: ignore[attr-defined]
        actor_error: BaseException | None = None
        try:
            self.actor_wg.save_checkpoint(
                local_path=str(actor_dir),
                hdfs_path=None,
                global_step=self.global_steps,
                # Actor-only retention can delete the checkpoint still named by
                # the trainer's latest marker before dispatcher data.pt commits.
                # The trainer prunes complete directories after advancing that
                # marker instead.
                max_ckpt_to_keep=None,
            )
        except BaseException as exc:
            actor_error = exc
        try:
            await cast(Any, dispatcher_ref)
        except BaseException as dispatcher_error:
            if actor_error is not None:
                actor_error.add_note(f"prompt cursor checkpoint also failed: {dispatcher_error!r}")
                raise actor_error from dispatcher_error
            raise
        if actor_error is not None:
            raise actor_error

        checkpoint_root.mkdir(parents=True, exist_ok=True)
        # Advance the marker only after both actor state and data.pt completed.
        tracker_file = get_checkpoint_tracker_filename(str(checkpoint_root))
        temp_tracker_file = tracker_file + ".tmp"
        with open(temp_tracker_file, "w") as f:
            f.write(f"{self.global_steps}\n")
        os.rename(temp_tracker_file, tracker_file)
        retention_pruned_count, retention_failed_count = self._prune_complete_local_checkpoints(
            checkpoint_root,
            current_step=self.global_steps,
        )
        save_time_s = time.perf_counter() - save_started
        logger.info(
            "Saved async-RL trainer checkpoint step=%s path=%s time_s=%.3f",
            self.global_steps,
            checkpoint_dir,
            save_time_s,
        )
        return {
            "checkpoint/completed": 1.0,
            "checkpoint/step": float(self.global_steps),
            "checkpoint/save_time_s": save_time_s,
            "checkpoint/path": str(checkpoint_dir),
            "checkpoint/retention_pruned_count": float(retention_pruned_count),
            "checkpoint/retention_failed_count": float(retention_failed_count),
        }

    async def _load_checkpoint(self) -> bool:
        """Restore the actor train state and prompt cursor, if configured."""
        resume_mode = str(self.config.trainer.resume_mode)
        if resume_mode == "disable":
            return False
        if self.rollout_dispatcher is None:
            raise RuntimeError("AsyncRLTrainer.connect_checkpoint_sources must run before resume")
        self._validate_checkpoint_storage_config()

        if resume_mode == "auto":
            checkpoint_path = find_latest_ckpt_path(str(self._checkpoint_root()))
            if checkpoint_path is None:
                logger.info("No async-RL trainer checkpoint found; training from scratch")
                return False
            checkpoint_dir = Path(checkpoint_path)
        elif resume_mode == "resume_path":
            resume_from_path = self.config.trainer.resume_from_path
            if not isinstance(resume_from_path, str):
                raise ValueError("trainer.resume_from_path must be a string in resume_path mode")
            checkpoint_dir = Path(resume_from_path)
            if not checkpoint_dir.is_absolute():
                checkpoint_dir = Path.cwd() / checkpoint_dir
        else:
            raise ValueError(f"Unknown resume_mode: {resume_mode}")

        try:
            step = int(checkpoint_dir.name.removeprefix("global_step_"))
        except ValueError as exc:
            raise RuntimeError(f"invalid async-RL checkpoint directory: {checkpoint_dir}") from exc
        if checkpoint_dir.name != f"global_step_{step}" or step < 0:
            raise RuntimeError(f"invalid async-RL checkpoint directory: {checkpoint_dir}")
        actor_dir = checkpoint_dir / "actor"
        data_path = checkpoint_dir / "data.pt"
        if not actor_dir.is_dir() or not data_path.is_file():
            raise RuntimeError(
                "incomplete async-RL trainer checkpoint: "
                f"actor_exists={actor_dir.is_dir()} data_exists={data_path.is_file()} path={checkpoint_dir}"
            )

        self.actor_wg.load_checkpoint(
            local_path=str(actor_dir),
            hdfs_path=None,
            del_local_after_load=bool(self.config.trainer.del_local_ckpt_after_load),
        )
        await cast(
            Any,
            self.rollout_dispatcher.load_checkpoint.remote(str(data_path)),  # type: ignore[attr-defined]
        )
        self.global_steps = step
        logger.info(
            "Restored async-RL trainer checkpoint step=%s path=%s; "
            "ExperienceBuffer and runtime in-flight state start empty",
            step,
            checkpoint_dir,
        )
        return True

    # ----------------------------------------------------- Sample → update --

    def _materialize_sampled_batch(
        self,
        batch: SampledBatch,
        *,
        metrics: dict[str, Any],
    ) -> TensorDict:
        """Materialize a sampled batch into a contiguous TensorDict.

        Fields fetched match what the actor PPO loss expects:
            * ``input_ids`` (prompt+response, padded)
            * ``attention_mask`` (matching ``input_ids``)
            * ``position_ids`` (producer layout, including VL RoPE rows)
            * ``response_mask`` (over the response window)
            * ``behav_log_probs`` / ``rollout_log_probs`` from rollout

        TODO(async_rl): when ref-policy support is added, append ``ref_log_prob``
        whenever the trainer has run ``_compute_ref_log_prob`` earlier in the
        step (so the actor mini-batch carries it for KL-in-loss).
        """
        rollout_cfg = self.config.actor_rollout_ref.rollout
        require_routed_experts = bool(rollout_cfg.get("enable_rollout_routing_replay", False))
        if require_routed_experts:
            actor_cfg = self.config.actor_rollout_ref.actor
            require_routed_experts = actor_cfg.strategy == "megatron" and actor_cfg.megatron.router_replay.mode == "R3"
        return materialize_batch(
            batch,
            metrics=metrics,
            require_routed_experts=require_routed_experts,
            use_remove_padding=bool(self.config.actor_rollout_ref.model.use_remove_padding),
            temperature=float(self.config.actor_rollout_ref.rollout.temperature),
        )

    def _compute_advantages(
        self,
        batch: SampledBatch,
        td: TensorDict,
        metrics: dict[str, Any],
    ) -> None:
        """Compute session advantages from complete terminal prompt context.

        The buffer samples trajectories independently, so the sampled rows are
        not necessarily a complete prompt group. ``SampledBatch.group_context``
        supplies one small terminal-reward record per successful session. The
        estimator runs over those session records, then each selected
        trajectory receives its session's scalar advantage.

        Failed sessions have no trajectory key and therefore contribute no
        synthetic zero-reward member.
        """
        response_mask = td["response_mask"]
        rm_scores = td["rm_scores"]
        if response_mask.is_nested or rm_scores.is_nested:
            raise RuntimeError("trainer-side advantage expects padded response_mask and rm_scores")

        unit_groups: list[str] = []
        unit_rewards: list[float] = []
        session_unit: dict[tuple[str, int], int] = {}
        for uid, members in batch.group_context.items():
            for member in members:
                session_id = int(member["session_id"])
                identity = (uid, session_id)
                if identity in session_unit:
                    raise RuntimeError(
                        "prompt-group context must contain exactly one final trajectory per session: "
                        f"uid={uid!r} session_id={session_id}"
                    )
                session_unit[identity] = len(unit_rewards)
                unit_groups.append(uid)
                unit_rewards.append(float(member["reward_score"]))

        row_to_unit: list[int] = []
        for key in batch.keys:
            uid, session_id, _output_index = trajectory_key_facts(key)
            identity = (uid, session_id)
            unit_index = session_unit.get(identity)
            if unit_index is None:
                raise RuntimeError(
                    "sampled trajectory is missing complete prompt-group advantage context: "
                    f"key={key!r} uid={uid!r} session_id={session_id}"
                )
            row_to_unit.append(unit_index)

        if not unit_rewards:
            raise RuntimeError("cannot compute advantages for an empty trajectory batch")

        algorithm_config = self.config.algorithm
        adv_estimator = algorithm_config.adv_estimator
        adv_name = (
            adv_estimator.value if isinstance(adv_estimator, core_algos.AdvantageEstimator) else str(adv_estimator)
        )
        if adv_name not in {core_algos.AdvantageEstimator.GRPO, core_algos.AdvantageEstimator.RLOO}:
            raise NotImplementedError(
                f"async_rl trainer-side session advantage currently supports only GRPO and RLOO; got {adv_name!r}"
            )
        reward_tensor = torch.tensor(unit_rewards, dtype=torch.float32, device=rm_scores.device).unsqueeze(-1)
        unit_mask = torch.ones_like(reward_tensor)
        group_index = np.asarray(unit_groups, dtype=object)
        advantage_start = time.perf_counter()
        unit_advantages, unit_returns = core_algos.get_adv_estimator_fn(adv_estimator)(
            token_level_rewards=reward_tensor,
            response_mask=unit_mask,
            index=group_index,
            config=algorithm_config,
            norm_adv_by_std_in_grpo=bool(algorithm_config.norm_adv_by_std_in_grpo),
        )
        if unit_advantages.shape != reward_tensor.shape or unit_returns.shape != reward_tensor.shape:
            raise RuntimeError(
                "async_rl outcome estimator must return one scalar per session: "
                f"advantages={tuple(unit_advantages.shape)} returns={tuple(unit_returns.shape)}"
            )

        row_indices = torch.tensor(row_to_unit, dtype=torch.long, device=unit_advantages.device)
        row_advantages = unit_advantages[:, 0].index_select(0, row_indices).unsqueeze(-1)
        row_returns = unit_returns[:, 0].index_select(0, row_indices).unsqueeze(-1)
        td["advantages"] = row_advantages * response_mask
        td["returns"] = row_returns * response_mask
        metrics["timing/advantage"] = time.perf_counter() - advantage_start
        metrics["advantage/source_prompt_group_count"] = float(len(batch.group_context))
        metrics["advantage/source_session_count"] = float(len(session_unit))

    def _compute_old_log_probs(self, td: TensorDict, metrics: dict[str, Any]) -> None:
        """Recompute the update-start policy log probabilities before training."""
        infer_td = td.clone(recurse=False)
        if not infer_td["input_ids"].is_nested:
            infer_td = left_right_2_no_padding(infer_td)
        actor_cfg = self.config.actor_rollout_ref.actor
        calculate_entropy = bool(actor_cfg.calculate_entropy or actor_cfg.entropy_coeff != 0.0)
        calculate_sum_pi_squared = bool(actor_cfg.calculate_sum_pi_squared)
        tu.assign_non_tensor(
            infer_td,
            calculate_entropy=calculate_entropy,
            calculate_sum_pi_squared=calculate_sum_pi_squared,
            compute_loss=False,
        )
        metrics["old_log_prob/calculate_entropy"] = float(calculate_entropy)

        old_log_prob_start = time.perf_counter()
        output = self.actor_wg.compute_log_prob(infer_td)
        metrics["timing/old_log_prob"] = time.perf_counter() - old_log_prob_start
        metrics["old_log_prob/recomputed"] = 1.0

        log_probs = tu.get(output, "log_probs")
        if log_probs is None:
            raise RuntimeError("actor_wg.compute_log_prob did not return log_probs")
        td["old_log_probs"] = no_padding_2_padding(log_probs, infer_td).float()

        entropy = tu.get(output, "entropy")
        if entropy is not None:
            td["entropy"] = no_padding_2_padding(entropy, infer_td).float()
        output_metrics = tu.get(output, "metrics", default=None)
        if isinstance(output_metrics, dict):
            raw_output_metrics: dict[str, list[Any]] = {}
            for key, value in output_metrics.items():
                _append_metric_values(raw_output_metrics, key, value)
            reduced_output_metrics = reduce_metrics(raw_output_metrics)
            if "mfu" in reduced_output_metrics:
                metrics["perf/mfu/old_log_prob"] = float(reduced_output_metrics["mfu"])
            for metric_key in (
                "perf/dynamic_micro_batches",
                "perf/dynamic_micro_batch_valid_tokens_per_gpu",
                "perf/dynamic_micro_batch_valid_tokens_per_gpu/max",
                "perf/dynamic_micro_batch_valid_tokens_per_gpu/min",
            ):
                if metric_key in reduced_output_metrics:
                    metrics[f"old_log_prob/{metric_key}"] = float(reduced_output_metrics[metric_key])

    def _apply_rollout_correction(self, td: TensorDict, metrics: dict[str, Any]) -> None:
        """Apply baseline decoupled IS/RS after the stable old-policy forward."""
        if not _rollout_correction_enabled(self.config):
            return
        if "rollout_log_probs" not in td:
            raise RuntimeError("decoupled rollout correction requires rollout_log_probs in every sampled trajectory")

        correction = self.config.algorithm.rollout_correction
        weights, response_mask, correction_metrics = compute_rollout_correction_and_rejection_mask(
            old_log_prob=td["old_log_probs"],
            rollout_log_prob=td["rollout_log_probs"],
            response_mask=td["response_mask"],
            rollout_is=correction.rollout_is,
            rollout_is_threshold=correction.rollout_is_threshold,
            rollout_is_batch_normalize=correction.rollout_is_batch_normalize,
            rollout_rs=correction.rollout_rs,
            rollout_rs_threshold=correction.rollout_rs_threshold,
        )
        td["response_mask"] = response_mask
        td["loss_mask"] = response_mask
        if "advantages" in td:
            td["advantages"] = td["advantages"] * response_mask
        if "returns" in td:
            td["returns"] = td["returns"] * response_mask
        if weights is not None:
            td["rollout_is_weights"] = weights.batch["rollout_is_weights"]
        metrics.update(correction_metrics)

    # TODO(async_rl): add critic support.

    # TODO(async_rl): restore ``_compute_ref_log_prob`` here when the
    # ref-policy branch is wired back in. It should run
    # ``self.actor_wg.compute_ref_log_prob(td)`` (the parent
    # ``ActorRolloutRefWorker`` already exposes this method when role
    # contains ``ref``) and make ``ref_log_prob`` available before either
    # trainer-side KL-in-reward or KL-in-loss consumes it.

    # TODO(async_rl): add critic update support.

    def _balance_actor_batch(self, td: TensorDict, metrics: dict[str, Any]) -> TensorDict:
        """Reorder trajectories so equal-size DP shards have balanced transformer workload."""
        dispatch_info = self.actor_wg._dispatch_info
        if "actor" not in dispatch_info:
            dispatch_info["actor"] = self.actor_wg._query_dispatch_info("actor")
        dp_size = max(dispatch_info["actor"]) + 1

        input_ids = td["input_ids"]
        if input_ids.is_nested:
            sequence_lengths = input_ids.offsets().diff().long()
        else:
            sequence_lengths = td["attention_mask"].view(len(td), -1).sum(dim=-1).long()
        if len(sequence_lengths) % dp_size != 0:
            raise RuntimeError(
                "trainer.balance_batch requires the trajectory batch size to be divisible by actor DP size: "
                f"batch_size={len(sequence_lengths)} dp_size={dp_size}"
            )

        workloads = calculate_workload(sequence_lengths).cpu().tolist()
        partitions = get_seqlen_balanced_partitions(workloads, k_partitions=dp_size, equal_size=True)
        for index, partition in enumerate(partitions):
            partition.sort(key=lambda item: (workloads[item], item))
            partitions[index] = partition[::2] + partition[1::2][::-1]

        sequence_length_list = sequence_lengths.cpu().tolist()
        metrics.update(log_seqlen_unbalance(sequence_length_list, partitions, prefix="global_seqlen"))
        metrics.update(log_seqlen_unbalance(workloads, partitions, prefix="global_workload"))
        metrics["trainer/dp_balance/dp_size"] = float(dp_size)

        global_indices = [item for partition in partitions for item in partition]
        td = tu.index_select_tensor_dict(td, global_indices)
        td["global_token_num"] = NonTensorData([sequence_length_list[index] for index in global_indices])
        return td

    def _materialize_actor_batch(self, batch: SampledBatch, metrics: dict) -> TensorDict:
        logger.info(
            "AsyncRLTrainer step=%s materialize_actor_batch: materializing %d keys from partition=%s",
            self.global_steps,
            len(batch.keys),
            batch.partition_id,
        )
        materialize_start = time.perf_counter()
        td = self._materialize_sampled_batch(batch, metrics=metrics)
        self._compute_advantages(batch, td, metrics)
        balance_batch = self.trainer_config.balance_batch
        metrics["trainer/dp_balance/enabled"] = float(balance_batch)
        if balance_batch:
            td = self._balance_actor_batch(td, metrics)
        metrics["timing/materialize_batch"] = time.perf_counter() - materialize_start
        batch_shape = tuple(td["input_ids"].shape) if "input_ids" in td else None
        response_shape = tuple(td["responses"].shape) if "responses" in td else None
        logger.info(
            "AsyncRLTrainer step=%s materialize_actor_batch: materialized input_ids=%s responses=%s",
            self.global_steps,
            batch_shape,
            response_shape,
        )
        metrics.update(_extract_reward_group_metrics(batch))
        metrics.update(_extract_train_batch_metrics(td))
        metrics.update(sampled_metadata_metrics(batch, trainer_version=self.global_steps - 1))
        num_action_tokens = float(metrics.get("train_batch/num_action_tokens", 0.0) or 0.0)
        if num_action_tokens <= 0:
            raise RuntimeError(
                "AsyncRLTrainer sampled a batch with zero action tokens; "
                "refusing actor update because the batch contains no trainable rollout tokens."
            )
        return td

    def _compute_actor_update_inputs(self, td: TensorDict, metrics: dict) -> None:
        """Compute read-only actor inputs without blocking the driver event loop."""
        self._compute_old_log_probs(td, metrics)
        self._apply_rollout_correction(td, metrics)
        restore_train_mode = bool(metrics.get("old_log_prob/recomputed", 0.0))
        metrics["old_log_prob/restore_train_mode"] = float(restore_train_mode)
        if restore_train_mode:
            restore_start = time.perf_counter()
            self.actor_wg.restore_actor_train_mode()
            metrics["timing/old_log_prob_restore_train_mode"] = time.perf_counter() - restore_start
        else:
            metrics["timing/old_log_prob_restore_train_mode"] = 0.0

    async def _prepare_actor_update(self, td: TensorDict, metrics: dict) -> None:
        """Prepare read-only PPO inputs while allowing the previous publish to finish."""
        # Actor worker-group calls are blocking. Run this read-only phase off
        # the driver loop so the pending publish can process its broadcast ack
        # and advance loadable_version while old-log-prob is executing.
        await asyncio.to_thread(self._compute_actor_update_inputs, td, metrics)

    def _update_actor(self, td: TensorDict, metrics: dict) -> None:
        """Mutate actor weights after the previous Relay publish fence."""

        if "rm_scores" in td:
            td.pop("rm_scores")
        actor_cfg = self.config.actor_rollout_ref.actor
        trajectory_batch_size = self.train_batch_size
        ppo_mini_batch_size = self.actor_mini_batch_size()
        num_mini_batches = self.actor_num_mini_batches()
        if td.shape[0] != trajectory_batch_size:
            raise RuntimeError(
                "async actor materialized batch size does not match the configured trajectory batch: "
                f"materialized={td.shape[0]}, configured={trajectory_batch_size}"
            )
        # Mainline verl defines global_batch_size as the global batch used by each
        # optimizer update, not the full PPO batch that is split into updates.
        tu.assign_non_tensor(
            td,
            mini_batch_size=ppo_mini_batch_size,
            global_batch_size=ppo_mini_batch_size,
            epochs=actor_cfg.ppo_epochs,
            calculate_entropy=bool(actor_cfg.calculate_entropy or actor_cfg.entropy_coeff != 0.0),
            seed=actor_cfg.data_loader_seed,
            dataloader_kwargs={"shuffle": bool(actor_cfg.shuffle)},
            compute_loss=True,
        )
        metrics["actor/update/trajectory_batch_size"] = float(trajectory_batch_size)
        metrics["actor/update/trajectory_mini_batch_size"] = float(ppo_mini_batch_size)
        metrics["actor/update/mini_batches_per_epoch"] = float(num_mini_batches)
        metrics["actor/update/optimizer_updates_expected"] = float(num_mini_batches * int(actor_cfg.ppo_epochs))
        logger.info(
            "AsyncRLTrainer step=%s _update_actor: calling actor_wg.update_actor "
            "trajectory_batch_size=%s prompt_mini_batch_size=%s "
            "trajectory_mini_batch_size=%s mini_batches_per_epoch=%s epochs=%s",
            self.global_steps,
            trajectory_batch_size,
            actor_cfg.ppo_mini_batch_size,
            ppo_mini_batch_size,
            num_mini_batches,
            actor_cfg.ppo_epochs,
        )
        actor_update_start = time.perf_counter()
        output = self.actor_wg.update_actor(td)
        metrics["timing/actor_wg_update_actor"] = time.perf_counter() - actor_update_start
        metrics.update(_extract_actor_update_metrics(output))
        metrics.setdefault("actor/update/optimizer_updates_valid", 0.0)
        if metrics["actor/update/optimizer_updates_valid"] == 1.0:
            metrics["actor/update/optimizer_updates"] = metrics["actor/update/optimizer_updates_completed"]
        logger.info(
            "AsyncRLTrainer step=%s _update_actor: actor_wg.update_actor done metric_keys=%s",
            self.global_steps,
            sorted(metrics.keys()),
        )

    # ----------------------------------------------------------- Fit loop --

    async def prepare(self) -> None:
        """Restore train state and seed Relay before rollout dispatch starts."""
        if self._prepared:
            return

        self._validate_checkpoint_config()
        logger.info("AsyncRLTrainer prepare: load_checkpoint start")
        restored = bool(await self._load_checkpoint())
        logger.info(
            "AsyncRLTrainer prepare: load_checkpoint done restored=%s global_steps=%s",
            restored,
            self.global_steps,
        )
        # Restore the dispatcher cursor before starting either rollout dispatch or
        # Store polling. ExperienceBuffer intentionally restarts empty.
        await cast(Any, self.experience_buffer.start.remote())
        await cast(Any, self.tracker.set_next_step_to_log.remote(self.global_steps))

        logger.info("Pushing initial/restored weights version=%s to relay", self.global_steps)
        await self.publish_actor_weights_to_relay(version=self.global_steps)

        if self.colocate:
            if self._colocate_rollout_manager is None:
                raise RuntimeError("colocate_async requires a connected rollout lifecycle manager")
            self._rollout_weight_version = self.global_steps
            await self._resume_colocate_rollout()

        self._prepared = True

    async def _commit_step_metrics(
        self, metrics: dict[str, Any], timing_raw: dict[str, float], *, step_start_time: float
    ) -> tuple[dict[str, Any], int]:
        """Collect live metrics, close the step timer, and commit its immutable row."""
        # The live-source collection ack is part of the trainer-owned step
        # wall. Tracker commit/persistence is the causal tail after that
        # timer and is reported separately.
        with simple_timer("tracker_collect_remote_metrics", timing_raw):
            await cast(
                Any,
                self.tracker.collect_and_stage_remote_metrics.remote(step=self.global_steps),
            )

        timing_raw["step"] = time.perf_counter() - step_start_time
        metrics.update({f"timing/{k}": v for k, v in timing_raw.items()})
        trainer_cfg = self.config.trainer
        rollout_cfg = self.config.actor_rollout_ref.rollout
        trainer_gpus = int(trainer_cfg.n_gpus_per_node) * int(trainer_cfg.nnodes)
        rollout_gpus = int(rollout_cfg.n_gpus_per_node) * int(rollout_cfg.nnodes)
        total_gpus = trainer_gpus if self.colocate else trainer_gpus + rollout_gpus
        _add_train_throughput_metrics(
            metrics,
            timing_raw,
            trainer_gpus=trainer_gpus,
            total_gpus=total_gpus,
            rollout_gpus=rollout_gpus,
        )
        metrics["training/global_step"] = self.global_steps
        metrics = export_flat_metrics(metrics)

        await cast(
            Any,
            self.tracker.commit_metrics.remote(step=self.global_steps, train_metrics=metrics),
        )
        logger.info(f"[Step {self.global_steps}] {metrics}")
        return metrics, total_gpus

    async def _save_checkpoint_and_resume_rollout(self, *, checkpoint_due: bool, is_last_step: bool) -> None:
        """Finish checkpoint I/O before handing colocated GPUs back to rollout."""
        # Trainer checkpoint I/O is a causal tail after this step's committed
        # metrics row. Its complete latency is carried to the next row and
        # is never back-written into timing/step.
        if checkpoint_due:
            checkpoint_metrics = await self._save_checkpoint()
            self._pending_checkpoint_metrics = checkpoint_metrics
            await cast(
                Any,
                self.tracker.record_event.remote(
                    step=self.global_steps,
                    event_type="checkpoint_completed",
                    event_id=f"checkpoint:{self.global_steps}",
                    metrics=checkpoint_metrics,
                ),
            )
            logger.info(
                "AsyncRLTrainer checkpoint tail: step=%s metrics=%s",
                self.global_steps,
                checkpoint_metrics,
            )

        if self.colocate and not is_last_step:
            resume_start = time.perf_counter()
            await self._resume_colocate_rollout()
            # Like checkpoint I/O, handoff is a tail after committed metrics.
            # Carry its latency forward without rewriting the previous row.
            if self._pending_checkpoint_metrics is None:
                self._pending_checkpoint_metrics = {}
            self._pending_checkpoint_metrics["timing/colocate_resume"] = time.perf_counter() - resume_start

    async def _record_training_cycle(
        self,
        metrics: dict[str, Any],
        *,
        step_start_time: float,
        fit_started_at: float,
        total_gpus: int,
    ) -> None:
        """Record the full cycle after the durable step row and its tails."""
        # The complete cycle includes commit, checkpoint and handoff tails.
        # It is a separate immutable event because those tails finish after
        # the step row. Resource counts are declared pool sizes, not billing.
        cycle_time_s = time.perf_counter() - step_start_time
        cycle_metrics = {
            "timing/cycle": cycle_time_s,
            "training/elapsed_s": time.perf_counter() - fit_started_at,
            "resource/configured_gpu_seconds/cycle": cycle_time_s * total_gpus,
            "resource/gpu_count/total": float(total_gpus),
        }
        if "train_batch/num_attention_tokens" in metrics and cycle_time_s > 0:
            cycle_metrics["train/throughput/attention_token/cycle"] = (
                metrics["train_batch/num_attention_tokens"] / cycle_time_s
            )
        await cast(
            Any,
            self.tracker.record_event.remote(
                step=self.global_steps,
                event_type="training_cycle_completed",
                event_id=f"cycle:{self.global_steps}",
                metrics=cycle_metrics,
            ),
        )

    async def fit(self) -> None:
        """Main training-loop implementation on the driver event loop."""
        fit_started_at = time.perf_counter()
        save_freq = self.config.trainer.save_freq
        if save_freq and save_freq > 0 and save_freq % self.trainer_config.weight_publish_interval != 0:
            raise ValueError(
                "trainer.save_freq must align with trainer.weight_publish_interval so every "
                "trainer checkpoint has an exact committed Relay version: "
                f"save_freq={save_freq}, "
                f"weight_publish_interval={self.trainer_config.weight_publish_interval}"
            )
        logger.info("AsyncRLTrainer fit: start global_steps=%s", self.global_steps)
        await self.prepare()

        total_steps = self.total_training_steps
        if total_steps is not None and self.global_steps >= total_steps:
            logger.info(
                "Restored global_steps=%s already reached total_training_steps=%s; exiting fit loop",
                self.global_steps,
                total_steps,
            )
            await cast(Any, self.tracker.finalize.remote())
            return

        while True:
            self.global_steps += 1
            is_last_step = total_steps is not None and self.global_steps >= total_steps
            step_start_time = time.perf_counter()
            metrics: dict[str, Any] = {}
            metrics["timestamp/step"] = time.time()
            pending_checkpoint_metrics = self._pending_checkpoint_metrics
            if pending_checkpoint_metrics is not None:
                metrics.update(pending_checkpoint_metrics)
                self._pending_checkpoint_metrics = None
            timing_raw: dict[str, float] = {}
            logger.info(
                "AsyncRLTrainer step=%s start total_steps=%s is_last_step=%s",
                self.global_steps,
                total_steps,
                is_last_step,
            )

            # 1. Pull a fresh trajectory batch and its terminal group context.
            # Waits if not enough trajectories have arrived.
            logger.info(
                "AsyncRLTrainer step=%s sample: waiting for batch_size=%s",
                self.global_steps,
                self.train_batch_size,
            )
            with simple_timer("sample", timing_raw):
                sample_ref = cast(
                    Any,
                    self.experience_buffer_client.sample_remote(
                        batch_size=self.train_batch_size,
                    ),
                )
                batch, relay_wait_time_s = await self._await_sample_or_relay_failure(
                    sample_ref,
                    metrics,
                )
                partition_id = batch.partition_id
                assert partition_id, "sampled batch must carry a TransferQueue partition"
            logger.info(
                "AsyncRLTrainer step=%s sample: got %d keys partition=%s",
                self.global_steps,
                len(batch.keys),
                partition_id,
            )

            if self.colocate:
                with simple_timer("colocate_pause", timing_raw):
                    relay_wait_time_s += await self._wait_for_pending_relay_publish(metrics)
                    await cast(Any, self._colocate_rollout_manager.pause_for_training.remote())
                metrics["colocate/rotation"] = 1.0

            # 2. Materialization and old-log-prob do not mutate actor weights.
            # Keep their blocking worker/TQ calls off the control-plane event
            # loop so a pending publish can commit concurrently. Megatron's
            # worker fences its export collectives before old-log-prob inference.
            td = await asyncio.to_thread(self._materialize_actor_batch, batch, metrics)
            materialize_time_s = float(metrics["timing/materialize_batch"])
            prepare_actor_update_start = time.perf_counter()
            await self._prepare_actor_update(td, metrics)
            prepare_actor_update_time_s = time.perf_counter() - prepare_actor_update_start

            # TODO(async_rl): add ref-policy KL support. KL-in-reward needs
            # the relevant ref log-probs before trainer-side advantage.
            # TODO(async_rl): add critic support.

            # 3. Synchronize with the previous publish immediately before the
            # first operation that may mutate actor parameters.
            logger.info("AsyncRLTrainer step=%s wait_for_prev_relay: start", self.global_steps)
            relay_wait_time_s += await self._wait_for_pending_relay_publish(metrics)
            timing_raw["wait_for_prev_relay"] = relay_wait_time_s
            logger.info("AsyncRLTrainer step=%s wait_for_prev_relay: done", self.global_steps)

            # TODO(async_rl): add critic update support.

            # 4. Actor update.
            logger.info("AsyncRLTrainer step=%s update_actor: start", self.global_steps)
            update_actor_start = time.perf_counter()
            self._update_actor(td, metrics)
            # Keep the historical compute-work denominator comparable even
            # though materialization and old-log-prob now overlap the previous
            # publish. Relay residual wait remains outside this denominator.
            timing_raw["update_actor"] = (
                materialize_time_s + prepare_actor_update_time_s + (time.perf_counter() - update_actor_start)
            )
            logger.info("AsyncRLTrainer step=%s update_actor: done", self.global_steps)

            checkpoint_due = bool(save_freq and save_freq > 0 and (is_last_step or self.global_steps % save_freq == 0))

            # 5. Push new version to relay. A normal final step is not
            # published because no training rollout can consume it. A final
            # checkpoint boundary is published so the checkpoint never waits
            # for a Relay version that cannot exist.
            logger.info("AsyncRLTrainer step=%s schedule_relay_publish: start", self.global_steps)
            with simple_timer("schedule_relay_publish", timing_raw):
                metrics["trainer/weight_publish_interval"] = float(self.trainer_config.weight_publish_interval)
                if self._step_should_publish_weights(
                    self.global_steps,
                    total_steps,
                    checkpoint_due=checkpoint_due,
                ):
                    metrics["trainer/weight_publish_scheduled"] = 1.0
                    metrics["trainer/weight_publish_version"] = float(self.global_steps)
                    self._schedule_relay_publish(version=self.global_steps)
                else:
                    metrics["trainer/weight_publish_scheduled"] = 0.0
                    logger.info(
                        "Skipping weight push for step=%s: weight_publish_interval=%s total_steps=%s",
                        self.global_steps,
                        self.trainer_config.weight_publish_interval,
                        total_steps,
                    )
            logger.info("AsyncRLTrainer step=%s schedule_relay_publish: done", self.global_steps)

            # 6. Fence the Relay version needed by a trainer checkpoint.
            if checkpoint_due or self.colocate:
                relay_wait_time_s = await self._wait_for_pending_relay_publish(metrics)
                timing_raw["wait_for_current_relay"] = relay_wait_time_s

            # 7. Buffer cleanup releases the rollout-produced store records.
            with simple_timer("experience_buffer_remove", timing_raw):
                await self.experience_buffer_client.release(batch.keys)

            metrics, total_gpus = await self._commit_step_metrics(metrics, timing_raw, step_start_time=step_start_time)
            await self._save_checkpoint_and_resume_rollout(checkpoint_due=checkpoint_due, is_last_step=is_last_step)
            await self._record_training_cycle(
                metrics, step_start_time=step_start_time, fit_started_at=fit_started_at, total_gpus=total_gpus
            )

            if is_last_step:
                logger.info(f"Reached total_training_steps={total_steps}; exiting fit loop")
                await cast(Any, self.tracker.finalize.remote())
                return
