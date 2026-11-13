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
"""Trajectory-level scheduling over TransferQueue metadata.

TransferQueue owns payloads. ``ExperienceBufferServer`` tracks successful trajectory
records, waits for the raw ``uid`` terminal marker, and returns trajectory
keys to the trainer. Advantage estimation and payload materialization
remain trainer-owned.

For the current stage, terminal keys are assumed immutable while live and a
successful Store release is assumed not to reappear in later metadata lists.
There is intentionally no digest, tombstone, release fence, or retry ledger.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from verl.utils.transferqueue_utils import tq

from laminar.metrics import MetricSnapshot, MetricSource, cumulative_counter_specs
from laminar.train.experience_buffer.config import ExperienceBufferConfig
from laminar.train.experience_buffer.tags import (
    _as_padded_tensor,
    _finite_float,
    _tag_weight_version,
    trajectory_key_facts,
)
from laminar.utils.concurrency import log_exception_callback

logger = logging.getLogger(__name__)

# Buffer-owned lifecycle counters for commit / activate / sample / release / drop.
_BUFFER_COUNTER_KEYS = (
    "lifecycle/committed_trajectory_count",
    "lifecycle/committed_attention_tokens",
    "lifecycle/activated_trajectory_count",
    "lifecycle/sampled_trajectory_count",
    "lifecycle/released_trajectory_count",
    "eviction/after_commit_trajectory_count",
    "drop/trajectory_count",
    "sink/trajectory_clear_failure_count",
    "sink/prompt_terminal_clear_failure_count",
)


@dataclass
class SampledBatch:
    """TQ keys for one trainer step, plus prompt-group terminal context.

    ``keys`` are TransferQueue trajectory keys. ``partition_id`` is the TQ
    partition. ``group_context`` is keyed by framework ``uid`` and has no
    tensor payloads.
    """

    keys: list[str]
    partition_id: str = "train"
    group_context: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    trajectory_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)


class ExperienceBufferServer:
    """Lightweight trajectory metadata/control plane over the experience Store."""

    def __init__(self, config: ExperienceBufferConfig | None = None):
        config = config or ExperienceBufferConfig()

        self._config = config
        self.max_size = int(config.max_size)
        self.sampler = config.sampler
        self.store_namespace = config.store_namespace
        self.poll_interval = config.poll_interval

        # Activated, unconsumed trajectories stay in arrival order for FIFO
        # capacity eviction regardless of sampling order. Staged and sampled
        # trajectories are deliberately outside this trainer-backlog limit.
        self._ready: deque[str] = deque()
        self._tags: dict[str, dict[str, Any]] = {}

        # Trajectories wait for the raw uid terminal marker. A prompt group
        # can span multiple sampled batches, so its compact final-session
        # context lives until the last trajectory is released.
        self._prompt_group_keys: dict[str, list[str]] = defaultdict(list)
        self._prompt_group_context: dict[str, list[dict[str, Any]]] = {}
        self._prompt_terminals: dict[str, dict[str, Any]] = {}
        self._activated_prompt_groups: set[str] = set()
        self._activating_prompt_groups: set[str] = set()

        self._condition = asyncio.Condition()
        self._poll_task: asyncio.Task[None] | None = None
        self._metrics: dict[str, int] = {key: 0 for key in _BUFFER_COUNTER_KEYS}
        self._metric_source = MetricSource(cumulative_counter_specs("experience_buffer", _BUFFER_COUNTER_KEYS))

    async def add(self, items: dict[str, dict]) -> None:
        """Ingest completed trajectories; activate when their raw ``uid`` group closes."""
        self._validate_items(items)
        await self._add(items)

    async def _add(self, items: dict[str, dict] | None = None) -> None:
        activation_plans: list[tuple[str, list[str], list[dict[str, Any]], dict[str, Any]]] = []
        prompt_terminal_keys: list[str] = []
        evicted_keys: list[str] = []
        async with self._condition:
            if items is None:
                # A metadata snapshot must be admitted before release or eviction
                # can forget keys that were live when the Store read began.
                data = await tq.async_kv_list()
                items = dict(data.get(self.store_namespace, {})) if data else {}
                self._validate_items(items)
            prompt_uids_to_check: set[str] = set()

            for key, tag in items.items():
                status = str(tag.get("status", "unknown"))
                if status != "success":
                    continue
                success_tag = dict(tag)
                uid, session_id, output_index = trajectory_key_facts(key)
                tagged_uid = success_tag.get("uid")
                if tagged_uid is not None and str(tagged_uid) != uid:
                    raise RuntimeError(
                        "trajectory key uid conflicts with its tag: "
                        f"store_namespace={self.store_namespace!r} key={key!r} tag_uid={tagged_uid!r}"
                    )
                success_tag.update(uid=uid, session_id=session_id, output_index=output_index)
                if self._terminal_metadata_is_duplicate_locked(key, success_tag):
                    continue
                self._tags[key] = success_tag
                self._metrics["lifecycle/committed_trajectory_count"] += 1
                self._metrics["lifecycle/committed_attention_tokens"] += success_tag["seq_len"]
                self._prompt_group_keys[uid].append(key)
                prompt_uids_to_check.add(uid)

            for key, tag in items.items():
                status = str(tag.get("status", "unknown"))
                if status not in {"finished", "failure"}:
                    continue
                if "_" in key:
                    raise RuntimeError(
                        "prompt terminal key must be the raw uid without underscores: "
                        f"store_namespace={self.store_namespace!r} key={key!r}"
                    )
                previous = self._prompt_terminals.get(key)
                if previous is not None and previous != dict(tag):
                    raise RuntimeError(
                        "ExperienceBuffer observed conflicting prompt terminal metadata: "
                        f"store_namespace={self.store_namespace!r} uid={key!r}"
                    )
                self._prompt_terminals[key] = dict(tag)
                prompt_uids_to_check.add(key)

            for uid in sorted(prompt_uids_to_check):
                activation = self._prepare_prompt_group_activation_locked(uid)
                if activation is None:
                    continue
                ready_keys, final_records, terminal = activation
                if final_records:
                    activation_plans.append((uid, ready_keys, final_records, terminal))
                else:
                    prompt_terminal_keys.append(uid)

        try:
            loaded_contexts = [
                await self._load_prompt_group_context(uid, final_records)
                for uid, _ready_keys, final_records, _terminal in activation_plans
            ]
        except BaseException:
            async with self._condition:
                for uid, _ready_keys, _final_records, _terminal in activation_plans:
                    self._activating_prompt_groups.discard(uid)
            raise

        async with self._condition:
            ready_changed = False
            for (uid, ready_keys, _final_records, terminal), group_context in zip(
                activation_plans, loaded_contexts, strict=True
            ):
                if uid not in self._activating_prompt_groups:
                    continue
                if self._prompt_terminals.get(uid) != terminal or self._prompt_group_keys.get(uid) != ready_keys:
                    self._activating_prompt_groups.discard(uid)
                    continue
                self._prompt_group_context[uid] = group_context
                self._activating_prompt_groups.discard(uid)
                self._activated_prompt_groups.add(uid)
                self._ready.extend(ready_keys)
                self._metrics["lifecycle/activated_trajectory_count"] += len(ready_keys)
                prompt_terminal_keys.append(uid)
                ready_changed = True

            if len(self._ready) > self.max_size:
                eviction_count = len(self._ready) - self.max_size
                evicted_keys = list(self._ready)[:eviction_count]
                self._drop_trajectory_state_locked(evicted_keys)
                self._metrics["eviction/after_commit_trajectory_count"] += len(evicted_keys)
                self._metrics["drop/trajectory_count"] += len(evicted_keys)

            if ready_changed or evicted_keys:
                self._condition.notify_all()

            # Keep the synchronous Store clears in the same critical section
            # as metadata cleanup, so the next poll cannot see forgotten keys.
            if prompt_terminal_keys:
                prompt_terminal_keys = list(dict.fromkeys(prompt_terminal_keys))
                try:
                    tq.kv_clear(keys=prompt_terminal_keys, partition_id=self.store_namespace)
                except Exception:
                    self._metrics["sink/prompt_terminal_clear_failure_count"] += 1
                    raise
                self._cleanup_prompt_terminals_locked(prompt_terminal_keys)
            if evicted_keys:
                try:
                    tq.kv_clear(keys=evicted_keys, partition_id=self.store_namespace)
                except Exception:
                    self._metrics["sink/trajectory_clear_failure_count"] += 1
                    raise

    async def sample(self, batch_size: int) -> SampledBatch:
        """Sample exactly ``batch_size`` trajectories in configured order."""
        if batch_size <= 0:
            raise ValueError(f"sample batch_size must be positive, got {batch_size}")
        if batch_size > self.max_size:
            raise ValueError(f"sample batch_size {batch_size} exceeds buffer max_size {self.max_size}")

        async with self._condition:
            await self._condition.wait_for(lambda: len(self._ready) >= batch_size)
            chosen_keys = self._pop_ready_locked(batch_size=batch_size)
            self._metrics["lifecycle/sampled_trajectory_count"] += len(chosen_keys)

            group_ids = {self._tags[key]["uid"] for key in chosen_keys}
            group_context = {
                group_id: [dict(item) for item in self._prompt_group_context[group_id]]
                for group_id in sorted(group_ids)
            }
            return SampledBatch(
                keys=chosen_keys,
                partition_id=self.store_namespace,
                group_context=group_context,
                trajectory_metadata={key: dict(self._tags[key]) for key in chosen_keys},
            )

    async def remove(self, keys: list[str]) -> None:
        """Release sampled Store payloads, then forget their metadata."""
        ordered_keys = list(dict.fromkeys(keys))
        if not ordered_keys:
            return
        async with self._condition:
            missing = [key for key in ordered_keys if key not in self._tags]
            if missing:
                raise RuntimeError(f"ExperienceBuffer cannot release unknown trajectory keys: {missing!r}")

        # Keep metadata until TQ clear succeeds. No additional fencing is
        # needed: sampled keys have already left the ready queue.
        try:
            tq.kv_clear(keys=ordered_keys, partition_id=self.store_namespace)
        except Exception:
            async with self._condition:
                self._metrics["sink/trajectory_clear_failure_count"] += 1
            raise
        async with self._condition:
            self._metrics["lifecycle/released_trajectory_count"] += len(ordered_keys)
            self._drop_trajectory_state_locked(ordered_keys)

    def __len__(self) -> int:
        return len(self._ready)

    async def get_metrics(self) -> MetricSnapshot:
        """Return lifecycle counters, inventory and a snapshot timestamp.

        Commit counts first ingestion of successful trajectory metadata, before
        group activation. Attention tokens use the Store's ``seq_len`` (prompt
        + response, including tool context), without trainability filtering.
        ``activated = ready + sampled + evicted``.
        """
        async with self._condition:
            ready_count = len(self._ready)
            metrics: dict[str, int | float] = dict(self._metrics)
            metrics["timestamp/snapshot_s"] = time.time()
            metrics["inventory/ready_trajectory_count"] = ready_count
            metrics["conservation/activation_residual"] = (
                self._metrics["lifecycle/activated_trajectory_count"]
                - ready_count
                - self._metrics["lifecycle/sampled_trajectory_count"]
                - self._metrics["eviction/after_commit_trajectory_count"]
            )
            return self._metric_source.snapshot(metrics)

    async def start(self) -> None:
        """Start the background Store metadata poll task exactly once."""
        if self._poll_task is not None:
            return
        self._poll_task = asyncio.create_task(self._poll_from_store())
        self._poll_task.add_done_callback(log_exception_callback)

    async def _poll_from_store(self) -> None:
        """Poll terminal metadata; any Store/ingest error fails the actor."""
        try:
            while True:
                await self._add()
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("ExperienceBuffer poll task failed; exiting actor")
            self._fail_fast_job()
            raise

    def _fail_fast_job(self) -> None:
        """Kill the hosting Ray actor when polling dies; no-op in unit tests."""
        ray_mod = sys.modules.get("ray")
        if ray_mod is None or not ray_mod.is_initialized():
            return
        try:
            if ray_mod.get_runtime_context().get_actor_id() is None:
                return
            from ray.actor import exit_actor

            exit_actor()
        except Exception:  # noqa: BLE001
            logger.exception("ExperienceBuffer failed to exit Ray actor after poll crash")

    def _validate_items(self, items: Mapping[str, Mapping[str, Any]]) -> None:
        """Validate one metadata snapshot before mutating Buffer state."""
        for key, tag in items.items():
            status = str(tag.get("status", "unknown"))
            if status == "success":
                try:
                    trajectory_key_facts(key)
                except ValueError as exc:
                    raise RuntimeError(
                        "ExperienceBuffer rejected invalid trajectory key: "
                        f"store_namespace={self.store_namespace!r} key={key!r}: {exc}"
                    ) from exc
                seq_len = tag.get("seq_len")
                if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len < 0:
                    raise RuntimeError(
                        "ExperienceBuffer requires non-negative integer seq_len metadata: "
                        f"store_namespace={self.store_namespace!r} key={key!r} seq_len={seq_len!r}"
                    )
                continue
            if status in {"finished", "failure"} and "_" in key:
                raise RuntimeError(
                    "ExperienceBuffer prompt terminal key must be a raw uid without underscores: "
                    f"store_namespace={self.store_namespace!r} key={key!r}"
                )

    def _prepare_prompt_group_activation_locked(
        self, uid: str
    ) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]] | None:
        terminal = self._prompt_terminals.get(uid)
        if terminal is None:
            return None
        if uid in self._activated_prompt_groups:
            return [], [], dict(terminal)
        if uid in self._activating_prompt_groups:
            return None

        keys = list(self._prompt_group_keys.get(uid, ()))
        status = str(terminal["status"])
        if status == "failure":
            if keys:
                raise RuntimeError(
                    f"failed prompt terminal cannot close successful trajectories: uid={uid!r} keys={keys!r}"
                )
            self._activated_prompt_groups.add(uid)
            return [], [], dict(terminal)
        if not keys:
            raise RuntimeError(f"finished prompt terminal requires at least one successful trajectory: uid={uid!r}")

        final_key_by_session: dict[int, tuple[int, str]] = {}
        for key in keys:
            key_uid, session_id, output_index = trajectory_key_facts(key)
            if key_uid != uid:
                raise RuntimeError(f"prompt group contains a trajectory for another uid: uid={uid!r} key={key!r}")
            previous = final_key_by_session.get(session_id)
            if previous is None or output_index > previous[0]:
                final_key_by_session[session_id] = (output_index, key)

        final_records = sorted(
            (
                {"key": key, "session_id": session_id, "output_index": output_index}
                for session_id, (output_index, key) in final_key_by_session.items()
            ),
            key=lambda item: (item["session_id"], item["output_index"], item["key"]),
        )
        self._activating_prompt_groups.add(uid)
        return keys, final_records, dict(terminal)

    async def _load_prompt_group_context(self, uid: str, final_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        final_keys = [str(record["key"]) for record in final_records]
        reward_data = await tq.async_kv_batch_get(
            keys=final_keys,
            partition_id=self.store_namespace,
            select_fields=["rm_scores"],
        )
        if "rm_scores" not in reward_data:
            raise RuntimeError(f"experience store prompt group is missing rm_scores: uid={uid!r}")
        rm_scores = _as_padded_tensor(reward_data["rm_scores"])
        if rm_scores.ndim == 1 and len(final_keys) == 1:
            rm_scores = rm_scores.unsqueeze(0)
        if rm_scores.ndim < 2 or rm_scores.shape[0] != len(final_keys):
            raise RuntimeError(
                "experience store rm_scores must align with final successful sessions: "
                f"uid={uid!r} shape={tuple(rm_scores.shape)} sessions={len(final_keys)}"
            )
        rewards = rm_scores.reshape(len(final_keys), -1).sum(dim=1).tolist()
        for record, reward in zip(final_records, rewards, strict=True):
            reward_score = _finite_float(reward)
            if reward_score is None:
                raise RuntimeError(
                    "experience store prompt group has a non-finite reward: "
                    f"uid={uid!r} key={record['key']!r} reward={reward!r}"
                )
            record["reward_score"] = reward_score

        return final_records

    def _terminal_metadata_is_duplicate_locked(self, key: str, tag: Mapping[str, Any]) -> bool:
        previous = self._tags.get(key)
        if previous is None:
            previous = self._prompt_terminals.get(key)
        if previous is None:
            return False
        if previous != dict(tag):
            raise RuntimeError(
                "ExperienceBuffer observed conflicting immutable terminal metadata for one active Store key: "
                f"store_namespace={self.store_namespace!r} key={key!r} "
                f"previous_status={previous.get('status')!r} current_status={tag.get('status')!r}"
            )
        return True

    def _drop_trajectory_state_locked(self, keys: Iterable[str]) -> None:
        keys_set = set(keys)
        if not keys_set:
            return
        self._ready = deque(key for key in self._ready if key not in keys_set)
        affected_uids = set()
        for key in keys_set:
            tag = self._tags.pop(key, None)
            if tag is not None:
                affected_uids.add(tag["uid"])
        for uid in affected_uids:
            group_keys = self._prompt_group_keys.get(uid)
            if group_keys is None:
                continue
            remaining = [group_key for group_key in group_keys if group_key not in keys_set]
            if remaining:
                self._prompt_group_keys[uid] = remaining
                continue
            self._prompt_group_keys.pop(uid, None)
            self._prompt_group_context.pop(uid, None)
            if uid not in self._prompt_terminals:
                self._activated_prompt_groups.discard(uid)

    def _cleanup_prompt_terminals_locked(self, keys: Iterable[str]) -> None:
        for uid in keys:
            self._prompt_terminals.pop(uid, None)
            if not self._prompt_group_keys.get(uid):
                self._activated_prompt_groups.discard(uid)

    def _pop_ready_locked(self, *, batch_size: int) -> list[str]:
        if self.sampler != "version_lifo":
            pop = self._ready.pop if self.sampler == "lifo" else self._ready.popleft
            return [pop() for _ in range(batch_size)]
        indexed_keys = list(enumerate(self._ready))
        selected = self._select_version_lifo_ready_locked(indexed_keys, batch_size)
        selected_indices = {index for index, _key in selected}
        self._ready = deque(key for index, key in indexed_keys if index not in selected_indices)
        return [key for _index, key in selected]

    def _select_version_lifo_ready_locked(
        self,
        indexed_keys: list[tuple[int, str]],
        batch_size: int,
    ) -> list[tuple[int, str]]:
        def rank(item: tuple[int, str]) -> tuple[int, int, int]:
            index, key = item
            version = _tag_weight_version(self._tags.get(key, {}))
            if version is None:
                return (0, -1, index)
            return (1, version, index)

        return sorted(indexed_keys, key=rank, reverse=True)[:batch_size]
