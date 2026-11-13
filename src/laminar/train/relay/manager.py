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
"""Async weight-sync relay: actor → CPU-resident RDMA buffer → rollouts.

The relay decouples the trainer from rollout servers on the weight path:

* :class:`RelayWorker` runs one per node, hosts a :class:`UCXTensorServer` on
  pinned CPU memory, and serves both as an actor ingress (full tensors, or
  shards written as byte ranges of them) and as a chain-broadcast hop.
* :class:`RelayManager` is the Ray driver-side wrapper: it pins one
  ``RelayWorker`` per rollout node and owns the publish/commit protocol used
  to expose complete trainer versions to rollout.
* :class:`AsyncRLvLLMColocateWorkerExtension` is the rollout-side worker
  extension that mmap()'s the shared CPU buffer and loads weights directly
  from it (no extra IPC hop).

The async-RL recipe spawns one ``RelayManager`` on the driver; each
publishing ``ActorRolloutRefWorker`` rank attaches to the chain head
(``Relay 0``) via ``connect_to_relay_server`` and pushes its full tensors or
shards with ``send_weights``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

from laminar.metrics import MetricSnapshot, MetricSource, cumulative_counter_specs
from laminar.train.relay.client import RelayBroadcastStats, application_goodput_bytes_per_s
from laminar.train.relay.storage import make_shared_tensor_from_meta
from laminar.train.relay.transport import (
    EMPTY_VERSION,
    TensorChunkSender,
    TensorStorageMeta,
    TensorTransferHeader,
    TensorTransferMode,
    UCXTensorServer,
    VersionedTensor,
    complete_tensor_transfer,
    create_endpoint,
    relay_backend,
    send_obj,
    send_tensor,
)
from laminar.utils.concurrency import ReadWriteLock, await_shielded_completion, gather_cancel_on_error
from laminar.utils.tensor import TensorMeta

if TYPE_CHECKING:
    from ray.actor import ActorHandle
    from vllm.v1.worker.worker_base import WorkerBase

logger = logging.getLogger(__name__)

_RELAY_CHAIN_SEND_WINDOW_SIZE = 16
_RELAY_COUNTER_KEYS = (
    "publish/started",
    "publish/committed",
    "broadcast/started",
    "broadcast/completed",
    "broadcast/failed",
    "broadcast/latency_s",
)

# Fail-closed deadline for one tensor of a weight version to land on Relay 0.
# The ingress wait streams tensors published by the actor worker-group; if a
# rank's shard never arrives (e.g. the publisher stalls before/at the initial
# version=0 broadcast), the wait would otherwise loop forever, occupying GPUs
# while only logging warnings. A generous ceiling (far above a healthy
# publish) turns an unbounded hang into a fast, restartable failure without
# killing a legitimately slow large-MoE publish.
_RELAY_INGRESS_TENSOR_WAIT_TIMEOUT_S = 1800.0


@dataclass
class _PublishState:
    """Manager-owned state for the single in-flight publish."""

    broadcast_task: asyncio.Task[RelayBroadcastStats] | None = None


@dataclass(frozen=True)
class _RelayHopStats:
    """Result needed by the manager from one fixed direct-send chain hop."""

    logical_payload_bytes: int
    elapsed_s: float


class _ThrottledWaitLogger:
    """Log expected async waits only if they stay stuck long enough."""

    def __init__(
        self,
        *,
        owner: str,
        warn_after_s: float = 5.0,
        repeat_s: float = 30.0,
        clock: Any = time.perf_counter,
    ) -> None:
        self.owner = owner
        self.warn_after_s = warn_after_s
        self.repeat_s = repeat_s
        self.clock = clock
        self.start_time = clock()
        self.last_warn_time: float | None = None
        self.num_checks = 0

    def maybe_warn(self, message: str) -> None:
        self.num_checks += 1
        now = self.clock()
        elapsed = now - self.start_time
        if self.num_checks == 1:
            logger.debug("%s waiting: %s", self.owner, message)
        if elapsed < self.warn_after_s:
            return
        if self.last_warn_time is not None and now - self.last_warn_time < self.repeat_s:
            return
        self.last_warn_time = now
        logger.warning("%s still waiting after %.1fs (%d checks): %s", self.owner, elapsed, self.num_checks, message)


class RelayManager:
    """Ray-driver-side façade for the :class:`RelayWorker` chain.

    Hides per-worker RPC plumbing from end users. Weight versions are strictly
    increasing; only the active version can become loadable after both the
    actor-send and Relay-broadcast barriers complete.
    """

    def __init__(self, node_ids: list[str], *, num_cpus: int = 1) -> None:
        if isinstance(num_cpus, bool) or not isinstance(num_cpus, int) or num_cpus < 1:
            raise ValueError("relay.num_cpus must be a positive integer")
        worker_env = {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}
        if num_cpus > 1:
            worker_env.update(OMP_NUM_THREADS=str(num_cpus), MKL_NUM_THREADS=str(num_cpus))
        worker_cls = ray.remote(RelayWorker)
        self.workers = [
            worker_cls.options(
                name=f"relay_{i}",
                # Match the explicit CPU copy thread budget to Ray resources.
                num_cpus=num_cpus,
                # pin_memory needs the CUDA context, so Ray must preserve the
                # node's CUDA visibility. UCX policy is inherited job-wide.
                runtime_env={"env_vars": worker_env},
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
            ).remote(name=f"Relay {i}")
            for i, node_id in enumerate(node_ids)
        ]
        self.node_id2worker = {node_id: worker for node_id, worker in zip(node_ids, self.workers, strict=True)}
        # Trainer-side version/intent. This advances before transport so
        # rollout dispatch stops sending trajectories to older behavior versions.
        self.trainer_version: int = EMPTY_VERSION
        self._active_publish: _PublishState | None = None
        # RelayManager is an async Ray actor, so its methods interleave at every
        # await. Serialize begin/commit transitions across their worker barriers
        # so exactly one terminal outcome can win for a publish.
        self._publish_transition_lock = asyncio.Lock()
        # Rollout-side committed version. It advances only after both barriers.
        self.loadable_version: int = EMPTY_VERSION
        self.full_state_meta_dict: dict[str, TensorMeta] = {}
        self._metrics: dict[str, int | float] = {
            "publish/started": 0,
            "publish/committed": 0,
            "broadcast/started": 0,
            "broadcast/completed": 0,
            "broadcast/failed": 0,
            "broadcast/latency_s": 0.0,
            "broadcast/latency_s/last": 0.0,
            "broadcast/latency_s/max": 0.0,
            "broadcast/tensor_count/last": 0,
            "broadcast/logical_payload_bytes/last": 0,
            "broadcast/end_to_end_goodput_bytes_per_s/last": 0.0,
            "broadcast/bottleneck_hop_index/last": -1,
        }
        self._metric_source = MetricSource(cumulative_counter_specs("relay", _RELAY_COUNTER_KEYS))
        # Wire up the chain topology: each ``src`` connects to the next ``dst``.
        # We do this eagerly (``ray.get``) so a hop-init failure crashes startup
        # rather than the first broadcast.
        ray.get(
            [
                src_worker.connect_to_dst_worker.remote(
                    dst_worker,
                    hop_index=hop_index,
                    dst_name=f"Relay {hop_index + 1}",
                )
                for hop_index, (src_worker, dst_worker) in enumerate(
                    zip(self.chain_src_workers, self.workers[1:], strict=True)
                )
            ]
        )

    # --- Init / topology ---

    async def init_full_state_dict_in_shm(self, full_state_meta_dict: dict[str, TensorMeta]) -> None:
        """Pre-allocate the full-state-dict shared-memory backing on every worker."""
        self.full_state_meta_dict = full_state_meta_dict
        await asyncio.gather(
            *[
                worker.init_full_state_dict_in_shm.remote(full_state_meta_dict=self.full_state_meta_dict)
                for worker in self.workers
            ]
        )

    @property
    def master_worker(self) -> ActorHandle:
        return self.workers[0]

    @property
    def chain_src_workers(self) -> list[ActorHandle]:
        return self.workers[:-1]

    async def get_master_server_ip_port(self) -> tuple[str, int]:
        return await self.master_worker.get_server_ip_port.remote()

    # --- Version queries ---

    def get_trainer_version(self) -> int:
        """Return the trainer-side target/intent weight version."""
        return self.trainer_version

    def get_loadable_version(self) -> int:
        """Return the newest version committed after both publish barriers."""
        return self.loadable_version

    # --- Publish: begin -> broadcast -> commit ---

    async def begin_publish(self, version: int) -> None:
        """Fence the single Relay slot for a new, strictly newer version.

        Publish failure is fatal for the job. There is no abort/drain/retry:
        leave the in-flight fence and version intent as-is and raise so the
        trainer ends the run. Retry / fault tolerance can be reconsidered only
        after real publish failures show up in production.
        """
        async with self._publish_transition_lock:
            self._validate_can_begin_publish(version=version)
            self.trainer_version = version
            self._active_publish = _PublishState()
            results = await asyncio.gather(
                *[worker.begin_publish.remote(version=version) for worker in self.workers],
                return_exceptions=True,
            )
            errors = [result for result in results if isinstance(result, BaseException)]
            if errors:
                raise RuntimeError(f"Failed to begin Relay publish version={version}") from errors[0]
            self._metrics["publish/started"] += 1

    async def broadcast(self, version: int) -> RelayBroadcastStats:
        """Stage a version through the full Relay chain.

        Broadcast failure is fatal for the job. We intentionally await the full
        chain so the trainer publish path receives the original exception.
        """
        async with self._publish_transition_lock:
            publish = self._require_active_publish(version=version)
            if publish.broadcast_task is not None:
                raise RuntimeError(f"Relay broadcast already started: version={version}")
            self._metrics["broadcast/started"] += 1
            publish.broadcast_task = asyncio.create_task(
                self._broadcast(version=version, publish=publish),
                name=f"relay.broadcast:{version}",
            )
        try:
            return await publish.broadcast_task
        except asyncio.CancelledError as exc:
            raise RuntimeError(f"Relay broadcast aborted: version={version}") from exc

    async def _broadcast(
        self,
        *,
        version: int,
        publish: _PublishState,
    ) -> RelayBroadcastStats:
        """Chain-broadcast one weight version as the actor writes it into Relay 0."""
        start_time = time.perf_counter()
        tensor_count, logical_payload_bytes = self._full_state_dict_stats()
        try:
            # Actor ingress overlaps early chain hops; neither waits on the other.
            ingress_ref = self.master_worker.wait_for_full_state_dict.remote(version=version)
            chain_refs = [
                src_worker.send_full_state_dict_to_dst_worker.remote(
                    wait_for_version=version,
                )
                for src_worker in self.chain_src_workers
            ]
            logger.info(
                "Broadcasting version=%s to all workers: tensors=%s bytes=%s workers=%s",
                version,
                tensor_count,
                logical_payload_bytes,
                len(self.workers),
            )
            results = await asyncio.gather(*chain_refs, ingress_ref)
            hop_stats = results[: len(chain_refs)]
            broadcast_stats = self._broadcast_stats_from_hops(
                version=version,
                publish=publish,
                hop_stats=hop_stats,
                tensor_count=tensor_count,
                logical_payload_bytes=logical_payload_bytes,
                elapsed_s=time.perf_counter() - start_time,
            )
        except asyncio.CancelledError:
            raise
        except BaseException:
            elapsed_s = time.perf_counter() - start_time
            self._metrics["broadcast/failed"] += 1
            self._metrics["broadcast/latency_s/last"] = elapsed_s
            self._metrics["broadcast/latency_s/max"] = max(
                float(self._metrics["broadcast/latency_s/max"]),
                elapsed_s,
            )
            logger.exception("Relay broadcast failed: version=%s", version)
            raise

        elapsed_s = broadcast_stats.relay_broadcast_time_s
        self._metrics["broadcast/latency_s/last"] = elapsed_s
        self._metrics["broadcast/latency_s/max"] = max(
            float(self._metrics["broadcast/latency_s/max"]),
            elapsed_s,
        )
        self._metrics["broadcast/tensor_count/last"] = broadcast_stats.tensor_count
        self._metrics["broadcast/logical_payload_bytes/last"] = broadcast_stats.logical_payload_bytes
        self._metrics["broadcast/end_to_end_goodput_bytes_per_s/last"] = broadcast_stats.end_to_end_goodput_bytes_per_s
        self._metrics["broadcast/bottleneck_hop_index/last"] = broadcast_stats.bottleneck_hop_index
        self._metrics["broadcast/completed"] += 1
        self._metrics["broadcast/latency_s"] += elapsed_s
        logger.info(
            "Broadcasted version=%s to all workers: tensors=%s logical_payload_bytes=%s "
            "elapsed=%.3fs application_goodput=%.3fGiB/s",
            version,
            broadcast_stats.tensor_count,
            broadcast_stats.logical_payload_bytes,
            elapsed_s,
            broadcast_stats.end_to_end_goodput_bytes_per_s / 1024**3,
        )
        return broadcast_stats

    async def complete_publish(self, version: int) -> None:
        """Commit a version after actor-send and broadcast barriers."""
        async with self._publish_transition_lock:
            self._require_successful_broadcast(version=version)
            worker_results = await asyncio.gather(
                *[worker.complete_publish.remote(version=version) for worker in self.workers],
                return_exceptions=True,
            )
            errors = [result for result in worker_results if isinstance(result, BaseException)]
            if errors:
                raise RuntimeError(f"Failed to complete Relay publish version={version}") from errors[0]
            self.loadable_version = version
            self._active_publish = None
            self._metrics["publish/committed"] += 1

    # --- Rollout load ---

    async def acquire_rollout_version(
        self,
        node_ids: Iterable[str],
        *,
        minimum_version: int,
    ) -> int:
        """Return and read-lock the newest committed rollout version when needed.

        Relay storage is a single in-place slot. Once a newer publish starts,
        the old bytes are no longer safe to reload even though its version was
        previously committed. Readers therefore wait while a publish is active,
        require ``trainer_version == loadable_version``, and validate the exact
        version after taking each worker's publish read lock. A failed publish
        leaves the slot fenced and ends the job; there is no abort/retry path.
        """
        if type(minimum_version) is not int or minimum_version < 0:
            raise ValueError(f"minimum_version must be a non-negative integer, got {minimum_version!r}")
        relay_workers = self._unique_workers_for_nodes(node_ids)

        while True:
            version = await self._wait_for_acquireable_rollout_version(minimum_version=minimum_version)
            if await self._try_acquire_consistent_rollout_read(relay_workers, version=version):
                return version
            await asyncio.sleep(RelayWorker.DEFAULT_INTERVAL_S)

    async def release_full_read(self, node_ids: Iterable[str]) -> None:
        """Release the read-locks acquired by ``acquire_rollout_version``."""
        relay_workers = self._unique_workers_for_nodes(node_ids)
        results = await asyncio.gather(
            *[relay_worker.release_full_read.remote() for relay_worker in relay_workers],
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise RuntimeError("Failed to release one or more Relay full-read fences") from errors[0]

    # --- Metrics ---

    def get_metrics(self) -> MetricSnapshot:
        metrics = dict(self._metrics)
        metrics["trainer_version"] = self.trainer_version
        metrics["loadable_version"] = self.loadable_version
        publish = self._active_publish
        broadcast_task = None if publish is None else publish.broadcast_task
        metrics["publish/active_version"] = EMPTY_VERSION if publish is None else self.trainer_version
        metrics["publish/broadcast_done"] = int(
            broadcast_task is not None
            and broadcast_task.done()
            and not broadcast_task.cancelled()
            and broadcast_task.exception() is None
        )
        metrics["broadcast/inflight"] = 1 if broadcast_task is not None and not broadcast_task.done() else 0
        metrics["broadcast/tensor_count"] = len(self.full_state_meta_dict)
        metrics["broadcast/logical_payload_bytes"] = self._full_state_dict_stats()[1]
        return self._metric_source.snapshot(metrics)

    # --- Private helpers ---

    def _validate_can_begin_publish(self, *, version: int) -> None:
        if self._active_publish is not None:
            raise RuntimeError(
                f"Cannot overlap Relay publishes: active_version={self.trainer_version} next_version={version}"
            )
        if self.trainer_version != self.loadable_version:
            raise RuntimeError(
                "Cannot publish after a failed Relay publish: "
                f"trainer_version={self.trainer_version} loadable_version={self.loadable_version}"
            )
        if version <= self.trainer_version:
            raise RuntimeError(
                "Relay publish versions must be strictly increasing: "
                f"version={version} trainer_version={self.trainer_version}"
            )

    def _require_active_publish(self, *, version: int) -> _PublishState:
        publish = self._active_publish
        if publish is None or self.trainer_version != version:
            raise RuntimeError(
                "Relay publish is not active: "
                f"version={version} active_version={self.trainer_version if publish is not None else None}"
            )
        return publish

    def _require_successful_broadcast(self, *, version: int) -> None:
        publish = self._require_active_publish(version=version)
        broadcast_task = publish.broadcast_task
        if broadcast_task is None or not broadcast_task.done() or broadcast_task.cancelled():
            raise RuntimeError(f"Cannot commit an incompletely broadcast Relay version: version={version}")
        broadcast_error = broadcast_task.exception()
        if broadcast_error is not None:
            raise RuntimeError(f"Cannot commit a failed Relay broadcast: version={version}") from broadcast_error

    def _broadcast_stats_from_hops(
        self,
        *,
        version: int,
        publish: _PublishState,
        hop_stats: list[_RelayHopStats],
        tensor_count: int,
        logical_payload_bytes: int,
        elapsed_s: float,
    ) -> RelayBroadcastStats:
        """Validate hop coverage/payload and build the typed broadcast result."""
        expected_hop_count = len(self.chain_src_workers)
        if len(hop_stats) != expected_hop_count:
            raise RuntimeError(
                "Relay broadcast hop coverage mismatch: "
                f"version={version} expected={expected_hop_count} observed={len(hop_stats)}"
            )
        for hop_index, hop_stat in enumerate(hop_stats):
            if hop_stat.logical_payload_bytes != logical_payload_bytes:
                raise RuntimeError(
                    "Relay broadcast hop logical payload mismatch: "
                    f"version={version} hop_index={hop_index} "
                    f"expected={logical_payload_bytes} observed={hop_stat.logical_payload_bytes}"
                )
            if logical_payload_bytes > 0 and hop_stat.elapsed_s <= 0.0:
                raise RuntimeError(
                    f"Relay broadcast hop {hop_index} elapsed time must be positive for a non-empty payload, "
                    f"got {hop_stat.elapsed_s!r}"
                )
        bottleneck_hop_index = (
            max(range(len(hop_stats)), key=lambda index: hop_stats[index].elapsed_s) if hop_stats else -1
        )
        if self._active_publish is not publish or self.trainer_version != version:
            raise RuntimeError(
                "Relay broadcast completed for a stale version: "
                f"version={version} active_version="
                f"{self.trainer_version if self._active_publish is not None else None}"
            )
        return RelayBroadcastStats(
            relay_broadcast_time_s=elapsed_s,
            logical_payload_bytes=logical_payload_bytes,
            tensor_count=tensor_count,
            hop_count=expected_hop_count,
            bottleneck_hop_index=bottleneck_hop_index,
            end_to_end_goodput_bytes_per_s=application_goodput_bytes_per_s(
                logical_payload_bytes=logical_payload_bytes,
                elapsed_s=elapsed_s,
                boundary="Relay broadcast end-to-end",
            ),
        )

    def _unique_workers_for_nodes(self, node_ids: Iterable[str]) -> list[ActorHandle]:
        relay_workers = list(dict.fromkeys(self.node_id2worker[node_id] for node_id in node_ids))
        assert relay_workers, f"There are no relay workers running in {node_ids=}"
        return relay_workers

    async def _wait_for_acquireable_rollout_version(self, *, minimum_version: int) -> int:
        """Wait until a committed, non-poisoned version meets ``minimum_version``."""
        while True:
            if self._active_publish is not None:
                await asyncio.sleep(1.0)
                continue
            # A failed publish poisons the in-place slot. Fail instead of
            # waiting for a retry that this runtime intentionally forbids.
            if self.trainer_version != self.loadable_version:
                raise RuntimeError(
                    "Relay publish failed before commit: "
                    f"trainer_version={self.trainer_version} loadable_version={self.loadable_version}"
                )
            version = self.loadable_version
            if version < minimum_version:
                await asyncio.sleep(1.0)
                continue
            return version

    async def _try_acquire_consistent_rollout_read(
        self,
        relay_workers: list[ActorHandle],
        *,
        version: int,
    ) -> bool:
        """Acquire a consistent read fence, or release a raced attempt and return False."""
        acquire_results, cancellation = await await_shielded_completion(
            asyncio.gather(
                *[worker.acquire_full_read_if_publish.remote(version=version) for worker in relay_workers],
                return_exceptions=True,
            )
        )
        acquired = [result is True for result in acquire_results]
        acquire_errors = [result for result in acquire_results if isinstance(result, BaseException)]
        state_unchanged = (
            self._active_publish is None and self.trainer_version == version and self.loadable_version == version
        )
        if cancellation is None and not acquire_errors and all(acquired) and state_unchanged:
            return True

        # Release any partial fence before retrying so the next publish is not stalled.
        release_results, release_cancellation = await await_shielded_completion(
            asyncio.gather(
                *[
                    worker.release_full_read.remote()
                    for worker, did_acquire in zip(relay_workers, acquired, strict=True)
                    if did_acquire
                ],
                return_exceptions=True,
            )
        )
        cancellation = cancellation or release_cancellation
        release_errors = [result for result in release_results if isinstance(result, BaseException)]
        if release_errors:
            # A partial fence whose release failed is unsafe to retry, including
            # when the caller was cancelled while the rollback was in flight.
            raise RuntimeError(f"Failed to release a partial Relay read fence: version={version}") from (
                release_errors[0]
            )
        if cancellation is not None:
            raise cancellation
        if acquire_errors:
            raise RuntimeError(f"Failed to acquire a consistent Relay read fence: version={version}") from (
                acquire_errors[0]
            )
        return False

    def _full_state_dict_stats(self) -> tuple[int, int]:
        return len(self.full_state_meta_dict), sum(meta.nbytes for meta in self.full_state_meta_dict.values())


class RelayWorker:
    """Per-node relay: UCX server + chain-broadcast forwarder.

    Each worker hosts a :class:`UCXTensorServer` on its node's pinned CPU
    memory. Actor publishers write the full ``key`` tensors directly: a
    full-state publisher sends whole tensors, sharded publishers send each
    shard as a byte range of them. The first worker in the chain is the
    ingress site; subsequent workers pull from the previous hop instead.
    """

    DEFAULT_INTERVAL_S: float = 1e-3

    def __init__(self, name: str) -> None:
        self.name = name
        self.server = UCXTensorServer(buffer_kwargs={"device": torch.device("cpu"), "pin_memory": True})

        # A publish holds the write side from begin through successful commit;
        # failure leaves it fenced, with no abort/retry path. Rollout loads hold
        # the read side until vLLM has copied every tensor, so an uncommitted
        # single-slot overwrite can never escape.
        self._publish_lock = ReadWriteLock()
        self._active_version: int | None = None
        # Preserve the publisher's export order on every Relay hop. The same
        # metadata dict is broadcast to all workers, so insertion order is
        # deterministic and lets each hop forward tensors as they arrive.
        self.full_tensor_keys: list[str] = []

    # --- Init ---

    def init_full_state_dict_in_shm(self, full_state_meta_dict: dict[str, TensorMeta]) -> None:
        """Materialize the shared buffer once before the first publish."""
        self.full_tensor_keys = list(full_state_meta_dict)

        shared_full_state = {
            k: VersionedTensor(
                tensor=make_shared_tensor_from_meta(meta, key=k),
            )
            for k, meta in full_state_meta_dict.items()
        }
        self.server.buffer.update(shared_full_state)

    # --- Publish fence: begin -> commit ---

    async def begin_publish(self, *, version: int) -> None:
        """Fence rollout reads for one manager-owned weight version."""
        await self._publish_lock.acquire_write()
        if self._active_version is not None:
            self._publish_lock.release_write()
            raise RuntimeError(f"[{self.name}] cannot overlap publishes: active={self._active_version} next={version}")
        self._active_version = version

    def complete_publish(self, *, version: int) -> None:
        """Release the publish fence after this worker received the full version."""
        self._require_active_publish(version=version)
        for key in self.full_tensor_keys:
            entry = self.server.buffer[key]
            if entry.version != version:
                raise RuntimeError(
                    f"[{self.name}] cannot commit incomplete tensor {key!r}: "
                    f"expected_version={version} observed={entry.version}"
                )
        self._active_version = None
        self._publish_lock.release_write()

    # --- Endpoint ---

    async def get_server_ip_port(self) -> tuple[str, int]:
        """For ``create_endpoint(addr, port)``.

        NOTE: ``lf.ip`` is usually ``"::"`` or ``"0.0.0.0"``.
        """
        listener = await self.server.wait_until_ready(self.DEFAULT_INTERVAL_S)
        return (self.server.ip, listener.port)

    # --- Pipeline transport ---

    async def wait_for_full_state_dict(self, version: int) -> None:
        """Wait until the actor publishers have written every full tensor of ``version``.

        Called only on the head of the chain. Publishers write straight into
        the full tensors (sharded publishers by byte range), so there is
        nothing to consolidate; subsequent hops receive the same tensors via
        ``send_full_state_dict_to_dst_worker``.
        """
        start_time = time.perf_counter()
        self._require_active_publish(version=version)
        logger.info("[%s] [Pipeline: Actor -> Relay 0 CPU] Enter: version=%s", self.name, version)
        wait_logger = _ThrottledWaitLogger(
            owner=f"[{self.name}] wait for full state dict version={version}",
            warn_after_s=5.0,
            repeat_s=30.0,
        )
        for i, full_tensor_key in enumerate(self.full_tensor_keys):
            await self._wait_for_tensor(
                full_tensor_key=full_tensor_key,
                tensor_index=i,
                version=version,
                wait_logger=wait_logger,
            )
        logger.info(
            "[%s] [Pipeline: Actor -> Relay 0 CPU] Exit: version=%s, latency=%.3f",
            self.name,
            version,
            time.perf_counter() - start_time,
        )

    async def connect_to_dst_worker(
        self,
        dst_worker: ActorHandle[RelayWorker],
        *,
        hop_index: int,
        dst_name: str,
    ) -> None:
        """Probe one chain hop and retain the destination address."""
        ip, port = await dst_worker.get_server_ip_port.remote()
        endpoint = await create_endpoint(ip, port)
        await endpoint.close()
        self.dst_worker = dst_worker
        self.dst_server_address = (ip, port)
        logger.info(
            "Relay hop connected: hop_index=%s source_name=%s destination_name=%s destination=%s:%s",
            hop_index,
            self.name,
            dst_name,
            ip,
            port,
        )

    async def send_full_state_dict_to_dst_worker(
        self,
        wait_for_version: int,
    ) -> _RelayHopStats:
        """Forward one exact version through a bounded send window."""
        self._require_active_publish(version=wait_for_version)
        start_time = time.perf_counter()
        logical_payload_bytes = self._logical_payload_bytes()
        logger.info(
            "[%s] [Pipeline: %s CPU -> Next CPU] Enter: version=%s tensors=%s bytes=%s",
            self.name,
            self.name,
            wait_for_version,
            len(self.full_tensor_keys),
            logical_payload_bytes,
        )
        wait_logger = _ThrottledWaitLogger(
            owner=f"[{self.name}] send full state dict version={wait_for_version}",
            warn_after_s=5.0,
            repeat_s=30.0,
        )
        # Delay endpoint creation until the first tensor is ready to avoid UCXX idle timeout.
        await self._wait_for_first_tensor_version(version=wait_for_version, wait_logger=wait_logger)
        await self._send_tagged_windows(
            version=wait_for_version,
            logical_payload_bytes=logical_payload_bytes,
            wait_logger=wait_logger,
        )
        elapsed_s = time.perf_counter() - start_time
        logger.info(
            "[%s] [Pipeline: %s CPU -> Next CPU] Exit: version=%s tensors=%s bytes=%s elapsed=%.3fs",
            self.name,
            self.name,
            wait_for_version,
            len(self.full_tensor_keys),
            logical_payload_bytes,
            elapsed_s,
        )
        return _RelayHopStats(logical_payload_bytes=logical_payload_bytes, elapsed_s=elapsed_s)

    # --- Rollout load ---

    def acquire_full_read_if_publish(self, *, version: int) -> bool:
        """Read-lock the committed slot immediately, or decline a busy publish fence.

        The manager may already hold reads on other workers. Waiting here for a
        writer would prevent it from releasing those partial reads, while that
        writer's publish barrier waits for the same workers.
        """
        if not self._publish_lock.try_acquire_read():
            return False
        keep_read_lock = False
        try:
            keep_read_lock = self._full_publish_matches(version=version)
            return keep_read_lock
        finally:
            if not keep_read_lock:
                self._publish_lock.release_read()

    def release_full_read(self) -> None:
        """Release one slot-wide read acquired by the manager."""
        self._publish_lock.release_read()

    # --- Private helpers ---

    def _require_active_publish(self, *, version: int) -> None:
        if self._active_version != version:
            raise RuntimeError(
                f"[{self.name}] stale publish version: expected={self._active_version} observed={version}"
            )

    async def _wait_for_tensor(
        self,
        *,
        full_tensor_key: str,
        tensor_index: int,
        version: int,
        wait_logger: _ThrottledWaitLogger,
    ) -> None:
        n = len(self.full_tensor_keys)
        wait_start = time.perf_counter()
        tensor_entry = self.server.buffer[full_tensor_key]
        while tensor_entry.version < version:
            self._require_active_publish(version=version)
            elapsed = time.perf_counter() - wait_start
            if elapsed > _RELAY_INGRESS_TENSOR_WAIT_TIMEOUT_S:
                raise RuntimeError(
                    f"[{self.name}] ingress tensor wait exceeded deadline: "
                    f"key={full_tensor_key!r} tensor_index={tensor_index}/{n} expected_version={version} "
                    f"observed_version={tensor_entry.version} elapsed_s={elapsed:.1f} "
                    f"timeout_s={_RELAY_INGRESS_TENSOR_WAIT_TIMEOUT_S}; "
                    "publisher likely never supplied this tensor (stalled initial broadcast)"
                )
            wait_logger.maybe_warn(f"[{tensor_index}/{n}]{full_tensor_key=}, {tensor_entry.version=} < {version=}")
            await asyncio.sleep(self.DEFAULT_INTERVAL_S)
        if tensor_entry.version != version:
            raise RuntimeError(
                f"[{self.name}] tensor publish overtook ingress wait: "
                f"key={full_tensor_key!r} expected={version} observed={tensor_entry.version}"
            )

    def _logical_payload_bytes(self) -> int:
        return sum(self.server.buffer[key].tensor.nbytes for key in self.full_tensor_keys)

    async def _wait_for_first_tensor_version(
        self,
        *,
        version: int,
        wait_logger: _ThrottledWaitLogger,
    ) -> None:
        first_key = self.full_tensor_keys[0]
        first_entry = self.server.buffer[first_key]
        first_version = first_entry.version
        while first_version != version:
            self._require_active_publish(version=version)
            if first_version > version:
                raise RuntimeError(
                    f"[{self.name}] first tensor overtook chain send: "
                    f"key={first_key!r} expected={version} observed={first_version}"
                )
            wait_logger.maybe_warn(
                f"[0/{len(self.full_tensor_keys)}]{first_key=}, {first_version=} != {version=} before endpoint creation"
            )
            await asyncio.sleep(self.DEFAULT_INTERVAL_S)
            first_version = first_entry.version

    async def _send_tagged_windows(
        self,
        *,
        version: int,
        logical_payload_bytes: int,
        wait_logger: _ThrottledWaitLogger,
    ) -> None:
        storage_metas = [
            TensorStorageMeta(
                key=key,
                data_meta=TensorMeta(
                    dtype=self.server.buffer[key].tensor.dtype,
                    shape=self.server.buffer[key].tensor.shape,
                ),
                version=version,
            )
            for key in self.full_tensor_keys
        ]
        ip, port = self.dst_server_address
        endpoint = await create_endpoint(ip, port)
        try:
            if relay_backend() == "nccl":
                # NCCL sends one payload at a time, so a round trip per tensor
                # dominates MoE models with thousands of small experts. Pack
                # them into bounded chunks like the actor publishers do.
                sender = TensorChunkSender(endpoint)
                for i_tensor, storage_meta in enumerate(storage_metas):
                    tensor, lock = await self._read_lock_version(i_tensor, version=version, wait_logger=wait_logger)
                    try:
                        await sender.add(storage_meta, tensor)
                    finally:
                        lock.release_read()
                await sender.finish()
                return
            await send_obj(
                endpoint,
                TensorTransferHeader(
                    storage_metas=storage_metas,
                    mode=TensorTransferMode.TAGGED,
                    window_size=_RELAY_CHAIN_SEND_WINDOW_SIZE,
                ),
            )
            for start in range(0, len(self.full_tensor_keys), _RELAY_CHAIN_SEND_WINDOW_SIZE):
                stop = min(start + _RELAY_CHAIN_SEND_WINDOW_SIZE, len(self.full_tensor_keys))
                window: list[tuple[int, torch.Tensor, Any]] = []
                try:
                    for i_tensor in range(start, stop):
                        tensor, lock = await self._read_lock_version(i_tensor, version=version, wait_logger=wait_logger)
                        window.append((i_tensor, tensor, lock))
                    await gather_cancel_on_error(
                        *(send_tensor(endpoint, tensor, tag=i_tensor) for i_tensor, tensor, _lock in window)
                    )
                finally:
                    for _i_tensor, _tensor, lock in window:
                        lock.release_read()
            await complete_tensor_transfer(
                endpoint,
                tensor_count=len(self.full_tensor_keys),
                payload_nbytes=logical_payload_bytes,
            )
        finally:
            await endpoint.close()

    async def _read_lock_version(
        self,
        i_tensor: int,
        *,
        version: int,
        wait_logger: _ThrottledWaitLogger,
    ) -> tuple[torch.Tensor, ReadWriteLock]:
        """Wait until tensor ``i_tensor`` holds ``version`` and return it read-locked."""
        key = self.full_tensor_keys[i_tensor]
        logger.debug(f"[{self.name}] Sending: [{i_tensor}/{len(self.full_tensor_keys)}]{key=}")
        entry = self.server.buffer[key]
        lock = self.server.tensor_rwlocks[key]
        while True:
            self._require_active_publish(version=version)
            await lock.acquire_read()
            observed_version = entry.version
            if observed_version == version:
                return entry.tensor, lock
            lock.release_read()
            if observed_version > version:
                raise RuntimeError(
                    f"[{self.name}] tensor overtook chain send: "
                    f"key={key!r} expected={version} observed={observed_version}"
                )
            wait_logger.maybe_warn(f"[{i_tensor}/{len(self.full_tensor_keys)}]{key=}, {observed_version=} < {version=}")
            await asyncio.sleep(self.DEFAULT_INTERVAL_S)

    def _full_publish_matches(self, *, version: int) -> bool:
        """Return whether every full tensor contains one exact version."""
        if not self.full_tensor_keys:
            return False
        return all(
            key in self.server.buffer and self.server.buffer[key].version == version for key in self.full_tensor_keys
        )


class AsyncRLvLLMColocateWorkerExtension(vLLMColocateWorkerExtension):
    """vLLM colocated-worker extension that loads weights from the relay's shm buffer.

    Injected into the rollout vLLM workers so that when the relay broadcasts
    a new version, the rollout side can mmap() the same backing tensors and
    call ``model.load_weights`` directly — no extra D2D / D2H staging hop.
    """

    def init_shm_state_dict(self: WorkerBase, state_meta_dict: dict[str, TensorMeta]) -> None:  # type: ignore[misc]
        """Materialize the shared CPU buffer mirroring the relay's full state dict.

        vLLM ``collective_rpc`` may msgpack each ``TensorMeta`` as
        ``(dtype_name, shape)``; accept that wire form as well as the NamedTuple.
        """
        shm_state_dict = {}
        for key, meta in state_meta_dict.items():
            if not isinstance(meta, TensorMeta):
                dtype_name, shape = meta
                meta = TensorMeta(
                    dtype=getattr(torch, str(dtype_name).removeprefix("torch.")),
                    shape=torch.Size(shape),
                )
            shm_state_dict[key] = make_shared_tensor_from_meta(meta, key=key)
        self.shm_state_dict = shm_state_dict

    def load_weights_from_shm_state_dict(self: WorkerBase) -> None:  # type: ignore[misc]
        """Counterpart to ``vLLMColocateWorkerExtension.update_weights_from_ipc``.

        Reads from the shared CPU buffer prepared by :meth:`init_shm_state_dict`
        instead of an IPC handle; semantics otherwise mirror the standard
        colocated vLLM update path.
        """
        if not hasattr(self, "shm_state_dict"):
            raise RuntimeError("Relay shared state must be initialized before loading weights")

        from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

        for model in self._iter_all_models():
            patch_vllm_moe_model_weight_loader(model)

        assert self.device is not None
        self._update_weights(list(self.shm_state_dict.items()), peft_config=None, base_sync_done=False)

        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        for model, model_config in self._iter_all_models_with_config():
            process_weights_after_loading(model, model_config, self.device)
