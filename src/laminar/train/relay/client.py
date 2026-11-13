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
"""Local client facade for async-RL relay operations.

``RelayManager`` is a Ray actor. ``RelayClient`` is the lightweight local
wrapper passed to trainers, rollout servers, validation managers, or other
framework code so relay protocol calls do not leak ``.remote()`` plumbing into
each component.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Iterable

import ray

from laminar.utils.tensor import TensorMeta

if TYPE_CHECKING:
    from ray.actor import ActorHandle

    from laminar.train.relay.manager import RelayManager

logger = logging.getLogger(__name__)


def application_goodput_bytes_per_s(*, logical_payload_bytes: int, elapsed_s: float, boundary: str) -> float:
    """Return logical-payload goodput; empty payloads are 0."""
    if logical_payload_bytes == 0:
        return 0.0
    if elapsed_s <= 0.0:
        raise RuntimeError(
            f"{boundary} elapsed time must be positive for a non-empty payload: "
            f"logical_payload_bytes={logical_payload_bytes} elapsed_s={elapsed_s}"
        )
    return logical_payload_bytes / elapsed_s


@dataclass(frozen=True)
class ActorPublishStats:
    """Aggregate of completed actor sends for one version, not scheduled sends."""

    publisher_count: int
    tensor_count: int = 0
    chunk_count: int = 0
    payload_bytes: int = 0
    export_time_s: float = 0.0
    connect_time_s: float = 0.0
    sender_time_s: float = 0.0
    tensor_prepare_time_s: float = 0.0
    pack_time_s: float = 0.0
    staging_time_s: float = 0.0
    transport_time_s: float = 0.0
    completion_ack_time_s: float = 0.0
    total_time_s: float = 0.0

    TIME_METRICS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("export_time_s", "timing/relay_publish_actor_export"),
        ("connect_time_s", "timing/relay_publish_actor_connect"),
        ("sender_time_s", "timing/relay_publish_actor_sender"),
        ("tensor_prepare_time_s", "timing/relay_publish_actor_tensor_prepare"),
        ("pack_time_s", "timing/relay_publish_actor_pack"),
        ("staging_time_s", "timing/relay_publish_actor_staging"),
        ("transport_time_s", "timing/relay_publish_actor_transport"),
        ("completion_ack_time_s", "timing/relay_publish_actor_completion_ack"),
        ("total_time_s", "timing/relay_publish_actor_worker_total"),
    )
    COUNT_METRICS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("tensor_count", "relay_publish/actor_tensor_count"),
        ("chunk_count", "relay_publish/actor_chunk_count"),
        ("payload_bytes", "relay_publish/actor_logical_payload_bytes"),
    )

    def as_metrics(self) -> dict[str, float]:
        """Return the canonical trainer metrics derived from this publish."""
        metrics = {
            metric_name: float(getattr(self, field_name))
            for field_name, metric_name in (*self.TIME_METRICS, *self.COUNT_METRICS)
        }
        metrics["relay_publish/actor_publisher_count"] = float(self.publisher_count)
        return metrics

    @classmethod
    def from_results(cls, *results: object) -> ActorPublishStats:
        """Collect one level of worker-group results and aggregate a publish."""
        values: list[ActorPublishStats] = []
        for result in results:
            items = result if isinstance(result, list | tuple) else (result,)
            for item in items:
                if item is None:
                    continue
                if not isinstance(item, cls):
                    raise RuntimeError(f"Unexpected Relay actor send result: {type(item).__name__}")
                values.append(item)
        if not values:
            raise RuntimeError("Relay actor send completed without publisher stats")

        counts = {
            field_name: sum(getattr(value, field_name) for value in values) for field_name, _ in cls.COUNT_METRICS
        }
        timings = {
            field_name: max(getattr(value, field_name) for value in values) for field_name, _ in cls.TIME_METRICS
        }
        return cls(
            publisher_count=sum(value.publisher_count for value in values),
            **counts,
            **timings,
        )


@dataclass(frozen=True)
class RelayBroadcastStats:
    """Canonical result of one completed Relay chain broadcast."""

    relay_broadcast_time_s: float
    logical_payload_bytes: int
    tensor_count: int
    hop_count: int
    bottleneck_hop_index: int
    end_to_end_goodput_bytes_per_s: float

    def as_metrics(self) -> dict[str, float]:
        """Return the canonical trainer metrics derived from this broadcast."""
        return {
            "timing/relay_publish_relay_broadcast": self.relay_broadcast_time_s,
            "relay/broadcast/logical_payload_bytes": float(self.logical_payload_bytes),
            "relay/broadcast/tensor_count": float(self.tensor_count),
            "relay/broadcast/hop_count": float(self.hop_count),
            "relay/broadcast/bottleneck_hop_index": float(self.bottleneck_hop_index),
            "relay/broadcast/end_to_end_goodput_bytes_per_s": self.end_to_end_goodput_bytes_per_s,
        }


@dataclass(frozen=True)
class RelayClient:
    """Small typed facade around the Ray ``RelayManager`` handle."""

    manager: ActorHandle[RelayManager]

    # ---- Manager version/state RPCs ---------------------------------

    async def get_trainer_version(self) -> int:
        return await self.manager.get_trainer_version.remote()

    async def get_loadable_version(self) -> int:
        return await self.manager.get_loadable_version.remote()

    async def acquire_rollout_version(
        self,
        node_ids: Iterable[str],
        *,
        minimum_version: int,
    ) -> int:
        return await self.manager.acquire_rollout_version.remote(
            node_ids,
            minimum_version=minimum_version,
        )

    async def release_full_read_async(self, node_ids: Iterable[str]) -> None:
        await self.manager.release_full_read.remote(node_ids)

    async def get_metrics(self) -> dict[str, int | float]:
        return await self.manager.get_metrics.remote()

    # ---- Actor/rollout initialization -------------------------------

    def connect_actor_workers(self, actor_wg: Any) -> None:
        ip, port = ray.get(self.manager.get_master_server_ip_port.remote())
        actor_wg.connect_to_relay_server(ip=ip, port=port)

    def init_full_state_dict_in_shm(
        self,
        *,
        full_state_meta_dict: dict[str, TensorMeta],
    ) -> Any:
        return self.manager.init_full_state_dict_in_shm.remote(
            full_state_meta_dict=full_state_meta_dict,
        )

    def init_rollout_state_dict_in_shm(
        self,
        *,
        rollout_servers: Iterable[Any],
        state_meta_dict: dict[str, TensorMeta],
    ) -> list[Any]:
        return [
            server.init_shm_state_dict_in_workers.remote(
                state_meta_dict=state_meta_dict,
            )
            for server in rollout_servers
        ]

    # ---- Weight publish ---------------------------------------------

    async def publish(
        self,
        actor_wg: Any,
        *,
        version: int,
        dtype: str,
    ) -> dict[str, float]:
        """Run one fail-fast weight publish and return flat trainer metrics.

        Owns begin → concurrent actor-send ∥ broadcast → commit. Trainer only
        schedules / awaits this outer fence; barrier orchestration stays here.
        Failure ends the job: local sibling awaiters are cancelled and the
        first error is raised, with no Relay abort/retry. Cancellation cannot
        stop an actor send already running in ``to_thread``; the poisoned slot
        stays fenced while process teardown owns any residual work.
        """
        publish_start_time = time.perf_counter()
        await self.manager.begin_publish.remote(version=version)
        concurrent_phase_start_time = time.perf_counter()

        def _send_and_wait_for_actor_weights() -> tuple[ActorPublishStats, float]:
            # Worker-group dispatch may synchronously ray.get() a blocking
            # worker method. Keep dispatch and its failure fence off the caller
            # event loop (trainer asyncio control plane).
            result = self._send_actor_weights(
                actor_wg,
                version=version,
                dtype=dtype,
            )
            return result, time.perf_counter()

        async def _broadcast_and_timestamp() -> tuple[RelayBroadcastStats, float]:
            broadcast_stats = await self.manager.broadcast.remote(version=version)
            return broadcast_stats, time.perf_counter()

        actor_task = asyncio.create_task(
            asyncio.to_thread(_send_and_wait_for_actor_weights),
            name=f"relay-actor-send:{version}",
        )
        broadcast_task = asyncio.create_task(
            _broadcast_and_timestamp(),
            name=f"relay-broadcast:{version}",
        )
        try:
            await asyncio.wait(
                {actor_task, broadcast_task},
                return_when=asyncio.FIRST_EXCEPTION,
            )
            actor_error = self._publish_task_error(actor_task)
            broadcast_error = self._publish_task_error(broadcast_task)
            if actor_error is not None or broadcast_error is not None:
                raise broadcast_error or actor_error
        except BaseException:
            actor_task.cancel()
            broadcast_task.cancel()
            await asyncio.gather(actor_task, broadcast_task, return_exceptions=True)
            raise

        actor_stats, actor_flush_done_time = actor_task.result()
        broadcast_stats, broadcast_done_time = broadcast_task.result()
        if actor_stats.payload_bytes != broadcast_stats.logical_payload_bytes:
            raise RuntimeError(
                "Relay publish logical payload mismatch: "
                f"version={version} actor={actor_stats.payload_bytes} "
                f"relay={broadcast_stats.logical_payload_bytes}"
            )
        # Reject a completed send that reported no transport time before commit.
        application_goodput_bytes_per_s(
            logical_payload_bytes=actor_stats.payload_bytes,
            elapsed_s=actor_stats.transport_time_s,
            boundary="Relay actor active transport",
        )

        commit_start_time = time.perf_counter()
        await self.manager.complete_publish.remote(version=version)
        commit_time_s = time.perf_counter() - commit_start_time
        actor_flush_time_s = actor_flush_done_time - concurrent_phase_start_time
        broadcast_time_s = broadcast_done_time - concurrent_phase_start_time
        publish_time_s = time.perf_counter() - publish_start_time
        logger.info(
            "Relay publish version=%s committed: actor=%.2fs broadcast=%.2fs total=%.2fs",
            version,
            actor_flush_time_s,
            broadcast_time_s,
            publish_time_s,
        )
        publish_metrics = {
            "relay_publish/completed_version": float(version),
            "timing/relay_publish_actor_flush": actor_flush_time_s,
            "timing/relay_publish_broadcast": broadcast_time_s,
            "timing/relay_publish_commit": commit_time_s,
            "timing/relay_publish_total": publish_time_s,
            "relay_publish/effective_goodput_bytes_per_s": application_goodput_bytes_per_s(
                logical_payload_bytes=actor_stats.payload_bytes,
                elapsed_s=publish_time_s,
                boundary="Relay full publish",
            ),
        }
        publish_metrics.update(broadcast_stats.as_metrics())
        publish_metrics.update(actor_stats.as_metrics())
        return publish_metrics

    def _send_actor_weights(
        self,
        actor_wg: Any,
        *,
        version: int,
        dtype: str,
    ) -> ActorPublishStats:
        """Dispatch actor weights and drain every worker send barrier."""
        try:
            send_ref = actor_wg.send_weights(
                version=version,
                dtype=dtype,
            )
            send_result = self._materialize_send_ref(send_ref)
        except BaseException as send_error:
            try:
                self._materialize_send_ref(actor_wg.wait_for_sending_to_relay())
            except BaseException as fence_error:
                send_error.add_note(f"Relay actor-send fence also failed: {fence_error!r}")
            raise
        send_results = send_result if isinstance(send_result, list | tuple) else (send_result,)
        if any(isinstance(result, ActorPublishStats) for result in send_results):
            # Synchronous publishers return stats only after every send RPC
            # completes. A second RPC can queue behind next-step computation.
            return ActorPublishStats.from_results(send_result)
        wait_result = self._materialize_send_ref(actor_wg.wait_for_sending_to_relay())
        return ActorPublishStats.from_results(send_result, wait_result)

    @staticmethod
    def _publish_task_error(task: asyncio.Task[Any]) -> BaseException | None:
        if not task.done():
            return None
        if task.cancelled():
            return asyncio.CancelledError(f"Relay publish task {task.get_name()!r} was cancelled")
        return task.exception()

    @staticmethod
    def _materialize_send_ref(send_ref: Any | None) -> Any:
        if send_ref is None:
            return None
        if isinstance(send_ref, ray.ObjectRef):
            return ray.get(send_ref)
        if (
            isinstance(send_ref, list | tuple)
            and send_ref
            and all(isinstance(item, ray.ObjectRef) for item in send_ref)
        ):
            results = []
            errors: list[BaseException] = []
            for item in send_ref:
                try:
                    results.append(ray.get(item))
                except BaseException as error:
                    errors.append(error)
            if errors:
                primary_error = errors[0]
                for error in errors[1:]:
                    primary_error.add_note(f"Another Relay actor worker also failed: {error!r}")
                raise primary_error
            return results
        return send_ref
