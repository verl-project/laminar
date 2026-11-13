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
"""Lifecycle orchestration for Router-committed rollout updates and GPU handoff."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Coroutine
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from laminar.metrics import MetricSnapshot, MetricSource, cumulative_counter_specs
from laminar.multi_version_rollout.lifecycle import (
    _DRAIN_TIMEOUT_S,
    begin_dead_server_recovery,
    restart_and_finish_recovery,
    wait_for_source_drain,
)
from laminar.multi_version_rollout.repack import (
    CommittedAbortSource,
    CommittedRepackGroup,
    RepackPlanResult,
)
from laminar.train.relay import EMPTY_VERSION, RelayClient
from laminar.utils.concurrency import gather_cancel_on_error, log_exception_callback

if TYPE_CHECKING:
    from ray.actor import ActorHandle

logger = logging.getLogger(__name__)
# TODO: tune caller budgets from measured drain and planned training durations.
_COLOCATE_PAUSE_TIMEOUT_S = 3600.0
_MANAGER_COUNTER_DEFAULTS: dict[str, int | float] = {
    "colocate/pause_succeeded": 0,
    "colocate/resume_succeeded": 0,
    "colocate/handoff_failed": 0,
    "colocate/pause_latency_s": 0.0,
    "colocate/resume_latency_s": 0.0,
    "recovery/attempts": 0,
    "recovery/succeeded": 0,
    "recovery/failed": 0,
    "recovery/latency_s": 0.0,
    "repack/plan_and_begin_attempts": 0,
    "repack/plan_and_begin_succeeded": 0,
    "repack/plan_and_begin_failed": 0,
    "repack/plan_and_begin_latency_s": 0.0,
    "repack/transfer_failed": 0,
    "repack/no_plan_calls": 0,
    "repack/bins_accepted": 0,
    "repack/bins_failed": 0,
    "repack/packed_sources_accepted": 0,
    "repack/accepted_sources_per_bin/sum": 0,
    "repack/accepted_sources_per_bin/count": 0,
    "repack/source_drain_wait_s": 0.0,
    "repack/group_drain_critical_path_s": 0.0,
    "repack/group_drain_straggler_gap_s": 0.0,
    "repack/group_source_drain_wait_s/sum": 0.0,
    "repack/group_source_drain_wait_s/count": 0,
    "repack/load_weights_attempts": 0,
    "repack/load_weights_succeeded": 0,
    "repack/load_weights_failed": 0,
    "repack/load_weights_latency_s": 0.0,
    "repack/reopen_failed": 0,
    "repack/reopen_latency_s": 0.0,
    "repack/idle_update_attempts": 0,
    "repack/idle_update_succeeded": 0,
    "repack/idle_update_failed": 0,
    "repack/abort_update_attempts": 0,
    "repack/abort_update_succeeded": 0,
    "repack/abort_update_failed": 0,
    "repack/operation_latency_s": 0.0,
    "repack/group_operation_latency_s": 0.0,
}


@dataclass(frozen=True)
class MultiVersionRolloutRepackConfig:
    """Typed config for request migration between same-version rollout engines."""

    enable: bool = True
    source_max_kv_cache_usage: float = 0.5
    destination_max_kv_cache_usage: float = 0.95

    def __post_init__(self) -> None:
        for name, value in (
            ("source_max_kv_cache_usage", self.source_max_kv_cache_usage),
            ("destination_max_kv_cache_usage", self.destination_max_kv_cache_usage),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"multi_version_rollout.repack.{name} must be in [0, 1], got {value}")


@dataclass(frozen=True)
class MultiVersionRolloutManagerConfig:
    """Typed config for multi-version rollout lifecycle control."""

    repack: MultiVersionRolloutRepackConfig = field(default_factory=MultiVersionRolloutRepackConfig)
    schedule_interval_s: float = 5.0
    colocate: bool = False

    def __post_init__(self) -> None:
        if self.schedule_interval_s <= 0:
            raise ValueError(
                f"multi_version_rollout.schedule_interval_s must be positive, got {self.schedule_interval_s}"
            )


class MultiVersionRolloutManager:
    """Execute lifecycle operations selected and committed by the Router.

    The router owns weight versions, lifecycle state, route bindings, request
    release, Repack placement, and route transfer. This manager orchestrates
    source abort, weight loading, and reopening.
    """

    def __init__(self, config: MultiVersionRolloutManagerConfig):
        self.config = config
        self.servers: dict[str, ActorHandle] = {}
        self.router: ActorHandle | None = None
        self.relay_client: RelayClient | None = None
        self._run_task: asyncio.Task | None = None
        self._repack_tasks: dict[asyncio.Task[None], frozenset[str]] = {}
        self._handoff_lock = asyncio.Lock()
        # Colocated replicas initialize asleep before actor model construction.
        self._colocate_phase = "training"
        self._metrics: dict[str, int | float] = {
            **_MANAGER_COUNTER_DEFAULTS,
            "colocate/pause_latency_s/max": 0.0,
            "colocate/resume_latency_s/max": 0.0,
            "recovery/latency_s/max": 0.0,
            "repack/plan_and_begin_latency_s/max": 0.0,
            "repack/accepted_sources_per_bin/min": 0,
            "repack/accepted_sources_per_bin/max": 0,
            "repack/source_drain_wait_s/max": 0.0,
            "repack/group_drain_critical_path_s/max": 0.0,
            "repack/group_drain_straggler_gap_s/max": 0.0,
            "repack/group_source_drain_wait_s/min": 0.0,
            "repack/group_source_drain_wait_s/max": 0.0,
            "repack/load_weights_latency_s/max": 0.0,
            "repack/reopen_latency_s/max": 0.0,
            "repack/operation_latency_s/max": 0.0,
            "repack/group_operation_latency_s/max": 0.0,
        }
        self._metric_source = MetricSource(cumulative_counter_specs("multi_version_rollout", _MANAGER_COUNTER_DEFAULTS))

    def connect(
        self,
        *,
        servers: dict[str, ActorHandle],
        router: ActorHandle,
        relay_client: RelayClient,
    ) -> None:
        """Connect cross-component actors after they all exist."""
        if not servers:
            raise ValueError("MultiVersionRolloutManager requires at least one rollout engine")
        if router is None or relay_client is None:
            raise ValueError("MultiVersionRolloutManager requires both router and relay_client")
        self.servers = dict(servers)
        self.router = router
        self.relay_client = relay_client

    async def start(self) -> None:
        """Launch the periodic multi-version lifecycle loop inside this actor."""
        self._require_connected()
        if self._run_task is not None and not self._run_task.done():
            return
        self._run_task = asyncio.create_task(self._loop(), name="multi_version_rollout_manager.loop")
        self._run_task.add_done_callback(log_exception_callback)

    async def wait_for_stop(self) -> None:
        """Wait for the background loop to stop and surface fatal errors."""
        await self._run_task

    async def pause_for_training(self) -> None:
        """Abort unfinished turns, drain release ACKs, and sleep the entire pool."""
        self._require_colocate()
        async with self._handoff_lock:
            if self._colocate_phase != "rollout":
                raise RuntimeError(f"cannot pause colocated rollout in phase {self._colocate_phase!r}")
            self._colocate_phase = "pausing"
            start = time.perf_counter()
            try:
                deadline = asyncio.get_running_loop().time() + _COLOCATE_PAUSE_TIMEOUT_S
                async with asyncio.timeout_at(deadline):
                    versions = await self.router.begin_colocate_pause.remote()
                    if not isinstance(versions, dict) or set(versions) != set(self.servers):
                        raise RuntimeError("Router pause must fence every colocated rollout server")
                    await gather_cancel_on_error(
                        *(
                            self._wait_for_source_drain(server_id, expected_version=version, deadline=deadline)
                            for server_id, version in versions.items()
                        )
                    )
                    await gather_cancel_on_error(
                        *(server.sleep_for_training.remote() for server in self.servers.values())
                    )
            except BaseException:
                self._colocate_phase = "failed"
                self._metrics["colocate/handoff_failed"] += 1
                raise
            self._colocate_phase = "training"
            self._metrics["colocate/pause_succeeded"] += 1
            self._record_latency("colocate/pause_latency_s", time.perf_counter() - start)

    async def resume_rollout(self, version: int) -> None:
        """Wake all replicas at the committed version before reopening routing."""
        self._require_colocate()
        if type(version) is not int or version < 0:
            raise ValueError("colocated rollout version must be a non-negative integer")
        async with self._handoff_lock:
            if self._colocate_phase != "training":
                raise RuntimeError(f"cannot resume colocated rollout in phase {self._colocate_phase!r}")
            self._colocate_phase = "resuming"
            start = time.perf_counter()
            try:
                await gather_cancel_on_error(
                    *(self._load_colocate_source(server_id, version) for server_id in self.servers)
                )
                # Engines resume while Router still fences the whole cohort.
                # If one resume fails, no server is made routable.
                await gather_cancel_on_error(*(server.resume_generation.remote() for server in self.servers.values()))
                await gather_cancel_on_error(
                    *(
                        self.router.mark_active.remote(server_id=server_id, weight_version=version)
                        for server_id in self.servers
                    )
                )
            except BaseException:
                self._colocate_phase = "failed"
                self._metrics["colocate/handoff_failed"] += 1
                raise
            self._colocate_phase = "rollout"
            self._metrics["colocate/resume_succeeded"] += 1
            self._record_latency("colocate/resume_latency_s", time.perf_counter() - start)

    async def _load_colocate_source(self, server_id: str, version: int) -> None:
        server = self.servers[server_id]
        loaded_version = await server.load_weights.remote(wait_for_version=version, resume_generation=False)
        if type(loaded_version) is not int or loaded_version != version:
            raise RuntimeError(
                f"colocated rollout engine {server_id!r} loaded version {loaded_version!r}; expected exactly {version}"
            )
        await server.check_health.remote()

    def _require_colocate(self) -> None:
        self._require_connected()
        if not self.config.colocate:
            raise RuntimeError("GPU handoff requires multi_version_rollout.colocate=True")

    async def _loop(self) -> None:
        try:
            while True:
                if self.config.colocate:
                    # A dead rollout engine can prevent the next batch from
                    # ever reaching the trainer's handoff boundary. Surface
                    # that failure through wait_for_stop instead of leaving
                    # sample waiting forever. Sleeping/loading replicas are
                    # owned by the explicit handoff and checked on resume.
                    if self._colocate_phase == "rollout":
                        failed_servers = await self.router.failed_server_ids.remote()
                        if failed_servers:
                            self._colocate_phase = "failed"
                            raise RuntimeError(
                                f"colocated rollout engines failed: {failed_servers}; engine recovery is unsupported"
                            )
                else:
                    await self._reap_repack_tasks()
                    await self._schedule_recoveries()
                    await self._schedule_repack_once()
                await asyncio.sleep(self.config.schedule_interval_s)
        finally:
            await self._cancel_repack_tasks()

    async def _reap_repack_tasks(self) -> None:
        """Surface operation failures and release their server ownership."""
        done_tasks = [task for task in self._repack_tasks if task.done()]
        if not done_tasks:
            return
        for task in done_tasks:
            self._repack_tasks.pop(task)
        results = await asyncio.gather(*done_tasks, return_exceptions=True)
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise errors[0]

    async def _cancel_repack_tasks(self) -> None:
        tasks = list(self._repack_tasks)
        self._repack_tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _schedule_recoveries(self) -> None:
        """Recover confirmed dead engines without blocking healthy replicas."""
        assert self.router is not None
        busy = self._active_operation_servers()
        for server_id in await self.router.failed_server_ids.remote():
            if server_id in busy:
                continue
            # Router observations may predate the completion of a restart.
            if not await begin_dead_server_recovery(self.router, self.servers[server_id], server_id):
                continue
            self._start_operation(
                self._restart_source(server_id),
                name=f"multi_version_rollout_manager.recovery.{server_id}",
                participants=frozenset((server_id,)),
            )

    async def _recover_dead_source(self, source: str) -> bool:
        """A failed lifecycle RPC is recoverable only for a confirmed dead engine."""
        assert self.router is not None
        if not await begin_dead_server_recovery(self.router, self.servers[source], source):
            return False
        await self._restart_source(source)
        return True

    async def _restart_source(self, source: str) -> None:
        assert self.router is not None
        self._metrics["recovery/attempts"] += 1
        start = time.perf_counter()
        try:
            loadable = await self._get_loadable_version()
            if loadable <= EMPTY_VERSION:
                raise RuntimeError("cannot restart a rollout engine without published weights")
            await restart_and_finish_recovery(
                self.router,
                self.servers[source],
                source,
                wait_for_version=loadable,
            )
        except Exception:
            self._metrics["recovery/failed"] += 1
            raise
        else:
            self._metrics["recovery/succeeded"] += 1
        finally:
            self._record_latency("recovery/latency_s", time.perf_counter() - start)

    async def _schedule_repack_once(self) -> None:
        """Ask Router to commit one pass, then launch lifecycle operations."""
        self._require_connected()
        loadable_version = await self._get_loadable_version()
        if loadable_version <= EMPTY_VERSION:
            return
        plan = await self._plan_and_begin_repack(
            loadable_version=loadable_version,
            excluded_servers=self._active_operation_servers(),
        )
        self._start_operations(plan=plan, loadable_version=loadable_version)

    async def _plan_and_begin_repack(
        self,
        *,
        loadable_version: int,
        excluded_servers: frozenset[str],
    ) -> RepackPlanResult:
        assert self.router is not None
        self._metrics["repack/plan_and_begin_attempts"] += 1
        start_time = time.perf_counter()
        try:
            plan: RepackPlanResult = await self.router.plan_and_begin_repack.remote(
                loadable_version=loadable_version,
                source_max_kv_cache_usage=self.config.repack.source_max_kv_cache_usage,
                destination_max_kv_cache_usage=self.config.repack.destination_max_kv_cache_usage,
                excluded_servers=sorted(excluded_servers),
                repack_enabled=self.config.repack.enable,
            )
            if not isinstance(plan, RepackPlanResult):
                raise RuntimeError(f"Router returned invalid plan-and-begin response: {plan!r}")
            operation_servers = (
                {server_id for group in plan.groups for server_id in (group.destination, *group.sources)}
                | set(plan.idle_sources)
                | {source.server_id for source in plan.abort_sources}
            )
            unknown_servers = operation_servers - self.servers.keys()
            if unknown_servers:
                raise RuntimeError(f"Router committed unknown rollout servers: {unknown_servers}")
            excluded_participants = operation_servers & excluded_servers
            if excluded_participants:
                raise RuntimeError(f"Router committed excluded rollout servers: {excluded_participants}")
            source_counts = [len(group.sources) for group in plan.groups]
            self._metrics["repack/bins_accepted"] += len(plan.groups)
            self._metrics["repack/packed_sources_accepted"] += sum(source_counts)
            self._record_distribution("repack/accepted_sources_per_bin", source_counts)
            if not plan.groups:
                self._metrics["repack/no_plan_calls"] += 1
        except Exception:
            self._metrics["repack/plan_and_begin_failed"] += 1
            self._metrics["repack/transfer_failed"] += 1
            raise
        else:
            self._metrics["repack/plan_and_begin_succeeded"] += 1
            return plan
        finally:
            self._record_latency("repack/plan_and_begin_latency_s", time.perf_counter() - start_time)

    def _start_operations(
        self,
        *,
        plan: RepackPlanResult,
        loadable_version: int,
    ) -> None:
        for source in plan.abort_sources:
            self._start_operation(
                self._execute_abort_update(source, loadable_version=loadable_version),
                name=f"multi_version_rollout_manager.abort_update.{source.server_id}",
                participants=frozenset((source.server_id,)),
            )
        for source in plan.idle_sources:
            self._start_operation(
                self._execute_idle_update(source=source, loadable_version=loadable_version),
                name=f"multi_version_rollout_manager.idle_update.{source}",
                participants=frozenset((source,)),
            )
        for group in plan.groups:
            self._start_operation(
                self._execute_repack_group(group, loadable_version=loadable_version),
                name=f"multi_version_rollout_manager.repack_group.{group.destination}",
                participants=frozenset((group.destination, *group.sources)),
            )

    def _start_operation(
        self, operation: Coroutine[Any, Any, None], *, name: str, participants: frozenset[str]
    ) -> None:
        if (
            not participants
            or participants - self.servers.keys()
            or participants & self._active_operation_servers()
            or len(self._repack_tasks) >= len(self.servers)
        ):
            operation.close()
            raise RuntimeError("rollout lifecycle operations must own distinct registered replicas")
        task = asyncio.create_task(operation, name=name)
        task.add_done_callback(log_exception_callback)
        self._repack_tasks[task] = participants

    async def _wait_for_source_drain(
        self, server_id: str, *, expected_version: int, deadline: float | None = None
    ) -> None:
        """Use the shared Repack source drain before loading training weights."""
        assert self.router is not None
        if deadline is None:
            deadline = asyncio.get_running_loop().time() + _DRAIN_TIMEOUT_S
        await wait_for_source_drain(self.router, server_id, expected_version=expected_version, deadline=deadline)

    def _active_operation_servers(self) -> frozenset[str]:
        return frozenset(server_id for participants in self._repack_tasks.values() for server_id in participants)

    async def _execute_repack_group(self, group: CommittedRepackGroup, *, loadable_version: int) -> None:
        group_start = time.perf_counter()
        drain_waits_s: list[float] = []
        try:
            await gather_cancel_on_error(
                *(
                    self._execute_repack_source(
                        source,
                        drain_waits_s,
                        expected_version=group.weight_version,
                        loadable_version=loadable_version,
                    )
                    for source in group.sources
                )
            )
        except Exception:
            self._metrics["repack/transfer_failed"] += 1
            self._metrics["repack/bins_failed"] += 1
            raise
        finally:
            self._record_latency("repack/group_operation_latency_s", time.perf_counter() - group_start)
        self._record_group_drain_waits(drain_waits_s)

    async def _execute_abort_update(self, source: CommittedAbortSource, *, loadable_version: int) -> None:
        self._metrics["repack/abort_update_attempts"] += 1
        try:
            await self._execute_repack_source(
                source.server_id,
                [],
                expected_version=source.weight_version,
                loadable_version=loadable_version,
            )
        except Exception:
            self._metrics["repack/abort_update_failed"] += 1
            raise
        else:
            self._metrics["repack/abort_update_succeeded"] += 1

    async def _execute_repack_source(
        self,
        source: str,
        drain_waits_s: list[float],
        *,
        expected_version: int,
        loadable_version: int,
    ) -> None:
        assert self.router is not None
        operation_start = time.perf_counter()
        drain_start = time.perf_counter()
        try:
            await self._wait_for_source_drain(source, expected_version=expected_version)
            drain_wait_s = time.perf_counter() - drain_start
            self._record_latency("repack/source_drain_wait_s", drain_wait_s)
            drain_waits_s.append(drain_wait_s)

            await self._load_and_reopen_source(
                source=source,
                loadable_version=loadable_version,
            )
        except Exception:
            if not await self._recover_dead_source(source):
                raise
        finally:
            self._record_latency("repack/operation_latency_s", time.perf_counter() - operation_start)

    async def _execute_idle_update(self, *, source: str, loadable_version: int) -> None:
        self._metrics["repack/idle_update_attempts"] += 1
        operation_start = time.perf_counter()
        try:
            await self._load_and_reopen_source(
                source=source,
                loadable_version=loadable_version,
            )
        except Exception:
            self._metrics["repack/idle_update_failed"] += 1
            if not await self._recover_dead_source(source):
                raise
        else:
            self._metrics["repack/idle_update_succeeded"] += 1
        finally:
            self._record_latency("repack/operation_latency_s", time.perf_counter() - operation_start)

    async def _load_and_reopen_source(self, *, source: str, loadable_version: int) -> None:
        assert self.router is not None
        loaded_version = await self._load_source_weights(
            source=source,
            loadable_version=loadable_version,
        )
        reopen_start = time.perf_counter()
        try:
            await self.router.mark_active.remote(
                server_id=source,
                weight_version=loaded_version,
            )
        except Exception:
            self._metrics["repack/reopen_failed"] += 1
            raise
        finally:
            self._record_latency("repack/reopen_latency_s", time.perf_counter() - reopen_start)

    async def _load_source_weights(self, *, source: str, loadable_version: int) -> int:
        self._metrics["repack/load_weights_attempts"] += 1
        start_time = time.perf_counter()
        try:
            loaded_version = await self.servers[source].load_weights.remote(wait_for_version=loadable_version)
            if type(loaded_version) is not int:
                raise RuntimeError(
                    f"rollout engine {source!r} returned invalid loaded version {loaded_version!r}; expected int"
                )
            if loaded_version < loadable_version:
                raise RuntimeError(
                    f"rollout engine {source!r} loaded stale version {loaded_version}; "
                    f"expected at least {loadable_version}"
                )
        except Exception:
            self._metrics["repack/load_weights_failed"] += 1
            raise
        latency_s = time.perf_counter() - start_time
        self._metrics["repack/load_weights_succeeded"] += 1
        self._record_latency("repack/load_weights_latency_s", latency_s)
        return loaded_version

    async def _get_loadable_version(self) -> int:
        assert self.relay_client is not None
        loadable_version = await self.relay_client.get_loadable_version()
        if type(loadable_version) is not int or loadable_version < EMPTY_VERSION:
            raise RuntimeError(
                f"relay returned invalid loadable version {loadable_version!r}; expected an integer >= {EMPTY_VERSION}"
            )
        return loadable_version

    def get_metrics(self) -> MetricSnapshot:
        """Return cumulative manager metrics for Tracker consumption."""
        metrics = dict(self._metrics)
        metrics["repack/scheduler_inflight_operations"] = sum(not task.done() for task in self._repack_tasks)
        return self._metric_source.snapshot(metrics)

    def _record_latency(self, metric_name: str, latency_s: float) -> None:
        self._metrics[metric_name] += latency_s
        max_name = f"{metric_name}/max"
        self._metrics[max_name] = max(float(self._metrics[max_name]), latency_s)

    def _record_group_drain_waits(self, drain_waits_s: list[float]) -> None:
        if not drain_waits_s:
            return
        critical_path_s = max(drain_waits_s)
        straggler_gap_s = critical_path_s - min(drain_waits_s)
        self._record_latency("repack/group_drain_critical_path_s", critical_path_s)
        self._record_latency("repack/group_drain_straggler_gap_s", straggler_gap_s)
        self._record_distribution("repack/group_source_drain_wait_s", drain_waits_s)

    def _record_distribution(self, metric_name: str, values: list[int | float]) -> None:
        if not values:
            return
        count_key = f"{metric_name}/count"
        previous_count = int(self._metrics[count_key])
        self._metrics[count_key] = previous_count + len(values)
        self._metrics[f"{metric_name}/sum"] += sum(values)
        minimum = min(values)
        min_key = f"{metric_name}/min"
        self._metrics[min_key] = minimum if previous_count == 0 else min(self._metrics[min_key], minimum)
        max_key = f"{metric_name}/max"
        self._metrics[max_key] = max(self._metrics[max_key], max(values))

    def _require_connected(self) -> None:
        if not self.servers or self.router is None:
            raise RuntimeError("MultiVersionRolloutManager.connect must be called first with a rollout router")
