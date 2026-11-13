# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from laminar.global_request_router.router import RouteUnavailableError
from laminar.multi_version_rollout import (
    CommittedAbortSource,
    CommittedRepackGroup,
    MultiVersionRolloutManager,
    MultiVersionRolloutManagerConfig,
    MultiVersionRolloutRepackConfig,
    RepackPlanResult,
)
from laminar.multi_version_rollout import manager as manager_module


class _RemoteMethod:
    def __init__(
        self,
        *,
        func: Callable[..., Any] | None = None,
        result: Any = None,
        exception: BaseException | None = None,
    ) -> None:
        self.func = func
        self.result = result
        self.exception = exception
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    async def remote(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        if self.exception is not None:
            raise self.exception
        result = self.func(*args, **kwargs) if self.func is not None else self.result
        if inspect.isawaitable(result):
            return await result
        return result


class _FakeRelay:
    def __init__(self, version: int | None) -> None:
        self.version = version

    async def get_loadable_version(self) -> int | None:
        return self.version


class _FakeServer:
    def __init__(
        self,
        *,
        loaded_version: int | None = None,
        exception: BaseException | None = None,
        on_load: Callable[[], Any] | None = None,
    ) -> None:
        self.loaded_version = loaded_version
        self.exception = exception
        self.on_load = on_load
        self.load_weights = _RemoteMethod(func=self._load_weights)
        self.snapshot = _RemoteMethod(result=SimpleNamespace(backend_health="healthy"))
        self.restart = _RemoteMethod(result=8)
        self.get_server_handle = _RemoteMethod(result="rebuilt-handle")

    async def _load_weights(self, *, wait_for_version: int) -> int:
        if self.exception is not None:
            raise self.exception
        if self.on_load is not None:
            result = self.on_load()
            if inspect.isawaitable(result):
                await result
        return wait_for_version if self.loaded_version is None else self.loaded_version


class _FakeRouter:
    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.drain_status: dict[str, bool] = {}
        self.drain_delay_s: dict[str, float] = {}
        self.server_handles: dict[str, Any] = {}
        self.get_server_handle = _RemoteMethod(func=self._get_server_handle)
        self.plan_and_begin_repack = _RemoteMethod(func=self._plan_and_begin_repack)
        self.try_mark_updating = _RemoteMethod(func=self._try_mark_updating)
        self.mark_active = _RemoteMethod(result={"ok": True})
        self.failed_server_ids = _RemoteMethod(result=[])
        self.begin_recovery = _RemoteMethod()
        self.finish_recovery = _RemoteMethod()

    def _plan_and_begin_repack(self, **_: Any) -> Any:
        if not self.responses:
            raise AssertionError("unexpected plan_and_begin_repack call")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def _get_server_handle(self, *, server_id: str):
        if server_id not in self.server_handles:

            async def abort(*, abort_only):
                assert abort_only
                await asyncio.sleep(self.drain_delay_s.get(server_id, 0.0))

            self.server_handles[server_id] = SimpleNamespace(abort_all_requests=_RemoteMethod(func=abort))
        return self.server_handles[server_id]

    def _try_mark_updating(
        self,
        *,
        server_id: str,
        expected_version: int,
    ) -> None:
        del expected_version
        if not self.drain_status.get(server_id, True):
            raise RouteUnavailableError("source is still draining")


def _committed_group(
    *,
    sources: tuple[str, ...] = ("s0",),
    destination: str = "d0",
    weight_version: int = 1,
) -> CommittedRepackGroup:
    return CommittedRepackGroup(
        sources=sources,
        destination=destination,
        weight_version=weight_version,
    )


def _router_response(
    *,
    groups: list[CommittedRepackGroup] | None = None,
    idle_sources: list[str] | None = None,
) -> RepackPlanResult:
    return RepackPlanResult(
        groups=tuple(groups or ()),
        idle_sources=tuple(idle_sources or ()),
    )


def _manager(
    *,
    router: _FakeRouter,
    servers: dict[str, _FakeServer],
    version: int | None = 2,
    repack_enabled: bool = True,
) -> MultiVersionRolloutManager:
    manager = MultiVersionRolloutManager(
        MultiVersionRolloutManagerConfig(
            repack=MultiVersionRolloutRepackConfig(
                enable=repack_enabled,
                source_max_kv_cache_usage=0.4,
                destination_max_kv_cache_usage=0.9,
            ),
            schedule_interval_s=0.01,
        )
    )
    manager.connect(servers=servers, router=router, relay_client=_FakeRelay(version))
    return manager


async def _run_scheduled_pass(manager: MultiVersionRolloutManager) -> int:
    """Drive the production scheduler once and surface background task errors."""
    previous_operations = set(manager._repack_tasks)
    await manager._schedule_repack_once()
    started_count = len(set(manager._repack_tasks) - previous_operations)
    tasks = list(manager._repack_tasks)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
        await manager._reap_repack_tasks()
    return started_count


@pytest.mark.parametrize("field", ["source_max_kv_cache_usage", "destination_max_kv_cache_usage"])
@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_repack_config_rejects_invalid_kv_threshold(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        MultiVersionRolloutRepackConfig(**{field: value})


@pytest.mark.asyncio
async def test_scheduled_pass_uses_router_plan_and_begin_and_executes_sources_concurrently() -> None:
    group = _committed_group(sources=("s0", "s1"))
    router = _FakeRouter([_router_response(groups=[group])])
    active_loads = 0
    max_active_loads = 0
    both_started = asyncio.Event()
    release = asyncio.Event()

    async def on_load() -> None:
        nonlocal active_loads, max_active_loads
        active_loads += 1
        max_active_loads = max(max_active_loads, active_loads)
        if active_loads == 2:
            both_started.set()
        await release.wait()
        active_loads -= 1

    servers = {
        "s0": _FakeServer(loaded_version=3, on_load=on_load),
        "s1": _FakeServer(loaded_version=3, on_load=on_load),
        "d0": _FakeServer(),
    }
    manager = _manager(router=router, servers=servers)

    task = asyncio.create_task(_run_scheduled_pass(manager))
    await asyncio.wait_for(both_started.wait(), timeout=1)
    release.set()
    assert await task == 1

    assert max_active_loads == 2
    assert router.plan_and_begin_repack.calls == [
        (
            (),
            {
                "loadable_version": 2,
                "source_max_kv_cache_usage": 0.4,
                "destination_max_kv_cache_usage": 0.9,
                "excluded_servers": [],
                "repack_enabled": True,
            },
        )
    ]
    assert len(router.try_mark_updating.calls) == 2
    assert {call[1]["server_id"] for call in router.try_mark_updating.calls} == {"s0", "s1"}
    assert {call[1]["expected_version"] for call in router.try_mark_updating.calls} == {1}
    assert not hasattr(router, "mark_updating")
    assert [servers[source].load_weights.calls[0][1]["wait_for_version"] for source in ("s0", "s1")] == [2, 2]
    assert [call[1]["weight_version"] for call in router.mark_active.calls] == [3, 3]
    metrics = manager.get_metrics()
    assert metrics["repack/plan_and_begin_attempts"] == 1
    assert metrics["repack/bins_accepted"] == 1
    assert metrics["repack/packed_sources_accepted"] == 2
    assert metrics["repack/accepted_sources_per_bin/count"] == 1
    assert metrics["repack/accepted_sources_per_bin/sum"] == 2
    assert metrics["repack/accepted_sources_per_bin/min"] == 2
    assert metrics["repack/accepted_sources_per_bin/max"] == 2
    assert metrics["repack/group_source_drain_wait_s/count"] == 2
    assert metrics["repack/group_drain_critical_path_s/max"] >= 0


@pytest.mark.asyncio
async def test_latest_version_abort_sources_drain_load_and_reopen_without_destination() -> None:
    router = _FakeRouter(
        [
            RepackPlanResult(
                groups=(),
                idle_sources=("idle",),
                abort_sources=(CommittedAbortSource(server_id="busy", weight_version=1),),
            )
        ]
    )
    servers = {"busy": _FakeServer(), "idle": _FakeServer()}
    manager = _manager(router=router, servers=servers)

    assert await _run_scheduled_pass(manager) == 2

    assert [call[1] for call in router.try_mark_updating.calls] == [{"server_id": "busy", "expected_version": 1}]
    assert [server.load_weights.calls[0][1]["wait_for_version"] for server in servers.values()] == [2, 2]
    assert sorted(call[1]["server_id"] for call in router.mark_active.calls) == ["busy", "idle"]
    metrics = manager.get_metrics()
    assert metrics["repack/abort_update_attempts"] == 1
    assert metrics["repack/abort_update_succeeded"] == 1
    assert metrics["repack/idle_update_succeeded"] == 1


@pytest.mark.asyncio
async def test_group_drain_metrics_capture_critical_path_and_straggler_gap() -> None:
    group = _committed_group(sources=("s0", "s1"))
    router = _FakeRouter([_router_response(groups=[group])])
    router.drain_delay_s.update({"s0": 0.01, "s1": 0.04})
    manager = _manager(
        router=router,
        servers={"s0": _FakeServer(), "s1": _FakeServer(), "d0": _FakeServer()},
    )

    assert await _run_scheduled_pass(manager) == 1

    metrics = manager.get_metrics()
    assert metrics["repack/group_source_drain_wait_s/count"] == 2
    assert metrics["repack/group_drain_critical_path_s"] >= 0.035
    assert metrics["repack/group_drain_straggler_gap_s"] >= 0.02
    assert metrics["repack/group_source_drain_wait_s/min"] <= metrics["repack/group_source_drain_wait_s/max"]
    assert metrics["repack/group_source_drain_wait_s/max"] == metrics["repack/group_drain_critical_path_s"]


@pytest.mark.asyncio
async def test_background_schedule_excludes_every_active_group_participant() -> None:
    first_group = _committed_group(sources=("s0",), destination="d0")
    router = _FakeRouter([_router_response(groups=[first_group]), _router_response()])
    router.drain_status["s0"] = False
    manager = _manager(
        router=router,
        servers={"s0": _FakeServer(), "d0": _FakeServer(), "other": _FakeServer()},
    )

    await manager._schedule_repack_once()
    operations = set(manager._repack_tasks)
    assert len(operations) == 1
    await asyncio.sleep(0)
    await manager._schedule_repack_once()
    assert set(manager._repack_tasks) == operations
    assert router.plan_and_begin_repack.calls[1][1]["excluded_servers"] == ["d0", "s0"]
    await manager._cancel_repack_tasks()


@pytest.mark.asyncio
async def test_repack_disabled_is_router_option_and_keeps_idle_updates() -> None:
    response = _router_response(
        idle_sources=["s0"],
    )
    router = _FakeRouter([response])
    server = _FakeServer(loaded_version=0)
    manager = _manager(router=router, servers={"s0": server}, version=0, repack_enabled=False)

    assert await _run_scheduled_pass(manager) == 1

    assert router.plan_and_begin_repack.calls[0][1]["repack_enabled"] is False
    assert router.try_mark_updating.calls == []
    assert server.load_weights.calls == [((), {"wait_for_version": 0})]
    assert router.mark_active.calls[0][1] == {"server_id": "s0", "weight_version": 0}


@pytest.mark.asyncio
async def test_non_bootstrap_idle_update_reuses_router_updating_claim() -> None:
    response = _router_response(
        idle_sources=["s0"],
    )
    router = _FakeRouter([response])
    manager = _manager(router=router, servers={"s0": _FakeServer()})

    assert await _run_scheduled_pass(manager) == 1

    assert router.try_mark_updating.calls == []
    assert router.mark_active.calls[0][1] == {"server_id": "s0", "weight_version": 2}


@pytest.mark.asyncio
async def test_empty_router_plan_is_normal_without_loading() -> None:
    router = _FakeRouter([_router_response()])
    server = _FakeServer()
    manager = _manager(router=router, servers={"s0": server})

    assert await _run_scheduled_pass(manager) == 0

    metrics = manager.get_metrics()
    assert metrics["repack/no_plan_calls"] == 1
    assert metrics["repack/transfer_failed"] == 0
    assert server.load_weights.calls == []


@pytest.mark.asyncio
async def test_plan_and_begin_failure_is_fatal() -> None:
    router = _FakeRouter([RuntimeError("planning failed")])
    manager = _manager(router=router, servers={"s0": _FakeServer()})

    with pytest.raises(RuntimeError, match="planning failed"):
        await _run_scheduled_pass(manager)

    metrics = manager.get_metrics()
    assert metrics["repack/plan_and_begin_failed"] == 1
    assert metrics["repack/transfer_failed"] == 1


@pytest.mark.asyncio
async def test_invalid_router_response_fails_fast_without_local_replanning() -> None:
    router = _FakeRouter([{"groups": [_committed_group()]}])
    manager = _manager(router=router, servers={"s0": _FakeServer(), "d0": _FakeServer()})

    with pytest.raises(RuntimeError, match="invalid plan-and-begin response"):
        await _run_scheduled_pass(manager)

    assert router.plan_and_begin_repack.calls
    assert manager.get_metrics()["repack/plan_and_begin_failed"] == 1


@pytest.mark.asyncio
async def test_repack_waits_persistently_for_source_drain_before_loading() -> None:
    router = _FakeRouter([_router_response(groups=[_committed_group()])])
    router.drain_status["s0"] = False
    source = _FakeServer()
    manager = _manager(router=router, servers={"s0": source, "d0": _FakeServer()})

    task = asyncio.create_task(_run_scheduled_pass(manager))
    while not router.try_mark_updating.calls:
        await asyncio.sleep(0)
    await asyncio.sleep(0.02)

    assert not task.done()
    assert source.load_weights.calls == []
    assert len(router.try_mark_updating.calls) == 1
    assert router.mark_active.calls == []
    assert not hasattr(router, "restore_active")

    router.drain_status["s0"] = True
    assert await asyncio.wait_for(task, timeout=1) == 1

    assert len(source.load_weights.calls) == 1
    assert len(router.try_mark_updating.calls) >= 2
    assert len(router.mark_active.calls) == 1
    assert len(router.server_handles["s0"].abort_all_requests.calls) >= 2
    assert "repack/source_drain_timeout" not in manager.get_metrics()


@pytest.mark.asyncio
async def test_source_drain_timeout_and_cancellation_stop_backend_wait(monkeypatch):
    router = _FakeRouter()
    manager = _manager(router=router, servers={"s0": _FakeServer()})
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def abort(*, abort_only):
        assert abort_only
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    router.server_handles["s0"] = SimpleNamespace(abort_all_requests=_RemoteMethod(func=abort))
    monkeypatch.setattr(manager_module, "_DRAIN_TIMEOUT_S", 0.02)
    with pytest.raises(TimeoutError):
        await manager._wait_for_source_drain("s0", expected_version=1)
    assert cancelled.is_set() and router.try_mark_updating.calls == []

    started.clear()
    cancelled.clear()
    monkeypatch.setattr(manager_module, "_DRAIN_TIMEOUT_S", 10)
    pending = asyncio.create_task(manager._wait_for_source_drain("s0", expected_version=1))
    await started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert cancelled.is_set() and router.try_mark_updating.calls == []


@pytest.mark.asyncio
async def test_backend_abort_failure_does_not_load_weights_or_reopen_source():
    router = _FakeRouter()
    source = _FakeServer()
    manager = _manager(router=router, servers={"s0": source})
    router.server_handles["s0"] = SimpleNamespace(
        abort_all_requests=_RemoteMethod(exception=RuntimeError("backend abort failed"))
    )
    with pytest.raises(RuntimeError, match="backend abort failed"):
        await manager._execute_repack_source("s0", [], expected_version=1, loadable_version=2)
    assert router.try_mark_updating.calls == []
    assert source.load_weights.calls == [] and router.mark_active.calls == []


@pytest.mark.asyncio
async def test_lifecycle_task_admission_rejects_overlapping_replica_ownership():
    manager = _manager(router=_FakeRouter(), servers={"s0": _FakeServer()})

    async def operation():
        await asyncio.Event().wait()

    manager._start_operation(operation(), name="first", participants=frozenset({"s0"}))
    with pytest.raises(RuntimeError, match="distinct registered replicas"):
        manager._start_operation(operation(), name="overlap", participants=frozenset({"s0"}))
    assert len(manager._repack_tasks) == 1
    await manager._cancel_repack_tasks()


@pytest.mark.asyncio
async def test_load_failure_propagates_without_mark_unavailable_fallback() -> None:
    router = _FakeRouter([_router_response(groups=[_committed_group()])])
    manager = _manager(
        router=router,
        servers={"s0": _FakeServer(exception=RuntimeError("load failed")), "d0": _FakeServer()},
    )

    with pytest.raises(RuntimeError, match="load failed"):
        await _run_scheduled_pass(manager)

    assert not hasattr(router, "mark_unavailable")
    metrics = manager.get_metrics()
    assert metrics["repack/load_weights_failed"] == 1
    assert metrics["repack/bins_failed"] == 1


@pytest.mark.asyncio
async def test_load_result_requires_exact_integer_version() -> None:
    server = _FakeServer()
    server.load_weights = _RemoteMethod(result={"weight_version": 2})
    manager = _manager(router=_FakeRouter([]), servers={"s0": server})

    with pytest.raises(RuntimeError, match="invalid loaded version"):
        await manager._load_source_weights(source="s0", loadable_version=2)

    assert manager.get_metrics()["repack/load_weights_failed"] == 1


def test_manager_connect_requires_relay_loadable_version_source() -> None:
    manager = MultiVersionRolloutManager(MultiVersionRolloutManagerConfig())
    with pytest.raises(ValueError, match="router and relay_client"):
        manager.connect(servers={"s0": _FakeServer()}, router=_FakeRouter([]), relay_client=None)


@pytest.mark.asyncio
async def test_invalid_loadable_version_fails_before_calling_router() -> None:
    router = _FakeRouter([])
    manager = _manager(router=router, servers={"s0": _FakeServer()}, version=None)

    with pytest.raises(RuntimeError, match="invalid loadable version"):
        await _run_scheduled_pass(manager)
    assert router.plan_and_begin_repack.calls == []


@pytest.mark.asyncio
async def test_dead_engine_recovery_is_single_flight_and_does_not_block_healthy_updates():
    dead = _FakeServer()
    dead.snapshot.result = SimpleNamespace(backend_health="unavailable")
    restart_started = asyncio.Event()
    restart_release = asyncio.Event()

    async def restart(**kwargs):
        restart_started.set()
        await restart_release.wait()
        dead.snapshot.result = SimpleNamespace(backend_health="healthy")
        return 8

    dead.restart = _RemoteMethod(func=restart)
    router = _FakeRouter()
    router.failed_server_ids.result = ["s0"]
    manager = MultiVersionRolloutManager(MultiVersionRolloutManagerConfig())
    manager.connect(servers={"s0": dead, "s1": _FakeServer()}, router=router, relay_client=_FakeRelay(4))
    await manager._schedule_recoveries()
    await restart_started.wait()
    await manager._schedule_recoveries()
    await manager._execute_idle_update(source="s1", loadable_version=4)
    assert len(dead.restart.calls) == 1
    assert router.mark_active.calls[-1][1]["server_id"] == "s1"
    restart_release.set()
    await asyncio.gather(*manager._repack_tasks)
    await manager._reap_repack_tasks()
    # A delayed unavailable Router observation must not restart the fresh engine.
    await manager._schedule_recoveries()
    assert len(dead.restart.calls) == 1
    assert router.finish_recovery.calls == [((), dict(server_id="s0", handle="rebuilt-handle", weight_version=8))]
    assert manager.get_metrics()["recovery/succeeded"] == 1


@pytest.mark.asyncio
async def test_engine_death_during_weight_loading_uses_recovery():
    dead = _FakeServer(exception=RuntimeError("engine died during load"))
    dead.snapshot.result = SimpleNamespace(backend_health="unavailable")
    router = _FakeRouter()
    manager = MultiVersionRolloutManager(MultiVersionRolloutManagerConfig())
    manager.connect(servers={"s0": dead}, router=router, relay_client=_FakeRelay(4))
    await manager._execute_idle_update(source="s0", loadable_version=4)
    assert len(router.begin_recovery.calls) == 1
    assert len(router.finish_recovery.calls) == 1
    assert not router.mark_active.calls


@pytest.mark.asyncio
async def test_restart_failure_stays_quarantined_and_surfaces():
    dead = _FakeServer()
    dead.snapshot.result = SimpleNamespace(backend_health="unavailable")
    dead.restart.exception = RuntimeError("old GPU worker still alive")
    router = _FakeRouter()
    router.failed_server_ids.result = ["s0"]
    manager = MultiVersionRolloutManager(MultiVersionRolloutManagerConfig())
    manager.connect(servers={"s0": dead}, router=router, relay_client=_FakeRelay(4))
    await manager._schedule_recoveries()
    await asyncio.gather(*manager._repack_tasks, return_exceptions=True)
    with pytest.raises(RuntimeError, match="old GPU worker still alive"):
        await manager._reap_repack_tasks()
    assert not router.finish_recovery.calls
    assert manager.get_metrics()["recovery/failed"] == 1
