# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from laminar.global_request_router import GlobalRequestRouter, GlobalRequestRouterConfig, RouteUnavailableError
from laminar.global_request_router.router import ACTIVE, DRAINING, UPDATING


class _Server:
    def __init__(self):
        self.abort_calls = []
        self.abort_all_requests = SimpleNamespace(remote=self._abort)

    async def _abort(self, **kwargs):
        self.abort_calls.append(kwargs)


def _snapshot(server_id, *, running=0, waiting=0, version=1):
    return {
        "server_id": server_id,
        "backend_health": "healthy",
        "kv_cache_usage": 0.1,
        "num_running_requests": running,
        "num_waiting_requests": waiting,
        "snapshot_version": version,
    }


def _router(*, latest=False, count=1):
    servers = {f"s{index}": _Server() for index in range(count)}
    router = GlobalRequestRouter(
        servers,
        config=GlobalRequestRouterConfig(continuation="latest_version" if latest else "same_version"),
    )
    for server_id in servers:
        router.mark_active(server_id, weight_version=7)
    router.update_routing_snapshot({server_id: _snapshot(server_id) for server_id in servers})
    return router


def test_colocate_pause_drops_routes_but_waits_for_real_lease_release() -> None:
    router = _router(latest=True)
    router.bind_route("session")
    router.acquire_lease("session", lease_id="running")
    assert router.begin_colocate_pause() == {"s0": 7}
    assert router._session_id_to_server == {}
    with pytest.raises(RouteUnavailableError) as error:
        router.acquire_lease("session", lease_id="next")
    assert error.value.planned_pause
    with pytest.raises(RouteUnavailableError):
        router.try_mark_updating("s0", expected_version=7)
    router.release_lease("session", "running")
    router.try_mark_updating("s0", expected_version=7)
    assert router._server_states["s0"].lifecycle == UPDATING


def test_planned_pause_clears_only_after_the_last_replica_reopens() -> None:
    router = _router(latest=True, count=2)
    router.begin_colocate_pause()
    router.try_mark_updating("s0", expected_version=7)
    router.try_mark_updating("s1", expected_version=7)
    router.mark_active("s0", weight_version=8)
    with pytest.raises(RouteUnavailableError) as error:
        router.bind_route("early")
    assert error.value.planned_pause
    router.mark_active("s1", weight_version=8)
    router.bind_route("ready")


def test_colocate_pause_rejects_same_version_and_incomplete_pool() -> None:
    with pytest.raises(ValueError, match="latest_version"):
        _router().begin_colocate_pause()
    router = _router(latest=True, count=2)
    router.begin_recovery("s1")
    with pytest.raises(RuntimeError, match="entire rollout pool"):
        router.begin_colocate_pause()
    assert router._server_states["s0"].lifecycle == ACTIVE


def test_route_and_lease_are_distinct_drain_preconditions() -> None:
    router = _router()
    router.bind_route("session", min_weight_version=7)
    router.acquire_lease("session", lease_id="running", weight_version=7)
    router._server_states["s0"].lifecycle = DRAINING
    router.release_lease("session", "running")
    with pytest.raises(RouteUnavailableError):
        router.try_mark_updating("s0", expected_version=7)
    router.release_route("session", weight_version=7)
    router.try_mark_updating("s0", expected_version=7)
    assert router._server_states["s0"].lifecycle == UPDATING


def test_runtime_zero_cannot_replace_a_lease_release_ack() -> None:
    router = _router(latest=True)
    router.bind_route("session")
    router.acquire_lease("session", lease_id="running")
    router.begin_colocate_pause()
    router.update_routing_snapshot({"s0": _snapshot("s0", version=2)})
    with pytest.raises(RouteUnavailableError):
        router.try_mark_updating("s0", expected_version=7)
    assert router._inflight_count("s0") == 1
    router.release_lease("session", "running")
    router.try_mark_updating("s0", expected_version=7)


def test_backend_runtime_activity_is_an_independent_drain_precondition() -> None:
    router = _router()
    router._server_states["s0"].lifecycle = DRAINING
    router.update_routing_snapshot({"s0": _snapshot("s0", running=1, version=2)})
    with pytest.raises(RouteUnavailableError):
        router.try_mark_updating("s0", expected_version=7)
    router.update_routing_snapshot({"s0": _snapshot("s0", version=3)})
    router.try_mark_updating("s0", expected_version=7)
