# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pytest

from laminar.global_request_router import (
    GlobalRequestRouter,
    GlobalRequestRouterConfig,
    RouteUnavailableError,
    SessionRouteClosedError,
)


def _snapshot(server_id: str, *, available: bool = True, version: int = 1):
    return {
        "server_id": server_id,
        "backend_health": "healthy" if available else "unavailable",
        "kv_cache_usage": 0.1,
        "num_waiting_requests": 0,
        "num_running_requests": 0,
        "snapshot_version": version,
        "timestamp": float(version),
    }


def _router(*, latest: bool, versions: dict[str, int]) -> GlobalRequestRouter:
    router = GlobalRequestRouter(
        servers=dict.fromkeys(versions),
        config=GlobalRequestRouterConfig(continuation="latest_version" if latest else "same_version"),
    )
    for server_id, version in versions.items():
        router.mark_active(server_id, weight_version=version)
    router.update_routing_snapshot({server_id: _snapshot(server_id) for server_id in versions})
    return router


@pytest.mark.parametrize("latest", [False, True])
def test_minimum_bind_and_acquire_use_the_current_ready_cohort(latest: bool) -> None:
    versions = {"higher": 7, "lower": 3, "unavailable": 11, "middle": 5}
    router = _router(latest=latest, versions=versions)
    router.update_routing_snapshot(
        {server_id: _snapshot(server_id, available=server_id != "unavailable", version=2) for server_id in versions}
    )
    assert router.get_routable_version(2) == 3
    assert router.bind_route("first", min_weight_version=2) == (None if latest else 3)
    first, _ = router.acquire_lease("first", lease_id="first")
    assert first == ("higher" if latest else "lower")
    router.release_lease("first", "first")
    assert router.bind_route("second", min_weight_version=4) == (None if latest else 5)
    second, _ = router.acquire_lease("second", lease_id="second")
    assert second == ("higher" if latest else "middle")


def test_latest_rechecks_the_whole_cohort_on_every_short_attempt() -> None:
    router = _router(latest=True, versions={"lower": 1, "higher": 1})
    assert set(router.plan_and_begin_repack(loadable_version=2, repack_enabled=False).idle_sources) == {
        "lower",
        "higher",
    }
    with pytest.raises(RouteUnavailableError):
        router.bind_route("session", min_weight_version=2)
    router.mark_active("lower", weight_version=3)
    with pytest.raises(RouteUnavailableError):
        router.bind_route("session", min_weight_version=2)
    router.mark_active("higher", weight_version=3)
    assert router.bind_route("session", min_weight_version=2) is None
    assert router.acquire_lease("session", lease_id="lease")[0] in {"lower", "higher"}


@pytest.mark.parametrize(
    ("session_id", "lease_id", "weight_version", "error"),
    [
        ("", "lease", None, "session_id"),
        ("missing", "lease", None, "not bound"),
        ("session", "", None, "lease_id"),
        ("session", "lease", True, "built-in non-negative integer"),
        ("session", "lease", 3, "got a pin"),
    ],
)
def test_latest_acquire_validates_identity_and_pin(
    session_id: str,
    lease_id: str,
    weight_version: object,
    error: str,
) -> None:
    router = _router(latest=True, versions={"only": 3})
    if session_id == "session":
        router.bind_route(session_id)
    exception = SessionRouteClosedError if error == "not bound" else ValueError
    with pytest.raises(exception, match=error):
        router.acquire_lease(session_id, lease_id=lease_id, weight_version=weight_version)
    assert sum(len(session.leases) for session in router._sessions.values()) == 0


@pytest.mark.parametrize("operation", ["version", "bind"])
def test_unavailable_attempt_cannot_commit_when_weights_arrive_later(operation: str) -> None:
    router = GlobalRequestRouter(
        servers={"only": None},
        config=GlobalRequestRouterConfig(continuation="latest_version"),
    )
    with pytest.raises(RouteUnavailableError):
        if operation == "version":
            router.get_routable_version(0)
        else:
            router.bind_route("failed", min_weight_version=0)
    router.mark_active("only", weight_version=0)
    assert router._session_id_to_server == {}
    assert router.get_active_session_count() == 0
