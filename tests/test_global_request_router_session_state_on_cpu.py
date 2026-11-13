# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pytest

from laminar.global_request_router import (
    GlobalRequestRouter,
    GlobalRequestRouterConfig,
    RouteUnavailableError,
    SessionRouteClosedError,
)


def _router(*, ready: bool = True, latest: bool = False) -> GlobalRequestRouter:
    router = GlobalRequestRouter(
        {"s0": object(), "s1": object()},
        config=GlobalRequestRouterConfig(continuation="latest_version" if latest else "same_version"),
    )
    if ready:
        for server_id in ("s0", "s1"):
            router.mark_active(server_id, weight_version=1)
    return router


def _inflight(router: GlobalRequestRouter) -> int:
    return sum(router._inflight_count(server_id) for server_id in router._server_states)


def test_acquire_requires_an_explicit_live_session_and_never_auto_binds() -> None:
    router = _router()
    with pytest.raises(SessionRouteClosedError):
        router.acquire_lease("absent", lease_id="late")
    assert router.get_active_session_count() == 0
    assert router._session_id_to_server == {}
    assert _inflight(router) == 0


def test_unavailable_bind_and_acquire_leave_no_pending_state() -> None:
    router = _router(ready=False, latest=True)
    with pytest.raises(RouteUnavailableError):
        router.bind_route("session")
    assert router.get_active_session_count() == 0

    router.mark_active("s0", weight_version=1)
    router.bind_route("session")
    router.begin_colocate_pause()
    for _ in range(3):
        with pytest.raises(RouteUnavailableError):
            router.acquire_lease("session", lease_id="retry")
    assert _inflight(router) == 0
    assert router._sessions["session"].leases == {}


def test_committed_lease_and_release_are_idempotent() -> None:
    router = _router()
    router.bind_route("session")
    first = router.acquire_lease("session", lease_id="lease", weight_version=1)
    assert router.acquire_lease("session", lease_id="lease", weight_version=1) == first
    assert _inflight(router) == 1
    router.release_lease("session", "lease")
    router.release_lease("session", "lease")
    assert _inflight(router) == 0


def test_two_leases_release_independently_and_preserve_the_route() -> None:
    router = _router()
    router.bind_route("session")
    first = router.acquire_lease("session", lease_id="first")
    second = router.acquire_lease("session", lease_id="second")
    assert first[0] == second[0]
    assert _inflight(router) == 2
    router.release_lease("session", "first")
    assert _inflight(router) == 1
    assert "session" in router._session_id_to_server
    router.release_lease("session", "second")
    router.release_route("session")
    assert _inflight(router) == 0
    assert router.get_active_session_count() == 0


def test_close_rejects_live_lease_and_preserves_session_for_retry() -> None:
    router = _router()
    router.bind_route("session")
    router.acquire_lease("session", lease_id="running")
    with pytest.raises(RuntimeError, match="still owns 1 request lease"):
        router.release_route("session")
    assert router.get_active_session_count() == 1
    assert _inflight(router) == 1
    router.release_lease("session", "running")
    router.release_route("session")
    assert router.get_active_session_count() == 0


def test_route_can_be_dropped_while_its_committed_lease_still_releases() -> None:
    router = _router(latest=True)
    router.bind_route("session")
    server_id, _ = router.acquire_lease("session", lease_id="running")
    router.begin_colocate_pause()
    assert router._session_id_to_server == {}
    assert router._inflight_count(server_id) == 1
    router.release_lease("session", "running")
    router.release_route("session")
    assert _inflight(router) == 0


def test_closed_session_rejects_a_late_acquire_without_reopening() -> None:
    router = _router()
    router.bind_route("session")
    router.release_route("session")
    with pytest.raises(SessionRouteClosedError):
        router.acquire_lease("session", lease_id="late")
    router.release_lease("session", "late")
    assert router.get_active_session_count() == 0


def test_release_is_scoped_by_session_and_lease() -> None:
    router = _router()
    router.bind_route("one")
    router.bind_route("two")
    router.acquire_lease("one", lease_id="same")
    router.acquire_lease("two", lease_id="same")
    router.release_lease("one", "same")
    assert _inflight(router) == 1
    assert router._sessions["two"].leases
