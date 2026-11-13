# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from laminar.global_request_router import GlobalRequestRouter, GlobalRequestRouterConfig, RouteUnavailableError
from laminar.global_request_router.router import DRAINING, UPDATING
from laminar.multi_version_rollout import CommittedAbortSource


def _snapshot(server_id: str, *, running: int = 0, healthy: bool = True, version: int = 1):
    return {
        "server_id": server_id,
        "backend_health": "healthy" if healthy else "unavailable",
        "max_num_seqs": 8,
        "kv_cache_usage": 0.1,
        "num_waiting_requests": 0,
        "num_running_requests": running,
        "token_counts": {},
        "snapshot_version": version,
        "timestamp": float(version),
    }


def _router(versions: dict[str, int]) -> GlobalRequestRouter:
    router = GlobalRequestRouter(
        {server_id: SimpleNamespace() for server_id in versions},
        config=GlobalRequestRouterConfig(continuation="latest_version"),
    )
    for server_id, version in versions.items():
        router.mark_active(server_id, weight_version=version)
    router.update_routing_snapshot({server_id: _snapshot(server_id) for server_id in versions})
    return router


def test_latest_session_reuses_a_newest_route_until_the_cohort_changes() -> None:
    router = _router({"old": 1, "new": 2})
    assert router.bind_route("session", min_weight_version=0) is None
    first, _ = router.acquire_lease("session", lease_id="one")
    router.release_lease("session", "one")
    assert first == "new"
    assert router.acquire_lease("session", lease_id="two")[0] == first


def test_latest_rejects_pinned_acquire() -> None:
    router = _router({"s0": 1})
    router.bind_route("session")
    with pytest.raises(ValueError, match="newest version"):
        router.acquire_lease("session", lease_id="lease", weight_version=1)


def test_latest_update_drops_routes_but_preserves_source_leases() -> None:
    router = _router({"s0": 1, "s1": 1})
    router.bind_route("session")
    source, _ = router.acquire_lease("session", lease_id="running")
    plan = router.plan_and_begin_repack(loadable_version=2)
    assert plan.abort_sources
    assert all(isinstance(item, CommittedAbortSource) for item in plan.abort_sources)
    assert router._session_id_to_server == {}
    assert router._inflight_count(source) == 1
    router.release_lease("session", "running")
    assert router._inflight_count(source) == 0


def test_latest_cohort_reopens_together_after_update() -> None:
    router = _router({"s0": 1, "s1": 1})
    plan = router.plan_and_begin_repack(loadable_version=2, repack_enabled=False)
    participants = set(plan.idle_sources) | {source.server_id for source in plan.abort_sources}
    assert participants == {"s0", "s1"}
    assert all(router._server_states[server_id].lifecycle in {DRAINING, UPDATING} for server_id in participants)
    router.mark_active("s0", weight_version=2)
    with pytest.raises(RouteUnavailableError):
        router.bind_route("early")
    router.mark_active("s1", weight_version=2)
    router.bind_route("ready")
    assert router._session_id_to_server["ready"].weight_version == 2


def test_latest_bind_is_fail_fast_while_every_replica_is_unavailable() -> None:
    router = _router({"s0": 1})
    router.update_routing_snapshot({"s0": _snapshot("s0", healthy=False, version=2)})
    with pytest.raises(RouteUnavailableError):
        router.bind_route("session")
    assert router.get_active_session_count() == 0


def test_latest_planned_pause_marks_failures_for_the_longer_client_budget() -> None:
    router = _router({"s0": 1})
    router.begin_colocate_pause()
    with pytest.raises(RouteUnavailableError) as error:
        router.bind_route("session")
    assert error.value.planned_pause is True


def test_latest_routable_version_reports_the_lowest_current_cohort_version() -> None:
    router = _router({"v2": 2, "v3": 3})
    assert router.get_routable_version(2) == 2
    assert router.bind_route("session", min_weight_version=2) is None
    assert router._session_id_to_server["session"].weight_version == 3
