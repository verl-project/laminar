# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from laminar.global_request_router import GlobalRequestRouter, GlobalRequestRouterConfig, RouteUnavailableError
from laminar.global_request_router.router import DRAINING


def _snapshot(server_id: str, *, kv: float = 0.1, healthy: bool = True, version: int = 1):
    return {
        "server_id": server_id,
        "backend_health": "healthy" if healthy else "unavailable",
        "max_num_seqs": 8,
        "kv_cache_usage": kv,
        "num_waiting_requests": 0,
        "num_running_requests": 0,
        "token_counts": {},
        "snapshot_version": version,
        "timestamp": float(version),
    }


def _router(versions: dict[str, int]) -> GlobalRequestRouter:
    router = GlobalRequestRouter({server_id: object() for server_id in versions})
    for server_id, version in versions.items():
        router.mark_active(server_id, weight_version=version)
    router.update_routing_snapshot({server_id: _snapshot(server_id) for server_id in versions})
    return router


def test_bind_prefers_lower_kv_then_balances_route_occupancy() -> None:
    router = _router({"s0": 7, "s1": 7})
    router.update_routing_snapshot({"s0": _snapshot("s0", kv=0.7), "s1": _snapshot("s1", kv=0.1)})
    router.bind_route("one", min_weight_version=7)
    assert router._session_id_to_server["one"].server_id == "s1"

    router.update_routing_snapshot({"s0": _snapshot("s0", kv=0.1, version=2), "s1": _snapshot("s1", kv=0.1, version=2)})
    router.bind_route("two", min_weight_version=7)
    assert router._session_id_to_server["two"].server_id == "s0"
    router.release_route("one", weight_version=7)
    router.bind_route("three", min_weight_version=7)
    assert router._session_id_to_server["three"].server_id == "s1"


def test_bind_minimum_resolves_the_lowest_routable_version() -> None:
    router = _router({"v8": 8, "v9": 9})
    assert router.bind_route("session", min_weight_version=7) == 8
    server_id, _ = router.acquire_lease("session", lease_id="lease", weight_version=8)
    assert server_id == "v8"


def test_bound_session_is_sticky_and_rejects_a_conflicting_version() -> None:
    router = _router({"s0": 7, "s1": 7})
    assert router.bind_route("session", min_weight_version=7) == 7
    first = router.acquire_lease("session", lease_id="one", weight_version=7)
    router.release_lease("session", "one")
    assert router.acquire_lease("session", lease_id="two", weight_version=7)[0] == first[0]
    with pytest.raises(RuntimeError, match="version conflicts"):
        router.acquire_lease("session", lease_id="wrong", weight_version=8)
    assert router._inflight_count(first[0]) == 1


def test_fail_fast_bind_does_not_open_a_session_without_a_routable_server() -> None:
    router = GlobalRequestRouter({"s0": object()})
    with pytest.raises(RouteUnavailableError):
        router.bind_route("session", min_weight_version=7)
    assert router.get_active_session_count() == 0


def test_existing_draining_route_can_finish_but_new_routes_cannot_use_it() -> None:
    router = _router({"s0": 7})
    router.bind_route("existing", min_weight_version=7)
    router._server_states["s0"].lifecycle = DRAINING
    assert router.acquire_lease("existing", lease_id="lease", weight_version=7)[0] == "s0"
    with pytest.raises(RouteUnavailableError):
        router.bind_route("new", min_weight_version=7)


def test_unhealthy_bound_route_fails_without_committing_a_lease() -> None:
    router = _router({"s0": 7})
    router.bind_route("session", min_weight_version=7)
    router.update_routing_snapshot({"s0": _snapshot("s0", healthy=False, version=2)})
    with pytest.raises(RouteUnavailableError):
        router.acquire_lease("session", lease_id="lease", weight_version=7)
    assert router._sessions["session"].leases == {}


def test_recovery_replaces_handle_but_old_committed_lease_remains_releasable() -> None:
    original = object()
    replacement = object()
    router = GlobalRequestRouter({"s0": original, "s1": object()})
    for server_id in ("s0", "s1"):
        router.mark_active(server_id, weight_version=7)
    router.update_routing_snapshot({server_id: _snapshot(server_id) for server_id in ("s0", "s1")})
    router.bind_route("old", min_weight_version=7)
    old_server, _ = router.acquire_lease("old", lease_id="old-lease", weight_version=7)
    router.begin_recovery(old_server)
    router.finish_recovery(old_server, replacement, weight_version=8)
    router.update_routing_snapshot(
        {
            old_server: _snapshot(old_server, version=2),
            ("s1" if old_server == "s0" else "s0"): _snapshot("s1" if old_server == "s0" else "s0", version=2),
        }
    )
    router.bind_route("new", min_weight_version=8)
    assert router.acquire_lease("new", lease_id="new-lease", weight_version=8) == (old_server, replacement)
    assert router._inflight_count(old_server) == 2
    router.release_lease("old", "old-lease")
    router.release_lease("new", "new-lease")
    assert router._inflight_count(old_server) == 0


def test_get_routable_version_is_an_immediate_check() -> None:
    router = _router({"s0": 7})
    assert router.get_routable_version(6) == 7
    with pytest.raises(RouteUnavailableError):
        router.get_routable_version(8)
    router.mark_active("s0", weight_version=8)
    assert router.get_routable_version(8) == 8


def test_configuration_keeps_only_policy_not_actor_concurrency() -> None:
    config = GlobalRequestRouterConfig()
    assert config.continuation == "same_version"
    assert not hasattr(config, "max_concurrency")
