# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pytest

from laminar.global_request_router import GlobalRequestRouter, GlobalRequestRouterConfig
from laminar.global_request_router.runtime import kv_below_threshold


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _snapshot(
    server_id: str,
    *,
    kv: float,
    snapshot_version: int,
) -> dict:
    return {
        "server_id": server_id,
        "backend_health": "healthy",
        "max_num_seqs": 1024,
        "kv_cache_usage": kv,
        "kv_cache_usage_change_rate": 0.0,
        "num_waiting_requests": 0,
        "num_running_requests": 0,
        "num_preemptions": 0,
        "token_counts": {},
        "snapshot_version": snapshot_version,
        "timestamp": float(snapshot_version),
    }


def _router(
    clock: _FakeClock,
    *,
    mode: str,
    servers: tuple[str, ...] = ("s0",),
    dynamic_overrides: dict | None = None,
) -> GlobalRequestRouter:
    dynamic_config = {
        "mode": mode,
        "max_snapshot_age_s": 5.0,
        "kv_cache_usage_threshold": 0.8,
    }
    dynamic_config.update(dynamic_overrides or {})
    router = GlobalRequestRouter(
        servers={server_id: None for server_id in servers},
        config=GlobalRequestRouterConfig(
            dynamic_overload_admission=dynamic_config,
        ),
        monotonic_fn=clock,
    )
    for server_id in servers:
        router.mark_active(server_id=server_id, weight_version=1)
    return router


@pytest.mark.parametrize(
    ("kv_cache_usage", "expected"),
    ((None, False), (0.79, True), (0.80, False), (0.95, False)),
)
def test_kv_below_threshold_is_shared_stateless_predicate(kv_cache_usage, expected) -> None:
    assert kv_below_threshold(kv_cache_usage, 0.80) is expected


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("mode", "invalid", "mode must be one of"),
        ("max_snapshot_age_s", 0, "max_snapshot_age_s must be positive"),
        ("kv_cache_usage_threshold", 1.1, "must be a number in"),
        ("resume_kv_cache_usage", 0.65, "unknown dynamic overload admission fields"),
    ),
)
def test_router_rejects_invalid_dynamic_overload_config(field, value, error) -> None:
    with pytest.raises(ValueError, match=error):
        _router(_FakeClock(), mode="off", dynamic_overrides={field: value})


def test_bind_route_prefers_runtime_open_server() -> None:
    clock = _FakeClock()
    router = _router(clock, mode="enforce", servers=("s0", "s1"))
    router.update_routing_snapshot(
        {
            "s0": _snapshot("s0", kv=0.95, snapshot_version=1),
            "s1": _snapshot("s1", kv=0.10, snapshot_version=1),
        }
    )

    router.bind_route("route-0", min_weight_version=1)

    assert router._session_id_to_server["route-0"].server_id == "s1"
    assert all(router._inflight_count(server_id) == 0 for server_id in router._server_states)


def test_bind_route_falls_back_to_base_server_when_runtime_blocks_all() -> None:
    clock = _FakeClock()
    router = _router(clock, mode="enforce")
    router.update_routing_snapshot({"s0": _snapshot("s0", kv=0.95, snapshot_version=1)})

    router.bind_route("route-0", min_weight_version=1)

    assert router._session_id_to_server["route-0"].server_id == "s0"
    assert router._inflight_count("s0") == 0


def test_bind_route_rejects_missing_base_version_without_mutation() -> None:
    clock = _FakeClock()
    router = _router(clock, mode="enforce")
    router.update_routing_snapshot({"s0": _snapshot("s0", kv=0.1, snapshot_version=1)})

    with pytest.raises(RuntimeError, match="No available servers"):
        router.bind_route("route-0", min_weight_version=2)

    assert router._session_id_to_server == {}


def test_off_mode_binds_without_fresh_runtime_snapshot() -> None:
    router = _router(_FakeClock(), mode="off")

    router.bind_route("route-0", min_weight_version=1)

    assert router._session_id_to_server["route-0"].server_id == "s0"


def test_stateless_threshold_applies_to_each_fresh_snapshot() -> None:
    clock = _FakeClock()
    router = _router(clock, mode="enforce")
    router.update_routing_snapshot({"s0": _snapshot("s0", kv=0.95, snapshot_version=1)})
    assert router._runtime.admission_open("s0") is False

    clock.advance(1)
    router.update_routing_snapshot({"s0": _snapshot("s0", kv=0.8, snapshot_version=2)})
    assert router._runtime.admission_open("s0") is False

    clock.advance(1)
    router.update_routing_snapshot({"s0": _snapshot("s0", kv=0.799, snapshot_version=3)})
    assert router._runtime.admission_open("s0") is True


def test_older_runtime_snapshot_does_not_replace_newer_server_state() -> None:
    clock = _FakeClock()
    router = _router(clock, mode="enforce")
    router.update_routing_snapshot({"s0": _snapshot("s0", kv=0.95, snapshot_version=2)})
    router.update_routing_snapshot({"s0": _snapshot("s0", kv=0.1, snapshot_version=1)})

    assert router._runtime.server_state("s0").kv_cache_usage == 0.95
    assert router._runtime.snapshot_version == 2


def test_stale_snapshot_blocks_runtime_preference_but_not_session_bind() -> None:
    clock = _FakeClock()
    router = _router(clock, mode="enforce")
    router.update_routing_snapshot({"s0": _snapshot("s0", kv=0.1, snapshot_version=1)})
    clock.advance(6)

    router.bind_route("route-0", min_weight_version=1)

    assert router._runtime.admission_open("s0") is False
    assert router._session_id_to_server["route-0"].server_id == "s0"


def test_existing_and_new_routes_continue_after_runtime_overload() -> None:
    clock = _FakeClock()
    router = _router(clock, mode="enforce")
    router.update_routing_snapshot({"s0": _snapshot("s0", kv=0.1, snapshot_version=1)})
    router.bind_route("sticky", min_weight_version=1)
    assert router.acquire_lease("sticky", lease_id="first", weight_version=1)[0] == "s0"
    router.release_lease("sticky", "first")

    clock.advance(1)
    router.update_routing_snapshot({"s0": _snapshot("s0", kv=0.95, snapshot_version=2)})

    assert router.acquire_lease("sticky", lease_id="second", weight_version=1)[0] == "s0"
    router.release_lease("sticky", "second")
    router.bind_route("new-session", min_weight_version=1)
    assert router.acquire_lease("new-session", lease_id="new", weight_version=1)[0] == "s0"
    router.release_lease("new-session", "new")
