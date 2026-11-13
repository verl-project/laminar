# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import ray

from laminar.eval.router import (
    ValidationRequestRouter,
    validation_replica_groups,
)
from laminar.global_request_router import GlobalRequestRouterConfig, RouteUnavailableError
from laminar.multi_version_rollout.lifecycle import wait_for_source_drain


def _snapshot(server_id: str, *, running: int = 0, version: int = 1):
    return {
        "server_id": server_id,
        "backend_health": "healthy",
        "kv_cache_usage": 0.1,
        "num_waiting_requests": 0,
        "num_running_requests": running,
        "snapshot_version": version,
        "timestamp": float(version),
    }


def _router(count=4):
    router = ValidationRequestRouter({f"s{index}": object() for index in range(count)})
    for server_id in router._server_states:
        router.mark_active(server_id, weight_version=0)
    router.update_routing_snapshot({server_id: _snapshot(server_id) for server_id in router._server_states})
    return router


def _loaded_groups():
    router = _router()
    groups = router.set_validation_versions([1, 2])
    for version, server_ids in groups.items():
        for server_id in server_ids:
            router.mark_active(server_id, weight_version=version)
    return router, groups


def test_validation_replica_groups_are_deterministic_and_cover_every_server() -> None:
    groups = validation_replica_groups(["s0", "s1", "s2", "s3", "s4"], [1, 2])
    assert groups == {1: ["s0", "s2", "s4"], 2: ["s1", "s3"]}
    with pytest.raises(ValueError):
        validation_replica_groups(["s0"], [])
    with pytest.raises(ValueError):
        validation_replica_groups(["s0"], [1, 2])


def test_validation_groups_reject_latest_version_continuation() -> None:
    router = ValidationRequestRouter(
        {"s0": object()},
        config=GlobalRequestRouterConfig(continuation="latest_version"),
    )
    with pytest.raises(RuntimeError, match="same_version"):
        router.set_validation_versions([1])


def test_assignment_pins_routes_to_exact_versions() -> None:
    router, groups = _loaded_groups()
    router.bind_route("v1", min_weight_version=1)
    router.bind_route("v2", min_weight_version=2)
    assert router._session_id_to_server["v1"].server_id in groups[1]
    assert router._session_id_to_server["v2"].server_id in groups[2]
    with pytest.raises(ValueError, match="assigned"):
        router.bind_route("v3", min_weight_version=3)


def test_unready_validation_group_does_not_fall_forward_to_a_newer_version() -> None:
    router = _router(count=2)
    groups = router.set_validation_versions([1, 2])
    router.mark_active(groups[2][0], weight_version=2)
    with pytest.raises(RouteUnavailableError):
        router.bind_route("v1", min_weight_version=1)
    assert router.bind_route("v2", min_weight_version=2) == 2


def test_validation_barrier_checks_only_the_selected_replica_group() -> None:
    router, groups = _loaded_groups()
    router.update_routing_snapshot(
        {
            server_id: _snapshot(server_id, running=int(server_id in groups[2]), version=2)
            for server_id in router._server_states
        }
    )
    assert router.get_idle_server_versions(1, server_ids=groups[1]) == dict.fromkeys(groups[1], 1)
    with pytest.raises(RouteUnavailableError):
        router.get_idle_server_versions(2, server_ids=groups[2])


def test_validation_rechecks_live_routes_and_release_ack() -> None:
    router = ValidationRequestRouter(servers={"only": object()})
    router.mark_active("only", weight_version=7)
    router.update_routing_snapshot({"only": _snapshot("only")})
    router.bind_route("session", min_weight_version=7)
    assert router.plan_and_begin_repack(loadable_version=7, repack_enabled=False).idle_sources == ()
    with pytest.raises(RouteUnavailableError):
        router.get_idle_server_versions(7)
    with pytest.raises(RuntimeError, match="active and idle"):
        router.set_validation_versions([7])
    router.release_route("session", weight_version=7)
    assert router.get_idle_server_versions(7) == {"only": 7}
    assert router.set_validation_versions([7]) == {7: ["only"]}
    router.bind_route("next-session", min_weight_version=7)
    router.acquire_lease("next-session", lease_id="running", weight_version=7)
    with pytest.raises(RouteUnavailableError):
        router.get_idle_server_versions(7)
    with pytest.raises(RuntimeError, match="active and idle"):
        router.set_validation_versions([8])
    router.release_lease("next-session", "running")
    router.release_route("next-session", weight_version=7)
    assert router.get_idle_server_versions(7) == {"only": 7}
    assert router.set_validation_versions([8]) == {8: ["only"]}
    router.mark_active("only", weight_version=8)
    assert router.get_idle_server_versions(8) == {"only": 8}


def test_repack_respects_validation_targets() -> None:
    router, groups = _loaded_groups()
    plan = router.plan_and_begin_repack(loadable_version=2, repack_enabled=False)
    assert set(plan.idle_sources).issubset(groups[2])
    assert not (set(plan.idle_sources) & set(groups[1]))


def test_validation_repack_preserves_exact_routes_and_fences_leases_before_reassignment() -> None:
    router, groups = _loaded_groups()
    router.bind_route("old", min_weight_version=1)
    source, _ = router.acquire_lease("old", lease_id="old-turn", weight_version=1)
    router.bind_route("other-checkpoint", min_weight_version=2)
    other_route = router._session_id_to_server["other-checkpoint"]

    plan = router.plan_validation_repack(groups[1])
    assert len(plan.groups) == 1 and plan.idle_sources == ()
    group = plan.groups[0]
    assert group.sources == (source,) and group.weight_version == 1
    assert router._session_id_to_server["old"].server_id == group.destination
    assert router._session_id_to_server["old"].weight_version == 1
    assert router._session_id_to_server["other-checkpoint"] == other_route
    assert all(router._server_states[server_id].lifecycle == "active" for server_id in groups[2])
    with pytest.raises(RuntimeError, match="updating and drained"):
        router.set_validation_versions([3], server_ids=[source], repacked=True)
    with pytest.raises(RouteUnavailableError, match="in-flight"):
        router.try_mark_updating(source, expected_version=1)

    continued, _ = router.acquire_lease("old", lease_id="continued", weight_version=1)
    assert continued == group.destination
    router.release_lease("old", "old-turn")
    router.try_mark_updating(source, expected_version=1)
    assert router.set_validation_versions([3], server_ids=[source], repacked=True) == {3: [source]}
    assert router._server_states[source].lifecycle == "updating"
    with pytest.raises(RouteUnavailableError):
        router.bind_route("new-checkpoint", min_weight_version=3)
    with pytest.raises(RuntimeError, match="outside its validation target"):
        router.mark_active(source, weight_version=2)
    router.mark_active(source, weight_version=3)
    assert router.bind_route("new-checkpoint", min_weight_version=3) == 3
    assert router._session_id_to_server["old"].server_id == group.destination
    assert router._session_id_to_server["old"].weight_version == 1


def test_validation_repack_leaves_idle_and_single_replica_groups_active() -> None:
    router, groups = _loaded_groups()
    assert router.plan_validation_repack(groups[1]).groups == ()
    assert router.plan_validation_repack(groups[1]).idle_sources == ()
    assert all(state.lifecycle == "active" for state in router._server_states.values())
    router.bind_route("only", min_weight_version=1)
    server_id = router._session_id_to_server["only"].server_id
    plan = router.plan_validation_repack([server_id])
    assert plan.groups == () and plan.idle_sources == ()
    assert router._server_states[server_id].lifecycle == "active"


def test_validation_repack_rejects_mixed_or_unloaded_targets_without_mutation() -> None:
    router, groups = _loaded_groups()
    with pytest.raises(ValueError, match="one loaded checkpoint"):
        router.plan_validation_repack([groups[1][0], groups[2][0]])
    router.set_validation_versions([3], server_ids=groups[1])
    with pytest.raises(ValueError, match="one loaded checkpoint"):
        router.plan_validation_repack(groups[1])
    assert all(state.lifecycle == "active" for state in router._server_states.values())
    with pytest.raises(ValueError, match="explicit server_ids"):
        router.set_validation_versions([4], repacked=True)
    with pytest.raises(RuntimeError, match="updating and drained"):
        router.set_validation_versions([4], server_ids=groups[1], repacked=True)


@pytest.mark.asyncio
async def test_shared_drain_sweeps_late_requests_and_waits_for_the_source_lease_release() -> None:
    router, groups = _loaded_groups()
    router.bind_route("old", min_weight_version=1)
    source, _ = router.acquire_lease("old", lease_id="late-turn", weight_version=1)
    group = router.plan_validation_repack(groups[1]).groups[0]
    assert source in group.sources
    sweeps = []

    class Remote:
        def __init__(self, function):
            self.function = function

        async def remote(self, *args, **kwargs):
            return self.function(*args, **kwargs)

    def abort(**kwargs):
        sweeps.append(kwargs)
        if len(sweeps) == 2:
            router.release_lease("old", "late-turn")

    backend = SimpleNamespace(abort_all_requests=Remote(abort))
    actor = SimpleNamespace(
        get_server_handle=Remote(lambda **kwargs: backend),
        try_mark_updating=Remote(router.try_mark_updating),
    )
    await wait_for_source_drain(actor, source, expected_version=1)
    assert sweeps == [{"abort_only": True}, {"abort_only": True}]
    assert router._server_states[source].lifecycle == "updating"
    assert router._session_id_to_server["old"].server_id == group.destination
    assert router._session_id_to_server["old"].weight_version == 1


def test_subset_barrier_is_fail_fast_on_a_real_local_actor() -> None:
    assert not ray.is_initialized()
    ray.init(
        address="local",
        num_cpus=2,
        include_dashboard=False,
        _temp_dir=tempfile.mkdtemp(prefix="validation-router-"),
        runtime_env={
            "env_vars": {
                "PYTHONPATH": os.pathsep.join(
                    filter(
                        None,
                        (
                            str(Path(__file__).resolve().parents[1] / "src"),
                            str(Path(__file__).parent),
                            os.environ.get("PYTHONPATH", ""),
                        ),
                    )
                ),
                "OMP_NUM_THREADS": "1",
            }
        },
    )
    router = ray.remote(ValidationRequestRouter).remote(servers={"s0": None, "s1": None})
    try:
        for server_id in ("s0", "s1"):
            ray.get(router.mark_active.remote(server_id, weight_version=0), timeout=5)
        ray.get(
            router.update_routing_snapshot.remote({"s0": _snapshot("s0"), "s1": _snapshot("s1")}),
            timeout=5,
        )
        groups = ray.get(router.set_validation_versions.remote([1, 2]), timeout=5)
        for version, server_ids in groups.items():
            for server_id in server_ids:
                ray.get(router.mark_active.remote(server_id, weight_version=version), timeout=5)
        ray.get(router.bind_route.remote("busy", min_weight_version=1), timeout=5)
        with pytest.raises(ray.exceptions.RayTaskError, match="not yet idle"):
            ray.get(router.get_idle_server_versions.remote(1, server_ids=groups[1]), timeout=5)
        assert ray.get(router.get_idle_server_versions.remote(2, server_ids=groups[2]), timeout=5) == dict.fromkeys(
            groups[2], 2
        )
    finally:
        ray.kill(router, no_restart=True)
        ray.shutdown()
