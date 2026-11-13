# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from pathlib import Path

import pytest
import ray

from laminar.global_request_router import GlobalRequestRouter, GlobalRequestRouterConfig, SessionRouteClosedError


class _InspectableRouter(GlobalRequestRouter):
    def inspect_state(self):
        return {
            "routes": len(self._session_id_to_server),
            "sessions": self.get_active_session_count(),
            "leases": sum(len(session.leases) for session in self._sessions.values()),
        }


@pytest.fixture(scope="module")
def local_ray():
    assert not ray.is_initialized(), "This regression owns an isolated local runtime"
    ray.init(
        address="local",
        num_cpus=4,
        include_dashboard=False,
        _temp_dir=tempfile.mkdtemp(prefix="router-fast-rpc-"),
        object_store_memory=128 * 1024 * 1024,
        runtime_env={
            "env_vars": {
                "PYTHONPATH": str(Path(__file__).parent) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                "OMP_NUM_THREADS": "1",
            }
        },
    )
    yield
    ray.shutdown()


def test_many_fail_fast_calls_do_not_occupy_actor_slots_or_block_control(local_ray):
    router = ray.remote(max_concurrency=2)(_InspectableRouter).remote(
        servers={"s0": None},
        config=GlobalRequestRouterConfig(continuation="latest_version"),
    )
    try:
        ray.get(router.mark_active.remote("s0", weight_version=1), timeout=10)
        session_ids = [f"session-{index}" for index in range(336)]
        ray.get([router.bind_route.remote(session_id) for session_id in session_ids], timeout=10)
        ray.get(router.acquire_lease.remote(session_ids[0], lease_id="running"), timeout=5)
        ray.get(router.begin_colocate_pause.remote(), timeout=5)
        attempts = [
            router.acquire_lease.remote(session_id, lease_id=f"next-{index}")
            for index, session_id in enumerate(session_ids)
        ]
        ray.get(router.release_lease.remote(session_ids[0], "running"), timeout=10)
        ray.get(router.try_mark_updating.remote("s0", expected_version=1), timeout=10)
        for attempt in attempts:
            with pytest.raises(ray.exceptions.RayTaskError, match="No available servers"):
                ray.get(attempt, timeout=10)
        state = ray.get(router.inspect_state.remote(), timeout=5)
        assert state["routes"] == state["leases"] == 0
        ray.get(router.mark_active.remote("s0", weight_version=2), timeout=5)
        assert ray.get(router.acquire_lease.remote(session_ids[-1], lease_id="retry"), timeout=5)[0] == "s0"
    finally:
        ray.kill(router, no_restart=True)


def test_public_actor_needs_no_large_max_concurrency_configuration(local_ray):
    config = GlobalRequestRouterConfig(continuation="latest_version")
    assert not hasattr(config, "max_concurrency")
    router = ray.remote(GlobalRequestRouter).remote(servers={"s0": None}, config=config)
    try:
        with pytest.raises(ray.exceptions.RayTaskError, match="No available servers"):
            ray.get(router.bind_route.remote("session"), timeout=10)
        ray.get(router.mark_active.remote("s0", weight_version=1), timeout=5)
        ray.get(router.bind_route.remote("session"), timeout=5)
        assert ray.get(router.acquire_lease.remote("session", lease_id="lease"), timeout=5)[0] == "s0"
    finally:
        ray.kill(router, no_restart=True)


def test_closed_session_cannot_be_reopened_by_a_late_acquire(local_ray):
    router = ray.remote(GlobalRequestRouter).remote(servers={"s0": None})
    try:
        ray.get(router.mark_active.remote("s0", weight_version=1), timeout=5)
        ray.get(router.bind_route.remote("session"), timeout=5)
        ray.get(router.release_route.remote("session"), timeout=5)
        with pytest.raises(ray.exceptions.RayTaskError) as error:
            ray.get(router.acquire_lease.remote("session", lease_id="late"), timeout=5)
        assert isinstance(error.value.as_instanceof_cause(), SessionRouteClosedError)
    finally:
        ray.kill(router, no_restart=True)
