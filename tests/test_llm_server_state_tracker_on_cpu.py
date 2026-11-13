# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import math
import os
import tempfile
from pathlib import Path

import pytest
import ray

from laminar.global_request_router import GlobalRequestRouter
from laminar.llm_server.controller import LLMServerSnapshot
from laminar.llm_server.state_tracker import LLMServerStateTracker


@ray.remote
class FakeLLMServerController:
    def __init__(self, snapshot: LLMServerSnapshot | None = None):
        self.snapshot_value = snapshot
        self.fail = False

    def set_fail(self, fail: bool) -> None:
        self.fail = fail

    def snapshot(self) -> LLMServerSnapshot:
        if self.fail:
            raise RuntimeError("snapshot failed")
        assert self.snapshot_value is not None
        return self.snapshot_value


@ray.remote
class FakeRouter:
    def __init__(self):
        self.routing_snapshots = []
        self.fail_update = False

    def update_routing_snapshot(self, snapshot):
        if self.fail_update:
            raise RuntimeError("router update failed")
        self.routing_snapshots.append(snapshot)

    def set_fail_update(self, fail_update: bool):
        self.fail_update = fail_update

    def get_routing_snapshots(self):
        return list(self.routing_snapshots)


@pytest.fixture(scope="module")
def ray_for_tracker():
    # Use a bounded local Ray instance with explicit test/source PYTHONPATH so
    # full-suite runs never attach to an ambient RAY_ADDRESS or starve CPU/shm.
    owned = not ray.is_initialized()
    with tempfile.TemporaryDirectory(prefix="state-tracker-cpu-") as temp_dir:
        if owned:
            test_dir = Path(__file__).resolve().parent
            worker_pythonpath = os.pathsep.join(
                [str(test_dir.parent / "src"), str(test_dir), os.environ.get("PYTHONPATH", "")]
            )
            ray.init(
                address="local",
                num_cpus=2,
                include_dashboard=False,
                object_store_memory=128 * 1024 * 1024,
                _temp_dir=temp_dir,
                runtime_env={"env_vars": {"PYTHONPATH": worker_pythonpath, "OMP_NUM_THREADS": "1"}},
            )
        try:
            yield
        finally:
            if owned:
                ray.shutdown()


def _snapshot(
    *,
    kv_usage: float,
    waiting: int = 0,
    running: int = 0,
) -> LLMServerSnapshot:
    return LLMServerSnapshot(
        kv_cache_usage=kv_usage,
        num_waiting_requests=waiting,
        num_running_requests=running,
    )


def test_tracker_exposes_compact_routing_snapshot(ray_for_tracker):
    s0 = FakeLLMServerController.remote(_snapshot(kv_usage=0.25, waiting=1))
    s1 = FakeLLMServerController.remote(_snapshot(kv_usage=0.10, running=3))
    tracker = ray.remote(LLMServerStateTracker).remote()
    ray.get(tracker.connect.remote({"s0": s0, "s1": s1}))

    ray.get(tracker.refresh_once.remote())
    routing_snapshot = ray.get(tracker.get_routing_snapshot.remote())

    assert routing_snapshot["s0"] == {
        "backend_health": "healthy",
        "kv_cache_usage": 0.25,
        "num_waiting_requests": 1,
        "num_running_requests": 0,
        "snapshot_version": 1,
    }
    assert routing_snapshot["s1"]["kv_cache_usage"] == 0.10
    assert routing_snapshot["s1"]["num_running_requests"] == 3


def test_tracker_push_updates_router_selection(ray_for_tracker):
    s0 = FakeLLMServerController.remote(_snapshot(kv_usage=0.25))
    s1 = FakeLLMServerController.remote(_snapshot(kv_usage=0.10))
    router = ray.remote(GlobalRequestRouter).remote(servers={"s0": None, "s1": None})
    tracker = ray.remote(LLMServerStateTracker).remote()
    ray.get(router.mark_active.remote(server_id="s0", weight_version=3))
    ray.get(router.mark_active.remote(server_id="s1", weight_version=3))
    ray.get(tracker.connect.remote({"s0": s0, "s1": s1}, router))
    ray.get(tracker.refresh_once.remote())

    assert ray.get(router.bind_route.remote(session_id="session-0", min_weight_version=3)) == 3
    selected = ray.get(router.acquire_lease.remote(session_id="session-0", lease_id="request", weight_version=3))[0]

    assert selected == "s1"


def test_tracker_refresh_pushes_one_routing_snapshot_to_router(ray_for_tracker):
    s0 = FakeLLMServerController.remote(_snapshot(kv_usage=0.25, waiting=1))
    s1 = FakeLLMServerController.remote(_snapshot(kv_usage=0.10, running=3))
    router = FakeRouter.remote()
    tracker = ray.remote(LLMServerStateTracker).remote()
    ray.get(tracker.connect.remote({"s0": s0, "s1": s1}, router))

    ray.get(tracker.refresh_once.remote())

    snapshots = ray.get(router.get_routing_snapshots.remote())
    assert len(snapshots) == 1
    assert snapshots[0]["s0"]["backend_health"] == "healthy"
    assert snapshots[0]["s0"]["num_waiting_requests"] == 1
    assert snapshots[0]["s1"]["num_running_requests"] == 3


def test_tracker_fails_fast_when_router_snapshot_update_fails(ray_for_tracker):
    s0 = FakeLLMServerController.remote(_snapshot(kv_usage=0.25))
    router = FakeRouter.remote()
    ray.get(router.set_fail_update.remote(True))
    tracker = ray.remote(LLMServerStateTracker).remote()
    ray.get(tracker.connect.remote({"s0": s0}, router))

    with pytest.raises(RuntimeError, match="router update failed"):
        ray.get(tracker.refresh_once.remote())


def test_tracker_fails_fast_on_poll_failure(ray_for_tracker):
    s0 = FakeLLMServerController.remote(_snapshot(kv_usage=0.2))
    ray.get(s0.set_fail.remote(True))
    tracker = ray.remote(LLMServerStateTracker).remote()
    ray.get(tracker.connect.remote({"s0": s0}))

    with pytest.raises(RuntimeError, match="failed to poll LLM server 's0'"):
        ray.get(tracker.refresh_once.remote())


@pytest.mark.parametrize(
    "field_name",
    ["kv_cache_usage", "num_waiting_requests", "num_running_requests"],
)
def test_tracker_fails_fast_on_incomplete_snapshot(ray_for_tracker, field_name: str):
    snapshot = _snapshot(kv_usage=0.2)
    delattr(snapshot, field_name)
    s0 = FakeLLMServerController.remote(snapshot)
    tracker = ray.remote(LLMServerStateTracker).remote()
    ray.get(tracker.connect.remote({"s0": s0}))

    with pytest.raises(RuntimeError, match="invalid snapshot from LLM server 's0'"):
        ray.get(tracker.refresh_once.remote())


@pytest.mark.parametrize(
    "snapshot",
    [
        _snapshot(kv_usage=math.nan),
        _snapshot(kv_usage=1.1),
        _snapshot(kv_usage=0.2, waiting=-1),
    ],
)
def test_tracker_rejects_invalid_routing_snapshot(ray_for_tracker, snapshot):
    s0 = FakeLLMServerController.remote(snapshot)
    tracker = ray.remote(LLMServerStateTracker).remote()
    ray.get(tracker.connect.remote({"s0": s0}))

    with pytest.raises(RuntimeError, match="invalid snapshot from LLM server 's0'"):
        ray.get(tracker.refresh_once.remote())


def test_tracker_wait_for_stop_surfaces_background_poll_failure(ray_for_tracker):
    s0 = FakeLLMServerController.remote(_snapshot(kv_usage=0.2))
    ray.get(s0.set_fail.remote(True))
    tracker = ray.remote(LLMServerStateTracker).remote()
    ray.get(tracker.connect.remote({"s0": s0}))
    ray.get(tracker.start.remote())

    with pytest.raises(RuntimeError, match="failed to poll LLM server 's0'"):
        ray.get(tracker.wait_for_stop.remote())


def test_tracker_preserves_confirmed_unavailable_without_stale_gauges():
    tracker = LLMServerStateTracker()
    snapshot = LLMServerSnapshot(
        kv_cache_usage=0.0, num_waiting_requests=0, num_running_requests=0, backend_health="unavailable"
    )
    state = tracker._state_from_snapshot(snapshot, snapshot_version=2)
    assert state["backend_health"] == "unavailable"
    assert state["snapshot_version"] == 2


def test_tracker_rejects_unknown_health():
    tracker = LLMServerStateTracker()
    snapshot = LLMServerSnapshot(
        kv_cache_usage=0.0, num_waiting_requests=0, num_running_requests=0, backend_health="unknown"
    )
    with pytest.raises(ValueError, match="invalid backend health"):
        tracker._state_from_snapshot(snapshot)
