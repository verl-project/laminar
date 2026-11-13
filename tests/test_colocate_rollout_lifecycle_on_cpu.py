# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest

from laminar.global_request_router.router import RouteUnavailableError
from laminar.llm_server.controller import LLMServerController
from laminar.multi_version_rollout import MultiVersionRolloutManager, MultiVersionRolloutManagerConfig


class _Remote:
    def __init__(self, method):
        self.remote = method


class _Engine:
    def __init__(self):
        self.events = []
        self._server_handle = SimpleNamespace(
            collective_rpc=_Remote(self.collective_rpc),
            clear_kv_cache=_Remote(self.clear_kv_cache),
            set_global_steps=_Remote(self.set_global_steps),
        )

    async def init_colocated(self, *, resource_pool):
        self.events.append(("init_colocated", resource_pool))

    async def abort_all_requests(self, **kwargs):
        self.events.append(("abort", kwargs))

    async def sleep(self):
        self.events.append(("sleep",))

    async def wake_up(self):
        self.events.append(("wake",))

    async def resume_generation(self):
        self.events.append(("resume",))

    async def collective_rpc(self, method):
        self.events.append((method,))

    async def clear_kv_cache(self):
        self.events.append(("clear_kv_cache",))

    async def set_global_steps(self, version):
        self.events.append(("set_global_steps", version))


class _Relay:
    def __init__(self, events):
        self.events = events

    async def acquire_rollout_version(self, nodes, *, minimum_version):
        self.events.append(("acquire", nodes, minimum_version))
        return minimum_version

    async def release_full_read_async(self, nodes):
        self.events.append(("release", nodes))


@pytest.mark.asyncio
async def test_initial_colocated_engine_sleeps_before_trainer_initialization():
    engine = _Engine()
    controller = LLMServerController(relay_client=object(), engine=engine, name="test")
    await controller.init_engine(resource_pool="shared-pool", colocate=True)
    assert engine.events == [
        ("init_colocated", "shared-pool"),
        ("abort", {"reset_prefix_cache": False}),
        ("sleep",),
    ]
    assert controller._asleep
    # The backend has no snapshot method: sleeping polling must avoid vLLM's
    # scheduler while its allocations belong to the trainer.
    snapshot = await controller.snapshot()
    assert snapshot.backend_health == "healthy"
    assert (snapshot.kv_cache_usage, snapshot.num_waiting_requests, snapshot.num_running_requests) == (0, 0, 0)


@pytest.mark.asyncio
async def test_same_published_version_wakes_without_reloading_or_opening_gate():
    engine = _Engine()
    controller = LLMServerController(relay_client=object(), engine=engine, name="test")
    controller._colocate = controller._asleep = True
    controller.weight_version = 7
    assert await controller.load_weights(7, resume_generation=False) == 7
    assert engine.events == [("wake",)]
    assert not controller._asleep
    await controller.resume_generation()
    assert engine.events == [("wake",), ("resume",)]


@pytest.mark.asyncio
async def test_new_version_wakes_loads_and_releases_relay_before_cohort_resume():
    engine = _Engine()
    controller = LLMServerController(relay_client=_Relay(engine.events), engine=engine, name="test")
    controller._colocate = controller._asleep = True
    controller._server_node_ids = ("node",)
    controller.weight_version = 7
    assert await controller.load_weights(8, resume_generation=False) == 8
    assert engine.events == [
        ("acquire", ("node",), 8),
        ("wake",),
        ("abort", {"reset_prefix_cache": False}),
        ("load_weights_from_shm_state_dict",),
        ("clear_kv_cache",),
        ("set_global_steps", 8),
        ("release", ("node",)),
    ]


@pytest.mark.asyncio
async def test_wake_failure_releases_relay_read_lock_without_changing_version():
    engine = _Engine()
    controller = LLMServerController(relay_client=_Relay(engine.events), engine=engine, name="test")
    controller._colocate = controller._asleep = True
    controller._server_node_ids = ("node",)
    controller.weight_version = 7

    async def fail_wake():
        raise RuntimeError("wake failed")

    engine.wake_up = fail_wake
    with pytest.raises(RuntimeError, match="wake failed"):
        await controller.load_weights(8, resume_generation=False)
    assert controller.weight_version == 7
    assert controller._asleep
    assert engine.events == [("acquire", ("node",), 8), ("release", ("node",))]


class _Controller:
    def __init__(self, server_id, events, *, loaded_version=8, fail=None):
        self.server_id, self.events = server_id, events
        self.loaded_version, self.fail = loaded_version, fail
        self.load_weights = _Remote(self._load)
        self.check_health = _Remote(self._check_health)
        self.resume_generation = _Remote(self._resume)
        self.sleep_for_training = _Remote(self._sleep)
        self.abort_all_requests = _Remote(self._abort)

    async def _load(self, *, wait_for_version, resume_generation):
        assert not resume_generation
        assert wait_for_version == 8
        self.events.append(("load", self.server_id))
        if self.fail == "load":
            raise RuntimeError("load failed")
        return self.loaded_version

    async def _check_health(self):
        self.events.append(("health", self.server_id))
        if self.fail == "health":
            raise RuntimeError("health failed")

    async def _resume(self):
        self.events.append(("resume", self.server_id))
        if self.fail == "resume":
            raise RuntimeError("resume failed")

    async def _abort(self, *, abort_only):
        assert abort_only
        self.events.append(("abort_wait_release", self.server_id))

    async def _sleep(self):
        self.events.append(("sleep", self.server_id))
        if self.fail == "sleep":
            raise RuntimeError("sleep failed")


class _Router:
    def __init__(self, events):
        self.events = events
        self.begin_colocate_pause = _Remote(self._begin)
        self.try_mark_updating = _Remote(self._drain)
        self.get_server_handle = _Remote(self._get_server_handle)
        self.server_handles = {}
        self.mark_active = _Remote(self._active)
        self.pending_release = asyncio.Event()
        self.failed_servers = []
        self.failed_server_ids = _Remote(self._failed)

    async def _failed(self):
        return list(self.failed_servers)

    async def _begin(self):
        self.events.append(("fence",))
        return {"s0": 8, "s1": 8}

    async def _get_server_handle(self, *, server_id):
        return self.server_handles[server_id]

    async def _drain(self, *, server_id, expected_version):
        assert expected_version == 8
        if not self.pending_release.is_set():
            raise RouteUnavailableError("request release is still pending")
        self.events.append(("drained", server_id))

    async def _active(self, *, server_id, weight_version):
        assert weight_version == 8
        self.events.append(("active", server_id))


def _manager(*, fail=None, loaded_version=8):
    events = []
    router = _Router(events)
    manager = MultiVersionRolloutManager(MultiVersionRolloutManagerConfig(colocate=True, schedule_interval_s=0.001))
    manager.connect(
        servers={
            "s0": _Controller("s0", events),
            "s1": _Controller("s1", events, fail=fail, loaded_version=loaded_version),
        },
        router=router,
        relay_client=object(),
    )
    router.server_handles = manager.servers
    return manager, router, events


@pytest.mark.asyncio
async def test_resume_checks_entire_cohort_then_opens_routing():
    manager, _, events = _manager()
    await manager.resume_rollout(8)
    stage_names = [event[0] for event in events]
    assert stage_names.count("health") == stage_names.count("resume") == stage_names.count("active") == 2
    assert max(i for i, name in enumerate(stage_names) if name == "health") < stage_names.index("resume")
    assert max(i for i, name in enumerate(stage_names) if name == "resume") < stage_names.index("active")
    assert manager._colocate_phase == "rollout"


@pytest.mark.parametrize("fail", ["load", "health", "resume"])
@pytest.mark.asyncio
async def test_resume_failure_never_reopens_routing_and_cannot_retry(fail):
    manager, _, events = _manager(fail=fail)
    with pytest.raises(RuntimeError):
        await manager.resume_rollout(8)
    assert all(event[0] != "active" for event in events)
    assert manager._colocate_phase == "failed"
    with pytest.raises(RuntimeError, match="phase 'failed'"):
        await manager.resume_rollout(8)


@pytest.mark.parametrize("loaded_version", [7, 9, True])
@pytest.mark.asyncio
async def test_resume_requires_exact_published_version(loaded_version):
    manager, _, events = _manager(loaded_version=loaded_version)
    with pytest.raises(RuntimeError, match="expected exactly 8"):
        await manager.resume_rollout(8)
    assert all(event[0] not in {"resume", "active"} for event in events)


@pytest.mark.asyncio
async def test_pause_waits_for_all_request_release_acks_before_sleeping():
    manager, router, events = _manager()
    await manager.resume_rollout(8)
    events.clear()
    pause = asyncio.create_task(manager.pause_for_training())
    while sum(event[0] == "abort_wait_release" for event in events) != 2:
        await asyncio.sleep(0)
    assert events[0] == ("fence",)
    assert all(event[0] != "sleep" for event in events)
    router.pending_release.set()
    await pause
    names = [event[0] for event in events]
    assert names.count("drained") == names.count("sleep") == 2
    assert max(i for i, name in enumerate(names) if name == "drained") < names.index("sleep")
    assert names.count("abort_wait_release") >= 2
    assert manager._colocate_phase == "training"


@pytest.mark.asyncio
async def test_sleep_failure_leaves_handoff_failed():
    manager, router, events = _manager(fail="sleep")
    await manager.resume_rollout(8)
    router.pending_release.set()
    with pytest.raises(RuntimeError, match="sleep failed"):
        await manager.pause_for_training()
    with pytest.raises(RuntimeError, match="phase 'failed'"):
        await manager.resume_rollout(8)


@pytest.mark.asyncio
async def test_colocate_background_loop_never_loads_weights_or_recovers_during_training():
    manager, _, events = _manager()
    await manager.start()
    await asyncio.sleep(0.005)
    manager._run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await manager.wait_for_stop()
    assert events == []
    assert manager._repack_tasks == {}


@pytest.mark.asyncio
async def test_rollout_engine_death_surfaces_without_waiting_for_another_training_batch():
    manager, router, events = _manager()
    await manager.resume_rollout(8)
    events.clear()
    router.failed_servers = ["s1"]
    await manager.start()
    with pytest.raises(RuntimeError, match="colocated rollout engines failed.*s1"):
        await asyncio.wait_for(manager.wait_for_stop(), timeout=1)
    assert manager._colocate_phase == "failed"
    # No reload/recovery attempt can allocate rollout memory in the trainer's
    # pool after the lifecycle owner has reported the engine failure.
    assert events == []


@pytest.mark.asyncio
async def test_training_handoff_owns_health_checks_until_resume():
    manager, router, events = _manager()
    router.failed_servers = ["s1"]
    await manager.start()
    await asyncio.sleep(0.005)
    assert not manager._run_task.done()
    manager._run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await manager.wait_for_stop()
    assert events == []
