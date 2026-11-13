# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from verl.workers.config import RolloutConfig

from laminar.llm_server.controller import LLMServerController
from laminar.utils.tensor import TensorMeta


def _backend_snapshot() -> dict:
    return {
        "kv_cache_usage": 0.25,
        "num_waiting_requests": 1,
        "num_running_requests": 2,
    }


def _controller_with_snapshot(snapshot: dict) -> LLMServerController:
    class _Snapshot:
        async def remote(self, **kwargs):  # noqa: ANN003
            assert kwargs == {}
            return snapshot

    controller = object.__new__(LLMServerController)
    controller._unavailable = False
    controller.relay_client = object()
    controller.name = "llm-server-controller-test"
    controller.weight_version = 3
    controller._engine = SimpleNamespace(
        _server_handle=SimpleNamespace(snapshot=_Snapshot(), check_health=_Snapshot()),
    )
    return controller


def test_llm_server_controller_snapshot_uses_actor_level_concurrency():
    assert "__ray_concurrency_group__" not in vars(LLMServerController.snapshot)


def test_llm_server_controller_snapshot_exposes_routing_state():
    controller = _controller_with_snapshot(_backend_snapshot())

    snapshot = asyncio.run(controller.snapshot())

    assert snapshot.kv_cache_usage == 0.25
    assert snapshot.num_waiting_requests == 1
    assert snapshot.num_running_requests == 2


def test_health_check_uses_backend_without_routing_metrics():
    controller = _controller_with_snapshot({})
    calls = []

    async def health():
        calls.append("health")

    controller._engine._server_handle.check_health = SimpleNamespace(remote=health)
    assert asyncio.run(controller.check_health()) is None
    controller._asleep = True
    assert asyncio.run(controller.check_health()) is None
    assert calls == ["health"]
    controller._unavailable = True
    with pytest.raises(RuntimeError, match="unavailable"):
        asyncio.run(controller.check_health())


@pytest.mark.parametrize(
    "missing_path",
    [
        ("kv_cache_usage",),
        ("num_waiting_requests",),
        ("num_running_requests",),
    ],
)
def test_llm_server_controller_snapshot_requires_backend_observations(missing_path: tuple[str, ...]):
    backend_snapshot = _backend_snapshot()
    owner = backend_snapshot
    for field_name in missing_path[:-1]:
        owner = owner[field_name]
    del owner[missing_path[-1]]

    with pytest.raises(KeyError, match=missing_path[-1]):
        asyncio.run(_controller_with_snapshot(backend_snapshot).snapshot())


def test_llm_server_controller_initializes_fixed_pinned_worker_storage():
    class _RayCall:
        def __init__(self):
            self.calls = 0

        async def remote(self, func):  # noqa: ANN001
            self.calls += 1
            return "node-a"

    class _EngineServer:
        def __init__(self):
            self.__ray_call__ = _RayCall()

    class _CollectiveRPC:
        def __init__(self):
            self.calls = []

        async def remote(self, method, *, args):
            self.calls.append((method, args))

    collective_rpc = _CollectiveRPC()
    server = _EngineServer()
    controller = object.__new__(LLMServerController)
    controller._server_node_ids = None
    controller._engine = SimpleNamespace(
        servers=[server],
        _server_handle=SimpleNamespace(collective_rpc=collective_rpc),
    )
    meta = TensorMeta(dtype=torch.bfloat16, shape=torch.Size([2, 3]))

    asyncio.run(controller.init_shm_state_dict_in_workers(state_meta_dict={"weight": meta}))

    assert len(collective_rpc.calls) == 1
    method, args = collective_rpc.calls[0]
    assert method == "init_shm_state_dict"
    assert len(args) == 1
    assert set(args[0]) == {"weight"}
    assert controller._server_node_ids == ("node-a",)
    assert asyncio.run(controller._resolve_server_node_ids()) == ("node-a",)
    assert server.__ray_call__.calls == 1


class _LoadEngine:
    def __init__(self, fail: str | None = None):
        self.config = RolloutConfig(name="vllm")
        self.calls: list[tuple[str, dict[str, Any], int]] = []
        self.fail = fail
        self.controller: LLMServerController | None = None
        self._server_handle = SimpleNamespace(
            abort_all_requests=SimpleNamespace(remote=self.abort_all_requests),
            collective_rpc=SimpleNamespace(remote=self.collective_rpc),
            clear_kv_cache=SimpleNamespace(remote=self.clear_kv_cache),
            set_global_steps=SimpleNamespace(remote=self.set_global_steps),
            resume_generation=SimpleNamespace(remote=self.resume_generation),
        )

    def record(self, method: str, **kwargs: Any) -> None:
        assert self.controller is not None
        self.calls.append((method, kwargs, self.controller.weight_version))
        if self.fail == method:
            raise RuntimeError(f"{method} failed")

    async def abort_all_requests(self, **kwargs: Any) -> None:
        self.record("pause", **kwargs)

    async def collective_rpc(self, method: str) -> None:
        assert method == "load_weights_from_shm_state_dict"
        self.record("load")

    async def clear_kv_cache(self, **kwargs: Any) -> None:
        self.record("clear_kv_cache", **kwargs)

    async def set_global_steps(self, version: int) -> None:
        self.record("set_global_steps", version=version)

    async def resume_generation(self) -> None:
        self.record("resume")


class _LoadRelay:
    def __init__(self, engine: _LoadEngine):
        self.engine = engine
        self.releases: list[tuple[str, ...]] = []

    async def acquire_rollout_version(self, server_node_ids: tuple[str, ...], *, minimum_version: int) -> int:
        assert server_node_ids == ("node-a", "node-b")
        assert minimum_version == 3
        return 3

    async def release_full_read_async(self, server_node_ids: tuple[str, ...]) -> None:
        self.releases.append(server_node_ids)
        self.engine.record("release_read")


def _load_controller(fail: str | None = None) -> tuple[LLMServerController, _LoadEngine]:
    engine = _LoadEngine(fail)
    controller = LLMServerController(relay_client=_LoadRelay(engine), engine=engine, name="test-controller")
    controller.weight_version = 1
    controller._server_node_ids = ("node-a", "node-b")
    engine.controller = controller
    return controller, engine


@pytest.mark.parametrize("failed_stage", ["pause", "load", "clear_kv_cache", "set_global_steps", "resume"])
def test_llm_server_controller_load_weights_fails_fast_and_releases_read_lock(failed_stage: str):
    controller, engine = _load_controller(fail=failed_stage)
    stages = ["pause", "load", "clear_kv_cache", "set_global_steps", "resume"]

    with pytest.raises(RuntimeError, match=f"{failed_stage} failed"):
        asyncio.run(controller.load_weights(wait_for_version=3))

    assert controller.weight_version == 1
    assert [call[0] for call in engine.calls] == stages[: stages.index(failed_stage) + 1] + ["release_read"]
    assert all(call[2] == 1 for call in engine.calls)
    assert controller.relay_client.releases == [("node-a", "node-b")]


def test_llm_server_controller_load_weights_sets_server_global_steps_after_success():
    controller, engine = _load_controller()

    assert asyncio.run(controller.load_weights(wait_for_version=3)) == 3

    assert controller.weight_version == 3
    assert engine.calls == [
        ("pause", {"reset_prefix_cache": False}, 1),
        ("load", {}, 1),
        ("clear_kv_cache", {}, 1),
        ("set_global_steps", {"version": 3}, 1),
        ("resume", {}, 1),
        ("release_read", {}, 1),
    ]
    assert controller.relay_client.releases == [("node-a", "node-b")]


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("operation", ["snapshot", "check_health"])
def test_backend_check_marks_only_engine_dead_unavailable(wrapped, operation):
    import ray
    from vllm.v1.engine.exceptions import EngineDeadError

    controller = _controller_with_snapshot(_backend_snapshot())
    error = EngineDeadError()
    if wrapped:
        error = ray.exceptions.RayTaskError("snapshot", "backend traceback", error)

    async def failed():
        raise error

    setattr(controller._engine._server_handle, operation, SimpleNamespace(remote=failed))
    if operation == "check_health":
        with pytest.raises(type(error)) as raised:
            asyncio.run(controller.check_health())
        assert raised.value is error
    else:
        assert asyncio.run(controller.snapshot()).backend_health == "unavailable"
    assert asyncio.run(controller.snapshot()).backend_health == "unavailable"


@pytest.mark.parametrize("operation", ["snapshot", "check_health"])
def test_backend_check_does_not_swallow_unknown_backend_error(operation):
    controller = _controller_with_snapshot(_backend_snapshot())

    async def failed():
        raise ValueError("invalid backend observation")

    setattr(controller._engine._server_handle, operation, SimpleNamespace(remote=failed))
    with pytest.raises(ValueError, match="invalid backend"):
        asyncio.run(getattr(controller, operation)())
    assert not controller._unavailable


@pytest.mark.parametrize("failed_stage", [None, "restart", "shm", "weights", "health"])
def test_restart_reloads_storage_weights_and_health_before_reopening(failed_stage):
    controller = _controller_with_snapshot(_backend_snapshot())
    controller.relay_client = object()
    controller._server_node_ids = ("old-node",)
    controller._state_meta_dict = {"weight": object()}
    events = []

    async def stage(name):
        events.append(name)
        assert (await controller.snapshot()).backend_health == "unavailable"
        if failed_stage == name:
            raise RuntimeError(name)

    async def restart():
        assert controller.weight_version == -1
        assert controller._server_node_ids is None
        await stage("restart")

    async def shm(meta):
        assert meta == controller._state_meta_dict
        await stage("shm")

    async def weights(minimum):
        assert minimum == 3
        await stage("weights")
        controller.weight_version = 4
        return 4

    async def health():
        await stage("health")

    controller._engine.restart = restart
    controller.init_shm_state_dict_in_workers = shm
    controller.load_weights = weights
    controller._engine._server_handle.check_health = SimpleNamespace(remote=health)
    if failed_stage:
        with pytest.raises(RuntimeError, match=failed_stage):
            asyncio.run(controller.restart(3))
        assert controller._unavailable
        assert events == ["restart", "shm", "weights", "health"][: events.index(failed_stage) + 1]
    else:
        assert asyncio.run(controller.restart(3)) == 4
        assert not controller._unavailable
        assert events == ["restart", "shm", "weights", "health"]


@pytest.mark.parametrize("old_result", ["healthy", "dead", "actor_dead"])
@pytest.mark.parametrize("operation", ["snapshot", "check_health"])
def test_late_backend_check_cannot_poison_replacement_engine(old_result, operation):
    import ray
    from vllm.v1.engine.exceptions import EngineDeadError

    async def run():
        controller = _controller_with_snapshot(_backend_snapshot())
        started, finish = asyncio.Event(), asyncio.Event()

        async def old_snapshot():
            started.set()
            await finish.wait()
            if old_result == "dead":
                raise EngineDeadError()
            if old_result == "actor_dead":
                raise ray.exceptions.RayActorError()
            return _backend_snapshot()

        setattr(controller._engine._server_handle, operation, SimpleNamespace(remote=old_snapshot))
        poll = asyncio.create_task(getattr(controller, operation)())
        await started.wait()
        controller._engine._server_handle = _controller_with_snapshot(_backend_snapshot())._engine._server_handle
        finish.set()
        if operation == "check_health":
            expected_error = {
                "healthy": RuntimeError,
                "dead": EngineDeadError,
                "actor_dead": ray.exceptions.RayActorError,
            }[old_result]
            with pytest.raises(expected_error):
                await poll
        else:
            assert (await poll).backend_health == "unavailable"
        assert not controller._unavailable
        assert (await controller.snapshot()).backend_health == "healthy"

    asyncio.run(run())


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("operation", ["snapshot", "check_health"])
def test_evaluation_backend_check_marks_engine_dead_unavailable(wrapped, operation):
    import ray
    from vllm.v1.engine.exceptions import EngineDeadError

    controller = _controller_with_snapshot(_backend_snapshot())
    controller.relay_client = None
    error = EngineDeadError()
    if wrapped:
        error = ray.exceptions.RayTaskError("snapshot", "backend traceback", error)

    async def failed():
        raise error

    setattr(controller._engine._server_handle, operation, SimpleNamespace(remote=failed))
    if operation == "check_health":
        with pytest.raises(type(error)) as raised:
            asyncio.run(controller.check_health())
        assert raised.value is error
    else:
        assert asyncio.run(controller.snapshot()).backend_health == "unavailable"
    assert controller._unavailable
    assert asyncio.run(controller.snapshot()).backend_health == "unavailable"
