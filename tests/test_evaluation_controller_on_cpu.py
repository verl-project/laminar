# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any

import pytest
from verl.workers.config import RolloutConfig

from laminar.global_request_router.router import EMPTY_WEIGHT_VERSION
from laminar.llm_server.controller import LLMServerController


class _Engine:
    def __init__(self, fail: str | None = None):
        self.config = RolloutConfig(name="vllm")
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any], int]] = []
        self.fail = fail
        self.controller: LLMServerController | None = None
        self._server_handle = SimpleNamespace(
            abort_all_requests=SimpleNamespace(remote=self.abort_all_requests),
            collective_rpc=SimpleNamespace(remote=self.collective_rpc),
            clear_kv_cache=SimpleNamespace(remote=self.clear_kv_cache),
            set_global_steps=SimpleNamespace(remote=self.set_global_steps),
            resume_generation=SimpleNamespace(remote=self.resume_generation),
        )

    def record(self, method: str, *args: Any, **kwargs: Any) -> None:
        assert self.controller is not None
        self.calls.append((method, args, kwargs, self.controller.weight_version))
        if self.fail == method:
            raise RuntimeError(f"{method} failed")

    async def init_standalone(self, *, resource_pool: Any) -> None:
        self.record("init_standalone", resource_pool=resource_pool)

    async def collective_rpc(self, method: str, **kwargs: Any) -> None:
        self.record(method, **kwargs)

    async def abort_all_requests(self, **kwargs: Any) -> None:
        self.record("pause", **kwargs)

    async def clear_kv_cache(self, **kwargs: Any) -> None:
        self.record("clear_kv_cache", **kwargs)

    async def set_global_steps(self, version: int) -> None:
        self.record("set_global_steps", version)

    async def resume_generation(self) -> None:
        self.record("resume")


def _controller(*, fail: str | None = None, relay: Any = None) -> tuple[LLMServerController, _Engine]:
    engine = _Engine(fail)
    controller = LLMServerController(relay_client=relay, engine=engine, name="evaluation-replica")
    engine.controller = controller
    return controller, engine


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [None, 1], ids=["training", "evaluation"])
async def test_engine_initialization_loads_once_and_sets_only_an_explicit_version(version: int | None) -> None:
    relay = object() if version is None else None
    controller, engine = _controller(relay=relay)
    pool = object()

    if version is None:
        await controller.init_engine(pool)
    else:
        await controller.init_engine(pool, weight_version=version)

    expected = [("init_standalone", (), {"resource_pool": pool}, EMPTY_WEIGHT_VERSION)]
    if version is not None:
        expected.append(("set_global_steps", (version,), {}, EMPTY_WEIGHT_VERSION))
    assert engine.calls == expected
    assert controller.weight_version == (EMPTY_WEIGHT_VERSION if version is None else version)
    assert controller.relay_client is relay


@pytest.mark.asyncio
async def test_hf_reload_commits_version_after_reload_cache_clear_and_global_step() -> None:
    controller, engine = _controller()
    controller.weight_version = 1
    path = "/mounted/checkpoints/step128/actor/model/huggingface"

    assert await controller.load_hf_checkpoint(path, weight_version=2) == 2

    assert engine.calls == [
        ("pause", (), {"reset_prefix_cache": False}, 1),
        ("reload_weights", (), {"kwargs": {"weights_path": path}}, 1),
        ("clear_kv_cache", (), {}, 1),
        ("set_global_steps", (2,), {}, 1),
        ("resume", (), {}, 1),
    ]
    assert controller.weight_version == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_stage", ["pause", "reload_weights", "clear_kv_cache", "set_global_steps", "resume"])
async def test_failed_hf_reload_never_marks_new_version_or_runs_later_stages(failed_stage: str) -> None:
    controller, engine = _controller(fail=failed_stage)
    controller.weight_version = 1
    stages = ["pause", "reload_weights", "clear_kv_cache", "set_global_steps", "resume"]

    with pytest.raises(RuntimeError, match=f"{failed_stage} failed"):
        await controller.load_hf_checkpoint("/mounted/checkpoint", weight_version=2)

    assert [call[0] for call in engine.calls] == stages[: stages.index(failed_stage) + 1]
    assert all(call[3] == 1 for call in engine.calls)
    assert controller.weight_version == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_stage", ["init_standalone", "set_global_steps"])
async def test_failed_initial_hf_load_does_not_mark_version(failed_stage: str) -> None:
    controller, engine = _controller(fail=failed_stage)
    with pytest.raises(RuntimeError, match=f"{failed_stage} failed"):
        await controller.init_engine(object(), weight_version=1)
    assert controller.weight_version == EMPTY_WEIGHT_VERSION
    assert [call[0] for call in engine.calls] == (
        ["init_standalone"] if failed_stage == "init_standalone" else ["init_standalone", "set_global_steps"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [-1, True, 0, 1])
async def test_hf_reload_rejects_non_integer_negative_and_non_increasing_versions(version: Any) -> None:
    controller, engine = _controller()
    controller.weight_version = 1
    with pytest.raises(ValueError):
        await controller.load_hf_checkpoint("/mounted/checkpoint", weight_version=version)
    assert engine.calls == []
    assert controller.weight_version == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("version", [-1, True])
async def test_initial_invalid_version_is_rejected_before_loading(version: Any) -> None:
    controller, engine = _controller()
    with pytest.raises(ValueError):
        await controller.init_engine(object(), weight_version=version)
    assert engine.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("has_relay", [False, True])
async def test_controller_rejects_the_other_weight_transport_before_touching_engine(has_relay: bool) -> None:
    controller, engine = _controller(relay=object() if has_relay else None)
    with pytest.raises(RuntimeError, match="Relay"):
        if has_relay:
            await controller.load_hf_checkpoint("/mounted/checkpoint", weight_version=1)
        else:
            await controller.load_weights(wait_for_version=1)
    assert engine.calls == []
    assert controller.weight_version == EMPTY_WEIGHT_VERSION


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_stage", [None, "restart", "set_global_steps", "check_health"])
async def test_standalone_hf_controller_restart_reloads_checkpoint_and_reopens(failed_stage: str | None) -> None:
    controller, engine = _controller(fail=failed_stage)
    engine.model_config = SimpleNamespace(path="/mounted/initial", local_path="/mounted/initial")

    async def restart() -> None:
        engine.record("restart")

    async def check_health() -> None:
        engine.record("check_health")

    engine.restart = restart
    engine._server_handle.check_health = SimpleNamespace(remote=check_health)
    controller.weight_version = 1
    controller._unavailable = True

    if failed_stage is not None:
        with pytest.raises(RuntimeError, match=f"{failed_stage} failed"):
            await controller.restart(2, checkpoint_path="/mounted/reloaded")
        assert controller._unavailable
    else:
        assert await controller.restart(2, checkpoint_path="/mounted/reloaded") == 2
        assert not controller._unavailable
        assert controller.weight_version == 2
        assert engine.model_config.path == "/mounted/reloaded"
        assert engine.model_config.local_path == "/mounted/reloaded"
        assert [call[0] for call in engine.calls] == ["restart", "set_global_steps", "check_health"]
