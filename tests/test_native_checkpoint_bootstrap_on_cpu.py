# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from verl.workers.config import McoreEngineConfig

from laminar.workers import megatron_workers


class _Engine:
    def __init__(self, engine_config, events, *, fail_at=None):
        self.engine_config = engine_config
        self.original_config = engine_config
        self.events = events
        self.fail_at = fail_at
        self.optimizer_config = OmegaConf.create({"optimizer": "adam"})
        self.is_value_model = False
        self.model_config = OmegaConf.create({"hf_config": {}, "local_path": "/models/hf"})
        self.bridge = SimpleNamespace(load_hf_weights=self.load_hf_weights)
        self.provider = object()
        self.peft_cls = None
        self.enable_routing_replay = False

    def load_hf_weights(self, module, path, **kwargs):
        self.events.append(("hf", path, self.engine_config.use_dist_checkpointing))
        if self.fail_at == "build":
            raise RuntimeError("module failed")

    def _resolve_override_ddp_config(self):
        return {}

    def _build_megatron_module(self):
        from verl.workers.engine.megatron import transformer_impl

        # Exercise the pinned backend's actual loader-selection branch.
        return transformer_impl.MegatronEngine._build_megatron_module(self)

    def initialize(self):
        if self.fail_at == "before_build":
            raise RuntimeError("before module failed")
        self.module = self._build_megatron_module()
        assert self.engine_config is self.original_config
        self.events.append(("manager", self.engine_config.use_dist_checkpointing))
        if self.fail_at == "after_build":
            raise RuntimeError("after module failed")
        self.checkpoint_mananager = SimpleNamespace(use_dist_checkpointing=self.engine_config.use_dist_checkpointing)
        return "initialized"


@pytest.fixture
def build_worker(monkeypatch):
    import verl.utils.megatron_utils as megatron_utils
    from verl.workers.engine.megatron import transformer_impl

    monkeypatch.setattr(megatron_utils, "make_megatron_module", lambda **kwargs: ([object()], object()))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    def factory(*, native=True, path=None, fail_at=None):
        events = []
        config = McoreEngineConfig(use_dist_checkpointing=native, dist_checkpointing_path=path)
        engine = _Engine(config, events, fail_at=fail_at)
        monkeypatch.setattr(
            transformer_impl,
            "load_mcore_dist_weights",
            lambda module, native_path, **kwargs: events.append(("native", native_path)),
        )
        worker = object.__new__(megatron_workers._NativeCheckpointTrainingWorker)
        worker.engine = engine
        worker.engine_config = config
        return worker, engine, events

    return factory


def test_hf_bootstrap_restores_native_format_before_manager_construction(build_worker):
    worker, engine, events = build_worker()
    original_build = engine._build_megatron_module

    worker.reset()

    assert events == [("hf", "/models/hf", False), ("manager", True)]
    assert engine.checkpoint_mananager.use_dist_checkpointing is True
    assert engine.engine_config is worker.engine_config is engine.original_config
    assert engine.engine_config.use_dist_checkpointing is True
    assert engine.engine_config.dist_checkpointing_path is None
    assert engine._build_megatron_module == original_build
    assert megatron_workers.AsyncMegatronActorWorker.actor_worker_cls is type(worker)


@pytest.mark.parametrize("fail_at", ["before_build", "build", "after_build"])
def test_initialization_errors_restore_native_config_and_instance_method(build_worker, fail_at):
    worker, engine, events = build_worker(fail_at=fail_at)
    original_build = engine._build_megatron_module

    with pytest.raises(RuntimeError, match="failed"):
        worker.reset()

    assert engine.engine_config is worker.engine_config is engine.original_config
    assert engine.engine_config.use_dist_checkpointing is True
    assert engine._build_megatron_module == original_build
    assert not hasattr(engine, "checkpoint_mananager")


@pytest.mark.parametrize(
    ("native", "path", "expected"),
    [
        (True, "/checkpoints/bootstrap/model/dist_ckpt", ("native", "/checkpoints/bootstrap/model/dist_ckpt")),
        (True, "", ("native", "")),
        (False, None, ("hf", "/models/hf", False)),
    ],
)
def test_explicit_native_bootstrap_and_existing_hf_path_keep_backend_semantics(build_worker, native, path, expected):
    worker, engine, events = build_worker(native=native, path=path)
    original_build = engine._build_megatron_module

    worker.reset()

    assert events == [expected, ("manager", native)]
    assert engine.engine_config is worker.engine_config is engine.original_config
    assert engine.engine_config.use_dist_checkpointing is native
    assert engine.engine_config.dist_checkpointing_path == path
    assert engine._build_megatron_module == original_build


def test_bootstrap_hook_is_local_to_the_initializing_engine(build_worker):
    first, first_engine, first_events = build_worker()
    second, second_engine, second_events = build_worker()
    original_first_build = first_engine._build_megatron_module

    def build_and_check_other_engine():
        assert first_engine.engine_config.use_dist_checkpointing is False
        assert second_engine.engine_config.use_dist_checkpointing is True
        assert second_engine._build_megatron_module.__func__ is _Engine._build_megatron_module
        return original_first_build()

    first_engine._build_megatron_module = build_and_check_other_engine
    first.reset()
    second.reset()

    assert first_events == second_events == [("hf", "/models/hf", False), ("manager", True)]
    assert first_engine._build_megatron_module is build_and_check_other_engine


@pytest.mark.parametrize("async_enabled", [False, True])
def test_native_async_reload_releases_cached_allocations_after_the_loader(async_enabled, monkeypatch):
    events = []
    worker = object.__new__(megatron_workers.AsyncMegatronActorWorker)
    worker._async_checkpoint = object() if async_enabled else None

    def load(_self, path, remote, delete):
        events.append(("load", path, remote, delete))

    monkeypatch.setattr(megatron_workers.ActorRolloutRefWorker, "load_checkpoint", load)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append(("fence",)))
    monkeypatch.setattr(
        megatron_workers, "aggressive_empty_cache", lambda *, force_sync: events.append(("cleanup", force_sync))
    )

    worker.load_checkpoint("/checkpoint/actor", del_local_after_load=False)

    expected = [("load", "/checkpoint/actor", None, False)]
    if async_enabled:
        expected.extend((("fence",), ("cleanup", True)))
    assert events == expected
