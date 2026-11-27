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


@pytest.mark.parametrize(
    ("enabled", "manager_api"),
    [
        (False, None),
        (False, "enable_trainer_owned_threaded_checkpointing"),
        (True, None),
        (True, "enable_threaded_checkpointing"),
        (True, "enable_trainer_owned_threaded_checkpointing"),
    ],
)
def test_actor_selects_shared_writer_only_when_enabled(enabled, manager_api, monkeypatch):
    selected = []
    manager = SimpleNamespace(
        checkpoint_config=SimpleNamespace(async_save=enabled),
        use_dist_checkpointing=True,
        use_megatron_fsdp=False,
        peft_cls=None,
        use_checkpoint_opt_param_scheduler=True,
    )
    if manager_api is not None:
        setattr(manager, manager_api, lambda **kwargs: selected.append(kwargs))
    worker = object.__new__(megatron_workers.AsyncMegatronActorWorker)
    worker._is_rollout = False
    worker.actor = SimpleNamespace(engine=SimpleNamespace(checkpoint_mananager=manager))
    worker.loss_fn = object()
    worker._async_checkpoint_manager = None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(megatron_workers.ActorRolloutRefWorker, "init_model", lambda _self: None)
    monkeypatch.setattr(worker, "_prewarm_grad_stats_communicators", lambda: None)

    compatible = manager_api == "enable_trainer_owned_threaded_checkpointing"
    if enabled and not compatible:
        with pytest.raises(RuntimeError, match=r"trainer\.async_checkpoint requires a verl build"):
            worker.init_model()
    else:
        worker.init_model()

    assert selected == ([{"cache_plans": True}] if enabled and compatible else [])
    assert worker._async_checkpoint_manager is (manager if enabled and compatible else None)


@pytest.mark.parametrize("async_enabled", [False, True])
def test_native_async_reload_releases_cached_allocations_after_the_loader(async_enabled, monkeypatch):
    events = []
    worker = object.__new__(megatron_workers.AsyncMegatronActorWorker)
    worker._async_checkpoint_manager = object() if async_enabled else None

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


@pytest.mark.parametrize("async_enabled", [False, True])
def test_async_checkpoint_prewarms_default_world_communicator(async_enabled, monkeypatch):
    events = []
    grad_group = object()
    world_group = object()
    probe = object()
    worker = object.__new__(megatron_workers.AsyncMegatronActorWorker)
    worker._async_checkpoint_manager = object() if async_enabled else None
    worker.actor = SimpleNamespace(
        engine=SimpleNamespace(optimizer=SimpleNamespace(get_grad_stats_parallel_group=lambda: grad_group))
    )
    worker._log_cuda_memory = lambda event: events.append(event)
    monkeypatch.setattr(megatron_workers.torch.distributed, "group", SimpleNamespace(WORLD=world_group))
    monkeypatch.setattr(megatron_workers.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(megatron_workers.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(megatron_workers.torch.cuda, "synchronize", lambda: events.append("cuda_synchronize"))
    monkeypatch.setattr(megatron_workers.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(megatron_workers.torch, "zeros", lambda *_args, **_kwargs: probe)
    monkeypatch.setattr(
        megatron_workers.torch.distributed,
        "all_reduce",
        lambda tensor, *, op, group: events.append(("all_reduce", tensor, op, group)),
    )

    worker._prewarm_grad_stats_communicators()

    expected = [
        "grad_stats_nccl_prewarm_before",
        ("all_reduce", probe, torch.distributed.ReduceOp.SUM, grad_group),
    ]
    if async_enabled:
        expected.append(("all_reduce", probe, torch.distributed.ReduceOp.SUM, world_group))
    expected.extend(("cuda_synchronize", "grad_stats_nccl_prewarm_after"))
    assert events == expected


def test_async_checkpoint_prewarm_rejects_missing_world_group(monkeypatch):
    worker = object.__new__(megatron_workers.AsyncMegatronActorWorker)
    worker._async_checkpoint_manager = object()
    worker.actor = SimpleNamespace(
        engine=SimpleNamespace(optimizer=SimpleNamespace(get_grad_stats_parallel_group=lambda: object()))
    )
    monkeypatch.setattr(megatron_workers.torch.distributed, "group", SimpleNamespace(WORLD=None))
    monkeypatch.setattr(megatron_workers.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(megatron_workers.torch.distributed, "is_initialized", lambda: True)

    with pytest.raises(RuntimeError, match="Megatron default process group is missing"):
        worker._prewarm_grad_stats_communicators()
