# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest
import torch

from laminar.workers.metrics import add_optimizer_update_metrics, observe_optimizer_updates


class _FSDPEngine:
    def __init__(self, *, scaler=False):
        self.parameter = torch.nn.Parameter(torch.tensor(2.0))
        self.optimizer = torch.optim.SGD([self.parameter], lr=0.1)
        self.scaler = torch.amp.GradScaler("cpu") if scaler else None
        self.skip = False

    def optimizer_step(self):
        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(self.parameter * 2).backward()
            if self.skip:
                self.parameter.grad.fill_(float("inf"))
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            grad_norm = torch.tensor(float("inf") if self.skip else 2.0)
            self.parameter.grad = torch.tensor(2.0)
            if torch.isfinite(grad_norm):
                self.optimizer.step()
            else:
                self.optimizer.zero_grad()
        return "original gradient norm"


@pytest.mark.parametrize("scaler", [False, True], ids=["nonfinite_clip", "grad_scaler"])
def test_fsdp_observes_actual_step_and_skip_without_changing_optimizer(scaler):
    engine = _FSDPEngine(scaler=scaler)
    original_step = engine.optimizer_step
    original_hooks = dict(engine.optimizer._optimizer_step_post_hooks)

    with observe_optimizer_updates(engine, backend="fsdp") as stats:
        assert engine.optimizer_step() == "original gradient norm"
        updated_parameter = engine.parameter.detach().clone()
        engine.skip = True
        assert engine.optimizer_step() == "original gradient norm"

    assert updated_parameter.item() == pytest.approx(1.8)
    torch.testing.assert_close(engine.parameter, updated_parameter)
    assert stats.metrics() == {
        "update/optimizer_updates_attempted": 2,
        "update/optimizer_updates_completed": 1,
        "update/optimizer_updates_skipped": 1,
        "update/optimizer_updates_valid": 1,
    }
    assert engine.optimizer_step == original_step
    assert "optimizer_step" not in vars(engine)
    assert dict(engine.optimizer._optimizer_step_post_hooks) == original_hooks


def test_fsdp_fused_amp_step_outcome_is_unknown():
    engine = _FSDPEngine()
    engine.scaler = SimpleNamespace()
    engine.optimizer._step_supports_amp_scaling = True
    # An AMP-aware optimizer can be called yet skip internally; an ordinary
    # post-step hook alone cannot prove whether that call changed parameters.
    engine.optimizer_step = lambda: None

    with observe_optimizer_updates(engine, backend="fsdp") as stats:
        engine.optimizer_step()

    assert stats.metrics() == {
        "update/optimizer_updates_attempted": 1,
        "update/optimizer_updates_valid": 0,
    }


def test_fsdp_observation_without_public_optimizer_hook_is_unknown():
    engine = SimpleNamespace(optimizer=SimpleNamespace(step=lambda: None), optimizer_step=lambda: "norm")
    with observe_optimizer_updates(engine, backend="fsdp") as stats:
        assert engine.optimizer_step() == "norm"
    assert stats.attempted == 1
    assert stats.valid is False
    assert "update/optimizer_updates_completed" not in stats.metrics()


def test_observation_is_scoped_to_engine_instance():
    observed_engine = _FSDPEngine()
    other_engine = _FSDPEngine()
    with observe_optimizer_updates(observed_engine, backend="fsdp") as stats:
        other_engine.optimizer_step()
        observed_engine.optimizer_step()
    assert stats.completed == 1
    assert stats.attempted == 1
    assert other_engine.parameter.item() == pytest.approx(1.8)


def test_fsdp_failure_preserves_exception_and_restores_methods_and_hooks():
    engine = _FSDPEngine()
    failure = RuntimeError("optimizer failed")

    def fail():
        raise failure

    engine.optimizer_step = fail
    with pytest.raises(RuntimeError) as caught:
        with observe_optimizer_updates(engine, backend="fsdp") as stats:
            engine.optimizer_step()

    assert caught.value is failure
    assert engine.optimizer_step is fail
    assert not engine.optimizer._optimizer_step_post_hooks
    assert stats.attempted == 1
    assert stats.valid is False
    assert stats.skipped == 0


class _MegatronOptimizer:
    def __init__(self):
        self.successful = True
        self.calls = []

    def step(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.successful, "norm", "zero count"


class _MegatronEngine:
    def __init__(self, *, raise_on_skip=False):
        self.optimizer = _MegatronOptimizer()
        self.raise_on_skip = raise_on_skip

    def optimizer_step(self, *args, **kwargs):
        successful, norm, _zeros = self.optimizer.step(*args, **kwargs)
        if not successful and self.raise_on_skip:
            raise NotImplementedError("Megatron optimizer step failed")
        return norm


def test_megatron_observes_backend_success_flag_and_preserves_arguments():
    engine = _MegatronEngine()
    original_step = engine.optimizer.step
    with observe_optimizer_updates(engine, backend="megatron") as stats:
        assert engine.optimizer_step("closure", test_argument=3) == "norm"
        engine.optimizer.successful = False
        assert engine.optimizer_step() == "norm"

    assert engine.optimizer.calls == [(("closure",), {"test_argument": 3}), ((), {})]
    assert engine.optimizer.step == original_step
    assert "step" not in vars(engine.optimizer)
    assert stats.attempted == 2
    assert stats.completed == stats.skipped == 1
    assert stats.valid is True


def test_megatron_skip_keeps_existing_backend_error_behavior():
    engine = _MegatronEngine(raise_on_skip=True)
    engine.optimizer.successful = False
    original_step = engine.optimizer.step
    with pytest.raises(NotImplementedError, match="Megatron optimizer step failed"):
        with observe_optimizer_updates(engine, backend="megatron") as stats:
            engine.optimizer_step()
    assert engine.optimizer.step == original_step
    assert "optimizer_step" not in vars(engine)
    assert stats.attempted == 1
    assert stats.completed == 0
    assert stats.valid is False


def test_megatron_unrecognized_step_return_is_unknown_and_unchanged():
    result = object()
    engine = SimpleNamespace(optimizer=SimpleNamespace(step=lambda: result))
    engine.optimizer_step = lambda: engine.optimizer.step()
    with observe_optimizer_updates(engine, backend="megatron") as stats:
        assert engine.optimizer_step() is result
    assert stats.attempted == 1
    assert stats.valid is False
    assert "update/optimizer_updates_completed" not in stats.metrics()


def test_worker_counts_survive_tensordict_roundtrip_and_none_rank_output():
    from verl.utils import tensordict_utils as tu

    engine = _FSDPEngine()
    with observe_optimizer_updates(engine, backend="fsdp") as stats:
        engine.optimizer_step()
    output = tu.get_tensordict({}, non_tensor_dict={"metrics": {"loss": [0.5, 1.5]}})
    assert add_optimizer_update_metrics(output, stats) is output
    assert tu.get(output, "metrics") == {"loss": [0.5, 1.5], **stats.metrics()}
    assert add_optimizer_update_metrics(None, stats) is None


@pytest.mark.parametrize("backend", ["fsdp", "megatron"])
def test_actor_worker_exports_observed_rpc_counts_after_relay_fence(monkeypatch, backend):
    from verl.utils import tensordict_utils as tu

    from laminar.workers import fsdp_workers, megatron_workers

    events = []
    owner = fsdp_workers if backend == "fsdp" else megatron_workers
    worker_class = owner.AsyncActorWorker if backend == "fsdp" else owner.AsyncMegatronActorWorker
    engine = _FSDPEngine() if backend == "fsdp" else _MegatronEngine()
    worker = object.__new__(worker_class)
    worker.actor = SimpleNamespace(engine=engine)
    worker._first_actor_update_memory_logged = True

    async def wait_for_sending_to_relay():
        events.append("relay_done")

    def update_actor(_worker, _data):
        events.append("update")
        engine.optimizer_step()
        if backend == "fsdp":
            engine.skip = True
        else:
            engine.optimizer.successful = False
        engine.optimizer_step()
        return tu.get_tensordict({}, non_tensor_dict={"metrics": {"loss": [0.25, 0.75]}})

    worker.wait_for_sending_to_relay = wait_for_sending_to_relay
    monkeypatch.setattr(owner.ActorRolloutRefWorker, "update_actor", update_actor)
    output = asyncio.run(worker.update_actor(object()))

    assert events == ["relay_done", "update"]
    assert tu.get(output, "metrics") == {
        "loss": [0.25, 0.75],
        "update/optimizer_updates_attempted": 2,
        "update/optimizer_updates_completed": 1,
        "update/optimizer_updates_skipped": 1,
        "update/optimizer_updates_valid": 1,
    }
