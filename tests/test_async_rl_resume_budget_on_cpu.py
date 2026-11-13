# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from laminar.train.experience_buffer.server import SampledBatch
from laminar.train.trainer import AsyncRLTrainer, AsyncRLTrainerConfig


class _RemoteMethod:
    def __init__(self, callback: Callable[..., Any]):
        self._callback = callback

    def remote(self, *args, **kwargs):  # noqa: ANN201
        return self._callback(*args, **kwargs)


class _ExperienceBufferProbe:
    def __init__(self):
        self.sample_calls = 0
        self.release_calls = 0

    def sample_remote(self, *, batch_size: int) -> object:
        assert batch_size == 1
        self.sample_calls += 1
        return object()

    async def release(self, refs: list[str]) -> None:
        assert refs == ["sample-0"]
        self.release_calls += 1


class _TrackerProbe:
    def __init__(self):
        self.commit_calls = 0
        self.collect_and_stage_remote_metrics = _RemoteMethod(self._collect)
        self.commit_metrics = _RemoteMethod(self._commit)
        self.record_event = _RemoteMethod(self._record_event)
        self.finalize = _RemoteMethod(self._finalize)

    async def _collect(self, *, step: int) -> None:
        assert step >= 0

    async def _commit(self, *, step: int, train_metrics: dict[str, Any]) -> None:
        assert train_metrics["training/global_step"] == step
        self.commit_calls += 1

    async def _record_event(self, **_event: Any) -> None:
        pass

    async def _finalize(self) -> None:
        pass


class _StopUnboundedLoop(RuntimeError):
    pass


def _build_trainer(*, restored_step: int, total_steps: int | None, stop_on_update: bool = False):
    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = SimpleNamespace(
        data=SimpleNamespace(train_batch_size=1),
        trainer=SimpleNamespace(save_freq=0),
        actor_rollout_ref=SimpleNamespace(
            actor=SimpleNamespace(nnodes=1, num_gpus_per_node=1),
            rollout=SimpleNamespace(n=1, nnodes=1, n_gpus_per_node=1),
        ),
    )
    trainer.trainer_config = AsyncRLTrainerConfig(total_training_steps=total_steps)
    trainer.global_steps = 0
    trainer._pending_relay_publish = None
    trainer._pending_checkpoint_metrics = None

    calls = {"prepare": 0, "actor_update": 0}
    buffer = _ExperienceBufferProbe()
    tracker = _TrackerProbe()
    trainer.experience_buffer_client = buffer
    trainer.tracker = tracker

    async def prepare() -> None:
        calls["prepare"] += 1
        trainer.global_steps = restored_step

    async def await_sample(_sample_ref: object, _metrics: dict[str, Any]):
        return SampledBatch(keys=["sample-0"], partition_id="train", group_context={}), 0.0

    def materialize(_batch: object, metrics: dict[str, Any]) -> object:
        metrics["timing/materialize_batch"] = 0.0
        return object()

    async def prepare_actor_update(_td: object, _metrics: dict[str, Any]) -> None:
        return None

    async def wait_for_publish(_metrics: dict[str, Any]) -> float:
        return 0.0

    def update_actor(_td: object, _metrics: dict[str, Any]) -> None:
        calls["actor_update"] += 1
        if stop_on_update:
            raise _StopUnboundedLoop

    trainer.prepare = prepare
    trainer._await_sample_or_relay_failure = await_sample
    trainer._materialize_actor_batch = materialize
    trainer._prepare_actor_update = prepare_actor_update
    trainer._wait_for_pending_relay_publish = wait_for_publish
    trainer._update_actor = update_actor
    trainer._schedule_relay_publish = lambda *, version: None
    return trainer, buffer, tracker, calls


@pytest.mark.parametrize("restored_step", [3, 4])
def test_fit_does_not_start_an_optimizer_step_when_resume_already_reached_budget(restored_step):
    trainer, buffer, tracker, calls = _build_trainer(restored_step=restored_step, total_steps=3)

    asyncio.run(trainer.fit())

    assert trainer.global_steps == restored_step
    assert calls == {"prepare": 1, "actor_update": 0}
    assert buffer.sample_calls == 0
    assert buffer.release_calls == 0
    assert tracker.commit_calls == 0


def test_fit_keeps_none_training_budget_unbounded():
    trainer, buffer, _tracker, calls = _build_trainer(
        restored_step=3,
        total_steps=None,
        stop_on_update=True,
    )

    with pytest.raises(_StopUnboundedLoop):
        asyncio.run(trainer.fit())

    assert trainer.global_steps == 4
    assert calls == {"prepare": 1, "actor_update": 1}
    assert buffer.sample_calls == 1
