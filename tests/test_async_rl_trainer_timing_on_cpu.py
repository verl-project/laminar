# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from omegaconf import OmegaConf

from laminar.train.experience_buffer.client import ExperienceBufferClient
from laminar.train.experience_buffer.server import SampledBatch
from laminar.train.trainer import AsyncRLTrainer, AsyncRLTrainerConfig


class _RemoteMethod:
    def __init__(self, callback: Callable[..., Any]):
        self._callback = callback

    def remote(self, *args, **kwargs):  # noqa: ANN201
        return self._callback(*args, **kwargs)


class _FakeTracker:
    def __init__(self, *, collect_delay_s: float, commit_delay_s: float):
        self.events: list[str] = []
        self.committed_metrics: dict[int, dict[str, Any]] = {}
        self.collect_delay_s = collect_delay_s
        self.commit_delay_s = commit_delay_s
        self.set_next_step_to_log = _RemoteMethod(self._set_next_step_to_log)
        self.collect_and_stage_remote_metrics = _RemoteMethod(self._collect_and_stage_remote_metrics)
        self.commit_metrics = _RemoteMethod(self._commit_metrics)
        self.event_records: list[dict[str, Any]] = []
        self.record_event = _RemoteMethod(self._record_event)
        self.finalize = _RemoteMethod(self._finalize)

    async def _set_next_step_to_log(self, step: int) -> None:
        self.events.append(f"set-next:{step}")

    async def _collect_and_stage_remote_metrics(self, step: int) -> None:
        self.events.append(f"collect-start:{step}")
        await asyncio.sleep(self.collect_delay_s)
        self.events.append(f"collect-end:{step}")

    async def _commit_metrics(self, step: int, train_metrics: dict[str, Any]) -> None:
        self.events.append(f"commit-start:{step}")
        self.committed_metrics[step] = dict(train_metrics)
        await asyncio.sleep(self.commit_delay_s)
        self.events.append(f"commit-end:{step}")

    async def _record_event(self, **event: Any) -> None:
        self.event_records.append(event)
        self.events.append(f"event:{event['event_type']}:{event['step']}")

    async def _finalize(self) -> None:
        self.events.append("tracker-finalize")


class _FakeExperienceBuffer:
    def __init__(self, *, remove_delay_s: float, events: list[str]):
        self.remove_delay_s = remove_delay_s
        self.events = events
        self.sample = _RemoteMethod(lambda **_kwargs: object())
        self.remove = _RemoteMethod(self._remove)
        self.start = _RemoteMethod(self._start)

    async def _start(self) -> None:
        return None

    async def _remove(self, keys: list[str]) -> None:
        assert keys == ["sample-0"]
        self.events.append("remove-start")
        await asyncio.sleep(self.remove_delay_s)
        self.events.append("remove-end")


async def _run_one_step(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sample_delay_s: float = 0.0,
    relay_residual_delay_s: float = 0.0,
    remove_delay_s: float = 0.0,
    collect_delay_s: float = 0.0,
    commit_delay_s: float = 0.0,
    reported_publish_latency_s: float = 0.0,
    exercise_actor_update_pipeline: bool = False,
    save_freq: int = 0,
    total_training_steps: int = 1,
    initial_global_step: int = 0,
    weight_publish_interval: int = 1,
    training_strategy: str = "separate_async",
    checkpoint_delay_s: float = 0.0,
    captured_events: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    tracker = _FakeTracker(
        collect_delay_s=collect_delay_s,
        commit_delay_s=commit_delay_s,
    )
    events = tracker.events
    experience_buffer = _FakeExperienceBuffer(remove_delay_s=remove_delay_s, events=events)
    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 1},
            "trainer": {"save_freq": save_freq, "nnodes": 1, "n_gpus_per_node": 1},
            "actor_rollout_ref": {
                "rollout": {
                    "n": 1,
                    "nnodes": 1,
                    "n_gpus_per_node": 1,
                },
            },
        }
    )
    trainer.trainer_config = AsyncRLTrainerConfig(
        total_training_steps=total_training_steps,
        weight_publish_interval=weight_publish_interval,
        training_strategy=training_strategy,
    )
    trainer.experience_buffer = experience_buffer
    trainer.experience_buffer_client = ExperienceBufferClient(experience_buffer)
    trainer.tracker = tracker
    trainer.global_steps = initial_global_step
    trainer._pending_relay_publish = None
    trainer._prepared = False
    trainer._pending_checkpoint_metrics = None
    if trainer.colocate:

        async def pause_for_training() -> None:
            events.append("colocate-pause")

        async def resume_rollout(version: int) -> None:
            events.append(f"colocate-resume:{version}")

        def offload(device: str, **kwargs) -> None:
            assert device == "cpu"
            assert kwargs == {"model": True, "optimizer": True, "grad": True}
            events.append("actor-offload")

        trainer.actor_wg = SimpleNamespace(to=offload)
        trainer._colocate_rollout_manager = SimpleNamespace(
            pause_for_training=_RemoteMethod(pause_for_training),
            resume_rollout=_RemoteMethod(resume_rollout),
        )

    async def load_checkpoint() -> None:
        return None

    async def publish_actor_weights_to_relay(version: int) -> None:
        assert version >= 0

    async def await_sample(_sample_ref: Any, _metrics: dict[str, Any]):
        events.append("sample-start")
        await asyncio.sleep(sample_delay_s)
        events.append("sample-end")
        return SampledBatch(keys=["sample-0"], partition_id="train"), 0.0

    relay_wait_calls = 0

    async def wait_for_pending_relay_publish(metrics: dict[str, Any]) -> float:
        nonlocal relay_wait_calls
        relay_wait_calls += 1
        if relay_wait_calls > 1:
            return 0.0
        metrics["timing/relay_publish_total"] = reported_publish_latency_s
        events.append("relay-residual-start")
        await asyncio.sleep(relay_residual_delay_s)
        events.append("relay-residual-end")
        return relay_residual_delay_s

    trainer._validate_checkpoint_config = lambda: None
    trainer._load_checkpoint = load_checkpoint
    trainer.publish_actor_weights_to_relay = publish_actor_weights_to_relay
    trainer._await_sample_or_relay_failure = await_sample
    trainer._wait_for_pending_relay_publish = wait_for_pending_relay_publish
    trainer._schedule_relay_publish = lambda *, version: events.append(f"schedule-publish:{version}")

    async def save_checkpoint() -> dict[str, float]:
        events.append(f"save-checkpoint:{trainer.global_steps}")
        await asyncio.sleep(checkpoint_delay_s)
        return {"checkpoint/completed": 1.0}

    trainer._save_checkpoint = save_checkpoint

    def materialize_actor_batch(_batch: Any, metrics: dict[str, Any]) -> object:
        if exercise_actor_update_pipeline:
            events.append("materialize")
        metrics["timing/materialize_batch"] = 0.0
        return object()

    async def prepare_actor_update(_td: object, _metrics: dict[str, Any]) -> None:
        if exercise_actor_update_pipeline:
            events.append("old-log-prob")

    def update_actor(_td: object, _metrics: dict[str, Any]) -> None:
        if exercise_actor_update_pipeline:
            events.append("actor-update")

    trainer._materialize_actor_batch = materialize_actor_batch
    trainer._prepare_actor_update = prepare_actor_update
    trainer._update_actor = update_actor

    await trainer.fit()
    if captured_events is not None:
        captured_events.extend(tracker.event_records)
    return tracker.committed_metrics[initial_global_step + 1], events


def test_colocate_fences_gpu_work_until_rollout_sleeps_and_saves_before_resume(monkeypatch):
    metrics, events = asyncio.run(
        _run_one_step(
            monkeypatch,
            exercise_actor_update_pipeline=True,
            save_freq=1,
            total_training_steps=2,
            training_strategy="colocate_async",
        )
    )
    assert events.index("actor-offload") < events.index("colocate-resume:0") < events.index("sample-start")
    assert events.index("sample-end") < events.index("colocate-pause") < events.index("old-log-prob")
    # The first step returns memory only after its checkpoint. The final step
    # remains fenced so final checkpoint I/O cannot compete with rollout.
    checkpoint_index = events.index("save-checkpoint:1")
    assert events[checkpoint_index + 1 : checkpoint_index + 4] == [
        "event:checkpoint_completed:1",
        "actor-offload",
        "colocate-resume:0",
    ]
    assert events[-3:] == ["event:checkpoint_completed:2", "event:training_cycle_completed:2", "tracker-finalize"]
    assert metrics["colocate/rotation"] == 1.0


def test_colocate_resume_uses_the_completed_publish_version():
    async def exercise():
        trainer = object.__new__(AsyncRLTrainer)
        trainer.trainer_config = AsyncRLTrainerConfig(training_strategy="colocate_async")
        trainer._rollout_weight_version = 3

        async def publish_result():
            return {"relay/publish_succeeded": 1.0}

        task = asyncio.create_task(publish_result())
        trainer._pending_relay_publish = (7, task)
        metrics = {}
        await trainer._wait_for_pending_relay_publish(metrics)
        assert trainer._rollout_weight_version == 7
        assert trainer._pending_relay_publish is None
        assert metrics["relay/publish_succeeded"] == 1.0

    asyncio.run(exercise())


def test_delayed_remote_collection_increases_trainer_owned_step_wall(monkeypatch):
    fast_metrics, _ = asyncio.run(_run_one_step(monkeypatch, collect_delay_s=0.0))
    slow_metrics, _ = asyncio.run(_run_one_step(monkeypatch, collect_delay_s=0.05))

    assert slow_metrics["timing/tracker_collect_remote_metrics"] >= 0.045
    assert slow_metrics["timing/step"] - fast_metrics["timing/step"] >= 0.04


def test_cleanup_buffer_remove_and_remote_collection_are_inside_step_timer(monkeypatch):
    metrics, events = asyncio.run(
        _run_one_step(
            monkeypatch,
            remove_delay_s=0.02,
            collect_delay_s=0.02,
        )
    )

    assert metrics["timing/experience_buffer_remove"] >= 0.018
    assert metrics["timing/tracker_collect_remote_metrics"] >= 0.018
    assert metrics["timing/step"] >= 0.035
    assert events.index("remove-end") < events.index("collect-start:1")
    assert events.index("collect-end:1") < events.index("commit-start:1")


def test_sample_and_relay_publish_are_counted_only_by_natural_wall(monkeypatch):
    metrics, _ = asyncio.run(
        _run_one_step(
            monkeypatch,
            sample_delay_s=0.02,
            relay_residual_delay_s=0.02,
            commit_delay_s=0.05,
            reported_publish_latency_s=10.0,
        )
    )

    assert metrics["timing/sample"] >= 0.018
    assert metrics["timing/wait_for_prev_relay"] == pytest.approx(0.02)
    assert metrics["timing/relay_publish_total"] == 10.0
    assert 0.035 <= metrics["timing/step"] < 0.5


def test_old_log_prob_precedes_publish_fence_and_actor_update_follows_it(monkeypatch):
    _, events = asyncio.run(
        _run_one_step(
            monkeypatch,
            exercise_actor_update_pipeline=True,
        )
    )

    assert events.index("materialize") < events.index("old-log-prob")
    assert events.index("old-log-prob") < events.index("relay-residual-start")
    assert events.index("relay-residual-end") < events.index("actor-update")


def test_prepare_actor_update_runs_read_phase_off_driver_event_loop():
    trainer = object.__new__(AsyncRLTrainer)
    trainer.trainer_config = AsyncRLTrainerConfig()
    trainer.global_steps = 1
    driver_thread = threading.get_ident()
    read_phase_threads: list[int] = []

    def compute_actor_update_inputs(_td: object, _metrics: dict[str, Any]) -> None:
        read_phase_threads.append(threading.get_ident())

    trainer._compute_actor_update_inputs = compute_actor_update_inputs

    result = asyncio.run(trainer._prepare_actor_update(object(), {}))

    assert result is None
    assert read_phase_threads and read_phase_threads[0] != driver_thread


def test_tracker_commit_tail_is_outside_same_row_step_timer(monkeypatch):
    fast_metrics, _ = asyncio.run(_run_one_step(monkeypatch, commit_delay_s=0.0))
    slow_metrics, _ = asyncio.run(_run_one_step(monkeypatch, commit_delay_s=0.05))

    assert abs(slow_metrics["timing/step"] - fast_metrics["timing/step"]) < 0.025


def test_checkpoint_frequency_must_align_with_weight_publish_interval():
    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = OmegaConf.create({"trainer": {"save_freq": 3}})
    trainer.trainer_config = AsyncRLTrainerConfig(weight_publish_interval=2)
    trainer._pending_relay_publish = None

    with pytest.raises(ValueError, match="save_freq must align"):
        asyncio.run(trainer.fit())


def test_final_checkpoint_forces_publish_off_normal_cadence(monkeypatch):
    metrics, events = asyncio.run(
        _run_one_step(
            monkeypatch,
            save_freq=10,
            total_training_steps=12,
            initial_global_step=11,
            weight_publish_interval=5,
        )
    )

    assert metrics["trainer/weight_publish_scheduled"] == 1.0
    assert metrics["trainer/weight_publish_version"] == 12.0
    assert events.index("schedule-publish:12") < events.index("save-checkpoint:12")


def test_final_cycle_persists_checkpoint_and_commit_tails_without_rewriting_step(monkeypatch):
    captured = []
    step_metrics, events = asyncio.run(
        _run_one_step(
            monkeypatch,
            commit_delay_s=0.03,
            checkpoint_delay_s=0.03,
            save_freq=1,
            captured_events=captured,
        )
    )
    checkpoint, cycle = captured
    assert checkpoint["event_type"] == "checkpoint_completed"
    assert checkpoint["step"] == 1
    assert checkpoint["metrics"]["checkpoint/completed"] == 1
    assert cycle["event_type"] == "training_cycle_completed"
    assert cycle["metrics"]["timing/cycle"] - step_metrics["timing/step"] >= 0.05
    assert cycle["metrics"]["resource/configured_gpu_seconds/cycle"] == pytest.approx(
        2 * cycle["metrics"]["timing/cycle"]
    )
    assert "timing/cycle" not in step_metrics
    assert events[-1] == "tracker-finalize"
