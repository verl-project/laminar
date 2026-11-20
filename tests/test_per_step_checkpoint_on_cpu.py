# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Driver ownership and commit protocol for native async per-step snapshots."""

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf
from verl.utils.checkpoint.checkpoint_manager import get_checkpoint_tracker_filename

from laminar.train.trainer import AsyncRLTrainer


class _RemoteMethod:
    def __init__(self, callback):
        self.callback = callback

    def remote(self, *args, **kwargs):
        # Ray starts remote work when submitted, independently of driver await.
        return asyncio.create_task(self.callback(*args, **kwargs))


class _Dispatcher:
    def __init__(self, *, error=None):
        self.error = error
        self.started = {}
        self.release = {}
        self.completed = []
        self.save_checkpoint = _RemoteMethod(self._save)

    async def _save(self, path):
        step = int(Path(path).parent.name.removeprefix("global_step_"))
        self.started.setdefault(step, asyncio.Event()).set()
        await self.release.setdefault(step, asyncio.Event()).wait()
        self.completed.append(step)
        if self.error is not None:
            raise self.error
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(f"cursor-at-step-{step}\n")

    async def wait_started(self, step):
        await asyncio.wait_for(self.started.setdefault(step, asyncio.Event()).wait(), timeout=1)

    def complete(self, step):
        self.release.setdefault(step, asyncio.Event()).set()


class _ActorWorkerGroup:
    def __init__(self, *, save_error=None, finalize_error=None):
        self.save_error = save_error
        self.finalize_error = finalize_error
        self.save_calls = []
        self.finalized_steps = []
        self.pending = None

    def save_checkpoint(self, **kwargs):
        assert self.pending is None, "driver must drain the previous native snapshot before staging another"
        self.save_calls.append(kwargs)
        self.pending = kwargs
        Path(kwargs["local_path"]).mkdir(parents=True, exist_ok=True)
        if self.save_error is not None:
            raise self.save_error

    def finalize_checkpoint(self):
        assert self.pending is not None
        pending, self.pending = self.pending, None
        self.finalized_steps.append(pending["global_step"])
        if self.finalize_error is not None:
            raise self.finalize_error
        (Path(pending["local_path"]) / "actor-state").write_text(f"step={pending['global_step']}\n")


class _Tracker:
    def __init__(self):
        self.events = []
        self.record_event = _RemoteMethod(self._record_event)

    async def _record_event(self, **kwargs):
        self.events.append(kwargs)


def _trainer(tmp_path, *, dispatcher=None, actor_wg=None):
    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "async_checkpoint": True,
                "default_local_dir": str(tmp_path),
                "default_hdfs_dir": None,
            },
            "actor_rollout_ref": {
                "model": {"lora_rank": 0},
                "actor": {
                    "strategy": "megatron",
                    "optim": {"use_checkpoint_opt_param_scheduler": True},
                    "megatron": {"use_dist_checkpointing": True, "use_megatron_fsdp": False},
                    "checkpoint": {
                        "async_save": True,
                        "save_contents": ["model", "optimizer", "extra"],
                        "load_contents": ["model", "optimizer", "extra"],
                    },
                },
            },
        }
    )
    trainer.trainer_config = SimpleNamespace(
        max_actor_ckpt_to_keep=1, training_strategy="separate_async", async_checkpoint=True
    )
    trainer.global_steps = 12
    trainer.rollout_dispatcher = dispatcher or _Dispatcher()
    trainer.actor_wg = actor_wg or _ActorWorkerGroup()
    # An async actor snapshot does not require publishing this step to Relay.
    trainer.relay_client = SimpleNamespace(get_loadable_version=AsyncMock(return_value=8))
    trainer.tracker = _Tracker()
    trainer._pending_checkpoint = None
    trainer._pending_checkpoint_metrics = None
    return trainer


def _old_checkpoint(root, step=8):
    directory = root / f"global_step_{step}"
    (directory / "actor").mkdir(parents=True)
    (directory / "data.pt").write_text("old cursor\n")
    Path(get_checkpoint_tracker_filename(str(root))).write_text(f"{step}\n")
    return directory


def _marker(root):
    return Path(get_checkpoint_tracker_filename(str(root))).read_text()


def test_async_stage_returns_before_io_and_completion_keeps_the_snapshot_step(tmp_path):
    async def scenario():
        old = _old_checkpoint(tmp_path)
        trainer = _trainer(tmp_path)
        dispatcher = trainer.rollout_dispatcher

        scheduled = await trainer._save_checkpoint()
        await dispatcher.wait_started(12)
        assert scheduled["checkpoint/scheduled"] == 1.0
        assert scheduled["checkpoint/staged_step"] == 12.0
        assert _marker(tmp_path) == "8\n"
        assert old.is_dir()
        assert trainer.actor_wg.finalized_steps == []
        trainer.relay_client.get_loadable_version.assert_not_awaited()

        # Completion must run at an explicit safe driver boundary. Even cursor
        # I/O completion must not launch worker collectives in the background.
        dispatcher.complete(12)
        await asyncio.sleep(0)
        assert _marker(tmp_path) == "8\n"
        assert trainer.actor_wg.finalized_steps == []
        trainer.global_steps = 13
        completed = await trainer._finish_pending_checkpoint()

        assert completed["checkpoint/completed"] == 1.0
        assert completed["checkpoint/step"] == 12.0
        assert _marker(tmp_path) == "12\n"
        assert trainer.actor_wg.finalized_steps == [12]
        assert (tmp_path / "global_step_12" / "actor" / "actor-state").read_text() == "step=12\n"
        assert not old.exists()
        assert trainer.actor_wg.save_calls[0]["max_ckpt_to_keep"] is None
        assert [event["step"] for event in trainer.tracker.events] == [12]
        assert trainer.tracker.events[0]["event_type"] == "checkpoint_completed"
        assert trainer.tracker.events[0]["event_id"] == "checkpoint:12"
        assert trainer._pending_checkpoint is None
        assert await trainer._finish_pending_checkpoint() == {}

    asyncio.run(scenario())


@pytest.mark.parametrize("failed_owner", ["actor", "cursor", "both"])
def test_failed_async_io_preserves_previous_commit_and_settles_both_owners(tmp_path, failed_owner):
    async def scenario():
        old = _old_checkpoint(tmp_path)
        dispatcher = _Dispatcher(error=RuntimeError("cursor save failed") if failed_owner != "actor" else None)
        actor = _ActorWorkerGroup(
            finalize_error=RuntimeError("actor finalize failed") if failed_owner != "cursor" else None
        )
        trainer = _trainer(tmp_path, dispatcher=dispatcher, actor_wg=actor)
        await trainer._save_checkpoint()
        await dispatcher.wait_started(12)
        finish_task = asyncio.create_task(trainer._finish_pending_checkpoint())
        await asyncio.sleep(0)
        assert not finish_task.done()
        assert _marker(tmp_path) == "8\n"
        dispatcher.complete(12)
        expected_error = "cursor save failed" if failed_owner == "cursor" else "actor finalize failed"
        with pytest.raises(RuntimeError, match=expected_error):
            await finish_task

        assert actor.finalized_steps == [12]
        assert dispatcher.completed == [12]
        assert _marker(tmp_path) == "8\n"
        assert old.is_dir()
        assert trainer.tracker.events == []
        assert trainer._pending_checkpoint is not None

    asyncio.run(scenario())


def test_next_save_applies_backpressure_before_staging_another_snapshot(tmp_path):
    async def scenario():
        _old_checkpoint(tmp_path)
        trainer = _trainer(tmp_path)
        dispatcher = trainer.rollout_dispatcher
        await trainer._save_checkpoint()
        await dispatcher.wait_started(12)
        trainer.global_steps = 13

        next_save = asyncio.create_task(trainer._save_checkpoint())
        await asyncio.sleep(0)
        assert not next_save.done()
        assert [call["global_step"] for call in trainer.actor_wg.save_calls] == [12]
        assert _marker(tmp_path) == "8\n"
        dispatcher.complete(12)
        metrics = await asyncio.wait_for(next_save, timeout=1)
        await dispatcher.wait_started(13)
        assert [call["global_step"] for call in trainer.actor_wg.save_calls] == [12, 13]
        assert metrics["checkpoint/staged_step"] == 13.0
        assert metrics["checkpoint/backpressure_time_s"] >= 0
        assert _marker(tmp_path) == "12\n"
        assert (tmp_path / "global_step_12").is_dir()

        dispatcher.complete(13)
        await trainer._finish_pending_checkpoint()
        assert _marker(tmp_path) == "13\n"
        assert sorted(path.name for path in tmp_path.glob("global_step_*")) == ["global_step_13"]
        assert [event["step"] for event in trainer.tracker.events] == [12, 13]

    asyncio.run(scenario())


def test_previous_async_failure_prevents_staging_next_checkpoint(tmp_path):
    async def scenario():
        _old_checkpoint(tmp_path)
        trainer = _trainer(tmp_path, actor_wg=_ActorWorkerGroup(finalize_error=RuntimeError("disk full")))
        await trainer._save_checkpoint()
        await trainer.rollout_dispatcher.wait_started(12)
        trainer.rollout_dispatcher.complete(12)
        trainer.global_steps = 13
        with pytest.raises(RuntimeError, match="disk full"):
            await trainer._save_checkpoint()
        assert [call["global_step"] for call in trainer.actor_wg.save_calls] == [12]
        assert _marker(tmp_path) == "8\n"

    asyncio.run(scenario())


def test_actor_staging_failure_settles_cursor_without_advancing_marker(tmp_path):
    async def scenario():
        old = _old_checkpoint(tmp_path)
        dispatcher = _Dispatcher(error=RuntimeError("cursor save also failed"))
        actor = _ActorWorkerGroup(save_error=RuntimeError("actor staging failed"))
        trainer = _trainer(tmp_path, dispatcher=dispatcher, actor_wg=actor)
        save_task = asyncio.create_task(trainer._save_checkpoint())
        await dispatcher.wait_started(12)
        await asyncio.sleep(0)
        assert not save_task.done()
        dispatcher.complete(12)
        with pytest.raises(RuntimeError, match="actor staging failed") as exc_info:
            await save_task
        assert any("cursor save also failed" in note for note in getattr(exc_info.value, "__notes__", ()))
        assert actor.finalized_steps == [12]
        assert dispatcher.completed == [12]
        assert _marker(tmp_path) == "8\n"
        assert old.is_dir()
        assert trainer.tracker.events == []

    asyncio.run(scenario())


def test_async_per_step_checkpoint_configuration_is_accepted(tmp_path):
    _trainer(tmp_path)._validate_checkpoint_config()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("actor_rollout_ref.actor.checkpoint.async_save", False, "async_save"),
        ("actor_rollout_ref.actor.strategy", "fsdp2", "Megatron|megatron"),
        ("actor_rollout_ref.actor.megatron.use_dist_checkpointing", False, "use_dist_checkpointing"),
        ("actor_rollout_ref.actor.checkpoint.save_contents", ["model", "optimizer", "extra", "hf_model"], "hf_model"),
    ],
)
def test_async_checkpoint_rejects_unsupported_snapshot_configuration(tmp_path, field, value, message):
    trainer = _trainer(tmp_path)
    OmegaConf.update(trainer.config, field, value)
    with pytest.raises(ValueError, match=message):
        trainer._validate_checkpoint_config()


@pytest.mark.parametrize("checkpoint_due", [False, True])
def test_final_step_flushes_pending_checkpoint_even_when_no_new_save_is_due(tmp_path, checkpoint_due):
    async def scenario():
        trainer = _trainer(tmp_path)
        trainer._resume_colocate_rollout = AsyncMock()
        if not checkpoint_due:
            await trainer._save_checkpoint()
            await trainer.rollout_dispatcher.wait_started(12)
        trainer.rollout_dispatcher.complete(12)
        await trainer._save_checkpoint_and_resume_rollout(checkpoint_due=checkpoint_due, is_last_step=True)
        assert _marker(tmp_path) == "12\n"
        assert trainer._pending_checkpoint is None
        assert trainer.actor_wg.finalized_steps == [12]
        assert trainer.tracker.events[-1]["event_type"] == "checkpoint_completed"

    asyncio.run(scenario())


def test_async_checkpoint_does_not_force_per_step_relay_publication(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.trainer_config.weight_publish_interval = 4
    assert trainer._step_should_publish_weights(1, 9, checkpoint_due=True) is False
    assert trainer._step_should_publish_weights(4, 9, checkpoint_due=True) is True
    assert trainer._step_should_publish_weights(9, 9, checkpoint_due=True) is False


def test_async_fit_allows_per_step_saving_with_sparse_relay_and_flushes_before_tracker_finalize(monkeypatch):
    from test_async_rl_trainer_timing_on_cpu import _run_one_step

    monkeypatch.setattr(AsyncRLTrainer, "async_checkpoint", property(lambda _self: True))

    async def finish(trainer):
        trainer.tracker.events.append("checkpoint-flushed")
        return {}

    monkeypatch.setattr(AsyncRLTrainer, "_finish_pending_checkpoint", finish)
    _, events = asyncio.run(_run_one_step(monkeypatch, save_freq=1, total_training_steps=3, weight_publish_interval=4))
    assert not any(event.startswith("schedule-publish:") for event in events)
    assert [event for event in events if event.startswith("save-checkpoint:")] == [
        "save-checkpoint:1",
        "save-checkpoint:2",
        "save-checkpoint:3",
    ]
    assert events.index("checkpoint-flushed") < events.index("tracker-finalize")


def test_async_fit_does_not_finalize_tracker_when_final_checkpoint_fails(monkeypatch):
    from test_async_rl_trainer_timing_on_cpu import _run_one_step

    monkeypatch.setattr(AsyncRLTrainer, "async_checkpoint", property(lambda _self: True))
    events = []

    async def fail_finish(trainer):
        events.append(trainer.tracker.events)
        raise RuntimeError("final checkpoint failed")

    monkeypatch.setattr(AsyncRLTrainer, "_finish_pending_checkpoint", fail_finish)
    with pytest.raises(RuntimeError, match="final checkpoint failed"):
        asyncio.run(_run_one_step(monkeypatch, save_freq=1))
    assert len(events) == 1
    assert "tracker-finalize" not in events[0]


def test_async_fit_fences_an_actual_relay_publish_before_checkpoint_collectives(monkeypatch):
    from test_async_rl_trainer_timing_on_cpu import _run_one_step

    monkeypatch.setattr(AsyncRLTrainer, "async_checkpoint", property(lambda _self: True))
    original_should_publish = AsyncRLTrainer._step_should_publish_weights

    def should_publish(trainer, step, total_steps, *, checkpoint_due):
        # Install after the update, so only this step's current-publication
        # fence can place this event before the checkpoint staging RPC.
        async def wait_for_current_relay(_metrics):
            trainer.tracker.events.append(f"current-relay-fenced:{trainer.global_steps}")
            return 0.0

        trainer._wait_for_pending_relay_publish = wait_for_current_relay
        return original_should_publish(trainer, step, total_steps, checkpoint_due=checkpoint_due)

    monkeypatch.setattr(AsyncRLTrainer, "_step_should_publish_weights", should_publish)
    _, events = asyncio.run(
        _run_one_step(
            monkeypatch,
            save_freq=1,
            initial_global_step=3,
            total_training_steps=5,
            weight_publish_interval=4,
        )
    )
    assert events.index("schedule-publish:4") < events.index("current-relay-fenced:4")
    assert events.index("current-relay-fenced:4") < events.index("save-checkpoint:4")
    assert "schedule-publish:5" not in events
