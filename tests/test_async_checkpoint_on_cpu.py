# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
from omegaconf import OmegaConf
from verl.utils.checkpoint.checkpoint_manager import get_checkpoint_tracker_filename


class _RemoteMethod:
    def __init__(self, callback):
        self.callback = callback
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.callback(*args, **kwargs)


class _Dispatcher:
    def __init__(self):
        self.saved_paths = []
        self.loaded_paths = []
        self.save_checkpoint = _RemoteMethod(self._save)
        self.load_checkpoint = _RemoteMethod(self._load)

    async def _save(self, path):
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"prompt_cursor": 17}, checkpoint_path)
        self.saved_paths.append(path)

    async def _load(self, path):
        self.loaded_paths.append(path)


class _BlockingDispatcher(_Dispatcher):
    def __init__(self, *, error=None):
        super().__init__()
        self.error = error
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.completed = asyncio.Event()

    async def _save(self, path):
        self.started.set()
        try:
            await self.release.wait()
            await super()._save(path)
            if self.error is not None:
                raise self.error
        finally:
            self.completed.set()


class _ActorWorkerGroup:
    def __init__(self, *, save_error=None):
        self.save_error = save_error
        self.save_calls = []
        self.load_calls = []

    def save_checkpoint(self, **kwargs):
        self.save_calls.append(kwargs)
        if self.save_error is not None:
            raise self.save_error
        actor_path = Path(kwargs["local_path"])
        actor_path.mkdir(parents=True, exist_ok=True)
        (actor_path / "actor-state").write_text("complete\n")

    def load_checkpoint(self, **kwargs):
        self.load_calls.append(kwargs)


def _checkpoint_config(checkpoint_root: Path, *, resume_mode="resume_path", resume_from_path=None):
    if resume_from_path is None:
        resume_from_path = checkpoint_root / "global_step_12"
    return OmegaConf.create(
        {
            "trainer": {
                "resume_mode": resume_mode,
                "resume_from_path": str(resume_from_path),
                "default_local_dir": str(checkpoint_root),
                "default_hdfs_dir": None,
                "del_local_ckpt_after_load": False,
            },
            "actor_rollout_ref": {
                "actor": {
                    "strategy": "fsdp2",
                    "checkpoint": {
                        "save_contents": ["model", "optimizer", "extra"],
                        "load_contents": ["model", "optimizer", "extra"],
                        "async_save": False,
                    },
                }
            },
        }
    )


def _trainer(
    checkpoint_root: Path,
    *,
    resume_mode="resume_path",
    resume_from_path=None,
    rollout_dispatcher=None,
    actor_wg=None,
    relay_version=12,
):
    from laminar.train.trainer import AsyncRLTrainer

    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = _checkpoint_config(
        checkpoint_root,
        resume_mode=resume_mode,
        resume_from_path=resume_from_path,
    )
    trainer.rollout_dispatcher = rollout_dispatcher or _Dispatcher()
    trainer.actor_wg = actor_wg or _ActorWorkerGroup()
    trainer.relay_client = SimpleNamespace(get_loadable_version=AsyncMock(return_value=relay_version))
    trainer.trainer_config = SimpleNamespace(max_actor_ckpt_to_keep=3)
    trainer.global_steps = 12
    return trainer


def _write_minimal_checkpoint(checkpoint_root: Path, *, step=12):
    checkpoint_dir = checkpoint_root / f"global_step_{step}"
    (checkpoint_dir / "actor").mkdir(parents=True)
    torch.save({"prompt_cursor": 17}, checkpoint_dir / "data.pt")
    return checkpoint_dir


def test_async_trainer_checkpoint_saves_actor_prompt_cursor_and_latest_marker_only(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    trainer = _trainer(checkpoint_root)

    metrics = asyncio.run(trainer._save_checkpoint())

    checkpoint_dir = checkpoint_root / "global_step_12"
    assert trainer.actor_wg.save_calls == [
        {
            "local_path": str(checkpoint_dir / "actor"),
            "hdfs_path": None,
            "global_step": 12,
            "max_ckpt_to_keep": None,
        }
    ]
    assert trainer.rollout_dispatcher.saved_paths == [str(checkpoint_dir / "data.pt")]
    assert Path(get_checkpoint_tracker_filename(str(checkpoint_root))).read_text() == "12\n"
    assert sorted(path.name for path in checkpoint_dir.iterdir()) == ["actor", "data.pt"]
    assert not (checkpoint_dir / "experience_buffer").exists()
    assert metrics["checkpoint/completed"] == 1.0
    assert metrics["checkpoint/step"] == 12.0


def test_async_trainer_checkpoint_rejects_actor_only_hdfs_before_owner_calls(tmp_path):
    trainer = _trainer(tmp_path / "checkpoints")
    trainer.config.trainer.default_hdfs_dir = "hdfs://checkpoint-root/experiment"

    with pytest.raises(NotImplementedError, match="actor-only default_hdfs_dir"):
        asyncio.run(trainer._save_checkpoint())

    trainer.relay_client.get_loadable_version.assert_not_awaited()
    assert trainer.actor_wg.save_calls == []
    assert trainer.rollout_dispatcher.save_checkpoint.calls == []


def test_async_trainer_auto_resume_rejects_actor_only_hdfs_without_local_marker(tmp_path):
    trainer = _trainer(tmp_path / "checkpoints", resume_mode="auto")
    trainer.config.trainer.default_hdfs_dir = "hdfs://checkpoint-root/experiment"

    with pytest.raises(NotImplementedError, match="actor-only default_hdfs_dir"):
        asyncio.run(trainer._load_checkpoint())

    assert trainer.actor_wg.load_calls == []
    assert trainer.rollout_dispatcher.load_checkpoint.calls == []


def test_async_trainer_retention_prunes_only_after_complete_checkpoint_commit(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    _write_minimal_checkpoint(checkpoint_root, step=8)
    _write_minimal_checkpoint(checkpoint_root, step=10)
    Path(get_checkpoint_tracker_filename(str(checkpoint_root))).write_text("10\n")
    trainer = _trainer(checkpoint_root)
    trainer.trainer_config.max_actor_ckpt_to_keep = 1

    asyncio.run(trainer._save_checkpoint())

    assert Path(get_checkpoint_tracker_filename(str(checkpoint_root))).read_text() == "12\n"
    assert sorted(path.name for path in checkpoint_root.glob("global_step_*")) == ["global_step_12"]
    assert trainer.actor_wg.save_calls[0]["max_ckpt_to_keep"] is None


def test_async_trainer_retention_preserves_current_step_when_higher_steps_exist(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    _write_minimal_checkpoint(checkpoint_root, step=10)
    _write_minimal_checkpoint(checkpoint_root, step=12)
    trainer = _trainer(checkpoint_root, resume_from_path=checkpoint_root / "global_step_8")
    trainer.trainer_config.max_actor_ckpt_to_keep = 1
    trainer.global_steps = 8
    trainer.relay_client.get_loadable_version.return_value = 8

    asyncio.run(trainer._save_checkpoint())

    assert Path(get_checkpoint_tracker_filename(str(checkpoint_root))).read_text() == "8\n"
    assert sorted(path.name for path in checkpoint_root.glob("global_step_*")) == ["global_step_8"]


def test_async_trainer_retention_failure_does_not_fail_committed_checkpoint(tmp_path, monkeypatch):
    checkpoint_root = tmp_path / "checkpoints"
    old_checkpoint = _write_minimal_checkpoint(checkpoint_root, step=8)
    trainer = _trainer(checkpoint_root)
    trainer.trainer_config.max_actor_ckpt_to_keep = 1

    def fail_prune(_path):
        raise OSError("transient FUSE failure")

    monkeypatch.setattr("laminar.train.trainer.shutil.rmtree", fail_prune)

    metrics = asyncio.run(trainer._save_checkpoint())

    assert metrics["checkpoint/completed"] == 1.0
    assert metrics["checkpoint/retention_pruned_count"] == 0.0
    assert metrics["checkpoint/retention_failed_count"] == 1.0
    assert Path(get_checkpoint_tracker_filename(str(checkpoint_root))).read_text() == "12\n"
    assert old_checkpoint.is_dir()
    assert (checkpoint_root / "global_step_12" / "actor").is_dir()


def test_failed_prompt_cursor_save_preserves_last_committed_checkpoint_with_retention_one(tmp_path):
    async def scenario():
        checkpoint_root = tmp_path / "checkpoints"
        old_checkpoint = _write_minimal_checkpoint(checkpoint_root, step=8)
        Path(get_checkpoint_tracker_filename(str(checkpoint_root))).write_text("8\n")
        dispatcher = _BlockingDispatcher(error=RuntimeError("cursor save failed"))
        trainer = _trainer(checkpoint_root, rollout_dispatcher=dispatcher)
        trainer.trainer_config.max_actor_ckpt_to_keep = 1

        save_task = asyncio.create_task(trainer._save_checkpoint())
        await asyncio.wait_for(dispatcher.started.wait(), timeout=1)
        dispatcher.release.set()
        with pytest.raises(RuntimeError, match="cursor save failed"):
            await save_task

        assert Path(get_checkpoint_tracker_filename(str(checkpoint_root))).read_text() == "8\n"
        assert (old_checkpoint / "actor").is_dir()
        assert (old_checkpoint / "data.pt").is_file()
        assert trainer.actor_wg.save_calls[0]["max_ckpt_to_keep"] is None

        restored = _trainer(checkpoint_root, resume_mode="auto")
        restored.global_steps = 0
        assert await restored._load_checkpoint() is True
        assert restored.global_steps == 8

    asyncio.run(scenario())


def test_async_trainer_checkpoint_requires_exact_committed_relay_version(tmp_path):
    trainer = _trainer(tmp_path / "checkpoints", relay_version=11)

    with pytest.raises(RuntimeError, match="exact committed Relay version"):
        asyncio.run(trainer._save_checkpoint())

    assert trainer.actor_wg.save_calls == []
    assert trainer.rollout_dispatcher.save_checkpoint.calls == []


def test_actor_save_failure_still_settles_prompt_cursor_save_before_failing(tmp_path):
    async def scenario():
        dispatcher = _BlockingDispatcher(error=RuntimeError("cursor save also failed"))
        actor_wg = _ActorWorkerGroup(save_error=RuntimeError("actor save failed"))
        trainer = _trainer(tmp_path / "checkpoints", rollout_dispatcher=dispatcher, actor_wg=actor_wg)

        save_task = asyncio.create_task(trainer._save_checkpoint())
        await asyncio.wait_for(dispatcher.started.wait(), timeout=1)
        await asyncio.sleep(0)
        assert not save_task.done()
        dispatcher.release.set()
        with pytest.raises(RuntimeError, match="actor save failed") as exc_info:
            await save_task

        assert dispatcher.completed.is_set()
        assert any("cursor save also failed" in note for note in getattr(exc_info.value, "__notes__", ()))
        assert not Path(get_checkpoint_tracker_filename(str(tmp_path / "checkpoints"))).exists()

    asyncio.run(scenario())


def test_async_trainer_checkpoint_restores_actor_prompt_cursor_and_step(tmp_path):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_dir = _write_minimal_checkpoint(checkpoint_root)
    trainer = _trainer(checkpoint_root, resume_from_path=checkpoint_dir)
    trainer.global_steps = 0

    restored = asyncio.run(trainer._load_checkpoint())

    assert restored is True
    assert trainer.global_steps == 12
    assert trainer.actor_wg.load_calls == [
        {
            "local_path": str(checkpoint_dir / "actor"),
            "hdfs_path": None,
            "del_local_after_load": False,
        }
    ]
    assert trainer.rollout_dispatcher.loaded_paths == [str(checkpoint_dir / "data.pt")]


@pytest.mark.parametrize("missing", ["actor", "data.pt"])
def test_async_trainer_checkpoint_rejects_incomplete_checkpoint_before_actor_load(tmp_path, missing):
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_dir = _write_minimal_checkpoint(checkpoint_root)
    target = checkpoint_dir / missing
    if target.is_dir():
        target.rmdir()
    else:
        target.unlink()
    trainer = _trainer(checkpoint_root, resume_from_path=checkpoint_dir)

    with pytest.raises(RuntimeError, match="incomplete async-RL trainer checkpoint"):
        asyncio.run(trainer._load_checkpoint())

    assert trainer.actor_wg.load_calls == []
    assert trainer.rollout_dispatcher.load_checkpoint.calls == []


def test_async_trainer_auto_resume_without_latest_marker_starts_fresh(tmp_path):
    trainer = _trainer(tmp_path / "checkpoints", resume_mode="auto")

    assert asyncio.run(trainer._load_checkpoint()) is False
    assert trainer.actor_wg.load_calls == []
    assert trainer.rollout_dispatcher.load_checkpoint.calls == []


def test_async_trainer_checkpoint_requires_full_actor_train_state(tmp_path):
    trainer = _trainer(tmp_path / "checkpoints")
    trainer.config.actor_rollout_ref.actor.checkpoint.save_contents = ["model", "extra"]

    with pytest.raises(ValueError, match="model.*optimizer.*extra"):
        trainer._validate_checkpoint_config()


def test_async_trainer_megatron_checkpoint_requires_optimizer_scheduler_state(tmp_path):
    trainer = _trainer(tmp_path / "checkpoints")
    actor_config = trainer.config.actor_rollout_ref.actor
    actor_config.strategy = "megatron"
    actor_config.optim = {"use_checkpoint_opt_param_scheduler": False}

    with pytest.raises(ValueError, match="use_checkpoint_opt_param_scheduler=True"):
        trainer._validate_checkpoint_config()

    actor_config.optim.use_checkpoint_opt_param_scheduler = True
    trainer._validate_checkpoint_config()


class _RecordingRemoteMethod:
    def __init__(self):
        self.calls = []

    def remote(self, *args, **kwargs):
        async def call():
            self.calls.append((args, kwargs))

        return call()


def test_async_trainer_prepare_publishes_restored_version_exactly_once():
    from laminar.train.trainer import AsyncRLTrainer, AsyncRLTrainerConfig

    trainer = object.__new__(AsyncRLTrainer)
    trainer.trainer_config = AsyncRLTrainerConfig()
    trainer.global_steps = 12
    trainer._prepared = False
    trainer._validate_checkpoint_config = lambda: None
    trainer._load_checkpoint = AsyncMock(return_value=True)
    trainer.publish_actor_weights_to_relay = AsyncMock(return_value={})
    trainer.experience_buffer = SimpleNamespace(start=_RecordingRemoteMethod())
    trainer.tracker = SimpleNamespace(
        set_next_step_to_log=_RecordingRemoteMethod(),
    )

    assert asyncio.run(trainer.prepare()) is None
    assert asyncio.run(trainer.prepare()) is None

    trainer._load_checkpoint.assert_awaited_once_with()
    trainer.publish_actor_weights_to_relay.assert_awaited_once_with(version=12)
    assert trainer.experience_buffer.start.calls == [((), {})]
    assert trainer.tracker.set_next_step_to_log.calls == [((12,), {})]
