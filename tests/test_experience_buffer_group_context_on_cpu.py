# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
import torch
from tensordict import TensorDict

from fake_transfer_queue import install_fake_tq
from laminar.train.experience_buffer.config import ExperienceBufferConfig
from laminar.train.experience_buffer.server import ExperienceBufferServer


def _run(awaitable):
    return asyncio.run(awaitable)


def _success_tag(uid: str, *, version: int = 2) -> dict:
    return {
        "status": "success",
        "uid": uid,
        "prompt_len": 3,
        "response_len": 2,
        "seq_len": 5,
        "global_steps": version,
        "min_global_steps": version,
        "max_global_steps": version,
    }


def _install_reward_store(monkeypatch, rewards: dict[str, float]):
    async def get_many(*, keys, namespace, fields):  # noqa: ANN001
        assert namespace == "train"
        assert fields == ["rm_scores"]
        rows = [torch.tensor([0.0, rewards[key]]) for key in keys]
        return TensorDict(
            {"rm_scores": torch.nested.as_nested_tensor(rows, layout=torch.jagged)},
            batch_size=[len(keys)],
        )

    return install_fake_tq(monkeypatch, get_many_handler=get_many)


def _expected_context(uid: str) -> dict[str, list[dict]]:
    return {
        uid: [
            {"key": f"{uid}_0_1", "session_id": 0, "output_index": 1, "reward_score": 1.0},
            {"key": f"{uid}_1_0", "session_id": 1, "output_index": 0, "reward_score": 3.0},
        ]
    }


def test_uid_terminal_gates_sampling_and_builds_context_from_final_rm_scores(monkeypatch):
    uid = "prompt-a"
    rewards = {
        f"{uid}_0_1": 1.0,
        f"{uid}_1_0": 3.0,
    }
    storage = _install_reward_store(monkeypatch, rewards)
    buffer = ExperienceBufferServer(ExperienceBufferConfig(sampler="fifo"))
    trajectories = {
        f"{uid}_0_0": _success_tag(uid, version=2),
        f"{uid}_0_1": _success_tag(uid, version=5),
        f"{uid}_1_0": _success_tag(uid, version=3),
    }

    _run(buffer.add(trajectories))
    assert list(buffer._ready) == []

    _run(buffer.add({uid: {"status": "finished"}}))
    first = _run(buffer.sample(batch_size=1))
    second = _run(buffer.sample(batch_size=2))

    assert first.keys == [f"{uid}_0_0"]
    assert second.keys == [f"{uid}_0_1", f"{uid}_1_0"]
    assert first.group_context == _expected_context(uid)
    assert second.group_context == _expected_context(uid)
    assert storage.get_many_calls == [
        {
            "keys": [f"{uid}_0_1", f"{uid}_1_0"],
            "namespace": "train",
            "fields": ["rm_scores"],
        }
    ]
    assert storage.release_calls == [([uid], "train")]


def test_version_lifo_still_orders_activated_trajectories(monkeypatch):
    uid = "prompt-b"
    _install_reward_store(monkeypatch, {f"{uid}_0_1": 1.0, f"{uid}_1_0": 3.0})
    buffer = ExperienceBufferServer(ExperienceBufferConfig(sampler="version_lifo"))
    _run(
        buffer.add(
            {
                f"{uid}_0_0": _success_tag(uid, version=2),
                f"{uid}_0_1": _success_tag(uid, version=5),
                f"{uid}_1_0": _success_tag(uid, version=3),
                uid: {"status": "finished"},
            }
        )
    )

    assert _run(buffer.sample(batch_size=2)).keys == [f"{uid}_0_1", f"{uid}_1_0"]
    assert _run(buffer.sample(batch_size=1)).keys == [f"{uid}_0_0"]


def test_group_context_survives_split_batch_release(monkeypatch):
    uid = "prompt-c"
    _install_reward_store(monkeypatch, {f"{uid}_0_1": 1.0, f"{uid}_1_0": 3.0})
    buffer = ExperienceBufferServer(ExperienceBufferConfig(sampler="version_lifo"))
    _run(
        buffer.add(
            {
                f"{uid}_0_0": _success_tag(uid, version=2),
                f"{uid}_0_1": _success_tag(uid, version=5),
                f"{uid}_1_0": _success_tag(uid, version=3),
                uid: {"status": "finished"},
            }
        )
    )

    first = _run(buffer.sample(batch_size=1))
    assert first.keys == [f"{uid}_0_1"]
    _run(buffer.remove(first.keys))

    siblings = _run(buffer.sample(batch_size=2))
    assert siblings.group_context == _expected_context(uid)
    _run(buffer.remove(siblings.keys))
    assert uid not in buffer._prompt_group_context


def test_partial_success_uses_only_actual_sessions_without_zero_fill(monkeypatch):
    uid = "prompt-d"
    _install_reward_store(monkeypatch, {f"{uid}_0_0": 1.0, f"{uid}_2_0": 5.0})
    buffer = ExperienceBufferServer(ExperienceBufferConfig(sampler="fifo"))
    _run(
        buffer.add(
            {
                f"{uid}_0_0": _success_tag(uid),
                f"{uid}_2_0": _success_tag(uid),
                # Session 1 failed and therefore has no trajectory record.
                uid: {"status": "finished"},
            }
        )
    )

    batch = _run(buffer.sample(batch_size=2))
    assert batch.keys == [f"{uid}_0_0", f"{uid}_2_0"]
    assert batch.group_context == {
        uid: [
            {"key": f"{uid}_0_0", "session_id": 0, "output_index": 0, "reward_score": 1.0},
            {"key": f"{uid}_2_0", "session_id": 2, "output_index": 0, "reward_score": 5.0},
        ]
    }


def test_all_failure_terminal_releases_marker_without_trainable_rows(monkeypatch):
    async def unexpected_get_many(**_kwargs):
        raise AssertionError("all-failure prompt must not read trajectory payloads")

    storage = install_fake_tq(monkeypatch, get_many_handler=unexpected_get_many)
    buffer = ExperienceBufferServer()
    uid = "prompt-e"

    _run(buffer.add({uid: {"status": "failure"}}))

    assert list(buffer._ready) == []
    assert storage.get_many_calls == []
    assert storage.release_calls == [([uid], "train")]


def test_failure_terminal_rejects_successful_trajectory_records(monkeypatch):
    _install_reward_store(monkeypatch, {})
    buffer = ExperienceBufferServer()
    uid = "prompt-f"

    with pytest.raises(RuntimeError, match="failed prompt terminal cannot close successful trajectories"):
        _run(buffer.add({f"{uid}_0_0": _success_tag(uid), uid: {"status": "failure"}}))


def test_reward_read_does_not_hold_the_buffer_condition(monkeypatch):
    async def scenario():
        slow_uid = "prompt-slow"
        ready_uid = "prompt-ready"
        read_started = asyncio.Event()
        allow_read = asyncio.Event()

        async def get_many(*, keys, namespace, fields):  # noqa: ANN001
            assert namespace == "train"
            assert fields == ["rm_scores"]
            if keys == [f"{slow_uid}_0_0"]:
                read_started.set()
                await allow_read.wait()
            return {"rm_scores": torch.ones((len(keys), 1), dtype=torch.float32)}

        install_fake_tq(monkeypatch, get_many_handler=get_many)
        buffer = ExperienceBufferServer(ExperienceBufferConfig(sampler="fifo"))
        await buffer.add({f"{ready_uid}_0_0": _success_tag(ready_uid), ready_uid: {"status": "finished"}})

        slow_add = asyncio.create_task(
            buffer.add({f"{slow_uid}_0_0": _success_tag(slow_uid), slow_uid: {"status": "finished"}})
        )
        await asyncio.wait_for(read_started.wait(), timeout=1.0)
        ready_batch = await asyncio.wait_for(buffer.sample(batch_size=1), timeout=1.0)
        assert ready_batch.keys == [f"{ready_uid}_0_0"]

        allow_read.set()
        await slow_add

    _run(scenario())


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_group_reward_is_rejected_before_activation(monkeypatch, reward):
    uid = "prompt-nonfinite"
    storage = _install_reward_store(monkeypatch, {f"{uid}_0_0": reward})
    buffer = ExperienceBufferServer()

    with pytest.raises(RuntimeError, match="non-finite reward"):
        _run(buffer.add({f"{uid}_0_0": _success_tag(uid), uid: {"status": "finished"}}))

    assert list(buffer._ready) == []
    assert uid not in buffer._prompt_group_context
    assert uid not in buffer._activating_prompt_groups
    assert storage.release_calls == []


def test_failed_reward_read_clears_activation_for_an_explicit_retry(monkeypatch):
    uid = "prompt-retry"
    attempts = 0

    async def get_many(*, keys, namespace, fields):  # noqa: ANN001
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary read failure")
        assert keys == [f"{uid}_0_0"]
        assert namespace == "train"
        assert fields == ["rm_scores"]
        return {"rm_scores": torch.tensor([[2.0]])}

    storage = install_fake_tq(monkeypatch, get_many_handler=get_many)
    buffer = ExperienceBufferServer()
    items = {f"{uid}_0_0": _success_tag(uid), uid: {"status": "finished"}}

    with pytest.raises(RuntimeError, match="temporary read failure"):
        _run(buffer.add(items))
    metrics = _run(buffer.get_metrics())
    assert metrics["lifecycle/committed_trajectory_count"] == 1
    assert metrics["lifecycle/committed_attention_tokens"] == 5
    assert metrics["lifecycle/activated_trajectory_count"] == 0
    _run(buffer.add(items))
    assert _run(buffer.get_metrics())["lifecycle/committed_attention_tokens"] == 5

    batch = _run(buffer.sample(batch_size=1))
    assert batch.keys == [f"{uid}_0_0"]
    assert batch.group_context[uid][0]["reward_score"] == 2.0
    assert len(storage.get_many_calls) == 2
