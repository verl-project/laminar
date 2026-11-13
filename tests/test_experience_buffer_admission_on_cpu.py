# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
import torch

from fake_transfer_queue import install_fake_tq
from laminar.train.experience_buffer.server import ExperienceBufferServer


def _run(awaitable):
    return asyncio.run(awaitable)


def _success_tag(uid: str, **overrides) -> dict:
    tag = {
        "global_steps": 2,
        "min_global_steps": 2,
        "max_global_steps": 2,
        "status": "success",
        "prompt_len": 3,
        "response_len": 2,
        "seq_len": 5,
        "uid": uid,
    }
    tag.update(overrides)
    return tag


def _install_reward_store(monkeypatch, **handlers):
    async def get_many(*, keys, namespace, fields):  # noqa: ANN001
        assert namespace == "train"
        assert fields == ["rm_scores"]
        return {"rm_scores": torch.ones((len(keys), 1), dtype=torch.float32)}

    handlers.setdefault("get_many_handler", get_many)
    return install_fake_tq(monkeypatch, **handlers)


def test_success_waits_for_raw_uid_finished_marker(monkeypatch):
    storage = _install_reward_store(monkeypatch)
    buffer = ExperienceBufferServer()
    uid = "admission-a"
    key = f"{uid}_0_0"

    _run(buffer.add({key: _success_tag(uid)}))

    assert list(buffer._ready) == []
    assert storage.get_many_calls == []

    _run(buffer.add({uid: {"status": "finished"}}))

    assert list(buffer._ready) == [key]
    assert storage.get_many_calls == [{"keys": [key], "namespace": "train", "fields": ["rm_scores"]}]
    assert storage.release_calls == [([uid], "train")]

    batch = _run(buffer.sample(batch_size=1))
    assert batch.keys == [key]


def test_legacy_status_only_terminal_preserves_admission(monkeypatch):
    _install_reward_store(monkeypatch)
    buffer = ExperienceBufferServer()
    uid = "legacy-terminal"
    key = f"{uid}_0_0"

    _run(buffer.add({key: _success_tag(uid), uid: {"status": "finished"}}))

    assert _run(buffer.sample(batch_size=1)).keys == [key]
    assert _run(buffer.get_metrics())["lifecycle/activated_trajectory_count"] == 1


@pytest.mark.parametrize(
    "key",
    [
        "uid",
        "uid_0",
        "uid_with_underscore_0_0",
        "uid_session_0",
        "uid_0_output",
        "uid_-1_0",
        "uid_0_-1",
    ],
)
def test_invalid_trajectory_key_is_rejected_before_mutation(monkeypatch, key):
    storage = _install_reward_store(monkeypatch)
    buffer = ExperienceBufferServer()

    with pytest.raises(RuntimeError, match="invalid trajectory key"):
        _run(buffer.add({key: _success_tag("uid")}))

    assert buffer._tags == {}
    assert storage.release_calls == []


def test_trajectory_key_uid_must_match_original_tag(monkeypatch):
    _install_reward_store(monkeypatch)
    buffer = ExperienceBufferServer()

    with pytest.raises(RuntimeError, match="trajectory key uid conflicts with its tag"):
        _run(buffer.add({"right_0_0": _success_tag("wrong")}))

    assert buffer._tags == {}


@pytest.mark.parametrize("seq_len", [None, -1, 1.5, "5", True])
def test_invalid_attention_length_is_rejected_before_snapshot_mutation(monkeypatch, seq_len):
    storage = _install_reward_store(monkeypatch)
    buffer = ExperienceBufferServer()
    invalid_tag = _success_tag("invalid", seq_len=seq_len)
    if seq_len is None:
        invalid_tag.pop("seq_len")

    with pytest.raises(RuntimeError, match="non-negative integer seq_len"):
        _run(buffer.add({"valid_0_0": _success_tag("valid"), "invalid_0_0": invalid_tag}))

    assert buffer._tags == {}
    assert storage.get_many_calls == []
    assert storage.release_calls == []
    assert _run(buffer.get_metrics())["lifecycle/committed_attention_tokens"] == 0


def test_exact_duplicate_success_is_idempotent_but_changed_metadata_is_fatal(monkeypatch):
    _install_reward_store(monkeypatch)
    buffer = ExperienceBufferServer()
    uid = "duplicate"
    key = f"{uid}_0_0"
    tag = _success_tag(uid)

    _run(buffer.add({key: tag}))
    _run(buffer.add({key: dict(tag)}))
    with pytest.raises(RuntimeError, match="conflicting immutable terminal metadata"):
        _run(buffer.add({key: _success_tag(uid, prompt_len=4, seq_len=6)}))

    _run(buffer.add({uid: {"status": "finished"}}))
    assert _run(buffer.sample(batch_size=1)).keys == [key]


def test_all_failure_marker_has_no_trajectory_and_releases_only_the_marker(monkeypatch):
    async def unexpected_get_many(**_kwargs):
        raise AssertionError("an all-failure group must not read trajectory payloads")

    storage = _install_reward_store(monkeypatch, get_many_handler=unexpected_get_many)
    buffer = ExperienceBufferServer()
    uid = "all-failed"

    _run(buffer.add({uid: {"status": "failure"}}))

    assert list(buffer._ready) == []
    assert buffer._tags == {}
    assert storage.get_many_calls == []
    assert storage.release_calls == [([uid], "train")]
    assert _run(buffer.get_metrics())["lifecycle/activated_trajectory_count"] == 0


def test_failure_marker_rejects_a_group_with_successful_trajectories(monkeypatch):
    _install_reward_store(monkeypatch)
    buffer = ExperienceBufferServer()
    uid = "mixed-terminal"

    with pytest.raises(RuntimeError, match="failed prompt terminal cannot close successful trajectories"):
        _run(
            buffer.add(
                {
                    f"{uid}_0_0": _success_tag(uid),
                    uid: {"status": "failure"},
                }
            )
        )


def test_finished_marker_requires_at_least_one_successful_trajectory(monkeypatch):
    storage = _install_reward_store(monkeypatch)
    buffer = ExperienceBufferServer()

    with pytest.raises(RuntimeError, match="finished prompt terminal requires at least one successful trajectory"):
        _run(buffer.add({"empty": {"status": "finished"}}))

    assert storage.get_many_calls == []
    assert storage.release_calls == []


@pytest.mark.parametrize("status", ["finished", "failure"])
def test_prompt_terminal_key_must_be_the_raw_underscore_free_uid(monkeypatch, status):
    storage = _install_reward_store(monkeypatch)
    buffer = ExperienceBufferServer()
    terminal = {"status": status}

    with pytest.raises(RuntimeError, match="raw uid without underscores"):
        _run(buffer.add({"uid_0_0": terminal}))

    assert buffer._prompt_terminals == {}
    assert storage.release_calls == []
