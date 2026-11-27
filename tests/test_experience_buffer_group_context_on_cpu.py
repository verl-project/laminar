# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
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


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("rewards", [[0.0] * 8, [1.0] * 8, [2.5] * 8, [0.0]])
def test_zero_advantage_groups_are_filtered_only_when_enabled(monkeypatch, enabled, rewards):
    uid = "homogeneous"
    keys = [f"{uid}_{session_id}_0" for session_id in range(len(rewards))]
    storage = _install_reward_store(monkeypatch, dict(zip(keys, rewards, strict=True)))
    items = {**{key: _success_tag(uid) for key in keys}, uid: {"status": "finished"}}
    storage.metadata_by_namespace["train"] = dict(items)
    buffer = ExperienceBufferServer(ExperienceBufferConfig(filter_groups_enable=enabled))

    _run(buffer._add())

    metrics = _run(buffer.get_metrics())
    assert metrics["lifecycle/committed_trajectory_count"] == len(keys)
    assert metrics["conservation/activation_residual"] == 0
    if enabled:
        assert storage.release_calls == [([*keys, uid], "train")]
        assert not storage.metadata_by_namespace["train"]
        assert not buffer._ready and not buffer._tags and not buffer._prompt_group_keys
        assert not buffer._prompt_group_context and not buffer._prompt_terminals
        assert not buffer._activated_prompt_groups and not buffer._activating_prompt_groups
        assert metrics["lifecycle/activated_trajectory_count"] == 0
        assert metrics["filter_groups/filtered_group_count"] == 1
        assert metrics["filter_groups/filtered_trajectory_count"] == len(keys)
        assert metrics["filter_groups/zero_reward_group_count"] == int(rewards[0] == 0.0)
        assert metrics["filter_groups/one_reward_group_count"] == int(rewards[0] == 1.0)
        assert metrics["drop/trajectory_count"] == len(keys)
        _run(buffer._add())
        assert _run(buffer.get_metrics())["filter_groups/filtered_group_count"] == 1
    else:
        assert _run(buffer.sample(batch_size=len(keys))).keys == keys
        assert storage.release_calls == [([uid], "train")]
        assert metrics["filter_groups/filtered_group_count"] == 0


@pytest.mark.parametrize("rewards", [[0.0, 1.0], [0.0, 1e-8], [1.0], [-1.0]])
def test_filtering_preserves_groups_with_nonzero_session_advantages(monkeypatch, rewards):
    uid = "signal"
    keys = [f"{uid}_{session_id}_0" for session_id in range(len(rewards))]
    storage = _install_reward_store(monkeypatch, dict(zip(keys, rewards, strict=True)))
    buffer = ExperienceBufferServer(ExperienceBufferConfig(filter_groups_enable=True))

    _run(buffer.add({**{key: _success_tag(uid) for key in keys}, uid: {"status": "finished"}}))

    assert _run(buffer.sample(batch_size=len(keys))).keys == keys
    assert storage.release_calls == [([uid], "train")]
    assert _run(buffer.get_metrics())["filter_groups/filtered_group_count"] == 0


def test_filtering_waits_for_terminal_and_clears_all_segments_using_final_rewards(monkeypatch):
    uid = "final-rewards"
    keys = [f"{uid}_0_0", f"{uid}_0_1", f"{uid}_2_0"]
    storage = _install_reward_store(monkeypatch, {keys[0]: 0.0, keys[1]: 1.0, keys[2]: 1.0})
    buffer = ExperienceBufferServer(ExperienceBufferConfig(filter_groups_enable=True))

    _run(buffer.add({key: _success_tag(uid) for key in keys}))
    assert storage.get_many_calls == [] and storage.release_calls == []
    assert not buffer._ready
    _run(buffer.add({uid: {"status": "finished"}}))

    assert storage.get_many_calls[0]["keys"] == keys[1:]
    assert storage.release_calls == [([*keys, uid], "train")]
    assert not buffer._tags and not buffer._ready
    assert _run(buffer.get_metrics())["filter_groups/filtered_trajectory_count"] == 3


def test_filtered_groups_do_not_complete_a_batch_until_signal_groups_fill_it(monkeypatch):
    async def scenario():
        bad_uid, first_uid, second_uid = "bad", "signal-first", "signal-second"
        rewards = {
            f"{bad_uid}_0_0": 0.0,
            f"{bad_uid}_1_0": 0.0,
            f"{first_uid}_0_0": 0.0,
            f"{first_uid}_1_0": 1.0,
            f"{second_uid}_0_0": 0.0,
            f"{second_uid}_1_0": 1.0,
        }
        storage = _install_reward_store(monkeypatch, rewards)
        buffer = ExperienceBufferServer(ExperienceBufferConfig(max_size=4, filter_groups_enable=True))
        sample_started = asyncio.Event()

        async def take_batch():
            sample_started.set()
            return await buffer.sample(batch_size=4)

        sample = asyncio.create_task(take_batch())
        await sample_started.wait()

        async def finish_group(uid):
            keys = [f"{uid}_0_0", f"{uid}_1_0"]
            await buffer.add({**{key: _success_tag(uid) for key in keys}, uid: {"status": "finished"}})
            return keys

        await finish_group(bad_uid)
        assert not sample.done() and len(buffer) == 0
        first_keys = await finish_group(first_uid)
        assert not sample.done() and len(buffer) == 2
        second_keys = await finish_group(second_uid)
        batch = await asyncio.wait_for(sample, timeout=1.0)

        assert batch.keys == first_keys + second_keys
        assert set(batch.group_context) == {first_uid, second_uid}
        assert all(key.split("_")[0] != bad_uid for key in batch.trajectory_metadata)
        assert storage.release_calls[0] == ([f"{bad_uid}_0_0", f"{bad_uid}_1_0", bad_uid], "train")
        metrics = await buffer.get_metrics()
        assert metrics["lifecycle/activated_trajectory_count"] == 4
        assert metrics["lifecycle/sampled_trajectory_count"] == 4
        assert metrics["filter_groups/filtered_trajectory_count"] == 2
        assert metrics["eviction/after_commit_trajectory_count"] == 0
        assert metrics["conservation/activation_residual"] == 0

    _run(asyncio.wait_for(scenario(), timeout=3.0))


def test_filtered_payload_clear_failure_preserves_metadata_and_fails_the_add(monkeypatch):
    uid, other_uid = "filter-failure", "filter-pending"
    keys = [f"{uid}_0_0", f"{uid}_1_0"]
    other_keys = [f"{other_uid}_0_0", f"{other_uid}_1_0"]
    storage = _install_reward_store(monkeypatch, dict.fromkeys(keys + other_keys, 0.0))

    def fail_clear(_keys, *, namespace):
        assert namespace == "train"
        raise RuntimeError("filtered payload clear failed")

    storage._release_handler = fail_clear
    buffer = ExperienceBufferServer(ExperienceBufferConfig(filter_groups_enable=True))
    with pytest.raises(RuntimeError, match="filtered payload clear failed"):
        _run(
            buffer.add(
                {
                    **{key: _success_tag(uid) for key in keys},
                    **{key: _success_tag(other_uid) for key in other_keys},
                    uid: {"status": "finished"},
                    other_uid: {"status": "finished"},
                }
            )
        )

    assert set(buffer._tags) == set(keys + other_keys)
    assert set(buffer._prompt_terminals) == {uid, other_uid}
    assert not buffer._ready and not buffer._activating_prompt_groups
    metrics = _run(buffer.get_metrics())
    assert metrics["sink/trajectory_clear_failure_count"] == 1
    assert metrics["filter_groups/filtered_group_count"] == 0
    assert metrics["drop/trajectory_count"] == 0


def test_concurrent_metadata_poll_cannot_recommit_a_filtered_group(monkeypatch):
    async def scenario():
        uid = "filter-race"
        keys = [f"{uid}_0_0", f"{uid}_1_0"]
        items = {**{key: _success_tag(uid) for key in keys}, uid: {"status": "finished"}}
        reward_started, allow_reward = asyncio.Event(), asyncio.Event()
        read_started, allow_read = asyncio.Event(), asyncio.Event()

        async def get_rewards(*, keys, namespace, fields):
            assert namespace == "train" and fields == ["rm_scores"]
            reward_started.set()
            await allow_reward.wait()
            return {"rm_scores": torch.zeros((len(keys), 1))}

        async def list_metadata(*, namespace):
            assert namespace is None
            snapshot = {ns: dict(entries) for ns, entries in storage.metadata_by_namespace.items()}
            read_started.set()
            await allow_read.wait()
            return snapshot

        storage = install_fake_tq(
            monkeypatch,
            get_many_handler=get_rewards,
            list_metadata_handler=list_metadata,
            metadata_by_namespace={"train": items},
        )
        buffer = ExperienceBufferServer(ExperienceBufferConfig(filter_groups_enable=True))
        activation = asyncio.create_task(buffer.add(items))
        await reward_started.wait()
        poll = asyncio.create_task(buffer._add())
        await read_started.wait()
        allow_reward.set()
        allow_read.set()
        await asyncio.gather(activation, poll)
        await buffer._add()

        assert storage.release_calls == [([*keys, uid], "train")]
        assert not storage.metadata_by_namespace["train"]
        assert not buffer._ready and not buffer._tags and not buffer._prompt_group_keys
        metrics = await buffer.get_metrics()
        assert metrics["lifecycle/committed_trajectory_count"] == 2
        assert metrics["filter_groups/filtered_group_count"] == 1
        assert metrics["conservation/activation_residual"] == 0

    _run(asyncio.wait_for(scenario(), timeout=3.0))


@pytest.mark.parametrize("adv_estimator", ["grpo", "rloo"])
@pytest.mark.parametrize("filter_override", ["algorithm.filter_groups=True", "algorithm.filter_groups.enable=True"])
def test_algorithm_filter_switch_removes_zero_signal_groups_before_advantage_computation(
    monkeypatch, adv_estimator, filter_override
):
    from laminar.train.trainer import AsyncRLTrainer

    async def scenario():
        config_dir = Path(__file__).resolve().parents[1] / "src/laminar/train/config"
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            overrides = [f"algorithm.adv_estimator={adv_estimator}", filter_override]
            if filter_override.endswith(".enable=True"):
                overrides.append("algorithm.filter_groups.metric=reward")
            config = compose(
                config_name="async_rl",
                overrides=overrides,
            )
        rewards = {
            f"{uid}_{session_id}_0": reward
            for uid, group_rewards in {
                "all-zero": [0.0] * 8,
                "all-one": [1.0] * 8,
                "mixed": [0.0, 1.0] * 4,
            }.items()
            for session_id, reward in enumerate(group_rewards)
        }
        _install_reward_store(monkeypatch, rewards)
        buffer = ExperienceBufferServer(ExperienceBufferConfig.from_config(config))
        await buffer.add(
            {
                **{key: _success_tag(key.split("_")[0]) for key in rewards},
                **{uid: {"status": "finished"} for uid in ("all-zero", "all-one", "mixed")},
            }
        )
        batch = await buffer.sample(batch_size=8)
        assert batch.keys == [f"mixed_{session_id}_0" for session_id in range(8)]
        trainer = object.__new__(AsyncRLTrainer)
        trainer.config = config
        td = TensorDict(
            {
                "rm_scores": torch.tensor([[rewards[key]] for key in batch.keys]),
                "response_mask": torch.ones((8, 1)),
            },
            batch_size=[8],
        )
        metrics = {}
        trainer._compute_advantages(batch, td, metrics)

        assert (td["advantages"][::2] < 0).all()
        assert (td["advantages"][1::2] > 0).all()
        assert metrics["advantage/source_prompt_group_count"] == 1
        buffer_metrics = await buffer.get_metrics()
        assert buffer_metrics["filter_groups/filtered_group_count"] == 2
        assert buffer_metrics["filter_groups/filtered_trajectory_count"] == 16

    _run(scenario())
