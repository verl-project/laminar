# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time

import pytest
import torch

from fake_transfer_queue import install_fake_tq
from laminar.train.experience_buffer.config import ExperienceBufferConfig
from laminar.train.experience_buffer.server import ExperienceBufferServer
from laminar.train.experience_buffer.tags import _tag_weight_version


def _run(awaitable):
    return asyncio.run(awaitable)


def _key(uid: str, session_id: int = 0, output_index: int = 0) -> str:
    return f"{uid}_{session_id}_{output_index}"


def _success_tag(
    uid: str,
    *,
    attention_tokens: int = 5,
    response_tokens: int = 2,
    version: int | None = 2,
    **extra,
) -> dict:
    tag = {
        "status": "success",
        "prompt_len": attention_tokens - response_tokens,
        "response_len": response_tokens,
        "seq_len": attention_tokens,
        "uid": uid,
    }
    if version is not None:
        tag.update(
            global_steps=version,
            min_global_steps=version,
            max_global_steps=version,
        )
    tag.update(extra)
    return tag


async def _get_rm_scores(*, keys, namespace, fields):  # noqa: ANN001
    assert namespace == "train"
    assert fields == ["rm_scores"]
    return {"rm_scores": torch.ones((len(keys), 1), dtype=torch.float32)}


def _install_store(monkeypatch, **handlers):
    handlers.setdefault("get_many_handler", _get_rm_scores)
    return install_fake_tq(monkeypatch, **handlers)


@pytest.fixture(autouse=True)
def _fake_store(monkeypatch):
    return _install_store(monkeypatch)


def _add_finished(buffer: ExperienceBufferServer, uid: str, tags: list[dict]) -> list[str]:
    keys = [_key(uid, session_id) for session_id in range(len(tags))]
    _run(
        buffer.add(
            {
                **dict(zip(keys, tags, strict=True)),
                uid: {"status": "finished" if tags else "failure"},
            }
        )
    )
    return keys


def test_owner_metrics_are_present_and_zero_before_activity():
    before = time.time()
    metrics = _run(ExperienceBufferServer().get_metrics())

    assert metrics
    assert before <= metrics.pop("timestamp/snapshot_s") <= time.time()
    assert set(metrics.values()) == {0}
    assert {
        "lifecycle/committed_trajectory_count",
        "lifecycle/committed_attention_tokens",
        "lifecycle/activated_trajectory_count",
        "inventory/ready_trajectory_count",
        "eviction/after_commit_trajectory_count",
        "drop/trajectory_count",
        "sink/trajectory_clear_failure_count",
        "sink/prompt_terminal_clear_failure_count",
        "conservation/activation_residual",
    }.issubset(metrics)


def test_committed_metrics_count_ingestion_before_activation_and_survive_consumption():
    buffer = ExperienceBufferServer(ExperienceBufferConfig(max_size=2, sampler="fifo"))
    uid = "committed"
    keys = [_key(uid, 0, 0), _key(uid, 0, 1), _key(uid, 1, 0)]
    items = dict(zip(keys, [_success_tag(uid, attention_tokens=n) for n in (7, 11, 13)], strict=True))

    _run(buffer.add(items))
    staged = _run(buffer.get_metrics())
    assert staged["lifecycle/committed_trajectory_count"] == 3
    assert staged["lifecycle/committed_attention_tokens"] == 31
    assert staged["lifecycle/activated_trajectory_count"] == 0
    assert staged["inventory/ready_trajectory_count"] == 0

    _run(buffer.add(items))
    duplicate = _run(buffer.get_metrics())
    assert duplicate["lifecycle/committed_trajectory_count"] == 3
    assert duplicate["lifecycle/committed_attention_tokens"] == 31

    _run(buffer.add({uid: {"status": "finished"}}))
    assert _run(buffer.sample(batch_size=2)).keys == keys[1:]
    _run(buffer.remove(keys[1:]))
    consumed = _run(buffer.get_metrics())
    assert consumed["lifecycle/committed_trajectory_count"] == 3
    assert consumed["lifecycle/committed_attention_tokens"] == 31
    assert consumed["lifecycle/activated_trajectory_count"] == 3
    assert consumed["drop/trajectory_count"] == 1
    assert consumed["lifecycle/released_trajectory_count"] == 2


def test_owner_metrics_conserve_activate_sample_and_release():
    buffer = ExperienceBufferServer(ExperienceBufferConfig(sampler="fifo"))
    uid = "owner-flow"
    key = _add_finished(buffer, uid, [_success_tag(uid)])[0]

    activated = _run(buffer.get_metrics())
    assert activated["lifecycle/activated_trajectory_count"] == 1
    assert activated["inventory/ready_trajectory_count"] == 1
    assert activated["conservation/activation_residual"] == 0

    assert _run(buffer.sample(batch_size=1)).keys == [key]
    sampled = _run(buffer.get_metrics())
    assert sampled["lifecycle/sampled_trajectory_count"] == 1
    assert sampled["inventory/ready_trajectory_count"] == 0
    assert sampled["conservation/activation_residual"] == 0

    _run(buffer.remove([key]))
    released = _run(buffer.get_metrics())
    assert released["lifecycle/released_trajectory_count"] == 1
    assert released["conservation/activation_residual"] == 0


def test_sample_rejects_batch_larger_than_buffer_capacity():
    buffer = ExperienceBufferServer(ExperienceBufferConfig(max_size=1))

    with pytest.raises(ValueError, match="batch_size 2 exceeds buffer max_size 1"):
        _run(buffer.sample(batch_size=2))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("trajectory_weight_version", -1),
        ("max_global_steps", "invalid"),
        ("global_steps", 1.5),
    ],
)
def test_experience_buffer_rejects_invalid_weight_version_fact(field, value):
    with pytest.raises(ValueError, match=field):
        _tag_weight_version({field: value})


def test_conflicting_trajectory_metadata_fails_closed_for_the_same_store_key():
    buffer = ExperienceBufferServer(ExperienceBufferConfig(sampler="fifo"))
    uid = "conflict"
    key = _key(uid)
    _run(buffer.add({key: _success_tag(uid)}))

    with pytest.raises(RuntimeError, match="conflicting immutable terminal metadata"):
        _run(buffer.add({key: _success_tag(uid, attention_tokens=6)}))

    assert list(buffer._ready) == []
    _run(buffer.add({uid: {"status": "finished"}}))
    assert _run(buffer.sample(batch_size=1)).keys == [key]


def test_remove_clears_trajectory_and_group_state(monkeypatch):
    storage = _install_store(monkeypatch)
    buffer = ExperienceBufferServer(ExperienceBufferConfig(sampler="fifo"))
    uid = "remove-state"
    key = _add_finished(buffer, uid, [_success_tag(uid)])[0]

    batch = _run(buffer.sample(batch_size=1))
    assert batch.keys == [key]
    _run(buffer.remove(batch.keys))

    assert storage.release_calls == [([uid], "train"), ([key], "train")]
    assert key not in buffer._tags
    assert uid not in buffer._prompt_group_context


def test_fifo_samples_oldest_ready_key():
    buffer = ExperienceBufferServer(ExperienceBufferConfig(sampler="fifo"))
    uid = "fifo"
    keys = _add_finished(buffer, uid, [_success_tag(uid), _success_tag(uid)])

    batch = _run(buffer.sample(batch_size=1))
    assert batch.keys == [keys[0]]


def test_lifo_samples_newest_ready_key():
    buffer = ExperienceBufferServer(ExperienceBufferConfig(sampler="lifo"))
    uid = "lifo"
    keys = _add_finished(buffer, uid, [_success_tag(uid), _success_tag(uid)])

    batch = _run(buffer.sample(batch_size=1))
    assert batch.keys == [keys[1]]


def test_version_lifo_samples_highest_version_then_newest_ready_key():
    buffer = ExperienceBufferServer(ExperienceBufferConfig(sampler="version_lifo"))
    uid = "version-order"
    keys = _add_finished(
        buffer,
        uid,
        [
            _success_tag(uid, version=0),
            _success_tag(uid, version=8),
            _success_tag(uid, version=None),
            _success_tag(uid, version=0),
            _success_tag(uid, version=8),
        ],
    )

    first = _run(buffer.sample(batch_size=3))
    second = _run(buffer.sample(batch_size=2))
    assert first.keys == [keys[4], keys[1], keys[3]]
    assert second.keys == [keys[0], keys[2]]


def test_duplicate_success_does_not_requeue_the_trajectory():
    buffer = ExperienceBufferServer(ExperienceBufferConfig(sampler="fifo"))
    uid = "duplicate-order"
    key = _key(uid)
    tag = _success_tag(uid)

    _run(buffer.add({key: tag}))
    _run(buffer.add({key: dict(tag)}))
    _run(buffer.add({uid: {"status": "finished"}}))
    first = _run(buffer.sample(batch_size=1))
    _run(buffer.add({key: dict(tag)}))

    assert first.keys == [key]
    assert list(buffer._ready) == []


def test_eviction_releases_the_oldest_trajectory_after_activation(monkeypatch):
    storage = _install_store(monkeypatch)
    buffer = ExperienceBufferServer(ExperienceBufferConfig(max_size=1, sampler="fifo"))
    old_uid = "evict-old"
    new_uid = "evict-new"
    old_key = _add_finished(buffer, old_uid, [_success_tag(old_uid)])[0]
    new_key = _add_finished(buffer, new_uid, [_success_tag(new_uid)])[0]

    batch = _run(buffer.sample(batch_size=1))

    assert storage.release_calls == [
        ([old_uid], "train"),
        ([new_uid], "train"),
        ([old_key], "train"),
    ]
    assert batch.keys == [new_key]
    metrics = _run(buffer.get_metrics())
    assert metrics["lifecycle/activated_trajectory_count"] == 2
    assert metrics["eviction/after_commit_trajectory_count"] == 1
    assert metrics["drop/trajectory_count"] == 1
    assert metrics["conservation/activation_residual"] == 0


def test_eviction_clears_old_keys_in_one_transfer_queue_call(monkeypatch):
    storage = _install_store(monkeypatch)
    buffer = ExperienceBufferServer(ExperienceBufferConfig(max_size=2, sampler="fifo"))
    uid = "bulk-evict"
    keys = _add_finished(buffer, uid, [_success_tag(uid) for _ in range(5)])

    assert storage.release_calls == [([uid], "train"), (keys[:3], "train")]


def test_remove_clears_transfer_queue_before_dropping_metadata(monkeypatch):
    order: list[str] = []

    def fake_release(keys, *, namespace):  # noqa: ANN001
        assert namespace == "train"
        order.append(f"release:{','.join(keys)}")

    storage = _install_store(monkeypatch, release_handler=fake_release)
    buffer = ExperienceBufferServer(ExperienceBufferConfig(max_size=8, sampler="fifo"))
    uid = "remove-order"
    keys = _add_finished(buffer, uid, [_success_tag(uid), _success_tag(uid)])
    batch = _run(buffer.sample(batch_size=2))
    order.append("sampled")

    _run(buffer.remove(batch.keys))
    order.append("removed")

    assert storage.release_calls == [([uid], "train"), (keys, "train")]
    assert order == [f"release:{uid}", "sampled", f"release:{','.join(keys)}", "removed"]
    assert len(buffer) == 0
    assert all(key not in buffer._tags for key in keys)


def test_remove_release_failure_fails_fast_and_preserves_metadata(monkeypatch):
    fail_key: str | None = None

    def fail_trajectory_release(keys, *, namespace):  # noqa: ANN001
        assert namespace == "train"
        if keys == [fail_key]:
            raise RuntimeError("release failed")

    _install_store(monkeypatch, release_handler=fail_trajectory_release)
    buffer = ExperienceBufferServer(ExperienceBufferConfig(max_size=8, sampler="fifo"))
    uid = "remove-failure"
    fail_key = _add_finished(buffer, uid, [_success_tag(uid)])[0]
    assert _run(buffer.sample(batch_size=1)).keys == [fail_key]

    with pytest.raises(RuntimeError, match="release failed"):
        _run(buffer.remove([fail_key]))

    assert fail_key in buffer._tags
    metrics = _run(buffer.get_metrics())
    assert metrics["sink/trajectory_clear_failure_count"] == 1
    assert metrics["lifecycle/released_trajectory_count"] == 0
    assert metrics["conservation/activation_residual"] == 0


@pytest.mark.parametrize("sampler", ["fifo", "lifo", "version_lifo"])
def test_inflight_metadata_snapshot_cannot_recommit_a_released_trajectory(monkeypatch, sampler):
    async def scenario():
        read_started = asyncio.Event()
        allow_read = asyncio.Event()
        uid = "inflight-release"
        key = _key(uid)

        async def list_metadata(*, namespace):  # noqa: ANN001
            assert namespace is None
            snapshot = {ns: dict(entries) for ns, entries in storage.metadata_by_namespace.items()}
            read_started.set()
            await allow_read.wait()
            return snapshot

        storage = _install_store(
            monkeypatch,
            list_metadata_handler=list_metadata,
            metadata_by_namespace={"train": {key: _success_tag(uid), uid: {"status": "finished"}}},
        )
        buffer = ExperienceBufferServer(ExperienceBufferConfig(sampler=sampler))
        await buffer.add(dict(storage.metadata_by_namespace["train"]))
        batch = await buffer.sample(batch_size=1)
        poll = asyncio.create_task(buffer._add())
        await read_started.wait()
        remove = asyncio.create_task(buffer.remove(batch.keys))
        await asyncio.sleep(0)
        allow_read.set()
        await asyncio.gather(poll, remove)

        metrics = await buffer.get_metrics()
        assert metrics["lifecycle/committed_trajectory_count"] == 1
        assert metrics["lifecycle/released_trajectory_count"] == 1
        assert metrics["conservation/activation_residual"] == 0
        assert not storage.metadata_by_namespace["train"]
        assert not buffer._tags and not buffer._prompt_group_keys and not buffer._ready

    _run(asyncio.wait_for(scenario(), timeout=3.0))


@pytest.mark.parametrize("sampler", ["fifo", "lifo", "version_lifo"])
@pytest.mark.parametrize("schedule", ["list_before_activation", "list_queued_after_activation"])
def test_inflight_metadata_snapshot_cannot_recommit_a_capacity_eviction(monkeypatch, sampler, schedule):
    async def scenario():
        reward_started = asyncio.Event()
        allow_reward = asyncio.Event()
        read_started = asyncio.Event()
        allow_read = asyncio.Event()
        uid = "inflight-eviction"
        keys = [_key(uid, session_id) for session_id in range(5)]
        items = {**{key: _success_tag(uid) for key in keys}, uid: {"status": "finished"}}

        async def get_rewards(**kwargs):
            reward_started.set()
            await allow_reward.wait()
            return await _get_rm_scores(**kwargs)

        async def list_metadata(*, namespace):  # noqa: ANN001
            assert namespace is None
            snapshot = {ns: dict(entries) for ns, entries in storage.metadata_by_namespace.items()}
            read_started.set()
            await allow_read.wait()
            return snapshot

        storage = _install_store(
            monkeypatch,
            get_many_handler=get_rewards,
            list_metadata_handler=list_metadata,
            metadata_by_namespace={"train": items},
        )
        buffer = ExperienceBufferServer(ExperienceBufferConfig(max_size=4, sampler=sampler))
        activation = asyncio.create_task(buffer.add(items))
        await reward_started.wait()
        if schedule == "list_before_activation":
            poll = asyncio.create_task(buffer._add())
            await read_started.wait()
            allow_reward.set()
            await asyncio.sleep(0)
        else:
            # Queue activation first, then a poll, without changing metadata.
            async with buffer._condition:
                allow_reward.set()
                await asyncio.sleep(0)
                poll = asyncio.create_task(buffer._add())
                await asyncio.sleep(0)
            await read_started.wait()
        allow_read.set()
        await asyncio.gather(activation, poll)

        metrics = await buffer.get_metrics()
        assert metrics["lifecycle/committed_trajectory_count"] == 5
        assert metrics["lifecycle/activated_trajectory_count"] == 5
        assert metrics["drop/trajectory_count"] == 1
        assert metrics["conservation/activation_residual"] == 0
        assert list(buffer._ready) == keys[1:]
        assert set(buffer._tags) == set(storage.metadata_by_namespace["train"]) == set(keys[1:])
        await buffer.remove((await buffer.sample(batch_size=4)).keys)
        assert not buffer._tags and not buffer._prompt_group_keys and not buffer._ready

    _run(asyncio.wait_for(scenario(), timeout=3.0))


def test_nonterminal_statuses_do_not_affect_capacity():
    buffer = ExperienceBufferServer(ExperienceBufferConfig(max_size=2, sampler="fifo"))
    uid = "live-metrics"
    key = _key(uid)

    _run(
        buffer.add(
            {
                "running": {"status": "running"},
                "pending": {"status": "pending"},
                key: _success_tag(uid),
                uid: {"status": "finished"},
            }
        )
    )

    assert list(buffer._ready) == [key]
    assert list(buffer._tags) == [key]


def test_sample_wait_is_woken_only_when_uid_marker_closes_the_group():
    buffer = ExperienceBufferServer(ExperienceBufferConfig(poll_interval=10.0))
    uid = "wake-sample"
    key = _key(uid)

    async def scenario():
        sample_task = asyncio.create_task(buffer.sample(batch_size=1))
        await asyncio.sleep(0)
        assert not sample_task.done()

        await buffer.add({key: _success_tag(uid)})
        await asyncio.sleep(0)
        assert not sample_task.done()

        await buffer.add({uid: {"status": "finished"}})
        return await asyncio.wait_for(sample_task, timeout=1.0)

    batch = _run(scenario())
    assert batch.keys == [key]


def test_competing_sample_waiters_recheck_capacity_and_cancellation_preserves_data():
    async def scenario():
        buffer = ExperienceBufferServer(ExperienceBufferConfig(sampler="fifo", poll_interval=10.0))
        first = asyncio.create_task(buffer.sample(batch_size=2))
        second = asyncio.create_task(buffer.sample(batch_size=2))
        cancelled = asyncio.create_task(buffer.sample(batch_size=1))
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled

        async with buffer._condition:
            buffer._condition.notify_all()
        await asyncio.sleep(0)
        assert not first.done() and not second.done()

        first_keys = [_key("first", session_id) for session_id in range(2)]
        await buffer.add({**{key: _success_tag("first") for key in first_keys}, "first": {"status": "finished"}})
        assert (await first).keys == first_keys
        assert not second.done()

        for uid in ("second", "third"):
            await buffer.add({_key(uid): _success_tag(uid), uid: {"status": "finished"}})
            if uid == "second":
                await asyncio.sleep(0)
                assert not second.done()
        assert (await second).keys == [_key("second"), _key("third")]
        assert len(buffer) == 0
        assert (await buffer.get_metrics())["lifecycle/sampled_trajectory_count"] == 4

    _run(asyncio.wait_for(scenario(), timeout=1.0))


def test_background_poll_all_failure_marker_is_released_and_polling_continues(monkeypatch):
    poll_count = 0
    uid = "poll-success"
    key = _key(uid)

    async def fake_list_metadata(*, namespace=None):  # noqa: ANN001
        nonlocal poll_count
        poll_count += 1
        if poll_count == 1:
            return {"train": {"poll-failure": {"status": "failure"}}}
        return {
            "train": {
                key: _success_tag(uid, version=7),
                uid: {"status": "finished"},
            }
        }

    storage = _install_store(monkeypatch, list_metadata_handler=fake_list_metadata)
    buffer = ExperienceBufferServer(ExperienceBufferConfig(poll_interval=0.001))

    async def exercise():
        await buffer.start()
        return await buffer.sample(batch_size=1)

    batch = _run(exercise())

    assert batch.keys == [key]
    assert storage.release_calls[:2] == [(["poll-failure"], "train"), ([uid], "train")]


def test_background_poll_transport_failure_fails_poll_task(monkeypatch):
    async def fake_list_metadata(*, namespace=None):  # noqa: ANN001
        raise ConnectionError("transfer queue unavailable")

    _install_store(monkeypatch, list_metadata_handler=fake_list_metadata)
    buffer = ExperienceBufferServer(ExperienceBufferConfig(poll_interval=0.001))

    async def exercise():
        await buffer.start()
        poll_task = buffer._poll_task
        assert poll_task is not None
        with pytest.raises(ConnectionError, match="transfer queue unavailable"):
            await poll_task

    _run(exercise())


def test_background_poll_failure_marker_release_error_fails_poll_task(monkeypatch):
    async def fake_list_metadata(*, namespace=None):  # noqa: ANN001
        return {"train": {"failed-prompt": {"status": "failure"}}}

    def fake_release(keys, *, namespace):  # noqa: ANN001
        raise ConnectionError("transfer queue release unavailable")

    _install_store(
        monkeypatch,
        list_metadata_handler=fake_list_metadata,
        release_handler=fake_release,
    )
    buffer = ExperienceBufferServer(ExperienceBufferConfig(poll_interval=0.001))

    async def exercise():
        await buffer.start()
        poll_task = buffer._poll_task
        assert poll_task is not None
        with pytest.raises(ConnectionError, match="transfer queue release unavailable"):
            await poll_task
        return await buffer.get_metrics()

    metrics = _run(exercise())
    assert metrics["sink/prompt_terminal_clear_failure_count"] == 1


def test_poll_only_ingests_configured_partition(monkeypatch):
    uid = "train-prompt"
    key = _key(uid)

    async def fake_list_metadata(*, namespace=None):  # noqa: ANN001
        return {
            "val": {"val-prompt": {"status": "failure"}},
            "train": {
                key: _success_tag(uid, version=0),
                uid: {"status": "finished"},
            },
        }

    storage = _install_store(monkeypatch, list_metadata_handler=fake_list_metadata)
    buffer = ExperienceBufferServer(ExperienceBufferConfig(poll_interval=0.001))

    async def exercise():
        await buffer.start()
        return await asyncio.wait_for(buffer.sample(batch_size=1), timeout=1.0)

    batch = _run(exercise())
    assert batch.keys == [key]
    assert set(buffer._tags) == {key}
    assert storage.release_calls == [([uid], "train")]
    assert _run(buffer.get_metrics())["lifecycle/committed_trajectory_count"] == 1
