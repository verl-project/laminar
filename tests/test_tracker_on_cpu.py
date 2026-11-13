# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import json
import time
from typing import Any, cast

import orjson
import pytest

from laminar.metrics import MetricSnapshot, SnapshotMetadata
from laminar.train.relay import RelayClient
from laminar.train.tracker import Tracker, TrackerConfig


class _RemoteMetricMethod:
    def __init__(self, metrics: dict[str, float | int]):
        self.metrics = metrics
        self.calls = 0

    async def remote(self) -> dict[str, float | int]:
        self.calls += 1
        return dict(self.metrics)


class _MetricSource:
    def __init__(self, metrics: dict[str, float | int]):
        self.get_metrics = _RemoteMetricMethod(metrics)


class _CoordinatedRemoteMetricMethod:
    def __init__(self, entered: asyncio.Event, release: asyncio.Event, metrics: dict[str, float | int]):
        self.entered = entered
        self.release = release
        self.metrics = metrics

    async def remote(self) -> dict[str, float | int]:
        self.entered.set()
        await self.release.wait()
        return dict(self.metrics)


class _CoordinatedMetricSource:
    def __init__(self, entered: asyncio.Event, release: asyncio.Event, metrics: dict[str, float | int]):
        self.get_metrics = _CoordinatedRemoteMetricMethod(entered, release, metrics)


class _ExceptionRemoteMetricMethod:
    async def remote(self) -> dict[str, float]:
        raise RuntimeError("metric source failed")


class _ExceptionMetricSource:
    get_metrics = _ExceptionRemoteMetricMethod()


class _CapturingLogger:
    def __init__(self):
        self.calls = []
        self.finish_calls = 0

    def log(self, data, step):
        self.calls.append((step, dict(data)))

    def finish(self, exit_code=0):
        self.finish_calls += 1


async def _collect_and_commit(
    tracker: Tracker,
    *,
    step: int,
    train_metrics: dict[str, float | int],
) -> None:
    await tracker.collect_and_stage_remote_metrics(step=step)
    await tracker.commit_metrics(step=step, train_metrics=train_metrics)


def _tracker_with_sources(config: TrackerConfig, **sources) -> Tracker:
    tracker = Tracker(config=config)
    tracker.connect_metric_sources(**sources)
    return tracker


def test_tracker_resume_drops_stale_metric_state():
    tracker = Tracker(config=TrackerConfig())
    tracker.metric_buffer[1] = {"train/loss": 0.25}

    tracker.set_next_step_to_log(2)

    assert tracker.next_step_to_log == 2
    assert 1 not in tracker.metric_buffer


def test_tracker_dump_writes_training_metrics(tmp_path):
    dump_path = tmp_path / "metrics.jsonl"
    tracker = Tracker(config=TrackerConfig(dump_path=str(dump_path)))

    tracker.maybe_dump(step=5, metrics={"train/loss": 0.25})

    rows = [orjson.loads(line) for line in dump_path.read_bytes().splitlines()]
    assert rows == [{"step": 5, "train/loss": 0.25}]


def test_tracker_preflights_persistent_metrics_path(tmp_path):
    dump_path = tmp_path / "metrics" / "metrics.jsonl"

    tracker = Tracker(
        config=TrackerConfig(
            dump_path=str(dump_path),
            dump_require_persistent_path=True,
            dump_persistent_roots=(str(tmp_path),),
        )
    )

    assert tracker.config.dump_path == str(dump_path)
    assert dump_path.exists()


def test_tracker_rejects_non_persistent_metrics_path(tmp_path):
    with pytest.raises(ValueError, match="must be under a persistent HDFS mount root"):
        Tracker(
            config=TrackerConfig(
                dump_path=str(tmp_path / "metrics.jsonl"),
                dump_require_persistent_path=True,
                dump_persistent_roots=("/mnt/hdfs",),
            )
        )


def test_tracker_rejects_relative_persistent_metrics_path():
    with pytest.raises(ValueError, match="must be an absolute path under a persistent HDFS mount root"):
        Tracker(
            config=TrackerConfig(
                dump_path="metrics.jsonl",
                dump_require_persistent_path=True,
                dump_persistent_roots=("/mnt/hdfs",),
            )
        )


def test_tracker_rejects_hdfs_uri_metrics_path():
    with pytest.raises(ValueError, match="must use a mounted writable filesystem path"):
        Tracker(config=TrackerConfig(dump_path="hdfs://cluster/path/metrics.jsonl"))


def test_tracker_dump_oserror_is_fatal(monkeypatch, tmp_path):
    tracker = Tracker(config=TrackerConfig(dump_path=str(tmp_path / "metrics.jsonl")))

    def fail_open(*_args, **_kwargs):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr("builtins.open", fail_open)
    with pytest.raises(RuntimeError, match="Failed to dump tracker metrics for step=7"):
        tracker.maybe_dump(step=7, metrics={"train/loss": 0.25})


def test_tracker_commit_metrics_flushes_ready_steps(tmp_path):
    dump_path = tmp_path / "metrics.jsonl"
    tracker = Tracker(config=TrackerConfig(dump_path=str(dump_path)))

    asyncio.run(_collect_and_commit(tracker, step=1, train_metrics={"train/loss": 0.25}))

    rows = [orjson.loads(line) for line in dump_path.read_bytes().splitlines()]
    assert rows == [{"step": 1, "train/loss": 0.25}]
    assert tracker.next_step_to_log == 2


def test_tracker_commit_uses_staged_remote_metrics_without_recollecting(tmp_path):
    dump_path = tmp_path / "metrics.jsonl"
    rollout_dispatcher = _MetricSource({"request/num_finished/last_step/sum": 3})
    tracker = _tracker_with_sources(
        config=TrackerConfig(dump_path=str(dump_path)),
        rollout_dispatcher=rollout_dispatcher,
    )

    async def _run() -> None:
        await tracker.collect_and_stage_remote_metrics(step=0)
        await tracker.commit_metrics(step=0, train_metrics={"timing/step": 12.0})

    asyncio.run(_run())

    row = orjson.loads(dump_path.read_bytes().splitlines()[0])
    assert row["timing/step"] == 12.0
    assert row["rollout/request/num_finished/last_step/sum"] == 3
    assert rollout_dispatcher.get_metrics.calls == 1


def test_tracker_collects_classic_remote_metric_sources(tmp_path):
    dump_path = tmp_path / "metrics.jsonl"
    rollout_dispatcher = _MetricSource({"request/num_finished/last_step/sum": 3})
    experience_buffer = _MetricSource(
        {
            "conservation/activation_residual": 0,
            "lifecycle/committed_attention_tokens": 125,
            "lifecycle/committed_trajectory_count": 3,
        }
    )
    multi_version_rollout_manager = _MetricSource({"repack/plan_and_begin_attempts": 4})
    relay_manager = _MetricSource({"broadcast/latency_s/last": 5.5})
    tracker = _tracker_with_sources(
        config=TrackerConfig(dump_path=str(dump_path)),
        rollout_dispatcher=rollout_dispatcher,
        experience_buffer=experience_buffer,
        multi_version_rollout_manager=multi_version_rollout_manager,
        relay_client=RelayClient(cast(Any, relay_manager)),
    )

    asyncio.run(_collect_and_commit(tracker, step=0, train_metrics={}))
    metrics = orjson.loads(dump_path.read_bytes().splitlines()[0])

    assert metrics["rollout/request/num_finished/last_step/sum"] == 3
    assert metrics["experience_buffer/conservation/activation_residual"] == 0
    assert metrics["experience_buffer/lifecycle/committed_attention_tokens"] == 125
    assert metrics["experience_buffer/lifecycle/committed_trajectory_count"] == 3
    assert metrics["multi_version_rollout/repack/plan_and_begin_attempts"] == 4
    assert metrics["relay/broadcast/latency_s/last"] == 5.5
    assert metrics["tracker/remote_metrics/source_count"] == 4.0
    assert metrics["tracker/remote_metrics/success_count"] == 4.0
    for source in ("rollout", "experience_buffer", "multi_version_rollout", "relay"):
        assert metrics[f"tracker/remote_metrics/source/{source}/success"] == 1.0


def test_tracker_collects_remote_metrics_concurrently(tmp_path):
    async def _run() -> None:
        rollout_entered = asyncio.Event()
        relay_entered = asyncio.Event()
        release = asyncio.Event()
        tracker = _tracker_with_sources(
            config=TrackerConfig(
                dump_path=str(tmp_path / "metrics.jsonl"),
                remote_metrics_timeout_s=1.0,
            ),
            rollout_dispatcher=_CoordinatedMetricSource(rollout_entered, release, {"ready": 1}),
            relay_client=RelayClient(cast(Any, _CoordinatedMetricSource(relay_entered, release, {"ready": 2}))),
        )
        collect_task = asyncio.create_task(tracker.collect_and_stage_remote_metrics(step=0))
        await asyncio.wait_for(asyncio.gather(rollout_entered.wait(), relay_entered.wait()), timeout=0.1)
        release.set()
        await collect_task
        await tracker.commit_metrics(step=0, train_metrics={})
        metrics = orjson.loads((tmp_path / "metrics.jsonl").read_bytes().splitlines()[0])
        assert metrics["rollout/ready"] == 1
        assert metrics["relay/ready"] == 2

    asyncio.run(_run())


def test_tracker_timeout_and_failure_do_not_block_healthy_source(tmp_path):
    async def _run() -> None:
        entered = asyncio.Event()
        never_release = asyncio.Event()
        tracker = _tracker_with_sources(
            config=TrackerConfig(
                dump_path=str(tmp_path / "metrics.jsonl"),
                remote_metrics_timeout_s=0.01,
            ),
            rollout_dispatcher=_CoordinatedMetricSource(entered, never_release, {"unreachable": 1}),
            relay_client=RelayClient(cast(Any, _ExceptionMetricSource())),
            multi_version_rollout_manager=_MetricSource({"ready": 8}),
        )
        await _collect_and_commit(tracker, step=0, train_metrics={"train/loss": 0.25})
        metrics = orjson.loads((tmp_path / "metrics.jsonl").read_bytes().splitlines()[0])
        assert entered.is_set()
        assert metrics["multi_version_rollout/ready"] == 8
        assert "rollout/unreachable" not in metrics
        assert metrics["tracker/remote_metrics/source/rollout/timeout"] == 1.0
        assert metrics["tracker/remote_metrics/source/relay/failure"] == 1.0
        assert metrics["tracker/remote_metrics/source/multi_version_rollout/success"] == 1.0

    asyncio.run(_run())


def test_tracker_two_phase_handoff_rejects_missing_duplicate_and_out_of_order_calls():
    async def _run() -> None:
        tracker = Tracker(config=TrackerConfig())

        with pytest.raises(RuntimeError, match="missing staged remote metrics"):
            await tracker.commit_metrics(step=0, train_metrics={})

        await tracker.collect_and_stage_remote_metrics(step=0)
        with pytest.raises(RuntimeError, match="duplicate remote metrics collect"):
            await tracker.collect_and_stage_remote_metrics(step=0)
        with pytest.raises(RuntimeError, match="missing metrics commit for staged step=0"):
            await tracker.collect_and_stage_remote_metrics(step=1)
        with pytest.raises(RuntimeError, match="out-of-order metrics commit"):
            await tracker.commit_metrics(step=1, train_metrics={})

        await tracker.commit_metrics(step=0, train_metrics={"train/loss": 0.25})
        with pytest.raises(RuntimeError, match="missing staged remote metrics"):
            await tracker.commit_metrics(step=0, train_metrics={})

    asyncio.run(_run())


def test_tracker_rejects_duplicate_collect_while_first_collect_is_in_flight():
    async def _run() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        tracker = _tracker_with_sources(
            config=TrackerConfig(remote_metrics_timeout_s=1.0),
            rollout_dispatcher=_CoordinatedMetricSource(entered, release, {"ready": 1}),
        )
        collect_task = asyncio.create_task(tracker.collect_and_stage_remote_metrics(step=0))
        await asyncio.wait_for(entered.wait(), timeout=0.1)
        with pytest.raises(RuntimeError, match="duplicate remote metrics collect already in flight"):
            await tracker.collect_and_stage_remote_metrics(step=0)
        release.set()
        await collect_task
        await tracker.commit_metrics(step=0, train_metrics={"train/loss": 0.25})

    asyncio.run(_run())


def test_tracker_commit_persistence_failure_propagates(monkeypatch, tmp_path):
    tracker = Tracker(config=TrackerConfig(dump_path=str(tmp_path / "metrics.jsonl")))

    def fail_open(*_args, **_kwargs):
        raise OSError(5, "I/O error")

    async def _run() -> None:
        await tracker.collect_and_stage_remote_metrics(step=0)
        monkeypatch.setattr("builtins.open", fail_open)
        with pytest.raises(RuntimeError, match="Failed to dump tracker metrics"):
            await tracker.commit_metrics(step=0, train_metrics={"train/loss": 0.25})

    asyncio.run(_run())


def test_tracker_reports_complete_commit_latency_on_next_row(tmp_path):
    dump_path = tmp_path / "metrics.jsonl"
    tracker = Tracker(config=TrackerConfig(dump_path=str(dump_path)))

    class _DelayedLogger(_CapturingLogger):
        def log(self, data, step):
            time.sleep(0.01)
            super().log(data, step)

    tracker._logger = _DelayedLogger()

    async def _run() -> None:
        await _collect_and_commit(tracker, step=0, train_metrics={"timing/step": 1.0})
        await _collect_and_commit(tracker, step=1, train_metrics={"timing/step": 2.0})

    asyncio.run(_run())

    rows = [orjson.loads(line) for line in dump_path.read_bytes().splitlines()]
    assert "tracker/metrics_commit_latency_s" not in rows[0]
    assert rows[1]["tracker/metrics_commit_latency_committed_step"] == 0.0
    assert rows[1]["tracker/metrics_commit_latency_s"] >= 0.01


def test_tracker_rejects_non_positive_remote_metrics_timeout():
    with pytest.raises(ValueError, match="remote_metrics_timeout_s must be positive"):
        Tracker(config=TrackerConfig(remote_metrics_timeout_s=0.0))


class _SnapshotRemoteMethod:
    def __init__(self, *snapshots):
        self.snapshots = iter(snapshots)

    async def remote(self):
        snapshot = next(self.snapshots)
        if isinstance(snapshot, Exception):
            raise snapshot
        return snapshot


def _snapshot(value, *, epoch="a", sequence=1, timestamp=10):
    return MetricSnapshot(
        {"committed": value, "latest_count": 9},
        SnapshotMetadata(epoch, timestamp, sequence, 0.0, ("committed",)),
    )


def test_tracker_counter_windows_survive_failed_poll_and_rebaseline_on_restart(tmp_path):
    source = type("Source", (), {})()
    source.get_metrics = _SnapshotRemoteMethod(
        _snapshot(10),
        RuntimeError("unreachable"),
        _snapshot(18, sequence=3, timestamp=14),
        _snapshot(2, epoch="b", timestamp=15),
        _snapshot(5, epoch="b", sequence=2, timestamp=17),
    )
    dump_path = tmp_path / "metrics.jsonl"
    tracker = _tracker_with_sources(TrackerConfig(dump_path=str(dump_path)), experience_buffer=source)
    tracker._logger = _CapturingLogger()

    async def run():
        for step in range(5):
            await _collect_and_commit(tracker, step=step, train_metrics={})

    asyncio.run(run())
    rows = [orjson.loads(line) for line in dump_path.read_bytes().splitlines()]
    source_key = "tracker/remote_metrics/source/experience_buffer"
    assert "experience_buffer/committed/window_delta" not in rows[0]
    assert rows[1]["tracker/remote_metrics/coverage"] == 0
    assert rows[2]["experience_buffer/committed/window_delta"] == 8
    assert rows[2]["experience_buffer/committed/window_rate_per_s"] == 2
    assert rows[2][f"{source_key}/poll_gap_count"] == 1
    assert rows[2][f"{source_key}/snapshot_sequence_gap"] == 1
    assert rows[3][f"{source_key}/counter_reset"] == 1
    assert "experience_buffer/committed/window_delta" not in rows[3]
    assert rows[4]["experience_buffer/committed/window_delta"] == 3
    assert "experience_buffer/latest_count/window_delta" not in rows[4]
    assert rows[4]["source_metadata"]["experience_buffer"]["source_epoch"] == "b"
    assert "source_metadata" not in tracker._logger.calls[-1][1]


def test_tracker_same_epoch_counter_drop_is_an_explicit_reset(tmp_path):
    source = type("Source", (), {})()
    source.get_metrics = _SnapshotRemoteMethod(_snapshot(10), _snapshot(1, sequence=2, timestamp=11))
    tracker = _tracker_with_sources(TrackerConfig(), experience_buffer=source)
    tracker._logger = _CapturingLogger()

    async def run():
        await _collect_and_commit(tracker, step=0, train_metrics={})
        await _collect_and_commit(tracker, step=1, train_metrics={})

    asyncio.run(run())
    metrics = tracker._logger.calls[-1][1]
    assert metrics["tracker/remote_metrics/source/experience_buffer/counter_reset"] == 1
    assert "experience_buffer/committed/window_delta" not in metrics


def test_tracker_stale_snapshot_keeps_latest_counter_baseline():
    source = type("Source", (), {})()
    source.get_metrics = _SnapshotRemoteMethod(
        _snapshot(10),
        _snapshot(20, sequence=2, timestamp=11),
        _snapshot(10),
        _snapshot(30, sequence=3, timestamp=12),
    )
    tracker = _tracker_with_sources(TrackerConfig(), experience_buffer=source)
    tracker._logger = _CapturingLogger()

    async def run():
        for step in range(4):
            await _collect_and_commit(tracker, step=step, train_metrics={})

    asyncio.run(run())
    stale_metrics = tracker._logger.calls[2][1]
    assert stale_metrics["tracker/remote_metrics/source/experience_buffer/stale_snapshot"] == 1
    assert stale_metrics["tracker/remote_metrics/source/experience_buffer/counter_reset"] == 0
    assert "experience_buffer/committed/window_delta" not in stale_metrics
    assert tracker._logger.calls[3][1]["experience_buffer/committed/window_delta"] == 10
    assert tracker._logger.calls[3][1]["tracker/remote_metrics/source/experience_buffer/poll_gap_count"] == 1


def test_tracker_invalid_counter_snapshot_is_isolated_from_other_sources():
    source = type("Source", (), {})()
    source.get_metrics = _SnapshotRemoteMethod(_snapshot(float("nan")))
    tracker = _tracker_with_sources(
        TrackerConfig(), experience_buffer=source, rollout_dispatcher=_MetricSource({"ready": 3})
    )
    tracker._logger = _CapturingLogger()
    asyncio.run(_collect_and_commit(tracker, step=0, train_metrics={}))
    metrics = tracker._logger.calls[0][1]
    assert metrics["rollout/ready"] == 3
    assert "experience_buffer/committed" not in metrics
    assert metrics["tracker/remote_metrics/source/experience_buffer/failure"] == 1
    assert metrics["tracker/remote_metrics/coverage"] == 0.5


def test_tracker_rejects_owner_collision_without_silent_overwrite():
    tracker = _tracker_with_sources(TrackerConfig(), rollout_dispatcher=_MetricSource({"ready": 1}))
    tracker._logger = _CapturingLogger()

    async def run():
        await tracker.collect_and_stage_remote_metrics(step=0)
        with pytest.raises(ValueError, match="owners collide"):
            await tracker.commit_metrics(step=0, train_metrics={"rollout/ready": 2})

    asyncio.run(run())
    assert tracker._logger.calls == []


def test_tracker_publish_event_and_live_snapshot_keep_separate_owners():
    tracker = _tracker_with_sources(
        TrackerConfig(), relay_client=RelayClient(cast(Any, _MetricSource({"broadcast/tensor_count": 4})))
    )
    tracker._logger = _CapturingLogger()
    asyncio.run(_collect_and_commit(tracker, step=0, train_metrics={"relay/broadcast/tensor_count": 3}))
    metrics = tracker._logger.calls[0][1]
    assert metrics["relay/broadcast/tensor_count"] == 4
    assert metrics["relay_publish/broadcast/tensor_count"] == 3


def test_tracker_final_events_persist_without_rewriting_scalar_steps(tmp_path):
    dump_path = tmp_path / "metrics.jsonl"
    tracker = Tracker(TrackerConfig(dump_path=str(dump_path)))
    tracker._logger = _CapturingLogger()

    async def run():
        await _collect_and_commit(tracker, step=1, train_metrics={"timing/step": 2})
        await tracker.record_event(1, "checkpoint", "checkpoint:1", {"checkpoint/completed": 1})
        await tracker.record_event(1, "checkpoint", "checkpoint:1", {"checkpoint/completed": 1})
        with pytest.raises(ValueError, match="different contents"):
            await tracker.record_event(1, "checkpoint", "checkpoint:1", {"checkpoint/completed": 0})
        await tracker.finalize()
        await tracker.finalize()

    asyncio.run(run())
    rows = [orjson.loads(line) for line in dump_path.read_bytes().splitlines()]
    events = [orjson.loads(line) for line in (tmp_path / "metrics.events.jsonl").read_bytes().splitlines()]
    assert len(rows) == 1
    assert rows[0]["timing_s/step"] == 2
    assert [step for step, _ in tracker._logger.calls] == [1]
    assert [event["event_type"] for event in events] == ["checkpoint", "metrics_commit"]
    assert events[1]["step"] == 1
    assert events[1]["metrics"]["tracker/metrics_commit_latency_committed_step"] == 1


def test_tracker_event_persistence_failure_propagates(tmp_path, monkeypatch):
    tracker = Tracker(TrackerConfig(dump_path=str(tmp_path / "metrics.jsonl")))

    def fail_open(*args, **kwargs):
        raise OSError(5, "I/O error")

    monkeypatch.setattr("builtins.open", fail_open)
    with pytest.raises(RuntimeError, match="Failed to dump tracker event"):
        asyncio.run(tracker.record_event(1, "checkpoint", "checkpoint:1", {"checkpoint/completed": 1}))


def test_tracker_finalizes_existing_tracking_backend_once_after_tail_persistence(tmp_path):
    from verl.utils.tracking import Tracking

    dump_path = tmp_path / "metrics.jsonl"
    tracker = Tracker(TrackerConfig(dump_path=str(dump_path)))
    recorded = _CapturingLogger()
    tracking = Tracking.__new__(Tracking)
    tracking.logger = {"file": recorded}
    tracker._logger = tracking

    async def run():
        await _collect_and_commit(tracker, step=1, train_metrics={"train/loss": 0.5})
        await tracker.finalize()
        await tracker.finalize()

    asyncio.run(run())
    assert recorded.finish_calls == 1
    assert len(recorded.calls) == 1
    events = [orjson.loads(line) for line in (tmp_path / "metrics.events.jsonl").read_bytes().splitlines()]
    assert len(events) == 1
    assert events[0]["event_type"] == "metrics_commit"
    assert tracker._finalized


def test_tracker_finish_failure_propagates_without_marking_finalized(tmp_path):
    class FailingLogger(_CapturingLogger):
        def finish(self, exit_code=0):
            raise RuntimeError("backend flush failed")

    tracker = Tracker(TrackerConfig(dump_path=str(tmp_path / "metrics.jsonl")))
    tracker._logger = FailingLogger()

    async def run():
        await _collect_and_commit(tracker, step=1, train_metrics={"train/loss": 0.5})
        with pytest.raises(RuntimeError, match="backend flush failed"):
            await tracker.finalize()

    asyncio.run(run())
    assert not tracker._finalized
    events = (tmp_path / "metrics.events.jsonl").read_bytes().splitlines()
    assert len(events) == 1


def test_tracker_finalize_does_not_initialize_unused_tracking(monkeypatch):
    def fail_init(*args, **kwargs):
        raise AssertionError("unused Tracking must not initialize during finalize")

    monkeypatch.setattr("laminar.train.tracker.Tracking", fail_init)
    tracker = Tracker(TrackerConfig())
    asyncio.run(tracker.finalize())
    assert tracker._logger is None
    assert tracker._finalized


class _FinishingLogger(_CapturingLogger):
    def __init__(self, error=None):
        super().__init__()
        self.finish_calls = []
        self.pending = None
        self.persisted = []
        self.error = error

    def log(self, data, step):
        super().log(data, step)
        if self.pending is not None:
            self.persisted.append(self.pending)
        self.pending = (step, dict(data))

    def finish(self, exit_code=0):
        self.finish_calls.append(exit_code)
        if self.error is not None:
            raise self.error
        if self.pending is not None:
            self.persisted.append(self.pending)
            self.pending = None


def test_tracker_finish_flushes_last_backend_point_once_without_replay(tmp_path):
    tracker = Tracker(TrackerConfig(dump_path=str(tmp_path / "metrics.jsonl")))
    backend = _FinishingLogger()
    tracker._logger = backend
    asyncio.run(_collect_and_commit(tracker, step=0, train_metrics={"train/loss": 0.25}))
    assert backend.persisted == []
    asyncio.run(tracker.finalize())
    tracker.finish()
    tracker.finish()
    assert backend.finish_calls == [0]
    assert backend.persisted == backend.calls == [(0, {"train/loss": 0.25})]
    assert len((tmp_path / "metrics.jsonl").read_bytes().splitlines()) == 1
    events = [orjson.loads(line) for line in (tmp_path / "metrics.events.jsonl").read_bytes().splitlines()]
    assert [event["event_type"] for event in events] == ["metrics_commit"]
    assert events[0]["step"] == 0
    with pytest.raises(RuntimeError, match="already finalized"):
        asyncio.run(tracker.collect_and_stage_remote_metrics(1))


def test_tracker_finish_without_logger_does_not_create_backend(monkeypatch):
    def forbidden_backend(**_kwargs):
        raise AssertionError("finish must not initialize a backend")

    monkeypatch.setattr("laminar.train.tracker.Tracking", forbidden_backend)
    tracker = Tracker(TrackerConfig())
    tracker.finish()
    tracker.finish()
    assert tracker._logger is None


def test_tracker_error_finish_preserves_uncommitted_staging():
    tracker = Tracker(TrackerConfig())
    backend = _FinishingLogger()
    tracker._logger = backend
    asyncio.run(tracker.collect_and_stage_remote_metrics(0))
    with pytest.raises(RuntimeError, match="uncommitted metric work"):
        tracker.finish()
    assert backend.finish_calls == []
    tracker.finish(exit_code=1)
    assert backend.finish_calls == [1]
    assert tracker._staged_remote_metrics is not None
    assert backend.calls == []
    with pytest.raises(RuntimeError, match="already finalized"):
        asyncio.run(tracker.commit_metrics(0, {"train/loss": 0.25}))


def test_tracker_failed_training_finish_persists_committed_tail_without_replaying_staged_step(tmp_path):
    tracker = Tracker(TrackerConfig(dump_path=str(tmp_path / "metrics.jsonl")))
    backend = _FinishingLogger()
    tracker._logger = backend

    async def run():
        await _collect_and_commit(tracker, step=0, train_metrics={"train/loss": 0.25})
        await tracker.record_event(0, "checkpoint_completed", "checkpoint:0", {"checkpoint/completed": 1})
        await tracker.collect_and_stage_remote_metrics(1)
        tracker.finish(exit_code=1)
        await tracker.finalize()
        with pytest.raises(RuntimeError, match="already finalized"):
            await tracker.record_event(1, "training_cycle_completed", "cycle:1", {"timing/cycle": 1})

    asyncio.run(run())
    assert backend.finish_calls == [1]
    assert backend.persisted == backend.calls == [(0, {"train/loss": 0.25})]
    assert tracker._staged_remote_metrics.step == 1
    rows = [orjson.loads(line) for line in (tmp_path / "metrics.jsonl").read_bytes().splitlines()]
    events = [orjson.loads(line) for line in (tmp_path / "metrics.events.jsonl").read_bytes().splitlines()]
    assert [row["step"] for row in rows] == [0]
    assert [event["event_type"] for event in events] == ["checkpoint_completed", "metrics_commit"]
    assert events[-1]["metrics"]["tracker/metrics_commit_latency_committed_step"] == 0


def test_tracker_finish_event_write_failure_keeps_backend_open_for_tail_retry(tmp_path, monkeypatch):
    tracker = Tracker(TrackerConfig(dump_path=str(tmp_path / "metrics.jsonl")))
    backend = _FinishingLogger()
    tracker._logger = backend
    asyncio.run(_collect_and_commit(tracker, step=0, train_metrics={"train/loss": 0.25}))

    def failed_append(_event):
        raise RuntimeError("event persistence failed")

    with monkeypatch.context() as patch:
        patch.setattr(tracker, "_append_event", failed_append)
        with pytest.raises(RuntimeError, match="event persistence failed"):
            tracker.finish()
    assert backend.finish_calls == []
    assert tracker._finish_exit_code is None
    assert tracker._pending_metrics_commit_observation is not None
    asyncio.run(tracker.finalize())
    tracker.finish()
    assert backend.finish_calls == [0]
    events = [orjson.loads(line) for line in (tmp_path / "metrics.events.jsonl").read_bytes().splitlines()]
    assert [event["event_type"] for event in events] == ["metrics_commit"]


def test_tracker_error_finish_does_not_stage_a_collect_completed_after_close():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        tracker = _tracker_with_sources(
            TrackerConfig(),
            rollout_dispatcher=_CoordinatedMetricSource(entered, release, {"ready": 1}),
        )
        backend = _FinishingLogger()
        tracker._logger = backend
        collect = asyncio.create_task(tracker.collect_and_stage_remote_metrics(0))
        await entered.wait()
        with pytest.raises(RuntimeError, match="uncommitted metric work"):
            tracker.finish()
        tracker.finish(exit_code=1)
        release.set()
        with pytest.raises(RuntimeError, match="already finalized"):
            await collect
        assert tracker._staged_remote_metrics is None
        assert backend.finish_calls == [1]

    asyncio.run(run())


def test_tracker_backend_finish_failure_is_not_silently_retried():
    tracker = Tracker(TrackerConfig())
    backend = _FinishingLogger(OSError("backend finish failed"))
    tracker._logger = backend
    with pytest.raises(OSError, match="backend finish failed"):
        tracker.finish()
    with pytest.raises(RuntimeError, match="previously failed"):
        tracker.finish(exit_code=1)
    assert backend.finish_calls == [0]


def test_tracker_finish_does_not_replay_a_buffered_uncommitted_step():
    tracker = Tracker(TrackerConfig())
    backend = _FinishingLogger()
    tracker._logger = backend
    tracker.metric_buffer[0] = {"train/loss": 0.25}
    with pytest.raises(RuntimeError, match="uncommitted metric work"):
        tracker.finish()
    tracker.finish(exit_code=1)
    assert tracker.metric_buffer == {0: {"train/loss": 0.25}}
    assert backend.calls == []
    assert backend.finish_calls == [1]


@pytest.mark.parametrize("exit_code", [None, True])
def test_tracker_finish_rejects_invalid_exit_code_without_backend_creation(exit_code):
    tracker = Tracker(TrackerConfig())
    with pytest.raises(ValueError, match="exit_code must be an integer"):
        tracker.finish(exit_code)
    assert tracker._logger is None


def test_tracker_sanitized_config_sidecar_uses_short_name_and_preserves_science(tmp_path):
    config = {
        "trainer": {"total_training_steps": None},
        "actor_rollout_ref": {"rollout": {"n": 8}},
        "long_" + "x" * 65: "scientific value",
        "ray_kwargs": {"runtime_env": {"env_vars": {"SENSITIVE": "private environment"}}},
        "api_key": "private key",
    }
    original = copy.deepcopy(config)
    tracker = Tracker(
        TrackerConfig(
            dump_path=str(tmp_path / ("m" * 238 + ".jsonl")),
            resolved_config=config,
            config_key_max_bytes=64,
        )
    )
    assert config == original
    sidecar = next(tmp_path.glob("config-*.json"))
    assert len(sidecar.name.encode()) < 64
    assert tracker.config_dump_path == str(sidecar)
    payload = json.loads(sidecar.read_text())
    assert payload["resolved_config"]["trainer"] == config["trainer"]
    assert payload["resolved_config"]["actor_rollout_ref"] == config["actor_rollout_ref"]
    assert "private key" not in sidecar.read_text()
    assert "private environment" not in sidecar.read_text()
    assert set(payload["omitted_paths"]) == {"api_key", "ray_kwargs.runtime_env"}
    assert payload["projection"]["logger_config"] == tracker._logger_config
    assert tracker._logger_config["actor_rollout_ref"]["rollout"]["n"] == 8


def test_tracker_config_sidecar_failure_prevents_backend_initialization(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("laminar.train.tracker.Tracking", lambda **kwargs: calls.append(kwargs))

    def failed_replace(*_args):
        raise OSError("sidecar write failed")

    monkeypatch.setattr("laminar.train.tracker.os.replace", failed_replace)
    with pytest.raises(RuntimeError, match="Failed to persist Tracker config sidecar"):
        Tracker(
            TrackerConfig(resolved_config={"data": {"train_batch_size": 16}}, dump_path=str(tmp_path / "metrics.jsonl"))
        )
    assert calls == []
    assert not list(tmp_path.glob(".config-*"))
    assert not list(tmp_path.glob("config-*.json"))


def test_tracker_existing_sidecar_cannot_be_overwritten_with_another_config(tmp_path):
    path = str(tmp_path / "metrics.jsonl")
    Tracker(TrackerConfig(resolved_config={"data": {"train_batch_size": 16}}, dump_path=path))
    with pytest.raises(RuntimeError, match="different configuration"):
        Tracker(TrackerConfig(resolved_config={"data": {"train_batch_size": 32}}, dump_path=path))
