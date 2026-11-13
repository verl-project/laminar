# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from laminar.eval import main as validation
from laminar.eval.worker import evaluation_profile
from test_validation_on_cpu import _config


def observer(tmp_path, manager):
    config = _config(tmp_path)
    validation._validate_config(config)
    runtime = SimpleNamespace(agent_loop_manager=manager, service_started_at="service-identity")
    return validation._EfficiencySnapshots(config, runtime)


def test_cumulative_zero_and_repeated_snapshots_do_not_reset(tmp_path):
    metrics = {
        "task_reports/count": 0,
        "admission/wait_s": 0.0,
        "sessions/admitted": 0,
        "sessions/completed": 0,
        "sessions/in_flight": 0,
        "model/gateway_count": 1,
        "model/request_elapsed_s": 0.0,
    }

    async def getter():
        return metrics

    sink = observer(tmp_path, SimpleNamespace(get_efficiency_metrics=getter))
    try:
        sink.capture()
        metrics.update({"task_reports/count": 2, "model/request_elapsed_s": 12.0})
        sink.capture()
        rows = [json.loads(p.read_text()) for p in sorted(sink.path.glob("efficiency_*.json"))]
        assert [r["status"] for r in rows] == ["PASS", "PASS"]
        assert rows[0]["metrics"]["task_reports/count"] == 0
        assert rows[1]["metrics"] == metrics
        assert metrics["model/request_elapsed_s"] == 12.0
        assert rows[1]["semantics"]["reset"] is False
        assert "not GPU wall fractions" in rows[1]["semantics"]["timers"]
    finally:
        sink.close()


def test_model_only_snapshot_is_unknown_and_preserves_real_partial_counters(tmp_path):
    metrics = {"model/gateway_count": 4, "model/input_tokens": 0}

    async def getter():
        return metrics

    sink = observer(tmp_path, SimpleNamespace(get_efficiency_metrics=getter))
    try:
        sink.capture()
        row = json.loads(next(sink.path.glob("efficiency_*.json")).read_text())
        assert row["status"] == "UNKNOWN" and row["metrics"] == metrics
        assert "task_reports/count" in row["missing_required_metrics"]
        assert "sessions/admitted" in row["missing_required_metrics"]
        assert "not atomic" in row["semantics"]["consistency"]
        assert "not physical prefill" in row["semantics"]["tokens"]
    finally:
        sink.close()


@pytest.mark.parametrize("fault", ["missing", "empty", "failure", "nonfinite", "timeout"])
def test_observation_failures_are_unknown_without_checkpoint_failure(tmp_path, fault):
    async def getter():
        if fault == "failure":
            raise RuntimeError("do not persist a raw exception payload")
        if fault == "timeout":
            await asyncio.sleep(10)
        return {} if fault == "empty" else {"reward/total_s": float("nan")}

    manager = SimpleNamespace() if fault == "missing" else SimpleNamespace(get_efficiency_metrics=getter)
    sink = observer(tmp_path, manager)
    sink.timeout = 0.01
    try:
        sink.capture()
        row = json.loads(next(sink.path.glob("efficiency_*.json")).read_text())
        assert row["status"] == "UNKNOWN" and row["metrics"] == {}
        assert "do not persist" not in json.dumps(row)
        if fault == "timeout":
            assert row["error_type"] == "TimeoutError"
    finally:
        sink.close()


def test_pending_getter_is_single_and_does_not_block_coordinator(tmp_path):
    started, finish = threading.Event(), threading.Event()
    calls = []

    async def getter():
        calls.append(1)
        started.set()
        while not finish.is_set():
            await asyncio.sleep(0.001)
        return {"sessions/completed": 1}

    sink = observer(tmp_path, SimpleNamespace(get_efficiency_metrics=getter))
    try:
        sink.poll()
        assert started.wait(2)
        sink.next_read = 0
        sink.poll()
        assert calls == [1] and not sink.future.done()
        # The coordinator continues without waiting for this pending RPC.
        finish.set()
        sink.future.result(timeout=2)
    finally:
        finish.set()
        sink.close()


def test_persistence_failure_is_logged_unknown_not_a_pass(tmp_path, monkeypatch, caplog):
    async def getter():
        return {"sessions/completed": 1}

    sink = observer(tmp_path, SimpleNamespace(get_efficiency_metrics=getter))
    monkeypatch.setattr(validation, "_write_result", lambda *args: (_ for _ in ()).throw(OSError("disk failure")))
    try:
        with pytest.raises(OSError):
            sink.capture()
        assert "snapshot persistence UNKNOWN" in caplog.text
        assert not list(sink.path.glob("efficiency_*.json"))
    finally:
        sink.close()


@pytest.mark.parametrize("interval", [0, -1, float("inf"), True])
def test_invalid_observation_interval_fails_before_runtime(tmp_path, interval):
    config = _config(tmp_path)
    config.validation.efficiency_interval_s = interval
    with pytest.raises(ValueError, match="efficiency_interval_s"):
        validation._validate_config(config)


def test_observation_interval_does_not_change_scoring_profile(tmp_path, monkeypatch):
    config = _config(tmp_path)
    task = tmp_path / "task.yaml"
    task.write_text("formal scoring")
    # Use a complete model profile rather than interpreting observation settings as protocol.
    monkeypatch.setattr("laminar.eval.worker.model_profile", lambda *args: {"architecture": "same"})
    config.actor_rollout_ref.model = {"path": config.validation.checkpoints[0]}
    config.actor_rollout_ref.rollout.update(
        prompt_length=1024,
        response_length=2048,
        custom={"agent_framework": {"agent_runners": {"task": {"runner_kwargs": {"task_config_path": str(task)}}}}},
    )
    before = evaluation_profile(config)
    config.validation.efficiency_interval_s = 30
    assert evaluation_profile(config) == before
