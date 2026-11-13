# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""CPU coverage for dispatcher pause on LLM-server KV pressure."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from laminar.llm_server.state_tracker import LLMServerStateTracker
from laminar.train.dispatcher import RolloutDispatcher


class _InProcessTracker:
    def __init__(self, snapshot: dict):
        self.snapshot = snapshot
        self.calls = 0

    def get_routing_snapshot(self) -> dict:
        self.calls += 1
        return self.snapshot


class _RemoteSnapshot:
    def __init__(self, snapshot: dict):
        self.snapshot = snapshot
        self.calls = 0

    def __call__(self, **kwargs):
        raise TypeError("Actor method must be invoked with .remote")

    def remote(self):
        self.calls += 1
        future = asyncio.get_event_loop().create_future()
        future.set_result(self.snapshot)
        return future


class _ActorHandleTracker:
    def __init__(self, snapshot: dict):
        self.get_routing_snapshot = _RemoteSnapshot(snapshot)


def _dispatcher(*, mode: str, tracker, threshold: float = 0.80) -> RolloutDispatcher:
    from laminar.metrics import MetricSource, cumulative_counter_specs
    from laminar.train.dispatcher import _DISPATCHER_COUNTER_KEYS

    producer = object.__new__(RolloutDispatcher)
    producer._metric_source = MetricSource(cumulative_counter_specs("rollout", _DISPATCHER_COUNTER_KEYS))
    producer.config = SimpleNamespace(
        schedule_interval_s=0.001,
        state_batch_size=1,
        num_samples_per_prompt=1,
        pressure_admission_mode=mode,
        kv_cache_usage_threshold=threshold,
    )
    producer.llm_server_state_tracker = tracker
    producer._metrics = {
        "dispatcher/prompt_batches_dispatched": 0,
        "dispatcher/llm_server_pressure_skip_count": 0,
    }
    return producer


async def _one_loop_tick(producer: RolloutDispatcher) -> None:
    original_sleep = asyncio.sleep

    async def _sleep(interval: float):
        if interval == producer.config.schedule_interval_s:
            raise asyncio.CancelledError
        return await original_sleep(interval)

    # Patch only the dispatcher cadence sleep so awaited tracker futures still resolve.
    asyncio.sleep = _sleep  # type: ignore[method-assign]
    try:
        with pytest.raises(asyncio.CancelledError):
            await producer._loop()
    finally:
        asyncio.sleep = original_sleep  # type: ignore[method-assign]


def test_pressure_open_when_mode_is_off():
    tracker = _InProcessTracker({"s0": {"kv_cache_usage": 0.95}})
    producer = _dispatcher(mode="off", tracker=tracker)

    assert asyncio.run(producer._llm_server_pressure_open()) is True
    assert tracker.calls == 0


def test_pressure_open_when_tracker_is_missing():
    producer = _dispatcher(mode="enforce", tracker=None)

    assert asyncio.run(producer._llm_server_pressure_open()) is True


def test_pressure_open_uses_in_process_tracker():
    tracker = _InProcessTracker(
        {
            "s0": {"kv_cache_usage": 0.90},
            "s1": {"kv_cache_usage": 0.50},
        }
    )
    producer = _dispatcher(mode="enforce", tracker=tracker, threshold=0.75)

    assert asyncio.run(producer._llm_server_pressure_open()) is True
    assert tracker.calls == 1


def test_pressure_closed_uses_actor_handle_tracker():
    tracker = _ActorHandleTracker(
        {
            "s0": {"kv_cache_usage": 0.80},
            "s1": {"kv_cache_usage": 0.99},
        }
    )
    producer = _dispatcher(mode="enforce", tracker=tracker, threshold=0.80)

    assert asyncio.run(producer._llm_server_pressure_open()) is False
    assert tracker.get_routing_snapshot.calls == 1


def test_empty_tracker_snapshot_allows_dispatch():
    producer = _dispatcher(mode="enforce", tracker=LLMServerStateTracker())

    assert asyncio.run(producer._llm_server_pressure_open()) is True


def test_pressure_recovers_after_kv_drops_below_threshold():
    tracker = _InProcessTracker({"s0": {"kv_cache_usage": 0.95}})
    producer = _dispatcher(mode="enforce", tracker=tracker)

    assert asyncio.run(producer._llm_server_pressure_open()) is False
    tracker.snapshot = {"s0": {"kv_cache_usage": 0.79}}
    assert asyncio.run(producer._llm_server_pressure_open()) is True


def test_loop_skips_dispatch_when_every_replica_is_full():
    tracker = _InProcessTracker(
        {
            "s0": {"kv_cache_usage": 0.80},
            "s1": {"kv_cache_usage": 0.99},
        }
    )
    producer = _dispatcher(mode="enforce", tracker=tracker)
    consumed = []

    async def _dispatch() -> None:
        consumed.append("batch")

    producer._dispatch_next_batch = _dispatch  # type: ignore[method-assign]
    asyncio.run(_one_loop_tick(producer))

    assert consumed == []
    assert producer._metrics["dispatcher/llm_server_pressure_skip_count"] == 1


def test_loop_dispatches_when_any_replica_is_below_threshold():
    tracker = _InProcessTracker(
        {
            "s0": {"kv_cache_usage": 0.80},
            "s1": {"kv_cache_usage": 0.79},
        }
    )
    producer = _dispatcher(mode="enforce", tracker=tracker)
    consumed = []

    async def _dispatch() -> None:
        consumed.append("batch")

    producer._dispatch_next_batch = _dispatch  # type: ignore[method-assign]
    asyncio.run(_one_loop_tick(producer))

    assert consumed == ["batch"]
    assert producer._metrics["dispatcher/llm_server_pressure_skip_count"] == 0


def test_dispatcher_collects_optional_rollout_efficiency():
    class Adapter:
        async def get_efficiency_metrics(self):
            return {"model/output_tokens": 123.0, "tool/total_s": 7.0}

    producer = _dispatcher(mode="off", tracker=None)
    producer.agent_loop_manager = Adapter()
    metrics = asyncio.run(producer.get_metrics())
    assert metrics["efficiency/model/output_tokens"] == 123.0
    assert metrics["efficiency/tool/total_s"] == 7.0
    assert metrics["dispatcher/prompt_batches_dispatched"] == 0
    producer.agent_loop_manager = object()
    assert asyncio.run(producer.get_metrics()) == producer._metrics
