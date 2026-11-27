# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from laminar.eval.router import ValidationRequestRouter
from laminar.global_request_router import GlobalRequestRouter, GlobalRequestRouterConfig, RouteUnavailableError
from laminar.llm_server_client import LLMServerClient
from laminar.train import dispatcher as dispatcher_module
from laminar.train.dispatcher import RolloutDispatcher


def _router(
    versions: dict[str, int],
    *,
    continuation: str = "same_version",
    max_sessions: int = 2,
    overload_mode: str = "off",
) -> GlobalRequestRouter:
    router = GlobalRequestRouter(
        servers={sid: SimpleNamespace(server_id=sid) for sid in versions},
        config=GlobalRequestRouterConfig(
            continuation=continuation,
            dynamic_overload_admission={"mode": overload_mode, "kv_cache_usage_threshold": 0.80},
            replica_concurrency={"max_sessions": max_sessions},
        ),
    )
    for sid, version in versions.items():
        router.mark_active(sid, weight_version=version)
    router.update_routing_snapshot(
        {sid: {"backend_health": "healthy", "kv_cache_usage": 0.05, "snapshot_version": 1} for sid in versions}
    )
    return router


def test_max_sessions_admission_burst_balance_and_validation_isolation() -> None:
    with pytest.raises(ValueError, match="replica_concurrency"):
        GlobalRequestRouterConfig(replica_concurrency={"max_sessions": 0})
    with pytest.raises(ValueError, match="unknown replica_concurrency"):
        GlobalRequestRouterConfig(replica_concurrency={"extra": 1})

    router = _router({"s0": 1, "s1": 2}, max_sessions=2)
    with pytest.raises(RouteUnavailableError) as missing_ver:
        router.get_routable_version(3)
    assert missing_ver.value.capacity_full is False

    assert [router.bind_route("a", min_weight_version=1), router.bind_route("b", min_weight_version=1)] == [1, 1]
    assert router.bind_route("a", min_weight_version=1) == 1
    assert router.get_routable_version(1) == 2
    assert [router.bind_route("c", min_weight_version=1), router.bind_route("d", min_weight_version=1)] == [2, 2]
    with pytest.raises(RouteUnavailableError) as full_ver:
        router.get_routable_version(1)
    assert full_ver.value.capacity_full is True
    with pytest.raises(RouteUnavailableError) as full_bind:
        router.bind_route("overflow", min_weight_version=1)
    assert full_bind.value.capacity_full is True

    router.release_route("a", weight_version=1)
    assert router.bind_route("overflow", min_weight_version=1) == 1

    # Same-version Repack destination is intentionally unconstrained by admission max_sessions.
    repack_router = _router({"s0": 1, "s1": 1}, max_sessions=1)
    repack_router.bind_route("x0", min_weight_version=1)
    repack_router.bind_route("x1", min_weight_version=1)
    assert len(repack_router.plan_and_begin_repack(loadable_version=2).groups) == 1

    # Bursts between KV tracker polls spread evenly by slot occupancy instead of herding onto stale kv=0.0.
    balanced = _router({"s0": 1, "s1": 1, "s2": 1}, max_sessions=6)
    balanced.update_routing_snapshot(
        {
            "s0": {"backend_health": "healthy", "kv_cache_usage": 0.0, "snapshot_version": 2},
            "s1": {"backend_health": "healthy", "kv_cache_usage": 0.01, "snapshot_version": 2},
            "s2": {"backend_health": "healthy", "kv_cache_usage": 0.70, "snapshot_version": 2},
        }
    )
    for idx in range(6):
        balanced.bind_route(f"burst-{idx}", min_weight_version=1)
    assert {sid: balanced._count_session_routes(sid) for sid in ("s0", "s1", "s2")} == {"s0": 3, "s1": 3, "s2": 0}

    # ValidationRequestRouter isolates saturated checkpoint group 1 without spilling onto group 2.
    val_router = ValidationRequestRouter(
        servers={"g1": object(), "g2": object()},
        config=GlobalRequestRouterConfig(replica_concurrency={"max_sessions": 1}),
    )
    for sid in ("g1", "g2"):
        val_router.mark_active(sid, weight_version=0)
    val_router.update_routing_snapshot(
        {sid: {"backend_health": "healthy", "kv_cache_usage": 0.05, "snapshot_version": 1} for sid in ("g1", "g2")}
    )
    val_router.set_validation_versions([1, 2])
    with pytest.raises(RouteUnavailableError) as unloaded:
        val_router.get_routable_version(1)
    assert unloaded.value.capacity_full is False
    val_router.mark_active("g1", weight_version=1)
    val_router.mark_active("g2", weight_version=2)
    assert val_router.bind_route("v1-ok", min_weight_version=1) == 1
    with pytest.raises(RouteUnavailableError) as g1_full:
        val_router.get_routable_version(1)
    assert g1_full.value.capacity_full is True
    with pytest.raises(RouteUnavailableError) as g1_bind_full:
        val_router.bind_route("v1-overflow", min_weight_version=1)
    assert g1_bind_full.value.capacity_full is True
    assert val_router.get_routable_version(2) == 2
    assert val_router.bind_route("v2-ok", min_weight_version=2) == 2


def test_latest_version_rebind_enforces_per_replica_cap() -> None:
    # 1) dynamic_overload_admission=enforce with skewed KV snapshots must not pile all rebinds onto s0.
    router = _router({"s0": 1, "s1": 1}, continuation="latest_version", max_sessions=1, overload_mode="enforce")
    router.bind_route("s-0", min_weight_version=1)
    router.bind_route("s-1", min_weight_version=1)
    router.begin_colocate_pause()
    router.mark_active("s0", weight_version=2)
    router.mark_active("s1", weight_version=2)
    router.update_routing_snapshot(
        {
            "s0": {"backend_health": "healthy", "kv_cache_usage": 0.10, "snapshot_version": 2},
            "s1": {"backend_health": "healthy", "kv_cache_usage": 0.95, "snapshot_version": 2},
        }
    )
    with pytest.raises(RouteUnavailableError) as full_err:
        router.bind_route("newcomer", min_weight_version=2)
    assert full_err.value.capacity_full is True
    assert {router.acquire_lease("s-0", lease_id="r0")[0], router.acquire_lease("s-1", lease_id="r1")[0]} == {
        "s0",
        "s1",
    }

    # 2) Single-replica failover / newer-version recovery must not exceed max_sessions on surviving replica.
    router.release_lease("s-0", "r0")
    router.release_lease("s-1", "r1")
    router.begin_recovery("s1")
    first_owner = "s-0" if router._sessions["s-0"].route.server_id == "s0" else "s-1"
    failover_sess = "s-1" if first_owner == "s-0" else "s-0"
    assert router.acquire_lease(first_owner, lease_id="keep")[0] == "s0"
    with pytest.raises(RouteUnavailableError) as failover_full:
        router.acquire_lease(failover_sess, lease_id="blocked")
    assert failover_full.value.capacity_full is True

    router.finish_recovery("s1", SimpleNamespace(server_id="s1"), weight_version=3)
    router.update_routing_snapshot(
        {sid: {"backend_health": "healthy", "kv_cache_usage": 0.10, "snapshot_version": 3} for sid in ("s0", "s1")}
    )
    assert router.acquire_lease(failover_sess, lease_id="v3-slot")[0] == "s1"
    with pytest.raises(RouteUnavailableError) as v3_full:
        router.acquire_lease(first_owner, lease_id="v3-overflow")
    assert v3_full.value.capacity_full is True


@pytest.mark.asyncio
async def test_capacity_wait_decouples_from_readiness_timeouts(monkeypatch) -> None:
    router = _router({"s0": 1}, max_sessions=1)
    router.bind_route("holder", min_weight_version=1)

    async def remote_bind(**kwargs):
        return router.bind_route(**kwargs)

    cfg = GlobalRequestRouterConfig(
        client_wait={
            "timeout_s": 0.015,
            "planned_pause_timeout_s": 5.0,
            "initial_backoff_s": 0.005,
            "max_backoff_s": 0.005,
        },
        replica_concurrency={"max_sessions": 1},
    )
    client = LLMServerClient(
        SimpleNamespace(rollout_router=cfg), SimpleNamespace(bind_route=SimpleNamespace(remote=remote_bind))
    )
    waiter = asyncio.create_task(client.bind_route("queued", weight_version=1))
    await asyncio.sleep(0.04)  # Exceeds client_wait.timeout_s (0.015s) without expiring on capacity_full=True.
    assert not waiter.done()
    router.release_route("holder", weight_version=1)
    assert await asyncio.wait_for(waiter, timeout=1.0) == 1

    # Post-pause capacity_full resets the 5.0s pause grace back to normal 0.015s timeout_s on a subsequent failure.
    seq = iter(
        [
            RouteUnavailableError("paused", planned_pause=True),
            RouteUnavailableError("resumed-full", capacity_full=True),
        ]
    )

    async def remote_seq(**_kwargs):
        raise next(seq, RouteUnavailableError("dead", capacity_full=False))

    seq_client = LLMServerClient(
        SimpleNamespace(rollout_router=cfg), SimpleNamespace(bind_route=SimpleNamespace(remote=remote_seq))
    )
    with pytest.raises(TimeoutError, match="Router availability wait expired"):
        await asyncio.wait_for(seq_client.bind_route("after-pause", weight_version=1), timeout=0.3)

    # RolloutDispatcher likewise waits past _VERSION_WAIT_TIMEOUT_S on capacity_full=True and resets pause grace.
    monkeypatch.setattr(dispatcher_module, "_VERSION_WAIT_TIMEOUT_S", 0.015)
    monkeypatch.setattr(dispatcher_module, "_PLANNED_PAUSE_WAIT_TIMEOUT_S", 5.0)
    monkeypatch.setattr(dispatcher_module, "_VERSION_POLL_INTERVAL_S", 0.005)
    dispatcher = object.__new__(RolloutDispatcher)
    dispatcher.relay_client = SimpleNamespace(get_trainer_version=AsyncMock(return_value=1))
    slot_open = False

    async def get_routable_version(*, weight_version: int) -> int:
        if not slot_open:
            raise RouteUnavailableError("full", capacity_full=True)
        return weight_version

    dispatcher.rollout_router = SimpleNamespace(get_routable_version=SimpleNamespace(remote=get_routable_version))
    disp_task = asyncio.create_task(dispatcher._wait_for_available_version())
    await asyncio.sleep(0.04)
    assert not disp_task.done()
    slot_open = True
    assert await asyncio.wait_for(disp_task, timeout=1.0) == (1, 1)

    disp_seq = iter(
        [
            RouteUnavailableError("paused", planned_pause=True),
            RouteUnavailableError("resumed-full", capacity_full=True),
        ]
    )
    dispatcher.rollout_router = SimpleNamespace(
        get_routable_version=SimpleNamespace(
            remote=AsyncMock(
                side_effect=lambda **_kw: (_ for _ in ()).throw(next(disp_seq, RouteUnavailableError("dead")))
            )
        )
    )
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(dispatcher._wait_for_available_version(), timeout=0.3)
