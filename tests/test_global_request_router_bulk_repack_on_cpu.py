# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from laminar.global_request_router import GlobalRequestRouter, GlobalRequestRouterConfig, RouteUnavailableError
from laminar.global_request_router.router import ACTIVE, DRAINING, UPDATING
from laminar.multi_version_rollout import CommittedAbortSource, CommittedRepackGroup


class _FakeServer:
    def __init__(self) -> None:
        self.abort_calls: list[dict[str, bool]] = []
        self.abort_all_requests = SimpleNamespace(remote=self._abort)

    async def _abort(self, **kwargs: bool) -> None:
        self.abort_calls.append(kwargs)


def _snapshot(
    server_id: str,
    *,
    kv: float = 0.1,
    waiting: int = 0,
    running: int = 0,
    health: str = "healthy",
    version: int = 1,
) -> dict[str, object]:
    return {
        "server_id": server_id,
        "backend_health": health,
        "max_num_seqs": 8,
        "kv_cache_usage": kv,
        "num_waiting_requests": waiting,
        "num_running_requests": running,
        "token_counts": {},
        "snapshot_version": version,
        "timestamp": float(version),
    }


def _router(versions: dict[str, int], *, continuation: str = "same_version") -> GlobalRequestRouter:
    router = GlobalRequestRouter(
        servers={server_id: _FakeServer() for server_id in versions},
        config=GlobalRequestRouterConfig(continuation=continuation),
    )
    for server_id, version in versions.items():
        router.mark_active(server_id, weight_version=version)
    return router


def _bind_to(
    router: GlobalRequestRouter,
    server_ids: dict[str, int],
    *,
    session_id: str,
    server_id: str,
    weight_version: int | None,
    snapshot_version: int,
) -> None:
    router.update_routing_snapshot(
        {
            candidate: _snapshot(candidate, kv=0.0 if candidate == server_id else 0.9, version=snapshot_version)
            for candidate in server_ids
        }
    )
    router.bind_route(session_id, min_weight_version=weight_version)
    assert router._session_id_to_server[session_id].server_id == server_id


def test_bulk_repack_preserves_all_version_routes_and_rechecks_after_release() -> None:
    versions = {"v3-source": 3, "idle": 1, "v2-destination": 2, "v3-destination": 3, "v2-source": 1}
    router = _router(versions)
    _bind_to(router, versions, session_id="old-route", server_id="v2-source", weight_version=1, snapshot_version=1)
    # Recovery can retain a previous-version binding while its owner loads new weights.
    router.begin_recovery("v2-source")
    router.finish_recovery("v2-source", _FakeServer(), 2)
    for index in range(2):
        _bind_to(
            router,
            versions,
            session_id=f"v3-route-{index}",
            server_id="v3-source",
            weight_version=3,
            snapshot_version=index + 2,
        )
    router.update_routing_snapshot(
        {
            "v3-source": _snapshot("v3-source", kv=0.2, version=4),
            "idle": _snapshot("idle", kv=0.99, version=4),
            "v2-destination": _snapshot("v2-destination", kv=0.65, version=4),
            "v3-destination": _snapshot("v3-destination", kv=0.6, version=4),
            "v2-source": _snapshot("v2-source", kv=0.1, version=4),
        }
    )

    plan = router.plan_and_begin_repack(loadable_version=4)

    assert plan.groups == (
        CommittedRepackGroup(sources=("v2-source",), destination="v2-destination", weight_version=2),
        CommittedRepackGroup(sources=("v3-source",), destination="v3-destination", weight_version=3),
    )
    assert plan.idle_sources == ("idle",)
    assert plan.abort_sources == ()
    assert {
        session_id: (binding.server_id, binding.weight_version)
        for session_id, binding in router._session_id_to_server.items()
    } == {
        "old-route": ("v2-destination", 1),
        "v3-route-0": ("v3-destination", 3),
        "v3-route-1": ("v3-destination", 3),
    }
    assert {server_id: state.lifecycle for server_id, state in router._server_states.items()} == {
        "v3-source": DRAINING,
        "idle": UPDATING,
        "v2-destination": ACTIVE,
        "v3-destination": ACTIVE,
        "v2-source": DRAINING,
    }
    assert router.acquire_lease("v3-route-0", lease_id="request", weight_version=3)[0] == "v3-destination"
    router.release_lease("v3-route-0", "request")

    # The v1 binding still occupies its v2 destination on the next bulk decision.
    assert router.plan_and_begin_repack(loadable_version=4, repack_enabled=False).idle_sources == ()
    with pytest.raises(RuntimeError, match="version conflicts"):
        router.release_route("old-route", weight_version=2)
    router.release_route("old-route", weight_version=1)
    next_plan = router.plan_and_begin_repack(loadable_version=4, repack_enabled=False)
    assert next_plan.groups == ()
    assert next_plan.idle_sources == ("v2-destination",)
    assert router._server_states["v2-destination"].lifecycle == UPDATING
    assert router._server_states["v3-destination"].lifecycle == ACTIVE


@pytest.mark.parametrize("continuation", ["same_version", "latest_version"])
def test_bulk_idle_gates_and_latest_route_retirement(continuation: str) -> None:
    versions = dict.fromkeys(
        (
            "route-only",
            "inflight-only",
            "queued",
            "running",
            "idle",
            "excluded",
            "current",
            "unavailable",
            "recovering",
        ),
        1,
    )
    router = _router(versions, continuation=continuation)
    requested_version = None if continuation == "latest_version" else 1
    route_owners = ("route-only", "inflight-only", "excluded", "current", "unavailable", "recovering")
    for index, server_id in enumerate(route_owners):
        _bind_to(
            router,
            versions,
            session_id=server_id,
            server_id=server_id,
            weight_version=requested_version,
            snapshot_version=index + 1,
        )
    assert (
        router.acquire_lease("inflight-only", lease_id="running", weight_version=requested_version)[0]
        == "inflight-only"
    )
    router.mark_active("current", weight_version=2)
    router.begin_recovery("recovering")
    router.update_routing_snapshot(
        {
            server_id: _snapshot(
                server_id,
                waiting=int(server_id == "queued"),
                running=int(server_id == "running"),
                health="unavailable" if server_id == "unavailable" else "healthy",
                version=7,
            )
            for server_id in versions
        }
    )

    plan = router.plan_and_begin_repack(loadable_version=2, excluded_servers=["excluded"], repack_enabled=False)

    assert plan.groups == ()
    assert plan.idle_sources == ("idle",)
    busy = ("route-only", "inflight-only", "queued", "running")
    latest = continuation == "latest_version"
    assert plan.abort_sources == (tuple(CommittedAbortSource(server_id, 1) for server_id in busy) if latest else ())
    assert {server_id: state.lifecycle for server_id, state in router._server_states.items()} == {
        **{server_id: DRAINING if latest else ACTIVE for server_id in busy},
        "idle": UPDATING,
        "excluded": ACTIVE,
        "current": ACTIVE,
        "unavailable": ACTIVE,
        "recovering": UPDATING,
    }
    remaining_routes = {"excluded", "current", "unavailable", "recovering"}
    assert set(router._session_id_to_server) == (
        remaining_routes if latest else remaining_routes | {"route-only", "inflight-only"}
    )
    assert all(binding.weight_version == 1 for binding in router._session_id_to_server.values())
    assert router._inflight_count("inflight-only") == 1
    if latest:
        with pytest.raises(RouteUnavailableError):
            router.try_mark_updating("inflight-only", expected_version=1)
        assert router._server_states["inflight-only"].handle.abort_calls == []
        router.release_lease("inflight-only", "running")
        router.try_mark_updating("inflight-only", expected_version=1)
        assert router._server_states["inflight-only"].lifecycle == UPDATING
