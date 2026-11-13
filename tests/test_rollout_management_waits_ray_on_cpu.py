# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import tempfile
from pathlib import Path

import pytest
import ray

from laminar.global_request_router import GlobalRequestRouter, GlobalRequestRouterConfig
from laminar.multi_version_rollout import MultiVersionRolloutManager, MultiVersionRolloutManagerConfig


@ray.remote(max_concurrency=4)
class _PausedAbortServer:
    def __init__(self):
        self.started = asyncio.Event()
        self.allowed = asyncio.Event()

    async def abort_all_requests(self, *, abort_only):
        assert abort_only
        self.started.set()
        await self.allowed.wait()

    async def wait_for_abort(self):
        await self.started.wait()

    def allow_abort(self):
        self.allowed.set()


class _InspectableRouter(GlobalRequestRouter):
    def lifecycle(self, server_id):
        return self._server_states[server_id].lifecycle


@pytest.mark.asyncio
async def test_manager_backend_abort_wait_does_not_occupy_router_rpc_capacity():
    assert not ray.is_initialized()
    ray.init(
        address="local",
        num_cpus=2,
        include_dashboard=False,
        _temp_dir=tempfile.mkdtemp(prefix="rollout-management-"),
        object_store_memory=128 * 1024 * 1024,
        runtime_env={
            "env_vars": {
                "PYTHONPATH": str(Path(__file__).parent) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        },
    )
    pending = None
    try:
        server = _PausedAbortServer.remote()
        router = ray.remote(_InspectableRouter).remote(
            servers={"s0": server}, config=GlobalRequestRouterConfig(continuation="latest_version")
        )
        await router.mark_active.remote("s0", weight_version=1)
        await router.bind_route.remote("active")
        await router.acquire_lease.remote("active", lease_id="running")
        await router.begin_colocate_pause.remote()
        manager = MultiVersionRolloutManager(MultiVersionRolloutManagerConfig())
        manager.connect(servers={"s0": object()}, router=router, relay_client=object())
        pending = asyncio.create_task(manager._wait_for_source_drain("s0", expected_version=1))
        await asyncio.wait_for(server.wait_for_abort.remote(), timeout=5)
        assert not pending.done()
        await asyncio.wait_for(router.release_lease.remote("active", "running"), timeout=1)
        assert await asyncio.wait_for(router.lifecycle.remote("s0"), timeout=1) == "draining"
        await server.allow_abort.remote()
        await asyncio.wait_for(pending, timeout=2)
        assert await router.lifecycle.remote("s0") == "updating"
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        ray.shutdown()
