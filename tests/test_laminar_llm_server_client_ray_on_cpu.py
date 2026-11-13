# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import ray
from verl.workers.rollout.replica import TokenOutput

from laminar.llm_server_client import LLMServerClient

_RAY_ACK_TIMEOUT_S = 10


async def _wait(awaitable):
    return await asyncio.wait_for(asyncio.shield(awaitable), timeout=_RAY_ACK_TIMEOUT_S)


@ray.remote(max_concurrency=16)
class _ControlledServer:
    def __init__(self, block: bool):
        self.started = asyncio.Event()
        self.allowed = asyncio.Event()
        if not block:
            self.allowed.set()
        self.calls = 0
        self.completed = 0

    async def generate(self, **_kwargs):
        self.calls += 1
        self.started.set()
        await self.allowed.wait()
        self.completed += 1
        return TokenOutput(token_ids=[7], stop_reason="completed", extra_fields={"global_steps": 3})

    async def wait_started(self):
        await self.started.wait()

    def allow(self):
        self.allowed.set()

    def snapshot(self):
        return self.calls, self.completed


@ray.remote(max_concurrency=16)
class _ControlledRouter:
    def __init__(self, server, *, block_acquire: bool, block_reply: bool, block_release: bool):
        self.server = server
        self.acquire_started = asyncio.Event()
        self.acquire_allowed = asyncio.Event()
        self.committed = asyncio.Event()
        self.reply_allowed = asyncio.Event()
        self.release_started = asyncio.Event()
        self.release_allowed = asyncio.Event()
        if not block_acquire:
            self.acquire_allowed.set()
        if not block_reply:
            self.reply_allowed.set()
        if not block_release:
            self.release_allowed.set()
        self.leases = set()
        self.release_count = 0

    async def acquire_lease(self, session_id, lease_id, weight_version=None):
        self.acquire_started.set()
        await self.acquire_allowed.wait()
        self.leases.add((session_id, lease_id))
        self.committed.set()
        await self.reply_allowed.wait()
        return "server-0", self.server

    async def release_lease(self, session_id, lease_id):
        self.release_started.set()
        await self.release_allowed.wait()
        self.leases.discard((session_id, lease_id))
        self.release_count += 1

    async def wait_acquire_started(self):
        await self.acquire_started.wait()

    def allow_acquire(self):
        self.acquire_allowed.set()

    async def wait_committed(self):
        await self.committed.wait()

    def allow_reply(self):
        self.reply_allowed.set()

    async def wait_release_started(self):
        await self.release_started.wait()

    def allow_release(self):
        self.release_allowed.set()

    def snapshot(self):
        return {"leases": len(self.leases), "release_count": self.release_count}


@pytest.fixture(scope="module")
def ray_runtime():
    owned = not ray.is_initialized()
    ray.init(
        address="local",
        num_cpus=2,
        include_dashboard=False,
        _temp_dir=tempfile.mkdtemp(prefix="client-router-cpu-"),
        object_store_memory=128 * 1024 * 1024,
        runtime_env={
            "env_vars": {
                "PYTHONPATH": str(Path(__file__).parent) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                "OMP_NUM_THREADS": "1",
            }
        },
    )
    yield
    if owned:
        ray.shutdown()


def _task(router):
    client = LLMServerClient(config=SimpleNamespace(), router=router)
    return asyncio.create_task(
        client.generate(
            session_id="session",
            weight_version=3,
            prompt_ids=[1],
            sampling_params={},
        )
    )


@pytest.mark.asyncio
async def test_cancelled_queued_acquire_settles_then_releases(ray_runtime):
    server = _ControlledServer.remote(False)
    router = _ControlledRouter.remote(
        server,
        block_acquire=True,
        block_reply=False,
        block_release=False,
    )
    task = _task(router)
    await _wait(router.wait_acquire_started.remote())
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    await _wait(router.allow_acquire.remote())
    with pytest.raises(asyncio.CancelledError):
        await _wait(task)
    assert await _wait(router.snapshot.remote()) == {"leases": 0, "release_count": 1}
    assert await _wait(server.snapshot.remote()) == (0, 0)


@pytest.mark.asyncio
async def test_cancel_after_commit_before_reply_releases_the_lease(ray_runtime):
    server = _ControlledServer.remote(False)
    router = _ControlledRouter.remote(
        server,
        block_acquire=False,
        block_reply=True,
        block_release=False,
    )
    task = _task(router)
    await _wait(router.wait_committed.remote())
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    await _wait(router.allow_reply.remote())
    with pytest.raises(asyncio.CancelledError):
        await _wait(task)
    assert await _wait(router.snapshot.remote()) == {"leases": 0, "release_count": 1}
    assert await _wait(server.snapshot.remote()) == (0, 0)


@pytest.mark.asyncio
async def test_backend_cancel_settles_generation_before_release(ray_runtime):
    server = _ControlledServer.remote(True)
    router = _ControlledRouter.remote(
        server,
        block_acquire=False,
        block_reply=False,
        block_release=False,
    )
    task = _task(router)
    await _wait(server.wait_started.remote())
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert (await _wait(router.snapshot.remote()))["release_count"] == 0
    await _wait(server.allow.remote())
    with pytest.raises(asyncio.CancelledError):
        await _wait(task)
    assert await _wait(server.snapshot.remote()) == (1, 1)
    assert await _wait(router.snapshot.remote()) == {"leases": 0, "release_count": 1}


@pytest.mark.asyncio
async def test_release_cancel_settles_ack_before_propagating(ray_runtime):
    server = _ControlledServer.remote(False)
    router = _ControlledRouter.remote(
        server,
        block_acquire=False,
        block_reply=False,
        block_release=True,
    )
    task = _task(router)
    await _wait(router.wait_release_started.remote())
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    await _wait(router.allow_release.remote())
    with pytest.raises(asyncio.CancelledError):
        await _wait(task)
    assert await _wait(router.snapshot.remote()) == {"leases": 0, "release_count": 1}
