# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Exercise Relay's real publish/read fences on two CPU-only Ray actors."""

import asyncio
import inspect
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import ray
import torch

from laminar.train.relay.manager import RelayManager, RelayWorker, _RelayHopStats
from laminar.utils.concurrency import ReadWriteLock
from laminar.utils.tensor import TensorMeta


class _CpuFenceRelayWorker(RelayWorker):
    """Keep real lock/version methods, replacing CUDA storage and UCX transport."""

    def __init__(self, *, delay_first_read: bool):
        self.name = "cpu-relay"
        self._publish_lock = ReadWriteLock()
        self._active_version = None
        self.full_tensor_keys = ["weight"]
        self.server = SimpleNamespace(buffer={"weight": SimpleNamespace(version=7)})
        self._delay_first_read = delay_first_read
        self._first_read_acquired = asyncio.Event()
        self._first_write_acquired = asyncio.Event()

    async def acquire_full_read_if_publish(self, *, version: int) -> bool:
        if self._delay_first_read and version == 7:
            # Force opposite lock acquisition order on the second worker. This
            # simulates async actor reordering; it does not assert its frequency.
            await self._first_write_acquired.wait()
        acquired = super().acquire_full_read_if_publish(version=version)
        if inspect.isawaitable(acquired):
            # Also exercise the former async method when checking this test
            # against the blocking implementation: fail on the lock cycle.
            acquired = await acquired
        if acquired:
            self._first_read_acquired.set()
        return acquired

    async def begin_publish(self, *, version: int) -> None:
        await super().begin_publish(version=version)
        self._first_write_acquired.set()

    async def wait_for_first_read(self) -> None:
        await self._first_read_acquired.wait()

    async def wait_for_queued_writer(self) -> None:
        while not self._publish_lock._waiting_writers:
            await asyncio.sleep(0.001)

    def configure_destination(self, destination) -> None:
        self.dst_worker = destination

    async def wait_for_full_state_dict(self, *, version: int) -> None:
        self.receive_version(version=version)

    def receive_version(self, *, version: int) -> None:
        self._require_active_publish(version=version)
        self.server.buffer["weight"].version = version

    async def send_full_state_dict_to_dst_worker(self, *, wait_for_version: int) -> _RelayHopStats:
        self._require_active_publish(version=wait_for_version)
        await self.dst_worker.receive_version.remote(version=wait_for_version)
        return _RelayHopStats(logical_payload_bytes=8, elapsed_s=0.001)


def _manager_for_workers(workers) -> RelayManager:
    manager = object.__new__(RelayManager)
    manager.workers = workers
    manager.node_id2worker = {f"node-{index}": worker for index, worker in enumerate(workers)}
    manager.trainer_version = manager.loadable_version = 7
    manager._active_publish = None
    manager._publish_transition_lock = asyncio.Lock()
    manager.full_state_meta_dict = {"weight": TensorMeta(dtype=torch.float32, shape=torch.Size([2]))}
    manager._metrics = {
        "publish/started": 0,
        "publish/committed": 0,
        "broadcast/started": 0,
        "broadcast/completed": 0,
        "broadcast/failed": 0,
        "broadcast/latency_s": 0.0,
        "broadcast/latency_s/max": 0.0,
    }
    return manager


@pytest.fixture(scope="module")
def local_ray_cluster():
    # Keep Ray's Unix socket path below its platform length limit; pytest's
    # per-test temporary directories can be too deeply nested.
    with tempfile.TemporaryDirectory(prefix="relay-fence-") as temp_dir:
        test_dir = Path(__file__).resolve().parent
        worker_pythonpath = os.pathsep.join(
            [str(test_dir.parent / "src"), str(test_dir), os.environ.get("PYTHONPATH", "")]
        )
        try:
            ray.init(
                address="local",
                num_cpus=2,
                include_dashboard=False,
                object_store_memory=128 * 1024**2,
                _temp_dir=temp_dir,
                # Pytest adds test/source roots only to the driver's sys.path.
                # Bind both explicitly so workers import this exact checkout.
                runtime_env={"env_vars": {"PYTHONPATH": worker_pythonpath}},
            )
            yield
        finally:
            ray.shutdown()


@pytest.mark.parametrize("cancel_acquire", [False, True])
def test_opposite_worker_lock_order_releases_partial_read_and_finishes_publish(cancel_acquire, local_ray_cluster):
    """A read on A and a publish write on B must never form a lock cycle."""
    workers = []
    try:
        worker_cls = ray.remote(_CpuFenceRelayWorker)
        workers = [worker_cls.remote(delay_first_read=delay) for delay in (False, True)]

        async def run():
            manager = _manager_for_workers(workers)
            await workers[0].configure_destination.remote(workers[1])
            read_task = asyncio.create_task(manager._try_acquire_consistent_rollout_read(workers, version=7))
            tasks = [read_task]
            try:
                await asyncio.wait_for(workers[0].wait_for_first_read.remote(), timeout=20)
                if cancel_acquire:
                    read_task.cancel()
                    await asyncio.sleep(0)
                    assert not read_task.done()
                publish_task = asyncio.create_task(manager.begin_publish(version=8))
                tasks.append(publish_task)
                # Unlike wait_for, wait is bounded even if shielded acquisition
                # cannot finish under a regression to blocking worker reads.
                _, pending = await asyncio.wait(tasks, timeout=20)
                assert not pending, "opposite worker lock ordering stalled the read/publish barriers"
                if cancel_acquire:
                    with pytest.raises(asyncio.CancelledError):
                        await read_task
                else:
                    assert await read_task is False
                await publish_task
                await manager.broadcast(version=8)
                await manager.complete_publish(version=8)

                assert await manager.acquire_rollout_version(["node-0", "node-1"], minimum_version=8) == 8
                next_publish = asyncio.create_task(manager.begin_publish(version=9))
                tasks.append(next_publish)
                await asyncio.wait_for(
                    asyncio.gather(*[worker.wait_for_queued_writer.remote() for worker in workers]), timeout=20
                )
                assert not next_publish.done()
                await manager.release_full_read(["node-0", "node-1"])
                await asyncio.wait_for(next_publish, timeout=20)
                await manager.broadcast(version=9)
                await manager.complete_publish(version=9)
                assert manager.trainer_version == manager.loadable_version == 9
                assert manager._active_publish is None
            finally:
                # Terminate only this test's actors to unblock any failed RPCs
                # before draining manager tasks, including shielded operations.
                for worker in workers:
                    ray.kill(worker, no_restart=True)
                await asyncio.gather(*tasks, return_exceptions=True)

        asyncio.run(run())
    finally:
        for worker in workers:
            ray.kill(worker, no_restart=True)
