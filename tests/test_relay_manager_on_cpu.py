# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from laminar.train.relay import RelayBroadcastStats
from laminar.train.relay import manager as relay_manager_module
from laminar.train.relay import storage as relay_storage_module
from laminar.train.relay import transport as relay_transport_module
from laminar.train.relay.manager import (
    EMPTY_VERSION,
    RelayManager,
    RelayWorker,
)
from laminar.train.relay.transport import UCXTensorServer
from laminar.utils.concurrency import ReadWriteLock
from laminar.utils.tensor import TensorMeta


class _RemoteMethod:
    def __init__(self, callback):
        self._callback = callback

    def remote(self, *args, **kwargs):  # noqa: ANN201
        return self._callback(*args, **kwargs)


def _hop_stats(logical_payload_bytes: int, elapsed_s: float):  # noqa: ANN202
    return relay_manager_module._RelayHopStats(
        logical_payload_bytes=logical_payload_bytes,
        elapsed_s=elapsed_s,
    )


def _fake_relay_destination(address: tuple[str, int]):
    async def get_server_ip_port():
        return address

    return SimpleNamespace(get_server_ip_port=_RemoteMethod(get_server_ip_port))


class _FakeRelayWorker:
    def __init__(self, *, hop_stats=None, send_error: Exception | None = None):
        self.hop_stats = hop_stats
        self.send_error = send_error
        self.active_version: int | None = None
        self.physical_version = EMPTY_VERSION
        self.events: list[tuple[str, int]] = []
        self.acquire_calls: list[int] = []
        self.read_locks = 0
        self.begin_publish = _RemoteMethod(self._begin_publish)
        self.complete_publish = _RemoteMethod(self._complete_publish)
        self.wait_for_full_state_dict = _RemoteMethod(self._wait_for_full_state_dict)
        self.send_full_state_dict_to_dst_worker = _RemoteMethod(self._send_full_state_dict_to_dst_worker)
        self.acquire_full_read_if_publish = _RemoteMethod(self._acquire_full_read_if_publish)
        self.release_full_read = _RemoteMethod(self._release_full_read)

    async def _begin_publish(self, *, version: int) -> None:
        assert self.active_version is None
        self.active_version = version
        self.events.append(("begin", version))

    async def _complete_publish(self, *, version: int) -> None:
        assert self.active_version == version
        self.physical_version = version
        self.active_version = None
        self.events.append(("complete", version))

    async def _wait_for_full_state_dict(self, *, version: int) -> None:
        assert self.active_version == version

    async def _send_full_state_dict_to_dst_worker(
        self,
        *,
        wait_for_version: int,
    ) -> relay_manager_module._RelayHopStats:
        assert self.active_version == wait_for_version
        self.events.append(("send", wait_for_version))
        if self.send_error is not None:
            raise self.send_error
        assert self.hop_stats is not None
        self.physical_version = wait_for_version
        return self.hop_stats

    async def _acquire_full_read_if_publish(self, *, version: int) -> bool:
        self.acquire_calls.append(version)
        if self.physical_version != version:
            return False
        self.read_locks += 1
        return True

    async def _release_full_read(self) -> None:
        assert self.read_locks > 0
        self.read_locks -= 1


def _make_lightweight_relay_manager(*workers: _FakeRelayWorker) -> RelayManager:
    from laminar.metrics import MetricSource, cumulative_counter_specs
    from laminar.train.relay.manager import _RELAY_COUNTER_KEYS

    manager = object.__new__(RelayManager)
    manager._metric_source = MetricSource(cumulative_counter_specs("relay", _RELAY_COUNTER_KEYS))
    manager.trainer_version = EMPTY_VERSION
    manager._active_publish = None
    manager._publish_transition_lock = asyncio.Lock()
    manager.loadable_version = EMPTY_VERSION
    manager.full_state_meta_dict = {}
    manager.workers = list(workers)
    manager.node_id2worker = {f"node-{index}": worker for index, worker in enumerate(workers)}
    manager._metrics = {
        "publish/started": 0,
        "publish/committed": 0,
        "broadcast/started": 0,
        "broadcast/completed": 0,
        "broadcast/failed": 0,
        "broadcast/latency_s": 0.0,
        "broadcast/latency_s/last": 0.0,
        "broadcast/latency_s/max": 0.0,
        "broadcast/tensor_count/last": 0,
        "broadcast/logical_payload_bytes/last": 0,
        "broadcast/end_to_end_goodput_bytes_per_s/last": 0.0,
        "broadcast/bottleneck_hop_index/last": -1,
    }
    return manager


def test_relay_worker_poll_interval_avoids_busy_spin():
    assert RelayWorker.DEFAULT_INTERVAL_S >= 1e-4
    assert RelayWorker.DEFAULT_INTERVAL_S <= 1e-2


def test_ucx_tensor_server_retries_auto_port_when_listener_bind_is_busy(monkeypatch):
    calls: list[int] = []

    def fake_create_listener(_handler, *, port):
        calls.append(port)
        if len(calls) == 1:
            raise RuntimeError("Device is busy")
        return SimpleNamespace(ip="0.0.0.0", port=port)

    ports = iter([(59015, None), (59016, None)])
    monkeypatch.setattr(relay_transport_module.ucxx, "create_listener", fake_create_listener)
    monkeypatch.setattr(relay_transport_module, "get_free_port", lambda _ip: next(ports))

    async def run_test():
        server = object.__new__(UCXTensorServer)
        server.backend = "ucx"
        server.ip = "127.0.0.1"
        server._server_error = None
        server.MAX_LISTENER_BIND_ATTEMPTS = 2
        task = asyncio.create_task(server.run_server())
        server._server_task = task
        try:
            listener = await server.wait_until_ready(interval_s=0)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert listener.port == 59016

    asyncio.run(run_test())
    assert calls == [59015, 59016]


def test_ucx_tensor_server_wait_until_ready_surfaces_startup_error():
    async def run_test():
        server = object.__new__(UCXTensorServer)
        server.backend = "ucx"
        server._server_error = RuntimeError("bind failed")
        server._server_task = asyncio.create_task(asyncio.sleep(60))
        try:
            with pytest.raises(RuntimeError, match="UCX tensor server failed to start") as exc_info:
                await server.wait_until_ready(interval_s=0)
            assert exc_info.value.__cause__ is server._server_error
        finally:
            server._server_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await server._server_task

    asyncio.run(run_test())


def test_relay_worker_uses_fixed_pinned_cpu_backing(monkeypatch):
    captured = {}

    def fake_server(*, buffer_kwargs):
        captured.update(buffer_kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(relay_manager_module, "UCXTensorServer", fake_server)

    RelayWorker("relay-test")

    assert captured == {"device": torch.device("cpu"), "pin_memory": True}


def test_relay_shared_tensor_storage_is_always_cuda_registered(monkeypatch):
    tensor = torch.empty(6, dtype=torch.float32)
    register_calls = []
    filenames = []
    monkeypatch.delenv("VERL_RAY_JOB_ID", raising=False)

    monkeypatch.setattr(
        relay_storage_module.torch,
        "from_file",
        lambda filename, **_kwargs: filenames.append(filename) or tensor,
    )
    monkeypatch.setattr(
        relay_storage_module.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_job_id=lambda: "job-a"),
    )
    monkeypatch.setattr(
        relay_storage_module.torch.cuda,
        "cudart",
        lambda: SimpleNamespace(
            cudaHostRegister=lambda data_ptr, nbytes, flags: register_calls.append((data_ptr, nbytes, flags)) or 0
        ),
    )

    shared_tensor = relay_storage_module.make_shared_tensor_from_meta(
        TensorMeta(dtype=torch.float32, shape=torch.Size([2, 3])),
        key="dense.weight",
    )

    assert shared_tensor.shape == (2, 3)
    assert filenames == ["/dev/shm/job-a-dense.weight"]
    assert register_calls == [(shared_tensor.data_ptr(), shared_tensor.nbytes, 0)]


def test_relay_worker_preserves_publisher_tensor_order(monkeypatch):
    worker = object.__new__(RelayWorker)
    registered = {}
    worker.server = SimpleNamespace(buffer=registered)
    metas = {
        "model.embed_tokens.weight": TensorMeta(dtype=torch.float32, shape=torch.Size([2, 2])),
        "model.layers.0.self_attn.q_proj.weight": TensorMeta(dtype=torch.float32, shape=torch.Size([2, 2])),
        "lm_head.weight": TensorMeta(dtype=torch.float32, shape=torch.Size([2, 2])),
    }

    def fake_make_shared_tensor_from_meta(meta, *, key):
        assert meta is metas[key]
        return torch.empty(meta.shape, dtype=meta.dtype)

    monkeypatch.setattr(relay_manager_module, "make_shared_tensor_from_meta", fake_make_shared_tensor_from_meta)

    worker.init_full_state_dict_in_shm(metas)

    assert worker.full_tensor_keys == list(metas)
    assert list(registered) == list(metas)


def test_relay_worker_ingress_wait_returns_once_every_tensor_has_version():
    worker = object.__new__(RelayWorker)
    worker.name = "relay-test"
    worker.full_tensor_keys = ["dense.weight", "dense.bias"]
    worker._active_version = 7
    worker.DEFAULT_INTERVAL_S = 1e-3
    bias = SimpleNamespace(tensor=torch.zeros(2), version=EMPTY_VERSION)
    worker.server = SimpleNamespace(
        buffer={"dense.weight": SimpleNamespace(tensor=torch.zeros(2), version=7), "dense.bias": bias},
    )

    async def run():
        wait_task = asyncio.create_task(worker.wait_for_full_state_dict(version=7))
        await asyncio.sleep(0.01)
        assert not wait_task.done()
        bias.version = 7
        await asyncio.wait_for(wait_task, timeout=1)

    asyncio.run(run())


def test_rollout_worker_initializes_shared_state_without_mutating_wire_metadata(monkeypatch):
    worker = object.__new__(relay_manager_module.AsyncRLvLLMColocateWorkerExtension)
    wire_metas = {"dense.weight": ["torch.float32", [2, 3]]}

    def fake_make_shared_tensor_from_meta(meta, *, key):
        assert meta == TensorMeta(dtype=torch.float32, shape=torch.Size([2, 3]))
        assert key == "dense.weight"
        return torch.empty(meta.shape, dtype=meta.dtype)

    monkeypatch.setattr(relay_manager_module, "make_shared_tensor_from_meta", fake_make_shared_tensor_from_meta)

    worker.init_shm_state_dict(wire_metas)

    assert wire_metas == {"dense.weight": ["torch.float32", [2, 3]]}
    assert worker.shm_state_dict["dense.weight"].shape == (2, 3)


def test_rollout_worker_requires_shared_state_before_loading_weights():
    worker = object.__new__(relay_manager_module.AsyncRLvLLMColocateWorkerExtension)

    with pytest.raises(RuntimeError, match="shared state must be initialized"):
        worker.load_weights_from_shm_state_dict()


def test_relay_worker_chain_keeps_cpu_tensors_direct(monkeypatch):
    tensors = {
        "dense.bias": torch.arange(2, dtype=torch.float32),
        "dense.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
    }
    worker = object.__new__(RelayWorker)
    worker.name = "relay-test"
    worker.full_tensor_keys = sorted(tensors)
    worker._active_version = 7
    worker.server = SimpleNamespace(
        buffer={key: SimpleNamespace(tensor=tensor, version=7) for key, tensor in tensors.items()},
        tensor_rwlocks=defaultdict(ReadWriteLock),
    )
    endpoint = SimpleNamespace(closed=False)

    async def close_endpoint():
        endpoint.closed = True

    endpoint.close = close_endpoint

    async def fake_create_endpoint(ip, port):
        assert (ip, port) == ("127.0.0.1", 12345)
        return endpoint

    worker.dst_server_address = ("127.0.0.1", 12345)
    headers = []
    payloads = []
    completions = []

    async def fake_send_obj(actual_endpoint, header):
        assert actual_endpoint is endpoint
        headers.append(header)

    async def fake_send_tensor(actual_endpoint, tensor, *, tag=None):
        assert actual_endpoint is endpoint
        payloads.append((tag, tensor))

    async def fake_complete_tensor_transfer(actual_endpoint, *, tensor_count, payload_nbytes):
        assert actual_endpoint is endpoint
        completions.append((tensor_count, payload_nbytes))

    monkeypatch.setattr(relay_manager_module, "send_obj", fake_send_obj)
    monkeypatch.setattr(relay_manager_module, "send_tensor", fake_send_tensor)
    monkeypatch.setattr(relay_manager_module, "complete_tensor_transfer", fake_complete_tensor_transfer)
    monkeypatch.setattr(relay_manager_module, "create_endpoint", fake_create_endpoint)

    stats = asyncio.run(worker.send_full_state_dict_to_dst_worker(wait_for_version=7))

    assert stats.logical_payload_bytes == 32
    assert stats.elapsed_s > 0
    assert len(headers) == 1
    assert [meta.key for meta in headers[0].storage_metas] == worker.full_tensor_keys
    assert [meta.version for meta in headers[0].storage_metas] == [7, 7]
    assert headers[0].mode is relay_manager_module.TensorTransferMode.TAGGED
    assert headers[0].window_size == 16
    assert [tag for tag, _payload in payloads] == [0, 1]
    assert all(payload is tensors[key] for (_tag, payload), key in zip(payloads, worker.full_tensor_keys, strict=True))
    assert completions == [(2, 32)]
    assert endpoint.closed is True


def test_relay_worker_nccl_chain_packs_tensors_in_publisher_order(monkeypatch):
    monkeypatch.setenv("LAMINAR_RELAY_BACKEND", "nccl")
    tensors = {f"tensor-{index}": torch.full((index + 1,), float(index)) for index in range(5)}
    worker = object.__new__(RelayWorker)
    worker.name = "relay-test"
    worker.full_tensor_keys = list(tensors)
    worker._active_version = 7
    worker.server = SimpleNamespace(
        buffer={key: SimpleNamespace(tensor=tensor, version=7) for key, tensor in tensors.items()},
        tensor_rwlocks=defaultdict(ReadWriteLock),
    )
    endpoint = SimpleNamespace(closed=False)

    async def close_endpoint():
        endpoint.closed = True

    endpoint.close = close_endpoint
    worker.dst_server_address = ("127.0.0.1", 12345)
    sent = []

    async def fake_create_endpoint(_ip, _port):
        return endpoint

    async def fake_send_obj(actual_endpoint, header):
        assert actual_endpoint is endpoint
        sent.append(header)

    async def fake_send_tensor(actual_endpoint, tensor, **_kwargs):
        assert actual_endpoint is endpoint
        sent.append(tensor.clone())
        return SimpleNamespace(staging_time_s=0.0, transport_time_s=0.0)

    async def fake_complete_tensor_transfer(actual_endpoint, *, tensor_count, payload_nbytes):
        assert actual_endpoint is endpoint
        sent.append((tensor_count, payload_nbytes))

    monkeypatch.setattr(relay_manager_module, "create_endpoint", fake_create_endpoint)
    monkeypatch.setattr(relay_transport_module, "send_obj", fake_send_obj)
    monkeypatch.setattr(relay_transport_module, "send_tensor", fake_send_tensor)
    monkeypatch.setattr(relay_transport_module, "complete_tensor_transfer", fake_complete_tensor_transfer)

    async def run():
        await worker.send_full_state_dict_to_dst_worker(wait_for_version=7)
        for lock in worker.server.tensor_rwlocks.values():
            await asyncio.wait_for(lock.acquire_write(), timeout=0.1)
            lock.release_write()

    asyncio.run(run())
    header, payload, completion = sent
    assert header.mode is relay_manager_module.TensorTransferMode.PACKED
    assert [meta.key for meta in header.storage_metas] == worker.full_tensor_keys
    assert [meta.version for meta in header.storage_metas] == [7] * 5
    assert torch.equal(payload.view(torch.float32), torch.cat(list(tensors.values())))
    assert completion == (5, 60)
    assert endpoint.closed is True


def test_relay_worker_chain_bounds_concurrent_zero_copy_sends(monkeypatch):
    tensors = {f"tensor-{index}": torch.arange(index + 1, dtype=torch.float32) for index in range(5)}
    worker = object.__new__(RelayWorker)
    worker.name = "relay-test"
    worker.full_tensor_keys = list(tensors)
    worker._active_version = 7
    worker.server = SimpleNamespace(
        buffer={key: SimpleNamespace(tensor=tensor, version=7) for key, tensor in tensors.items()},
        tensor_rwlocks=defaultdict(ReadWriteLock),
    )
    endpoint = SimpleNamespace()

    async def close_endpoint():
        return None

    endpoint.close = close_endpoint
    worker.dst_server_address = ("127.0.0.1", 12345)
    active = 0
    max_active = 0
    tags = []
    completions = []

    async def fake_create_endpoint(_ip, _port):
        return endpoint

    async def fake_send_obj(_endpoint, header):
        assert _endpoint is endpoint
        assert header.mode is relay_manager_module.TensorTransferMode.TAGGED
        assert header.window_size == 2

    async def fake_send_tensor(_endpoint, tensor, *, tag=None):
        nonlocal active, max_active
        assert _endpoint is endpoint
        assert tensor is tensors[f"tensor-{tag}"]
        active += 1
        max_active = max(max_active, active)
        tags.append(tag)
        await asyncio.sleep(0)
        active -= 1

    async def fake_complete_tensor_transfer(_endpoint, *, tensor_count, payload_nbytes):
        assert _endpoint is endpoint
        completions.append((tensor_count, payload_nbytes))

    monkeypatch.setattr(relay_manager_module, "_RELAY_CHAIN_SEND_WINDOW_SIZE", 2)
    monkeypatch.setattr(relay_manager_module, "create_endpoint", fake_create_endpoint)
    monkeypatch.setattr(relay_manager_module, "send_obj", fake_send_obj)
    monkeypatch.setattr(relay_manager_module, "send_tensor", fake_send_tensor)
    monkeypatch.setattr(relay_manager_module, "complete_tensor_transfer", fake_complete_tensor_transfer)

    stats = asyncio.run(worker.send_full_state_dict_to_dst_worker(wait_for_version=7))

    assert stats.logical_payload_bytes == 60
    assert stats.elapsed_s > 0
    assert tags == [0, 1, 2, 3, 4]
    assert max_active == 2
    assert completions == [(5, 60)]


def test_relay_worker_chain_cancels_siblings_and_releases_locks_on_send_failure(monkeypatch):
    tensors = {f"tensor-{index}": torch.arange(2, dtype=torch.float32) for index in range(2)}
    worker = object.__new__(RelayWorker)
    worker.name = "relay-test"
    worker.full_tensor_keys = list(tensors)
    worker._active_version = 7
    worker.server = SimpleNamespace(
        buffer={key: SimpleNamespace(tensor=tensor, version=7) for key, tensor in tensors.items()},
        tensor_rwlocks=defaultdict(ReadWriteLock),
    )
    endpoint = SimpleNamespace(closed=False)
    events = []

    async def close_endpoint():
        endpoint.closed = True

    endpoint.close = close_endpoint
    worker.dst_server_address = ("127.0.0.1", 12345)

    async def fake_create_endpoint(_ip, _port):
        return endpoint

    async def fake_send_obj(_endpoint, _header):
        assert _endpoint is endpoint

    async def fake_send_tensor(_endpoint, _tensor, *, tag=None):
        assert _endpoint is endpoint
        if tag == 0:
            events.append("started")
            try:
                await asyncio.Event().wait()
            finally:
                events.append("drained")
        await asyncio.sleep(0)
        raise RuntimeError("send failed")

    monkeypatch.setattr(relay_manager_module, "create_endpoint", fake_create_endpoint)
    monkeypatch.setattr(relay_manager_module, "send_obj", fake_send_obj)
    monkeypatch.setattr(relay_manager_module, "send_tensor", fake_send_tensor)

    async def run():
        with pytest.raises(RuntimeError, match="send failed"):
            await worker.send_full_state_dict_to_dst_worker(wait_for_version=7)
        assert events == ["started", "drained"]
        for lock in worker.server.tensor_rwlocks.values():
            await asyncio.wait_for(lock.acquire_write(), timeout=0.1)
            lock.release_write()

    asyncio.run(run())
    assert endpoint.closed is True


def test_relay_worker_waits_for_first_tensor_before_creating_endpoint(monkeypatch):
    tensor = torch.arange(2, dtype=torch.float32)
    entry = SimpleNamespace(tensor=tensor, version=6)
    worker = object.__new__(RelayWorker)
    worker.name = "relay-test"
    worker.full_tensor_keys = ["dense.weight"]
    worker._active_version = 7
    worker.server = SimpleNamespace(
        buffer={"dense.weight": entry},
        tensor_rwlocks=defaultdict(ReadWriteLock),
    )
    worker.dst_server_address = ("127.0.0.1", 12345)
    endpoint = SimpleNamespace()
    sleeps = []
    completions = []

    async def fake_sleep(interval_s):
        sleeps.append(interval_s)
        entry.version = 7

    async def fake_create_endpoint(ip, port):
        assert (ip, port) == worker.dst_server_address
        assert entry.version == 7
        assert sleeps == [RelayWorker.DEFAULT_INTERVAL_S]
        return endpoint

    async def fake_send_obj(_endpoint, _header):
        assert _endpoint is endpoint

    async def fake_send_tensor(_endpoint, actual_tensor, *, tag=None):
        assert _endpoint is endpoint
        assert actual_tensor is tensor
        assert tag == 0

    async def fake_complete_tensor_transfer(_endpoint, *, tensor_count, payload_nbytes):
        assert _endpoint is endpoint
        completions.append((tensor_count, payload_nbytes))

    async def close_endpoint():
        return None

    endpoint.close = close_endpoint
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(relay_manager_module, "create_endpoint", fake_create_endpoint)
    monkeypatch.setattr(relay_manager_module, "send_obj", fake_send_obj)
    monkeypatch.setattr(relay_manager_module, "send_tensor", fake_send_tensor)
    monkeypatch.setattr(relay_manager_module, "complete_tensor_transfer", fake_complete_tensor_transfer)

    stats = asyncio.run(worker.send_full_state_dict_to_dst_worker(wait_for_version=7))

    assert stats.logical_payload_bytes == 8
    assert stats.elapsed_s > 0
    assert completions == [(1, 8)]


def test_relay_worker_uses_server_ready_state():
    worker = object.__new__(RelayWorker)
    worker.name = "relay-test"
    intervals: list[float] = []

    class FakeServer:
        ip = "127.0.0.1"

        async def wait_until_ready(self, interval_s):
            intervals.append(interval_s)
            return SimpleNamespace(port=12345)

    worker.server = FakeServer()
    endpoint = asyncio.run(worker.get_server_ip_port())

    assert endpoint == ("127.0.0.1", 12345)
    assert intervals == [RelayWorker.DEFAULT_INTERVAL_S]


def test_relay_worker_startup_connect_commits_destination(monkeypatch):
    worker = object.__new__(RelayWorker)
    worker.name = "Relay 0"
    destination = _fake_relay_destination(("fd00::4", 12345))
    close_calls = []

    class FakeEndpoint:
        async def close(self):
            close_calls.append("close")

    async def fake_create_endpoint(ip, port):
        assert (ip, port) == ("fd00::4", 12345)
        return FakeEndpoint()

    monkeypatch.setattr(relay_manager_module, "create_endpoint", fake_create_endpoint)

    asyncio.run(worker.connect_to_dst_worker(destination, hop_index=0, dst_name="Relay 1"))

    assert close_calls == ["close"]
    assert worker.dst_worker is destination
    assert worker.dst_server_address == ("fd00::4", 12345)


def test_relay_worker_startup_connect_does_not_commit_on_failure(monkeypatch):
    worker = object.__new__(RelayWorker)
    worker.name = "Relay 0"
    destination = _fake_relay_destination(("127.0.0.1", 1))

    async def fake_create_endpoint(_ip, _port):
        raise TimeoutError("peer timed out")

    monkeypatch.setattr(relay_manager_module, "create_endpoint", fake_create_endpoint)

    with pytest.raises(TimeoutError, match="peer timed out"):
        asyncio.run(worker.connect_to_dst_worker(destination, hop_index=0, dst_name="Relay 1"))

    assert not hasattr(worker, "dst_worker")
    assert not hasattr(worker, "dst_server_address")


def test_relay_worker_full_read_requires_exact_version():
    worker = object.__new__(RelayWorker)
    worker.name = "relay-test"
    worker.full_tensor_keys = ["dense.weight", "dense.bias"]
    worker._publish_lock = ReadWriteLock()
    worker.server = SimpleNamespace(
        buffer={
            "dense.weight": SimpleNamespace(version=4),
            "dense.bias": SimpleNamespace(version=4),
        }
    )

    async def run() -> None:
        assert worker.acquire_full_read_if_publish(version=3) is False
        await asyncio.wait_for(worker._publish_lock.acquire_write(), timeout=0.1)
        worker._publish_lock.release_write()

        assert worker.acquire_full_read_if_publish(version=4) is True
        worker.release_full_read()

    asyncio.run(run())


def test_relay_worker_full_read_releases_lock_when_metadata_check_raises():
    worker = object.__new__(RelayWorker)
    worker._publish_lock = ReadWriteLock()

    def fail_match(*, version: int) -> bool:
        del version
        raise RuntimeError("metadata check failed")

    worker._full_publish_matches = fail_match

    async def run() -> None:
        with pytest.raises(RuntimeError, match="metadata check failed"):
            worker.acquire_full_read_if_publish(version=4)
        await asyncio.wait_for(worker._publish_lock.acquire_write(), timeout=0.1)
        worker._publish_lock.release_write()

    asyncio.run(run())


def test_relay_worker_complete_publish_requires_every_tensor_from_exact_version():
    worker = object.__new__(RelayWorker)
    worker.name = "relay-test"
    worker.full_tensor_keys = ["dense.weight", "dense.bias"]
    worker._publish_lock = ReadWriteLock()
    worker._active_version = None
    worker.server = SimpleNamespace(
        buffer={
            "dense.weight": SimpleNamespace(version=5),
            "dense.bias": SimpleNamespace(version=4),
        }
    )

    async def run() -> None:
        await worker.begin_publish(version=5)
        with pytest.raises(RuntimeError, match="cannot commit incomplete tensor"):
            worker.complete_publish(version=5)

        worker.server.buffer["dense.bias"].version = 5
        worker.complete_publish(version=5)
        assert worker._active_version is None
        await asyncio.wait_for(worker._publish_lock.acquire_write(), timeout=0.1)
        worker._publish_lock.release_write()

    asyncio.run(run())


def test_relay_manager_broadcast_failure_raises_to_caller():
    source = _FakeRelayWorker(send_error=RuntimeError("chain failed for v7"))
    manager = _make_lightweight_relay_manager(source, _FakeRelayWorker())

    async def run() -> None:
        await manager.begin_publish(version=7)
        with pytest.raises(RuntimeError, match="chain failed for v7"):
            await manager.broadcast(version=7)

    asyncio.run(run())

    metrics = manager.get_metrics()
    assert metrics["trainer_version"] == 7
    assert metrics["broadcast/started"] == 1
    assert metrics["broadcast/failed"] == 1
    assert metrics["broadcast/completed"] == 0
    assert metrics["loadable_version"] == EMPTY_VERSION
    assert metrics["publish/active_version"] == 7
    assert metrics["publish/broadcast_done"] == 0


def test_relay_manager_begin_broadcast_complete_commits_version():
    first_hop = _FakeRelayWorker(hop_stats=_hop_stats(28, 2.0))
    second_hop = _FakeRelayWorker(hop_stats=_hop_stats(28, 3.0))
    last_worker = _FakeRelayWorker()
    manager = _make_lightweight_relay_manager(first_hop, second_hop, last_worker)
    manager.full_state_meta_dict = {
        "dense.weight": TensorMeta(dtype=torch.float16, shape=torch.Size([2, 3])),
        "dense.bias": TensorMeta(dtype=torch.float32, shape=torch.Size([4])),
    }

    async def run() -> RelayBroadcastStats:
        await manager.begin_publish(version=8)
        broadcast_stats = await manager.broadcast(version=8)
        assert manager.get_loadable_version() == EMPTY_VERSION
        await manager.complete_publish(version=8)
        return broadcast_stats

    broadcast_stats = asyncio.run(run())
    assert broadcast_stats == RelayBroadcastStats(
        relay_broadcast_time_s=broadcast_stats.relay_broadcast_time_s,
        logical_payload_bytes=28,
        tensor_count=2,
        hop_count=2,
        bottleneck_hop_index=1,
        end_to_end_goodput_bytes_per_s=28 / broadcast_stats.relay_broadcast_time_s,
    )
    assert broadcast_stats.relay_broadcast_time_s == manager._metrics["broadcast/latency_s/last"]
    assert first_hop.events == [("begin", 8), ("send", 8), ("complete", 8)]
    assert second_hop.events == [("begin", 8), ("send", 8), ("complete", 8)]
    assert last_worker.events == [("begin", 8), ("complete", 8)]

    metrics = manager.get_metrics()
    assert metrics["publish/started"] == 1
    assert metrics["publish/committed"] == 1
    assert metrics["broadcast/started"] == 1
    assert metrics["broadcast/failed"] == 0
    assert metrics["broadcast/completed"] == 1
    assert metrics["broadcast/tensor_count/last"] == 2
    assert metrics["broadcast/tensor_count"] == 2
    assert metrics["broadcast/logical_payload_bytes/last"] == 28
    assert metrics["broadcast/logical_payload_bytes"] == 28
    assert metrics["broadcast/latency_s/last"] > 0
    assert metrics["broadcast/latency_s/max"] == metrics["broadcast/latency_s/last"]
    assert metrics["broadcast/latency_s"] == metrics["broadcast/latency_s/last"]
    assert metrics["broadcast/end_to_end_goodput_bytes_per_s/last"] > 0
    assert metrics["broadcast/bottleneck_hop_index/last"] == 1
    assert metrics["trainer_version"] == 8
    assert metrics["loadable_version"] == 8
    assert metrics["publish/active_version"] == EMPTY_VERSION
    assert metrics["publish/broadcast_done"] == 0
    assert manager.get_loadable_version() == 8


def test_relay_manager_single_hop_records_bottleneck_index():
    source = _FakeRelayWorker(hop_stats=_hop_stats(8, 2.0))
    manager = _make_lightweight_relay_manager(source, _FakeRelayWorker())
    manager.full_state_meta_dict = {
        "weight": TensorMeta(dtype=torch.float32, shape=torch.Size([2])),
    }

    async def run() -> RelayBroadcastStats:
        await manager.begin_publish(version=1)
        return await manager.broadcast(version=1)

    broadcast_stats = asyncio.run(run())

    assert broadcast_stats.hop_count == 1
    assert broadcast_stats.bottleneck_hop_index == 0


def test_relay_manager_rejects_zero_duration_for_nonempty_hop():
    source = _FakeRelayWorker(hop_stats=_hop_stats(8, 0.0))
    manager = _make_lightweight_relay_manager(source, _FakeRelayWorker())
    manager.full_state_meta_dict = {
        "weight": TensorMeta(dtype=torch.float32, shape=torch.Size([2])),
    }

    async def run() -> None:
        await manager.begin_publish(version=1)
        with pytest.raises(RuntimeError, match="must be positive"):
            await manager.broadcast(version=1)

    asyncio.run(run())

    assert manager.get_metrics()["broadcast/failed"] == 1
    assert manager.get_metrics()["broadcast/completed"] == 0


def test_relay_manager_rejects_publish_commit_before_broadcast():
    manager = _make_lightweight_relay_manager(_FakeRelayWorker())

    async def run() -> None:
        await manager.begin_publish(version=3)
        with pytest.raises(RuntimeError, match="incompletely broadcast"):
            await manager.complete_publish(version=3)

    asyncio.run(run())

    assert manager.get_loadable_version() == EMPTY_VERSION


def test_relay_manager_requires_strictly_increasing_versions():
    source = _FakeRelayWorker(hop_stats=_hop_stats(8, 2.0))
    manager = _make_lightweight_relay_manager(source, _FakeRelayWorker())
    manager.full_state_meta_dict = {
        "weight": TensorMeta(dtype=torch.float32, shape=torch.Size([2])),
    }

    async def run() -> None:
        await manager.begin_publish(version=1)
        await manager.broadcast(version=1)
        await manager.complete_publish(version=1)
        with pytest.raises(RuntimeError, match="strictly increasing"):
            await manager.begin_publish(version=1)

        await manager.begin_publish(version=2)
        with pytest.raises(RuntimeError, match="Cannot overlap"):
            await manager.begin_publish(version=3)

    asyncio.run(run())

    assert manager.get_trainer_version() == 2
    assert manager.get_loadable_version() == 1
    assert manager.get_metrics()["publish/started"] == 2
    assert manager.get_metrics()["publish/committed"] == 1
    assert manager._active_publish is not None


def test_relay_manager_begin_failure_leaves_publish_active_and_blocks_retry():
    healthy_worker = _FakeRelayWorker()
    failing_worker = _FakeRelayWorker()
    manager = _make_lightweight_relay_manager(healthy_worker, failing_worker)

    async def fail_begin(*, version: int) -> None:
        del version
        raise RuntimeError("begin failed")

    failing_worker.begin_publish = _RemoteMethod(fail_begin)

    async def run() -> None:
        with pytest.raises(RuntimeError, match="Failed to begin Relay publish"):
            await manager.begin_publish(version=3)
        assert manager._active_publish is not None
        assert manager.get_trainer_version() == 3
        assert manager.get_loadable_version() == EMPTY_VERSION
        with pytest.raises(RuntimeError, match="Cannot overlap"):
            await manager.begin_publish(version=4)

    asyncio.run(run())


def test_relay_manager_committed_load_gate_uses_exact_version():
    source = _FakeRelayWorker(hop_stats=_hop_stats(8, 2.0))
    destination = _FakeRelayWorker()
    manager = _make_lightweight_relay_manager(source, destination)
    manager.full_state_meta_dict = {
        "weight": TensorMeta(dtype=torch.float32, shape=torch.Size([2])),
    }

    async def run() -> None:
        await manager.begin_publish(version=4)
        await manager.broadcast(version=4)
        await manager.complete_publish(version=4)

        loaded = await manager.acquire_rollout_version(
            node_ids=["node-0", "node-1"],
            minimum_version=4,
        )
        assert loaded == 4
        assert source.acquire_calls == [4]
        assert destination.acquire_calls == [4]
        assert await manager.release_full_read(["node-0", "node-1"]) is None

    asyncio.run(run())
    assert source.read_locks == 0
    assert destination.read_locks == 0


def test_relay_manager_read_fanout_error_releases_successful_acquisitions():
    source = _FakeRelayWorker()
    failing_worker = _FakeRelayWorker()
    manager = _make_lightweight_relay_manager(source, failing_worker)
    manager.trainer_version = 7
    manager.loadable_version = 7
    source.physical_version = 7
    failing_worker.physical_version = 7

    async def fail_acquire(*, version: int) -> bool:
        del version
        raise RuntimeError("acquire failed")

    failing_worker.acquire_full_read_if_publish = _RemoteMethod(fail_acquire)

    async def run() -> None:
        with pytest.raises(RuntimeError, match="Failed to acquire a consistent Relay read fence"):
            await manager.acquire_rollout_version(
                node_ids=["node-0", "node-1"],
                minimum_version=7,
            )

    asyncio.run(run())
    assert source.read_locks == 0
    assert failing_worker.read_locks == 0


def test_relay_manager_cancelled_read_fanout_waits_then_releases_every_lock():
    source = _FakeRelayWorker()
    slow_worker = _FakeRelayWorker()
    manager = _make_lightweight_relay_manager(source, slow_worker)
    manager.trainer_version = 7
    manager.loadable_version = 7
    source.physical_version = 7
    slow_worker.physical_version = 7
    slow_started = asyncio.Event()
    slow_release = asyncio.Event()
    original_acquire = slow_worker._acquire_full_read_if_publish

    async def slow_acquire(*, version: int) -> bool:
        slow_started.set()
        await slow_release.wait()
        return await original_acquire(version=version)

    slow_worker.acquire_full_read_if_publish = _RemoteMethod(slow_acquire)

    async def run() -> None:
        load_task = asyncio.create_task(
            manager.acquire_rollout_version(
                node_ids=["node-0", "node-1"],
                minimum_version=7,
            )
        )
        await slow_started.wait()
        await asyncio.sleep(0)
        assert source.read_locks == 1

        load_task.cancel()
        await asyncio.sleep(0)
        assert load_task.done() is False
        assert source.read_locks == 1

        slow_release.set()
        with pytest.raises(asyncio.CancelledError):
            await load_task
        assert source.read_locks == 0
        assert slow_worker.read_locks == 0

    asyncio.run(run())


@pytest.mark.parametrize("cancel_acquire", [False, True])
def test_relay_manager_partial_read_release_error_waits_for_siblings_and_never_retries(cancel_acquire):
    source = _FakeRelayWorker()
    sibling = _FakeRelayWorker()
    unavailable = _FakeRelayWorker()
    manager = _make_lightweight_relay_manager(source, sibling, unavailable)
    manager.trainer_version = manager.loadable_version = 7
    source.physical_version = sibling.physical_version = 7
    sibling_releasing = asyncio.Event()
    finish_sibling_release = asyncio.Event()
    release_error = RuntimeError("read release failed")

    async def fail_release():
        raise release_error

    async def slow_release():
        sibling_releasing.set()
        await finish_sibling_release.wait()
        await sibling._release_full_read()

    source.release_full_read = _RemoteMethod(fail_release)
    sibling.release_full_read = _RemoteMethod(slow_release)

    async def run():
        load_task = asyncio.create_task(
            manager.acquire_rollout_version(["node-0", "node-1", "node-2"], minimum_version=7)
        )
        try:
            await asyncio.wait_for(sibling_releasing.wait(), timeout=1)
            assert not load_task.done()
            if cancel_acquire:
                load_task.cancel()
                await asyncio.sleep(0)
                assert not load_task.done()
            finish_sibling_release.set()
            with pytest.raises(RuntimeError, match="Failed to release a partial Relay read fence") as exc_info:
                await load_task
            assert exc_info.value.__cause__ is release_error
        finally:
            finish_sibling_release.set()
            await asyncio.gather(load_task, return_exceptions=True)

    asyncio.run(run())
    # Failed release is an error, never a normal race eligible for a retry.
    assert source.acquire_calls == sibling.acquire_calls == unavailable.acquire_calls == [7]
    assert sibling.read_locks == unavailable.read_locks == 0
    assert source.read_locks == 1


def test_relay_manager_repeated_cancellation_during_partial_read_rollback_waits_for_release():
    source = _FakeRelayWorker()
    unavailable = _FakeRelayWorker()
    manager = _make_lightweight_relay_manager(source, unavailable)
    manager.trainer_version = manager.loadable_version = source.physical_version = 7
    release_started = asyncio.Event()
    finish_release = asyncio.Event()

    async def slow_release():
        release_started.set()
        await finish_release.wait()
        await source._release_full_read()

    source.release_full_read = _RemoteMethod(slow_release)

    async def run():
        load_task = asyncio.create_task(manager.acquire_rollout_version(["node-0", "node-1"], minimum_version=7))
        try:
            await asyncio.wait_for(release_started.wait(), timeout=1)
            for _ in range(2):
                load_task.cancel()
                await asyncio.sleep(0)
                assert not load_task.done()
                assert source.read_locks == 1
            finish_release.set()
            with pytest.raises(asyncio.CancelledError):
                await load_task
            assert source.read_locks == 0
        finally:
            finish_release.set()
            await asyncio.gather(load_task, return_exceptions=True)

    asyncio.run(run())


def test_relay_manager_active_failed_publish_blocks_new_reads():
    source = _FakeRelayWorker(hop_stats=_hop_stats(8, 2.0))
    destination = _FakeRelayWorker()
    manager = _make_lightweight_relay_manager(source, destination)
    manager.full_state_meta_dict = {
        "weight": TensorMeta(dtype=torch.float32, shape=torch.Size([2])),
    }

    async def run() -> None:
        await manager.begin_publish(version=1)
        await manager.broadcast(version=1)
        await manager.complete_publish(version=1)

        await manager.begin_publish(version=2)
        assert manager.get_trainer_version() == 2
        assert manager.get_loadable_version() == 1
        assert manager._active_publish is not None

        load_task = asyncio.create_task(
            manager.acquire_rollout_version(
                node_ids=["node-0", "node-1"],
                minimum_version=1,
            )
        )
        await asyncio.sleep(0)
        assert load_task.done() is False
        load_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await load_task
        assert source.acquire_calls == []
        assert destination.acquire_calls == []

    asyncio.run(run())


def test_ingress_tensor_wait_fails_closed_on_deadline(monkeypatch):
    """A never-arriving shard makes the ingress wait fail-fast, not hang forever.

    Reproduces the run4/run5 hang (Relay 0 waiting on a version=0 shard the
    publisher never supplied): the wait must raise a clear deadline error rather
    than loop indefinitely occupying GPUs. The deadline is monkeypatched tiny so
    the test does not actually wait the production ceiling.
    """
    monkeypatch.setattr(relay_manager_module, "_RELAY_INGRESS_TENSOR_WAIT_TIMEOUT_S", 0.05)

    worker = object.__new__(RelayWorker)
    worker.name = "relay-test"
    worker.full_tensor_keys = ["dense.weight"]
    worker._active_version = 7
    worker.DEFAULT_INTERVAL_S = 1e-3
    # A missing FSDP shard keeps the full tensor below the requested version until the deadline.
    worker.server = SimpleNamespace(
        buffer={"dense.weight": SimpleNamespace(tensor=torch.tensor([0.0, 0.0]), version=EMPTY_VERSION)},
    )

    with pytest.raises(RuntimeError, match="ingress tensor wait exceeded deadline"):
        asyncio.run(worker.wait_for_full_state_dict(version=7))


def test_relay_manager_scopes_worker_resources_without_polluting_driver(monkeypatch):
    captured_options = []
    workers = []
    connections = []

    class FakeRemoteWorker:
        def options(self, **options):
            captured_options.append(options)
            return self

        def remote(self, *, name):
            worker = SimpleNamespace(name=name)
            worker.connect_to_dst_worker = _RemoteMethod(
                lambda destination, **kwargs: connections.append((worker, destination, kwargs))
            )
            workers.append(worker)
            return worker

    monkeypatch.setenv("UCX_TLS", "shm,tcp")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("MKL_NUM_THREADS", "8")
    monkeypatch.setattr(relay_manager_module.ray, "remote", lambda _worker_cls: FakeRemoteWorker())
    monkeypatch.setattr(relay_manager_module.ray, "get", lambda refs: refs)

    RelayManager(node_ids=["a" * 56, "b" * 56], num_cpus=3)

    worker_env = {
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        "OMP_NUM_THREADS": "3",
        "MKL_NUM_THREADS": "3",
    }
    assert [options["num_cpus"] for options in captured_options] == [3, 3]
    assert [options["runtime_env"] for options in captured_options] == [{"env_vars": worker_env}] * 2
    assert os.environ["OMP_NUM_THREADS"] == os.environ["MKL_NUM_THREADS"] == "8"
    assert "UCX_TLS" not in worker_env
    assert connections == [(workers[0], workers[1], {"hop_index": 0, "dst_name": "Relay 1"})]


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "invalid"])
def test_relay_manager_rejects_invalid_cpu_budget_before_creating_workers(monkeypatch, value):
    def unexpected_remote(*args, **kwargs):
        pytest.fail("invalid CPU budget must not create actors")

    monkeypatch.setattr(relay_manager_module.ray, "remote", unexpected_remote)
    with pytest.raises(ValueError, match="relay.num_cpus must be a positive integer"):
        RelayManager(node_ids=["a" * 56], num_cpus=value)  # type: ignore[arg-type]
