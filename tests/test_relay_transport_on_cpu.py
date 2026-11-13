# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
import errno
import os
import pickle
import socket
from collections import defaultdict
from types import SimpleNamespace

import pytest
import ray
import torch

from laminar.train.relay import transport
from laminar.utils.concurrency import ReadWriteLock, gather_cancel_on_error
from laminar.utils.tensor import TensorMeta


class _CaptureEndpoint:
    def __init__(self):
        self.headers = []
        self.payloads = []

    async def send_obj(self, data):
        self.headers.append(pickle.loads(data))

    async def send(self, data):
        self.payloads.append(data.copy())

    async def recv_obj(self):
        complete = self.headers[-1]
        assert isinstance(complete, transport.TensorTransferComplete)
        return pickle.dumps(
            transport.TensorTransferCompleteAck(
                tensor_count=complete.tensor_count,
                payload_nbytes=complete.payload_nbytes,
            )
        )


class _ReplayEndpoint:
    def __init__(self, headers, payloads):
        self.headers = iter(headers)
        self.payloads = iter(payloads)

    async def recv_obj(self):
        return pickle.dumps(next(self.headers))

    async def recv(self, destination):
        source = next(self.payloads)
        assert destination.nbytes == source.nbytes
        destination.reshape(-1)[:] = source.reshape(-1)


class _TaggedReplayEndpoint:
    def __init__(self, header, payloads):
        self.header = header
        self.payloads = payloads
        self.active = 0
        self.max_active = 0
        self.tags = []

    async def recv_obj(self):
        return pickle.dumps(self.header)

    async def recv(self, destination, *, tag=None):
        assert tag is not None
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.tags.append(tag)
        await asyncio.sleep(0)
        source = self.payloads[tag]
        assert destination.nbytes == source.nbytes
        destination.reshape(-1)[:] = source.reshape(-1)
        self.active -= 1


class _ClosingEndpoint:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class _ResettingRecvEndpoint:
    async def recv(self, destination):
        raise transport.UCXConnectionResetError("peer reset")


class _DuplexEndpoint:
    def __init__(self, incoming, outgoing):
        self.incoming = incoming
        self.outgoing = outgoing

    async def send_obj(self, data):
        await self.outgoing.put(data)

    async def recv_obj(self):
        return await self.incoming.get()

    async def send(self, data):
        await self.outgoing.put(data.copy())

    async def recv(self, destination):
        source = await self.incoming.get()
        assert destination.nbytes == source.nbytes
        destination.reshape(-1)[:] = source.reshape(-1)


def _duplex_endpoint_pair():
    client_to_server = asyncio.Queue()
    server_to_client = asyncio.Queue()
    return (
        _DuplexEndpoint(server_to_client, client_to_server),
        _DuplexEndpoint(client_to_server, server_to_client),
    )


def test_ucx_tensor_server_treats_client_cancellation_as_disconnect(monkeypatch):
    server = object.__new__(transport.UCXTensorServer)
    server.backend = "ucx"
    endpoint = _ClosingEndpoint()

    async def raise_canceled(*args, **kwargs):
        raise transport.UCXCanceledError("client closed")

    monkeypatch.setattr(server, "_send_tensors_handler", raise_canceled)

    asyncio.run(server._handler(endpoint))

    assert endpoint.closed is True


def test_ucx_tensor_server_prints_unexpected_handler_failure(monkeypatch, capsys):
    server = object.__new__(transport.UCXTensorServer)
    server.backend = "ucx"
    endpoint = _ClosingEndpoint()

    async def raise_failure(*args, **kwargs):
        raise ValueError("metadata mismatch")

    monkeypatch.setattr(server, "_send_tensors_handler", raise_failure)

    asyncio.run(server._handler(endpoint))

    assert endpoint.closed is True
    output = capsys.readouterr().out
    assert "Relay transport fatal error: ValueError: metadata mismatch" in output
    assert "Traceback (most recent call last)" in output


@pytest.mark.parametrize(
    ("listener_host", "expected"),
    [(None, "192.0.2.91"), ("0.0.0.0", "192.0.2.91"), ("::", "2001:db8::1"), ("2001:db8::1", "2001:db8::1")],
)
def test_get_ucx_ip_matches_listener_family_on_dual_stack_interface(monkeypatch, listener_host, expected):
    monkeypatch.setenv("UCXX_IFNAME", "eth0")
    if listener_host is None:
        monkeypatch.delenv("UCXX_LISTENER_HOST", raising=False)
    else:
        monkeypatch.setenv("UCXX_LISTENER_HOST", listener_host)
    monkeypatch.setattr(
        transport.ucxx,
        "get_address",
        lambda ifname=None, use_ipv6=False: "2001:db8::1" if use_ipv6 else "192.0.2.91",
    )
    assert transport.get_ucx_ip() == expected


def test_get_ucx_ip_uses_selected_interface_ipv6_when_ipv4_is_unavailable(monkeypatch):
    monkeypatch.delenv("UCXX_LISTENER_HOST", raising=False)
    monkeypatch.setenv("UCXX_IFNAME", "eth0")
    calls = []

    def get_address(ifname=None, use_ipv6=False):
        calls.append((ifname, use_ipv6))
        if not use_ipv6:
            raise OSError(errno.EADDRNOTAVAIL, "Cannot assign requested address")
        return "2605:340:cd51:7700::1"

    monkeypatch.setattr(transport.ucxx, "get_address", get_address)
    monkeypatch.setattr(
        transport,
        "get_first_rdma_interface_addresses",
        lambda: [SimpleNamespace(family=socket.AF_INET6, address="fe80::1")],
    )

    assert transport.get_ucx_ip() == "2605:340:cd51:7700::1"
    assert calls == [("eth0", False), ("eth0", True)]


def test_get_ucx_ip_propagates_other_address_errors(monkeypatch):
    monkeypatch.delenv("UCXX_LISTENER_HOST", raising=False)
    monkeypatch.setenv("UCXX_IFNAME", "eth0")
    error = OSError(errno.ENODEV, "No such device")
    calls = []

    def get_address(ifname=None, use_ipv6=False):
        calls.append((ifname, use_ipv6))
        raise error

    monkeypatch.setattr(transport.ucxx, "get_address", get_address)

    with pytest.raises(OSError) as caught:
        transport.get_ucx_ip()

    assert caught.value is error
    assert calls == [("eth0", False)]


def test_get_ucx_ip_fails_closed_without_tcp_fallback(monkeypatch):
    monkeypatch.delenv("UCXX_LISTENER_HOST", raising=False)
    monkeypatch.delenv("LAMINAR_RELAY_UCX_TCP_FALLBACK", raising=False)
    monkeypatch.setattr(transport.ucxx, "get_address", lambda ifname=None: None)
    monkeypatch.setattr(transport, "get_first_rdma_interface_addresses", lambda: None)

    with pytest.raises(RuntimeError, match="No IP address found for UCX"):
        transport.get_ucx_ip()


def test_get_ucx_ip_can_fallback_to_ray_node_ip(monkeypatch):
    monkeypatch.delenv("UCXX_LISTENER_HOST", raising=False)
    monkeypatch.setenv("LAMINAR_RELAY_UCX_TCP_FALLBACK", "1")
    monkeypatch.delenv("UCX_TLS", raising=False)
    monkeypatch.setattr(transport.ucxx, "get_address", lambda ifname=None: None)
    monkeypatch.setattr(transport, "get_first_rdma_interface_addresses", lambda: None)
    monkeypatch.setattr(ray.util, "get_node_ip_address", lambda: "127.0.0.2")

    assert transport.get_ucx_ip() == "127.0.0.2"
    assert os.environ["UCX_TLS"] == "shm,tcp"


def test_get_ucx_ip_uses_first_rdma_interface_ipv6_address(monkeypatch):
    monkeypatch.delenv("UCXX_LISTENER_HOST", raising=False)
    monkeypatch.delenv("UCXX_IFNAME", raising=False)
    monkeypatch.setattr(transport.ucxx, "get_address", lambda ifname=None: None)
    monkeypatch.setattr(
        transport,
        "get_first_rdma_interface_addresses",
        lambda: [SimpleNamespace(family=socket.AF_INET6, address="2001:db8::1")],
    )

    assert transport.get_ucx_ip() == "2001:db8::1"


def test_ucx_job_runtime_env_vars_from_env_forwards_driver_overrides(monkeypatch):
    for key in (
        "LAMINAR_RELAY_BACKEND",
        "UCX_TLS",
        "UCX_NET_DEVICES",
        "UCX_MAX_RNDV_RAILS",
        "UCX_RNDV_THRESH",
        "UCX_RNDV_FRAG_MEM_TYPE",
        "UCX_PROTO_ENABLE",
        "UCX_MEMTYPE_CACHE",
        "UCX_LOG_LEVEL",
        "UCXX_IFNAME",
        "LAMINAR_RELAY_UCX_TCP_FALLBACK",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LAMINAR_RELAY_BACKEND", "nccl")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("MKL_NUM_THREADS", "8")
    monkeypatch.setenv("UCX_TLS", "shm,tcp")
    monkeypatch.setenv("UCX_LOG_LEVEL", "warn")
    monkeypatch.setenv("LAMINAR_RELAY_UCX_TCP_FALLBACK", "1")
    monkeypatch.setenv("UNRELATED_UCX_TEST", "ignored")

    env_vars = transport.ucx_job_runtime_env_vars_from_env()

    assert env_vars == {
        "LAMINAR_RELAY_BACKEND": "nccl",
        "UCX_TLS": "shm,tcp",
        "UCX_LOG_LEVEL": "warn",
        "LAMINAR_RELAY_UCX_TCP_FALLBACK": "1",
    }


def test_cuda_cpu_staging_reuses_pinned_sender_buffer(monkeypatch):
    events = []

    class FakeDevice:
        type = "cuda"

    class FakeCudaTensor:
        device = FakeDevice()

        def numel(self):
            return 8

    class FakePinnedBuffer:
        def numel(self):
            return 16

    pinned_buffer = FakePinnedBuffer()

    def fake_empty(size, *, dtype, device, pin_memory):
        events.append(("allocate", size, dtype, device, pin_memory))
        return pinned_buffer

    async def fake_send_tensor(ep, tensor, *, cpu_staging_buffer):
        del ep, tensor
        events.append(("send", cpu_staging_buffer))
        return transport.TensorSendStats(staging_time_s=0.25, transport_time_s=0.5)

    monkeypatch.setattr(transport.torch, "empty", fake_empty)
    monkeypatch.setattr(transport, "send_tensor", fake_send_tensor)
    sender = transport.TensorChunkSender(SimpleNamespace(), chunk_bytes=16)

    asyncio.run(sender._send_payload(FakeCudaTensor()))
    asyncio.run(sender._send_payload(FakeCudaTensor()))

    assert events == [
        ("allocate", 16, torch.uint8, "cpu", True),
        ("send", pinned_buffer),
        ("send", pinned_buffer),
    ]
    assert sender.staging_time_s >= 0.5
    assert sender.transport_time_s == 1.0


def test_cuda_cpu_staging_copies_into_pinned_buffer_before_send(monkeypatch):
    events = []

    class FakeDevice:
        type = "cuda"

    class FakeCudaTensor:
        device = FakeDevice()

        def detach(self):
            events.append("detach")
            return self

        def contiguous(self):
            events.append("contiguous")
            return self

        def view(self, dtype):
            assert dtype == torch.uint8
            events.append("view")
            return self

        def numel(self):
            return 8

    class FakePinnedBuffer:
        device = SimpleNamespace(type="cpu")
        dtype = torch.uint8

        def is_pinned(self):
            return True

        def numel(self):
            return 16

        def __getitem__(self, item):
            assert item == slice(None, 8)
            events.append("slice")
            return self

        def copy_(self, tensor, *, non_blocking):
            assert isinstance(tensor, FakeCudaTensor)
            assert non_blocking is True
            events.append("copy")

        def numpy(self):
            events.append("numpy")
            return "pinned-view"

    class FakeEndpoint:
        async def send(self, data):
            assert data == "pinned-view"
            events.append("send")

    monkeypatch.setattr(
        transport,
        "_synchronize_current_cuda_stream",
        lambda tensor: events.append("sync"),
    )
    stats = asyncio.run(
        transport.send_tensor(
            FakeEndpoint(),
            FakeCudaTensor(),
            cpu_staging_buffer=FakePinnedBuffer(),
        )
    )

    assert events == ["detach", "contiguous", "view", "slice", "copy", "sync", "numpy", "send"]
    assert stats.staging_time_s >= 0.0
    assert stats.transport_time_s >= 0.0


def test_tensor_chunk_sender_packs_small_tensors_and_round_trips_metadata():
    send_ep = _CaptureEndpoint()
    tensors = {
        "small-a": torch.tensor([1.5, 2.5], dtype=torch.float32),
        "small-b": torch.tensor([3, 4], dtype=torch.int32),
        "large": torch.arange(5, dtype=torch.float32),
    }

    async def send_all():
        sender = transport.TensorChunkSender(send_ep, chunk_bytes=16)
        for key, tensor in tensors.items():
            await sender.add(
                transport.TensorStorageMeta(
                    key=key,
                    data_meta=TensorMeta(dtype=tensor.dtype, shape=tensor.shape),
                    version=7,
                ),
                tensor,
            )
        return await sender.finish()

    stats = asyncio.run(send_all())

    assert stats.chunk_count == 3
    assert stats.payload_bytes == 36
    assert stats.tensor_prepare_time_s >= 0.0
    assert stats.pack_time_s >= 0.0
    assert stats.staging_time_s == 0.0
    assert stats.transport_time_s >= 0.0
    assert stats.completion_ack_time_s >= 0.0
    assert len(send_ep.headers) == 3
    assert send_ep.headers[0].mode is transport.TensorTransferMode.PACKED
    assert [meta.key for meta in send_ep.headers[0].storage_metas] == ["small-a", "small-b"]
    assert send_ep.headers[0].payload_nbytes == 16
    assert send_ep.headers[1].mode is transport.TensorTransferMode.CHUNKED
    assert send_ep.headers[1].chunk_bytes == 16
    assert send_ep.headers[1].payload_nbytes == 20
    assert [meta.key for meta in send_ep.headers[1].storage_metas] == ["large"]
    assert send_ep.headers[2] == transport.TensorTransferComplete(tensor_count=3, payload_nbytes=36)
    assert [payload.nbytes for payload in send_ep.payloads] == [16, 16, 4]

    server = object.__new__(transport.UCXTensorServer)
    server.backend = "ucx"
    server.buffer = {}
    server.buffer_kwargs = {"device": torch.device("cpu"), "pin_memory": False}
    server.tensor_rwlocks = defaultdict(ReadWriteLock)
    replay_ep = _ReplayEndpoint(send_ep.headers, send_ep.payloads)
    transfer_state = transport._TensorReceiveState()

    packed_recv_buffer = asyncio.run(server._send_tensors_handler(replay_ep, transfer_state=transfer_state))
    assert packed_recv_buffer is not None
    assert packed_recv_buffer.numel() == transport.DEFAULT_TENSOR_TRANSFER_CHUNK_BYTES
    assert (
        asyncio.run(
            server._send_tensors_handler(
                replay_ep,
                transfer_state=transfer_state,
                packed_recv_buffer=packed_recv_buffer,
            )
        )
        is packed_recv_buffer
    )

    for key, expected in tensors.items():
        assert torch.equal(server.buffer[key].tensor, expected)
        assert server.buffer[key].version == 7


def test_tensor_transfer_completion_waits_for_receiver_commit():
    send_ep, recv_ep = _duplex_endpoint_pair()
    tensor = torch.arange(4, dtype=torch.float32)
    server = object.__new__(transport.UCXTensorServer)
    server.backend = "ucx"
    server.buffer = {}
    server.buffer_kwargs = {"device": torch.device("cpu"), "pin_memory": False}
    server.tensor_rwlocks = defaultdict(ReadWriteLock)

    async def round_trip():
        sender = transport.TensorChunkSender(send_ep, chunk_bytes=32)
        transfer_state = transport._TensorReceiveState()
        payload_committed = asyncio.Event()
        allow_ack = asyncio.Event()

        async def receive():
            packed_buffer = await server._send_tensors_handler(
                recv_ep,
                transfer_state=transfer_state,
            )
            payload_committed.set()
            await allow_ack.wait()
            await server._send_tensors_handler(
                recv_ep,
                packed_recv_buffer=packed_buffer,
                transfer_state=transfer_state,
            )

        receive_task = asyncio.create_task(receive())
        await sender.add(
            transport.TensorStorageMeta(
                key="dense.weight",
                data_meta=TensorMeta(dtype=tensor.dtype, shape=tensor.shape),
                version=5,
            ),
            tensor,
        )
        finish_task = asyncio.create_task(sender.finish())
        await payload_committed.wait()
        await asyncio.sleep(0)
        assert not finish_task.done()
        allow_ack.set()
        stats = await finish_task
        await receive_task
        return stats

    stats = asyncio.run(round_trip())

    assert stats.completion_ack_time_s >= 0.0
    assert torch.equal(server.buffer["dense.weight"].tensor, tensor)
    assert server.buffer["dense.weight"].version == 5


def test_tensor_transfer_completion_rejects_sender_count_mismatch():
    send_ep, recv_ep = _duplex_endpoint_pair()
    server = object.__new__(transport.UCXTensorServer)
    server.backend = "ucx"

    async def complete_without_payload():
        return await asyncio.gather(
            transport.complete_tensor_transfer(
                send_ep,
                tensor_count=1,
                payload_nbytes=16,
            ),
            server._send_tensors_handler(
                recv_ep,
                transfer_state=transport._TensorReceiveState(),
            ),
            return_exceptions=True,
        )

    sender_error, receiver_error = asyncio.run(complete_without_payload())

    assert isinstance(sender_error, RuntimeError)
    assert "completion acknowledgement mismatch" in str(sender_error)
    assert isinstance(receiver_error, RuntimeError)
    assert "tensor transfer completion mismatch" in str(receiver_error)


def test_tensor_transfer_completion_timeout_fails_closed():
    class _NeverAckEndpoint:
        async def send_obj(self, _data):
            return None

        async def recv_obj(self):
            await asyncio.Event().wait()

    with pytest.raises(RuntimeError, match="Timed out waiting for Relay tensor transfer completion"):
        asyncio.run(
            transport.complete_tensor_transfer(
                _NeverAckEndpoint(),
                tensor_count=1,
                payload_nbytes=8,
                timeout_s=0.01,
            )
        )


def test_relay_object_send_timeout_fails_closed():
    class _NeverSendEndpoint:
        async def send_obj(self, _data):
            await asyncio.Event().wait()

    with pytest.raises(RuntimeError, match=r"Timed out during Relay UCX send: kind=object .* timeout_s=0.01"):
        asyncio.run(
            transport.send_obj(
                _NeverSendEndpoint(),
                {"version": 7},
                timeout_s=0.01,
            )
        )


def test_relay_tensor_send_timeout_fails_closed_with_payload_context():
    class _NeverSendEndpoint:
        async def send(self, _data):
            await asyncio.Event().wait()

    with pytest.raises(
        RuntimeError,
        match=r"Timed out during Relay UCX send: kind=tensor payload_nbytes=16 tag=None timeout_s=0.01",
    ):
        asyncio.run(
            transport.send_tensor(
                _NeverSendEndpoint(),
                torch.ones(4),
                timeout_s=0.01,
            )
        )


@pytest.mark.parametrize("send_kind", ["object", "tensor"])
def test_relay_send_timeout_must_be_positive(send_kind):
    endpoint = SimpleNamespace()
    if send_kind == "object":
        call = transport.send_obj(endpoint, {}, timeout_s=0)
    else:
        call = transport.send_tensor(endpoint, torch.ones(1), timeout_s=0)
    with pytest.raises(ValueError, match="Relay UCX send timeout must be positive"):
        asyncio.run(call)


def test_tagged_tensor_receive_uses_bounded_concurrency_without_pack_copy():
    tensors = {f"tensor-{index}": torch.arange(index + 1, dtype=torch.float32) for index in range(5)}
    storage_metas = [
        transport.TensorStorageMeta(
            key=key,
            data_meta=TensorMeta(dtype=tensor.dtype, shape=tensor.shape),
            version=11,
        )
        for key, tensor in tensors.items()
    ]
    header = transport.TensorTransferHeader(
        storage_metas=storage_metas,
        mode=transport.TensorTransferMode.TAGGED,
        window_size=2,
    )
    endpoint = _TaggedReplayEndpoint(header, [tensor.view(torch.uint8).numpy() for tensor in tensors.values()])
    server = object.__new__(transport.UCXTensorServer)
    server.backend = "ucx"
    server.buffer = {}
    server.buffer_kwargs = {"device": torch.device("cpu"), "pin_memory": False}
    server.tensor_rwlocks = defaultdict(ReadWriteLock)

    assert (
        asyncio.run(
            server._send_tensors_handler(
                endpoint,
                transfer_state=transport._TensorReceiveState(),
            )
        )
        is None
    )

    assert endpoint.tags == [0, 1, 2, 3, 4]
    assert endpoint.max_active == 2
    for key, expected in tensors.items():
        assert torch.equal(server.buffer[key].tensor, expected)
        assert server.buffer[key].version == 11


def test_gather_cancel_on_error_drains_siblings_before_raising():
    events = []

    async def wait_until_cancelled():
        events.append("started")
        try:
            await asyncio.Event().wait()
        finally:
            events.append("drained")

    async def fail():
        await asyncio.sleep(0)
        raise RuntimeError("transfer failed")

    async def run():
        with pytest.raises(RuntimeError, match="transfer failed"):
            await gather_cancel_on_error(wait_until_cancelled(), fail())
        assert events == ["started", "drained"]

    asyncio.run(run())


def test_chunked_receive_reports_tensor_progress_on_connection_reset():
    server = object.__new__(transport.UCXTensorServer)
    server.backend = "ucx"
    server.buffer = {}
    server.buffer_kwargs = {"device": torch.device("cpu"), "pin_memory": False}
    server.tensor_rwlocks = defaultdict(ReadWriteLock)
    header = transport.TensorTransferHeader(
        storage_metas=[
            transport.TensorStorageMeta(
                key="model.embed_tokens.weight",
                data_meta=TensorMeta(dtype=torch.float32, shape=torch.Size([5])),
                version=1,
            )
        ],
        mode=transport.TensorTransferMode.CHUNKED,
        payload_nbytes=20,
        chunk_bytes=16,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Relay chunked payload receive failed: key='model.embed_tokens.weight', "
            "offset=0, chunk_nbytes=16, payload_nbytes=20"
        ),
    ):
        asyncio.run(server._handle_chunked_tensor(_ResettingRecvEndpoint(), header))


def _range_meta(key, *, shape, version, byte_offset, dtype=torch.float32):
    return transport.TensorStorageMeta(
        key=key,
        data_meta=TensorMeta(dtype=dtype, shape=torch.Size(shape)),
        version=version,
        byte_offset=byte_offset,
    )


def test_sharded_publishers_write_byte_ranges_into_one_full_tensor():
    full_weight = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    full_bias = torch.tensor([10.0, 11.0, 12.0])
    server = object.__new__(transport.UCXTensorServer)
    server.backend = "ucx"
    server.buffer = {
        "dense.weight": transport.VersionedTensor(torch.zeros(3, 2)),
        "dense.bias": transport.VersionedTensor(torch.zeros(3)),
    }
    server.buffer_kwargs = {"device": torch.device("cpu"), "pin_memory": False}
    server.tensor_rwlocks = defaultdict(ReadWriteLock)
    storage = {key: entry.tensor for key, entry in server.buffer.items()}
    # Rank 0 sends a CHUNKED weight shard plus a PACKED bias shard; rank 1 packs both.
    rank_shards = [
        [
            (_range_meta("dense.weight", shape=(2, 2), version=7, byte_offset=0), full_weight[:2]),
            (_range_meta("dense.bias", shape=(2,), version=7, byte_offset=0), full_bias[:2]),
        ],
        [
            (_range_meta("dense.weight", shape=(1, 2), version=7, byte_offset=16), full_weight[2:]),
            (_range_meta("dense.bias", shape=(1,), version=7, byte_offset=8), full_bias[2:]),
        ],
    ]

    async def publish(shards, message_count):
        send_ep, recv_ep = _duplex_endpoint_pair()
        sender = transport.TensorChunkSender(send_ep, chunk_bytes=16)

        async def send():
            for storage_meta, tensor in shards:
                await sender.add(storage_meta, tensor)
            await sender.finish()

        async def receive():
            transfer_state = transport._TensorReceiveState()
            packed_recv_buffer = None
            for _ in range(message_count):
                packed_recv_buffer = await server._send_tensors_handler(
                    recv_ep, transfer_state=transfer_state, packed_recv_buffer=packed_recv_buffer
                )

        await asyncio.gather(send(), receive())

    asyncio.run(publish(rank_shards[0], message_count=3))
    assert [entry.version for entry in server.buffer.values()] == [transport.EMPTY_VERSION] * 2

    asyncio.run(publish(rank_shards[1], message_count=2))
    assert [entry.version for entry in server.buffer.values()] == [7, 7]
    assert torch.equal(server.buffer["dense.weight"].tensor, full_weight)
    assert torch.equal(server.buffer["dense.bias"].tensor, full_bias)
    assert all(server.buffer[key].tensor is tensor for key, tensor in storage.items())


def test_versioned_tensor_rejects_byte_ranges_that_do_not_tile():
    entry = transport.VersionedTensor(torch.zeros(4))
    entry.commit(_range_meta("w", shape=(2,), version=1, byte_offset=0))

    with pytest.raises(RuntimeError, match="do not tile 16 bytes"):
        entry.commit(_range_meta("w", shape=(2,), version=1, byte_offset=4))
    assert entry.version == transport.EMPTY_VERSION


def test_versioned_tensor_rejects_version_overtaking_incomplete_ranges():
    entry = transport.VersionedTensor(torch.zeros(4))
    entry.commit(_range_meta("w", shape=(2,), version=1, byte_offset=0))

    with pytest.raises(RuntimeError, match="version 2 overtook incomplete version 1"):
        entry.commit(_range_meta("w", shape=(2,), version=2, byte_offset=8))


def test_byte_range_destination_requires_allocated_in_bounds_storage():
    server = object.__new__(transport.UCXTensorServer)
    server.buffer = {"w": transport.VersionedTensor(torch.zeros(4))}
    server.buffer_kwargs = {"device": torch.device("cpu"), "pin_memory": False}

    with pytest.raises(AssertionError, match="out of bounds"):
        server._destination(_range_meta("w", shape=(2,), version=1, byte_offset=12))
    with pytest.raises(AssertionError, match="needs pre-allocated storage"):
        server._destination(_range_meta("missing", shape=(2,), version=1, byte_offset=8))
