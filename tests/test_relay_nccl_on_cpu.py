# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Exercise NCCL's real socket protocol with only CUDA/NCCL hardware replaced."""

import asyncio
import ctypes
import sys
import threading
from collections import defaultdict
from contextlib import asynccontextmanager, nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import ray
import torch

from laminar.train.relay import nccl, transport
from laminar.utils.concurrency import ReadWriteLock
from laminar.utils.tensor import TensorMeta


class FakeCudaTensor(torch.Tensor):
    """CPU storage with CUDA identity; no host conversion is permitted."""

    @property
    def is_cuda(self):
        return True

    @property
    def device(self):
        return torch.device("cuda", 0)

    def cpu(self, *args, **kwargs):
        raise AssertionError("GPU-direct send must not copy to CPU")

    def numpy(self, *args, **kwargs):
        raise AssertionError("GPU-direct send must not convert to NumPy")


@pytest.fixture
def hardware(monkeypatch):
    calls = []

    class UniqueId(ctypes.Structure):
        _fields_ = [("internal", ctypes.c_byte * 128)]

    class Library:
        def ncclGetUniqueId(self):
            uid = UniqueId()
            uid.internal[0] = 42
            return uid

        def ncclCommInitRank(self, size, uid, rank):
            calls.append(("init", size, uid.internal[0], rank))
            return object()

        def ncclCommDestroy(self, comm):
            calls.append(("destroy", comm))

    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.device_communicators.pynccl_wrapper",
        SimpleNamespace(NCCLLibrary=Library, ncclUniqueId=UniqueId),
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "Stream", lambda **_: object())
    empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: empty(*a, **{**kw, "device": "cpu"}))
    monkeypatch.setattr(nccl, "CHUNK_BYTES", 32)
    monkeypatch.setattr(nccl, "_IDLE", {})
    monkeypatch.setattr(
        nccl.NCCLEndpoint,
        "_transfer",
        lambda self, array, *, sending: calls.append(("transfer", self.rank, sending, array.nbytes)),
    )
    return calls


@asynccontextmanager
async def relay_server():
    accepted = asyncio.Queue()

    async def accept(reader, writer):
        await accepted.put(nccl.NCCLEndpoint(reader, writer, rank=1))

    server = await asyncio.start_server(accept, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    @asynccontextmanager
    async def connect():
        sender = await nccl.create_endpoint("127.0.0.1", port)
        receiver = await accepted.get()
        try:
            yield sender, receiver
        finally:
            await asyncio.wait_for(asyncio.gather(sender.close(), receiver.close()), timeout=5)

    try:
        yield connect
    finally:
        server.close()
        await server.wait_closed()


@asynccontextmanager
async def endpoints():
    async with relay_server() as connect, connect() as pair:
        yield pair


async def transfer(sender, receiver, nbytes=17, tag=0):
    await asyncio.wait_for(
        asyncio.gather(
            sender.send(np.zeros(nbytes, dtype=np.uint8), tag=tag),
            receiver.recv(np.zeros(nbytes, dtype=np.uint8), tag=tag),
        ),
        timeout=5,
    )


def test_socket_frames_bootstrap_once_and_keep_ack_boundary(hardware):
    async def run():
        async with endpoints() as (sender, receiver):
            await sender.send_obj(b"metadata\x00\xff")
            assert await receiver.recv_obj() == b"metadata\x00\xff"
            for tag in (0, 1):
                await transfer(sender, receiver, tag=tag)
            await receiver.send_obj(b"completion-ack")
            assert await sender.recv_obj() == b"completion-ack"

    asyncio.run(run())
    assert sorted(call[1:] for call in hardware if call[0] == "init") == [(2, 42, 0), (2, 42, 1)]
    assert len([call for call in hardware if call[0] == "transfer"]) == 4
    assert not [call for call in hardware if call[0] == "destroy"]


def test_reconnect_reuses_communicator_without_bootstrap(hardware):
    async def run():
        async with relay_server() as connect:
            comms = []
            for _ in range(3):
                async with connect() as (sender, receiver):
                    await transfer(sender, receiver)
                    comms.append((sender.comm, receiver.comm))
            return comms

    comms = asyncio.run(run())
    assert comms[0] == comms[1] == comms[2]
    assert sorted(call[1:] for call in hardware if call[0] == "init") == [(2, 42, 0), (2, 42, 1)]
    assert len([call for call in hardware if call[0] == "transfer"]) == 6


def test_completion_ack_timeout_survives_close_with_a_stalled_peer(hardware):
    async def run():
        async with endpoints() as (sender, receiver):
            await transfer(sender, receiver)
            comm = sender.comm

            async def publish():
                try:
                    # The live peer keeps its socket open without reading or acknowledging.
                    await transport.complete_tensor_transfer(sender, tensor_count=1, payload_nbytes=17, timeout_s=0.01)
                finally:
                    await sender.close()

            with pytest.raises(RuntimeError, match="completion acknowledgement"):
                await asyncio.wait_for(publish(), timeout=1)
            assert sender.failed
            assert sender.writer.is_closing()
            assert sender.comm is comm  # Preserve buffers that a failed transfer may still own.
            assert sender.idle_key not in nccl._IDLE

    asyncio.run(run())


def test_close_times_out_and_does_not_cache_a_stalled_communicator(hardware):
    async def run():
        async with endpoints() as (sender, receiver):
            await transfer(sender, receiver)
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(sender.close(timeout_s=0.01), timeout=1)
            assert sender.failed
            assert sender.writer.is_closing()
            assert sender.idle_key not in nccl._IDLE
            # Failed close is retryable without another peer handshake.
            await asyncio.wait_for(sender.close(timeout_s=0.01), timeout=1)

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["handshake", "socket"])
def test_cancelled_close_closes_socket_and_discards_communicator(hardware, monkeypatch, phase):
    async def run():
        async with endpoints() as (sender, receiver):
            await transfer(sender, receiver)
            if phase == "socket":
                await receiver.close()
            started = asyncio.Event()
            owner, method = (sender.reader, "read") if phase == "handshake" else (sender.writer, "wait_closed")
            original = getattr(owner, method)

            async def stalled(*args, **kwargs):
                started.set()
                result = await original(*args, **kwargs)
                if phase == "socket":
                    await asyncio.Event().wait()
                return result

            with monkeypatch.context() as patch:
                patch.setattr(owner, method, stalled)
                closing = asyncio.create_task(sender.close(timeout_s=1))
                await asyncio.wait_for(started.wait(), timeout=1)
                closing.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(closing, timeout=1)
            assert sender.failed
            assert sender.writer.is_closing()
            assert sender.idle_key not in nccl._IDLE
            await asyncio.wait_for(sender.close(timeout_s=0.01), timeout=1)

    asyncio.run(run())


def test_socket_cleanup_failure_does_not_replace_completion_ack_timeout(hardware, monkeypatch):
    async def run():
        async with endpoints() as (sender, receiver):
            await transfer(sender, receiver)
            wait_closed = sender.writer.wait_closed

            async def fail_wait_closed():
                await wait_closed()
                raise ConnectionResetError("socket cleanup failed")

            async def publish():
                try:
                    await transport.complete_tensor_transfer(sender, tensor_count=1, payload_nbytes=17, timeout_s=0.01)
                finally:
                    await sender.close(timeout_s=0.02)

            with monkeypatch.context() as patch:
                patch.setattr(sender.writer, "wait_closed", fail_wait_closed)
                with pytest.raises(RuntimeError, match="completion acknowledgement") as caught:
                    await asyncio.wait_for(publish(), timeout=1)
            assert any("socket cleanup failed" in note for note in caught.value.__notes__)
            assert sender.failed
            assert sender.writer.is_closing()
            assert sender.idle_key not in nccl._IDLE

    asyncio.run(run())


def test_concurrent_connections_never_share_a_communicator(hardware):
    async def run():
        async with relay_server() as connect:
            async with connect() as first, connect() as second:
                await asyncio.gather(transfer(*first), transfer(*second))
                assert first[0].comm is not second[0].comm
                assert first[1].comm is not second[1].comm

    asyncio.run(run())
    assert len([call for call in hardware if call[0] == "init"]) == 4


def test_failed_receiver_discards_communicator_and_reuse_fails_closed(hardware):
    async def run():
        async with relay_server() as connect:
            async with connect() as (sender, receiver):
                await transfer(sender, receiver)
                receiver.failed = True
            async with connect() as (sender, receiver):
                with pytest.raises(RuntimeError, match="does not hold"):
                    await transfer(sender, receiver)
                sender.failed = True
            async with connect() as (sender, receiver):
                await transfer(sender, receiver)

    asyncio.run(run())
    assert len([call for call in hardware if call[0] == "init"]) == 4


@pytest.mark.parametrize("nbytes,tag", [(3, 7), (4, 8)])
def test_payload_mismatch_rejected_before_nccl_bootstrap(hardware, nbytes, tag):
    async def run():
        async with endpoints() as (sender, receiver):
            await sender.send_obj(nccl._PAYLOAD.pack(nbytes, tag))
            with pytest.raises(RuntimeError, match="payload mismatch"):
                await receiver.recv(np.zeros(4, dtype=np.uint8), tag=7)
            assert receiver.failed

    asyncio.run(run())
    assert hardware == []


def test_cancelled_transfer_does_not_destroy_inflight_communicator(hardware, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def transfer(self, array, *, sending):
        started.set()
        try:
            assert release.wait(timeout=5), "test did not release the hardware transfer"
        finally:
            finished.set()

    monkeypatch.setattr(nccl.NCCLEndpoint, "_transfer", transfer)

    async def run():
        async with endpoints() as (sender, receiver):
            await asyncio.gather(sender._initialize(), receiver._initialize())
            task = asyncio.create_task(sender.send(np.zeros(4, dtype=np.uint8)))
            try:

                async def wait_started():
                    while not started.is_set():
                        await asyncio.sleep(0.001)

                await asyncio.wait_for(wait_started(), timeout=3)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert sender.failed
                failed_comm = sender.comm
                await sender.close()
                assert not finished.is_set()
                assert not any(call == ("destroy", failed_comm) for call in hardware)
            finally:
                release.set()
        assert not any(call == ("destroy", failed_comm) for call in hardware)

    asyncio.run(run())
    assert finished.is_set()


def test_backend_default_and_invalid_value(monkeypatch):
    monkeypatch.delenv("LAMINAR_RELAY_BACKEND", raising=False)
    assert transport.relay_backend() == "ucx"
    monkeypatch.setenv("LAMINAR_RELAY_BACKEND", "nccl")
    assert transport.relay_backend() == "nccl"
    monkeypatch.setenv("LAMINAR_RELAY_BACKEND", "typo")
    with pytest.raises(ValueError, match="Unsupported Relay backend"):
        transport.relay_backend()


def test_cuda_transfer_uses_source_pointers_after_producer_event(monkeypatch):
    calls = []
    ready = object()
    source = torch.arange(69, dtype=torch.uint8).as_subclass(FakeCudaTensor)
    endpoint = object.__new__(nccl.NCCLEndpoint)
    endpoint.device, endpoint.rank, endpoint.comm = 0, 0, object()
    endpoint.stream = SimpleNamespace(
        cuda_stream=123,
        wait_event=lambda event: calls.append(("wait", event)),
        synchronize=lambda: calls.append(("sync",)),
    )
    endpoint.nccl = SimpleNamespace(ncclSend=lambda ptr, count, *_: calls.append(("send", ptr.value, count)))
    # Deliberately no staging attribute: CUDA sends must use the source directly.
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.device_communicators.pynccl_wrapper",
        SimpleNamespace(ncclDataTypeEnum=SimpleNamespace(ncclUint8=1)),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(nccl, "CHUNK_BYTES", 32)
    endpoint._transfer(source, sending=True, ready=ready)
    assert calls == [
        ("wait", ready),
        ("send", source.data_ptr(), 32),
        ("sync",),
        ("send", source.data_ptr() + 32, 32),
        ("sync",),
        ("send", source.data_ptr() + 64, 5),
        ("sync",),
    ]


def test_cuda_socket_send_retains_source_until_transfer_and_ack(hardware, monkeypatch):
    producer_event = object()
    caller_thread = threading.get_ident()
    started, release = threading.Event(), threading.Event()
    order = []
    source = torch.arange(17, dtype=torch.uint8).as_subclass(FakeCudaTensor)

    def record_event():
        assert threading.get_ident() == caller_thread
        order.append("record")
        return producer_event

    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: SimpleNamespace(record_event=record_event))

    def transfer(self, array, *, sending, ready=None):
        if sending:
            assert ready is producer_event
            assert array.data_ptr() == source.data_ptr()
            order.append("transfer")
            started.set()
            assert release.wait(timeout=5)
            assert torch.equal(array.as_subclass(torch.Tensor), torch.arange(17, dtype=torch.uint8))
            order.append("synchronized")

    monkeypatch.setattr(nccl.NCCLEndpoint, "_transfer", transfer)
    monkeypatch.setenv("LAMINAR_RELAY_BACKEND", "nccl")

    async def run():
        async with endpoints() as (sender, receiver):
            original_send_obj = sender.send_obj

            async def send_obj(payload):
                assert order[0] == "record", "producer event must precede control-socket awaits"
                await original_send_obj(payload)

            monkeypatch.setattr(sender, "send_obj", send_obj)
            sending = asyncio.create_task(transport.send_tensor(sender, source, tag=7))
            receiving = asyncio.create_task(receiver.recv(np.zeros(17, dtype=np.uint8), tag=7))
            try:

                async def wait_started():
                    while not started.is_set():
                        await asyncio.sleep(0.001)

                await asyncio.wait_for(wait_started(), timeout=3)
                assert not sending.done(), "caller must not reuse source before NCCL completion"
                release.set()
                stats, _ = await asyncio.wait_for(asyncio.gather(sending, receiving), timeout=3)
                assert stats.staging_time_s == 0
                assert order == ["record", "transfer", "synchronized"]
                source.fill_(99)  # Reuse is safe only after send has returned.
                await receiver.send_obj(b"complete")
                assert await sender.recv_obj() == b"complete"
            finally:
                release.set()

    asyncio.run(run())


def test_cuda_chunk_sender_does_not_allocate_host_staging(monkeypatch):
    monkeypatch.setenv("LAMINAR_RELAY_BACKEND", "nccl")
    source = torch.arange(7, dtype=torch.uint8).as_subclass(FakeCudaTensor)
    sent = []

    class Endpoint:
        async def send(self, payload):
            assert payload.is_cuda
            sent.append(payload.data_ptr())

    sender = transport.TensorChunkSender(Endpoint(), chunk_bytes=32)
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: pytest.fail("unexpected CPU staging allocation"))
    asyncio.run(sender._send_payload(source))
    assert sent == [source.data_ptr()]
    assert sender._cpu_staging_buffer is None
    assert sender.staging_time_s == 0


@pytest.mark.parametrize("shard_device", [0, 3, 7])
def test_first_cuda_payload_binds_device_and_later_device_change_is_rejected(hardware, monkeypatch, shard_device):
    class ShardTensor(FakeCudaTensor):
        @property
        def device(self):
            return torch.device("cuda", shard_device)

    class OtherDeviceTensor(FakeCudaTensor):
        @property
        def device(self):
            return torch.device("cuda", (shard_device + 1) % 8)

    stream_devices = []
    monkeypatch.setattr(torch.cuda, "Stream", lambda *, device: stream_devices.append(device) or object())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: SimpleNamespace(record_event=lambda: object()))
    monkeypatch.setattr(nccl.NCCLEndpoint, "_transfer", lambda self, array, *, sending, ready=None: None)

    async def run():
        async with endpoints() as (sender, receiver):
            assert sender.device == 0  # Fresh auxiliary thread's default device.
            source = torch.arange(5, dtype=torch.uint8).as_subclass(ShardTensor)
            for tag in (0, 1):
                await asyncio.wait_for(
                    asyncio.gather(
                        sender.send(source, tag=tag),
                        receiver.recv(np.zeros(5, dtype=np.uint8), tag=tag),
                    ),
                    timeout=3,
                )
                assert sender.device == shard_device
                assert receiver.device == 0  # CPU receiver keeps its own device.
            comm = sender.comm
            sent_headers = []

            async def unexpected_header(payload):
                sent_headers.append(payload)

            monkeypatch.setattr(sender, "send_obj", unexpected_header)
            other = torch.zeros(5, dtype=torch.uint8).as_subclass(OtherDeviceTensor)
            with pytest.raises(ValueError, match="endpoint CUDA device"):
                await sender.send(other)
            assert sender.comm is comm
            assert sender.device == shard_device
            assert not sent_headers

    asyncio.run(run())
    assert sorted(stream_devices) == sorted([0, 0, shard_device, shard_device])


@pytest.mark.parametrize("sending", [True, False])
def test_host_transfer_alternates_staging_and_waits_for_buffer_release(monkeypatch, sending):
    calls = []

    class Stream:
        def __init__(self, name):
            self.name, self.cuda_stream, self.events = name, 0, 0

        def wait_event(self, event):
            calls.append(("wait", self.name, event))

        def record_event(self):
            self.events += 1
            return (self.name, self.events)

        def synchronize(self):
            calls.append(("sync", self.name))

    payload = torch.arange(80, dtype=torch.uint8)
    received = torch.zeros(80, dtype=torch.uint8)
    endpoint = object.__new__(nccl.NCCLEndpoint)
    endpoint.device, endpoint.rank, endpoint.comm = 0, 0, object()
    endpoint.stream, endpoint.copy_stream = Stream("nccl"), Stream("copy")
    endpoint.staging = torch.zeros(2, 32, dtype=torch.uint8)

    def send(ptr, count, *_):
        staging = next(row for row in endpoint.staging if row.data_ptr() == ptr.value)
        calls.append(("send", int(staging[0]), count))

    def recv(ptr, count, *_):
        index = next(i for i, row in enumerate(endpoint.staging) if row.data_ptr() == ptr.value)
        offset = sum(call[0] == "recv" for call in calls) * 32
        endpoint.staging[index][:count] = payload[offset : offset + count]
        calls.append(("recv", index, count))

    endpoint.nccl = SimpleNamespace(ncclSend=send, ncclRecv=recv)
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.device_communicators.pynccl_wrapper",
        SimpleNamespace(ncclDataTypeEnum=SimpleNamespace(ncclUint8=1)),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(nccl, "CHUNK_BYTES", 32)
    endpoint._transfer(payload if sending else received.numpy(), sending=sending)

    first, second = ("copy", "nccl") if sending else ("nccl", "copy")
    waits = [call for call in calls if call[0] == "wait"]
    assert waits == [
        ("wait", second, (first, 1)),
        ("wait", second, (first, 2)),
        ("wait", first, (second, 1)),
        ("wait", second, (first, 3)),
    ]
    assert calls[-1] == ("sync", second)
    if sending:
        assert [call for call in calls if call[0] == "send"] == [("send", 0, 32), ("send", 32, 32), ("send", 64, 16)]
    else:
        assert [call for call in calls if call[0] == "recv"] == [("recv", 0, 32), ("recv", 1, 32), ("recv", 0, 16)]
        assert torch.equal(received, payload)


def test_tagged_receive_is_ordered_and_failure_does_not_publish_version():
    server = object.__new__(transport.UCXTensorServer)
    server.backend = "nccl"
    server.buffer_kwargs = {"device": "cpu"}
    server.buffer = {}
    server.tensor_rwlocks = defaultdict(ReadWriteLock)
    metas = [transport.TensorStorageMeta(str(i), TensorMeta(torch.float32, torch.Size([2])), 9) for i in range(4)]
    header = transport.TensorTransferHeader(metas, transport.TensorTransferMode.TAGGED, window_size=3)
    entered = []
    active = 0

    class Receiver:
        async def recv(self, destination, *, tag):
            nonlocal active
            active += 1
            assert active == 1, "NCCL payloads must not be received concurrently"
            entered.append(tag)
            await asyncio.sleep(0)
            destination[:] = tag
            active -= 1
            if tag == 2:
                raise RuntimeError("incomplete D2H")

    with pytest.raises(RuntimeError, match="incomplete D2H"):
        asyncio.run(server._handle_tagged_tensors(Receiver(), header))
    assert entered == [0, 1, 2]
    assert server.buffer["0"].version == server.buffer["1"].version == 9
    assert server.buffer["2"].version == transport.EMPTY_VERSION
    assert "3" not in server.buffer


@pytest.mark.parametrize("node_ip", ["[::1]", "::1", "127.0.0.1"])
def test_node_address_binds_real_control_socket_without_cuda(monkeypatch, node_ip):
    monkeypatch.setenv("LAMINAR_RELAY_BACKEND", "nccl")
    monkeypatch.setattr(ray.util, "get_node_ip_address", lambda: node_ip)
    # A socket-only control round trip must not construct CUDA/NCCL state.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    async def run():
        server = transport.UCXTensorServer(buffer_kwargs={"device": "cpu"})
        endpoint = None
        try:
            listener = await asyncio.wait_for(server.wait_until_ready(), timeout=3)
            assert server.ip == node_ip.strip("[]")
            endpoint = await transport.create_endpoint(server.ip, listener.port)
            await transport.complete_tensor_transfer(endpoint, tensor_count=0, payload_nbytes=0)
            assert endpoint.comm is None
        finally:
            if endpoint is not None:
                await endpoint.close()
            server._server_task.cancel()
            await asyncio.gather(server._server_task, return_exceptions=True)

    asyncio.run(run())
