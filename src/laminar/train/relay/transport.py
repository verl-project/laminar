# Copyright 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Versioned tensor protocol with UCX or NCCL transport for Relay weight sync.

The :class:`UCXTensorServer` accepts packed, chunked, or tagged transfers that
update its CPU buffer in-place. The companion ``send_tensors`` / ``send_obj`` /
``send_tensor`` functions are the client side; actor publishers and Relay
chain hops use this protocol to populate each per-node Relay snapshot.
"""

from __future__ import annotations

import asyncio
import errno
import logging
import os
import pickle
import re
import socket
import subprocess
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Hashable, Iterable

import psutil
import torch
import ucxx
from ucxx.exceptions import UCXCanceledError, UCXConnectionResetError
from verl.utils.net_utils import get_free_port

from laminar.utils.concurrency import ReadWriteLock, gather_cancel_on_error
from laminar.utils.tensor import TensorMeta

if TYPE_CHECKING:
    from ucxx._lib_async import Endpoint, Listener


logger = logging.getLogger(__name__)
DEFAULT_TENSOR_TRANSFER_CHUNK_BYTES = 64 * 1024**2
DEFAULT_TENSOR_TRANSFER_IO_TIMEOUT_S = 120.0
DEFAULT_TENSOR_TRANSFER_ACK_TIMEOUT_S = 120.0
_UCX_JOB_RUNTIME_ENV_KEYS = (
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
)


def relay_backend() -> str:
    backend = os.environ.get("LAMINAR_RELAY_BACKEND", "ucx")
    if backend not in {"ucx", "nccl"}:
        raise ValueError(f"Unsupported Relay backend: {backend!r}")
    return backend


async def create_endpoint(ip: str, port: int) -> Any:
    if relay_backend() == "nccl":
        from laminar.train.relay.nccl import create_endpoint as create_nccl_endpoint

        return await create_nccl_endpoint(ip, port)
    return await ucxx.create_endpoint(ip, port)


def ucx_job_runtime_env_vars_from_env() -> dict[str, str]:
    """Return UCX-related env vars for the Ray job runtime environment.

    Platform Ray heads may define their own UCX defaults. The async-RL driver
    can override them for single-node TCP fallback runs. Publishing the
    overrides once at job scope makes every transport participant inherit the
    same policy without component-level copies.
    """
    return {key: value for key in _UCX_JOB_RUNTIME_ENV_KEYS if (value := os.environ.get(key)) is not None}


@dataclass(frozen=True)
class TensorStorageMeta:
    """Per-payload envelope sent ahead of the bytes on the wire.

    Carries the destination ``key`` (used as the buffer dict key on the
    receiver), the payload's ``data_meta`` (shape + dtype for sanity-check),
    and the actor weight ``version`` committed after its payload arrives.
    A sharded publisher sends each shard as the contiguous byte range of the
    full ``key`` tensor starting at ``byte_offset``.
    """

    key: Hashable
    data_meta: TensorMeta
    version: int
    byte_offset: int = 0


class TensorTransferMode(Enum):
    PACKED = "packed"
    CHUNKED = "chunked"
    TAGGED = "tagged"


@dataclass
class TensorTransferHeader:
    """Describe one packed, bounded-chunk, or tagged transfer."""

    storage_metas: list[TensorStorageMeta]
    mode: TensorTransferMode
    payload_nbytes: int = 0
    chunk_bytes: int = 0
    window_size: int = 0


@dataclass(frozen=True)
class TensorTransferComplete:
    """Sender-side fence after every payload in one logical transfer."""

    tensor_count: int
    payload_nbytes: int


@dataclass(frozen=True)
class TensorTransferCompleteAck:
    """Receiver acknowledgement after all preceding payloads are committed."""

    tensor_count: int
    payload_nbytes: int


@dataclass
class _TensorReceiveState:
    """Receiver-observed progress for one logical tensor transfer."""

    tensor_count: int = 0
    payload_nbytes: int = 0


@dataclass(frozen=True)
class TensorTransferStats:
    """Canonical sender-side statistics for one completed tensor transfer."""

    chunk_count: int = 0
    payload_bytes: int = 0
    tensor_prepare_time_s: float = 0.0
    pack_time_s: float = 0.0
    staging_time_s: float = 0.0
    transport_time_s: float = 0.0
    completion_ack_time_s: float = 0.0


@dataclass(frozen=True)
class TensorSendStats:
    """Timing for one tensor payload send."""

    staging_time_s: float = 0.0
    transport_time_s: float = 0.0


# Sentinel for "no trainer / tensor version has been published yet".
EMPTY_VERSION = -1


@dataclass
class VersionedTensor:
    """One Relay tensor and the latest version written over all of its bytes.

    Payloads may cover disjoint byte ranges (one per actor shard); ``version``
    only advances once the ranges committed for a version tile the tensor.
    Bytes are overwritten in-place before that label advances; RelayWorker's
    publish fence keeps rollout readers out until the full publish commits.
    """

    tensor: torch.Tensor
    version: int = EMPTY_VERSION
    _pending_version: int = field(default=EMPTY_VERSION, repr=False)
    _pending_ranges: list[tuple[int, int]] = field(default_factory=list, repr=False)
    _pending_nbytes: int = field(default=0, repr=False)

    def commit(self, storage_meta: TensorStorageMeta) -> None:
        """Record that the byte range described by ``storage_meta`` has landed."""
        version = storage_meta.version
        if self._pending_ranges and version != self._pending_version:
            raise RuntimeError(f"Relay tensor version {version} overtook incomplete version {self._pending_version}")
        self._pending_version = version
        self._pending_ranges.append((storage_meta.byte_offset, storage_meta.data_meta.nbytes))
        self._pending_nbytes += storage_meta.data_meta.nbytes
        if self._pending_nbytes < self.tensor.nbytes:
            return
        ranges = sorted(self._pending_ranges)
        self._pending_ranges, self._pending_nbytes = [], 0
        ends = [offset + nbytes for offset, nbytes in ranges]
        if [offset for offset, _ in ranges] != [0, *ends[:-1]] or ends[-1] != self.tensor.nbytes:
            raise RuntimeError(
                f"Relay byte ranges of version {version} do not tile {self.tensor.nbytes} bytes: {ranges}"
            )
        self.version = version


class UCXTensorServer:
    """In-place tensor receiver shared by the UCX and NCCL Relay backends.

    Each connection carries an ordered stream of transfer headers and
    completion fences. Transfers for different keys may proceed concurrently;
    per-key write locks serialize all ranges targeting the same key, while
    :class:`VersionedTensor` range accounting exposes a version only after its
    full tensor has arrived.
    """

    DEFAULT_INTERVAL_S: float = 1e-3
    MAX_LISTENER_BIND_ATTEMPTS: int = 32

    def __init__(
        self,
        buffer_kwargs: dict[str, Any],
    ) -> None:
        self.backend = relay_backend()
        if self.backend == "nccl":
            import ray

            # Ray may return URL-style brackets; socket APIs need the raw IPv6 host.
            self.ip = ray.util.get_node_ip_address().strip("[]")
        else:
            self.ip = get_ucx_ip()

        # Per-tensor storage.
        self.buffer: dict[Hashable, VersionedTensor] = {}
        self.buffer_kwargs = buffer_kwargs
        # Per-tensor RW locks so writes & reads on different keys never
        # serialize against each other.
        self.tensor_rwlocks: dict[Hashable, ReadWriteLock] = defaultdict(ReadWriteLock)
        self._server_error: BaseException | None = None

        self._server_task = asyncio.create_task(self.run_server(), name="relay.ucx_tensor_server")
        self._server_task.add_done_callback(self._record_server_task_result)

    def _record_server_task_result(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self._server_error = exc
            logger.error(
                "UCX tensor server task exited before serving",
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    @staticmethod
    def _is_listener_bind_retryable(exc: BaseException) -> bool:
        busy_error = getattr(ucxx, "UCXXBusyError", None)
        if busy_error is not None and isinstance(exc, busy_error):
            return True
        if isinstance(exc, OSError) and exc.errno == errno.EADDRINUSE:
            return True
        message = str(exc).lower()
        return "address already in use" in message or "device is busy" in message

    async def wait_until_ready(self, interval_s: float | None = None) -> Listener:
        """Wait for the UCX listener and surface startup failures."""
        if interval_s is None:
            interval_s = self.DEFAULT_INTERVAL_S
        while True:
            listener = getattr(self, "lf", None)
            if listener is not None:
                return listener
            if self._server_error is not None:
                raise RuntimeError("UCX tensor server failed to start") from self._server_error
            if self._server_task.done():
                if self._server_task.cancelled():
                    raise RuntimeError("UCX tensor server was cancelled before listener was ready")
                exc = self._server_task.exception()
                if exc is not None:
                    self._server_error = exc
                    raise RuntimeError("UCX tensor server failed to start") from exc
                raise RuntimeError("UCX tensor server exited before listener was ready")
            await asyncio.sleep(interval_s)

    async def run_server(self) -> None:
        if self.backend == "nccl":
            from types import SimpleNamespace

            from laminar.train.relay.nccl import NCCLEndpoint

            async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                endpoint = NCCLEndpoint(reader, writer, rank=1)
                try:
                    await self._handler(endpoint)
                finally:
                    await endpoint.close()

            listener = await asyncio.start_server(accept, self.ip, 0)
            self.lf = SimpleNamespace(ip=self.ip, port=listener.sockets[0].getsockname()[1])
            async with listener:
                await listener.serve_forever()
            return
        for attempt in range(1, self.MAX_LISTENER_BIND_ATTEMPTS + 1):
            port, _ = get_free_port(self.ip)
            try:
                self.lf = ucxx.create_listener(self._handler, port=port)
                break
            except Exception as exc:
                if not self._is_listener_bind_retryable(exc) or attempt >= self.MAX_LISTENER_BIND_ATTEMPTS:
                    self._server_error = exc
                    raise
                logger.warning(
                    "UCX listener bind failed on auto-selected port %s (attempt %d/%d), retrying with a new port: %r",
                    port,
                    attempt,
                    self.MAX_LISTENER_BIND_ATTEMPTS,
                    exc,
                )
                await asyncio.sleep(self.DEFAULT_INTERVAL_S)
        logger.info(f"Created listener on {self.lf.ip=}, {self.lf.port=}")
        # create_listener starts UCXX's own progress machinery. Keep this
        # task alive until cancellation without periodic wakeups.
        await asyncio.get_running_loop().create_future()

    async def _handler(self, ep: Endpoint) -> None:
        """Each ``ep`` corresponds to a client, supporting concurrent writes natively."""
        packed_recv_buffer = None
        transfer_state = _TensorReceiveState()
        try:
            while True:
                packed_recv_buffer = await self._send_tensors_handler(
                    ep,
                    packed_recv_buffer=packed_recv_buffer,
                    transfer_state=transfer_state,
                )
        except (UCXCanceledError, UCXConnectionResetError) as e:
            logger.debug("UCXTensorServer client disconnected: %s", e)
            await ep.close()
        except asyncio.IncompleteReadError:
            if self.backend != "nccl":
                raise
        except Exception as e:
            trace = traceback.format_exc()
            message = f"Relay transport fatal error: {type(e).__name__}: {e}\n{trace}"
            # Ray actor logging configuration is not guaranteed to forward
            # module loggers. Flush the fatal receiver cause explicitly so a
            # sender-side connection reset never hides the original failure.
            print(message, flush=True)
            logger.error(message)
            await ep.close()

    async def _send_tensors_handler(
        self,
        ep: Endpoint,
        *,
        transfer_state: _TensorReceiveState,
        packed_recv_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Handle one transfer header and its optional tensor payload."""
        message = pickle.loads(await ep.recv_obj())
        if isinstance(message, TensorTransferComplete):
            actual_tensor_count = transfer_state.tensor_count
            actual_payload_nbytes = transfer_state.payload_nbytes
            mismatch = None
            if actual_tensor_count != message.tensor_count or actual_payload_nbytes != message.payload_nbytes:
                mismatch = (
                    "Relay tensor transfer completion mismatch: "
                    f"received_tensors={actual_tensor_count} expected_tensors={message.tensor_count} "
                    f"received_bytes={actual_payload_nbytes} expected_bytes={message.payload_nbytes}"
                )
            await send_obj(
                ep,
                TensorTransferCompleteAck(
                    tensor_count=actual_tensor_count,
                    payload_nbytes=actual_payload_nbytes,
                ),
            )
            if mismatch is not None:
                raise RuntimeError(mismatch)
            transfer_state.tensor_count = 0
            transfer_state.payload_nbytes = 0
            return packed_recv_buffer

        header = message
        assert isinstance(header, TensorTransferHeader), f"Unexpected Relay transfer header: {type(header)}"
        storage_metas = header.storage_metas
        assert storage_metas, "Relay transfer header must contain at least one tensor"
        logger.info(f"Received {len(storage_metas)=}, {storage_metas[0]=}")
        if header.mode is TensorTransferMode.TAGGED:
            await self._handle_tagged_tensors(ep, header)
        elif header.mode is TensorTransferMode.PACKED:
            packed_recv_buffer = await self._handle_packed_tensors(
                ep,
                header,
                packed_recv_buffer=packed_recv_buffer,
            )
        elif header.mode is TensorTransferMode.CHUNKED:
            await self._handle_chunked_tensor(ep, header)
        else:
            raise RuntimeError(f"Unexpected Relay transfer mode: {header.mode!r}")
        transfer_state.tensor_count += len(storage_metas)
        transfer_state.payload_nbytes += sum(meta.data_meta.nbytes for meta in storage_metas)
        logger.info(f"Handled {len(storage_metas)=}, {storage_metas[0]=}")
        return packed_recv_buffer

    async def _handle_tagged_tensors(
        self,
        ep: Endpoint,
        header: TensorTransferHeader,
    ) -> None:
        """Receive direct tensors concurrently without copying them through a pack buffer."""
        assert header.window_size > 0, f"Tagged Relay payload requires a positive window: {header.window_size}"
        for start in range(0, len(header.storage_metas), header.window_size):
            stop = min(start + header.window_size, len(header.storage_metas))
            if self.backend == "nccl":
                for index in range(start, stop):
                    await self._handle_sent_tensor(ep, header.storage_metas[index], tag=index)
                continue
            await gather_cancel_on_error(
                *(self._handle_sent_tensor(ep, header.storage_metas[index], tag=index) for index in range(start, stop))
            )

    def _destination(self, storage_meta: TensorStorageMeta) -> tuple[VersionedTensor, torch.Tensor]:
        """Return the entry ``storage_meta`` writes and the byte view its payload lands in."""
        key = storage_meta.key
        data_meta = storage_meta.data_meta
        offset = storage_meta.byte_offset
        if key not in self.buffer:
            assert offset == 0, f"Relay byte-range write needs pre-allocated storage: {key=} {offset=}"
            tensor = torch.empty(data_meta.shape, dtype=data_meta.dtype, **self.buffer_kwargs)  # type: ignore[call-overload]
            self.buffer[key] = VersionedTensor(tensor)
            logger.info(f"Created storage for {key=} with {data_meta=}, now {len(self.buffer)=}")
        entry = self.buffer[key]
        tensor = entry.tensor
        assert tensor.dtype == data_meta.dtype, f"Tensor dtype mismatch for {key=}: {tensor.dtype=} != {data_meta=}"
        assert 0 <= offset and offset + data_meta.nbytes <= tensor.nbytes, (
            f"Tensor byte range out of bounds for {key=}: {offset=} {data_meta=} {tensor.nbytes=}"
        )
        assert data_meta.nbytes < tensor.nbytes or tensor.shape == data_meta.shape, (
            f"Tensor shape mismatch for {key=}: {tensor.shape=} != {data_meta.shape=}"
        )
        return entry, tensor.detach().reshape(-1).view(torch.uint8)[offset : offset + data_meta.nbytes]

    async def _handle_packed_tensors(
        self,
        ep: Endpoint,
        header: TensorTransferHeader,
        packed_recv_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        storage_metas = header.storage_metas
        expected_nbytes = sum(meta.data_meta.nbytes for meta in storage_metas)
        assert header.payload_nbytes == expected_nbytes, (
            f"Packed Relay payload size mismatch: {header.payload_nbytes=} != {expected_nbytes=}"
        )
        destinations = [self._destination(meta) for meta in storage_metas]
        if packed_recv_buffer is None or packed_recv_buffer.numel() < header.payload_nbytes:
            capacity = max(DEFAULT_TENSOR_TRANSFER_CHUNK_BYTES, header.payload_nbytes)
            packed_recv_buffer = torch.empty(capacity, dtype=torch.uint8, **self.buffer_kwargs)  # type: ignore[call-overload]
        packed = packed_recv_buffer[: header.payload_nbytes]
        try:
            await ep.recv(packed.numpy())
        except (UCXCanceledError, UCXConnectionResetError) as exc:
            raise RuntimeError(
                "Relay packed payload receive failed: "
                f"tensor_count={len(storage_metas)}, payload_nbytes={header.payload_nbytes}"
            ) from exc
        offset = 0
        for storage_meta, (entry, destination) in zip(storage_metas, destinations, strict=True):
            nbytes = destination.numel()
            async with self.tensor_rwlocks[storage_meta.key].write_lock():
                destination.copy_(packed[offset : offset + nbytes])
                entry.commit(storage_meta)
            offset += nbytes
        assert offset == header.payload_nbytes
        return packed_recv_buffer

    async def _handle_chunked_tensor(
        self,
        ep: Endpoint,
        header: TensorTransferHeader,
    ) -> None:
        assert len(header.storage_metas) == 1, "Chunked Relay payload must contain exactly one tensor"
        storage_meta = header.storage_metas[0]
        expected_nbytes = storage_meta.data_meta.nbytes
        assert header.payload_nbytes == expected_nbytes, (
            f"Chunked Relay payload size mismatch: {header.payload_nbytes=} != {expected_nbytes=}"
        )
        assert 0 < header.chunk_bytes <= header.payload_nbytes, (
            f"Invalid chunked Relay payload size: {header.chunk_bytes=}, {header.payload_nbytes=}"
        )

        entry, tensor_bytes = self._destination(storage_meta)
        async with self.tensor_rwlocks[storage_meta.key].write_lock():
            for offset in range(0, header.payload_nbytes, header.chunk_bytes):
                chunk = tensor_bytes[offset : min(offset + header.chunk_bytes, header.payload_nbytes)]
                try:
                    await ep.recv(chunk.numpy())
                except (UCXCanceledError, UCXConnectionResetError) as exc:
                    raise RuntimeError(
                        "Relay chunked payload receive failed: "
                        f"key={storage_meta.key!r}, offset={offset}, chunk_nbytes={chunk.numel()}, "
                        f"payload_nbytes={header.payload_nbytes}"
                    ) from exc
            entry.commit(storage_meta)

    async def _handle_sent_tensor(
        self,
        ep: Endpoint,
        storage_meta: TensorStorageMeta,
        tag: int | None = None,
    ) -> None:
        key = storage_meta.key

        entry, destination = self._destination(storage_meta)

        async with self.tensor_rwlocks[key].write_lock():
            try:
                payload = destination.numpy()
                if tag is None:
                    await ep.recv(payload)
                else:
                    await ep.recv(payload, tag=tag)
            except (UCXCanceledError, UCXConnectionResetError) as exc:
                raise RuntimeError(
                    f"Relay direct payload receive failed: key={key!r}, payload_nbytes={storage_meta.data_meta.nbytes}"
                ) from exc
            # Advance the version only after all bytes land. A partial recv
            # can overwrite old bytes without changing this label; rollout
            # reads remain protected by RelayWorker's publish fence.
            entry.commit(storage_meta)


async def send_tensors(
    ep: Endpoint,
    storage_metas: list[TensorStorageMeta],
    tensors: Iterable[torch.Tensor],
) -> TensorTransferStats:
    """Pack or chunk a tensor batch, then wait for receiver-side commit.

    Small tensors share bounded ``PACKED`` payloads; tensors at least one
    chunk wide use one ``CHUNKED`` header followed by bounded payload chunks.
    ``TensorChunkSender.finish`` closes the logical transfer with a completion
    acknowledgement covering the aggregate tensor and byte counts.
    """
    sender = TensorChunkSender(ep)
    for storage_meta, tensor in zip(storage_metas, tensors, strict=True):
        await sender.add(storage_meta, tensor)
    return await sender.finish()


class TensorChunkSender:
    """Pack small tensors and segment large tensors into bounded UCX messages."""

    def __init__(self, ep: Endpoint, chunk_bytes: int = DEFAULT_TENSOR_TRANSFER_CHUNK_BYTES) -> None:
        if chunk_bytes <= 0:
            raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes}")
        self.ep = ep
        self.chunk_bytes = chunk_bytes
        self._buffer: torch.Tensor | None = None
        self._cpu_staging_buffer: torch.Tensor | None = None
        self._used = 0
        self._storage_metas: list[TensorStorageMeta] = []
        self.chunk_count = 0
        self.tensor_count = 0
        self.payload_bytes = 0
        self.tensor_prepare_time_s = 0.0
        self.pack_time_s = 0.0
        self.staging_time_s = 0.0
        self.transport_time_s = 0.0
        self.completion_ack_time_s = 0.0

    async def add(self, storage_meta: TensorStorageMeta, tensor: torch.Tensor) -> None:
        prepare_start = time.perf_counter()
        data_tensor = tensor.to(storage_meta.data_meta.dtype).detach().contiguous()
        data_bytes = data_tensor.view(torch.uint8).reshape(-1)
        self.tensor_prepare_time_s += time.perf_counter() - prepare_start
        nbytes = data_bytes.numel()
        self.tensor_count += 1
        self.payload_bytes += nbytes
        expected_nbytes = storage_meta.data_meta.nbytes
        assert nbytes == expected_nbytes, f"Tensor byte size mismatch: {nbytes=} != {expected_nbytes=}"

        if nbytes >= self.chunk_bytes:
            await self.flush()
            await self._send_header(
                TensorTransferHeader(
                    storage_metas=[storage_meta],
                    mode=TensorTransferMode.CHUNKED,
                    payload_nbytes=nbytes,
                    chunk_bytes=self.chunk_bytes,
                )
            )
            for offset in range(0, nbytes, self.chunk_bytes):
                await self._send_payload(data_bytes[offset : min(offset + self.chunk_bytes, nbytes)])
                self.chunk_count += 1
            return

        if self._buffer is not None and self._buffer.device != data_tensor.device:
            await self.flush()
            self._buffer = None
        if self._used + nbytes > self.chunk_bytes:
            await self.flush()
        pack_start = time.perf_counter()
        if self._buffer is None:
            buffer_kwargs: dict[str, Any] = {"device": data_tensor.device}
            if data_tensor.device.type == "cpu":
                buffer_kwargs["pin_memory"] = data_tensor.is_pinned()
            self._buffer = torch.empty(self.chunk_bytes, dtype=torch.uint8, **buffer_kwargs)

        self._buffer[self._used : self._used + nbytes].copy_(data_bytes)
        self.pack_time_s += time.perf_counter() - pack_start
        self._used += nbytes
        self._storage_metas.append(storage_meta)

    async def flush(self) -> None:
        if not self._storage_metas:
            return
        assert self._buffer is not None and self._used > 0
        await self._send_header(
            TensorTransferHeader(
                storage_metas=self._storage_metas,
                mode=TensorTransferMode.PACKED,
                payload_nbytes=self._used,
            )
        )
        await self._send_payload(self._buffer[: self._used])
        self.chunk_count += 1
        self._used = 0
        self._storage_metas = []

    async def _send_header(self, header: TensorTransferHeader) -> None:
        transport_start = time.perf_counter()
        await send_obj(self.ep, header)
        self.transport_time_s += time.perf_counter() - transport_start

    async def _send_payload(self, tensor: torch.Tensor) -> None:
        staging_time_s = 0.0
        if tensor.device.type == "cuda" and relay_backend() != "nccl":
            staging_start = time.perf_counter()
            if self._cpu_staging_buffer is None or self._cpu_staging_buffer.numel() < tensor.numel():
                self._cpu_staging_buffer = torch.empty(
                    max(self.chunk_bytes, tensor.numel()),
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=True,
                )
            staging_time_s = time.perf_counter() - staging_start
        send_stats = await send_tensor(
            self.ep,
            tensor,
            cpu_staging_buffer=self._cpu_staging_buffer,
        )
        self.staging_time_s += staging_time_s + send_stats.staging_time_s
        self.transport_time_s += send_stats.transport_time_s

    async def finish(self) -> TensorTransferStats:
        await self.flush()
        completion_start = time.perf_counter()
        await complete_tensor_transfer(
            self.ep,
            tensor_count=self.tensor_count,
            payload_nbytes=self.payload_bytes,
        )
        self.completion_ack_time_s = time.perf_counter() - completion_start
        return TensorTransferStats(
            chunk_count=self.chunk_count,
            payload_bytes=self.payload_bytes,
            tensor_prepare_time_s=self.tensor_prepare_time_s,
            pack_time_s=self.pack_time_s,
            staging_time_s=self.staging_time_s,
            transport_time_s=self.transport_time_s,
            completion_ack_time_s=self.completion_ack_time_s,
        )


async def complete_tensor_transfer(
    ep: Endpoint,
    *,
    tensor_count: int,
    payload_nbytes: int,
    timeout_s: float = DEFAULT_TENSOR_TRANSFER_ACK_TIMEOUT_S,
) -> None:
    """Fence endpoint teardown on receiver-side payload commit."""
    if tensor_count < 0 or payload_nbytes < 0:
        raise ValueError(f"Relay tensor transfer counts must be non-negative, got {tensor_count=} {payload_nbytes=}")
    await send_obj(
        ep,
        TensorTransferComplete(
            tensor_count=tensor_count,
            payload_nbytes=payload_nbytes,
        ),
    )
    try:
        raw_ack = await asyncio.wait_for(ep.recv_obj(), timeout=timeout_s)
    except TimeoutError as exc:
        raise RuntimeError(
            "Timed out waiting for Relay tensor transfer completion acknowledgement: "
            f"tensor_count={tensor_count} payload_nbytes={payload_nbytes} timeout_s={timeout_s}"
        ) from exc
    ack = pickle.loads(raw_ack)
    if not isinstance(ack, TensorTransferCompleteAck):
        raise RuntimeError(f"Unexpected Relay tensor transfer completion acknowledgement: {type(ack)}")
    if ack.tensor_count != tensor_count or ack.payload_nbytes != payload_nbytes:
        raise RuntimeError(
            "Relay tensor transfer completion acknowledgement mismatch: "
            f"ack_tensors={ack.tensor_count} expected_tensors={tensor_count} "
            f"ack_bytes={ack.payload_nbytes} expected_bytes={payload_nbytes}"
        )


async def _await_ucx_send(
    send: Any,
    *,
    kind: str,
    payload_nbytes: int,
    tag: int | None,
    timeout_s: float,
) -> None:
    try:
        await asyncio.wait_for(send, timeout=timeout_s)
    except TimeoutError as exc:
        raise RuntimeError(
            "Timed out during Relay UCX send: "
            f"kind={kind} payload_nbytes={payload_nbytes} tag={tag} timeout_s={timeout_s}"
        ) from exc


async def send_obj(
    ep: Endpoint,
    obj: object,
    *,
    timeout_s: float = DEFAULT_TENSOR_TRANSFER_IO_TIMEOUT_S,
) -> None:
    if timeout_s <= 0:
        raise ValueError(f"Relay UCX send timeout must be positive, got {timeout_s}")
    payload = pickle.dumps(obj)
    await _await_ucx_send(
        ep.send_obj(payload),
        kind="object",
        payload_nbytes=len(payload),
        tag=None,
        timeout_s=timeout_s,
    )


async def send_tensor(
    ep: Endpoint,
    tensor: torch.Tensor,
    *,
    cpu_staging_buffer: torch.Tensor | None = None,
    tag: int | None = None,
    timeout_s: float = DEFAULT_TENSOR_TRANSFER_IO_TIMEOUT_S,
) -> TensorSendStats:
    if timeout_s <= 0:
        raise ValueError(f"Relay UCX send timeout must be positive, got {timeout_s}")
    if tensor.device.type == "cpu":
        transport_start = time.perf_counter()
        payload = tensor.detach().contiguous().view(torch.uint8).numpy()
        send = ep.send(payload) if tag is None else ep.send(payload, tag=tag)
        await _await_ucx_send(
            send,
            kind="tensor",
            payload_nbytes=payload.nbytes,
            tag=tag,
            timeout_s=timeout_s,
        )
        return TensorSendStats(transport_time_s=time.perf_counter() - transport_start)

    if relay_backend() == "nccl":
        transport_start = time.perf_counter()
        payload = tensor.detach().contiguous().view(torch.uint8)
        send = ep.send(payload) if tag is None else ep.send(payload, tag=tag)
        await _await_ucx_send(
            send,
            kind="tensor",
            payload_nbytes=payload.numel(),
            tag=tag,
            timeout_s=timeout_s,
        )
        return TensorSendStats(transport_time_s=time.perf_counter() - transport_start)

    # TODO: GPU-direct UCX send once expandable-segment CUDA tensors can be
    # registered. Relay already lands on a CPU buffer; staging to pinned host
    # memory is the production path because UCX CUDA export fails with the
    # actor allocator.
    staging_start = time.perf_counter()
    data_tensor = tensor.detach().contiguous().view(torch.uint8)
    if cpu_staging_buffer is None:
        data = data_tensor.cpu().numpy()
    else:
        assert cpu_staging_buffer.device.type == "cpu"
        assert cpu_staging_buffer.dtype == torch.uint8
        assert cpu_staging_buffer.is_pinned()
        assert cpu_staging_buffer.numel() >= data_tensor.numel()
        staged = cpu_staging_buffer[: data_tensor.numel()]
        staged.copy_(data_tensor, non_blocking=True)
        _synchronize_current_cuda_stream(data_tensor)
        data = staged.numpy()
    staging_time_s = time.perf_counter() - staging_start
    transport_start = time.perf_counter()
    send = ep.send(data) if tag is None else ep.send(data, tag=tag)
    await _await_ucx_send(
        send,
        kind="tensor",
        payload_nbytes=data_tensor.numel(),
        tag=tag,
        timeout_s=timeout_s,
    )
    return TensorSendStats(
        staging_time_s=staging_time_s,
        transport_time_s=time.perf_counter() - transport_start,
    )


def _synchronize_current_cuda_stream(tensor: torch.Tensor) -> None:
    if tensor.device.type != "cuda":
        return
    # Wait for the non-blocking D2H copy before exposing the pinned buffer to UCX.
    torch.cuda.current_stream(tensor.device).synchronize()


def get_first_rdma_interface_addresses() -> list | None:
    """Return addresses on the first RDMA netdev, or ``None`` if ``rdma`` is missing."""
    try:
        result = subprocess.run(["rdma", "link", "show"], capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    pattern = re.compile(r"link\s+\S+/\d+\s+.*?\s+netdev\s+(?P<net_iface>\S+)")
    for line in result.stdout.strip().split("\n"):
        match = pattern.search(line)
        if match:
            return psutil.net_if_addrs().get(match.group("net_iface"), [])
    return []


def get_ucx_ip() -> str:
    """Pick an IP for UCX to bind to.

    Priority: ``UCXX_IFNAME`` env var → UCX's own default discovery → first
    IPv6 address of the first RDMA interface. Fails closed if none of these
    yield an address.
    """
    ucx_ifname = os.getenv("UCXX_IFNAME", None)
    if ucx_ifname is not None:
        logger.info(f"Using custom UCXX_IFNAME: {ucx_ifname}")

    try:
        # Advertise the same address family as the configured UCXX listener.
        if ":" in os.getenv("UCXX_LISTENER_HOST", ""):
            ip = ucxx.get_address(ifname=ucx_ifname, use_ipv6=True)
        else:
            ip = ucxx.get_address(ifname=ucx_ifname)
    except OSError as exc:
        if exc.errno != errno.EADDRNOTAVAIL:
            raise
        ip = ucxx.get_address(ifname=ucx_ifname, use_ipv6=True)

    if ip is None:
        # Fallback: probe the RDMA-attached netdev directly.
        rdma_addrs = get_first_rdma_interface_addresses()
        if rdma_addrs:
            for addr in rdma_addrs:
                if addr.family == socket.AF_INET6:
                    logger.info(f"Found for UCX {addr=}")
                    ip = addr.address
                    break

    if ip is None and os.getenv("LAMINAR_RELAY_UCX_TCP_FALLBACK", "0") == "1":
        # Local smoke runs may not expose an RDMA netdev to UCX. Keep this
        # opt-in so production multi-node runs still fail fast when RDMA
        # discovery is broken.
        try:
            import ray

            ip = ray.util.get_node_ip_address()
        except Exception:  # noqa: BLE001
            ip = socket.gethostbyname(socket.gethostname())
        os.environ.setdefault("UCX_TLS", "shm,tcp")
        logger.warning(f"Falling back to UCX TCP transport on {ip=}")

    if ip is None:
        raise RuntimeError("No IP address found for UCX")
    logger.info(f"Found IP address {ip=} for UCX")
    return ip
