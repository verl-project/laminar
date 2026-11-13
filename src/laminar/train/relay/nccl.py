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
"""NCCL tensor endpoints; sockets carry only protocol metadata and acknowledgements.

Each connection uses a two-rank communicator, independent of training process
groups. Payloads pass through bounded CUDA staging into the existing CPU snapshot.
Connections stay per-transfer, but a pair of processes reuses one communicator
across them: setup costs about a second per pair, and repeatedly creating and
destroying communicators between the same processes fails intermittently on
GCP FasTrak.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import struct
import sys
import uuid
from collections.abc import Hashable
from typing import Any

import torch

logger = logging.getLogger(__name__)
CHUNK_BYTES = 32 * 1024**2
DEFAULT_ENDPOINT_CLOSE_TIMEOUT_S = 120.0
_FRAME = struct.Struct("!Q")
_PAYLOAD = struct.Struct("!Qq")
_KEY_BYTES = 16

# Idle communicators of this process, keyed by (peer, device) on the sender
# (rank 0) and by the sender-chosen key on the receiver (rank 1). An endpoint
# takes its communicator out while connected, so two connections never share one.
_IDLE: dict[Hashable, tuple[bytes, Any, Any, Any, torch.Tensor]] = {}


class NCCLEndpoint:
    """FIFO endpoint implementing the Relay tensor protocol without UCX."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        rank: int,
        peer: Hashable = None,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.rank = rank
        self.peer = peer
        self.device = torch.cuda.current_device()
        self.comm: Any = None
        self.failed = False
        self._send_lock = asyncio.Lock()
        self._recv_lock = asyncio.Lock()

    async def send_obj(self, payload: bytes) -> None:
        self.writer.write(_FRAME.pack(len(payload)) + payload)
        await self.writer.drain()

    async def recv_obj(self) -> bytes:
        (size,) = _FRAME.unpack(await self.reader.readexactly(_FRAME.size))
        return await self.reader.readexactly(size)

    async def _initialize(self) -> None:
        if self.comm is not None:
            return
        from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary, ncclUniqueId

        self.nccl = NCCLLibrary()
        # Bootstrap frame: the communicator key, plus a unique id when it is new.
        if self.rank == 0:
            self.idle_key = (self.peer, self.device)
            idle = _IDLE.pop(self.idle_key, None)
            if idle is not None:
                self.key, self.comm, self.stream, self.copy_stream, self.staging = idle
                await self.send_obj(self.key)
                return
            self.key, uid = uuid.uuid4().bytes, self.nccl.ncclGetUniqueId()
            await self.send_obj(self.key + bytes(uid))
        else:
            frame = await self.recv_obj()
            self.key = self.idle_key = frame[:_KEY_BYTES]
            if len(frame) == _KEY_BYTES:
                idle = _IDLE.pop(self.key, None)
                if idle is None:
                    raise RuntimeError("Relay NCCL sender reused a communicator this receiver does not hold")
                _, self.comm, self.stream, self.copy_stream, self.staging = idle
                return
            uid = ncclUniqueId.from_buffer_copy(frame[_KEY_BYTES:])

        def initialize() -> None:
            with torch.cuda.device(self.device):
                self.comm = self.nccl.ncclCommInitRank(2, uid, self.rank)
                self.stream = torch.cuda.Stream(device=self.device)
                self.copy_stream = torch.cuda.Stream(device=self.device)
                self.staging = torch.empty(2, CHUNK_BYTES, dtype=torch.uint8, device=self.device)

        await asyncio.to_thread(initialize)
        logger.info("Relay NCCL communicator ready: rank=%s device=%s", self.rank, self.device)

    def _transfer(self, array: Any, *, sending: bool, ready: torch.cuda.Event | None = None) -> None:
        from vllm.distributed.device_communicators.pynccl_wrapper import ncclDataTypeEnum

        data = array if isinstance(array, torch.Tensor) else torch.from_numpy(array)
        data = data.view(torch.uint8).reshape(-1)
        operation = self.nccl.ncclSend if sending else self.nccl.ncclRecv

        def transfer(buffer: torch.Tensor) -> None:
            operation(
                ctypes.c_void_p(buffer.data_ptr()),
                buffer.numel(),
                ncclDataTypeEnum.ncclUint8,
                1 - self.rank,
                self.comm,
                ctypes.c_void_p(self.stream.cuda_stream),
            )

        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            if ready is not None:
                self.stream.wait_event(ready)
            if data.is_cuda:
                for offset in range(0, data.numel(), CHUNK_BYTES):
                    transfer(data[offset : offset + CHUNK_BYTES])
                    self.stream.synchronize()
                return
            # Host payloads alternate between two staging buffers, so the PCIe
            # copy of one chunk overlaps the NCCL transfer of the other. Each
            # chunk runs a first and a second stage; a buffer is refilled only
            # after the second stage of its previous chunk.
            if sending:
                first, second = self.copy_stream, self.stream
                run_first = lambda view, staging: staging.copy_(view, non_blocking=True)
                run_second = lambda view, staging: transfer(staging)
            else:
                first, second = self.stream, self.copy_stream
                run_first = lambda view, staging: transfer(staging)
                run_second = lambda view, staging: view.copy_(staging, non_blocking=True)
            released: list[torch.cuda.Event | None] = [None, None]
            for i, offset in enumerate(range(0, data.numel(), CHUNK_BYTES)):
                view = data[offset : offset + CHUNK_BYTES]
                staging = self.staging[i % 2][: view.numel()]
                if released[i % 2] is not None:
                    first.wait_event(released[i % 2])
                with torch.cuda.stream(first):
                    run_first(view, staging)
                second.wait_event(first.record_event())
                with torch.cuda.stream(second):
                    run_second(view, staging)
                released[i % 2] = second.record_event()
            second.synchronize()

    async def send(self, array: Any, tag: int | None = None) -> None:
        ready = None
        if isinstance(array, torch.Tensor) and array.is_cuda:
            # Packing runs on the caller's stream; the transfer runs on
            # a different stream and thread. Keep the source alive until
            # _transfer synchronizes before allowing pack-buffer reuse.
            ready = torch.cuda.current_stream(array.device).record_event()
        async with self._send_lock:
            try:
                if isinstance(array, torch.Tensor) and array.is_cuda:
                    # FSDP's send-loop thread may still have CUDA device 0 as
                    # its default. Bind the communicator to the actual shard.
                    if self.comm is None:
                        self.device = array.device.index
                    elif array.device.index != self.device:
                        raise ValueError("Relay NCCL payload must be on the endpoint CUDA device")
                await self.send_obj(_PAYLOAD.pack(array.nbytes, -1 if tag is None else tag))
                await self._initialize()
                if ready is None:
                    await asyncio.to_thread(self._transfer, array, sending=True)
                else:
                    await asyncio.to_thread(self._transfer, array, sending=True, ready=ready)
            except BaseException:
                self.failed = True
                raise

    async def recv(self, array: Any, tag: int | None = None) -> None:
        async with self._recv_lock:
            try:
                actual = _PAYLOAD.unpack(await self.recv_obj())
                expected = (array.nbytes, -1 if tag is None else tag)
                if actual != expected:
                    raise RuntimeError(f"Relay NCCL payload mismatch: expected={expected} received={actual}")
                await self._initialize()
                await asyncio.to_thread(self._transfer, array, sending=False)
            except BaseException:
                self.failed = True
                raise

    async def close(self, *, timeout_s: float = DEFAULT_ENDPOINT_CLOSE_TIMEOUT_S) -> None:
        publish_error = sys.exception()
        if publish_error is not None:
            self.failed = True
        cached = None
        close_error: BaseException | None = None
        try:
            if self.comm is not None and not self.failed:
                if self.rank == 0:
                    # Cache only after the receiver releases its communicator,
                    # without letting a stalled peer mask a publish deadline.
                    self.writer.write_eof()
                    await asyncio.wait_for(self.reader.read(), timeout=timeout_s)
                cached = (self.key, self.comm, self.stream, self.copy_stream, self.staging)
                _IDLE[self.idle_key] = cached
        except BaseException as exc:
            close_error = exc
        finally:
            try:
                self.writer.close()
                await asyncio.wait_for(self.writer.wait_closed(), timeout=timeout_s)
            except BaseException as exc:
                if close_error is None:
                    close_error = exc
                else:
                    close_error.add_note(f"Relay NCCL socket cleanup also failed: {exc!r}")
        if close_error is not None:
            self.failed = True
            if cached is not None and _IDLE.get(self.idle_key) is cached:
                del _IDLE[self.idle_key]
            if publish_error is None:
                raise close_error
            publish_error.add_note(f"Relay NCCL endpoint cleanup also failed: {close_error!r}")
        elif cached is not None:
            self.comm = None
            del self.staging
        # A cancelled to_thread operation may still own CUDA buffers. Failed
        # publishes end the job; leave those resources for process teardown.


async def create_endpoint(ip: str, port: int) -> NCCLEndpoint:
    reader, writer = await asyncio.open_connection(ip, port)
    return NCCLEndpoint(reader, writer, rank=0, peer=(ip, port))
