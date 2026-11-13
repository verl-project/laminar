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
"""Asyncio primitives shared by async_rl components."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Awaitable
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)


async def await_shielded_completion[T](awaitable: Awaitable[T]) -> tuple[T, asyncio.CancelledError | None]:
    """Finish an owned operation even if its caller is cancelled.

    The returned cancellation must be re-raised by the caller after it has
    restored the invariant protected by the operation. ``asyncio.shield``
    alone is insufficient because it returns immediately on cancellation,
    before the owned operation has actually finished.
    """
    future = asyncio.ensure_future(awaitable)
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            return await asyncio.shield(future), cancellation
        except asyncio.CancelledError as exc:
            if future.done():
                return future.result(), cancellation or exc
            if cancellation is None:
                cancellation = exc


async def gather_cancel_on_error[T](*awaitables: Awaitable[T]) -> list[T]:
    """Gather awaitables, cancelling and draining siblings before propagating an error."""
    tasks = [asyncio.ensure_future(awaitable) for awaitable in awaitables]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def log_exception_callback(future: Any) -> None:
    """``add_done_callback`` wrapper so fire-and-forget task exceptions are not swallowed."""
    try:
        future.result()
    except asyncio.CancelledError:
        logger.info("Task was cancelled.")
    except BaseException:
        logger.exception("Task crashed")


class ReadWriteLock:
    """Writer-preference async RW lock with synchronous release.

    ``release_read`` / ``release_write`` are sync because Ray worker methods
    such as ``complete_publish`` cannot await. Wake-up reserves ownership
    before resolving the waiter so a newly arriving reader cannot sneak in
    ahead of a granted writer.
    """

    def __init__(self):
        self._readers = 0
        self._writer_active = False
        self._waiting_writers: deque[asyncio.Future[None]] = deque()
        self._waiting_readers: deque[asyncio.Future[None]] = deque()
        self._granted_writer: asyncio.Future[None] | None = None
        self._granted_readers: set[asyncio.Future[None]] = set()

    @staticmethod
    def _discard_done_waiters(waiters: deque[asyncio.Future[None]]) -> None:
        for future in tuple(waiters):
            if future.done():
                waiters.remove(future)

    def _wake_waiters(self) -> None:
        self._discard_done_waiters(self._waiting_writers)
        self._discard_done_waiters(self._waiting_readers)
        if self._writer_active or self._granted_readers:
            return
        if self._waiting_writers:
            if self._readers > 0:
                return
            writer_future = self._waiting_writers.popleft()
            self._writer_active = True
            self._granted_writer = writer_future
            writer_future.set_result(None)
            return
        if self._waiting_readers:
            reader_futures = list(self._waiting_readers)
            self._waiting_readers.clear()
            self._readers += len(reader_futures)
            self._granted_readers.update(reader_futures)
            for reader_future in reader_futures:
                reader_future.set_result(None)

    def try_acquire_read(self) -> bool:
        """Take a read immediately without queuing behind a writer or reserved grant."""
        self._wake_waiters()
        if self._writer_active or self._waiting_writers or self._granted_readers:
            return False
        self._readers += 1
        return True

    async def acquire_read(self) -> None:
        self._wake_waiters()
        if self._writer_active or self._waiting_writers or self._granted_readers:
            future = asyncio.get_running_loop().create_future()
            self._waiting_readers.append(future)
            try:
                await future
            except asyncio.CancelledError:
                if future in self._granted_readers:
                    self._granted_readers.remove(future)
                    self._readers -= 1
                elif future in self._waiting_readers:
                    self._waiting_readers.remove(future)
                self._wake_waiters()
                raise
            self._granted_readers.discard(future)
            self._wake_waiters()
            return
        self._readers += 1

    def release_read(self) -> None:
        self._readers -= 1
        self._wake_waiters()

    async def acquire_write(self) -> None:
        self._wake_waiters()
        if self._readers > 0 or self._writer_active:
            future = asyncio.get_running_loop().create_future()
            self._waiting_writers.append(future)
            try:
                await future
            except asyncio.CancelledError:
                if self._granted_writer is future:
                    self._granted_writer = None
                    self._writer_active = False
                elif future in self._waiting_writers:
                    self._waiting_writers.remove(future)
                self._wake_waiters()
                raise
            self._granted_writer = None
            return
        self._writer_active = True

    def release_write(self) -> None:
        self._writer_active = False
        self._wake_waiters()

    @asynccontextmanager
    async def write_lock(self):
        await self.acquire_write()
        try:
            yield
        finally:
            self.release_write()
