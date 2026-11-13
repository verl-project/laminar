# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from laminar.utils.concurrency import ReadWriteLock


def test_try_read_does_not_bypass_queued_or_granted_writer():
    async def run():
        lock = ReadWriteLock()
        assert lock.try_acquire_read()
        assert lock.try_acquire_read()
        writer = asyncio.create_task(lock.acquire_write())
        await asyncio.sleep(0)
        assert not writer.done()
        assert not lock.try_acquire_read()
        lock.release_read()
        assert not lock.try_acquire_read()
        lock.release_read()
        # The writer owns its grant before its coroutine resumes.
        assert not writer.done()
        assert not lock.try_acquire_read()
        await asyncio.wait_for(writer, timeout=1)
        assert not lock.try_acquire_read()
        lock.release_write()
        assert lock.try_acquire_read()
        lock.release_read()
        await asyncio.wait_for(lock.acquire_write(), timeout=1)
        lock.release_write()

    asyncio.run(run())


def test_try_read_allows_readers_after_waiting_writer_is_cancelled():
    async def run():
        lock = ReadWriteLock()
        assert lock.try_acquire_read()
        writer = asyncio.create_task(lock.acquire_write())
        await asyncio.sleep(0)
        assert not lock.try_acquire_read()
        writer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await writer
        assert lock.try_acquire_read()
        lock.release_read()
        lock.release_read()
        await asyncio.wait_for(lock.acquire_write(), timeout=1)
        lock.release_write()

    asyncio.run(run())


def test_try_read_preserves_an_existing_reader_grant():
    async def run():
        lock = ReadWriteLock()
        await lock.acquire_write()
        reader = asyncio.create_task(lock.acquire_read())
        await asyncio.sleep(0)
        lock.release_write()
        # The queued reader has reserved ownership but has not resumed yet.
        assert not reader.done()
        assert not lock.try_acquire_read()
        await asyncio.wait_for(reader, timeout=1)
        assert lock.try_acquire_read()
        lock.release_read()
        lock.release_read()
        await asyncio.wait_for(lock.acquire_write(), timeout=1)
        lock.release_write()

    asyncio.run(run())
