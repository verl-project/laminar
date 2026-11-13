# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Shared replica lifecycle operations for Repack draining and dead-engine recovery."""

from __future__ import annotations

import asyncio
from typing import Any

import ray

from laminar.global_request_router.router import RouteUnavailableError

_DRAIN_TIMEOUT_S = 600.0
_DRAIN_POLL_INTERVAL_S = 0.1


def is_engine_dead_error(exc: BaseException) -> bool:
    """Return True when ``exc`` is a vLLM ``EngineDeadError`` (direct or wrapped in ``RayTaskError``)."""
    cause = exc.cause if isinstance(exc, ray.exceptions.RayTaskError) else exc
    try:
        from vllm.v1.engine.exceptions import EngineDeadError
    except ImportError:
        return False
    return isinstance(cause, EngineDeadError)


def is_recoverable_backend_error(exc: BaseException) -> bool:
    """Return True when a backend RPC failed due to a dead engine or dead server actor."""
    return isinstance(exc, ray.exceptions.RayActorError) or is_engine_dead_error(exc)


async def begin_dead_server_recovery(router: Any, server: Any, server_id: str) -> bool:
    """Fence ``server_id`` for recovery only after confirming its backend is unavailable."""
    snapshot = await server.snapshot.remote()
    if snapshot.backend_health != "unavailable":
        return False
    await router.begin_recovery.remote(server_id=server_id)
    return True


async def restart_and_finish_recovery(
    router: Any,
    server: Any,
    server_id: str,
    *,
    wait_for_version: int,
    checkpoint_path: str | None = None,
    exact_version: bool = False,
) -> int:
    """Restart a fenced replica and publish its replacement handle to the Router."""
    restart_kwargs: dict[str, Any] = {"wait_for_version": wait_for_version}
    if checkpoint_path is not None:
        restart_kwargs["checkpoint_path"] = checkpoint_path
    version = await server.restart.remote(**restart_kwargs)
    if type(version) is not int or (version != wait_for_version if exact_version else version < wait_for_version):
        raise RuntimeError(f"restarted engine {server_id!r} returned invalid weight version {version!r}")
    handle = await server.get_server_handle.remote()
    await router.finish_recovery.remote(server_id=server_id, handle=handle, weight_version=version)
    return version


async def recover_dead_server(
    router: Any,
    server: Any,
    server_id: str,
    *,
    wait_for_version: int,
    checkpoint_path: str | None = None,
    exact_version: bool = False,
) -> bool:
    """Confirm a replica is dead, fence it in the Router, restart it, and finish recovery."""
    if not await begin_dead_server_recovery(router, server, server_id):
        return False
    await restart_and_finish_recovery(
        router,
        server,
        server_id,
        wait_for_version=wait_for_version,
        checkpoint_path=checkpoint_path,
        exact_version=exact_version,
    )
    return True


async def wait_for_source_drain(
    router: Any, server_id: str, *, expected_version: int, deadline: float | None = None
) -> None:
    """Abort old turns and wait for their releases before fencing a source.

    The Router has already moved future-turn routes. Committed request leases
    still belong to their source and must release before it can load weights.
    """
    if deadline is None:
        deadline = asyncio.get_running_loop().time() + _DRAIN_TIMEOUT_S
    async with asyncio.timeout_at(deadline):
        server = await router.get_server_handle.remote(server_id=server_id)
        while True:
            # A previously acquired request may reach the engine after a sweep.
            await server.abort_all_requests.remote(abort_only=True)
            try:
                await router.try_mark_updating.remote(server_id=server_id, expected_version=expected_version)
                return
            except RouteUnavailableError:
                await asyncio.sleep(_DRAIN_POLL_INTERVAL_S)
