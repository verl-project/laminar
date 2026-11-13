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
"""LLM-server runtime state source for async routing."""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING, Any

from ray.actor import ActorHandle

from laminar.utils.concurrency import log_exception_callback

if TYPE_CHECKING:
    from laminar.llm_server.controller import LLMServerSnapshot

HEALTHY = "healthy"


class LLMServerStateTracker:
    """Poll LLM servers, cache validated snapshots, and push them to the Router."""

    def __init__(self):
        self.servers: dict[str, ActorHandle] = {}
        self.router: ActorHandle | None = None
        self._server_states: dict[str, dict[str, Any]] = {}
        self._snapshot_version = 0
        self._run_task: asyncio.Task | None = None

    def connect(
        self,
        servers: dict[str, ActorHandle],
        router: ActorHandle | None = None,
    ) -> None:
        """Connect LLM server snapshot handles and optional router handle."""
        if not servers:
            raise ValueError("LLMServerStateTracker requires at least one engine")
        self.servers = dict(servers)
        self.router = router

    async def start(self) -> None:
        """Launch the polling loop inside this actor."""
        if not self.servers:
            raise RuntimeError("LLMServerStateTracker.connect must be called before start.")
        if self._run_task is not None and not self._run_task.done():
            return
        self._run_task = asyncio.create_task(self._loop(), name="llm_server_state_tracker.loop")
        self._run_task.add_done_callback(log_exception_callback)

    async def wait_for_stop(self) -> None:
        """Wait for the polling loop and surface its first fatal error."""
        await self._run_task

    async def _loop(self) -> None:
        while True:
            await self.refresh_once()
            await asyncio.sleep(1.0)

    async def refresh_once(self) -> None:
        """Poll every server once and update the Router."""
        if not self.servers:
            raise RuntimeError("LLMServerStateTracker.connect must be called before refresh_once.")
        snapshot_version = self._snapshot_version + 1
        server_ids = list(self.servers)
        results = await asyncio.gather(
            *(self._poll_server(server_id, snapshot_version=snapshot_version) for server_id in server_ids)
        )
        self._server_states = dict(results)
        self._snapshot_version = snapshot_version
        if self.router is not None:
            await self.router.update_routing_snapshot.remote(self.get_routing_snapshot())

    async def _poll_server(self, server_id: str, *, snapshot_version: int) -> tuple[str, dict[str, Any]]:
        try:
            snapshot = await self.servers[server_id].snapshot.remote()
        except Exception as exc:
            raise RuntimeError(f"failed to poll LLM server {server_id!r}") from exc
        try:
            state = self._state_from_snapshot(snapshot, snapshot_version=snapshot_version)
        except Exception as exc:
            raise RuntimeError(f"invalid snapshot from LLM server {server_id!r}") from exc
        return server_id, state

    def get_routing_snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a copy of the compact per-replica runtime snapshot."""
        return {server_id: dict(state) for server_id, state in self._server_states.items()}

    def _state_from_snapshot(
        self,
        snapshot: LLMServerSnapshot,
        *,
        snapshot_version: int | None = None,
    ) -> dict[str, Any]:
        if snapshot.backend_health not in (HEALTHY, "unavailable"):
            raise ValueError(f"invalid backend health: {snapshot.backend_health!r}")
        kv_cache_usage = float(snapshot.kv_cache_usage)
        if not math.isfinite(kv_cache_usage) or not 0.0 <= kv_cache_usage <= 1.0:
            raise ValueError(f"kv_cache_usage must be in [0, 1], got {kv_cache_usage!r}")
        num_waiting_requests = int(snapshot.num_waiting_requests)
        num_running_requests = int(snapshot.num_running_requests)
        if num_waiting_requests < 0 or num_running_requests < 0:
            raise ValueError("request counts must be non-negative")
        return {
            "backend_health": snapshot.backend_health,
            "kv_cache_usage": kv_cache_usage,
            "num_waiting_requests": num_waiting_requests,
            "num_running_requests": num_running_requests,
            "snapshot_version": self._snapshot_version if snapshot_version is None else snapshot_version,
        }
