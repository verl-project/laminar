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
"""Per-replica LLM server controller for the AgentLoop async-RL architecture.

This control-plane endpoint owns only rollout engine lifecycle and weight
loading. It deliberately does not build trajectories, call rewards, write
TransferQueue samples, or run a direct prompt producer. Uni-Agent / AgentLoop
owns rollout semantics; the router talks to the underlying OpenAI-compatible
vLLM server handle exposed here.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import ray

from laminar.global_request_router.router import EMPTY_WEIGHT_VERSION
from laminar.utils.tensor import TensorMeta

if TYPE_CHECKING:
    from verl.single_controller.ray import RayResourcePool

    from laminar.train.relay import RelayClient

logger = logging.getLogger(__name__)


@dataclass
class LLMServerSnapshot:
    kv_cache_usage: float
    num_waiting_requests: int
    num_running_requests: int
    backend_health: Literal["healthy", "unavailable"] = "healthy"


class LLMServerController:
    """Ray actor control-plane endpoint for one rollout LLM server replica.

    Router generation traffic goes directly to ``get_server_handle()``. Tracker /
    Relay / multi-version control use ``snapshot`` and ``load_weights``.
    """

    def __init__(
        self,
        relay_client: RelayClient | None,
        engine: Any,  # RolloutReplica
        name: str,
    ):
        self.relay_client = relay_client
        self.weight_version = EMPTY_WEIGHT_VERSION
        self._engine = engine
        self.name = name

        self._server_node_ids: tuple[str, ...] | None = None
        self._unavailable = False
        self._state_meta_dict: dict[str, TensorMeta] | None = None
        self._checkpoint_path: str | None = self._extract_model_path(engine)
        self._colocate = False
        self._asleep = False

    @staticmethod
    def _extract_model_path(engine: Any) -> str | None:
        model_config = getattr(engine, "model_config", None)
        path = getattr(model_config, "path", None)
        if path is None and isinstance(model_config, Mapping):
            path = model_config.get("path")
        return str(path) if isinstance(path, str) and path else None

    def _set_model_path(self, checkpoint_path: str) -> None:
        self._checkpoint_path = checkpoint_path
        model_config = getattr(self._engine, "model_config", None)
        if model_config is None:
            return
        if isinstance(model_config, MutableMapping):
            model_config["path"] = checkpoint_path
        elif hasattr(model_config, "path"):
            model_config.path = checkpoint_path
        if hasattr(model_config, "local_path") and getattr(model_config, "local_path", None) is not None:
            model_config.local_path = checkpoint_path

    async def init_engine(
        self, resource_pool: RayResourcePool, weight_version: int | None = None, *, colocate: bool = False
    ) -> None:
        """Heavy init that connects the underlying rollout replica to resources."""
        if weight_version is not None and (type(weight_version) is not int or weight_version < 0):
            raise ValueError("weight_version must be a non-negative integer")
        self._colocate = colocate
        if colocate:
            await self._engine.init_colocated(resource_pool=resource_pool)
        else:
            await self._engine.init_standalone(resource_pool=resource_pool)
        if weight_version is not None:
            await self._engine._server_handle.set_global_steps.remote(weight_version)
            self.weight_version = weight_version
        if colocate:
            await self.sleep_for_training()

    async def sleep_for_training(self) -> None:
        """Release rollout GPU memory after Router has drained request releases."""
        if not self._colocate:
            raise RuntimeError("sleep_for_training requires a colocated rollout engine")
        if self._asleep:
            return
        await self._engine.abort_all_requests(reset_prefix_cache=False)
        await self._engine.sleep()
        self._asleep = True

    async def resume_generation(self) -> None:
        """Open generation only after the shared GPU pool has returned to rollout."""
        if self._asleep:
            raise RuntimeError("cannot resume generation while rollout memory is asleep")
        if self._colocate:
            await self._engine.resume_generation()
        else:
            await self._engine._server_handle.resume_generation.remote()

    async def load_hf_checkpoint(self, checkpoint_path: str, weight_version: int) -> int:
        """Reload a drained standalone evaluation replica using vLLM's HF loader.

        The caller fences routing before this operation and reopens the replica
        only after it succeeds.
        """
        if self.relay_client is not None:
            raise RuntimeError("HF checkpoint reload requires a standalone evaluation controller without Relay")
        if type(weight_version) is not int or weight_version < 0:
            raise ValueError("weight_version must be a non-negative integer")
        if weight_version <= self.weight_version:
            raise ValueError("evaluation checkpoint versions must strictly increase")
        server = self._engine._server_handle
        try:
            await server.abort_all_requests.remote(reset_prefix_cache=False)
            await server.collective_rpc.remote("reload_weights", kwargs={"weights_path": checkpoint_path})
            await server.clear_kv_cache.remote()
            await server.set_global_steps.remote(weight_version)
            await server.resume_generation.remote()
        except Exception as exc:
            self._mark_backend_unavailable(server, exc)
            raise
        self._set_model_path(checkpoint_path)
        self.weight_version = weight_version
        return weight_version

    def get_server_handle(self) -> Any:
        """Return the underlying OpenAI-compatible rollout server handle."""
        if self._engine._server_handle is None:
            raise RuntimeError("LLM server must be initialized before exposing its server handle")
        return self._engine._server_handle

    async def check_health(self) -> None:
        """Check backend health without requiring routing metrics."""
        if self._unavailable:
            raise RuntimeError("LLM server is unavailable")
        if getattr(self, "_asleep", False):
            return
        server = self.get_server_handle()
        try:
            await server.check_health.remote()
        except Exception as exc:
            self._mark_backend_unavailable(server, exc)
            raise
        if self._unavailable or server is not self._engine._server_handle:
            raise RuntimeError("LLM server changed or became unavailable during its health check")

    def _mark_backend_unavailable(self, server: Any, exc: Exception) -> bool:
        if self._unavailable or server is not self._engine._server_handle:
            return True
        # RayTaskError may wrap the original backend exception.
        cause = exc.cause if isinstance(exc, ray.exceptions.RayTaskError) else exc
        try:
            from vllm.v1.engine.exceptions import EngineDeadError
        except ImportError:
            raise exc from None
        if not isinstance(cause, EngineDeadError):
            return False
        self._unavailable = True
        return True

    async def snapshot(self) -> LLMServerSnapshot:
        """Return the runtime state used by routing and Repack."""
        if self._unavailable:
            return LLMServerSnapshot(0.0, 0, 0, backend_health="unavailable")
        if getattr(self, "_asleep", False):
            # Sleeping is an intentional handoff, not an engine failure. The
            # Router's lifecycle fence keeps this healthy but idle view closed.
            return LLMServerSnapshot(0.0, 0, 0)
        server = self._engine._server_handle
        try:
            engine_snapshot: dict[str, Any] = await server.snapshot.remote()
        except Exception as exc:
            if not self._mark_backend_unavailable(server, exc):
                raise
            return LLMServerSnapshot(0.0, 0, 0, backend_health="unavailable")
        if self._unavailable or server is not self._engine._server_handle:
            return LLMServerSnapshot(0.0, 0, 0, backend_health="unavailable")
        return LLMServerSnapshot(
            kv_cache_usage=float(engine_snapshot["kv_cache_usage"]),
            num_waiting_requests=int(engine_snapshot["num_waiting_requests"]),
            num_running_requests=int(engine_snapshot["num_running_requests"]),
        )

    async def restart(self, wait_for_version: int, *, checkpoint_path: str | None = None) -> int:
        """Rebuild an isolated replica; the routing owner publishes its new handle."""
        if getattr(self, "_colocate", False):
            raise RuntimeError("colocated rollout engine recovery is unsupported")
        if type(wait_for_version) is not int or wait_for_version < 0:
            raise ValueError("wait_for_version must be a non-negative integer")
        target_checkpoint = (
            checkpoint_path
            if checkpoint_path is not None
            else (getattr(self, "_checkpoint_path", None) or self._extract_model_path(getattr(self, "_engine", None)))
        )
        if self.relay_client is not None:
            if self._state_meta_dict is None:
                raise RuntimeError("restart requires initialized Relay weight storage")
        elif not isinstance(target_checkpoint, str) or not target_checkpoint:
            raise RuntimeError("restart requires initialized Relay weight storage or a checkpoint path")
        self._unavailable = True
        self.weight_version = EMPTY_WEIGHT_VERSION
        self._server_node_ids = None
        if self.relay_client is None:
            self._set_model_path(target_checkpoint)
        await self._engine.restart()
        if self.relay_client is not None:
            await self.init_shm_state_dict_in_workers(self._state_meta_dict)
            version = await self.load_weights(wait_for_version)
        else:
            await self._engine._server_handle.set_global_steps.remote(wait_for_version)
            self.weight_version = wait_for_version
            version = wait_for_version
        # Check the replacement directly while our unavailable view stays closed.
        await self._engine._server_handle.check_health.remote()
        self._unavailable = False
        return version

    async def init_shm_state_dict_in_workers(
        self,
        state_meta_dict: dict[str, TensorMeta],
    ) -> None:
        """Forward the shm state-dict layout to the underlying vLLM workers."""
        self._state_meta_dict = dict(state_meta_dict)
        while self._engine._server_handle is None:
            await asyncio.sleep(0.1)
        await self._resolve_server_node_ids()
        # vLLM collective_rpc msgpack cannot carry torch.dtype; send (name, shape).
        state_meta_payload = {
            key: (str(value.dtype).removeprefix("torch."), tuple(value.shape)) for key, value in state_meta_dict.items()
        }
        await self._engine._server_handle.collective_rpc.remote(
            "init_shm_state_dict",
            args=(state_meta_payload,),
        )

    async def _resolve_server_node_ids(self) -> tuple[str, ...]:
        if self._server_node_ids is None:
            self._server_node_ids = tuple(
                await asyncio.gather(
                    *[
                        server.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())
                        for server in self._engine.servers
                    ]
                )
            )
        return self._server_node_ids

    async def load_weights(self, wait_for_version: int, *, resume_generation: bool = True) -> int:
        """Pull a new weight version from Relay and return the exact loaded version."""
        if self.relay_client is None:
            raise RuntimeError("load_weights requires Relay; evaluation uses load_hf_checkpoint")
        if type(wait_for_version) is not int or wait_for_version < 0:
            raise ValueError(f"wait_for_version must be a non-negative integer, got {wait_for_version!r}")
        if self.weight_version >= wait_for_version:
            if self._asleep:
                await self._engine.wake_up()
                self._asleep = False
                if resume_generation:
                    await self.resume_generation()
            return self.weight_version

        server_node_ids = await self._resolve_server_node_ids()
        relay_version = await self.relay_client.acquire_rollout_version(
            server_node_ids,
            minimum_version=wait_for_version,
        )

        logger.info("[%s] Updating weights %s -> %s", self.name, self.weight_version, relay_version)
        server_handle = self._engine._server_handle
        try:
            if self._asleep:
                await self._engine.wake_up()
                self._asleep = False
            # Router request slots have been released; pause the core before
            # changing weights. clear_kv_cache below drops stale prefix KV.
            if relay_version > self.weight_version:
                if self._colocate:
                    await self._engine.abort_all_requests(reset_prefix_cache=False)
                else:
                    await server_handle.abort_all_requests.remote(reset_prefix_cache=False)
                await server_handle.collective_rpc.remote("load_weights_from_shm_state_dict")
                await server_handle.clear_kv_cache.remote()
                await server_handle.set_global_steps.remote(relay_version)
            if resume_generation:
                await self.resume_generation()
        except BaseException as exc:
            if isinstance(exc, Exception):
                self._mark_backend_unavailable(server_handle, exc)
            try:
                await self.relay_client.release_full_read_async(server_node_ids)
            except Exception:
                logger.exception("[%s] Failed to release Relay read lock after load_weights failure", self.name)
            raise

        await self.relay_client.release_full_read_async(server_node_ids)
        self.weight_version = max(self.weight_version, relay_version)
        return self.weight_version
