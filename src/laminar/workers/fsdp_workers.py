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
"""``AsyncActorWorker``: new-engine ``ActorRolloutRefWorker`` + async-RL relay hooks.

Bridges the model-engine actor worker (``verl.workers.engine_workers.ActorRolloutRefWorker``)
with the CPU-RDMA relay (``laminar.train.relay``) used by the
async-RL recipe. The base actor handles model construction, dispatch
registration, and ``update_actor``; this subclass layers on:

* A dedicated send-weights thread (``send_loop`` / ``send_thread``) so
  weight broadcasts don't block the trainer's main asyncio loop.
* ``connect_to_relay_server`` / ``send_weights`` RPCs that drive the relay
  protocol from the actor side; each FSDP rank writes its shards straight
  into their byte ranges of Relay's full tensors.
* Mainline ``ppo_loss`` wired via ``set_loss_fn`` during ``init_model`` —
  configures behavior-policy PPO clipping and the recipe's preferred
  sequence-level denominator.

The custom ``update_actor`` is a thin wrapper that waits for any in-flight
weight send to drain before mutating the parameters (avoids tearing the
relay buffer mid-stream).
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from concurrent.futures import Future
from typing import TYPE_CHECKING

import torch
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from verl.single_controller.base.decorator import Dispatch, Execute, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.engine_workers import ActorRolloutRefWorker

from laminar.train.relay import ActorPublishStats
from laminar.train.relay.manager import _ThrottledWaitLogger
from laminar.train.relay.transport import (
    TensorStorageMeta,
    TensorTransferStats,
    create_endpoint,
    send_tensors,
)
from laminar.utils.concurrency import log_exception_callback
from laminar.utils.tensor import TensorMeta
from laminar.workers.metrics import add_optimizer_update_metrics, observe_optimizer_updates

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from tensordict import TensorDict

logger = logging.getLogger(__name__)


# Sharded GPU state dict — no offload (we copy to the relay's CPU buffer ourselves).
SHARDED_GPU_STATE_DICT_OPTIONS = StateDictOptions(full_state_dict=False, cpu_offload=False, broadcast_from_rank0=False)


def _local_dim0_shard(value: torch.Tensor) -> tuple[torch.Tensor, int] | None:
    """Return this rank's shard of ``value`` and its element offset in the full tensor.

    Relay receives each shard as one contiguous byte range of the full tensor,
    so a shard must span every dimension but dim 0 (how FSDP1 and FSDP2 shard).
    Returns ``None`` when this rank holds no element of ``value``.
    """
    if isinstance(value, DTensor):
        local = value.to_local()
        _, offsets = compute_local_shape_and_global_offset(value.shape, value.device_mesh, value.placements)
    elif isinstance(value, ShardedTensor):
        shards = value.local_shards()
        if not shards:
            return None
        (shard,) = shards
        local, offsets = shard.tensor, shard.metadata.shard_offsets
    else:
        local, offsets = value, [0] * value.dim()
    if local.numel() == 0:
        return None
    full_shape = value.size()
    if local.shape[1:] != full_shape[1:]:
        raise ValueError(
            f"Relay needs dim-0 shards: full_shape={tuple(full_shape)} local_shape={tuple(local.shape)} {offsets=}"
        )
    return local, (offsets[0] * math.prod(full_shape[1:]) if offsets else 0)


def _restore_actor_train_mode(worker) -> dict[str, object]:
    """Restore actor modules after read-only log-prob inference."""
    assert "actor" in worker.role, "restore_actor_train_mode only supports actor role"
    engine = worker.actor.engine
    modules = getattr(engine, "module", None)
    if modules is None:
        modules = getattr(engine, "model", None)
    if modules is None:
        actor_modules = []
    elif isinstance(modules, list | tuple):
        actor_modules = list(modules)
    else:
        actor_modules = [modules]
    before = sum(1 for module in actor_modules if bool(getattr(module, "training", False)))
    for module in actor_modules:
        module.train()
    after = sum(1 for module in actor_modules if bool(getattr(module, "training", False)))
    clear_router_replay_state = getattr(engine, "clear_router_replay_state", None)
    router_replay_state = clear_router_replay_state() if callable(clear_router_replay_state) else None
    return {
        "engine_class": engine.__class__.__name__,
        "module_count": len(actor_modules),
        "module_training_count_before": before,
        "module_training_count_after": after,
        "router_replay_state": router_replay_state,
    }


class AsyncActorWorker(ActorRolloutRefWorker):
    """``ActorRolloutRefWorker`` with relay-side weight-send + actor PPO loss.

    This subclass is not a rollout worker: ``init_model`` rejects rollout
    roles, and its Relay hooks require an actor engine. The async-RL recipe
    runs it as ``role="actor"`` while a separate vLLM replica fleet owns
    rollout.
    """

    def __init__(
        self,
        config: DictConfig,
        role: str,
    ):
        super().__init__(config=config, role=role)
        # Most worker methods are blocking; we offload the weight-send
        # path to a dedicated event loop on a daemon thread so the
        # send_weights RPC returns immediately to the trainer.
        self.send_loop = asyncio.new_event_loop()
        self.send_thread = threading.Thread(
            target=self.send_loop.run_forever, name="send_weights_from_cpu", daemon=True
        )
        self.send_thread.start()
        self._relay_send_future: Future[ActorPublishStats] | None = None
        self.relay_server_address: tuple[str, int] | None = None

    @property
    def _actor_engine(self):
        assert self.actor is not None, "actor engine is not initialized; call init_model() first"
        return self.actor.engine

    @property
    def _actor_device_mesh(self):
        return self._actor_engine.device_mesh

    @property
    def _actor_module_fsdp(self):
        return self._actor_engine.module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def init_model(self):
        """Build the model and wire the configured PPO loss as the engine loss fn."""
        assert not self._is_rollout, "AsyncActorWorker should not be initialized as a rollout worker"
        super().init_model()

        # Reuse the mainline actor loss; async-specific behavior is expressed
        # through actor config fields so DP/global-batch normalization stays
        # on the same engine path as synchronous PPO.
        assert self.loss_fn is not None, "ActorRolloutRefWorker.init_model() did not register a loss function"

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def restore_actor_train_mode(self) -> dict[str, object]:
        """Restore training flags without entering the engine's context manager."""
        return _restore_actor_train_mode(self)

    # === Relay ===

    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO, blocking=True)  # type: ignore[attr-defined]
    async def get_actor_state_meta_dict(self, dtype: torch.dtype | str) -> dict[str, TensorMeta]:
        """Driver-visible: full state dict metas, sorted by key for determinism."""
        sharded_state_dict = get_model_state_dict(self._actor_module_fsdp, options=SHARDED_GPU_STATE_DICT_OPTIONS)
        sorted_full_tensor_keys = sorted(sharded_state_dict.keys())
        relay_dtype = dtype if isinstance(dtype, torch.dtype) else PrecisionType.to_dtype(dtype)
        state_meta_dict: dict[str, TensorMeta] = {
            key: TensorMeta(
                dtype=relay_dtype,
                shape=sharded_state_dict[key].data.shape,
            )
            for key in sorted_full_tensor_keys
        }
        return state_meta_dict

    @property
    def ddp_rank(self) -> int:
        """Local DDP rank (0 if no DDP mesh dim)."""
        ddp_rank = 0
        device_mesh = self._actor_device_mesh
        if "ddp" in device_mesh.mesh_dim_names:
            ddp_rank = device_mesh["ddp"].get_local_rank()
        return ddp_rank

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)  # type: ignore[attr-defined]
    async def connect_to_relay_server(self, ip: str, port: int) -> None:
        """Validate Relay connectivity without retaining an idle UCX endpoint."""
        endpoint = await create_endpoint(ip, port)
        await endpoint.close()
        self.relay_server_address = (ip, port)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)  # type: ignore[attr-defined]
    async def send_weights(
        self,
        version: int,
        dtype: torch.dtype | str,
    ) -> None:
        """Schedule a weight-send to the relay on the dedicated send thread.

        The in-flight future is stored before this RPC returns so callers can
        immediately use ``wait_for_sending_to_relay`` as a real send barrier.
        """
        if self._relay_send_future is not None:
            raise RuntimeError(f"Relay actor weight send is already in progress on rank {self.rank}")
        # Other DDP replicas hold duplicates of DDP rank 0's FSDP shards.
        if self.ddp_rank > 0:
            return

        dtype = PrecisionType.to_dtype(dtype)
        logger.info(
            "[Rank %s] send_weights: scheduling version=%s dtype=%s",
            self.rank,
            version,
            dtype,
        )
        send_coro = self._send_weights(version=version, dtype=dtype)
        # Hand off to the dedicated send thread so GPUs can keep training.
        try:
            self._relay_send_future = asyncio.run_coroutine_threadsafe(send_coro, self.send_loop)
            self._relay_send_future.add_done_callback(log_exception_callback)
        except BaseException:
            send_coro.close()
            raise

    async def _send_weights(
        self,
        version: int,
        dtype: torch.dtype | None = None,
    ) -> ActorPublishStats:
        rank = self._actor_device_mesh["fsdp"].get_local_rank()

        _send_start_time = time.perf_counter()
        logger.info(
            "[Rank %s] _send_weights: start version=%s fsdp_rank=%s",
            self.rank,
            version,
            rank,
        )
        export_time_s = 0.0
        connect_time_s = 0.0
        sender_time_s = 0.0
        transfer_stats = TensorTransferStats()
        export_start = time.perf_counter()
        local_tensors, local_storage_metas = self._get_local_tensors_and_storage_metas(
            version=version,
            dtype=dtype,
        )
        export_time_s = time.perf_counter() - export_start

        # TODO: overlap D2H and CPU RDMA transport (currently serialized
        # inside ``send_tensors``).
        ip, port = self._require_relay_server_address()
        connect_start = time.perf_counter()
        endpoint = await create_endpoint(ip, port)
        connect_time_s = time.perf_counter() - connect_start
        try:
            sender_start = time.perf_counter()
            transfer_stats = await send_tensors(
                endpoint,
                storage_metas=local_storage_metas,
                tensors=local_tensors,
            )
            sender_time_s = time.perf_counter() - sender_start
        finally:
            await endpoint.close()

        _send_done_time = time.perf_counter()
        elapsed = _send_done_time - _send_start_time
        stats = ActorPublishStats(
            publisher_count=1,
            tensor_count=len(local_storage_metas),
            chunk_count=transfer_stats.chunk_count,
            payload_bytes=transfer_stats.payload_bytes,
            export_time_s=export_time_s,
            connect_time_s=connect_time_s,
            sender_time_s=sender_time_s,
            tensor_prepare_time_s=transfer_stats.tensor_prepare_time_s,
            pack_time_s=transfer_stats.pack_time_s,
            staging_time_s=transfer_stats.staging_time_s,
            transport_time_s=transfer_stats.transport_time_s,
            completion_ack_time_s=transfer_stats.completion_ack_time_s,
            total_time_s=elapsed,
        )
        logger.info(
            "[Rank %s] _send_weights: done version=%s fsdp_rank=%s elapsed=%.3fs",
            self.rank,
            version,
            rank,
            elapsed,
        )
        return stats

    def _require_relay_server_address(self) -> tuple[str, int]:
        if self.relay_server_address is None:
            raise RuntimeError("Relay server is not connected; call connect_to_relay_server first")
        return self.relay_server_address

    def _get_local_tensors_and_storage_metas(
        self,
        *,
        version: int,
        dtype: torch.dtype | str | None = None,
    ) -> tuple[list[torch.Tensor], list[TensorStorageMeta]]:
        """Materialize the non-empty local FSDP shards + their relay metas.

        Each shard targets its byte range of the full Relay tensor. No
        ``cpu_offload`` here: callers are expected to ``wait_for_sending_to_relay``
        before mutating parameters, so the GPU tensors stay valid for the
        duration of the send.
        """
        sharded_state_dict = get_model_state_dict(self._actor_module_fsdp, options=SHARDED_GPU_STATE_DICT_OPTIONS)

        local_tensors: list[torch.Tensor] = []
        storage_metas: list[TensorStorageMeta] = []
        relay_dtype = None if dtype is None else PrecisionType.to_dtype(dtype)

        for param_key in sorted(sharded_state_dict):
            local_shard = _local_dim0_shard(sharded_state_dict[param_key])
            if local_shard is None:
                continue
            local_tensor, element_offset = local_shard
            if relay_dtype is not None:
                local_tensor = local_tensor.to(relay_dtype)
            local_tensors.append(local_tensor)

            # Strip FSDP1's "_fsdp_wrapped_module." prefix so the relay's
            # full-key namespace matches the rollout's vLLM model.
            storage_meta = TensorStorageMeta(
                key=param_key.replace("_fsdp_wrapped_module.", ""),
                data_meta=TensorMeta(dtype=local_tensor.dtype, shape=local_tensor.shape),
                version=version,
                byte_offset=element_offset * local_tensor.element_size(),
            )
            storage_metas.append(storage_meta)

        return local_tensors, storage_metas

    # RelayClient materializes these refs one by one so every rank is drained
    # even when one rank reports a send failure.
    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL, blocking=False)  # type: ignore[attr-defined]
    async def wait_for_sending_to_relay(self) -> ActorPublishStats | None:
        """Block until the in-flight send (if any) finishes. Call before mutating params."""
        send_future = self._relay_send_future
        if send_future is None:
            return None
        try:
            if not send_future.done():
                _wait_start_time = time.perf_counter()
                wait_logger = _ThrottledWaitLogger(owner=f"[Rank {self.rank}] wait_for_sending_to_relay")
                while not send_future.done():
                    wait_logger.maybe_warn("previous relay send is still in flight")
                    await asyncio.sleep(0.1)
                _wait_done_time = time.perf_counter()
                wait_time_s = _wait_done_time - _wait_start_time
                logger.info(f"[Rank {self.rank}] wait_for_sending_to_relay: waited {wait_time_s:.3f}s")
            return send_future.result()
        finally:
            if send_future.done() and self._relay_send_future is send_future:
                self._relay_send_future = None

    @register(
        dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"),
        execute_mode=Execute.ALL,
        blocking=True,
    )  # type: ignore[attr-defined]
    async def update_actor(self, data: TensorDict) -> TensorDict:
        """Wait for in-flight relay sends, then delegate to the base engine.

        The base ``ActorRolloutRefWorker.update_actor`` runs the engine
        mini-batch loop with whatever ``loss_fn`` we registered in
        ``init_model``.
        """
        await self.wait_for_sending_to_relay()
        # ``super().update_actor`` is sync in the new engine; awaiting a
        # non-coroutine is a TypeError, so we call it directly.
        with observe_optimizer_updates(self._actor_engine, backend="fsdp") as update_stats:
            output = super().update_actor(data)
        return add_optimizer_update_metrics(output, update_stats)
