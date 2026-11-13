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
"""Megatron actor worker with async-RL Relay publishing hooks."""

from __future__ import annotations

import asyncio
import gc
import logging
import time
from typing import TYPE_CHECKING

import torch
from verl.models.mcore.patch import neutralize_broken_flash_attn_cute
from verl.single_controller.base.decorator import Dispatch, Execute, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import get_torch_device
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.engine_workers import ActorRolloutRefWorker

from laminar.train.relay import ActorPublishStats
from laminar.train.relay.manager import _ThrottledWaitLogger
from laminar.train.relay.transport import (
    TensorChunkSender,
    TensorStorageMeta,
    TensorTransferStats,
    create_endpoint,
)
from laminar.utils.tensor import TensorMeta
from laminar.workers.fsdp_workers import _restore_actor_train_mode
from laminar.workers.metrics import add_optimizer_update_metrics, observe_optimizer_updates

# Must precede any Megatron import: Megatron's optional FA4 probe only catches
# ImportError, so broken optional flash_attn.cute installs need to be hidden first.
neutralize_broken_flash_attn_cute()
from megatron.core.optimizer import ChainedOptimizer, LayerWiseDistributedOptimizer  # noqa: E402

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from tensordict import TensorDict

logger = logging.getLogger(__name__)

_RELAY_CONNECT_TIMEOUT_S = 30.0


class AsyncMegatronActorWorker(ActorRolloutRefWorker):
    """Megatron ``ActorRolloutRefWorker`` plus Relay full-weight publish.

    Megatron/MBridge exports rollout-loadable full HF/vLLM tensors from one
    logical publisher rank: MP output-source rank and data-parallel rank 0.
    Other MP ranks still have to enter the export iterator because MBridge may
    perform collectives while materializing those tensors. That rank sends
    whole tensors straight into Relay's full-state storage.
    """

    def __init__(
        self,
        config: DictConfig,
        role: str,
    ):
        super().__init__(config=config, role=role)
        self.is_sending_to_relay = False
        self.relay_server_address: tuple[str, int] | None = None
        self._first_actor_update_memory_logged = False

    @property
    def _actor_engine(self):
        assert self.actor is not None, "actor engine is not initialized; call init_model() first"
        return self.actor.engine

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def init_model(self):
        assert not self._is_rollout, "AsyncMegatronActorWorker should not be initialized as a rollout worker"
        if torch.cuda.is_available():
            self._log_cuda_memory("cublas_prewarm_before")
            torch.cuda.current_blas_handle()
            self._log_cuda_memory("cublas_prewarm_after")
        super().init_model()
        self._prewarm_grad_stats_communicators()
        assert self.loss_fn is not None, "ActorRolloutRefWorker.init_model() did not register a loss function"

    def _prewarm_grad_stats_communicators(self) -> None:
        if not torch.cuda.is_available():
            return
        if not torch.distributed.is_initialized():
            raise RuntimeError("distributed process group is not initialized")

        optimizer = self._actor_engine.optimizer
        if optimizer is None:
            raise RuntimeError("Megatron optimizer is not initialized")

        groups = []
        for group in self._grad_stats_parallel_groups(optimizer):
            if all(group is not existing_group for existing_group in groups):
                groups.append(group)
        if not groups:
            raise RuntimeError("Megatron optimizer has no grad-stats process groups")

        self._log_cuda_memory("grad_stats_nccl_prewarm_before")
        probe = torch.zeros(
            1,
            dtype=torch.float32,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        for group in groups:
            torch.distributed.all_reduce(
                probe,
                op=torch.distributed.ReduceOp.SUM,
                group=group,
            )
        torch.cuda.synchronize()
        self._log_cuda_memory("grad_stats_nccl_prewarm_after")

    def _grad_stats_parallel_groups(self, optimizer) -> list[torch.distributed.ProcessGroup]:
        """Mirror the optimizer's ``get_grad_norm`` collective-group control flow."""

        def require_group(group):
            if group is None:
                raise RuntimeError("Megatron grad-stats process group is missing")
            return group

        # LayerWiseDistributedOptimizer intentionally overrides ChainedOptimizer.get_grad_norm()
        # and reduces on the default/world group, irrespective of its child optimizer groups.
        if isinstance(optimizer, LayerWiseDistributedOptimizer):
            return [require_group(torch.distributed.group.WORLD)]

        if not isinstance(optimizer, ChainedOptimizer):
            return [require_group(optimizer.get_grad_stats_parallel_group())]

        children = optimizer.chained_optimizers
        if not children:
            raise RuntimeError("Megatron chained optimizer has no child optimizers")
        if len(children) == 1:
            return self._grad_stats_parallel_groups(children[0])
        if optimizer.grads_states_parallel_group_is_shared():
            return [require_group(optimizer.get_grad_stats_parallel_group())]

        groups = []
        for child_optimizer in children:
            for group in self._grad_stats_parallel_groups(child_optimizer):
                if all(group is not existing_group for existing_group in groups):
                    groups.append(group)
        return groups

    def _log_cuda_memory(self, event: str) -> None:
        try:
            device_index = torch.cuda.current_device()
            allocated_bytes = torch.cuda.memory_allocated(device_index)
            reserved_bytes = torch.cuda.memory_reserved(device_index)
            rank = self.rank
        except Exception:
            logger.warning("Async Megatron CUDA memory observation failed: event=%s", event, exc_info=True)
            return
        logger.info(
            "[Rank %s] LAMINAR_ASYNC_MEGATRON_CUDA_MEMORY event=%s device=%s allocated_bytes=%s reserved_bytes=%s",
            rank,
            event,
            device_index,
            allocated_bytes,
            reserved_bytes,
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    def restore_actor_train_mode(self) -> dict[str, object]:
        """Restore training flags without entering the engine's context manager."""
        return _restore_actor_train_mode(self)

    # === Relay ===

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, execute_mode=Execute.ALL, blocking=True)
    def get_actor_state_meta_dict(self, dtype: torch.dtype | str) -> dict[str, TensorMeta] | None:
        """Return full actor tensor metas from the single Relay publisher rank."""
        should_publish = self._should_publish_relay_actor_weights()
        relay_dtype = self._normalize_relay_dtype(dtype)
        assert relay_dtype is not None
        try:
            state_meta_dict = {
                name: TensorMeta(dtype=relay_dtype, shape=tensor.shape)
                for name, tensor in self._iter_relay_actor_tensors(dtype=None, publish_only=True)
            }
            return state_meta_dict if should_publish else None
        finally:
            self._cleanup_relay_actor_export()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    async def connect_to_relay_server(self, ip: str, port: int) -> None:
        if not self._should_publish_relay_actor_weights():
            return
        # Validate connectivity now, but do not retain an endpoint across
        # potentially long rollout-engine cold compilation.
        endpoint = await create_endpoint(ip, port)
        await endpoint.close()
        self.relay_server_address = (ip, port)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=True)
    async def send_weights(
        self,
        version: int,
        dtype: torch.dtype | str,
    ) -> ActorPublishStats | None:
        """Publish a Megatron full-weight snapshot to Relay.

        MBridge weight export may perform collectives, so it runs on the Ray
        RPC stack on every rank. Only the selected publisher rank sends tensors
        to Relay; non-publisher ranks iterate and drop tensors immediately.
        """
        if self.is_sending_to_relay:
            raise RuntimeError(f"Relay actor weight send is already in progress on rank {self.rank}")
        should_publish = self._should_publish_relay_actor_weights()
        if should_publish:
            self._require_relay_server_address()
        relay_dtype = self._normalize_relay_dtype(dtype)
        self.is_sending_to_relay = True
        return await self._send_weights(version=version, dtype=relay_dtype)

    async def _send_weights(
        self,
        version: int,
        dtype: torch.dtype | None = None,
    ) -> ActorPublishStats | None:
        should_publish = self._should_publish_relay_actor_weights()
        _send_start_time = time.perf_counter()
        tensor_count = 0
        sent_tensor_count = 0
        export_time_s = 0.0
        connect_time_s = 0.0
        sender_time_s = 0.0
        transfer_stats = TensorTransferStats()
        logger.info(
            "[Rank %s] Megatron _send_weights: start version=%s publish=%s",
            self.rank,
            version,
            should_publish,
        )
        endpoint = None
        chunk_sender = None
        send_failure: tuple[RuntimeError, Exception] | None = None
        try:
            try:
                # MBridge export may contain collectives, so all ranks perform the
                # export on this RPC stack. Only the publisher rank ships tensors.
                tensor_iter = iter(self._iter_relay_actor_tensors(dtype=dtype, publish_only=False))
                while True:
                    export_start = time.perf_counter()
                    try:
                        name, tensor = next(tensor_iter)
                    except StopIteration:
                        export_time_s += time.perf_counter() - export_start
                        break
                    export_time_s += time.perf_counter() - export_start
                    tensor_count += 1
                    if not should_publish or send_failure is not None:
                        del tensor
                        continue
                    storage_meta = TensorStorageMeta(
                        key=name,
                        data_meta=TensorMeta(dtype=tensor.dtype, shape=tensor.shape),
                        version=version,
                    )
                    try:
                        if chunk_sender is None:
                            ip, port = self._require_relay_server_address()
                            connect_start = time.perf_counter()
                            endpoint = await asyncio.wait_for(
                                create_endpoint(ip, port),
                                timeout=_RELAY_CONNECT_TIMEOUT_S,
                            )
                            connect_time_s += time.perf_counter() - connect_start
                            chunk_sender = TensorChunkSender(endpoint)
                        sender_start = time.perf_counter()
                        await chunk_sender.add(storage_meta, tensor)
                        sender_time_s += time.perf_counter() - sender_start
                    except Exception as exc:
                        send_failure = (
                            RuntimeError(
                                "Megatron Relay tensor send failed: "
                                f"version={version} tensor_index={tensor_count - 1} "
                                f"sent_tensor_count={sent_tensor_count} name={name!r} "
                                f"shape={tuple(tensor.shape)} dtype={tensor.dtype} numel={tensor.numel()} "
                                f"chunk_count={chunk_sender.chunk_count if chunk_sender is not None else 0}"
                            ),
                            exc,
                        )
                    else:
                        sent_tensor_count += 1
                    del tensor
                if chunk_sender is not None and send_failure is None:
                    sender_start = time.perf_counter()
                    try:
                        transfer_stats = await chunk_sender.finish()
                    except Exception as exc:
                        send_failure = (RuntimeError(f"Megatron Relay transfer failed for version={version}"), exc)
                    finally:
                        sender_time_s += time.perf_counter() - sender_start
            finally:
                if endpoint is not None:
                    await endpoint.close()
        finally:
            self.is_sending_to_relay = False
            self._cleanup_relay_actor_export()

        if send_failure is not None:
            error, cause = send_failure
            raise error from cause
        logger.info(
            "[Rank %s] Megatron _send_weights: done version=%s publish=%s tensors=%s sent=%s elapsed=%.3fs",
            self.rank,
            version,
            should_publish,
            tensor_count,
            sent_tensor_count,
            time.perf_counter() - _send_start_time,
        )
        if not should_publish:
            return None
        return ActorPublishStats(
            publisher_count=1,
            tensor_count=tensor_count,
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
            total_time_s=time.perf_counter() - _send_start_time,
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, execute_mode=Execute.ALL, blocking=False)
    async def wait_for_sending_to_relay(self) -> None:
        if self.is_sending_to_relay:
            wait_logger = _ThrottledWaitLogger(owner=f"[Rank {self.rank}] wait_for_megatron_relay_send")
            _wait_start_time = time.perf_counter()
            while self.is_sending_to_relay:
                wait_logger.maybe_warn("previous Megatron Relay send is still in flight")
                await asyncio.sleep(0.1)
            logger.info(
                "[Rank %s] wait_for_sending_to_relay: waited %.3fs",
                self.rank,
                time.perf_counter() - _wait_start_time,
            )

    @register(
        dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"),
        execute_mode=Execute.ALL,
        blocking=True,
    )
    async def compute_log_prob(self, data: TensorDict) -> TensorDict:
        """Fence MBridge export collectives before actor inference collectives."""
        await self.wait_for_sending_to_relay()
        return super().compute_log_prob(data)

    @register(
        dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"),
        execute_mode=Execute.ALL,
        blocking=True,
    )
    async def update_actor(self, data: TensorDict) -> TensorDict:
        await self.wait_for_sending_to_relay()
        log_first_update_memory = not getattr(self, "_first_actor_update_memory_logged", False)
        if log_first_update_memory:
            self._first_actor_update_memory_logged = True
            self._log_cuda_memory("first_actor_update_before")
        try:
            with observe_optimizer_updates(self._actor_engine, backend="megatron") as update_stats:
                output = super().update_actor(data)
            return add_optimizer_update_metrics(output, update_stats)
        finally:
            if log_first_update_memory:
                self._log_cuda_memory("first_actor_update_after")

    def _require_relay_server_address(self) -> tuple[str, int]:
        assert "actor" in self.role, "Relay weight publish only supports actor role"
        if self.relay_server_address is None:
            raise RuntimeError("Relay server is not connected; call connect_to_relay_server first")
        return self.relay_server_address

    def _should_publish_relay_actor_weights(self) -> bool:
        assert "actor" in self.role, "Relay weight publish only supports actor role"
        engine = self._actor_engine
        return bool(engine.is_mp_src_rank_with_outputs() and engine.get_data_parallel_rank() == 0)

    def _iter_relay_actor_tensors(
        self,
        dtype: torch.dtype | str | None = None,
        *,
        publish_only: bool = False,
    ):
        assert "actor" in self.role, "Relay weight publish only supports actor role"
        should_publish = self._should_publish_relay_actor_weights()
        relay_dtype = self._normalize_relay_dtype(dtype)
        per_tensor_param, _ = self._actor_engine.get_per_tensor_param()
        for name, tensor in per_tensor_param:
            # MBridge export may run collectives; all ranks must consume the iterator.
            if publish_only and not should_publish:
                continue
            if should_publish and relay_dtype is not None and tensor.dtype != relay_dtype:
                tensor = tensor.to(dtype=relay_dtype)
            yield name, tensor

    def _normalize_relay_dtype(self, dtype: torch.dtype | str | None) -> torch.dtype | None:
        if dtype is None or isinstance(dtype, torch.dtype):
            return dtype
        return PrecisionType.to_dtype(dtype)

    def _cleanup_relay_actor_export(self) -> None:
        try:
            engine = self._actor_engine
            clear_router_replay_state = getattr(engine, "clear_router_replay_state", None)
            if callable(clear_router_replay_state):
                clear_router_replay_state()
            bridge_export_buffer = getattr(getattr(engine, "bridge", None), "export_weights_buff", None)
            if isinstance(bridge_export_buffer, dict):
                bridge_export_buffer.clear()
            if engine.is_param_offload_enabled:
                engine.to(device="cpu", model=True, optimizer=False, grad=False)
            gc.collect()
            ipc_collect = getattr(get_torch_device(), "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
            aggressive_empty_cache(force_sync=True)
        except Exception:
            logger.warning("Failed to cleanup Megatron Relay actor export memory", exc_info=True)
