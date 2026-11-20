# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Laminar ownership of Megatron's staged asynchronous checkpoint writes.

The pinned verl manager builds the state dictionaries and checkpoint layout.
Laminar supplies the MCore queue and owns completion; only the trainer commits
the joint actor/prompt-cursor latest marker. No Megatron training globals are
initialized or shared with the training engine.
"""

from __future__ import annotations

import inspect
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from types import MethodType
from typing import Any, Callable


class ThreadAsyncCaller:
    """Native request staging on the actor thread, CPU writes on one I/O thread.

    The caller never runs completion callbacks. Native AsyncCallsQueue keeps
    those callbacks and their collectives on the actor RPC thread. Completion
    waits do not raise a local Future exception before ranks agree on failure.
    """

    def __init__(self, executor, *, synchronize, sync_all_done, sync_failure):
        self._executor = executor
        self._synchronize = synchronize
        self._sync_all_done = sync_all_done
        self._sync_failure = sync_failure
        self._future: Future | None = None

    def schedule_async_call(self, request):
        args = list(request.async_fn_args)
        if request.preload_fn is not None:
            args[1] = request.preload_fn()
        self._synchronize()
        if request.async_fn is None:
            self._future = Future()
            self._future.set_result(None)
        else:
            self._future = self._executor.submit(request.async_fn, *args, **request.async_fn_kwargs)

    def is_current_async_call_done(self, blocking=False, no_dist=False):
        if self._future is None:
            return True
        is_alive = int(not self._future.done())
        is_done = not is_alive if no_dist else self._sync_all_done(is_alive)
        if not (is_done or blocking):
            return False
        # wait(), unlike result(), leaves exceptions for the all-rank guard.
        self.close()
        error = self._future.exception()
        if no_dist:
            if error is not None:
                raise error
        else:
            self._sync_failure(error)
        return True

    def close(self, abort=False):
        # Running native file I/O cannot be safely cancelled. Supported normal
        # shutdown drains it before the executor or checkpoint owner exits.
        if self._future is not None:
            wait([self._future])

    def __del__(self):
        # No collectives or blocking work during Python object destruction.
        pass


def _raise_if_writer_failed(error):
    import torch
    from torch.distributed.checkpoint import CheckpointException
    from torch.distributed.checkpoint.api import _wrap_exception

    failed = torch.tensor([int(error is not None)], dtype=torch.int, device=torch.cuda.current_device())
    torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
    if failed.item():
        messages = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(messages, repr(error) if error is not None else None)
        failures = {rank: _wrap_exception(RuntimeError(message)) for rank, message in enumerate(messages) if message}
        raise CheckpointException("Laminar async CPU checkpoint writer", failures)


def create_thread_checkpoint_queue(*, synchronize=None, sync_all_done=None, sync_failure=None):
    """Keep native queue/finalization with an injectable CPU-only caller gate."""
    import torch
    from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCaller, AsyncCallsQueue

    class NativeThreadCaller(ThreadAsyncCaller, AsyncCaller):
        def __init__(self, executor):
            super().__init__(
                executor,
                synchronize=synchronize or torch.cuda.synchronize,
                sync_all_done=sync_all_done or self.sync_all_async_calls,
                sync_failure=sync_failure or _raise_if_writer_failed,
            )

    class NativeThreadQueue(AsyncCallsQueue):
        def __init__(self):
            super().__init__(persistent=False)
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="laminar-checkpoint")

        def _get_async_caller(self):
            return NativeThreadCaller(self._executor)

        def close(self, abort=False):
            try:
                super().close(abort=abort)
            finally:
                self._executor.shutdown(wait=True)

    return NativeThreadQueue()


def _native_runtime() -> tuple[Any, Callable[..., Any]]:
    """Select the built-in MCore writer explicitly, without optional NVRx."""
    import queue as thread_queue
    from pathlib import Path

    from megatron.core import dist_checkpointing, mpu
    from megatron.core.dist_checkpointing.strategies import filesystem_async
    from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelSaveStrategyWrapper
    from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy

    if "async_strategy" not in inspect.signature(dist_checkpointing.save).parameters:
        raise NotImplementedError("Laminar async checkpoint requires MCore's explicit async_strategy save API")
    if filesystem_async._results_queue is None:
        # Both staging/finalization and the writer thread live in this actor
        # process; avoid spawning per-rank SyncManager/resource_tracker children
        # and pickling WriteResult lists over an IPC socket on every save.
        filesystem_async._results_queue = thread_queue.Queue()
    strategies: dict[tuple[str, tuple[str, ...]], Any] = {}

    def save_dist_checkpoint(manager, path, state_dict):
        # Cache sharding distribution and DCP plans per content tree ("model",
        # "optimizer", "extra") across steps, matching Megatron-Bridge's
        # constant-structure per-step save path after validating the first save.
        slot = (Path(path).parent.name, tuple(sorted(state_dict)))
        strategy = strategies.get(slot)
        first_save = strategy is None
        if first_save:
            strategy = FullyParallelSaveStrategyWrapper(
                TorchDistSaveShardedStrategy(backend="torch_dist", version=1, cached_metadata=True),
                mpu.get_data_parallel_group(with_context_parallel=True),
                do_cache_distribution=True,
            )
        request = dist_checkpointing.save(
            state_dict,
            path,
            sharded_strategy=strategy,
            async_sharded_save=True,
            async_strategy="mcore",
            validate_access_integrity=first_save,
            content_metadata=manager._build_sharded_state_dict_metadata(),
        )
        if first_save:
            strategies[slot] = strategy
        return request

    # Native preload finishes D2H before schedule returns. A thread avoids
    # forking the live actor address space on every checkpoint.
    return create_thread_checkpoint_queue(), save_dist_checkpoint


def _isolate_cpu_snapshots(request, metrics=None):
    """Clone CPU sources that MCore's D2H preload would otherwise alias.

    GPU tensors already get independent host storage from native preload.
    CPU/offloaded tensors may be backed by shared storage, for which fork's
    copy-on-write does not isolate subsequent optimizer mutations.
    """
    if request.preload_fn is None:
        return request
    source_buckets = request.async_fn_args[1]
    native_preload = request.preload_fn

    def preload():
        staged_buckets = native_preload()
        result = []
        snapshot_bytes = 0
        cpu_copy_bytes = 0
        for source_bucket, staged_bucket in zip(source_buckets, staged_buckets, strict=True):
            _, _, (_, source_tensors) = source_bucket
            path, key, (byte_data, staged_tensors) = staged_bucket
            tensors = [
                (item, tensor.detach().clone() if source.device.type == "cpu" else tensor)
                for (_, source), (item, tensor) in zip(source_tensors, staged_tensors, strict=True)
            ]
            for (_, source), (_, tensor) in zip(source_tensors, tensors, strict=True):
                size_bytes = tensor.numel() * tensor.element_size()
                snapshot_bytes += size_bytes
                if source.device.type == "cpu":
                    cpu_copy_bytes += size_bytes
            result.append((path, key, (byte_data, tensors)))
        if metrics is not None:
            for key, value in (
                ("checkpoint/worker_snapshot_bytes", snapshot_bytes),
                ("checkpoint/worker_cpu_snapshot_copy_bytes", cpu_copy_bytes),
            ):
                metrics[key] = metrics.get(key, 0.0) + float(value)
        return result

    return request._replace(preload_fn=preload)


def _write_checkpoint_contents(_rank, contents, _queue):
    # Only staged CPU tensors/BytesIO reach the native writers. Their pinned
    # implementation temporarily pauses process-wide automatic cyclic GC and
    # restores its prior enabled state on normal completion or exception.
    # Keep native fsync and serialization, and never change process priority.
    for write_fn, args, kwargs in contents:
        if write_fn is not None:
            write_fn(*args, **kwargs)


def _bundle_requests(requests, finalize_save_fn, metrics):
    """Stage all trees, then write and finalize them in matching FIFO order.

    Pinned MCore 0.19.2 uses one process-wide results queue without request
    IDs. Independent writers may finish out of order, causing one content's
    metadata to consume another content's results. One native request keeps
    matching order on the dedicated I/O thread.
    """
    isolated = [_isolate_cpu_snapshots(request, metrics) for request in requests]

    def preload():
        contents = []
        for request in isolated:
            args = list(request.async_fn_args)
            if request.preload_fn is not None:
                args[1] = request.preload_fn()
            contents.append((request.async_fn, args, request.async_fn_kwargs))
        return contents

    return requests[0]._replace(
        async_fn=_write_checkpoint_contents,
        async_fn_args=(None, None, None),
        async_fn_kwargs={},
        preload_fn=preload,
        finalize_fns=[callback for request in requests for callback in request.finalize_fns] + [finalize_save_fn],
    )


class NativeAsyncCheckpoint:
    """One staged save at a time, completed by an explicit all-rank RPC.

    ``save`` is synchronous through state capture and host staging. After it
    returns, actor updates can proceed while the native writer owns the
    snapshot. ``finalize`` runs between actor RPCs, because native completion
    performs distributed collectives. Failed saves remain fatal and pending.
    """

    def __init__(self, manager, *, queue=None, save_dist_checkpoint=None):
        self.manager = manager
        self._pending_step: int | None = None
        self._request_count = 0
        self._snapshot_metrics = {}
        required = {"model", "optimizer", "extra"}
        if not bool(manager.checkpoint_config.async_save):
            raise ValueError("NativeAsyncCheckpoint requires checkpoint.async_save=True")
        if not manager.use_dist_checkpointing or manager.use_megatron_fsdp:
            raise ValueError("async checkpoint requires Megatron dist checkpointing without Megatron-FSDP")
        if set(manager.checkpoint_save_contents) != required or set(manager.checkpoint_load_contents) != required:
            raise ValueError("async checkpoint save/load contents must be exactly model, optimizer, extra")
        if manager.peft_cls is not None or manager.should_save_hf_model or manager.should_load_hf_model:
            raise ValueError("async checkpoint does not support HF export or PEFT")
        if not manager.use_checkpoint_opt_param_scheduler:
            raise ValueError("async checkpoint requires use_checkpoint_opt_param_scheduler=True")
        for name in ("_dispatch_finalize", "_finalize_save", "_save_dist_checkpoint", "_write_checkpoint_manifest"):
            if not callable(getattr(manager, name, None)):
                raise NotImplementedError(f"unsupported verl Megatron checkpoint manager: missing {name}")
        if queue is None or save_dist_checkpoint is None:
            native_queue, native_save = _native_runtime()
            queue = queue if queue is not None else native_queue
            save_dist_checkpoint = save_dist_checkpoint or native_save
        self.queue = queue
        self._save_dist_checkpoint = save_dist_checkpoint

        # Adapt the pinned manager at its existing lifecycle hooks. The save
        # composition and load path stay in verl; no second checkpoint layout
        # or scheduler is introduced.
        def save_dist(_manager, path, state_dict):
            return self._save_dist_checkpoint(_manager, path, state_dict)

        def dispatch(_manager, async_requests, finalize_save_fn):
            if not async_requests:
                raise RuntimeError("native async checkpoint produced no asynchronous requests")
            request = _bundle_requests(async_requests, finalize_save_fn, self._snapshot_metrics)
            self.queue.schedule_async_request(request)
            self._request_count = len(async_requests)

        def finalize_save(_manager, *, local_path, hdfs_path, global_step, max_ckpt_to_keep, saved_any_dist_ckpt):
            if hdfs_path is not None or max_ckpt_to_keep is not None:
                raise RuntimeError("Laminar trainer owns complete checkpoint storage and retention")
            # The native per-content callbacks completed on every rank first.
            # The backend's normal async callback also writes trainer's latest
            # marker before data.pt is durable, so it cannot be used here.
            _manager._write_checkpoint_manifest(local_path, global_step, saved_any_dist_ckpt)
            # Retention belongs to the trainer. Registering with no backend
            # limit would still accumulate a path entry for every train step.

        manager._save_dist_checkpoint = MethodType(save_dist, manager)
        manager._dispatch_finalize = MethodType(dispatch, manager)
        manager._finalize_save = MethodType(finalize_save, manager)

    def save(self, save_callable, *, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        if self._pending_step is not None:
            raise RuntimeError(f"checkpoint step {self._pending_step} must be finalized before another save")
        if hdfs_path is not None or max_ckpt_to_keep is not None:
            raise ValueError("Laminar trainer owns complete checkpoint storage and retention")
        self._pending_step = int(global_step)
        started = time.perf_counter()
        save_callable(
            local_path=local_path,
            hdfs_path=hdfs_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )
        if self._request_count == 0:
            raise RuntimeError("native async checkpoint did not schedule any writes")
        return {
            "checkpoint/worker_stage_time_s": time.perf_counter() - started,
            "checkpoint/worker_async_request_count": float(self._request_count),
            **self._snapshot_metrics,
        }

    def finalize(self) -> dict[str, float]:
        if self._pending_step is None:
            return {}
        started = time.perf_counter()
        self.queue.maybe_finalize_async_calls(blocking=True, no_dist=False)
        metrics = {
            "checkpoint/worker_finalize_time_s": time.perf_counter() - started,
            "checkpoint/worker_async_request_count": float(self._request_count),
        }
        self._pending_step = None
        self._request_count = 0
        self._snapshot_metrics = {}
        return metrics
