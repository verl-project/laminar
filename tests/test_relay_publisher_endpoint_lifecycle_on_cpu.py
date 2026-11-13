# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch

from laminar.utils.tensor import TensorMeta
from laminar.workers import fsdp_workers, megatron_workers


class _Endpoint:
    def __init__(self, ip: str, port: int):
        self.address = (ip, port)
        self.closed = False

    async def close(self):
        self.closed = True


def test_fsdp_full_state_metadata_is_materialized_in_relay_dtype(monkeypatch):
    tensors = {
        "z.weight": SimpleNamespace(data=torch.empty(2, dtype=torch.float32)),
        "a.weight": SimpleNamespace(data=torch.empty(3, dtype=torch.float16)),
    }
    worker = object.__new__(fsdp_workers.AsyncActorWorker)
    worker.actor = SimpleNamespace(engine=SimpleNamespace(module=object()))
    monkeypatch.setattr(fsdp_workers, "get_model_state_dict", lambda *args, **kwargs: tensors)

    metas = asyncio.run(worker.get_actor_state_meta_dict(dtype="bfloat16"))

    assert list(metas) == ["a.weight", "z.weight"]
    assert metas == {
        "a.weight": TensorMeta(dtype=torch.bfloat16, shape=torch.Size([3])),
        "z.weight": TensorMeta(dtype=torch.bfloat16, shape=torch.Size([2])),
    }


def test_megatron_full_state_metadata_is_materialized_in_relay_dtype():
    worker = object.__new__(megatron_workers.AsyncMegatronActorWorker)
    worker._should_publish_relay_actor_weights = lambda: True
    worker._iter_relay_actor_tensors = lambda **kwargs: iter([("dense.weight", torch.empty(2, 3, dtype=torch.float32))])
    worker._cleanup_relay_actor_export = lambda: None

    metas = worker.get_actor_state_meta_dict(dtype="bfloat16")

    assert metas == {
        "dense.weight": TensorMeta(dtype=torch.bfloat16, shape=torch.Size([2, 3])),
    }


def _make_megatron_publisher(monkeypatch, *, tensors, sender_factory):
    endpoints = []

    async def create_endpoint(ip, port):
        endpoint = _Endpoint(ip, port)
        endpoints.append(endpoint)
        return endpoint

    worker = object.__new__(megatron_workers.AsyncMegatronActorWorker)
    worker.role = "actor"
    worker._rank = 0
    worker.is_sending_to_relay = True
    worker.relay_server_address = ("127.0.0.1", 1234)
    worker._should_publish_relay_actor_weights = lambda: True
    worker._cleanup_relay_actor_export = lambda: None
    worker._iter_relay_actor_tensors = lambda *args, **kwargs: iter(tensors)
    monkeypatch.setattr(megatron_workers, "create_endpoint", create_endpoint)
    monkeypatch.setattr(megatron_workers, "TensorChunkSender", sender_factory)
    return worker, endpoints


def test_megatron_publisher_reports_transfer_stats(monkeypatch):
    add_calls = []

    class _ChunkSender:
        def __init__(self, endpoint):
            self.endpoint = endpoint
            self.chunk_count = 1

        async def add(self, storage_meta, tensor):
            add_calls.append((storage_meta.key, storage_meta.version))

        async def finish(self):
            return megatron_workers.TensorTransferStats(chunk_count=1, payload_bytes=16)

    worker, endpoints = _make_megatron_publisher(
        monkeypatch,
        tensors=[("model.embed_tokens.weight", torch.ones(4))],
        sender_factory=_ChunkSender,
    )

    stats = asyncio.run(worker._send_weights(version=7))

    assert add_calls == [("model.embed_tokens.weight", 7)]
    assert len(endpoints) == 1
    assert all(endpoint.closed for endpoint in endpoints)
    assert stats is not None
    assert stats.tensor_count == 1
    assert stats.chunk_count == 1
    assert stats.payload_bytes == 16
    assert worker.is_sending_to_relay is False


def test_megatron_publisher_drains_export_after_send_failure(monkeypatch):
    add_calls = []
    exported = []

    class _FailingChunkSender:
        def __init__(self, endpoint):
            self.chunk_count = 1

        async def add(self, storage_meta, tensor):
            add_calls.append(storage_meta.key)
            if len(add_calls) == 2:
                raise ConnectionError("connection reset")

        async def finish(self):
            raise AssertionError("finish must not be reached after a failed send")

    def tensors():
        for name in ("embed", "layer.0", "layer.1"):
            exported.append(name)
            yield name, torch.ones(4)

    worker, endpoints = _make_megatron_publisher(
        monkeypatch,
        tensors=tensors(),
        sender_factory=_FailingChunkSender,
    )

    with pytest.raises(RuntimeError, match="Megatron Relay tensor send failed") as exc_info:
        asyncio.run(worker._send_weights(version=11))

    assert "tensor_index=1" in str(exc_info.value)
    assert "sent_tensor_count=1" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ConnectionError)
    assert add_calls == ["embed", "layer.0"]
    assert exported == ["embed", "layer.0", "layer.1"]
    assert len(endpoints) == 1
    assert all(endpoint.closed for endpoint in endpoints)
    assert worker.is_sending_to_relay is False


def test_megatron_compute_log_prob_fences_relay_export(monkeypatch):
    events = []
    worker = object.__new__(megatron_workers.AsyncMegatronActorWorker)

    async def wait_for_sending_to_relay():
        events.append("relay_export_done")

    def compute_log_prob(_worker, data):
        events.append(("compute_log_prob", data))
        return "result"

    worker.wait_for_sending_to_relay = wait_for_sending_to_relay
    monkeypatch.setattr(megatron_workers.ActorRolloutRefWorker, "compute_log_prob", compute_log_prob)
    data = object()

    result = asyncio.run(worker.compute_log_prob(data))

    assert result == "result"
    assert events == ["relay_export_done", ("compute_log_prob", data)]


def test_megatron_init_model_prewarms_cublas_before_model_allocation(monkeypatch):
    events = []
    worker = object.__new__(megatron_workers.AsyncMegatronActorWorker)
    worker._is_rollout = False
    worker._rank = 3
    grad_stats_group = object()
    probe = object()

    class _Optimizer:
        def get_grad_stats_parallel_group(self):
            events.append("get_grad_stats_parallel_group")
            return grad_stats_group

    def init_model(_worker):
        events.append("base_init_model")
        _worker.actor = SimpleNamespace(engine=SimpleNamespace(optimizer=_Optimizer()))
        _worker.loss_fn = object()

    worker._log_cuda_memory = lambda event: events.append(event)
    monkeypatch.setattr(megatron_workers.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(megatron_workers.torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(megatron_workers.torch.cuda, "current_blas_handle", lambda: events.append("cublas_handle"))
    monkeypatch.setattr(megatron_workers.torch.cuda, "synchronize", lambda: events.append("cuda_synchronize"))
    monkeypatch.setattr(megatron_workers.torch.distributed, "is_initialized", lambda: True)

    def zeros(size, *, dtype, device):
        events.append(("zeros", size, dtype, device))
        return probe

    def all_reduce(tensor, *, op, group):
        events.append(("all_reduce", tensor, op, group))

    monkeypatch.setattr(megatron_workers.torch, "zeros", zeros)
    monkeypatch.setattr(megatron_workers.torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(megatron_workers.ActorRolloutRefWorker, "init_model", init_model)

    worker.init_model()

    assert events == [
        "cublas_prewarm_before",
        "cublas_handle",
        "cublas_prewarm_after",
        "base_init_model",
        "get_grad_stats_parallel_group",
        "grad_stats_nccl_prewarm_before",
        ("zeros", 1, torch.float32, torch.device("cuda", 2)),
        ("all_reduce", probe, torch.distributed.ReduceOp.SUM, grad_stats_group),
        "cuda_synchronize",
        "grad_stats_nccl_prewarm_after",
    ]
    assert worker.loss_fn is not None


def test_megatron_init_model_skips_cublas_prewarm_without_cuda(monkeypatch):
    events = []
    worker = object.__new__(megatron_workers.AsyncMegatronActorWorker)
    worker._is_rollout = False

    def init_model(_worker):
        events.append("base_init_model")
        _worker.actor = SimpleNamespace(
            engine=SimpleNamespace(
                optimizer=SimpleNamespace(
                    get_grad_stats_parallel_group=lambda: (_ for _ in ()).throw(
                        AssertionError("NCCL prewarm must be skipped")
                    )
                )
            )
        )
        _worker.loss_fn = object()

    monkeypatch.setattr(megatron_workers.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        megatron_workers.torch.cuda,
        "current_blas_handle",
        lambda: (_ for _ in ()).throw(AssertionError("CUDA prewarm must be skipped")),
    )
    monkeypatch.setattr(megatron_workers.ActorRolloutRefWorker, "init_model", init_model)

    worker.init_model()

    assert events == ["base_init_model"]


def _make_grad_stats_prewarm_worker(group):
    worker = object.__new__(megatron_workers.AsyncMegatronActorWorker)
    worker.actor = SimpleNamespace(
        engine=SimpleNamespace(optimizer=SimpleNamespace(get_grad_stats_parallel_group=lambda: group))
    )
    worker._log_cuda_memory = lambda _event: None
    return worker


def test_megatron_grad_stats_prewarm_requires_initialized_distributed(monkeypatch):
    worker = _make_grad_stats_prewarm_worker(object())
    monkeypatch.setattr(megatron_workers.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(megatron_workers.torch.distributed, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="distributed process group is not initialized"):
        worker._prewarm_grad_stats_communicators()


def test_megatron_grad_stats_prewarm_rejects_missing_group(monkeypatch):
    worker = _make_grad_stats_prewarm_worker(None)
    monkeypatch.setattr(megatron_workers.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(megatron_workers.torch.distributed, "is_initialized", lambda: True)

    with pytest.raises(RuntimeError, match="Megatron grad-stats process group is missing"):
        worker._prewarm_grad_stats_communicators()


def test_megatron_grad_stats_prewarm_covers_unique_chained_optimizer_groups(monkeypatch):
    events = []
    first_group = object()
    second_group = object()
    probe = object()

    class _Optimizer:
        def __init__(self, group):
            self.group = group

        def get_grad_stats_parallel_group(self):
            return self.group

    class _ChainedOptimizer:
        def __init__(self):
            self.chained_optimizers = [
                _Optimizer(first_group),
                _Optimizer(second_group),
                _Optimizer(first_group),
            ]

        def grads_states_parallel_group_is_shared(self):
            return False

    worker = object.__new__(megatron_workers.AsyncMegatronActorWorker)
    worker.actor = SimpleNamespace(engine=SimpleNamespace(optimizer=_ChainedOptimizer()))
    worker._log_cuda_memory = lambda event: events.append(event)
    monkeypatch.setattr(megatron_workers, "ChainedOptimizer", _ChainedOptimizer)
    monkeypatch.setattr(megatron_workers.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(megatron_workers.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(megatron_workers.torch.cuda, "synchronize", lambda: events.append("cuda_synchronize"))
    monkeypatch.setattr(megatron_workers.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(megatron_workers.torch, "zeros", lambda *_args, **_kwargs: probe)
    monkeypatch.setattr(
        megatron_workers.torch.distributed,
        "all_reduce",
        lambda tensor, *, op, group: events.append(("all_reduce", tensor, op, group)),
    )

    worker._prewarm_grad_stats_communicators()

    assert events == [
        "grad_stats_nccl_prewarm_before",
        ("all_reduce", probe, torch.distributed.ReduceOp.SUM, first_group),
        ("all_reduce", probe, torch.distributed.ReduceOp.SUM, second_group),
        "cuda_synchronize",
        "grad_stats_nccl_prewarm_after",
    ]


def test_megatron_grad_stats_prewarm_uses_one_shared_chained_group(monkeypatch):
    events = []
    shared_group = object()
    probe = object()

    class _Optimizer:
        def get_grad_stats_parallel_group(self):
            raise AssertionError("shared outer chain must own group selection")

    class _ChainedOptimizer:
        chained_optimizers = [_Optimizer(), _Optimizer()]

        def grads_states_parallel_group_is_shared(self):
            return True

        def get_grad_stats_parallel_group(self):
            return shared_group

    worker = object.__new__(megatron_workers.AsyncMegatronActorWorker)
    worker.actor = SimpleNamespace(engine=SimpleNamespace(optimizer=_ChainedOptimizer()))
    worker._log_cuda_memory = lambda event: events.append(event)
    monkeypatch.setattr(megatron_workers, "ChainedOptimizer", _ChainedOptimizer)
    monkeypatch.setattr(megatron_workers.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(megatron_workers.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(megatron_workers.torch.cuda, "synchronize", lambda: events.append("cuda_synchronize"))
    monkeypatch.setattr(megatron_workers.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(megatron_workers.torch, "zeros", lambda *_args, **_kwargs: probe)
    monkeypatch.setattr(
        megatron_workers.torch.distributed,
        "all_reduce",
        lambda tensor, *, op, group: events.append(("all_reduce", tensor, op, group)),
    )

    worker._prewarm_grad_stats_communicators()

    assert events == [
        "grad_stats_nccl_prewarm_before",
        ("all_reduce", probe, torch.distributed.ReduceOp.SUM, shared_group),
        "cuda_synchronize",
        "grad_stats_nccl_prewarm_after",
    ]


def test_megatron_grad_stats_prewarm_uses_world_for_layerwise_and_nested_unshared_chain(monkeypatch):
    events = []
    world_group = object()
    other_group = object()
    probe = object()

    class _Optimizer:
        def __init__(self, group):
            self.group = group

        def get_grad_stats_parallel_group(self):
            return self.group

    class _ChainedOptimizer:
        def __init__(self, children):
            self.chained_optimizers = children

        def grads_states_parallel_group_is_shared(self):
            return False

    class _LayerWiseDistributedOptimizer(_ChainedOptimizer):
        pass

    layerwise = _LayerWiseDistributedOptimizer([_Optimizer(object()), _Optimizer(object())])
    outer = _ChainedOptimizer([layerwise, _Optimizer(other_group), layerwise])
    worker = object.__new__(megatron_workers.AsyncMegatronActorWorker)
    worker.actor = SimpleNamespace(engine=SimpleNamespace(optimizer=outer))
    worker._log_cuda_memory = lambda event: events.append(event)
    monkeypatch.setattr(megatron_workers, "ChainedOptimizer", _ChainedOptimizer)
    monkeypatch.setattr(megatron_workers, "LayerWiseDistributedOptimizer", _LayerWiseDistributedOptimizer)
    monkeypatch.setattr(megatron_workers.torch.distributed, "group", SimpleNamespace(WORLD=world_group))
    monkeypatch.setattr(megatron_workers.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(megatron_workers.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(megatron_workers.torch.cuda, "synchronize", lambda: events.append("cuda_synchronize"))
    monkeypatch.setattr(megatron_workers.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(megatron_workers.torch, "zeros", lambda *_args, **_kwargs: probe)
    monkeypatch.setattr(
        megatron_workers.torch.distributed,
        "all_reduce",
        lambda tensor, *, op, group: events.append(("all_reduce", tensor, op, group)),
    )

    worker._prewarm_grad_stats_communicators()

    assert events == [
        "grad_stats_nccl_prewarm_before",
        ("all_reduce", probe, torch.distributed.ReduceOp.SUM, world_group),
        ("all_reduce", probe, torch.distributed.ReduceOp.SUM, other_group),
        "cuda_synchronize",
        "grad_stats_nccl_prewarm_after",
    ]


@pytest.mark.parametrize("failing_call", ["all_reduce", "synchronize"])
def test_megatron_grad_stats_prewarm_propagates_collective_failure(monkeypatch, failing_call):
    group = object()
    probe = object()
    failure = RuntimeError(f"sentinel {failing_call} failure")
    worker = _make_grad_stats_prewarm_worker(group)
    monkeypatch.setattr(megatron_workers.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(megatron_workers.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(megatron_workers.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(megatron_workers.torch, "zeros", lambda *_args, **_kwargs: probe)

    def all_reduce(tensor, *, op, group):
        if failing_call == "all_reduce":
            raise failure

    def synchronize():
        if failing_call == "synchronize":
            raise failure

    monkeypatch.setattr(megatron_workers.torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(megatron_workers.torch.cuda, "synchronize", synchronize)

    with pytest.raises(RuntimeError) as exc_info:
        worker._prewarm_grad_stats_communicators()

    assert exc_info.value is failure


def test_megatron_update_actor_logs_first_update_memory_once(monkeypatch):
    from verl.utils import tensordict_utils as tu

    events = []
    worker = object.__new__(megatron_workers.AsyncMegatronActorWorker)
    worker._first_actor_update_memory_logged = False
    worker.actor = SimpleNamespace(engine=SimpleNamespace(optimizer_step=lambda: None))

    async def wait_for_sending_to_relay():
        events.append("relay_export_done")

    def update_actor(_worker, data):
        events.append(("update_actor", data))
        return data

    worker.wait_for_sending_to_relay = wait_for_sending_to_relay
    worker._log_cuda_memory = lambda event: events.append(event)
    monkeypatch.setattr(megatron_workers.ActorRolloutRefWorker, "update_actor", update_actor)
    first_data = tu.get_tensordict({}, non_tensor_dict={"metrics": {}})
    second_data = tu.get_tensordict({}, non_tensor_dict={"metrics": {}})

    assert asyncio.run(worker.update_actor(first_data)) is first_data
    assert asyncio.run(worker.update_actor(second_data)) is second_data

    assert events == [
        "relay_export_done",
        "first_actor_update_before",
        ("update_actor", first_data),
        "first_actor_update_after",
        "relay_export_done",
        ("update_actor", second_data),
    ]


def test_megatron_update_actor_logs_after_memory_without_masking_failure(monkeypatch):
    events = []
    worker = object.__new__(megatron_workers.AsyncMegatronActorWorker)
    worker._first_actor_update_memory_logged = False
    worker.actor = SimpleNamespace(engine=SimpleNamespace(optimizer_step=lambda: None))
    failure = RuntimeError("sentinel actor update failure")

    async def wait_for_sending_to_relay():
        events.append("relay_export_done")

    def update_actor(_worker, _data):
        events.append("update_actor")
        raise failure

    worker.wait_for_sending_to_relay = wait_for_sending_to_relay
    worker._log_cuda_memory = lambda event: events.append(event)
    monkeypatch.setattr(megatron_workers.ActorRolloutRefWorker, "update_actor", update_actor)

    with pytest.raises(RuntimeError) as exc_info:
        asyncio.run(worker.update_actor(object()))

    assert exc_info.value is failure
    assert events == [
        "relay_export_done",
        "first_actor_update_before",
        "update_actor",
        "first_actor_update_after",
    ]


def test_fsdp_publisher_uses_a_fresh_endpoint_after_materialization(monkeypatch):
    endpoints = []
    events = []

    async def create_endpoint(ip, port):
        endpoint = _Endpoint(ip, port)
        endpoints.append(endpoint)
        return endpoint

    async def send_tensors(endpoint, *, storage_metas, tensors):
        events.append(
            (
                "sent",
                endpoint,
                len(storage_metas),
                len(tensors),
                [storage_meta.version for storage_meta in storage_metas],
            )
        )
        return fsdp_workers.TensorTransferStats(
            chunk_count=1,
            payload_bytes=16,
            tensor_prepare_time_s=0.01,
            pack_time_s=0.02,
            staging_time_s=0.03,
            transport_time_s=0.04,
            completion_ack_time_s=0.05,
        )

    worker = object.__new__(fsdp_workers.AsyncActorWorker)
    worker.role = "actor"
    worker._rank = 0
    worker.actor = SimpleNamespace(
        engine=SimpleNamespace(device_mesh={"fsdp": SimpleNamespace(get_local_rank=lambda: 0)})
    )
    worker.relay_server_address = None

    def materialize(*args, **kwargs):
        events.append(("materialized", len(endpoints)))
        tensor = torch.ones(4)
        storage_meta = fsdp_workers.TensorStorageMeta(
            key=("model.layers.0.weight", 0),
            data_meta=TensorMeta(dtype=tensor.dtype, shape=tensor.shape),
            version=9,
        )
        return [tensor], [storage_meta]

    worker._get_local_tensors_and_storage_metas = materialize
    monkeypatch.setattr(fsdp_workers, "create_endpoint", create_endpoint)
    monkeypatch.setattr(fsdp_workers, "send_tensors", send_tensors)

    asyncio.run(worker.connect_to_relay_server("127.0.0.1", 5678))

    assert worker.relay_server_address == ("127.0.0.1", 5678)
    assert endpoints[0].closed is True
    assert not hasattr(worker, "relay_server_ep")

    stats = asyncio.run(worker._send_weights(version=9))

    assert len(endpoints) == 2
    assert endpoints[1].closed is True
    assert events[0] == ("materialized", 1)
    assert events[1] == ("sent", endpoints[1], 1, 1, [9])
    assert stats.payload_bytes == 16
    assert stats.sender_time_s > 0.0
    assert stats.transport_time_s == 0.04


def test_fsdp_relay_send_future_is_the_completion_state():
    worker = object.__new__(fsdp_workers.AsyncActorWorker)
    worker._rank = 0
    send_future = Future()
    stats = fsdp_workers.ActorPublishStats(publisher_count=1, payload_bytes=16)
    worker._relay_send_future = send_future
    send_future.set_result(stats)

    assert asyncio.run(worker.wait_for_sending_to_relay()) == stats
    assert worker._relay_send_future is None


def test_fsdp_relay_send_future_preserves_failure_and_cancellation():
    worker = object.__new__(fsdp_workers.AsyncActorWorker)
    worker._rank = 0
    failed_future = Future()
    worker._relay_send_future = failed_future
    failed_future.set_exception(ConnectionError("receiver closed"))

    with pytest.raises(ConnectionError, match="receiver closed"):
        asyncio.run(worker.wait_for_sending_to_relay())
    assert worker._relay_send_future is None

    pending_future = Future()
    worker._relay_send_future = pending_future

    async def cancel_wait():
        wait_task = asyncio.create_task(worker.wait_for_sending_to_relay())
        await asyncio.sleep(0)
        wait_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await wait_task

    asyncio.run(cancel_wait())

    assert worker._relay_send_future is pending_future
    pending_future.set_result(fsdp_workers.ActorPublishStats(publisher_count=1))
    assert asyncio.run(worker.wait_for_sending_to_relay()).publisher_count == 1
    assert worker._relay_send_future is None


def test_relay_actor_send_rejects_overlap():
    worker = object.__new__(fsdp_workers.AsyncActorWorker)
    worker._rank = 3
    worker._relay_send_future = Future()

    with pytest.raises(RuntimeError, match="already in progress on rank 3"):
        asyncio.run(worker.send_weights(version=10, dtype="bfloat16"))


def test_megatron_worker_neutralizes_broken_flash_attn_cute_and_preserves_working_fa4(monkeypatch):
    import importlib
    import importlib.util
    import sys
    import types

    from verl.models.mcore.patch import _MissingFlashAttnCute

    monkeypatch.setattr(
        sys,
        "meta_path",
        [finder for finder in sys.meta_path if not isinstance(finder, _MissingFlashAttnCute)],
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name == "flash_attn.cute" else None)

    partial = types.ModuleType("flash_attn.cute.utils")
    monkeypatch.setitem(sys.modules, "flash_attn.cute.utils", partial)

    def raise_broken_fa4(name):
        assert name == "flash_attn.cute"
        raise AttributeError("module 'cutlass.cute.core' has no attribute 'ThrMma'")

    monkeypatch.setattr(importlib, "import_module", raise_broken_fa4)

    megatron_workers.neutralize_broken_flash_attn_cute()
    megatron_workers.neutralize_broken_flash_attn_cute()  # Idempotent when called again.

    assert "flash_attn.cute.utils" not in sys.modules
    assert sum(isinstance(finder, _MissingFlashAttnCute) for finder in sys.meta_path) == 1
    finder = sys.meta_path[0]
    with pytest.raises(ModuleNotFoundError, match="flash_attn.cute"):
        finder.find_spec("flash_attn.cute")
    with pytest.raises(ModuleNotFoundError, match="flash_attn.cute.interface"):
        finder.find_spec("flash_attn.cute.interface")
    assert finder.find_spec("flash_attn.flash_attn_interface") is None


def test_megatron_worker_leaves_working_or_absent_flash_attn_cute_untouched(monkeypatch):
    import importlib
    import importlib.util
    import sys
    import types

    from verl.models.mcore.patch import _MissingFlashAttnCute

    monkeypatch.setattr(
        sys,
        "meta_path",
        [finder for finder in sys.meta_path if not isinstance(finder, _MissingFlashAttnCute)],
    )
    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(importlib, "import_module", lambda name: types.ModuleType(name))

    megatron_workers.neutralize_broken_flash_attn_cute()
    assert not any(isinstance(finder, _MissingFlashAttnCute) for finder in sys.meta_path)
