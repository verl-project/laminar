# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import threading
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from laminar.workers.checkpoint import NativeAsyncCheckpoint, ThreadAsyncCaller, _isolate_cpu_snapshots

_Request = namedtuple("_Request", "async_fn async_fn_args async_fn_kwargs finalize_fns preload_fn")


class _Queue:
    def __init__(self):
        self.requests = []
        self.error = None

    def schedule_async_request(self, request):
        args = list(request.async_fn_args)
        if request.preload_fn is not None:
            args[1] = request.preload_fn()
        request = request._replace(async_fn_args=args, preload_fn=None)
        self.requests.append(request)

    def maybe_finalize_async_calls(self, *, blocking, no_dist):
        assert blocking is True
        assert no_dist is False
        if self.error:
            raise self.error
        for request in self.requests:
            request.async_fn(*request.async_fn_args, **request.async_fn_kwargs)
            for callback in request.finalize_fns:
                callback()
        self.requests.clear()


class _Manager:
    def __init__(self):
        self.checkpoint_config = SimpleNamespace(async_save=True)
        self.checkpoint_save_contents = ["model", "optimizer", "extra"]
        self.checkpoint_load_contents = ["model", "optimizer", "extra"]
        self.use_dist_checkpointing = True
        self.use_megatron_fsdp = False
        self.use_checkpoint_opt_param_scheduler = True
        self.peft_cls = None
        self.should_save_hf_model = False
        self.should_load_hf_model = False
        self.events = []

    def _save_dist_checkpoint(self, path, state_dict):
        raise AssertionError("the incompatible backend default must be replaced")

    def _dispatch_finalize(self, requests, finalize):
        raise AssertionError("the backend singleton queue must be replaced")

    def _finalize_save(self, **kwargs):
        raise AssertionError("the actor-only backend latest marker must not be written")

    def _write_checkpoint_manifest(self, path, step, saved_any_dist_ckpt):
        assert saved_any_dist_ckpt is True
        assert self.events == ["model", "optimizer", "extra"]
        self.events.append("manifest")
        Path(path, "ckpt_contents.json").write_text(str(step))

    def register_checkpoint(self, path, max_to_keep):
        raise AssertionError("trainer-owned retention must not accumulate backend path history")

    def save_checkpoint(self, **kwargs):
        path = kwargs["local_path"]
        Path(path).mkdir(parents=True)
        requests = [self._save_dist_checkpoint(path, {name: 1}) for name in self.checkpoint_save_contents]
        self._dispatch_finalize(
            requests,
            lambda: self._finalize_save(**kwargs, saved_any_dist_ckpt=True),
        )


def _adapter(manager=None):
    manager = manager or _Manager()
    queue = _Queue()

    def native_save(manager, path, state_dict):
        name = next(iter(state_dict))
        return _Request(lambda: manager.events.append(name), (), {}, [], None)

    adapter = NativeAsyncCheckpoint(manager, queue=queue, save_dist_checkpoint=native_save)
    return manager, queue, adapter


def test_all_content_callbacks_finish_before_manifest_without_actor_latest_marker(tmp_path):
    manager, queue, adapter = _adapter()
    actor_dir = tmp_path / "global_step_12" / "actor"
    metrics = adapter.save(manager.save_checkpoint, local_path=str(actor_dir), global_step=12)

    assert metrics["checkpoint/worker_async_request_count"] == 3.0
    assert len(queue.requests) == 1
    assert manager.events == []
    assert not (actor_dir / "ckpt_contents.json").exists()
    assert not (tmp_path / "latest_checkpointed_iteration.txt").exists()

    metrics = adapter.finalize()

    assert manager.events == ["model", "optimizer", "extra", "manifest"]
    assert (actor_dir / "ckpt_contents.json").read_text() == "12"
    assert not (tmp_path / "latest_checkpointed_iteration.txt").exists()
    assert metrics["checkpoint/worker_finalize_time_s"] >= 0
    assert adapter.finalize() == {}


def test_another_save_requires_finalize_and_errors_remain_fatal(tmp_path):
    manager, queue, adapter = _adapter()
    adapter.save(manager.save_checkpoint, local_path=str(tmp_path / "actor"), global_step=12)
    with pytest.raises(RuntimeError, match="12 must be finalized"):
        adapter.save(manager.save_checkpoint, local_path=str(tmp_path / "other"), global_step=13)

    queue.error = OSError("writer failed")
    with pytest.raises(OSError, match="writer failed"):
        adapter.finalize()
    assert manager.events == []
    with pytest.raises(RuntimeError, match="12 must be finalized"):
        adapter.save(manager.save_checkpoint, local_path=str(tmp_path / "other"), global_step=13)


def test_cpu_shared_storage_is_snapshotted_before_training_mutates_it():
    source = torch.tensor([1.0, 2.0]).share_memory_()
    bucket = ("path", "key", (["serialized-object"], [("item", source)]))
    request_type = namedtuple("Request", "preload_fn async_fn_args")
    request = request_type(lambda: [bucket], (0, [bucket], None))

    staged = _isolate_cpu_snapshots(request).preload_fn()
    source.add_(10)

    staged_tensor = staged[0][2][1][0][1]
    assert torch.equal(staged_tensor, torch.tensor([1.0, 2.0]))
    assert staged_tensor.data_ptr() != source.data_ptr()
    assert staged[0][2][0] == ["serialized-object"]


def test_bundled_save_stages_every_tree_before_returning_and_preserves_result_order(tmp_path):
    manager = _Manager()
    queue = _Queue()
    sources = {
        name: torch.tensor([float(index)]).share_memory_() for index, name in enumerate(("model", "optimizer", "extra"))
    }
    written = {}
    results = []

    def native_save(_manager, path, state_dict):
        name = next(iter(state_dict))
        bucket = (path, name, ([], [(name, sources[name])]))

        def write(_rank, buckets, _queue):
            written[name] = buckets[0][2][1][0][1].item()
            manager.events.append(name)
            results.append(name)

        def finish():
            assert results.pop(0) == name

        return _Request(write, (0, [bucket], None), {}, [finish], lambda: [bucket])

    adapter = NativeAsyncCheckpoint(manager, queue=queue, save_dist_checkpoint=native_save)
    metrics = adapter.save(manager.save_checkpoint, local_path=str(tmp_path / "actor"), global_step=12)
    for source in sources.values():
        source.add_(10)

    assert metrics["checkpoint/worker_snapshot_bytes"] == 12.0
    assert metrics["checkpoint/worker_cpu_snapshot_copy_bytes"] == 12.0
    assert written == {}
    assert len(queue.requests) == 1
    adapter.finalize()
    assert written == {"model": 0.0, "optimizer": 1.0, "extra": 2.0}
    assert results == []


def test_gpu_native_host_snapshot_is_not_copied_twice():
    source = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    staged_tensor = torch.tensor([1.0, 2.0])
    source_bucket = ("path", "key", ([], [("item", source)]))
    staged_bucket = ("path", "key", ([], [("item", staged_tensor)]))
    request_type = namedtuple("Request", "preload_fn async_fn_args")
    request = request_type(lambda: [staged_bucket], (0, [source_bucket], None))

    staged = _isolate_cpu_snapshots(request).preload_fn()

    assert staged[0][2][1][0][1] is staged_tensor


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("use_dist_checkpointing", False, "Megatron dist checkpointing"),
        ("use_megatron_fsdp", True, "without Megatron-FSDP"),
        ("checkpoint_save_contents", ["model", "optimizer", "extra", "hf_model"], "exactly"),
        ("checkpoint_load_contents", ["model"], "exactly"),
        ("peft_cls", object(), "HF export or PEFT"),
        ("should_save_hf_model", True, "HF export or PEFT"),
        ("should_load_hf_model", True, "HF export or PEFT"),
        ("use_checkpoint_opt_param_scheduler", False, "use_checkpoint_opt_param_scheduler"),
    ],
)
def test_unsupported_native_save_paths_fail_before_install(field, value, message):
    manager = _Manager()
    setattr(manager, field, value)
    with pytest.raises(ValueError, match=message):
        _adapter(manager)
    assert "_dispatch_finalize" not in manager.__dict__


@pytest.mark.parametrize("kwargs", [{"hdfs_path": "hdfs://actor-only"}, {"max_ckpt_to_keep": 1}])
def test_storage_and_retention_stay_with_the_trainer(kwargs, tmp_path):
    manager, queue, adapter = _adapter()
    with pytest.raises(ValueError, match="trainer owns"):
        adapter.save(manager.save_checkpoint, local_path=str(tmp_path / "actor"), **kwargs)
    assert queue.requests == []
    assert adapter.finalize() == {}


def _raise_local_error(error):
    if error is not None:
        raise error


def test_thread_caller_stages_before_return_and_keeps_training_priority():
    events = []
    actor_thread = threading.get_ident()
    release = threading.Event()
    started = threading.Event()
    source = torch.tensor([1.0, 2.0])
    priority = os.getpriority(os.PRIO_PROCESS, 0)

    def preload():
        events.append(("preload", threading.get_ident()))
        return source.clone()

    def write(_rank, snapshot, _queue):
        started.set()
        assert release.wait(5)
        events.append(("write", threading.get_ident(), snapshot.tolist()))

    request = _Request(write, (None, None, None), {}, [], preload)
    with ThreadPoolExecutor(max_workers=1) as executor:
        caller = ThreadAsyncCaller(
            executor,
            synchronize=lambda: events.append(("synchronize", threading.get_ident())),
            sync_all_done=lambda alive: not alive,
            sync_failure=_raise_local_error,
        )
        try:
            caller.schedule_async_call(request)
            assert started.wait(5)
            assert caller.is_current_async_call_done(blocking=False) is False
            source.add_(10)
        finally:
            release.set()
        assert caller.is_current_async_call_done(blocking=True) is True
        caller.close()

    assert events[:2] == [("preload", actor_thread), ("synchronize", actor_thread)]
    assert events[2][0] == "write"
    assert events[2][1] != actor_thread
    assert events[2][2] == [1.0, 2.0]
    assert os.getpriority(os.PRIO_PROCESS, 0) == priority


def test_future_error_reaches_all_rank_guard_before_it_is_raised():
    events = []
    error = OSError("unexpected writer failure")

    def write():
        raise error

    def synchronize_failure(local_error):
        events.append(("failure_guard", local_error))
        raise RuntimeError("all-rank checkpoint failure")

    with ThreadPoolExecutor(max_workers=1) as executor:
        caller = ThreadAsyncCaller(
            executor,
            synchronize=lambda: None,
            sync_all_done=lambda alive: events.append(("readiness", alive)) or not alive,
            sync_failure=synchronize_failure,
        )
        caller.schedule_async_call(_Request(write, (), {}, [], None))
        with pytest.raises(RuntimeError, match="all-rank checkpoint failure"):
            caller.is_current_async_call_done(blocking=True)

    assert events[0][0] == "readiness"
    assert events[1] == ("failure_guard", error)


def test_successful_local_future_still_checks_for_other_rank_failure():
    def remote_failure(local_error):
        assert local_error is None
        raise RuntimeError("another rank failed")

    with ThreadPoolExecutor(max_workers=1) as executor:
        caller = ThreadAsyncCaller(
            executor,
            synchronize=lambda: None,
            sync_all_done=lambda alive: not alive,
            sync_failure=remote_failure,
        )
        caller.schedule_async_call(_Request(lambda: None, (), {}, [], None))
        with pytest.raises(RuntimeError, match="another rank failed"):
            caller.is_current_async_call_done(blocking=True)


@pytest.mark.parametrize("local_error", [None, OSError("local native writer failed")])
def test_default_failure_guard_raises_checkpoint_exception_on_every_rank(monkeypatch, local_error):
    from torch.distributed.checkpoint import CheckpointException

    from laminar.workers.checkpoint import _raise_if_writer_failed

    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda flag, **_kwargs: flag.fill_(1))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(messages, local_message):
        assert local_message == (repr(local_error) if local_error else None)
        messages[:] = [repr(local_error) if local_error else None, "OSError('remote native writer failed')"]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with pytest.raises(CheckpointException, match="remote native writer failed") as captured:
        _raise_if_writer_failed(local_error)
    assert 1 in captured.value.failures


@pytest.mark.parametrize("initially_enabled", [True, False])
@pytest.mark.parametrize("fail", [True, False])
def test_native_thread_writer_restores_process_gc_state(initially_enabled, fail):
    native = pytest.importorskip("megatron.core.dist_checkpointing.strategies.filesystem_async")
    previous_gc = gc.isenabled()
    observed = []

    class ResultsQueue:
        def put(self, _result):
            observed.append(gc.isenabled())
            if fail:
                raise OSError("results channel failed")

    if initially_enabled:
        gc.enable()
    else:
        gc.disable()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            caller = ThreadAsyncCaller(
                executor,
                synchronize=lambda: None,
                sync_all_done=lambda alive: not alive,
                sync_failure=_raise_local_error,
            )
            caller.schedule_async_call(
                _Request(
                    native.FileSystemWriterAsync.write_preloaded_data_multithread,
                    ([], False, 0, [], ResultsQueue()),
                    {},
                    [],
                    None,
                )
            )
            if fail:
                with pytest.raises(OSError, match="results channel failed"):
                    caller.is_current_async_call_done(blocking=True)
            else:
                caller.is_current_async_call_done(blocking=True)
        assert observed == [False]
        assert gc.isenabled() is initially_enabled
    finally:
        gc.enable() if previous_gc else gc.disable()


def test_native_thread_queue_keeps_collectives_on_actor_thread_and_drains_on_close(monkeypatch):
    native = pytest.importorskip("megatron.core.dist_checkpointing.strategies.async_utils")
    from laminar.workers.checkpoint import create_thread_checkpoint_queue

    actor_thread = threading.get_ident()
    events = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda *_args, **_kwargs: events.append(("all_reduce", threading.get_ident()))
    )
    queue = create_thread_checkpoint_queue(
        synchronize=lambda: events.append(("synchronize", threading.get_ident())),
        sync_all_done=lambda alive: not alive,
        sync_failure=_raise_local_error,
    )
    request = native.AsyncRequest(
        lambda: events.append(("write", threading.get_ident())),
        (),
        [lambda: events.append(("finalize", threading.get_ident()))],
    )
    queue.schedule_async_request(request)
    queue.close()

    assert [name for name, _ident in events] == ["synchronize", "write", "finalize", "all_reduce"]
    assert all(ident == actor_thread for name, ident in events if name != "write")
    assert next(ident for name, ident in events if name == "write") != actor_thread
    assert queue.get_num_unfinalized_calls() == 0
    with pytest.raises(RuntimeError, match="cannot schedule new futures after shutdown"):
        queue.schedule_async_request(request)


def test_native_runtime_uses_in_process_results_queue_and_caches_per_content_save_strategy(monkeypatch, tmp_path):
    import multiprocessing as mp
    import queue as thread_queue

    from megatron.core import dist_checkpointing, mpu
    from megatron.core.dist_checkpointing.strategies import filesystem_async

    from laminar.workers.checkpoint import _native_runtime

    calls = []
    fail_first = {"model": True}
    monkeypatch.setattr(filesystem_async, "_results_queue", None)
    monkeypatch.setattr(mpu, "get_data_parallel_group", lambda **_kwargs: "dp-cp-group")

    def fake_save(
        state_dict,
        path,
        *,
        sharded_strategy,
        validate_access_integrity,
        async_sharded_save=False,
        async_strategy="nvrx",
        **_kwargs,
    ):
        slot = Path(path).parent.name
        calls.append(
            (
                slot,
                sharded_strategy,
                validate_access_integrity,
                sharded_strategy.do_cache_distribution,
                sharded_strategy.base_strategy.use_cached_ckpt_structure,
                async_sharded_save,
                async_strategy,
            )
        )
        if fail_first.pop(slot, False):
            raise RuntimeError("integrity validation failed on first save")
        return SimpleNamespace(slot=slot)

    monkeypatch.setattr(dist_checkpointing, "save", fake_save)
    manager = SimpleNamespace(_build_sharded_state_dict_metadata=lambda: {"version": 1})

    native_queue, save_dist = _native_runtime()
    try:
        assert isinstance(filesystem_async._results_queue, thread_queue.Queue)
        assert mp.active_children() == []

        # A failed first save must not cache an unvalidated strategy.
        with pytest.raises(RuntimeError, match="integrity validation failed"):
            save_dist(manager, str(tmp_path / "step_1/actor/model/dist_ckpt"), {"model": 1})

        save_dist(manager, str(tmp_path / "step_1/actor/model/dist_ckpt"), {"model": 1})
        save_dist(manager, str(tmp_path / "step_1/actor/optimizer/dist_ckpt"), {"optimizer": 1, "lr_scheduler": 1})
        save_dist(manager, str(tmp_path / "step_2/actor/model/dist_ckpt"), {"model": 2})
        save_dist(manager, str(tmp_path / "step_2/actor/optimizer/dist_ckpt"), {"optimizer": 2, "lr_scheduler": 2})

        assert [row[0] for row in calls] == ["model", "model", "optimizer", "model", "optimizer"]
        assert [row[2] for row in calls] == [True, True, True, False, False]
        assert all(row[3] is True and row[4] is True and row[5] is True and row[6] == "mcore" for row in calls)
        # Step 2 reuses each slot's exact strategy instance without mixing model and optimizer caches.
        assert calls[1][1] is not calls[0][1]
        assert calls[3][1] is calls[1][1]
        assert calls[4][1] is calls[2][1]
        assert calls[1][1] is not calls[2][1]
    finally:
        native_queue.close(abort=True)
