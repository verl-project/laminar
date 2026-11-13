# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from laminar.eval import main as validation
from laminar.eval.server import EvaluationQueue, create_server
from laminar.eval.worker import (
    CheckpointRunWatcher,
    EvaluationTaskSource,
    evaluation_profile,
    model_profile,
)
from test_validation_on_cpu import _config, _fake_runtime, _parallel_runtime, _Queue


@pytest.fixture
def service(tmp_path):
    config = _config(tmp_path)
    config.actor_rollout_ref.model = {"path": config.validation.checkpoints[0]}
    task = tmp_path / "task.yaml"
    task.write_text("formal scoring protocol")
    config.actor_rollout_ref.rollout.update(
        nnodes=1,
        n_gpus_per_node=4,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        prompt_length=1024,
        response_length=2048,
        custom={
            "agent_framework": {
                "agent_runners": {
                    "task": {
                        "max_concurrent_sessions": 4,
                        "runner_kwargs": {"task_config_path": str(task)},
                    }
                }
            }
        },
    )
    queue = EvaluationQueue(tmp_path / "queue", evaluation_profile(config), [tmp_path])
    server = create_server(queue, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    config.validation.queue = {
        "endpoint": "http://127.0.0.1:" + str(server.server_port),
        "worker_id": "resident1/submission1",
        "result_uri": None,
    }
    try:
        yield config, queue
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_queue_worker_keeps_fast_group_moving_and_accepts_dynamic_checkpoint(service, tmp_path, monkeypatch):
    import asyncio

    config, queue = service
    seed = config.validation.checkpoints[0]
    paths = [seed]
    for name in ("second", "dynamic"):
        path = tmp_path / name
        path.mkdir()
        (path / "config.json").write_text("{}")
        paths.append(str(path))
    queue.submit(paths[0])
    queue.submit(paths[1])
    source = EvaluationTaskSource(config)
    source.initial = [source.claim(["server_0", "server_2"]), source.claim(["server_1", "server_3"])]
    validation._validate_config(config)
    tq, events = _Queue(), []
    runtime, _ = _fake_runtime(monkeypatch, tq, events, concurrency=2)
    monkeypatch.setattr(validation, "tq", tq)
    slow_started, dynamic_started, slow_finished = threading.Event(), threading.Event(), threading.Event()
    metrics_started = threading.Event()

    async def slow_metrics():
        metrics_started.set()
        while not dynamic_started.is_set():
            await asyncio.sleep(0.001)
        return {"sessions/completed": 0}

    runtime.agent_loop_manager.get_efficiency_metrics = slow_metrics

    def generate(prompts):
        from verl.utils import tensordict_utils as tu

        version = tu.get_non_tensor_data(prompts, "global_steps", None)
        if version == 1:
            slow_started.set()
            assert dynamic_started.wait(5), "queue worker waited for the slow checkpoint"
        elif version == 2:
            assert slow_started.wait(5)
            queue.submit(paths[2])  # This checkpoint did not exist in the initial task list.
        else:
            assert slow_started.is_set() and not slow_finished.is_set()
            assert metrics_started.wait(2)
            dynamic_started.set()
            queue.drain()
        for index in range(len(prompts)):
            tq.add(tu.get_non_tensor_data(prompts[index], "uid", None), version=version, success=2)
        if version == 1:
            slow_finished.set()

    runtime.agent_loop_manager.generate_sequences_and_wait = generate
    validation._run_checkpoints(
        config, runtime, [{"prompt": "task", "extra_info": {"tools_kwargs": {}}}], task_source=source
    )
    assert slow_finished.is_set() and dynamic_started.is_set()
    state = queue.read()
    assert len(state["tasks"]) == 3
    for task in state["tasks"].values():
        assert task["status"] == "FINISHED"
        result = task["result"]
        payload = Path(result["artifact"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == result["sha256"]
        data = json.loads(payload)
        assert data["queue_task_id"] == task["task_id"]
        assert data["checkpoint"] == task["checkpoint"]
        assert data["queue_owner"] == task["owner"]
        assert set(data["replica_versions"].values()) == {data["version"]}
    assert not tq.tags


def test_first_claims_seed_actual_groups_and_empty_groups_remain_available(service):
    config, queue = service
    queue.submit(config.validation.checkpoints[0])
    source = EvaluationTaskSource(config)
    prepared = source.prepare(config)
    assert source.initial == [(1, config.validation.checkpoints[0])]
    assert source.idle_groups == []
    assert list(prepared.validation.checkpoints) == [config.validation.checkpoints[0]]
    assert source.next_version == 2
    assert source.resume_config(config, 1).validation.resume_results == []


def test_worker_repack_release_uses_real_service_and_preserves_original_task_owner(service, tmp_path):
    config, queue = service
    first = queue.submit(config.validation.checkpoints[0])
    other = tmp_path / "next"
    other.mkdir()
    (other / "config.json").write_text("{}")
    second = queue.submit(str(other))
    source = EvaluationTaskSource(config)
    version, _ = source.claim(["s0", "s1"])
    original_owner = source.tasks[version]["owner"]
    with pytest.raises(ValueError, match="unreconciled"):
        source.claim(["s0"])
    released = source.release_replicas(version, ["s1"])
    assert released == source.tasks[version] == queue.read(first["task_id"])
    assert released["owner"] == original_owner
    assert source.claim(["s0"])[0] == 2
    assert source.tasks[2]["task_id"] == second["task_id"]
    artifact = tmp_path / "original-result.json"
    artifact.write_text("original complete result")
    source.finish(version, artifact, "PASS")
    finished = queue.read(first["task_id"])
    assert finished["owner"] == original_owner
    assert finished["remaining_replicas"] == ["s1"]
    assert finished["result"]["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()


def test_queue_repack_claims_released_source_before_old_tail_finishes(service, tmp_path, monkeypatch):
    config, queue = service
    config.validation.checkpoints = [config.validation.checkpoints[0]]
    config.validation.repack.schedule_interval_s = 0.01
    config.validation.batch_size = 2
    config.actor_rollout_ref.rollout.n_gpus_per_node = 2
    first = queue.submit(config.validation.checkpoints[0])
    other = tmp_path / "next"
    other.mkdir()
    (other / "config.json").write_text("{}")
    second = queue.submit(str(other))
    source = EvaluationTaskSource(config)
    source.initial = [source.claim(["server_0", "server_1"])]
    original_owner = source.tasks[1]["owner"]
    validation._validate_config(config)
    tq, events = _Queue(), []
    runtime, router, state = _parallel_runtime(monkeypatch, tq, events)
    monkeypatch.setattr(validation, "tq", tq)
    samples = [{"prompt": f"task {index}", "extra_info": {"tools_kwargs": {}}} for index in range(2)]
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(validation._run_checkpoints, config, runtime, samples, task_source=source)
        try:
            assert state.abort_repeated.wait(3)
            assert queue.read(first["task_id"])["owner"] == original_owner
            assert "remaining_replicas" not in queue.read(first["task_id"])
            assert queue.read(second["task_id"])["status"] == "QUEUED"
            state.acknowledge_abort.set()
            assert state.next_started.wait(3), "queue did not admit the released replica"
            assert not state.old_finished.is_set()
            queue.drain()
            state.old_complete.set()
            future.result(timeout=5)
        finally:
            state.acknowledge_abort.set()
            state.old_complete.set()
    previous = queue.read(first["task_id"])
    following = queue.read(second["task_id"])
    assert previous["owner"] == original_owner
    assert set(previous["remaining_replicas"]) == set(original_owner["replica_group"]) - set(router.sources)
    assert following["owner"]["replica_group"] == router.sources
    for task in (previous, following):
        assert task["status"] == "FINISHED"
        artifact = task["result"]
        assert hashlib.sha256(Path(artifact["artifact"]).read_bytes()).hexdigest() == artifact["sha256"]


def test_workers_reject_actual_protocol_drift_before_claiming(service):
    config, queue = service
    queue.submit(config.validation.checkpoints[0])
    config.actor_rollout_ref.rollout.val_kwargs.n += 1
    with pytest.raises(ValueError, match="sampling"):
        EvaluationTaskSource(config)
    assert all(task["status"] == "QUEUED" for task in queue.read()["tasks"].values())


def test_model_compatibility_preserves_tokenizer_rope_and_context_budget(tmp_path):
    paths = [tmp_path / "current", tmp_path / "historical"]
    for path in paths:
        path.mkdir()
        (path / "tokenizer.json").write_text('{"vocab": {"a": 0}}')
    current = {"max_position_embeddings": 262144, "attn_implementation": "flash_attention_2", "rope_theta": 1000000}
    historical = {"max_position_embeddings": 139264, "rope_theta": 1000000}
    (paths[0] / "config.json").write_text(json.dumps(current))
    (paths[1] / "config.json").write_text(json.dumps(historical))
    (paths[0] / "tokenizer_config.json").write_text('{"local_files_only": false, "eos_token": "end"}')
    (paths[1] / "tokenizer_config.json").write_text('{"eos_token": "end"}')
    assert model_profile(str(paths[0]), 135168) == model_profile(str(paths[1]), 135168)
    with pytest.raises(ValueError, match="context capacity"):
        model_profile(str(paths[1]), 150000)
    historical["rope_theta"] = 500000
    (paths[1] / "config.json").write_text(json.dumps(historical))
    assert model_profile(str(paths[0]), 135168) != model_profile(str(paths[1]), 135168)
    historical["rope_theta"] = current["rope_theta"]
    (paths[1] / "config.json").write_text(json.dumps(historical))
    (paths[1] / "tokenizer_config.json").write_text('{"eos_token": "different"}')
    assert model_profile(str(paths[0]), 135168) != model_profile(str(paths[1]), 135168)


def test_drained_empty_queue_does_not_initialize_inference(service):
    config, queue = service
    queue.drain()
    assert EvaluationTaskSource(config).prepare(config) is None


def test_profile_uses_task_bytes_and_ignores_resident_output_and_capacity(service, tmp_path):
    config, _ = service
    original = evaluation_profile(config)
    framework = config.actor_rollout_ref.rollout.custom.agent_framework
    task = Path(framework.agent_runners.task.runner_kwargs.task_config_path)
    other = tmp_path / "other-resident-task.yaml"
    other.write_bytes(task.read_bytes())
    framework.log_dir = "/different/persistent/submission/logs"
    framework.gateway_count = 16
    framework.agent_runners.task.max_concurrent_sessions = 224
    assert evaluation_profile(config) == original
    framework.agent_runners.task.runner_kwargs = {"task_config_path": str(task)}
    before = evaluation_profile(config)
    framework.agent_runners.task.runner_kwargs.task_config_path = str(other)
    assert evaluation_profile(config) == before
    other.write_text("changed scoring protocol")
    assert evaluation_profile(config) != before


def test_dynamic_resume_preserves_original_uid_and_reads_only_missing_samples(service, tmp_path):
    from test_validation_on_cpu import _resume_fixture

    config, queue = service
    samples, _, _, original = _resume_fixture(config, tmp_path, indices=(0, 2))
    task = queue.submit(
        config.validation.checkpoints[0],
        resume_results=OmegaConf.to_container(config.validation.resume_results),
        resume_protocol=OmegaConf.to_container(config.validation.resume_protocol),
    )
    source = EvaluationTaskSource(config)
    version, checkpoint = source.claim(["rollout_server_0"])
    restored_config = source.resume_config(config, version)
    restored = validation._read_resume_results(restored_config, samples)[checkpoint]
    assert [row["uid"] for row in restored] == [row["uid"] for row in original]
    assert [row["dataset_index"] for row in restored] == [0, 2]
    assert source.tasks[version]["task_id"] == task["task_id"]
    manifest = Path(restored_config.validation.resume_results[0].path)
    manifest.write_text("corrupted inherited evidence")
    with pytest.raises(ValueError, match="digest mismatch"):
        validation._read_resume_results(restored_config, samples)


def test_standard_eval_entry_can_serve_queue_without_starting_ray_or_inference(service, tmp_path, monkeypatch):
    config, _ = service
    config.validation.queue.update(
        endpoint=None,
        server_port=9000,
        server_host="127.0.0.1",
        coordinator_only=True,
        state_dir=str(tmp_path / "standalone-coordinator"),
    )
    served = []

    class Server:
        server_port = 9000

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def serve_forever(self):
            served.append(True)

    def create(queue, host, port):
        assert queue.read()["profile"] == evaluation_profile(config)
        assert host == "127.0.0.1" and port == 9000
        return Server()

    monkeypatch.setattr(validation, "create_server", create)
    monkeypatch.setattr(validation.ray, "init", lambda **kwargs: pytest.fail("coordinator started Ray"))
    monkeypatch.setattr(
        validation, "initialize_evaluation_runtime", lambda *args: pytest.fail("coordinator started GPU")
    )
    validation.main.__wrapped__(config)
    assert served == [True]


@pytest.mark.parametrize("worker_id", [None, "resident/submission"])
@pytest.mark.parametrize("server_host", ["127.0.0.1", "0.0.0.0"])
def test_standard_eval_entry_always_starts_embedded_service_and_enqueues_initial_checkpoints(
    service, tmp_path, monkeypatch, worker_id, server_host
):
    config, _ = service
    state_dir = tmp_path / "embedded-queue"
    config.validation.queue.update(
        endpoint=None,
        worker_id=worker_id,
        server_port=0,
        server_host=server_host,
        coordinator_only=False,
        state_dir=str(state_dir),
    )
    observed = {}

    driver_thread = threading.current_thread()

    def run_worker(worker_config):
        assert threading.current_thread() is driver_thread
        observed["endpoint"] = worker_config.validation.queue.endpoint
        observed["worker_id"] = worker_config.validation.queue.worker_id
        client = validation.EvaluationQueue(state_dir, evaluation_profile(worker_config), [tmp_path])
        observed["tasks"] = client.read()["tasks"]
        return "done"

    monkeypatch.setattr(validation.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(validation.ray, "get", lambda ref: ref)
    monkeypatch.setattr(
        validation.main_task, "remote", lambda _config: pytest.fail("embedded worker was scheduled on Ray")
    )
    monkeypatch.setattr(validation, "_run_worker", run_worker)
    validation.main.__wrapped__(config)

    assert observed["endpoint"].startswith("http://127.0.0.1:")
    assert (state_dir / "endpoint.txt").read_text().strip() == observed["endpoint"]
    if worker_id is None:
        assert observed["worker_id"].startswith("local-worker:")
    else:
        assert observed["worker_id"] == worker_id
    assert len(observed["tasks"]) == 1
    task = next(iter(observed["tasks"].values()))
    assert task["checkpoint"] == str(Path(config.validation.checkpoints[0]).resolve())
    assert task["status"] == "QUEUED"


def test_eval_service_supports_empty_initial_checkpoints_with_model_path_and_companion_watch_dir(
    service, tmp_path, monkeypatch
):
    config, _ = service
    base_model = Path(config.validation.checkpoints[0])
    watch_dir = tmp_path / "train-checkpoints"
    step_40 = watch_dir / "global_step_40" / "actor" / "model" / "huggingface"
    step_40.mkdir(parents=True)
    (step_40 / "config.json").write_text((base_model / "config.json").read_text())
    (step_40 / "model.safetensors").write_bytes(b"completed weights")
    (watch_dir / "global_step_40" / "data.pt").write_text("cursor")
    (watch_dir / "latest_checkpointed_iteration.txt").write_text("40\n")

    state_dir = tmp_path / "companion-queue"
    config.validation.checkpoints = []
    config.actor_rollout_ref.model.path = str(base_model)
    config.validation.queue.update(
        endpoint=None,
        server_port=0,
        server_host="127.0.0.1",
        coordinator_only=False,
        state_dir=str(state_dir),
        watch_dir=str(watch_dir),
        watch_interval_s=0.05,
    )
    observed = {}

    def fake_remote(worker_config):
        source = EvaluationTaskSource(worker_config)
        prepared = source.prepare(worker_config)
        observed["initial"] = source.initial
        observed["prepared_checkpoints"] = list(prepared.validation.checkpoints)
        return "done"

    monkeypatch.setattr(validation.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(validation.ray, "get", lambda ref: ref)
    monkeypatch.setattr(validation, "_run_worker", fake_remote)
    validation.main.__wrapped__(config)

    assert observed["initial"] == [(1, str(step_40.resolve()))]
    assert observed["prepared_checkpoints"][0] == str(step_40.resolve())


def test_embedded_eval_reopens_persistent_queue_with_a_new_worker_identity(service, tmp_path, monkeypatch):
    config, _ = service
    state_dir = tmp_path / "persistent-queue"
    config.validation.queue.update(
        endpoint=None,
        worker_id=None,
        server_port=0,
        server_host="127.0.0.1",
        coordinator_only=False,
        state_dir=str(state_dir),
    )
    observed = []

    def run_worker(worker_config):
        source = EvaluationTaskSource(worker_config)
        source.prepare(worker_config)
        version, checkpoint = source.initial[0]
        path = Path(worker_config.validation.result_dir) / "version_000001.json"
        path.write_text(json.dumps({"checkpoint": checkpoint, "status": "PASS"}))
        source.finish(version, path, "PASS")
        observed.append((source.worker, checkpoint))

    monkeypatch.setattr(validation.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(validation, "_run_worker", run_worker)
    validation.main.__wrapped__(config)

    second = tmp_path / "second-model"
    second.mkdir()
    (second / "config.json").write_text("{}")
    config.validation.checkpoints = [str(second)]
    config.validation.result_dir = str(tmp_path / "new-worker-results")
    validation.main.__wrapped__(config)

    assert len({worker for worker, _ in observed}) == 2
    assert [checkpoint for _, checkpoint in observed] == [config.actor_rollout_ref.model.path, str(second)]
    queue = EvaluationQueue(state_dir, evaluation_profile(config), [tmp_path])
    assert len(queue.read()["tasks"]) == 2
    assert all(task["status"] == "FINISHED" for task in queue.read()["tasks"].values())


def test_worker_drain_finishes_all_owned_groups_without_claiming_queued_task(service, tmp_path, monkeypatch):
    config, queue = service
    validation._validate_config(config)
    paths = [config.validation.checkpoints[0]]
    for name in ("second", "queued"):
        path = tmp_path / name
        path.mkdir()
        (path / "config.json").write_text("{}")
        paths.append(str(path))
    tasks = [queue.submit(path) for path in paths]
    source = EvaluationTaskSource(config)
    source.initial = [source.claim(["server_0", "server_2"]), source.claim(["server_1", "server_3"])]
    tq, events = _Queue(), []
    runtime, _ = _fake_runtime(monkeypatch, tq, events, concurrency=2)
    monkeypatch.setattr(validation, "tq", tq)
    first_started, drained = threading.Event(), threading.Event()

    def generate(prompts):
        from verl.utils import tensordict_utils as tu

        version = tu.get_non_tensor_data(prompts, "global_steps", None)
        assert version in (1, 2), "drained worker claimed the queued checkpoint"
        if version == 1:
            first_started.set()
            assert drained.wait(5)
        else:
            assert first_started.wait(5)
            queue.drain(source.worker)
            drained.set()
        for index in range(len(prompts)):
            tq.add(tu.get_non_tensor_data(prompts[index], "uid", None), version=version, success=2)

    runtime.agent_loop_manager.generate_sequences_and_wait = generate
    validation._run_checkpoints(
        config, runtime, [{"prompt": "task", "extra_info": {"tools_kwargs": {}}}], task_source=source
    )
    assert queue.read()["accepting"] is True
    assert source.draining is True
    assert [queue.read(task["task_id"])["status"] for task in tasks] == ["FINISHED", "FINISHED", "QUEUED"]
    assert not tq.tags
    for task in tasks[:2]:
        artifact = queue.read(task["task_id"])["result"]
        assert hashlib.sha256(Path(artifact["artifact"]).read_bytes()).hexdigest() == artifact["sha256"]


def test_queue_repack_waits_for_queued_task_and_recombines_finished_subsets(service, tmp_path, monkeypatch):
    config, queue = service
    config.validation.checkpoints = [config.validation.checkpoints[0]]
    config.validation.repack.schedule_interval_s = 0.01
    config.validation.batch_size = 2
    config.actor_rollout_ref.rollout.n_gpus_per_node = 2
    first = queue.submit(config.validation.checkpoints[0])
    source = EvaluationTaskSource(config)
    source.initial = [source.claim(["server_0", "server_1"])]
    assert source.has_queued() is False

    # Recombining finished subsets preserves the original admission group ordering.
    coalesced = validation._coalesce_available_groups(
        [
            (["server_1"], ("server_0", "server_1"), False),
            (["server_0"], ("server_0", "server_1"), False),
            (["server_2"], ("server_2", "server_3"), True),
        ]
    )
    assert coalesced == [
        (["server_0", "server_1"], ("server_0", "server_1"), False),
        (["server_2"], ("server_2", "server_3"), True),
    ]

    validation._validate_config(config)
    tq, events = _Queue(), []
    runtime, _router, state = _parallel_runtime(monkeypatch, tq, events, repack_enabled=False)
    monkeypatch.setattr(validation, "tq", tq)
    samples = [{"prompt": f"task {index}", "extra_info": {"tools_kwargs": {}}} for index in range(2)]
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(validation._run_checkpoints, config, runtime, samples, task_source=source)
        try:
            assert state.old_started.wait(3)
            # Without a queued task in EvaluationQueue, Repack must not split the active group.
            assert not state.abort_started.wait(0.05)
            assert not any(event[0] == "repack" for event in events)
            queue.drain()
            state.old_complete.set()
            future.result(timeout=5)
        finally:
            state.acknowledge_abort.set()
            state.old_complete.set()
    assert queue.read(first["task_id"])["status"] == "FINISHED"


def _make_exported_checkpoint(
    run_dir: Path, step: int, *, complete: bool = True, exported_config: dict | None = None
) -> Path:
    step_dir = run_dir / "Uni-Agent-SWE-Agent" / "checkpoints" / "exp" / f"global_step_{step}"
    hf_dir = step_dir / "actor" / "model" / "huggingface"
    hf_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "actor" / "ckpt_contents.json").write_text('{"actor": true}')
    default_cfg = {"Exported": True, "max_position_embeddings": 262144}
    (hf_dir / "config.json").write_text(json.dumps(exported_config or default_cfg))
    (hf_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.0.weight": "model-00001-of-00001.safetensors"}})
    )
    if complete:
        (hf_dir / "model-00001-of-00001.safetensors").write_bytes(b"safetensor-weights")
    return hf_dir


def test_checkpoint_run_watcher_discovers_multi_run_checkpoints_and_mirrors_results(service, tmp_path, monkeypatch):
    config, queue = service
    base_model = Path(config.validation.checkpoints[0])
    (base_model / "config.json").write_text('{"architectures": ["Qwen3MoeForCausalLM"]}')
    (base_model / "tokenizer.json").write_text('{"vocab": {}}')
    # Refresh queue profile to match updated base_model config/tokenizer.
    queue = EvaluationQueue(tmp_path / "multi-run-queue", evaluation_profile(config), [tmp_path])
    server = create_server(queue, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    config.validation.queue.endpoint = f"http://127.0.0.1:{server.server_port}"

    run_a = tmp_path / "runs" / "run_a"
    run_b = tmp_path / "runs" / "run_b"
    ckpt_a40 = _make_exported_checkpoint(run_a, 40)
    ckpt_a80 = _make_exported_checkpoint(run_a, 80, exported_config={"num_local_experts": 128})
    _make_exported_checkpoint(run_a, 120, complete=False)
    ckpt_b40 = _make_exported_checkpoint(run_b, 40)

    # Mark run_a global_step_40 as already evaluated in run_a/validation/completed_versions/.
    (run_a / "validation" / "completed_versions").mkdir(parents=True, exist_ok=True)
    (run_a / "validation" / "completed_versions" / "step_000040_version_000001.json").write_text(
        json.dumps({"status": "PASS", "version": 1, "checkpoint": str(ckpt_a40.resolve())})
    )

    watcher = CheckpointRunWatcher(
        queue=queue,
        watch_runs=[run_a, run_b],
        bootstrap_model_dir=base_model,
        persistent_roots=[tmp_path],
        interval_s=0.05,
        include_base=True,
    )
    try:
        submitted = watcher.poll_once()
        submitted_ckpts = [item["checkpoint"] for item in submitted]
        assert submitted_ckpts == [
            str(base_model.resolve()),
            str(ckpt_a80.resolve()),
            str(ckpt_b40.resolve()),
        ]
        # Original Megatron export metadata is backed up while bootstrap config/tokenizer are synced.
        backed_up = json.loads((ckpt_a80 / ".megatron_export_meta" / "config.json").read_text())
        assert backed_up == {"num_local_experts": 128}
        assert json.loads((ckpt_a80 / "config.json").read_text()) == {"architectures": ["Qwen3MoeForCausalLM"]}
        assert (ckpt_a80 / "tokenizer.json").read_text() == '{"vocab": {}}'

        config.validation.checkpoints = [config.validation.checkpoints[0]]
        source = EvaluationTaskSource(config)
        source.initial = [source.claim(["server_0", "server_1"])]
        validation._validate_config(config)
        tq, events = _Queue(), []
        runtime, _ = _fake_runtime(monkeypatch, tq, events, concurrency=1)
        monkeypatch.setattr(validation, "tq", tq)

        orig_generate = runtime.agent_loop_manager.generate_sequences_and_wait

        def generate(prompts):
            from verl.utils import tensordict_utils as tu

            if tu.get_non_tensor_data(prompts, "global_steps", None) == 3:
                queue.drain()
            orig_generate(prompts)

        runtime.agent_loop_manager.generate_sequences_and_wait = generate
        validation._run_checkpoints(
            config, runtime, [{"prompt": "task", "extra_info": {"tools_kwargs": {}}}], task_source=source
        )
        watcher.poll_once()

        run_a_results = [json.loads(p.read_text()) for p in sorted((run_a / "validation").glob("version_*.json"))]
        run_b_results = [json.loads(p.read_text()) for p in sorted((run_b / "validation").glob("version_*.json"))]
        assert [r["version"] for r in run_a_results] == [1, 2]
        assert {r["checkpoint"] for r in run_a_results} == {
            str(ckpt_a80.resolve()),
            str(base_model.resolve()),
        }
        assert [r["version"] for r in run_b_results] == [1, 2]
        assert {r["checkpoint"] for r in run_b_results} == {
            str(ckpt_b40.resolve()),
            str(base_model.resolve()),
        }
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_checkpoint_run_watcher_gates_tracker_marker_supports_fsdp_and_skips_prequeued_tasks(service, tmp_path):
    config, _ = service
    base_model = Path(config.validation.checkpoints[0])
    (base_model / "config.json").write_text('{"architectures": ["Qwen3MoeForCausalLM"]}')
    queue = EvaluationQueue(tmp_path / "run-watcher-gates", evaluation_profile(config), [tmp_path])
    # Pre-enqueue base model with default empty source={}; watcher must not conflict on startup.
    queue.submit(str(base_model))

    run_dir = tmp_path / "runs" / "fsdp_run"
    ckpt_root = run_dir / "checkpoints" / "exp"
    step_40_hf = ckpt_root / "global_step_40" / "actor" / "huggingface"
    step_80_hf = ckpt_root / "global_step_80" / "actor" / "huggingface"
    for hf_dir in (step_40_hf, step_80_hf):
        hf_dir.mkdir(parents=True)
        (hf_dir / "config.json").write_text('{"architectures": ["Qwen3MoeForCausalLM"]}')
        (hf_dir / "pytorch_model.bin").write_bytes(b"weights")

    watcher = CheckpointRunWatcher(
        queue=queue,
        watch_runs=[run_dir],
        bootstrap_model_dir=base_model,
        persistent_roots=[tmp_path],
        include_base=True,
    )
    # Without latest_checkpointed_iteration.txt, FSDP exports are not yet committed.
    assert watcher.poll_once() == []

    # While .tmp exists, commit is still in progress.
    tracker_tmp = ckpt_root / "latest_checkpointed_iteration.txt.tmp"
    tracker_tmp.write_text("40\n")
    assert watcher.poll_once() == []

    # Commit step 40; step 80 (> 40) must wait until the marker advances.
    tracker_tmp.replace(ckpt_root / "latest_checkpointed_iteration.txt")
    submitted = watcher.poll_once()
    assert [item["checkpoint"] for item in submitted] == [str(step_40_hf.resolve())]

    (ckpt_root / "latest_checkpointed_iteration.txt").write_text("80\n")
    submitted_next = watcher.poll_once()
    assert [item["checkpoint"] for item in submitted_next] == [str(step_80_hf.resolve())]


def test_embedded_eval_rejects_nonempty_result_dir_before_creating_queue_state(service, tmp_path):
    config, _ = service
    result_dir = Path(config.validation.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "version_000001.json").write_text('{"status": "PASS"}')
    state_dir = tmp_path / "untouched-state"
    config.validation.checkpoints = []
    config.actor_rollout_ref.model.path = str(config.actor_rollout_ref.model.path)
    config.validation.queue.update(endpoint=None, coordinator_only=False, state_dir=str(state_dir))

    with pytest.raises(FileExistsError, match="fresh result directory"):
        validation.main.__wrapped__(config)
    assert not state_dir.exists()


def test_resume_results_accept_symlinked_checkpoint_paths(tmp_path):
    from test_validation_on_cpu import _config, _resume_fixture

    config = _config(tmp_path)
    real_ckpt = Path(config.validation.checkpoints[0]).resolve()
    symlink_ckpt = tmp_path / "symlink-ckpt"
    symlink_ckpt.symlink_to(real_ckpt)
    samples, _, _, original = _resume_fixture(config, tmp_path, indices=(0, 2), checkpoint=str(real_ckpt))
    config.validation.checkpoints = [str(symlink_ckpt)]

    restored = validation._read_resume_results(config, samples)
    assert [row["uid"] for row in restored[str(real_ckpt)]] == [row["uid"] for row in original]
