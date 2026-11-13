# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import concurrent.futures
import hashlib
import json
import threading
from pathlib import Path

import pytest

from laminar.eval.server import (
    CheckpointDirectoryWatcher,
    EvaluationQueue,
    EvaluationQueueClient,
    create_server,
    run_queue_cli,
)


@pytest.fixture
def queue(tmp_path):
    return EvaluationQueue(tmp_path / "queue", {"dataset_sha256": "a" * 64, "n": 1}, [tmp_path])


def checkpoint(root, name):
    path = root / name
    path.mkdir()
    (path / "config.json").write_text("{}")
    return str(path)


def claim(queue, worker, claim_id, group=None):
    return queue.claim(worker, group or ["server0"], claim_id, queue.profile_sha256)["task"]


def result():
    return {"status": "PASS", "artifact": "hdfs://cluster/result.json", "sha256": hashlib.sha256(b"result").hexdigest()}


def test_two_residents_consume_one_queue_and_dynamic_tasks_survive_restart(queue, tmp_path):
    first = queue.submit(checkpoint(tmp_path, "a"), {"run": "baseline", "step": 10})
    second = queue.submit(checkpoint(tmp_path, "b"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(lambda worker: claim(queue, worker, "initial"), ["resident1/job1", "resident2/job2"]))
    assert {a["task_id"], b["task_id"]} == {first["task_id"], second["task_id"]}
    assert a["owner"]["worker"] != b["owner"]["worker"]
    assert queue.submit(first["checkpoint"], first["source"])["task_id"] == first["task_id"]
    assert queue.submit(first["checkpoint"])["task_id"] == first["task_id"]
    restored = EvaluationQueue(queue.path.parent, queue.read()["profile"], [tmp_path])
    assert claim(restored, a["owner"]["worker"], "initial") == a
    with pytest.raises(ValueError, match="unreconciled"):
        claim(restored, a["owner"]["worker"], "another")
    with pytest.raises(ValueError, match="unreconciled"):
        claim(restored, a["owner"]["worker"], "overlap", ["server0", "server1"])
    finished = restored.finish(a["task_id"], a["owner"], result())
    assert finished["status"] == "FINISHED"
    third = restored.submit(checkpoint(tmp_path, "c"))
    assert claim(restored, a["owner"]["worker"], "next")["task_id"] == third["task_id"]
    assert restored.read(b["task_id"]) == b
    assert restored.submit(a["checkpoint"], a["source"])["status"] == "FINISHED"
    assert restored.finish(a["task_id"], a["owner"], result()) == finished
    with pytest.raises(ValueError, match="different terminal"):
        restored.finish(a["task_id"], a["owner"], {**result(), "status": "FAIL"})


def test_profile_source_owner_and_persistence_fail_closed(queue, tmp_path, monkeypatch):
    path = checkpoint(tmp_path, "a")
    task = queue.submit(path, {"manifest_sha256": "c" * 64})
    with pytest.raises(ValueError, match="source identity"):
        queue.submit(path, {"manifest_sha256": "d" * 64})
    with pytest.raises(ValueError, match="profile"):
        queue.claim("worker", ["server0"], "claim", "wrong")
    with pytest.raises(ValueError, match="profile"):
        EvaluationQueue(queue.path.parent, {"n": 32}, [tmp_path])
    running = claim(queue, "worker", "claim")
    with pytest.raises(ValueError, match="owner"):
        queue.finish(task["task_id"], {**running["owner"], "worker": "other"}, result())
    with pytest.raises(ValueError, match="artifact"):
        queue.finish(task["task_id"], running["owner"], {"status": "PASS"})
    before = queue.read()
    real_replace = Path.replace

    def fail_write(path, target):
        if target == queue.path:
            raise OSError("persistent mount unavailable")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_write)
    with pytest.raises(OSError, match="mount unavailable"):
        queue.finish(task["task_id"], running["owner"], result())
    assert queue.read() == before
    assert json.loads(queue.path.read_text()) == before


def test_repack_release_persists_remaining_replicas_and_keeps_original_claim_and_finish_owner(queue, tmp_path):
    queue.submit(checkpoint(tmp_path, "tail"))
    next_task = queue.submit(checkpoint(tmp_path, "next"))
    running = claim(queue, "worker", "v1", ["s0", "s1", "s2"])
    with pytest.raises(ValueError, match="unreconciled"):
        claim(queue, "worker", "v2", ["s0", "s1"])
    released = queue.release_replicas(running["task_id"], running["owner"], ["s2"])
    assert released["remaining_replicas"] == ["s2"]
    assert released["owner"] == running["owner"]
    assert queue.release_replicas(running["task_id"], running["owner"], ["s2"]) == released

    restored = EvaluationQueue(queue.path.parent, queue.read()["profile"], [tmp_path])
    assert claim(restored, "worker", "v1", ["s2", "s1", "s0"]) == released
    with pytest.raises(ValueError, match="unreconciled"):
        claim(restored, "worker", "v2", ["s2"])
    assert claim(restored, "worker", "v2", ["s0", "s1"])["task_id"] == next_task["task_id"]
    finished = restored.finish(running["task_id"], running["owner"], result())
    assert finished["owner"] == running["owner"]
    assert finished["remaining_replicas"] == ["s2"]
    assert restored.finish(running["task_id"], running["owner"], result()) == finished


@pytest.mark.parametrize("remaining", [[], ["s0", "s0"], ["unknown"], [""], "s0", [1]])
def test_repack_release_rejects_invalid_subsets_without_changing_state(queue, tmp_path, remaining):
    queue.submit(checkpoint(tmp_path, "tail"))
    running = claim(queue, "worker", "v1", ["s0", "s1"])
    before = queue.read()
    with pytest.raises(ValueError, match="remaining_replicas"):
        queue.release_replicas(running["task_id"], running["owner"], remaining)
    assert queue.read() == before


def test_repack_release_requires_original_owner_and_cannot_expand_after_shrinking(queue, tmp_path):
    queue.submit(checkpoint(tmp_path, "tail"))
    running = claim(queue, "worker", "v1", ["s0", "s1"])
    before = queue.read()
    with pytest.raises(ValueError, match="original"):
        queue.release_replicas(running["task_id"], {**running["owner"], "claim_id": "other"}, ["s1"])
    assert queue.read() == before
    released = queue.release_replicas(running["task_id"], running["owner"], ["s1"])
    with pytest.raises(ValueError, match="only shrink"):
        queue.release_replicas(running["task_id"], running["owner"], ["s0", "s1"])
    assert queue.read(running["task_id"]) == released


def test_finished_task_can_release_late_drain_and_repair_clears_old_replica_subset(queue, tmp_path):
    queue.submit(checkpoint(tmp_path, "tail"))
    running = claim(queue, "worker", "v1", ["s0", "s1"])
    queue.finish(running["task_id"], running["owner"], result())
    released = queue.release_replicas(running["task_id"], running["owner"], ["s1"])
    assert released["status"] == "FINISHED" and released["owner"] == running["owner"]
    assert released["result"] == result()
    repaired = queue.requeue(
        running["task_id"],
        result()["sha256"],
        {"verdict": "UNKNOWN", "artifact": "hdfs://cluster/review", "sha256": "d" * 64},
        [],
        {},
    )
    assert "remaining_replicas" not in repaired
    assert repaired["previous_attempts"][0]["remaining_replicas"] == ["s1"]
    rerun = claim(queue, "worker", "v2", ["s2"])
    with pytest.raises(ValueError, match="unreconciled"):
        claim(queue, "worker", "v3", ["s2"])
    with pytest.raises(ValueError, match="original"):
        queue.release_replicas(running["task_id"], running["owner"], ["s1"])
    assert queue.read(rerun["task_id"]) == rerun


def test_repack_release_failed_persistence_does_not_publish_replica_availability(queue, tmp_path, monkeypatch):
    queue.submit(checkpoint(tmp_path, "tail"))
    running = claim(queue, "worker", "v1", ["s0", "s1"])
    before = queue.read()
    real_replace = Path.replace

    def fail_write(path, target):
        if target == queue.path:
            raise OSError("persistent mount unavailable")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_write)
    with pytest.raises(OSError, match="mount unavailable"):
        queue.release_replicas(running["task_id"], running["owner"], ["s1"])
    assert queue.read() == before
    assert json.loads(queue.path.read_text()) == before
    with pytest.raises(ValueError, match="unreconciled"):
        claim(queue, "worker", "v2", ["s0"])


def test_real_http_dynamic_submission_query_drain_and_duplicate(queue, tmp_path):
    server = create_server(queue, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    client = EvaluationQueueClient("http://127.0.0.1:" + str(server.server_port))
    try:
        task = client.request("/tasks", {"checkpoint": checkpoint(tmp_path, "a")})
        assert client.request("/tasks/" + task["task_id"]) == task
        request = {
            "worker": "resident1/job1",
            "group": ["server0"],
            "claim_id": "v1",
            "profile_sha256": queue.profile_sha256,
        }
        running = client.request("/claim", request)["task"]
        assert client.request("/claim", request)["task"] == running
        client.request("/drain", {})
        assert client.request("/tasks", {"checkpoint": task["checkpoint"]}) == running
        with pytest.raises(ValueError, match="draining"):
            client.request("/tasks", {"checkpoint": checkpoint(tmp_path, "b")})
        done = client.request("/finish", {"task_id": task["task_id"], "owner": running["owner"], "result": result()})
        assert done["status"] == "FINISHED"
        assert client.request("/claim", {**request, "claim_id": "v2"}) == {"task": None, "draining": True}
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_explicit_scoring_repair_preserves_previous_artifact_and_rejects_late_old_claim(queue, tmp_path):
    task = queue.submit(checkpoint(tmp_path, "a"))
    running = claim(queue, "resident1/job1", "v1")
    queue.finish(task["task_id"], running["owner"], result())
    review = {"verdict": "UNKNOWN", "artifact": "hdfs://cluster/audit.json", "sha256": "d" * 64}
    with pytest.raises(ValueError, match="terminal"):
        other = queue.submit(checkpoint(tmp_path, "b"))
        queue.requeue(other["task_id"], result()["sha256"], review, [], {})
    with pytest.raises(ValueError, match="different previous"):
        queue.requeue(task["task_id"], "b" * 64, review, [], {})
    repaired = queue.requeue(
        task["task_id"],
        result()["sha256"],
        review,
        [{"path": "/persistent/original499.json", "sha256": "e" * 64}],
        {"scorer": "original"},
    )
    assert repaired["status"] == "QUEUED"
    assert repaired["previous_attempts"][0]["result"] == result()
    assert repaired["previous_attempts"][0]["owner"] == running["owner"]
    with pytest.raises(ValueError, match="previous attempt"):
        claim(queue, "resident1/job1", "v1")
    assert claim(queue, "resident2/job2", "v1")["task_id"] == repaired["task_id"]


def test_queue_serves_ipv6_resident_endpoint(queue):
    server = create_server(queue, "::1", 0)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        client = EvaluationQueueClient("http://[::1]:" + str(server.server_port))
        assert client.request("/tasks")["profile_sha256"] == queue.profile_sha256
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_global_drain_blocks_queued_claims_but_preserves_running_reconciliation(queue, tmp_path):
    queue.submit(checkpoint(tmp_path, "a"))
    queued = queue.submit(checkpoint(tmp_path, "b"))
    running = claim(queue, "main/job", "v1")
    queue.drain()
    assert queue.claim("main/job", ["server0"], "v1", queue.profile_sha256) == {"task": running, "draining": True}
    assert queue.claim("another/job", ["server0"], "v1", queue.profile_sha256) == {"task": None, "draining": True}
    queue.finish(running["task_id"], running["owner"], result())
    assert queue.claim("main/job", ["server0"], "v2", queue.profile_sha256)["task"] is None
    assert queue.read(queued["task_id"]) == queued


def test_exact_worker_drain_is_durable_and_leaves_other_worker_and_admission_live(queue, tmp_path):
    queue.submit(checkpoint(tmp_path, "a"))
    queue.submit(checkpoint(tmp_path, "b"))
    running = claim(queue, "supp/job", "v1")
    before = queue.read()
    assert "drained_workers" not in before  # Existing native schema remains readable.
    with pytest.raises(ValueError, match="existing task claim"):
        queue.drain("typo/job")
    assert queue.read() == before
    assert queue.drain("supp/job") == {"worker": "supp/job", "draining": True}
    assert queue.drain("supp/job") == {"worker": "supp/job", "draining": True}
    restored = EvaluationQueue(queue.path.parent, queue.read()["profile"], [tmp_path])
    assert restored.read()["accepting"] is True
    assert restored.claim("supp/job", ["server1"], "v2", restored.profile_sha256) == {"task": None, "draining": True}
    assert restored.claim("supp/job", ["server0"], "v1", restored.profile_sha256)["task"] == running
    restored.finish(running["task_id"], running["owner"], result())
    assert claim(restored, "main/job", "v1") is not None
    assert restored.submit(checkpoint(tmp_path, "dynamic"))["status"] == "QUEUED"


def test_worker_drain_http_and_failed_persistence_do_not_publish_in_memory(queue, tmp_path, monkeypatch):
    queue.submit(checkpoint(tmp_path, "a"))
    claim(queue, "supp/job", "v1")
    before = queue.read()
    real_replace = Path.replace

    def fail_write(path, target):
        if target == queue.path:
            raise OSError("mount unavailable")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_write)
    with pytest.raises(OSError, match="mount unavailable"):
        queue.drain("supp/job")
    assert queue.read() == before
    assert json.loads(queue.path.read_text()) == before
    monkeypatch.setattr(Path, "replace", real_replace)
    server = create_server(queue, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        client = EvaluationQueueClient("http://127.0.0.1:" + str(server.server_port))
        assert client.request("/drain", {"worker": "supp/job"}) == {"worker": "supp/job", "draining": True}
        assert client.request("/tasks")["accepting"] is True
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_checkpoint_directory_watcher_orders_steps_and_waits_for_committed_tracker(queue, tmp_path):
    watch_dir = tmp_path / "checkpoints"
    for step in (80, 40, 120):
        hf_dir = watch_dir / f"global_step_{step}" / "actor" / "model" / "huggingface"
        hf_dir.mkdir(parents=True)
        (hf_dir / "config.json").write_text("{}")
        (hf_dir / "model.safetensors").write_bytes(b"weights")
        (watch_dir / f"global_step_{step}" / "data.pt").write_text("cursor")
    # Tracker commits through step 80; step 120 is still uncommitted.
    (watch_dir / "latest_checkpointed_iteration.txt").write_text("80\n")

    watcher = CheckpointDirectoryWatcher(watch_dir, queue, interval_s=0.05)
    initial = watcher.scan_once()
    assert [task["checkpoint"] for task in initial] == [
        str((watch_dir / "global_step_40" / "actor" / "model" / "huggingface").resolve()),
        str((watch_dir / "global_step_80" / "actor" / "model" / "huggingface").resolve()),
    ]
    assert watcher.scan_once() == []

    # Advance the tracker to 120; the next scan submits step 120 once.
    (watch_dir / "latest_checkpointed_iteration.txt").write_text("120\n")
    later = watcher.scan_once()
    assert [task["checkpoint"] for task in later] == [
        str((watch_dir / "global_step_120" / "actor" / "model" / "huggingface").resolve()),
    ]


def test_first_checkpoint_waits_for_tracker_commit_and_is_submitted_once(queue, tmp_path):
    watch_dir = tmp_path / "checkpoints"
    step_dir = watch_dir / "global_step_40"
    hf_dir = step_dir / "actor" / "model" / "huggingface"
    hf_dir.mkdir(parents=True)
    (hf_dir / "config.json").write_text("{}")
    watcher = CheckpointDirectoryWatcher(watch_dir, queue)

    # HF metadata is created before the first training checkpoint commits.
    assert watcher.scan_once() == []
    assert queue.read()["tasks"] == {}
    (hf_dir / "model.safetensors").write_bytes(b"weights")
    (step_dir / "data.pt").write_text("cursor")
    assert watcher.scan_once() == []

    temporary = watch_dir / "latest_checkpointed_iteration.txt.tmp"
    temporary.write_text("40\n")
    assert watcher.scan_once() == []
    temporary.replace(watch_dir / "latest_checkpointed_iteration.txt")
    submitted = watcher.scan_once()
    assert [task["checkpoint"] for task in submitted] == [str(hf_dir.resolve())]
    assert watcher.scan_once() == []
    assert CheckpointDirectoryWatcher(watch_dir, queue).scan_once() == []
    assert len(queue.read()["tasks"]) == 1


@pytest.mark.parametrize(
    ("index_name", "shard_suffix"),
    [("model.safetensors.index.json", "safetensors"), ("pytorch_model.bin.index.json", "bin")],
)
def test_checkpoint_watcher_waits_for_all_nonempty_indexed_shards(queue, tmp_path, index_name, shard_suffix):
    watch_dir = tmp_path / "checkpoints"
    hf_dir = watch_dir / "global_step_40" / "actor" / "model" / "huggingface"
    hf_dir.mkdir(parents=True)
    (hf_dir / "config.json").write_text("{}")
    (watch_dir / "latest_checkpointed_iteration.txt").write_text("40\n")
    watcher = CheckpointDirectoryWatcher(watch_dir, queue)
    assert watcher.scan_once() == []

    first = f"model-00001-of-00002.{shard_suffix}"
    second = f"model-00002-of-00002.{shard_suffix}"
    (hf_dir / index_name).write_text(json.dumps({"weight_map": {"a": first, "b": second}}))
    (hf_dir / first).write_bytes(b"first")
    assert watcher.scan_once() == []
    (hf_dir / second).touch()
    assert watcher.scan_once() == []
    assert queue.read()["tasks"] == {}

    (hf_dir / second).write_bytes(b"second")
    assert len(watcher.scan_once()) == 1
    assert watcher.scan_once() == []


@pytest.mark.parametrize("index", ["{", "[]", '{"weight_map": {}}', '{"weight_map": {"a": "weights", "b": null}}'])
def test_checkpoint_watcher_rejects_incomplete_weight_indexes(queue, tmp_path, index):
    watch_dir = tmp_path / "checkpoints"
    hf_dir = watch_dir / "global_step_40" / "actor" / "model" / "huggingface"
    hf_dir.mkdir(parents=True)
    (hf_dir / "config.json").write_text("{}")
    (hf_dir / "weights").write_bytes(b"weights")
    (hf_dir / "model.safetensors.index.json").write_text(index)
    (watch_dir / "latest_checkpointed_iteration.txt").write_text("40\n")

    assert CheckpointDirectoryWatcher(watch_dir, queue).scan_once() == []
    assert queue.read()["tasks"] == {}


@pytest.mark.parametrize("weight_name", ["model.safetensors", "pytorch_model.bin"])
def test_named_hf_checkpoint_watcher_needs_weights_but_no_training_tracker(queue, tmp_path, weight_name):
    watch_dir = tmp_path / "models"
    watch_dir.mkdir()
    hf_dir = Path(checkpoint(watch_dir, "model-release"))
    watcher = CheckpointDirectoryWatcher(watch_dir, queue)
    assert watcher.scan_once() == []
    (hf_dir / weight_name).touch()
    assert watcher.scan_once() == []
    (hf_dir / weight_name).write_bytes(b"weights")

    assert [task["checkpoint"] for task in watcher.scan_once()] == [str(hf_dir.resolve())]
    assert watcher.scan_once() == []


def test_queue_cli_submit_status_and_drain_discover_endpoint_from_result_dir(queue, tmp_path):
    result_dir = tmp_path / "eval-out"
    queue_dir = result_dir / "queue"
    queue_dir.mkdir(parents=True)
    server = create_server(queue, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        (queue_dir / "endpoint.txt").write_text(endpoint + "\n")
        step_dir = tmp_path / "global_step_400"
        hf_dir = step_dir / "actor" / "model" / "huggingface"
        hf_dir.mkdir(parents=True)
        (hf_dir / "config.json").write_text("{}")
        second = checkpoint(tmp_path, "second")

        submitted = run_queue_cli(["--result-dir", str(result_dir), "--submit", str(step_dir), second])
        assert len(submitted) == 2
        assert submitted[0]["checkpoint"] == str(hf_dir.resolve())
        assert submitted[1]["checkpoint"] == str(Path(second).resolve())

        status = run_queue_cli(["--result-dir", str(result_dir), "--status"])
        assert len(status["tasks"]) == 2
        single = run_queue_cli(["--result-dir", str(result_dir), "--status", submitted[0]["task_id"]])
        assert single["task_id"] == submitted[0]["task_id"]

        # Verify Python Client via laminar.eval.EvaluationQueueClient (1-to-1 with CLI)
        from laminar.eval import EvaluationQueueClient as PublicClient

        py_client = PublicClient(result_dir=result_dir)
        third = checkpoint(tmp_path, "third")
        fourth = checkpoint(tmp_path, "fourth")
        py_tasks = py_client.submit(third, fourth, source={"step": 480})
        assert len(py_tasks) == 2
        assert py_tasks[0]["checkpoint"] == str(Path(third).resolve())
        assert py_client.status(py_tasks[0]["task_id"])["source"] == {"step": 480}
        assert len(py_client.status()["tasks"]) == 4

        drained = py_client.drain()
        assert drained == {"accepting": False}
        assert queue.read()["accepting"] is False
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
