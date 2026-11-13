# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from uni_agent.framework.framework import GatewayAgentFramework
from verl.utils import tensordict_utils as tu

from laminar.eval import main as validation
from laminar.eval.metrics import reward_evidence
from laminar.eval.router import ValidationRequestRouter
from laminar.global_request_router.router import DRAINING, UPDATING, RouteUnavailableError


def _config(tmp_path):
    actor = tmp_path / "global_step_64" / "actor"
    actor.mkdir(parents=True)
    (actor / "config.json").write_text("{}")
    data = tmp_path / "input.parquet"
    data.write_bytes(b"dataset identity")
    return OmegaConf.create(
        {
            "validation": {
                "checkpoints": [str(actor), str(actor)],
                "data_path": str(data),
                "result_dir": str(tmp_path / "results"),
                "persistent_roots": [str(tmp_path)],
                "batch_size": 1,
                "stage_timeout_s": 5,
                "repack": {"schedule_interval_s": 5.0},
            },
            "actor_rollout_ref": {
                "rollout": {
                    "name": "vllm",
                    "load_format": "auto",
                    "data_parallel_size": 1,
                    "enable_sleep_mode": False,
                    "free_cache_engine": False,
                    "val_kwargs": {"n": 2, "temperature": 1.0, "do_sample": True},
                    "seed": 42,
                    "full_determinism": False,
                },
            },
        }
    )


@pytest.mark.parametrize("do_sample, expected_temperature", [(False, 0), (True, 1.0)])
def test_validation_launch_honors_greedy_flag_in_framework_sampling(tmp_path, do_sample, expected_temperature):
    config = _config(tmp_path)
    config.actor_rollout_ref.rollout.val_kwargs.update(do_sample=do_sample, top_p=0.8, top_k=20)
    config.actor_rollout_ref.rollout.update(temperature=1.0, top_p=0.9, top_k=30, calculate_log_probs=True)
    framework = object.__new__(GatewayAgentFramework)
    framework._rollout_config = config.actor_rollout_ref.rollout
    observed = []

    def generate(prompts):
        assert tu.get(prompts, "validate") is True
        for index in range(len(prompts)):
            fields = framework._extract_sample_fields(prompts=prompts, sample_index=index)
            observed.append(framework._build_session_sampling_params(partition_id="val", sample_fields=fields))

    runtime = SimpleNamespace(
        agent_loop_manager=SimpleNamespace(generate_sequences_and_wait=generate), admission_runners={}
    )
    run = validation._CheckpointRun(result={"version": 1}, path=tmp_path / "result.json", started=0.0, server_ids=None)
    samples = [{"prompt": "test task", "extra_info": {"tools_kwargs": {}}}]
    with ThreadPoolExecutor(max_workers=1) as executor:
        validation._launch_batch(config, runtime, samples, run, executor)
        run.future.result()
    assert len(observed) == 1
    assert observed[0]["temperature"] == expected_temperature
    assert observed[0]["top_p"] == (0.8 if do_sample else 1.0)
    assert observed[0]["top_k"] == (20 if do_sample else -1)


def test_validation_config_composes_real_verl_schema():
    with initialize_config_module(config_module="laminar.eval.config", version_base=None):
        config = compose(config_name="validation", overrides=["validation.checkpoints=[/hf/model]"])
    OmegaConf.resolve(config)
    assert not {"trainer", "experience_buffer", "multi_version_rollout", "relay"}.intersection(config)
    assert "actor" not in config.actor_rollout_ref
    assert config.actor_rollout_ref.rollout.load_format == "auto"
    assert config.rollout_router.client_wait.timeout_s == 600
    assert config.rollout_router.client_wait.planned_pause_timeout_s == 3600
    assert config.rollout_router.dynamic_overload_admission.mode == "off"
    assert config.validation.repack.source_max_kv_cache_usage == 0.5
    assert config.validation.repack.destination_max_kv_cache_usage == 0.95
    assert config.validation.repack.schedule_interval_s == 5.0


def test_validation_config_requires_durable_path_and_never_overwrites_results(tmp_path):
    config = _config(tmp_path)
    config.validation.persistent_roots = []
    with pytest.raises(ValueError, match="persistent_roots"):
        validation._validate_config(config)
    config.validation.persistent_roots = [str(tmp_path)]
    result_dir = validation._validate_config(config)
    (result_dir / "version_000002.json").write_text("previous result")
    with pytest.raises(FileExistsError):
        validation._validate_config(config)


def test_stage_timeout_and_stopped_control_loop_never_pass(monkeypatch):
    monkeypatch.setattr(validation.ray, "wait", lambda *_args, **_kwargs: ([], []))
    with pytest.raises(TimeoutError, match="timed out"):
        validation._wait(object(), [object()], 0.01)
    stopped = object()
    monkeypatch.setattr(validation.ray, "wait", lambda *_args, **_kwargs: ([stopped], []))
    monkeypatch.setattr(validation.ray, "get", lambda _ref: None)
    with pytest.raises(RuntimeError, match="control loop stopped"):
        validation._wait(object(), [stopped], 0.01)


def test_idle_poll_uses_original_run_deadline_and_observes_control_failure(monkeypatch, tmp_path):
    run = validation._CheckpointRun(
        result={"version": 1},
        path=tmp_path / "result.json",
        started=0,
        server_ids=["a"],
        deadline=time.monotonic() + 0.02,
    )
    calls = []

    def unavailable(**kwargs):
        calls.append(kwargs)
        raise RouteUnavailableError("replica is busy")

    runtime = SimpleNamespace(
        rollout_router=SimpleNamespace(get_idle_server_versions=SimpleNamespace(remote=unavailable))
    )
    monkeypatch.setattr(validation, "_check_control_loops", lambda _controls: None)
    with pytest.raises(TimeoutError, match="waiting for idle"):
        validation._wait_for_idle(runtime, run, [], timeout=10)
    assert calls == [{"weight_version": 1, "server_ids": ["a"]}]

    def stopped(_controls):
        raise RuntimeError("rollout control loop stopped during validation")

    monkeypatch.setattr(validation, "_check_control_loops", stopped)
    with pytest.raises(RuntimeError, match="control loop stopped"):
        validation._wait_for_idle(runtime, run, [], timeout=10)
    assert len(calls) == 1


def _snapshot(*, running=0, version=1):
    return {
        "backend_health": "healthy",
        "kv_cache_usage": 0.0,
        "num_waiting_requests": 0,
        "num_running_requests": running,
        "snapshot_version": version,
    }


def test_validation_barrier_requires_all_replicas_exact_version_snapshot_and_route_drain():
    router = ValidationRequestRouter({"a": None, "b": None})
    router.mark_active("a", weight_version=1)
    with pytest.raises(RouteUnavailableError):
        router.get_idle_server_versions(1)
    router.mark_active("b", weight_version=1)
    with pytest.raises(RouteUnavailableError):
        router.get_idle_server_versions(1)  # no real snapshot yet
    router.bind_route("session", min_weight_version=1)
    router.update_routing_snapshot({"a": _snapshot(), "b": _snapshot(running=1)})
    with pytest.raises(RouteUnavailableError):
        router.get_idle_server_versions(1)
    router.release_route("session", weight_version=1)
    with pytest.raises(RouteUnavailableError):
        router.get_idle_server_versions(1)  # backend is still running
    router.update_routing_snapshot({"a": _snapshot(version=2), "b": _snapshot(version=2)})
    assert router.get_idle_server_versions(1) == {"a": 1, "b": 1}


def test_validation_barrier_empty_missing_replica_and_newer_version_do_not_pass():
    with pytest.raises(ValueError, match="non-empty"):
        ValidationRequestRouter({}).get_idle_server_versions(1)
    router = ValidationRequestRouter({"a": None, "b": None})
    router.mark_active("a", weight_version=1)
    router.mark_active("b", weight_version=1)
    router.update_routing_snapshot({"a": _snapshot()})
    with pytest.raises(RouteUnavailableError):
        router.get_idle_server_versions(1)
    router.mark_active("b", weight_version=2)
    with pytest.raises(RuntimeError, match="advanced beyond"):
        router.get_idle_server_versions(1)


class _Queue:
    def __init__(self):
        self.tags = {}
        self.rewards = {}
        self.extra_fields = {}

    def add(self, uid, *, version=1, success=2):
        self.tags[uid] = {"status": "finished" if success else "failure"}
        for session in range(success):
            for index in range(2):
                key = f"{uid}_{session}_{index}"
                self.tags[key] = {
                    "status": "success",
                    "global_steps": version,
                    "min_global_steps": version,
                    "max_global_steps": version,
                }
                self.rewards[key] = float(session + 1)
                self.extra_fields[key] = {}

    def kv_list(self):
        return {"val": dict(self.tags)}

    def kv_batch_get(self, keys, *, partition_id, select_fields):
        assert partition_id == "val" and select_fields == ["rm_scores", "extra_fields"]
        return tu.get_tensordict(
            tensor_dict={
                "rm_scores": torch.tensor([[0.0, self.rewards[key]] for key in keys]),
                "extra_fields": [self.extra_fields[key] for key in keys],
            }
        )

    def kv_clear(self, keys, *, partition_id):
        assert partition_id == "val"
        for key in keys:
            del self.tags[key]
            self.rewards.pop(key, None)
            self.extra_fields.pop(key, None)


def test_scores_count_session_once_with_partial_and_failed_prompts(monkeypatch):
    queue = _Queue()
    queue.add("uid-a")
    queue.add("uid-b", success=1)
    queue.add("uid-c", success=0)
    monkeypatch.setattr(validation, "tq", queue)
    records, keys = validation._read_session_results(["uid-a", "uid-b", "uid-c"], version=1, n=2)
    assert [record["sessions"] for record in records] == [
        [
            {
                "session_index": 0,
                "reward": 1.0,
                **reward_evidence({}),
                "trajectory_metadata": [{"trajectory_index": 0}, {"trajectory_index": 1}],
            },
            {
                "session_index": 1,
                "reward": 2.0,
                **reward_evidence({}),
                "trajectory_metadata": [{"trajectory_index": 0}, {"trajectory_index": 1}],
            },
        ],
        [
            {
                "session_index": 0,
                "reward": 1.0,
                **reward_evidence({}),
                "trajectory_metadata": [{"trajectory_index": 0}, {"trajectory_index": 1}],
            }
        ],
        [],
    ]
    assert [record["terminal"] for record in records] == [
        {"status": "finished"},
        {"status": "finished"},
        {"status": "failure"},
    ]
    assert len(keys) == 9


def test_session_trajectory_lengths_use_existing_tags_without_changing_scoring(monkeypatch):
    queue = _Queue()
    queue.add("uid-a")
    queue.add("uid-b", success=1)
    lengths = {
        "uid-a_0_0": {"prompt_len": 10, "response_len": 20, "seq_len": 30},
        "uid-a_0_1": {"prompt_len": 40, "response_len": 50, "seq_len": 90},
        "uid-a_1_0": {"prompt_len": 0},
        "uid-a_1_1": {"seq_len": 120},
        "uid-b_0_0": {"response_len": 70},
        "uid-b_0_1": {},
    }
    for key, fields in lengths.items():
        queue.tags[key].update(fields)
    queue.tags = dict(reversed(list(queue.tags.items())))
    queue.rewards["uid-a_0_0"] = 99.0  # Earlier trajectories must not contribute another score.
    evidence = {"reward_context": {"resolved": True, "eval_report": {"status_map": {"test": "PASSED"}}}}
    queue.extra_fields["uid-a_0_1"] = {"runner_reward_info": evidence}
    reads = []
    original_read = queue.kv_batch_get

    def read(keys, **kwargs):
        reads.append(list(keys))
        return original_read(keys, **kwargs)

    monkeypatch.setattr(queue, "kv_batch_get", read)
    monkeypatch.setattr(validation, "tq", queue)
    records, keys = validation._read_session_results(["uid-a", "uid-b"], version=1, n=2)
    assert reads == [["uid-a_0_1", "uid-a_1_1", "uid-b_0_1"]]
    assert len(keys) == 8
    assert [session["reward"] for row in records for session in row["sessions"]] == [1.0, 2.0, 1.0]
    assert records[0]["sessions"][0]["runner_reward_info"] is evidence
    for row in records:
        for session in row["sessions"]:
            assert session["trajectory_metadata"] == [
                {"trajectory_index": index, **lengths[f"{row['uid']}_{session['session_index']}_{index}"]}
                for index in range(2)
            ]


@pytest.mark.parametrize("agent_error", [None, "", " ", "RuntimeError: upload failed"])
def test_evaluation_preserves_explicit_agent_error_and_normal_zero_scores(monkeypatch, agent_error):
    queue = _Queue()
    queue.add("test-uid", success=1)
    queue.rewards["test-uid_0_1"] = 0.0
    evidence = {"reward_context": {"agent_error": agent_error}}
    queue.extra_fields["test-uid_0_1"] = {"runner_reward_info": evidence}
    monkeypatch.setattr(validation, "tq", queue)
    records, _ = validation._read_session_results(["test-uid"], version=1, n=1)
    assert records[0]["sessions"] == [
        {
            "session_index": 0,
            "reward": 0.0,
            **reward_evidence({}),
            "trajectory_metadata": [{"trajectory_index": 0}, {"trajectory_index": 1}],
            "runner_reward_info": evidence,
        }
    ]


@pytest.mark.parametrize("exit_code", [0, 1, -1, 127])
@pytest.mark.parametrize("status_map", [{}, {"test_fix": "FAILED", "test_existing": "PASSED"}])
def test_evaluation_preserves_scoring_evidence_without_reclassifying_zero_reward(monkeypatch, exit_code, status_map):
    queue = _Queue()
    queue.add("test-uid", success=1)
    queue.rewards["test-uid_0_1"] = 0.0
    evidence = {
        "reward": 0.0,
        "metrics": {"acc": 0.0},
        "reward_context": {
            "eval_exit_code": exit_code,
            "eval_execution_time": 600.0,
            "eval_completed": exit_code == 0,
            "eval_report": {"status_map": status_map, "resolved": False},
        },
    }
    queue.extra_fields["test-uid_0_1"] = {"runner_reward_info": evidence}
    monkeypatch.setattr(validation, "tq", queue)
    records, _ = validation._read_session_results(["test-uid"], version=1, n=1)
    assert records[0]["sessions"] == [
        {
            "session_index": 0,
            "reward": 0.0,
            **reward_evidence({}),
            "trajectory_metadata": [{"trajectory_index": 0}, {"trajectory_index": 1}],
            "runner_reward_info": evidence,
        }
    ]


@pytest.mark.parametrize(
    "fault", ["terminal", "terminal_status", "version", "session", "trajectory_status", "missing_scores", "reward"]
)
def test_results_reject_missing_or_inconsistent_evidence(monkeypatch, fault):
    queue = _Queue()
    queue.add("uid")
    if fault == "terminal":
        del queue.tags["uid"]
    elif fault == "version":
        queue.tags["uid_0_0"]["min_global_steps"] = 0
    elif fault == "terminal_status":
        queue.tags["uid"]["status"] = "failure"
    elif fault == "session":
        queue.tags["uid_2_0"] = queue.tags.pop("uid_0_0")
    elif fault == "trajectory_status":
        queue.tags["uid_0_0"]["status"] = "failure"
    elif fault == "missing_scores":
        queue.tags = {"uid": queue.tags["uid"]}
    else:
        queue.rewards["uid_0_1"] = float("nan")
    monkeypatch.setattr(validation, "tq", queue)
    with pytest.raises(RuntimeError):
        validation._read_session_results(["uid"], version=1, n=2)


class _Remote:
    def __init__(self, call):
        self.call = call

    def remote(self, *args, **kwargs):
        return SimpleNamespace(value=self.call(*args, **kwargs))


def _fake_runtime(monkeypatch, queue, events, *, concurrency=1, fault=None):
    controls = [object()]
    versions = {f"server_{i}": i % concurrency + 1 for i in range(concurrency * 2)}
    targets = dict(versions)
    fenced = set()
    active = set()
    lock = threading.Lock()
    rendezvous = threading.Barrier(concurrency, timeout=5)
    simultaneous = []

    def fake_wait(refs, *, num_returns, timeout):
        ready = [ref for ref in refs if not any(ref is control for control in controls)]
        return ready[:num_returns], []

    monkeypatch.setattr(validation.ray, "wait", fake_wait)
    monkeypatch.setattr(validation.ray, "get", lambda ref, **_kwargs: ref.value)

    def assign(*, weight_versions, server_ids=None):
        selected = list(versions) if server_ids is None else server_ids
        assert not {versions[server] for server in selected}.intersection(active)
        groups = {version: [] for version in weight_versions}
        for index, server in enumerate(selected):
            version = weight_versions[index % len(weight_versions)]
            groups[version].append(server)
            targets[server] = version
        events.append(("wave", tuple(weight_versions)))
        return groups

    def fence(*, loadable_version, repack_enabled):
        assert not repack_enabled
        selected = [server for server in versions if targets[server] == loadable_version]
        assert not {versions[server] for server in selected}.intersection(active)
        fenced.update(selected)
        events.append(("fence", loadable_version))
        return SimpleNamespace(groups=(), idle_sources=tuple(selected))

    def load(server, path, *, weight_version):
        assert server in fenced and versions[server] not in active
        assert targets[server] == weight_version > versions[server]
        events.append(("reload", weight_version, server, path))
        if fault == "reload":
            raise RuntimeError("reload failed")
        versions[server] = weight_version
        return weight_version

    def reopen(server, *, weight_version):
        assert versions[server] == weight_version
        fenced.remove(server)
        events.append(("active", weight_version, server))

    def idle(*, weight_version, server_ids):
        assert server_ids and all(targets[server] == versions[server] == weight_version for server in server_ids)
        assert not fenced.intersection(server_ids)
        events.append(("idle", weight_version))
        return {server: versions[server] for server in server_ids}

    def generate(prompts):
        version = tu.get_non_tensor_data(prompts, "global_steps", None)
        assert version in versions.values()
        assert not any(targets[server] == version for server in fenced)
        with lock:
            active.add(version)
            if len(active) == concurrency:
                simultaneous.append(frozenset(active))
        try:
            rendezvous.wait()
            if fault == "rollout" and version == concurrency:
                raise RuntimeError("rollout failed")
            for index in range(len(prompts)):
                uid = tu.get_non_tensor_data(prompts[index], "uid", None)
                queue.add(
                    uid,
                    version=version + 1 if fault == "version" and version == 1 else version,
                    success=1 if fault == "partial" else 2,
                )
                for key in list(queue.rewards):
                    if key.startswith(uid + "_"):
                        queue.rewards[key] += version * 100
                        queue.extra_fields[key]["runner_reward_info"] = {
                            "reward": queue.rewards[key],
                            "metrics": {},
                            "reward_context": {"source_version": version},
                        }
                        if fault == "agent_error" and version == 1:
                            queue.extra_fields[key]["runner_reward_info"]["reward_context"]["agent_error"] = (
                                "RuntimeError: upload failed"
                            )
        finally:
            with lock:
                active.remove(version)

    runtime = SimpleNamespace(
        llm_server_controllers={
            server: SimpleNamespace(
                load_hf_checkpoint=_Remote(lambda path, server=server, **kw: load(server, path, **kw))
            )
            for server in versions
        },
        initial_versions=dict(versions),
        initial_load_timings=dict.fromkeys(versions, 0.25),
        service_started_at="2026-09-09T00:00:00+00:00",
        initialization_s=0.5,
        rollout_router=SimpleNamespace(
            get_idle_server_versions=_Remote(idle),
            set_validation_versions=_Remote(assign),
            plan_and_begin_repack=_Remote(fence),
            plan_validation_repack=_Remote(lambda **_kw: SimpleNamespace(groups=(), idle_sources=())),
            mark_active=_Remote(reopen),
        ),
        llm_server_state_tracker=SimpleNamespace(
            start=_Remote(lambda: None),
            refresh_once=_Remote(lambda: None),
            wait_for_stop=SimpleNamespace(remote=lambda: controls[0]),
        ),
        agent_loop_manager=SimpleNamespace(generate_sequences_and_wait=generate),
        admission_runners={},
    )
    return runtime, simultaneous


@pytest.mark.parametrize("concurrency", [1, 2])
@pytest.mark.parametrize("fault", [None, "rollout", "version", "reload", "agent_error", "partial"])
def test_offline_checkpoint_waves_reuse_services_and_keep_test_results_separate(
    tmp_path, monkeypatch, concurrency, fault
):
    config = _config(tmp_path)
    config.validation.checkpoints = [config.validation.checkpoints[0]] * (2 * concurrency)
    validation._validate_config(config)
    queue = _Queue()
    events = []
    runtime, simultaneous = _fake_runtime(monkeypatch, queue, events, concurrency=concurrency, fault=fault)
    clear = queue.kv_clear

    def checked_clear(keys, *, partition_id):
        # Every key cleared belongs to a result already durably recorded.
        payloads = [json.loads(path.read_text()) for path in Path(config.validation.result_dir).glob("*.json")]
        persisted_uids = {prompt["uid"] for payload in payloads for prompt in payload["prompts"]}
        assert all(key in persisted_uids or key.rsplit("_", 2)[0] in persisted_uids for key in keys)
        clear(keys, partition_id=partition_id)

    queue.kv_clear = checked_clear
    monkeypatch.setattr(validation, "tq", queue)
    samples = [{"prompt": "test task", "extra_info": {"tools_kwargs": {}}}]
    if fault not in (None, "partial", "agent_error"):
        with pytest.raises(RuntimeError, match="rollout failed|version mismatch|reload failed"):
            validation._run_checkpoints(config, runtime, samples)
        if fault != "reload":
            failed_version = concurrency if fault == "rollout" else 1
            failed_result = json.loads(
                (Path(config.validation.result_dir) / f"version_{failed_version:06d}.json").read_text()
            )
            # Another group may already have advanced before this failure was observed.
            failed_group = set(failed_result["replica_group"])
            assert not any(event[0] == "reload" and event[2] in failed_group for event in events)
        failed = [json.loads(p.read_text()) for p in Path(config.validation.result_dir).glob("*.json")]
        assert any(result["status"] == "FAIL" for result in failed)
        return

    validation._run_checkpoints(config, runtime, samples)
    assert simultaneous == [
        frozenset(range(1, concurrency + 1)),
        frozenset(range(concurrency + 1, 2 * concurrency + 1)),
    ]
    windows = {}
    for version in range(1, 2 * concurrency + 1):
        result = json.loads((Path(config.validation.result_dir) / f"version_{version:06d}.json").read_text())
        expected_status = (
            "FAIL" if fault == "agent_error" and version == 1 else ("PARTIAL" if fault == "partial" else "PASS")
        )
        assert result["status"] == expected_status
        assert result["num_scored_sessions"] == (1 if fault == "partial" else 2)
        assert result["num_failed_sessions"] == (1 if fault == "partial" else 0)
        assert result["mean_session_reward"] == version * 100 + (1.0 if fault == "partial" else 1.5)
        assert result["quality_complete"] is (expected_status == "PASS")
        assert any(key.startswith("val-core/") for key in result["metrics"]) is (expected_status == "PASS")
        for session in result["prompts"][0]["sessions"]:
            context = {"source_version": version}
            if fault == "agent_error" and version == 1:
                context["agent_error"] = "RuntimeError: upload failed"
            assert session["runner_reward_info"] == {
                "reward": session["reward"],
                "metrics": {},
                "reward_context": context,
            }
        assert result["checkpoint"] == config.validation.checkpoints[0]  # repeated path, independent result
        assert [prompt["dataset_index"] for prompt in result["prompts"]] == [0]
        assert len(result["replica_versions"]) == 2
        assert set(result["replica_versions"].values()) == {version}
        assert "publish_metrics" not in result
        assert result["service_initialization_s"] == 0.5
        assert ("initial_load" if version <= concurrency else "reload") in result["timing_s"]
        assert result["rollout_started_at"] <= result["finished_at"]
        windows[version] = (
            datetime.fromisoformat(result["rollout_started_at"]),
            datetime.fromisoformat(result["rollout_finished_at"]),
        )
    reloaded = [event for event in events if event[0] == "reload"]
    assert len(reloaded) == concurrency * 2  # only second wave, one reload per existing service
    assert {event[2] for event in reloaded} == set(runtime.llm_server_controllers)
    for event in reloaded:
        assert events.index(("fence", event[1])) < events.index(event) < events.index(("active", event[1], event[2]))
    if concurrency == 2:
        for first, second in ((1, 2), (3, 4)):
            assert max(windows[first][0], windows[second][0]) < min(windows[first][1], windows[second][1])
    assert queue.tags == {}


@pytest.mark.parametrize("stage", ["load", "warmup", "rollout"])
def test_checkpoint_stage_deadline_persists_failure_without_waiting_for_pending_work(tmp_path, monkeypatch, stage):
    config = _config(tmp_path)
    config.validation.checkpoints = config.validation.checkpoints[:1]
    config.validation.stage_timeout_s = 0.1
    validation._validate_config(config)
    queue, events = _Queue(), []
    runtime, _ = _fake_runtime(monkeypatch, queue, events)
    monkeypatch.setattr(validation, "tq", queue)
    entered, release = threading.Event(), threading.Event()
    clock = SimpleNamespace(now=0.0)
    actual_time = validation.time
    monkeypatch.setattr(
        validation,
        "time",
        SimpleNamespace(
            monotonic=lambda: clock.now,
            perf_counter=actual_time.perf_counter,
            sleep=actual_time.sleep,
        ),
    )

    def hold():
        entered.set()
        assert release.wait(5), "test did not release its pending stage"

    if stage == "load":
        original_load = validation._load_checkpoint

        def load(*args, **kwargs):
            hold()
            return original_load(*args, **kwargs)

        monkeypatch.setattr(validation, "_load_checkpoint", load)
    elif stage == "warmup":
        original_idle = validation._wait_for_idle

        def idle(*args, **kwargs):
            hold()
            return original_idle(*args, **kwargs)

        monkeypatch.setattr(validation, "_wait_for_idle", idle)
    else:
        original_generate = runtime.agent_loop_manager.generate_sequences_and_wait

        def generate(prompts):
            hold()
            return original_generate(prompts)

        runtime.agent_loop_manager.generate_sequences_and_wait = generate

    original_wait = validation.wait_futures

    def wait(futures, **kwargs):
        result = original_wait(futures, timeout=0.01, return_when=kwargs["return_when"])
        if entered.is_set():
            clock.now = 1.0
        return result

    monkeypatch.setattr(validation, "wait_futures", wait)
    executors = []

    class OwnedExecutor(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            executors.append(self)

    monkeypatch.setattr(validation, "ThreadPoolExecutor", OwnedExecutor)
    phase = "rollout" if stage == "rollout" else "load"
    try:
        with pytest.raises(TimeoutError, match=f"validation {phase} version=1 timed out"):
            validation._run_checkpoints(config, runtime, [{"prompt": "task", "extra_info": {"tools_kwargs": {}}}])
        assert entered.is_set() and not release.is_set()
        result = json.loads((Path(config.validation.result_dir) / "version_000001.json").read_text())
        assert result["status"] == result["execution_status"] == "FAIL"
        assert result["failed_stage"] == stage
        assert result["quality_complete"] is False
        assert result["num_completed_prompts"] == 0
        assert result["error"] == f"validation {phase} version=1 timed out"
    finally:
        release.set()
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)


def test_custom_scorer_without_final_proof_keeps_execution_success_and_scoring_unknown(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.validation.checkpoints = [config.validation.checkpoints[0]]
    config.reward = {"custom_reward_function": {"path": "/audited/custom_scorer.py"}}
    validation._validate_config(config)
    queue, events = _Queue(), []
    runtime, _ = _fake_runtime(monkeypatch, queue, events)
    monkeypatch.setattr(validation, "tq", queue)
    validation._run_checkpoints(
        config, runtime, [{"prompt": "task", "data_source": "test", "extra_info": {"tools_kwargs": {}}}]
    )
    result = json.loads((Path(config.validation.result_dir) / "version_000001.json").read_text())
    assert result["status"] == result["execution_status"] == "PASS"
    assert result["execution_coverage"] == 1.0
    assert result["scoring_status"] == "UNKNOWN" and result["scoring_coverage"] == 0.0
    assert result["num_completed_sessions"] == 2 and result["num_scored_sessions"] == 0
    assert result["mean_session_reward"] is None and result["mean_requested_session_reward"] is None
    assert not any(key.startswith("val-core/") for key in result["metrics"])
    assert [session["reward"] for session in result["prompts"][0]["sessions"]] == [101.0, 102.0]
    assert all(session["reward_valid"] is False for session in result["prompts"][0]["sessions"])


def test_fast_group_starts_next_checkpoint_before_slow_group_finishes(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.validation.checkpoints = [config.validation.checkpoints[0]] * 3
    validation._validate_config(config)
    queue, events = _Queue(), []
    runtime, _ = _fake_runtime(monkeypatch, queue, events, concurrency=2)
    monkeypatch.setattr(validation, "tq", queue)
    slow_started, next_started, slow_finished = threading.Event(), threading.Event(), threading.Event()

    def generate(prompts):
        version = tu.get_non_tensor_data(prompts, "global_steps", None)
        if version == 1:
            slow_started.set()
            assert next_started.wait(3), "fast group waited for the slow checkpoint"
        elif version == 2:
            assert slow_started.wait(3)
        else:
            assert slow_started.is_set() and not slow_finished.is_set()
            next_started.set()
        for index in range(len(prompts)):
            queue.add(tu.get_non_tensor_data(prompts[index], "uid", None), version=version, success=2)
        if version == 1:
            slow_finished.set()

    runtime.agent_loop_manager.generate_sequences_and_wait = generate
    validation._run_checkpoints(config, runtime, [{"prompt": "task", "extra_info": {"tools_kwargs": {}}}])
    assert next_started.is_set() and slow_finished.is_set()
    results = [json.loads(p.read_text()) for p in Path(config.validation.result_dir).glob("*.json")]
    assert len(results) == 3 and all(result["status"] == "PASS" for result in results)
    assert not queue.tags


class _AwaitableRef:
    """A local actor result usable by both ray.get and the shared async drain."""

    def __init__(self, value):
        self.value = value

    def __await__(self):
        return asyncio.sleep(0, result=self.value).__await__()


class _AwaitableRemote(_Remote):
    def remote(self, *args, **kwargs):
        return _AwaitableRef(self.call(*args, **kwargs))


class _ParallelRouter:
    """Serialize real Router calls like an actor while rollout threads run."""

    def __init__(self, servers, events):
        self.impl = ValidationRequestRouter(servers)
        self.lock = threading.RLock()
        self.events = events
        self.snapshot_version = 0
        self.sources = []
        for server in servers:
            self.impl.mark_active(server, weight_version=1)
        self.refresh()

    def refresh(self):
        with self.lock:
            self.snapshot_version += 1
            self.impl.update_routing_snapshot(
                {
                    server: {
                        **_snapshot(running=self.impl._inflight_count(server), version=self.snapshot_version),
                        "kv_cache_usage": 0.1 if self.impl._inflight_count(server) else 0.0,
                    }
                    for server in self.impl._server_states
                }
            )

    def __getattr__(self, name):
        def call(*args, **kwargs):
            with self.lock:
                value = getattr(self.impl, name)(*args, **kwargs)
                if name == "plan_validation_repack":
                    self.sources = list(value.idle_sources) + [s for g in value.groups for s in g.sources]
                    self.events.append(("repack", value))
                elif name == "set_validation_versions":
                    self.events.append(("assign", kwargs.get("repacked", False), value))
                return value

        return _AwaitableRemote(call)


def _parallel_runtime(monkeypatch, queue, events, *, repack_enabled=True, fault=None):
    control = object()
    state = SimpleNamespace(
        old_started=threading.Event(),
        abort_started=threading.Event(),
        abort_repeated=threading.Event(),
        acknowledge_abort=threading.Event(),
        old_complete=threading.Event(),
        old_finished=threading.Event(),
        next_started=threading.Event(),
        continued=threading.Event(),
    )

    def fake_wait(refs, *, num_returns, timeout):
        ready = [ref for ref in refs if ref is not control]
        return ready[:num_returns], []

    monkeypatch.setattr(validation.ray, "wait", fake_wait)
    monkeypatch.setattr(validation.ray, "get", lambda ref, **_kwargs: ref.value)

    def abort(server, *, abort_only):
        assert abort_only
        with router.lock:
            assert server in router.sources
            assert router.impl._server_states[server].lifecycle == DRAINING
            events.append(("abort", server))
            state.abort_started.set()
            if sum(event[0] == "abort" for event in events) > 1:
                state.abort_repeated.set()
        if fault == "abort":
            raise RuntimeError("source abort failed")

    def load(server, _checkpoint, *, weight_version):
        with router.lock:
            assert router.impl._server_states[server].lifecycle == UPDATING
            assert router.impl._inflight_count(server) == 0
            assert router.impl._validation_targets[server] == weight_version
            if repack_enabled:
                assert state.acknowledge_abort.is_set() and not state.old_finished.is_set()
            else:
                assert state.old_finished.is_set()
            events.append(("reload", server, weight_version))
        if fault == "reload":
            raise RuntimeError("source reload failed")
        return weight_version

    controllers = {
        server: SimpleNamespace(
            abort_all_requests=_AwaitableRemote(lambda server=server, **kw: abort(server, **kw)),
            load_hf_checkpoint=_AwaitableRemote(lambda path, server=server, **kw: load(server, path, **kw)),
        )
        for server in ("server_0", "server_1")
    }
    router = _ParallelRouter(controllers, events)

    def generate(prompts):
        version = tu.get_non_tensor_data(prompts, "global_steps", None)
        uids = [tu.get_non_tensor_data(prompts[index], "uid", None) for index in range(len(prompts))]
        with router.lock:
            for uid in uids:
                assert router.impl.bind_route(uid, min_weight_version=version) == version
                server, _ = router.impl.acquire_lease(uid, lease_id="request", weight_version=version)
                events.append(("request", version, uid, server))
            router.refresh()
        if version == 1:
            state.old_started.set()
            if repack_enabled and fault != "abort":
                assert state.acknowledge_abort.wait(5), "test did not acknowledge the aborted old request"
                with router.lock:
                    assert len(router.sources) == 1
                    source = router.sources[0]
                    migrated = [uid for uid in uids if router.impl._sessions[uid].leases["request"] == source]
                    assert len(migrated) == 1
                    uid = migrated[0]
                    router.impl.release_lease(uid, "request")
                    server, _ = router.impl.acquire_lease(uid, lease_id="continuation", weight_version=1)
                    assert server != source and router.impl._server_states[server].weight_version == 1
                    events.append(("continuation", 1, uid, server))
                    router.refresh()
                    state.continued.set()
            assert state.old_complete.wait(5), "test did not release its old checkpoint tail"
        else:
            assert version == 2
            assert state.old_finished.is_set() is not repack_enabled
            events.append(("next_rollout", version))
            state.next_started.set()
        with router.lock:
            for uid in uids:
                router.impl.release_lease(uid, "request")
                router.impl.release_lease(uid, "continuation")
                router.impl.release_route(uid, weight_version=version)
                queue.add(uid, version=version)
                for key in queue.rewards:
                    if key.startswith(uid + "_"):
                        queue.extra_fields[key]["runner_reward_info"] = {
                            "reward": queue.rewards[key],
                            "metrics": {},
                            "reward_context": {"source_version": version},
                        }
            router.refresh()
            if version == 1:
                state.old_finished.set()

    runtime = SimpleNamespace(
        llm_server_controllers=controllers,
        initial_versions=dict.fromkeys(controllers, 1),
        initial_load_timings=dict.fromkeys(controllers, 0.25),
        service_started_at="2026-10-27T00:00:00+00:00",
        initialization_s=0.5,
        rollout_router=router,
        llm_server_state_tracker=SimpleNamespace(
            start=_AwaitableRemote(lambda: None),
            refresh_once=_AwaitableRemote(router.refresh),
            wait_for_stop=SimpleNamespace(remote=lambda: control),
        ),
        agent_loop_manager=SimpleNamespace(generate_sequences_and_wait=generate),
        # A released subset must continue using its original admission runner.
        admission_runners={tuple(controllers): {"agent": ("scoped_runner", 4)}},
    )
    return runtime, router, state


def test_repack_released_source_runs_next_checkpoint_while_old_tail_keeps_exact_version(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.validation.repack.schedule_interval_s = 0.01
    config.validation.batch_size = 2
    validation._validate_config(config)
    queue, events = _Queue(), []
    runtime, router, state = _parallel_runtime(monkeypatch, queue, events)
    monkeypatch.setattr(validation, "tq", queue)
    samples = [{"prompt": f"task {index}", "extra_info": {"tools_kwargs": {}}} for index in range(2)]
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(validation._run_checkpoints, config, runtime, samples)
        try:
            assert state.old_started.wait(3)
            # Abort alone does not permit loading: the owner still owes its release ACK.
            assert state.abort_repeated.wait(3)
            with router.lock:
                source = router.sources[0]
                assert router.impl._server_states[source].lifecycle == DRAINING
                assert router.impl._inflight_count(source) == 1
                assert not any(event[0] == "reload" for event in events)
            assert not state.next_started.is_set()
            state.acknowledge_abort.set()
            assert state.continued.wait(3)
            assert state.next_started.wait(3), "released source did not evaluate the next checkpoint"
            assert not state.old_finished.is_set()
            state.old_complete.set()
            future.result(timeout=5)
        finally:
            state.acknowledge_abort.set()
            state.old_complete.set()
    results = [json.loads((Path(config.validation.result_dir) / f"version_{v:06d}.json").read_text()) for v in (1, 2)]
    assert all(result["status"] == "PASS" for result in results)
    for version, result in enumerate(results, start=1):
        assert [row["dataset_index"] for row in result["prompts"]] == [0, 1]
        assert len({row["uid"] for row in result["prompts"]}) == 2
        assert result["num_scored_sessions"] == 4
        assert result["admission"]["agent"]["runner"] == "scoped_runner"
        assert set(result["replica_versions"].values()) == {version}
    assert queue.tags == {}
    assert len(results[0]["remaining_replicas"]) == len(results[1]["replica_group"]) == 1
    assert results[0]["repack"][0]["sources"] == results[1]["replica_group"]
    assert any(event[0] == "assign" and event[1] is True for event in events)
    assert len([event for event in events if event[0] == "continuation"]) == 1
    assert len([event for event in events if event[0] == "request"]) == 4


@pytest.mark.parametrize("fault", ["abort", "reload"])
def test_repack_drain_or_reload_failure_stops_checkpoint_sequence(tmp_path, monkeypatch, fault):
    config = _config(tmp_path)
    config.validation.repack.schedule_interval_s = 0.01
    config.validation.batch_size = 2
    validation._validate_config(config)
    queue, events = _Queue(), []
    runtime, _router, state = _parallel_runtime(monkeypatch, queue, events, fault=fault)
    monkeypatch.setattr(validation, "tq", queue)
    samples = [{"prompt": f"task {index}", "extra_info": {"tools_kwargs": {}}} for index in range(2)]
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(validation._run_checkpoints, config, runtime, samples)
        try:
            assert state.abort_started.wait(3)
            if fault == "reload":
                state.acknowledge_abort.set()
                with pytest.raises(RuntimeError, match="source reload failed"):
                    future.result(timeout=5)
            else:
                with pytest.raises(ExceptionGroup) as raised:
                    future.result(timeout=5)
                assert isinstance(raised.value.exceptions[0], RuntimeError)
                assert str(raised.value.exceptions[0]) == "source abort failed"
            assert not state.next_started.is_set()
            results = [json.loads(path.read_text()) for path in Path(config.validation.result_dir).glob("*.json")]
            assert results and all(result["status"] == "FAIL" for result in results)
            if fault == "abort":
                assert not any(event[0] == "reload" for event in events)
        finally:
            state.acknowledge_abort.set()
            state.old_complete.set()


def test_disjoint_group_weight_reloads_overlap(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.validation.checkpoints = [config.validation.checkpoints[0]] * 4
    validation._validate_config(config)
    queue, events = _Queue(), []
    runtime, _ = _fake_runtime(monkeypatch, queue, events, concurrency=2)
    monkeypatch.setattr(validation, "tq", queue)
    both_loading = threading.Barrier(2, timeout=3)
    first_servers = {
        version: next(sid for sid, v in runtime.initial_versions.items() if v == version) for version in (1, 2)
    }
    for sid in first_servers.values():
        original = runtime.llm_server_controllers[sid].load_hf_checkpoint.call

        def load(path, *, weight_version, original=original):
            both_loading.wait()  # Serial reload scheduling cannot pass this rendezvous.
            return original(path, weight_version=weight_version)

        runtime.llm_server_controllers[sid].load_hf_checkpoint = _Remote(load)
    validation._run_checkpoints(config, runtime, [{"prompt": "task", "extra_info": {"tools_kwargs": {}}}])
    results = [json.loads(p.read_text()) for p in Path(config.validation.result_dir).glob("*.json")]
    assert len(results) == 4 and all(result["status"] == "PASS" for result in results)
    reloads = [result for result in results if "reload_started_at" in result]
    assert len(reloads) == 2
    assert max(result["reload_started_at"] for result in reloads) < min(
        result["reload_finished_at"] for result in reloads
    )


@pytest.mark.asyncio
async def test_evaluation_group_admission_keeps_fast_version_moving_when_slow_version_is_saturated(tmp_path):
    from uni_agent.framework.framework import _RunnerConfig

    from laminar.eval.runtime import _admission_config

    config = _config(tmp_path)
    config.actor_rollout_ref.rollout.custom = {
        "agent_framework": {
            "agent_runners": {
                "task": {
                    "runner_fqn": "uni_agent.framework.task_runner.run_task",
                    "dispatch_mode": "ray_task",
                    "max_concurrent_sessions": 4,
                    "runner_kwargs": {"task_config_path": "/same/task.yaml"},
                }
            }
        }
    }
    scoped, assigned = _admission_config(config, {1: ["slow"], 2: ["fast"]})
    framework = GatewayAgentFramework(
        None,
        runner_registry={
            name: _RunnerConfig.from_config(name, runner)
            for name, runner in scoped.actor_rollout_ref.rollout.custom.agent_framework.agent_runners.items()
        },
    )
    release_slow = asyncio.Event()
    release_fast = asyncio.Event()
    entered_fast = asyncio.Event()
    slow_name = assigned[("slow",)]["task"][0]
    fast_name = assigned[("fast",)]["task"][0]

    async def episode(**kwargs):
        if kwargs["runner_name"] == slow_name:
            await release_slow.wait()
        else:
            entered_fast.set()
            await release_fast.wait()
        return [], {}

    framework._run_agent_episode = episode
    observed = []

    async def submit(version, server):
        samples = [{"prompt": "unchanged", "extra_info": {"tools_kwargs": {}}}]
        run = validation._CheckpointRun(
            result={"version": version}, path=tmp_path / f"{version}.json", started=0.0, server_ids=[server]
        )
        runtime = SimpleNamespace(admission_runners=assigned, agent_loop_manager=None)
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Only capture the adapter call; admission itself uses the real Framework owner below.
            runtime.agent_loop_manager = SimpleNamespace(generate_sequences_and_wait=lambda prompts: None)
            validation._launch_batch(config, runtime, samples, run, executor)
            run.future.result()
        fields = framework._extract_sample_fields(prompts=run.prompts, sample_index=0)
        observed.append((fields["agent_name"], run.result["admission"]["task"]["max_concurrent_sessions"]))
        return await framework._submit_agent_episode(
            sample_fields=fields, sample_index=0, session_index=0, global_steps=version, sampling_params={}
        )

    tasks = [await submit(1, "slow"), await submit(1, "slow")]
    blocked = asyncio.create_task(submit(1, "slow"))
    try:
        await asyncio.sleep(0)
        assert not blocked.done()
        fast = await asyncio.wait_for(submit(2, "fast"), 1)
        tasks.append(fast)
        await asyncio.wait_for(entered_fast.wait(), 1)
        assert not blocked.done() and not release_slow.is_set()
        release_fast.set()
        await fast
        await asyncio.sleep(0)
        # The same physical group is reused with a new checkpoint version and the same admission owner.
        tasks.append(await asyncio.wait_for(submit(3, "fast"), 1))
        assert not blocked.done()
        assert observed[-1] == observed[-2] == (fast_name, 2)
    finally:
        release_slow.set()
        release_fast.set()
        tasks.append(await blocked)
        await asyncio.gather(*tasks)
    await asyncio.sleep(0)
    assert framework._admitted_sessions == framework._completed_sessions == 5


def test_evaluation_maps_each_original_runner_to_its_group_without_changing_individual_budgets(tmp_path):
    from uni_agent.framework.framework import _RunnerConfig

    from laminar.eval.runtime import _admission_config

    config = _config(tmp_path)
    config.validation.batch_size = 2
    runners = {
        name: {
            "runner_fqn": "uni_agent.framework.task_runner.run_task",
            "dispatch_mode": "ray_task",
            "max_concurrent_sessions": limit,
            "runner_kwargs": {"task_config_path": f"/{name}/task.yaml"},
        }
        for name, limit in [("task", 4), ("other", 6)]
    }
    config.actor_rollout_ref.rollout.custom = {"agent_framework": {"agent_runners": runners}}
    scoped, assigned = _admission_config(config, {1: ["a"], 2: ["b"]})
    framework = object.__new__(GatewayAgentFramework)
    framework.runner_registry = {
        name: _RunnerConfig.from_config(name, runner)
        for name, runner in scoped.actor_rollout_ref.rollout.custom.agent_framework.agent_runners.items()
    }
    for original, limit in [("task", 4), ("other", 6)]:
        assert sum(group[original][1] for group in assigned.values()) == limit
    samples = [
        {"prompt": name, "agent_name": name, "extra_info": {"tools_kwargs": {"task": {"name": name}}}}
        for name in runners
    ]
    runtime = SimpleNamespace(
        admission_runners=assigned,
        agent_loop_manager=SimpleNamespace(generate_sequences_and_wait=lambda prompts: None),
    )
    run = validation._CheckpointRun(result={"version": 2}, path=tmp_path / "result.json", started=0.0, server_ids=["b"])
    with ThreadPoolExecutor(max_workers=1) as executor:
        validation._launch_batch(config, runtime, samples, run, executor)
        run.future.result()
    for index, original in enumerate(runners):
        fields = framework._extract_sample_fields(prompts=run.prompts, sample_index=index)
        selected, runner = framework._resolve_runner(fields)
        assert selected == assigned[("b",)][original][0]
        assert runner.max_concurrent_sessions == assigned[("b",)][original][1]
        assert runner.runner_kwargs["task_config_path"] == f"/{original}/task.yaml"
        assert fields["raw_prompt"] == original
        assert fields["tools_kwargs"] == samples[index]["extra_info"]["tools_kwargs"]


def _resume_fixture(config, tmp_path, *, indices=(0, 2), checkpoint=None, version=17, rows=5):
    import hashlib

    checkpoint = checkpoint or config.validation.checkpoints[0]
    protocol = {"task_config_sha256": "caller-audited-task", "checkpoint_receipt_sha256": "caller-audited-weights"}
    config.validation.resume_protocol = protocol
    metadata = {
        "checkpoint": checkpoint,
        "dataset": {"sha256": hashlib.sha256(Path(config.validation.data_path).read_bytes()).hexdigest(), "rows": rows},
        "sampling": OmegaConf.to_container(config.actor_rollout_ref.rollout.val_kwargs),
        "seed": 42,
        "full_determinism": False,
    }
    prompts = [
        {
            "dataset_index": index,
            "uid": f"original-{version}-{index}",
            "terminal": {"status": "finished", "global_steps": version},
            "sessions": [
                {
                    "session_index": session,
                    "reward": 0.0,
                    "runner_reward_info": {
                        "reward_context": {"eval_exit_code": -1, "found_eval_status": False},
                    },
                }
                for session in range(config.actor_rollout_ref.rollout.val_kwargs.n)
            ],
        }
        for index in indices
    ]
    source = tmp_path / f"source-{version}.json"
    source.write_text(json.dumps({**metadata, "version": version, "status": "RUNNING", "prompts": prompts}))
    reference = {
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "submission_id": "original-job",
        "version": version,
    }
    manifest = tmp_path / f"resume-{version}.json"
    manifest.write_text(json.dumps({**metadata, "protocol": protocol, "sources": [reference]}))
    entry = {"path": str(manifest), "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    config.validation.resume_results = [entry]
    samples = [{"prompt": str(index), "extra_info": {"tools_kwargs": {}}} for index in range(rows)]
    return samples, source, manifest, prompts


def test_resume_only_generates_missing_original_indices_and_persists_before_clear(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.validation.checkpoints = [config.validation.checkpoints[0]]
    config.validation.batch_size = 500
    samples, _, _, original = _resume_fixture(config, tmp_path, indices=(0, 2, 499), rows=500)
    restored = validation._read_resume_results(config, samples)
    validation._validate_config(config)
    queue, events = _Queue(), []
    runtime, _ = _fake_runtime(monkeypatch, queue, events)
    generate = runtime.agent_loop_manager.generate_sequences_and_wait
    generated = []

    def capture(prompts):
        generated.extend(tu.get(prompts, "raw_prompt"))
        generate(prompts)

    runtime.agent_loop_manager.generate_sequences_and_wait = capture
    clear = queue.kv_clear

    def persisted_clear(keys, *, partition_id):
        result = json.loads((Path(config.validation.result_dir) / "version_000001.json").read_text())
        records = {record["uid"]: record for record in result["prompts"]}
        assert all(key in records or key.rsplit("_", 2)[0] in records for key in keys)
        assert all(record["uid"] in records for record in original)
        clear(keys, partition_id=partition_id)

    queue.kv_clear = persisted_clear
    monkeypatch.setattr(validation, "tq", queue)
    validation._run_checkpoints(config, runtime, samples, restored)
    result = json.loads((Path(config.validation.result_dir) / "version_000001.json").read_text())
    assert generated == [str(index) for index in range(500) if index not in (0, 2, 499)]
    assert [record["dataset_index"] for record in result["prompts"]] == list(range(500))
    assert result["num_expected_sessions"] == result["num_completed_sessions"] == 1000
    assert result["num_scored_sessions"] == 994
    assert result["num_unscored_sessions"] == 6
    assert result["scoring_status"] == "UNKNOWN"
    assert result["mean_requested_session_reward"] is None
    assert not any(key.startswith("val-core/") for key in result["metrics"])
    assert result["num_completed_prompts"] == 500 and result["num_inherited_prompts"] == 3
    assert result["mean_session_reward"] == 101.5
    for index in (0, 2, 499):
        record = result["prompts"][index]
        expected = next(item for item in original if item["dataset_index"] == index)
        assert record["uid"] == expected["uid"] and record["terminal"] == expected["terminal"]
        for session, original_session in zip(record["sessions"], expected["sessions"], strict=True):
            assert session["reward"] == original_session["reward"]
            assert session["runner_reward_info"] == original_session["runner_reward_info"]
            assert session["reward_valid"] is False and session["scorer_status"] == "UNKNOWN"
        assert record["source_result"]["version"] == 17
    # Owner completion is unchanged; UNKNOWN scorer context is never promoted to a scorer verdict.
    assert "scoring_integrity" not in result


def test_resume_multi_batch_keeps_original_indices_and_persists_prompt_progress(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config.validation.checkpoints = [config.validation.checkpoints[0]]
    config.validation.batch_size = 2
    samples, _, _, original = _resume_fixture(config, tmp_path, indices=(0, 3, 7), rows=8)
    restored = validation._read_resume_results(config, samples)
    validation._validate_config(config)
    queue, events = _Queue(), []
    runtime, _ = _fake_runtime(monkeypatch, queue, events)
    generate = runtime.agent_loop_manager.generate_sequences_and_wait
    generated, persisted_progress = [], []
    expected_batches = [[1, 2], [4, 5], [6]]
    inherited = {record["dataset_index"]: record for record in original}

    def capture(prompts):
        generated.append([int(value) for value in tu.get(prompts, "raw_prompt")])
        assert generated == expected_batches[: len(generated)]
        generate(prompts)

    runtime.agent_loop_manager.generate_sequences_and_wait = capture
    clear = queue.kv_clear

    def persisted_clear(keys, *, partition_id):
        result = json.loads((Path(config.validation.result_dir) / "version_000001.json").read_text())
        records = {record["dataset_index"]: record for record in result["prompts"]}
        expected_indices = sorted([*inherited, *(index for batch in generated for index in batch)])
        assert list(records) == expected_indices
        assert result["num_completed_prompts"] == len(records)
        assert result["num_inherited_prompts"] == len(inherited)
        for index, original_record in inherited.items():
            assert records[index]["uid"] == original_record["uid"]
            assert records[index]["terminal"] == original_record["terminal"]
        uids = {record["uid"] for record in records.values()}
        assert all(key in uids or key.rsplit("_", 2)[0] in uids for key in keys)
        persisted_progress.append(result["num_completed_prompts"])
        clear(keys, partition_id=partition_id)
        assert queue.tags == {}

    queue.kv_clear = persisted_clear
    monkeypatch.setattr(validation, "tq", queue)
    validation._run_checkpoints(config, runtime, samples, restored)
    result = json.loads((Path(config.validation.result_dir) / "version_000001.json").read_text())
    assert generated == expected_batches
    assert persisted_progress == [5, 7, 8]
    assert result["status"] == "PASS"
    assert [record["dataset_index"] for record in result["prompts"]] == list(range(8))
    assert len({record["uid"] for record in result["prompts"]}) == 8
    assert result["num_completed_prompts"] == 8 and result["num_inherited_prompts"] == 3
    assert result["num_expected_sessions"] == result["num_completed_sessions"] == 16
    assert result["num_scored_sessions"] == 10 and result["num_unscored_sessions"] == 6
    assert queue.tags == queue.rewards == queue.extra_fields == {}


@pytest.mark.parametrize(
    "fault",
    [
        "digest",
        "dataset",
        "sampling",
        "checkpoint",
        "protocol",
        "version",
        "version_bool",
        "path",
        "sessions",
        "agent_error",
        "full",
        "conflict",
        "duplicate_uid",
        "repeated_resume",
    ],
)
def test_resume_rejects_incompatible_or_conflicting_sources(tmp_path, fault):
    import hashlib

    config = _config(tmp_path)
    samples, source, manifest, _ = _resume_fixture(config, tmp_path)
    payload = json.loads(source.read_text())
    meta = json.loads(manifest.read_text())
    if fault == "digest":
        source.write_text(source.read_text() + " ")
    elif fault == "dataset":
        payload["dataset"]["sha256"] = "other"
    elif fault == "sampling":
        payload["sampling"]["temperature"] = 0.5
    elif fault == "checkpoint":
        payload["checkpoint"] = "/other"
    elif fault == "protocol":
        meta["protocol"] = {"task_config_sha256": "other"}
    elif fault == "version":
        payload["version"] = 18
    elif fault == "version_bool":
        payload["version"] = True
        meta["sources"][0]["version"] = 1
    elif fault == "path":
        config.validation.persistent_roots = [str(tmp_path / "other")]
    elif fault == "sessions":
        payload["prompts"][0]["sessions"].pop()
    elif fault == "agent_error":
        payload["prompts"][0]["sessions"][0]["runner_reward_info"]["reward_context"]["agent_error"] = "upload failed"
    elif fault == "full":
        samples, source, manifest, _ = _resume_fixture(config, tmp_path, indices=tuple(range(5)))
        payload = json.loads(source.read_text())
        meta = json.loads(manifest.read_text())
    elif fault == "conflict":
        record = json.loads(json.dumps(payload["prompts"][0]))
        record["sessions"][0]["reward"] = 1.0
        payload["prompts"].append(record)
    elif fault == "duplicate_uid":
        payload["prompts"][1]["uid"] = payload["prompts"][0]["uid"]
    elif fault == "repeated_resume":
        payload["prompts"][0]["source_result"] = dict(meta["sources"][0])
    if fault != "digest":
        source.write_text(json.dumps(payload))
        meta["sources"][0]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(meta))
        config.validation.resume_results[0].sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        validation._read_resume_results(config, samples)


def test_resume_checkpoint_isolation_and_equal_duplicate_deduplication(tmp_path):
    import hashlib

    config = _config(tmp_path)
    samples, _, manifest, _ = _resume_fixture(config, tmp_path, indices=(3, 0), version=17)
    first = OmegaConf.to_container(config.validation.resume_results[0])
    second_checkpoint = str(tmp_path / "other-checkpoint")
    config.validation.checkpoints[1] = second_checkpoint
    samples, _, _, _ = _resume_fixture(config, tmp_path, indices=(2,), checkpoint=second_checkpoint, version=18)
    second = OmegaConf.to_container(config.validation.resume_results[0])
    meta = json.loads(manifest.read_text())
    meta["sources"].append({**meta["sources"][0], "submission_id": "duplicate-job"})
    manifest.write_text(json.dumps(meta))
    first["sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
    config.validation.resume_results = [first, first, second]
    restored = validation._read_resume_results(config, samples)
    assert {
        checkpoint: [record["dataset_index"] for record in records] for checkpoint, records in restored.items()
    } == {
        config.validation.checkpoints[0]: [3, 0],
        second_checkpoint: [2],
    }
    assert all(
        record["source_result"]["submission_id"] == "original-job"
        for records in restored.values()
        for record in records
    )
    config.validation.resume_results = []
    assert validation._read_resume_results(config, samples) == {}


def test_invalid_resume_is_rejected_before_tq_or_gpu_creation(tmp_path, monkeypatch):
    import datasets

    config = _config(tmp_path)
    samples, source, _, _ = _resume_fixture(config, tmp_path)
    source.write_text("changed bytes")
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: SimpleNamespace(to_list=lambda: samples))
    entered = []
    monkeypatch.setattr(validation.tq, "init", lambda *args: entered.append("tq"))
    monkeypatch.setattr(validation, "initialize_evaluation_runtime", lambda *args: entered.append("gpu"))
    with pytest.raises(ValueError, match="digest mismatch"):
        validation.main_task._function(config)
    assert entered == []


def test_agent_error_in_earlier_batch_preserves_all_records_and_next_checkpoint(tmp_path, monkeypatch):
    config = _config(tmp_path)
    validation._validate_config(config)
    queue, events = _Queue(), []
    runtime, _ = _fake_runtime(monkeypatch, queue, events)
    generate = runtime.agent_loop_manager.generate_sequences_and_wait
    first_uid = None

    def fail_first_session(prompts):
        nonlocal first_uid
        generate(prompts)
        if first_uid is None:
            first_uid = tu.get(prompts, "uid")[0]
            queue.extra_fields[first_uid + "_0_1"]["runner_reward_info"]["reward_context"]["agent_error"] = (
                "Seed stateful shell tmux server lost"
            )

    runtime.agent_loop_manager.generate_sequences_and_wait = fail_first_session
    clear = queue.kv_clear

    def persisted_clear(keys, *, partition_id):
        records = [
            record
            for path in Path(config.validation.result_dir).glob("*.json")
            for record in json.loads(path.read_text())["prompts"]
        ]
        uids = {record["uid"] for record in records}
        assert all(key in uids or key.rsplit("_", 2)[0] in uids for key in keys)
        clear(keys, partition_id=partition_id)

    queue.kv_clear = persisted_clear
    monkeypatch.setattr(validation, "tq", queue)
    samples = [{"prompt": str(index), "extra_info": {"tools_kwargs": {}}} for index in range(2)]
    validation._run_checkpoints(config, runtime, samples)
    first = json.loads((Path(config.validation.result_dir) / "version_000001.json").read_text())
    second = json.loads((Path(config.validation.result_dir) / "version_000002.json").read_text())
    assert first["status"] == "FAIL" and second["status"] == "PASS"
    assert first["num_completed_prompts"] == second["num_completed_prompts"] == 2
    assert first["num_scored_sessions"] == second["num_scored_sessions"] == 4
    assert first["num_failed_sessions"] == second["num_failed_sessions"] == 0
    assert first["mean_session_reward"] == 101.5 and second["mean_session_reward"] == 201.5
    assert first["prompts"][0]["uid"] == first_uid
    assert first["prompts"][0]["sessions"][0]["runner_reward_info"]["reward_context"]["agent_error"] == (
        "Seed stateful shell tmux server lost"
    )
    assert "agent_error" not in first and "failed_stage" not in first
    assert [prompt["dataset_index"] for prompt in first["prompts"]] == [0, 1]
    assert any(event[0] == "reload" and event[1] == 2 for event in events)


@pytest.mark.parametrize("fault", ["partial", "agent_error"])
def test_multi_batch_task_failures_advance_by_prompts_and_preserve_next_checkpoint(tmp_path, monkeypatch, fault):
    config = _config(tmp_path)
    config.validation.batch_size = 2
    validation._validate_config(config)
    queue, events = _Queue(), []
    runtime, _ = _fake_runtime(monkeypatch, queue, events)
    generate = runtime.agent_loop_manager.generate_sequences_and_wait
    expected_batches = [(version, batch) for version in (1, 2) for batch in ([0, 1], [2, 3], [4])]
    generated, persisted_progress = [], []
    failed_uid = None

    def fail_first_batch(prompts):
        nonlocal failed_uid
        version = tu.get_non_tensor_data(prompts, "global_steps", None)
        generated.append((version, [int(value) for value in tu.get(prompts, "raw_prompt")]))
        assert generated == expected_batches[: len(generated)]
        generate(prompts)
        if len(generated) == 1:
            failed_uid, partial_uid = tu.get(prompts, "uid")
            if fault == "partial":
                # Both prompts complete, with zero and one successful session respectively.
                for key in list(queue.tags):
                    if key.startswith(failed_uid + "_") or key.startswith(partial_uid + "_1_"):
                        del queue.tags[key]
                        del queue.rewards[key]
                        del queue.extra_fields[key]
                queue.tags[failed_uid]["status"] = "failure"
            else:
                queue.extra_fields[failed_uid + "_0_1"]["runner_reward_info"]["reward_context"]["agent_error"] = (
                    "Seed stateful shell tmux server lost"
                )

    runtime.agent_loop_manager.generate_sequences_and_wait = fail_first_batch
    clear = queue.kv_clear

    def persisted_clear(keys, *, partition_id):
        version = generated[-1][0]
        result = json.loads((Path(config.validation.result_dir) / f"version_{version:06d}.json").read_text())
        records = result["prompts"]
        expected_indices = [index for batch_version, batch in generated if batch_version == version for index in batch]
        assert [record["dataset_index"] for record in records] == expected_indices
        assert result["num_completed_prompts"] == len(records)
        uids = {record["uid"] for record in records}
        assert all(key in uids or key.rsplit("_", 2)[0] in uids for key in keys)
        persisted_progress.append((version, result["num_completed_prompts"], result["num_scored_sessions"]))
        clear(keys, partition_id=partition_id)
        assert queue.tags == {}

    queue.kv_clear = persisted_clear
    monkeypatch.setattr(validation, "tq", queue)
    samples = [{"prompt": str(index), "extra_info": {"tools_kwargs": {}}} for index in range(5)]
    validation._run_checkpoints(config, runtime, samples)
    first = json.loads((Path(config.validation.result_dir) / "version_000001.json").read_text())
    second = json.loads((Path(config.validation.result_dir) / "version_000002.json").read_text())
    assert generated == expected_batches
    first_scores = (1, 5, 7) if fault == "partial" else (4, 8, 10)
    assert persisted_progress == [
        (version, completed, scored)
        for version, scores in ((1, first_scores), (2, (4, 8, 10)))
        for completed, scored in zip((2, 4, 5), scores, strict=True)
    ]
    assert first["status"] == ("PARTIAL" if fault == "partial" else "FAIL")
    assert second["status"] == "PASS"
    for result in (first, second):
        assert result["num_completed_prompts"] == len(result["prompts"]) == 5
        assert [record["dataset_index"] for record in result["prompts"]] == list(range(5))
        assert len({record["uid"] for record in result["prompts"]}) == 5
        assert result["num_expected_sessions"] == 10
    assert first["num_scored_sessions"] == first_scores[-1]
    assert first["num_failed_sessions"] == (3 if fault == "partial" else 0)
    assert second["num_scored_sessions"] == 10 and second["num_failed_sessions"] == 0
    assert first["prompts"][0]["uid"] == failed_uid
    if fault == "partial":
        assert first["prompts"][0]["terminal"]["status"] == "failure"
        assert first["prompts"][0]["sessions"] == []
        assert len(first["prompts"][1]["sessions"]) == 1
    else:
        assert first["prompts"][0]["sessions"][0]["runner_reward_info"]["reward_context"]["agent_error"] == (
            "Seed stateful shell tmux server lost"
        )
    assert any(event[0] == "reload" and event[1] == 2 for event in events)
    assert queue.tags == queue.rewards == queue.extra_fields == {}


@pytest.mark.parametrize("crash_phase", ["rollout", "reload"])
def test_evaluation_recovers_dead_replica_and_completes_checkpoint_pass(tmp_path, monkeypatch, crash_phase):
    config = _config(tmp_path)
    config.validation.batch_size = 2
    validation._validate_config(config)
    queue, events = _Queue(), []
    runtime, router, state = _parallel_runtime(monkeypatch, queue, events, repack_enabled=False)
    monkeypatch.setattr(validation, "tq", queue)

    dead_observed = threading.Event()
    restarted = threading.Event()
    health_state = {"server_0": "healthy", "server_1": "healthy"}

    def refresh():
        with router.lock:
            router.snapshot_version += 1
            router.impl.update_routing_snapshot(
                {
                    server: {
                        **_snapshot(running=router.impl._inflight_count(server), version=router.snapshot_version),
                        "backend_health": health_state[server],
                        "kv_cache_usage": 0.1 if router.impl._inflight_count(server) else 0.0,
                    }
                    for server in router.impl._server_states
                }
            )

    runtime.llm_server_state_tracker.refresh_once = _AwaitableRemote(refresh)

    for server, controller in runtime.llm_server_controllers.items():
        controller.snapshot = _AwaitableRemote(
            lambda server=server: SimpleNamespace(backend_health=health_state[server])
        )
        controller.get_server_handle = _AwaitableRemote(lambda server=server: f"rebuilt-{server}")

        def restart_replica(*, wait_for_version, checkpoint_path=None, server=server):
            assert checkpoint_path is not None
            health_state[server] = "healthy"
            events.append(("restart", server, wait_for_version, checkpoint_path))
            restarted.set()
            return wait_for_version

        controller.restart = _AwaitableRemote(restart_replica)

    if crash_phase == "reload":
        original_load = runtime.llm_server_controllers["server_0"].load_hf_checkpoint.call
        crashed_once = False

        def load_with_crash(path, *, weight_version):
            nonlocal crashed_once
            if not crashed_once and weight_version == 2:
                crashed_once = True
                health_state["server_0"] = "unavailable"
                refresh()
                raise RuntimeError("EngineDeadError during reload")
            return original_load(path, weight_version=weight_version)

        runtime.llm_server_controllers["server_0"].load_hf_checkpoint = _AwaitableRemote(load_with_crash)
    else:
        original_generate = runtime.agent_loop_manager.generate_sequences_and_wait

        def generate_with_rollout_crash(prompts):
            version = tu.get_non_tensor_data(prompts, "global_steps", None)
            if version == 1:
                health_state["server_0"] = "unavailable"
                refresh()
                dead_observed.set()
                assert restarted.wait(5), "coordinator did not recover dead replica during rollout"
            return original_generate(prompts)

        runtime.agent_loop_manager.generate_sequences_and_wait = generate_with_rollout_crash

    samples = [{"prompt": f"task {index}", "extra_info": {"tools_kwargs": {}}} for index in range(2)]
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(validation._run_checkpoints, config, runtime, samples)
        try:
            assert state.old_started.wait(3)
            state.old_complete.set()
            future.result(timeout=5)
        finally:
            state.old_complete.set()

    results = [json.loads((Path(config.validation.result_dir) / f"version_{v:06d}.json").read_text()) for v in (1, 2)]
    assert all(result["status"] == "PASS" and result["quality_complete"] is True for result in results)
    target_version = 1 if crash_phase == "rollout" else 2
    recovered_result = results[target_version - 1]
    assert len(recovered_result["recoveries"]) == 1
    assert recovered_result["recoveries"][0]["server_id"] == "server_0"
    assert recovered_result["recoveries"][0]["version"] == target_version
    assert any(event[0] == "restart" and event[1] == "server_0" and event[2] == target_version for event in events)
