# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from laminar.eval import runtime as evaluation_runtime


def test_evaluation_import_does_not_require_training_or_relay_runtime():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import laminar.eval.main; "
            "assert not any(name == 'laminar.train' or name.startswith('laminar.train.') for name in sys.modules); "
            "assert not {'laminar.multi_version_rollout.manager', 'ucxx'}.intersection(sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


class _Ref:
    def __init__(self, value=None):
        self.value = value


class _Remote:
    def __init__(self, call):
        self.call = call

    def remote(self, *args, **kwargs):
        return _Ref(self.call(*args, **kwargs))


@pytest.mark.parametrize("initialization_failure", [None, "timeout", "get_error"])
def test_standalone_initialization_starts_only_inference_and_waits_for_every_replica(
    monkeypatch, initialization_failure
):
    with initialize_config_module(config_module="laminar.eval.config", version_base=None):
        config = compose(
            config_name="validation",
            overrides=[
                "validation.checkpoints=[/hf/a,/hf/b]",
                "actor_rollout_ref.rollout.n_gpus_per_node=4",
                "actor_rollout_ref.rollout.ray_actor_max_concurrency=2048",
                "actor_rollout_ref.rollout.agent.agent_loop_manager_class=example.Manager",
            ],
        )
    assert "trainer" not in config and "actor" not in config.actor_rollout_ref
    OmegaConf.update(
        config,
        "actor_rollout_ref.rollout.custom.agent_framework.agent_runners",
        {"task": {"runner_fqn": "uni_agent.framework.task_runner.run_task", "max_concurrent_sessions": 4}},
        force_add=True,
    )
    initialized = []
    readbacks = []
    created = []
    engines = []
    init_refs = []
    clock = iter([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
    monkeypatch.setattr(
        evaluation_runtime, "time", SimpleNamespace(perf_counter=lambda: next(clock), monotonic=lambda: 0.0)
    )
    pg = object()
    pool = SimpleNamespace(get_placement_groups=lambda: [pg])

    def make_pool(**kwargs):
        assert kwargs["process_on_nodes"] == [4]
        return pool

    monkeypatch.setattr(evaluation_runtime, "RayResourcePool", make_pool)
    monkeypatch.setattr(
        evaluation_runtime,
        "split_resource_pool",
        lambda parent, *, split_size: [SimpleNamespace(pgs=[pg], start_bundle_index=i) for i in range(4)],
    )
    monkeypatch.setattr(evaluation_runtime, "PlacementGroupSchedulingStrategy", lambda **kwargs: kwargs)

    def make_engine(**kwargs):
        engines.append(kwargs)
        return SimpleNamespace(rank=kwargs["replica_rank"])

    monkeypatch.setattr(evaluation_runtime, "vLLMReplica", make_engine)

    class _ControllerFactory:
        def options(self, **kwargs):
            assert kwargs["max_concurrency"] == 2048
            return self

        def remote(self, *, relay_client, engine, name):
            assert relay_client is None
            created.append("controller")

            def init(*, resource_pool, weight_version):
                initialized.append((name, weight_version))
                ref = _Ref()
                init_refs.append(ref)
                return ref

            return SimpleNamespace(
                init_engine=SimpleNamespace(remote=init),
                get_server_handle=_Remote(lambda: name),
            )

    tracker = SimpleNamespace(connect=_Remote(lambda **kwargs: None))
    marked = []
    router = SimpleNamespace(
        mark_active=_Remote(lambda server_id, *, weight_version: marked.append((server_id, weight_version)))
    )

    def make_router(*, servers, config):
        assert len(readbacks) == 4, "Router exposed before all expected engines initialized"
        assert set(servers) == {f"rollout_server_{i}" for i in range(4)}
        created.append("router")
        return router

    def remote_factory(cls):
        if cls is evaluation_runtime.LLMServerController:
            return _ControllerFactory()
        if cls is evaluation_runtime.ValidationRequestRouter:
            return SimpleNamespace(remote=make_router)
        assert cls is evaluation_runtime.LLMServerStateTracker, "unexpected non-inference actor created"
        created.append("snapshot_tracker")
        return SimpleNamespace(remote=lambda: tracker)

    monkeypatch.setattr(evaluation_runtime.ray, "remote", remote_factory)

    def get(ref):
        if isinstance(ref, list):
            return [get(item) for item in ref]
        if ref in init_refs:
            readbacks.append(ref)
            if initialization_failure == "get_error" and ref is init_refs[2]:
                raise RuntimeError("initial engine failed")
        return ref.value

    monkeypatch.setattr(evaluation_runtime.ray, "get", get)
    monkeypatch.setattr(
        evaluation_runtime.ray,
        "wait",
        lambda refs, **kwargs: ([], refs) if initialization_failure == "timeout" else ([refs[-1]], refs[:-1]),
    )
    adapter = object()

    def make_adapter(**kwargs):
        created.append("adapter")
        assert kwargs["llm_client"]._router is router
        runners = kwargs["config"].actor_rollout_ref.rollout.custom.agent_framework.agent_runners
        assert {name: runner.max_concurrent_sessions for name, runner in runners.items()} == {
            "task": 4,
        }
        return adapter

    monkeypatch.setattr(
        evaluation_runtime,
        "load_class_from_fqn",
        lambda *_args: SimpleNamespace(create=make_adapter),
    )
    if initialization_failure is not None:
        error, message = (
            (TimeoutError, "initialization timed out")
            if initialization_failure == "timeout"
            else (RuntimeError, "initial engine failed")
        )
        with pytest.raises(error, match=message):
            evaluation_runtime.initialize_evaluation_runtime(config)
        assert created == ["controller"] * 4
        assert marked == []
        assert readbacks == ([] if initialization_failure == "timeout" else [init_refs[3], init_refs[2]])
        return
    runtime = evaluation_runtime.initialize_evaluation_runtime(config)
    assert created == ["controller"] * 4 + ["router", "snapshot_tracker", "adapter"]
    assert [entry["model_config"].path for entry in engines] == ["/hf/a"] * 4
    assert initialized == [(f"rollout_server_{i}", 1) for i in range(4)]
    assert dict(initialized) == dict(marked)
    assert runtime.initial_versions == dict(initialized)
    assert runtime.initial_load_timings == {
        "rollout_server_3": 10.0,
        "rollout_server_2": 30.0,
        "rollout_server_1": 50.0,
        "rollout_server_0": 70.0,
    }
    assert runtime.initialization_s == 90.0
    assert runtime.agent_loop_manager is adapter
    assert runtime.admission_runners == {}
    assert config.actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions == 4
    assert config.actor_rollout_ref.model.path == "/hf/a"


@pytest.mark.parametrize(
    "sizes, total_limit, expected", [([1] * 8, 224, [28] * 8), ([2, 1], 5, [3, 2]), ([1, 1], 0, [0, 0])]
)
def test_evaluation_runner_budgets_follow_fixed_replica_groups_without_changing_task(sizes, total_limit, expected):
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "custom": {
                        "agent_framework": {
                            "agent_runners": {
                                "task": {
                                    "runner_fqn": "uni_agent.framework.task_runner.run_task",
                                    "dispatch_mode": "ray_task",
                                    "max_concurrent_sessions": total_limit,
                                    "runner_kwargs": {"task_config_path": "/original/task.yaml"},
                                }
                            }
                        }
                    }
                }
            }
        }
    )
    original = OmegaConf.to_container(config, resolve=True)
    original_runner = original["actor_rollout_ref"]["rollout"]["custom"]["agent_framework"]["agent_runners"]["task"]
    groups = {index: [f"server-{index}-{replica}" for replica in range(size)] for index, size in enumerate(sizes)}
    scoped, assigned = evaluation_runtime._admission_config(config, groups)
    runners = scoped.actor_rollout_ref.rollout.custom.agent_framework.agent_runners
    assert [runner.max_concurrent_sessions for runner in runners.values()] == expected
    assert sum(runner.max_concurrent_sessions for runner in runners.values()) == total_limit
    assert len(set(name for group in assigned.values() for name, _ in group.values())) == len(sizes)
    for group, limit in zip(assigned.values(), expected, strict=True):
        name, actual_limit = group["task"]
        assert actual_limit == limit
        assert runners[name].runner_fqn == original_runner["runner_fqn"]
        assert runners[name].runner_kwargs.task_config_path == "/original/task.yaml"
    assert OmegaConf.to_container(config, resolve=True) == original


def test_evaluation_cannot_turn_a_small_finite_group_budget_into_unlimited_admission():
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"custom": {"agent_framework": {"agent_runners": {"task": {"max_concurrent_sessions": 1}}}}}
            }
        }
    )
    with pytest.raises(ValueError, match="cannot cover every validation replica group"):
        evaluation_runtime._admission_config(config, {1: ["a"], 2: ["b"]})
