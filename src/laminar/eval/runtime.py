# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Standalone inference services for offline checkpoint evaluation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import ray
from omegaconf import DictConfig, OmegaConf
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from verl.single_controller.ray import RayResourcePool
from verl.single_controller.ray.base import split_resource_pool
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.import_utils import load_class_from_fqn
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

from laminar.eval.router import ValidationRequestRouter, validation_replica_groups
from laminar.llm_server.controller import LLMServerController
from laminar.llm_server.state_tracker import LLMServerStateTracker
from laminar.llm_server_client import LLMServerClient


@dataclass(frozen=True)
class EvaluationRuntime:
    llm_server_controllers: dict[str, ActorHandle]
    llm_server_state_tracker: ActorHandle
    rollout_router: ActorHandle
    agent_loop_manager: Any
    admission_runners: dict[tuple[str, ...], dict[str, tuple[str, int]]]
    initial_versions: dict[str, int]
    initial_load_timings: dict[str, float]
    service_started_at: str
    initialization_s: float


def _admission_config(
    config: DictConfig, groups: dict[int, list[str]]
) -> tuple[DictConfig, dict[tuple[str, ...], dict[str, tuple[str, int]]]]:
    """Partition existing runner budgets across fixed evaluation replica groups.

    Framework's runner semaphores remain the admission owner. A slow version
    cannot take another group's session slots or overload its own replica.
    """
    runners = OmegaConf.select(config, "actor_rollout_ref.rollout.custom.agent_framework.agent_runners")
    if len(groups) == 1 or not runners:
        return config, {}
    sizes = [len(servers) for servers in groups.values()]
    total = sum(sizes)
    scoped = {}
    assigned = {tuple(sorted(servers)): {} for servers in groups.values()}
    for name, runner in runners.items():
        limit = int(runner.get("max_concurrent_sessions", 0) or 0)
        limits = [limit * size // total for size in sizes]
        for index in sorted(range(len(sizes)), key=lambda i: -(limit * sizes[i] % total))[: limit - sum(limits)]:
            limits[index] += 1
        if limit > 0 and any(value == 0 for value in limits):
            raise ValueError(f"agent runner {name!r} session limit cannot cover every validation replica group")
        for index, servers in enumerate(groups.values()):
            scoped_name = f"{name}__validation_{index}"
            scoped[scoped_name] = {
                **OmegaConf.to_container(runner, resolve=True),
                "max_concurrent_sessions": limits[index],
            }
            assigned[tuple(sorted(servers))][name] = (scoped_name, limits[index])
    result = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    result.actor_rollout_ref.rollout.custom.agent_framework.agent_runners = scoped
    return result, assigned


def initialize_evaluation_runtime(config: DictConfig) -> EvaluationRuntime:
    """Start rollout services and load each replica group's bootstrap checkpoint.

    All replicas must initialize before the Router receives their handles.
    Subsequent waves reload these actors in place; their lifetime is this Ray job.
    """
    started = time.perf_counter()
    service_started_at = datetime.now(UTC).isoformat()
    rollout_cfg = omega_conf_to_dataclass(config.actor_rollout_ref.rollout)
    manager_fqn = rollout_cfg.agent.agent_loop_manager_class
    if not manager_fqn:
        raise ValueError("actor_rollout_ref.rollout.agent.agent_loop_manager_class is required for evaluation")
    manager_cls = load_class_from_fqn(str(manager_fqn), "AgentLoopManager")
    rollout_pool = RayResourcePool(
        process_on_nodes=[rollout_cfg.n_gpus_per_node] * rollout_cfg.nnodes,
        use_gpu=True,
        max_colocate_count=2,
        name_prefix="evaluation_rollout",
    )
    rollout_pool.get_placement_groups()
    world_size = (
        rollout_cfg.tensor_model_parallel_size
        * rollout_cfg.data_parallel_size
        * rollout_cfg.pipeline_model_parallel_size
    )
    subpools = split_resource_pool(rollout_pool, split_size=world_size)
    groups = validation_replica_groups([f"rollout_server_{index}" for index in range(len(subpools))], [1])
    admission_config, admission_runners = _admission_config(config, groups)
    controllers: dict[str, ActorHandle] = {}
    initial_versions = {server_id: version for version, server_ids in groups.items() for server_id in server_ids}
    init_refs = {}
    controller_cls = ray.remote(LLMServerController)
    for index, subpool in enumerate(subpools):
        server_id = f"rollout_server_{index}"
        version = initial_versions[server_id]
        model_cfg = OmegaConf.create(OmegaConf.to_container(config.actor_rollout_ref.model, resolve=True))
        model_cfg.path = str(config.validation.checkpoints[version - 1])
        assert subpool.pgs is not None
        pg = subpool.pgs[subpool.start_bundle_index // rollout_cfg.n_gpus_per_node]
        controller = controller_cls.options(
            max_concurrency=rollout_cfg.ray_actor_max_concurrency,
            name=server_id,
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg),
        ).remote(
            relay_client=None,
            engine=vLLMReplica(
                replica_rank=index,
                config=rollout_cfg,
                model_config=model_cfg,
                gpus_per_node=rollout_cfg.n_gpus_per_node,
            ),
            name=server_id,
        )
        controllers[server_id] = controller
        before = time.perf_counter()
        init_refs[controller.init_engine.remote(resource_pool=subpool, weight_version=version)] = (server_id, before)
    initial_load_timings = {}
    deadline = time.monotonic() + config.timeout.init_model
    while init_refs:
        done, _ = ray.wait(list(init_refs), num_returns=1, timeout=max(0.0, deadline - time.monotonic()))
        if not done:
            raise TimeoutError("evaluation rollout initialization timed out")
        ray.get(done[0])
        server_id, before = init_refs.pop(done[0])
        initial_load_timings[server_id] = time.perf_counter() - before
    handles = ray.get([controller.get_server_handle.remote() for controller in controllers.values()])
    router_config = omega_conf_to_dataclass(config.rollout_router)
    router = ray.remote(ValidationRequestRouter).remote(
        servers=dict(zip(controllers, handles, strict=True)),
        config=router_config,
    )
    ray.get(
        [
            router.mark_active.remote(server_id, weight_version=version)
            for server_id, version in initial_versions.items()
        ]
    )
    tracker = ray.remote(LLMServerStateTracker).remote()
    ray.get(tracker.connect.remote(servers=controllers, router=router))
    adapter = manager_cls.create(
        config=admission_config,
        llm_client=LLMServerClient(config=config, router=router),
        teacher_client=None,
        reward_loop_worker_handles=None,
    )
    return EvaluationRuntime(
        llm_server_controllers=controllers,
        llm_server_state_tracker=tracker,
        rollout_router=router,
        agent_loop_manager=adapter,
        admission_runners=admission_runners,
        initial_versions=initial_versions,
        initial_load_timings=initial_load_timings,
        service_started_at=service_started_at,
        initialization_s=time.perf_counter() - started,
    )
