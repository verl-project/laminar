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
"""Entry point for the async-RL training recipe.

Hydra ``main`` starts Ray and dispatches to ``main_task`` so initialization
lands on a worker node. ``main_task`` owns TransferQueue lifecycle and
constructs resource pools, Relay, LLM servers, router, dispatcher,
ExperienceBuffer, Tracker, and ``AsyncRLTrainer``.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import ray
from omegaconf import OmegaConf, open_dict
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from verl.single_controller.ray import RayWorkerGroup
from verl.single_controller.ray.base import split_resource_pool
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.profiler.performance import simple_timer
from verl.utils.transferqueue_utils import tq
from verl.workers.config.rollout import RolloutConfig
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

from laminar.global_request_router import (
    GlobalRequestRouter,
    GlobalRequestRouterConfig,
)
from laminar.llm_server.controller import LLMServerController
from laminar.llm_server.state_tracker import LLMServerStateTracker
from laminar.llm_server_client import LLMServerClient
from laminar.multi_version_rollout import (
    MultiVersionRolloutManager,
    MultiVersionRolloutManagerConfig,
)
from laminar.ray_runtime import ray_runtime_env_from_env
from laminar.train.dispatcher import RolloutDispatcher, RolloutDispatcherConfig
from laminar.train.experience_buffer.config import ExperienceBufferConfig
from laminar.train.experience_buffer.server import ExperienceBufferServer
from laminar.train.relay import RelayClient, RelayManager
from laminar.train.relay.transport import ucx_job_runtime_env_vars_from_env
from laminar.train.tracker import Tracker, TrackerConfig
from laminar.train.trainer import AsyncRLTrainer, AsyncRLTrainerConfig

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from ray.actor import ActorHandle

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _AsyncRLRuntime:
    """Long-lived Ray actors started by the async-RL entry."""

    trainer: ActorHandle
    rollout_dispatcher: ActorHandle
    llm_server_state_tracker: ActorHandle
    multi_version_rollout_manager: ActorHandle
    tracker: ActorHandle | None = None

    def run(self) -> None:
        """Run training and acknowledge backend finalization before actor teardown."""
        try:
            self._run_training()
            self._finish_tracker(exit_code=0)
        except BaseException:
            try:
                self._finish_tracker(exit_code=1)
            except BaseException:
                logger.exception("Tracker failure cleanup failed; preserving the original training exception")
            raise

    def _finish_tracker(self, exit_code: int) -> None:
        if self.tracker is not None:
            ray.get(self.tracker.finish.remote(exit_code=exit_code))  # type: ignore[attr-defined]

    def _run_training(self) -> None:
        # In-flight rollout work still belongs to Ray actor teardown. Tracking
        # is finalized explicitly after the trainer's last acknowledged commit.
        logger.info("Async-RL runtime start: llm_server_state_tracker.refresh_once")
        ray.get(self.llm_server_state_tracker.refresh_once.remote())  # type: ignore[attr-defined]
        logger.info("Async-RL runtime start: llm_server_state_tracker.start")
        ray.get(self.llm_server_state_tracker.start.remote())  # type: ignore[attr-defined]
        logger.info("Async-RL runtime start: multi_version_rollout_manager.start")
        ray.get(self.multi_version_rollout_manager.start.remote())  # type: ignore[attr-defined]
        logger.info("Async-RL runtime start: trainer.prepare")
        ray.get(self.trainer.prepare.remote())  # type: ignore[attr-defined]
        logger.info("Async-RL runtime start: rollout_dispatcher.start")
        ray.get(self.rollout_dispatcher.start.remote())  # type: ignore[attr-defined]

        logger.info("Async-RL runtime start: trainer.fit")
        trainer_fit_ref = self.trainer.fit.remote()  # type: ignore[attr-defined]
        control_loop_refs = {
            self.llm_server_state_tracker.wait_for_stop.remote(): "LLM server state tracker",  # type: ignore[attr-defined]
            self.rollout_dispatcher.wait_for_stop.remote(): "rollout dispatcher",  # type: ignore[attr-defined]
            self.multi_version_rollout_manager.wait_for_stop.remote(): "multi-version rollout manager",  # type: ignore[attr-defined]
        }
        done_refs, _ = ray.wait([trainer_fit_ref, *control_loop_refs], num_returns=1)
        done_ref = done_refs[0]
        ray.get(done_ref)
        if done_ref != trainer_fit_ref:
            raise RuntimeError(f"{control_loop_refs[done_ref]} stopped before trainer.fit completed")


@ray.remote
def main_task(config: DictConfig) -> None:
    # TransferQueue is owned by the worker that consumes it, not the Hydra
    # entry on the head node.
    tq.init(config.transfer_queue)
    try:
        runtime = _initialize_runtime(config)
        runtime.run()
    finally:
        tq.close()


def _resolve_agent_loop_manager_class(config: DictConfig) -> type:
    """Load the configured AgentLoopManager class.

    Unlike main_ppo, async-RL has no default manager: recipes must set
    ``actor_rollout_ref.rollout.agent.agent_loop_manager_class``.
    """
    manager_class_fqn = config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
    if not manager_class_fqn:
        raise ValueError(
            "actor_rollout_ref.rollout.agent.agent_loop_manager_class is required for async-RL rollout dispatch"
        )
    return load_class_from_fqn(str(manager_class_fqn), "AgentLoopManager")


def _initialize_runtime(config: DictConfig) -> _AsyncRLRuntime:
    """Create resource pools and long-lived Ray actors.

    Kept separate so ``main_task`` is just TransferQueue pairing around init + run.
    """
    init_timing_raw: dict[str, float] = {}
    init_start_time = time.perf_counter()

    with simple_timer("config", init_timing_raw):
        colocate = _validate_training_strategy(config)
        if config.trainer.total_epochs is not None:
            raise ValueError("async_rl classic rollout requires trainer.total_epochs=null; use total_training_steps")
        rollout_cfg: RolloutConfig = omega_conf_to_dataclass(config.actor_rollout_ref.rollout)
        buffer_config: ExperienceBufferConfig = omega_conf_to_dataclass(config.experience_buffer)
        agent_loop_manager_cls = _resolve_agent_loop_manager_class(config)

        tracker_cfg: TrackerConfig = TrackerConfig.from_config(config)
        rollout_dispatcher_config: RolloutDispatcherConfig = omega_conf_to_dataclass(config.rollout_dispatcher)
        rollout_router_config: GlobalRequestRouterConfig = omega_conf_to_dataclass(config.rollout_router)
        multi_version_rollout_config: MultiVersionRolloutManagerConfig = omega_conf_to_dataclass(
            config.multi_version_rollout
        )
        multi_version_rollout_config = replace(multi_version_rollout_config, colocate=colocate)
        trainer_config: AsyncRLTrainerConfig = AsyncRLTrainerConfig.from_config(config)
        # Keep raw Hydra config instead of HFModelConfig: the latter eagerly
        # loads tokenizer/processor/HF config in this process, while rollout
        # server actors must initialize those on their own nodes.
        model_cfg: DictConfig = config.actor_rollout_ref.model

    # One rollout pool for the total node shape; TP × DP × PP world size
    # decides how many sibling replica slices ``split_resource_pool`` carves.
    with simple_timer("resource_pools", init_timing_raw):
        resource_pool_spec: dict[str, list[int]] = {
            "trainer_pool": ([config.trainer.n_gpus_per_node] * config.trainer.nnodes),
        }
        if not colocate:
            resource_pool_spec["rollout_pool"] = [rollout_cfg.n_gpus_per_node] * rollout_cfg.nnodes
        # TODO(async_rl): add critic support.
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping={Role.Actor: "trainer_pool", Role.Rollout: "trainer_pool" if colocate else "rollout_pool"},  # type: ignore[arg-type]
        )
        resource_pool_manager.create_resource_pool()
        _create_resource_pool_placement_groups(resource_pool_manager)

    with simple_timer("tokenizer", init_timing_raw):
        from verl.utils import hf_tokenizer
        from verl.utils.fs import copy_to_local

        # Pickle the tokenizer into the dispatcher so it can build its colocated
        # DataLoader without a second copy_to_local / HFModelConfig load.
        local_model_path = copy_to_local(model_cfg.path, use_shm=model_cfg.use_shm)
        tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.data.trust_remote_code)

    with simple_timer("relay_manager", init_timing_raw):
        relay_num_cpus = OmegaConf.select(config, "relay.num_cpus", default=1)
        relay_manager: ActorHandle = ray.remote(RelayManager).remote(
            node_ids=_resource_pool_node_ids(resource_pool_manager.get_resource_pool(Role.Rollout)),
            num_cpus=int(relay_num_cpus),
        )
        relay_client = RelayClient(relay_manager)

    with simple_timer("experience_buffer", init_timing_raw):
        experience_buffer: ActorHandle = ray.remote(ExperienceBufferServer).remote(
            config=buffer_config,
        )
    with simple_timer("tracker", init_timing_raw):
        tracker: ActorHandle = (
            ray.remote(Tracker).options(name="tracker").remote(config=tracker_cfg)  # pyright: ignore[reportCallIssue]
        )

    init_model_refs: list[ray.ObjectRef] = []

    with simple_timer("llm_server_controllers", init_timing_raw):
        llm_server_controllers: dict[str, ActorHandle] = {}
        rollout_parent_pool = resource_pool_manager.get_resource_pool(Role.Rollout)
        replica_world_size = (
            rollout_cfg.tensor_model_parallel_size
            * rollout_cfg.data_parallel_size
            * rollout_cfg.pipeline_model_parallel_size
        )
        rollout_subpools = split_resource_pool(rollout_parent_pool, split_size=replica_world_size)
        llm_server_controller_cls = ray.remote(LLMServerController)

        for i, sub_pool in enumerate(rollout_subpools):
            # Schedule the controller on the first PG of this replica slice.
            # The replica worker group spans the whole slice; the actor occupies
            # one bundle.
            assert sub_pool.pgs is not None
            first_pg_idx = sub_pool.start_bundle_index // rollout_cfg.n_gpus_per_node
            pg = sub_pool.pgs[first_pg_idx]
            server_id = f"rollout_server_{i}"
            controller: ActorHandle = llm_server_controller_cls.options(
                max_concurrency=rollout_cfg.ray_actor_max_concurrency,
                name=server_id,
                scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg),
            ).remote(
                relay_client=relay_client,
                engine=vLLMReplica(
                    replica_rank=i,
                    config=rollout_cfg,
                    model_config=model_cfg,
                    gpus_per_node=rollout_cfg.n_gpus_per_node,
                ),
                name=server_id,
            )
            llm_server_controllers[server_id] = controller
            if colocate:
                init_model_refs.append(controller.init_engine.remote(resource_pool=sub_pool, colocate=True))
            else:
                init_model_refs.append(controller.init_engine.remote(resource_pool=sub_pool))

        if colocate:
            # The replicas sleep after initialization. Do not allocate the
            # trainer model/optimizer until every replica has released memory.
            ray.get(init_model_refs, timeout=config.timeout.init_model)
            init_model_refs.clear()

    with simple_timer("llm_server_state_tracker", init_timing_raw):
        llm_server_state_tracker: ActorHandle = ray.remote(  # type: ignore[attr-defined]
            LLMServerStateTracker
        ).remote()

    with simple_timer("multi_version_rollout_manager", init_timing_raw):
        multi_version_rollout_manager: ActorHandle = ray.remote(  # type: ignore[attr-defined]
            MultiVersionRolloutManager
        ).remote(config=multi_version_rollout_config)

    with simple_timer("trainer", init_timing_raw):
        trainer: ActorHandle = ray.remote(AsyncRLTrainer).remote(
            config=config,
            resource_pool_manager=resource_pool_manager,
            experience_buffer=experience_buffer,
            relay_client=relay_client,
            trainer_config=trainer_config,
            tracker=tracker,
            ray_worker_group_cls=RayWorkerGroup,
        )
        init_model_refs.append(  # type: ignore[attr-defined]
            trainer.init_workers.remote(llm_server_controllers=list(llm_server_controllers.values()))
        )

    with simple_timer("wait_init_model", init_timing_raw):
        ray.get(init_model_refs, timeout=config.timeout.init_model)

    with simple_timer("rollout_router", init_timing_raw):
        # Wait for init_engine so get_server_handle can expose the OpenAI-compatible server actor.
        server_handles: list[ActorHandle] = ray.get(
            [controller.get_server_handle.remote() for controller in llm_server_controllers.values()]  # type: ignore[attr-defined]
        )
        rollout_router: ActorHandle = ray.remote(GlobalRequestRouter).remote(
            servers=dict(zip(llm_server_controllers, server_handles, strict=True)),
            config=rollout_router_config,
        )
        ray.get(
            llm_server_state_tracker.connect.remote(  # type: ignore[attr-defined]
                servers=llm_server_controllers,
                router=rollout_router,
            )
        )
        ray.get(
            multi_version_rollout_manager.connect.remote(  # type: ignore[attr-defined]
                servers=llm_server_controllers,
                router=rollout_router,
                relay_client=relay_client,
            )
        )

    with simple_timer("rollout_dispatcher", init_timing_raw):
        agent_loop_manager = agent_loop_manager_cls.create(
            config=config,
            llm_client=LLMServerClient(config=config, router=rollout_router),
            teacher_client=None,
            reward_loop_worker_handles=None,
        )
        rollout_dispatcher: ActorHandle = ray.remote(  # type: ignore[attr-defined]
            RolloutDispatcher
        ).remote(
            config=rollout_dispatcher_config,
            data_config=config.data,
            tokenizer=tokenizer,
            agent_loop_manager=agent_loop_manager,
            relay_client=relay_client,
            rollout_router=rollout_router,
            llm_server_state_tracker=llm_server_state_tracker,
        )
        ray.get(
            trainer.connect_checkpoint_sources.remote(  # type: ignore[attr-defined]
                rollout_dispatcher=rollout_dispatcher,
            )
        )
        if colocate:
            ray.get(
                trainer.connect_colocate_rollout.remote(multi_version_rollout_manager=multi_version_rollout_manager)
            )
        ray.get(
            tracker.connect_metric_sources.remote(  # type: ignore[attr-defined]
                rollout_dispatcher=rollout_dispatcher,
                experience_buffer=experience_buffer,
                multi_version_rollout_manager=multi_version_rollout_manager,
                relay_client=relay_client,
            )
        )

    overall_s = time.perf_counter() - init_start_time
    stage_summary = ", ".join(f"{key}={value:.3f}s" for key, value in sorted(init_timing_raw.items()))
    logger.info("async RL initialization timing: overall=%.3fs, stages={%s}", overall_s, stage_summary)

    return _AsyncRLRuntime(
        trainer=trainer,
        rollout_dispatcher=rollout_dispatcher,
        llm_server_state_tracker=llm_server_state_tracker,
        multi_version_rollout_manager=multi_version_rollout_manager,
        tracker=tracker,
    )


def _if_colocate_resolver(colocate_value: Any, separate_value: Any, *, _root_: DictConfig) -> Any:
    """Select between colocated and decoupled defaults based on ``training_strategy``."""
    raw = colocate_value if _root_.get("training_strategy") == "colocate_async" else separate_value
    if isinstance(raw, str):
        selected = OmegaConf.select(_root_, raw)
        if selected is not None:
            return selected
    return raw


def _rollout_sessions_resolver(sessions_per_replica: int, *, _root_: DictConfig) -> int:
    """Scale concurrent agent sessions with the derived rollout replica count."""
    rollout = _root_.actor_rollout_ref.rollout
    pool_gpus = int(rollout.nnodes) * int(rollout.n_gpus_per_node)
    world_size = (
        int(rollout.tensor_model_parallel_size)
        * int(rollout.data_parallel_size)
        * int(rollout.pipeline_model_parallel_size)
    )
    if world_size <= 0 or pool_gpus <= 0 or pool_gpus % world_size != 0:
        raise ValueError(
            f"rollout pool GPUs ({pool_gpus}) must be positive and divisible by replica world size ({world_size})"
        )
    return (pool_gpus // world_size) * int(sessions_per_replica)


def _seq_token_cap_resolver(*, _root_: DictConfig) -> int:
    """Derive per-GPU dynamic batching token cap as ceil(max_seq_len / sequence_split)."""
    max_seq_len = int(_root_.data.max_prompt_length) + int(_root_.data.max_response_length)
    actor = _root_.actor_rollout_ref.actor
    if str(actor.get("strategy", "fsdp2")) == "megatron":
        seq_split = int(actor.megatron.context_parallel_size)
    else:
        seq_split = int(actor.get("ulysses_sequence_parallel_size", 1))
    if seq_split <= 0:
        raise ValueError("sequence/context parallel size must be positive")
    return (max_seq_len + seq_split - 1) // seq_split


OmegaConf.register_new_resolver("if_colocate", _if_colocate_resolver, replace=True)
OmegaConf.register_new_resolver("rollout_sessions", _rollout_sessions_resolver, replace=True)
OmegaConf.register_new_resolver("seq_token_cap", _seq_token_cap_resolver, replace=True)


def _validate_training_strategy(config: DictConfig) -> bool:
    strategy = str(config.get("training_strategy", "separate_async"))
    if strategy not in {"separate_async", "colocate_async"}:
        raise ValueError(f"unsupported training_strategy: {strategy!r}")
    if strategy == "separate_async":
        return False
    if config.rollout_router.continuation != "latest_version":
        raise ValueError("colocate_async requires rollout_router.continuation=latest_version")
    rollout = config.actor_rollout_ref.rollout
    if (config.trainer.nnodes, config.trainer.n_gpus_per_node) != (rollout.nnodes, rollout.n_gpus_per_node):
        raise ValueError("colocate_async requires identical trainer and rollout pool shapes")
    if not rollout.enable_sleep_mode or not rollout.free_cache_engine:
        raise ValueError("colocate_async requires rollout.enable_sleep_mode and rollout.free_cache_engine")
    actor = config.actor_rollout_ref.actor
    engine = actor.megatron if actor.strategy == "megatron" else actor.fsdp_config
    if not engine.param_offload or not engine.optimizer_offload:
        raise ValueError("colocate_async requires actor parameter and optimizer offload")
    return True


def resolve_train_config(config: DictConfig) -> DictConfig:
    """Resolve interpolations, apply engine defaults, and validate topology before starting Ray."""
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    if (
        config.get("model_engine") == "megatron" or actor.get("_target_") == "verl.workers.config.McoreActorConfig"
    ) and actor.get("strategy") in (None, "fsdp", "fsdp2"):
        actor.strategy = "megatron"
    strategy = str(actor.get("strategy", "fsdp2"))
    if strategy not in {"fsdp", "fsdp2", "megatron"}:
        raise ValueError(f"actor.strategy must be fsdp, fsdp2, or megatron, got {strategy!r}")
    if strategy != "megatron" and actor.get("megatron") is not None:
        with open_dict(actor):
            del actor["megatron"]

    OmegaConf.resolve(config)
    _validate_training_strategy(config)

    # Validate task recipe path early when the uni-agent task runner is configured.
    task_config_path = OmegaConf.select(
        config,
        "actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path",
    )
    if task_config_path is not None and (
        not isinstance(task_config_path, str)
        or not task_config_path
        or not Path(task_config_path).is_file()
        or not os.access(task_config_path, os.R_OK)
    ):
        raise ValueError("task_config_path must name an existing readable task recipe")

    relay_backend_cfg = OmegaConf.select(config, "relay.backend")
    if relay_backend_cfg is not None and str(relay_backend_cfg) not in {"ucx", "nccl"}:
        raise ValueError(f"relay.backend must be 'ucx' or 'nccl', got {relay_backend_cfg!r}")
    relay_num_cpus_cfg = OmegaConf.select(config, "relay.num_cpus")
    if relay_num_cpus_cfg is not None and (
        isinstance(relay_num_cpus_cfg, bool) or not isinstance(relay_num_cpus_cfg, int) or relay_num_cpus_cfg < 1
    ):
        raise ValueError("relay.num_cpus must be a positive integer")

    # Apply Megatron engine/optimizer defaults and validate Megatron-Bridge requirement.
    if strategy == "megatron":
        megatron = actor.megatron
        if megatron.get("use_mbridge") is not True:
            raise ValueError("Megatron requires USE_MBRIDGE=True (NVIDIA Megatron-Bridge)")
        megatron.use_remove_padding = bool(config.actor_rollout_ref.model.get("use_remove_padding", True))
        megatron.dtype = str(rollout.get("dtype", "bfloat16"))
        transformer_defaults = {
            "apply_rope_fusion": True,
            "masked_softmax_fusion": True,
            "bias_activation_fusion": True,
            "bias_dropout_fusion": True,
            "gradient_accumulation_fusion": True,
            "deallocate_pipeline_outputs": True,
            "persist_layer_norm": True,
            "recompute_method": "uniform",
            "recompute_granularity": "full",
            "recompute_num_layers": 1,
            "moe_grouped_gemm": True,
            "moe_permute_fusion": True,
            "moe_token_dispatcher_type": "alltoall",
            "moe_router_dtype": "fp32",
        }
        with open_dict(megatron.override_transformer_config):
            for key, value in transformer_defaults.items():
                if megatron.override_transformer_config.get(key) is None:
                    megatron.override_transformer_config[key] = value

        warmup_steps = actor.optim.get("lr_warmup_steps", -1)
        if warmup_steps is not None and int(warmup_steps) < 0:
            actor.optim.lr_warmup_steps = 0
        total_steps = config.trainer.get("total_training_steps")
        if int(actor.optim.get("total_training_steps", -1)) <= 0:
            actor.optim.total_training_steps = int(total_steps) if total_steps is not None else 1
        if int(actor.optim.total_training_steps) <= 0:
            raise ValueError("Megatron optimizer total training steps must be positive")
        if actor.optim.get("lr_decay_steps") is None:
            actor.optim.lr_decay_steps = int(actor.optim.total_training_steps)
        optim_override_defaults = {
            "optimizer_offload_fraction": 1.0,
            "overlap_cpu_optimizer_d2h_h2d": True,
            "use_precision_aware_optimizer": True,
            "optimizer_cpu_offload": False,
        }
        with open_dict(actor.optim.override_optimizer_config):
            for key, value in optim_override_defaults.items():
                if actor.optim.override_optimizer_config.get(key) is None:
                    actor.optim.override_optimizer_config[key] = value

    # Derive per-GPU dynamic batching token caps from sequence/context parallelism when unset.
    max_seq_len = int(config.data.max_prompt_length) + int(config.data.max_response_length)
    seq_split_size = (
        int(actor.megatron.context_parallel_size)
        if strategy == "megatron"
        else int(actor.get("ulysses_sequence_parallel_size", 1))
    )
    if seq_split_size <= 0:
        raise ValueError("sequence/context parallel size must be positive")
    derived_token_cap = (max_seq_len + seq_split_size - 1) // seq_split_size
    if actor.get("ppo_max_token_len_per_gpu") is None:
        actor.ppo_max_token_len_per_gpu = derived_token_cap
    if rollout.get("log_prob_max_token_len_per_gpu") is None:
        rollout.log_prob_max_token_len_per_gpu = int(actor.ppo_max_token_len_per_gpu)

    if strategy == "megatron" and bool(actor.get("use_dynamic_bsz", False)):
        actor_max_token_len = int(actor.ppo_max_token_len_per_gpu) * seq_split_size
        if actor_max_token_len < max_seq_len:
            raise ValueError(
                "ppo_max_token_len_per_gpu * context_parallel_size must cover max_seq_len "
                f"for Megatron dynamic batching: {actor_max_token_len} < {max_seq_len}"
            )
        infer_max_token_len = int(rollout.log_prob_max_token_len_per_gpu) * seq_split_size
        if infer_max_token_len < max_seq_len:
            raise ValueError(
                "log_prob_max_token_len_per_gpu * context_parallel_size must cover max_seq_len "
                f"for Megatron dynamic batching: {infer_max_token_len} < {max_seq_len}"
            )

    # Validate trainer and rollout pool geometry.
    trainer_nnodes = int(config.trainer.nnodes)
    trainer_gpus_per_node = int(config.trainer.n_gpus_per_node)
    if trainer_nnodes <= 0 or trainer_gpus_per_node <= 0:
        raise ValueError("trainer.nnodes and trainer.n_gpus_per_node must be positive")
    trainer_gpus = trainer_nnodes * trainer_gpus_per_node
    if strategy == "megatron":
        mp_size = (
            int(actor.megatron.tensor_model_parallel_size)
            * int(actor.megatron.pipeline_model_parallel_size)
            * int(actor.megatron.context_parallel_size)
        )
        if mp_size <= 0 or trainer_gpus % mp_size != 0:
            raise ValueError(
                f"trainer GPUs ({trainer_gpus}) must be divisible by TP * PP * CP ({mp_size}) for Megatron"
            )

    rollout_nnodes = int(rollout.nnodes)
    rollout_gpus_per_node = int(rollout.n_gpus_per_node)
    if rollout_nnodes <= 0 or rollout_gpus_per_node <= 0:
        raise ValueError("rollout.nnodes and rollout.n_gpus_per_node must be positive")
    rollout_pool_gpus = rollout_nnodes * rollout_gpus_per_node
    replica_world_size = (
        int(rollout.tensor_model_parallel_size)
        * int(rollout.data_parallel_size)
        * int(rollout.pipeline_model_parallel_size)
    )
    if replica_world_size <= 0 or rollout_pool_gpus % replica_world_size != 0:
        raise ValueError("rollout pool GPUs must be divisible by TP * DP * PP")
    if replica_world_size <= rollout_gpus_per_node:
        if rollout_gpus_per_node % replica_world_size != 0:
            raise ValueError("rollout TP * DP * PP must divide rollout.n_gpus_per_node")
    elif replica_world_size % rollout_gpus_per_node != 0:
        raise ValueError("rollout TP * DP * PP must be a multiple of rollout.n_gpus_per_node")

    # Size TransferQueue storage to cover ExperienceBuffer capacity with headroom.
    buffer_max_size = int(config.experience_buffer.max_size)
    if buffer_max_size <= 0:
        raise ValueError("experience_buffer.max_size must be positive")
    storage_cfg = config.transfer_queue.backend.SimpleStorage
    num_units = int(storage_cfg.get("num_data_storage_units", 8))
    min_tq_size = max(100000, buffer_max_size * num_units * 2)
    if storage_cfg.get("total_storage_size") is None or int(storage_cfg.total_storage_size) < min_tq_size:
        storage_cfg.total_storage_size = min_tq_size

    return config


def _create_resource_pool_placement_groups(resource_pool_manager: ResourcePoolManager) -> None:
    """Create placement groups largest-first so STRICT_PACK does not fragment an exactly-sized cluster."""
    resource_pools = list(resource_pool_manager.resource_pool_dict.values())
    resource_pools.sort(key=lambda rp: max(rp.store), reverse=True)
    for rp in resource_pools:
        rp.get_placement_groups()


def _resource_pool_node_ids(resource_pool: Any) -> list[str]:
    """Unique Ray node IDs from already-created placement groups."""
    node_ids: list[str] = []
    seen_node_ids: set[str] = set()
    for placement_group in resource_pool.get_placement_groups():
        bundles_to_node_id = ray.util.placement_group_table(placement_group).get("bundles_to_node_id")
        if not bundles_to_node_id:
            raise RuntimeError("placement group has no assigned node IDs")
        for node_id in bundles_to_node_id.values():
            if node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                node_ids.append(node_id)
    return node_ids


def _ray_init_kwargs_from_env(config: DictConfig | None = None) -> dict[str, Any]:
    """Build ``ray.init`` kwargs, optionally reusing an existing Ray head.

    Ray's ``RAY_ADDRESS`` reuses an already-created head. ``LAMINAR_RAY_ENV_VARS``
    supplies an explicit JSON object of additional worker environment variables.
    """
    os.environ.setdefault("RAY_OVERRIDE_JOB_RUNTIME_ENV", "1")
    env_vars = {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}
    if config is not None and (relay_backend_cfg := OmegaConf.select(config, "relay.backend")):
        env_vars["LAMINAR_RELAY_BACKEND"] = str(relay_backend_cfg)
    env_vars.update(ucx_job_runtime_env_vars_from_env())
    kwargs: dict[str, Any] = {"runtime_env": ray_runtime_env_from_env(env_vars)}
    address = os.environ.get("RAY_ADDRESS")
    if address:
        kwargs["address"] = address
    return kwargs


@hydra.main(config_path="config", config_name="async_rl", version_base=None)
def main(config) -> None:
    resolve_train_config(config)
    logger.info(f"Async RL trainer config: {config}")
    if not ray.is_initialized():
        ray.init(**_ray_init_kwargs_from_env(config))
    ray.get(main_task.remote(config))


if __name__ == "__main__":
    main()
