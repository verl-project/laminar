#!/usr/bin/env bash
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

# Qwen3-Coder-30B-A3B SWE agent-framework async-RL recipe.
#
# Default layout targets 64x H800 (16 trainer + 48 rollout GPUs, rollout
# TP2 / 24 replicas, 336 concurrent sessions): Megatron TP1/PP1/CP8/EP8/ETP1/DP2,
# full recompute, flash attention, fused kernels, activation offload on, and
# train/infer token cap = ceil(max_seq_len / sequence_split) derived by
# ``laminar.train``. Separate pools keep parameters and the optimizer on GPU;
# colocated mode (``training_strategy=colocate_async``) shares the trainer pool
# and offloads both during rollout handoff.
#
# Path/cluster environment variables:
#   MODEL_PATH, TRAIN_FILE, RAY_DATA_HOME, PERSISTENT_OUTPUT_ROOT
#   (or PERSISTENT_JSONL_OUTPUT_ROOTS), TASK_CONFIG_PATH, UNI_AGENT_LOG_DIR, EXP_NAME.
# All model, algorithm, and resource knobs can be overridden via Hydra CLI
# arguments passed to this script (e.g. ``training_strategy=colocate_async`` or
# ``trainer.nnodes=4 actor_rollout_ref.rollout.nnodes=4 relay.num_cpus=4`` for 64x H20).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
PROJECT_NAME=${PROJECT_NAME:-"Laminar-SWE"}
MODEL_ID=${MODEL_ID:-"qwen3-coder-30b-a3b-instruct"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-Coder-30B-A3B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/swe_agent/swe_rebench_filtered.parquet"}
TASK_CONFIG_PATH=${TASK_CONFIG_PATH:-"${SCRIPT_DIR}/task_config.yaml"}
PERSISTENT_OUTPUT_ROOT=${PERSISTENT_OUTPUT_ROOT:-"${RAY_DATA_HOME}"}
PERSISTENT_JSONL_OUTPUT_ROOTS=${PERSISTENT_JSONL_OUTPUT_ROOTS:-"${PERSISTENT_OUTPUT_ROOT}"}

timestamp=$(date +%Y%m%d-%H%M%S)
commit_full=$(git rev-parse HEAD 2>/dev/null || true)
commit_id="${commit_full:0:8}"
EXP_NAME=${EXP_NAME:-"${MODEL_ID}-swe-agentloop"}
run_name="${EXP_NAME}-${commit_id:-nogit}-${timestamp}"
log_home="${RAY_DATA_HOME}/${PROJECT_NAME}/logs/train"
log_path="${log_home}/${run_name}.log"

extra_args=()
if [ -n "${UNI_AGENT_LOG_DIR:-}" ]; then
    extra_args+=(++actor_rollout_ref.rollout.custom.agent_framework.log_dir="${UNI_AGENT_LOG_DIR}")
fi

cmd=(
    python3 -m laminar.train
    model_engine=megatron
    training_strategy=separate_async
    data.train_files="${TRAIN_FILE}"
    data.val_files=[]
    data.max_prompt_length=8192
    data.max_response_length=131072
    data.train_batch_size=16
    data.gen_batch_size=1
    data.filter_overlong_prompts=True
    data.truncation=error
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.enable_activation_offload=True
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.model.fused_kernel_options.impl_backend=triton
    actor_rollout_ref.actor.strategy=megatron
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.use_dynamic_bsz=True
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=\${seq_token_cap:}"
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.clip_ratio_high=4e-4
    actor_rollout_ref.actor.clip_ratio_low=4e-4
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.loss_scale_factor=1.0
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo
    "++actor_rollout_ref.actor.megatron.param_offload=\${if_colocate:True,False}"
    "++actor_rollout_ref.actor.megatron.optimizer_offload=\${if_colocate:True,False}"
    ++actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1
    ++actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
    ++actor_rollout_ref.actor.megatron.context_parallel_size=8
    ++actor_rollout_ref.actor.megatron.expert_model_parallel_size=8
    ++actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.max_num_seqs=1024
    actor_rollout_ref.rollout.max_model_len=139264
    actor_rollout_ref.rollout.max_num_batched_tokens=139264
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=\${seq_token_cap:}"
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8
    actor_rollout_ref.rollout.tensor_model_parallel_size=2
    "actor_rollout_ref.rollout.nnodes=\${if_colocate:trainer.nnodes,6}"
    "actor_rollout_ref.rollout.n_gpus_per_node=\${if_colocate:trainer.n_gpus_per_node,8}"
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    "actor_rollout_ref.rollout.free_cache_engine=\${if_colocate:True,False}"
    actor_rollout_ref.rollout.enable_sleep_mode=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=8
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path="${TASK_CONFIG_PATH}"
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name="${MODEL_ID}"
    "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=\${rollout_sessions:14}"
    ++actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker=False
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=True
    algorithm.rollout_correction.rollout_is=token
    experience_buffer.max_size=8192
    rollout_dispatcher.schedule_interval_s=1
    rollout_router.dynamic_overload_admission.mode=enforce
    rollout_router.dynamic_overload_admission.kv_cache_usage_threshold=0.80
    "rollout_router.continuation=\${if_colocate:latest_version,same_version}"
    "multi_version_rollout.repack.enable=\${if_colocate:False,True}"
    trainer.nnodes=2
    trainer.n_gpus_per_node=8
    trainer.total_training_steps=625
    trainer.save_freq=40
    trainer.max_actor_ckpt_to_keep=0
    trainer.resume_mode=disable
    trainer.weight_publish_interval=4
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${run_name}"
    "trainer.logger=['console']"
    trainer.default_local_dir="${RAY_DATA_HOME}/${PROJECT_NAME}/checkpoints/${EXP_NAME}"
    +trainer.worker_env.PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    "tracker.logger=[]"
    tracker.dump.path="${RAY_DATA_HOME}/${PROJECT_NAME}/metrics/metrics-${run_name}.jsonl"
    tracker.dump.require_persistent_path=True
    tracker.dump.persistent_roots="${PERSISTENT_JSONL_OUTPUT_ROOTS}"
    timeout.init_model=3600
    "${extra_args[@]}"
    "$@"
)

mkdir -p "${log_home}"
printf '%q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}" 2>&1 | tee "${log_path}"
