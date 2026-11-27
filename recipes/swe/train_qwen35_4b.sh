#!/usr/bin/env bash
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

# Qwen3.5-4B SWE: 32 Megatron Actor/Ref + 48 vLLM GPUs.
# Requires the reference-policy support in verl-project/laminar#1.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_ID=${MODEL_ID:-Qwen3.5-4B}
export MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME:-${HOME}/verl}/models/Qwen3.5-4B"}
export EXP_NAME=${EXP_NAME:-qwen35-4b-swe}
export TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL=0
export LAMINAR_RAY_ENV_VARS="$(
    python3 -c 'import json, os; env = json.loads(os.environ.get("LAMINAR_RAY_ENV_VARS", "{}")); env["TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL"] = "0"; print(json.dumps(env))'
)"

exec bash "${SCRIPT_DIR}/train.sh" \
    model_engine=megatron \
    data.train_batch_size=32 \
    data.gen_batch_size=1 \
    data.max_prompt_length=8192 \
    data.max_response_length=131072 \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.actor.strategy=megatron \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=34816 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    "actor_rollout_ref.actor.optim.betas=[0.9,0.98]" \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.lr_decay_style=constant \
    ++actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    ++actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0 \
    ++actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    ++actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.loss_scale_factor=1.0 \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.shuffle=False \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    ++actor_rollout_ref.actor.megatron.param_offload=True \
    ++actor_rollout_ref.actor.megatron.optimizer_offload=True \
    ++actor_rollout_ref.actor.megatron.context_parallel_size=4 \
    ++actor_rollout_ref.actor.megatron.expert_model_parallel_size=1 \
    ++actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=flash \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    ++actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.ref.strategy=megatron \
    actor_rollout_ref.ref.use_torch_compile=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=34816 \
    ++actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1 \
    ++actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1 \
    ++actor_rollout_ref.ref.megatron.context_parallel_size=4 \
    ++actor_rollout_ref.ref.megatron.expert_model_parallel_size=1 \
    ++actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1 \
    ++actor_rollout_ref.ref.megatron.param_offload=True \
    ++actor_rollout_ref.ref.megatron.optimizer_offload=False \
    ++actor_rollout_ref.ref.megatron.forward_only=True \
    ++actor_rollout_ref.ref.megatron.use_mbridge=True \
    ++actor_rollout_ref.ref.megatron.use_remove_padding=True \
    ++actor_rollout_ref.ref.megatron.use_dist_checkpointing=False \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.nnodes=6 \
    actor_rollout_ref.rollout.n_gpus_per_node=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_seqs=1024 \
    actor_rollout_ref.rollout.max_model_len=139264 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=34816 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_cache_mode=align \
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=48 \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=0 \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.rollout_correction.rollout_is_batch_normalize=False \
    algorithm.rollout_correction.rollout_rs=null \
    ++algorithm.filter_groups.enable=True \
    ++algorithm.filter_groups.metric=reward \
    ++algorithm.filter_groups.max_num_gen_batches=0 \
    ++algorithm.filter_groups.max_inflight_gen_batches=1 \
    experience_buffer.max_size=4096 \
    trainer.nnodes=4 \
    trainer.n_gpus_per_node=8 \
    trainer.total_training_steps=160 \
    trainer.weight_publish_interval=1 \
    trainer.save_freq=10 \
    trainer.max_actor_ckpt_to_keep=0 \
    trainer.resume_mode=disable \
    "$@"
