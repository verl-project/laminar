#!/usr/bin/env bash
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

# Qwen3-Coder-30B-A3B SWE-bench Verified evaluation service recipe.
#
# Default layout uses 8 GPUs (1 node x 8 GPUs, two TP4 vLLM replicas, up to 28
# concurrent agent sessions) and greedy decoding (do_sample=false, n=1, seed=42).
#
# Always starts the persistent evaluation queue service + rollout worker:
#   - Companion mode: watch training runs or checkpoint directories via WATCH_RUNS / WATCH_DIR.
#   - Offline mode: submit initial checkpoints via CHECKPOINT_PATH / CHECKPOINTS
#     at startup, and submit additional checkpoints anytime via:
#       bash recipes/swe/eval.sh --submit /path/to/checkpoint

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
PROJECT_NAME=${PROJECT_NAME:-"Laminar-SWE"}
VALIDATION_OUTPUT_DIR=${VALIDATION_OUTPUT_DIR:-"${RAY_DATA_HOME}/${PROJECT_NAME}/eval"}

# Forward client CLI control flags (--submit / --status / --drain / --requeue)
# directly to ``python3 -m laminar.eval``.
is_cli_control="False"
has_location_arg="False"
for arg in "$@"; do
    case "${arg}" in
        --submit|--submit=*|--status|--status=*|--drain|--requeue|--requeue=*)
            is_cli_control="True"
            ;;
        --endpoint|--endpoint=*|--state-dir|--state-dir=*|--result-dir|--result-dir=*)
            has_location_arg="True"
            ;;
    esac
done

if [ "${is_cli_control}" == "True" ]; then
    cli_args=()
    if [ "${has_location_arg}" == "False" ] && [ -n "${VALIDATION_OUTPUT_DIR:-}" ]; then
        cli_args+=(--result-dir "${VALIDATION_OUTPUT_DIR}")
    fi
    cli_args+=("$@")
    exec python3 -m laminar.eval "${cli_args[@]}"
fi

TASK_CONFIG_PATH=${TASK_CONFIG_PATH:-"${SCRIPT_DIR}/task_config.yaml"}
MODEL_PATH=${MODEL_PATH:-"${CHECKPOINT_PATH:-${RAY_DATA_HOME}/models/Qwen3-Coder-30B-A3B-Instruct}"}
VALIDATION_FILE=${VALIDATION_FILE:-"${RAY_DATA_HOME}/data/swe_agent/swe_bench_verified.parquet"}
PERSISTENT_OUTPUT_ROOT=${PERSISTENT_OUTPUT_ROOT:-"${RAY_DATA_HOME}"}
UNI_AGENT_LOG_DIR=${UNI_AGENT_LOG_DIR:-"${VALIDATION_OUTPUT_DIR}/trajectories"}

if [ -n "${CHECKPOINTS:-}" ]; then
    if [[ "${CHECKPOINTS}" == \[*\] ]]; then
        checkpoints_override="${CHECKPOINTS}"
    else
        checkpoints_override="[${CHECKPOINTS}]"
    fi
elif [ -n "${CHECKPOINT_PATH:-}" ]; then
    checkpoints_override="[${CHECKPOINT_PATH}]"
else
    checkpoints_override="[]"
fi

if [[ "${PERSISTENT_OUTPUT_ROOT}" == \[*\] ]]; then
    persistent_roots_override="${PERSISTENT_OUTPUT_ROOT}"
else
    persistent_roots_override="[${PERSISTENT_OUTPUT_ROOT}]"
fi

extra_args=()
if [ -n "${WATCH_RUNS:-}" ] && [ "${WATCH_RUNS}" != "null" ]; then
    if [[ "${WATCH_RUNS}" == \[*\] ]]; then
        extra_args+=(validation.queue.watch_runs="${WATCH_RUNS}")
    else
        extra_args+=(validation.queue.watch_runs="[${WATCH_RUNS}]")
    fi
fi
if [ -n "${WATCH_DIR:-}" ] && [ "${WATCH_DIR}" != "null" ]; then
    extra_args+=(validation.queue.watch_dir="${WATCH_DIR}")
fi

timestamp=$(date +%Y%m%d-%H%M%S)
log_home="${RAY_DATA_HOME}/${PROJECT_NAME}/logs/eval"
log_path="${log_home}/eval-${timestamp}.log"

cmd=(
    python3 -m laminar.eval
    validation.checkpoints="${checkpoints_override}"
    validation.data_path="${VALIDATION_FILE}"
    validation.result_dir="${VALIDATION_OUTPUT_DIR}"
    validation.persistent_roots="${persistent_roots_override}"
    validation.batch_size=500
    validation.stage_timeout_s=28800
    validation.queue.server_host=0.0.0.0
    validation.queue.server_port=9000
    data.max_prompt_length=4096
    data.max_response_length=131072
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.rollout.nnodes=1
    actor_rollout_ref.rollout.n_gpus_per_node=8
    actor_rollout_ref.rollout.tensor_model_parallel_size=4
    actor_rollout_ref.rollout.max_model_len=135168
    actor_rollout_ref.rollout.max_num_batched_tokens=16384
    actor_rollout_ref.rollout.enable_chunked_prefill=true
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9
    actor_rollout_ref.rollout.seed=42
    actor_rollout_ref.rollout.full_determinism=false
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=false
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
    actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=4
    ++actor_rollout_ref.rollout.custom.agent_framework.log_dir="${UNI_AGENT_LOG_DIR}"
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=28
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path="${TASK_CONFIG_PATH}"
    "${extra_args[@]}"
    "$@"
)

mkdir -p "${log_home}"
printf '%q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}" 2>&1 | tee "${log_path}"
