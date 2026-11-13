# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from hydra.utils import get_class
from omegaconf import OmegaConf
from verl.trainer.config import CheckpointConfig
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import McoreActorConfig

from laminar.train.main import resolve_train_config

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_CONFIG_DIR = str((_REPO_ROOT / "src/laminar/train/config").resolve())
_EVAL_CONFIG_DIR = str((_REPO_ROOT / "src/laminar/eval/config").resolve())


@pytest.fixture
def task_config_path(tmp_path):
    path = tmp_path / "task_config.yaml"
    path.write_text("{}\n")
    return path


def _compose_train(*overrides: str):
    with initialize_config_dir(config_dir=_TRAIN_CONFIG_DIR, version_base=None):
        config = compose(config_name="async_rl", overrides=list(overrides))
    return resolve_train_config(config)


def _capture_recipe_python_argv(
    script_path: Path,
    tmp_path: Path,
    *,
    env: dict[str, str] | None = None,
    cli_args: list[str] | None = None,
    cwd: Path | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok=True)
    argv_path = tmp_path / "python-argv.bin"
    python = fake_bin / "python3"
    python.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "${CAPTURED_ARGV_PATH}"\n')
    python.chmod(0o755)
    run_env = {
        **(env if env is not None else os.environ),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "CAPTURED_ARGV_PATH": str(argv_path),
    }
    result = subprocess.run(
        ["bash", str(script_path), *(cli_args or [])],
        cwd=cwd or _REPO_ROOT,
        check=False,
        env=run_env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if not argv_path.exists():
        return result, []
    encoded_argv = argv_path.read_bytes().split(b"\0")
    assert encoded_argv[-1] == b""
    return result, [part.decode() for part in encoded_argv[:-1]]


@pytest.mark.parametrize("recipe_path", ["", "missing.yaml", "."])
def test_train_config_requires_readable_task_recipe(tmp_path, recipe_path):
    target = str(tmp_path / recipe_path) if recipe_path else ""
    with pytest.raises(ValueError, match="task_config_path must name an existing readable task recipe"):
        _compose_train(
            f'++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path="{target}"'
        )


@pytest.mark.parametrize("backend", ["gloo", ""])
def test_train_config_rejects_invalid_relay_backend(backend):
    with pytest.raises(ValueError, match="relay.backend must be 'ucx' or 'nccl'"):
        _compose_train(f'relay.backend="{backend}"')


@pytest.mark.parametrize("num_cpus", [0, -2, True])
def test_train_config_rejects_invalid_relay_num_cpus(num_cpus):
    with pytest.raises(ValueError, match="relay.num_cpus must be a positive integer"):
        _compose_train(f"relay.num_cpus={num_cpus}")


def test_train_config_rejects_disabled_megatron_bridge():
    with pytest.raises(ValueError, match=r"Megatron requires USE_MBRIDGE=True \(NVIDIA Megatron-Bridge\)"):
        _compose_train(
            "model_engine=megatron",
            "actor_rollout_ref.actor.strategy=megatron",
            "actor_rollout_ref.actor.megatron.use_mbridge=False",
        )


@pytest.mark.parametrize("warmup_steps", [None, -1, 0, 5])
def test_train_config_preserves_megatron_warmup_ratio(warmup_steps):
    config = _compose_train(
        "model_engine=megatron",
        f"actor_rollout_ref.actor.optim.lr_warmup_steps={warmup_steps if warmup_steps is not None else 'null'}",
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1",
        "trainer.total_training_steps=100",
    )

    assert config.actor_rollout_ref.actor.optim.lr_warmup_steps == (0 if warmup_steps == -1 else warmup_steps)
    assert config.actor_rollout_ref.actor.optim.lr_warmup_steps_ratio == 0.1


@pytest.mark.parametrize("use_checkpoint_scheduler", [False, True])
def test_train_config_preserves_explicit_megatron_checkpoint_scheduler(use_checkpoint_scheduler):
    config = _compose_train(
        "model_engine=megatron",
        f"actor_rollout_ref.actor.optim.use_checkpoint_opt_param_scheduler={use_checkpoint_scheduler}",
    )

    assert config.actor_rollout_ref.actor.optim.use_checkpoint_opt_param_scheduler is use_checkpoint_scheduler


def test_train_config_maps_nondefault_algorithm_repack_and_rollout_shape():
    config = _compose_train(
        "algorithm.norm_adv_by_std_in_grpo=True",
        "rollout_dispatcher.schedule_interval_s=4.25",
        "multi_version_rollout.repack.source_max_kv_cache_usage=0.31",
        "multi_version_rollout.repack.destination_max_kv_cache_usage=0.87",
        "actor_rollout_ref.actor.loss_agg_mode=token-sum",
        "actor_rollout_ref.actor.loss_scale_factor=17.5",
        "actor_rollout_ref.actor.policy_loss.kl_cov_ratio=0.123",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.data_parallel_size=2",
        "actor_rollout_ref.rollout.pipeline_model_parallel_size=2",
        "trainer.n_gpus_per_node=2",
        "trainer.nnodes=2",
        "actor_rollout_ref.rollout.n_gpus_per_node=4",
        "actor_rollout_ref.rollout.nnodes=2",
    )

    expected = {
        "algorithm.norm_adv_by_std_in_grpo": True,
        "rollout_dispatcher.schedule_interval_s": 4.25,
        "multi_version_rollout.repack.source_max_kv_cache_usage": 0.31,
        "multi_version_rollout.repack.destination_max_kv_cache_usage": 0.87,
        "actor_rollout_ref.actor.loss_agg_mode": "token-sum",
        "actor_rollout_ref.actor.loss_scale_factor": 17.5,
        "actor_rollout_ref.actor.policy_loss.kl_cov_ratio": 0.123,
        "actor_rollout_ref.rollout.data_parallel_size": 2,
        "actor_rollout_ref.rollout.pipeline_model_parallel_size": 2,
        "trainer.n_gpus_per_node": 2,
        "trainer.nnodes": 2,
        "actor_rollout_ref.rollout.n_gpus_per_node": 4,
        "actor_rollout_ref.rollout.nnodes": 2,
    }
    assert {path: OmegaConf.select(config, path) for path in expected} == expected


@pytest.mark.parametrize("root_var", ["PERSISTENT_OUTPUT_ROOT", "PERSISTENT_JSONL_OUTPUT_ROOTS"])
@pytest.mark.parametrize("config_key_limit", [None, 64])
def test_swe_recipe_forwards_tracker_dump_overrides(tmp_path, task_config_path, root_var, config_key_limit):
    env = os.environ.copy()
    for name in ("RAY_RUNTIME_ENV", "PERSISTENT_OUTPUT_ROOT", "PERSISTENT_JSONL_OUTPUT_ROOTS"):
        env.pop(name, None)
    env.update(
        {
            "EXP_NAME": "test-persistent-jsonl-root",
            "RAY_DATA_HOME": str(tmp_path),
            root_var: str(tmp_path),
            "TASK_CONFIG_PATH": str(task_config_path),
        }
    )
    cli_args = []
    if config_key_limit is not None:
        cli_args.append(f"tracker.config_key_max_bytes={config_key_limit}")

    result, argv = _capture_recipe_python_argv(
        _REPO_ROOT / "recipes/swe/train.sh", tmp_path, env=env, cli_args=cli_args
    )

    assert result.returncode == 0
    config = _compose_train(*argv[2:])
    assert config.tracker.dump.require_persistent_path is True
    assert config.tracker.dump.persistent_roots == str(tmp_path)
    assert list(config.tracker.logger) == []
    assert config.tracker.config_key_max_bytes == config_key_limit


def test_swe_recipe_serializes_agent_log_dir_for_ray_workers(tmp_path, task_config_path, monkeypatch):
    log_dir = str(tmp_path / "durable-agent-logs")
    env = {
        **os.environ,
        "TASK_CONFIG_PATH": str(task_config_path),
        "EXP_NAME": "test-agent-log-dir",
        "RAY_DATA_HOME": str(tmp_path),
        "UNI_AGENT_LOG_DIR": log_dir,
    }
    result, argv = _capture_recipe_python_argv(_REPO_ROOT / "recipes/swe/train.sh", tmp_path, env=env)
    assert result.returncode == 0
    assert argv[:2] == ["-m", "laminar.train"]
    config = _compose_train(*argv[2:])

    from uni_agent.framework.framework import GatewayAgentFramework

    monkeypatch.delenv("UNI_AGENT_LOG_DIR", raising=False)
    worker_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    framework = GatewayAgentFramework.from_config(config=worker_config, gateway_manager=object())
    assert framework._log_dir == log_dir


def test_train_config_preserves_explicit_fsdp_optimizer_offload():
    for train_engine in ("fsdp", "fsdp2"):
        for optim_offload in (True, False):
            config = _compose_train(
                f"actor_rollout_ref.actor.strategy={train_engine}",
                f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={optim_offload}",
            )
            assert config.actor_rollout_ref.actor.strategy == train_engine
            assert config.actor_rollout_ref.actor.fsdp_config.optimizer_offload is optim_offload


def test_swe_recipe_expands_public_agent_framework_payload_with_fsdp2_override(tmp_path, task_config_path):
    env = os.environ.copy()
    env.pop("RAY_RUNTIME_ENV", None)
    env.update(
        {
            "TASK_CONFIG_PATH": str(task_config_path),
            "RAY_DATA_HOME": str(tmp_path),
        }
    )
    cli_args = [
        "model_engine=dp",
        "actor_rollout_ref.actor.strategy=fsdp2",
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size=2",
        "actor_rollout_ref.actor.fsdp_config.fsdp_size=2",
        "trainer.n_gpus_per_node=2",
        "trainer.nnodes=1",
        "actor_rollout_ref.rollout.n_gpus_per_node=4",
        "actor_rollout_ref.rollout.nnodes=1",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
        "data.train_batch_size=2",
        "actor_rollout_ref.actor.ppo_mini_batch_size=2",
        "actor_rollout_ref.rollout.max_num_seqs=8",
        "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=336",
        "+actor_rollout_ref.rollout.server.max_attempts=13",
        "+actor_rollout_ref.rollout.server.retry_delay=4.0",
    ]

    result, argv = _capture_recipe_python_argv(
        _REPO_ROOT / "recipes/swe/train.sh", tmp_path, env=env, cli_args=cli_args
    )

    assert result.returncode == 0
    config = _compose_train(*argv[2:])
    assert config.trainer.project_name == "Laminar-SWE"
    assert config.data.train_files.endswith("data/swe_agent/swe_rebench_filtered.parquet")
    assert list(config.data.val_files) == []
    runner = config.actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task
    assert runner.runner_kwargs.task_config_path == str(task_config_path)
    assert runner.max_concurrent_sessions == 336
    assert runner.runner_fqn == "uni_agent.framework.task_runner.run_task"
    assert config.actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker is False
    assert config.actor_rollout_ref.model.path.endswith("Qwen3-Coder-30B-A3B-Instruct")
    assert config.data.max_prompt_length == 8192
    assert config.data.max_response_length == 131072
    assert config.actor_rollout_ref.rollout.n == 8
    assert config.actor_rollout_ref.rollout.tensor_model_parallel_size == 2
    assert config.actor_rollout_ref.actor.ppo_epochs == 1
    assert config.actor_rollout_ref.rollout.server.max_attempts == 13
    assert config.actor_rollout_ref.rollout.server.retry_delay == 4.0
    assert config.actor_rollout_ref.actor.ppo_mini_batch_size == 2
    assert config.actor_rollout_ref.actor.ulysses_sequence_parallel_size == 2
    assert config.actor_rollout_ref.actor.fsdp_config.param_offload is False
    assert config.actor_rollout_ref.actor.fsdp_config.optimizer_offload is False
    assert config.actor_rollout_ref.model.use_fused_kernels is True
    assert config.actor_rollout_ref.model.enable_activation_offload is True
    assert config.actor_rollout_ref.model.enable_gradient_checkpointing is True
    assert config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu == 69632
    assert config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu == 69632
    assert config.algorithm.adv_estimator == "grpo"
    assert config.actor_rollout_ref.actor.policy_loss.loss_mode == "gspo"
    assert config.algorithm.rollout_correction.rollout_is == "token"
    assert config.algorithm.rollout_correction.rollout_is_threshold == 2.0
    assert config.algorithm.rollout_correction.rollout_rs is None
    assert config.algorithm.rollout_correction.rollout_rs_threshold is None
    assert config.actor_rollout_ref.rollout.calculate_log_probs is True
    assert (
        config.actor_rollout_ref.rollout.agent.agent_loop_manager_class
        == "uni_agent.framework.entry.AgentFrameworkRolloutAdapter"
    )
    assert config.actor_rollout_ref.rollout.multi_turn.format == "qwen3_coder"
    assert config.trainer.weight_publish_interval == 4
    assert config.multi_version_rollout.repack.enable is True
    assert config.rollout_router.dynamic_overload_admission.mode == "enforce"
    assert config.rollout_router.dynamic_overload_admission.max_snapshot_age_s == 5.0
    assert config.rollout_router.dynamic_overload_admission.kv_cache_usage_threshold == 0.80
    assert config.rollout_router.continuation == "same_version"
    assert "" not in argv


@pytest.mark.parametrize("use_dist_ckpt", [False, True])
def test_train_30b_swe_megatron_expands_tuned_16_train_48_rollout_payload(tmp_path, task_config_path, use_dist_ckpt):
    env = os.environ.copy()
    env.pop("RAY_RUNTIME_ENV", None)
    env.update(
        {
            "TASK_CONFIG_PATH": str(task_config_path),
            "RAY_DATA_HOME": str(tmp_path),
        }
    )
    cli_args = [
        "trainer.total_training_steps=3",
        f"actor_rollout_ref.actor.megatron.use_dist_checkpointing={use_dist_ckpt}",
    ]

    result, argv = _capture_recipe_python_argv(
        _REPO_ROOT / "recipes/swe/train.sh", tmp_path, env=env, cli_args=cli_args
    )

    assert result.returncode == 0
    config = _compose_train(*argv[2:])
    assert config.actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions == 336
    assert config.rollout_router.dynamic_overload_admission.mode == "enforce"
    assert config.rollout_router.dynamic_overload_admission.kv_cache_usage_threshold == 0.80
    assert config.actor_rollout_ref.actor.optim.total_training_steps == 3
    assert config.actor_rollout_ref.actor.optim.use_checkpoint_opt_param_scheduler is True
    assert config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu == 17408
    assert config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu == 17408
    assert config.actor_rollout_ref.rollout.tensor_model_parallel_size == 2
    assert config.trainer.nnodes == 2
    assert config.actor_rollout_ref.rollout.nnodes == 6
    assert config.actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion is True
    assert config.actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm is True

    actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
    assert isinstance(actor_config, McoreActorConfig)
    assert isinstance(actor_config.checkpoint, CheckpointConfig)
    assert actor_config.checkpoint.save_contents == ["model", "optimizer", "extra"]
    assert actor_config.checkpoint.load_contents == ["model", "optimizer", "extra"]
    assert actor_config.checkpoint.async_save is False
    assert actor_config.megatron.use_mbridge is True
    assert actor_config.megatron.use_dist_checkpointing is use_dist_ckpt
    extension_cls = get_class(config.actor_rollout_ref.rollout.engine_kwargs.vllm.worker_extension_cls)
    assert callable(extension_cls.load_weights_from_shm_state_dict)


def test_train_30b_swe_megatron_scopes_cuda_allocator_to_trainer_workers(tmp_path, task_config_path):
    env = os.environ.copy()
    for name in ("PYTORCH_CUDA_ALLOC_CONF", "RAY_RUNTIME_ENV"):
        env.pop(name, None)
    env.update(
        {
            "TASK_CONFIG_PATH": str(task_config_path),
            "RAY_DATA_HOME": str(tmp_path),
        }
    )

    result, argv = _capture_recipe_python_argv(_REPO_ROOT / "recipes/swe/train.sh", tmp_path, env=env)

    assert result.returncode == 0
    allocator_override = "+trainer.worker_env.PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    assert allocator_override in argv
    with initialize_config_dir(config_dir=_TRAIN_CONFIG_DIR, version_base=None):
        with pytest.raises(ConfigCompositionException, match="Could not override"):
            compose(config_name="async_rl", overrides=[allocator_override.removeprefix("+")])
        config = compose(config_name="async_rl", overrides=[allocator_override])
    assert config.trainer.worker_env == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


@pytest.mark.parametrize("train_engine", ["fsdp2", "megatron"])
def test_colocate_swe_rendered_payload_composes_to_one_shared_pool(tmp_path, task_config_path, train_engine):
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ["HOME"],
        "EXP_NAME": "test-colocate-compose",
        "RAY_DATA_HOME": str(tmp_path),
        "TASK_CONFIG_PATH": str(task_config_path),
    }
    cli_args = ["training_strategy=colocate_async"]
    if train_engine == "fsdp2":
        cli_args.extend(
            [
                "model_engine=dp",
                "actor_rollout_ref.actor.strategy=fsdp2",
                "actor_rollout_ref.actor.fsdp_config.param_offload=True",
                "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
            ]
        )
    result, argv = _capture_recipe_python_argv(
        _REPO_ROOT / "recipes/swe/train.sh", tmp_path, env=env, cli_args=cli_args
    )
    assert result.returncode == 0
    config = _compose_train(*argv[2:])
    assert config.training_strategy == "colocate_async"
    assert (config.trainer.nnodes, config.trainer.n_gpus_per_node) == (2, 8)
    rollout = config.actor_rollout_ref.rollout
    assert (rollout.nnodes, rollout.n_gpus_per_node) == (2, 8)
    assert rollout.enable_sleep_mode and rollout.free_cache_engine
    actor = config.actor_rollout_ref.actor
    engine = actor.megatron if train_engine == "megatron" else actor.fsdp_config
    assert engine.param_offload and engine.optimizer_offload
    assert config.rollout_router.continuation == "latest_version"


@pytest.mark.parametrize(
    "cap_field",
    [
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu",
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu",
    ],
)
def test_train_30b_swe_megatron_rejects_cp_token_cap_below_max_sequence(tmp_path, cap_field, task_config_path):
    env = os.environ.copy()
    env.pop("RAY_RUNTIME_ENV", None)
    env.update(
        {
            "TASK_CONFIG_PATH": str(task_config_path),
            "RAY_DATA_HOME": str(tmp_path),
        }
    )
    cli_args = [
        "actor_rollout_ref.actor.megatron.context_parallel_size=2",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=69632",
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=69632",
        f"{cap_field}=57344",
    ]
    result, argv = _capture_recipe_python_argv(
        _REPO_ROOT / "recipes/swe/train.sh", tmp_path, env=env, cli_args=cli_args
    )
    assert result.returncode == 0
    with pytest.raises(ValueError, match=r"context_parallel_size must cover max_seq_len.*114688 < 139264"):
        _compose_train(*argv[2:])


def test_swe_recipe_train_script_runs_from_arbitrary_cwd_with_default_task_config(tmp_path):
    script_path = _REPO_ROOT / "recipes" / "swe" / "train.sh"
    default_task_config = _REPO_ROOT / "recipes" / "swe" / "task_config.yaml"
    env = os.environ.copy()
    for name in (
        "RAY_RUNTIME_ENV",
        "TASK_CONFIG_PATH",
        "PERSISTENT_OUTPUT_ROOT",
        "PERSISTENT_JSONL_OUTPUT_ROOTS",
    ):
        env.pop(name, None)
    env["RAY_DATA_HOME"] = str(tmp_path)

    result, argv = _capture_recipe_python_argv(script_path, tmp_path, env=env, cwd=tmp_path)

    assert result.returncode == 0
    config = _compose_train(*argv[2:])
    assert (
        config.actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path
        == str(default_task_config)
    )
    assert config.tracker.dump.persistent_roots == str(tmp_path)
    assert config.trainer.total_training_steps == 625
    assert config.trainer.save_freq == 40
    assert config.trainer.max_actor_ckpt_to_keep == 0


def test_swe_recipe_eval_script_supports_companion_offline_and_cli_control(tmp_path):
    eval_script = _REPO_ROOT / "recipes" / "swe" / "eval.sh"
    default_task_config = _REPO_ROOT / "recipes" / "swe" / "task_config.yaml"
    base_env = os.environ.copy()
    for name in (
        "RAY_RUNTIME_ENV",
        "TASK_CONFIG_PATH",
        "CHECKPOINT_PATH",
        "CHECKPOINTS",
        "WATCH_DIR",
        "VALIDATION_OUTPUT_DIR",
    ):
        base_env.pop(name, None)

    # 1. Companion watch mode (no initial checkpoints required; uses MODEL_PATH and WATCH_DIR).
    companion_env = {
        **base_env,
        "RAY_DATA_HOME": str(tmp_path),
        "WATCH_DIR": str(tmp_path / "checkpoints"),
    }
    companion, companion_argv = _capture_recipe_python_argv(eval_script, tmp_path, env=companion_env, cwd=tmp_path)
    assert companion.returncode == 0
    assert companion_argv[:2] == ["-m", "laminar.eval"]
    with initialize_config_dir(config_dir=_EVAL_CONFIG_DIR, version_base=None):
        companion_cfg = compose(config_name="validation", overrides=companion_argv[2:])
    OmegaConf.resolve(companion_cfg)
    assert list(companion_cfg.validation.checkpoints) == []
    assert companion_cfg.validation.queue.watch_dir == str(tmp_path / "checkpoints")
    assert companion_cfg.validation.queue.server_port == 9000
    assert (
        companion_cfg.actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path
        == str(default_task_config)
    )

    # 2. Offline initial submission mode + Hydra config composition check.
    offline_env = {
        **base_env,
        "RAY_DATA_HOME": str(tmp_path),
        "CHECKPOINT_PATH": "/shared/ckpt/global_step_400/actor/model/huggingface",
    }
    offline, offline_argv = _capture_recipe_python_argv(eval_script, tmp_path, env=offline_env, cwd=tmp_path)
    assert offline.returncode == 0
    assert offline_argv[:2] == ["-m", "laminar.eval"]
    with initialize_config_dir(config_dir=_EVAL_CONFIG_DIR, version_base=None):
        composed = compose(config_name="validation", overrides=offline_argv[2:])
    OmegaConf.resolve(composed)
    assert list(composed.validation.checkpoints) == ["/shared/ckpt/global_step_400/actor/model/huggingface"]
    assert composed.validation.queue.server_port == 9000
    assert composed.actor_rollout_ref.rollout.tensor_model_parallel_size == 4
    assert composed.actor_rollout_ref.rollout.val_kwargs.do_sample is False

    # 3. Subsequent CLI submission / status / drain forwarding.
    cli_submit, cli_argv = _capture_recipe_python_argv(
        eval_script,
        tmp_path,
        env={**base_env, "RAY_DATA_HOME": str(tmp_path)},
        cli_args=["--submit", "/shared/ckpt/global_step_440"],
        cwd=tmp_path,
    )
    assert cli_submit.returncode == 0
    assert "--result-dir" in cli_argv
    assert "--submit" in cli_argv
    assert "/shared/ckpt/global_step_440" in cli_argv
