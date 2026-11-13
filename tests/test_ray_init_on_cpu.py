# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Ray driver configuration without backend worker-class fixtures."""

import json
import os
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from laminar.eval import main as validation
from laminar.train import main as training
from laminar.train.main import _ray_init_kwargs_from_env


def test_ray_init_kwargs_reuses_ray_jobs_address(monkeypatch):
    monkeypatch.delenv("RAY_OVERRIDE_JOB_RUNTIME_ENV", raising=False)
    monkeypatch.setenv("RAY_ADDRESS", "192.0.2.1:6379")

    kwargs = _ray_init_kwargs_from_env()

    assert kwargs["address"] == "192.0.2.1:6379"
    assert os.environ["RAY_OVERRIDE_JOB_RUNTIME_ENV"] == "1"
    assert kwargs["runtime_env"]["env_vars"]["TOKENIZERS_PARALLELISM"] == "true"
    assert kwargs["runtime_env"]["env_vars"]["NCCL_DEBUG"] == "WARN"


def test_ray_init_kwargs_propagates_driver_ucx_env(monkeypatch):
    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.setenv("UCX_TLS", "shm,tcp")
    monkeypatch.setenv("LAMINAR_RELAY_UCX_TCP_FALLBACK", "1")

    kwargs = _ray_init_kwargs_from_env()

    env_vars = kwargs["runtime_env"]["env_vars"]
    assert env_vars["UCX_TLS"] == "shm,tcp"
    assert env_vars["LAMINAR_RELAY_UCX_TCP_FALLBACK"] == "1"


def test_ray_init_kwargs_propagates_relay_backend_from_config(monkeypatch):
    monkeypatch.delenv("LAMINAR_RELAY_BACKEND", raising=False)
    config = OmegaConf.create({"relay": {"backend": "nccl"}})

    env_vars = _ray_init_kwargs_from_env(config)["runtime_env"]["env_vars"]

    assert env_vars["LAMINAR_RELAY_BACKEND"] == "nccl"


def test_ray_init_kwargs_propagates_explicit_worker_environment(monkeypatch):
    monkeypatch.setenv("LAMINAR_RAY_ENV_VARS", json.dumps({"TASK_SERVICE_URL": "https://tasks.example/api"}))
    monkeypatch.setenv("UNRELATED_API_KEY", "do-not-copy")

    env_vars = _ray_init_kwargs_from_env()["runtime_env"]["env_vars"]
    eval_env_vars = validation.ray_runtime_env_from_env({"TOKENIZERS_PARALLELISM": "true"})["env_vars"]

    assert env_vars["TASK_SERVICE_URL"] == "https://tasks.example/api"
    assert eval_env_vars["TASK_SERVICE_URL"] == "https://tasks.example/api"
    assert eval_env_vars["TOKENIZERS_PARALLELISM"] == "true"
    assert "UNRELATED_API_KEY" not in env_vars
    assert "UNRELATED_API_KEY" not in eval_env_vars


@pytest.mark.parametrize("entrypoint", [training, validation], ids=["train", "eval"])
def test_ray_entrypoints_use_shared_runtime_env_helper(monkeypatch, entrypoint):
    monkeypatch.delenv("LAMINAR_RAY_ENV_VARS", raising=False)
    monkeypatch.setenv("RAY_ADDRESS", "192.0.2.1:6379")
    runtime_env = {"env_vars": {"SHARED_HELPER_SENTINEL": "1"}}
    helper_inputs = []
    observed = {}
    monkeypatch.setattr(
        entrypoint,
        "ray_runtime_env_from_env",
        lambda env_vars: helper_inputs.append(env_vars) or runtime_env,
    )
    monkeypatch.setattr(entrypoint.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(entrypoint.ray, "init", lambda **kwargs: observed.update(kwargs))
    monkeypatch.setattr(entrypoint.ray, "get", lambda _ref: None)
    monkeypatch.setattr(entrypoint, "main_task", SimpleNamespace(remote=lambda _config: None))
    if entrypoint is validation:
        monkeypatch.setattr(validation, "_validate_config", lambda _config: None)
        config = OmegaConf.create({"validation": {"queue": {"endpoint": "http://127.0.0.1:9000", "server_port": None}}})
    else:
        monkeypatch.setattr(training, "resolve_train_config", lambda _config: _config)
        config = OmegaConf.create({})

    entrypoint.main.__wrapped__(config)

    assert observed == {"address": "192.0.2.1:6379", "runtime_env": runtime_env}
    assert len(helper_inputs) == 1
    assert helper_inputs[0]["TOKENIZERS_PARALLELISM"] == "true"
    if entrypoint is training:
        assert helper_inputs[0]["NCCL_DEBUG"] == "WARN"


@pytest.mark.parametrize("value", ["[]", '{"TASK_SERVICE_URL": 42}', "null", "invalid-json"])
def test_ray_init_kwargs_rejects_invalid_worker_environment(monkeypatch, value):
    monkeypatch.setenv("LAMINAR_RAY_ENV_VARS", value)
    with pytest.raises(ValueError):
        _ray_init_kwargs_from_env()
