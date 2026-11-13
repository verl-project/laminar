# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import pytest

from laminar.ray_runtime import ray_runtime_env_from_env, uv_environment_from_env


def test_uv_environment_requires_explicit_selection(monkeypatch):
    monkeypatch.delenv("UV_PROJECT", raising=False)
    monkeypatch.delenv("UV_PROJECT_ENVIRONMENT", raising=False)
    monkeypatch.setenv("UV_CACHE_DIR", "unrelated-cache")

    assert uv_environment_from_env() == {}


def test_uv_environment_preserves_environment_without_project(monkeypatch):
    monkeypatch.delenv("UV_PROJECT", raising=False)
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/shared/venv")

    assert uv_environment_from_env() == {"UV_PROJECT_ENVIRONMENT": "/shared/venv"}


def test_uv_environment_resolves_relative_project_in_driver_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UV_PROJECT", "../environment")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", ".venv-custom")

    assert uv_environment_from_env() == {
        "UV_PROJECT": str(tmp_path.parent / "environment"),
        "UV_PROJECT_ENVIRONMENT": ".venv-custom",
    }
    assert os.environ["UV_PROJECT"] == "../environment"


def test_ray_runtime_environment_keeps_default_interpreter(monkeypatch):
    for key in ("UV_PROJECT", "UV_PROJECT_ENVIRONMENT", "LAMINAR_RAY_PY_EXECUTABLE"):
        monkeypatch.delenv(key, raising=False)
    env_vars = {"TOKENIZERS_PARALLELISM": "true"}

    assert ray_runtime_env_from_env(env_vars) == {"env_vars": env_vars}


@pytest.mark.parametrize("selection", ["UV_PROJECT", "UV_PROJECT_ENVIRONMENT"])
def test_ray_runtime_environment_uses_selected_driver_interpreter(monkeypatch, selection, tmp_path):
    for key in ("UV_PROJECT", "UV_PROJECT_ENVIRONMENT", "LAMINAR_RAY_PY_EXECUTABLE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(selection, str(tmp_path / "environment"))

    runtime_env = ray_runtime_env_from_env({})

    assert runtime_env["py_executable"] == sys.executable
    assert runtime_env["env_vars"][selection] == str(tmp_path / "environment")


def test_ray_runtime_environment_explicit_interpreter_takes_precedence(monkeypatch):
    monkeypatch.delenv("UV_PROJECT_ENVIRONMENT", raising=False)
    monkeypatch.setenv("UV_PROJECT", "/driver/project")
    monkeypatch.setenv("LAMINAR_RAY_PY_EXECUTABLE", "/worker/python")

    assert ray_runtime_env_from_env({})["py_executable"] == "/worker/python"


def test_ray_runtime_environment_preserves_caller_overrides(monkeypatch):
    monkeypatch.setenv("UV_PROJECT", "/driver/project")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", ".venv-driver")
    env_vars = {"UV_PROJECT": "/worker/project", "TOKENIZERS_PARALLELISM": "false"}

    runtime_env = ray_runtime_env_from_env(env_vars)

    assert runtime_env["env_vars"] == {
        "UV_PROJECT": "/worker/project",
        "UV_PROJECT_ENVIRONMENT": ".venv-driver",
        "TOKENIZERS_PARALLELISM": "false",
    }
    assert env_vars == {"UV_PROJECT": "/worker/project", "TOKENIZERS_PARALLELISM": "false"}
