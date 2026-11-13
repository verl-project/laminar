# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Environment selection shared by Laminar Ray drivers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def uv_environment_from_env() -> dict[str, str]:
    """Forward explicit uv selection, resolving the project before workers change cwd."""
    env_vars = {key: os.environ[key] for key in ("UV_PROJECT", "UV_PROJECT_ENVIRONMENT") if key in os.environ}
    if "UV_PROJECT" in env_vars:
        env_vars["UV_PROJECT"] = str(Path(env_vars["UV_PROJECT"]).resolve())
    return env_vars


def explicit_worker_env_from_env() -> dict[str, str]:
    """Parse explicit worker environment variables from ``LAMINAR_RAY_ENV_VARS``."""
    extra_env = json.loads(os.environ.get("LAMINAR_RAY_ENV_VARS", "{}"))
    if not isinstance(extra_env, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in extra_env.items()
    ):
        raise ValueError("LAMINAR_RAY_ENV_VARS must be a JSON object of string environment values")
    return extra_env


def ray_runtime_env_from_env(env_vars: dict[str, str]) -> dict[str, Any]:
    """Select the prepared driver interpreter for uv jobs, with an explicit override.

    The selected executable must already be available on every worker node.
    Caller environment variables take precedence over the driver's uv selection.
    """
    worker_env = {**uv_environment_from_env(), **env_vars, **explicit_worker_env_from_env()}
    runtime_env: dict[str, Any] = {"env_vars": worker_env}
    executable = os.environ.get("LAMINAR_RAY_PY_EXECUTABLE")
    if executable:
        runtime_env["py_executable"] = executable
    elif "UV_PROJECT" in worker_env or "UV_PROJECT_ENVIRONMENT" in worker_env:
        runtime_env["py_executable"] = sys.executable
    return runtime_env
