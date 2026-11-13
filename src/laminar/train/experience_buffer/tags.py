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

"""Experience metadata helpers shared by ingest and commit stats."""

from __future__ import annotations

import math
from typing import Any

import torch


def trajectory_key_facts(key: str) -> tuple[str, int, int]:
    """Parse the framework's ``{uid}_{session}_{index}`` trajectory key."""
    parts = key.rsplit("_", 2)
    if len(parts) != 3 or not parts[0] or "_" in parts[0]:
        raise ValueError(f"expected {{uid}}_{{session}}_{{index}} with an underscore-free uid, got {key!r}")
    try:
        session_id = int(parts[1])
        output_index = int(parts[2])
    except ValueError as exc:
        raise ValueError(f"trajectory session and index must be integers, got {key!r}") from exc
    if session_id < 0 or output_index < 0:
        raise ValueError(f"trajectory session and index must be non-negative, got {key!r}")
    return parts[0], session_id, output_index


def _strict_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    item = getattr(value, "item", None)
    if callable(item) and type(value) is not int:
        try:
            value = item()
        except (RuntimeError, TypeError, ValueError):
            return None
    if type(value) is not int:
        return None
    return value if value >= 0 else None


def _tag_weight_version(tag: dict[str, Any]) -> int | None:
    """Return the behavior/weight version recorded on experience metadata, if any."""
    for name in ("trajectory_weight_version", "max_global_steps", "global_steps"):
        value = tag.get(name)
        if value is None:
            continue
        normalized = _strict_nonnegative_int(value)
        if normalized is None:
            raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
        return normalized
    return None


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_padded_tensor(value: Any, *, padding: float = 0.0) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.is_nested:
            return value.to_padded_tensor(padding=padding)
        return value

    to_padded_tensor = getattr(value, "to_padded_tensor", None)
    if to_padded_tensor is None:
        raise TypeError(f"expected tensor-like experience field with to_padded_tensor(), got {type(value).__name__}")
    return to_padded_tensor(padding=padding)
