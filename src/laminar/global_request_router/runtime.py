# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Runtime snapshots and KV-pressure routing for the Laminar Router."""

import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

HEALTHY = "healthy"
UNAVAILABLE = "unavailable"
_DYNAMIC_OVERLOAD_MODES = frozenset({"off", "enforce"})


def _required_non_negative_float(value: Any, *, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative number, got {value!r}") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{field_name} must be a non-negative number, got {value!r}")
    return parsed


def _optional_non_negative_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a non-negative integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} must be a non-negative integer, got {parsed}")
    return parsed


def _required_fraction(value: Any, *, field_name: str) -> float:
    if value is None:
        raise ValueError(f"{field_name} must be a number in [0, 1], got None")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number in [0, 1], got {value!r}") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{field_name} must be a number in [0, 1], got {value!r}")
    return parsed


def kv_below_threshold(kv_cache_usage: Any | None, threshold: Any) -> bool:
    """Return whether one replica's observed KV usage is below the shared threshold."""
    if kv_cache_usage is None:
        return False
    usage = _required_fraction(kv_cache_usage, field_name="kv_cache_usage")
    limit = _required_fraction(threshold, field_name="kv_cache_usage_threshold")
    return usage < limit


@dataclass
class _ServerRuntimeState:
    """Runtime-only observations for one rollout server."""

    backend_health: Any = HEALTHY
    kv_cache_usage: Any | None = None
    num_waiting_requests: Any = 0
    num_running_requests: Any = 0


class _RuntimeRouting:
    """Own pushed runtime state and interpret it for request routing."""

    def __init__(
        self,
        server_ids: Iterable[str],
        dynamic_overload_admission: dict[str, Any],
        monotonic_fn: Callable[[], float] | None = None,
    ) -> None:
        self._server_states = {server_id: _ServerRuntimeState() for server_id in server_ids}
        self._snapshot_version = 0
        self._snapshot_observed_at: float | None = None
        self._monotonic = monotonic_fn or time.monotonic

        dynamic_config = dict(dynamic_overload_admission)
        unknown = set(dynamic_config) - {"mode", "max_snapshot_age_s", "kv_cache_usage_threshold"}
        if unknown:
            raise ValueError(f"unknown dynamic overload admission fields: {sorted(unknown)}")
        self._dynamic_overload_mode = str(dynamic_config.get("mode", "off")).lower()
        if self._dynamic_overload_mode not in _DYNAMIC_OVERLOAD_MODES:
            raise ValueError(
                "dynamic overload admission mode must be one of "
                f"{sorted(_DYNAMIC_OVERLOAD_MODES)}, got {self._dynamic_overload_mode!r}"
            )
        self._max_snapshot_age_s = _required_non_negative_float(
            dynamic_config.get("max_snapshot_age_s", 5.0),
            field_name="dynamic_overload_admission.max_snapshot_age_s",
        )
        if self._max_snapshot_age_s == 0:
            raise ValueError("dynamic_overload_admission.max_snapshot_age_s must be positive")
        self._kv_cache_usage_threshold = _required_fraction(
            dynamic_config.get("kv_cache_usage_threshold", 0.8),
            field_name="dynamic_overload_admission.kv_cache_usage_threshold",
        )

    @property
    def snapshot_version(self) -> int:
        return self._snapshot_version

    def server_state(self, server_id: str) -> _ServerRuntimeState:
        return self._server_states[server_id]

    def update_snapshot(self, snapshot: dict[str, dict[str, Any]]) -> bool:
        """Replace runtime observations, ignoring an older snapshot batch."""
        incoming_version = max(
            (
                _optional_non_negative_int(
                    entry.get("snapshot_version"),
                    field_name=f"routing snapshot version for server {server_id!r}",
                )
                or 0
                for server_id, entry in snapshot.items()
                if server_id in self._server_states
            ),
            default=0,
        )
        if self._snapshot_observed_at is not None and incoming_version < self._snapshot_version:
            return False
        if self._snapshot_observed_at is None or incoming_version > self._snapshot_version:
            self._snapshot_observed_at = self._monotonic()
            self._snapshot_version = incoming_version

        for server_id, state in self._server_states.items():
            entry = snapshot.get(server_id)
            if entry is None:
                state.backend_health = UNAVAILABLE
                state.kv_cache_usage = None
                state.num_waiting_requests = 0
                state.num_running_requests = 0
                continue
            state.backend_health = entry.get("backend_health", HEALTHY)
            state.kv_cache_usage = entry.get("kv_cache_usage")
            state.num_waiting_requests = entry.get("num_waiting_requests", 0)
            state.num_running_requests = entry.get("num_running_requests", 0)
        return True

    def admission_open(self, server_id: str, *, now: float | None = None) -> bool:
        if self._dynamic_overload_mode == "off":
            return True
        if self._snapshot_observed_at is None:
            return False
        observed_at = self._monotonic() if now is None else float(now)
        if observed_at - self._snapshot_observed_at > self._max_snapshot_age_s:
            return False
        return kv_below_threshold(
            self._server_states[server_id].kv_cache_usage,
            self._kv_cache_usage_threshold,
        )

    def admission_open_servers(self, server_ids: Iterable[str]) -> list[str]:
        server_ids = list(server_ids)
        if self._dynamic_overload_mode == "off":
            return server_ids
        now = self._monotonic()
        return [server_id for server_id in server_ids if self.admission_open(server_id, now=now)]

    def is_available(self, server_id: str) -> bool:
        return self._server_states[server_id].backend_health != UNAVAILABLE

    def new_route_score(
        self,
        server_id: str,
        *,
        route_count: int,
        inflight_request_count: int,
    ) -> tuple[float, int, int, int, str]:
        state = self._server_states[server_id]
        kv_cache_usage = float(state.kv_cache_usage or 0.0)
        waiting_requests = int(state.num_waiting_requests or 0)
        request_pressure = int(state.num_running_requests or 0) + inflight_request_count
        return (kv_cache_usage, route_count, waiting_requests, request_pressure, server_id)
