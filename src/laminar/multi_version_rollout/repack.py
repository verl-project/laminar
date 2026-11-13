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
"""Pure same-version Repack planning for multi-version rollout.

The Router applies this plan atomically because it owns route bindings.
"""

import math
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class CommittedRepackGroup:
    """One destination group already atomically committed by the Router."""

    sources: tuple[str, ...]
    destination: str
    weight_version: int


@dataclass(frozen=True)
class CommittedAbortSource:
    """A busy stale server already set draining with no destination (latest-version continuation)."""

    server_id: str
    weight_version: int


@dataclass(frozen=True)
class RepackPlanResult:
    """Minimal Router plan-and-commit result consumed by the lifecycle manager."""

    groups: tuple[CommittedRepackGroup, ...]
    idle_sources: tuple[str, ...]
    abort_sources: tuple[CommittedAbortSource, ...] = ()


@dataclass(frozen=True)
class _RepackServerView:
    """Immutable Router state consumed by one Repack planning pass."""

    server_id: str
    weight_version: int
    active: bool
    runtime_available: bool
    kv_cache_usage: Any | None
    route_count: int


def _valid_kv_cache_usage(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        return None
    return parsed


def _plan_repack_groups(
    servers: Iterable[_RepackServerView],
    *,
    loadable_version: int,
    source_max_kv_cache_usage: float,
    destination_max_kv_cache_usage: float,
    excluded_servers: frozenset[str],
) -> list[CommittedRepackGroup]:
    """Plan same-version Best-Fit groups without mutating Router state."""
    version_groups: dict[int, list[tuple[_RepackServerView, float]]] = {}
    for server in servers:
        if server.server_id in excluded_servers:
            continue
        if not 0 <= server.weight_version < loadable_version:
            continue
        if not server.active:
            continue
        if not server.runtime_available:
            raise ValueError(f"active stale server {server.server_id!r} is not runtime available for repack planning")
        kv_cache_usage = _valid_kv_cache_usage(server.kv_cache_usage)
        if kv_cache_usage is None:
            raise ValueError(f"server {server.server_id!r} has missing or invalid kv_cache_usage for repack planning")
        version_groups.setdefault(server.weight_version, []).append((server, kv_cache_usage))

    groups: list[CommittedRepackGroup] = []
    for version in sorted(version_groups):
        servers_in_version = version_groups[version]
        source_candidates = sorted(
            (kv_cache_usage, server.server_id)
            for server, kv_cache_usage in servers_in_version
            if server.route_count != 0 and kv_cache_usage <= source_max_kv_cache_usage
        )
        if not source_candidates or len(servers_in_version) < 2:
            continue

        planned_sources: dict[str, list[str]] = {}
        planned_destination_kv = {server.server_id: kv_cache_usage for server, kv_cache_usage in servers_in_version}
        for source_kv_cache_usage, source_id in source_candidates:
            if source_id not in planned_destination_kv or source_id in planned_sources:
                continue
            best_fit: tuple[float, str] | None = None
            for destination_id, destination_kv_cache_usage in planned_destination_kv.items():
                if destination_id == source_id:
                    continue
                if destination_kv_cache_usage > destination_max_kv_cache_usage:
                    continue
                projected_destination_kv_cache_usage = destination_kv_cache_usage + source_kv_cache_usage
                if projected_destination_kv_cache_usage > destination_max_kv_cache_usage:
                    continue
                candidate = (projected_destination_kv_cache_usage, destination_id)
                if (
                    best_fit is None
                    or candidate[0] > best_fit[0]
                    or (candidate[0] == best_fit[0] and candidate[1] < best_fit[1])
                ):
                    best_fit = candidate
            if best_fit is None:
                continue
            projected_destination_kv_cache_usage, destination_id = best_fit
            del planned_destination_kv[source_id]
            planned_sources.setdefault(destination_id, []).append(source_id)
            planned_destination_kv[destination_id] = projected_destination_kv_cache_usage

        for destination_id in sorted(planned_sources):
            groups.append(
                CommittedRepackGroup(
                    sources=tuple(planned_sources[destination_id]),
                    destination=destination_id,
                    weight_version=version,
                )
            )
    return groups
