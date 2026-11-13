# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Validation-only exact-version routing."""

from collections.abc import Callable
from typing import Any

from ray.actor import ActorHandle

from laminar.global_request_router.router import (
    ACTIVE,
    NO_AVAILABLE_SERVER_ERROR,
    UPDATING,
    GlobalRequestRouter,
    GlobalRequestRouterConfig,
    RouteUnavailableError,
    _required_fraction,
    _required_non_negative_int,
    _ServerState,
    _strict_optional_non_negative_int,
)
from laminar.multi_version_rollout.repack import RepackPlanResult


def validation_replica_groups(server_ids: list[str], weight_versions: list[int]) -> dict[int, list[str]]:
    """Deterministically partition validation replicas across checkpoints."""
    if not weight_versions or any(type(version) is not int or version < 0 for version in weight_versions):
        raise ValueError("validation weight_versions must be non-empty non-negative integers")
    if len(set(weight_versions)) != len(weight_versions):
        raise ValueError("validation weight_versions must be distinct")
    if len(weight_versions) > len(server_ids):
        raise ValueError("validation requires at least one registered replica per checkpoint")
    groups = {version: [] for version in weight_versions}
    for index, server_id in enumerate(server_ids):
        groups[weight_versions[index % len(weight_versions)]].append(server_id)
    return groups


class ValidationRequestRouter(GlobalRequestRouter):
    """GlobalRequestRouter with validation checkpoint groups pinned exactly."""

    def __init__(
        self,
        servers: dict[str, ActorHandle],
        config: GlobalRequestRouterConfig | None = None,
        monotonic_fn: Callable[[], float] | None = None,
    ) -> None:
        super().__init__(servers=servers, config=config, monotonic_fn=monotonic_fn)
        self._validation_targets: dict[str, int] | None = None

    def _place_exact_route(
        self,
        session_id: str,
        *,
        weight_version: Any | None = None,
        eligible_servers: dict[str, _ServerState] | None = None,
        open_session: bool = False,
    ) -> None:
        session = self._sessions.get(session_id)
        if (
            self._validation_targets is not None
            and (session is None or session.route is None)
            and (session is not None or open_session)
        ):
            exact_version = _strict_optional_non_negative_int(weight_version, field_name="weight_version")
            if exact_version not in set(self._validation_targets.values()):
                raise ValueError("validation routes require an exact assigned weight_version")
        super()._place_exact_route(
            session_id,
            weight_version=weight_version,
            eligible_servers=eligible_servers,
            open_session=open_session,
        )

    def _exact_routable_cohort(self, minimum_weight_version: int) -> tuple[int, dict[str, _ServerState]]:
        if self._validation_targets is None:
            return super()._exact_routable_cohort(minimum_weight_version)
        if minimum_weight_version not in set(self._validation_targets.values()):
            raise ValueError(f"version {minimum_weight_version} is not assigned to a validation replica group")
        eligible_servers = self._eligible_exact_route_servers()
        if not any(state.weight_version == minimum_weight_version for state in eligible_servers.values()):
            raise RouteUnavailableError(NO_AVAILABLE_SERVER_ERROR, planned_pause=self._planned_pause)
        return minimum_weight_version, eligible_servers

    def _eligible_exact_route_servers(self) -> dict[str, _ServerState]:
        eligible_servers = super()._eligible_exact_route_servers()
        if self._validation_targets is None:
            return eligible_servers
        # A changed target removes the replica from routing until that checkpoint is loaded.
        return {
            server_id: state
            for server_id, state in eligible_servers.items()
            if self._validation_targets[server_id] == state.weight_version
        }

    def _validation_servers(self, server_ids: list[str] | None, *, operation: str) -> list[str]:
        selected = list(self._server_states) if server_ids is None else list(server_ids)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError(f"validation {operation} requires non-empty distinct registered servers")
        for server_id in selected:
            self._validate_registered_server(server_id)
        return selected

    def set_validation_versions(
        self, weight_versions: list[int], server_ids: list[str] | None = None, *, repacked: bool = False
    ) -> dict[int, list[str]]:
        """Assign idle replicas while preserving other validation groups and routes."""
        if self._latest_version:
            raise RuntimeError("validation replica groups require same_version continuation")
        if type(repacked) is not bool or (repacked and server_ids is None):
            raise ValueError("repacked validation assignment requires explicit server_ids and a boolean repacked flag")
        selected_servers = self._validation_servers(server_ids, operation="assignment")
        if server_ids is None:
            targets: dict[str, int] = {}
        elif self._validation_targets is None:
            raise RuntimeError("initialize all validation groups before reusing selected replicas")
        else:
            targets = dict(self._validation_targets)
        groups = validation_replica_groups(selected_servers, weight_versions)
        if repacked:
            if not all(
                self._server_states[server_id].lifecycle == UPDATING and self._validation_server_is_drained(server_id)
                for server_id in selected_servers
            ):
                raise RuntimeError(
                    "repacked validation assignment requires selected servers observed updating and drained"
                )
        elif not all(self._validation_server_is_idle(server_id) for server_id in selected_servers):
            raise RuntimeError("validation replica assignment requires selected servers observed active and idle")
        for version, servers in groups.items():
            if any(
                version < self._server_states[server_id].weight_version
                or (repacked and version == self._server_states[server_id].weight_version)
                for server_id in servers
            ):
                raise ValueError("validation replica targets cannot move backwards")
        targets.update({server_id: version for version, servers in groups.items() for server_id in servers})
        self._validation_targets = targets
        return groups

    def _validation_server_is_idle(self, server_id: str) -> bool:
        return self._server_states[server_id].lifecycle == ACTIVE and self._validation_server_is_drained(server_id)

    def _validation_server_is_drained(self, server_id: str) -> bool:
        runtime = self._runtime.server_state(server_id)
        return (
            self._runtime.snapshot_version > 0
            and self._runtime.is_available(server_id)
            and runtime.kv_cache_usage is not None
            and runtime.num_waiting_requests == 0
            and runtime.num_running_requests == 0
            and self._server_is_drained(server_id)
        )

    def plan_validation_repack(
        self,
        server_ids: list[str],
        *,
        source_max_kv_cache_usage: Any = 0.5,
        destination_max_kv_cache_usage: Any = 0.95,
    ) -> RepackPlanResult:
        """Consolidate one live checkpoint with the existing Router planner.

        Idle completed groups remain available to the evaluation coordinator.
        Every selected replica must still serve its assigned exact checkpoint;
        the shared planner atomically transfers routes and fences its sources.
        """
        selected = self._validation_servers(server_ids, operation="repack")
        if self._validation_targets is None:
            raise RuntimeError("initialize validation groups before repacking")
        versions = {self._server_states[server_id].weight_version for server_id in selected}
        if len(versions) != 1 or any(
            self._validation_targets[server_id] != self._server_states[server_id].weight_version
            for server_id in selected
        ):
            raise ValueError("validation repack requires one loaded checkpoint matching every selected target")
        source_max_kv_cache_usage = _required_fraction(
            source_max_kv_cache_usage, field_name="source_max_kv_cache_usage"
        )
        destination_max_kv_cache_usage = _required_fraction(
            destination_max_kv_cache_usage, field_name="destination_max_kv_cache_usage"
        )
        selected_set = set(selected)
        if len(selected) < 2 or not any(binding.server_id in selected_set for binding in self._route_bindings()):
            return RepackPlanResult(groups=(), idle_sources=())
        return super().plan_and_begin_repack(
            loadable_version=versions.pop() + 1,
            source_max_kv_cache_usage=source_max_kv_cache_usage,
            destination_max_kv_cache_usage=destination_max_kv_cache_usage,
            excluded_servers=set(self._server_states) - selected_set,
        )

    def get_idle_server_versions(
        self,
        weight_version: int,
        server_ids: list[str] | None = None,
    ) -> dict[str, int]:
        """Check one validation barrier without retaining an actor RPC slot."""
        weight_version = _required_non_negative_int(weight_version, field_name="weight_version")
        selected_servers = self._validation_servers(server_ids, operation="barrier")
        if any(self._server_states[server_id].weight_version > weight_version for server_id in selected_servers):
            raise RuntimeError(f"rollout server advanced beyond validation version {weight_version}")
        if not all(
            self._server_states[server_id].weight_version == weight_version
            and self._validation_server_is_idle(server_id)
            for server_id in selected_servers
        ):
            raise RouteUnavailableError("selected validation replicas are not yet idle at the requested version")
        return {server_id: self._server_states[server_id].weight_version for server_id in selected_servers}

    def _validate_loaded_weight_version(self, server_id: str, weight_version: int) -> None:
        if self._validation_targets is not None and self._validation_targets[server_id] != weight_version:
            raise RuntimeError(f"server {server_id!r} loaded a version outside its validation target")

    def plan_and_begin_repack(
        self,
        *,
        loadable_version: Any,
        source_max_kv_cache_usage: Any = 0.5,
        destination_max_kv_cache_usage: Any = 0.95,
        excluded_servers: list[str] | tuple[str, ...] | set[str] | frozenset[str] | None = None,
        repack_enabled: bool = True,
    ) -> RepackPlanResult:
        if isinstance(excluded_servers, str | bytes):
            combined_exclusions = excluded_servers
        else:
            combined_exclusions = set(excluded_servers or ())
            if self._validation_targets is not None:
                combined_exclusions.update(
                    server_id for server_id, target in self._validation_targets.items() if target != loadable_version
                )
        return super().plan_and_begin_repack(
            loadable_version=loadable_version,
            source_max_kv_cache_usage=source_max_kv_cache_usage,
            destination_max_kv_cache_usage=destination_max_kv_cache_usage,
            excluded_servers=combined_exclusions,
            repack_enabled=repack_enabled,
        )
