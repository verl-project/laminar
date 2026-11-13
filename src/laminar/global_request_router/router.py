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
"""Laminar GlobalRequestRouter: versioned sticky-route owner for async_rl.

Extracted from the public rollout llm_server module so versioned routing does
not live on the upstream GlobalRequestLoadBalancer. Same-version Repack
planning lives in ``multi_version_rollout``; this actor applies and commits it.
The configured continuation selects same-version pinning or latest-version
routing, where each request lease follows the newest currently routable weights.
"""

import math
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import ray

from laminar.global_request_router.runtime import _optional_non_negative_int, _required_fraction, _RuntimeRouting
from laminar.multi_version_rollout.repack import (
    CommittedAbortSource,
    RepackPlanResult,
    _plan_repack_groups,
    _RepackServerView,
)

NO_AVAILABLE_SERVER_ERROR = "No available servers in load balancer"
EMPTY_WEIGHT_VERSION = -1
ACTIVE = "active"
DRAINING = "draining"
UPDATING = "updating"
SAME_VERSION = "same_version"
LATEST_VERSION = "latest_version"


class RouteUnavailableError(RuntimeError):
    """A valid routing operation cannot commit against the current replica state."""

    def __init__(self, message: str, *, planned_pause: bool = False) -> None:
        super().__init__(message)
        self.planned_pause = planned_pause


class SessionRouteClosedError(RuntimeError):
    """The operation does not belong to a live explicitly bound session."""


def _required_non_negative_int(value: Any, *, field_name: str) -> int:
    parsed = _optional_non_negative_int(value, field_name=field_name)
    if parsed is None:
        raise ValueError(f"{field_name} must be a non-negative integer, got None")
    return parsed


def _strict_optional_non_negative_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a built-in non-negative integer, got {value!r}")
    return value


def _required_non_empty_str(value: Any, *, field_name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")
    return value


@dataclass(frozen=True)
class RouterClientWaitConfig:
    """Client-owned retry budget for temporarily unavailable routes."""

    timeout_s: float = 600.0
    planned_pause_timeout_s: float = 3600.0
    initial_backoff_s: float = 0.1
    max_backoff_s: float = 2.0

    def __post_init__(self) -> None:
        for field_name in ("timeout_s", "planned_pause_timeout_s", "initial_backoff_s", "max_backoff_s"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field_name} must be finite and positive, got {value!r}")
            object.__setattr__(self, field_name, float(value))
        if self.planned_pause_timeout_s < self.timeout_s:
            raise ValueError("planned_pause_timeout_s must be greater than or equal to timeout_s")
        if self.max_backoff_s < self.initial_backoff_s:
            raise ValueError("max_backoff_s must be greater than or equal to initial_backoff_s")

    @classmethod
    def from_value(cls, value: Any) -> "RouterClientWaitConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(f"client_wait must be a mapping or RouterClientWaitConfig, got {type(value)!r}")
        return cls(**dict(value))


@dataclass(frozen=True)
class GlobalRequestRouterConfig:
    """Typed Hydra surface for Router policy."""

    dynamic_overload_admission: dict[str, Any] = field(default_factory=dict)
    # This policy is consumed by LLMServerClient, not by the Router actor.
    client_wait: RouterClientWaitConfig | dict[str, Any] = field(default_factory=RouterClientWaitConfig)
    # Version a turn continues on after its request is aborted for a weight
    # update. ``same_version`` pins each session to its bound version;
    # ``latest_version`` rebinds before each request lease, so aborted turns
    # resume on the newest version routable at reacquisition time.
    continuation: str = SAME_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "client_wait", RouterClientWaitConfig.from_value(self.client_wait))


@dataclass(frozen=True)
class _RouteBinding:
    """Router-owned identity and placement for one live sticky route."""

    server_id: str
    weight_version: int | None


@dataclass
class _ServerState:
    """Router-local version state for one rollout server."""

    handle: ray.actor.ActorHandle
    weight_version: int = EMPTY_WEIGHT_VERSION
    lifecycle: str = ACTIVE


@dataclass
class _SessionState:
    """One explicitly bound session owns its route and committed request leases."""

    session_id: str
    route: _RouteBinding | None = None
    leases: dict[str, str] = field(default_factory=dict)


class GlobalRequestRouter:
    """Version-aware sticky routing, in-flight accounting, and Repack ownership."""

    def __init__(
        self,
        servers: dict[str, ray.actor.ActorHandle],
        config: GlobalRequestRouterConfig | None = None,
        monotonic_fn: Callable[[], float] | None = None,
    ):
        router_config = GlobalRequestRouterConfig() if config is None else config
        if not isinstance(router_config, GlobalRequestRouterConfig):
            raise TypeError(f"config must be GlobalRequestRouterConfig, got {type(router_config)!r}")
        if not servers:
            raise ValueError("servers must be non-empty")
        if router_config.continuation not in {SAME_VERSION, LATEST_VERSION}:
            raise ValueError(
                f"continuation must be {SAME_VERSION!r} or {LATEST_VERSION!r}, got {router_config.continuation!r}"
            )
        self._latest_version = router_config.continuation == LATEST_VERSION
        self._server_states = {server_id: _ServerState(handle=handle) for server_id, handle in servers.items()}
        self._sessions: dict[str, _SessionState] = {}
        self._recovering_servers: set[str] = set()
        self._runtime = _RuntimeRouting(
            server_ids=servers,
            dynamic_overload_admission=router_config.dynamic_overload_admission,
            monotonic_fn=monotonic_fn,
        )
        self._planned_pause = False

    @property
    def _session_id_to_server(self) -> dict[str, _RouteBinding]:
        """Read-only derived route view for diagnostics; SessionState owns mutations."""
        return {session_id: state.route for session_id, state in self._sessions.items() if state.route is not None}

    def _route_bindings(self):
        return (state.route for state in self._sessions.values() if state.route is not None)

    def _live_session(self, session_id: Any) -> _SessionState:
        session_id = _required_non_empty_str(session_id, field_name="session_id")
        state = self._sessions.get(session_id)
        if state is None:
            raise SessionRouteClosedError(f"session {session_id!r} is not bound")
        return state

    def get_active_session_count(self) -> int:
        return len(self._sessions)

    def _inflight_count(self, server_id: str) -> int:
        return sum(
            leased_server == server_id
            for session in self._sessions.values()
            for leased_server in session.leases.values()
        )

    def _inflight_counts(self) -> Counter[str]:
        """Snapshot lease ownership for one synchronous routing or drain pass."""
        return Counter(server_id for session in self._sessions.values() for server_id in session.leases.values())

    def bind_route(
        self,
        session_id: str,
        *,
        min_weight_version: Any | None = None,
    ) -> int | None:
        """Open a session and bind it according to the configured continuation policy."""
        session_id = _required_non_empty_str(session_id, field_name="session_id")
        minimum = _strict_optional_non_negative_int(min_weight_version, field_name="min_weight_version")
        if self._latest_version:
            self._open_latest_route(session_id, minimum)
            return None
        return self._open_exact_route(session_id, minimum)

    def _open_latest_route(self, session_id: str, minimum: int | None) -> None:
        eligible_servers = self._latest_routable_servers(0 if minimum is None else minimum)
        self._place_latest_route(session_id, eligible_servers=eligible_servers, open_session=True)

    def _open_exact_route(self, session_id: str, minimum: int | None) -> int | None:
        session = self._sessions.get(session_id)
        if session is not None and session.route is not None:
            if minimum is not None and session.route.weight_version < minimum:
                raise RuntimeError(
                    f"sticky session {session_id!r} is bound below the requested version: "
                    f"bound {session.route.weight_version!r}, requested at least {minimum!r}"
                )
            self._place_exact_route(session_id, weight_version=session.route.weight_version)
            return session.route.weight_version if minimum is not None else None
        if minimum is None:
            self._place_exact_route(session_id, open_session=True)
            return None
        version, eligible_servers = self._exact_routable_cohort(minimum)
        self._place_exact_route(
            session_id,
            weight_version=version,
            eligible_servers=eligible_servers,
            open_session=True,
        )
        return version

    def _place_exact_route(
        self,
        session_id: str,
        *,
        weight_version: Any | None = None,
        eligible_servers: dict[str, _ServerState] | None = None,
        open_session: bool = False,
    ) -> None:
        """Keep or place one same-version route; only ``bind_route`` may open it."""
        session_id = _required_non_empty_str(session_id, field_name="session_id")
        weight_version = _strict_optional_non_negative_int(weight_version, field_name="weight_version")
        session = self._sessions.get(session_id)
        if session is None and not open_session:
            raise SessionRouteClosedError(f"session {session_id!r} is not bound")
        if session is not None and session.route is not None:
            self._validate_exact_route(session, weight_version)
            return

        if eligible_servers is None:
            eligible_servers = self._eligible_exact_route_servers()
        matching_servers = [
            server_id
            for server_id, state in eligible_servers.items()
            if weight_version is None or state.weight_version == weight_version
        ]
        server_id = self._select_exact_route_server(matching_servers)
        state = self._server_states[server_id]
        bound_weight_version = state.weight_version if weight_version is None else weight_version
        if session is None:
            if not open_session:
                raise SessionRouteClosedError(f"session {session_id!r} is not bound")
            session = self._sessions.setdefault(session_id, _SessionState(session_id))
        session.route = _RouteBinding(
            server_id=server_id,
            weight_version=bound_weight_version,
        )

    def _validate_exact_route(
        self,
        session: _SessionState,
        weight_version: int | None,
    ) -> _RouteBinding:
        """Validate and return the immutable route of a same-version session."""
        binding = session.route
        assert binding is not None
        if weight_version is not None and binding.weight_version != weight_version:
            raise RuntimeError(
                f"sticky session {session.session_id!r} version conflicts with binding: "
                f"expected {binding.weight_version!r}, got {weight_version!r}"
            )
        server_id = binding.server_id
        if self._server_can_keep_exact_route(server_id, binding.weight_version):
            return binding
        state = self._server_states[server_id]
        raise RouteUnavailableError(
            f"sticky session {session.session_id!r} cannot use bound server {server_id!r}: "
            f"lifecycle_state={state.lifecycle!r}, "
            f"server_weight_version={state.weight_version!r}",
            planned_pause=self._planned_pause,
        )

    def _route_selection_inputs(self, base_eligible_servers: list[str]) -> tuple[list[str], Counter[str], Counter[str]]:
        # Runtime admission is a routing preference, not a hard session gate.
        if not base_eligible_servers:
            raise RouteUnavailableError(NO_AVAILABLE_SERVER_ERROR, planned_pause=self._planned_pause)
        runtime_eligible_servers = self._runtime.admission_open_servers(base_eligible_servers)
        eligible_servers = runtime_eligible_servers or base_eligible_servers
        route_counts = Counter(binding.server_id for binding in self._route_bindings())
        return eligible_servers, route_counts, self._inflight_counts()

    def _select_exact_route_server(self, base_eligible_servers: list[str]) -> str:
        eligible_servers, route_counts, inflight_counts = self._route_selection_inputs(base_eligible_servers)

        def exact_route_score(server_id: str) -> tuple[float, int, int, int, str]:
            return self._runtime.new_route_score(
                server_id,
                route_count=route_counts[server_id],
                inflight_request_count=inflight_counts[server_id],
            )

        return min(eligible_servers, key=exact_route_score)

    def _select_latest_route_server(self, base_eligible_servers: list[str]) -> str:
        eligible_servers, route_counts, inflight_counts = self._route_selection_inputs(base_eligible_servers)

        def latest_route_score(server_id: str) -> tuple[int, float, int, int, str]:
            score = self._runtime.new_route_score(
                server_id,
                route_count=route_counts[server_id],
                inflight_request_count=inflight_counts[server_id],
            )
            # A resumed cohort may still report pre-update KV usage; balance routes first.
            return (score[1], score[0], score[2], score[3], score[4])

        return min(eligible_servers, key=latest_route_score)

    def _place_latest_route(
        self,
        session_id: str,
        *,
        eligible_servers: dict[str, _ServerState] | None = None,
        open_session: bool = False,
    ) -> None:
        """Keep a newest-version route when possible, otherwise rebind it."""
        if eligible_servers is None:
            eligible_servers = self._eligible_latest_route_servers()
        if not eligible_servers:
            raise RouteUnavailableError(NO_AVAILABLE_SERVER_ERROR, planned_pause=self._planned_pause)
        newest_version = max(state.weight_version for state in eligible_servers.values())
        newest_servers = [
            server_id for server_id, state in eligible_servers.items() if state.weight_version == newest_version
        ]
        session = self._sessions.get(session_id)
        binding = session.route if session is not None else None
        if binding is not None and binding.server_id in newest_servers:
            server_id = binding.server_id
            if binding.weight_version == newest_version:
                return
        else:
            server_id = self._select_latest_route_server(newest_servers)
        if session is None:
            if not open_session:
                raise SessionRouteClosedError(f"session {session_id!r} is not bound")
            session = self._sessions.setdefault(session_id, _SessionState(session_id))
        session.route = _RouteBinding(server_id=server_id, weight_version=newest_version)

    def get_routable_version(self, weight_version: Any) -> int:
        """Immediately return the lowest routable version at or above the minimum."""
        minimum = _required_non_negative_int(weight_version, field_name="weight_version")
        if self._latest_version:
            eligible_servers = self._latest_routable_servers(minimum)
            return min(state.weight_version for state in eligible_servers.values() if state.weight_version >= minimum)
        version, _ = self._exact_routable_cohort(minimum)
        return version

    def _latest_routable_servers(self, minimum_weight_version: int) -> dict[str, _ServerState]:
        eligible_servers = self._eligible_latest_route_servers()
        if not any(state.weight_version >= minimum_weight_version for state in eligible_servers.values()):
            raise RouteUnavailableError(NO_AVAILABLE_SERVER_ERROR, planned_pause=self._planned_pause)
        return eligible_servers

    def _exact_routable_cohort(self, minimum_weight_version: int) -> tuple[int, dict[str, _ServerState]]:
        """Return the lowest accepted version and the snapshot used to select it."""
        eligible_servers = self._eligible_exact_route_servers()
        versions = [
            state.weight_version
            for state in eligible_servers.values()
            if state.weight_version >= minimum_weight_version
        ]
        if not versions:
            raise RouteUnavailableError(NO_AVAILABLE_SERVER_ERROR, planned_pause=self._planned_pause)
        return min(versions), eligible_servers

    def acquire_lease(
        self,
        session_id: str,
        *,
        lease_id: str,
        weight_version: Any | None = None,
    ) -> tuple[str, ray.actor.ActorHandle]:
        """Atomically commit one idempotent request lease on a live session."""
        session = self._live_session(session_id)
        lease_id = _required_non_empty_str(lease_id, field_name="lease_id")
        weight_version = _strict_optional_non_negative_int(weight_version, field_name="weight_version")
        if self._latest_version and weight_version is not None:
            raise ValueError("latest_version continuation routes every turn to the newest version; got a pin")
        server_id = session.leases.get(lease_id)
        if server_id is not None:
            return server_id, self._server_states[server_id].handle
        if self._latest_version:
            eligible_servers = self._latest_routable_servers(0)
            self._place_latest_route(session_id, eligible_servers=eligible_servers)
            binding = session.route
            assert binding is not None
        else:
            binding = self._validate_exact_route(session, weight_version)
        server_id = binding.server_id
        # Leases retain the serving replica even if a later update moves or drops the route.
        session.leases[lease_id] = server_id
        return server_id, self._server_states[server_id].handle

    def release_lease(self, session_id: str, lease_id: str) -> None:
        """Idempotently release one committed request lease."""
        session_id = _required_non_empty_str(session_id, field_name="session_id")
        lease_id = _required_non_empty_str(lease_id, field_name="lease_id")
        session = self._sessions.get(session_id)
        if session is not None:
            session.leases.pop(lease_id, None)

    def release_route(
        self,
        session_id: str,
        *,
        weight_version: Any | None = None,
    ) -> None:
        """Close an idle session; generation owners must settle leases first."""
        session_id = _required_non_empty_str(session_id, field_name="session_id")
        weight_version = _strict_optional_non_negative_int(weight_version, field_name="weight_version")
        session = self._sessions.get(session_id)
        if session is None:
            return
        binding = session.route
        if weight_version is not None and (binding is None or binding.weight_version != weight_version):
            raise RuntimeError(
                f"route release version conflicts for {session_id!r}: "
                f"expected {None if binding is None else binding.weight_version!r}, got {weight_version!r}"
            )
        if session.leases:
            raise RuntimeError(f"session {session_id!r} still owns {len(session.leases)} request lease(s)")
        self._sessions.pop(session_id)

    def update_routing_snapshot(
        self,
        snapshot: dict[str, dict[str, Any]],
    ) -> None:
        """Replace the runtime cache pushed by ``LLMServerStateTracker``."""
        self._runtime.update_snapshot(snapshot)

    def failed_server_ids(self) -> list[str]:
        """Return unavailable replicas outside an owned recovery or weight load."""
        return [
            server_id
            for server_id, state in self._server_states.items()
            if server_id not in self._recovering_servers
            and state.lifecycle != UPDATING
            and not self._runtime.is_available(server_id)
        ]

    def begin_recovery(self, server_id: str) -> None:
        """Fence a failed replica without discarding live routes or release ACKs."""
        self._validate_registered_server(server_id)
        self._recovering_servers.add(server_id)
        self._set_lifecycle_state(server_id, UPDATING)

    def finish_recovery(self, server_id: str, handle: ray.actor.ActorHandle, weight_version: int) -> None:
        """Install a rebuilt, loaded replica; runtime health still comes from snapshots."""
        self._validate_registered_server(server_id)
        weight_version = _required_non_negative_int(weight_version, field_name="weight_version")
        if server_id not in self._recovering_servers:
            raise RuntimeError(f"server {server_id!r} is not recovering")
        self._validate_loaded_weight_version(server_id, weight_version)
        state = self._server_states[server_id]
        state.handle = handle
        state.weight_version = weight_version
        self._recovering_servers.remove(server_id)
        self._set_lifecycle_state(server_id, ACTIVE)

    def plan_and_begin_repack(
        self,
        *,
        loadable_version: Any,
        source_max_kv_cache_usage: Any = 0.5,
        destination_max_kv_cache_usage: Any = 0.95,
        excluded_servers: list[str] | tuple[str, ...] | set[str] | frozenset[str] | None = None,
        repack_enabled: bool = True,
    ) -> RepackPlanResult:
        """Atomically plan and commit one Repack pass from the pushed runtime cache."""
        loadable_version = _required_non_negative_int(loadable_version, field_name="loadable_version")
        source_max_kv_cache_usage = _required_fraction(
            source_max_kv_cache_usage,
            field_name="source_max_kv_cache_usage",
        )
        destination_max_kv_cache_usage = _required_fraction(
            destination_max_kv_cache_usage,
            field_name="destination_max_kv_cache_usage",
        )
        if isinstance(excluded_servers, str | bytes):
            raise ValueError("excluded_servers must be a sequence of server ids")
        if not isinstance(repack_enabled, bool):
            raise ValueError(f"repack_enabled must be a boolean, got {repack_enabled!r}")
        excluded = frozenset(excluded_servers or ())
        unknown_exclusions = sorted(excluded.difference(self._server_states))
        if unknown_exclusions:
            raise ValueError(f"excluded_servers contains unregistered servers: {unknown_exclusions}")
        excluded |= self._recovering_servers | {
            server_id for server_id in self._server_states if not self._runtime.is_available(server_id)
        }
        if self._latest_version:
            return self._begin_latest_version_updates(loadable_version=loadable_version, excluded_servers=excluded)
        return self._begin_exact_version_repack(
            loadable_version=loadable_version,
            source_max_kv_cache_usage=source_max_kv_cache_usage,
            destination_max_kv_cache_usage=destination_max_kv_cache_usage,
            excluded_servers=excluded,
            repack_enabled=repack_enabled,
        )

    def begin_colocate_pause(self) -> dict[str, int]:
        """Fence the full pool and let interrupted turns rebind after training."""
        if not self._latest_version:
            raise ValueError("colocate_async requires latest_version continuation")
        if self._recovering_servers or any(
            state.lifecycle != ACTIVE or not self._runtime.is_available(server_id)
            for server_id, state in self._server_states.items()
        ):
            raise RuntimeError("colocate_async requires the entire rollout pool to be healthy and active")
        versions = {server_id: state.weight_version for server_id, state in self._server_states.items()}
        for session in self._sessions.values():
            # Leases stay attached to old replicas and continue to gate the update fence.
            session.route = None
        self._planned_pause = True
        for state in self._server_states.values():
            state.lifecycle = DRAINING
        return versions

    def get_server_handle(self, server_id: str):
        self._validate_registered_server(server_id)
        return self._server_states[server_id].handle

    def try_mark_updating(
        self,
        server_id: str,
        *,
        expected_version: Any,
    ) -> None:
        """Fence a drained source; its manager owns engine aborts and retry timing."""
        expected_version = _required_non_negative_int(expected_version, field_name="expected_version")
        self._validate_server_expected_version(server_id, expected_version)
        lifecycle = self._server_states[server_id].lifecycle
        if lifecycle not in {DRAINING, UPDATING}:
            raise ValueError(f"server {server_id!r} must be draining before updating, got {lifecycle!r}")

        if not self._server_is_drained(server_id):
            raise RouteUnavailableError(f"server {server_id!r} still owns routes or in-flight requests")
        self._set_lifecycle_state(server_id, UPDATING)

    def _server_is_drained(
        self, server_id: str, *, route_count: int | None = None, inflight_count: int | None = None
    ) -> bool:
        runtime_state = self._runtime.server_state(server_id)
        if route_count is None:
            route_count = self._count_session_routes(server_id)
        return (
            route_count == 0
            and (self._inflight_count(server_id) if inflight_count is None else inflight_count) == 0
            and int(runtime_state.num_waiting_requests or 0) == 0
            and int(runtime_state.num_running_requests or 0) == 0
        )

    def mark_active(
        self,
        server_id: str,
        *,
        weight_version: Any,
    ) -> None:
        """Open a server for routing after a successful weight load."""
        if server_id in self._recovering_servers:
            raise RuntimeError(f"server {server_id!r} is recovering")
        weight_version = _required_non_negative_int(weight_version, field_name="weight_version")
        self._validate_registered_server(server_id)
        self._validate_loaded_weight_version(server_id, weight_version)
        state = self._server_states[server_id]
        state.weight_version = weight_version
        if state.lifecycle != ACTIVE:
            self._set_lifecycle_state(server_id, ACTIVE)
        if self._planned_pause and not any(
            candidate.lifecycle in {DRAINING, UPDATING}
            for replica_id, candidate in self._server_states.items()
            if replica_id not in self._recovering_servers
        ):
            self._planned_pause = False

    def _count_session_routes(self, server_id: str) -> int:
        """Count live sticky session routes owned by one server."""
        return sum(1 for binding in self._route_bindings() if binding.server_id == server_id)

    def _set_lifecycle_state(self, server_id: str, lifecycle_state: str) -> None:
        self._validate_registered_server(server_id)
        if lifecycle_state not in {ACTIVE, DRAINING, UPDATING}:
            raise ValueError(f"invalid lifecycle_state {lifecycle_state!r}")
        state = self._server_states[server_id]
        if state.lifecycle == lifecycle_state:
            return
        state.lifecycle = lifecycle_state

    def _validate_registered_server(self, server_id: str) -> None:
        if server_id not in self._server_states:
            raise ValueError(f"server {server_id!r} is not registered")

    def _validate_loaded_weight_version(self, server_id: str, weight_version: int) -> None:
        """Allow specialized routers to enforce additional load invariants."""

    def _validate_server_expected_version(self, server_id: str, expected_version: int) -> None:
        self._validate_registered_server(server_id)
        actual_version = self._server_states[server_id].weight_version
        if actual_version != expected_version:
            raise ValueError(
                f"server {server_id!r} version {actual_version!r} does not match expected {expected_version!r}"
            )

    def _begin_exact_version_repack(
        self,
        *,
        loadable_version: int,
        source_max_kv_cache_usage: float,
        destination_max_kv_cache_usage: float,
        excluded_servers: frozenset[str],
        repack_enabled: bool,
    ) -> RepackPlanResult:
        # Classification shares one synchronous snapshot; route mutations happen below.
        route_counts = Counter(binding.server_id for binding in self._route_bindings()) if repack_enabled else None
        groups = (
            _plan_repack_groups(
                (
                    _RepackServerView(
                        server_id=server_id,
                        weight_version=state.weight_version,
                        active=state.lifecycle == ACTIVE,
                        runtime_available=self._runtime.is_available(server_id),
                        kv_cache_usage=self._runtime.server_state(server_id).kv_cache_usage,
                        route_count=route_counts[server_id],
                    )
                    for server_id, state in self._server_states.items()
                ),
                loadable_version=loadable_version,
                source_max_kv_cache_usage=source_max_kv_cache_usage,
                destination_max_kv_cache_usage=destination_max_kv_cache_usage,
                excluded_servers=excluded_servers,
            )
            if repack_enabled
            else []
        )
        participants = {server_id for group in groups for server_id in (group.destination, *group.sources)}
        inflight_counts = self._inflight_counts()
        idle_sources: list[str] = []
        for server_id, state in self._server_states.items():
            if (
                server_id in excluded_servers | participants
                or state.weight_version >= loadable_version
                or state.lifecycle != ACTIVE
                or not self._runtime.is_available(server_id)
            ):
                continue
            if route_counts is None:
                route_counts = Counter(binding.server_id for binding in self._route_bindings())
            if self._server_is_drained(
                server_id, route_count=route_counts[server_id], inflight_count=inflight_counts[server_id]
            ):
                idle_sources.append(server_id)

        if groups or idle_sources:
            source_to_destination = {source: group.destination for group in groups for source in group.sources}
            # Move only future turns; committed leases continue draining on their source.
            for session in self._sessions.values():
                binding = session.route
                if binding is None:
                    continue
                destination = source_to_destination.get(binding.server_id)
                if destination is not None:
                    session.route = _RouteBinding(
                        server_id=destination,
                        weight_version=binding.weight_version,
                    )
            for source in source_to_destination:
                self._server_states[source].lifecycle = DRAINING
            for source in idle_sources:
                self._server_states[source].lifecycle = UPDATING

        return RepackPlanResult(groups=tuple(groups), idle_sources=tuple(idle_sources))

    def _begin_latest_version_updates(
        self,
        *,
        loadable_version: int,
        excluded_servers: frozenset[str],
    ) -> RepackPlanResult:
        """Retire every stale server at once: latest-version rollout serves only the newest weights."""
        route_counts = None
        inflight_counts = self._inflight_counts()
        idle_sources: list[str] = []
        abort_sources: list[CommittedAbortSource] = []
        for server_id, state in self._server_states.items():
            if server_id in excluded_servers or state.lifecycle != ACTIVE or state.weight_version >= loadable_version:
                continue
            can_idle = self._runtime.is_available(server_id)
            if can_idle:
                if route_counts is None:
                    route_counts = Counter(binding.server_id for binding in self._route_bindings())
                can_idle = self._server_is_drained(
                    server_id, route_count=route_counts[server_id], inflight_count=inflight_counts[server_id]
                )
            if can_idle:
                idle_sources.append(server_id)
            else:
                abort_sources.append(CommittedAbortSource(server_id=server_id, weight_version=state.weight_version))

        if idle_sources or abort_sources:
            draining = {source.server_id for source in abort_sources}
            # Drop future-turn routes; committed leases still drain on their serving replica.
            for session in self._sessions.values():
                if session.route is not None and session.route.server_id in draining:
                    session.route = None
            for server_id in draining:
                self._server_states[server_id].lifecycle = DRAINING
            for server_id in idle_sources:
                self._server_states[server_id].lifecycle = UPDATING
        return RepackPlanResult(groups=(), idle_sources=tuple(idle_sources), abort_sources=tuple(abort_sources))

    def _server_can_keep_exact_route(self, server_id: str, weight_version: int | None) -> bool:
        if server_id not in self._server_states:
            return False
        if not self._runtime.is_available(server_id):
            return False
        if weight_version is not None and self._server_states[server_id].weight_version != weight_version:
            return False
        # Draining rejects new routes, but an existing route remains valid until the update fence.
        return self._server_states[server_id].lifecycle != UPDATING

    def _active_route_servers(self) -> dict[str, _ServerState]:
        return {
            server_id: state
            for server_id, state in self._server_states.items()
            if state.lifecycle == ACTIVE
            and state.weight_version != EMPTY_WEIGHT_VERSION
            and self._runtime.is_available(server_id)
        }

    def _eligible_exact_route_servers(self) -> dict[str, _ServerState]:
        return self._active_route_servers()

    def _eligible_latest_route_servers(self) -> dict[str, _ServerState]:
        # Reopen an updated pool as one cohort, not as each replica finishes loading.
        if any(
            state.lifecycle in {DRAINING, UPDATING}
            for replica_id, state in self._server_states.items()
            if replica_id not in self._recovering_servers
        ):
            return {}
        return self._active_route_servers()
