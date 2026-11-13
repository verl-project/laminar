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
"""LLM client: Router-owned session_id / weight_version generate.

This client owns Laminar's ``session_id`` contract and optional
``weight_version``, and routes through ``GlobalRequestRouter``. The deprecated
``request_id`` alias accepts callers using verl's upstream client interface;
the engine's per-request ``request_id`` remains a separate identifier.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time
import warnings
from typing import Any
from uuid import uuid4

from omegaconf import DictConfig
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

from laminar.global_request_router.router import RouterClientWaitConfig, RouteUnavailableError
from laminar.multi_version_rollout.lifecycle import is_recoverable_backend_error
from laminar.utils.concurrency import await_shielded_completion

logger = logging.getLogger(__name__)
_REQUEST_ID_ALIAS_WARNED = False


def _caller_deadline(remaining_timeout_s):
    if remaining_timeout_s is None:
        return None
    if (
        isinstance(remaining_timeout_s, bool)
        or not isinstance(remaining_timeout_s, (int, float))
        or not math.isfinite(remaining_timeout_s)
        or remaining_timeout_s <= 0
    ):
        raise ValueError("remaining_timeout_s must be finite and positive")
    return time.monotonic() + remaining_timeout_s


def warn_request_id_deprecated() -> None:
    """Warn once per process about the old session-key spelling."""
    global _REQUEST_ID_ALIAS_WARNED
    if not _REQUEST_ID_ALIAS_WARNED:
        message = "request_id is deprecated as the sticky grouping id; use session_id instead"
        warnings.warn(message, DeprecationWarning, stacklevel=3)
        logger.warning(message)
        _REQUEST_ID_ALIAS_WARNED = True


def resolve_session_id(session_id: str | None = None, *, request_id: str | None = None) -> str:
    """Resolve a session key; engine/OpenAI request IDs are separate identifiers."""
    if session_id is not None and request_id is not None and session_id != request_id:
        raise ValueError(
            "session_id and request_id must match when both are provided, "
            f"got session_id={session_id!r}, request_id={request_id!r}"
        )
    if request_id is not None:
        warn_request_id_deprecated()
        if session_id is None:
            session_id = request_id
    if type(session_id) is not str or not session_id:
        raise ValueError(f"session_id must be a non-empty string, got {session_id!r}")
    return session_id


def _strict_optional_non_negative_int(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a built-in non-negative integer, got {value!r}")
    return value


class LLMServerClient:
    """Proxy route occupancy and generation onto the Router."""

    def __init__(
        self,
        config: DictConfig,
        router,
        **kwargs,
    ):
        if router is None:
            raise ValueError("LLMServerClient requires a GlobalRequestRouter handle")
        self.config = config
        self._router = router
        router_config = getattr(config, "rollout_router", None)
        self._wait = RouterClientWaitConfig.from_value(getattr(router_config, "client_wait", None))

    @staticmethod
    async def _settle_rpc(awaitable, deadline: float | None = None):
        """Await a short Router RPC to a known outcome before propagating interruption."""
        task = asyncio.ensure_future(awaitable)
        interruption: asyncio.CancelledError | TimeoutError | None = None
        try:
            if deadline is None:
                await asyncio.shield(task)
            else:
                remaining = max(0.0, deadline - time.monotonic())
                done, _ = await asyncio.wait({task}, timeout=remaining)
                if not done:
                    interruption = TimeoutError("Router availability wait expired")
        except asyncio.CancelledError as exc:
            interruption = exc
        except BaseException:
            # The task itself failed and is inspected below.
            pass

        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                interruption = exc
            except BaseException:
                break
        if interruption is not None:
            try:
                task.result()
            except BaseException as error:
                raise interruption from error
            raise interruption
        return task.result()

    async def _retry_router_call(
        self,
        method: str,
        kwargs: dict[str, Any],
        *,
        planned_pause: bool,
        caller_deadline: float | None,
    ):
        if type(planned_pause) is not bool:
            raise ValueError("planned_pause must be a boolean")
        started = time.monotonic()
        hard_deadline = math.inf if caller_deadline is None else caller_deadline
        normal_deadline = min(hard_deadline, started + self._wait.timeout_s)
        pause_deadline = min(hard_deadline, started + self._wait.planned_pause_timeout_s)
        deadline = pause_deadline if planned_pause else normal_deadline
        interval = self._wait.initial_backoff_s
        while True:
            if deadline <= time.monotonic():
                raise TimeoutError("Router availability wait expired")
            try:
                remote = getattr(self._router, method).remote(**kwargs)
                return await self._settle_rpc(remote, deadline)
            except RouteUnavailableError as error:
                if getattr(error, "planned_pause", False):
                    deadline = pause_deadline
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Router availability wait expired") from error
                await asyncio.sleep(min(remaining, random.uniform(interval / 2, interval)))
                interval = min(self._wait.max_backoff_s, interval * 2)

    async def _acquire_lease(
        self,
        session_id: str,
        *,
        lease_id: str,
        weight_version: int | None,
        planned_pause: bool,
        caller_deadline: float | None,
    ) -> tuple[str, Any]:
        try:
            return await self._retry_router_call(
                "acquire_lease",
                dict(session_id=session_id, lease_id=lease_id, weight_version=weight_version),
                planned_pause=planned_pause,
                caller_deadline=caller_deadline,
            )
        except BaseException as error:
            _, cleanup_cancellation = await await_shielded_completion(
                self._router.release_lease.remote(session_id=session_id, lease_id=lease_id)
            )
            if cleanup_cancellation is not None:
                raise cleanup_cancellation from error
            raise

    async def _release_lease(self, session_id: str, lease_id: str) -> None:
        acknowledgement = await self._router.release_lease.remote(session_id=session_id, lease_id=lease_id)
        if acknowledgement is not None:
            raise RuntimeError(f"invalid request-lease release acknowledgement: {acknowledgement!r}")

    def bind_route(
        self,
        session_id: str | None = None,
        *,
        request_id: str | None = None,
        weight_version: Any | None = None,
        planned_pause: bool = False,
        remaining_timeout_s: float | None = None,
    ) -> Any:
        """Submit an idempotent sticky-route bind and return its completion handle.

        ``weight_version`` is the lowest acceptable version. Callers pick it at
        dispatch, but sessions can wait for admission long enough for replicas
        at that version to move on, so the Router resolves the routable version
        at bind time. The handle resolves to the bound version (``None`` when
        unversioned or when latest-version continuation leaves the session
        unpinned); callers must enforce that version, not the requested one.
        """
        session_id = resolve_session_id(session_id, request_id=request_id)
        weight_version = _strict_optional_non_negative_int(weight_version, field_name="weight_version")
        caller_deadline = _caller_deadline(remaining_timeout_s)
        return self._retry_router_call(
            "bind_route",
            dict(session_id=session_id, min_weight_version=weight_version),
            planned_pause=planned_pause,
            caller_deadline=caller_deadline,
        )

    def generate(
        self,
        session_id: str | None = None,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str | None = None,
        weight_version: Any | None = None,
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
        audio_data: list[Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        planned_pause: bool = False,
        remaining_timeout_s: float | None = None,
        **kwargs: Any,
    ) -> TokenOutput:
        resolved = resolve_session_id(session_id, request_id=request_id)
        caller_deadline = _caller_deadline(remaining_timeout_s)
        return self._generate(
            session_id=resolved,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            weight_version=weight_version,
            image_data=image_data,
            video_data=video_data,
            audio_data=audio_data,
            mm_processor_kwargs=mm_processor_kwargs,
            planned_pause=planned_pause,
            caller_deadline=caller_deadline,
            **kwargs,
        )

    @rollout_trace_op
    async def _generate(
        self,
        session_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        weight_version: Any | None = None,
        image_data: list[Any] | None = None,
        video_data: list[Any] | None = None,
        audio_data: list[Any] | None = None,
        mm_processor_kwargs: dict[str, Any] | None = None,
        planned_pause: bool = False,
        caller_deadline: float | None = None,
        **kwargs: Any,
    ) -> TokenOutput:
        """Generate tokens on the Router-selected server for one sticky session."""
        weight_version = _strict_optional_non_negative_int(weight_version, field_name="weight_version")
        priority = kwargs.pop("priority", 0)
        rollout = getattr(getattr(self.config, "actor_rollout_ref", None), "rollout", None)
        priority_kwargs = {"priority": priority} if priority != 0 and getattr(rollout, "name", None) == "vllm" else {}
        multimodal_kwargs: dict[str, Any] = {}
        if audio_data is not None:
            multimodal_kwargs["audio_data"] = audio_data
        if mm_processor_kwargs:
            multimodal_kwargs["mm_processor_kwargs"] = mm_processor_kwargs

        # A weight update aborts in-flight turns after the Router moved or dropped
        # their routes, and a backend crash can fail an in-flight turn before its
        # replica is recovered. The turn keeps its already generated tokens and
        # continues on the next acquired server.
        generated_ids: list[int] = []
        generated_log_probs: list[float] = []
        segment_versions: list[Any] = []
        request_sampling_params = sampling_params
        recovery_deadline: float | None = None
        recovery_interval = self._wait.initial_backoff_s
        while True:
            # Build fallible local request identity before Router acquisition.
            # Every acquired request slot must be released before cancellation
            # can reach the Gateway owner.
            backend_request_id = uuid4().hex
            server_id, server = await self._acquire_lease(
                session_id,
                lease_id=backend_request_id,
                weight_version=weight_version,
                planned_pause=planned_pause,
                caller_deadline=caller_deadline,
            )

            output: TokenOutput | None = None
            backend_error: Exception | None = None
            try:
                output = await self._settle_rpc(
                    server.generate.remote(
                        request_id=backend_request_id,
                        session_id=session_id,
                        prompt_ids=[*prompt_ids, *generated_ids] if generated_ids else prompt_ids,
                        sampling_params=request_sampling_params,
                        image_data=image_data,
                        video_data=video_data,
                        **multimodal_kwargs,
                        **priority_kwargs,
                        **kwargs,
                    )
                )
            except Exception as exc:
                if not is_recoverable_backend_error(exc):
                    raise
                backend_error = exc
            finally:
                _, release_cancellation = await await_shielded_completion(
                    self._release_lease(session_id, backend_request_id)
                )
            if release_cancellation is not None:
                if backend_error is not None:
                    raise release_cancellation from backend_error
                raise release_cancellation
            if backend_error is not None:
                now = time.monotonic()
                if recovery_deadline is None:
                    wait_budget = self._wait.planned_pause_timeout_s if planned_pause else self._wait.timeout_s
                    hard_deadline = math.inf if caller_deadline is None else caller_deadline
                    recovery_deadline = min(hard_deadline, now + wait_budget)
                remaining = recovery_deadline - now
                if remaining <= 0:
                    raise TimeoutError("Router availability wait expired") from backend_error
                logger.warning(
                    "Turn %s encountered dead engine on %s after %s tokens; retrying.",
                    backend_request_id,
                    server_id,
                    len(generated_ids),
                )
                await asyncio.sleep(min(remaining, random.uniform(recovery_interval / 2, recovery_interval)))
                recovery_interval = min(self._wait.max_backoff_s, recovery_interval * 2)
                continue
            recovery_deadline = None
            recovery_interval = self._wait.initial_backoff_s
            assert output is not None
            aborted = output.stop_reason == "aborted"
            # Only segments that produced tokens contribute to the version span.
            if output.token_ids or not aborted:
                segment_versions.append(output.extra_fields.get("global_steps"))
            if not aborted:
                break
            generated_ids.extend(output.token_ids)
            generated_log_probs.extend(output.log_probs or [])
            budget_key, budget = self._response_budget(sampling_params, prompt_length=len(prompt_ids))
            remaining = budget - len(generated_ids)
            if remaining <= 0:
                # The abort landed on the budget boundary, so the turn is complete.
                output = TokenOutput(
                    token_ids=[],
                    log_probs=[] if generated_log_probs else None,
                    routed_experts=output.routed_experts,
                    stop_reason="completed",
                    extra_fields=output.extra_fields,
                )
                break
            request_sampling_params = {**sampling_params, budget_key: remaining}
            logger.info(
                "Turn %s was aborted on %s after %s tokens; continuing.",
                backend_request_id,
                server_id,
                len(generated_ids),
            )

        if generated_ids:
            # The final request's routed_experts already cover the continued prefix.
            output.token_ids = generated_ids + list(output.token_ids)
            if output.log_probs is not None:
                output.log_probs = generated_log_probs + list(output.log_probs)

        extra_fields = dict(output.extra_fields)
        global_steps = extra_fields.get("global_steps")
        expected_identity: dict[str, Any] = {
            "session_id": session_id,
            "request_id": backend_request_id,
            "server_id": server_id,
        }
        if weight_version is not None:
            for segment_version in segment_versions or [global_steps]:
                segment_version = _strict_optional_non_negative_int(segment_version, field_name="global_steps")
                if segment_version != weight_version:
                    raise RuntimeError(
                        "backend global_steps conflicts with requested weight_version: "
                        f"expected {weight_version}, got {segment_version!r}"
                    )
            expected_identity.update(
                weight_version=weight_version,
                global_steps=weight_version,
                min_global_steps=weight_version,
                max_global_steps=weight_version,
            )
        for field_name, expected_value in expected_identity.items():
            if field_name in extra_fields and extra_fields[field_name] != expected_value:
                raise RuntimeError(
                    f"backend {field_name} conflicts with Router request identity: "
                    f"expected {expected_value!r}, got {extra_fields[field_name]!r}"
                )
            extra_fields[field_name] = expected_value
        if weight_version is None and "weight_version" not in extra_fields and "global_steps" in extra_fields:
            extra_fields["weight_version"] = extra_fields["global_steps"]
            # Latest-version continuation can span versions within one turn.
            versions = [version for version in segment_versions if version is not None] or [global_steps]
            extra_fields["min_global_steps"] = min(versions)
            extra_fields["max_global_steps"] = max(versions)
        output.extra_fields = extra_fields
        return output

    def _response_budget(self, sampling_params: dict[str, Any], *, prompt_length: int) -> tuple[str, int]:
        """Return the max-token key and the whole turn's response budget."""
        for key in ("max_tokens", "max_new_tokens"):
            if key in sampling_params:
                return key, int(sampling_params[key])
        # Mirror the server default so a continued turn keeps the original budget.
        rollout = self.config.actor_rollout_ref.rollout
        return "max_tokens", min(
            rollout.response_length,
            rollout.prompt_length + rollout.response_length - prompt_length,
        )

    def release_route(
        self,
        session_id: str | None = None,
        *,
        request_id: str | None = None,
    ) -> Any:
        """Submit sticky-route cleanup and return its completion handle."""
        session_id = resolve_session_id(session_id, request_id=request_id)
        return self._router.release_route.remote(session_id=session_id)
