# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest
from verl.workers.rollout.replica import TokenOutput

from laminar.global_request_router.router import RouteUnavailableError
from laminar.llm_server_client import LLMServerClient, resolve_session_id


class _Remote:
    def __init__(self, function):
        self.function = function

    def remote(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class _Server:
    def __init__(self, outputs=None):
        self.calls = []
        self.outputs = list(
            outputs or [TokenOutput(token_ids=[7, 8], stop_reason="completed", extra_fields={"global_steps": 3})]
        )
        self.generate = _Remote(self._generate)

    async def _generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs.pop(0)


class _Router:
    def __init__(self, server):
        self.server = server
        self.bind_calls = []
        self.acquire_calls = []
        self.release_calls = []
        self.release_route_calls = []
        self.bind_route = _Remote(self._bind)
        self.acquire_lease = _Remote(self._acquire)
        self.release_lease = _Remote(self._release)
        self.release_route = _Remote(self._release_route)

    async def _bind(self, **kwargs):
        self.bind_calls.append(kwargs)
        return kwargs["min_weight_version"]

    async def _acquire(self, **kwargs):
        self.acquire_calls.append(kwargs)
        return "server-0", self.server

    async def _release(self, **kwargs):
        self.release_calls.append(kwargs)

    def _release_route(self, **kwargs):
        self.release_route_calls.append(kwargs)
        return "released"


def _client(router, *, client_wait=None):
    config = SimpleNamespace()
    if client_wait is not None:
        config.rollout_router = SimpleNamespace(client_wait=client_wait)
    return LLMServerClient(config=config, router=router)


def test_client_requires_router_and_valid_session_identity() -> None:
    with pytest.raises(ValueError, match="GlobalRequestRouter"):
        LLMServerClient(config=SimpleNamespace(), router=None)
    for invalid in (None, "", True):
        with pytest.raises(ValueError, match="session_id"):
            resolve_session_id(invalid)
    with pytest.raises(ValueError, match="must match"):
        resolve_session_id("one", request_id="two")


@pytest.mark.asyncio
async def test_bind_uses_one_api_and_treats_version_as_a_minimum() -> None:
    router = _Router(_Server())
    client = _client(router)
    assert await client.bind_route("session", weight_version=3) == 3
    assert await client.bind_route(request_id="other") is None
    assert router.bind_calls == [
        {"session_id": "session", "min_weight_version": 3},
        {"session_id": "other", "min_weight_version": None},
    ]


@pytest.mark.parametrize("weight_version", [True, "3", 3.0, -1])
def test_bind_rejects_non_builtin_nonnegative_versions(weight_version) -> None:
    client = _client(_Router(_Server()))
    with pytest.raises(ValueError, match="weight_version"):
        client.bind_route("session", weight_version=weight_version)


@pytest.mark.asyncio
async def test_generate_uses_one_committed_lease_and_releases_it() -> None:
    server = _Server()
    router = _Router(server)
    output = await _client(router).generate(
        session_id="session",
        weight_version=3,
        prompt_ids=[1, 2],
        sampling_params={"temperature": 0.0},
    )
    lease_id = server.calls[0]["request_id"]
    assert router.acquire_calls == [{"session_id": "session", "lease_id": lease_id, "weight_version": 3}]
    assert router.release_calls == [{"session_id": "session", "lease_id": lease_id}]
    assert server.calls[0]["session_id"] == "session"
    assert output.extra_fields == {
        "global_steps": 3,
        "session_id": "session",
        "request_id": lease_id,
        "server_id": "server-0",
        "weight_version": 3,
        "min_global_steps": 3,
        "max_global_steps": 3,
    }


@pytest.mark.asyncio
async def test_cancelled_acquire_settles_remote_outcome_then_releases_without_backend_call() -> None:
    server = _Server()
    router = _Router(server)
    started = asyncio.Event()
    acknowledged = asyncio.Event()
    committed = False

    async def acquire(**kwargs):
        nonlocal committed
        router.acquire_calls.append(kwargs)
        started.set()
        await acknowledged.wait()
        committed = True
        return "server-0", server

    router.acquire_lease = _Remote(acquire)
    task = asyncio.create_task(_client(router).generate(session_id="session", prompt_ids=[1], sampling_params={}))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert router.release_calls == []
    acknowledged.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert committed
    assert router.release_calls == [{"session_id": "session", "lease_id": router.acquire_calls[0]["lease_id"]}]
    assert server.calls == []


@pytest.mark.asyncio
async def test_unavailable_acquire_retries_with_the_same_lease() -> None:
    server = _Server()
    router = _Router(server)
    attempts = 0

    async def acquire(**kwargs):
        nonlocal attempts
        router.acquire_calls.append(kwargs)
        attempts += 1
        if attempts < 3:
            raise RouteUnavailableError("not ready")
        return "server-0", server

    router.acquire_lease = _Remote(acquire)
    client = _client(
        router,
        client_wait={
            "timeout_s": 1,
            "planned_pause_timeout_s": 1,
            "initial_backoff_s": 0.001,
            "max_backoff_s": 0.001,
        },
    )
    await client.generate(session_id="session", prompt_ids=[1], sampling_params={})
    assert len(router.acquire_calls) == 3
    assert len({call["lease_id"] for call in router.acquire_calls}) == 1
    assert len(router.release_calls) == 1


@pytest.mark.asyncio
async def test_acquire_timeout_still_settles_and_releases_a_late_commit() -> None:
    server = _Server()
    router = _Router(server)
    started = asyncio.Event()
    acknowledged = asyncio.Event()

    async def acquire(**kwargs):
        router.acquire_calls.append(kwargs)
        started.set()
        await acknowledged.wait()
        return "server-0", server

    router.acquire_lease = _Remote(acquire)
    task = asyncio.create_task(
        _client(router).generate(
            session_id="session",
            prompt_ids=[1],
            sampling_params={},
            remaining_timeout_s=0.01,
        )
    )
    await started.wait()
    await asyncio.sleep(0.02)
    assert not task.done()
    acknowledged.set()
    with pytest.raises(TimeoutError):
        await task
    assert len(router.release_calls) == 1
    assert server.calls == []


@pytest.mark.asyncio
async def test_backend_cancellation_settles_generation_before_release() -> None:
    started = asyncio.Event()
    finished = asyncio.Event()

    class BlockingServer(_Server):
        async def _generate(self, **kwargs):
            self.calls.append(kwargs)
            started.set()
            await finished.wait()
            return self.outputs.pop(0)

    server = BlockingServer()
    router = _Router(server)
    task = asyncio.create_task(_client(router).generate(session_id="session", prompt_ids=[1], sampling_params={}))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert router.release_calls == []
    finished.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(router.release_calls) == 1


@pytest.mark.asyncio
async def test_release_cancellation_settles_ack_before_propagating() -> None:
    router = _Router(_Server())
    started = asyncio.Event()
    acknowledged = asyncio.Event()

    async def release(**kwargs):
        router.release_calls.append(kwargs)
        started.set()
        await acknowledged.wait()

    router.release_lease = _Remote(release)
    task = asyncio.create_task(_client(router).generate(session_id="session", prompt_ids=[1], sampling_params={}))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    acknowledged.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(router.release_calls) == 1


@pytest.mark.asyncio
async def test_aborted_segments_continue_and_record_the_version_span() -> None:
    server = _Server(
        [
            TokenOutput(
                token_ids=[10, 11],
                log_probs=[-0.1, -0.2],
                stop_reason="aborted",
                extra_fields={"global_steps": 3},
            ),
            TokenOutput(
                token_ids=[12],
                log_probs=[-0.3],
                stop_reason="completed",
                extra_fields={"global_steps": 4},
            ),
        ]
    )
    output = await _client(_Router(server)).generate(
        session_id="session",
        prompt_ids=[1],
        sampling_params={"max_tokens": 8, "logprobs": True},
    )
    assert [call["prompt_ids"] for call in server.calls] == [[1], [1, 10, 11]]
    assert server.calls[1]["sampling_params"]["max_tokens"] == 6
    assert output.token_ids == [10, 11, 12]
    assert output.log_probs == [-0.1, -0.2, -0.3]
    assert output.extra_fields["min_global_steps"] == 3
    assert output.extra_fields["max_global_steps"] == 4


@pytest.mark.parametrize("global_steps", [None, 2, 4])
@pytest.mark.asyncio
async def test_versioned_generate_rejects_backend_version_mismatch(global_steps) -> None:
    router = _Router(
        _Server([TokenOutput(token_ids=[1], stop_reason="completed", extra_fields={"global_steps": global_steps})])
    )
    with pytest.raises((ValueError, RuntimeError), match="global_steps"):
        await _client(router).generate(
            session_id="session",
            weight_version=3,
            prompt_ids=[1],
            sampling_params={},
        )
    assert len(router.release_calls) == 1


def test_release_route_forwards_the_session_identity() -> None:
    router = _Router(_Server())
    client = _client(router)
    assert client.release_route("session") == "released"
    assert client.release_route(request_id="other") == "released"
    assert router.release_route_calls == [{"session_id": "session"}, {"session_id": "other"}]


@pytest.mark.asyncio
async def test_cancelled_bind_backoff_finishes_before_route_cleanup(monkeypatch) -> None:
    from laminar.global_request_router import GlobalRequestRouter

    owner = GlobalRequestRouter({"server-0": object()})
    backoff_started = asyncio.Event()
    sleep = asyncio.sleep

    async def backoff(delay):
        backoff_started.set()
        await sleep(delay)

    monkeypatch.setattr("laminar.llm_server_client.asyncio.sleep", backoff)

    async def bind(**kwargs):
        return owner.bind_route(**kwargs)

    async def release(**kwargs):
        return owner.release_route(**kwargs)

    router = SimpleNamespace(bind_route=_Remote(bind), release_route=_Remote(release))
    client = _client(
        router,
        client_wait={
            "timeout_s": 5,
            "planned_pause_timeout_s": 5,
            "initial_backoff_s": 1,
            "max_backoff_s": 1,
        },
    )
    binding = asyncio.create_task(client.bind_route("unique-session", weight_version=3))
    try:
        await asyncio.wait_for(backoff_started.wait(), timeout=1)
        binding.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(binding), timeout=1)
        await asyncio.wait_for(client.release_route("unique-session"), timeout=1)
        owner.mark_active("server-0", weight_version=3)
        assert owner.get_active_session_count() == 0
        assert owner._session_id_to_server == {}
    finally:
        binding.cancel()
        await asyncio.gather(binding, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_bind_rpc_settles_before_route_cleanup() -> None:
    from laminar.global_request_router import GlobalRequestRouter

    owner = GlobalRequestRouter({"server-0": object()})
    owner.mark_active("server-0", weight_version=3)
    committed, allow_reply = asyncio.Event(), asyncio.Event()
    releases = []

    async def bind(**kwargs):
        result = owner.bind_route(**kwargs)
        committed.set()
        await allow_reply.wait()
        return result

    async def release(**kwargs):
        releases.append(kwargs)
        return owner.release_route(**kwargs)

    client = _client(SimpleNamespace(bind_route=_Remote(bind), release_route=_Remote(release)))
    binding = asyncio.create_task(client.bind_route("unique-session", weight_version=3))
    try:
        await asyncio.wait_for(committed.wait(), timeout=1)
        binding.cancel()
        await asyncio.sleep(0)
        assert not binding.done()
        assert releases == []
        assert owner.get_active_session_count() == 1
        allow_reply.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(binding), timeout=1)
        await asyncio.wait_for(client.release_route("unique-session"), timeout=1)
        assert releases == [{"session_id": "unique-session"}]
        assert owner.get_active_session_count() == 0
    finally:
        allow_reply.set()
        binding.cancel()
        await asyncio.gather(binding, return_exceptions=True)


@pytest.mark.asyncio
async def test_generate_retries_after_recoverable_engine_death_and_preserves_aborted_prefix() -> None:
    import ray
    from vllm.v1.engine.exceptions import EngineDeadError

    calls = []
    outcomes = [
        TokenOutput(
            token_ids=[10, 11],
            log_probs=[-0.1, -0.2],
            stop_reason="aborted",
            extra_fields={"global_steps": 3},
        ),
        ray.exceptions.RayTaskError("generate", "engine died", EngineDeadError()),
        TokenOutput(
            token_ids=[12],
            log_probs=[-0.3],
            stop_reason="completed",
            extra_fields={"global_steps": 3},
        ),
    ]

    async def generate(**kwargs):
        calls.append(kwargs)
        item = outcomes.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    server = SimpleNamespace(generate=_Remote(generate))
    router = _Router(server)
    client = _client(
        router,
        client_wait={
            "timeout_s": 1,
            "planned_pause_timeout_s": 1,
            "initial_backoff_s": 0.001,
            "max_backoff_s": 0.002,
        },
    )
    output = await client.generate(
        session_id="session",
        weight_version=3,
        prompt_ids=[1],
        sampling_params={"max_tokens": 8, "logprobs": True},
    )
    assert [call["prompt_ids"] for call in calls] == [[1], [1, 10, 11], [1, 10, 11]]
    assert calls[1]["sampling_params"]["max_tokens"] == 6
    assert calls[2]["sampling_params"]["max_tokens"] == 6
    assert len(router.acquire_calls) == 3
    assert len(router.release_calls) == 3
    assert output.token_ids == [10, 11, 12]
    assert output.log_probs == [-0.1, -0.2, -0.3]
    assert output.extra_fields["weight_version"] == 3


@pytest.mark.asyncio
async def test_generate_cancelled_during_engine_death_propagates_cancellation_after_release() -> None:
    from vllm.v1.engine.exceptions import EngineDeadError

    started = asyncio.Event()
    fail_now = asyncio.Event()

    async def generate(**kwargs):
        started.set()
        await fail_now.wait()
        raise EngineDeadError()

    router = _Router(SimpleNamespace(generate=_Remote(generate)))
    task = asyncio.create_task(_client(router).generate(session_id="session", prompt_ids=[1], sampling_params={}))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    fail_now.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(router.acquire_calls) == 1
    assert len(router.release_calls) == 1


@pytest.mark.asyncio
async def test_generate_engine_death_respects_wait_timeout_and_releases_leases() -> None:
    import ray

    async def generate(**kwargs):
        raise ray.exceptions.RayActorError()

    router = _Router(SimpleNamespace(generate=_Remote(generate)))
    client = _client(
        router,
        client_wait={
            "timeout_s": 0.02,
            "planned_pause_timeout_s": 0.02,
            "initial_backoff_s": 0.005,
            "max_backoff_s": 0.005,
        },
    )
    with pytest.raises(TimeoutError, match="Router availability wait expired"):
        await client.generate(session_id="session", prompt_ids=[1], sampling_params={})
    assert len(router.acquire_calls) == len(router.release_calls) >= 1
