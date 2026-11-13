# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Real Gateway, Laminar client and small-capacity Router terminal ownership."""

import asyncio
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import ray
from verl.workers.rollout.replica import TokenOutput

from laminar.global_request_router import GlobalRequestRouter, GlobalRequestRouterConfig, SessionRouteClosedError
from laminar.llm_server_client import LLMServerClient


class _CharacterTokenizer:
    eos_token_id = ord("\n")

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True, **kwargs):
        text = "".join(f"{message['role']}:{message.get('content', '')}\n" for message in messages)
        if add_generation_prompt:
            text += "assistant:"
        return [ord(character) for character in text] if tokenize else text

    def decode(self, token_ids, **kwargs):
        return "".join(chr(int(token_id)) for token_id in token_ids)

    def encode(self, text, **kwargs):
        return [ord(character) for character in text]


class _InspectableRouter(GlobalRequestRouter):
    def state(self):
        return {
            "sessions": self.get_active_session_count(),
            "routes": len(self._session_id_to_server),
            "leases": sum(len(session.leases) for session in self._sessions.values()),
        }


@ray.remote(max_concurrency=4)
class _GatedServer:
    def __init__(self):
        self.entered = False
        self.allow = asyncio.Event()
        self.calls = 0

    async def generate(self, **kwargs):
        self.entered = True
        self.calls += 1
        await self.allow.wait()
        return TokenOutput(token_ids=[65], stop_reason="aborted", extra_fields={"global_steps": 1})

    def state(self):
        return self.entered, self.calls

    def finish(self):
        self.allow.set()


class _LocalRemote:
    def __init__(self, function):
        self.function = function

    def remote(self, *args, **kwargs):
        async def invoke():
            return self.function(*args, **kwargs)

        return invoke()


class _LocalHandle:
    def __init__(self, owner):
        self.owner = owner

    def __getattr__(self, name):
        return _LocalRemote(getattr(self.owner, name))


@pytest.fixture(scope="module")
def local_ray():
    assert not ray.is_initialized(), "This suite owns an isolated local runtime"
    ray.init(
        address="local",
        num_cpus=4,
        include_dashboard=False,
        _temp_dir=tempfile.mkdtemp(prefix="session-owner-"),
        object_store_memory=128 * 1024 * 1024,
        runtime_env={
            "env_vars": {
                "PYTHONPATH": str(Path(__file__).parent) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        },
    )
    yield
    ray.shutdown()


@pytest.mark.asyncio
async def test_local_gateway_create_generate_finalize_clears_route_and_lease():
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import _GatewayActor

    class Server:
        def __init__(self):
            self.generate = _LocalRemote(self._generate)

        def _generate(self, **_kwargs):
            return TokenOutput(token_ids=[65], stop_reason="completed", extra_fields={"global_steps": 1})

    server = Server()
    owner = _InspectableRouter({"s0": server})
    owner.mark_active("s0", weight_version=1)
    gateway = _GatewayActor(
        GatewayActorConfig(tokenizer=_CharacterTokenizer()),
        LLMServerClient(SimpleNamespace(), _LocalHandle(owner)),
    )
    gateway._server_base_url = "http://test"

    await gateway.create_session("session", sampling_params={"max_tokens": 4}, weight_version=1)
    response = await gateway._handle_openai_chat_completions(
        "session",
        {"messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 200
    assert owner.state() == {"sessions": 1, "routes": 1, "leases": 0}
    await gateway.finalize_session("session")
    assert owner.state() == {"sessions": 0, "routes": 0, "leases": 0}


@pytest.mark.asyncio
async def test_actual_gateway_abort_settles_laminar_generation_then_fences_late_router_requests(local_ray):
    from uni_agent.gateway.config import GatewayActorConfig
    from uni_agent.gateway.gateway import _GatewayActor

    server = _GatedServer.remote()
    router = ray.remote(_InspectableRouter).remote(
        {"s0": server},
        config=GlobalRequestRouterConfig(continuation="latest_version"),
    )
    generation = abort = None
    try:
        await router.mark_active.remote("s0", weight_version=1)
        client = LLMServerClient(SimpleNamespace(), router)
        gateway = _GatewayActor(GatewayActorConfig(tokenizer=_CharacterTokenizer()), client)
        gateway._server_base_url = "http://test"
        await gateway.create_session("session", sampling_params={"max_tokens": 4})
        generation = asyncio.create_task(
            gateway._handle_openai_chat_completions(
                "session",
                {"messages": [{"role": "user", "content": "hello"}]},
            )
        )
        deadline = time.monotonic() + 10
        while not (await server.state.remote())[0]:
            assert time.monotonic() < deadline, "backend generation did not begin"
            await asyncio.sleep(0.01)
        abort = asyncio.create_task(gateway.abort_session("session"))
        # Repeated caller cancellation must leave the shared generation cleanup intact.
        for _ in range(2):
            await asyncio.sleep(0)
            abort.cancel()
        state = await router.state.remote()
        assert state == {"sessions": 1, "routes": 1, "leases": 1}
        assert not abort.done()
        # Backend completion, not session closure, makes request occupancy releasable.
        await server.finish.remote()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(abort, timeout=10)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(generation, timeout=10)
        assert (await server.state.remote())[1] == 1  # no partial-output continuation after abort
        assert await router.state.remote() == {"sessions": 0, "routes": 0, "leases": 0}
        assert gateway._sessions == {}
        # A late acquire after terminal cleanup cannot revive the route.
        with pytest.raises(ray.exceptions.RayTaskError) as error:
            await router.acquire_lease.remote("session", lease_id="late-request")
        assert isinstance(error.value.as_instanceof_cause(), SessionRouteClosedError)
        assert await router.state.remote() == {"sessions": 0, "routes": 0, "leases": 0}
    finally:
        await server.finish.remote()
        if generation is not None:
            await asyncio.gather(generation, return_exceptions=True)
        if abort is not None:
            await asyncio.gather(abort, return_exceptions=True)
        ray.kill(router, no_restart=True)
        ray.kill(server, no_restart=True)
