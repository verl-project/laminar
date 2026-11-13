# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Session routing must preserve vLLM's per-request engine and OpenAI contracts.

The HTTP tests use the installed vLLM schema and ASGI handler, with only model
serving stubbed. They do not start an inference engine or exercise GPU generation.
"""

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI, Request
from omegaconf import OmegaConf

pytest.importorskip("vllm")

from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

from laminar.global_request_router import GlobalRequestRouter
from laminar.llm_server_client import LLMServerClient


@pytest.mark.asyncio
@pytest.mark.parametrize("identity_key", ["session_id", "request_id"])
async def test_session_routing_preserves_vllm_generate_signature(identity_key):
    engine_request_ids = []
    signature = inspect.signature(vLLMHttpServer.generate)

    async def generate(**kwargs):
        # Bind the real server API: an accidental session_id kwarg must fail,
        # even though the lightweight remote-call wrapper accepts **kwargs.
        arguments = signature.bind(None, **kwargs).arguments
        engine_request_ids.append(arguments["request_id"])
        return TokenOutput(token_ids=[2], stop_reason="completed", extra_fields={"global_steps": 3})

    server = SimpleNamespace(generate=SimpleNamespace(remote=generate))
    routing_owner = GlobalRequestRouter(servers={"replica-0": server})
    routing_owner.mark_active(server_id="replica-0", weight_version=3)
    routing_owner.bind_route("conversation-1", min_weight_version=3)

    def remote_method(name):
        async def invoke(**kwargs):
            value = getattr(routing_owner, name)(**kwargs)
            return await value if inspect.isawaitable(value) else value

        return SimpleNamespace(remote=AsyncMock(side_effect=invoke))

    router = SimpleNamespace(
        **{
            name: remote_method(name)
            for name in (
                "acquire_lease",
                "release_lease",
            )
        }
    )
    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"name": "vllm"}}})
    client = LLMServerClient(config=config, router=router)

    for _ in range(2):
        result = await client.generate(
            **{identity_key: "conversation-1"}, prompt_ids=[1], sampling_params={}, weight_version=3
        )
        assert result.token_ids == [2]
        assert result.extra_fields["session_id"] == "conversation-1"
        assert result.extra_fields["request_id"] == engine_request_ids[-1]

    assert len(engine_request_ids) == len(set(engine_request_ids)) == 2
    assert "conversation-1" not in engine_request_ids
    assert routing_owner._inflight_count("replica-0") == 0
    assert router.acquire_lease.remote.await_count == router.release_lease.remote.await_count == 2
    router.acquire_lease.remote.assert_awaited_with(
        session_id="conversation-1", lease_id=engine_request_ids[-1], weight_version=3
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("identity_source", ["generated", "body", "header"])
async def test_native_openai_request_and_response_ids_remain_per_request(identity_source):
    # Keep optional HTTP layouts from preventing collection of the client test.
    api = pytest.importorskip("vllm.entrypoints.openai.chat_completion.api_router")
    protocol = pytest.importorskip("vllm.entrypoints.openai.chat_completion.protocol")
    try:
        from vllm.entrypoints.serve.engine.serving import BaseServing
    except ModuleNotFoundError as error:
        if error.name not in {"vllm.entrypoints.serve.engine", "vllm.entrypoints.serve.engine.serving"}:
            raise
        # vLLM 0.18 keeps the same serving helper under the OpenAI package.
        BaseServing = pytest.importorskip("vllm.entrypoints.openai.engine.serving").OpenAIServing
    ChatCompletionRequest = protocol.ChatCompletionRequest
    ChatCompletionResponse = protocol.ChatCompletionResponse
    received = []

    class Serving:
        async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
            received.append(request)
            request_id = BaseServing._base_request_id(raw_request, request.request_id)
            return ChatCompletionResponse(
                id=f"chatcmpl-{request_id}",
                model=request.model,
                choices=[{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            )

    app = FastAPI()
    app.state.openai_serving_chat = Serving()
    app.include_router(api.router)
    responses = []
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for turn in range(2):
            payload = {"model": "model", "messages": [{"role": "user", "content": "hello"}]}
            if identity_source != "generated":
                payload["request_id"] = f"body-turn-{turn}"
            headers = {"X-Request-Id": f"header-turn-{turn}"} if identity_source == "header" else {}
            response = await client.post("/v1/chat/completions", json=payload, headers=headers)
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["object"] == "chat.completion"
            assert body["choices"][0]["message"]["content"] == "ok"
            assert "session_id" not in body
            responses.append(body)

    assert len(received) == 2
    for turn, request in enumerate(received):
        assert isinstance(request, ChatCompletionRequest)
        assert "session_id" not in request.model_fields_set
        if identity_source != "generated":
            assert request.request_id == f"body-turn-{turn}"
        expected_id = f"header-turn-{turn}" if identity_source == "header" else request.request_id
        assert responses[turn]["id"] == f"chatcmpl-{expected_id}"
    assert received[0].request_id != received[1].request_id
    assert responses[0]["id"] != responses[1]["id"]
