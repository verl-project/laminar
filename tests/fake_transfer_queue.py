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

"""Fake TransferQueue helpers for async-RL unit tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import laminar.train.experience_buffer.client as buffer_client
import laminar.train.experience_buffer.server as buffer_server


class FakeTQ:
    """Recording TQ stand-in with optional per-method handlers."""

    def __init__(
        self,
        *,
        get_many_handler: Callable[..., Any] | None = None,
        get_many_sync_handler: Callable[..., Any] | None = None,
        release_handler: Callable[..., Any] | None = None,
        list_metadata_handler: Callable[..., Any] | None = None,
        metadata_by_namespace: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
    ) -> None:
        self.get_many_calls: list[dict[str, Any]] = []
        self.get_many_sync_calls: list[dict[str, Any]] = []
        self.release_calls: list[tuple[list[str], str]] = []
        self.list_metadata_calls: list[str | None] = []
        self.init_calls: list[Any] = []
        self.close_calls = 0
        self._get_many_handler = get_many_handler
        self._get_many_sync_handler = get_many_sync_handler
        self._release_handler = release_handler
        self._list_metadata_handler = list_metadata_handler
        self.metadata_by_namespace: dict[str, dict[str, dict[str, Any]]] = {
            str(ns): {str(key): dict(meta) for key, meta in entries.items()}
            for ns, entries in dict(metadata_by_namespace or {}).items()
        }

    def init(self, config: Any | None = None) -> None:
        self.init_calls.append(config)

    def close(self) -> None:
        self.close_calls += 1

    async def async_kv_list(self, partition_id: str | None = None) -> dict[str, dict[str, dict[str, Any]]]:
        self.list_metadata_calls.append(partition_id)
        if self._list_metadata_handler is not None:
            return await self._list_metadata_handler(namespace=partition_id)
        if partition_id is None:
            return {ns: dict(entries) for ns, entries in self.metadata_by_namespace.items()}
        return {partition_id: dict(self.metadata_by_namespace.get(partition_id, {}))}

    async def async_kv_batch_get(
        self,
        keys: Sequence[str],
        partition_id: str,
        select_fields: Sequence[str] | None = None,
    ) -> Any:
        self.get_many_calls.append(
            {
                "keys": list(keys),
                "namespace": partition_id,
                "fields": None if select_fields is None else list(select_fields),
            }
        )
        if self._get_many_handler is None:
            raise AssertionError("FakeTQ.async_kv_batch_get handler not configured")
        return await self._get_many_handler(keys=list(keys), namespace=partition_id, fields=select_fields)

    def kv_batch_get(
        self,
        keys: Sequence[str],
        partition_id: str,
        select_fields: Sequence[str] | None = None,
    ) -> Any:
        self.get_many_sync_calls.append(
            {
                "keys": list(keys),
                "namespace": partition_id,
                "fields": None if select_fields is None else list(select_fields),
            }
        )
        if self._get_many_sync_handler is None:
            raise AssertionError("FakeTQ.kv_batch_get handler not configured")
        return self._get_many_sync_handler(keys=list(keys), namespace=partition_id, fields=select_fields)

    def kv_clear(self, keys: Sequence[str], partition_id: str) -> None:
        key_list = list(keys)
        self.release_calls.append((key_list, partition_id))
        bucket = self.metadata_by_namespace.get(partition_id)
        if bucket is not None:
            for key in key_list:
                bucket.pop(key, None)
        if self._release_handler is not None:
            result = self._release_handler(key_list, namespace=partition_id)
            if hasattr(result, "__await__"):
                raise TypeError("FakeTQ.kv_clear is synchronous; release_handler must not be async")


def install_fake_tq(monkeypatch, engine: FakeTQ | None = None, **handlers) -> FakeTQ:
    """Install ``engine`` as the process-local TransferQueue for one test."""
    fake = engine if engine is not None else FakeTQ(**handlers)
    monkeypatch.setattr(buffer_server, "tq", fake)
    monkeypatch.setattr(buffer_client, "tq", fake)
    return fake
