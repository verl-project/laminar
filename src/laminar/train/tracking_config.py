# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Portable configuration projection for backends with bounded key lengths."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_VALUE_PREFIX = "laminar_cfg_value."
_PATH_PREFIX = "laminar_cfg_path."
_JSON_PREFIX = "laminar_cfg_json."
_CREDENTIAL_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "token",
    "password",
    "secret",
    "secret_key",
    "credentials",
    "authorization",
    "auth_headers",
    "authentication_headers",
}


def sanitize_config(config: Any) -> tuple[Any, list[str]]:
    """Copy the resolved configuration, redacting runtime-environment and credential values."""
    omitted: list[str] = []

    def visit(value: Any, path: tuple[str, ...]) -> Any:
        if isinstance(value, Mapping):
            result = {}
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValueError("Tracking configuration keys must be strings")
                child_path = (*path, key)
                name = key.lower()
                if name == "runtime_env":
                    result[key] = "[OMITTED_RUNTIME_ENV]"
                    omitted.append(".".join(child_path))
                elif (
                    name in _CREDENTIAL_KEYS or name.endswith(("_api_key", "_token", "_password", "_secret"))
                ) and child not in (None, "", {}, []):
                    result[key] = "[REDACTED_CREDENTIAL]"
                    omitted.append(".".join(child_path))
                else:
                    result[key] = visit(child, child_path)
            return result
        if isinstance(value, list):
            return [visit(child, (*path, str(index))) for index, child in enumerate(value)]
        return copy.deepcopy(value)

    return visit(config, ()), omitted


def flatten_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten dictionaries with dot paths, rejecting ambiguous literal dot keys."""
    flat: dict[str, Any] = {}

    def visit(value: Mapping[str, Any], prefix: str = "") -> None:
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("Tracking configuration keys must be strings")
            path = prefix + key
            if isinstance(child, Mapping) and child:
                visit(child, path + ".")
            else:
                if path in flat:
                    raise ValueError(f"Ambiguous Tracking configuration path: {path}")
                flat[path] = copy.deepcopy(child)

    visit(config)
    return flat


@dataclass(frozen=True)
class ConfigProjection:
    logger_config: dict[str, Any]
    path_mapping: dict[str, str]
    json_value_paths: tuple[str, ...]
    metadata_keys: tuple[str, ...]

    def restore_flat_config(self) -> dict[str, Any]:
        """Recover original flattened keys and values without metadata entries."""
        restored = {}
        for key, value in flatten_config(self.logger_config).items():
            if key in self.metadata_keys:
                continue
            path = self.path_mapping.get(key, key)
            restored[path] = json.loads(value) if path in self.json_value_paths else copy.deepcopy(value)
        return restored


def project_config(config: Mapping[str, Any], max_bytes: int) -> ConfigProjection:
    """Keep short paths; alias long paths with reversible, collision-safe metadata."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 32:
        raise ValueError("tracker.config_key_max_bytes must be null or an integer >= 32")
    flat = flatten_config(config)
    occupied = set(flat)
    projected: dict[str, Any] = {}
    mapping: dict[str, str] = {}
    json_paths: list[str] = []
    metadata: list[str] = []
    digest_size = min(32, max_bytes - max(map(len, (_VALUE_PREFIX, _PATH_PREFIX, _JSON_PREFIX))))

    def allocate(path: str, prefixes: tuple[str, ...]) -> tuple[str, ...]:
        digest = hashlib.sha256(path.encode("utf-8")).hexdigest()[:digest_size]
        for attempt in range(len(occupied) + 1):
            suffix = digest if attempt == 0 else digest[:-9] + f"-{attempt:08x}"
            keys = tuple(prefix + suffix for prefix in prefixes)
            if not any(key in occupied for key in keys):
                occupied.update(keys)
                return keys
        raise ValueError("Unable to allocate collision-free Tracking config aliases")

    for path, value in sorted(flat.items()):
        if len(path.encode("utf-8")) > max_bytes:
            key, path_key = allocate(path, (_VALUE_PREFIX, _PATH_PREFIX))
            mapping[key] = path
            projected[path_key] = path
            metadata.append(path_key)
        else:
            key = path
        # A second backend flatten must not silently drop an empty dictionary.
        if isinstance(value, Mapping) or (value is None and "." not in key):
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            json_paths.append(path)
            (type_key,) = allocate(path, (_JSON_PREFIX,))
            projected[type_key] = path
            metadata.append(type_key)
        projected[key] = value
    # Keep root values as nonempty dictionaries where possible. Backends may
    # skip a top-level null while retaining null leaves in a nested config.
    nested: dict[str, Any] = {}
    for path, value in projected.items():
        parts = path.split(".")
        cursor = nested
        for part in parts[:-1]:
            existing = cursor.setdefault(part, {})
            if not isinstance(existing, dict):
                raise ValueError(f"Conflicting Tracking configuration path prefix: {path}")
            cursor = existing
        if parts[-1] in cursor:
            raise ValueError(f"Conflicting Tracking configuration path prefix: {path}")
        cursor[parts[-1]] = value
    return ConfigProjection(nested, mapping, tuple(json_paths), tuple(metadata))
