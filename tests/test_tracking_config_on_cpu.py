# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib

import pytest

from laminar.train.tracking_config import flatten_config, project_config, sanitize_config


def test_projection_roundtrips_nested_config_and_keeps_short_paths():
    config = {
        "trainer": {"total_training_steps": None},
        "global_profiler": {"global_tool_config": {"precision_debugger": {"config_path": "example"}}},
        "empty": {},
        "list": [{"config": "value"}],
    }
    original = copy.deepcopy(config)
    projection = project_config(config, 64)
    assert projection.restore_flat_config() == flatten_config(config)
    assert config == original
    assert projection.logger_config["trainer"]["total_training_steps"] is None
    assert all(len(key.encode("utf-8")) <= 64 for key in flatten_config(projection.logger_config))
    assert "global_profiler.global_tool_config.precision_debugger.config_path" in projection.path_mapping.values()
    assert projection.logger_config["empty"] == "{}"


def test_projection_unicode_limit_uses_bytes():
    key = "汉" * 30
    assert len(key) < 64 < len(key.encode("utf-8"))
    projection = project_config({key: 17}, 64)
    assert projection.restore_flat_config() == {key: 17}
    assert key not in projection.logger_config
    assert all(len(key.encode("utf-8")) <= 64 for key in projection.logger_config)


def test_alias_does_not_overwrite_original_short_path():
    long_path = "x" * 65
    digest = hashlib.sha256(long_path.encode()).hexdigest()[:32]
    original_alias = "laminar_cfg_value." + digest
    original_mapping_alias = "laminar_cfg_path." + digest
    config = {long_path: [1, 2], original_alias: "original", original_mapping_alias: "original mapping"}
    projection = project_config(config, 64)
    assert projection.restore_flat_config() == config
    flat = flatten_config(projection.logger_config)
    assert flat[original_alias] == "original"
    assert flat[original_mapping_alias] == "original mapping"


def test_hash_collisions_have_deterministic_distinct_aliases(monkeypatch):
    class _Digest:
        def hexdigest(self):
            return "f" * 64

    monkeypatch.setattr("laminar.train.tracking_config.hashlib.sha256", lambda _data: _Digest())
    first, second = "a" * 65, "b" * 65
    projection = project_config({first: 1, second: 2}, 64)
    reordered = project_config({second: 2, first: 1}, 64)
    assert projection.logger_config == reordered.logger_config
    assert len(projection.path_mapping) == 2
    assert projection.restore_flat_config() == {first: 1, second: 2}


def test_ambiguous_literal_dot_path_is_rejected():
    with pytest.raises(ValueError, match="Ambiguous Tracking configuration path"):
        project_config({"a.b": 1, "a": {"b": 2}}, 64)


@pytest.mark.parametrize("limit", [True, "64", 0, 31])
def test_invalid_projection_limit_is_rejected(limit):
    with pytest.raises(ValueError, match="integer >= 32"):
        project_config({}, limit)


def test_config_sanitization_omits_environments_and_only_nonempty_credentials():
    config = {
        "runtime_env": {"env_vars": {"PRIVATE_VALUE": "private environment"}},
        "trainer": {"token": "private token", "api_key": None, "password": "", "max_tokens": 123},
        "workers": [{"auth_headers": {"Authorization": "private header"}}],
    }
    original = copy.deepcopy(config)
    sanitized, omitted = sanitize_config(config)
    assert config == original
    assert sanitized["trainer"] == {
        "token": "[REDACTED_CREDENTIAL]",
        "api_key": None,
        "password": "",
        "max_tokens": 123,
    }
    assert sanitized["runtime_env"] == "[OMITTED_RUNTIME_ENV]"
    assert omitted == ["runtime_env", "trainer.token", "workers.0.auth_headers"]
