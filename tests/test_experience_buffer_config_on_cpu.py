# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf
from verl.utils.config import omega_conf_to_dataclass


def _buffer_cfg(**overrides):
    return OmegaConf.create({"_target_": "laminar.train.experience_buffer.config.ExperienceBufferConfig", **overrides})


def test_experience_buffer_config_preserves_explicit_behavior():
    buffer_config = omega_conf_to_dataclass(
        _buffer_cfg(
            max_size=128,
            sampler="version_lifo",
            store_namespace="rollout-train",
            poll_interval=0.25,
        )
    )

    assert (
        buffer_config.max_size,
        buffer_config.sampler,
        buffer_config.store_namespace,
        buffer_config.poll_interval,
    ) == (128, "version_lifo", "rollout-train", 0.25)


def test_experience_buffer_config_defaults_to_fifo_sampler():
    buffer_config = omega_conf_to_dataclass(_buffer_cfg())

    assert buffer_config.sampler == "fifo"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_size": 0}, "max_size must be positive"),
        ({"max_size": None}, "max_size"),
        ({"old_ordering": "submission"}, "old_ordering"),
        ({"sampler": "unknown_sampler"}, "unknown_sampler"),
        ({"store_namespace": ""}, "store_namespace must be non-empty"),
    ],
)
def test_experience_buffer_config_fails_fast(overrides, message):
    with pytest.raises(Exception, match=message):
        omega_conf_to_dataclass(_buffer_cfg(**overrides))
