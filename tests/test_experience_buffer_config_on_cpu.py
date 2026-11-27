# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf
from verl.utils.config import omega_conf_to_dataclass

from laminar.train.experience_buffer.config import ExperienceBufferConfig


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
    assert buffer_config.filter_groups_enable is False


@pytest.mark.parametrize("filter_groups", [None, {}, False, {"enable": False, "metric": "acc"}])
def test_algorithm_group_filtering_is_disabled_without_an_enabled_switch(filter_groups):
    config = OmegaConf.create({"experience_buffer": _buffer_cfg(), "algorithm": {"filter_groups": filter_groups}})

    assert ExperienceBufferConfig.from_config(config).filter_groups_enable is False


@pytest.mark.parametrize("filter_groups", [True, {"enable": True, "metric": "reward"}])
def test_algorithm_reward_filtering_is_bound_to_the_buffer(filter_groups):
    config = OmegaConf.create(
        {
            "experience_buffer": _buffer_cfg(max_size=512, sampler="version_lifo"),
            "algorithm": {"filter_groups": filter_groups},
        }
    )
    buffer_config = ExperienceBufferConfig.from_config(config)

    assert buffer_config.filter_groups_enable is True
    assert (buffer_config.max_size, buffer_config.sampler) == (512, "version_lifo")


@pytest.mark.parametrize("metric", [None, "acc", "entropy"])
def test_algorithm_group_filtering_rejects_unsupported_metrics(metric):
    config = OmegaConf.create(
        {
            "experience_buffer": _buffer_cfg(),
            "algorithm": {"filter_groups": {"enable": True, "metric": metric}},
        }
    )

    with pytest.raises(ValueError, match="algorithm.filter_groups.metric=reward"):
        ExperienceBufferConfig.from_config(config)


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
