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

"""Typed config for ExperienceBuffer."""

from __future__ import annotations

from dataclasses import dataclass, replace

from omegaconf import DictConfig


@dataclass(frozen=True)
class ExperienceBufferConfig:
    """Typed config surface for the ExperienceBufferServer metadata actor."""

    max_size: int = 4096
    sampler: str = "fifo"
    store_namespace: str = "train"
    poll_interval: float = 1.0
    filter_groups_enable: bool = False

    @classmethod
    def from_config(cls, config: DictConfig) -> ExperienceBufferConfig:
        """Bind reward filtering to the algorithm switch."""
        from verl.utils.config import omega_conf_to_dataclass

        buffer_config: ExperienceBufferConfig = omega_conf_to_dataclass(config.experience_buffer)
        filter_config = config.algorithm.get("filter_groups")
        if isinstance(filter_config, bool):
            enabled = filter_config
        else:
            enabled = filter_config is not None and bool(filter_config.get("enable", False))
            if enabled and filter_config.get("metric") != "reward":
                raise ValueError("async_rl group filtering requires algorithm.filter_groups.metric=reward")
        return replace(buffer_config, filter_groups_enable=enabled)

    def __post_init__(self) -> None:
        if self.max_size is None or self.max_size <= 0:
            raise ValueError(f"experience_buffer.max_size must be positive, got {self.max_size}")
        if self.sampler not in ("fifo", "lifo", "version_lifo"):
            raise ValueError(
                f"experience_buffer.sampler must be 'fifo', 'lifo', or 'version_lifo', got {self.sampler!r}"
            )
        if not self.store_namespace:
            raise ValueError("experience_buffer.store_namespace must be non-empty")
        if self.poll_interval <= 0:
            raise ValueError(f"experience_buffer.poll_interval must be positive, got {self.poll_interval}")
