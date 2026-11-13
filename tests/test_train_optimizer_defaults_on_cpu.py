# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Compose optimizer defaults without importing accelerator runtime modules."""

from importlib.util import find_spec
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

_TRAIN_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "src/laminar/train/config")


def _compose_train(*overrides: str):
    # Hydra's pkg:// search path imports verl.__init__, which loads torch.
    # Config composition only needs its installed YAML files.
    verl_spec = find_spec("verl")
    assert verl_spec is not None and verl_spec.submodule_search_locations
    verl_config_dir = Path(next(iter(verl_spec.submodule_search_locations))) / "trainer/config"
    with initialize_config_dir(config_dir=_TRAIN_CONFIG_DIR, version_base=None):
        return compose(
            config_name="async_rl",
            overrides=[f"hydra.searchpath=[file://{verl_config_dir}]", *overrides],
        )


@pytest.mark.parametrize("use_checkpoint_scheduler", [None, False, True])
def test_megatron_checkpoint_scheduler_default_is_overridable(use_checkpoint_scheduler):
    overrides = ["model_engine=megatron"]
    if use_checkpoint_scheduler is not None:
        overrides.append(f"actor_rollout_ref.actor.optim.use_checkpoint_opt_param_scheduler={use_checkpoint_scheduler}")

    config = _compose_train(*overrides)

    expected = True if use_checkpoint_scheduler is None else use_checkpoint_scheduler
    assert config.actor_rollout_ref.actor.optim.use_checkpoint_opt_param_scheduler is expected


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2"])
def test_fsdp_does_not_receive_megatron_optimizer_defaults(strategy):
    config = _compose_train("model_engine=dp", f"actor_rollout_ref.actor.strategy={strategy}")

    assert config.actor_rollout_ref.actor.optim._target_ == "verl.workers.config.FSDPOptimizerConfig"
    assert "use_checkpoint_opt_param_scheduler" not in config.actor_rollout_ref.actor.optim
