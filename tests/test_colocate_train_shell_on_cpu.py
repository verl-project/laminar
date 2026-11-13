# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Exercise shared-pool resource accounting and incompatible launch overrides."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir

from laminar.train.main import resolve_train_config

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG_DIR = str((ROOT / "src/laminar/train/config").resolve())


def _capture_recipe_python_argv(script: str, scratch: Path, *cli_overrides: str) -> list[str]:
    fake_bin = scratch / "bin"
    fake_bin.mkdir(exist_ok=True)
    argv_path = scratch / "python-argv.bin"
    python = fake_bin / "python3"
    python.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "${CAPTURED_ARGV_PATH}"\n')
    python.chmod(0o755)
    task_config = scratch / "task_config.yaml"
    task_config.write_text("{}\n")
    result = subprocess.run(
        ["bash", script, *cli_overrides],
        cwd=ROOT,
        env={
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "HOME": os.environ["HOME"],
            "CAPTURED_ARGV_PATH": str(argv_path),
            "EXP_NAME": "colocate-shell-test",
            "RAY_DATA_HOME": str(scratch),
            "TASK_CONFIG_PATH": str(task_config),
        },
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert "python3 -m laminar.train" in result.stdout
    encoded = argv_path.read_bytes().split(b"\0")
    assert encoded[-1] == b""
    return [part.decode() for part in encoded[:-1]]


class ColocateTrainShellTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch = tempfile.TemporaryDirectory(prefix="laminar-colocate-shell-")
        self.addCleanup(self.scratch.cleanup)
        self.scratch_path = Path(self.scratch.name)

    def compose_swe(self, *cli_overrides: str):
        argv = _capture_recipe_python_argv("recipes/swe/train.sh", self.scratch_path, *cli_overrides)
        self.assertEqual(argv[:2], ["-m", "laminar.train"])
        with initialize_config_dir(config_dir=TRAIN_CONFIG_DIR, version_base=None):
            config = compose(config_name="async_rl", overrides=argv[2:])
        return resolve_train_config(config)

    def test_swe_default_uses_one_16_gpu_pool_and_latest_version(self) -> None:
        config = self.compose_swe("training_strategy=colocate_async")
        self.assertEqual(config.training_strategy, "colocate_async")
        self.assertEqual((config.trainer.nnodes, config.trainer.n_gpus_per_node), (2, 8))
        rollout = config.actor_rollout_ref.rollout
        self.assertEqual((rollout.nnodes, rollout.n_gpus_per_node), (2, 8))
        self.assertEqual(
            rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions,
            112,
        )
        self.assertTrue(rollout.free_cache_engine)
        self.assertTrue(rollout.enable_sleep_mode)
        megatron = config.actor_rollout_ref.actor.megatron
        self.assertTrue(megatron.param_offload)
        self.assertTrue(megatron.optimizer_offload)
        self.assertEqual(config.rollout_router.continuation, "latest_version")
        self.assertFalse(config.multi_version_rollout.repack.enable)
        # Keep the science budget while changing resource placement.
        self.assertEqual(config.data.train_batch_size, 16)
        self.assertEqual(rollout.n, 8)
        self.assertEqual(config.trainer.weight_publish_interval, 4)

    def test_generic_fsdp_can_share_a_subnode_pool(self) -> None:
        with initialize_config_dir(config_dir=TRAIN_CONFIG_DIR, version_base=None):
            config = compose(
                config_name="async_rl",
                overrides=[
                    "training_strategy=colocate_async",
                    "trainer.nnodes=1",
                    "trainer.n_gpus_per_node=2",
                    "actor_rollout_ref.rollout.nnodes=1",
                    "actor_rollout_ref.rollout.n_gpus_per_node=2",
                    "actor_rollout_ref.rollout.free_cache_engine=True",
                    "actor_rollout_ref.rollout.enable_sleep_mode=True",
                    "actor_rollout_ref.actor.fsdp_config.fsdp_size=2",
                    "actor_rollout_ref.actor.fsdp_config.param_offload=True",
                    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True",
                    "rollout_router.continuation=latest_version",
                ],
            )
        resolve_train_config(config)
        self.assertEqual((config.trainer.nnodes, config.trainer.n_gpus_per_node), (1, 2))
        self.assertEqual(
            (config.actor_rollout_ref.rollout.nnodes, config.actor_rollout_ref.rollout.n_gpus_per_node),
            (1, 2),
        )
        self.assertTrue(config.actor_rollout_ref.actor.fsdp_config.param_offload)
        self.assertTrue(config.actor_rollout_ref.actor.fsdp_config.optimizer_offload)

    def test_rejects_incompatible_settings_before_starting_ray(self) -> None:
        for override, message in (
            ("rollout_router.continuation=same_version", "requires rollout_router.continuation=latest_version"),
            ("actor_rollout_ref.actor.megatron.param_offload=False", "requires actor parameter and optimizer offload"),
            (
                "actor_rollout_ref.actor.megatron.optimizer_offload=False",
                "requires actor parameter and optimizer offload",
            ),
            (
                "actor_rollout_ref.rollout.enable_sleep_mode=False",
                "requires rollout.enable_sleep_mode and rollout.free_cache_engine",
            ),
            (
                "actor_rollout_ref.rollout.free_cache_engine=False",
                "requires rollout.enable_sleep_mode and rollout.free_cache_engine",
            ),
            ("actor_rollout_ref.rollout.nnodes=4", "requires identical trainer and rollout pool shapes"),
            ("actor_rollout_ref.rollout.n_gpus_per_node=4", "requires identical trainer and rollout pool shapes"),
        ):
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, message):
                    self.compose_swe("training_strategy=colocate_async", override)


if __name__ == "__main__":
    unittest.main()
