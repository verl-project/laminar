# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from tensordict import NonTensorData, TensorDict

from laminar.train.batch_metrics import sampled_metadata_metrics, tensor_stats, train_batch_metrics
from laminar.train.experience_buffer.server import SampledBatch
from laminar.train.trainer import _add_train_throughput_metrics, _extract_actor_update_metrics


def test_stored_context_lengths_are_distinct_from_trainable_action_lengths():
    td = TensorDict(
        {
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 0, 0]]),
            "response_mask": torch.tensor([[1, 0, 0, 1], [1, 0, 0, 0]]),
            "advantages": torch.tensor([[1.0, 0, 0, 2], [-1.0, 0, 0, 0]]),
        },
        batch_size=[2],
    )
    metrics = train_batch_metrics(td)
    assert metrics["train_batch/num_prompt_tokens"] == 3
    assert metrics["train_batch/num_response_tokens"] == 6
    assert metrics["train_batch/num_action_tokens"] == 3
    assert metrics["response_length/mean"] == 3
    assert metrics["train_batch/action_length/mean"] == 1.5
    assert metrics["response_length/p99"] == pytest.approx(3.98)
    assert "train_batch/response_length/clip_ratio" not in metrics
    assert metrics["train_batch/response_length/clip_ratio_valid"] == 0
    assert metrics["train_batch/advantages/count"] == 3
    assert metrics["train_batch/advantages/sum"] == 2


def test_empty_population_is_missing_and_real_zero_is_measured():
    empty = tensor_stats("empty", torch.tensor([]))
    zero = tensor_stats("zero", torch.tensor([0.0]))
    assert empty == {"empty/count": 0.0, "empty/valid": 0.0, "empty/sum": 0.0, "empty/m2": 0.0}
    assert zero["zero/mean"] == 0
    assert zero["zero/valid"] == 1
    invalid = tensor_stats("invalid", torch.tensor([float("nan")]))
    assert invalid["invalid/valid"] == 0
    assert "invalid/mean" not in invalid


def test_sample_versions_and_truncation_use_only_consumed_metadata():
    batch = SampledBatch(
        keys=["p_0_0", "p_0_1", "p_1_0"],
        trajectory_metadata={
            "p_0_0": {"min_global_steps": 1, "max_global_steps": 3, "finish_reason": "length"},
            "p_0_1": {"global_steps": 3, "materialization_reason": "max_trajectory_length"},
            "p_1_0": {"global_steps": 4, "truncated": False},
            "unconsumed": {"global_steps": 1000, "finish_reason": "length"},
        },
    )
    metrics = sampled_metadata_metrics(batch, trainer_version=4)
    assert metrics["train_batch/num_prompts"] == 1
    assert metrics["train_batch/num_sessions"] == 2
    assert metrics["training/off_policy/version_tag_age/mean"] == pytest.approx(2 / 3)
    assert metrics["training/partial_rollout/multi_version_count"] == 1
    assert metrics["training/partial_rollout/length_materialized_count"] == 1
    assert metrics["train_batch/truncation/known_count"] == 2
    assert metrics["train_batch/truncation/observed_ratio"] == 0.5
    assert metrics["train_batch/response_length/clip_ratio_valid"] == 0
    assert "train_batch/response_length/clip_ratio" not in metrics
    batch.trajectory_metadata["p_0_1"]["finish_reason"] = "stop"
    complete = sampled_metadata_metrics(batch, trainer_version=4)
    assert complete["train_batch/response_length/clip_ratio_valid"] == 1
    assert complete["train_batch/response_length/clip_ratio"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("completed,valid", [([3, 3], [1, 1]), ([3, 2], [1, 1]), ([3, 3], [1, 0])])
def test_optimizer_rank_counts_are_replicated_and_disagreement_is_unknown(completed, valid):
    raw = {
        "update/optimizer_updates_attempted": [4, 4],
        "update/optimizer_updates_completed": completed,
        "update/optimizer_updates_skipped": [4 - value for value in completed],
        "update/optimizer_updates_valid": valid,
        "loss": [1.0, 3.0],
    }
    output = TensorDict({}, batch_size=[])
    output["metrics"] = NonTensorData(raw)
    metrics = _extract_actor_update_metrics(output)
    assert metrics["actor/update/optimizer_updates_attempted"] == 4
    assert metrics["actor/loss"] == 2
    assert raw["loss"] == [1.0, 3.0]
    if len(set(completed)) == 1 and min(valid) == 1:
        assert metrics["actor/update/optimizer_updates_completed"] == 3
        assert metrics["actor/update/optimizer_updates_skipped"] == 1
    else:
        assert metrics["actor/update/optimizer_updates_valid"] == 0
        assert "actor/update/optimizer_updates_completed" not in metrics


def test_shared_resource_roles_do_not_double_count_unique_gpu_cost():
    metrics = {"train_batch/num_attention_tokens": 100}
    _add_train_throughput_metrics(metrics, {"step": 2}, trainer_gpus=4, rollout_gpus=4, total_gpus=4)
    assert metrics["resource/gpu_count/rollout"] == 4
    assert metrics["resource/gpu_count/shared"] == 4
    assert metrics["resource/gpu_count/total"] == 4
    assert metrics["perf/throughput"] == 12.5
