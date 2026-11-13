# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from laminar.eval.metrics import reward_evidence, summarize_evaluation


def _scored(index, reward):
    return {
        "session_index": index,
        "reward": reward,
        "reward_valid": True,
        "reward_source": "agent_runner",
        "scorer_status": "PASS",
    }


def test_numeric_reward_and_runner_context_do_not_prove_final_scoring_without_contract():
    assert reward_evidence({}, raw_reward=0.0)["reward_valid"] is False
    evidence = {"runner_reward_info": {"reward": 0.0, "reward_context": {"eval_completed": True}}}
    unknown = reward_evidence(evidence, raw_reward=0.0)
    assert unknown["scorer_status"] == "UNKNOWN"
    assert unknown["reward_validity_reason"] == "legacy_reward_proof_missing"
    proven = reward_evidence(evidence, raw_reward=0.0, allow_runner_reward=True)
    assert proven["reward_valid"] is True
    assert proven["reward_source"] == "agent_runner"
    assert proven["reward_validity_reason"] == "producer_contract_runner_reward"
    mismatch = reward_evidence(evidence, raw_reward=1.0, allow_runner_reward=True)
    assert mismatch["reward_valid"] is False
    assert mismatch["reward_validity_reason"] == "runner_reward_disagrees_with_final_reward"


def test_explicit_final_scorer_proof_takes_priority_over_runner_inference():
    extra = {
        "reward_valid": True,
        "reward_source": "custom_reward_worker",
        "scorer_status": "PASS",
        "runner_reward_info": {"reward": 1.0},
    }
    assert reward_evidence(extra, raw_reward=0.0, allow_runner_reward=True)["reward_source"] == "custom_reward_worker"
    extra["reward_valid"] = False
    extra["scorer_status"] = "FAIL"
    failed = reward_evidence(extra, raw_reward=1.0, allow_runner_reward=True)
    assert failed["reward_valid"] is False and failed["scorer_status"] == "FAIL"
    extra["scorer_status"] = "PASS"
    assert reward_evidence(extra, raw_reward=1.0, allow_runner_reward=True)["scorer_status"] == "UNKNOWN"


def test_coverage_and_dataset_means_preserve_missing_unscored_and_real_zero_populations():
    samples = [{"data_source": "complete"}, {"data_source": "partial"}, {"data_source": "partial"}]
    prompts = [
        {"dataset_index": 0, "sessions": [_scored(0, 0.0), _scored(1, 1.0)]},
        {"dataset_index": 1, "sessions": [_scored(0, 3.0), {"session_index": 1, "reward": 100.0}]},
        {"dataset_index": 2, "sessions": []},
    ]
    summary = summarize_evaluation(prompts, samples, n=2)
    assert summary["num_completed_sessions"] == 4
    assert summary["num_scored_sessions"] == 3
    assert summary["num_missing_sessions"] == 2
    assert summary["num_unscored_sessions"] == 1
    assert summary["execution_coverage"] == 4 / 6
    assert summary["scoring_coverage"] == 3 / 6
    assert summary["mean_session_reward"] == 4 / 3
    assert summary["mean_requested_session_reward"] is None
    assert summary["scoring_status"] == "UNKNOWN"
    assert summary["metrics"]["val-core/complete/reward/mean@2"] == 0.5
    assert "val-core/partial/reward/mean@2" not in summary["metrics"]
    assert summary["metrics"]["val-aux/partial/reward/mean_observed"] == 3.0
    assert summary["groups"]["partial"]["observed_samples_per_prompt"] == {"2": 1, "0": 1}
    assert summary["groups"]["partial"]["scored_samples_per_prompt"] == {"1": 1, "0": 1}
    assert prompts[1]["sessions"][1]["reward"] == 100.0


def test_no_scores_have_no_zero_mean_or_core_metric_and_explicit_failure_remains_failure():
    summary = summarize_evaluation([{"dataset_index": 0, "sessions": [{"session_index": 0, "reward": 0.0}]}], [{}], n=1)
    assert summary["execution_coverage"] == 1.0 and summary["scoring_coverage"] == 0.0
    assert summary["mean_session_reward"] is None
    assert summary["mean_requested_session_reward"] is None
    assert not any("/reward/" in key for key in summary["metrics"])
    failed = {"session_index": 0, "reward": 0.0, "reward_valid": False, "scorer_status": "FAIL"}
    assert summarize_evaluation([{"dataset_index": 0, "sessions": [failed]}], [{}], n=1)["scoring_status"] == "FAIL"


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), True])
def test_invalid_reward_never_reaches_metrics(reward):
    with pytest.raises(ValueError, match="finite"):
        summarize_evaluation([{"dataset_index": 0, "sessions": [_scored(0, reward)]}], [{}], n=1)


@pytest.mark.parametrize("allow_runner_reward", [False, True])
def test_mixed_reward_evidence_preserves_source_populations_and_input(allow_runner_reward):
    samples = [
        {"data_source": "explicit"},
        {"data_source": "runner"},
        {"data_source": "mixed"},
        {"data_source": "explicit"},
        {"data_source": "mixed"},
    ]
    explicit_zero = {
        "session_index": 2,
        "reward": 0.0,
        "reward_valid": True,
        "reward_source": "custom_reward_worker",
        "scorer_status": "PASS",
        "runner_reward_info": {"reward": 999.0},
    }
    prompts = [
        {"dataset_index": 3, "sessions": [_scored(1, 1.0), _scored(3, 3.0), _scored(0, 5.0), _scored(2, 7.0)]},
        {
            "dataset_index": 2,
            "sessions": [
                {"session_index": 3, "reward": 100.0, "reward_valid": False, "scorer_status": "FAIL"},
                {"session_index": 0, "reward": 200.0, "reward_valid": False, "scorer_status": "UNSCORED"},
                {"session_index": 2, "reward": 300.0, "reward_valid": False, "scorer_status": "UNKNOWN"},
                {"session_index": 1, "reward": 400.0, "runner_reward_info": {"reward": 399.0}},
            ],
        },
        {"dataset_index": 0, "sessions": [explicit_zero, _scored(0, 2.0), _scored(3, 4.0), _scored(1, 6.0)]},
        {
            "dataset_index": 1,
            "sessions": [
                {"session_index": 2, "reward": 0.0, "runner_reward_info": {"reward": 0.0}},
                {"session_index": 0, "reward": 2.0, "runner_reward_info": {"reward": 2.0}},
                {"session_index": 3, "reward": 4.0, "runner_reward_info": {"reward": 4.0}},
                {"session_index": 1, "reward": 6.0, "runner_reward_info": {"reward": 6.0}},
            ],
        },
    ]
    original_inputs = deepcopy((prompts, samples))

    summary = summarize_evaluation(prompts, samples, n=4, allow_runner_reward=allow_runner_reward)

    assert (prompts, samples) == original_inputs
    assert summary["num_prompts"] == 5
    assert summary["requested_samples_per_prompt"] == 4
    assert summary["num_expected_sessions"] == 20
    assert summary["num_completed_sessions"] == 16
    assert summary["num_missing_sessions"] == 4
    assert summary["num_scored_sessions"] == (12 if allow_runner_reward else 8)
    assert summary["num_unscored_sessions"] == (4 if allow_runner_reward else 8)
    assert summary["execution_coverage"] == 0.8
    assert summary["scoring_coverage"] == (0.6 if allow_runner_reward else 0.4)
    assert summary["scorer_status_counts"] == {
        "PASS": 12 if allow_runner_reward else 8,
        "FAIL": 1,
        "UNSCORED": 1,
        "UNKNOWN": 2 if allow_runner_reward else 6,
    }
    assert summary["observed_samples_per_prompt"] == {"4": 4, "0": 1}
    assert summary["scored_samples_per_prompt"] == ({"4": 3, "0": 2} if allow_runner_reward else {"4": 2, "0": 3})
    assert summary["mean_session_reward"] == (10 / 3 if allow_runner_reward else 3.5)
    assert summary["mean_requested_session_reward"] is None
    assert summary["scoring_status"] == "FAIL"
    groups = summary["groups"]
    assert groups["explicit"]["mean_requested_session_reward"] == 3.5
    assert groups["runner"]["num_scored_sessions"] == (4 if allow_runner_reward else 0)
    assert groups["runner"]["mean_requested_session_reward"] == (3.0 if allow_runner_reward else None)
    assert groups["runner"]["scoring_status"] == ("PASS" if allow_runner_reward else "UNKNOWN")
    assert groups["mixed"]["scorer_status_counts"] == {"FAIL": 1, "UNSCORED": 1, "UNKNOWN": 2}
    assert groups["mixed"]["mean_session_reward"] is None
    assert groups["mixed"]["scoring_status"] == "FAIL"
    reward_metrics = {key: value for key, value in summary["metrics"].items() if "/reward/" in key}
    expected_reward_metrics = {
        "val-aux/explicit/reward/mean_observed": 3.5,
        "val-core/explicit/reward/mean@4": 3.5,
    }
    if allow_runner_reward:
        expected_reward_metrics.update(
            {"val-aux/runner/reward/mean_observed": 3.0, "val-core/runner/reward/mean@4": 3.0}
        )
    assert reward_metrics == expected_reward_metrics
    assert summary["metrics"]["val-aux/mixed/coverage/num_expected_sessions"] == 8
    assert summary["metrics"]["val-aux/mixed/coverage/num_missing_sessions"] == 4
    assert summary["metrics"]["val-aux/mixed/coverage/num_unscored_sessions"] == 4
