# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Evaluation coverage and score aggregation over explicit final reward evidence."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


def reward_evidence(
    extra: Mapping[str, Any], *, raw_reward: float | None = None, allow_runner_reward: bool = False
) -> dict[str, Any]:
    """A numeric reward alone cannot distinguish scoring from a default zero.

    These fields describe the final scorer output, rather than the runner's
    reward, which a reward worker may subsequently replace. Legacy producers
    without this evidence remain readable. When the caller has verified that
    no custom scorer is configured, Uni-Agent's runner-reward priority contract
    also proves a finite runner reward that agrees with the final rm_scores.
    """
    fields = ("reward_valid", "reward_source", "scorer_status")
    proof = {key: extra[key] for key in fields if key in extra}
    source = extra.get("reward_source")
    source = source if isinstance(source, str) and source.strip() else None
    declared = extra.get("scorer_status")
    valid = extra.get("reward_valid") is True and source is not None and declared == "PASS"
    if valid:
        status, reason = "PASS", "explicit_final_reward_proof"
    elif not proof:
        runner = extra.get("runner_reward_info")
        runner_reward = runner.get("reward") if isinstance(runner, Mapping) else None
        runner_valid = type(runner_reward) in (int, float) and math.isfinite(runner_reward)
        if allow_runner_reward and runner_valid and raw_reward is not None:
            if math.isclose(raw_reward, runner_reward, rel_tol=1e-6, abs_tol=1e-7):
                valid, source = True, "agent_runner"
                status, reason = "PASS", "producer_contract_runner_reward"
            else:
                status, reason = "UNKNOWN", "runner_reward_disagrees_with_final_reward"
        else:
            status, reason = "UNKNOWN", "legacy_reward_proof_missing"
    elif extra.get("reward_valid") is False and declared in ("FAIL", "UNSCORED", "UNKNOWN"):
        status, reason = declared, "producer_reward_invalid"
    else:
        status, reason = "UNKNOWN", "incomplete_or_inconsistent_reward_proof"
    result = {
        "reward_valid": valid,
        "reward_source": source,
        "scorer_status": status,
        "reward_validity_reason": reason,
    }
    if proof:
        result["producer_reward_proof"] = proof
    return result


def _source(sample: Mapping[str, Any]) -> str:
    value = sample.get("data_source")
    return value if isinstance(value, str) and value else "unknown"


def summarize_evaluation(
    prompts: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    *,
    n: int,
    allow_runner_reward: bool = False,
) -> dict[str, Any]:
    """Keep successful execution and valid scoring as separate populations.

    Observed means exclude missing/invalid scores. Requested-population means
    and verl-compatible mean@N exist only when every requested score is valid;
    this function never imputes a zero for infrastructure or scorer failures.
    """
    if type(n) is not int or n <= 0:
        raise ValueError("evaluation sample count n must be a positive integer")
    by_index: dict[int, Mapping[str, Any]] = {}
    for prompt in prompts:
        index = prompt["dataset_index"]
        if type(index) is not int or not 0 <= index < len(samples) or index in by_index:
            raise ValueError("evaluation dataset index is invalid or duplicated")
        by_index[index] = prompt
    groups: dict[str, list[list[Mapping[str, Any]]]] = {}
    for index, sample in enumerate(samples):
        sessions = list(by_index.get(index, {}).get("sessions", []))
        indices = [session["session_index"] for session in sessions]
        if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < n for i in indices):
            raise ValueError("evaluation session index is invalid or duplicated")
        if any(
            type(session["reward"]) not in (int, float) or not math.isfinite(session["reward"]) for session in sessions
        ):
            raise ValueError("evaluation reward must be finite")
        groups.setdefault(_source(sample), []).append(sessions)

    def summarize(session_groups: list[list[Mapping[str, Any]]]) -> dict[str, Any]:
        scored_groups = []
        statuses = Counter()
        for group in session_groups:
            scored_group = []
            for session in group:
                proof = reward_evidence(session, raw_reward=session["reward"], allow_runner_reward=allow_runner_reward)
                statuses[proof["scorer_status"]] += 1
                if proof["reward_valid"]:
                    scored_group.append(session)
            scored_groups.append(scored_group)
        scores = [float(session["reward"]) for group in scored_groups for session in group]
        expected = len(session_groups) * n
        completed = sum(len(group) for group in session_groups)
        scored = len(scores)
        complete = scored == expected
        return {
            "num_prompts": len(session_groups),
            "requested_samples_per_prompt": n,
            "num_expected_sessions": expected,
            "num_completed_sessions": completed,
            "num_scored_sessions": scored,
            "num_missing_sessions": expected - completed,
            "num_unscored_sessions": completed - scored,
            "execution_coverage": completed / expected if expected else None,
            "scoring_coverage": scored / expected if expected else None,
            "scorer_status_counts": dict(statuses),
            "observed_samples_per_prompt": dict(Counter(str(len(group)) for group in session_groups)),
            "scored_samples_per_prompt": dict(Counter(str(len(group)) for group in scored_groups)),
            "mean_session_reward": sum(scores) / scored if scored else None,
            "mean_requested_session_reward": sum(scores) / expected if complete and expected else None,
            "scoring_status": "PASS" if complete and expected else ("FAIL" if statuses["FAIL"] else "UNKNOWN"),
        }

    group_summaries = {source: summarize(session_groups) for source, session_groups in groups.items()}
    total = summarize([group for session_groups in groups.values() for group in session_groups])
    metrics: dict[str, float | int] = {}
    for source, summary in group_summaries.items():
        for field in (
            "num_prompts",
            "num_expected_sessions",
            "num_completed_sessions",
            "num_scored_sessions",
            "num_missing_sessions",
            "num_unscored_sessions",
            "execution_coverage",
            "scoring_coverage",
        ):
            if summary[field] is not None:
                metrics[f"val-aux/{source}/coverage/{field}"] = summary[field]
        if summary["mean_session_reward"] is not None:
            metrics[f"val-aux/{source}/reward/mean_observed"] = summary["mean_session_reward"]
        if summary["mean_requested_session_reward"] is not None:
            metrics[f"val-core/{source}/reward/mean@{n}"] = summary["mean_requested_session_reward"]
    return {**total, "groups": group_summaries, "metrics": metrics}
