# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Consumed-batch observations, independent of asynchronous producer snapshots."""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from laminar.train.experience_buffer.server import SampledBatch
from laminar.train.experience_buffer.tags import trajectory_key_facts


def tensor_stats(prefix: str, values: torch.Tensor, *, quantiles: bool = False) -> dict[str, float]:
    """Keep additive moments; quantiles describe this population only.

    An empty or non-finite population has no mean/min/max. Quantiles must never
    be averaged across ranks or logging windows.
    """
    values = values.detach().flatten().to(dtype=torch.float64)
    count = values.numel()
    metrics = {f"{prefix}/count": float(count), f"{prefix}/valid": float(count > 0)}
    if count == 0:
        metrics.update({f"{prefix}/sum": 0.0, f"{prefix}/m2": 0.0})
        return metrics
    if not bool(torch.isfinite(values).all().item()):
        metrics[f"{prefix}/valid"] = 0.0
        return metrics
    mean = values.mean()
    metrics.update(
        {
            f"{prefix}/sum": float(values.sum().item()),
            f"{prefix}/mean": float(mean.item()),
            f"{prefix}/max": float(values.max().item()),
            f"{prefix}/min": float(values.min().item()),
            f"{prefix}/m2": float(((values - mean) ** 2).sum().item()),
        }
    )
    if quantiles:
        points = torch.tensor([0.5, 0.9, 0.99], dtype=values.dtype, device=values.device)
        for name, value in zip(("p50", "p90", "p99"), torch.quantile(values, points).tolist(), strict=True):
            metrics[f"{prefix}/{name}"] = float(value)
    return metrics


def train_batch_metrics(td: TensorDict) -> dict[str, float | int]:
    """Separate stored sequence/context lengths from trainable action tokens."""
    attention_mask = td["attention_mask"]
    response_mask = td["response_mask"].to(dtype=torch.bool)
    response_width = response_mask.shape[-1]
    prompt_width = attention_mask.shape[-1] - response_width
    prompt_lengths = attention_mask[:, :prompt_width].sum(dim=-1)
    response_lengths = attention_mask[:, prompt_width:].sum(dim=-1)
    action_lengths = response_mask.sum(dim=-1)
    sequence_lengths = prompt_lengths + response_lengths
    metrics: dict[str, float | int] = {
        "train_batch/num_trajectories": response_mask.shape[0],
        "train_batch/num_attention_tokens": int(sequence_lengths.sum().item()),
        "train_batch/num_prompt_tokens": int(prompt_lengths.sum().item()),
        "train_batch/num_response_tokens": int(response_lengths.sum().item()),
        "train_batch/num_action_tokens": int(action_lengths.sum().item()),
        # Real truncation requires producer evidence, never padding width.
        "train_batch/response_length/clip_ratio_valid": 0.0,
        "metrics/schema_version": 2,
    }
    for name, lengths in (
        ("prompt_length", prompt_lengths),
        ("response_length", response_lengths),
        ("action_length", action_lengths),
        ("sequence_length", sequence_lengths),
    ):
        stats = tensor_stats(f"train_batch/{name}", lengths, quantiles=True)
        metrics.update(stats)
        if name in {"prompt_length", "response_length"}:
            metrics.update({key.removeprefix("train_batch/"): value for key, value in stats.items()})
    rm_scores = td.get("rm_scores")
    if isinstance(rm_scores, torch.Tensor):
        metrics.update(tensor_stats("train_batch/rewards", rm_scores.sum(dim=-1)))
    metrics.update(tensor_stats("train_batch/advantages", td["advantages"][response_mask]))
    return metrics


def sampled_metadata_metrics(batch: SampledBatch, *, trainer_version: int) -> dict[str, Any]:
    """Observe only sampled trajectories; missing metadata stays unobserved."""
    identities = [trajectory_key_facts(key) for key in batch.keys]
    metrics: dict[str, Any] = {
        "train_batch/num_prompts": len({uid for uid, _, _ in identities}),
        "train_batch/num_sessions": len({(uid, session) for uid, session, _ in identities}),
        "training/off_policy/observation_count": len(batch.keys),
    }
    metadata = getattr(batch, "trajectory_metadata", {})
    version_tag_ages: list[int] = []
    version_tag_spans: list[int] = []
    truncated: list[bool] = []
    length_materialized = 0
    for key in batch.keys:
        tag = metadata.get(key, {})
        oldest = tag.get("min_global_steps", tag.get("global_steps"))
        newest = tag.get("max_global_steps", tag.get("global_steps"))
        if (
            isinstance(oldest, int)
            and not isinstance(oldest, bool)
            and isinstance(newest, int)
            and not isinstance(newest, bool)
            and 0 <= oldest <= newest
        ):
            version_tag_ages.append(trainer_version - newest)
            version_tag_spans.append(newest - oldest)
        if isinstance(tag.get("truncated"), bool):
            truncated.append(tag["truncated"])
        elif tag.get("finish_reason") in {"length", "stop", "eos", "tool_calls"}:
            truncated.append(tag["finish_reason"] == "length")
        # A length-materialized trajectory can continue in the same session;
        # it is not evidence that the session or generation was truncated.
        length_materialized += tag.get("materialization_reason") == "max_trajectory_length"
    count = len(batch.keys)
    metrics["training/off_policy/version_tag_observation_count"] = len(version_tag_ages)
    metrics["training/off_policy/version_tag_coverage"] = len(version_tag_ages) / count if count else 0.0
    metrics["training/partial_rollout/length_materialized_count"] = length_materialized
    metrics["training/partial_rollout/version_tag_observation_count"] = len(version_tag_spans)
    if version_tag_ages:
        metrics.update(tensor_stats("training/off_policy/version_tag_age", torch.tensor(version_tag_ages)))
        metrics["training/off_policy/future_version_count"] = sum(age < 0 for age in version_tag_ages)
        metrics["training/partial_rollout/multi_version_count"] = sum(span > 0 for span in version_tag_spans)
        metrics.update(tensor_stats("training/partial_rollout/version_tag_span", torch.tensor(version_tag_spans)))
    metrics["train_batch/truncation/known_count"] = len(truncated)
    metrics["train_batch/truncation/truncated_count"] = sum(truncated)
    metrics["train_batch/truncation/coverage"] = len(truncated) / count if count else 0.0
    metrics["train_batch/response_length/clip_ratio_valid"] = float(bool(count) and len(truncated) == count)
    if truncated:
        metrics["train_batch/truncation/observed_ratio"] = sum(truncated) / len(truncated)
    if count and len(truncated) == count:
        metrics["train_batch/response_length/clip_ratio"] = sum(truncated) / count
        metrics["response_length/clip_ratio"] = sum(truncated) / count
    return metrics
