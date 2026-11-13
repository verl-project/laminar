# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Observe optimizer updates on one engine without changing its update policy."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass
class OptimizerUpdateStats:
    """Counts for one actor RPC, replicated rather than additive across ranks.

    An attempt starts when the engine enters ``optimizer_step``. A completed
    update requires a successful real optimizer step, rather than a configured
    minibatch count. Unsupported outcome observation leaves completion unknown.
    """

    attempted: int = 0
    completed: int = 0
    skipped: int = 0
    valid: bool = False

    def metrics(self) -> dict[str, int]:
        metrics = {
            "update/optimizer_updates_attempted": self.attempted,
            "update/optimizer_updates_valid": int(self.valid),
        }
        if self.valid:
            metrics.update(
                {
                    "update/optimizer_updates_completed": self.completed,
                    "update/optimizer_updates_skipped": self.skipped,
                }
            )
        return metrics


@contextmanager
def _instance_method(instance, name: str, replacement):
    """Temporarily observe this instance, preserving existing bound overrides."""
    had_override = name in vars(instance)
    previous_override = vars(instance).get(name)
    setattr(instance, name, replacement)
    try:
        yield
    finally:
        if had_override:
            setattr(instance, name, previous_override)
        else:
            delattr(instance, name)


@contextmanager
def observe_optimizer_updates(engine, *, backend: str) -> Iterator[OptimizerUpdateStats]:
    """Observe actual attempts and outcomes during an actor update RPC.

    FSDP uses the optimizer's public post-step hook: GradScaler and non-finite
    clipping skips do not call ``optimizer.step``. AMP-aware optimizers may
    implement skipping inside that call, so their outcome is deliberately
    unknown when a scaler is active. Megatron exposes ``update_successful`` in
    its optimizer's returned tuple. All methods, arguments, return values and
    exceptions remain owned by the original backend.
    """
    if backend not in {"fsdp", "megatron"}:
        raise ValueError(f"Unsupported optimizer observation backend: {backend}")

    stats = OptimizerUpdateStats()
    original_engine_step = engine.optimizer_step
    optimizer = getattr(engine, "optimizer", None)
    outcomes: list[bool] = []
    hook_handle = None
    original_optimizer_step = None

    if backend == "fsdp":
        register_hook = getattr(optimizer, "register_step_post_hook", None)
        internal_amp_skip = getattr(engine, "scaler", None) is not None and bool(
            getattr(optimizer, "_step_supports_amp_scaling", False)
        )
        if callable(register_hook) and not internal_amp_skip:
            hook_handle = register_hook(lambda *_args, **_kwargs: outcomes.append(True))
            stats.valid = True
    elif optimizer is not None and callable(getattr(optimizer, "step", None)):
        original_optimizer_step = optimizer.step
        stats.valid = True

    def optimizer_step(*args, **kwargs):
        result = original_optimizer_step(*args, **kwargs)
        if isinstance(result, tuple) and result and isinstance(result[0], bool):
            outcomes.append(result[0])
        else:
            # Forward compatible: an unfamiliar return contract is unknown,
            # never evidence that an update completed.
            stats.valid = False
        return result

    def engine_step(*args, **kwargs):
        stats.attempted += 1
        before = len(outcomes)
        try:
            result = original_engine_step(*args, **kwargs)
        except BaseException:
            # A backend may fail before or after mutating weights. Preserve the
            # exception and do not classify that attempt as an ordinary skip.
            stats.valid = False
            raise
        if stats.valid:
            observed = outcomes[before:]
            if backend == "fsdp" and not observed:
                stats.skipped += 1
            elif len(observed) == 1:
                stats.completed += int(observed[0])
                stats.skipped += int(not observed[0])
            else:
                stats.valid = False
        return result

    try:
        with _instance_method(engine, "optimizer_step", engine_step):
            if original_optimizer_step is None:
                yield stats
            else:
                with _instance_method(optimizer, "step", optimizer_step):
                    yield stats
    finally:
        if hook_handle is not None:
            hook_handle.remove()


def add_optimizer_update_metrics(output, stats: OptimizerUpdateStats):
    """Attach one RPC's counts after verl's minibatch reduction has finished."""
    if output is not None:
        from verl.utils import tensordict_utils as tu

        metrics = dict(tu.get(output, "metrics"))
        metrics.update(stats.metrics())
        tu.assign_non_tensor(output, metrics=metrics)
    return output
