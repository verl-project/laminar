# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Metric contracts and transport-independent reduction.

Reduction is declared by the producer, never inferred from a metric name.
``WeightedMean`` and ``Ratio`` retain sufficient statistics until all contributors
have been merged. An empty population is missing, rather than a measured zero.
The flat exporter preserves existing dashboard keys and adds timing aliases.
"""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class Aggregation(StrEnum):
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    LAST = "last"
    WEIGHTED_MEAN = "weighted_mean"
    RATIO = "ratio"


@dataclass(frozen=True)
class MetricSpec:
    name: str
    aggregation: Aggregation = Aggregation.LAST
    kind: str = "gauge"
    unit: str | None = None
    owner: str | None = None
    population: str | None = None
    denominator: str | None = None
    reset: str | None = None
    window: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WeightedMean:
    """The sum of weighted values and their total weight, not a local mean."""

    total: float
    weight: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.total) or not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("weighted mean requires finite total and nonnegative finite weight")
        if self.weight == 0 and self.total != 0:
            raise ValueError("an empty weighted mean must have zero total")

    @classmethod
    def from_value(cls, value: float, weight: float) -> WeightedMean:
        return cls(total=value * weight, weight=weight)


@dataclass(frozen=True)
class Ratio:
    numerator: float
    denominator: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.numerator) or not math.isfinite(self.denominator) or self.denominator < 0:
            raise ValueError("ratio requires a finite numerator and nonnegative finite denominator")


def reduce_metric_values(spec: MetricSpec, values: Iterable[Any]) -> float | None:
    values = list(values)
    if not values:
        return None
    if spec.aggregation == Aggregation.WEIGHTED_MEAN:
        if not all(isinstance(value, WeightedMean) for value in values):
            raise TypeError(f"{spec.name} requires WeightedMean sufficient statistics")
        weight = sum(value.weight for value in values)
        return sum(value.total for value in values) / weight if weight else None
    if spec.aggregation == Aggregation.RATIO:
        if not all(isinstance(value, Ratio) for value in values):
            raise TypeError(f"{spec.name} requires Ratio sufficient statistics")
        denominator = sum(value.denominator for value in values)
        return sum(value.numerator for value in values) / denominator if denominator else None
    scalars = [float(value) for value in values]
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError(f"{spec.name} contains a non-finite observation")
    match spec.aggregation:
        case Aggregation.SUM:
            return sum(scalars)
        case Aggregation.MAX:
            return max(scalars)
        case Aggregation.MIN:
            return min(scalars)
        case Aggregation.LAST:
            return scalars[-1]
    raise ValueError(f"unsupported aggregation for {spec.name}: {spec.aggregation}")


def reduce_metric_snapshots(
    specs: Mapping[str, MetricSpec],
    snapshots: Iterable[Mapping[str, Any]],
    *,
    legacy_reducer: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reduce declared keys first; undeclared legacy keys need an explicit adapter.

    New producers should register every key. Passing e.g. verl's reducer is a
    compatibility boundary for existing worker outputs, not a default policy.
    """
    gathered: dict[str, list[Any]] = {}
    for snapshot in snapshots:
        for name, value in snapshot.items():
            gathered.setdefault(name, []).extend(value if isinstance(value, list | tuple) else [value])
    result: dict[str, Any] = {}
    legacy: dict[str, Any] = {}
    for name, values in gathered.items():
        if name not in specs:
            self_described = [
                hasattr(value, "aggregation")
                and all(callable(getattr(value, method, None)) for method in ("aggregate", "init_list", "extend"))
                for value in values
            ]
            if any(self_described):
                if not all(self_described):
                    raise TypeError(f"{name} mixes self-described legacy Metric objects with plain observations")
                if len(values) == 1:
                    legacy[name] = values[0]
                else:
                    # Preserve the producer's explicit Metric aggregation. In
                    # particular, [Metric] must never become np.mean([Metric]).
                    merged = values[0].init_list()
                    for value in values:
                        merged.extend(value)
                    legacy[name] = merged
            else:
                legacy[name] = values
            continue
        value = reduce_metric_values(specs[name], values)
        if value is not None:
            result[name] = value
    if legacy:
        if legacy_reducer is None:
            raise KeyError(f"undeclared metrics require an explicit legacy reducer: {sorted(legacy)}")
        result.update(legacy_reducer(legacy))
    return result


def export_flat_metrics(metrics: Mapping[str, Any], *, timing_aliases: bool = True) -> dict[str, Any]:
    """Finalize individual sufficient statistics and preserve legacy flat values.

    This does not reduce across workers or windows. Non-scalar legacy values are
    retained for the backend's existing adapter; metadata belongs outside here.
    """
    result: dict[str, Any] = {}
    for name, value in metrics.items():
        if isinstance(value, WeightedMean):
            value = value.total / value.weight if value.weight else None
        elif isinstance(value, Ratio):
            value = value.numerator / value.denominator if value.denominator else None
        if value is not None:
            result[name] = value
    if timing_aliases:
        for name, value in tuple(result.items()):
            if name.startswith("timing/"):
                alias = f"timing_s/{name.removeprefix('timing/')}"
                if alias in result and result[alias] != value:
                    raise ValueError(f"conflicting timing alias {alias}")
                result[alias] = value
    return result


@dataclass(frozen=True)
class SnapshotMetadata:
    source_epoch: str
    timestamp_s: float
    sequence: int
    window_start_s: float
    cumulative_keys: tuple[str, ...] = ()


class MetricSnapshot(dict[str, int | float]):
    """A backwards-compatible numeric dictionary with separate source metadata."""

    def __init__(self, metrics: Mapping[str, int | float], metadata: SnapshotMetadata):
        super().__init__(metrics)
        self.metadata = metadata


class MetricSource:
    """Non-destructive source snapshots; epoch changes whenever an actor restarts."""

    def __init__(self, specs: Iterable[MetricSpec], *, source_epoch: str | None = None):
        self.specs = {spec.name: spec for spec in specs}
        self.source_epoch = source_epoch or uuid.uuid4().hex
        self.started_at_s = time.time()
        self.sequence = 0

    def snapshot(self, metrics: Mapping[str, int | float]) -> MetricSnapshot:
        self.sequence += 1
        return MetricSnapshot(
            metrics,
            SnapshotMetadata(
                source_epoch=self.source_epoch,
                timestamp_s=time.time(),
                sequence=self.sequence,
                window_start_s=self.started_at_s,
                cumulative_keys=tuple(
                    name
                    for name, spec in self.specs.items()
                    if spec.kind == "counter" and spec.window == "runtime_epoch"
                ),
            ),
        )


def cumulative_counter_specs(owner: str, names: Iterable[str]) -> tuple[MetricSpec, ...]:
    """Declare an owner's explicitly enumerated cumulative counters."""
    return tuple(
        MetricSpec(
            name=name,
            aggregation=Aggregation.SUM,
            kind="counter",
            owner=owner,
            reset="source_epoch",
            window="runtime_epoch",
        )
        for name in names
    )
