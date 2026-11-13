# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pickle

import pytest

from laminar.metrics import (
    Aggregation,
    MetricSource,
    MetricSpec,
    Ratio,
    WeightedMean,
    cumulative_counter_specs,
    export_flat_metrics,
    reduce_metric_snapshots,
    reduce_metric_values,
)


def test_weighted_mean_and_ratio_merge_sufficient_statistics():
    mean = MetricSpec("loss", Aggregation.WEIGHTED_MEAN, denominator="action_tokens")
    ratio = MetricSpec("acceptance", Aggregation.RATIO, denominator="candidates")
    assert reduce_metric_values(mean, [WeightedMean(2, 1), WeightedMean(30, 9)]) == pytest.approx(3.2)
    assert reduce_metric_values(ratio, [Ratio(1, 2), Ratio(9, 18)]) == pytest.approx(0.5)
    assert reduce_metric_values(mean, [WeightedMean(0, 0)]) is None
    assert reduce_metric_values(ratio, [Ratio(0, 0)]) is None
    with pytest.raises(TypeError, match="sufficient statistics"):
        reduce_metric_values(mean, [2.0, 3.0])


def test_reducer_obeys_contract_instead_of_key_suffix():
    specs = {
        "metric/max": MetricSpec("metric/max", Aggregation.SUM),
        "metric/min": MetricSpec("metric/min", Aggregation.LAST),
        "peak": MetricSpec("peak", Aggregation.MAX),
        "floor": MetricSpec("floor", Aggregation.MIN),
    }
    result = reduce_metric_snapshots(
        specs,
        [
            {"metric/max": 2, "metric/min": 7, "peak": 2, "floor": 3},
            {"metric/max": 4, "metric/min": 9, "peak": 4, "floor": 1},
        ],
    )
    assert result == {"metric/max": 6, "metric/min": 9, "peak": 4, "floor": 1}


def test_legacy_adapter_receives_only_undeclared_keys():
    observed = []

    def legacy(metrics):
        observed.append(metrics)
        return {"old": sum(metrics["old"]) / len(metrics["old"])}

    specs = {"new": MetricSpec("new", Aggregation.SUM)}
    snapshots = [{"new": 2, "old": 3}, {"new": 4, "old": 5}]
    with pytest.raises(KeyError, match="explicit legacy reducer"):
        reduce_metric_snapshots(specs, snapshots)
    assert reduce_metric_snapshots(specs, snapshots, legacy_reducer=legacy) == {"new": 6, "old": 4}
    assert observed == [{"old": [3, 5]}]


def test_legacy_verl_metric_keeps_its_explicit_aggregation_and_original_values():
    from verl.utils.metric import Metric, reduce_metrics

    first = Metric("sum", value=2)
    second = Metric("sum", value=5)
    assert reduce_metric_snapshots({}, [{"legacy": first}], legacy_reducer=reduce_metrics) == {"legacy": 2}
    assert reduce_metric_snapshots({}, [{"legacy": first}, {"legacy": second}], legacy_reducer=reduce_metrics) == {
        "legacy": 7
    }
    assert first.values == [2]
    assert second.values == [5]
    with pytest.raises(ValueError, match="Aggregation type mismatch"):
        reduce_metric_snapshots(
            {}, [{"legacy": first}, {"legacy": Metric("max", value=5)}], legacy_reducer=reduce_metrics
        )


def test_timing_alias_preserves_legacy_key_and_omits_empty_population():
    assert export_flat_metrics({"timing/step": 2, "empty": WeightedMean(0, 0)}) == {
        "timing/step": 2,
        "timing_s/step": 2,
    }
    with pytest.raises(ValueError, match="conflicting timing alias"):
        export_flat_metrics({"timing/step": 2, "timing_s/step": 3})


def test_snapshot_metadata_survives_transport_without_polluting_numeric_dictionary():
    source = MetricSource(cumulative_counter_specs("test", ["committed"]), source_epoch="attempt-a")
    first = source.snapshot({"committed": 8, "latest_count": 3})
    received = pickle.loads(pickle.dumps(first))
    assert received == {"committed": 8, "latest_count": 3}
    assert received.metadata.source_epoch == "attempt-a"
    assert received.metadata.cumulative_keys == ("committed",)
    assert source.snapshot({"committed": 9}).metadata.sequence == 2
