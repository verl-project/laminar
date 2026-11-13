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
"""Centralized metric Tracker actor for the async-RL trainer.

``AsyncRLTrainer.fit`` first asks Tracker to collect and stage a step's remote
metrics, then commits the trainer-owned metrics after its wall timer stops.
Tracker synchronously flushes ready steps to Tracking and the optional JSONL
dump when ``commit_metrics`` runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import tempfile
import time
import uuid
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, NamedTuple

import torch
from omegaconf import DictConfig, OmegaConf
from verl.utils.tracking import Tracking

from laminar.metrics import MetricSnapshot, SnapshotMetadata, export_flat_metrics
from laminar.train.relay import RelayClient
from laminar.train.tracking_config import project_config, sanitize_config

if TYPE_CHECKING:
    from ray.actor import ActorHandle

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackerConfig:
    """Typed view of the top-level async-RL ``tracker`` configuration."""

    project_name: str = "async_rl"
    experiment_name: str = "async_rl"
    logger: str | list[str] = field(default_factory=lambda: ["console"])
    dump_path: str | None = None
    dump_require_persistent_path: bool = False
    dump_persistent_roots: tuple[str, ...] = ("/mnt/hdfs",)
    remote_metrics_timeout_s: float = 5.0
    config_key_max_bytes: int | None = None
    resolved_config: Any | None = None

    @classmethod
    def from_config(cls, config: DictConfig) -> TrackerConfig:
        trainer_cfg = config.trainer
        tracker_cfg = config.tracker
        dump_cfg = tracker_cfg.dump

        logger_cfg: Any = tracker_cfg.logger
        if OmegaConf.is_config(logger_cfg):
            logger_cfg = OmegaConf.to_container(logger_cfg, resolve=True)
        logger_cfg = [str(item) for item in logger_cfg] if isinstance(logger_cfg, list) else str(logger_cfg)

        require_persistent_path = dump_cfg.require_persistent_path
        if not isinstance(require_persistent_path, bool):
            raise ValueError("tracker.dump.require_persistent_path must be a boolean")
        dump_path = dump_cfg.path
        return cls(
            project_name=str(trainer_cfg.project_name),
            experiment_name=str(trainer_cfg.experiment_name),
            logger=logger_cfg,
            dump_path=None if dump_path is None else str(dump_path),
            dump_require_persistent_path=require_persistent_path,
            dump_persistent_roots=_normalize_persistent_roots(dump_cfg.persistent_roots),
            remote_metrics_timeout_s=float(tracker_cfg.remote_metrics_timeout_s),
            config_key_max_bytes=tracker_cfg.get("config_key_max_bytes"),
            resolved_config=OmegaConf.to_container(config, resolve=True),
        )


class _RemoteMetricsResult(NamedTuple):
    prefix: str
    metrics: Mapping[str, Any]
    status: str
    latency_s: float
    metadata: SnapshotMetadata | None = None


class _StagedRemoteMetrics(NamedTuple):
    step: int
    metrics: dict[str, Any]
    source_metadata: dict[str, SnapshotMetadata]


class _SourceBaseline(NamedTuple):
    metadata: SnapshotMetadata
    counters: dict[str, float]


def _json_safe_metric_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            value = value.detach().cpu().item()
        else:
            return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        with suppress(TypeError, ValueError):
            value = value.item()
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _normalize_persistent_roots(value: Any) -> tuple[str, ...]:
    if value is None:
        return ("/mnt/hdfs",)
    roots = value.split(":") if isinstance(value, str) else [str(item) for item in value]
    normalized = tuple(root.rstrip(os.sep) or os.sep for root in roots if root)
    return normalized or ("/mnt/hdfs",)


def _is_under_persistent_roots(path: str, roots: tuple[str, ...]) -> bool:
    if not os.path.isabs(path):
        return False
    abs_path = os.path.abspath(path)
    for root in roots:
        abs_root = os.path.abspath(root)
        if abs_path == abs_root or abs_path.startswith(f"{abs_root.rstrip(os.sep)}{os.sep}"):
            return True
    return False


class Tracker:
    """Step-aligned training metric aggregator, intended to be a Ray actor."""

    def __init__(self, config: TrackerConfig) -> None:
        if config.remote_metrics_timeout_s <= 0:
            raise ValueError("tracker.remote_metrics_timeout_s must be positive")
        self.config = config
        self._logger: Tracking | None = None
        self._finish_exit_code: int | None = None
        self._finish_error: BaseException | None = None
        self.config_dump_path: str | None = None
        sanitized_config, omitted_paths = sanitize_config(config.resolved_config)
        self._logger_config = config.resolved_config
        projection = None
        if config.config_key_max_bytes is not None:
            if not isinstance(sanitized_config, Mapping):
                raise ValueError("bounded Tracking config requires a resolved configuration mapping")
            if config.dump_path is None:
                raise ValueError("bounded Tracking config requires tracker.dump.path for its complete config sidecar")
            projection = project_config(sanitized_config, config.config_key_max_bytes)
            self._logger_config = projection.logger_config

        self.rollout_dispatcher: ActorHandle | None = None
        self.experience_buffer: ActorHandle | None = None
        self.multi_version_rollout_manager: ActorHandle | None = None
        self.relay_client: RelayClient | None = None

        self._collecting_remote_metrics_step: int | None = None
        self._staged_remote_metrics: _StagedRemoteMetrics | None = None

        self.next_step_to_log = 0
        self.metric_buffer: dict[int, dict[str, Any]] = {}
        self._pending_metrics_commit_observation: dict[str, float] | None = None
        self._source_metadata_buffer: dict[int, dict[str, SnapshotMetadata]] = {}
        self._source_baselines: dict[str, _SourceBaseline] = {}
        self._source_poll_gaps: dict[str, int] = {}
        self._collected_source_metadata: dict[str, SnapshotMetadata] = {}
        self._tracker_epoch = uuid.uuid4().hex
        self._recorded_events: dict[str, dict[str, Any]] = {}
        self._finalized = False

        self._preflight_dump_path()
        self._persist_config_sidecar(
            sanitized_config,
            omitted_paths,
            None
            if projection is None
            else {
                "logger_config": projection.logger_config,
                "path_mapping": projection.path_mapping,
                "json_value_paths": projection.json_value_paths,
                "metadata_keys": projection.metadata_keys,
            },
        )
        if self.config.dump_path is None:
            logger.info("Tracker JSONL metrics dump is disabled.")
        else:
            logger.info("Tracker JSONL metrics dump path: %s", self.config.dump_path)

    def connect_metric_sources(
        self,
        rollout_dispatcher: ActorHandle | None = None,
        multi_version_rollout_manager: ActorHandle | None = None,
        relay_client: RelayClient | None = None,
        experience_buffer: ActorHandle | None = None,
    ) -> None:
        """Connect optional remote metric sources after runtime actors exist."""
        self.rollout_dispatcher = rollout_dispatcher
        self.experience_buffer = experience_buffer
        self.multi_version_rollout_manager = multi_version_rollout_manager
        self.relay_client = relay_client

    @property
    def logger(self) -> Tracking:
        self._ensure_open()
        if self._logger is None:
            self._logger = Tracking(
                project_name=self.config.project_name,
                experiment_name=self.config.experiment_name,
                default_backend=self.config.logger,
                config=self._logger_config,
            )
        return self._logger

    def finish(self, exit_code: int = 0) -> None:
        """Persist the committed tail and close existing sinks without replay."""
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise ValueError("Tracker finish exit_code must be an integer")
        if self._finish_exit_code is not None:
            if self._finish_error is not None:
                raise RuntimeError("Tracker finalization previously failed") from self._finish_error
            return
        if exit_code == 0 and (
            self._collecting_remote_metrics_step is not None
            or self._staged_remote_metrics is not None
            or self.metric_buffer
        ):
            raise RuntimeError("cannot finish Tracker successfully with uncommitted metric work")
        # Even a failed training run can have a previously committed step. Its
        # final latency belongs in the event stream, never an uncommitted row.
        pending = self._pending_metrics_commit_observation
        if pending is not None:
            step = int(pending["tracker/metrics_commit_latency_committed_step"])
            self._record_event(step, "metrics_commit", f"metrics_commit:{step}", pending)
            self._pending_metrics_commit_observation = None
        self._finish_exit_code = exit_code
        try:
            if self._logger is not None:
                self._logger.finish(exit_code=exit_code)
        except BaseException as exc:
            self._finish_error = exc
            raise
        self._finalized = True

    def _ensure_open(self) -> None:
        if self._finish_exit_code is not None:
            raise RuntimeError("Tracker is already finalized")

    async def collect_and_stage_remote_metrics(self, step: int) -> None:
        """Collect one live snapshot and stage it for exactly one commit."""
        self._validate_collect_step(step)
        self._collecting_remote_metrics_step = step
        try:
            remote_metrics = await self._collect_remote_metrics()
            self._ensure_open()
            self._staged_remote_metrics = _StagedRemoteMetrics(
                step=step,
                metrics=remote_metrics,
                source_metadata=dict(self._collected_source_metadata),
            )
        finally:
            self._collecting_remote_metrics_step = None

    async def commit_metrics(self, step: int, train_metrics: Mapping[str, Any]) -> None:
        """Consume ``step``'s staged snapshot and persist its metric row."""
        staged = self._validate_commit_step(step)
        reserved_keys = {
            "tracker/metrics_commit_latency_s",
            "tracker/metrics_commit_latency_committed_step",
        }
        conflicting_keys = reserved_keys.intersection(train_metrics)
        if conflicting_keys:
            raise ValueError(f"train_metrics contains Tracker-owned commit latency keys: {sorted(conflicting_keys)}")

        logger.debug("Committing metrics: step=%s, |train_metrics|=%s", step, len(train_metrics))
        # A completed publish is an event; RelayManager owns the legacy live
        # snapshot namespace. Preserve both instead of silently overwriting one.
        merged_metrics = {}
        for key, value in train_metrics.items():
            destination = (
                f"relay_publish/broadcast/{key.removeprefix('relay/broadcast/')}"
                if key.startswith("relay/broadcast/")
                else key
            )
            if destination in merged_metrics:
                raise ValueError(f"duplicate trainer metric after namespace migration: {destination}")
            merged_metrics[destination] = value
        collisions = merged_metrics.keys() & staged.metrics.keys()
        if collisions:
            raise ValueError(f"trainer and remote metric owners collide: {sorted(collisions)}")
        merged_metrics.update(staged.metrics)
        if step in self.metric_buffer or step < self.next_step_to_log:
            raise RuntimeError(f"duplicate metrics commit for step={step}")
        self.metric_buffer[step] = merged_metrics
        self._source_metadata_buffer[step] = staged.source_metadata
        self._staged_remote_metrics = None
        self._log_ready_steps()

    def set_next_step_to_log(self, step: int) -> None:
        """Realign the ordered log cursor when resuming from a checkpoint."""
        self._ensure_open()
        if self._collecting_remote_metrics_step is not None or self._staged_remote_metrics is not None:
            raise RuntimeError("cannot reset Tracker logging order with a remote metrics collect in flight or staged")
        self.next_step_to_log = step
        for old_step in [buffered_step for buffered_step in self.metric_buffer if buffered_step < step]:
            self.metric_buffer.pop(old_step, None)
            self._source_metadata_buffer.pop(old_step, None)
        self._log_ready_steps()

    def maybe_dump(self, step: int, metrics: dict[str, Any]) -> None:
        """Append one step row to the configured JSONL dump path, if any."""
        dump_path = self.config.dump_path
        if dump_path is None:
            return
        import orjson

        dump_metrics = {key: _json_safe_metric_value(value) for key, value in metrics.items()}
        dump_metrics["step"] = step
        source_metadata = self._source_metadata_buffer.get(step)
        if source_metadata:
            dump_metrics["source_metadata"] = {source: asdict(metadata) for source, metadata in source_metadata.items()}
        parent_dir = os.path.dirname(dump_path)
        try:
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(dump_path, "ab") as handle:
                handle.write(orjson.dumps(dump_metrics) + b"\n")
        except OSError as exc:
            raise RuntimeError(f"Failed to dump tracker metrics for step={step} to {dump_path}") from exc

    def _validate_collect_step(self, step: int) -> None:
        self._ensure_open()
        collecting_step = self._collecting_remote_metrics_step
        if collecting_step is not None:
            if step == collecting_step:
                raise RuntimeError(f"duplicate remote metrics collect already in flight for step={step}")
            raise RuntimeError(
                f"out-of-order remote metrics collect for step={step}; step={collecting_step} is still collecting"
            )

        staged = self._staged_remote_metrics
        if staged is not None:
            if step == staged.step:
                raise RuntimeError(f"duplicate remote metrics collect for staged step={step}")
            raise RuntimeError(f"missing metrics commit for staged step={staged.step} before collect step={step}")

        if step < self.next_step_to_log or step in self.metric_buffer:
            raise RuntimeError(f"duplicate remote metrics collect for committed step={step}")

    def _validate_commit_step(self, step: int) -> _StagedRemoteMetrics:
        self._ensure_open()
        staged = self._staged_remote_metrics
        if staged is None:
            if self._collecting_remote_metrics_step == step:
                raise RuntimeError(f"remote metrics collect for step={step} has not acknowledged staging")
            raise RuntimeError(f"missing staged remote metrics for commit step={step}")
        if step != staged.step:
            raise RuntimeError(f"out-of-order metrics commit for step={step}; staged step={staged.step}")
        return staged

    async def _collect_remote_metrics(self) -> dict[str, Any]:
        self._collected_source_metadata = {}
        fetches: list[tuple[str, Any]] = []
        if self.rollout_dispatcher is not None:
            fetches.append(("rollout", self.rollout_dispatcher.get_metrics.remote()))
        if self.experience_buffer is not None:
            fetches.append(("experience_buffer", self.experience_buffer.get_metrics.remote()))
        if self.multi_version_rollout_manager is not None:
            fetches.append(("multi_version_rollout", self.multi_version_rollout_manager.get_metrics.remote()))
        if self.relay_client is not None:
            fetches.append(("relay", self.relay_client.get_metrics()))
        if not fetches:
            return {}

        started_at = time.perf_counter()
        results = await asyncio.gather(*(self._get_remote_source_metrics(prefix, fetch) for prefix, fetch in fetches))
        metrics: dict[str, Any] = {}
        status_counts = {"success": 0, "timeout": 0, "failure": 0}
        for result in results:
            status_counts[result.status] += 1
            metrics.update({f"{result.prefix}/{key}": value for key, value in result.metrics.items()})
            source_prefix = f"tracker/remote_metrics/source/{result.prefix}"
            metrics[f"{source_prefix}/latency_s"] = result.latency_s
            for status in status_counts:
                metrics[f"{source_prefix}/{status}"] = float(result.status == status)
            metrics.update(self._observe_source(result))

        metrics["tracker/remote_metrics/latency_s"] = time.perf_counter() - started_at
        metrics["tracker/remote_metrics/source_count"] = float(len(results))
        metrics["tracker/remote_metrics/coverage"] = status_counts["success"] / len(results)
        for status, count in status_counts.items():
            metrics[f"tracker/remote_metrics/{status}_count"] = float(count)
        return metrics

    def _observe_source(self, result: _RemoteMetricsResult) -> dict[str, float]:
        """Diff only owner-declared counters, using the last accepted snapshot.

        Failed polls never replace a baseline. The next valid cumulative read
        covers the full elapsed interval, with an explicit poll gap observation.
        Restarted sources establish a new baseline without inventing a delta.
        """
        prefix = result.prefix
        source_prefix = f"tracker/remote_metrics/source/{prefix}"
        if result.status != "success":
            self._source_poll_gaps[prefix] = self._source_poll_gaps.get(prefix, 0) + 1
            return {f"{source_prefix}/metadata_valid": 0.0}
        metadata = result.metadata
        if metadata is None:
            return {f"{source_prefix}/metadata_valid": 0.0}
        self._collected_source_metadata[prefix] = metadata
        counters = {key: float(result.metrics[key]) for key in metadata.cumulative_keys if key in result.metrics}
        baseline = self._source_baselines.get(prefix)
        stale = (
            baseline is not None
            and metadata.source_epoch == baseline.metadata.source_epoch
            and metadata.sequence <= baseline.metadata.sequence
        )
        reset = baseline is not None and (
            metadata.source_epoch != baseline.metadata.source_epoch
            or (not stale and any(value < baseline.counters.get(key, value) for key, value in counters.items()))
        )
        elapsed_s = 0.0 if baseline is None or reset else metadata.timestamp_s - baseline.metadata.timestamp_s
        valid = baseline is not None and not reset and not stale and elapsed_s > 0
        observed_keys = counters.keys() & (baseline.counters.keys() if baseline is not None else set())
        if stale:
            self._source_poll_gaps[prefix] = self._source_poll_gaps.get(prefix, 0) + 1
        poll_gaps = self._source_poll_gaps.get(prefix, 0)
        if not stale:
            self._source_poll_gaps.pop(prefix, None)
        metrics = {
            f"{source_prefix}/metadata_valid": 1.0,
            f"{source_prefix}/snapshot_timestamp_s": metadata.timestamp_s,
            f"{source_prefix}/snapshot_age_s": max(0.0, time.time() - metadata.timestamp_s),
            f"{source_prefix}/snapshot_sequence": float(metadata.sequence),
            f"{source_prefix}/stale_snapshot": float(stale),
            f"{source_prefix}/counter_reset": float(reset),
            f"{source_prefix}/counter_window_valid": float(valid),
            f"{source_prefix}/counter_window_s": max(0.0, elapsed_s),
            f"{source_prefix}/counter_window_start_s": baseline.metadata.timestamp_s if valid else metadata.timestamp_s,
            f"{source_prefix}/counter_coverage": (
                len(counters) / len(metadata.cumulative_keys) if metadata.cumulative_keys else 1.0
            ),
            f"{source_prefix}/counter_window_coverage": (
                len(observed_keys) / len(metadata.cumulative_keys)
                if valid and metadata.cumulative_keys
                else float(valid)
            ),
            f"{source_prefix}/poll_gap_count": float(poll_gaps),
        }
        if valid and baseline is not None:
            metrics[f"{source_prefix}/snapshot_sequence_gap"] = float(
                metadata.sequence - baseline.metadata.sequence - 1
            )
            for key in observed_keys:
                delta = counters[key] - baseline.counters[key]
                metrics[f"{prefix}/{key}/window_delta"] = delta
                metrics[f"{prefix}/{key}/window_rate_per_s"] = delta / elapsed_s
        if not stale:
            self._source_baselines[prefix] = _SourceBaseline(metadata, counters)
        return metrics

    async def _get_remote_source_metrics(
        self,
        prefix: str,
        fetch: Any,
    ) -> _RemoteMetricsResult:
        started_at = time.perf_counter()
        try:
            source_metrics = await asyncio.wait_for(
                fetch,
                timeout=self.config.remote_metrics_timeout_s,
            )
        except TimeoutError:
            latency_s = time.perf_counter() - started_at
            logger.warning(
                "%s.get_metrics timed out after %.3fs",
                prefix,
                self.config.remote_metrics_timeout_s,
            )
            return _RemoteMetricsResult(prefix, {}, "timeout", latency_s)
        except Exception:  # noqa: BLE001
            logger.warning("%s.get_metrics failed; skipping", prefix, exc_info=True)
            return _RemoteMetricsResult(prefix, {}, "failure", time.perf_counter() - started_at)

        if not isinstance(source_metrics, Mapping):
            logger.warning("%s.get_metrics returned non-mapping metrics; skipping", prefix)
            return _RemoteMetricsResult(prefix, {}, "failure", time.perf_counter() - started_at)
        metadata = source_metrics.metadata if isinstance(source_metrics, MetricSnapshot) else None
        if metadata is not None:
            try:
                if (
                    not isinstance(metadata, SnapshotMetadata)
                    or not metadata.source_epoch
                    or metadata.sequence < 1
                    or not math.isfinite(metadata.timestamp_s)
                    or not math.isfinite(metadata.window_start_s)
                ):
                    raise ValueError("invalid snapshot identity or time")
                for key in metadata.cumulative_keys:
                    if key in source_metrics:
                        value = float(source_metrics[key])
                        if not math.isfinite(value) or value < 0:
                            raise ValueError(f"invalid cumulative counter {key!r}")
            except (AttributeError, TypeError, ValueError):
                logger.warning("%s.get_metrics returned invalid snapshot metadata; skipping", prefix, exc_info=True)
                return _RemoteMetricsResult(prefix, {}, "failure", time.perf_counter() - started_at)
        return _RemoteMetricsResult(
            prefix,
            source_metrics,
            "success",
            time.perf_counter() - started_at,
            metadata,
        )

    def _log_ready_steps(self) -> None:
        """Log every ready step from ``next_step_to_log`` onward."""
        while True:
            step = self.next_step_to_log
            # A resumed run can receive a later train step first. Steps before
            # the first buffered step belong to the previous run.
            step_ready = step in self.metric_buffer or (bool(self.metric_buffer) and step < min(self.metric_buffer))
            if not step_ready:
                break
            commit_started_at = time.perf_counter()
            step_metrics = export_flat_metrics(self.metric_buffer.get(step, {}))
            if step_metrics:
                pending_observation = self._pending_metrics_commit_observation
                if pending_observation is not None:
                    conflicting_keys = pending_observation.keys() & step_metrics.keys()
                    if conflicting_keys:
                        raise RuntimeError(
                            f"step metrics contain Tracker-owned commit latency keys: {sorted(conflicting_keys)}"
                        )
                    step_metrics.update(pending_observation)
                self.logger.log(step_metrics, step=step)
                self.maybe_dump(step, step_metrics)
                self.metric_buffer.pop(step, None)
                self._source_metadata_buffer.pop(step, None)
                self._pending_metrics_commit_observation = {
                    "tracker/metrics_commit_latency_s": time.perf_counter() - commit_started_at,
                    "tracker/metrics_commit_latency_committed_step": float(step),
                }
            else:
                self.metric_buffer.pop(step, None)
                logger.warning("step=%s produced no metrics, advancing anyway", step)
            self.next_step_to_log += 1

    @property
    def event_dump_path(self) -> str | None:
        path = self.config.dump_path
        return None if path is None else f"{path.removesuffix('.jsonl')}.events.jsonl"

    async def record_event(self, step: int, event_type: str, event_id: str, metrics: Mapping[str, Any]) -> None:
        """Persist a completion or tail without rewriting a historical step.

        Events use their own JSONL stream. They are not sent through scalar
        Tracking.log at an already committed training step. Idempotence applies
        within this Tracker process epoch; every event records that epoch so a
        resumed process's new epoch remains distinguishable in the same stream.
        """
        self._record_event(step, event_type, event_id, metrics)

    def _record_event(self, step: int, event_type: str, event_id: str, metrics: Mapping[str, Any]) -> None:
        self._ensure_open()
        if not event_type or not event_id:
            raise ValueError("event_type and event_id must be nonempty")
        flat_metrics = export_flat_metrics(metrics)
        event = {
            "step": step,
            "event_type": event_type,
            "event_id": event_id,
            "tracker_epoch": self._tracker_epoch,
            "metrics": {key: _json_safe_metric_value(value) for key, value in flat_metrics.items()},
        }
        previous = self._recorded_events.get(event_id)
        if previous is not None:
            if previous != event:
                raise ValueError(f"event_id={event_id!r} reused with different contents")
            return
        self._append_event(event | {"timestamp_s": time.time()})
        self._recorded_events[event_id] = event

    def _append_event(self, event: dict[str, Any]) -> None:
        path = self.event_dump_path
        if path is None:
            logger.info("Tracker event: %s", event)
            return
        import orjson

        try:
            with open(path, "ab") as handle:
                handle.write(orjson.dumps(event) + b"\n")
        except OSError as exc:
            raise RuntimeError(f"Failed to dump tracker event {event['event_id']!r} to {path}") from exc

    async def finalize(self) -> None:
        """Persist the last commit observation and finish already-open sinks."""
        self.finish(exit_code=0)

    def _preflight_dump_path(self) -> None:
        dump_path = self.config.dump_path
        if dump_path is None:
            if self.config.dump_require_persistent_path:
                raise ValueError("Tracker metrics JSONL persistence is required, but dump path is not configured")
            return
        self._preflight_dump_file(dump_path, "Tracker metrics JSONL path")
        event_path = self.event_dump_path
        if event_path is not None:
            self._preflight_dump_file(event_path, "Tracker events JSONL path")

    def _persist_config_sidecar(self, sanitized_config: Any, omitted_paths: list[str], projection: Any) -> None:
        if self.config.dump_path is None or sanitized_config is None:
            return
        parent = os.path.dirname(self.config.dump_path) or "."
        identity = hashlib.sha256(os.path.basename(self.config.dump_path).encode("utf-8")).hexdigest()[:16]
        path = os.path.join(parent, f"config-{identity}.json")
        payload = {
            "scope": "SCIENTIFIC_CONFIG_AND_KEY_MAPPING_EXCLUDES_RUNTIME_ENV_AND_CREDENTIAL_VALUES",
            "omitted_paths": omitted_paths,
            "resolved_config": sanitized_config,
            "config_key_max_bytes": self.config.config_key_max_bytes,
            "projection": projection,
        }
        encoded = (
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
        )
        temporary_path = None
        try:
            if os.path.exists(path):
                with open(path, "rb") as handle:
                    if handle.read() != encoded:
                        raise RuntimeError(
                            f"Tracker config sidecar already exists with different configuration: {path}"
                        )
            else:
                with tempfile.NamedTemporaryFile(mode="wb", dir=parent, prefix=".config-", delete=False) as handle:
                    temporary_path = handle.name
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_path, path)
                temporary_path = None
            self.config_dump_path = path
        except OSError as exc:
            raise RuntimeError(f"Failed to persist Tracker config sidecar: {path}") from exc
        finally:
            if temporary_path is not None:
                with suppress(OSError):
                    os.unlink(temporary_path)

    def _preflight_dump_file(self, path: str, label: str) -> None:
        if path.startswith("hdfs://"):
            raise ValueError(f"{label} must use a mounted writable filesystem path, got {path}")
        if self.config.dump_require_persistent_path and not os.path.isabs(path):
            raise ValueError(
                f"{label} must be an absolute path under a persistent HDFS mount root "
                f"{self.config.dump_persistent_roots}, got {path}"
            )
        if self.config.dump_require_persistent_path and not _is_under_persistent_roots(
            path,
            self.config.dump_persistent_roots,
        ):
            raise ValueError(
                f"{label} must be under a persistent HDFS mount root {self.config.dump_persistent_roots}, got {path}"
            )
        parent_dir = os.path.dirname(path)
        try:
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(path, "ab"):
                pass
        except OSError as exc:
            raise RuntimeError(f"Failed to preflight {label}: {path}") from exc
