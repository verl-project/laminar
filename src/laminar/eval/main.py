# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint validation over resident Laminar rollout groups pinned to exact versions.

Use ``python -m laminar.eval`` with the model/resource/task settings from
your recipe. Checkpoints are Hugging Face model directories loaded directly by
vLLM; no trainer, optimizer or weight-publishing service is created. Results must
be on a caller-declared persistent mount. A failed stage stops the sequence
without loading the next checkpoint.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import math
import os
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import hydra
import ray
from omegaconf import DictConfig, OmegaConf
from verl.utils import tensordict_utils as tu
from verl.utils.transferqueue_utils import tq

from laminar.eval.metrics import reward_evidence, summarize_evaluation
from laminar.eval.runtime import EvaluationRuntime, initialize_evaluation_runtime
from laminar.eval.server import (
    CheckpointDirectoryWatcher,
    EvaluationQueue,
    EvaluationQueueClient,
    create_server,
    format_client_endpoint,
    has_queue_cli_args,
    resolve_checkpoint_path,
    run_queue_cli,
)
from laminar.eval.worker import (
    CheckpointRunWatcher,
    EvaluationTaskSource,
    evaluation_profile,
    mirror_run_validation_result,
)
from laminar.global_request_router.router import RouteUnavailableError
from laminar.multi_version_rollout.lifecycle import recover_dead_server, wait_for_source_drain
from laminar.ray_runtime import ray_runtime_env_from_env

logger = logging.getLogger(__name__)


def _write_result(path: Path, result: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class _EfficiencySnapshots:
    """Read the existing adapter off the checkpoint coordinator's critical path."""

    def __init__(self, config: DictConfig, runtime: EvaluationRuntime) -> None:
        self.runtime = runtime
        self.path = Path(config.validation.result_dir) / "efficiency"
        self.interval = float(OmegaConf.select(config, "validation.efficiency_interval_s", default=60))
        self.timeout = min(self.interval, float(config.validation.stage_timeout_s))
        self.started = time.monotonic()
        self.next_read = self.started
        self.future: Future[None] | None = None
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eval-efficiency")
        self.worker = OmegaConf.select(config, "validation.queue.worker_id")

    def poll(self) -> None:
        if self.future is not None:
            if not self.future.done():
                return
            # capture logs persistence failure, including after the last poll.
            # Observability failure must not become a checkpoint failure.
            self.future.exception()
            self.future = None
        if time.monotonic() >= self.next_read:
            self.next_read = time.monotonic() + self.interval
            self.future = self.executor.submit(self.capture)

    def capture(self) -> None:
        started = time.monotonic()
        result: dict[str, Any] = {
            "status": "UNKNOWN",
            "observed_at": datetime.now(UTC).isoformat(),
            "service_started_at": self.runtime.service_started_at,
            "observer_elapsed_s": started - self.started,
            "worker_id": self.worker,
            "metrics": {},
            "semantics": {
                "timers": "Cumulative concurrent operation seconds; nested stages overlap, not GPU wall fractions.",
                "task_reports": "Terminal runners only; active and force-killed runners excluded.",
                "model": "Backend call elapsed includes queue/version waits; request_elapsed includes in-flight age.",
                "tokens": "Input tokens count repeated contexts, not physical prefill work after prefix caching.",
                "consistency": "Sequential actor reads, not atomic across actors; observed_at is read start.",
                "sessions": "Lifecycle counters, not accepted scores; compare deduplicated TQ UIDs and score evidence.",
                "reset": False,
            },
        }
        try:
            getter = getattr(self.runtime.agent_loop_manager, "get_efficiency_metrics", None)
            if getter is None:
                result["reason"] = "getter_missing"
            else:

                async def read() -> dict[str, float]:
                    return await asyncio.wait_for(getter(), timeout=self.timeout)

                metrics = asyncio.run(read())
                if not isinstance(metrics, dict) or not metrics:
                    result["reason"] = "empty_or_invalid_metrics"
                elif any(
                    not isinstance(key, str) or type(value) not in (int, float) or not math.isfinite(value)
                    for key, value in metrics.items()
                ):
                    result["reason"] = "non_finite_or_invalid_metrics"
                else:
                    result["metrics"] = dict(metrics)
                    required = (
                        "task_reports/count",
                        "admission/wait_s",
                        "sessions/admitted",
                        "sessions/completed",
                        "sessions/in_flight",
                        "model/gateway_count",
                    )
                    missing = [key for key in required if key not in metrics]
                    if missing:
                        result.update(reason="missing_components", missing_required_metrics=missing)
                    elif metrics["model/gateway_count"] <= 0:
                        result["reason"] = "no_gateway_metrics"
                    else:
                        result["status"] = "PASS"
        except Exception as exc:
            result["error_type"] = type(exc).__name__
        result["getter_elapsed_s"] = time.monotonic() - started
        try:
            self.path.mkdir(parents=True, exist_ok=True)
            _write_result(self.path / f"efficiency_{time.time_ns()}.json", result)
        except Exception:
            logger.exception("Evaluation efficiency snapshot persistence UNKNOWN")
            raise

    def close(self) -> None:
        # Like rollout futures, the Ray job owns any read still in progress.
        self.executor.shutdown(wait=False, cancel_futures=True)


def _check_control_loops(control_refs: list[ray.ObjectRef]) -> None:
    done, _ = ray.wait(control_refs, num_returns=1, timeout=0)
    if done:
        ray.get(done[0])
        raise RuntimeError("rollout control loop stopped during validation")


def _wait(ref: ray.ObjectRef, control_refs: list[ray.ObjectRef], timeout: float) -> Any:
    done, _ = ray.wait([*control_refs, ref], num_returns=1, timeout=timeout)
    if not done:
        raise TimeoutError("validation stage timed out")
    _check_control_loops(control_refs)
    return ray.get(ref)


def _read_session_results(
    uids: list[str], *, version: int, n: int, allow_runner_reward: bool = False
) -> tuple[list[dict[str, Any]], list[str]]:
    """Read one raw reward per successful session, retaining scoring validity.

    Typed TaskResult broadcasts the session reward to all its trajectories. Only
    the last trajectory is scored; all trajectories must agree on the version.
    A prompt terminal marks completion. Session coverage is derived from the
    returned trajectories and the requested rollout count.
    """
    partition = (tq.kv_list() or {}).get("val", {}) or {}
    input_uids = set(uids)
    final: dict[tuple[str, int], tuple[int, str]] = {}
    trajectory_metadata: dict[tuple[str, int], list[dict[str, Any]]] = {}
    cleanup_keys = []
    for key, tag in partition.items():
        if key in input_uids:
            cleanup_keys.append(key)
            continue
        parts = key.rsplit("_", 2)
        if len(parts) != 3 or parts[0] not in input_uids:
            continue
        uid, session_text, index_text = parts
        session, index = int(session_text), int(index_text)
        if not 0 <= session < n or index < 0 or tag.get("status") != "success":
            raise RuntimeError(f"invalid validation trajectory tag for {key}")
        if any(tag.get(field) != version for field in ("global_steps", "min_global_steps", "max_global_steps")):
            raise RuntimeError(f"validation trajectory version mismatch for {key}")
        cleanup_keys.append(key)
        # Stored token lengths include continuation context, not only model output.
        trajectory_metadata.setdefault((uid, session), []).append(
            {
                "trajectory_index": index,
                **{field: tag[field] for field in ("prompt_len", "response_len", "seq_len") if field in tag},
            }
        )
        if (uid, session) not in final or final[uid, session][0] < index:
            final[uid, session] = (index, key)

    final_items = sorted(final.items())
    scores: dict[str, list[dict[str, Any]]] = {uid: [] for uid in uids}
    if final_items:
        data = tq.kv_batch_get(
            keys=[item[1][1] for item in final_items], partition_id="val", select_fields=["rm_scores", "extra_fields"]
        )
        rewards = data["rm_scores"].sum(dim=-1).tolist()
        extra_fields = tu.get(data, "extra_fields")
        for ((uid, session), _), reward, extra in zip(final_items, rewards, extra_fields, strict=True):
            reward = float(reward)
            if not math.isfinite(reward):
                raise RuntimeError(f"non-finite validation reward for {uid}")
            session_result: dict[str, Any] = {
                "session_index": session,
                "reward": reward,
                **reward_evidence(extra, raw_reward=reward, allow_runner_reward=allow_runner_reward),
                "trajectory_metadata": sorted(
                    trajectory_metadata[uid, session], key=lambda item: item["trajectory_index"]
                ),
            }
            if "runner_reward_info" in extra:
                # Preserve the runner's scoring evidence without interpreting
                # task-specific exit codes as model or infrastructure failures.
                session_result["runner_reward_info"] = extra["runner_reward_info"]
            scores[uid].append(session_result)

    results = []
    for uid in uids:
        terminal = partition.get(uid, {})
        if terminal.get("status") not in ("finished", "failure"):
            raise RuntimeError(f"missing or invalid validation terminal for {uid}")
        if terminal["status"] != ("finished" if scores[uid] else "failure"):
            raise RuntimeError(f"incomplete or inconsistent validation output for {uid}")
        results.append({"uid": uid, "terminal": dict(terminal), "sessions": scores[uid]})
    return results, cleanup_keys


def _summarize_scores(config: DictConfig, samples: list[dict[str, Any]], result: dict[str, Any]) -> None:
    # Uni-Agent gives the runner reward priority unless a custom scorer exists.
    # Its persisted runner_reward_info proves the final reward only in that case.
    allow_runner_reward = OmegaConf.select(config, "reward.custom_reward_function.path", default=None) is None
    for prompt in result["prompts"]:
        prompt.setdefault("data_source", samples[prompt["dataset_index"]].get("data_source", "unknown"))
        for session in prompt["sessions"]:
            evidence = reward_evidence(session, raw_reward=session["reward"], allow_runner_reward=allow_runner_reward)
            canonical = all(
                session.get(key) == evidence[key] for key in ("reward_valid", "reward_source", "scorer_status")
            )
            if not canonical or "reward_validity_reason" not in session:
                session.update(evidence)
    result.update(summarize_evaluation(result["prompts"], samples, n=config.actor_rollout_ref.rollout.val_kwargs.n))
    result["metrics_schema_version"] = 2
    result["quality_complete"] = result["status"] == "PASS" and result["scoring_coverage"] == 1.0
    if not result["quality_complete"]:
        result["metrics"] = {key: value for key, value in result["metrics"].items() if not key.startswith("val-core/")}
    result["reward_proof_policy"] = {
        "explicit_final_metadata": True,
        "runner_contract_without_custom_scorer": allow_runner_reward,
        "raw_reward_field": "reward",
        "missing_score_policy": "observed_mean_excludes_missing; requested_mean_requires_complete_valid_scores",
    }
    # Preserve the execution-failure counter's meaning independently of scoring.
    result["num_failed_sessions"] = result["num_missing_sessions"]


def _validate_config(config: DictConfig) -> Path:
    validation = config.validation
    if validation.checkpoints:
        for index, checkpoint in enumerate(validation.checkpoints):
            try:
                resolved = resolve_checkpoint_path(str(checkpoint))
            except ValueError:
                raise ValueError(
                    f"validation checkpoint must be a mounted Hugging Face directory: {checkpoint}"
                ) from None
            if not (Path(checkpoint) / "config.json").is_file():
                validation.checkpoints[index] = str(resolved)
    else:
        model_path = OmegaConf.select(config, "actor_rollout_ref.model.path")
        if not model_path:
            raise ValueError("validation.checkpoints must contain Hugging Face model directories")
        try:
            resolved = resolve_checkpoint_path(str(model_path))
        except ValueError:
            raise ValueError(f"validation checkpoint must be a mounted Hugging Face directory: {model_path}") from None
        if not (Path(str(model_path)) / "config.json").is_file():
            config.actor_rollout_ref.model.path = str(resolved)
    rollout = config.actor_rollout_ref.rollout
    if rollout.name != "vllm" or rollout.load_format != "auto":
        raise ValueError("evaluation requires vllm with load_format=auto")
    if rollout.data_parallel_size != 1:
        raise ValueError("evaluation HF reload requires rollout data_parallel_size=1; use independent replicas")
    rollout_nnodes = int(rollout.get("nnodes", 1))
    rollout_gpus_per_node = int(rollout.get("n_gpus_per_node", 1))
    rollout_pool_gpus = rollout_nnodes * rollout_gpus_per_node
    replica_world_size = (
        int(rollout.get("tensor_model_parallel_size", 1))
        * int(rollout.get("data_parallel_size", 1))
        * int(rollout.get("pipeline_model_parallel_size", 1))
    )
    if rollout_pool_gpus <= 0:
        raise ValueError("rollout.nnodes * rollout.n_gpus_per_node must be positive")
    if replica_world_size <= 0 or rollout_pool_gpus % replica_world_size != 0:
        raise ValueError(
            "rollout.nnodes * rollout.n_gpus_per_node must be divisible by "
            "tensor_model_parallel_size * data_parallel_size * pipeline_model_parallel_size"
        )
    if rollout.enable_sleep_mode or rollout.free_cache_engine:
        raise ValueError("evaluation requires resident rollout weights and KV cache (sleep/cache freeing disabled)")
    for name, value in (
        ("validation.batch_size", validation.batch_size),
        ("rollout.val_kwargs.n", config.actor_rollout_ref.rollout.val_kwargs.n),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(validation.stage_timeout_s) or validation.stage_timeout_s <= 0:
        raise ValueError("validation.stage_timeout_s must be finite and positive")
    interval = OmegaConf.select(config, "validation.efficiency_interval_s", default=60)
    if type(interval) not in (int, float) or not math.isfinite(interval) or interval <= 0:
        raise ValueError("validation.efficiency_interval_s must be finite and positive")
    repack = OmegaConf.select(config, "validation.repack", default={})
    for name, default in (("source_max_kv_cache_usage", 0.5), ("destination_max_kv_cache_usage", 0.95)):
        value = repack.get(name, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"validation.repack.{name} must be in [0, 1]")
    schedule_interval = repack.get("schedule_interval_s", 5.0)
    if type(schedule_interval) not in (int, float) or not math.isfinite(schedule_interval) or schedule_interval <= 0:
        raise ValueError("validation.repack.schedule_interval_s must be finite and positive")
    watch_dir = OmegaConf.select(config, "validation.queue.watch_dir")
    if watch_dir is not None and not Path(str(watch_dir)).is_absolute():
        raise ValueError("validation.queue.watch_dir must be an absolute path")
    watch_interval = OmegaConf.select(config, "validation.queue.watch_interval_s", default=5.0)
    if type(watch_interval) not in (int, float) or not math.isfinite(watch_interval) or watch_interval <= 0:
        raise ValueError("validation.queue.watch_interval_s must be finite and positive")
    if not Path(validation.data_path).is_file():
        raise ValueError("validation.data_path must be a readable parquet file")
    result_dir = Path(validation.result_dir)
    roots = [Path(root).resolve() for root in validation.persistent_roots]
    if not result_dir.is_absolute() or not any(result_dir.resolve().is_relative_to(root) for root in roots):
        raise ValueError("validation.result_dir must be under a declared validation.persistent_roots mount")
    coordinator_only = bool(OmegaConf.select(config, "validation.queue.coordinator_only", default=False))
    if not coordinator_only and any(result_dir.glob("version_*.json")):
        raise FileExistsError("queue workers require a fresh result directory and submission identity")
    result_dir.mkdir(parents=True, exist_ok=True)
    if OmegaConf.select(config, "rollout_router.continuation", default="same_version") != "same_version":
        raise ValueError("standalone queue evaluation requires same_version")
    watch_runs = OmegaConf.select(config, "validation.queue.watch_runs", default=[]) or []
    if watch_runs:
        watch_interval = OmegaConf.select(config, "validation.queue.watch_interval_s", default=20.0)
        if type(watch_interval) not in (int, float) or not math.isfinite(watch_interval) or watch_interval <= 0:
            raise ValueError("validation.queue.watch_interval_s must be finite and positive")
        include_base = OmegaConf.select(config, "validation.queue.watch_include_base", default=False)
        if type(include_base) is not bool:
            raise ValueError("validation.queue.watch_include_base must be a boolean")
        for run_dir in watch_runs:
            run_path = Path(run_dir)
            if not run_path.is_absolute() or not any(run_path.resolve().is_relative_to(root) for root in roots):
                raise ValueError(
                    "validation.queue.watch_runs must be under a declared validation.persistent_roots mount"
                )
    return result_dir


def _canonical_checkpoint_key(checkpoint: str) -> str:
    """Normalize a checkpoint or trainer step path without requiring nonexistent test paths to have config.json."""
    try:
        return str(resolve_checkpoint_path(checkpoint).resolve())
    except ValueError:
        return str(Path(checkpoint).resolve())


def _read_resume_results(config: DictConfig, samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Import completed prompts, never active sessions or task-specific score verdicts.

    resume_protocol is the caller's task/scorer/checkpoint binding, not independent
    proof of those identities; the submitting owner must audit primary evidence.
    Source result bytes and their original dataset/checkpoint/version are checked here.
    """
    entries = OmegaConf.select(config, "validation.resume_results", default=[]) or []
    if not entries:
        return {}
    protocol = OmegaConf.to_container(config.validation.resume_protocol, resolve=True)
    if not protocol:
        raise ValueError("validation.resume_protocol must bind the audited evaluation protocol")
    dataset_sha256 = hashlib.sha256(Path(config.validation.data_path).read_bytes()).hexdigest()
    sampling = OmegaConf.to_container(config.actor_rollout_ref.rollout.val_kwargs, resolve=True)
    expected = {
        "sampling": sampling,
        "seed": config.actor_rollout_ref.rollout.seed,
        "full_determinism": config.actor_rollout_ref.rollout.full_determinism,
    }
    n = config.actor_rollout_ref.rollout.val_kwargs.n
    requested = {_canonical_checkpoint_key(str(checkpoint)) for checkpoint in config.validation.checkpoints}
    restored: dict[str, dict[int, dict[str, Any]]] = {}
    seen_uids: set[str] = set()

    def read(reference: dict[str, Any]) -> dict[str, Any]:
        path = Path(reference["path"])
        roots = [Path(root).resolve() for root in config.validation.persistent_roots]
        if not path.is_absolute() or not any(path.resolve().is_relative_to(root) for root in roots):
            raise ValueError("resume source paths must be under declared persistent roots")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != reference["sha256"]:
            raise ValueError(f"resume source digest mismatch: {path}")
        return json.loads(data)

    def check_metadata(payload: dict[str, Any], checkpoint: str) -> None:
        if _canonical_checkpoint_key(str(payload["checkpoint"])) != checkpoint:
            raise ValueError("resume checkpoint identity mismatch")
        if payload["dataset"]["sha256"] != dataset_sha256 or payload["dataset"]["rows"] != len(samples):
            raise ValueError("resume dataset identity mismatch")
        if any(payload[key] != value for key, value in expected.items()):
            raise ValueError("resume sampling protocol mismatch")

    for entry in entries:
        manifest = read(entry)
        checkpoint = _canonical_checkpoint_key(str(manifest["checkpoint"]))
        if checkpoint not in requested:
            raise ValueError("resume checkpoint is not requested")
        check_metadata(manifest, checkpoint)
        if manifest["protocol"] != protocol:
            raise ValueError("resume audited protocol mismatch")
        records = restored.setdefault(checkpoint, {})
        if not manifest["sources"]:
            raise ValueError("resume manifest has no source results")
        for source in manifest["sources"]:
            payload = read(source)
            check_metadata(payload, checkpoint)
            version = source["version"]
            if (
                type(version) is not int
                or version < 1
                or type(payload["version"]) is not int
                or payload["version"] != version
                or not isinstance(source["submission_id"], str)
                or not source["submission_id"]
            ):
                raise ValueError("resume source version/submission identity mismatch")
            for original in payload["prompts"]:
                record = copy.deepcopy(original)
                if "source_result" in record:
                    raise ValueError("repeated resume is unsupported; original generation sources are required")
                index, uid = record["dataset_index"], record["uid"]
                if type(index) is not int or not 0 <= index < len(samples):
                    raise ValueError("resume dataset index is invalid or duplicated")
                existing = records.get(index)
                if existing is not None:
                    prior = {key: value for key, value in existing.items() if key != "source_result"}
                    if prior == record:
                        continue
                    raise ValueError("conflicting resume records for dataset index")
                if not isinstance(uid, str) or not uid or uid in seen_uids:
                    raise ValueError("resume uid is invalid or duplicated")
                terminal = record["terminal"]
                if terminal.get("status") != "finished":
                    raise ValueError("resume requires finished prompts")
                if any(
                    terminal[field] != version
                    for field in ("global_steps", "min_global_steps", "max_global_steps")
                    if field in terminal
                ):
                    raise ValueError("resume terminal version mismatch")
                sessions = record["sessions"]
                if len(sessions) != n or {session["session_index"] for session in sessions} != set(range(n)):
                    raise ValueError("resume requires every requested session of a prompt")
                for session in sessions:
                    if (
                        type(session["session_index"]) is not int
                        or type(session["reward"]) not in (int, float)
                        or not math.isfinite(session["reward"])
                    ):
                        raise ValueError("resume session index/reward is invalid")
                    error = session.get("runner_reward_info", {}).get("reward_context", {}).get("agent_error")
                    if isinstance(error, str) and error.strip():
                        raise ValueError("resume prompt has an agent error")
                record["source_result"] = dict(source)
                seen_uids.add(uid)
                records[index] = record
        if len(records) == len(samples):
            raise ValueError("fully completed resume checkpoint must be excluded before GPU initialization")
    return {checkpoint: list(records.values()) for checkpoint, records in restored.items()}


@dataclass
class _CheckpointRun:
    result: dict[str, Any]
    path: Path
    started: float
    server_ids: list[str] | None
    stage: str = "persist"
    rollout_started: float = 0.0
    future: Future[tuple[str, str] | None] | None = None
    prompts: Any = None
    uids: list[str] | None = None
    deadline: float = 0.0
    pending_indices: list[int] | None = None
    batch_indices: list[int] | None = None
    # A split keeps its original Framework semaphore, including across checkpoints.
    admission_key: tuple[str, ...] | None = None


def _recover_eval_replica(
    runtime: EvaluationRuntime,
    server_id: str,
    *,
    version: int,
    checkpoint: str,
    timeout: float,
) -> bool:
    """Recover a confirmed dead evaluation replica and refresh routing health."""
    controller = getattr(runtime, "llm_server_controllers", {}).get(server_id)
    if (
        controller is None
        or not hasattr(controller, "snapshot")
        or not hasattr(runtime.rollout_router, "begin_recovery")
    ):
        return False

    async def recover() -> bool:
        async with asyncio.timeout(timeout):
            recovered = await recover_dead_server(
                runtime.rollout_router,
                controller,
                server_id,
                wait_for_version=version,
                checkpoint_path=checkpoint,
                exact_version=True,
            )
            if recovered:
                await runtime.llm_server_state_tracker.refresh_once.remote()
            return recovered

    return asyncio.run(recover())


def _wait_for_idle(
    runtime: EvaluationRuntime, run: _CheckpointRun, controls: list[ray.ObjectRef], timeout: float
) -> Any:
    kwargs: dict[str, Any] = {"weight_version": run.result["version"]}
    if run.server_ids is not None:
        kwargs["server_ids"] = run.server_ids
    deadline = time.monotonic() + timeout
    if run.deadline > 0:
        deadline = min(deadline, run.deadline)
    while True:
        _check_control_loops(controls)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("validation stage timed out while waiting for idle replicas")
        try:
            return _wait(runtime.rollout_router.get_idle_server_versions.remote(**kwargs), controls, remaining)
        except RouteUnavailableError:
            checkpoint = run.result.get("checkpoint")
            if checkpoint and hasattr(runtime.rollout_router, "failed_server_ids"):
                failed = _wait(
                    runtime.rollout_router.failed_server_ids.remote(),
                    controls,
                    max(0.01, deadline - time.monotonic()),
                )
                target_servers = set(run.server_ids) if run.server_ids is not None else set(failed)
                for server_id in failed:
                    if server_id in target_servers and _recover_eval_replica(
                        runtime,
                        server_id,
                        version=run.result["version"],
                        checkpoint=str(checkpoint),
                        timeout=max(0.01, deadline - time.monotonic()),
                    ):
                        run.result.setdefault("recoveries", []).append(
                            {
                                "server_id": server_id,
                                "stage": run.stage,
                                "version": run.result["version"],
                                "recovered_at": datetime.now(UTC).isoformat(),
                            }
                        )
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))


def _remaining_stage_timeout(run: _CheckpointRun, timeout: float) -> float:
    remaining = timeout if run.deadline <= 0 else min(timeout, run.deadline - time.monotonic())
    if remaining <= 0:
        raise TimeoutError("validation stage timed out")
    return remaining


def _launch_batch(
    config: DictConfig,
    runtime: EvaluationRuntime,
    samples: list[dict[str, Any]],
    run: _CheckpointRun,
    executor: ThreadPoolExecutor,
) -> None:
    indices = run.pending_indices if run.pending_indices is not None else list(range(len(samples)))
    offset = run.result.get("num_completed_prompts", 0) - run.result.get("num_inherited_prompts", 0)
    run.batch_indices = indices[offset : offset + config.validation.batch_size]
    batch = [samples[index] for index in run.batch_indices]
    run.uids = [str(uuid.uuid4()) for _ in batch]
    tensor_fields = {
        "raw_prompt": [sample.get("prompt") for sample in batch],
        "uid": run.uids,
        "tools_kwargs": [sample["extra_info"]["tools_kwargs"] for sample in batch],
    }
    if runtime.admission_runners:
        runners = runtime.admission_runners[run.admission_key or tuple(sorted(run.server_ids))]
        tensor_fields["agent_name"] = [
            runners[next(iter(runners)) if len(runners) == 1 else sample["agent_name"]][0] for sample in batch
        ]
        run.result["admission"] = {
            name: {"runner": scoped_name, "max_concurrent_sessions": limit}
            for name, (scoped_name, limit) in runners.items()
        }
    run.prompts = tu.get_tensordict(
        tensor_dict=tensor_fields,
        non_tensor_dict={
            "global_steps": run.result["version"],
            "validate": True,
            "__do_sample__": config.actor_rollout_ref.rollout.val_kwargs.do_sample,
        },
    )
    run.stage = "rollout"
    run.deadline = time.monotonic() + config.validation.stage_timeout_s
    run.future = executor.submit(
        _generate_batch, runtime.agent_loop_manager, run.prompts, float(config.validation.stage_timeout_s)
    )


def _generate_batch(agent_loop_manager: Any, prompts: Any, timeout: float) -> tuple[str, str]:
    """Timestamp the blocking rollout call in its executing thread."""
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    agent_loop_manager.generate_sequences_and_wait(prompts)
    if time.monotonic() - started > timeout:
        raise TimeoutError("validation rollout timed out")
    return started_at, datetime.now(UTC).isoformat()


def _finish_batch(
    config: DictConfig,
    runtime: EvaluationRuntime,
    samples: list[dict[str, Any]],
    run: _CheckpointRun,
    controls: list[ray.ObjectRef],
    executor: ThreadPoolExecutor,
) -> None:
    assert run.future is not None and run.uids is not None
    started_at, finished_at = run.future.result()
    result = run.result
    result.setdefault("rollout_started_at", started_at)
    result["rollout_finished_at"] = finished_at
    version = result["version"]
    if tu.get_non_tensor_data(run.prompts, "global_steps", None) != version:
        raise RuntimeError("adapter changed the requested validation version")
    timeout = float(config.validation.stage_timeout_s)
    run.stage = "drain"
    _wait(runtime.llm_server_state_tracker.refresh_once.remote(), controls, _remaining_stage_timeout(run, timeout))
    _wait_for_idle(runtime, run, controls, timeout)
    run.stage = "read_results"
    records, keys = _read_session_results(
        run.uids,
        version=version,
        n=config.actor_rollout_ref.rollout.val_kwargs.n,
        allow_runner_reward=OmegaConf.select(config, "reward.custom_reward_function.path", default=None) is None,
    )
    assert run.batch_indices is not None
    for row_index, record in zip(run.batch_indices, records, strict=True):
        record["dataset_index"] = row_index
    result["prompts"].extend(records)
    result["prompts"].sort(key=lambda record: record["dataset_index"])
    result["num_completed_prompts"] += len(records)
    _summarize_scores(config, samples, result)
    run.stage = "persist"
    _write_result(run.path, result)
    run.stage = "clear_results"
    tq.kv_clear(keys=keys, partition_id="val")
    logger.info(
        "Validation version=%s completed_prompts=%s/%s scored_sessions=%s",
        version,
        result["num_completed_prompts"],
        len(samples),
        result["num_scored_sessions"],
    )
    if result["num_completed_prompts"] < len(samples):
        _launch_batch(config, runtime, samples, run, executor)
        return
    result["timing_s"]["rollout_and_drain"] = time.perf_counter() - run.rollout_started
    agent_errors = [
        session.get("runner_reward_info", {}).get("reward_context", {}).get("agent_error")
        for prompt in result["prompts"]
        for session in prompt["sessions"]
    ]
    # A task failure invalidates this checkpoint, while other replica groups
    # and the next checkpoint can continue. Preserve its original scoring evidence.
    if any(isinstance(error, str) and error.strip() for error in agent_errors):
        result["status"] = "FAIL"
    else:
        result["status"] = "PASS" if result["num_failed_sessions"] == 0 else "PARTIAL"
    result["execution_status"] = result["status"]
    _summarize_scores(config, samples, result)
    result["timing_s"]["total"] = time.perf_counter() - run.started
    result["finished_at"] = datetime.now(UTC).isoformat()
    _check_control_loops(controls)
    run.stage = "persist"
    _write_result(run.path, result)
    run.future = None
    logger.info("Validation version=%s status=%s result=%s", version, result["status"], run.path)


def _load_checkpoint(
    runtime: EvaluationRuntime,
    run: _CheckpointRun,
    checkpoint: str,
    controls: list[ray.ObjectRef],
    timeout: float,
    *,
    initial: bool,
) -> None:
    version = run.result["version"]
    before = time.perf_counter()
    if initial:
        if any(runtime.initial_versions[server_id] != version for server_id in run.server_ids):
            raise RuntimeError("initial checkpoint groups do not match Router assignment")
        run.result["timing_s"]["initial_load"] = max(
            runtime.initial_load_timings[server_id] for server_id in run.server_ids
        )
    else:
        run.result["reload_started_at"] = datetime.now(UTC).isoformat()
        refs: dict[str, Any] = {}
        errors: dict[str, Exception] = {}
        for server_id in run.server_ids:
            try:
                refs[server_id] = runtime.llm_server_controllers[server_id].load_hf_checkpoint.remote(
                    str(checkpoint), weight_version=version
                )
            except Exception as exc:
                errors[server_id] = exc
        for server_id in run.server_ids:
            try:
                if server_id in errors:
                    raise errors[server_id]
                loaded_version = _wait(refs[server_id], controls, _remaining_stage_timeout(run, timeout))
            except Exception as exc:
                if not _recover_eval_replica(
                    runtime,
                    server_id,
                    version=version,
                    checkpoint=str(checkpoint),
                    timeout=_remaining_stage_timeout(run, timeout),
                ):
                    raise exc from None
                run.result.setdefault("recoveries", []).append(
                    {
                        "server_id": server_id,
                        "stage": "reload",
                        "version": version,
                        "recovered_at": datetime.now(UTC).isoformat(),
                    }
                )
            else:
                if loaded_version != version:
                    raise RuntimeError("replica loaded an unexpected checkpoint version")
                _wait(
                    runtime.rollout_router.mark_active.remote(server_id, weight_version=version),
                    controls,
                    _remaining_stage_timeout(run, timeout),
                )
        run.result["timing_s"]["reload"] = time.perf_counter() - before
        run.result["reload_finished_at"] = datetime.now(UTC).isoformat()
    _wait(runtime.llm_server_state_tracker.refresh_once.remote(), controls, _remaining_stage_timeout(run, timeout))
    run.stage = "warmup"
    before = time.perf_counter()
    run.result["replica_versions"] = _wait_for_idle(runtime, run, controls, timeout)
    run.result["timing_s"]["warmup"] = time.perf_counter() - before


def _drain_repacked_sources(router: Any, sources: list[str], version: int, timeout: float) -> None:
    """Run the shared abort/release-ACK barrier off the coordinator thread."""

    async def drain() -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        async with asyncio.TaskGroup() as tasks:
            for source in sources:
                tasks.create_task(wait_for_source_drain(router, source, expected_version=version, deadline=deadline))

    asyncio.run(drain())


def _coalesce_available_groups(
    available: list[tuple[list[str], tuple[str, ...] | None, bool]],
) -> list[tuple[list[str], tuple[str, ...] | None, bool]]:
    """Recombine finished Repack subsets that share the same original admission group."""
    merged: list[tuple[list[str], tuple[str, ...] | None, bool]] = []
    by_admission: dict[tuple[str, ...], list[str]] = {}
    for server_ids, admission_key, repacked in available:
        if repacked or admission_key is None:
            merged.append((list(server_ids), admission_key, repacked))
            continue
        if admission_key not in by_admission:
            combined = list(server_ids)
            by_admission[admission_key] = combined
            merged.append((combined, admission_key, False))
        else:
            combined = by_admission[admission_key]
            for server_id in server_ids:
                if server_id not in combined:
                    combined.append(server_id)
            order = {server_id: idx for idx, server_id in enumerate(admission_key)}
            combined.sort(key=lambda sid: order.get(sid, len(order)))
    return merged


def _run_checkpoints(
    config: DictConfig,
    runtime: EvaluationRuntime,
    samples: list[dict[str, Any]],
    resume_results: dict[str, list[dict[str, Any]]] | None = None,
    task_source: EvaluationTaskSource | None = None,
) -> None:
    """Reuse completed groups and Repack released replicas while tails continue.

    Loads and blocking adapter calls overlap across disjoint groups. Result
    reads, persistence, routing fences and TQ cleanup stay in this coordinator.
    """
    validation = config.validation
    timeout = float(validation.stage_timeout_s)
    result_dir = Path(validation.result_dir)
    with Path(validation.data_path).open("rb") as stream:
        dataset_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    ray.get(runtime.llm_server_state_tracker.refresh_once.remote(), timeout=timeout)
    ray.get(runtime.llm_server_state_tracker.start.remote(), timeout=timeout)
    controls = [runtime.llm_server_state_tracker.wait_for_stop.remote()]
    executor = ThreadPoolExecutor(max_workers=max(1, len(runtime.llm_server_controllers) * 2))
    efficiency = _EfficiencySnapshots(config, runtime)
    checkpoints = iter(enumerate(validation.checkpoints, start=1))
    initial_group_count = len(set(runtime.initial_versions.values()))
    initial_checkpoints = (
        task_source.initial
        if task_source is not None
        else [next(checkpoints) for _ in range(min(initial_group_count, len(validation.checkpoints)))]
    )
    versions = [v for v, _ in initial_checkpoints]
    runs: list[_CheckpointRun] = []
    pending_checkpoint = None if task_source else next(checkpoints, None)
    repack = OmegaConf.select(config, "validation.repack", default={})
    repack_interval = repack.get("schedule_interval_s", 5.0)
    next_repack = time.monotonic() + repack_interval
    # Drains retain their sources after the old checkpoint has completed.
    repacking: dict[Future, tuple[list[str], _CheckpointRun, dict[str, Any]]] = {}
    recovering: dict[Future[bool], tuple[str, int, str]] = {}
    server_targets: dict[str, tuple[int, str]] = {
        server_id: (version, str(validation.checkpoints[version - 1]))
        for server_id, version in runtime.initial_versions.items()
        if 1 <= version <= len(validation.checkpoints)
    }

    def claim(server_ids: list[str]) -> tuple[int, str] | None:
        nonlocal pending_checkpoint
        if task_source is not None:
            return task_source.claim(server_ids)
        selected = pending_checkpoint
        pending_checkpoint = next(checkpoints, None) if selected is not None else None
        return selected

    def start(
        version: int,
        checkpoint: str,
        server_ids: list[str],
        *,
        initial: bool,
        repacked: bool = False,
        admission_key: tuple[str, ...] | None = None,
    ) -> None:
        for server_id in server_ids:
            server_targets[server_id] = (version, str(checkpoint))
        started = time.perf_counter()
        result: dict[str, Any] = {
            "status": "RUNNING",
            "checkpoint": str(checkpoint),
            "version": version,
            "replica_group": list(server_ids),
            "started_at": datetime.now(UTC).isoformat(),
            "service_started_at": runtime.service_started_at,
            "service_initialization_s": runtime.initialization_s,
            "dataset": {"path": str(validation.data_path), "sha256": dataset_sha256, "rows": len(samples)},
            "sampling": OmegaConf.to_container(config.actor_rollout_ref.rollout.val_kwargs, resolve=True),
            "seed": config.actor_rollout_ref.rollout.seed,
            "full_determinism": config.actor_rollout_ref.rollout.full_determinism,
            "num_expected_sessions": len(samples) * config.actor_rollout_ref.rollout.val_kwargs.n,
            "num_completed_prompts": 0,
            "num_scored_sessions": 0,
            "timing_s": {},
            "prompts": [],
        }
        run = _CheckpointRun(
            result=result,
            path=result_dir / f"version_{version:06d}.json",
            started=started,
            server_ids=server_ids,
            admission_key=admission_key or tuple(sorted(server_ids)),
        )
        restore_config = config
        restored = resume_results or {}
        if task_source is not None:
            restore_config = task_source.resume_config(config, version)
            restored = _read_resume_results(restore_config, samples)
            result["queue_task_id"] = task_source.tasks[version]["task_id"]
            result["queue_owner"] = task_source.tasks[version]["owner"]
            if task_source.tasks[version].get("source"):
                result["queue_source"] = copy.deepcopy(task_source.tasks[version]["source"])
            result["evaluation_profile_sha256"] = task_source.profile_sha256
        inherited = copy.deepcopy(restored.get(_canonical_checkpoint_key(str(checkpoint)), []))
        if inherited:
            result["prompts"] = sorted(inherited, key=lambda record: record["dataset_index"])
            result["num_completed_prompts"] = len(inherited)
            result["num_inherited_prompts"] = len(inherited)
            result["resume_protocol"] = OmegaConf.to_container(restore_config.validation.resume_protocol, resolve=True)
            result["resume_results"] = OmegaConf.to_container(restore_config.validation.resume_results, resolve=True)
            complete = {record["dataset_index"] for record in inherited}
            run.pending_indices = [index for index in range(len(samples)) if index not in complete]
        _summarize_scores(restore_config, samples, result)
        runs.append(run)
        _write_result(run.path, result)
        logger.info("Validation version=%s checkpoint=%s started", version, checkpoint)
        run.stage = "load"
        if not initial:
            _wait(runtime.llm_server_state_tracker.refresh_once.remote(), controls, timeout)
            assignment = {"weight_versions": [version], "server_ids": server_ids}
            if repacked:
                assignment["repacked"] = True
            assigned = _wait(
                runtime.rollout_router.set_validation_versions.remote(**assignment),
                controls,
                timeout,
            )
            if set(assigned[version]) != set(server_ids):
                raise RuntimeError("checkpoint reload changed its replica group")
            if not repacked:
                plan = _wait(
                    runtime.rollout_router.plan_and_begin_repack.remote(loadable_version=version, repack_enabled=False),
                    controls,
                    timeout,
                )
                if plan.groups or set(plan.idle_sources) != set(run.server_ids):
                    raise RuntimeError("checkpoint reload could not fence every assigned replica")
        run.deadline = time.monotonic() + timeout
        run.future = executor.submit(
            _load_checkpoint, runtime, run, str(checkpoint), controls, timeout, initial=initial
        )

    try:
        groups = _wait(
            runtime.rollout_router.set_validation_versions.remote(weight_versions=versions), controls, timeout
        )
        for version, checkpoint in initial_checkpoints:
            start(version, checkpoint, groups[version], initial=True)
        idle_groups = [(group, tuple(sorted(group)), False) for group in task_source.idle_groups] if task_source else []
        while runs or repacking or recovering or (task_source is not None and not task_source.draining):
            _check_control_loops(controls)
            efficiency.poll()
            # Settle all completed stages before admitting another checkpoint.
            available = idle_groups
            idle_groups = []
            for future in list(recovering):
                if future.done():
                    server_id, version, _checkpoint = recovering.pop(future)
                    if future.result():
                        for owner_run in runs:
                            if owner_run.result["version"] == version:
                                owner_run.result.setdefault("recoveries", []).append(
                                    {
                                        "server_id": server_id,
                                        "stage": owner_run.stage,
                                        "version": version,
                                        "recovered_at": datetime.now(UTC).isoformat(),
                                    }
                                )
                                _write_result(owner_run.path, owner_run.result)
            for future in list(repacking):
                if future.done():
                    server_ids, original_run, event = repacking[future]
                    try:
                        future.result()
                        if task_source is not None:
                            task_source.release_replicas(original_run.result["version"], original_run.server_ids)
                    except Exception:
                        original_run.stage = "repack"
                        original_run.future = future
                        raise
                    repacking.pop(future)
                    event["drained_at"] = datetime.now(UTC).isoformat()
                    _write_result(original_run.path, original_run.result)
                    available.append((server_ids, original_run.admission_key, True))
            for run in list(runs):
                if run.future is not None:
                    phase = "load" if run.stage in {"load", "warmup"} else "rollout"
                    if run.future.done():
                        if phase == "load":
                            run.future.result()
                            run.future = None
                            _write_result(run.path, run.result)
                            run.rollout_started = time.perf_counter()
                            _launch_batch(config, runtime, samples, run, executor)
                        elif not any(server_id in (run.server_ids or ()) for server_id, _, _ in recovering.values()):
                            _finish_batch(config, runtime, samples, run, controls, executor)
                    elif time.monotonic() >= run.deadline:
                        raise TimeoutError(f"validation {phase} version={run.result['version']} timed out")
                if (
                    run.future is None
                    and not any(owner is run for _, owner, _ in repacking.values())
                    and not any(version == run.result["version"] for _, version, _ in recovering.values())
                ):
                    if task_source is not None:
                        source_meta = task_source.tasks[run.result["version"]].get("source") or {}
                        target_result_dir = source_meta.get("result_dir")
                        if isinstance(target_result_dir, str) and target_result_dir:
                            mirror_run_validation_result(
                                run.result,
                                Path(target_result_dir),
                                [Path(root) for root in config.validation.persistent_roots],
                            )
                        task_source.finish(run.result["version"], run.path, run.result["status"])
                    available.append((run.server_ids, run.admission_key, False))
                    runs.remove(run)
            available = _coalesce_available_groups(available)
            for server_ids, admission_key, repacked in available:
                next_checkpoint = claim(server_ids)
                if next_checkpoint is not None:
                    start(*next_checkpoint, server_ids, initial=False, repacked=repacked, admission_key=admission_key)
                elif task_source is not None:
                    idle_groups.append((server_ids, admission_key, repacked))
            if hasattr(runtime.rollout_router, "failed_server_ids"):
                failed_servers = _wait(runtime.rollout_router.failed_server_ids.remote(), controls, timeout)
                busy_servers = (
                    {server_id for server_id, _, _ in recovering.values()}
                    | {server_id for sources, _, _ in repacking.values() for server_id in sources}
                    | {
                        server_id
                        for run in runs
                        if run.stage in {"load", "warmup"} and run.future is not None
                        for server_id in (run.server_ids or ())
                    }
                )
                for server_id in failed_servers:
                    if server_id in busy_servers or server_id not in server_targets:
                        continue
                    target_version, target_checkpoint = server_targets[server_id]
                    owner_run = next((r for r in runs if r.result["version"] == target_version), None)
                    recover_timeout = _remaining_stage_timeout(owner_run, timeout) if owner_run is not None else timeout
                    future = executor.submit(
                        _recover_eval_replica,
                        runtime,
                        server_id,
                        version=target_version,
                        checkpoint=target_checkpoint,
                        timeout=recover_timeout,
                    )
                    recovering[future] = (server_id, target_version, target_checkpoint)
            if time.monotonic() >= next_repack:
                next_repack = time.monotonic() + repack_interval
                repack_candidates = [
                    run
                    for run in runs
                    if (
                        # Repack only the final admitted batch; future batches keep
                        # their original capacity until all prompts have been sent.
                        run.stage == "rollout"
                        and run.future is not None
                        and not run.future.done()
                        and len(run.server_ids) >= 2
                        and run.batch_indices is not None
                        and run.result["num_completed_prompts"] + len(run.batch_indices) == len(samples)
                        and not any(owner.admission_key == run.admission_key for _, owner, _ in repacking.values())
                        and not any(server_id in (run.server_ids or ()) for server_id, _, _ in recovering.values())
                    )
                ]
                if repack_candidates and (
                    pending_checkpoint is not None or (task_source is not None and task_source.has_queued())
                ):
                    for run in repack_candidates:
                        plan = _wait(
                            runtime.rollout_router.plan_validation_repack.remote(
                                server_ids=run.server_ids,
                                source_max_kv_cache_usage=repack.get("source_max_kv_cache_usage", 0.5),
                                destination_max_kv_cache_usage=repack.get("destination_max_kv_cache_usage", 0.95),
                            ),
                            controls,
                            _remaining_stage_timeout(run, timeout),
                        )
                        sources = list(plan.idle_sources) + [
                            source for group in plan.groups for source in group.sources
                        ]
                        if not sources:
                            continue
                        event = {
                            "committed_at": datetime.now(UTC).isoformat(),
                            "sources": sources,
                            "destinations": [group.destination for group in plan.groups],
                            "weight_version": run.result["version"],
                        }
                        run.result.setdefault("repack", []).append(event)
                        run.server_ids = [server for server in run.server_ids if server not in sources]
                        run.result["remaining_replicas"] = list(run.server_ids)
                        _write_result(run.path, run.result)
                        future = executor.submit(
                            _drain_repacked_sources,
                            runtime.rollout_router,
                            sources,
                            run.result["version"],
                            _remaining_stage_timeout(run, timeout),
                        )
                        repacking[future] = (sources, run, event)
            pending = [run.future for run in runs if run.future is not None] + list(repacking) + list(recovering)
            if pending:
                wait_futures(pending, timeout=min(1.0, repack_interval), return_when=FIRST_COMPLETED)
            elif task_source is not None and not task_source.draining:
                time.sleep(1.0)
    except Exception as exc:
        for run in runs:
            if run.future is None and run.result["status"] in ("PASS", "PARTIAL", "FAIL"):
                continue
            run.result.update(status="FAIL", failed_stage=run.stage, error=str(exc))
            run.result["execution_status"] = "FAIL"
            run.result["quality_complete"] = False
            run.result["metrics"] = {
                key: value for key, value in run.result["metrics"].items() if not key.startswith("val-core/")
            }
            run.result["timing_s"]["total"] = time.perf_counter() - run.started
            run.result["finished_at"] = datetime.now(UTC).isoformat()
            _write_result(run.path, run.result)
            if run.future is not None:
                run.future.cancel()
        raise
    finally:
        # Ray job teardown owns unfinished requests and loads.
        executor.shutdown(wait=False, cancel_futures=True)
        efficiency.close()


def _build_watcher(config: DictConfig, queue: EvaluationQueue | EvaluationQueueClient) -> CheckpointRunWatcher | None:
    watch_runs = OmegaConf.select(config, "validation.queue.watch_runs", default=[]) or []
    if not watch_runs:
        return None
    bootstrap = (
        config.validation.checkpoints[0]
        if config.validation.checkpoints
        else OmegaConf.select(config, "actor_rollout_ref.model.path")
    )
    return CheckpointRunWatcher(
        queue=queue,
        watch_runs=[Path(run_dir) for run_dir in watch_runs],
        bootstrap_model_dir=Path(str(bootstrap)),
        persistent_roots=[Path(root) for root in config.validation.persistent_roots],
        interval_s=float(OmegaConf.select(config, "validation.queue.watch_interval_s", default=20.0)),
        include_base=bool(OmegaConf.select(config, "validation.queue.watch_include_base", default=False)),
    )


def _run_worker(config: DictConfig) -> None:
    from datasets import load_dataset

    _validate_config(config)
    samples = load_dataset("parquet", data_files={"test": str(config.validation.data_path)}, split="test").to_list()
    if not samples:
        raise ValueError("validation dataset is empty")
    resume_results = _read_resume_results(config, samples)
    colocated_server = None
    colocated_thread = None
    watcher = None
    if OmegaConf.select(config, "validation.queue.server_port") and OmegaConf.select(
        config, "validation.queue.endpoint"
    ):
        queue = EvaluationQueue(
            Path(config.validation.queue.state_dir or Path(config.validation.result_dir) / "queue"),
            evaluation_profile(config),
            [Path(root) for root in config.validation.persistent_roots],
        )
        colocated_server = create_server(
            queue, config.validation.queue.server_host, config.validation.queue.server_port
        )
        colocated_thread = threading.Thread(
            target=colocated_server.serve_forever, name="eval-queue-server", daemon=True
        )
        colocated_thread.start()
        watcher = _build_watcher(config, queue)
        if watcher is not None:
            watcher.start()
    elif OmegaConf.select(config, "validation.queue.endpoint"):
        watcher = _build_watcher(
            config,
            EvaluationQueueClient(str(config.validation.queue.endpoint), config.validation.stage_timeout_s),
        )
        if watcher is not None:
            watcher.start()

    try:
        task_source = None
        if OmegaConf.select(config, "validation.queue.endpoint"):
            task_source = EvaluationTaskSource(config)
            config = task_source.prepare(config)
            if config is None:
                return
        tq.init(config.transfer_queue)
        try:
            runtime = initialize_evaluation_runtime(config)
            _run_checkpoints(config, runtime, samples, resume_results, task_source)
        finally:
            tq.close()
    finally:
        if watcher is not None:
            watcher.stop()
        if colocated_server is not None:
            colocated_server.shutdown()
            if colocated_thread is not None:
                colocated_thread.join(timeout=5.0)
            colocated_server.server_close()


@ray.remote
def main_task(config: DictConfig) -> None:
    _run_worker(config)


def _enqueue_initial_checkpoints(config: DictConfig, queue: EvaluationQueue) -> None:
    if not config.validation.checkpoints:
        return
    raw_entries = OmegaConf.select(config, "validation.resume_results", default=[])
    resume_entries = (
        OmegaConf.to_container(raw_entries, resolve=True) if OmegaConf.is_config(raw_entries) else raw_entries
    ) or []
    raw_protocol = OmegaConf.select(config, "validation.resume_protocol", default={})
    resume_protocol = (
        OmegaConf.to_container(raw_protocol, resolve=True) if OmegaConf.is_config(raw_protocol) else raw_protocol
    ) or {}
    for checkpoint in config.validation.checkpoints:
        resolved = str(resolve_checkpoint_path(str(checkpoint)).resolve())
        matched_entries = []
        for entry in resume_entries:
            try:
                manifest = json.loads(Path(entry["path"]).read_bytes())
                manifest_ckpt = str(resolve_checkpoint_path(manifest["checkpoint"]).resolve())
            except Exception:
                manifest_ckpt = ""
            if manifest_ckpt == resolved or len(config.validation.checkpoints) == 1:
                matched_entries.append(entry)
        queue.submit(
            resolved,
            resume_results=matched_entries or None,
            resume_protocol=resume_protocol if matched_entries else None,
        )


@hydra.main(config_path="config", config_name="validation", version_base=None)
def main(config: DictConfig) -> None:
    OmegaConf.resolve(config)
    _validate_config(config)
    if OmegaConf.select(config, "validation.queue.endpoint"):
        if not ray.is_initialized():
            os.environ.setdefault("RAY_OVERRIDE_JOB_RUNTIME_ENV", "1")
            ray.init(
                address=os.environ.get("RAY_ADDRESS"),
                runtime_env=ray_runtime_env_from_env({"TOKENIZERS_PARALLELISM": "true"}),
            )
        ray.get(main_task.remote(config))
        return

    state_dir = Path(
        OmegaConf.select(config, "validation.queue.state_dir") or Path(config.validation.result_dir) / "queue"
    )
    queue = EvaluationQueue(
        state_dir,
        evaluation_profile(config),
        [Path(root) for root in config.validation.persistent_roots],
    )
    _enqueue_initial_checkpoints(config, queue)
    watch_dir = OmegaConf.select(config, "validation.queue.watch_dir")
    dir_watcher = (
        CheckpointDirectoryWatcher(
            Path(str(watch_dir)),
            queue,
            interval_s=float(OmegaConf.select(config, "validation.queue.watch_interval_s", default=5.0)),
        )
        if watch_dir
        else None
    )
    run_watcher = _build_watcher(config, queue)
    host = str(OmegaConf.select(config, "validation.queue.server_host") or "127.0.0.1")
    port = int(OmegaConf.select(config, "validation.queue.server_port") or 0)
    coordinator_only = bool(OmegaConf.select(config, "validation.queue.coordinator_only", default=False))
    if dir_watcher is not None:
        dir_watcher.start()
    if run_watcher is not None:
        run_watcher.start()
    try:
        with create_server(queue, host, port) as server:
            server_port = int(getattr(server, "server_port", port) or port)
            endpoint = format_client_endpoint(host, server_port)
            (state_dir / "endpoint.txt").write_text(endpoint + "\n")
            logger.info("Evaluation queue service listening on %s (state_dir=%s)", endpoint, state_dir)
            if coordinator_only:
                server.serve_forever()
                return
            server_thread = threading.Thread(target=server.serve_forever, name="eval-queue-server", daemon=True)
            server_thread.start()
            try:
                worker_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
                worker_config.validation.queue.endpoint = endpoint
                worker_config.validation.queue.server_port = None
                worker_config.validation.queue.watch_dir = None
                worker_config.validation.queue.watch_runs = []
                if not OmegaConf.select(worker_config, "validation.queue.worker_id"):
                    worker_config.validation.queue.worker_id = f"local-worker:{uuid.uuid4().hex}"
                if not ray.is_initialized():
                    os.environ.setdefault("RAY_OVERRIDE_JOB_RUNTIME_ENV", "1")
                    ray.init(
                        address=os.environ.get("RAY_ADDRESS"),
                        runtime_env=ray_runtime_env_from_env({"TOKENIZERS_PARALLELISM": "true"}),
                    )
                # The embedded queue lives in this driver, so its loopback URL
                # must stay local. Inference actors still use the Ray GPU pool.
                _run_worker(worker_config)
            finally:
                if hasattr(server, "shutdown"):
                    server.shutdown()
                server_thread.join(timeout=5.0)
    finally:
        if run_watcher is not None:
            run_watcher.stop()
        if dir_watcher is not None:
            dir_watcher.stop()


def cli_main(argv: list[str] | None = None) -> Any:
    import sys

    args = sys.argv[1:] if argv is None else argv
    if has_queue_cli_args(args):
        return run_queue_cli(args)
    return main()


if __name__ == "__main__":
    cli_main()
