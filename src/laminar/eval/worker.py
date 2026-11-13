# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint queue consumption; inference and scoring remain with eval owners."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from laminar.eval.router import validation_replica_groups
from laminar.eval.server import (
    EvaluationQueue,
    EvaluationQueueClient,
    _digest,
    has_complete_hf_weights,
    resolve_checkpoint_path,
)

logger = logging.getLogger(__name__)

_METADATA_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


def model_profile(checkpoint: str, context_length: int | None = None) -> dict[str, Any]:
    root = Path(checkpoint)
    if not (root / "config.json").is_file():
        root = resolve_checkpoint_path(root)
    config = json.loads((root / "config.json").read_text())
    # vLLM owns attention kernels; this Transformers loading option is not a
    # weight/architecture difference. Keep rotary/scaling settings in the profile.
    for key in ("_name_or_path", "transformers_version", "attn_implementation"):
        config.pop(key, None)
    if context_length is not None and "max_position_embeddings" in config:
        if config["max_position_embeddings"] < context_length:
            raise ValueError("checkpoint context capacity is below the evaluation token budget")
        config["max_position_embeddings"] = context_length
    tokenizer = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")
        if (root / name).is_file()
    }
    if "tokenizer_config.json" in tokenizer:
        tokenizer_config = json.loads((root / "tokenizer_config.json").read_text())
        # Local mounted checkpoints do not use this Hub download preference.
        tokenizer_config.pop("local_files_only", None)
        tokenizer["tokenizer_config.json"] = _digest(tokenizer_config)
    return {"config": config, "tokenizer": tokenizer}


def evaluation_profile(config: DictConfig) -> dict[str, Any]:
    """Bind the concrete dataset, model, task and sampling used by this worker."""
    rollout = config.actor_rollout_ref.rollout
    configured_task_path = OmegaConf.select(
        rollout, "custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path"
    )
    task_config_sha256 = (
        hashlib.sha256(Path(str(configured_task_path)).read_bytes()).hexdigest() if configured_task_path else None
    )
    raw_framework = OmegaConf.select(rollout, "custom.agent_framework")
    framework = OmegaConf.to_container(raw_framework, resolve=True) if raw_framework is not None else {}
    framework.pop("log_dir", None)
    framework.pop("gateway_count", None)
    for runner in framework.get("agent_runners", {}).values():
        runner.pop("max_concurrent_sessions", None)
        kwargs = runner.get("runner_kwargs", {})
        if "task_config_path" in kwargs:
            kwargs["task_config_path"] = hashlib.sha256(Path(kwargs["task_config_path"]).read_bytes()).hexdigest()
    model_options = OmegaConf.to_container(config.actor_rollout_ref.model, resolve=True)
    model_path = model_options.pop("path", None)
    reference_checkpoint = config.validation.checkpoints[0] if config.validation.checkpoints else model_path
    if not reference_checkpoint:
        raise ValueError("validation.checkpoints or actor_rollout_ref.model.path is required for evaluation_profile")
    return {
        "dataset_sha256": hashlib.sha256(Path(config.validation.data_path).read_bytes()).hexdigest(),
        "sampling": OmegaConf.to_container(rollout.val_kwargs, resolve=True),
        "seed": rollout.seed,
        "full_determinism": rollout.full_determinism,
        "task_config_sha256": task_config_sha256,
        "framework": framework,
        "model": model_profile(
            str(reference_checkpoint),
            OmegaConf.select(rollout, "max_model_len") or rollout.prompt_length + rollout.response_length,
        ),
        "model_options": model_options,
        "prompt_length": rollout.prompt_length,
        "response_length": rollout.response_length,
        "rollout_model": {
            name: OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
            for name in ("dtype", "quantization", "multi_turn", "max_model_len")
            for value in [OmegaConf.select(rollout, name)]
        },
    }


class EvaluationTaskSource:
    def __init__(self, config: DictConfig):
        if not config.validation.queue.worker_id:
            raise ValueError("queue worker_id must identify the exact resident submission")
        self.worker = str(config.validation.queue.worker_id)
        self.client = EvaluationQueueClient(str(config.validation.queue.endpoint), config.validation.stage_timeout_s)
        state = self.client.request("/tasks")
        self.profile = evaluation_profile(config)
        self.profile_sha256 = _digest(self.profile)
        if state["profile"] != self.profile or state["profile_sha256"] != self.profile_sha256:
            raise ValueError("worker dataset/model/task/sampling differs from the queue profile")
        self.tasks: dict[int, dict[str, Any]] = {}
        self.next_version = 1
        self.draining = not state["accepting"]
        self.initial: list[tuple[int, str]] = []
        self.idle_groups: list[list[str]] = []
        self.result_uri = config.validation.queue.result_uri

    def claim(self, group: list[str], *, version: int | None = None) -> tuple[int, str] | None:
        version = self.next_version if version is None else version
        response = self.client.request(
            "/claim",
            {
                "worker": self.worker,
                "group": group,
                "claim_id": f"{self.worker}:version:{version}",
                "profile_sha256": self.profile_sha256,
            },
        )
        self.draining = response["draining"]
        task = response["task"]
        if task is None:
            return None
        if task["status"] != "RUNNING":
            raise ValueError("claim already finished; reconcile the previous worker before restarting")
        context_length = self.profile["rollout_model"]["max_model_len"] or (
            self.profile["prompt_length"] + self.profile["response_length"]
        )
        if model_profile(task["checkpoint"], context_length) != self.profile["model"]:
            raise ValueError("claimed checkpoint architecture/tokenizer differs from the queue profile")
        self.tasks[version] = task
        self.next_version = max(self.next_version, version + 1)
        return version, task["checkpoint"]

    def prepare(self, config: DictConfig) -> DictConfig | None:
        rollout = config.actor_rollout_ref.rollout
        size = rollout.tensor_model_parallel_size * rollout.data_parallel_size * rollout.pipeline_model_parallel_size
        count = rollout.nnodes * rollout.n_gpus_per_node // size
        groups = validation_replica_groups(
            [f"rollout_server_{index}" for index in range(count)],
            [1],
        )
        # Claim before initialization so occupied groups load their first task
        # only once. Unclaimed groups bootstrap the same compatible model and
        # remain idle until a later claim reloads them.
        while not self.initial:
            self.idle_groups = []
            for version, group in groups.items():
                task = self.claim(group, version=version)
                if task is None:
                    self.idle_groups.append(group)
                else:
                    self.initial.append(task)
            if not self.initial:
                if self.draining:
                    return None
                time.sleep(1.0)
        self.next_version = len(groups) + 1
        prepared = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        bootstrap = self.initial[0][1]
        prepared.validation.checkpoints = [
            self.tasks.get(version, {}).get("checkpoint", bootstrap) for version in groups
        ]
        return prepared

    def resume_config(self, config: DictConfig, version: int) -> DictConfig:
        task = self.tasks[version]
        result = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        result.validation.checkpoints = [task["checkpoint"]]
        result.validation.resume_results = task.get("resume_results", [])
        result.validation.resume_protocol = task.get("resume_protocol", {})
        return result

    def finish(self, version: int, path: Path, status: str) -> None:
        task = self.tasks[version]
        artifact = str(path) if not self.result_uri else self.result_uri.rstrip("/") + "/" + path.name
        self.client.request(
            "/finish",
            {
                "task_id": task["task_id"],
                "owner": task["owner"],
                "result": {
                    "status": status,
                    "artifact": artifact,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                },
            },
        )

    def release_replicas(self, version: int, remaining_replicas: list[str]) -> dict[str, Any]:
        """Release drained sources while retaining the original claim identity."""
        task = self.tasks[version]
        released = self.client.request(
            "/release-replicas",
            {"task_id": task["task_id"], "owner": task["owner"], "remaining_replicas": remaining_replicas},
        )
        self.tasks[version] = released
        return released

    def has_queued(self) -> bool:
        """Return whether the queue is accepting and has a waiting task to claim."""
        if self.draining:
            return False
        state = self.client.request("/tasks")
        self.draining = not state["accepting"] or self.worker in state.get("drained_workers", [])
        if self.draining:
            return False
        return any(task.get("status") == "QUEUED" for task in state.get("tasks", {}).values())


def sync_checkpoint_model_metadata(checkpoint_dir: Path, bootstrap_model_dir: Path) -> None:
    """Align exported HF config/tokenizer files with the bootstrap model while keeping a backup."""
    checkpoint_dir = checkpoint_dir.resolve()
    bootstrap_model_dir = bootstrap_model_dir.resolve()
    if checkpoint_dir == bootstrap_model_dir:
        return
    backup_dir = checkpoint_dir / ".megatron_export_meta"
    if not backup_dir.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        for name in _METADATA_FILES:
            target = checkpoint_dir / name
            if target.is_file():
                shutil.copy2(target, backup_dir / name)
    for name in _METADATA_FILES:
        source = bootstrap_model_dir / name
        target = checkpoint_dir / name
        if source.is_file():
            shutil.copy2(source, target)
        elif target.is_file():
            target.unlink()


def is_complete_hf_checkpoint(step_dir: Path) -> Path | None:
    """Return the exported Hugging Face directory once committed and all weight shards are non-empty."""
    try:
        step = int(step_dir.name.rsplit("global_step_", 1)[-1])
    except ValueError:
        return None
    if step_dir.name != f"global_step_{step}" or step < 0:
        return None
    checkpoint_root = step_dir.parent
    if (checkpoint_root / "latest_checkpointed_iteration.txt.tmp").exists():
        return None
    tracker_file = checkpoint_root / "latest_checkpointed_iteration.txt"
    latest_step: int | None = None
    if tracker_file.is_file():
        try:
            latest_step = int(tracker_file.read_text().strip())
        except ValueError:
            return None
        if step > latest_step:
            return None

    megatron_hf_dir = step_dir / "actor" / "model" / "huggingface"
    if megatron_hf_dir.is_dir():
        # Megatron rank 0 writes ckpt_contents.json last after model/huggingface shards finish.
        if not (step_dir / "actor" / "ckpt_contents.json").is_file():
            return None
        return megatron_hf_dir if has_complete_hf_weights(megatron_hf_dir) else None

    fsdp_hf_dir = step_dir / "actor" / "huggingface"
    if fsdp_hf_dir.is_dir() and latest_step is not None:
        # FSDP writes config/tokenizer before weights; require the committed tracker marker.
        return fsdp_hf_dir if has_complete_hf_weights(fsdp_hf_dir) else None
    return None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def mirror_run_validation_result(
    result: dict[str, Any], target_result_dir: Path, persistent_roots: list[Path]
) -> Path | None:
    """Mirror a finished evaluation result into its originating training run's validation directory."""
    resolved_target = target_result_dir.resolve()
    roots = [root.resolve() for root in persistent_roots]
    if not target_result_dir.is_absolute() or not any(resolved_target.is_relative_to(root) for root in roots):
        raise ValueError(f"target validation result directory must be under persistent_roots: {target_result_dir}")
    target_result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_resolved = str(Path(result["checkpoint"]).resolve())
    queue_task_id = result.get("queue_task_id")
    max_version = 0
    for existing_path in sorted(target_result_dir.glob("version_*.json")):
        try:
            stem_version = int(existing_path.stem.split("version_")[-1])
            max_version = max(max_version, stem_version)
            existing = json.loads(existing_path.read_text())
        except (OSError, ValueError):
            continue
        existing_ckpt = existing.get("checkpoint")
        if isinstance(existing_ckpt, str) and str(Path(existing_ckpt).resolve()) == checkpoint_resolved:
            return existing_path
        if queue_task_id and existing.get("queue_task_id") == queue_task_id:
            return existing_path
    next_version = max_version + 1
    mirrored = dict(result)
    mirrored["worker_version"] = result.get("version")
    mirrored["version"] = next_version
    destination = target_result_dir / f"version_{next_version:06d}.json"
    _atomic_write_json(destination, mirrored)
    return destination


class CheckpointRunWatcher:
    """Watch one or more training run directories and enqueue completed HF checkpoints."""

    def __init__(
        self,
        queue: EvaluationQueue | EvaluationQueueClient,
        watch_runs: list[Path],
        bootstrap_model_dir: Path,
        persistent_roots: list[Path],
        *,
        interval_s: float = 20.0,
        include_base: bool = False,
    ) -> None:
        self.queue = queue
        self.watch_runs = [Path(run_dir).resolve() for run_dir in watch_runs]
        self.bootstrap_model_dir = Path(bootstrap_model_dir).resolve()
        self.persistent_roots = [Path(root).resolve() for root in persistent_roots]
        self.interval_s = float(interval_s)
        self.include_base = bool(include_base)
        self._submitted: set[tuple[str, str]] = set()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def _submit(self, checkpoint: Path, source: dict[str, Any]) -> dict[str, Any]:
        if isinstance(self.queue, EvaluationQueue):
            return self.queue.submit(str(checkpoint), source=source)
        return self.queue.request("/tasks", {"checkpoint": str(checkpoint), "source": source})

    def _read_state(self) -> dict[str, Any]:
        if isinstance(self.queue, EvaluationQueue):
            return self.queue.read()
        return self.queue.request("/tasks")

    @staticmethod
    def _evaluated_checkpoints(val_dir: Path) -> set[str]:
        evaluated: set[str] = set()
        if not val_dir.is_dir():
            return evaluated
        candidates = list(val_dir.glob("version_*.json"))
        completed_dir = val_dir / "completed_versions"
        if completed_dir.is_dir():
            candidates.extend(completed_dir.glob("*.json"))
        for result_file in candidates:
            try:
                data = json.loads(result_file.read_text())
            except (OSError, ValueError):
                continue
            if data.get("status") in ("PASS", "PARTIAL") and isinstance(data.get("checkpoint"), str):
                evaluated.add(str(Path(data["checkpoint"]).resolve()))
        return evaluated

    @staticmethod
    def _discover_step_dirs(run_dir: Path) -> list[tuple[int, Path]]:
        patterns = (
            "*/checkpoints/*/global_step_*",
            "checkpoints/*/global_step_*",
            "checkpoints/global_step_*",
            "global_step_*",
        )
        found: dict[Path, int] = {}
        for pattern in patterns:
            for step_dir in run_dir.glob(pattern):
                if not step_dir.is_dir():
                    continue
                try:
                    step = int(step_dir.name.rsplit("global_step_", 1)[-1])
                except ValueError:
                    continue
                found[step_dir.resolve()] = step
        return sorted(((step, path) for path, step in found.items()), key=lambda item: (item[0], str(item[1])))

    def poll_once(self) -> list[dict[str, Any]]:
        """Scan watched runs once, enqueue new checkpoints, and mirror finished results."""
        submitted_tasks: list[dict[str, Any]] = []
        state = self._read_state()
        accepting = state.get("accepting", True)
        existing = {task["checkpoint"] for task in state.get("tasks", {}).values()}
        try:
            base_resolved_path = resolve_checkpoint_path(self.bootstrap_model_dir).resolve()
        except ValueError:
            base_resolved_path = self.bootstrap_model_dir
        base_resolved = str(base_resolved_path)
        for run_dir in self.watch_runs:
            if not run_dir.is_dir():
                continue
            val_dir = run_dir / "validation"
            evaluated = self._evaluated_checkpoints(val_dir)
            if self.include_base and accepting and base_resolved not in evaluated:
                key = ("__base__", base_resolved)
                if base_resolved in existing:
                    self._submitted.add(key)
                elif key not in self._submitted:
                    try:
                        task = self._submit(base_resolved_path, {"step": 0, "kind": "base"})
                    except ValueError:
                        logger.warning("Skipping base checkpoint submission due to queue conflict: %s", base_resolved)
                    else:
                        self._submitted.add(key)
                        existing.add(base_resolved)
                        submitted_tasks.append(task)
            if not accepting:
                continue
            for step, step_dir in self._discover_step_dirs(run_dir):
                hf_dir = is_complete_hf_checkpoint(step_dir)
                if hf_dir is None:
                    continue
                hf_resolved = str(hf_dir.resolve())
                key = (str(run_dir), hf_resolved)
                if hf_resolved in evaluated or hf_resolved in existing or key in self._submitted:
                    if hf_resolved in existing:
                        self._submitted.add(key)
                    continue
                sync_checkpoint_model_metadata(hf_dir, base_resolved_path)
                source = {
                    "run": run_dir.name,
                    "run_dir": str(run_dir),
                    "result_dir": str(val_dir),
                    "step": step,
                    "kind": "training_checkpoint",
                }
                try:
                    task = self._submit(hf_dir, source)
                except ValueError:
                    logger.warning("Skipping watched checkpoint submission due to queue conflict: %s", hf_resolved)
                    continue
                self._submitted.add(key)
                existing.add(hf_resolved)
                submitted_tasks.append(task)

        for task in self._read_state().get("tasks", {}).values():
            if task.get("status") != "FINISHED":
                continue
            artifact_path = Path(task.get("result", {}).get("artifact", ""))
            if not artifact_path.is_file():
                continue
            try:
                payload = json.loads(artifact_path.read_text())
            except (OSError, ValueError):
                continue
            source = task.get("source") or {}
            target_dir = source.get("result_dir")
            if isinstance(target_dir, str) and target_dir:
                mirror_run_validation_result(payload, Path(target_dir), self.persistent_roots)
            elif self.include_base and source.get("kind") == "base":
                for run_dir in self.watch_runs:
                    if run_dir.is_dir():
                        mirror_run_validation_result(payload, run_dir / "validation", self.persistent_roots)
        return submitted_tasks

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.poll_once()
            except Exception:
                logger.exception("CheckpointRunWatcher poll failed")
            self._stop_event.wait(self.interval_s)

    def start(self) -> None:
        self.poll_once()
        self._thread = threading.Thread(target=self._run_loop, name="eval-checkpoint-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
