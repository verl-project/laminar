# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Persistent checkpoint admission shared by independent evaluation residents.

The service schedules checkpoint tasks only. Worker-local Laminar owners retain
session admission, version routing, engine reload and scoring responsibility.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import socket
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def resolve_checkpoint_path(checkpoint: str | Path) -> Path:
    """Resolve a direct Hugging Face model directory or a trainer step/actor directory."""
    path = Path(checkpoint)
    if not path.is_absolute():
        raise ValueError("checkpoint must be a mounted Hugging Face model directory")
    for candidate in (
        path,
        path / "actor" / "model" / "huggingface",
        path / "actor" / "huggingface",
        path / "actor",
        path / "model" / "huggingface",
        path / "huggingface",
    ):
        if (candidate / "config.json").is_file():
            return candidate
    raise ValueError("checkpoint must be a mounted Hugging Face model directory")


def has_complete_hf_weights(path: Path) -> bool:
    """Check a watched HF export before admitting it to the durable queue."""
    try:
        for index_name, weight_name in (
            ("model.safetensors.index.json", "model.safetensors"),
            ("pytorch_model.bin.index.json", "pytorch_model.bin"),
        ):
            index_path = path / index_name
            if index_path.is_file():
                index = json.loads(index_path.read_text())
                weight_map = index.get("weight_map") if isinstance(index, dict) else None
                if not isinstance(weight_map, dict) or not weight_map:
                    return False
                shards = list(weight_map.values())
                if any(not isinstance(shard, str) or not shard for shard in shards):
                    return False
                return all((path / shard).is_file() and (path / shard).stat().st_size > 0 for shard in set(shards))
            weight_path = path / weight_name
            if weight_path.is_file():
                return weight_path.stat().st_size > 0
    except (OSError, ValueError):
        return False
    return False


class EvaluationQueue:
    """Single coordinator; accepted changes are persisted before acknowledgement.

    Running claims survive coordinator restart. There is deliberately no expiry
    or automatic retry: an unknown worker outcome must be reconciled first.
    """

    def __init__(self, state_dir: Path, profile: dict[str, Any], persistent_roots: list[Path]):
        if not state_dir.is_absolute() or not any(
            state_dir.resolve().is_relative_to(root.resolve()) for root in persistent_roots
        ):
            raise ValueError("evaluation queue state must be under a declared persistent root")
        if not profile:
            raise ValueError("an audited evaluation profile is required")
        state_dir.mkdir(parents=True, exist_ok=True)
        self.path = state_dir / "queue.json"
        self.lock = threading.Lock()
        self.profile_sha256 = _digest(profile)
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state["profile_sha256"] != self.profile_sha256:
                raise ValueError("existing queue has a different evaluation profile")
        else:
            self.state = {
                "profile": copy.deepcopy(profile),
                "profile_sha256": self.profile_sha256,
                "accepting": True,
                "tasks": {},
            }
            self._persist(self.state)

    def _persist(self, state: dict[str, Any]) -> None:
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(state, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.path)
        # Publish in memory only after the fsynced file replacement succeeds.
        self.state = state

    def submit(
        self,
        checkpoint: str,
        source: dict[str, Any] | None = None,
        resume_results: list[dict[str, Any]] | None = None,
        resume_protocol: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        path = resolve_checkpoint_path(checkpoint)
        checkpoint = str(path.resolve())
        task_id = _digest({"checkpoint": checkpoint, "profile": self.profile_sha256})
        with self.lock:
            old = self.state["tasks"].get(task_id)
            if old is not None:
                if (
                    (source is not None and old["source"] != source)
                    or (resume_results is not None and old.get("resume_results", []) != resume_results)
                    or (resume_protocol is not None and old.get("resume_protocol", {}) != resume_protocol)
                ):
                    raise ValueError("duplicate checkpoint has conflicting source identity")
                return copy.deepcopy(old)
            if not self.state["accepting"]:
                raise ValueError("evaluation queue is draining")
            state = copy.deepcopy(self.state)
            task = {
                "task_id": task_id,
                "checkpoint": checkpoint,
                "source": source or {},
                "resume_results": resume_results or [],
                "resume_protocol": resume_protocol or {},
                "profile_sha256": self.profile_sha256,
                "status": "QUEUED",
                "submitted_at": datetime.now(UTC).isoformat(),
            }
            state["tasks"][task_id] = copy.deepcopy(task)
            self._persist(state)
            return copy.deepcopy(task)

    def claim(self, worker: str, group: list[str], claim_id: str, profile_sha256: str) -> dict[str, Any]:
        if not worker or not claim_id or not group or len(set(group)) != len(group):
            raise ValueError("worker, unique replica group and claim identity are required")
        if profile_sha256 != self.profile_sha256:
            raise ValueError("worker evaluation profile differs from the queue")
        owner = {"worker": worker, "replica_group": sorted(group), "claim_id": claim_id}
        with self.lock:
            draining = not self.state["accepting"] or worker in self.state.get("drained_workers", [])
            for task in self.state["tasks"].values():
                if any(attempt.get("owner") == owner for attempt in task.get("previous_attempts", [])):
                    raise ValueError("claim belongs to a previous attempt; use a new worker claim identity")
                previous = task.get("owner")
                if previous == owner:
                    return {"task": copy.deepcopy(task), "draining": draining}
                if previous and task["status"] == "RUNNING" and previous["worker"] == worker:
                    if set(task.get("remaining_replicas", previous["replica_group"])).intersection(group):
                        raise ValueError("worker replica group already has an unreconciled task")
            if draining:
                return {"task": None, "draining": True}
            state = copy.deepcopy(self.state)
            task = next((task for task in state["tasks"].values() if task["status"] == "QUEUED"), None)
            if task is not None:
                task.update(status="RUNNING", owner=owner, started_at=datetime.now(UTC).isoformat())
                self._persist(state)
            return {"task": copy.deepcopy(task), "draining": not self.state["accepting"]}

    def release_replicas(self, task_id: str, owner: dict[str, Any], remaining_replicas: list[str]) -> dict[str, Any]:
        """Record a drained Repack subset without changing the immutable claim.

        The worker calls this only after the Router's source drain barrier. A
        finished task may still have a source drain completing independently.
        """
        if (
            not isinstance(remaining_replicas, list)
            or not remaining_replicas
            or any(not isinstance(server_id, str) or not server_id for server_id in remaining_replicas)
            or len(set(remaining_replicas)) != len(remaining_replicas)
        ):
            raise ValueError("remaining_replicas must be a non-empty list of distinct replica ids")
        with self.lock:
            task = self.state["tasks"][task_id]
            if task.get("owner") != owner or task["status"] not in ("RUNNING", "FINISHED"):
                raise ValueError("replica release requires the original running or finished task owner")
            current = set(task.get("remaining_replicas", owner["replica_group"]))
            remaining = set(remaining_replicas)
            if not remaining.issubset(current):
                raise ValueError("remaining_replicas may only shrink the task's current replica set")
            if remaining == current:
                return copy.deepcopy(task)
            state = copy.deepcopy(self.state)
            task = state["tasks"][task_id]
            task["remaining_replicas"] = sorted(remaining)
            self._persist(state)
            return copy.deepcopy(task)

    def finish(self, task_id: str, owner: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        if result.get("status") not in ("PASS", "PARTIAL", "FAIL") or not result.get("artifact"):
            raise ValueError("terminal result status and persistent artifact reference are required")
        digest = result.get("sha256", "")
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("result artifact SHA256 is required")
        with self.lock:
            task = self.state["tasks"][task_id]
            if task.get("owner") != owner:
                raise ValueError("result owner differs from the task claim")
            if task["status"] == "FINISHED":
                if task["result"] != result:
                    raise ValueError("task already has a different terminal result")
                return copy.deepcopy(task)
            state = copy.deepcopy(self.state)
            task = state["tasks"][task_id]
            task.update(status="FINISHED", result=result, finished_at=datetime.now(UTC).isoformat())
            self._persist(state)
            return copy.deepcopy(task)

    def read(self, task_id: str | None = None) -> dict[str, Any]:
        with self.lock:
            return copy.deepcopy(self.state["tasks"][task_id] if task_id else self.state)

    def requeue(
        self,
        task_id: str,
        previous_result_sha256: str,
        review: dict[str, Any],
        resume_results: list[dict[str, Any]],
        resume_protocol: dict[str, Any],
    ) -> dict[str, Any]:
        """Explicit audited repair; retain the previous attempt and imported evidence."""
        if review.get("verdict") not in ("FAIL", "UNKNOWN") or not review.get("artifact") or not review.get("sha256"):
            raise ValueError("requeue requires a caller-audited failure or unknown scoring review")
        with self.lock:
            task = self.state["tasks"][task_id]
            if not self.state["accepting"] or task["status"] != "FINISHED":
                raise ValueError("only a terminal task in an accepting queue can be explicitly requeued")
            if task["result"]["sha256"] != previous_result_sha256:
                raise ValueError("requeue refers to a different previous result")
            state = copy.deepcopy(self.state)
            task = state["tasks"][task_id]
            previous = copy.deepcopy(task)
            previous.pop("previous_attempts", None)
            task.setdefault("previous_attempts", []).append(previous)
            for key in ("owner", "remaining_replicas", "result", "started_at", "finished_at"):
                task.pop(key, None)
            task.update(
                status="QUEUED", resume_results=resume_results, resume_protocol=resume_protocol, repair_review=review
            )
            self._persist(state)
            return copy.deepcopy(task)

    def drain(self, worker: str | None = None) -> dict[str, Any]:
        with self.lock:
            state = copy.deepcopy(self.state)
            if worker is not None:
                if not isinstance(worker, str) or not worker:
                    raise ValueError("worker drain requires an exact worker identity")
                workers = state.setdefault("drained_workers", [])
                if worker not in workers:
                    if not any(task.get("owner", {}).get("worker") == worker for task in state["tasks"].values()):
                        raise ValueError("worker has no existing task claim")
                    workers.append(worker)
                    self._persist(state)
                return {"worker": worker, "draining": True}
            state["accepting"] = False
            self._persist(state)
            return {"accepting": False}


def format_client_endpoint(host: str, port: int) -> str:
    """Format a connectable HTTP URL for a bound queue server host/port."""
    if host in ("0.0.0.0", ""):
        client_host = "127.0.0.1"
    elif host == "::":
        client_host = "[::1]"
    elif ":" in host and not host.startswith("["):
        client_host = f"[{host}]"
    else:
        client_host = host
    return f"http://{client_host}:{port}"


def resolve_queue_endpoint(
    *,
    endpoint: str | None = None,
    state_dir: Path | str | None = None,
    result_dir: Path | str | None = None,
) -> str:
    """Resolve the active evaluation queue HTTP endpoint from args or endpoint.txt."""
    if endpoint:
        return endpoint.rstrip("/")
    candidates: list[Path] = []
    if state_dir:
        candidates.append(Path(state_dir) / "endpoint.txt")
    if result_dir:
        candidates.append(Path(result_dir) / "queue" / "endpoint.txt")
        candidates.append(Path(result_dir) / "endpoint.txt")
    for candidate in candidates:
        if candidate.is_file():
            content = candidate.read_text().strip()
            if content:
                return content.rstrip("/")
    return format_client_endpoint("127.0.0.1", 9000)


class EvaluationQueueClient:
    """Python Client for submitting and controlling tasks on an EvaluationQueue service."""

    def __init__(
        self,
        endpoint: str | None = None,
        timeout: float | None = 30.0,
        *,
        state_dir: Path | str | None = None,
        result_dir: Path | str | None = None,
    ):
        self.endpoint = resolve_queue_endpoint(endpoint=endpoint, state_dir=state_dir, result_dir=result_dir)
        self.opener = build_opener(ProxyHandler({}))
        self.timeout = timeout

    def request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = json.dumps(payload, allow_nan=False).encode() if payload is not None else None
        request = Request(self.endpoint + path, data=body, headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as error:
            raise ValueError(json.load(error)["error"]) from error
        # No retry after an ambiguous mutation. The caller reads the exact task/claim.

    def submit(
        self,
        checkpoint: str | Path | list[str | Path] | tuple[str | Path, ...],
        *extra_checkpoints: str | Path,
        source: dict[str, Any] | None = None,
        resume_results: list[dict[str, Any]] | None = None,
        resume_protocol: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Submit one or more checkpoint directories for evaluation (1-to-1 with ``--submit``)."""
        raw_items = list(checkpoint) if isinstance(checkpoint, (list, tuple)) else [checkpoint]
        raw_items.extend(extra_checkpoints)
        if not raw_items:
            raise ValueError("at least one checkpoint path is required")
        results = []
        for raw_ckpt in raw_items:
            ckpt_path = Path(raw_ckpt).expanduser()
            if not ckpt_path.is_absolute():
                ckpt_path = (Path.cwd() / ckpt_path).resolve()
            payload: dict[str, Any] = {"checkpoint": str(ckpt_path)}
            if source is not None:
                payload["source"] = source
            if resume_results is not None:
                payload["resume_results"] = resume_results
            if resume_protocol is not None:
                payload["resume_protocol"] = resume_protocol
            results.append(self.request("/tasks", payload))
        return results[0] if len(results) == 1 else results

    def status(self, task_id: str | None = None) -> dict[str, Any]:
        """Query the full evaluation queue state or a single task by task_id (1-to-1 with ``--status``)."""
        return self.request(f"/tasks/{task_id}" if task_id else "/tasks")

    read = status

    def drain(self, worker: str | None = None) -> dict[str, Any]:
        """Stop accepting new tasks globally, or drain a specific worker (1-to-1 with ``--drain``)."""
        return self.request("/drain", {"worker": worker} if worker else {})

    def requeue(
        self,
        task_id: str,
        previous_sha256: str | None = None,
        review: dict[str, Any] | None = None,
        resume_results: list[dict[str, Any]] | None = None,
        resume_protocol: dict[str, Any] | None = None,
        *,
        previous_result_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Requeue a finished task with audited repair evidence (1-to-1 with ``--requeue``)."""
        resolved_sha256 = previous_sha256 or previous_result_sha256
        if not resolved_sha256 or review is None:
            raise ValueError("requeue requires previous_sha256 and review")
        return self.request(
            "/requeue",
            {
                "task_id": task_id,
                "previous_result_sha256": resolved_sha256,
                "review": review,
                "resume_results": resume_results or [],
                "resume_protocol": resume_protocol or {},
            },
        )


def create_server(queue: EvaluationQueue, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            self.respond()

        def do_POST(self) -> None:
            self.respond()

        def respond(self) -> None:
            try:
                body = None
                if self.command == "POST":
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.command == "GET" and self.path == "/tasks":
                    value = queue.read()
                elif self.command == "GET" and self.path.startswith("/tasks/"):
                    value = queue.read(self.path.removeprefix("/tasks/"))
                elif self.command == "POST" and self.path == "/tasks":
                    value = queue.submit(**body)
                elif self.command == "POST" and self.path == "/claim":
                    value = queue.claim(**body)
                elif self.command == "POST" and self.path == "/finish":
                    value = queue.finish(**body)
                elif self.command == "POST" and self.path == "/release-replicas":
                    value = queue.release_replicas(**body)
                elif self.command == "POST" and self.path == "/requeue":
                    value = queue.requeue(**body)
                elif self.command == "POST" and self.path == "/drain":
                    value = queue.drain(**body)
                else:
                    raise KeyError(self.path)
                status = 200
            except (ValueError, TypeError, KeyError) as error:
                status = 400
                value = {"error": str(error)}
            data = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    class Server(ThreadingHTTPServer):
        address_family = socket.AF_INET6 if ":" in host else socket.AF_INET

    return Server((host, port), Handler)


class CheckpointDirectoryWatcher:
    """Poll a training checkpoint directory and submit completed checkpoints to EvaluationQueue."""

    def __init__(self, watch_dir: Path, queue: EvaluationQueue, *, interval_s: float = 5.0):
        if not watch_dir.is_absolute():
            raise ValueError("watch_dir must be an absolute path")
        if interval_s <= 0:
            raise ValueError("watch_interval_s must be positive")
        self.watch_dir = watch_dir
        self.queue = queue
        self.interval_s = float(interval_s)
        self._submitted: set[str] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _discover_ready_checkpoints(self) -> list[Path]:
        if not self.watch_dir.is_dir():
            return []
        tracker_tmp = self.watch_dir / "latest_checkpointed_iteration.txt.tmp"
        if tracker_tmp.exists():
            return []
        tracker_file = self.watch_dir / "latest_checkpointed_iteration.txt"
        latest_step: int | None = None
        if tracker_file.is_file():
            try:
                latest_step = int(tracker_file.read_text().strip())
            except ValueError:
                return []

        step_entries: list[tuple[int, Path]] = []
        named_entries: list[tuple[str, Path]] = []
        try:
            children = list(self.watch_dir.iterdir())
        except OSError:
            return []

        for entry in children:
            if not entry.is_dir() or entry.name.startswith((".", "_")):
                continue
            if entry.name.startswith("global_step_"):
                suffix = entry.name.removeprefix("global_step_")
                try:
                    step = int(suffix)
                except ValueError:
                    continue
                if entry.name != f"global_step_{step}" or step < 0:
                    continue
                # The trainer commits this marker only after actor state and
                # data.pt are saved, including the very first checkpoint.
                if latest_step is None or step > latest_step:
                    continue
                try:
                    hf_path = resolve_checkpoint_path(entry).resolve()
                except ValueError:
                    continue
                if not has_complete_hf_weights(hf_path):
                    continue
                step_entries.append((step, hf_path))
            else:
                if any(entry.glob("*.tmp")):
                    continue
                try:
                    hf_path = resolve_checkpoint_path(entry).resolve()
                except ValueError:
                    continue
                if not has_complete_hf_weights(hf_path):
                    continue
                named_entries.append((entry.name, hf_path))

        step_entries.sort(key=lambda item: item[0])
        named_entries.sort(key=lambda item: item[0])
        return [path for _, path in step_entries] + [path for _, path in named_entries]

    def scan_once(self) -> list[dict[str, Any]]:
        """Discover and enqueue any newly completed checkpoints in watch_dir."""
        state = self.queue.read()
        if not state.get("accepting", False):
            return []
        existing = {task["checkpoint"] for task in state.get("tasks", {}).values()}
        submitted_tasks: list[dict[str, Any]] = []
        for hf_path in self._discover_ready_checkpoints():
            resolved = str(hf_path)
            if resolved in self._submitted or resolved in existing:
                self._submitted.add(resolved)
                continue
            try:
                task = self.queue.submit(resolved)
            except ValueError:
                continue
            self._submitted.add(resolved)
            submitted_tasks.append(task)
        return submitted_tasks

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.scan_once()
            if not self.queue.read().get("accepting", False):
                break

    def start(self) -> None:
        self.scan_once()
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="eval-checkpoint-watcher", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=max(1.0, self.interval_s))
        self._thread = None


def has_queue_cli_args(argv: list[str]) -> bool:
    """Return True when argv invokes a queue client control operation."""
    control_flags = {"--submit", "--status", "--drain", "--requeue"}
    return any(arg in control_flags or any(arg.startswith(f"{flag}=") for flag in control_flags) for arg in argv)


def run_queue_cli(argv: list[str] | None = None) -> Any:
    """Execute client operations (--submit, --status, --drain, --requeue) against an evaluation service."""
    parser = argparse.ArgumentParser(
        description="Submit or manage checkpoints on a running Laminar evaluation service."
    )
    parser.add_argument("--endpoint", help="Evaluation queue HTTP endpoint (e.g. http://127.0.0.1:9000)")
    parser.add_argument("--state-dir", type=Path, help="Evaluation queue state directory containing endpoint.txt")
    parser.add_argument("--result-dir", type=Path, help="Evaluation result directory containing queue/endpoint.txt")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP request timeout in seconds")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--submit", nargs="+", metavar="CHECKPOINT", help="Submit one or more checkpoints for evaluation"
    )
    group.add_argument(
        "--status",
        nargs="?",
        const="",
        metavar="TASK_ID",
        help="Query queue state or a specific task_id",
    )
    group.add_argument("--drain", action="store_true", help="Drain the evaluation queue (or a specific --worker)")
    group.add_argument("--requeue", metavar="TASK_ID", help="Requeue a finished task with audited repair evidence")

    parser.add_argument("--worker", help="Worker ID for --drain")
    parser.add_argument("--source", help="Optional JSON object describing checkpoint provenance for --submit")
    parser.add_argument("--resume-results", help="Optional JSON list of resume manifest references")
    parser.add_argument("--resume-protocol", help="Optional JSON object binding the audited resume protocol")
    parser.add_argument("--previous-sha256", help="Previous result SHA256 required for --requeue")
    parser.add_argument("--review", help="Audited review JSON object required for --requeue")

    args = parser.parse_args(argv)
    client = EvaluationQueueClient(
        endpoint=args.endpoint,
        timeout=args.timeout,
        state_dir=args.state_dir,
        result_dir=args.result_dir,
    )

    if args.submit is not None:
        output: Any = client.submit(
            *args.submit,
            source=json.loads(args.source) if args.source else None,
            resume_results=json.loads(args.resume_results) if args.resume_results else None,
            resume_protocol=json.loads(args.resume_protocol) if args.resume_protocol else None,
        )
    elif args.status is not None:
        output = client.status(args.status or None)
    elif args.drain:
        output = client.drain(worker=args.worker)
    else:
        if not args.previous_sha256 or not args.review:
            parser.error("--requeue requires --previous-sha256 and --review")
        output = client.requeue(
            args.requeue,
            previous_sha256=args.previous_sha256,
            review=json.loads(args.review),
            resume_results=json.loads(args.resume_results) if args.resume_results else None,
            resume_protocol=json.loads(args.resume_protocol) if args.resume_protocol else None,
        )
    print(json.dumps(output, indent=2, allow_nan=False))
    return output


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        import sys

        argv = sys.argv[1:]
    if has_queue_cli_args(argv):
        run_queue_cli(argv)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--persistent-root", type=Path, action="append", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--watch-dir", type=Path, default=None)
    parser.add_argument("--watch-interval-s", type=float, default=5.0)
    args = parser.parse_args(argv)
    queue = EvaluationQueue(args.state_dir, json.loads(args.profile.read_text()), args.persistent_root)
    watcher = (
        CheckpointDirectoryWatcher(args.watch_dir, queue, interval_s=args.watch_interval_s)
        if args.watch_dir is not None
        else None
    )
    if watcher is not None:
        watcher.start()
    try:
        with create_server(queue, args.host, args.port) as server:
            endpoint_file = args.state_dir / "endpoint.txt"
            endpoint_file.write_text(format_client_endpoint(args.host, server.server_port) + "\n")
            server.serve_forever()
    finally:
        if watcher is not None:
            watcher.stop()


if __name__ == "__main__":
    main()
