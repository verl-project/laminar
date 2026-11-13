# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from laminar.eval import main as validation
from laminar.eval.server import EvaluationQueue, create_server, format_client_endpoint
from laminar.eval.worker import EvaluationTaskSource, evaluation_profile
from test_validation_on_cpu import _config, _resume_fixture


def _queue_config(tmp_path: Path) -> DictConfig:
    config = _config(tmp_path)
    second = tmp_path / "global_step_128" / "actor"
    second.mkdir(parents=True)
    (second / "config.json").write_text("{}")
    config.validation.checkpoints = [config.validation.checkpoints[0], str(second)]
    config.actor_rollout_ref.model = {"path": config.validation.checkpoints[0]}
    task = tmp_path / "task.yaml"
    task.write_text("formal scoring protocol")
    config.actor_rollout_ref.rollout.update(
        nnodes=1,
        n_gpus_per_node=4,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        prompt_length=1024,
        response_length=2048,
        custom={
            "agent_framework": {
                "agent_runners": {
                    "task": {
                        "max_concurrent_sessions": 4,
                        "runner_kwargs": {"task_config_path": str(task)},
                    }
                }
            }
        },
    )
    return config


@contextmanager
def _embedded_task_source(config: DictConfig):
    state_dir = Path(config.validation.result_dir) / "queue"
    queue = EvaluationQueue(
        state_dir,
        evaluation_profile(config),
        [Path(root) for root in config.validation.persistent_roots],
    )
    validation._enqueue_initial_checkpoints(config, queue)
    with create_server(queue, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            worker_cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
            worker_cfg.validation.queue = {
                "endpoint": format_client_endpoint("127.0.0.1", server.server_port),
                "worker_id": "local-worker:test",
                "result_uri": None,
            }
            yield EvaluationTaskSource(worker_cfg), queue
        finally:
            server.shutdown()
            thread.join(timeout=5.0)


def test_batch_resume_manifests_stay_with_their_checkpoint_after_queue_claim(tmp_path):
    config = _queue_config(tmp_path)
    first, second = config.validation.checkpoints
    samples, _, _, _ = _resume_fixture(config, tmp_path, checkpoint=first, indices=(0,), version=17)
    first_entry = OmegaConf.to_container(config.validation.resume_results[0])
    _resume_fixture(config, tmp_path, checkpoint=second, indices=(2,), version=18)
    second_entry = OmegaConf.to_container(config.validation.resume_results[0])
    config.validation.resume_results = [first_entry, second_entry]

    with _embedded_task_source(config) as (source, _queue):
        for checkpoint, expected_entry, expected_index in ((first, first_entry, 0), (second, second_entry, 2)):
            version, claimed = source.claim([f"replica-{expected_index}"])
            assert claimed == checkpoint
            restored_config = source.resume_config(config, version)
            assert OmegaConf.to_container(restored_config.validation.resume_results) == [expected_entry]
            restored = validation._read_resume_results(restored_config, samples)
            assert set(restored) == {checkpoint}
            assert [record["dataset_index"] for record in restored[checkpoint]] == [expected_index]
        assert source.claim(["unused"]) is None


def test_batch_resume_preserves_evidence_through_canonical_checkpoint_identity(tmp_path):
    config = _queue_config(tmp_path)
    canonical = config.validation.checkpoints[0]
    alias = tmp_path / "checkpoint-alias"
    alias.symlink_to(canonical, target_is_directory=True)
    config.validation.checkpoints = [str(alias)]
    samples, artifact, manifest, _ = _resume_fixture(config, tmp_path, indices=(0,), checkpoint=str(alias))
    original_bytes = artifact.read_bytes(), manifest.read_bytes()

    with _embedded_task_source(config) as (source, _queue):
        prepared = source.prepare(config)
        assert prepared is not None
        version, checkpoint = source.initial[0]
        assert checkpoint == str(Path(canonical).resolve())
        restored = validation._read_resume_results(source.resume_config(config, version), samples)
        assert list(restored) == [str(Path(canonical).resolve())]
        assert restored[str(Path(canonical).resolve())][0]["source_result"]["path"] == str(artifact)
        assert (artifact.read_bytes(), manifest.read_bytes()) == original_bytes


def test_non_empty_result_dir_is_rejected_before_mutating_queue_state(tmp_path):
    config = _queue_config(tmp_path)
    with _embedded_task_source(config) as (source, _queue):
        assert source.prepare(config) is not None
    (Path(config.validation.result_dir) / "version_000001.json").write_text("{}")
    state_path = Path(config.validation.result_dir) / "queue" / "queue.json"
    previous = state_path.read_bytes()
    with pytest.raises(FileExistsError, match="fresh result directory"):
        validation.main.__wrapped__(config)
    assert state_path.read_bytes() == previous


def test_resident_profile_uses_explicit_bootstrap_without_a_finite_task_list(tmp_path):
    config = _queue_config(tmp_path)
    expected = evaluation_profile(config)
    config.validation.checkpoints = []
    config.validation.queue = {"endpoint": "http://localhost:9000"}
    validation._validate_config(config)
    assert evaluation_profile(config) == expected


def test_custom_runner_profile_does_not_require_a_task_recipe_path(tmp_path):
    config = _queue_config(tmp_path)
    config.actor_rollout_ref.rollout.custom = {
        "agent_framework": {
            "agent_runners": {"qualification": {"runner_fqn": "qualification.run", "runner_kwargs": {"max_tokens": 8}}}
        }
    }
    before = evaluation_profile(config)
    assert before["task_config_sha256"] is None
    config.actor_rollout_ref.rollout.custom.agent_framework.agent_runners.qualification.runner_kwargs.max_tokens = 16
    assert evaluation_profile(config) != before
