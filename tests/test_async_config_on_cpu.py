# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
import torch
from hydra.errors import InstantiationException
from omegaconf import OmegaConf
from tensordict import NonTensorData, TensorDict
from verl.single_controller.base.decorator import MAGIC_ATTR
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import collate_fn
from verl.workers.config import FSDPActorConfig, RolloutConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.engine_workers import ActorRolloutRefWorker

from fake_transfer_queue import install_fake_tq
from laminar.global_request_router import GlobalRequestRouterConfig, RouterClientWaitConfig
from laminar.global_request_router.router import RouteUnavailableError
from laminar.metrics import MetricSource
from laminar.train import dispatcher as rollout_dispatcher_module
from laminar.train import main as main_async_rl_module
from laminar.train import trainer as async_rl_trainer_module
from laminar.train.dispatcher import RolloutDispatcher
from laminar.train.experience_buffer.server import SampledBatch
from laminar.train.main import (
    _create_resource_pool_placement_groups,
    _initialize_runtime,
    _resolve_agent_loop_manager_class,
    _resource_pool_node_ids,
    _validate_training_strategy,
)
from laminar.train.relay import ActorPublishStats, RelayBroadcastStats
from laminar.train.tracker import Tracker, TrackerConfig
from laminar.train.trainer import (
    AsyncRLTrainer,
    AsyncRLTrainerConfig,
    _add_train_throughput_metrics,
    _extract_actor_update_metrics,
    _extract_reward_group_metrics,
    _extract_train_batch_metrics,
    _select_async_actor_worker_cls,
)
from laminar.utils.tensor import TensorMeta
from laminar.workers.fsdp_workers import AsyncActorWorker
from laminar.workers.megatron_workers import AsyncMegatronActorWorker

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _verl_trainer_yaml(name: str) -> Path:
    import verl

    return Path(verl.__file__).resolve().parent / "trainer/config" / name


def _composed_async_rl_config():
    ppo_cfg = OmegaConf.load(_verl_trainer_yaml("_generated_ppo_trainer.yaml"))
    async_rl_cfg = OmegaConf.load(_REPO_ROOT / "src/laminar/train/config/async_rl_base.yaml")
    return OmegaConf.merge(ppo_cfg, async_rl_cfg)


def test_colocate_configuration_rejects_unsupported_continuation_and_memory_shape():
    config = _composed_async_rl_config()
    config.training_strategy = "colocate_async"
    with pytest.raises(ValueError, match="latest_version"):
        _validate_training_strategy(config)
    config.rollout_router.continuation = "latest_version"
    with pytest.raises(ValueError, match="identical"):
        _validate_training_strategy(config)
    config.actor_rollout_ref.rollout.nnodes = config.trainer.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.trainer.n_gpus_per_node
    config.actor_rollout_ref.rollout.free_cache_engine = True
    with pytest.raises(ValueError, match="offload"):
        _validate_training_strategy(config)
    config.actor_rollout_ref.actor.fsdp_config.param_offload = True
    config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = True
    assert _validate_training_strategy(config) is True


def test_async_rl_requires_configured_rollout_manager():
    config = OmegaConf.create({"actor_rollout_ref": {"rollout": {"agent": {}}}})

    with pytest.raises(ValueError, match="agent_loop_manager_class is required"):
        _resolve_agent_loop_manager_class(config)


def test_async_rl_loads_configured_agent_loop_manager(monkeypatch: pytest.MonkeyPatch):
    sentinel = type("ConfiguredAgentLoopManager", (), {})
    calls = []
    monkeypatch.setattr(
        "laminar.train.main.load_class_from_fqn",
        lambda fqn, description: calls.append((fqn, description)) or sentinel,
    )
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "agent": {"agent_loop_manager_class": ("uni_agent.framework.entry.AgentFrameworkRolloutAdapter")}
                }
            }
        }
    )

    assert _resolve_agent_loop_manager_class(config) is sentinel
    assert calls == [
        (
            "uni_agent.framework.entry.AgentFrameworkRolloutAdapter",
            "AgentLoopManager",
        )
    ]


def test_async_rl_config_overrides_ppo_total_epochs_to_unbounded():
    ppo_cfg = OmegaConf.load(_verl_trainer_yaml("ppo_trainer.yaml"))
    async_rl_cfg = OmegaConf.load(_REPO_ROOT / "src/laminar/train/config/async_rl_base.yaml")

    cfg = OmegaConf.merge(ppo_cfg, async_rl_cfg)

    assert cfg.trainer.total_epochs is None


@pytest.mark.parametrize("worker_cls", [AsyncActorWorker, AsyncMegatronActorWorker])
def test_restore_actor_train_mode_only_sets_module_training_flag(worker_cls):
    modules = [torch.nn.Dropout(), torch.nn.LayerNorm(2)]
    for module in modules:
        module.eval()

    class Engine:
        module = modules

        def train_mode(self, **kwargs):
            raise AssertionError("restore must not enter engine.train_mode()")

    worker = object.__new__(worker_cls)
    worker.role = "actor"
    worker.actor = SimpleNamespace(engine=Engine())

    summary = worker_cls.restore_actor_train_mode(worker)

    assert all(module.training for module in modules)
    assert summary == {
        "engine_class": "Engine",
        "module_count": 2,
        "module_training_count_before": 0,
        "module_training_count_after": 2,
        "router_replay_state": None,
    }


def test_resource_pool_placement_groups_create_largest_pools_first():
    calls = []

    class Pool:
        def __init__(self, name, store):
            self.name = name
            self.store = store
            self.calls = 0

        def get_placement_groups(self):
            calls.append(self.name)
            self.calls += 1

    trainer = Pool("trainer", [4])
    rollout = Pool("rollout", [8, 8])
    small = Pool("small", [4])
    manager = SimpleNamespace(
        resource_pool_dict={
            "trainer_pool": trainer,
            "rollout_pool": rollout,
            "small_pool": small,
        }
    )

    _create_resource_pool_placement_groups(cast(Any, manager))

    assert calls == ["rollout", "trainer", "small"]
    assert [trainer.calls, rollout.calls, small.calls] == [1, 1, 1]


def test_resource_pool_node_ids_are_resolved_from_public_placement_groups(monkeypatch):
    placement_groups = [object(), object()]
    pool = SimpleNamespace(get_placement_groups=lambda: placement_groups)
    tables = {
        placement_groups[0]: {"bundles_to_node_id": {0: "node-a", 1: "node-a"}},
        placement_groups[1]: {"bundles_to_node_id": {0: "node-b"}},
    }
    monkeypatch.setattr(main_async_rl_module.ray.util, "placement_group_table", tables.__getitem__)

    assert _resource_pool_node_ids(pool) == ["node-a", "node-b"]


def test_resource_pool_node_ids_fail_closed_on_unassigned_placement_group(monkeypatch):
    placement_group = object()
    pool = SimpleNamespace(get_placement_groups=lambda: [placement_group])
    monkeypatch.setattr(
        main_async_rl_module.ray.util,
        "placement_group_table",
        lambda _pg: {"bundles_to_node_id": {}},
    )

    with pytest.raises(RuntimeError, match="no assigned node IDs"):
        _resource_pool_node_ids(pool)


def test_async_rl_rollout_config_uses_mainline_rollout_config():
    ppo_cfg = OmegaConf.load(_verl_trainer_yaml("_generated_ppo_trainer.yaml"))
    async_rl_cfg = OmegaConf.load(_REPO_ROOT / "src/laminar/train/config/async_rl_base.yaml")
    cfg = OmegaConf.merge(ppo_cfg, async_rl_cfg)
    cfg.actor_rollout_ref.rollout.name = "vllm"

    rollout_cfg = omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout)

    assert isinstance(rollout_cfg, RolloutConfig)
    assert rollout_cfg.disable_log_stats is False
    assert not hasattr(rollout_cfg, "validation")
    assert not hasattr(rollout_cfg, "log_interval")
    assert not hasattr(rollout_cfg, "num_gpus_per_node")


def test_async_rl_rollout_config_owns_pool_resource_fields():
    ppo_cfg = OmegaConf.load(_verl_trainer_yaml("_generated_ppo_trainer.yaml"))
    async_rl_cfg = OmegaConf.load(_REPO_ROOT / "src/laminar/train/config/async_rl_base.yaml")
    cfg = OmegaConf.merge(ppo_cfg, async_rl_cfg)
    cfg.actor_rollout_ref.rollout.name = "vllm"
    cfg.actor_rollout_ref.rollout.n_gpus_per_node = 4
    cfg.actor_rollout_ref.rollout.nnodes = 2
    cfg.actor_rollout_ref.rollout.tensor_model_parallel_size = 2
    cfg.actor_rollout_ref.rollout.data_parallel_size = 2
    cfg.actor_rollout_ref.rollout.pipeline_model_parallel_size = 1

    rollout_cfg = omega_conf_to_dataclass(cfg.actor_rollout_ref.rollout)

    assert isinstance(rollout_cfg, RolloutConfig)
    assert rollout_cfg.n_gpus_per_node == 4
    assert rollout_cfg.nnodes == 2
    assert (
        rollout_cfg.tensor_model_parallel_size
        * rollout_cfg.data_parallel_size
        * rollout_cfg.pipeline_model_parallel_size
        == 4
    )


def test_async_rl_relay_uses_rollout_config_defaults():
    ppo_cfg = OmegaConf.load(_verl_trainer_yaml("_generated_ppo_trainer.yaml"))
    async_rl_cfg = OmegaConf.load(_REPO_ROOT / "src/laminar/train/config/async_rl_base.yaml")
    cfg = OmegaConf.merge(ppo_cfg, async_rl_cfg)
    cfg.actor_rollout_ref.rollout.dtype = "bfloat16"
    swe_train_script = Path("recipes/swe/train.sh").read_text()

    assert cfg.relay.backend == "ucx"
    assert cfg.relay.num_cpus == 1
    assert "dtype" not in cfg.relay
    assert "pin_memory" not in cfg.relay
    assert cfg.actor_rollout_ref.rollout.dtype == "bfloat16"
    assert "RELAY_DTYPE" not in swe_train_script
    assert "RELAY_PIN_MEMORY" not in swe_train_script
    assert "relay.dtype=" not in swe_train_script
    assert "relay.pin_memory=" not in swe_train_script


def test_async_rl_trainer_config_owns_pool_resource_fields():
    ppo_cfg = OmegaConf.load(_verl_trainer_yaml("_generated_ppo_trainer.yaml"))
    async_rl_cfg = OmegaConf.load(_REPO_ROOT / "src/laminar/train/config/async_rl_base.yaml")
    cfg = OmegaConf.merge(ppo_cfg, async_rl_cfg)
    cfg.trainer.n_gpus_per_node = 4
    cfg.trainer.nnodes = 2

    actor_cfg = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)

    assert cfg.trainer.n_gpus_per_node == 4
    assert cfg.trainer.nnodes == 2
    assert cfg.trainer.worker_env == {}
    assert isinstance(actor_cfg, FSDPActorConfig)
    assert not hasattr(actor_cfg, "num_gpus_per_node")
    assert not hasattr(actor_cfg, "worker_env")


def test_async_rl_actor_config_uses_mainline_fsdp_target():
    ppo_cfg = OmegaConf.load(_verl_trainer_yaml("_generated_ppo_trainer.yaml"))
    async_rl_cfg = OmegaConf.load(_REPO_ROOT / "src/laminar/train/config/async_rl_base.yaml")
    cfg = OmegaConf.merge(ppo_cfg, async_rl_cfg)
    cfg.actor_rollout_ref.actor.rollout_n = 2
    cfg.actor_rollout_ref.actor.use_dynamic_bsz = True

    actor_cfg = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)

    assert cfg.actor_rollout_ref.actor._target_ == "verl.workers.config.FSDPActorConfig"
    assert actor_cfg.fsdp_config.strategy == "fsdp2"
    assert not hasattr(actor_cfg.fsdp_config, "gradient_checkpointing_kwargs")
    assert not hasattr(actor_cfg, "num_gpus_per_node")
    assert not hasattr(actor_cfg, "worker_env")


def test_async_trainer_selects_actor_worker_by_strategy():
    assert _select_async_actor_worker_cls("fsdp2") is AsyncActorWorker
    assert _select_async_actor_worker_cls("fsdp") is AsyncActorWorker
    assert _select_async_actor_worker_cls("megatron") is AsyncMegatronActorWorker

    with pytest.raises(ValueError, match="not supported"):
        _select_async_actor_worker_cls("unknown")


def test_async_actor_init_model_is_blocking_before_relay_connect():
    attrs = getattr(AsyncActorWorker.init_model, MAGIC_ATTR)

    assert attrs["blocking"] is True


def test_async_actor_update_actor_uses_actor_mesh_dispatch():
    attrs = getattr(AsyncActorWorker.update_actor, MAGIC_ATTR)
    dispatch_mode = attrs["dispatch_mode"]

    assert attrs["blocking"] is True
    assert isinstance(dispatch_mode, dict)
    assert getattr(dispatch_mode["dispatch_fn"], "args", ()) == ("actor",)


def test_async_actor_init_model_keeps_mainline_actor_loss_config(monkeypatch):
    sentinel_loss_fn = object()

    class FakeActor:
        def __init__(self):
            self.loss_fns = []

        def set_loss_fn(self, loss_fn):  # noqa: ANN001
            self.loss_fns.append(loss_fn)

    def fake_base_init_model(self):  # noqa: ANN001
        self.actor = FakeActor()
        self.loss_fn = sentinel_loss_fn
        self.actor.set_loss_fn(self.loss_fn)

    monkeypatch.setattr(ActorRolloutRefWorker, "init_model", fake_base_init_model)

    worker = object.__new__(AsyncActorWorker)
    worker._is_rollout = False
    worker.config = OmegaConf.create({"actor": {"global_batch_info": None}})

    AsyncActorWorker.init_model(worker)

    assert worker.actor.loss_fns == [sentinel_loss_fn]


def test_actor_update_metrics_extracts_worker_group_outputs():
    td = TensorDict({}, batch_size=[])
    td["metrics"] = NonTensorData(
        {
            "loss": [1.0, 3.0],
            "grad_norm": [3.0, 5.0],
            "actor/pg_clipfrac": [0.25, 0.75],
            "actor/policy_over_anchor/log_ratio_abs_max": [4.0, 9.0],
            "actor/policy_over_anchor/ratio_gt_2_fraction": [0.2, 0.2],
            "actor/policy_over_anchor/ratio_gt_10_fraction": [0.1, 0.1],
            "actor/policy_over_anchor/action_token_observation_count": [100.0, 100.0],
            "mfu": [0.2, 0.6],
        }
    )

    metrics = _extract_actor_update_metrics(td)

    assert metrics["actor/loss"] == 2.0
    assert metrics["actor/grad_norm"] == 4.0
    assert metrics["actor/pg_clipfrac"] == 0.5
    assert metrics["actor/policy_over_anchor/log_ratio_abs_max"] == 9.0
    assert metrics["actor/policy_over_anchor/ratio_gt_2_fraction"] == 0.2
    assert metrics["actor/policy_over_anchor/ratio_gt_10_fraction"] == 0.1
    assert metrics["actor/policy_over_anchor/action_token_observation_count"] == 100.0
    assert metrics["perf/mfu/actor"] == 0.4
    assert "actor/mfu" not in metrics


def test_reward_group_metrics_summarize_uid_group_context():
    batch = SampledBatch(
        keys=["prompt-1_0_0"],
        group_context={
            "prompt-1": [{"reward_score": 1.0}, {"reward_score": 0.0}],
            "prompt-2": [{"reward_score": 1.0}, {"reward_score": 1.0}],
            "prompt-3": [{"reward_score": 0.0}],
        },
    )

    metrics = _extract_reward_group_metrics(batch)

    assert metrics["train_batch/reward_group/count"] == 3.0
    assert metrics["train_batch/reward_group/heterogeneous_count"] == 1.0
    assert metrics["train_batch/reward_group/homogeneous_count"] == 2.0
    assert metrics["train_batch/reward_group/singleton_count"] == 1.0
    assert metrics["train_batch/reward_group/size/mean"] == pytest.approx(5 / 3)
    assert metrics["train_batch/reward_group/size/max"] == 2.0
    assert metrics["train_batch/reward_group/reward_span/max"] == 1.0


def _collated_prompt_batch(item: dict[str, Any]) -> dict[str, Any]:
    return collate_fn([item])


def test_rollout_dispatcher_builds_mainline_raw_prompt_dataloader(monkeypatch):
    class Tokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return [len(self.calls), 99]

    class Dataset:
        def __init__(self):
            self.items = [
                {
                    "raw_prompt": [{"role": "user", "content": "same"}],
                    "data_source": "math_dapo",
                    "extra_info": {"index": "problem-1", "tools_kwargs": {"env": {"timeout": 30}}},
                    "tools_kwargs": {"env": {"timeout": 30}},
                    "agent_name": "uni_agent",
                    "reward_model": {"ground_truth": "1", "style": "rule"},
                },
                {
                    "raw_prompt": [{"role": "user", "content": "same"}],
                    "data_source": "math_dapo",
                    "extra_info": {"index": "problem-1"},
                    "tools_kwargs": {},
                    "agent_name": "uni_agent",
                    "reward_model": {"ground_truth": "1", "style": "rule"},
                },
                {
                    "raw_prompt": [{"role": "user", "content": "other"}],
                    "data_source": "math_dapo",
                    "extra_info": {"index": "problem-2"},
                    "tools_kwargs": {},
                    "agent_name": "single_turn_agent",
                    "reward_model": {"ground_truth": "2", "style": "rule"},
                },
            ]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    create_calls = []

    def fake_create_rl_dataset(files, data_config, tokenizer, processor, is_train, max_samples):  # noqa: ANN001
        create_calls.append((files, data_config, tokenizer, processor, is_train, max_samples))
        return Dataset()

    monkeypatch.setattr(rollout_dispatcher_module, "create_rl_dataset", fake_create_rl_dataset)

    tokenizer = Tokenizer()
    data_config = OmegaConf.create(
        {
            "train_files": "train.parquet",
            "train_max_samples": -1,
            "prompt_key": "prompt",
            "dataloader_num_workers": 0,
            "shuffle": False,
        }
    )
    producer = RolloutDispatcher(
        config=cast(Any, SimpleNamespace(state_batch_size=3)),
        data_config=data_config,
        tokenizer=tokenizer,
        agent_loop_manager=object(),
        relay_client=object(),
        rollout_router=object(),
    )
    batch = next(iter(producer.prompt_dataloader))

    assert len(create_calls) == 1
    assert create_calls[0][0] == "train.parquet"
    assert create_calls[0][4] is True
    assert tokenizer.calls == []
    assert "prompt_token_ids" not in batch
    assert list(batch["raw_prompt"]) == [
        [{"role": "user", "content": "same"}],
        [{"role": "user", "content": "same"}],
        [{"role": "user", "content": "other"}],
    ]
    assert batch["tools_kwargs"][0] == {"env": {"timeout": 30}}
    assert batch["agent_name"][0] == "uni_agent"
    assert batch["extra_info"][2] == {"index": "problem-2"}
    assert batch["reward_model"][0] == {"ground_truth": "1", "style": "rule"}


def _fake_agent_loop_tq_batch(*, include_routed_experts: bool = False, rows: int = 2) -> TensorDict:
    def jagged(*rows: torch.Tensor) -> torch.Tensor:
        return torch.nested.as_nested_tensor(list(rows), layout=torch.jagged)

    def selected(*values: torch.Tensor) -> torch.Tensor:
        return jagged(*values[:rows])

    fields = {
        "prompts": selected(torch.tensor([11, 12]), torch.tensor([13, 14, 15])),
        "responses": selected(torch.tensor([21, 22]), torch.tensor([23])),
        "position_ids": selected(torch.arange(4), torch.arange(4)),
        "response_mask": selected(torch.tensor([1, 1]), torch.tensor([1])),
        "rm_scores": selected(torch.tensor([0.0, 1.0]), torch.tensor([2.0])),
        "rollout_log_probs": selected(torch.tensor([-0.1, -0.2]), torch.tensor([-0.3])),
    }
    if include_routed_experts:
        fields["routed_experts"] = selected(
            torch.arange(16, dtype=torch.uint8).reshape(4, 2, 2),
            torch.arange(16, 32, dtype=torch.uint8).reshape(4, 2, 2),
        )
    return TensorDict(fields, batch_size=[rows])


@pytest.mark.parametrize(("adv_estimator", "expected_advantage"), [("grpo", -1.0), ("rloo", -2.0)])
def test_trainer_computes_session_advantage_from_complete_prompt_context(adv_estimator, expected_advantage):
    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": adv_estimator,
                "norm_adv_by_std_in_grpo": False,
            }
        }
    )
    batch = SampledBatch(
        keys=["prompt-1_0_0", "prompt-1_0_1"],
        group_context={
            "prompt-1": [
                {"key": "prompt-1_0_1", "session_id": 0, "output_index": 1, "reward_score": 1.0},
                {"key": "prompt-1_1_0", "session_id": 1, "output_index": 0, "reward_score": 3.0},
            ]
        },
    )
    td = TensorDict(
        {
            "response_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
            # The selected subset does not contain session 1; group context
            # still supplies its terminal reward for the GRPO baseline.
            "rm_scores": torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
        },
        batch_size=[2],
    )
    metrics: dict[str, Any] = {}

    trainer._compute_advantages(batch, td, metrics)

    torch.testing.assert_close(
        td["advantages"],
        torch.tensor(
            [
                [expected_advantage, expected_advantage, 0.0],
                [expected_advantage, 0.0, 0.0],
            ]
        ),
    )
    torch.testing.assert_close(td["returns"], td["advantages"])
    assert metrics["advantage/source_prompt_group_count"] == 1.0
    assert metrics["advantage/source_session_count"] == 2.0


def test_materialize_sampled_batch_preserves_ragged_prompt_response_boundary(monkeypatch):
    def fake_get_many_sync(*, keys, namespace, fields):  # noqa: ANN001
        assert keys == ["traj-a", "traj-b"]
        assert namespace == "train"
        assert fields == [
            "prompts",
            "responses",
            "position_ids",
            "response_mask",
            "rm_scores",
            "rollout_log_probs",
        ]
        return _fake_agent_loop_tq_batch()

    install_fake_tq(monkeypatch, get_many_sync_handler=fake_get_many_sync)

    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"rollout_correction": {"bypass_mode": False}},
            "actor_rollout_ref": {
                "actor": {"strategy": "fsdp2", "policy_loss": {"loss_mode": "vanilla"}},
                "model": {"use_remove_padding": True},
                "rollout": {"temperature": 0.7},
            },
        }
    )
    trainer.trainer_config = AsyncRLTrainerConfig()
    td = trainer._materialize_sampled_batch(SampledBatch(keys=["traj-a", "traj-b"], partition_id="train"), metrics={})

    assert td["input_ids"].is_nested
    assert td["position_ids"].is_nested
    torch.testing.assert_close(
        td["input_ids"].offsets().diff(),
        torch.tensor([4, 4], dtype=td["input_ids"].offsets().dtype),
    )
    torch.testing.assert_close(td["input_ids"].values(), torch.tensor([11, 12, 21, 22, 13, 14, 15, 23]))
    torch.testing.assert_close(td["position_ids"].values(), torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]))
    padded_response_log_probs = async_rl_trainer_module.no_padding_2_padding(
        torch.tensor([100.0, 101.0, 102.0, 103.0, 200.0, 201.0, 202.0, 203.0]),
        td,
    )
    torch.testing.assert_close(
        padded_response_log_probs,
        torch.tensor([[101.0, 102.0], [202.0, 0.0]]),
    )
    assert "old_log_probs" not in td.keys()
    torch.testing.assert_close(td["loss_mask"], td["response_mask"])
    torch.testing.assert_close(
        td["rm_scores"],
        torch.tensor([[0.0, 1.0], [2.0, 0.0]]),
    )


def test_materialize_sampled_batch_builds_loss_mask_without_remove_padding(monkeypatch):
    install_fake_tq(monkeypatch, get_many_sync_handler=lambda **_: _fake_agent_loop_tq_batch())

    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"use_remove_padding": False},
                "rollout": {"temperature": 0.7},
            }
        }
    )
    td = trainer._materialize_sampled_batch(SampledBatch(keys=["traj-a", "traj-b"], partition_id="train"), metrics={})

    assert not td["input_ids"].is_nested
    torch.testing.assert_close(td["loss_mask"], td["response_mask"])


def _materialize_fake_batch(monkeypatch, batch: TensorDict, *, use_remove_padding: bool) -> TensorDict:
    install_fake_tq(monkeypatch, get_many_sync_handler=lambda **_: batch)
    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {"use_remove_padding": use_remove_padding},
                "rollout": {"temperature": 0.7},
            }
        }
    )
    return trainer._materialize_sampled_batch(SampledBatch(keys=["traj-a", "traj-b"], partition_id="train"), metrics={})


@pytest.mark.parametrize("use_remove_padding", [False, True])
def test_materialize_sampled_batch_keeps_producer_rope_position_rows(monkeypatch, use_remove_padding):
    rope_a = torch.arange(4).expand(4, 4) + torch.arange(4).unsqueeze(1) * 10
    rope_b = rope_a + 100
    batch = _fake_agent_loop_tq_batch()
    batch["position_ids"] = torch.nested.as_nested_tensor([rope_a, rope_b], layout=torch.jagged)

    td = _materialize_fake_batch(monkeypatch, batch, use_remove_padding=use_remove_padding)

    if use_remove_padding:
        rows = td["position_ids"].unbind()
        torch.testing.assert_close(rows[0], rope_a)
        torch.testing.assert_close(rows[1], rope_b)
    else:
        expected = torch.zeros(2, 4, 5, dtype=torch.long)
        expected[0, :, 1:] = rope_a
        expected[1, :, :4] = rope_b
        torch.testing.assert_close(td["position_ids"], expected)


@pytest.mark.parametrize("second_response_length", [1, 2])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_materialize_multimodal_positions_pack_for_fsdp(monkeypatch, second_response_length, position_dtype):
    batch = _fake_agent_loop_tq_batch()
    for field in ("responses", "response_mask", "rm_scores", "rollout_log_probs"):
        first, second = batch[field].unbind()
        second = second.repeat(second_response_length)
        batch[field] = torch.nested.as_nested_tensor([first, second], layout=torch.jagged)
    lengths = [4, 3 + second_response_length]
    position_rows = [
        torch.arange(length, dtype=position_dtype).expand(4, length)
        + torch.arange(4, dtype=position_dtype).unsqueeze(1) * 10
        for length in lengths
    ]
    expected_values = torch.cat(position_rows, dim=-1)
    batch["position_ids"] = torch.nested.nested_tensor_from_jagged(
        values=expected_values,
        offsets=torch.tensor([0, lengths[0], sum(lengths)]),
        jagged_dim=2,
    )

    td = _materialize_fake_batch(monkeypatch, batch, use_remove_padding=True)

    torch.testing.assert_close(td["position_ids"].values(), expected_values)
    torch.testing.assert_close(td["position_ids"].offsets(), td["input_ids"].offsets())
    engine = SimpleNamespace(pad_to_length=0, use_ulysses_sp=False, _get_packed_pad_size=lambda _: 0)
    model_inputs, _ = FSDPEngineWithLMHead.prepare_model_inputs(engine, td)
    assert model_inputs["input_ids"].shape == (1, sum(lengths))
    torch.testing.assert_close(model_inputs["position_ids"], expected_values.unsqueeze(1))


@pytest.mark.parametrize("use_remove_padding", [False, True])
def test_materialize_sampled_batch_keeps_nonstandard_1d_position_ids(monkeypatch, use_remove_padding):
    positions_a = torch.tensor([101, 103, 107, 109])
    positions_b = torch.tensor([211, 223, 227, 229])
    batch = _fake_agent_loop_tq_batch()
    batch["position_ids"] = torch.nested.as_nested_tensor([positions_a, positions_b], layout=torch.jagged)

    td = _materialize_fake_batch(monkeypatch, batch, use_remove_padding=use_remove_padding)

    if use_remove_padding:
        rows = td["position_ids"].unbind()
        torch.testing.assert_close(rows[0], positions_a)
        torch.testing.assert_close(rows[1], positions_b)
    else:
        torch.testing.assert_close(
            td["position_ids"],
            torch.tensor(
                [
                    [0, 101, 103, 107, 109],
                    [211, 223, 227, 229, 0],
                ]
            ),
        )


def test_materialize_sampled_batch_rejects_misaligned_position_ids(monkeypatch):
    batch = _fake_agent_loop_tq_batch()
    batch["position_ids"] = torch.nested.as_nested_tensor([torch.arange(4), torch.arange(3)], layout=torch.jagged)

    with pytest.raises(RuntimeError, match="position_ids lengths do not match prompt\\+response lengths"):
        _materialize_fake_batch(monkeypatch, batch, use_remove_padding=True)


def test_materialize_sampled_batch_requires_position_ids(monkeypatch):
    batch = _fake_agent_loop_tq_batch()
    batch.pop("position_ids")

    with pytest.raises(RuntimeError, match="missing required sequence-aligned field 'position_ids'"):
        _materialize_fake_batch(monkeypatch, batch, use_remove_padding=False)


def test_materialize_sampled_batch_rejects_misaligned_rollout_logprobs(monkeypatch):
    batch = _fake_agent_loop_tq_batch()
    batch["rollout_log_probs"] = torch.nested.as_nested_tensor(
        [torch.tensor([-0.1]), torch.tensor([-0.3])], layout=torch.jagged
    )
    install_fake_tq(monkeypatch, get_many_sync_handler=lambda **_: batch)

    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"rollout_correction": {"bypass_mode": False}},
            "actor_rollout_ref": {
                "actor": {"strategy": "fsdp2", "policy_loss": {"loss_mode": "vanilla"}},
                "model": {"use_remove_padding": True},
                "rollout": {"temperature": 0.7},
            },
        }
    )
    trainer.trainer_config = AsyncRLTrainerConfig()
    with pytest.raises(RuntimeError, match="rollout_log_probs.*do not match response lengths"):
        trainer._materialize_sampled_batch(SampledBatch(keys=["traj-a", "traj-b"], partition_id="train"), metrics={})


def test_materialize_sampled_batch_requires_rollout_logprobs_at_materialization(monkeypatch):
    batch = _fake_agent_loop_tq_batch()
    batch.pop("rollout_log_probs")
    install_fake_tq(monkeypatch, get_many_sync_handler=lambda **_: batch)

    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {"strategy": "fsdp2"},
                "model": {"use_remove_padding": False},
                "rollout": {"temperature": 0.7},
            }
        }
    )

    with pytest.raises(RuntimeError, match="missing required response-aligned field 'rollout_log_probs'"):
        trainer._materialize_sampled_batch(SampledBatch(keys=["traj-a", "traj-b"], partition_id="train"), metrics={})


def test_materialize_sampled_batch_fetches_routed_experts_for_r3(monkeypatch):
    def fake_get_many_sync(*, keys, namespace, fields):  # noqa: ANN001
        assert keys == ["traj-a", "traj-b"]
        assert namespace == "train"
        assert fields == [
            "prompts",
            "responses",
            "position_ids",
            "response_mask",
            "rm_scores",
            "rollout_log_probs",
            "routed_experts",
        ]
        return _fake_agent_loop_tq_batch(include_routed_experts=True)

    install_fake_tq(monkeypatch, get_many_sync_handler=fake_get_many_sync)

    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "strategy": "megatron",
                    "megatron": {"router_replay": {"mode": "R3"}},
                },
                "model": {"use_remove_padding": True},
                "rollout": {"enable_rollout_routing_replay": True, "temperature": 0.7},
            }
        }
    )
    metrics: dict[str, Any] = {}
    td = trainer._materialize_sampled_batch(
        SampledBatch(keys=["traj-a", "traj-b"], partition_id="train"),
        metrics=metrics,
    )

    assert td["routed_experts"].is_nested
    assert td["routed_experts"].values().dtype == torch.uint8
    torch.testing.assert_close(
        td["routed_experts"].offsets().diff(),
        torch.tensor([4, 4], dtype=td["routed_experts"].offsets().dtype),
    )
    routed_values = _fake_agent_loop_tq_batch(include_routed_experts=True)["routed_experts"].values()
    expected_routed_bytes = routed_values.numel() * routed_values.element_size()
    assert metrics["materialize_batch/routed_experts_bytes"] == expected_routed_bytes
    assert metrics["materialize_batch/routed_experts_fraction"] == pytest.approx(
        expected_routed_bytes / metrics["materialize_batch/store_payload_bytes"]
    )
    torch.testing.assert_close(td["loss_mask"], td["response_mask"])


def test_materialize_sampled_batch_ignores_disabled_engine_router_replay(monkeypatch):
    def fake_get_many_sync(*, fields, **_):  # noqa: ANN001
        assert "routed_experts" not in fields
        return _fake_agent_loop_tq_batch()

    install_fake_tq(monkeypatch, get_many_sync_handler=fake_get_many_sync)
    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "strategy": "megatron",
                    "megatron": {"router_replay": {"mode": "disabled"}},
                },
                "model": {"use_remove_padding": True},
                "rollout": {"enable_rollout_routing_replay": True, "temperature": 0.7},
            }
        }
    )

    td = trainer._materialize_sampled_batch(SampledBatch(keys=["traj-a", "traj-b"], partition_id="train"), metrics={})

    assert "routed_experts" not in td.keys()


def test_materialize_sampled_batch_requires_routed_experts_for_r3(monkeypatch):
    def fake_get_many_sync(*, fields, **_):  # noqa: ANN001
        assert "routed_experts" in fields
        return _fake_agent_loop_tq_batch(rows=1)

    install_fake_tq(monkeypatch, get_many_sync_handler=fake_get_many_sync)
    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {
                    "strategy": "megatron",
                    "megatron": {"router_replay": {"mode": "R3"}},
                },
                "model": {"use_remove_padding": False},
                "rollout": {"enable_rollout_routing_replay": True, "temperature": 0.7},
            }
        }
    )

    with pytest.raises(RuntimeError, match="missing routed_experts"):
        trainer._materialize_sampled_batch(SampledBatch(keys=["traj-a"], partition_id="train"), metrics={})


@pytest.mark.parametrize(
    ("mode", "rollout_enabled", "match"),
    [
        ("R2", False, "disabled or R3, not R2"),
        ("R3", False, "R3 requires rollout.enable_rollout_routing_replay=True"),
    ],
)
def test_init_actor_workers_rejects_unsupported_router_replay(mode, rollout_enabled, match):
    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "actor": {"strategy": "megatron", "megatron": {"router_replay": {"mode": mode}}},
                "rollout": {"enable_rollout_routing_replay": rollout_enabled},
            }
        }
    )

    with pytest.raises(ValueError, match=match):
        trainer._init_actor_workers()


@pytest.mark.parametrize(
    ("calculate_entropy", "entropy_coeff", "expect_entropy"),
    [
        (False, 0.0, False),
        (True, 0.0, True),
        (False, 0.01, True),
    ],
)
def test_compute_old_log_probs_uses_actor_worker_no_padding(calculate_entropy, entropy_coeff, expect_entropy):
    class ActorWG:
        def compute_log_prob(self, td):  # noqa: ANN001
            assert td["input_ids"].is_nested
            assert async_rl_trainer_module.tu.get(td, "compute_loss") is False
            assert async_rl_trainer_module.tu.get(td, "calculate_entropy") is expect_entropy
            output = {
                "log_probs": torch.arange(6, dtype=torch.float32),
                "metrics": NonTensorData(
                    {
                        "mfu": 0.5,
                        "perf/dynamic_micro_batches": [9.0, 11.0],
                        "perf/dynamic_micro_batch_valid_tokens_per_gpu": [60.0, 80.0],
                        "perf/dynamic_micro_batch_valid_tokens_per_gpu/max": [60.0, 80.0],
                        "perf/dynamic_micro_batch_valid_tokens_per_gpu/min": [60.0, 80.0],
                    }
                ),
            }
            if expect_entropy:
                output["entropy"] = torch.arange(10, 16, dtype=torch.float32)
            return TensorDict(output, batch_size=[])

    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"rollout_correction": {"bypass_mode": False}},
            "actor_rollout_ref": {
                "actor": {
                    "calculate_entropy": calculate_entropy,
                    "entropy_coeff": entropy_coeff,
                    "calculate_sum_pi_squared": False,
                    "policy_loss": {"loss_mode": "vanilla"},
                }
            },
        }
    )
    trainer.actor_wg = ActorWG()
    td = TensorDict(
        {
            "prompts": torch.tensor([[11, 12], [0, 13]]),
            "responses": torch.tensor([[21, 22, 0], [23, 0, 0]]),
            "input_ids": torch.tensor([[11, 12, 21, 22, 0], [0, 13, 23, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [0, 1, 1, 0, 0]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 0], [0, 0, 1, 0, 0]]),
            "response_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
        },
        batch_size=[2],
    )
    metrics: dict[str, Any] = {}

    trainer._compute_old_log_probs(td, metrics)

    torch.testing.assert_close(
        td["old_log_probs"],
        torch.tensor([[1.0, 2.0, 0.0], [4.0, 0.0, 0.0]]),
    )
    if expect_entropy:
        torch.testing.assert_close(
            td["entropy"],
            torch.tensor([[11.0, 12.0, 0.0], [14.0, 0.0, 0.0]]),
        )
    else:
        assert "entropy" not in td.keys()
    assert metrics["old_log_prob/recomputed"] == 1.0
    assert metrics["old_log_prob/calculate_entropy"] == float(expect_entropy)
    assert metrics["perf/mfu/old_log_prob"] == 0.5
    assert metrics["old_log_prob/perf/dynamic_micro_batches"] == 10.0
    assert metrics["old_log_prob/perf/dynamic_micro_batch_valid_tokens_per_gpu"] == 70.0
    assert metrics["old_log_prob/perf/dynamic_micro_batch_valid_tokens_per_gpu/max"] == 80.0
    assert metrics["old_log_prob/perf/dynamic_micro_batch_valid_tokens_per_gpu/min"] == 60.0


def test_async_trainer_applies_decoupled_rollout_correction():
    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "rollout_correction": {
                    "bypass_mode": False,
                    "rollout_is": "token",
                    "rollout_is_threshold": 2.0,
                    "rollout_is_batch_normalize": False,
                    "rollout_rs": None,
                    "rollout_rs_threshold": None,
                }
            },
            "actor_rollout_ref": {"actor": {"policy_loss": {"loss_mode": "dppo_tv"}}},
        }
    )
    td = TensorDict(
        {
            "old_log_probs": torch.tensor([[-0.5, -1.0]]),
            "rollout_log_probs": torch.tensor([[-1.0, -1.0]]),
            "response_mask": torch.ones((1, 2), dtype=torch.int64),
        },
        batch_size=[1],
    )
    metrics: dict[str, Any] = {}

    trainer._apply_rollout_correction(td, metrics)

    torch.testing.assert_close(td["rollout_is_weights"], torch.exp(torch.tensor([[0.5, 0.0]])))
    torch.testing.assert_close(td["response_mask"], torch.ones((1, 2), dtype=torch.int64))
    assert "rollout_corr/rollout_is_mean" in metrics


def test_async_trainer_rollout_correction_synchronizes_loss_mask(monkeypatch):
    corrected_mask = torch.tensor([[1, 0]], dtype=torch.int64)
    monkeypatch.setattr(
        async_rl_trainer_module,
        "compute_rollout_correction_and_rejection_mask",
        lambda **_: (None, corrected_mask, {}),
    )
    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "rollout_correction": {
                    "rollout_is": None,
                    "rollout_is_threshold": None,
                    "rollout_is_batch_normalize": False,
                    "rollout_rs": "token",
                    "rollout_rs_threshold": 2.0,
                }
            },
            "actor_rollout_ref": {"actor": {"policy_loss": {"loss_mode": "vanilla"}}},
        }
    )
    original_mask = torch.ones((1, 2), dtype=torch.int64)
    td = TensorDict(
        {
            "old_log_probs": torch.tensor([[-0.5, -1.0]]),
            "rollout_log_probs": torch.tensor([[-1.0, -1.0]]),
            "response_mask": original_mask,
            "loss_mask": original_mask,
        },
        batch_size=[1],
    )

    trainer._apply_rollout_correction(td, {})

    torch.testing.assert_close(td["response_mask"], corrected_mask)
    torch.testing.assert_close(td["loss_mask"], corrected_mask)


def test_async_trainer_rejects_correction_without_rollout_log_probs():
    config = OmegaConf.create(
        {
            "algorithm": {
                "rollout_correction": {
                    "bypass_mode": False,
                    "rollout_is": "token",
                    "rollout_rs": None,
                }
            },
            "actor_rollout_ref": {
                "actor": {"policy_loss": {"loss_mode": "vanilla"}},
                "rollout": {"calculate_log_probs": False},
            },
        }
    )

    with pytest.raises(ValueError, match="calculate_log_probs=True"):
        AsyncRLTrainer(
            config,
            resource_pool_manager=None,
            experience_buffer=None,
            relay_client=None,
            tracker=None,
            trainer_config=AsyncRLTrainerConfig(),
        )


def test_async_trainer_rejects_unsupported_ref_policy_kl():
    for path in ("algorithm.use_kl_in_reward", "actor_rollout_ref.actor.use_kl_loss"):
        config = _composed_async_rl_config()
        OmegaConf.update(config, path, True)

        with pytest.raises(ValueError, match="does not support ref-policy KL"):
            AsyncRLTrainer(
                config,
                resource_pool_manager=None,
                experience_buffer=None,
                relay_client=None,
                tracker=None,
                trainer_config=AsyncRLTrainerConfig(),
            )


def test_train_batch_metrics_summarize_rewards_advantages():
    td = TensorDict(
        {
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.int64),
            "response_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.int64),
            "rm_scores": torch.tensor([[0.0, 2.0, 0.0], [0.0, -1.0, 0.0]]),
            "advantages": torch.tensor([[1.0, -1.0, 0.0], [0.0, 2.0, -2.0]]),
        },
        batch_size=[2],
    )

    metrics = _extract_train_batch_metrics(td)

    assert metrics["train_batch/num_trajectories"] == 2
    assert metrics["train_batch/num_attention_tokens"] == 9
    assert metrics["train_batch/num_action_tokens"] == 5
    assert type(metrics["train_batch/num_trajectories"]) is int
    assert type(metrics["train_batch/num_attention_tokens"]) is int
    assert type(metrics["train_batch/num_action_tokens"]) is int
    assert metrics["train_batch/response_length/mean"] == pytest.approx(3.0)
    assert metrics["train_batch/action_length/mean"] == pytest.approx(2.5)
    assert "train_batch/response_length/clip_ratio" not in metrics
    assert metrics["train_batch/response_length/clip_ratio_valid"] == 0.0
    assert metrics["train_batch/rewards/mean"] == pytest.approx(0.5)
    assert metrics["train_batch/advantages/mean"] == pytest.approx(0.0)


def test_update_actor_rejects_zero_action_tokens(monkeypatch):
    store_batch = _fake_agent_loop_tq_batch()
    store_batch["response_mask"] = torch.nested.as_nested_tensor(
        [torch.zeros(2, dtype=torch.int64), torch.zeros(1, dtype=torch.int64)],
        layout=torch.jagged,
    )
    install_fake_tq(monkeypatch, get_many_sync_handler=lambda **_: store_batch)

    class ActorWG:
        def update_actor(self, td):  # noqa: ANN001
            raise AssertionError("actor update must not run for an all-masked batch")

    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = OmegaConf.merge(
        _composed_async_rl_config(),
        {
            "actor_rollout_ref": {
                "model": {"use_remove_padding": False},
                "rollout": {"temperature": 1.0},
                "actor": {"ppo_epochs": 1, "ppo_mini_batch_size": 2},
            }
        },
    )
    trainer.global_steps = 1
    trainer.actor_wg = ActorWG()
    trainer.actor_mini_batch_size = lambda: 2
    trainer.trainer_config = AsyncRLTrainerConfig(balance_batch=False)
    metrics: dict[str, Any] = {}
    batch = SampledBatch(
        keys=["uid-a_0_0", "uid-a_1_0"],
        partition_id="train",
        group_context={
            "uid-a": [
                {"key": "uid-a_0_0", "session_id": 0, "output_index": 0, "reward_score": 1.0},
                {"key": "uid-a_1_0", "session_id": 1, "output_index": 0, "reward_score": 2.0},
            ]
        },
    )
    with pytest.raises(RuntimeError, match="zero action tokens"):
        trainer._materialize_actor_batch(batch, metrics)

    assert metrics["train_batch/num_action_tokens"] == 0.0


def test_update_actor_records_four_by_32_trajectory_contract():
    class ActorWG:
        def __init__(self):
            self.received = None

        def update_actor(self, td):  # noqa: ANN001
            self.received = td
            return None

    actor_wg = ActorWG()
    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 16},
            "actor_rollout_ref": {
                "rollout": {"n": 8},
                "actor": {
                    "ppo_epochs": 1,
                    "ppo_mini_batch_size": 4,
                    "calculate_entropy": True,
                    "entropy_coeff": 0.0,
                    "data_loader_seed": 42,
                    "shuffle": False,
                },
            },
        }
    )
    trainer.actor_wg = actor_wg
    trainer.global_steps = 1
    td = TensorDict({"sample_id": torch.arange(128)}, batch_size=[128])
    metrics = {}

    trainer._update_actor(td, metrics)

    assert actor_wg.received is td
    assert int(td["mini_batch_size"]) == 32
    assert int(td["global_batch_size"]) == 32
    assert int(td["epochs"]) == 1
    assert bool(td["calculate_entropy"]) is True
    assert int(td["seed"]) == 42
    assert bool(td["compute_loss"]) is True
    assert tu.get(td, "dataloader_kwargs") == {"shuffle": False}
    assert metrics["actor/update/trajectory_batch_size"] == 128.0
    assert metrics["actor/update/trajectory_mini_batch_size"] == 32.0
    assert metrics["actor/update/mini_batches_per_epoch"] == 4.0
    assert metrics["actor/update/optimizer_updates_expected"] == 4.0
    assert metrics["actor/update/optimizer_updates_valid"] == 0.0
    assert "actor/update/optimizer_updates" not in metrics


def test_actor_num_mini_batches_rejects_non_divisible_trajectory_batch():
    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 15},
            "actor_rollout_ref": {
                "rollout": {"n": 8},
                "actor": {"ppo_mini_batch_size": 4},
            },
        }
    )

    with pytest.raises(ValueError, match="must be divisible"):
        trainer.actor_num_mini_batches()


@pytest.mark.parametrize("use_nested", [False, True])
def test_balance_actor_batch_reorders_equal_dp_shards_by_workload(use_nested):
    sequence_lengths = [9, 8, 1, 1]
    max_length = max(sequence_lengths)
    if use_nested:
        input_ids = torch.nested.as_nested_tensor(
            [torch.arange(length) for length in sequence_lengths], layout=torch.jagged
        )
        attention_mask = torch.nested.as_nested_tensor(
            [torch.ones(length, dtype=torch.int64) for length in sequence_lengths], layout=torch.jagged
        )
    else:
        input_ids = torch.zeros((len(sequence_lengths), max_length), dtype=torch.int64)
        attention_mask = torch.zeros_like(input_ids)
        for index, length in enumerate(sequence_lengths):
            attention_mask[index, :length] = 1

    td = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sample_id": torch.arange(len(sequence_lengths)),
            "global_token_num": NonTensorData(sequence_lengths),
        },
        batch_size=[len(sequence_lengths)],
    )
    trainer = object.__new__(AsyncRLTrainer)
    trainer.actor_wg = SimpleNamespace(_dispatch_info={"actor": [0, 1]})

    metrics: dict[str, Any] = {}
    balanced = trainer._balance_actor_batch(td, metrics)

    sample_ids = balanced["sample_id"].tolist()
    assert {sample_ids[0], sample_ids[1]} == {0, 2}
    assert {sample_ids[2], sample_ids[3]} == {1, 3}
    assert balanced["global_token_num"] == [sequence_lengths[index] for index in sample_ids]
    assert metrics["global_seqlen/min"] == 2
    assert metrics["global_seqlen/max"] == 17
    assert metrics["global_seqlen/balanced_min"] == 9
    assert metrics["global_seqlen/balanced_max"] == 10
    assert metrics["global_workload/balanced_max"] < metrics["global_workload/max"]
    assert metrics["trainer/dp_balance/dp_size"] == 2.0


def test_train_throughput_metrics_use_attention_tokens():
    metrics: dict[str, Any] = {
        "train_batch/num_attention_tokens": 960,
        "train_batch/num_action_tokens": 720,
        "train_batch/num_trajectories": 48,
    }
    timing_raw = {"step": 12.0, "update_actor": 4.0}

    _add_train_throughput_metrics(metrics, timing_raw, trainer_gpus=8, total_gpus=16)

    assert metrics["train/throughput/attention_token/step"] == pytest.approx(80.0)
    assert metrics["train/throughput/attention_token/update_actor"] == pytest.approx(240.0)
    assert metrics["train/throughput/attention_token_per_train_gpu/step"] == pytest.approx(10.0)
    assert metrics["train/throughput/attention_token_per_train_gpu/update_actor"] == pytest.approx(30.0)
    assert metrics["resource/gpu_count/trainer"] == 8.0
    assert metrics["resource/gpu_count/rollout"] == 8.0
    assert metrics["resource/gpu_count/total"] == 16.0


def _rollout_dispatcher_config(
    *,
    multi_version_rollout_overrides: dict[str, Any] | None = None,
    **rollout_dispatcher_overrides: Any,
):
    rollout_dispatcher = {
        "_target_": "laminar.train.dispatcher.RolloutDispatcherConfig",
        "schedule_interval_s": 2.5,
        "state_batch_size": 2,
        "num_samples_per_prompt": 3,
    }
    rollout_dispatcher.update(rollout_dispatcher_overrides)
    multi_version_rollout = {
        "_target_": "laminar.multi_version_rollout.MultiVersionRolloutManagerConfig",
        "schedule_interval_s": 2.5,
        "repack": {
            "_target_": "laminar.multi_version_rollout.MultiVersionRolloutRepackConfig",
            "enable": True,
            "source_max_kv_cache_usage": 0.5,
            "destination_max_kv_cache_usage": 0.95,
        },
    }
    if multi_version_rollout_overrides:
        for key, value in multi_version_rollout_overrides.items():
            if isinstance(value, dict) and isinstance(multi_version_rollout.get(key), dict):
                multi_version_rollout[key].update(value)
            else:
                multi_version_rollout[key] = value
    return OmegaConf.create(
        {
            "data": {"train_batch_size": 4, "gen_batch_size": 2},
            "actor_rollout_ref": {"rollout": {"n": 3, "max_num_seqs": 17}},
            "rollout_dispatcher": rollout_dispatcher,
            "multi_version_rollout": multi_version_rollout,
        }
    )


def test_rollout_dispatcher_config_from_explicit_fields():
    config = _rollout_dispatcher_config()
    cfg = omega_conf_to_dataclass(config.rollout_dispatcher)

    assert cfg.schedule_interval_s == 2.5
    assert cfg.state_batch_size == 2
    assert cfg.num_samples_per_prompt == 3
    assert cfg.pressure_admission_mode == "off"
    assert cfg.kv_cache_usage_threshold == 0.80


def test_rollout_dispatcher_config_from_existing_hydra_tree():
    ppo_cfg = OmegaConf.load(_verl_trainer_yaml("_generated_ppo_trainer.yaml"))
    async_rl_cfg = OmegaConf.load(_REPO_ROOT / "src/laminar/train/config/async_rl_base.yaml")
    config = OmegaConf.merge(ppo_cfg, async_rl_cfg)
    cfg = omega_conf_to_dataclass(config.rollout_dispatcher)

    assert cfg.state_batch_size == config.data.gen_batch_size
    assert cfg.num_samples_per_prompt == config.actor_rollout_ref.rollout.n
    assert cfg.pressure_admission_mode == config.rollout_router.dynamic_overload_admission.mode
    assert cfg.kv_cache_usage_threshold == config.rollout_router.dynamic_overload_admission.kv_cache_usage_threshold
    assert cfg.pressure_admission_mode == "off"
    assert cfg.kv_cache_usage_threshold == 0.80


def test_rollout_dispatcher_pressure_follows_router_overload_override():
    config = _composed_async_rl_config()
    OmegaConf.update(config, "rollout_router.dynamic_overload_admission.mode", "enforce")
    OmegaConf.update(config, "rollout_router.dynamic_overload_admission.kv_cache_usage_threshold", 0.65)
    cfg = omega_conf_to_dataclass(config.rollout_dispatcher)

    assert cfg.pressure_admission_mode == "enforce"
    assert cfg.kv_cache_usage_threshold == 0.65


def test_rollout_dispatcher_config_requires_explicit_fields():
    config = _rollout_dispatcher_config()
    del config.rollout_dispatcher.schedule_interval_s

    with pytest.raises(InstantiationException, match="schedule_interval_s"):
        omega_conf_to_dataclass(config.rollout_dispatcher)


def test_rollout_router_config_from_existing_hydra_tree():
    config = _composed_async_rl_config()
    cfg = omega_conf_to_dataclass(config.rollout_router)

    assert isinstance(cfg, GlobalRequestRouterConfig)
    assert cfg.client_wait == RouterClientWaitConfig()
    assert cfg.dynamic_overload_admission["mode"] == "off"
    assert cfg.dynamic_overload_admission["kv_cache_usage_threshold"] == 0.80


def test_rollout_router_config_from_explicit_override():
    config = OmegaConf.create(
        {
            "rollout_router": {
                "_target_": "laminar.global_request_router.GlobalRequestRouterConfig",
                "dynamic_overload_admission": {
                    "mode": "enforce",
                    "kv_cache_usage_threshold": 0.9,
                },
                "client_wait": {
                    "timeout_s": 10,
                    "planned_pause_timeout_s": 20,
                    "initial_backoff_s": 0.2,
                    "max_backoff_s": 1,
                },
            }
        }
    )
    cfg = omega_conf_to_dataclass(config.rollout_router)

    assert cfg.client_wait == RouterClientWaitConfig(
        timeout_s=10,
        planned_pause_timeout_s=20,
        initial_backoff_s=0.2,
        max_backoff_s=1,
    )
    assert cfg.dynamic_overload_admission == {
        "mode": "enforce",
        "kv_cache_usage_threshold": 0.9,
    }


def test_rollout_router_config_defaults_match_router_constructor():
    cfg = GlobalRequestRouterConfig()

    assert cfg.client_wait == RouterClientWaitConfig()
    assert cfg.dynamic_overload_admission == {}


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"timeout_s": 0}, "timeout_s"),
        ({"planned_pause_timeout_s": 10, "timeout_s": 20}, "planned_pause_timeout_s"),
        ({"initial_backoff_s": 2, "max_backoff_s": 1}, "max_backoff_s"),
    ],
)
def test_rollout_router_client_wait_rejects_invalid_values(override, message):
    with pytest.raises(ValueError, match=message):
        GlobalRequestRouterConfig(client_wait=override)


class _VersionRelay:
    def __init__(self, versions: list[int]):
        self.versions = list(versions)
        self.get_trainer_version_calls: list[dict[str, Any]] = []

    async def get_trainer_version(self) -> int:
        self.get_trainer_version_calls.append({"args": (), "kwargs": {}})
        assert self.versions
        if len(self.versions) > 1:
            return self.versions.pop(0)
        return self.versions[0]


class _DispatchManager:
    def __init__(
        self,
        *,
        fail: bool = False,
        events: list[str] | None = None,
    ):
        self.fail = fail
        self.events = [] if events is None else events
        self.batches: list[Any] = []

    def submit_sessions(self, batch: Any) -> None:
        self.events.append("dispatch")
        if self.fail:
            raise RuntimeError("dispatch failed")
        self.batches.append(batch)


class _IteratorCountingDataloader:
    def __init__(self, batches: list[dict[str, Any]]):
        self.batches = batches
        self.iter_calls = 0

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):  # noqa: ANN204
        self.iter_calls += 1
        return iter(self.batches)


class _CheckpointDataloader:
    def __init__(self, *, length: int, yielded: int):
        self.length = length
        self.yielded = yielded
        self.loaded_state = None

    def __len__(self) -> int:
        return self.length

    def state_dict(self) -> dict[str, int]:
        return {"yielded": self.yielded}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.loaded_state = dict(state)
        self.yielded = int(state["yielded"])

    def __iter__(self):  # noqa: ANN204
        return iter(())


def _dispatcher_for_checkpoint(*, length: int = 4, batch_size: int = 2) -> RolloutDispatcher:
    producer = object.__new__(RolloutDispatcher)
    producer.config = SimpleNamespace(state_batch_size=batch_size)
    producer.prompt_dataloader = _CheckpointDataloader(length=length, yielded=3)
    producer._prompt_iterator = object()
    producer._prompt_index = 19
    producer._run_task = None
    producer._metrics = {"dispatcher/prompts_dispatched": 17}
    return producer


def test_rollout_dispatcher_checkpoint_round_trip(tmp_path: Path):
    checkpoint_path = tmp_path / "producer" / "state.pt"
    source = _dispatcher_for_checkpoint()

    assert asyncio.run(source.save_checkpoint(str(checkpoint_path))) is None
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert set(payload) == {
        "schema_version",
        "prompt_dataloader",
        "prompt_index",
        "state_batch_size",
        "dataloader_length",
    }
    restored = _dispatcher_for_checkpoint()
    restored.prompt_dataloader.yielded = 0
    restored._prompt_index = 0
    restored._metrics = {"dispatcher/fresh_runtime": 1}
    stale_iterator = restored._prompt_iterator
    assert restored.load_checkpoint(str(checkpoint_path)) is None

    assert restored.prompt_dataloader.loaded_state == {"yielded": 3}
    assert restored._prompt_iterator is not stale_iterator
    assert restored._prompt_index == 19
    assert restored._metrics == {"dispatcher/fresh_runtime": 1}


def test_rollout_dispatcher_checkpoint_rejects_incompatible_or_started_restore(tmp_path: Path):
    checkpoint_path = tmp_path / "producer.pt"
    asyncio.run(_dispatcher_for_checkpoint().save_checkpoint(str(checkpoint_path)))

    incompatible = _dispatcher_for_checkpoint(length=5)
    with pytest.raises(RuntimeError, match="checkpoint is incompatible"):
        incompatible.load_checkpoint(str(checkpoint_path))

    started = _dispatcher_for_checkpoint()
    started._run_task = object()
    with pytest.raises(RuntimeError, match="after start"):
        started.load_checkpoint(str(checkpoint_path))


def test_rollout_dispatcher_checkpoint_rejects_corrupt_payload(tmp_path: Path):
    checkpoint_path = tmp_path / "producer.pt"
    torch.save(["not", "a", "mapping"], checkpoint_path)

    with pytest.raises(RuntimeError, match="invalid rollout dispatcher checkpoint payload"):
        _dispatcher_for_checkpoint().load_checkpoint(str(checkpoint_path))


def test_rollout_dispatcher_creates_prompt_iterator_at_init(monkeypatch: pytest.MonkeyPatch):
    dataloader = _IteratorCountingDataloader(
        [_collated_prompt_batch({"uid": "u0", "raw_prompt": [{"role": "user", "content": "prompt"}]})]
    )

    monkeypatch.setattr(rollout_dispatcher_module, "create_rl_dataset", lambda *args, **kwargs: object())
    monkeypatch.setattr(rollout_dispatcher_module, "create_rl_sampler", lambda *args, **kwargs: None)
    monkeypatch.setattr(rollout_dispatcher_module, "StatefulDataLoader", lambda **kwargs: dataloader)
    producer = RolloutDispatcher(
        config=cast(Any, SimpleNamespace(state_batch_size=1)),
        data_config=OmegaConf.create(
            {
                "train_files": "train.parquet",
                "train_max_samples": -1,
                "dataloader_num_workers": 0,
                "shuffle": False,
            }
        ),
        tokenizer=object(),
        agent_loop_manager=object(),
        relay_client=object(),
        rollout_router=object(),
    )

    assert dataloader.iter_calls == 1
    assert producer._prompt_iterator is not None
    batch = producer._next_prompt_batch()
    uids = tu.get(batch, "uid")
    assert len(uids) == 1
    assert isinstance(uids[0], str) and uids[0] and uids[0] != "u0"
    assert tu.get(batch, "raw_prompt")[0] == [{"role": "user", "content": "prompt"}]
    assert dataloader.iter_calls == 1


def test_initialize_runtime_rejects_finite_prompt_epochs():
    with pytest.raises(ValueError, match="total_epochs=null"):
        _initialize_runtime(OmegaConf.create({"trainer": {"total_epochs": 1}}))


def test_rollout_dispatcher_assigns_epoch_prefixed_uid_without_underscores(monkeypatch):
    monkeypatch.setattr(
        rollout_dispatcher_module.uuid,
        "uuid4",
        lambda: SimpleNamespace(hex="0123456789abcdef"),
    )
    producer = object.__new__(RolloutDispatcher)
    producer.config = SimpleNamespace(state_batch_size=1)
    prompt_batch = _collated_prompt_batch({"uid": "dataset-uid", "raw_prompt": [{"role": "user", "content": "hi"}]})
    producer.prompt_dataloader = [prompt_batch, prompt_batch]
    producer._prompt_iterator = iter(producer.prompt_dataloader)
    producer._prompt_index = 1
    batch = producer._next_prompt_batch()
    assert tu.get(batch, "uid") == ["0-0123456789abcdef"]

    producer._prompt_index = 2
    producer._prompt_iterator = iter(producer.prompt_dataloader)
    batch = producer._next_prompt_batch()
    assert tu.get(batch, "uid") == ["1-0123456789abcdef"]
    assert "_" not in tu.get(batch, "uid")[0]


def _dispatcher_for_version(*, versions: list[int]) -> RolloutDispatcher:
    producer = object.__new__(RolloutDispatcher)
    producer.config = SimpleNamespace(
        schedule_interval_s=0,
        state_batch_size=1,
    )
    producer.relay_client = _VersionRelay(versions)
    producer.agent_loop_manager = None
    # Keys that wait/dispatch paths increment; assignments do not need a prior entry.
    producer._metrics = {
        "dispatcher/prompt_batches_dispatched": 0,
        "dispatcher/prompts_dispatched": 0,
        "dispatcher/version_wait_duration_s": 0.0,
    }
    return producer


def _wait_for_test_version(producer: RolloutDispatcher) -> int:
    return asyncio.run(producer._wait_for_trainer_version())


def _dispatcher_for_dispatch(*, fail: bool = False) -> RolloutDispatcher:
    producer = _dispatcher_for_version(versions=[12])
    producer._metric_source = MetricSource([])
    producer.rollout_router = SimpleNamespace(
        get_routable_version=SimpleNamespace(remote=AsyncMock(return_value=12)),
    )
    producer.config.num_samples_per_prompt = 1
    events: list[str] = []
    producer.agent_loop_manager = _DispatchManager(fail=fail, events=events)
    producer.prompt_dataloader = [
        _collated_prompt_batch(
            {
                "uid": "u0",
                "raw_prompt": [{"role": "user", "content": "hi"}],
                "extra_info": {},
            }
        )
    ]
    producer._prompt_iterator = iter(producer.prompt_dataloader)
    producer._prompt_index = 0
    return producer


def test_rollout_dispatcher_requires_admission_without_generation_fallback():
    producer = _dispatcher_for_dispatch()
    generated = []
    producer.agent_loop_manager = SimpleNamespace(generate_sequences=generated.append)

    with pytest.raises(AttributeError, match="submit_sessions"):
        asyncio.run(producer._dispatch_next_batch())

    assert generated == []
    assert producer._metrics["dispatcher/prompt_batches_dispatched"] == 0


@pytest.mark.asyncio
async def test_rollout_dispatcher_waits_for_session_admission():
    producer = _dispatcher_for_dispatch()
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    admitted = asyncio.Event()

    def submit_sessions(batch: Any) -> None:
        loop.call_soon_threadsafe(started.set)
        asyncio.run_coroutine_threadsafe(admitted.wait(), loop).result(timeout=5)

    producer.agent_loop_manager = SimpleNamespace(submit_sessions=submit_sessions)
    dispatch = asyncio.create_task(producer._dispatch_next_batch())
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        assert not dispatch.done()
        assert producer._metrics["dispatcher/prompt_batches_dispatched"] == 0
    finally:
        admitted.set()
        await asyncio.wait_for(dispatch, timeout=2)

    assert producer._metrics["dispatcher/prompt_batches_dispatched"] == 1


def test_rollout_dispatcher_stamps_global_steps_before_dispatch():
    producer = _dispatcher_for_dispatch()
    producer.prompt_dataloader = [
        _collated_prompt_batch(
            {
                "uid": "u0",
                "raw_prompt": [{"role": "user", "content": "hi"}],
                "data_source": "swe",
                "reward_model": {"ground_truth": "pass"},
                "extra_info": {"index": 1},
                "tools_kwargs": {"env": {"image": "runner"}},
                "agent_name": "uni_agent",
            }
        )
    ]
    producer._prompt_iterator = iter(producer.prompt_dataloader)

    asyncio.run(producer._dispatch_next_batch())

    assert producer._metrics["dispatcher/prompt_batches_dispatched"] == 1
    batch = producer.agent_loop_manager.batches[0]
    assert batch["global_steps"] == 12
    assert tu.get(batch, "target_weight_version") is None
    assert tu.get(batch, "__rollout_n__") is None
    assert tu.get(batch, "__session_ids__") is None
    assert tu.get(batch, "raw_prompt")[0] == [{"role": "user", "content": "hi"}]
    assert tu.get(batch, "data_source") == ["swe"]
    assert tu.get(batch, "reward_model")[0] == {"ground_truth": "pass"}
    assert tu.get(batch, "extra_info")[0] == {"index": 1}
    assert tu.get(batch, "tools_kwargs")[0] == {"env": {"image": "runner"}}
    assert tu.get(batch, "agent_name") == ["uni_agent"]
    assert producer.agent_loop_manager.events == ["dispatch"]


def test_rollout_dispatcher_uses_configured_rollout_count_for_metrics():
    producer = _dispatcher_for_dispatch()
    producer.config.num_samples_per_prompt = 2

    asyncio.run(producer._dispatch_next_batch())

    batch = producer.agent_loop_manager.batches[0]
    assert tu.get(batch, "__rollout_n__") is None
    assert tu.get(batch, "__session_ids__") is None
    assert producer._metrics["dispatcher/latest_submitted_trajectory_count"] == 2
    metrics = asyncio.run(producer.get_metrics())
    assert metrics["dispatcher/latest_submitted_trajectory_count"] == 2


def test_rollout_dispatcher_records_the_actual_submitted_weight_version(monkeypatch: pytest.MonkeyPatch):
    producer = _dispatcher_for_dispatch()
    producer.rollout_router.get_routable_version.remote.return_value = 14
    asyncio.run(producer._dispatch_next_batch())

    assert tu.get(producer.agent_loop_manager.batches[0], "global_steps") == 14
    assert producer._metrics["dispatcher/trainer_version"] == 12
    assert producer._metrics["dispatcher/latest_target_weight_version"] == 14
    assert producer._metrics["dispatcher/version_wait_duration_s"] >= 0.0
    producer.rollout_router.get_routable_version.remote.assert_awaited_once_with(weight_version=12)


@pytest.mark.asyncio
async def test_rollout_dispatcher_waits_for_router_readiness_before_submit():
    producer = _dispatcher_for_dispatch()
    waiting = asyncio.Event()
    ready = asyncio.Event()

    async def wait_for_version(*, weight_version: int) -> int:
        assert weight_version == 12
        waiting.set()
        if not ready.is_set():
            raise RouteUnavailableError("replicas are updating")
        return 14

    producer.rollout_router.get_routable_version.remote.side_effect = wait_for_version
    dispatch = asyncio.create_task(producer._dispatch_next_batch())
    try:
        await asyncio.wait_for(waiting.wait(), timeout=1)
        assert not producer.agent_loop_manager.batches
        assert producer._metrics["dispatcher/prompt_batches_dispatched"] == 0
        assert (await producer.get_metrics())["dispatcher/prompt_batches_dispatched"] == 0
        ready.set()
        await asyncio.wait_for(dispatch, timeout=1)
    finally:
        if not dispatch.done():
            dispatch.cancel()
        await asyncio.gather(dispatch, return_exceptions=True)

    assert tu.get(producer.agent_loop_manager.batches[0], "global_steps") == 14
    assert producer._metrics["dispatcher/prompt_batches_dispatched"] == 1


@pytest.mark.asyncio
async def test_dispatcher_readiness_timeout_and_cancellation_do_not_submit(monkeypatch):
    producer = _dispatcher_for_dispatch()
    unavailable = producer.rollout_router.get_routable_version.remote
    unavailable.side_effect = RouteUnavailableError("replicas are updating")
    monkeypatch.setattr(rollout_dispatcher_module, "_VERSION_WAIT_TIMEOUT_S", 0.02)
    with pytest.raises(TimeoutError):
        await producer._dispatch_next_batch()
    assert unavailable.await_count == 1
    assert producer.agent_loop_manager.batches == []

    monkeypatch.setattr(rollout_dispatcher_module, "_VERSION_WAIT_TIMEOUT_S", 10)
    pending = asyncio.create_task(producer._dispatch_next_batch())
    while unavailable.await_count < 2:
        await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert producer.agent_loop_manager.batches == []


@pytest.mark.asyncio
async def test_dispatcher_planned_pause_budget_extends_once_from_original_start(monkeypatch):
    producer = _dispatcher_for_dispatch()
    unavailable = producer.rollout_router.get_routable_version.remote
    unavailable.side_effect = RouteUnavailableError("training owns colocated memory", planned_pause=True)
    monkeypatch.setattr(rollout_dispatcher_module, "_VERSION_WAIT_TIMEOUT_S", 0.01)
    monkeypatch.setattr(rollout_dispatcher_module, "_PLANNED_PAUSE_WAIT_TIMEOUT_S", 0.04)
    monkeypatch.setattr(rollout_dispatcher_module, "_VERSION_POLL_INTERVAL_S", 0.005)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(producer._dispatch_next_batch(), timeout=0.2)
    assert unavailable.await_count >= 2
    assert producer.agent_loop_manager.batches == []


@pytest.mark.parametrize("available_version", [11, True, None, 12.0])
def test_rollout_dispatcher_rejects_invalid_available_version_before_submit(available_version: Any):
    producer = _dispatcher_for_dispatch()
    producer.rollout_router.get_routable_version.remote.return_value = available_version

    with pytest.raises(RuntimeError, match="Router returned an invalid available weight version"):
        asyncio.run(producer._dispatch_next_batch())

    assert not producer.agent_loop_manager.batches
    assert producer._metrics["dispatcher/prompt_batches_dispatched"] == 0


def test_rollout_dispatcher_does_not_submit_when_router_wait_fails():
    producer = _dispatcher_for_dispatch()
    producer.rollout_router.get_routable_version.remote.side_effect = RuntimeError("router wait failed")

    with pytest.raises(RuntimeError, match="router wait failed"):
        asyncio.run(producer._dispatch_next_batch())

    assert not producer.agent_loop_manager.batches
    assert producer._metrics["dispatcher/prompt_batches_dispatched"] == 0


def test_rollout_dispatcher_does_not_count_failed_dispatch():
    producer = _dispatcher_for_dispatch(fail=True)

    with pytest.raises(RuntimeError, match="dispatch failed"):
        asyncio.run(producer._dispatch_next_batch())

    assert producer._metrics["dispatcher/prompt_batches_dispatched"] == 0
    assert producer.agent_loop_manager.events == ["dispatch"]


def test_rollout_dispatcher_waits_for_trainer_target_version():
    producer = _dispatcher_for_version(versions=[-1, 5])

    version = _wait_for_test_version(producer)

    assert version == 5


def test_rollout_dispatcher_uses_exact_trainer_version():
    producer = _dispatcher_for_version(versions=[10])

    version = _wait_for_test_version(producer)

    assert version == 10


def test_rollout_dispatcher_keeps_first_ready_trainer_version():
    producer = _dispatcher_for_version(versions=[5, 6])

    version = _wait_for_test_version(producer)

    assert version == 5
    assert producer.relay_client.get_trainer_version_calls == [{"args": (), "kwargs": {}}]


def test_rollout_dispatcher_requires_raw_prompt():
    producer = object.__new__(RolloutDispatcher)
    producer.config = SimpleNamespace(state_batch_size=1)
    producer.prompt_dataloader = [{"uid": ["u0"]}]
    producer._prompt_iterator = iter(producer.prompt_dataloader)
    producer._prompt_index = 0
    with pytest.raises(ValueError, match="requires raw_prompt"):
        producer._next_prompt_batch()


def test_runtime_multi_version_manager_config_from_existing_hydra_tree():
    ppo_cfg = OmegaConf.load(_verl_trainer_yaml("_generated_ppo_trainer.yaml"))
    async_rl_cfg = OmegaConf.load(_REPO_ROOT / "src/laminar/train/config/async_rl_base.yaml")
    config = OmegaConf.merge(ppo_cfg, async_rl_cfg)
    config.rollout_dispatcher.schedule_interval_s = 2.5
    multi_version_cfg = omega_conf_to_dataclass(config.multi_version_rollout)

    assert multi_version_cfg.repack.enable is True
    assert multi_version_cfg.repack.source_max_kv_cache_usage == 0.5
    assert multi_version_cfg.repack.destination_max_kv_cache_usage == 0.95
    assert multi_version_cfg.schedule_interval_s == 2.5


def test_runtime_repack_config_can_disable_repack_only():
    config = _rollout_dispatcher_config(multi_version_rollout_overrides={"repack": {"enable": False}})
    rollout_cfg = omega_conf_to_dataclass(config.rollout_dispatcher)
    multi_version_cfg = omega_conf_to_dataclass(config.multi_version_rollout)

    assert rollout_cfg.schedule_interval_s == 2.5
    assert multi_version_cfg.repack.enable is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_max_kv_cache_usage", -0.01),
        ("source_max_kv_cache_usage", 1.01),
        ("destination_max_kv_cache_usage", -0.01),
        ("destination_max_kv_cache_usage", 1.01),
    ],
)
def test_runtime_repack_config_rejects_invalid_kv_snapshot_threshold(field: str, value: float):
    config = _rollout_dispatcher_config(multi_version_rollout_overrides={"repack": {field: value}})
    with pytest.raises(InstantiationException, match=field):
        omega_conf_to_dataclass(config.multi_version_rollout)


def test_multi_turn_config_preserves_mainline_defaults():
    cfg = omega_conf_to_dataclass(
        OmegaConf.create(
            {
                "_target_": "verl.workers.config.MultiTurnConfig",
                "enable": True,
                "format": "llama3_json",
                "tool_config_path": "/tmp/tools.yaml",
                "max_assistant_turns": None,
                "max_user_turns": 5,
            }
        )
    )

    assert cfg.enable is True
    assert cfg.format == "llama3_json"
    assert cfg.tool_config_path == "/tmp/tools.yaml"
    assert cfg.max_assistant_turns is None
    assert cfg.max_user_turns == 5
    assert cfg.max_parallel_calls == 1
    assert cfg.max_tool_response_length == 256
    assert cfg.tool_response_truncate_side == "middle"


def test_async_rl_trainer_config_from_existing_hydra_tree():
    cfg = _composed_async_rl_config()
    cfg.trainer.total_training_steps = 10
    cfg.trainer.weight_publish_interval = 4
    cfg.trainer.max_actor_ckpt_to_keep = 2
    cfg.trainer.balance_batch = False

    trainer_cfg = AsyncRLTrainerConfig.from_config(cfg)

    assert trainer_cfg.total_training_steps == 10
    assert trainer_cfg.weight_publish_interval == 4
    assert trainer_cfg.max_actor_ckpt_to_keep == 2
    assert trainer_cfg.balance_batch is False


def test_async_rl_trainer_config_uses_composed_defaults():
    trainer_cfg = AsyncRLTrainerConfig.from_config(_composed_async_rl_config())

    assert trainer_cfg.weight_publish_interval == 1
    assert trainer_cfg.balance_batch is True


def test_async_rl_trainer_config_rejects_nonpositive_weight_publish_interval():
    cfg = _composed_async_rl_config()
    cfg.trainer.weight_publish_interval = 0
    with pytest.raises(ValueError, match="trainer.weight_publish_interval must be positive"):
        AsyncRLTrainerConfig.from_config(cfg)


def test_async_rl_trainer_derives_trajectory_mini_batch_from_prompt_level_config():
    trainer = object.__new__(AsyncRLTrainer)
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": 120},
            "actor_rollout_ref": {
                "actor": {"ppo_mini_batch_size": 60},
                "rollout": {"n": 16},
            },
        }
    )

    assert trainer.train_batch_size == 1920
    assert trainer.actor_mini_batch_size() == 960


def test_async_rl_trainer_skips_final_weight_publish():
    trainer = object.__new__(AsyncRLTrainer)
    trainer.trainer_config = AsyncRLTrainerConfig(weight_publish_interval=5)

    assert trainer._step_should_publish_weights(5, total_steps=None, checkpoint_due=False) is True
    assert trainer._step_should_publish_weights(6, total_steps=None, checkpoint_due=False) is False
    assert trainer._step_should_publish_weights(10, total_steps=10, checkpoint_due=False) is False
    assert trainer._step_should_publish_weights(12, total_steps=12, checkpoint_due=False) is False
    assert trainer._step_should_publish_weights(10, total_steps=10, checkpoint_due=True) is True


def _relay_broadcast_stats(*, logical_payload_bytes: int = 1024) -> RelayBroadcastStats:
    return RelayBroadcastStats(
        relay_broadcast_time_s=0.25,
        logical_payload_bytes=logical_payload_bytes,
        tensor_count=2,
        hop_count=1,
        bottleneck_hop_index=0,
        end_to_end_goodput_bytes_per_s=4096.0,
    )


def _relay_publish_metrics(*, version: int = 7) -> dict[str, float]:
    actor_stats = ActorPublishStats(
        publisher_count=1,
        tensor_count=12,
        chunk_count=3,
        payload_bytes=1000,
        export_time_s=0.1,
        connect_time_s=0.02,
        sender_time_s=0.5,
        tensor_prepare_time_s=0.03,
        pack_time_s=0.04,
        staging_time_s=0.2,
        transport_time_s=0.15,
        completion_ack_time_s=0.07,
        total_time_s=0.9,
    )
    metrics = {
        "relay_publish/completed_version": float(version),
        "timing/relay_publish_actor_flush": 1.0,
        "timing/relay_publish_broadcast": 2.0,
        "timing/relay_publish_commit": 0.05,
        "timing/relay_publish_total": 3.0,
        "relay_publish/effective_goodput_bytes_per_s": 1000 / 3.0,
    }
    metrics.update(_relay_broadcast_stats(logical_payload_bytes=1000).as_metrics())
    metrics.update(actor_stats.as_metrics())
    return metrics


def test_relay_actor_publisher_stats_aggregate_payload_without_protocol_ids():
    assert ActorPublishStats.from_results(
        [
            ActorPublishStats(publisher_count=1, payload_bytes=10, tensor_count=1, sender_time_s=0.2),
            None,
            ActorPublishStats(publisher_count=1, payload_bytes=20, tensor_count=2, sender_time_s=0.3),
        ]
    ) == ActorPublishStats(
        publisher_count=2,
        payload_bytes=30,
        tensor_count=3,
        sender_time_s=0.3,
    )


def test_async_rl_trainer_publish_actor_weights_delegates_to_relay_publish():
    class FakeRelay:
        def __init__(self):
            self.kwargs: dict[str, Any] | None = None

        async def publish(self, actor_wg, *, version, dtype):  # noqa: ANN001
            self.kwargs = {
                "actor_wg": actor_wg,
                "version": version,
                "dtype": dtype,
            }
            return _relay_publish_metrics(version=version)

    trainer = object.__new__(AsyncRLTrainer)
    trainer.actor_wg = "actor-wg"
    trainer.relay_client = FakeRelay()
    trainer.rollout_dtype = "bfloat16"

    publish_metrics = asyncio.run(trainer.publish_actor_weights_to_relay(version=3))

    assert trainer.relay_client.kwargs == {
        "actor_wg": "actor-wg",
        "version": 3,
        "dtype": "bfloat16",
    }
    assert publish_metrics == _relay_publish_metrics(version=3)


def test_async_rl_trainer_connect_relay_initializes_full_storage_from_actor_metadata(monkeypatch):
    events = []
    state_meta = {"dense.weight": TensorMeta(dtype=torch.bfloat16, shape=torch.Size([2, 3]))}

    class FakeActorWorkerGroup:
        def get_actor_state_meta_dict(self, *, dtype):
            assert dtype == "bfloat16"
            events.append("actor_metadata")
            return state_meta

    class FakeRelay:
        def connect_actor_workers(self, actor_wg):
            assert actor_wg is trainer.actor_wg
            events.append("connect")

        def init_full_state_dict_in_shm(self, *, full_state_meta_dict):
            assert full_state_meta_dict is state_meta
            events.append("final_storage")
            return "init-ref"

    trainer = object.__new__(AsyncRLTrainer)
    trainer.actor_wg = FakeActorWorkerGroup()
    trainer.relay_client = FakeRelay()
    trainer.rollout_dtype = "bfloat16"
    monkeypatch.setattr(async_rl_trainer_module.ray, "get", lambda ref: events.append(("ray_get", ref)))

    result = trainer._connect_relay()

    assert result is state_meta
    assert events == [
        "connect",
        "actor_metadata",
        "final_storage",
        ("ray_get", "init-ref"),
    ]


def test_async_rl_trainer_pipelines_relay_publish_with_sample_wait():
    async def run():
        publish_gate = asyncio.Event()
        sample_gate = asyncio.Event()
        trainer = object.__new__(AsyncRLTrainer)
        trainer._pending_relay_publish = None

        async def publish(version):  # noqa: ANN001
            assert version == 7
            await publish_gate.wait()
            return _relay_publish_metrics(version=7)

        async def sample():
            await sample_gate.wait()
            return SimpleNamespace(partition_id="train", keys=["k0"])

        trainer.publish_actor_weights_to_relay = publish  # type: ignore[method-assign]
        trainer._schedule_relay_publish(version=7)
        sample_task = asyncio.create_task(
            trainer._await_sample_or_relay_failure(sample(), {}),
        )
        await asyncio.sleep(0)
        assert not sample_task.done()

        sample_gate.set()
        batch, early_wait_time_s = await sample_task
        assert batch.keys == ["k0"]
        assert early_wait_time_s == 0.0
        assert trainer._pending_relay_publish is not None

        publish_gate.set()
        metrics: dict[str, Any] = {}
        residual_wait_time_s = await trainer._wait_for_pending_relay_publish(metrics)
        assert residual_wait_time_s >= 0.0
        assert metrics == _relay_publish_metrics(version=7)
        assert trainer._pending_relay_publish is None

    asyncio.run(run())


def test_async_rl_trainer_surfaces_relay_failure_while_sample_waits():
    async def run():
        publish_gate = asyncio.Event()
        trainer = object.__new__(AsyncRLTrainer)
        trainer._pending_relay_publish = None

        async def publish(version):  # noqa: ANN001
            del version
            await publish_gate.wait()
            raise RuntimeError("relay broadcast failed")

        async def sample():
            await asyncio.Event().wait()

        trainer.publish_actor_weights_to_relay = publish  # type: ignore[method-assign]
        trainer._schedule_relay_publish(version=8)
        sample_task = asyncio.create_task(
            trainer._await_sample_or_relay_failure(sample(), {}),
        )
        publish_gate.set()
        with pytest.raises(RuntimeError, match="relay broadcast failed"):
            await sample_task
        assert trainer._pending_relay_publish is None

    asyncio.run(run())


def test_async_rl_trainer_rejects_overlapping_relay_publish():
    async def run():
        publish_gate = asyncio.Event()
        trainer = object.__new__(AsyncRLTrainer)
        trainer._pending_relay_publish = None

        async def publish(version):  # noqa: ANN001
            assert version == 3
            await publish_gate.wait()
            return {}

        trainer.publish_actor_weights_to_relay = publish  # type: ignore[method-assign]
        trainer._schedule_relay_publish(version=3)
        with pytest.raises(RuntimeError, match="overlapping Relay publishes"):
            trainer._schedule_relay_publish(version=4)
        publish_gate.set()
        await trainer._wait_for_pending_relay_publish({})

    asyncio.run(run())


def test_async_rl_defaults_route_wandb_tracking_through_tracker():
    ppo_cfg = OmegaConf.load(_verl_trainer_yaml("_generated_ppo_trainer.yaml"))
    async_rl_cfg = OmegaConf.load(_REPO_ROOT / "src/laminar/train/config/async_rl_base.yaml")
    cfg = OmegaConf.merge(ppo_cfg, async_rl_cfg)

    tracker_cfg = TrackerConfig.from_config(cfg)

    assert list(cfg.trainer.logger) == ["console", "wandb"]
    assert "tracker" not in cfg.trainer
    assert "dump" not in cfg.trainer
    assert list(cfg.tracker.logger) == ["console", "wandb"]
    assert cfg.tracker.dump.path is None
    assert "enable" not in cfg.tracker
    assert not hasattr(tracker_cfg, "enable")
    assert tracker_cfg.logger == ["console", "wandb"]
    assert TrackerConfig().logger == ["console"]


def test_async_rl_tracker_logger_can_override_trainer_fallback():
    ppo_cfg = OmegaConf.load(_verl_trainer_yaml("_generated_ppo_trainer.yaml"))
    async_rl_cfg = OmegaConf.load(_REPO_ROOT / "src/laminar/train/config/async_rl_base.yaml")
    cfg = OmegaConf.merge(ppo_cfg, async_rl_cfg)
    cfg.trainer.logger = ["wandb"]
    cfg.tracker.logger = ["console"]

    tracker_cfg = TrackerConfig.from_config(cfg)

    assert tracker_cfg.logger == ["console"]


def test_train_config_exposes_explicit_train_state_resume_path():
    cfg = _composed_async_rl_config()

    assert cfg.trainer.resume_from_path is None
    cfg.trainer.resume_from_path = "/tmp/ckpt/global_step_40"
    assert cfg.trainer.resume_from_path == "/tmp/ckpt/global_step_40"


@pytest.mark.parametrize("config_key_limit", [None, 64])
def test_tracker_config_from_existing_hydra_tree(config_key_limit):
    cfg = _composed_async_rl_config()
    cfg.tracker.dump.path = "/tmp/metrics.jsonl"
    cfg.trainer.logger = ["console"]
    cfg.trainer.project_name = "proj"
    cfg.trainer.experiment_name = "exp"
    cfg.tracker.config_key_max_bytes = config_key_limit

    tracker_cfg = TrackerConfig.from_config(cfg)

    assert not hasattr(tracker_cfg, "enable")
    assert not hasattr(tracker_cfg, "interval_s")
    assert tracker_cfg.remote_metrics_timeout_s == 5.0
    assert not hasattr(tracker_cfg, "enable_validation")
    assert tracker_cfg.logger == ["console"]
    assert tracker_cfg.dump_path == "/tmp/metrics.jsonl"
    assert tracker_cfg.dump_require_persistent_path is False
    assert tracker_cfg.config_key_max_bytes == config_key_limit


@pytest.mark.parametrize("value", ["false", 0])
def test_tracker_config_rejects_non_boolean_persistent_path_requirement(value):
    cfg = _composed_async_rl_config()
    cfg.tracker.dump.require_persistent_path = value

    with pytest.raises(ValueError, match="require_persistent_path must be a boolean"):
        TrackerConfig.from_config(cfg)


def test_tracker_metric_sources_can_be_connected_after_construction():
    tracker = Tracker(config=TrackerConfig())
    rollout_dispatcher = cast(Any, object())
    experience_buffer = cast(Any, object())
    multi_version_rollout_manager = cast(Any, object())
    relay_client = cast(Any, object())

    tracker.connect_metric_sources(
        rollout_dispatcher=rollout_dispatcher,
        experience_buffer=experience_buffer,
        multi_version_rollout_manager=multi_version_rollout_manager,
        relay_client=relay_client,
    )

    assert tracker.rollout_dispatcher is rollout_dispatcher
    assert tracker.experience_buffer is experience_buffer
    assert tracker.multi_version_rollout_manager is multi_version_rollout_manager
    assert tracker.relay_client is relay_client
