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
"""Trainer-local ExperienceBuffer RPCs and process-local store materialization.

``ExperienceBufferServer`` remains the Ray-actor metadata/control-plane owner.
``materialize_batch`` fetches tensors in the trainer process (no Ray payload hop).
``ExperienceBufferClient`` only wraps sample / release RPCs.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

import torch
from tensordict import NonTensorData, TensorDict
from verl.utils.transferqueue_utils import tq
from verl.workers.utils.padding import left_right_2_no_padding

from laminar.train.experience_buffer.server import SampledBatch

if TYPE_CHECKING:
    from ray.actor import ActorHandle


def _tensor_payload_nbytes(value: Any) -> int:
    if not isinstance(value, torch.Tensor):
        return 0
    if value.is_nested:
        value = value.values()
    return int(value.numel() * value.element_size())


def _producer_position_rows(data: TensorDict, sequence_lengths: torch.Tensor) -> list[torch.Tensor]:
    """Consume VERL AgentLoop's producer-owned ``position_ids`` field.

    Each trajectory is ``(seq,)`` or ``(rope_rows, seq)``. The trainer may
    change padding, but must not derive this model-specific field again.
    """
    if "position_ids" not in data:
        raise RuntimeError("experience store batch is missing required sequence-aligned field 'position_ids'")
    value = data.pop("position_ids")
    if not isinstance(value, torch.Tensor) or not value.is_nested:
        raise RuntimeError("experience store field 'position_ids' must be a jagged tensor")
    rows = list(value.unbind())
    actual_lengths = [int(row.shape[-1]) for row in rows]
    expected_lengths = sequence_lengths.tolist()
    if actual_lengths != expected_lengths:
        raise RuntimeError(
            "experience store position_ids lengths do not match prompt+response lengths: "
            f"actual={actual_lengths} expected={expected_lengths}"
        )
    layouts = {tuple(row.shape[:-1]) for row in rows}
    if len(layouts) != 1 or len(next(iter(layouts))) > 1:
        raise RuntimeError(f"experience store position_ids must share one (seq,) or (rows, seq) layout: {layouts}")
    return rows


def _canonical_position_ids(
    rows: list[torch.Tensor],
    prompt_lengths: list[int],
    *,
    prompt_width: int,
    sequence_width: int,
) -> torch.Tensor:
    """Place producer position ids at the left-padded prompt / right-padded response boundary."""
    canonical = rows[0].new_zeros((len(rows), *rows[0].shape[:-1], sequence_width))
    for row, (positions, prompt_len) in enumerate(zip(rows, prompt_lengths, strict=True)):
        start = prompt_width - prompt_len
        canonical[row, ..., start : start + positions.shape[-1]] = positions
    return canonical


def materialize_batch(
    batch: SampledBatch,
    *,
    metrics: dict[str, Any],
    require_routed_experts: bool,
    use_remove_padding: bool,
    temperature: float,
) -> TensorDict:
    """Fetch TQ tensors for ``batch`` and rebuild a train-ready TensorDict.

    The store returns independent jagged fields. This helper rebuilds the
    canonical ``[left-padded prompt | right-padded response]`` boundary before
    optional no-padding conversion for the actor engine.
    """
    fields = [
        "prompts",
        "responses",
        "position_ids",
        "response_mask",
        "rm_scores",
        "rollout_log_probs",
    ]
    if require_routed_experts:
        fields.append("routed_experts")
    assert batch.partition_id, "sampled batch must carry a TransferQueue partition"
    fetch_start = time.perf_counter()
    data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
    store_fetch_time_s = time.perf_counter() - fetch_start
    store_payload_bytes = float(sum(_tensor_payload_nbytes(value) for value in data.values()))
    routed_experts_bytes = float(_tensor_payload_nbytes(data.get("routed_experts", None)))
    metrics["timing/materialize_batch_store_fetch"] = store_fetch_time_s
    metrics["materialize_batch/store_payload_bytes"] = store_payload_bytes
    metrics["materialize_batch/routed_experts_bytes"] = routed_experts_bytes
    metrics["materialize_batch/store_fetch_bytes_per_s"] = (
        store_payload_bytes / store_fetch_time_s if store_fetch_time_s > 0 else 0.0
    )
    metrics["materialize_batch/routed_experts_fraction"] = (
        routed_experts_bytes / store_payload_bytes if store_payload_bytes > 0 else 0.0
    )
    tensor_format_start = time.perf_counter()

    required_ragged_fields = ("prompts", "responses")
    for field in required_ragged_fields:
        value = data[field]
        if not isinstance(value, torch.Tensor) or not value.is_nested:
            raise RuntimeError(f"experience store field {field!r} must be a jagged tensor")

    prompt_lengths = data["prompts"].offsets().diff()
    response_lengths = data["responses"].offsets().diff()
    response_aligned_fields = (
        "response_mask",
        "rm_scores",
        "rollout_log_probs",
    )
    for field in response_aligned_fields:
        if field not in data:
            raise RuntimeError(f"experience store batch is missing required response-aligned field {field!r}")
        value = data[field]
        if not isinstance(value, torch.Tensor) or not value.is_nested:
            raise RuntimeError(f"experience store field {field!r} must be a jagged tensor")
        actual_lengths = value.offsets().diff()
        if not torch.equal(actual_lengths, response_lengths):
            raise RuntimeError(
                f"experience store field {field!r} lengths do not match response lengths: "
                f"actual={actual_lengths.tolist()} expected={response_lengths.tolist()}"
            )

    position_rows = _producer_position_rows(data, prompt_lengths + response_lengths)

    routed_experts = data.pop("routed_experts") if "routed_experts" in data else None
    if require_routed_experts and routed_experts is None:
        raise RuntimeError(
            "rollout routing replay is enabled but sampled experience batch is missing routed_experts; "
            "check rollout.enable_rollout_routing_replay and vLLM routed_experts output."
        )

    td = data.to_padded_tensor()

    prompt_width = int(td["prompts"].shape[1])
    canonical_prompts = torch.zeros_like(td["prompts"])
    prompt_attention_mask = torch.zeros_like(td["prompts"], dtype=torch.int64)
    response_attention_mask = torch.zeros_like(td["responses"], dtype=torch.int64)
    for row, (prompt_len, response_len) in enumerate(
        zip(prompt_lengths.tolist(), response_lengths.tolist(), strict=True)
    ):
        canonical_prompts[row, prompt_width - prompt_len :] = td["prompts"][row, :prompt_len]
        prompt_attention_mask[row, prompt_width - prompt_len :] = 1
        response_attention_mask[row, :response_len] = 1

    td["prompts"] = canonical_prompts
    td["input_ids"] = torch.cat([canonical_prompts, td["responses"]], dim=1)
    td["attention_mask"] = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)

    if routed_experts is not None:
        if not isinstance(routed_experts, torch.Tensor) or not routed_experts.is_nested:
            raise RuntimeError("experience store field 'routed_experts' must be a jagged tensor")
        sequence_lengths = prompt_lengths + response_lengths
        routed_lengths = routed_experts.offsets().diff()
        if not torch.equal(routed_lengths, sequence_lengths):
            raise RuntimeError(
                "experience store routed_experts lengths do not match prompt+response lengths: "
                f"actual={routed_lengths.tolist()} expected={sequence_lengths.tolist()}"
            )
        response_width = int(td["responses"].shape[1])
        routed_values = routed_experts.values()
        canonical_routed = routed_values.new_zeros(
            (len(batch.keys), prompt_width + response_width, *routed_values.shape[1:])
        )
        routed_offsets = routed_experts.offsets().tolist()
        for row, (prompt_len, response_len) in enumerate(
            zip(prompt_lengths.tolist(), response_lengths.tolist(), strict=True)
        ):
            routed_row = routed_values[routed_offsets[row] : routed_offsets[row + 1]]
            canonical_routed[row, prompt_width - prompt_len : prompt_width] = routed_row[:prompt_len]
            canonical_routed[row, prompt_width : prompt_width + response_len] = routed_row[prompt_len:]
        td["routed_experts"] = canonical_routed

    td["position_ids"] = _canonical_position_ids(
        position_rows,
        prompt_lengths.tolist(),
        prompt_width=prompt_width,
        sequence_width=int(td["input_ids"].shape[1]),
    )
    if use_remove_padding:
        td = left_right_2_no_padding(td)
        if position_rows[0].dim() == 2:
            # Equal sequence lengths otherwise make torch infer the RoPE rows as ragged.
            td["position_ids"] = torch.nested.nested_tensor_from_jagged(
                values=torch.cat(position_rows, dim=-1),
                offsets=td["input_ids"].offsets().to(device=position_rows[0].device),
                jagged_dim=2,
            )
    td["loss_mask"] = td["response_mask"]

    td["temperature"] = NonTensorData(temperature)
    td["global_token_num"] = NonTensorData(torch.sum(td["attention_mask"], dim=-1).tolist())
    metrics["timing/materialize_batch_tensor_format"] = time.perf_counter() - tensor_format_start
    return td


class ExperienceBufferClient:
    """Driver-local API: sample meta, materialize tensors, release consumed keys."""

    def __init__(self, experience_buffer: ActorHandle):
        self._experience_buffer = experience_buffer

    def sample_remote(self, *, batch_size: int) -> Any:
        """Return the Ray ObjectRef for a blocking buffer sample RPC."""
        return self._experience_buffer.sample.remote(  # type: ignore[attr-defined]
            batch_size=batch_size,
        )

    async def release(self, keys: list[str]) -> None:
        """Drop consumed store payloads and buffer metadata for ``keys``."""
        await cast(Any, self._experience_buffer.remove.remote(keys))
