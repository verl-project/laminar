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
"""Backing-store helpers for the CPU relay buffer."""

from __future__ import annotations

import os

import ray
import torch

from laminar.utils.tensor import TensorMeta

SHM_ROOT: str = "/dev/shm"


def make_shared_tensor_from_meta(meta: TensorMeta, key: str) -> torch.Tensor:
    """Create a tensor backed by this Ray job's shared-memory namespace.

    File-backed tensors cannot be pinned directly, hence the explicit
    ``cudaHostRegister`` required by Relay's fixed pinned-CPU storage.
    """
    # TODO(async_rl): sanitize/hash keys and add an owned cleanup path that
    # calls cudaHostUnregister (when pinned) and unlinks stale /dev/shm files.
    job_id = os.environ.get("VERL_RAY_JOB_ID") or ray.get_runtime_context().get_job_id()
    filename = f"{SHM_ROOT}/{job_id}-{key}"
    shared_tensor = torch.from_file(filename, shared=True, size=meta.shape.numel(), dtype=meta.dtype).reshape(
        meta.shape
    )
    lib = torch.cuda.cudart()
    err = lib.cudaHostRegister(shared_tensor.data_ptr(), shared_tensor.nbytes, 0)
    assert err == 0, f"cudaHostRegister failed with error {err}"
    return shared_tensor
