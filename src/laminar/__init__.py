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
"""Laminar training and evaluation over shared versioned rollout services.

Scope is intentionally narrow: ``recipes/swe/train.sh`` and ``recipes/swe/eval.sh``
are the Qwen3-Coder-30B-A3B SWE training and evaluation recipes on top of
``python -m laminar.train`` and ``python -m laminar.eval``.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
