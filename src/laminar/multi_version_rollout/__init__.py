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
"""Multi-version rollout: same-version Repack planning plus lifecycle orchestration.

Repack consolidates same-version traffic so stale engines can load newer
weights. The Router applies and commits the plan because it owns route
bindings; source turns are aborted and continue at the same version before
the manager loads new weights and reopens the source. Under latest-version
continuation the Router instead drains every stale server without a
destination, and aborted turns continue on the newest weights.
"""

from typing import TYPE_CHECKING, Any

from laminar.multi_version_rollout.repack import (
    CommittedAbortSource,
    CommittedRepackGroup,
    RepackPlanResult,
)

if TYPE_CHECKING:
    from laminar.multi_version_rollout.manager import (
        MultiVersionRolloutManager,
        MultiVersionRolloutManagerConfig,
        MultiVersionRolloutRepackConfig,
    )


def __getattr__(name: str) -> Any:
    # Standalone evaluation uses Router's repack plan/fence without loading
    # the training manager or its Relay/UCX transport dependencies.
    if name in {"MultiVersionRolloutManager", "MultiVersionRolloutManagerConfig", "MultiVersionRolloutRepackConfig"}:
        from laminar.multi_version_rollout import manager

        return getattr(manager, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CommittedAbortSource",
    "CommittedRepackGroup",
    "MultiVersionRolloutManager",
    "MultiVersionRolloutManagerConfig",
    "MultiVersionRolloutRepackConfig",
    "RepackPlanResult",
]
