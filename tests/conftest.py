# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Make ``import laminar`` and test helpers work without an editable install."""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
_SRC = _TESTS.parent / "src"
for _path in (_SRC, _TESTS):
    _resolved = str(_path)
    if _resolved not in sys.path:
        sys.path.insert(0, _resolved)
