# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Expose shared single-gate runtime helpers on ``sys.path``."""

from __future__ import annotations

import sys
from pathlib import Path

SINGLE_GATE_DIR = Path(__file__).resolve().parents[1] / "single_gate"
if str(SINGLE_GATE_DIR) not in sys.path:
    sys.path.append(str(SINGLE_GATE_DIR))

REPO_ROOT = Path(__file__).resolve().parents[2]
