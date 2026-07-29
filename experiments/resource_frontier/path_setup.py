# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Expose fixed_resources TFIM helpers and single_gate runtime on ``sys.path``."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = PACKAGE_DIR.parent
FIXED_RESOURCES_DIR = EXPERIMENTS_DIR / "fixed_resources"
SINGLE_GATE_DIR = EXPERIMENTS_DIR / "single_gate"
REPO_ROOT = EXPERIMENTS_DIR.parent

for path in (FIXED_RESOURCES_DIR, SINGLE_GATE_DIR):
    sp = str(path)
    if sp not in sys.path:
        sys.path.append(sp)
