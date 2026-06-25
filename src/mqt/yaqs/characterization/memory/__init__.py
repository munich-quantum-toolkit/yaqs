# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Non-Markovian memory characterization (split-cut diagnostics and reference combs)."""

from __future__ import annotations

from .diagnostics.probe import probe_process
from .diagnostics.results import ProbeResult

__all__ = [
    "ProbeResult",
    "probe_process",
]
