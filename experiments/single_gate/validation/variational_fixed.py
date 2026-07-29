# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Compatibility shim: corrected variational path now lives in ``variational.py``."""

from __future__ import annotations

from variational import (  # noqa: F401
    VariationalResult,
    _apply_gate_mpo,
    apply_variational_mpo_gate,
    tt_svd_from_vec,
)

# Older validation notes imported this name.
apply_variational_mpo_gate_multistart = apply_variational_mpo_gate
