# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Circuit optimization methods for MQT YAQS.

This package contains the Krotov-inspired discrete adjoint optimizer for
parameterized quantum circuits on the MPS/MPO backend, together with the
gate-list circuit representation it operates on.
"""

from __future__ import annotations

from .krotov import (
    KrotovOptions,
    KrotovReadout,
    KrotovResult,
    KrotovTruncation,
    empirical_loss,
    sample_contribution,
    train_krotov_batch,
    train_krotov_hybrid,
    train_krotov_online,
)
from .parameterized_circuit import ParameterizedCircuit, ParameterizedGate

__all__ = [
    "KrotovOptions",
    "KrotovReadout",
    "KrotovResult",
    "KrotovTruncation",
    "ParameterizedCircuit",
    "ParameterizedGate",
    "empirical_loss",
    "sample_contribution",
    "train_krotov_batch",
    "train_krotov_hybrid",
    "train_krotov_online",
]
