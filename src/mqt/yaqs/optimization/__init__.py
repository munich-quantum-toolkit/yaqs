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
    KrotovNoiseMap,
    KrotovOptions,
    KrotovReadout,
    KrotovResult,
    KrotovTJMOptions,
    KrotovTrajectory,
    KrotovTruncation,
    empirical_loss,
    noisy_sample_contribution,
    noisy_sample_loss,
    noisy_state_preparation_contribution,
    noisy_state_preparation_cross_contribution,
    noisy_state_preparation_loss,
    noisy_state_preparation_metrics,
    sample_contribution,
    state_preparation_contribution,
    state_preparation_loss,
    state_preparation_metrics,
    state_preparation_terminal_costate,
    train_krotov_batch,
    train_krotov_hybrid,
    train_krotov_noisy_state_preparation_batch,
    train_krotov_noisy_state_preparation_hybrid,
    train_krotov_noisy_state_preparation_online,
    train_krotov_online,
    train_krotov_state_preparation_batch,
    train_krotov_state_preparation_hybrid,
    train_krotov_state_preparation_online,
)
from .parameterized_circuit import (
    ParameterizedCircuit,
    ParameterizedGate,
    brickwall_matrix_product_disentangler_num_parameters,
    create_brickwall_matrix_product_disentangler_parameterized_circuit,
    create_sequential_matrix_product_disentangler_parameterized_circuit,
)

__all__ = [
    "KrotovNoiseMap",
    "KrotovOptions",
    "KrotovReadout",
    "KrotovResult",
    "KrotovTJMOptions",
    "KrotovTrajectory",
    "KrotovTruncation",
    "ParameterizedCircuit",
    "ParameterizedGate",
    "brickwall_matrix_product_disentangler_num_parameters",
    "create_brickwall_matrix_product_disentangler_parameterized_circuit",
    "create_sequential_matrix_product_disentangler_parameterized_circuit",
    "empirical_loss",
    "noisy_sample_contribution",
    "noisy_sample_loss",
    "noisy_state_preparation_contribution",
    "noisy_state_preparation_cross_contribution",
    "noisy_state_preparation_loss",
    "noisy_state_preparation_metrics",
    "sample_contribution",
    "state_preparation_contribution",
    "state_preparation_loss",
    "state_preparation_metrics",
    "state_preparation_terminal_costate",
    "train_krotov_batch",
    "train_krotov_hybrid",
    "train_krotov_noisy_state_preparation_batch",
    "train_krotov_noisy_state_preparation_hybrid",
    "train_krotov_noisy_state_preparation_online",
    "train_krotov_online",
    "train_krotov_state_preparation_batch",
    "train_krotov_state_preparation_hybrid",
    "train_krotov_state_preparation_online",
]
