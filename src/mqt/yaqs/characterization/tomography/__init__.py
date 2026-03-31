# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tomography module for YAQS (process tomography public API)."""

from .basis import build_basis_for_fixed_alphabet, intervention_from_alpha
from .combs import DenseComb, MPOComb, NNComb
from .estimator_class import TomographyEstimate
from .ml_dataset import build_rho_prev_rho_target
from .surrogates import (
    TrajectoryCombSample,
    mean_frobenius_mse_rho8,
    mean_trace_distance_rho8,
    trajectory_batch_to_tensors,
)
from .process_tomography import run, simulate_backend_labels, simulate_backend_trajectory_batch
from .predictor_encoding import (
    CHOI_FLAT_DIM,
    build_choi_feature_table,
    concat_choi_features,
    pack_rho8,
    random_density_matrix,
    state_prep_map_from_rho,
    unpack_rho8,
)
from .tomography_utils import build_initial_psi, make_mcwf_static_context
from .surrogates import TransformerComb

__all__ = [
    "CHOI_FLAT_DIM",
    "DenseComb",
    "MPOComb",
    "NNComb",
    "TrajectoryCombSample",
    "TomographyEstimate",
    "build_basis_for_fixed_alphabet",
    "build_choi_feature_table",
    "build_initial_psi",
    "build_rho_prev_rho_target",
    "concat_choi_features",
    "intervention_from_alpha",
    "make_mcwf_static_context",
    "mean_frobenius_mse_rho8",
    "mean_trace_distance_rho8",
    "pack_rho8",
    "random_density_matrix",
    "run",
    "simulate_backend_labels",
    "simulate_backend_trajectory_batch",
    "state_prep_map_from_rho",
    "trajectory_batch_to_tensors",
    "TransformerComb",
    "unpack_rho8",
]
