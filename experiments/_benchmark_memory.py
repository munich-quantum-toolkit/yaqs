# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact-rollout memory metrics for paper benchmark scripts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mqt.yaqs.characterization.memory.backends.exact import simulate_exact
from mqt.yaqs.characterization.memory.operational_memory.memory_matrix import (
    assemble_memory_matrix,
    compute_spectrum,
)

if TYPE_CHECKING:
    from mqt.yaqs.characterization.memory.operational_memory.samples import ProbeSet
    from mqt.yaqs.core.data_structures.mpo import MPO
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def delta_norm_of_centered(memory_matrix_raw: np.ndarray, memory_matrix: np.ndarray) -> float:
    """Ratio of squared Frobenius norms of centered to raw memory matrix."""
    fro_raw_sq = float(np.linalg.norm(memory_matrix_raw, ord="fro") ** 2)
    fro_c_sq = float(np.linalg.norm(memory_matrix, ord="fro") ** 2)
    return float(fro_c_sq / fro_raw_sq) if fro_raw_sq > 0.0 else 0.0


def entropy_from_singular_values(s: np.ndarray) -> float:
    """Shannon entropy of normalized squared singular values."""
    p = np.asarray(s, dtype=np.float64) ** 2
    ps = float(np.sum(p))
    if ps <= 0.0:
        return 0.0
    q = np.clip(p / ps, 1e-30, 1.0)
    return float(-np.sum(q * np.log(q)))


def linear_weighted_metrics(
    *,
    probe_set: ProbeSet,
    op: MPO,
    sim_params: AnalogSimParams,
    psi0: np.ndarray,
    parallel: bool,
    branch_weight_beta: float = 1.0,
    psi_pairs_list: list[list[Any]] | None = None,
) -> dict[str, float | int]:
    """Past-centered S_V with weighted memory matrix using mc-process branch weights."""
    pauli_xyz_ij, _, traces = simulate_exact(
        probe_set=probe_set,
        operator=op,
        sim_params=sim_params,
        initial_psi=psi0,
        parallel=parallel,
        psi_pairs_list=psi_pairs_list,
    )
    n_p, n_f = pauli_xyz_ij.shape[:2]
    weights_ij = np.zeros((n_p, n_f), dtype=np.float64)
    for ii in range(n_p):
        for jj in range(n_f):
            weights_ij[ii, jj] = float(traces[ii * n_f + jj]["cumulative_weight_final"])
    m_raw, memory_matrix = assemble_memory_matrix(
        pauli_xyz_ij,
        weights_ij,
        beta=branch_weight_beta,
        log_weight_warnings=False,
    )
    ana = compute_spectrum(memory_matrix)
    return {
        "entropy": float(ana["entropy"]),
        "delta_norm": float(delta_norm_of_centered(m_raw, memory_matrix)),
        "rank": float(ana["rank"]),
        "weight_min": float(np.min(weights_ij)),
        "weight_max": float(np.max(weights_ij)),
    }


def weighted_centered_singular_values(
    *,
    probe_set: ProbeSet,
    op: MPO,
    sim_params: AnalogSimParams,
    psi0: np.ndarray,
    parallel: bool,
    branch_weight_beta: float = 1.0,
) -> np.ndarray:
    """Singular values of the past-centered weighted memory matrix from exact rollouts."""
    pauli_xyz_ij, weights_ij, _ = simulate_exact(
        probe_set=probe_set,
        operator=op,
        sim_params=sim_params,
        initial_psi=psi0,
        parallel=parallel,
    )
    _m_raw, memory_matrix = assemble_memory_matrix(
        pauli_xyz_ij,
        weights_ij,
        beta=branch_weight_beta,
        log_weight_warnings=False,
    )
    return np.linalg.svd(np.asarray(memory_matrix, dtype=np.float64), compute_uv=False).astype(np.float64)
