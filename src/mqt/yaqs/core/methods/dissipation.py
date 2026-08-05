# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Dissipative sweep of the Tensor Jump Method.

This module implements a function to apply dissipation to a quantum state represented as an MPS.
The dissipative operator is computed from a noise model by exponentiating a weighted sum of jump operators,
and is then applied to each tensor in the MPS via tensor contraction. If no noise is present or if all
noise strengths are zero, the MPS is simply shifted to its canonical form.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import opt_einsum as oe

from .. import linalg
from ..data_structures.noise_model import is_pauli
from ..methods.decompositions import merge_two_site, split_two_site

if TYPE_CHECKING:
    from ..data_structures.mps import MPS
    from ..data_structures.noise_model import NoiseModel
    from ..data_structures.simulation_parameters import AnalogSimParams, DigitalSimParams
    from ..methods.decompositions import TruncMode

__all__ = ["apply_dissipation", "is_adjacent", "is_longrange", "is_pauli"]


def is_adjacent(proc: dict[str, Any]) -> bool:
    """Return True if the two-site process targets nearest neighbors.

    Assumes the process is two-site and checks ``|i-j| == 1``.
    """
    s = proc["sites"]
    return bool(abs(s[1] - s[0]) == 1)


def is_longrange(proc: dict[str, Any]) -> bool:
    """Return True if the two-site process is long-range (non-neighbor)."""
    s = proc["sites"]
    return bool(abs(s[1] - s[0]) > 1)


def apply_dissipation(
    state: MPS,
    noise_model: NoiseModel | None,
    dt: float,
    sim_params: AnalogSimParams | DigitalSimParams,
) -> None:
    """Apply dissipation to the system state using a given noise model and time step.

    This function modifies the state tensors of an MPS by applying a dissipative operator
    that is calculated from the noise model's jump operators and strengths. The operator is
    computed by exponentiating a matrix derived from these jump operators, and then applied to
    each tensor in the state using an Einstein summation contraction.

    Args:
        state: The Matrix Product State (MPS) representing the current state of the system.
        noise_model: The noise model containing jump operators and their
            corresponding strengths. If None or if all strengths are zero, no dissipation is applied.
        dt: The time step for the evolution, used in the exponentiation of the dissipative operator.
        sim_params: Simulation parameters that include settings.

    Notes:
        - If no noise is present (i.e. `noise_model` is None or all noise strengths are zero),
          the function shifts the orthogonality center of the MPS tensors and returns early.
        - The dissipation operator A is calculated as a sum over each jump operator, where each
          term is given by (noise strength) * (conjugate transpose of the jump operator) multiplied
          by the jump operator.
        - The dissipative operator is computed using the matrix exponential `expm(-0.5 * dt * A)`.
        - The operator is then applied to each tensor in the MPS via a contraction using `opt_einsum`.
    """
    if noise_model is None or all(proc["strength"] == 0 for proc in noise_model.processes):
        if state.orthogonality_center is not None:
            if state.orthogonality_center != 0:
                state.shift_center_to(0, decomposition="QR")
            state.shift_orthogonality_center_left(0, decomposition="QR")
        else:
            state.set_canonical_form(0, decomposition="QR")
        return

    if state.orthogonality_center is not None:
        if state.orthogonality_center != state.length - 1:
            state.shift_center_to(state.length - 1, decomposition="SVD")
    else:
        state.set_canonical_form(state.length - 1, decomposition="SVD")

    for i in reversed(range(state.length)):
        # 1. One-site dissipators on site i: expm(-0.5 dt * sum_k gamma_k L_k^dagger L_k)
        one_site = [
            process for process in noise_model.processes if len(process["sites"]) == 1 and process["sites"][0] == i
        ]
        if one_site:
            dim = state.physical_dimensions[i]
            generator = np.zeros((dim, dim), dtype=np.complex128)
            for process in one_site:
                gamma = process["strength"]
                if is_pauli(process):
                    generator += gamma * np.eye(dim, dtype=np.complex128)
                else:
                    jump_op_mat = process["matrix"]
                    generator += gamma * (np.conj(jump_op_mat).T @ jump_op_mat)
            if all(is_pauli(process) for process in one_site):
                state.tensors[i] *= np.exp(-0.5 * dt * float(np.real(generator[0, 0])))
            else:
                dissipative_op = linalg.expm(-0.5 * dt * generator)
                state.tensors[i] = oe.contract("ab, bcd->acd", dissipative_op, state.tensors[i])

        processes_here = [
            process for process in noise_model.processes if len(process["sites"]) == 2 and process["sites"][1] == i
        ]
        # 2. Two-site dissipators on (i-1, i): sum generators, one expm (Pauli long-range: scalar)
        if i != 0 and processes_here:
            longrange_procs = [process for process in processes_here if is_longrange(process)]
            adjacent_procs = [process for process in processes_here if not is_longrange(process)]

            for process in longrange_procs:
                if not is_pauli(process):
                    msg = "Non-Pauli Long-range processes are not implemented yet"
                    raise NotImplementedError(msg)
                state.tensors[i] *= np.exp(-0.5 * dt * process["strength"])

            if adjacent_procs:
                dim_left = state.physical_dimensions[i - 1]
                dim_right = state.physical_dimensions[i]
                dim = dim_left * dim_right
                generator = np.zeros((dim, dim), dtype=np.complex128)
                for process in adjacent_procs:
                    gamma = process["strength"]
                    if is_pauli(process):
                        generator += gamma * np.eye(dim, dtype=np.complex128)
                    else:
                        jump_op_mat = process["matrix"]
                        generator += gamma * (np.conj(jump_op_mat).T @ jump_op_mat)
                if all(is_pauli(process) for process in adjacent_procs):
                    state.tensors[i] *= np.exp(-0.5 * dt * float(np.real(generator[0, 0])))
                else:
                    dissipative_op = linalg.expm(-0.5 * dt * generator)
                    merged_tensor = merge_two_site(state.tensors[i - 1], state.tensors[i])
                    merged_tensor = oe.contract("ab, bcd->acd", dissipative_op, merged_tensor)
                    tensor_left, tensor_right = split_two_site(
                        merged_tensor,
                        [dim_left, dim_right],
                        svd_distribution="right",
                        trunc_mode=cast("TruncMode", sim_params.trunc_mode),
                        threshold=sim_params.svd_threshold,
                        max_bond_dim=sim_params.max_bond_dim,
                    )
                    state.tensors[i - 1], state.tensors[i] = tensor_left, tensor_right
                    state.update_center_after_split(i - 1, i, "right")

        # Shift orthogonality center
        if i != 0:
            if state.orthogonality_center is not None:
                if state.orthogonality_center != i:
                    state.shift_center_to(i, decomposition="SVD")
                state.shift_orthogonality_center_left(i, decomposition="SVD")
            else:
                state.set_canonical_form(i, decomposition="SVD")
                state.shift_orthogonality_center_left(i, decomposition="SVD")

    state.set_center(0)
