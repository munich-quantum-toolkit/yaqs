# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Restricted Process Tensor representation."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
from qiskit.quantum_info import DensityMatrix, entropy

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _vec_to_rho(vec4: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Convert a 4-element vector to a 2x2 density matrix.

    Args:
        vec4: A 4-element vector.

    Returns:
        A 2x2 density matrix.
    """
    assert len(vec4) == 4, "Vector must have 4 elements"
    rho = vec4.reshape(2, 2)
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho)
    if abs(tr) > 1e-15:
        rho /= tr
    return rho


class ProcessTensor:
    """Class to represent a Process Tensor.

    A Process Tensor generalizes the concept of a quantum channel to multiple time steps,
    capturing non-Markovian memory effects.

    Attributes:
        tensor (NDArray): The raw tensor data of shape (4, N, ..., N).
        weights (NDArray): The probabilities of each sequence of shape (N, ..., N).
        timesteps (list[float]): The time points where interventions/measurements occurred.
    """

    def __init__(
        self, tensor: NDArray[np.complex128], weights: NDArray[np.float64], timesteps: list[float]
    ) -> None:
        """Initialize the ProcessTensor.

        Args:
            tensor: The raw tensor data.
            weights: The probabilities of each sequence.
            timesteps: The time points where interventions/measurements occurred.
        """
        self.tensor = tensor
        self.weights = weights
        self.timesteps = timesteps

    def to_choi_matrix(self) -> NDArray[np.complex128]:
        """Convert to matrix view (final output vs all inputs).

        Returns matrix of shape (4, N^k) where:
        - 4: final output (vectorized 2x2 density matrix)
        - N: number of frame states per input slot
        - k: number of steps

        For Pauli frame (N=6), k=2 gives shape (4, 36).

        Returns:
            NDArray[np.complex128]: Matrix of shape (4, N^k).
        """
        # Shape is [4, N, N, ...] -> reshape to [4, N^k]
        num_inputs = np.prod(self.tensor.shape[1:])  # Product of all input dimensions
        return self.tensor.reshape(4, num_inputs)

    def predict_final_state(
        self,
        interventions: list[typing.Callable[[NDArray[np.complex128]], NDArray[np.complex128]]],
        duals: list[NDArray[np.complex128]],
    ) -> NDArray[np.complex128]:
        """Predict final state using dual-frame contraction for arbitrary interventions.

        Args:
            interventions: A list of callables representing CPTP maps for each intervention step.
                           The first callable is the initial state preparation at t=0.
            duals: Dual frame matrices from tomography.

        Returns:
            Predicted final density matrix (2x2)

        Raises:
            ValueError: If the number of interventions does not match num_steps.
        """
        import typing
        k_steps = len(self.timesteps)
        if len(interventions) != k_steps:
            msg = f"Expected {k_steps} interventions (including t=0 prep), got {len(interventions)}."
            raise ValueError(msg)

        # Precompute the Choi matrices and their projection onto the dual basis.
        # For a CP map E(\\rho), its Choi matrix is J(E) = sum_{i,j} E(|i><j|) \\otimes |i><j|^T
        # which in our basis choice maps directly to \\rho_p \\otimes E_m^T.
        
        c_maps = []
        for emap in interventions:
            J = np.zeros((4, 4), dtype=complex)
            for i in range(2):
                for j in range(2):
                    e_in = np.zeros((2, 2), dtype=complex)
                    e_in[i, j] = 1.0
                    rho_out = emap(e_in)
                    J += np.kron(rho_out, e_in)

            # Project onto duals: c_a = Tr(D_a^dag J)
            c_a = np.array([np.trace(d.conj().T @ J) for d in duals])
            c_maps.append(c_a)

        # Tensor contraction
        # self.tensor has shape (4, 16, 16, ..., 16).
        # We want to contract the k_steps indices (dimensions 1 to k_steps) with the c_maps coefficients.
        result_tensor = self.tensor
        for step in reversed(range(k_steps)):
            # Multiply and sum out the last axis (axis -1) with c_maps[step]
            result_tensor = np.tensordot(result_tensor, c_maps[step], axes=([-1], [0]))

        return result_tensor.reshape(2, 2)

    def quantum_mutual_information(self, base: int = 2) -> float:
        """Calculate the Quantum Mutual Information between the input sequence and the output state.

        I(A:B) = S(B) - sum_a p(a) S(B|a)
        where A is the sequence distribution, B is the output state.

        Args:
            base: Logarithm base for entropy calculation (default 2 for bits).

        Returns:
            float: The Quantum Mutual Information.

        Raises:
            ValueError: If the output dimension is not 4.
        """
        out_dim = self.tensor.shape[0]
        if out_dim != 4:
            msg = f"Expected output dimension 4, got {out_dim}."
            raise ValueError(msg)

        steps_k = self.tensor.ndim - 1
        if steps_k == 0:
            return 0.0

        states_n = self.tensor.shape[1]
        seqs = list(itertools.product(range(states_n), repeat=steps_k))

        # Build ensemble and average output
        rhos = {}
        rho_avg = np.zeros((2, 2), dtype=np.complex128)

        # Normalize weights if they are provided
        total_weight = np.sum(self.weights)
        if total_weight > 0:
            norm_weights = self.weights / total_weight
        else:
            norm_weights = np.ones_like(self.weights) / len(seqs)

        for seq in seqs:
            vec = self.tensor[(slice(None), *seq)]
            rho = _vec_to_rho(vec)
            rhos[seq] = rho
            rho_avg += norm_weights[seq] * rho

        # Compute Entropies
        if np.trace(rho_avg) > 1e-12:
            entropy_b = entropy(DensityMatrix(rho_avg), base=base)
        else:
            entropy_b = 0.0

        entropy_b_given_a = 0.0
        for seq in seqs:
            if norm_weights[seq] > 1e-12 and np.trace(rhos[seq]) > 1e-12:
                entropy_b_given_a += norm_weights[seq] * entropy(DensityMatrix(rhos[seq]), base=base)

        return entropy_b - entropy_b_given_a
