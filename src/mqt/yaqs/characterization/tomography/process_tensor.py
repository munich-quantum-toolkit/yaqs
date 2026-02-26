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
        initial_state: NDArray[np.complex128],
        interventions: list[typing.Callable[[NDArray[np.complex128]], NDArray[np.complex128]]],
        duals: list[NDArray[np.complex128]],
    ) -> NDArray[np.complex128]:
        """Predict final state using dual-frame contraction for arbitrary interventions.

        Args:
            initial_state: The initial density matrix (2x2).
            interventions: A list of callables representing CPTP maps for each intervention step.
            duals: Dual frame matrices from tomography.

        Returns:
            Predicted final density matrix (2x2)

        Raises:
            ValueError: If the number of interventions does not match num_steps - 1.
        """
        import typing
        k_steps = len(self.timesteps)
        if len(interventions) != k_steps - 1:
            msg = f"Expected {k_steps - 1} interventions, got {len(interventions)}."
            raise ValueError(msg)

        # Coefficients for initial state: c_0 = Tr(D_m^DAG rho_0)
        c_0 = [np.trace(d.conj().T @ initial_state) for d in duals]

        # Coefficients for each intervention map
        # c^{(s)}_{m, p} = Tr(D_p^DAG E_s(D_m))
        c_maps = []
        for emap in interventions:
            cmap = np.zeros((len(duals), len(duals)), dtype=complex)
            for m, dm in enumerate(duals):
                edm = emap(dm)
                for p, dp in enumerate(duals):
                    cmap[m, p] = np.trace(dp.conj().T @ edm)
            c_maps.append(cmap)

        result = np.zeros(4, dtype=complex)
        for idx in np.ndindex(*self.tensor.shape[1:]):
            coeff = c_0[idx[0]]
            for step in range(len(interventions)):
                m = idx[2 * step + 1]
                p = idx[2 * step + 2]
                coeff *= c_maps[step][m, p]
            result += coeff * self.tensor[(slice(None), *idx)]

        return result.reshape(2, 2)

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
        entropy_b = entropy(DensityMatrix(rho_avg), base=base)

        entropy_b_given_a = 0.0
        for seq in seqs:
            entropy_b_given_a += norm_weights[seq] * entropy(DensityMatrix(rhos[seq]), base=base)

        return entropy_b - entropy_b_given_a
