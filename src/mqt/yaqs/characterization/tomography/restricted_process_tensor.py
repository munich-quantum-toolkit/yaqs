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


class RestrictedProcessTensor:
    """Class to represent a Restricted Process Tensor.

    A Restricted Process Tensor generalizes the concept of a quantum channel to multiple time steps,
    capturing non-Markovian memory effects.

    Attributes:
        tensor (NDArray): The raw tensor data.
        timesteps (list[float]): The time points where interventions/measurements occurred.
    """

    def __init__(self, tensor: NDArray[np.complex128], weights: NDArray[np.float64], timesteps: list[float]) -> None:
        """Initialize the RestrictedProcessTensor.

        Args:
            tensor: The raw tensor data.
            weights: The initial state preparation weights.
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
        self, rho_sequence: list[NDArray[np.complex128]], duals: list[NDArray[np.complex128]]
    ) -> NDArray[np.complex128]:
        """Predict final state for a sequence of input states using dual-frame contraction.

        This method enables prediction on held-out state sequences not used during tomography.
        It computes dual-frame coefficients for each input state and contracts them with the
        restricted process tensor to predict the final output state.

        Args:
            rho_sequence: List of density matrices (2x2), one per time step
            duals: Dual frame matrices from tomography (same as used in reconstruction)

        Returns:
            Predicted final density matrix (2x2)

        Raises:
            ValueError: If the sequence length does not match the number of timesteps.

        Example:
            >>> # Build PT from tomography
            >>> pt = run_restricted_process_tensor_tomography(...)
            >>> # Predict on new sequence
            >>> rho0 = np.array([[0.5, 0.5], [0.5, 0.5]])  # |+⟩⟨+|
            >>> rho1 = np.array([[1, 0], [0, 0]])  # |0⟩⟨0|
            >>> duals = calculate_dual_frame(_get_pauli_basis_set())
            >>> predicted = pt.predict_final_state([rho0, rho1], duals)
        """
        if len(rho_sequence) != len(self.timesteps):
            msg = f"Sequence length {len(rho_sequence)} must match number of timesteps {len(self.timesteps)}"
            raise ValueError(msg)

        # Compute coefficients: c_k = Tr(D_k† rho) for each input state
        coeffs_per_step = []
        for rho in rho_sequence:
            coeffs = [np.trace(d.conj().T @ rho) for d in duals]
            coeffs_per_step.append(coeffs)

        # Contract with tensor: rho_pred = sum_{i0,...,ik} c_i0^(0) ... c_ik^(k) T[:,i0,...,ik]
        result = np.zeros(4, dtype=complex)
        for idx in np.ndindex(*self.tensor.shape[1:]):  # Iterate over all input frame indices
            # Compute product of coefficients for this index combination
            coeff = 1.0
            for step, frame_idx in enumerate(idx):
                coeff *= coeffs_per_step[step][frame_idx]
            # Add weighted contribution from this tensor element
            result += coeff * self.tensor[(slice(None), *idx)]

        return result.reshape(2, 2)

    def holevo_information(self, p: dict[tuple[int, ...], float] | None = None, base: int = 2) -> float:
        """Calculate the Holevo information of the restricted process tensor.

        Args:
            p: Optional dictionary mapping sequence tuples to probabilities.
               If None, assumes uniform distribution over all input sequences.
            base: Logarithm base for entropy calculation (default 2 for bits).

        Returns:
            float: The Holevo information.

        Raises:
            ValueError: If the output dimension is not 4.
        """
        # Tensor shape: (4, N, N, ..., N)
        # k is number of steps (input slots)
        steps_k = self.tensor.ndim - 1
        out_dim = self.tensor.shape[0]
        if out_dim != 4:
            msg = f"Expected output dimension 4, got {out_dim}."
            raise ValueError(msg)

        if steps_k == 0:
            # No inputs, just a single state
            return 0.0  # Information about inputs is zero if there are no inputs

        states_n = self.tensor.shape[1]

        # Generate all input sequences
        seqs = list(itertools.product(range(states_n), repeat=steps_k))

        if p is None:
            p = {seq: 1.0 / len(seqs) for seq in seqs}

        # Build ensemble and average output
        rhos = {}
        rho_avg = np.zeros((2, 2), dtype=np.complex128)

        for seq in seqs:
            # Extract output vector for this sequence
            vec = self.tensor[(slice(None), *seq)]  # shape (4,)
            rho = _vec_to_rho(vec)
            rhos[seq] = rho
            rho_avg += p[seq] * rho

        # Compute Entropies
        # Calculate entropy S(rho_avg)
        s_avg = entropy(DensityMatrix(rho_avg), base=base)

        # sum p_i S(rho_i)
        s_cond = 0.0
        for seq in seqs:
            s_cond += p[seq] * entropy(DensityMatrix(rhos[seq]), base=base)

        return s_avg - s_cond

    def holevo_information_conditional(self, fixed_step: int, fixed_idx: int, base: int = 2) -> float:
        """Calculate the Holevo information conditioned on a fixed input at a specific step.

        Args:
            fixed_step: The time step index to fix (0-indexed).
            fixed_idx: The basis state index to fix at that step.
            base: Logarithm base for entropy calculation (default 2 for bits).

        Returns:
            float: The conditional Holevo information.

        Raises:
            ValueError: If fixed_step or fixed_idx are out of bounds.
        """
        out_dim = self.tensor.shape[0]
        if out_dim != 4:
            msg = f"Expected output dimension 4, got {out_dim}."
            raise ValueError(msg)
        steps_k = self.tensor.ndim - 1

        if fixed_step < 0 or fixed_step >= steps_k:
            msg = f"fixed_step {fixed_step} out of bounds for {steps_k} steps."
            raise ValueError(msg)

        states_n = self.tensor.shape[1]
        if fixed_idx < 0 or fixed_idx >= states_n:
            msg = f"fixed_idx {fixed_idx} out of bounds for {states_n} basis states."
            raise ValueError(msg)

        # Generate all sequences of length steps_k where seq[fixed_step] == fixed_idx
        # We can optimize this by only generating the varying parts, but filtering is easier to write clearly
        seqs = [seq for seq in itertools.product(range(states_n), repeat=steps_k) if seq[fixed_step] == fixed_idx]

        if not seqs:
            return 0.0

        # Uniform probability over this subset
        p_val = 1.0 / len(seqs)
        p = dict.fromkeys(seqs, p_val)

        # Build ensemble and average output
        rhos = {}
        rho_avg = np.zeros((2, 2), dtype=np.complex128)

        for seq in seqs:
            vec = self.tensor[(slice(None), *seq)]
            rho = _vec_to_rho(vec)
            rhos[seq] = rho
            rho_avg += p[seq] * rho

        # Compute Entropies
        entropy_avg = entropy(DensityMatrix(rho_avg), base=base)

        entropy_cond = 0.0
        for seq in seqs:
            entropy_cond += p[seq] * entropy(DensityMatrix(rhos[seq]), base=base)

        return entropy_avg - entropy_cond
