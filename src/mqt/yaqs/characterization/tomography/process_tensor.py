from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qiskit.quantum_info import DensityMatrix, entropy

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _vec_to_rho(vec4: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Convert a 4-element vector to a 2x2 density matrix."""
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
        tensor (NDArray): The raw tensor data.
        timesteps (list[float]): The time points where interventions/measurements occurred.
    """

    def __init__(self, tensor: NDArray[np.complex128], timesteps: list[float]) -> None:
        self.tensor = tensor
        self.data = tensor
        self.timesteps = timesteps
        self.rank = len(tensor.shape) // 2

    def to_choi_matrix(self) -> NDArray[np.complex128]:
        """Convert to matrix view (final output vs all inputs).

        Returns matrix of shape (4, N^k) where:
        - 4: final output (vectorized 2x2 density matrix)
        - N: number of frame states per input slot
        - k: number of steps

        For Pauli frame (N=6), k=2 gives shape (4, 36).
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
        process tensor to predict the final output state.

        Args:
            rho_sequence: List of density matrices (2x2), one per time step
            duals: Dual frame matrices from tomography (same as used in reconstruction)

        Returns:
            Predicted final density matrix (2x2)

        Example:
            >>> # Build PT from tomography
            >>> pt = run_process_tensor_tomography(...)
            >>> # Predict on new sequence
            >>> rho0 = np.array([[0.5, 0.5], [0.5, 0.5]])  # |+⟩⟨+|
            >>> rho1 = np.array([[1, 0], [0, 0]])  # |0⟩⟨0|
            >>> duals = _calculate_dual_frame(_get_pauli_basis_set())
            >>> predicted = pt.predict_final_state([rho0, rho1], duals)
        """
        if len(rho_sequence) != len(self.timesteps):
            msg = f"Sequence length {len(rho_sequence)} must match number of timesteps {len(self.timesteps)}"
            raise ValueError(msg)

        # Compute coefficients: c_k = Tr(D_k† ρ) for each input state
        coeffs_per_step = []
        for rho in rho_sequence:
            coeffs = [np.trace(d.conj().T @ rho) for d in duals]
            coeffs_per_step.append(coeffs)

        # Contract with tensor: ρ_pred = Σ_{i0,...,ik} c_i0^(0) ... c_ik^(k) T[:,i0,...,ik]
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
        """Calculate the Holevo information of the process tensor.

        Args:
            p: Optional dictionary mapping sequence tuples to probabilities.
               If None, assumes uniform distribution over all input sequences.
            base: Logarithm base for entropy calculation (default 2 for bits).

        Returns:
            float: The Holevo information.
        """
        import itertools

        # Tensor shape: (4, N, N, ..., N)
        # N is number of basis states (6 for Pauli)
        # k is number of steps (input slots)
        out_dim = self.tensor.shape[0]
        assert out_dim == 4
        N = self.tensor.shape[1]
        k = self.tensor.ndim - 1

        # Generate all input sequences
        seqs = list(itertools.product(range(N), repeat=k))

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
        # S(rho_avg)
        S_avg = entropy(DensityMatrix(rho_avg), base=base)

        # sum p_i S(rho_i)
        S_cond = 0.0
        for seq in seqs:
            S_cond += p[seq] * entropy(DensityMatrix(rhos[seq]), base=base)

        return S_avg - S_cond

    def holevo_information_conditional(self, fixed_step: int, fixed_idx: int, base: int = 2) -> float:
        """Calculate the Holevo information conditioned on a fixed input at a specific step.

        Args:
            fixed_step: The time step index to fix (0-indexed).
            fixed_idx: The basis state index to fix at that step.
            base: Logarithm base for entropy calculation (default 2 for bits).

        Returns:
            float: The conditional Holevo information.
        """
        import itertools

        out_dim = self.tensor.shape[0]
        assert out_dim == 4
        N = self.tensor.shape[1]
        k = self.tensor.ndim - 1

        if fixed_step < 0 or fixed_step >= k:
            msg = f"fixed_step {fixed_step} out of bounds for {k} steps."
            raise ValueError(msg)
        if fixed_idx < 0 or fixed_idx >= N:
            msg = f"fixed_idx {fixed_idx} out of bounds for {N} basis states."
            raise ValueError(msg)

        # Generate all sequences of length k where seq[fixed_step] == fixed_idx
        # We can optimize this by only generating the varying parts, but filtering is easier to write clearly
        seqs = [seq for seq in itertools.product(range(N), repeat=k) if seq[fixed_step] == fixed_idx]

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
        S_avg = entropy(DensityMatrix(rho_avg), base=base)

        S_cond = 0.0
        for seq in seqs:
            S_cond += p[seq] * entropy(DensityMatrix(rhos[seq]), base=base)

        return S_avg - S_cond
