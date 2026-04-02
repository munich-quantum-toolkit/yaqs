from __future__ import annotations

import numpy as np

from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.characterization.tomography.estimate.basis import (
    calculate_dual_choi_basis,
    get_basis_states,
    get_choi_basis,
    _finalize_sequence_averages,
)
from mqt.yaqs.characterization.tomography.core.metrics import rel_fro_error
from mqt.yaqs.tomography import run_exhaustive


def test_choi_duality_biorthogonality() -> None:
    """Verify dual frame biorthogonality: Tr(D_i^† B_j) = δ_ij."""
    choi_basis, _ = get_choi_basis()
    duals = calculate_dual_choi_basis(choi_basis)

    for i, d_i in enumerate(duals):
        for j, b_j in enumerate(choi_basis):
            inner = np.trace(d_i.conj().T @ b_j)
            expected = 1.0 if i == j else 0.0
            np.testing.assert_allclose(inner, expected, atol=1e-10)


def test_reconstruction_identity_random_choi() -> None:
    """Verify Choi matrix reconstruction: J = Σ_k Tr(D_k^† J) B_k."""
    rng = np.random.default_rng(42)
    choi_basis, _ = get_choi_basis()
    duals = calculate_dual_choi_basis(choi_basis)

    j_rand = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    coeffs = np.array([np.trace(d.conj().T @ j_rand) for d in duals])
    j_rec = np.zeros((4, 4), dtype=complex)
    for c, b in zip(coeffs, choi_basis, strict=False):
        j_rec += c * b

    np.testing.assert_allclose(j_rec, j_rand, atol=1e-10)


def test_dual_extracts_one_hot_for_basis_maps() -> None:
    """Verify duals extract one-hot coefficients for basis maps under the Choi convention."""
    # Use the same basis label for states and Choi matrices (defaults differ per helper).
    basis = get_basis_states(basis="standard")
    choi_basis, choi_indices = get_choi_basis(basis="standard")
    duals = calculate_dual_choi_basis(choi_basis)

    for alpha in range(16):
        p, m = choi_indices[alpha]
        rho_p = basis[p][2]
        e_m = basis[m][2]

        def a_alpha(rho, e_m_sub=e_m, rho_p_sub=rho_p):
            return np.trace(e_m_sub @ rho) * rho_p_sub

        j_choi = np.zeros((4, 4), dtype=complex)
        for i in range(2):
            for j in range(2):
                e = np.zeros((2, 2), dtype=complex)
                e[i, j] = 1.0
                j_choi += np.kron(a_alpha(e), e)

        c = np.array([np.trace(d.conj().T @ j_choi) for d in duals])
        expected = np.zeros(16, dtype=complex)
        expected[alpha] = 1.0
        np.testing.assert_allclose(c, expected, atol=1e-10)


def test_finalize_sequence_averages_basic() -> None:
    """Smoke-test _finalize_sequence_averages normalization logic."""
    seq = (0,)
    rho = np.eye(2, dtype=np.complex128)
    weight_sum = 2.0
    count = 2
    acc = {seq: [rho * weight_sum, weight_sum, count]}
    final_seqs, outputs, weights = _finalize_sequence_averages(acc, weight_scale=1.0)
    assert final_seqs == [seq]
    np.testing.assert_allclose(outputs[0], np.eye(2, dtype=np.complex128))
    assert weights == [weight_sum]


def test_basis_reproduction_h0_identity_map() -> None:
    """End-to-end sanity: identity map yields correct prediction for H=0."""
    op = MPO.ising(length=2, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=16)
    comb = run_exhaustive(op, params, timesteps=[0.1], output="dense")

    def identity_map(rho: np.ndarray) -> np.ndarray:
        return rho

    rho_pred = comb.predict([identity_map])
    expected = np.array([[1.0, 0.0], [0.0, 0.0]])
    assert rel_fro_error(rho_pred, expected) < 1e-10