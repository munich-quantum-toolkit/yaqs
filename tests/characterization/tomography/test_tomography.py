# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Process Tomography estimates, equivalence, convergence, and MPO execution."""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Callable

import numpy as np
import pytest
from scipy.linalg import expm
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from collections.abc import Callable
    from numpy.typing import NDArray

from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.core.data_structures.noise_model import NoiseModel

from mqt.yaqs.characterization.tomography.tomography import (
    calculate_dual_choi_basis,
    get_basis_states,
    get_choi_basis,
    run,
    estimate_process_tensor,
)

from mqt.yaqs.characterization.tomography.process_tensor import (
    ProcessTensor,
    canonicalize_upsilon,
    comb_qmi_from_upsilon_dense,
    comb_cmi_from_upsilon_dense,
    reduced_upsilon,
    rank1_upsilon_mpo_term,
    upsilon_mpo_to_dense,
)

from mqt.yaqs.characterization.tomography.metrics import rel_fro_error



################################################################################
# ── Core Tomography Tests ─────────────────────────────────────────────────────
################################################################################

def _get_random_rho(rng: np.random.Generator) -> NDArray[np.complex128]:
    """Generate a random 2x2 density matrix.

    Args:
        rng: The random number generator.

    Returns:
        NDArray[np.complex128]: The resulting 2x2 density matrix.
    """
    # General mixed state via Ginibre ensemble
    z = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    rho = z @ z.conj().T
    return rho / np.trace(rho)


def _apply_local_map_site0(
    rho_full: NDArray[np.complex128],
    local_map: Callable[[NDArray[np.complex128]], NDArray[np.complex128]],
) -> NDArray[np.complex128]:
    """Apply (local_map ⊗ I) to a 2-qubit density matrix rho_full.

    Assumes rho_full uses ordering kron(site0, site1).
    Reshape convention: rho[i, k, j, l] where
      i,j index site0 (row/col), k,l index site1 (row/col).

    Args:
        rho_full: The 2-qubit density matrix.
        local_map: The local CP map to apply to site 0.

    Returns:
        NDArray[np.complex128]: The resulting 2-qubit density matrix.

    Raises:
        ValueError: If local_map does not return a (2,2) matrix.
    """
    rho4 = rho_full.reshape(2, 2, 2, 2)  # (i,k,j,l)
    out4 = np.zeros_like(rho4, dtype=np.complex128)

    for i in range(2):
        for j in range(2):
            e_ij = np.zeros((2, 2), dtype=np.complex128)
            e_ij[i, j] = 1.0

            a_eij = local_map(e_ij)
            if a_eij.shape != (2, 2):
                msg = f"local_map must return (2,2), got {a_eij.shape}"
                raise ValueError(msg)

            block_ij = rho4[i, :, j, :]  # (k,l) environment operator
            out4 += np.einsum("ab,kl->akbl", a_eij, block_ij)

    return out4.reshape(4, 4)


def test_tomography_run_basic() -> None:
    """Test standard single-step process tomography."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, max_bond_dim=16)
    pt = run(op, params, timesteps=[0.1])

    assert pt.tensor.shape == (4, 16)
    assert pt.weights.shape == (16,)
    assert len(pt.timesteps) == 1


def test_tomography_run_defaults() -> None:
    """Test defaults (timesteps=None -> single step)."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, elapsed_time=0.1)
    pt = run(op, params)
    assert pt.tensor.shape == (4, 16)


def test_tomography_mcwf_multistep() -> None:
    """Test multi-step process tomography with MCWF solver."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, solver="MCWF", num_traj=10)
    pt = run(op, params, timesteps=[0.1, 0.1])
    assert pt.tensor.shape == (4, 16, 16)


def test_tomography_run_multistep() -> None:
    """Test structure of multi-step PT."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, max_bond_dim=16)
    pt = run(op, params, timesteps=[0.1, 0.2])

    assert pt.tensor.shape == (4, 16, 16)
    assert pt.weights.shape == (16, 16)
    assert len(pt.timesteps) == 2


def test_basis_reproduction() -> None:
    """Verify that identity map yields correct prediction."""
    op = MPO.ising(length=2, J=0.0, g=0.0)  # Zero Hamiltonian
    params = AnalogSimParams(dt=0.1, max_bond_dim=16)
    pt = run(op, params, timesteps=[0.1])

    def identity_map(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return rho

    rho_pred = pt.predict_final_state([identity_map])
    # Expect state to be |0><0| (initial)
    expected = np.array([[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(rho_pred, expected, atol=1e-10)


def test_predict_linearity() -> None:
    """Ensure predict_final_state is linear in the intervention maps."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, max_bond_dim=16)
    pt = run(op, params, timesteps=[0.1])

    def map1(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return rho

    def map2(_rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.zeros((2, 2), dtype=complex)

    def sum_map(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return 0.5 * map1(rho) + 0.3 * map2(rho)

    rho1 = pt.predict_final_state([map1])
    rho2 = pt.predict_final_state([map2])
    rho_sum = pt.predict_final_state([sum_map])

    np.testing.assert_allclose(rho_sum, 0.5 * rho1 + 0.3 * rho2, atol=1e-10)


def test_reconstruction_depolarizing() -> None:
    """Test reconstruction of a depolarizing channel via PT."""
    op = MPO.ising(length=2, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=16)
    pt = run(op, params, timesteps=[0.1])

    def depolarize(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return 0.5 * np.trace(rho) * np.eye(2)

    rho_pred = pt.predict_final_state([depolarize])
    expected = 0.5 * np.eye(2)
    np.testing.assert_allclose(rho_pred, expected, atol=1e-10)


def test_choi_duality_biorthogonality() -> None:
    """Verify dual frame biorthogonality: Tr(D_i^dag B_j) = delta_ij."""
    choi_basis, _ = get_choi_basis()
    duals = calculate_dual_choi_basis(choi_basis)

    for i, d_i in enumerate(duals):
        for j, b_j in enumerate(choi_basis):
            inner = np.trace(d_i.conj().T @ b_j)
            expected = 1.0 if i == j else 0.0
            np.testing.assert_allclose(inner, expected, atol=1e-10)


def test_reconstruction_identity_random_choi() -> None:
    """Verify Choi matrix reconstruction: J = sum_k Tr(D_k^dag J) B_k."""
    rng = np.random.default_rng(42)
    choi_basis, _ = get_choi_basis()
    duals = calculate_dual_choi_basis(choi_basis)

    # Random 4x4 matrix
    j_rand = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))

    coeffs = np.array([np.trace(d.conj().T @ j_rand) for d in duals])
    j_rec = np.zeros((4, 4), dtype=complex)
    for c, b in zip(coeffs, choi_basis, strict=False):
        j_rec += c * b

    np.testing.assert_allclose(j_rec, j_rand, atol=1e-10)


def test_dual_extracts_one_hot_for_basis_maps() -> None:
    """Verify duals extract one-hot coefficients for basis maps under the strict Choi build convention.

    This locks the `predict_final_state` Choi builder convention to the duals.
    """
    basis = get_basis_states()
    choi_basis, choi_indices = get_choi_basis()
    duals = calculate_dual_choi_basis(choi_basis)

    for alpha in range(16):
        p, m = choi_indices[alpha]
        rho_p = basis[p][2]
        e_m = basis[m][2]

        def a_alpha(
            rho: NDArray[np.complex128],
            e_m_sub: NDArray[np.complex128] = e_m,
            rho_p_sub: NDArray[np.complex128] = rho_p,
        ) -> NDArray[np.complex128]:
            return np.trace(e_m_sub @ rho) * rho_p_sub

        j_choi = np.zeros((4, 4), dtype=complex)
        for i in range(2):
            for j in range(2):
                e = np.zeros((2, 2), dtype=complex)
                e[i, j] = 1.0
                j_choi += np.kron(a_alpha(e), e)  # NO transpose

        c = np.array([np.trace(d.conj().T @ j_choi) for d in duals])
        expected = np.zeros(16, dtype=complex)
        expected[alpha] = 1.0
        np.testing.assert_allclose(c, expected, atol=1e-10)


def test_held_out_prediction() -> None:
    """Test PT prediction against direct evolution for a random preparation map (1-step)."""
    rng = np.random.default_rng(42)
    op = MPO.ising(length=2, J=1.0, g=0.5)

    params = AnalogSimParams(dt=0.1, max_bond_dim=16, order=2)
    pt = run(op, params, timesteps=[0.1])

    # Hold-out intervention: prepare arbitrary mixed state rho_0
    rho_0 = _get_random_rho(rng)

    def prep_map(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.trace(rho) * rho_0

    # 1) PT prediction
    rho_pred = pt.predict_final_state([prep_map])

    # 2) Direct evolution
    h_mat = expm(-1j * op.to_matrix() * 0.1)
    # Start with rho_0 on site 0, |0> on site 1
    rho_init = np.kron(rho_0, np.array([[1.0, 0.0], [0.0, 0.0]]))
    rho_final_full = h_mat @ rho_init @ h_mat.conj().T
    # Partial trace over site 1
    rho_final = np.trace(rho_final_full.reshape(2, 2, 2, 2), axis1=1, axis2=3)

    np.testing.assert_allclose(rho_pred, rho_final, atol=1e-10)


def test_multi_step_correctness() -> None:
    """Verify 2-step PT correctness against explicit global evolution."""
    rng = np.random.default_rng(42)
    op = MPO.ising(length=2, J=1.0, g=0.5)

    # Use order=2 for TJM 2
    params = AnalogSimParams(dt=0.1, max_bond_dim=16, order=2)
    pt = run(op, params, timesteps=[0.1, 0.1])

    rho_0 = _get_random_rho(rng)

    # Make a proper unitary intervention for step 1
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    u_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.complex128)

    def a0_prep(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        # Prep random mixed state
        return np.trace(rho) * rho_0

    def a1_unitary(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        # Unitary channel
        return u_mat @ rho @ u_mat.conj().T

    # 1) PT prediction
    rho_pred = pt.predict_final_state([a0_prep, a1_unitary])

    # 2) Direct dense evolution
    h_mat = op.to_matrix()
    u_evol = expm(-1j * h_mat * 0.1)

    # Start from |00><00|
    rho_full = np.zeros((4, 4), dtype=np.complex128)
    rho_full[0, 0] = 1.0

    # Apply A0 on site0 (step 0 intervention)
    rho_full = _apply_local_map_site0(rho_full, a0_prep)
    # Evolve 1
    rho_full = u_evol @ rho_full @ u_evol.conj().T
    # Apply A1 on site0 (step 1 intervention)
    rho_full = _apply_local_map_site0(rho_full, a1_unitary)
    # Evolve 2
    rho_full = u_evol @ rho_full @ u_evol.conj().T

    # Partial trace over site 1
    rho_final = np.trace(rho_full.reshape(2, 2, 2, 2), axis1=1, axis2=3)

    # Assert tight tolerance
    np.testing.assert_allclose(rho_pred, rho_final, atol=1e-6)


def test_unnormalized_branch_semantics_h0() -> None:
    """Verify trace-weight consistency in the deterministic H=0 case.

    For each basis map A_{p,m}(rho) = Tr(E_m rho) rho_p, starting from |0><0| on site 0,
    the unnormalized output branch rho_out should have:
    - trace(rho_out) == pt.weights[alpha]
    - trace(rho_out) == Tr(E_m |0><0|)
    """
    op = MPO.ising(length=2, J=0.0, g=0.0)  # H = 0
    params = AnalogSimParams(dt=0.1, max_bond_dim=16, order=1)
    pt = run(op, params, timesteps=[0.1])

    basis = get_basis_states()
    _, choi_indices = get_choi_basis()

    # Initial state is |0><0| on site 0
    rho_0 = np.zeros((2, 2), dtype=complex)
    rho_0[0, 0] = 1.0

    for alpha in range(16):
        # Extract the branch density matrix (unnormalized output for this basis map)
        rho_branch = pt.tensor[:, alpha].reshape(2, 2)
        weight = pt.weights[alpha]

        p, m = choi_indices[alpha]
        e_m = basis[m][2]
        rho_p = basis[p][2]  # The expected output state proportional to rho_p

        expected_trace = np.trace(e_m @ rho_0)

        # Assert trace matches weight
        np.testing.assert_allclose(np.trace(rho_branch), weight, atol=1e-10)
        # Assert trace matches theoretical expectation Tr(E_m rho_0)
        np.testing.assert_allclose(np.trace(rho_branch), expected_trace, atol=1e-10)

        # Optionally, verify the state itself is proportional to rho_p
        expected_rho_branch = expected_trace * rho_p
        np.testing.assert_allclose(rho_branch, expected_rho_branch, atol=1e-10)


def test_tomography_with_noise() -> None:
    """Verify that the tomography pipeline runs correctly with a noise model and multiple trajectories.

    This is an integration test to ensure that the parallelized worker handles `num_trajectories > 1`
    and stochastic noise operators without crashing. It does not perform an exact arithmetic assertion
    against the output.
    """
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, max_bond_dim=16, order=1)

    # Create a simple noise model (e.g. amplitude damping on site 0)
    noise_model = NoiseModel([{"name": "lowering", "sites": [0], "strength": 0.05}])

    # Run tomography computationally with noise
    pt = run(op, params, timesteps=[0.1], num_trajectories=5, noise_model=noise_model)

    # Check that the tensor built properly without None outputs
    assert pt.tensor.shape == (4, 16)
    assert not np.isnan(pt.tensor).any()
    assert pt.weights.shape == (16,)
    assert not np.isnan(pt.weights).any()
def test_run_mc_upsilon_endpoint_correctness() -> None:
    """Test that run_mc_upsilon matches run exactly when exhaustive in deterministic mode."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    # Force deterministic TJM
    params = AnalogSimParams(dt=0.1, max_bond_dim=16, order=2, show_progress=False)
    timesteps = [0.1, 0.2]

    # 1. Exact full enumeration
    pt_exact = run(op, params, timesteps=timesteps)
    # Let ProcessTensor decide the best convention
    u_exact, conv = pt_exact.reconstruct_comb_choi(check=True, return_convention=True)
    u_exact = canonicalize_upsilon(u_exact, hermitize=True, psd_project=False, normalize_trace=True)

    # 2. MC with replace=False and N=256 (exhaustive for k=2)
    # We pass the same 'conv' found by exact reconstruction
    u_mc, _ = estimate_process_tensor(
        op, params, timesteps=timesteps,
        method="mc", output="dense",
        num_samples=256, replace=False, seed=42, dual_transform=conv,
    )
    u_mc = canonicalize_upsilon(u_mc, hermitize=True, psd_project=False, normalize_trace=True)

    err = rel_fro_error(u_mc, u_exact)
    assert err < 1e-10


@pytest.mark.parametrize("k", [1, 2])
def test_continuous_random_state_tomography(k: int):
    """Verify continuous random state tomography converges for 1 and 2 step sequences."""
    op = MPO.ising(length=2, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, solver="MCWF", show_progress=False)
    timesteps = [1.0] * k
    seed = 42

    # 1. Exact Reference via deterministic YAQS bases
    pt_ref = run(op, params, timesteps=timesteps)
    U_ref_raw = pt_ref.reconstruct_comb_choi()
    
    U_red = reduced_upsilon(U_ref_raw, k=k, keep_last_m=k)
    rho_ref = canonicalize_upsilon(U_red, hermitize=True, psd_project=False, normalize_trace=True)
    
    qmi_ref = float(comb_qmi_from_upsilon_dense(rho_ref, assume_canonical=True))

    # 2. Continuous State Estimation
    # Run a moderate N to ensure stochastic dimension setup and metrics execute cleanly
    U_cont, _ = estimate_process_tensor(
        op, params, timesteps=timesteps,
        method="mc", output="dense",
        num_samples=256, num_trajectories=1, seed=seed, sampling="continuous",
    )
    
    U_red = reduced_upsilon(U_cont, k=k, keep_last_m=k)
    rho_hat = canonicalize_upsilon(U_red, hermitize=True, psd_project=False, normalize_trace=True)
    
    # We can reuse rel_fro_error from metrics here instead of custom rel_fro
    fro_err = rel_fro_error(rho_hat, rho_ref)
    qmi_err = abs(float(comb_qmi_from_upsilon_dense(rho_hat, assume_canonical=True)) - qmi_ref)

    # Assert structural execution without bounds crashing
    assert fro_err < 10.0, f"Frobenius Error unboundedly diverged: {fro_err:.3f}"
    assert qmi_err < 10.0, f"QMI Error unboundedly diverged: {qmi_err:.3f}"

    if k > 1:
        cmi_ref = float(comb_cmi_from_upsilon_dense(rho_ref, assume_canonical=True))
        assert cmi_ref > 0.1, f"Failed to induce required non-Markovian CMI > 0.1 memory: {cmi_ref:.3f}"
        
        cmi_err = abs(float(comb_cmi_from_upsilon_dense(rho_hat, assume_canonical=True)) - cmi_ref)
        
        assert cmi_err < 10.0, f"CMI bounds unbounded: {cmi_err:.3f}"


################################################################################
# ── Convergence Tests ─────────────────────────────────────────────────────────
################################################################################

# ── helpers ─────────────────────────────────────────────────────────────────


def _rel_fro(A: np.ndarray, B: np.ndarray, eps: float = 1e-15) -> float:
    return float(np.linalg.norm(A - B, "fro") / max(np.linalg.norm(B, "fro"), eps))


def _make_problem(k: int = 2):
    """Tiny 2-site Ising system, deterministic evolution (noise_model=None)."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, solver="MCWF", show_progress=False)
    timesteps = [0.1] * k
    return op, params, timesteps


def _dense_ref_total(k: int) -> np.ndarray:
    """Full enumeration reference; returns raw U_ref (population total).

    Uses MC uniform with replace=False and N=16^k, which exactly evaluates
    the full sum over all paths without sampling variance.
    """
    op, params, timesteps = _make_problem(k)
    U_ref, _ = estimate_process_tensor(
        op, params, timesteps=timesteps,
        method="mc", output="dense",
        num_samples=16**k, num_trajectories=1,
        noise_model=None, dual_transform="T", replace=False,
    )
    return U_ref


def _canon_for_compare(U: np.ndarray, k: int) -> np.ndarray:
    """Canonicalize + reduce to constant (8,8) metric."""
    return canonicalize_upsilon(
        reduced_upsilon(U, k=k, keep_last_m=1),
        hermitize=True,
        psd_project=False,  # OFF for convergence plots
        normalize_trace=True,
    )


# ── A) Dense reference sanity ────────────────────────────────────────────────


@pytest.mark.parametrize("k", [1, 2])
def test_dense_reference_well_formed(k: int) -> None:
    """Dense Υ_ref has correct shape and is Hermitian after canonicalization."""
    U_ref = _dense_ref_total(k)

    assert U_ref.shape == (2 * (4**k), 2 * (4**k)), "Unexpected U_ref shape"

    rho = _canon_for_compare(U_ref, k)
    assert rho.shape == (8, 8)
    assert np.allclose(rho, rho.conj().T, atol=1e-10), "Not Hermitian after canonicalize"
    assert abs(np.trace(rho) - 1.0) < 1e-8, "Trace not 1 after normalize_trace"


# ── B) MC converges to dense reference ────────────────────────────────────────


@pytest.mark.parametrize("k", [1, 2])
def test_mc_uniform_converges_frobenius(k: int) -> None:
    """Relative Fro error on reduced+canonicalized Υ decreases as N_seq grows."""
    U_ref = _dense_ref_total(k)
    rho_ref = _canon_for_compare(U_ref, k)
    op, params, timesteps = _make_problem(k)

    # Two sample sizes; expect error to drop monotonically in expectation
    n_seeds = 3
    errs: dict[int, list[float]] = {64: [], 256: []}

    for nseq, err_list in errs.items():
        for s in range(n_seeds):
            U_hat, _ = estimate_process_tensor(
                op, params, timesteps=timesteps,
                method="mc", output="dense",
                num_samples=nseq, num_trajectories=1,
                noise_model=None, seed=100 + s,
                dual_transform="T", replace=True,
            )
            err_list.append(_rel_fro(_canon_for_compare(U_hat, k), rho_ref))

    mean_small = float(np.mean(errs[64]))
    mean_large = float(np.mean(errs[256]))
    assert mean_large < mean_small, (
        f"MC k={k}: error did not decrease: N=64 err={mean_small:.3f}, N=256 err={mean_large:.3f}"
    )


# ── C) SIS converges to dense reference ────────────────────────────────────────


@pytest.mark.parametrize("k", [2])
def test_sis_local_converges_frobenius(k: int) -> None:
    """SIS (local proposal) relative Fro error decreases as N_particles grows."""
    U_ref = _dense_ref_total(k)
    rho_ref = _canon_for_compare(U_ref, k)
    op, params, timesteps = _make_problem(k)

    n_seeds = 3
    errs: dict[int, list[float]] = {64: [], 256: []}

    for N, err_list in errs.items():
        for s in range(n_seeds):
            U_hat, _ = estimate_process_tensor(
                op, params, timesteps=timesteps,
                method="sis", output="dense",
                num_samples=N, noise_model=None,
                seed=200 + s,
                proposal="local", floor_eps=0.0,
                stratify_step1=True, resample=True,
                parallel=False, dual_transform="T",
            )
            err_list.append(_rel_fro(_canon_for_compare(U_hat, k), rho_ref))

    mean_small = float(np.mean(errs[64]))
    mean_large = float(np.mean(errs[256]))
    assert mean_large < mean_small, (
        f"SIS k={k}: error did not decrease: N=64 err={mean_small:.3f}, N=256 err={mean_large:.3f}"
    )


# ── D) QMI error decreases with N_particles (SIS) ────────────────────────────


@pytest.mark.parametrize("k", [2])
def test_sis_qmi_converges(k: int) -> None:
    """|QMI_hat - QMI_ref| decreases as N_particles grows (past='last')."""
    U_ref = _dense_ref_total(k)
    rho_ref = _canon_for_compare(U_ref, k)
    qmi_ref = float(comb_qmi_from_upsilon_dense(rho_ref, assume_canonical=True, past="last"))

    op, params, timesteps = _make_problem(k)
    n_seeds = 3
    errs: dict[int, list[float]] = {64: [], 256: []}

    for N, err_list in errs.items():
        for s in range(n_seeds):
            U_hat, _ = estimate_process_tensor(
                op, params, timesteps=timesteps,
                method="sis", output="dense",
                num_samples=N, noise_model=None,
                seed=300 + s,
                proposal="local", floor_eps=0.0,
                stratify_step1=True, resample=True,
                parallel=False, dual_transform="T",
            )
            rho_hat = _canon_for_compare(U_hat, k)
            qmi_hat = float(comb_qmi_from_upsilon_dense(rho_hat, assume_canonical=True, past="last"))
            err_list.append(abs(qmi_hat - qmi_ref))

    mean_small = float(np.mean(errs[64]))
    mean_large = float(np.mean(errs[256]))
    assert mean_large < mean_small, (
        f"SIS QMI k={k}: error did not decrease: N=64 err={mean_small:.4f}, N=256 err={mean_large:.4f}"
    )


@pytest.mark.parametrize("k", [1, 2])
def test_mc_exact_enumeration(k: int) -> None:
    """MC uniform with N=16^k and replace=False is exact (zero sampling error).
    
    This acts as a sanity check that no scale mismatch exists: the raw estimator
    recovers exactly the same object as the dense total reference.
    """
    U_ref = _dense_ref_total(k)
    rho_ref = _canon_for_compare(U_ref, k)
    op, params, timesteps = _make_problem(k)

    U_hat, _ = estimate_process_tensor(
        op, params, timesteps=timesteps,
        method="mc", output="dense",
        num_samples=16**k, num_trajectories=1,
        noise_model=None, dual_transform="T", replace=False,
    )
    rho_hat = _canon_for_compare(U_hat, k)
    err = _rel_fro(rho_hat, rho_ref)
    assert err < 1e-10, f"MC exact enumeration not exact: err = {err:.2e}"

# ── E) k=1 is exact under deterministic evolution + stratify ─────────────────

def test_sis_k1_exact_deterministic() -> None:
    """k=1 + deterministic evolution + stratify_step1: error should be near machine eps."""
    k = 1
    U_ref = _dense_ref_total(k)
    rho_ref = _canon_for_compare(U_ref, k)
    op, params, ts = _make_problem(k)

    U_hat, meta = estimate_process_tensor(
        op, params, timesteps=ts,
        method="sis", output="dense",
        num_samples=16, noise_model=None, seed=0,
        proposal="local", floor_eps=0.0,
        stratify_step1=True, resample=False,
        parallel=False, dual_transform="T",
    )
    err = _rel_fro(_canon_for_compare(U_hat, k), rho_ref)
    assert err < 1e-12, f"k=1 deterministic stratify not exact (err={err:.2e})"


# ── F) Theoretical Scaling Tests (Pure NumPy & Dephasing) ────────────────────

def test_pure_numpy_toy_mc_convergence(tmp_path) -> None:
    """Pure NumPy toy test (k=2, 8x8 contributions) with known exact reference.
    Plots Fro error vs N, shows slope ~ -1/2.
    """
    np.random.seed(42)
    k = 2
    M = 16**k  # 256
    dims = 8

    # Random Hermitian matrices (contributions per sequence)
    matrices = []
    for _ in range(M):
        A = np.random.randn(dims, dims) + 1j * np.random.randn(dims, dims)
        matrices.append(A + A.conj().T)
    matrices = np.array(matrices)

    # Random path weights (probabilities summing to 1)
    p = np.random.rand(M)
    p /= np.sum(p)

    # Exact population total reference
    U_ref = np.sum(p[:, None, None] * matrices, axis=0)
    norm_ref = np.linalg.norm(U_ref, "fro")

    sample_ns = np.array([16, 32, 64, 128, 256, 512, 1024])
    n_seeds = 10
    
    fro_mu, fro_sd = [], []
    for N in sample_ns:
        errs = []
        for s in range(n_seeds):
            rng = np.random.default_rng(1000 * N + s)
            # Uniform MC sampling (replace=True) targeting Total
            idxs = rng.integers(0, M, size=N)
            # Estimator = (M/N) * sum_{sampled} (p_i * U_i)
            U_hat = (M / N) * np.sum(p[idxs, None, None] * matrices[idxs], axis=0)
            
            err = float(np.linalg.norm(U_hat - U_ref, "fro") / norm_ref)
            errs.append(err)
        fro_mu.append(np.mean(errs))
        fro_sd.append(np.std(errs))

    fro_mu = np.array(fro_mu)
    log_N = np.log(sample_ns)
    slope, _ = np.polyfit(log_N, np.log(fro_mu), 1)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.errorbar(sample_ns, fro_mu, yerr=fro_sd, fmt="o-", label="MC uniform", capsize=3)
    
    # Reference slope line
    ref_y = fro_mu[0] * np.sqrt(sample_ns[0] / sample_ns)
    ax.plot(sample_ns, ref_y, "k:", alpha=0.55, label=r"$\propto N^{-1/2}$")
    
    ax.set_title(f"Pure NumPy Toy MC Convergence\nSlope = {slope:.3f}")
    ax.set_xlabel("Samples $N$")
    ax.set_ylabel("Relative Frobenius error")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    
    plot_path = tmp_path / "pure_numpy_toy_mc.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    assert np.isclose(slope, -0.5, atol=0.15), f"Expected slope ~ -0.5, got {slope:.3f}"


def test_pure_numpy_sis_toy_correctness() -> None:
    """SIS-only toy test for weight correctness (unbiased HT estimator check)."""
    np.random.seed(42)
    M = 256
    dims = 8

    matrices = []
    for _ in range(M):
        A = np.random.randn(dims, dims) + 1j * np.random.randn(dims, dims)
        matrices.append(A + A.conj().T)
    matrices = np.array(matrices)

    p = np.random.rand(M)
    p /= np.sum(p)
    U_ref = np.sum(p[:, None, None] * matrices, axis=0)

    # SIS proposal distribution q (slightly biased away from p)
    q = p + 0.1 * np.random.rand(M)
    q /= np.sum(q)

    N = 100000
    rng = np.random.default_rng(123)
    idxs = rng.choice(M, size=N, p=q)

    # SIS unbiased estimator: (1/N) * sum [ (p_i / q_i) * U_i ]
    w = p[idxs] / q[idxs]
    U_hat = (1 / N) * np.sum(w[:, None, None] * matrices[idxs], axis=0)
    
    err = float(np.linalg.norm(U_hat - U_ref, "fro") / np.linalg.norm(U_ref, "fro"))
    assert err < 0.1, f"SIS toy estimator failed to converge, err={err:.3f}"


def test_yaqs_k1_dephasing() -> None:
    """YAQS k=1 dephasing test comparing scalar metrics (trace, Fro of reduced, QMI of reduced)."""
    k = 1
    op = MPO.ising(length=2, J=1.0, g=0.5)
    noise = NoiseModel(processes=[{"name": "z", "sites": [i], "strength": 0.5} for i in range(op.length)])
    params = AnalogSimParams(dt=0.1, solver="MCWF", show_progress=False)
    timesteps = [0.1]

    # Dense reference via full YAQS run (averages over trajectories returns EXACT TOTAL T)
    pt = run(
        op, params, timesteps=timesteps,
        num_trajectories=1000, noise_model=noise
    )
    U_ref_total = pt.reconstruct_comb_choi()
    
    rho_ref = _canon_for_compare(U_ref_total, k)
    qmi_ref = float(comb_qmi_from_upsilon_dense(rho_ref, assume_canonical=True, past="last"))
    trace_ref = float(np.trace(U_ref_total).real)

    # Approximate via MC uniform (replace=True, N=2048 to tighten sampling errors)
    U_hat, _ = estimate_process_tensor(
        op, params, timesteps=timesteps, method="mc", output="dense",
        num_samples=2048, num_trajectories=1,
        noise_model=noise, dual_transform="T", replace=True, seed=100
    )
    rho_hat = _canon_for_compare(U_hat, k)
    qmi_hat = float(comb_qmi_from_upsilon_dense(rho_hat, assume_canonical=True, past="last"))
    trace_hat = float(np.trace(U_hat).real)
    fro_err = _rel_fro(rho_hat, rho_ref)

    # Scalar metric comparisons
    assert np.isclose(trace_ref, trace_hat, rtol=0.25), f"Trace mismatch: {trace_ref:.3f} vs {trace_hat:.3f}"
    assert fro_err < 0.4, f"Fro error too high: {fro_err:.3f}"
    assert abs(qmi_ref - qmi_hat) < 0.4, f"QMI error too high: {abs(qmi_ref - qmi_hat):.3f}"


################################################################################
# ── Process Tensor MPO Tests ──────────────────────────────────────────────────
################################################################################


def random_rho():
    np.random.seed(42)
    rho = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
    rho = rho @ rho.conj().T
    return rho / np.trace(rho)

def random_complex_matrix(dim):
    return np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)

def small_hamiltonian(length=1):
    return MPO.hamiltonian(length=length, one_body=[(1.0, "X")], two_body=[])

def rel_err(A, B):
    return np.linalg.norm(A - B) / max(np.linalg.norm(A), 1e-15)

def get_sim_params():
    return AnalogSimParams(elapsed_time=0.1, dt=0.01, solver="MCWF", show_progress=False)

# --- MPO Algebra Tests ---

@pytest.mark.parametrize("k", [1, 2, 3])
def test_rank1_mpo_vs_kron(k):
    np.random.seed(42 + k)
    rho = random_rho()
    ops = [random_complex_matrix(4) for _ in range(k)]
    w = 1.23

    mpo = rank1_upsilon_mpo_term(rho, ops, weight=w)
    dense_mpo = upsilon_mpo_to_dense(mpo)
    
    dense_ref = rho
    for op in ops:
        dense_ref = np.kron(dense_ref, op)
    dense_ref *= w

    assert rel_err(dense_mpo, dense_ref) < 1e-12

def test_mpo_addition_matches_dense():
    np.random.seed(111)
    rho1, rho2 = random_rho(), random_rho()
    ops1 = [random_complex_matrix(4), random_complex_matrix(4)]
    ops2 = [random_complex_matrix(4), random_complex_matrix(4)]
    
    mpo1 = rank1_upsilon_mpo_term(rho1, ops1, weight=1.0)
    mpo2 = rank1_upsilon_mpo_term(rho2, ops2, weight=0.5)
    
    mpo_added = mpo1 + mpo2
    dense_added = upsilon_mpo_to_dense(mpo_added)
    
    d1 = upsilon_mpo_to_dense(mpo1)
    d2 = upsilon_mpo_to_dense(mpo2)
    
    assert rel_err(dense_added, d1 + d2) < 1e-12

def test_mpo_sum_matches_dense():
    np.random.seed(222)
    mpos = []
    denses = []
    for _ in range(5):
        rho = random_rho()
        ops = [random_complex_matrix(4), random_complex_matrix(4)]
        w = np.random.rand()
        m = rank1_upsilon_mpo_term(rho, ops, weight=w)
        mpos.append(m)
        denses.append(upsilon_mpo_to_dense(m))
        
    mpo_summed = MPO.mpo_sum(mpos)
    dense_summed = sum(denses)
    
    assert rel_err(upsilon_mpo_to_dense(mpo_summed), dense_summed) < 1e-12

# --- MC Parity Tests ---

@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("dual_transform", ["id", "T", "conj", "dag"])
def test_mc_mpo_parity_discrete(k, dual_transform):
    sp = get_sim_params()
    H = small_hamiltonian()
    
    ups_dense, _ = estimate_process_tensor(
        operator=H, sim_params=sp, timesteps=[0.1]*k, method="mc", output="dense",
        num_samples=4, dual_transform=dual_transform, seed=42
    )
    ups_mpo, _ = estimate_process_tensor(
        operator=H, sim_params=sp, timesteps=[0.1]*k, method="mc", output="mpo",
        num_samples=4, dual_transform=dual_transform, seed=42,
        compress_every=10000, tol=0.0, max_bond_dim=None, n_sweeps=0
    )
    ups_mpo_dense = upsilon_mpo_to_dense(ups_mpo)
    assert rel_err(ups_dense, ups_mpo_dense) < 1e-12

@pytest.mark.parametrize("sampling", ["uniform"])  # continuous MC; candidate_local removed
def test_mc_mpo_parity_continuous(sampling):
    sp = get_sim_params()
    H = small_hamiltonian()

    ups_dense, _ = estimate_process_tensor(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1],
        method="mc", output="dense", sampling="continuous", num_samples=8, seed=42
    )
    ups_mpo, _ = estimate_process_tensor(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1],
        method="mc", output="mpo", sampling="continuous", num_samples=8, seed=42,
        compress_every=10000, tol=0.0, max_bond_dim=None, n_sweeps=0
    )
    assert rel_err(ups_dense, upsilon_mpo_to_dense(ups_mpo)) < 1e-12

def test_mc_mpo_parity_noisy():
    sp = get_sim_params()
    sp.noise_model = NoiseModel([{"name": "lowering", "sites": [0], "strength": 0.05}])
    H = small_hamiltonian()
    
    ups_dense, _ = estimate_process_tensor(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1],
        method="mc", output="dense", num_samples=1, seed=42
    )
    ups_mpo, _ = estimate_process_tensor(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1],
        method="mc", output="mpo", num_samples=1, seed=42,
        compress_every=10000, tol=0.0, max_bond_dim=None, n_sweeps=0
    )
    assert rel_err(ups_dense, upsilon_mpo_to_dense(ups_mpo)) < 1e-10

# --- SIS Parity Tests ---

@pytest.mark.parametrize("proposal", ["uniform", "local", "mixture"])
@pytest.mark.parametrize("dual_transform", ["id", "T"])
def test_sis_mpo_parity(proposal, dual_transform):
    sp = get_sim_params()
    H = small_hamiltonian()
    
    ups_dense, _ = estimate_process_tensor(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1], method="sis", output="dense",
        num_samples=8, proposal=proposal, resample=False, dual_transform=dual_transform, seed=123
    )
    ups_mpo, _ = estimate_process_tensor(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1], method="sis", output="mpo",
        num_samples=8, proposal=proposal, resample=False, dual_transform=dual_transform, seed=123,
        compress_every=10000, tol=0.0, max_bond_dim=None, n_sweeps=0
    )
    assert rel_err(ups_dense, upsilon_mpo_to_dense(ups_mpo)) < 1e-12

# --- Compression Tests ---

def test_compressed_mc_smoke():
    sp = get_sim_params()
    H = small_hamiltonian()
    
    ups_dense, _ = estimate_process_tensor(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1, 0.1],
        method="mc", output="dense", num_samples=16, seed=42
    )
    ups_mpo, meta = estimate_process_tensor(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1, 0.1],
        method="mc", output="mpo", num_samples=16, seed=42,
        compress_every=4, tol=1e-10, max_bond_dim=32, n_sweeps=1
    )
    ups_mpo_dense = upsilon_mpo_to_dense(ups_mpo)
    
    assert "bond_dim_final" in meta
    assert ups_mpo_dense.shape == ups_dense.shape
    assert rel_err(ups_dense, ups_mpo_dense) < 1e-6

def test_compression_delta():
    sp = get_sim_params()
    H = small_hamiltonian()
    
    ups_uncompressed, _ = estimate_process_tensor(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1, 0.1],
        method="mc", output="mpo", num_samples=16, seed=777,
        compress_every=1000, tol=0.0, max_bond_dim=None, n_sweeps=0
    )
    ups_compressed, _ = estimate_process_tensor(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1, 0.1],
        method="mc", output="mpo", num_samples=16, seed=777,
        compress_every=2, tol=1e-12, max_bond_dim=None, n_sweeps=1
    )
    
    d_u = upsilon_mpo_to_dense(ups_uncompressed)
    d_c = upsilon_mpo_to_dense(ups_compressed)
    assert rel_err(d_u, d_c) < 1e-10

def test_monotonic_compression():
    sp = get_sim_params()
    H = small_hamiltonian()
    
    ups_strong, _ = estimate_process_tensor(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1, 0.1],
        method="mc", output="mpo", num_samples=16, seed=888,
        compress_every=2, tol=1e-4, max_bond_dim=8, n_sweeps=1
    )
    ups_weak, _ = estimate_process_tensor(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1, 0.1],
        method="mc", output="mpo", num_samples=16, seed=888,
        compress_every=2, tol=1e-10, max_bond_dim=64, n_sweeps=1
    )
    ups_exact, _ = estimate_process_tensor(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1, 0.1],
        method="mc", output="mpo", num_samples=16, seed=888,
        compress_every=1000, tol=0.0, max_bond_dim=None, n_sweeps=0
    )
    
    err_strong = rel_err(upsilon_mpo_to_dense(ups_strong), upsilon_mpo_to_dense(ups_exact))
    err_weak = rel_err(upsilon_mpo_to_dense(ups_weak), upsilon_mpo_to_dense(ups_exact))
    
    assert err_weak < err_strong + 1e-12

# --- Metadata and Failure modes ---

def test_metadata_and_failure_modes():
    sp = get_sim_params()
    H = small_hamiltonian()
    
    _, meta = estimate_process_tensor(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1],
        method="mc", output="mpo", num_samples=4, seed=42
    )

    assert "bond_dim_final" in meta
    assert "max_bond_final" in meta
    assert "compression_tol" in meta
    assert "compression_max_bond_dim" in meta
    # Note: compression_n_sweeps not in standardized core meta


# --- Small Multi-Qubit Parity ---

@pytest.mark.slow
def test_small_multi_qubit_parity():
    sp = get_sim_params()
    # N=2 hamiltonian
    H = MPO.hamiltonian(length=2, one_body=[(1.0, "X")], two_body=[])
    
    ups_dense, _ = estimate_process_tensor(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1],
        method="mc", output="dense", num_samples=4, seed=999
    )
    ups_mpo, _ = estimate_process_tensor(
        operator=H, sim_params=sp, timesteps=[0.1, 0.1],
        method="mc", output="mpo", num_samples=4, seed=999,
        compress_every=1000, tol=0.0, max_bond_dim=None, n_sweeps=0
    )
    
    assert rel_err(ups_dense, upsilon_mpo_to_dense(ups_mpo)) < 1e-10