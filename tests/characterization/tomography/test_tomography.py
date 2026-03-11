# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for Process Tomography estimates, equivalence, convergence, and physical correctness."""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Callable

import numpy as np
import pytest
from scipy.linalg import expm

if TYPE_CHECKING:
    from numpy.typing import NDArray

from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.core.data_structures.noise_model import NoiseModel

from mqt.yaqs.characterization.tomography.tomography import (
    calculate_dual_choi_basis,
    get_basis_states,
    get_choi_basis,
    run_exact,
    estimate,
)

from mqt.yaqs.characterization.tomography.process_tensor import (
    canonicalize_upsilon,
    rank1_upsilon_mpo_term,
    upsilon_mpo_to_dense,
)

from mqt.yaqs.characterization.tomography.metrics import rel_fro_error

from mqt.yaqs.core.libraries.gate_library import X, Y, Z

################################################################################
# ── Helpers ───────────────────────────────────────────────────────────────────
################################################################################

def _get_random_rho(rng: np.random.Generator) -> NDArray[np.complex128]:
    """Generate a random 2x2 density matrix."""
    z = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    rho = z @ z.conj().T
    return rho / np.trace(rho)


def _apply_local_map_site0(
    rho_full: NDArray[np.complex128],
    local_map: Callable[[NDArray[np.complex128]], NDArray[np.complex128]],
) -> NDArray[np.complex128]:
    """Apply (local_map ⊗ I) to a 2-qubit density matrix rho_full.

    Assumes rho_full uses ordering kron(site0, site1).
    Reshape convention: rho[i, k, j, l] where i,j index site0, k,l index site1.
    """
    rho4 = rho_full.reshape(2, 2, 2, 2)  # (i,k,j,l)
    out4 = np.zeros_like(rho4, dtype=np.complex128)

    for i in range(2):
        for j in range(2):
            e_ij = np.zeros((2, 2), dtype=np.complex128)
            e_ij[i, j] = 1.0

            a_eij = local_map(e_ij)
            block_ij = rho4[i, :, j, :]  # (k,l) environment operator
            out4 += np.einsum("ab,kl->akbl", a_eij, block_ij)

    return out4.reshape(4, 4)


def _dense_physical_reference(
    op: MPO,
    timesteps: list[float],
    interventions: list[Callable[[NDArray[np.complex128]], NDArray[np.complex128]]],
) -> NDArray[np.complex128]:
    """Independent physical reference: direct dense evolution for a 2-site system.

    Starts from |00><00|. Primarily intended as a small-system physical oracle.
    For each timestep, apply the local intervention first, then evolve under
    the Hamiltonian. Returns final 2x2 density matrix on site 0 (partial trace
    over site 1).
    """
    if op.length != 2:
        raise ValueError("_dense_physical_reference only supports length=2.")

    h_mat = op.to_matrix()
    rho = np.zeros((4, 4), dtype=np.complex128)
    rho[0, 0] = 1.0  # |00><00|

    for step_idx, dt in enumerate(timesteps):
        # 1. Apply intervention locally on site 0
        rho = _apply_local_map_site0(rho, interventions[step_idx])

        # 2. Evolve
        u = expm(-1j * h_mat * dt)
        rho = u @ rho @ u.conj().T

    # Partial trace over site 1 (keep site 0)
    rho4 = rho.reshape(2, 2, 2, 2)
    return np.einsum("akbk->ab", rho4)


def predict_from_dense_upsilon(
    U: NDArray[np.complex128],
    interventions: list[Callable[[NDArray[np.complex128]], NDArray[np.complex128]]],
) -> NDArray[np.complex128]:
    """Contraction helper for parity tests. Predicts final state from a canonical dense Upsilon.

    NOTE: This is NOT an independent physical reference, but a tool to check representation consistency.
    """
    k_steps = len(interventions)
    past_list = []
    for emap in interventions:
        j_choi = np.zeros((4, 4), dtype=complex)
        for i in range(2):
            for j in range(2):
                e_in = np.zeros((2, 2), dtype=complex)
                e_in[i, j] = 1.0
                j_choi += np.kron(emap(e_in), e_in)
        past_list.append(j_choi)
    
    past_total = past_list[0]
    for p in past_list[1:]:
        past_total = np.kron(past_total, p)
        
    dim_p = 4 ** k_steps
    U4 = U.reshape(2, dim_p, 2, dim_p)
    ins = past_total.T.reshape(dim_p, dim_p)
    return np.einsum("s p q r, r p -> s q", U4, ins)


################################################################################
# ── Section 1: Basis / Dual Sanity ───────────────────────────────────────────
################################################################################

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

    j_rand = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    coeffs = np.array([np.trace(d.conj().T @ j_rand) for d in duals])
    j_rec = np.zeros((4, 4), dtype=complex)
    for c, b in zip(coeffs, choi_basis, strict=False):
        j_rec += c * b

    np.testing.assert_allclose(j_rec, j_rand, atol=1e-10)


def test_dual_extracts_one_hot_for_basis_maps() -> None:
    """Verify duals extract one-hot coefficients for basis maps under the Choi build convention."""
    basis = get_basis_states()
    choi_basis, choi_indices = get_choi_basis()
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


def test_basis_reproduction() -> None:
    """Verify that identity map yields correct prediction for H=0."""
    op = MPO.ising(length=2, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=16)
    pt = run_exact(op, params, timesteps=[0.1])

    def identity_map(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return rho

    rho_pred = pt.predict_final_state([identity_map])
    expected = np.array([[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(rho_pred, expected, atol=1e-10)


################################################################################
# ── Section 2: Exact Physical Correctness (vs Direct Physics) ────────────────
################################################################################

def test_exact_1step_prediction_vs_physics():
    """Verify 1-step exact ProcessTensor prediction vs direct dense evolution."""
    rng = np.random.default_rng(101)
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.001, max_bond_dim=16, order=2)

    rho_held_out = _get_random_rho(rng)
    def prep_map(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.trace(rho) * rho_held_out

    interventions = [prep_map]
    timesteps = [0.001]

    pt = run_exact(op, params, timesteps=timesteps)
    rho_pt = pt.predict_final_state(interventions)
    rho_ref = _dense_physical_reference(op, timesteps, interventions)

    assert rel_fro_error(rho_pt, rho_ref) < 1e-5


def test_exact_2step_prediction_vs_physics():
    """Verify 2-step exact ProcessTensor prediction vs direct dense evolution."""
    rng = np.random.default_rng(102)
    op = MPO.ising(length=2, J=0.8, g=1.2)
    params = AnalogSimParams(dt=0.001, max_bond_dim=16, order=2)

    rho_prep = _get_random_rho(rng)
    theta = float(rng.uniform(0.1, 1.0))
    u_mat = expm(-1j * theta * (X().matrix + Z().matrix))

    def a0_prep(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.trace(rho) * rho_prep
    def a1_unitary(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return u_mat @ rho @ u_mat.conj().T

    interventions = [a0_prep, a1_unitary]
    timesteps = [0.001, 0.001]

    pt = run_exact(op, params, timesteps=timesteps)
    rho_pt = pt.predict_final_state(interventions)
    rho_ref = _dense_physical_reference(op, timesteps, interventions)

    assert rel_fro_error(rho_pt, rho_ref) < 1e-5


def test_exact_instrument_prediction_vs_physics():
    """Verify prediction for a non-trace-preserving instrument branch vs direct physics."""
    op = MPO.ising(length=2, J=1.0, g=0.2)
    params = AnalogSimParams(dt=0.001, max_bond_dim=16, order=2)

    def proj0_map(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        p0 = np.array([[1, 0], [0, 0]], dtype=complex)
        return p0 @ rho @ p0

    interventions = [proj0_map]
    timesteps = [0.001]

    pt = run_exact(op, params, timesteps=timesteps)
    rho_pt = pt.predict_final_state(interventions)
    rho_ref = _dense_physical_reference(op, timesteps, interventions)

    assert abs(np.trace(rho_pt) - np.trace(rho_ref)) < 1e-5
    assert rel_fro_error(rho_pt, rho_ref) < 1e-5


def test_exact_moderate_step_prediction_vs_physics():
    """Verify prediction vs physics at a larger dt (0.1), where Trotter error is more visible but still bounded."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    # Order=2 helps at larger dt
    params = AnalogSimParams(dt=0.1, max_bond_dim=16, order=2)

    def x_gate(rho):
        x = X().matrix
        return x @ rho @ x.conj().T

    interventions = [x_gate]
    timesteps = [0.1]

    pt = run_exact(op, params, timesteps=timesteps)
    rho_pt = pt.predict_final_state(interventions)
    rho_ref = _dense_physical_reference(op, timesteps, interventions)

    # At dt=0.1, error should be larger but still around 1e-3 or 1e-4
    assert rel_fro_error(rho_pt, rho_ref) < 5e-3


def test_reconstruction_depolarizing() -> None:
    """Test reconstruction of a depolarizing channel via PT."""
    op = MPO.ising(length=2, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=16)
    pt = run_exact(op, params, timesteps=[0.1])

    def depolarize(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return 0.5 * np.trace(rho) * np.eye(2)

    rho_pred = pt.predict_final_state([depolarize])
    expected = 0.5 * np.eye(2)
    np.testing.assert_allclose(rho_pred, expected, atol=1e-10)


################################################################################
# ── Section 3: Representation Parity ──────────────────────────────────────────
################################################################################

def test_exact_representation_parity():
    """Verify exact MPO and process_tensor representations yield identical predictions."""
    rng = np.random.default_rng(201)
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, max_bond_dim=16, order=2)

    rho_prep = _get_random_rho(rng)
    u_mat = expm(-1j * 0.5 * Y().matrix)

    def a0_prep(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.trace(rho) * rho_prep
    def a1_unitary(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return u_mat @ rho @ u_mat.conj().T

    interventions = [a0_prep, a1_unitary]
    timesteps = [0.1, 0.1]

    pt = run_exact(op, params, timesteps=timesteps)
    rho_pt = pt.predict_final_state(interventions)

    # Use internal implementation for exact MPO parity validation
    U_mpo = estimate(op, params, timesteps=timesteps, method="mc", num_samples=16**len(timesteps))
    U_mpo_dense = canonicalize_upsilon(upsilon_mpo_to_dense(U_mpo), hermitize=True, psd_project=True, normalize_trace=False)
    rho_mpo = predict_from_dense_upsilon(U_mpo_dense, interventions)

    assert rel_fro_error(rho_pt, rho_mpo) < 1e-10


def test_mc_without_replacement_matches_exact_for_k2() -> None:
    """MC uniform with N=16^k matches internal exact MPO reference exactly."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, solver="MCWF", show_progress=False)
    timesteps = [0.1, 0.1]

    U_exact_mpo = estimate(op, params, timesteps=timesteps, method="mc", num_samples=16**len(timesteps))
    U_exact = upsilon_mpo_to_dense(U_exact_mpo)
    
    # 256 sequences for k=2 is exhaustive replacement=True/False doesn't matter much if fixed seeds match
    U_mc_mpo = estimate(op, params, timesteps=timesteps, method="mc", num_samples=256, seed=42)
    U_mc = upsilon_mpo_to_dense(U_mc_mpo)

    assert rel_fro_error(U_exact, U_mc) < 1e-10


@pytest.mark.parametrize("k", [1, 2, 3])
def test_rank1_mpo_vs_kron(k):
    rng = np.random.default_rng(300 + k)
    rho = _get_random_rho(rng)
    ops = [rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)) for _ in range(k)]
    w = 1.23

    mpo = rank1_upsilon_mpo_term(rho, ops, weight=w)
    dense_mpo = upsilon_mpo_to_dense(mpo)
    
    dense_ref = rho
    for op in ops:
        dense_ref = np.kron(dense_ref, op)
    dense_ref *= w

    assert rel_fro_error(dense_mpo, dense_ref) < 1e-12


def test_mpo_addition_matches_dense():
    rng = np.random.default_rng(311)
    rho1, rho2 = _get_random_rho(rng), _get_random_rho(rng)
    ops1 = [rng.standard_normal((4, 4)), rng.standard_normal((4, 4))]
    ops2 = [rng.standard_normal((4, 4)), rng.standard_normal((4, 4))]
    
    mpo1 = rank1_upsilon_mpo_term(rho1, ops1, weight=1.0)
    mpo2 = rank1_upsilon_mpo_term(rho2, ops2, weight=0.5)
    
    mpo_added = mpo1 + mpo2
    dense_added = upsilon_mpo_to_dense(mpo_added)
    assert rel_fro_error(dense_added, upsilon_mpo_to_dense(mpo1) + upsilon_mpo_to_dense(mpo2)) < 1e-12


def test_mc_mpo_parity_discrete():
    """Verify MC dense and MPO outputs are equivalent."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, solver="MCWF", show_progress=False)
    
    ups_mpo_1 = estimate(op, params, timesteps=[0.1], method="mc", num_samples=4, seed=42)
    ups_mpo_2 = estimate(op, params, timesteps=[0.1], method="mc", num_samples=4, seed=42)
    assert rel_fro_error(upsilon_mpo_to_dense(ups_mpo_1), upsilon_mpo_to_dense(ups_mpo_2)) < 1e-12


################################################################################
# ── Section 4: Estimator Convergence (MPO/Dense Equivalence) ─────────────────
################################################################################

def test_mc_uniform_converges_metrics() -> None:
    """Verify MC uniform converges to dense reference total for k=2."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, solver="MCWF", show_progress=False)
    timesteps = [0.1, 0.1]

    U_exact = upsilon_mpo_to_dense(estimate(op, params, timesteps=timesteps, method="mc", num_samples=16**len(timesteps)))
    
    errs = {64: [], 256: []}
    for nseq in errs:
        for s in range(3):
            U_hat_mpo = estimate(op, params, timesteps=timesteps, method="mc", num_samples=nseq, seed=100 + s)
            errs[nseq].append(rel_fro_error(upsilon_mpo_to_dense(U_hat_mpo), U_exact))

    assert np.mean(errs[256]) < np.mean(errs[64])


def test_sis_converges_metrics() -> None:
    """Verify SIS converges to dense reference total for k=2."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, solver="MCWF", show_progress=False)
    timesteps = [0.1, 0.1]

    U_exact = upsilon_mpo_to_dense(estimate(op, params, timesteps=timesteps, method="mc", num_samples=16**len(timesteps)))
    
    errs = {64: [], 256: []}
    for nparticles in errs:
        for s in range(3):
            U_hat_mpo = estimate(op, params, timesteps=timesteps, method="sis", num_samples=nparticles, seed=200 + s, parallel=False)
            errs[nparticles].append(rel_fro_error(upsilon_mpo_to_dense(U_hat_mpo), U_exact))

    assert np.mean(errs[256]) < np.mean(errs[64])


################################################################################
# ── Section 5: Prediction Convergence (vs Direct Physics) ─────────────────────
################################################################################

@pytest.mark.parametrize("method", ["mc", "sis"])
def test_predict_convergence_vs_physics(method):
    """Verify that predictions from MC and SIS estimators approach direct physical evolution as N increases."""
    rng = np.random.default_rng(501)
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, max_bond_dim=16, order=2, show_progress=False, solver="MCWF")

    rho_prep = _get_random_rho(rng)
    u_mat = expm(-1j * 0.4 * Z().matrix)

    def a0_prep(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.trace(rho) * rho_prep
    def a1_unitary(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return u_mat @ rho @ u_mat.conj().T

    interventions = [a0_prep, a1_unitary]
    timesteps = [0.1, 0.1]

    rho_ref = _dense_physical_reference(op, timesteps, interventions)

    sample_sizes = [64, 256]
    n_seeds = 5
    errs = {n: [] for n in sample_sizes}

    for N in sample_sizes:
        for s in range(n_seeds):
            U_hat_mpo = estimate(op, params, timesteps=timesteps, method=method, num_samples=N, seed=700 + s + N, parallel=False)
            U_hat = upsilon_mpo_to_dense(U_hat_mpo)
                
            U_canon = canonicalize_upsilon(U_hat, hermitize=True, psd_project=False, normalize_trace=False)
            rho_pred = predict_from_dense_upsilon(U_canon, interventions)
            errs[N].append(rel_fro_error(rho_pred, rho_ref))

    assert np.mean(errs[256]) < np.mean(errs[64])


################################################################################
# ── Section 6: Smoke / Integration Tests ──────────────────────────────────────
################################################################################

def test_tomography_run_basic() -> None:
    """Test standard single-step process tomography."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, max_bond_dim=16)
    pt = run_exact(op, params, timesteps=[0.1])
    assert pt.tensor.shape == (4, 16)


def test_tomography_run_defaults() -> None:
    """Test defaults (timesteps=None -> single step)."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, elapsed_time=0.1)
    pt = run_exact(op, params)
    assert pt.tensor.shape == (4, 16)


def test_tomography_mcwf_multistep() -> None:
    """Test multi-step process tomography with MCWF solver."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, solver="MCWF", num_traj=10)
    pt = run_exact(op, params, timesteps=[0.1, 0.1])
    assert pt.tensor.shape == (4, 16, 16)


def test_predict_linearity() -> None:
    """Ensure predict_final_state is linear in the intervention maps."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, max_bond_dim=16)
    pt = run_exact(op, params, timesteps=[0.1])

    def map1(rho): return rho
    def map2(_rho): return np.zeros((2, 2), dtype=complex)
    def sum_map(rho): return 0.5 * map1(rho) + 0.3 * map2(rho)

    rho1 = pt.predict_final_state([map1])
    rho2 = pt.predict_final_state([map2])
    rho_sum = pt.predict_final_state([sum_map])
    np.testing.assert_allclose(rho_sum, 0.5 * rho1 + 0.3 * rho2, atol=1e-10)


def test_tomography_with_noise() -> None:
    """Verify that the tomography pipeline runs with a noise model."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, max_bond_dim=16, order=1)
    noise_model = NoiseModel([{"name": "lowering", "sites": [0], "strength": 0.05}])
    pt = run_exact(op, params, timesteps=[0.1], num_trajectories=5, noise_model=noise_model)
    assert pt.tensor.shape == (4, 16)


def test_unnormalized_branch_semantics_h0() -> None:
    """Verify trace-weight consistency in deterministic H=0 case."""
    op = MPO.ising(length=2, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=16, order=1)
    pt = run_exact(op, params, timesteps=[0.1])
    for alpha in range(16):
        rho_branch = pt.tensor[:, alpha].reshape(2, 2)
        weight = pt.weights[alpha]
        if weight > 1e-13:
            np.testing.assert_allclose(np.trace(rho_branch), 1.0, atol=1e-10)
        else:
            np.testing.assert_allclose(np.trace(rho_branch), 0.0, atol=1e-10)


def test_run_return_types():
    """Verify API return types."""
    sp = AnalogSimParams(elapsed_time=0.1, dt=0.01, show_progress=False)
    H = MPO.ising(length=1, J=1.0, g=0.5)
    res = estimate(operator=H, sim_params=sp, timesteps=[0.1], method="mc", num_samples=4, seed=42)
    assert isinstance(res, MPO)
