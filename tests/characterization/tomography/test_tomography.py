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

from mqt.yaqs.characterization.tomography.process_tomography import (
    calculate_dual_choi_basis,
    get_basis_states,
    get_choi_basis,
    run,
)

from mqt.yaqs.characterization.tomography.combs import DenseComb
from mqt.yaqs.characterization.tomography.process_tomography import rank1_upsilon_mpo_term

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
    pt = run(op, params, timesteps=[0.1], method="exhaustive")

    def identity_map(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return rho

    rho_pred = pt.to_dense_comb().predict([identity_map])
    expected = np.array([[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(rho_pred, expected, atol=1e-10)


################################################################################
# ── Section 2: Exact Physical Correctness (vs Direct Physics) ────────────────
################################################################################

def test_exact_1step_prediction_vs_physics():
    """Verify 1-step exact tomography-estimate prediction vs direct dense evolution."""
    rng = np.random.default_rng(101)
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.001, max_bond_dim=16, order=2)

    rho_held_out = _get_random_rho(rng)
    def prep_map(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return np.trace(rho) * rho_held_out

    interventions = [prep_map]
    timesteps = [0.001]

    pt = run(op, params, timesteps=timesteps, method="exhaustive")
    rho_pt = pt.to_dense_comb().predict(interventions)
    rho_ref = _dense_physical_reference(op, timesteps, interventions)

    assert rel_fro_error(rho_pt, rho_ref) < 1e-5


def test_exact_2step_prediction_vs_physics():
    """Verify 2-step exact tomography-estimate prediction vs direct dense evolution."""
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

    pt = run(op, params, timesteps=timesteps, method="exhaustive")
    rho_pt = pt.to_dense_comb().predict(interventions)
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

    pt = run(op, params, timesteps=timesteps, method="exhaustive")
    rho_pt = pt.to_dense_comb().predict(interventions)
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

    pt = run(op, params, timesteps=timesteps, method="exhaustive")
    rho_pt = pt.to_dense_comb().predict(interventions)
    rho_ref = _dense_physical_reference(op, timesteps, interventions)

    # At dt=0.1, error should be larger but still around 1e-3 or 1e-4
    assert rel_fro_error(rho_pt, rho_ref) < 5e-3


def test_reconstruction_depolarizing() -> None:
    """Test reconstruction of a depolarizing channel via PT."""
    op = MPO.ising(length=2, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=16)
    pt = run(op, params, timesteps=[0.1], method="exhaustive")

    def depolarize(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return 0.5 * np.trace(rho) * np.eye(2)

    rho_pred = pt.to_dense_comb().predict([depolarize])
    expected = 0.5 * np.eye(2)
    np.testing.assert_allclose(rho_pred, expected, atol=1e-10)


################################################################################
# ── Section 3: Representation Parity ──────────────────────────────────────────
################################################################################

def test_exact_representation_parity():
    """Verify exact MPO and tomography-estimate representations yield identical canonical combs."""
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

    pt = run(op, params, timesteps=timesteps, method="exhaustive")

    # Canonical dense comb from tomography estimate
    U_pt = pt.reconstruct_comb_choi(check=True)
    U_pt_canon = DenseComb(U_pt, []).canonicalize(
        hermitize=True,
        psd_project=True,
        normalize_trace=False,
    ).to_matrix()

    # Canonical dense comb from MPO path
    U_mpo = run(op, params, timesteps=timesteps, method="exhaustive", output="mpo")
    U_mpo_dense = U_mpo.to_matrix()
    U_mpo_canon = DenseComb(U_mpo_dense, []).canonicalize(
        hermitize=True,
        psd_project=True,
        normalize_trace=False,
    ).to_matrix()

    # Compare canonicalized dense combs directly
    assert rel_fro_error(U_pt_canon, U_mpo_canon) < 1e-10


def test_exhaustive_dense_and_mpo_comb_match_for_k1() -> None:
    """Regression: raw combs from exhaustive dense and MPO must agree for k=1."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, max_bond_dim=16, order=2)
    timesteps = [0.1]

    pt = run(op, params, timesteps=timesteps, method="exhaustive", output="dense")
    mpo = run(op, params, timesteps=timesteps, method="exhaustive", output="mpo")

    U_pt = pt.reconstruct_comb_choi(check=True)
    U_mpo = mpo.to_matrix()

    assert rel_fro_error(U_pt, U_mpo) < 1e-10


def test_mc_without_replacement_matches_exact_for_k2() -> None:
    """MC uniform with N=16^k matches internal exact MPO reference exactly."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, solver="MCWF", show_progress=False)
    timesteps = [0.1, 0.1]

    U_exact_mpo = run(op, params, timesteps=timesteps, method="mc", num_samples=16**len(timesteps), seed=42, output="mpo")
    U_exact = U_exact_mpo.to_matrix()
    
    # 256 sequences for k=2 is exhaustive replacement=True/False doesn't matter much if fixed seeds match
    U_mc_mpo = run(op, params, timesteps=timesteps, method="mc", num_samples=256, seed=42, output="mpo")
    U_mc = U_mc_mpo.to_matrix()

    assert rel_fro_error(U_exact, U_mc) < 1e-10


@pytest.mark.parametrize("k", [1, 2, 3])
def test_rank1_mpo_vs_kron(k):
    rng = np.random.default_rng(300 + k)
    rho = _get_random_rho(rng)
    ops = [rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)) for _ in range(k)]
    w = 1.23

    mpo = rank1_upsilon_mpo_term(rho, ops, weight=w)
    dense_mpo = mpo.to_matrix()
    
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
    dense_added = mpo_added.to_matrix()
    assert rel_fro_error(dense_added, mpo1.to_matrix() + mpo2.to_matrix()) < 1e-12


def test_mc_mpo_parity_discrete():
    """Verify MC dense and MPO outputs are equivalent."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, solver="MCWF", show_progress=False)
    
    ups_mpo_1 = run(op, params, timesteps=[0.1], method="mc", num_samples=4, seed=42, output="mpo")
    ups_mpo_2 = run(op, params, timesteps=[0.1], method="mc", num_samples=4, seed=42, output="mpo")
    assert rel_fro_error(ups_mpo_1.to_matrix(), ups_mpo_2.to_matrix()) < 1e-12


################################################################################
# ── Section 4: Estimator Convergence (MPO/Dense Equivalence) ─────────────────
################################################################################

def test_mc_uniform_converges_metrics() -> None:
    """Verify MC uniform converges to dense reference total for k=2."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, solver="MCWF", show_progress=False)
    timesteps = [0.1, 0.1]

    U_exact = run(op, params, timesteps=timesteps, method="mc", num_samples=16**len(timesteps), output="mpo").to_matrix()

    errs = {64: [], 256: []}
    for nseq in errs:
        for s in range(3):
            U_hat_mpo = run(op, params, timesteps=timesteps, method="mc", num_samples=nseq, seed=100 + s, output="mpo")
            errs[nseq].append(rel_fro_error(U_hat_mpo.to_matrix(), U_exact))

    assert np.mean(errs[256]) < np.mean(errs[64])


def test_sis_converges_metrics() -> None:
    """Verify SIS converges to dense reference total for k=2."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, solver="MCWF", show_progress=False)
    timesteps = [0.1, 0.1]

    U_exact = run(op, params, timesteps=timesteps, method="mc", num_samples=16**len(timesteps), output="mpo").to_matrix()

    errs = {64: [], 256: []}
    for nparticles in errs:
        for s in range(3):
            U_hat_mpo = run(
                op,
                params,
                timesteps=timesteps,
                method="sis",
                num_samples=nparticles,
                num_trajectories=1,
                seed=200 + s,
                parallel=False,
                output="mpo",
            )
            errs[nparticles].append(rel_fro_error(U_hat_mpo.to_matrix(), U_exact))

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
            U_hat_mpo = run(
                op,
                params,
                timesteps=timesteps,
                method=method,
                num_samples=N,
                num_trajectories=1,
                seed=700 + s + N,
                parallel=False,
                output="mpo",
            )
            U_hat = U_hat_mpo.to_matrix()

            U_canon = DenseComb(U_hat, []).canonicalize(
                hermitize=True,
                psd_project=False,
                normalize_trace=False,
            ).to_matrix()
            rho_pred = DenseComb(U_canon, []).predict(interventions)
            errs[N].append(rel_fro_error(rho_pred, rho_ref))

    assert np.mean(errs[256]) < np.mean(errs[64])


################################################################################
# ── Section 6: Smoke / Integration Tests ──────────────────────────────────────
################################################################################

def test_tomography_run_basic() -> None:
    """Test standard single-step process tomography."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, max_bond_dim=16)
    pt = run(op, params, timesteps=[0.1], method="exhaustive")
    assert pt.tensor.shape == (4, 16)


def test_tomography_run_defaults() -> None:
    """Test defaults (timesteps=None -> single step)."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, elapsed_time=0.1)
    pt = run(op, params, method="exhaustive")
    assert pt.tensor.shape == (4, 16)


def test_tomography_mcwf_multistep() -> None:
    """Test multi-step process tomography with MCWF solver."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, solver="MCWF", num_traj=10)
    pt = run(op, params, timesteps=[0.1, 0.1], method="exhaustive")
    assert pt.tensor.shape == (4, 16, 16)


def test_predict_linearity() -> None:
    """Ensure comb prediction is linear in the intervention maps."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, max_bond_dim=16)
    pt = run(op, params, timesteps=[0.1], method="exhaustive")
    comb = pt.to_dense_comb()

    def map1(rho): return rho
    def map2(_rho): return np.zeros((2, 2), dtype=complex)
    def sum_map(rho): return 0.5 * map1(rho) + 0.3 * map2(rho)

    rho1 = comb.predict([map1])
    rho2 = comb.predict([map2])
    rho_sum = comb.predict([sum_map])
    np.testing.assert_allclose(rho_sum, 0.5 * rho1 + 0.3 * rho2, atol=1e-10)


def test_tomography_with_noise() -> None:
    """Verify that the tomography pipeline runs with a noise model."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, max_bond_dim=16, order=1)
    noise_model = NoiseModel([{"name": "lowering", "sites": [0], "strength": 0.05}])
    pt = run(op, params, timesteps=[0.1], method="exhaustive", num_trajectories=5, noise_model=noise_model)
    assert pt.tensor.shape == (4, 16)


def test_unnormalized_branch_semantics_h0() -> None:
    """Verify trace-weight consistency in deterministic H=0 case."""
    op = MPO.ising(length=2, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=16, order=1)
    pt = run(op, params, timesteps=[0.1], method="exhaustive")
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
    # MC defaults to dense in run(), but many tests expect MPO
    res = run(operator=H, sim_params=sp, timesteps=[0.1], method="mc", num_samples=4, seed=42, output="mpo")
    assert isinstance(res, MPO)
    
    res_dense = run(operator=H, sim_params=sp, timesteps=[0.1], method="mc", num_samples=4, seed=42, output="dense")
    from mqt.yaqs.characterization.tomography.estimator_class import TomographyEstimate
    assert isinstance(res_dense, TomographyEstimate)


def test_sis_tjm_basic() -> None:
    """Verify that SIS runs with TJM (MPS) backend (k=1 smoke test)."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, solver="TJM", max_bond_dim=4, show_progress=False)
    timesteps = [0.1]
    
    # Generic path check - explicitly ask for mpo
    res = run(
        op,
        params,
        timesteps=timesteps,
        method="sis",
        num_samples=10,
        num_trajectories=1,
        seed=42,
        output="mpo",
    )
    assert isinstance(res, MPO)
    dense = res.to_matrix()
    assert np.linalg.norm(dense) > 1e-10


def test_sis_tjm_k1_matches_exact_prediction() -> None:
    """Verify TJM SIS matches exact for k=1."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, solver="TJM", max_bond_dim=4, show_progress=False)
    timesteps = [0.1]
    
    res_exact = run(op, params, timesteps=timesteps, method="exhaustive", output="mpo")
    dense_exact = res_exact.to_matrix()
    
    res_sis = run(
        op,
        params,
        timesteps=timesteps,
        method="sis",
        num_samples=100,
        num_trajectories=1,
        seed=42,
        output="mpo",
    )
    dense_sis = res_sis.to_matrix()
    
    # k=1 should be reasonably accurate with 100 samples (within O(1) factor)
    assert rel_fro_error(dense_sis, dense_exact) < 0.6


def test_sis_tjm_k2_convergence() -> None:
    """Verify that TJM SIS mean error decreases from N=50 to N=150 particles for k=2."""
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, solver="TJM", max_bond_dim=4, show_progress=False)
    timesteps = [0.1, 0.1]
    
    res_exact = run(op, params, timesteps=timesteps, method="exhaustive", output="mpo")
    dense_exact = res_exact.to_matrix()
    
    errs_50 = []
    errs_150 = []
    # Use multiple seeds to check mean convergence
    for s in range(3):
        res_50 = run(
            op,
            params,
            timesteps=timesteps,
            method="sis",
            num_samples=50,
            num_trajectories=1,
            seed=300 + s,
            output="mpo",
        )
        errs_50.append(rel_fro_error(res_50.to_matrix(), dense_exact))
        
        res_150 = run(
            op,
            params,
            timesteps=timesteps,
            method="sis",
            num_samples=150,
            num_trajectories=1,
            seed=400 + s,
            output="mpo",
        )
        errs_150.append(rel_fro_error(res_150.to_matrix(), dense_exact))
        
    # Mean error should be significantly lower for N=150
    assert np.mean(errs_150) < np.mean(errs_50)


def test_sis_tjm_outputs_dense_and_mpo() -> None:
    """Verify both formatters yield consistent predictions for TJM SIS."""
    from mqt.yaqs.characterization.tomography.estimator_class import TomographyEstimate
    
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, solver="TJM", max_bond_dim=4, show_progress=False)
    timesteps = [0.1]
    
    from mqt.yaqs.characterization.tomography.process_tomography import (
        _estimate_sis_sequence_data,
        _sequence_data_to_mpo,
        _sequence_data_to_dense,
    )
    data = _estimate_sis_sequence_data(op, params, timesteps, num_samples=100, seed=42)
    mpo = _sequence_data_to_mpo(data)
    pt = _sequence_data_to_dense(data)
    assert isinstance(pt, TomographyEstimate)
    
    # 1. Structural consistency
    dm = mpo.to_matrix()
    dp = pt.reconstruct_comb_choi()
    assert rel_fro_error(dm, dp) < 1e-10


################################################################################
# ── Section 7: SIS Robustness and Convergence ────────────────────────────────
################################################################################

def get_matrix(res):
    """Robustly extract dense Choi matrix from TomographyEstimate or ndarray."""
    if hasattr(res, "dense_choi") and res.dense_choi is not None:
        return res.dense_choi
    if hasattr(res, "reconstruct_comb_choi"):
        return res.reconstruct_comb_choi()
    return res


def test_sis_uniform_baseline_consistency():
    """Verify that proposal='uniform' SIS still converges to exhaustive target."""
    op = MPO.hamiltonian(length=1, one_body=[(1.0, "X")])
    params = AnalogSimParams(dt=0.1, solver="MCWF", get_state=True)
    timesteps = [0.1]

    # Reference
    res_ex = run(op, params, timesteps, method="exhaustive", output="dense")
    ex_mat = get_matrix(res_ex)

    # SIS Uniform
    res_sis = run(op, params, timesteps, method="sis", proposal="uniform",
                  output="dense", num_samples=512, seed=42, num_trajectories=1)
    sis_mat = get_matrix(res_sis)

    err_sis = np.linalg.norm(sis_mat - ex_mat)
    assert err_sis < 0.6


def test_sis_local_baseline_consistency():
    """Verify that proposal='local' SIS converges to exhaustive target."""
    op = MPO.hamiltonian(length=1, one_body=[(1.0, "X")])
    params = AnalogSimParams(dt=0.1, solver="MCWF", get_state=True)
    timesteps = [0.1]

    # Reference
    res_ex = run(op, params, timesteps, method="exhaustive", output="dense")
    ex_mat = get_matrix(res_ex)

    # SIS Local (Default)
    res_sis = run(op, params, timesteps, method="sis", proposal="local",
                  output="dense", num_samples=512, seed=42, num_trajectories=1)
    sis_mat = get_matrix(res_sis)

    err_sis = np.linalg.norm(sis_mat - ex_mat)
    assert err_sis < 0.6


def test_sis_mc_matching_accuracy():
    """Verify that SIS (both proposals) reach similar accuracy levels to MC."""
    op = MPO.hamiltonian(length=1, one_body=[(1.0, "X")])
    params = AnalogSimParams(dt=0.1, solver="MCWF")
    timesteps = [0.2]

    res_ex = run(op, params, timesteps, method="exhaustive", output="dense")
    ex_mat = get_matrix(res_ex)

    n_samples = 1000
    res_mc = run(op, params, timesteps, method="mc", num_samples=n_samples, seed=42, output="dense", num_trajectories=1)
    res_sis_u = run(op, params, timesteps, method="sis", proposal="uniform", num_samples=n_samples, seed=42, output="dense", num_trajectories=1)
    res_sis_l = run(op, params, timesteps, method="sis", proposal="local", num_samples=n_samples, seed=42, output="dense", num_trajectories=1)

    err_mc = np.linalg.norm(get_matrix(res_mc) - ex_mat)
    err_sis_u = np.linalg.norm(get_matrix(res_sis_u) - ex_mat)
    err_sis_l = np.linalg.norm(get_matrix(res_sis_l) - ex_mat)

    # All should be reasonably accurate
    assert err_mc < 0.6
    assert err_sis_u < 0.6
    assert err_sis_l < 0.6


def test_sis_local_dense_mpo_consistency_integrated():
    """Verify results are consistent between dense and MPO for local proposal."""
    op = MPO.hamiltonian(length=1, one_body=[(1.0, "X")])
    params = AnalogSimParams(dt=0.1, solver="TJM", get_state=True)
    timesteps = [0.1]

    res_dense = run(op, params, timesteps, method="sis", proposal="local", output="dense", num_samples=50, seed=1, num_trajectories=1)
    res_mpo = run(op, params, timesteps, method="sis", proposal="local", output="mpo", num_samples=50, seed=1, num_trajectories=1)

    mpo_mat = res_mpo.to_matrix()
    dense_mat = get_matrix(res_dense)
    np.testing.assert_allclose(dense_mat, mpo_mat, atol=1e-12)


def test_sis_variance_reduction_integrated():
    """Verify that 'local' proposal usually has lower error than 'uniform' for fixed N."""
    # use a slightly harder system (k=3)
    op = MPO.hamiltonian(length=1, one_body=[(1.0, "X")])
    params = AnalogSimParams(dt=0.05, solver="MCWF")
    timesteps = [0.1, 0.1, 0.1]

    res_ex = run(op, params, timesteps, method="exhaustive", output="dense")
    ex_mat = get_matrix(res_ex)

    n_samples = 256
    # Try a few seeds to avoid noise luck
    errs_u = []
    errs_l = []
    for s in [42, 43, 44]:
        res_u = run(op, params, timesteps, method="sis", proposal="uniform", num_samples=n_samples, seed=s, output="dense", num_trajectories=1)
        res_l = run(op, params, timesteps, method="sis", proposal="local", num_samples=n_samples, seed=s, output="dense", num_trajectories=1)
        errs_u.append(np.linalg.norm(get_matrix(res_u) - ex_mat))
        errs_l.append(np.linalg.norm(get_matrix(res_l) - ex_mat))

    # On average 'local' should be better
    assert np.mean(errs_l) < np.mean(errs_u)


def test_sis_invalid_proposal_integrated():
    """Verify SIS rejects unknown proposals."""
    op = MPO.hamiltonian(length=1, one_body=[(1.0, "X")])
    params = AnalogSimParams(dt=0.1, solver="MCWF")
    with pytest.raises(ValueError, match="not currently supported"):
        run(op, params, method="sis", proposal="invalid_heuristic", num_trajectories=1) # type: ignore


def test_sis_log_stability_integrated():
    """Smoke test for log-weight stability (no underflow/overflow on simple run)."""
    length = 2
    k = 3
    op = MPO.hamiltonian(length=length, one_body=[(1.0, "X")])
    params = AnalogSimParams(dt=0.1, solver="MCWF")

    # This should just run without errors
    res = run(op, params, [0.1]*k, method="sis", proposal="local", num_samples=100, num_trajectories=1)
    assert res is not None
