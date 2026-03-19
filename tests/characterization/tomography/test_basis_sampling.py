from __future__ import annotations

import numpy as np

from mqt.yaqs.characterization.tomography.basis import (
    calculate_dual_choi_basis,
    dual_norm_metrics,
    get_basis_states,
    get_choi_basis,
)
from mqt.yaqs.characterization.tomography.process_tomography import run
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def test_estimate_exact_when_all_sequences_selected_k2() -> None:
    op = MPO.ising(length=2, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, solver="TJM", show_progress=False, max_bond_dim=16)
    timesteps = [0.1, 0.1]
    k = len(timesteps)
    n_all = 16**k

    comb_ex = run(op, params, timesteps=timesteps, method="exhaustive", output="dense", parallel=False)
    comb_bs = run(
        op,
        params,
        timesteps=timesteps,
        method="estimate",
        output="dense",
        parallel=False,
        num_samples=n_all,
        seed=123,
    )

    np.testing.assert_allclose(comb_bs.to_matrix(), comb_ex.to_matrix(), atol=1e-12)


def test_estimate_inclusion_corrected_is_unbiased_in_expectation_k1() -> None:
    # Deterministic backend (TJM) so only sampling randomness remains.
    op = MPO.ising(length=2, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, solver="TJM", show_progress=False, max_bond_dim=16)
    timesteps = [0.1]
    k = len(timesteps)
    n_all = 16**k

    comb_ex = run(op, params, timesteps=timesteps, method="exhaustive", output="dense", parallel=False)
    U_ex = comb_ex.to_matrix()

    n_pick = 8  # < 16
    # Horvitz–Thompson reweighting is unbiased but high-variance in the operator norm;
    # need enough subsamples so the empirical mean is close to exhaustive.
    n_seeds = 120
    Us = []
    for s in range(n_seeds):
        comb_bs = run(
            op,
            params,
            timesteps=timesteps,
            method="estimate",
            output="dense",
            parallel=False,
            num_samples=n_pick,
            seed=1000 + s,
        )
        Us.append(comb_bs.to_matrix())

    U_mean = np.mean(np.stack(Us, axis=0), axis=0)
    # Monte Carlo on subsets: tolerance scales ~1/sqrt(n_seeds) for entrywise error.
    np.testing.assert_allclose(U_mean, U_ex, atol=8e-2, rtol=8e-2)


def test_dual_norm_metrics_finite_for_all_bases() -> None:
    for basis_name, seed in [
        ("standard", None),
        ("tetrahedral", None),
        ("random", 12345),
    ]:
        choi_basis, _ = get_choi_basis(basis=basis_name, seed=seed)
        duals = calculate_dual_choi_basis(choi_basis)
        dn = dual_norm_metrics(duals)

        assert np.isfinite(dn["mean_dual_norm"])
        assert np.isfinite(dn["max_dual_norm"])


def test_tetrahedral_basis_states_are_physical_and_frame_is_full_rank() -> None:
    states = get_basis_states(basis="tetrahedral")
    assert len(states) == 4
    for _name, psi, rho in states:
        np.testing.assert_allclose(np.linalg.norm(psi), 1.0, atol=1e-12)
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-12)
        np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-12)
        # rank-1 projector: Tr(rho^2) == 1
        np.testing.assert_allclose(np.real(np.trace(rho @ rho)), 1.0, atol=1e-10)

    choi_basis, _ = get_choi_basis(basis="tetrahedral")
    frame_matrix = np.column_stack([m.reshape(-1) for m in choi_basis])
    s = np.linalg.svd(frame_matrix, compute_uv=False)
    assert s[-1] > 1e-12


def test_dual_biorthogonality_all_supported_bases() -> None:
    for basis_name, seed in [
        ("standard", None),
        ("tetrahedral", None),
        ("random", 12345),
    ]:
        choi_basis, _ = get_choi_basis(basis=basis_name, seed=seed)
        duals = calculate_dual_choi_basis(choi_basis)
        for i, d_i in enumerate(duals):
            for j, b_j in enumerate(choi_basis):
                inner = np.trace(d_i.conj().T @ b_j)
                expected = 1.0 if i == j else 0.0
                np.testing.assert_allclose(inner, expected, atol=1e-8)

