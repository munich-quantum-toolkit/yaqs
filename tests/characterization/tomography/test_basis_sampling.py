from __future__ import annotations

import numpy as np

from mqt.yaqs.characterization.tomography.process_tensor.basis import (
    calculate_dual_choi_basis,
    get_basis_states,
    get_choi_basis,
)
from mqt.yaqs.tomography import construct
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def test_construct_exhaustive_self_consistent_k2() -> None:
    op = MPO.ising(length=2, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, solver="TJM", show_progress=False, max_bond_dim=16)
    timesteps = [0.1, 0.1]

    comb_a = construct(op, params, timesteps=timesteps, parallel=False).to_dense_comb()
    comb_b = construct(op, params, timesteps=timesteps, parallel=False).to_dense_comb()

    np.testing.assert_allclose(comb_a.to_matrix(), comb_b.to_matrix(), atol=1e-12)


def test_dual_norm_metrics_finite_for_all_bases() -> None:
    for basis_name, seed in [
        ("standard", None),
        ("tetrahedral", None),
        ("random", 12345),
    ]:
        choi_basis, _ = get_choi_basis(basis=basis_name, seed=seed)
        duals = calculate_dual_choi_basis(choi_basis)
        norms = [float(np.linalg.norm(d, "fro")) for d in duals]
        mean_dual_norm = float(np.mean(norms))
        max_dual_norm = float(np.max(norms))

        assert np.isfinite(mean_dual_norm)
        assert np.isfinite(max_dual_norm)


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

