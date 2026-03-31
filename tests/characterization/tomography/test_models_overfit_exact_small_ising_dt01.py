#!/usr/bin/env python3
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from mqt.yaqs.characterization.tomography.basis import (
    build_basis_for_fixed_alphabet,
    intervention_from_alpha,
)
from mqt.yaqs.characterization.tomography.ml_dataset import (
    build_rho_prev_rho_target,
    mean_frobenius_mse_rho8,
    trajectory_batch_to_tensors,
)
from mqt.yaqs.characterization.tomography.predictor_encoding import (
    CHOI_FLAT_DIM,
    concat_choi_features,
    pack_rho8,
    random_density_matrix,
    state_prep_map_from_rho,
)
from mqt.yaqs.characterization.tomography.process_tomography import (
    run as tomography_run,
    simulate_backend_trajectory_batch,
)
from mqt.yaqs.characterization.tomography.sequence_models import TransformerComb
from mqt.yaqs.characterization.tomography.tomography_utils import (
    build_initial_psi,
    make_mcwf_static_context,
)
from mqt.yaqs.characterization.tomography.combs import NNComb
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def _build_first_intervention_map(alpha0: object, rho_in: np.ndarray):
    prep_map = state_prep_map_from_rho(rho_in)

    def first_map(sigma: np.ndarray, alpha0: object = alpha0, prep_map: object = prep_map) -> np.ndarray:
        return alpha0(prep_map(sigma))

    return first_map


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
def test_models_overfit_exact_small_ising_dt01() -> None:
    J = 1.0
    g = 1.0
    dt = 0.1
    L = 2
    k = 2

    n = 48
    seed = 0
    rng = np.random.default_rng(seed)

    op = MPO.ising(length=L, J=J, g=g)
    params = AnalogSimParams(dt=float(dt), solver="MCWF", show_progress=False, max_bond_dim=None)
    timesteps = [float(dt)] * k
    static_ctx = make_mcwf_static_context(op, params, noise_model=None)

    (basis_set, _choi_mats, choi_pm_pairs, choi_feat_table) = build_basis_for_fixed_alphabet(
        basis="tetrahedral", basis_seed=0
    )

    comb_ex_no_prep = tomography_run(
        operator=op,
        sim_params=params,
        timesteps=timesteps,
        method="exhaustive",
        output="dense",
        noise_model=None,
        parallel=False,
        num_samples=16**k,
        num_trajectories=1,
        seed=0,
        basis="tetrahedral",
        basis_seed=0,
    )

    rho_ins: list[np.ndarray] = []
    alphas_rows: list[np.ndarray] = []
    psi_pairs_list: list[list[tuple[np.ndarray, np.ndarray]]] = []
    initial_psis: list[np.ndarray] = []
    interventions_list: list[list[object]] = []

    for _ in range(n):
        rho_in = random_density_matrix(rng)
        alphas = rng.integers(0, 16, size=k, dtype=np.int64)
        pm = [choi_pm_pairs[int(a)] for a in alphas]
        pairs = [(basis_set[m][1], basis_set[p][1]) for (p, m) in pm]

        rho_ins.append(rho_in)
        alphas_rows.append(alphas)
        psi_pairs_list.append(pairs)
        initial_psis.append(build_initial_psi(rho_in, length=L, rng=rng, init_mode="eigenstate"))

        alpha0 = intervention_from_alpha(int(alphas[0]), basis_set=basis_set, choi_pm_pairs=choi_pm_pairs)
        first_map = _build_first_intervention_map(alpha0, rho_in)
        rest = [
            intervention_from_alpha(int(a), basis_set=basis_set, choi_pm_pairs=choi_pm_pairs) for a in alphas[1:]
        ]
        interventions_list.append([first_map, *rest])

    # Exact trajectory data (used for Transformer targets)
    samples = simulate_backend_trajectory_batch(
        operator=op,
        sim_params=params,
        timesteps=timesteps,
        psi_pairs_list=psi_pairs_list,
        alphas_rows=alphas_rows,
        initial_psis=initial_psis,
        choi_feat_table=choi_feat_table,
        parallel=False,
        static_ctx=static_ctx,
        context_vec=None,
    )
    rho0_np, E_np, rho_seq_np, _ctx = trajectory_batch_to_tensors(samples)

    # Exact final targets from exhaustive comb (consistency check vs backend)
    y_ex = []
    for ivs in interventions_list:
        rho = comb_ex_no_prep.predict(ivs)  # type: ignore[attr-defined]
        y_ex.append(pack_rho8(rho))
    y_ex_np = np.stack(y_ex, axis=0).astype(np.float32)
    assert mean_frobenius_mse_rho8(rho_seq_np[:, -1, :], y_ex_np) < 5e-8

    # NNComb overfit to exact comb targets.
    x_rows = []
    for rho_in, alphas in zip(rho_ins, alphas_rows, strict=True):
        x_rows.append(np.concatenate([pack_rho8(rho_in), concat_choi_features(alphas, choi_feat_table)], axis=0))
    x_np = np.stack(x_rows, axis=0).astype(np.float32)

    nn = NNComb(in_dim=8 + CHOI_FLAT_DIM * k, hidden=128, out_dim=8)
    nn.fit_features(x_np, y_ex_np, epochs=400, lr=3e-3, batch_size=32, device="cpu", verbose=False)
    y_nn = nn.predict_features(x_np, device="cpu")
    nn_train_frob = mean_frobenius_mse_rho8(y_nn, y_ex_np)
    assert nn_train_frob < 2e-4

    # TransformerComb overfit to exact trajectory labels.
    import torch

    device = torch.device("cpu")
    rho_prev_np, rho_tgt_np = build_rho_prev_rho_target(rho0_np, rho_seq_np)
    E_t = torch.as_tensor(E_np, dtype=torch.float32, device=device)
    prev_t = torch.as_tensor(rho_prev_np, dtype=torch.float32, device=device)
    tgt_t = torch.as_tensor(rho_tgt_np, dtype=torch.float32, device=device)
    rho0_t = torch.as_tensor(rho0_np, dtype=torch.float32, device=device)

    model = TransformerComb(d_e=CHOI_FLAT_DIM, d_rho=8, d_model=128, nhead=4, num_layers=2, dim_ff=256)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    mse = torch.nn.MSELoss()
    model.train()
    for _epoch in range(450):
        opt.zero_grad()
        pred = model(E_t, prev_t)
        loss = mse(pred, tgt_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred_teacher = model(E_t, prev_t).cpu().numpy()
        pred_rho0 = model(E_t, rho0_t).cpu().numpy()

    teacher_final = mean_frobenius_mse_rho8(pred_teacher[:, -1, :], rho_tgt_np[:, -1, :])
    rho0_final = mean_frobenius_mse_rho8(pred_rho0[:, -1, :], rho_tgt_np[:, -1, :])
    assert teacher_final < 5e-4
    assert rho0_final < 2e-3

