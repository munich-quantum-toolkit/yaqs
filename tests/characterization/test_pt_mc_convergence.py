# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Convergence tests: dense vs MC vs SIS process tomography.

Tests verify that approximate estimators (run_mc_upsilon, run_sis_upsilon)
recover the same result as dense enumeration (run + reconstruct_comb_choi)
with MC convergence as samples increase.

Convention
----------
All comparisons use the *canonicalized reduced* Upsilon:

    rho = canonicalize_upsilon(
        reduced_upsilon(U, k=k, keep_last_m=1),
        hermitize=True, psd_project=True, normalize_trace=True,
    )

This removes scale ambiguity (the raw estimators target the population total,
not the mean) and gives a constant-size (8,8) object independent of k.

Note on Deterministic Evolution
-------------------------------
For deterministic/no-noise and k>=2, uniform MC will look broken because the
measure is supported on a tiny set of sequences -> you’re in rare-event land,
not standard CLT scaling. Add tiny noise or use without-replacement enumeration.
"""

from __future__ import annotations

import numpy as np
import pytest
import matplotlib.pyplot as plt

from mqt.yaqs.characterization.tomography.tomography import run, run_mc_upsilon, run_sis_upsilon
from mqt.yaqs.characterization.tomography.process_tensor import (
    canonicalize_upsilon,
    comb_qmi_from_upsilon_dense,
    reduced_upsilon,
)
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.core.data_structures.noise_model import NoiseModel


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
    U_ref, _ = run_mc_upsilon(
        op, params, timesteps=timesteps,
        num_sequences=16**k, num_trajectories=1,
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
            U_hat, _ = run_mc_upsilon(
                op, params, timesteps=timesteps,
                num_sequences=nseq, num_trajectories=1,
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
            U_hat, _ = run_sis_upsilon(
                op, params, timesteps=timesteps,
                num_particles=N, noise_model=None,
                seed=200 + s,
                proposal="local", floor_eps=0.0,
                stratify_step1=True, resample=True,
                rejuvenate=False, parallel=False,
                dual_transform="T",
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
            U_hat, _ = run_sis_upsilon(
                op, params, timesteps=timesteps,
                num_particles=N, noise_model=None,
                seed=300 + s,
                proposal="local", floor_eps=0.0,
                stratify_step1=True, resample=True,
                rejuvenate=False, parallel=False,
                dual_transform="T",
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

    U_hat, _ = run_mc_upsilon(
        op, params, timesteps=timesteps,
        num_sequences=16**k, num_trajectories=1,
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

    U_hat, meta = run_sis_upsilon(
        op, params, timesteps=ts,
        num_particles=16, noise_model=None, seed=0,
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
    U_hat, _ = run_mc_upsilon(
        op, params, timesteps=timesteps,
        num_sequences=2048, num_trajectories=1,
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
