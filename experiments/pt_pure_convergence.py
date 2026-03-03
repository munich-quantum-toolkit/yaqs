"""experiments/pt_mc_convergence.py
Monte Carlo convergence plots: MC-uniform vs SIS-local vs N^{-1/2} guide.

Compares approximate Upsilon estimators against full-enumeration reference on
a constant-size canonicalized/reduced (8x8) object.

Usage:
    python experiments/pt_mc_convergence.py [--deterministic]

Outputs (in cwd):
    pt_convergence_fro.png   -- Frobenius relative error vs samples
    pt_convergence_qmi.png   -- |QMI_hat - QMI_ref| vs samples

Canonicalization policy (avoids scale ambiguity):
    The target is explicitly the population TOTAL T:
        T = Σ_{α∈[16]^k} p(α) · [ ρ_out(α) ⊗ D(α) ]
    Both MC uniform and SIS naturally target this TOTAL natively.
    For strict comparison, we reduce both to m=1, hermitize, and trace-normalize
    (PSD-projection is OFF to avoid artificially suppressing error noise):

        rho_can = canonicalize_upsilon(
            reduced_upsilon(U_total, k=k, keep_last_m=1),
            hermitize=True, psd_project=False, normalize_trace=True,
        )
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mqt.yaqs.characterization.tomography.tomography import run, run_mc_upsilon, run_sis_upsilon
from mqt.yaqs.characterization.tomography.process_tensor import (
    canonicalize_upsilon,
    comb_qmi_from_upsilon_dense,
    reduced_upsilon,
)
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


# ── helpers ──────────────────────────────────────────────────────────────────

def rel_fro(A: np.ndarray, B: np.ndarray, eps: float = 1e-15) -> float:
    return float(np.linalg.norm(A - B, "fro") / max(np.linalg.norm(B, "fro"), eps))


def canon_for_compare(U: np.ndarray, k: int) -> np.ndarray:
    U_red = reduced_upsilon(U, k=k, keep_last_m=1)
    if abs(np.trace(U_red)) < 1e-12:
        # MC uniform with replace=True might sample exactly 0 valid paths at very low N,
        # yielding exactly the 0 matrix. Avoid division by zero in canonicalize_upsilon
        return np.eye(U_red.shape[0], dtype=np.complex128) / U_red.shape[0]
    return canonicalize_upsilon(
        U_red,
        hermitize=True, psd_project=False, normalize_trace=True,
    )


def make_problem(k: int = 2):
    op = MPO.ising(length=2, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, solver="MCWF", show_progress=False)
    return op, params, [0.1] * k


def dense_ref_total(k: int, op, params, timesteps) -> np.ndarray:
    """Full enumeration reference; returns raw U_ref (population TOTAL T)."""
    U_ref, _ = run_mc_upsilon(
        op, params, timesteps=timesteps,
        num_sequences=16**k, num_trajectories=1,
        noise_model=None, dual_transform="T", replace=False,
    )
    return U_ref


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=2, help="Number of timesteps")
    parser.add_argument("--seeds", type=int, default=10, help="Seeds per sample size")
    args = parser.parse_args()

    k = args.k
    n_seeds = args.seeds
    op, params, timesteps = make_problem(k)

    print(f"Building dense reference (k={k}) TOTAL T…")
    t0 = time.perf_counter()
    U_ref = dense_ref_total(k, op, params, timesteps)
    rho_ref = canon_for_compare(U_ref, k)
    qmi_ref = float(comb_qmi_from_upsilon_dense(rho_ref, assume_canonical=True, past="last"))
    print(f"  done in {time.perf_counter()-t0:.1f}s  |  QMI_ref={qmi_ref:.5f} bits")

    sample_ns = np.array([16, 32, 64, 128, 256])

    def sweep(estimator_fn, label: str, base_seed: int) -> dict:
        fro_mu, fro_sd, qmi_mu, qmi_sd = [], [], [], []
        for N in sample_ns:
            fros, qerrs = [], []
            for s in range(n_seeds):
                U_hat, _ = estimator_fn(int(N), seed=base_seed + s)
                rho_hat = canon_for_compare(U_hat, k)
                fros.append(rel_fro(rho_hat, rho_ref))
                qerrs.append(abs(float(comb_qmi_from_upsilon_dense(rho_hat, assume_canonical=True, past="last")) - qmi_ref))
            fro_mu.append(float(np.mean(fros)))
            fro_sd.append(float(np.std(fros)))
            qmi_mu.append(float(np.mean(qerrs)))
            qmi_sd.append(float(np.std(qerrs)))
            print(f"  {label} N={N:4d}: Fro={fro_mu[-1]:.3e}±{fro_sd[-1]:.2e}  |QMI|={qmi_mu[-1]:.3e}")
        return dict(fro_mu=fro_mu, fro_sd=fro_sd, qmi_mu=qmi_mu, qmi_sd=qmi_sd)

    # ── MC uniform ───────────────────────────────────────────────────────────
    def mc_fn(N: int, seed: int):
        return run_mc_upsilon(
            op, params, timesteps=timesteps,
            num_sequences=N, num_trajectories=1,
            noise_model=None, seed=seed,
            dual_transform="T", replace=False,  # EXACT AT N=16^k
        )

    # ── SIS local ────────────────────────────────────────────────────────────
    def sis_fn(N: int, seed: int):
        return run_sis_upsilon(
            op, params, timesteps=timesteps,
            num_particles=N, noise_model=None, seed=seed,
            proposal="local", floor_eps=0.0,
            stratify_step1=True, resample=True,
            rejuvenate=False, parallel=True,
            dual_transform="T",
        )

    print(f"\n── MC sweep (n_seeds={n_seeds}) ──")
    mc = sweep(mc_fn, "MC-uniform", base_seed=1000)

    print(f"\n── SIS sweep (n_seeds={n_seeds}) ──")
    sis = sweep(sis_fn, "SIS-local", base_seed=2000)

    # ── Reference N^{-1/2} slope ─────────────────────────────────────────────
    ref_x = np.array([sample_ns[0], sample_ns[-1]], dtype=float)
    mc_fro_ref = mc["fro_mu"][0] * np.sqrt(sample_ns[0] / ref_x)
    mc_qmi_ref = mc["qmi_mu"][0] * np.sqrt(sample_ns[0] / ref_x)

    ec = dict(capsize=3, elinewidth=0.8)

    # ── Frobenius plot ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.errorbar(sample_ns, mc["fro_mu"], yerr=mc["fro_sd"],
                fmt="o-", label="MC uniform", **ec)
    ax.errorbar(sample_ns, sis["fro_mu"], yerr=sis["fro_sd"],
                fmt="s--", label="SIS local+stratify", **ec)
    ax.plot(ref_x, mc_fro_ref, "k:", alpha=0.55, label=r"$\propto N^{-1/2}$")
    ax.set_xlabel("Samples $N$ (sequences / particles)")
    ax.set_ylabel("Relative Frobenius error (reduced 8×8, canonicalized)")
    ax.set_title(f"Process-tensor convergence  |  k={k}  |  pure deterministic")
    ax.legend()
    # Clamp zeroes for log plot
    ax.set_ylim(bottom=1e-15)
    ax.grid(True, which="both", alpha=0.35)
    fig.tight_layout()
    fig.savefig("pt_convergence_fro.png", dpi=200)
    plt.close(fig)
    print("\nSaved pt_convergence_fro.png")

    # ── QMI error plot ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.errorbar(sample_ns, mc["qmi_mu"], yerr=mc["qmi_sd"],
                fmt="o-", label="MC uniform", **ec)
    ax.errorbar(sample_ns, sis["qmi_mu"], yerr=sis["qmi_sd"],
                fmt="s--", label="SIS local+stratify", **ec)
    ax.plot(ref_x, mc_qmi_ref, "k:", alpha=0.55, label=r"$\propto N^{-1/2}$")
    ax.set_xlabel("Samples $N$ (sequences / particles)")
    ax.set_ylabel(r"$|$QMI$_\hat{} - $ QMI$_\mathrm{ref}|$ (bits, past='last')")
    ax.set_title(f"QMI convergence  |  k={k}  |  pure deterministic")
    ax.legend()
    # Clamp zeroes for log plot
    ax.set_ylim(bottom=1e-15)
    ax.grid(True, which="both", alpha=0.35)
    fig.tight_layout()
    fig.savefig("pt_convergence_qmi.png", dpi=200)
    plt.close(fig)
    print("Saved pt_convergence_qmi.png")

    # ── Print log-log slopes ──────────────────────────────────────────────────
    # Exclude N=256 for slope fitting because error is EXACTLY 0.0, causing -inf log
    fit_ns = sample_ns[:-1]
    log_N = np.log(fit_ns.astype(float))
    for name, data in [("MC Fro", mc["fro_mu"]), ("SIS Fro", sis["fro_mu"]),
                       ("MC |QMI|", mc["qmi_mu"]), ("SIS |QMI|", sis["qmi_mu"])]:
        slope = float(np.polyfit(log_N, np.log(np.maximum(data[:-1], 1e-15)), 1)[0])
        print(f"  log-log approx slope {name:12s}: {slope:.3f} (till N=128)")


if __name__ == "__main__":
    main()
