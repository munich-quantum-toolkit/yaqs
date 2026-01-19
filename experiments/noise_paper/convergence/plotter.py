from __future__ import annotations

import glob
import os
import pickle
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext


# -------------------------
# I/O helpers
# -------------------------
def load_latest_pickle(prefix: str) -> Any:
    """Load the newest pickle matching f"{prefix}_*.pickle" based on integer suffix."""
    files = sorted(glob.glob(f"{prefix}_*.pickle"))
    if not files:
        raise FileNotFoundError(f"No files found for prefix '{prefix}_*.pickle'")

    def suffix_int(path: str) -> int:
        base = os.path.basename(path)
        num = base.replace(".pickle", "").split("_")[-1]
        return int(num)

    latest = max(files, key=suffix_int)
    with open(latest, "rb") as f:
        return pickle.load(f)


def extract_qutip_final(qutip_loaded: list, j: int) -> float:
    """QuTiP reference: take final time-point of dp index j."""
    series = np.asarray(qutip_loaded[j], dtype=float).squeeze()
    if series.ndim != 1:
        raise ValueError(f"Expected QuTiP series to be 1D; got shape {series.shape}")
    return float(series[-1])


# -------------------------
# YAQS trajectory extraction (final time only)
# -------------------------
def extract_final_samples(observable_obj: Any) -> np.ndarray:
    """Return final-time samples across trajectories as shape (num_traj,)."""
    if not hasattr(observable_obj, "trajectories"):
        raise AttributeError("Observable has no attribute 'trajectories'.")
    arr = np.asarray(getattr(observable_obj, "trajectories"), dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError(f"Expected final samples to squeeze to 1D; got shape {arr.shape}")
    return arr


def extract_u_final_samples(u_loaded: list, j: int, *, obs_index: int = 0) -> np.ndarray:
    """u_loaded[j] is list of Observable objects; pick obs_index and return final samples."""
    obs = u_loaded[j][obs_index]
    return extract_final_samples(obs)


# -------------------------
# Convergence estimation
# -------------------------
def make_N_grid(M: int, *, max_N: int = 1000) -> np.ndarray:
    """Dense at small N, log-spaced up to max_N."""
    max_N = min(max_N, M)
    linear = np.arange(1, min(31, max_N + 1))
    log = np.unique(np.round(np.logspace(np.log10(30), np.log10(max_N), 28)).astype(int))
    N = np.unique(np.concatenate([linear, log]))
    return N[(N >= 1) & (N <= max_N)]


def error_vs_N_batched(
    samples: np.ndarray,
    ref_final: float,
    *,
    N_grid: np.ndarray,
    n_batches: int = 999,
    replace: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each N, draw n_batches resamples of size N, compute mean, and return:
    N, mean(|mean-ref|), q16, q84 (quantile band is nicer on log axes).
    """
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples, dtype=float).reshape(-1)
    M = samples.size
    N_grid = np.asarray(N_grid, dtype=int)
    N_grid = N_grid[(N_grid >= 1) & (N_grid <= M)]

    mean_err = np.empty(N_grid.size, dtype=float)
    q16 = np.empty(N_grid.size, dtype=float)
    q84 = np.empty(N_grid.size, dtype=float)

    for i, N in enumerate(N_grid):
        idx = rng.choice(M, size=(n_batches, N), replace=replace)
        batch_means = np.mean(samples[idx], axis=1)
        errs = np.abs(batch_means - ref_final)
        mean_err[i] = float(np.mean(errs))
        q16[i] = float(np.quantile(errs, 0.16))
        q84[i] = float(np.quantile(errs, 0.84))

    return N_grid, mean_err, q16, q84


def monotone_envelope_decreasing(y: np.ndarray) -> np.ndarray:
    """Non-increasing envelope: y_env[i] = min(y[:i+1])."""
    y = np.asarray(y, dtype=float)
    return np.minimum.accumulate(y)


def N_for_error_threshold(N: np.ndarray, err: np.ndarray, eps_grid: np.ndarray) -> np.ndarray:
    """Smallest N such that err(N) <= eps; NaN if never achieved."""
    N = np.asarray(N, dtype=float)
    err = monotone_envelope_decreasing(err)
    out = np.full(eps_grid.shape, np.nan, dtype=float)
    for i, eps in enumerate(eps_grid):
        idx = np.where(err <= eps)[0]
        if idx.size:
            out[i] = N[idx[0]]
    return out


def compute_kappa_on_eps_grid(
    N: np.ndarray, err_A: np.ndarray, err_B: np.ndarray, eps_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (eps, kappa(eps)=N_B/N_A) for eps where both are finite."""
    NA = N_for_error_threshold(N, err_A, eps_grid)
    NB = N_for_error_threshold(N, err_B, eps_grid)
    kappa = NB / NA
    mask = np.isfinite(kappa) & (kappa > 0)
    return eps_grid[mask], kappa[mask]


# -------------------------
# Plot helpers
# -------------------------
def style_axes_prx(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", alpha=0.18, linewidth=0.6)
    ax.grid(False, which="minor")


def add_kappa_inset(
    ax: plt.Axes,
    *,
    N: np.ndarray,
    err_A: np.ndarray,
    err_B: np.ndarray,
    eps_grid: np.ndarray,
    loc: list[float] = [0.10, 0.14, 0.32, 0.32],
) -> None:
    """Add inset showing kappa(eps)=N_B/N_A."""
    eps_i, kappa = compute_kappa_on_eps_grid(N, err_A, err_B, eps_grid)
    if eps_i.size == 0:
        return

    inset = ax.inset_axes(loc)
    inset.plot(eps_i, kappa, ls="--", lw=1.1)

    inset.set_xscale("log")
    inset.set_xlim(eps_grid.min(), eps_grid.max())
    inset.set_ylim(1, 10)

    inset.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
    inset.xaxis.set_major_formatter(LogFormatterMathtext())
    inset.minorticks_off()

    inset.grid(True, which="major", alpha=0.15, linewidth=0.5)
    inset.tick_params(labelsize=7, length=2, pad=1)
    inset.axhline(1.0, lw=0.8, alpha=0.6)

    inset.text(
        0.03,
        0.95,
        r"$\kappa(\varepsilon)=N_B/N_A$",
        transform=inset.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.8),
    )


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # PRX-ish defaults
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "legend.fontsize": 9,
            "savefig.dpi": 300,
        }
    )

    # Load data
    uA_loaded = load_latest_pickle("convergence_u1")
    uB_loaded = load_latest_pickle("convergence_u2")
    qutip_loaded = load_latest_pickle("convergence_qutip")

    dp_list = [1e-3, 1e-2, 1e-1, 1.0]
    obs_index = 0
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    # Grid for N
    M0 = extract_u_final_samples(uA_loaded, 0, obs_index=obs_index).size
    N_grid = make_N_grid(M0, max_N=1000)

    # Inset epsilon range (global)
    eps_grid_global = np.logspace(-2, -1, 40)  # 1e-2 .. 1e-1
    eps_floor = 1e-3  # floor for log fill-bands

    # ---- Figure*: wider, less tall (still 2x2) ----
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 3.8), sharex=True, sharey=True)
    axes = axes.ravel()
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.13, top=0.83, wspace=0.12, hspace=0.18)

    legend_handles = None

    for j, ax in enumerate(axes):
        dp = dp_list[j]
        ref_final = extract_qutip_final(qutip_loaded, j)

        samples_A = extract_u_final_samples(uA_loaded, j, obs_index=obs_index)
        samples_B = extract_u_final_samples(uB_loaded, j, obs_index=obs_index)

        N, errA, q16A, q84A = error_vs_N_batched(samples_A, ref_final, N_grid=N_grid, seed=10 + j)
        _, errB, q16B, q84B = error_vs_N_batched(samples_B, ref_final, N_grid=N_grid, seed=110 + j)

        # Main curves
        l1, = ax.plot(N, errA, lw=1.6, label="Unraveling A")
        l2, = ax.plot(N, errB, lw=1.6, label="Unraveling B")

        ax.fill_between(N, np.maximum(q16A, eps_floor), np.maximum(q84A, eps_floor), alpha=0.18)
        ax.fill_between(N, np.maximum(q16B, eps_floor), np.maximum(q84B, eps_floor), alpha=0.18)

        ax.set_xscale("log")
        ax.set_yscale("log")
        style_axes_prx(ax)

        # Panel annotations
        ax.text(0.03, 0.95, panel_labels[j], transform=ax.transAxes, ha="left", va="top", fontweight="bold")
        ax.text(
            0.97,
            0.92,
            rf"$dp={dp:g}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
        )

        # Inset: kappa(eps)
        add_kappa_inset(ax, N=N, err_A=errA, err_B=errB, eps_grid=eps_grid_global)

        if legend_handles is None:
            legend_handles = (l1, l2)

        # Axis labels (outer only)
        row, col = divmod(j, 2)
        if col == 0:
            ax.set_ylabel(r"Error $\langle XX^{[L/2]}\rangle$")
        if row == 1:
            ax.set_xlabel(r"$N$ trajectories")

    # Global legend in reserved top margin
    fig.legend(
        legend_handles,
        ["Unraveling A", "Unraveling B"],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )

    # Limits (optional, keep consistent)
    for ax in axes:
        ax.set_xlim(1, 500)

    fig.savefig("convergence.pdf", dpi=300)
    plt.show()
