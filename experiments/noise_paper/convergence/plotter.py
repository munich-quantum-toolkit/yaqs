from __future__ import annotations

import glob
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Loading helpers
# -------------------------
def load_latest_pickle(prefix: str) -> object:
    """
    Loads the latest (highest suffix) pickle file matching prefix_*.pickle.

    Your loop dumped cumulative lists; this loads the newest file which
    should contain all dp entries.
    """
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
    """
    qutip_loaded is your results3 list over dp.
    Each entry is a 1D list/array over time; we take the final time.
    """
    series = np.asarray(qutip_loaded[j], dtype=float).squeeze()
    if series.ndim != 1:
        raise ValueError(f"Expected QuTiP series to be 1D; got shape {series.shape}")
    return float(series[-1])


# -------------------------
# YAQS trajectory extraction (final time only)
# -------------------------
def extract_final_samples(observable_obj) -> np.ndarray:
    """
    Extract final-time samples per trajectory as a 1D array shape (num_traj,).

    You said: observable.trajectories has shape (1000, 1).
    """
    if not hasattr(observable_obj, "trajectories"):
        raise AttributeError("Observable has no attribute 'trajectories'.")

    arr = np.asarray(getattr(observable_obj, "trajectories"), dtype=float)

    # Accept (num_traj, 1) or (num_traj,)
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError(f"Expected final samples to squeeze to 1D; got shape {arr.shape}")

    return arr


def extract_u_final_samples(u_loaded: list, j: int, obs_index: int = 0) -> np.ndarray:
    """
    u_loaded[j] is a list of Observable objects for dp index j.
    We pick observable obs_index and return final-time samples across trajectories.
    """
    observables_list = u_loaded[j]
    obs = observables_list[obs_index]
    return extract_final_samples(obs)


# -------------------------
# Error vs N (scalar reference)
# -------------------------
def error_vs_N_batched(
    samples: np.ndarray,
    ref_final: float,
    *,
    N_grid: np.ndarray | None = None,
    max_N: int = 1000,
    n_batches: int = 200,
    replace: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each N in N_grid, draw n_batches subsets of size N from 'samples',
    compute mean(subset), then error = |mean - ref_final|.

    Returns:
        N_grid, mean_error, q16_error, q84_error   (quantile band is nicer on log axes)
    """
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples, dtype=float).reshape(-1)
    M = samples.size

    if N_grid is None:
        N_grid = np.unique(
            np.concatenate([
                np.arange(1, min(51, M + 1)),
                np.round(np.logspace(np.log10(50), np.log10(min(max_N, M)), 35)).astype(int),
            ])
        )
    else:
        N_grid = np.asarray(N_grid, dtype=int)

    N_grid = N_grid[(N_grid >= 1) & (N_grid <= min(max_N, M))]

    mean_err = np.empty(N_grid.size, dtype=float)
    q16 = np.empty(N_grid.size, dtype=float)
    q84 = np.empty(N_grid.size, dtype=float)

    for i, N in enumerate(N_grid):
        idx = rng.choice(M, size=(n_batches, N), replace=replace)  # (B, N)
        batch_means = np.mean(samples[idx], axis=1)
        errs = np.abs(batch_means - ref_final)

        mean_err[i] = float(np.mean(errs))
        q16[i] = float(np.quantile(errs, 0.16))
        q84[i] = float(np.quantile(errs, 0.84))

    return N_grid, mean_err, q16, q84

def monotone_envelope_decreasing(y: np.ndarray) -> np.ndarray:
    """Return a non-increasing envelope: y_env[i] = min(y[:i+1])."""
    y = np.asarray(y, dtype=float)
    return np.minimum.accumulate(y)


def N_for_error_threshold(N: np.ndarray, err: np.ndarray, eps_grid: np.ndarray) -> np.ndarray:
    """
    For each eps in eps_grid, return the smallest N such that err(N) <= eps.
    If never achieved, return np.nan.
    """
    N = np.asarray(N, dtype=float)
    err = np.asarray(err, dtype=float)

    # enforce monotone decreasing so inversion is well-defined
    err_m = monotone_envelope_decreasing(err)

    out = np.full(eps_grid.shape, np.nan, dtype=float)
    for i, eps in enumerate(eps_grid):
        idx = np.where(err_m <= eps)[0]
        if idx.size:
            out[i] = N[idx[0]]
    return out


def compute_kappa_on_eps_grid(N: np.ndarray, err_A: np.ndarray, err_B: np.ndarray, eps_grid: np.ndarray):
    errA = monotone_envelope_decreasing(err_A)
    errB = monotone_envelope_decreasing(err_B)

    NA = N_for_error_threshold(N, errA, eps_grid)
    NB = N_for_error_threshold(N, errB, eps_grid)

    kappa = NB / NA
    mask = np.isfinite(kappa) & (kappa > 0)
    return eps_grid[mask], kappa[mask]


# -------------------------
# Main script: load + compute + plot
# -------------------------
if __name__ == "__main__":
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "legend.fontsize": 9,
        "savefig.dpi": 300,
    })

    u1_loaded = load_latest_pickle("convergence_u1")
    u2_loaded = load_latest_pickle("convergence_u2")
    qutip_loaded = load_latest_pickle("convergence_qutip")

    dp_list = [1e-3, 1e-2, 1e-1, 1.0]
    obs_index = 0

    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    def make_N_grid(M: int, max_N: int = 1000) -> np.ndarray:
        max_N = min(max_N, M)
        linear = np.arange(1, min(31, max_N + 1))  # slightly less dense
        log = np.unique(np.round(np.logspace(np.log10(30), np.log10(max_N), 28)).astype(int))
        N = np.unique(np.concatenate([linear, log]))
        return N[(N >= 1) & (N <= max_N)]

    M0 = extract_u_final_samples(u1_loaded, 0, obs_index=obs_index).size
    N_grid = make_N_grid(M0, max_N=1000)

    # --- layout: use subplots_adjust so legend never overlaps ---
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.0), sharex=True, sharey=True)
    axes = axes.ravel()
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.88, wspace=0.10, hspace=0.12)

    # We'll build legend handles once
    legend_handles = None

    # For consistent log plotting
    eps = 1e-3
    # Choose epsilon range you care about for kappa(eps) (global, same for all panels)
    eps_grid_global = np.logspace(-2, -1, 40)  # 1e-3 .. 1e-1

    for j, ax in enumerate(axes):
        dp = dp_list[j]
        ref_final = extract_qutip_final(qutip_loaded, j)

        samples_u1 = extract_u_final_samples(u1_loaded, j, obs_index=obs_index)
        samples_u2 = extract_u_final_samples(u2_loaded, j, obs_index=obs_index)

        N, u1_mean, u1_q16, u1_q84 = error_vs_N_batched(
            samples_u1, ref_final, N_grid=N_grid, n_batches=999, seed=10 + j, replace=True
        )
        _, u2_mean, u2_q16, u2_q84 = error_vs_N_batched(
            samples_u2, ref_final, N_grid=N_grid, n_batches=999, seed=110 + j, replace=True
        )

        l1, = ax.plot(N, u1_mean, lw=1.6, label="Unraveling A")
        l2, = ax.plot(N, u2_mean, lw=1.6, label="Unraveling B")

        ax.fill_between(N, np.maximum(u1_q16, eps), np.maximum(u1_q84, eps), alpha=0.18)
        ax.fill_between(N, np.maximum(u2_q16, eps), np.maximum(u2_q84, eps), alpha=0.18)

        ax.set_xscale("log")
        ax.set_yscale("log")

        # PRX-ish: no top/right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Major grid only, light
        ax.grid(True, which="major", alpha=0.18, linewidth=0.6)
        ax.grid(False, which="minor")

        ax.text(0.03, 0.95, panel_labels[j], transform=ax.transAxes,
                ha="left", va="top", fontweight="bold")
        ax.text(
            0.97, 0.92, rf"$dp = {dp:g}$",
            transform=ax.transAxes,
            ha="right", va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
        )


        if legend_handles is None:
            legend_handles = (l1, l2)

        # Row / column indices
        row = j // 2
        col = j % 2

        # Y label on left column only
        if col == 0:
            ax.set_ylabel(r"Error $\langle XX^{[L/2]}\rangle$")

        # X label on bottom row only
        if row == 1:
            ax.set_xlabel("N trajectories")

        from matplotlib.ticker import LogLocator, LogFormatterMathtext

        # ---- kappa inset: kappa(eps) = N_B/N_A ----
        eps_i, kappa = compute_kappa_on_eps_grid(N, u1_mean, u2_mean, eps_grid_global)

        if eps_i.size:
            inset = ax.inset_axes([0.10, 0.12, 0.38, 0.34])

            inset.plot(eps_i, kappa, ls="--", lw=1.1)

            inset.set_xscale("log")
            # inset.set_yscale("log")

            # Use the SAME x-limits in every inset for comparability
            inset.set_xlim(eps_grid_global.min(), eps_grid_global.max())
            inset.set_ylim(1, 10)
            # Nice log ticks
            inset.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
            inset.xaxis.set_major_formatter(LogFormatterMathtext())
            inset.minorticks_off()

            inset.grid(True, which="major", alpha=0.15, linewidth=0.5)
            inset.tick_params(labelsize=7, length=2, pad=1)

            inset.axhline(1.0, lw=0.8, alpha=0.6)

            inset.text(
                0.03, 0.95, r"$\kappa(\varepsilon)=N_B/N_A$",
                transform=inset.transAxes,
                ha="left", va="top",
                fontsize=7,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.8),
            )


    # global legend in reserved top margin
    fig.legend(legend_handles, ["Unraveling A", "Unraveling B"],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.97))
    plt.xlim(1, 500)
    plt.show()
