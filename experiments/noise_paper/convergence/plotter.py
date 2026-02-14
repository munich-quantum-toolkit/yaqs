from __future__ import annotations

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import matplotlib.patheffects as pe

# -------------------------
# Parameters (must match convergence.py)
# -------------------------
dt_list = [0.01, 0.0125, 0.02, 0.025, 0.04, 0.05, 0.1]
gamma_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
gamma_min = 0.01
gamma_max = 50
dp_levels = [1e-3, 1e-2, 1e-1, 1.0]

dt = np.array(dt_list, dtype=float)
g = np.array(gamma_list, dtype=float)

# -------------------------
# Computation: N for error
# -------------------------
def make_N_grid(M: int, *, max_N: int = 1000) -> np.ndarray:
    max_N = min(max_N, M)
    linear = np.arange(1, min(31, max_N + 1))
    log = np.unique(np.round(np.logspace(np.log10(30), np.log10(max_N), 28)).astype(int))
    N = np.unique(np.concatenate([linear, log]))
    return N[(N >= 1) & (N <= max_N)]

def monotone_envelope_decreasing(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    return np.minimum.accumulate(y)

def error_for_N(samples: np.ndarray, ref_val: float, N_grid: np.ndarray, n_batches: int = 1000) -> np.ndarray:
    """Returns mean error for each N."""
    rng = np.random.default_rng(42)
    samples = samples.flatten()
    M = samples.size
    
    mean_err = np.empty(N_grid.size, dtype=float)
    
    for i, N in enumerate(N_grid):
        # Bootstrap n_batches
        idx = rng.choice(M, size=(n_batches, N), replace=True)
        batch_means = np.mean(samples[idx], axis=1)
        errs = np.abs(batch_means - ref_val)
        mean_err[i] = np.mean(errs)
        
    return monotone_envelope_decreasing(mean_err)

def get_N_required(err_vs_N: np.ndarray, N_grid: np.ndarray, target_error: float) -> float:
    """Find smallest N such that error <= target_error."""
    idx = np.where(err_vs_N <= target_error + 1e-3)[0]
    if idx.size > 0:
        return float(N_grid[idx[0]])
    return np.nan

# -------------------------
# Load Data
# -------------------------
def load_grid_N_req(target_error: float = 0.02, obs_index: int = 0):
    """
    Load all pickles, compute N_required grid for U1 and U2.
    """
    
    grid_N1 = np.full((len(gamma_list), len(dt_list)), np.nan, dtype=float)
    grid_N2 = np.full((len(gamma_list), len(dt_list)), np.nan, dtype=float)
    
    gamma_to_idx = {val: i for i, val in enumerate(gamma_list)}
    
    for k, dt_val in enumerate(dt_list):
        # Filenames
        f_u1 = f"convergence_u1_{k}.pickle"
        f_u2 = f"convergence_u2_{k}.pickle"
        f_qu = f"convergence_qutip_{k}.pickle"
        
        if not (Path(f_u1).exists() and Path(f_u2).exists() and Path(f_qu).exists()):
             continue
             
        with open(f_u1, 'rb') as f: res1 = pickle.load(f)
        with open(f_u2, 'rb') as f: res2 = pickle.load(f)
        with open(f_qu, 'rb') as f: resq = pickle.load(f)
        
        # Each res is a list over gamma
        for j, g_val in enumerate(gamma_list):
            if j >= len(res1) or res1[j] is None: continue
            
            # QuTiP reference (last time point)
            # resq[j] is the Z_results list
            # Wait, convergence.py: qutip returns (Z_results, tlist) 
            # OR (cost, tlist) where cost = average? 
            # In convergence.py: cost, tlist = qutip_lindblad_simulator(...)
            # And cost = list(np.real(res.expect[0]))
            # So resq[j] is list of exp vals. We want the last one.
            ref_series = resq[j]
            ref_val = ref_series[-1]
            
            # YAQS samples
            # res1[j] is sim_params.observables (list of Observable)
            # Obs 0 is XX. trajectories is shape (num_traj, time?) or (num_traj,)
            # convergence.py: "sample_timesteps=False" -> only final?
            # actually usually trajectories is (traj, time_steps)
            # check if sample_timesteps=False, results is just final?
            # SimParams defaults: sample_timesteps=False.
            # Then trajectories is typically independent of time, just final values?
            # Or (traj, 1). Let's assume (traj, 1) or (traj,)
            
            obs1 = res1[j][obs_index]
            obs2 = res2[j][obs_index]
            
            s1 = np.array(obs1.trajectories).flatten()
            s2 = np.array(obs2.trajectories).flatten()
            
            # N grid
            M1 = s1.size
            if M1 < 10: continue 
            N_grid = make_N_grid(M1)
            
            # Curves
            err1 = error_for_N(s1, ref_val, N_grid)
            err2 = error_for_N(s2, ref_val, N_grid)
            
            # N req
            n1 = get_N_required(err1, N_grid, target_error)
            n2 = get_N_required(err2, N_grid, target_error)
            
            row = gamma_to_idx[g_val]
            grid_N1[row, k] = n1
            grid_N2[row, k] = n2
            
    return grid_N1, grid_N2

# -------------------------
# Plotting
# -------------------------
def plot_heatmaps(N1, N2, target_error):
    # Setup styling (match practical style)
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
    })
    
    # Figure geometry (match 3-panel geometry)
    fig = plt.figure(figsize=(12, 3.8))
    gs = fig.add_gridspec(
        1, 6,
        left=0.08, right=0.92, bottom=0.2, top=0.85,
        width_ratios=[1.0, 1.0, 0.04, 0.15, 1.0, 0.04], 
        wspace=0.15
    )
    
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    cax_n = fig.add_subplot(gs[0, 2])
    axKappa = fig.add_subplot(gs[0, 4])
    cax_k = fig.add_subplot(gs[0, 5])
    
    # Calculate log-edges for pcolormesh
    def get_log_edges(centers):
        centers = np.asarray(centers)
        log_c = np.log10(centers)
        d_log = np.diff(log_c)
        edges_log = np.concatenate([
            [log_c[0] - d_log[0]/2],
            log_c[:-1] + d_log/2,
            [log_c[-1] + d_log[-1]/2]
        ])
        return 10**edges_log

    dt_edges = get_log_edges(dt_list)
    g_edges = get_log_edges(gamma_list)
    
    # ------------------------
    # 1. N required heatmaps
    # ------------------------
    vmin = max(1, min(np.nanmin(N1), np.nanmin(N2)))
    vmax = max(np.nanmax(N1), np.nanmax(N2))
    norm_n = LogNorm(vmin=vmin, vmax=vmax)
    cmap_n = "magma_r"

    pc_opts = dict(shading="flat", edgecolors="none", antialiased=False)
    
    pcmA = axA.pcolormesh(dt_edges, g_edges, N1, cmap=cmap_n, norm=norm_n, **pc_opts)
    pcmB = axB.pcolormesh(dt_edges, g_edges, N2, cmap=cmap_n, norm=norm_n, **pc_opts)
    
    # ------------------------
    # 2. Kappa heatmap
    # ------------------------
    Kappa = N2 / N1
    norm_k = Normalize(vmin=1, vmax=6) # Matching alpha vmin/vmax range
    cmap_k = plt.get_cmap("plasma").copy()
    cmap_k.set_bad(color="0.85")
    cmap_k.set_under("black")
    
    pcm_k = axKappa.pcolormesh(dt_edges, g_edges, Kappa, cmap=cmap_k, norm=norm_k, **pc_opts)
    
    # Subtle Grid
    for ax in (axA, axB, axKappa):
        ax.grid(which="both", color="w", alpha=0.12, linewidth=0.5)

    # ------------------------
    # Axes & Ticking
    # ------------------------
    from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter
    def set_scientific_log_ticks(ax, axis='x'):
        target = ax.xaxis if axis == 'x' else ax.yaxis
        target.set_major_locator(LogLocator(base=10))
        target.set_major_formatter(LogFormatterMathtext(base=10))
        target.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
        target.set_minor_formatter(NullFormatter())

    for ax, label in zip([axA, axB, axKappa], ["(a)", "(b)", "(c)"]):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Time step $\delta t$")
        set_scientific_log_ticks(ax, 'x')
        set_scientific_log_ticks(ax, 'y')
        ax.tick_params(axis='x', rotation=15)
        
        # Internal Labels
        ax.text(0.04, 0.96, label, transform=ax.transAxes, va="top", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))

    axA.set_ylabel(r"Noise strength $\gamma$")
    axB.tick_params(labelleft=False)
    axKappa.tick_params(labelleft=False)
    
    axA.set_title("Unraveling A")
    axB.set_title("Unraveling B")
    axKappa.set_title("Sampling Inflation")

    # ------------------------
    # Colorbars
    # ------------------------
    cb_n = fig.colorbar(pcmA, cax=cax_n)
    cax_n.set_title(rf"$N(\epsilon={target_error})$", pad=12, fontsize=10)
    # Add specific ticks to N colorbar
    tick_vals = [5, 10, 20, 30, 50]
    cb_n.ax.yaxis.set_major_locator(plt.FixedLocator(tick_vals))
    cb_n.ax.yaxis.set_major_formatter(plt.FixedFormatter([f"{v}" for v in tick_vals]))
    
    cb_k = fig.colorbar(pcm_k, cax=cax_k, ticks=[2.0, 4.0, 6.0, 8.0])
    cax_k.set_title(r"$\kappa = N_B / N_A$", pad=12, fontsize=10)
    
    # ------------------------
    # Stats box for Kappa
    # ------------------------
    valid_kappa = Kappa[~np.isnan(Kappa)]
    mean_k = np.nanmean(valid_kappa) if valid_kappa.size > 0 else 0
    std_k = np.nanstd(valid_kappa) if valid_kappa.size > 0 else 0
    stats_text = rf"$\mu_\kappa = {mean_k:.2f}$"+"\n"+rf"$\sigma_\kappa = {std_k:.2f}$"
    axKappa.text(0.95, 0.95, stats_text, transform=axKappa.transAxes, 
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))

    # Kappa=1 contour
    if np.any(~np.isnan(Kappa)):
        cs_k = axKappa.contour(dt_list, gamma_list, Kappa, levels=[1.0], colors="black", linewidths=1.3, zorder=6)
        axKappa.clabel(cs_k, fmt={1.0: r"$\kappa=1$"}, inline=True, fontsize=8, colors="black")

    # ------------------------
    # DP Lines
    # ------------------------
    def add_dp_lines(ax, annotate=False):
        dt_fine = np.logspace(np.log10(min(dt_list)), np.log10(max(dt_list)), 100)
        dt_label = 0.04
        for dp in dp_levels:
            g_fine = dp / dt_fine
            mask = (g_fine >= min(gamma_list)) & (g_fine <= max(gamma_list))
            if np.any(mask):
                ax.plot(dt_fine[mask], g_fine[mask], "w--", lw=1.1, alpha=0.7, zorder=5)
                if annotate and dp in [1e-3, 1e-2, 1e-1]:
                    g_at_dt = 1.5 * (dp / dt_label)
                    if min(gamma_list) <= g_at_dt <= max(gamma_list):
                        label_text = rf"$\delta p = 10^{{{int(np.log10(dp))}}}$"
                        txt = ax.text(dt_label, g_at_dt, label_text, fontsize=8, rotation=-15, ha='center', va='center',
                                      color='white', weight='bold')
                        txt.set_path_effects([pe.withStroke(linewidth=2, foreground="black", alpha=0.6)])
            
    add_dp_lines(axA, annotate=True)
    add_dp_lines(axB, annotate=False)
    
    plt.savefig("convergence_heatmap.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("convergence_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    target_err = 0.04 # Match user requested epsilon
    N1, N2 = load_grid_N_req(target_error=target_err)
    
    # Check if we have data
    if np.all(np.isnan(N1)):
        print("No data found or all NaNs. Run convergence.py first.")
    else:
        # Diagnostics
        Kappa = N2 / N1
        total_points = Kappa.size
        nan_count = np.count_nonzero(np.isnan(Kappa))
        print(f"Kappa Stats:")
        print(f"  Mean:   {np.nanmean(Kappa):.3f}")
        print(f"  Median: {np.nanmedian(Kappa):.3f}")
        print(f"  Range:  [{np.nanmin(Kappa):.3f}, {np.nanmax(Kappa):.3f}]")
        print(f"  NaNs:   {nan_count} / {total_points} ({100*nan_count/total_points:.1f}%)")
        
        plot_heatmaps(N1, N2, target_err)

