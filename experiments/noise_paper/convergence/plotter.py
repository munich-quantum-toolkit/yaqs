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

def error_for_N(samples: np.ndarray, ref_val: float, N_grid: np.ndarray, n_batches: int = 200) -> np.ndarray:
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
    idx = np.where(err_vs_N <= target_error)[0]
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
    # Setup
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9
    })
    
    fig = plt.figure(figsize=(10.5, 2.6))
    gs = fig.add_gridspec(
        1, 6,
        left=0.06, right=0.92, bottom=0.18, top=0.88,
        width_ratios=[1.0, 1.0, 0.03, 0.12, 1.0, 0.03], 
        wspace=0.08
    )
    
    ax_u1 = fig.add_subplot(gs[0, 0])
    ax_u2 = fig.add_subplot(gs[0, 1])
    cax_n = fig.add_subplot(gs[0, 2])
    
    ax_al = fig.add_subplot(gs[0, 4])
    cax_al = fig.add_subplot(gs[0, 5])
    
    # Grid edges
    dt_edges = np.arange(len(dt_list) + 1, dtype=float) - 0.5
    g_edges = np.arange(len(gamma_list) + 1, dtype=float) - 0.5
    dt_centers = np.arange(len(dt_list), dtype=float)
    g_centers = np.arange(len(gamma_list), dtype=float)
    
    # 1. N required heatmaps
    # Shared norm
    vmin = min(np.nanmin(N1), np.nanmin(N2))
    vmax = max(np.nanmax(N1), np.nanmax(N2))
    # LogNorm for N? N varies from 1 to 1000.
    norm_n = LogNorm(vmin=max(vmin, 1), vmax=vmax)
    
    cmap_n = "viridis_r" # Less N is better -> brighter/better color? Or standard?
    # Usually dark is low, bright is high.
    # If using viridis_r: Yellow (low N, good) -> Purple (high N, bad)
    
    m1 = ax_u1.pcolormesh(dt_edges, g_edges, N1, cmap=cmap_n, norm=norm_n, shading="auto")
    m2 = ax_u2.pcolormesh(dt_edges, g_edges, N2, cmap=cmap_n, norm=norm_n, shading="auto")
    
    # 2. Kappa (N2 / N1) - wait, user said N_B / N_A. U2 is B.
    Kappa = N2 / N1
    
    # Kappa norm
    # typically kappa > 1 means B is worse (needs more traj).
    # kappa < 1 means B is better.
    # Let's center around 1? or just linear/log?
    # Usually we look for improvement.
    k_min, k_max = np.nanmin(Kappa), np.nanmax(Kappa)
    norm_k = Normalize(vmin=1, vmax=max(5, 2)) # Adjust based on data
    
    m3 = ax_al.pcolormesh(dt_edges, g_edges, Kappa, cmap="RdBu_r", norm=norm_k, shading="auto")
    # Red = High Kappa (B needs more = A better). Blue = Low Kappa (B needs less = B better). 
    # Or cividis like practical?
    # user said "alpha plot... what we call a kappa plot".
    # practical.py used cividis.
    
    # Labels
    tick_idx = np.arange(len(dt_list))
    tick_labels = [f"{x:g}" for x in dt_list]
    
    for ax in (ax_u1, ax_u2, ax_al):
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel(r"$\delta t$")
        ax.set_ylim(-0.5, len(gamma_list)-0.5)
        
    # Y labels
    g_tick_idx = np.arange(len(gamma_list))
    g_tick_labels = [f"{x:g}" for x in gamma_list]
    ax_u1.set_yticks(g_tick_idx)
    ax_u1.set_yticklabels(g_tick_labels)
    ax_u1.set_ylabel(r"$\gamma$")
    
    ax_u2.tick_params(labelleft=False)
    ax_al.tick_params(labelleft=False)
    
    # Titles
    ax_u1.set_title(r"$N_{req}$ (Unraveling A)")
    ax_u2.set_title(r"$N_{req}$ (Unraveling B)")
    ax_al.set_title(r"$\kappa = N_B / N_A$")
    
    # Colorbars
    fig.colorbar(m1, cax=cax_n, label="Trajectories")
    fig.colorbar(m3, cax=cax_al, label=r"$\kappa$")
    
    # DP lines
    def add_dp_lines(ax):
        logg = np.log(g)
        dt_dense = np.linspace(dt.min(), dt.max(), 100)
        x_dense = np.interp(dt_dense, dt, dt_centers)
        
        for dp in dp_levels:
            gamma_dense = dp / dt_dense
            # Filter range
            mask = (gamma_dense >= np.min(g)) & (gamma_dense <= np.max(g))
            if not np.any(mask): continue
            
            y_dense = np.interp(np.log(gamma_dense[mask]), logg, g_centers)
            ax.plot(x_dense[mask], y_dense, "k--", lw=0.8, alpha=0.7)
            
    for ax in (ax_u1, ax_u2):
        add_dp_lines(ax)
        
    plt.savefig("convergence_heatmap.pdf", dpi=300)
    plt.show()

if __name__ == "__main__":
    target_err = 0.05 # Example target error
    N1, N2 = load_grid_N_req(target_error=target_err)
    
    # Check if we have data
    if np.all(np.isnan(N1)):
        print("No data found or all NaNs. Run convergence.py first.")
    else:
        plot_heatmaps(N1, N2, target_err)
