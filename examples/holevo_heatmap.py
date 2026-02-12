import sys, os
sys.path.append(os.path.abspath("src"))

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.tomography.tomography import run_process_tensor_tomography
from mqt.yaqs.core.libraries.gate_library import X, Y, Z


# ---------------------------
# helpers
# ---------------------------

def vec_to_rho(vec4: np.ndarray) -> np.ndarray:
    rho = vec4.reshape(2, 2)
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho)
    if abs(tr) > 1e-15:
        rho = rho / tr
    return rho

def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    # 0.5 * ||rho - sigma||_1 ; for 2x2 Hermitian delta use eigenvalues
    delta = rho - sigma
    delta = 0.5 * (delta + delta.conj().T)
    w = np.linalg.eigvalsh(delta)
    return 0.5 * float(np.sum(np.abs(w)))

def mean_pairwise_trace_distance(rhos: list[np.ndarray]) -> float:
    dists = []
    for i in range(len(rhos)):
        for j in range(i + 1, len(rhos)):
            dists.append(trace_distance(rhos[i], rhos[j]))
    return float(np.mean(dists)) if dists else 0.0

def conditional_memory_distance_two_step(pt, fixed_step: int, fixed_idx: int) -> float:
    """
    pt.tensor shape (4, N, N). Fix one step, vary the other uniformly.
    Return mean pairwise trace distance of the resulting outputs.
    """
    N = pt.tensor.shape[1]
    assert pt.tensor.ndim - 1 == 2, "Expected 2-step PT tensor (4,N,N)."

    seqs = [seq for seq in itertools.product(range(N), repeat=2) if seq[fixed_step] == fixed_idx]
    rhos = [vec_to_rho(pt.tensor[(slice(None),) + seq]) for seq in seqs]
    return mean_pairwise_trace_distance(rhos)


# ---------------------------
# main scan
# ---------------------------

def main():
    # ----- model -----
    num_sites = 2
    J = 1.0
    g = 1.0
    operator = MPO.ising(num_sites, J=J, g=g)

    # ----- simulation params -----
    dt = 0.1
    sim_params = AnalogSimParams(dt=dt, num_traj=1, get_state=True)

    # ----- scan settings -----
    T_max = 10
    step = 1
    t_vals = np.round(np.arange(0.0, T_max + 1e-12, step))
    n = len(t_vals)

    # Selective settings
    ntraj = 5  # increase if noisy
    mode = "selective"

    # Memory probe choice:
    # fix_step=1 means: fix later injection (step 1), vary earlier injection (step 0)
    fixed_step = 1
    fixed_idx = 0  # "zeros" at the fixed step; try 2 or 4 as well
    base = 2

    # ----- caching -----
    cache_file = "selective_memory_grid_cache.npz"
    if os.path.exists(cache_file):
        data = np.load(cache_file, allow_pickle=True)
        chi_map = data["chi_map"]
        dist_map = data["dist_map"]
        done = data["done"].astype(bool)
        
        # Consistency check: If marked done but value is NaN, mark as not done
        nan_mask = np.isnan(chi_map)
        inconsistent = done & nan_mask
        if np.any(inconsistent):
            print(f"Fixing cache: Found {np.sum(inconsistent)} points marked 'done' but with NaN values. Resetting status.")
            done[inconsistent] = False
            
        print(f"Loaded cache: {cache_file} (done {done.sum()}/{done.size})")
    else:
        chi_map = np.full((n, n), np.nan, dtype=float)   # rows: t2 index, cols: t1 index
        dist_map = np.full((n, n), np.nan, dtype=float)
        done = np.zeros((n, n), dtype=bool)

    # ----- scan loop -----
    for i in range(n):
        t1 = t_vals[i]
        for j in range(n):
            t2 = t_vals[j]
            if done[j, i]:
                continue

            timesteps = [float(t1), float(t2)]
            print(f"\n=== point t1={t1:.1f}, t2={t2:.1f} ===")

            pt = run_process_tensor_tomography(
                operator=operator,
                sim_params=sim_params,
                timesteps=timesteps,
                num_trajectories=ntraj,
                mode=mode,
            )

            chi = pt.holevo_information_conditional(
                fixed_step=fixed_step,
                fixed_idx=fixed_idx,
                base=base
            )
            dtr = conditional_memory_distance_two_step(pt, fixed_step=fixed_step, fixed_idx=fixed_idx)

            chi_map[j, i] = chi
            dist_map[j, i] = dtr
            done[j, i] = True

            np.savez(cache_file, t_vals=t_vals, chi_map=chi_map, dist_map=dist_map, done=done)
            print(f"χ={chi:.6e} bits,  mean-pairwise-TrDist={dtr:.6e}   (cached)")

    # ----- plotting -----
    def plot_heat(M, title, fname, cbar_label):
        plt.figure(figsize=(7.8, 6.2))
        plt.imshow(
            M,
            origin="lower",
            extent=[t_vals[0] - step/2, t_vals[-1] + step/2, t_vals[0] - step/2, t_vals[-1] + step/2],
            aspect="auto",
            # norm=LogNorm(vmin=1e-2, vmax=1e0)
        )
        plt.colorbar(label=cbar_label)
        plt.xlabel("t1")
        plt.ylabel("t2")
        plt.title(title)
        plt.xticks(t_vals)
        plt.yticks(t_vals)
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        print(f"Saved: {fname}")

    plot_heat(
        chi_map,
        title=f"Selective conditional Holevo χ (bits)\nfix_step={fixed_step}, fixed_idx={fixed_idx}, ntraj={ntraj}, dt={dt}",
        fname="selective_memory_grid_holevo.png",
        cbar_label="χ (bits)",
    )

    plot_heat(
        dist_map,
        title=f"Selective mean pairwise trace distance\nfix_step={fixed_step}, fixed_idx={fixed_idx}, ntraj={ntraj}, dt={dt}",
        fname="selective_memory_grid_tracedist.png",
        cbar_label="mean pairwise trace distance",
    )

    # Optional: overlay the t1+t2=1 diagonal on the heatmaps by printing the indices
    # (If you want it drawn, we can add a line in axes coordinates too.)

if __name__ == "__main__":
    main()
