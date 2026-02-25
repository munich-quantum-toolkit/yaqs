import itertools
import pathlib
import sys

sys.path.append(str(pathlib.Path("src").resolve()))

import matplotlib.pyplot as plt
import numpy as np

from mqt.yaqs.characterization.tomography.tomography import run
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def calculate_holevo_map(num_sites: int, gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # ----- model -----
    J = 1.0
    g = 1.0  # transverse field (hx = 1.0)
    operator = MPO.ising(num_sites, J=J, g=g)

    # ----- simulation params -----
    # To get exact results matching Mathematica, we use TJM with max_bond_dim=4
    # which is exact for L=2. Here we keep it max_bond_dim=4.
    dt = 0.5
    ntraj = 10
    sim_params = AnalogSimParams(
        dt=dt, num_traj=ntraj, max_bond_dim=4, order=2, get_state=True, solver="MCWF", show_progress=False
    )

    noise_model = NoiseModel([
        {"name": name, "sites": [i], "strength": gamma} for i in range(1, num_sites) for name in ["lowering"]
    ])

    # ----- scan settings -----
    T1_max = 6.0
    T2_max = 6.0
    
    # We use step=0.2 for a 51x51 resolution. Change to 0.1 for 101x101 (exact match to Mathematica notebook).
    # Setting to 0.5 as in the original provided file to keep test relatively fast.
    step1 = 0.5
    step2 = 0.5

    t1_grid = np.arange(0.0, T1_max + 1e-12, step1)
    t2_grid = np.arange(0.0, T2_max + 1e-12, step2)
    n1 = len(t1_grid)
    n2 = len(t2_grid)

    # Memory probe choice:
    # fix_step=1 means: fix later injection (step 1), vary earlier injection (step 0)
    fixed_step = 1
    fixed_idx = 0  # "zeros" state at the intermediate step
    base = np.e  # Use natural log to match Mathematica's S[x] := -x * Log[x]

    # ----- caching -----
    cache_file = f"holevo_heatmap_cache_L{num_sites}_g{gamma}.npz"
    if pathlib.Path(cache_file).exists():
        data = np.load(cache_file, allow_pickle=True)
        chi_map = data["chi_map"]
        done = data["done"].astype(bool)

        # Check grid shapes match
        if chi_map.shape != (n1, n2):
            print(f"Cache shape mismatch for L={num_sites}, gamma={gamma}. Re-calculating.")
            chi_map = np.full((n1, n2), np.nan, dtype=float)
            done = np.zeros((n1, n2), dtype=bool)
        else:
            inconsistent = done & np.isnan(chi_map)
            done[inconsistent] = False
    else:
        chi_map = np.full((n1, n2), np.nan, dtype=float)  # rows: t1, cols: t2
        done = np.zeros((n1, n2), dtype=bool)

    # ----- scan loop -----
    print(f"Starting {n1}x{n2} grid scan for L={num_sites}, gamma={gamma}...")
    for i in range(n1):
        t1 = t1_grid[i]
        for j in range(n2):
            print("Grid point:", i, j)
            if done[i, j]:
                continue

            t2 = t2_grid[j]
            # t2 is the first evolution duration (t), t1 is the second evolution duration (tau).
            # init -> t2 -> inject -> t1 -> output.
            timesteps = [float(t2), float(t1)]

            # To speed up the exact case, we restrict intermediate bases to Z
            pt = run(
                operator=operator,
                sim_params=sim_params,
                timesteps=timesteps,
                num_trajectories=ntraj,
                noise_model=noise_model,
                measurement_bases=["Z"],
            )

            chi = pt.holevo_information_conditional(fixed_step=fixed_step, fixed_idx=fixed_idx, base=base)
            chi_map[i, j] = chi
            done[i, j] = True

            if (i * n2 + j) % 10 == 0:
                np.savez(cache_file, t1_grid=t1_grid, t2_grid=t2_grid, chi_map=chi_map, done=done)

    # Final save
    np.savez(cache_file, t1_grid=t1_grid, t2_grid=t2_grid, chi_map=chi_map, done=done)
    print(f"Scan complete for L={num_sites}, gamma={gamma}.")

    return t1_grid, t2_grid, chi_map


def main() -> None:
    gammas = [0.001, 0.01, 0.1]
    system_sizes = [2, 3, 4]
    
    # 1) Run simulations and gather data
    results = {}
    for gamma in gammas:
        for L in system_sizes:
            print("gamma:", gamma, "L:", L)
            results[(gamma, L)] = calculate_holevo_map(L, gamma)

    # 2) Plotting (Contours, FFT, Log-FFT)
    plot_types = ["contours", "fft", "fft_log"]

    for plot_type in plot_types:
        fig, axes = plt.subplots(len(gammas), len(system_sizes), figsize=(14, 12), sharex=True, sharey=True)
        
        # Calculate global min and max for normalization
        plot_data_dict = {}
        global_min = np.inf
        global_max = -np.inf
        
        for gamma in gammas:
            for L in system_sizes:
                t1_grid, t2_grid, chi_map = results[(gamma, L)]
                
                if plot_type == "contours":
                    plot_data = chi_map
                    cbar_label = "Conditional Holevo Information"
                elif plot_type == "fft" or plot_type == "fft_log":
                    F = np.fft.fftshift(np.fft.fft2(chi_map))
                    Fmag = np.abs(F)
                    if plot_type == "fft_log":
                        plot_data = np.log10(Fmag + 1e-12)
                        cbar_label = "log10(|FFT2(Z)|)"
                    else:
                        plot_data = Fmag
                        cbar_label = "|FFT2(Z)|"
                
                plot_data_dict[(gamma, L)] = plot_data
                global_min = min(global_min, np.nanmin(plot_data))
                global_max = max(global_max, np.nanmax(plot_data))

        for i, gamma in enumerate(gammas):
            for j, L in enumerate(system_sizes):
                ax = axes[i, j]
                t1_grid, t2_grid, chi_map = results[(gamma, L)]
                plot_data = plot_data_dict[(gamma, L)]
                
                n1 = len(t1_grid)
                n2 = len(t2_grid)
                
                if plot_type == "contours":
                    im = ax.imshow(
                        plot_data,
                        origin="lower",
                        aspect="auto",
                        extent=[t2_grid[0], t2_grid[-1], t1_grid[0], t1_grid[-1]],
                        interpolation="none",
                        vmin=global_min,
                        vmax=global_max
                    )
                elif plot_type == "fft" or plot_type == "fft_log":
                    dt1 = float(t1_grid[1] - t1_grid[0])
                    dt2 = float(t2_grid[1] - t2_grid[0])
                    
                    f1 = np.fft.fftshift(np.fft.fftfreq(n1, d=dt1))
                    f2 = np.fft.fftshift(np.fft.fftfreq(n2, d=dt2))
                        
                    im = ax.imshow(
                        plot_data,
                        origin="lower",
                        aspect="auto",
                        extent=[f2[0], f2[-1], f1[0], f1[-1]],
                        interpolation="none",
                        vmin=global_min,
                        vmax=global_max
                    )

                ax.set_title(f"L = {L}, $\\gamma = {gamma}$")
                
                if i == len(gammas) - 1:
                    ax.set_xlabel("t2 (first step)" if plot_type == "contours" else "frequency along t2")
                if j == 0:
                    ax.set_ylabel("t1 (second step)" if plot_type == "contours" else "frequency along t1")
                
        plt.suptitle(f"{plot_type.replace('_', ' ').title().replace('Fft', 'FFT')} Grid (Rows: $\\gamma$, Cols: L)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label=cbar_label)
        plt.savefig(f"holevo_multi_grid_{plot_type}.png", dpi=200)
        plt.close(fig)

    # 3) Plotting Max Memory vs L and vs Gamma
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Max memory vs L (for fixed gammas)
    for gamma in gammas:
        max_mems = [np.nanmax(results[(gamma, L)][2]) for L in system_sizes]
        axes[0].plot(system_sizes, max_mems, marker='o', label=f"$\\gamma = {gamma}$")
    
    axes[0].set_xlabel("System Size (L)")
    axes[0].set_ylabel("Max Conditional Memory")
    axes[0].set_title("Max Memory vs. System Size")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.7)
    axes[0].set_xticks(system_sizes)

    # Plot 2: Max memory vs Gamma (for fixed Ls)
    for L in system_sizes:
        max_mems = [np.nanmax(results[(gamma, L)][2]) for gamma in gammas]
        axes[1].plot(gammas, max_mems, marker='s', label=f"L = {L}")

    axes[1].set_xlabel(r"Noise Strength ($\gamma$)")
    axes[1].set_ylabel("Max Conditional Memory")
    axes[1].set_title("Max Memory vs. Noise Strength")
    axes[1].set_xscale("log")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("holevo_max_memory_scaling.png", dpi=200)
    plt.close(fig)

    print("All multi-plots generated successfully.")


if __name__ == "__main__":
    main()
