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


def main() -> None:
    # ----- model -----
    # 2-site Ising model to match the exact diagonalization calculation
    num_sites = 2
    J = 1.0
    g = 1.0  # transverse field (hx = 1.0)
    operator = MPO.ising(num_sites, J=J, g=g)

    # ----- simulation params -----
    # To get exact results matching Mathematica, we use TJM with max_bond_dim=4
    # which is exact for L=2.
    dt = 0.5
    ntraj = 10
    sim_params = AnalogSimParams(dt=dt, num_traj=ntraj, max_bond_dim=4, order=2, get_state=True, solver="MCWF", show_progress=False)
    gamma = 0.1
    noise_model = NoiseModel([
        {"name": name, "sites": [i], "strength": gamma} for i in range(num_sites) for name in ["lowering"]
    ])
    # ----- scan settings -----
    T1_max = 6.0
    T2_max = 6.0
    
    # We use step=0.2 for a 51x51 resolution. Change to 0.1 for 101x101 (exact match to Mathematica notebook, but slower).
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
    cache_file = "holevo_heatmap_cache.npz"
    if pathlib.Path(cache_file).exists():
        data = np.load(cache_file, allow_pickle=True)
        chi_map = data["chi_map"]
        done = data["done"].astype(bool)

        # Check grid shapes match
        if chi_map.shape != (n1, n2):
            print("Cache shape mismatch. Re-calculating.")
            chi_map = np.full((n1, n2), np.nan, dtype=float)
            done = np.zeros((n1, n2), dtype=bool)
        else:
            inconsistent = done & np.isnan(chi_map)
            done[inconsistent] = False
    else:
        chi_map = np.full((n1, n2), np.nan, dtype=float)  # rows: t1, cols: t2
        done = np.zeros((n1, n2), dtype=bool)

    # ----- scan loop -----
    print(f"Starting {n1}x{n2} grid scan...")
    for i in range(n1):
        t1 = t1_grid[i]
        for j in range(n2):
            print(i, j)
            t2 = t2_grid[j]
            if done[i, j]:
                continue

            # As per the Mathematica notebook, t2 is the first evolution duration (t)
            # and t1 is the second evolution duration (tau).
            # We want: init -> t2 -> inject -> t1 -> output.
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
    print("Scan complete.")

    Z = chi_map
    print(f"min(Z) = {np.nanmin(Z):.6f}")
    print(f"max(Z) = {np.nanmax(Z):.6f}")

    # ============================================================
    # Plotting (adapted from user's script)
    # ============================================================

    # --- contour plot ---
    plt.figure(figsize=(7, 6))
    X, Y = np.meshgrid(t2_grid, t1_grid)  # X=t2 (cols), Y=t1 (rows)
    cs = plt.contourf(X, Y, Z, levels=40)
    plt.colorbar(cs, label="Conditional Holevo Information")
    plt.xlabel("t2 (first step)")
    plt.ylabel("t1 (second step)")
    plt.title("Contours of Conditional Memory")
    plt.tight_layout()
    plt.savefig("holevo_contours.png", dpi=200)

    # --- 2D FFT (t1 -> f1 axis, t2 -> f2 axis) ---
    dt1 = float(t1_grid[1] - t1_grid[0])
    dt2 = float(t2_grid[1] - t2_grid[0])

    # Z has shape (n1, n2) with axes (t1, t2).
    F = np.fft.fftshift(np.fft.fft2(Z))
    Fmag = np.abs(F)

    f1 = np.fft.fftshift(np.fft.fftfreq(n1, d=dt1))  # for t1 axis
    f2 = np.fft.fftshift(np.fft.fftfreq(n2, d=dt2))  # for t2 axis

    # --- Heatmap of |FFT2(Z)| ---
    plt.figure(figsize=(7, 6))
    imF = plt.imshow(
        Fmag,
        origin="lower",
        aspect="auto",
        extent=[f2[0], f2[-1], f1[0], f1[-1]],  # x=f2, y=f1
    )
    plt.colorbar(imF, label="|FFT2(Z)|")
    plt.xlabel("frequency along t2 (cycles / unit time)")
    plt.ylabel("frequency along t1 (cycles / unit time)")
    plt.title("2D Fourier transform magnitude")
    plt.tight_layout()
    plt.savefig("holevo_fft.png", dpi=200)

    # --- log-magnitude heatmap ---
    plt.figure(figsize=(7, 6))
    imFlog = plt.imshow(
        np.log10(Fmag + 1e-12),
        origin="lower",
        aspect="auto",
        extent=[f2[0], f2[-1], f1[0], f1[-1]],
    )
    plt.colorbar(imFlog, label="log10(|FFT2(Z)|)")
    plt.xlabel("frequency along t2 (cycles / unit time)")
    plt.ylabel("frequency along t1 (cycles / unit time)")
    plt.title("log-magnitude 2D FFT")
    plt.tight_layout()
    plt.savefig("holevo_fft_log.png", dpi=200)


if __name__ == "__main__":
    main()
