"""Educational analog Ising simulation that saves observable dynamics to .npy.

Edit the parameters in the CONFIG section, then run this file.
The saved array has shape (num_sites, num_time_points), so each row is one site's <X>(t).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import X, Y, Z


def main():
    """Run the analog simulation and save observable dynamics as a .npy file.

    For new users:
        1. Change values in the CONFIG section below.
        2. Run this script.
        3. Load the .npy output with NumPy.
    """

    # ============================== CONFIG ==================================
    # 1) System and Hamiltonian
    length = 3 # Total length of the chain
    j_coupling = 1.0 # How much neighboring sites interact
    g_field = 1.0 # How much a field interacts with individual sites
    initial_state = "zeros" # Starting state of the chain
    hamiltonian = MPO.ising(length, j_coupling, g_field) # Model being simulated
    observables = [Observable(Z(), site) for site in range(length)] # What we want to measure. Options X, Y, Z

    # 2) Simulation controls
    elapsed_time = 5.0 # Total time we want to simulate
    dt = 0.1 # Size of individual time steps. Coarse vs. fine grain view.
    num_traj = 100 # Number of Monte Carlo simulations we want to run. More is better, but slower.
    max_bond_dim = 2**31 - 1  # very large value approximates "no cap"
    threshold = 1e-6 # Accuracy of the simulation. Lower is more accurate, but slower.

    # 3) Noise model and scheduled jumps
    gamma = 0  # Noise strength acting on every site. Use between 1e-3 and 1e-1.
    scheduled_jump_time = None # Time of a scheduled jump. Set to None to disable.
    scheduled_jump_site = 2 # Location of scheduled jump
    scheduled_jump_operator = "lowering" # Type of noise scheduled

    # 4) Heatmap interpolation
    heatmap_interpolation = "bicubic"  # e.g. "nearest", "bilinear", "bicubic", None

    # Output
    output_path = Path("data/dynamics.npy")
    heatmap_png_path = Path("data/dynamics.png")
    # ========================================================================

    # Simulation
    state = MPS(length, state=initial_state)

    # Build noise model directly in this script to keep it easy to modify.
    processes = [{"name": "pauli_z", "sites": [site], "strength": gamma} for site in range(length)]
    scheduled_jumps = []
    if scheduled_jump_time is not None:
        if scheduled_jump_site < 0 or scheduled_jump_site >= length:
            msg = f"scheduled_jump_site must be in [0, {length - 1}]"
            raise ValueError(msg)
        scheduled_jumps.append({"time": scheduled_jump_time, "sites": [scheduled_jump_site], "name": scheduled_jump_operator})
    noise_model = NoiseModel(processes=processes, scheduled_jumps=scheduled_jumps)

    sim_params = AnalogSimParams(
        observables=observables,
        elapsed_time=elapsed_time,
        dt=dt,
        num_traj=num_traj,
        max_bond_dim=max_bond_dim,
        threshold=threshold,
        order=1,
        sample_timesteps=True,
    )

    simulator.run(state, hamiltonian, sim_params, noise_model)

    # Shape: (num_sites, num_times), ready for np.load(...)/imshow(...)
    heatmap = np.asarray([observable.results for observable in sim_params.observables], dtype=np.float64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, heatmap)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    image = ax.imshow(
        heatmap,
        aspect="auto",
        extent=(0, elapsed_time, length, 0),
        interpolation=heatmap_interpolation,
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Site")
    # Center integer site labels on each heatmap row.
    ax.set_yticks(np.arange(length) + 0.5)
    ax.set_yticklabels([str(site) for site in range(length)])
    ax.set_title("Observable Dynamics Heatmap")
    fig.colorbar(image, ax=ax, label="<X>")
    fig.tight_layout()
    fig.savefig(heatmap_png_path, dpi=200)
    plt.close(fig)

    print(f"Saved observable dynamics to: {output_path.resolve()}")
    print(f"Saved heatmap image to: {heatmap_png_path.resolve()}")
    print(f"Array shape: {heatmap.shape} (sites, sampled_times)")
    if scheduled_jumps:
        print(f"Scheduled jump: {scheduled_jumps[0]}")
    else:
        print("No scheduled lowering jumps configured.")


if __name__ == "__main__":
    main()
