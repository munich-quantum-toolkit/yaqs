import numpy as np
import matplotlib.pyplot as plt
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.simulator import run

def create_ising_jump_example():
    L = 10
    T = 10.0
    dt = 0.1
    J = 1.0
    g = 1.0
    jump_time = 1
    
    # 1. Setup Hamiltonian
    hamiltonian = MPO.ising(length=L, J=J, g=g)
    
    # 2. RUN BASELINE (No jump)
    print("Simulating baseline (no jump)...")
    initial_state_baseline = MPS(L, state="zeros")
    initial_state_baseline.normalize("B")
    
    z_obs_baseline = Observable(GateLibrary.z(), sites=0)
    params_baseline = AnalogSimParams(
        elapsed_time=T,
        dt=dt,
        num_traj=1,
        observables=[z_obs_baseline],
        order=1,
        show_progress=False
    )
    run(initial_state_baseline, hamiltonian, params_baseline)
    baseline_results = z_obs_baseline.results
    times = params_baseline.times
    
    # 3. SETUP PLOTS
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    cmap = plt.get_cmap("winter")
    
    # Plot baseline
    ax1.plot(times, baseline_results, label="Baseline", color="black", linestyle="--", lw=2)
    
    # 4. LOOP OVER SITES TO APPLY JUMP
    for jump_site in range(L):
        print(f"Simulating jump on site {jump_site}...")
        
        initial_state = MPS(L, state="zeros")
        initial_state.normalize("B")
        
        scheduled_jumps = [{"time": jump_time, "sites": [jump_site], "name": "lowering"}]
        noise_model = NoiseModel(scheduled_jumps=scheduled_jumps)
        
        z_obs = Observable(GateLibrary.z(), sites=0)
        params = AnalogSimParams(
            elapsed_time=T,
            dt=dt,
            num_traj=1,
            observables=[z_obs],
            show_progress=False
        )
        
        run(initial_state, hamiltonian, params, noise_model=noise_model)
        jump_results = z_obs.results
        
        # Color: site 0 is dark (1.0), site L-1 is light (0.3)
        color = cmap(1.0 - 0.7 * (jump_site / (L - 1)))
        
        # Plot expectation value
        ax1.plot(times, jump_results, label=f"Jump on site {jump_site}", color=color)
        
        # Plot log difference
        diff = np.abs(jump_results - baseline_results)
        ax2.plot(times, diff, label=f"Jump on site {jump_site}", color=color)
        ax2.set_yscale("log")

    # 5. FINALIZE PLOTS
    ax1.axvline(x=jump_time, color='r', linestyle='--', alpha=0.5, label="Jump Time")
    ax1.set_title(f"Ising Chain (L={L}): <Z_0> Dynamics with Scheduled X Jump at T={jump_time}")
    ax1.set_ylabel("<Z_0>")
    ax1.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    ax1.grid(True, alpha=0.3)
    
    ax2.axvline(x=jump_time, color='r', linestyle='--', alpha=0.5)
    ax2.set_title("Absolute Difference from Baseline")
    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("|ΔZ_0|)")
    ax2.set_ylim(1e-6, 2)  # Show very small differences up to significant ones
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = "ising_jumps_comparison.png"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    create_ising_jump_example()
