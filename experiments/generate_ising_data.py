"""
Generate Lindblad time evolution data for a 10-site Ising model.

System: 10 sites, J=1, g=1 (Transverse Field Ising Model)
Dynamics: Lindblad master equation with Z noise at each site.
Observables: <Z_i(t)> for all i, t.
Sweep: Gamma from 10^-4 to 10^-1 (1000 points).
"""

import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs import simulator

def generate_data(test_run: bool = False):
    # System Parameters
    if test_run:
        L = 4
    else:
        L = 6
    J = 1.0
    g = 1.0
    
    # Time Evolution Parameters
    if test_run:
        T = 1.0
        dt = 0.5
        n_gammas = 2
    else:
        T = 10.0
        dt = 0.1
        n_gammas = 1000
    
    # Gammas to sweep
    # Logspace from 10^-4 to 10^-1
    gammas = np.logspace(-4, -1, n_gammas)
    
    print(f"Starting data generation for {n_gammas} gamma values...")
    if test_run:
        print("Running in TEST mode with reduced parameters.")
    
    all_results = []
    times = None # Will store time points from first successful run
    
    # Static Objects
    # Hamiltonian: MPO.ising(L, J, g) -> -J sum Z_i Z_{i+1} - g sum X_i
    H = MPO.ising(L, J, g)
    
    # We create initial state and observables inside loop or once outside if they are reusable?
    # MPS is modified by simulator (time evolution happens in place on copied state depending on implementation)
    # simulator.run takes (initial_state, ...)
    # Let's check if simulator.run modifies initial_state.
    # The docstring for simulator.run usually implies it runs simulation. 
    # Usually it's safer to create fresh objects or look at implementation.
    # Looking at `simulator.py`, `_run_analog` copies state?
    # `_run_analog` calls `lindblad`.
    # `lindblad` starts with `psi = initial_state.to_vec()`. 
    # It does not modify `initial_state`.
    # So we can reuse `psi0` if we wanted, but `MPS` creation is cheap compared to Lindblad.
    
    psi0 = MPS(L, state="zeros")
    
    for gamma in tqdm(gammas, desc="Gamma Sweep"):
        # 1. Noise Model: Z noise at strength gamma on each site
        # Lindblad solver uses these strengths directly
        noise_processes = [
            {"name": "z", "sites": [i], "strength": gamma}
            for i in range(L)
        ]
        noise_model = NoiseModel(processes=noise_processes)
        
        # 2. Simulation Parameters
        # Need fresh observables each time because results are stored in them
        observables = [Observable(Z(), sites=i) for i in range(L)]
        
        sim_params = AnalogSimParams(
            observables=observables,
            elapsed_time=T,
            dt=dt,
            solver="Lindblad",
            sample_timesteps=True,
            show_progress=False # Disable inner progress bar
        )
        
        # 3. Run Simulation
        # We need to pass the initial state. Since it's read-only in Lindblad, passing reused object is fine.
        simulator.run(psi0, H, sim_params, noise_model)
        
        # 4. Collect Results
        # sim_params.sorted_observables contains the results
        # AnalogSimParams sorts observables by site.
        # Since we added them 0..L-1, they should be in order.
        
        # Collect results for this gamma
        # Each obs.results is array of shape (n_timesteps,)
        # We stack them to get (n_sites, n_timesteps)
        
        # Sort back if needed? They are sorted by site key in SimParams init.
        # Since we inserted site 0, 1, ... L-1, sorting is stable/identity.
        
        current_gamma_results = []
        for obs in sim_params.sorted_observables:
            current_gamma_results.append(obs.results)
        
        all_results.append(np.array(current_gamma_results))
        
        if times is None:
            times = sim_params.times

    # Convert to numpy array
    # all_results is list of (n_sites, n_timesteps) arrays
    # final shape: (n_gammas, n_sites, n_timesteps)
    all_results_np = np.array(all_results) 
    
    # Save Data
    # Use absolute path relative to script location or current working dir?
    # This script is in experiments/. Data in experiments/data/
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ising_lindblad_data.npz"
    
    np.savez(output_file, 
             gammas=gammas, 
             times=times, 
             observables=all_results_np, 
             system_params={"L": L, "J": J, "g": g, "T": T, "dt": dt}
            )
    
    print(f"Data saved to {output_file}")
    print(f"Observables shape: {all_results_np.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run a fast test simulation")
    args = parser.parse_args()
    
    generate_data(test_run=args.test) # Pass boolean flag
