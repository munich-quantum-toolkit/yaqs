"""
Generate MCWF time evolution data for a 12-site Ising model.

System: 12 sites, J=1, g=1 (Transverse Field Ising Model)
Dynamics: MCWF with Z noise at each site, 1000 trajectories.
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

def generate_mcwf_data(test_run: bool = False):
    # System Parameters
    if test_run:
        L = 4
        num_traj = 10
    else:
        L = 9
        num_traj = 200
        
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
        # 1000 gammas * 1000 traj * L=12 (serial) is too slow (>24hrs).
        # We reduce to 50 points to make it feasible (~1-2 hours).
        n_gammas = 1000
    
    # Gammas to sweep
    # Logspace from 10^-4 to 10^-1
    gammas = np.logspace(-4, -1, n_gammas)
    
    print(f"Starting MCWF data generation for {n_gammas} gamma values...")
    print(f"System L={L}, Trajectories={num_traj}")
    if test_run:
        print("Running in TEST mode with reduced parameters.")
    
    all_results = []
    times = None
    
    # Static Objects
    H = MPO.ising(L, J, g)
    psi0 = MPS(L, state="Neel")
    
    for gamma in tqdm(gammas, desc="Gamma Sweep"):
        # 1. Noise Model
        noise_processes = [
            {"name": "z", "sites": [i], "strength": gamma}
            for i in range(L)
        ]
        noise_model = NoiseModel(processes=noise_processes)
        
        # 2. Simulation Parameters
        # MCWF averages over trajectories automatically in aggregate_trajectories if we use the standard flow?
        # Let's check AnalogSimParams.aggregate_trajectories.
        # It computes mean of observable.trajectories and stores in observable.results.
        # Yes, standard flow works.
        observables = [Observable(Z(), sites=i) for i in range(L)]
        
        sim_params = AnalogSimParams(
            observables=observables,
            elapsed_time=T,
            dt=dt,
            solver="MCWF",
            num_traj=num_traj, # 1000 trajectories
            sample_timesteps=True,
            show_progress=False,
        )
        
        # 3. Run Simulation
        # TJM (MPS) is safe for parallel execution.
        simulator.run(psi0, H, sim_params, noise_model, parallel=True)
        
        # 4. Collect Results
        current_gamma_results = []
        for obs in sim_params.sorted_observables:
            current_gamma_results.append(obs.results)
        
        all_results.append(np.array(current_gamma_results))
        
        if times is None:
            times = sim_params.times

    # Final shape: (n_gammas, n_sites, n_timesteps)
    all_results_np = np.array(all_results) 
    
    # Save Data
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ising_mcwf_data.npz"
    
    np.savez(output_file, 
             gammas=gammas, 
             times=times, 
             observables=all_results_np, 
             system_params={"L": L, "J": J, "g": g, "T": T, "dt": dt, "solver": "MCWF", "num_traj": num_traj}
            )
    
    print(f"Data saved to {output_file}")
    print(f"Observables shape: {all_results_np.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run a fast test simulation")
    args = parser.parse_args()
    
    generate_mcwf_data(test_run=args.test)
