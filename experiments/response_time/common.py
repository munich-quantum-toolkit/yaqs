
import numpy as np
import matplotlib.pyplot as plt
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.simulator import run

def run_ising_jump_simulation(L, T, dt, J, g, jump_time, jump_site, jump_op_name, gamma=0):
    """
    Runs an Ising simulation with a scheduled jump.
    
    Args:
        L (int): System size.
        T (float): Total simulation time.
        dt (float): Time step.
        J (float): Coupling constant.
        g (float): Transverse field.
        jump_time (float): Time at which the jump occurs.
        jump_site (int): Site where the jump occurs.
        jump_op_name (str): Name of the jump operator (e.g., "x", "y", "z", "plus", "minus").
        
    Returns:
        tuple: (times, baseline_results, jump_results)
    """
    # 1. Setup Hamiltonian
    hamiltonian = MPO.ising(length=L, J=J, g=g)
    
    # 2. RUN BASELINE (No jump)
    # Check if we can cache baseline? For now just re-run it to keep it simple and stateless
    initial_state_baseline = MPS(L, state="zeros")
    initial_state_baseline.normalize("B")

    z_obs_baseline = Observable(GateLibrary.z(), sites=0)

    params_baseline = AnalogSimParams(
        elapsed_time=T,
        dt=dt,
        num_traj=1,
        observables=[z_obs_baseline],
        order=2,
        show_progress=False
    )
    run(initial_state_baseline, hamiltonian, params_baseline)
    baseline_results = z_obs_baseline.results
    times = params_baseline.times
    
    # 3. RUN JUMP SIMULATION
    initial_state = MPS(L, state="zeros")
    initial_state.normalize("B")
    
    scheduled_jumps = [{"time": jump_time, "sites": [jump_site], "name": jump_op_name}]
    if gamma == 0:
        noise_model = NoiseModel(scheduled_jumps=scheduled_jumps)
    else:
        noise_model = NoiseModel([{"name": name, "sites": [i], "strength": gamma} for i in range(L) for name in [jump_op_name]], scheduled_jumps=scheduled_jumps)
    
    z_obs = Observable(GateLibrary.z(), sites=0)
    if gamma == 0:
        num_traj = 1
    else:
        num_traj = 10
    params = AnalogSimParams(
        elapsed_time=T,
        dt=dt,
        num_traj=num_traj,
        observables=[z_obs],
        show_progress=False
    )
    
    run(initial_state, hamiltonian, params, noise_model=noise_model)
    jump_results = z_obs.results
    
    return times, baseline_results, jump_results

def calculate_response_time(times, baseline_results, jump_results, jump_time, threshold=1e-6):
    """
    Calculates the response time (time lag until deviation > threshold).
    
    Args:
        times (np.ndarray): Array of time points.
        baseline_results (np.ndarray): Expectation values for the baseline.
        jump_results (np.ndarray): Expectation values for the jump simulation.
        jump_time (float): Time at which the jump occurred.
        threshold (float): Deviation threshold.
        
    Returns:
        float: Response time, or np.nan if threshold is never crossed.
    """
    diff = np.abs(jump_results - baseline_results)
    mask = (times > jump_time) & (diff > threshold)
    if np.any(mask):
        first_idx = np.argmax(mask)
        t_detection = times[first_idx]
        return t_detection - jump_time
    else:
        return np.nan
