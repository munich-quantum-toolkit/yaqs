import concurrent.futures
import copy
import multiprocessing
import numpy as np
from tqdm import tqdm

from yaqs.methods.dissipation import apply_dissipation
from yaqs.methods.dynamic_TDVP import dynamic_TDVP
from yaqs.methods.stochastic_process import stochastic_process

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from yaqs.data_structures.MPO import MPO
    from yaqs.data_structures.MPS import MPS
    from yaqs.data_structures.noise_model import NoiseModel
    from yaqs.data_structures.simulation_parameters import SimulationParams


def initialize(state: 'MPS', noise_model: 'NoiseModel', sim_params: 'SimulationParams') -> 'MPS':
    """
    Initialize the sampling MPS for second-order Trotterization.
    Corresponds to F0 in the TJM paper.

    Args:
        state (MPS): Initial state of the system.
        noise_model (NoiseModel): Noise model to apply to the system.
        sim_params (SimulationParams): Simulation parameters, including time step.

    Returns:
        MPS: The initialized sampling MPS Phi(0).
    """
    previous_state = copy.deepcopy(state)
    apply_dissipation(state, noise_model, sim_params.dt/2)
    return stochastic_process(previous_state, state, noise_model, sim_params.dt)


def step_through(state: 'MPS', H: 'MPO', noise_model: 'NoiseModel', sim_params: 'SimulationParams') -> 'MPS':
    """
    Perform a single time step of the TJM of the system state.
    Corresponds to Fj in the TJM paper.

    Args:
        state (MPS): Current state of the system.
        H (MPO): Hamiltonian operator for the system.
        noise_model (NoiseModel): Noise model to apply to the system.
        sim_params (SimulationParams): Simulation parameters, including time step and max bond dimension.

    Returns:
        MPS: The updated state after performing the time step evolution.
    """
    previous_state = copy.deepcopy(state)
    dynamic_TDVP(state, H, sim_params.dt, sim_params.max_bond_dim)
    apply_dissipation(state, noise_model, sim_params.dt)
    return stochastic_process(previous_state, state, noise_model, sim_params.dt)


def sample(phi: 'MPS', H: 'MPO', noise_model: 'NoiseModel', sim_params: 'SimulationParams') -> 'MPS':
    """
    Sample the quantum state and measure an observable from the sampling MPS.
    Corresponds to Fn in the TJM paper.

    Args:
        phi (MPS): State of the system before sampling.
        H (MPO): Hamiltonian operator for the system.
        noise_model (NoiseModel): Noise model to apply to the system.
        sim_params (SimulationParams): Simulation parameters, including time step and measurements.

    Returns:
        MPS: The measured observable value.
    """
    psi = copy.deepcopy(phi)
    dynamic_TDVP(psi, H, sim_params.dt, sim_params.max_bond_dim)
    apply_dissipation(psi, noise_model, sim_params.dt/2)
    psi.normalize('B')
    return psi.measure_observable(list(sim_params.measurements.keys())[0], list(sim_params.measurements.values())[0])


def run_trajectory(args):
    """
    Run a single trajectory of the TJM.

    Args:
        args (tuple): Tuple containing index, initial state, noise model, simulation parameters, observables, sites, times, and Hamiltonian.

    Returns:
        list: Expectation values for the trajectory over time.
    """
    i, initial_state, noise_model, sim_params, observables, sites, times, H = args

    # Create deep copies of the shared inputs to avoid race conditions
    state = copy.deepcopy(initial_state)
    single_trajectory_exp_values = [state.measure_observable(observables[0], sites[0])]

    phi = initialize(state, noise_model, sim_params)
    single_trajectory_exp_values.append(sample(phi, H, noise_model, sim_params))

    for _ in times[2:]:
        phi = step_through(phi, H, noise_model, sim_params)
        single_trajectory_exp_values.append(sample(phi, H, noise_model, sim_params))

    return single_trajectory_exp_values


def TJM(initial_state: 'MPS', H: 'MPO', noise_model: 'NoiseModel', sim_params: 'SimulationParams', full_data=False, multi_core=True) -> np.ndarray:
    """
    Perform the Tensor Jump Method (TJM) to simulate the noisy evolution of a quantum system.

    Args:
        initial_state (MPS): Initial state of the system.
        H (MPO): System Hamiltonian.
        noise_model (NoiseModel): Noise model to apply to the system.
        sim_params (SimulationParams): Simulation parameters, including time step, number of trajectories, and measurements.
        full_data (bool, optional): Whether to return the results of all trajectories. Defaults to False.
        multi_core (bool, optional): Whether to use multiple cores for parallel processing. Defaults to True.

    Returns:
        np.ndarray: Array containing times and expectation values. If full_data is True, this is for each trajectory.
                    Otherwise, it only contains the average over N trajectories.
    """
    all_trajectories_exp_values = []
    observables = list(sim_params.measurements.keys())
    sites = list(sim_params.measurements.values())

    times = np.arange(0, sim_params.T + sim_params.dt, sim_params.dt)
    args = [(i, initial_state, noise_model, sim_params, observables, sites, times, H) for i in range(sim_params.N)]

    if multi_core:
        max_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one core for the system
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_trajectory, arg): arg[0] for arg in args}
            with tqdm(total=sim_params.N, desc="Processing trajectories", ncols=80) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    i = futures[future]
                    try:
                        result = future.result()
                        all_trajectories_exp_values.append(result)
                    except Exception as e:
                        print(f"\nTrajectory {i} failed with exception: {e}. Retrying...")
                        retry_result = run_trajectory(args[i])
                        all_trajectories_exp_values.append(retry_result)
                    finally:
                        pbar.update(1)

    else:
        with tqdm(total=sim_params.N, desc="Processing trajectories") as pbar:
            for i in range(sim_params.N):
                single_trajectory_exp_values = run_trajectory(args[i])
                all_trajectories_exp_values.append(single_trajectory_exp_values)
                pbar.update(1)

    all_trajectories_exp_values = np.array(all_trajectories_exp_values)

    if full_data:
        return times, all_trajectories_exp_values
    else:
        return times, np.mean(all_trajectories_exp_values, axis=0)
