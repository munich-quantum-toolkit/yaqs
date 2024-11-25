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
    apply_dissipation(state, noise_model, sim_params.dt/2)
    return stochastic_process(state, noise_model, sim_params.dt)


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
    dynamic_TDVP(state, H, sim_params)
    apply_dissipation(state, noise_model, sim_params.dt)
    return stochastic_process(state, noise_model, sim_params.dt)


def sample(phi: 'MPS', H: 'MPO', noise_model: 'NoiseModel', sim_params: 'SimulationParams', results: np.ndarray, j: int) -> 'MPS':
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
    dynamic_TDVP(psi, H, sim_params)
    apply_dissipation(psi, noise_model, sim_params.dt/2)
    psi.normalize('B')
    for obs_index, observable in enumerate(sim_params.observables):
        results[obs_index, j] = copy.deepcopy(psi).measure(observable)


def run_trajectory_second_order(args):
    """
    Run a single trajectory of the TJM.

    Args:
        args (tuple): Tuple containing index, initial state, noise model, simulation parameters, observables, sites, times, and Hamiltonian.

    Returns:
        list: Expectation values for the trajectory over time.
    """
    i, initial_state, noise_model, sim_params, times, H = args

    # Create deep copies of the shared inputs to avoid race conditions
    state = copy.deepcopy(initial_state)
    
    results = np.zeros((len(sim_params.observables), len(times)))

    for obs_index, observable in enumerate(sim_params.observables):
        results[obs_index, 0] = copy.deepcopy(state).measure(observable)

    phi = initialize(state, noise_model, sim_params)
    sample(phi, H, noise_model, sim_params, results, j=1)

    for j, _ in enumerate(times[2:], start=2):
        phi = step_through(phi, H, noise_model, sim_params)
        sample(phi, H, noise_model, sim_params, results, j)

    return results

def run_trajectory_first_order(args):
    i, initial_state, noise_model, sim_params, times, H = args
    state = copy.deepcopy(initial_state)
    results = np.zeros((len(sim_params.observables), len(times)))

    for obs_index, observable in enumerate(sim_params.observables):
        results[obs_index, 0] = copy.deepcopy(state).measure(observable)

    for j, _ in enumerate(times[1:], start=1):
        dynamic_TDVP(state, H, sim_params)
        apply_dissipation(state, noise_model, sim_params.dt)
        state = stochastic_process(state, noise_model, sim_params.dt)
        for obs_index, observable in enumerate(sim_params.observables):
            results[obs_index, j] = copy.deepcopy(state).measure(observable)

    return results


def TJM(initial_state: 'MPS', H: 'MPO', noise_model: 'NoiseModel', sim_params: 'SimulationParams', order=1) -> np.ndarray:
    """
    Perform the Tensor Jump Method (TJM) to simulate the noisy evolution of a quantum system.

    Args:
        initial_state (MPS): Initial state of the system.
        H (MPO): System Hamiltonian.
        noise_model (NoiseModel): Noise model to apply to the system.
        sim_params (SimulationParams): Simulation parameters, including time step, number of trajectories, and measurements.
        order (int): First or second order Trotterization.

    Returns:
        np.ndarray: Array containing times and expectation values. If full_data is True, this is for each trajectory.
                    Otherwise, it only contains the average over N trajectories.
    """
    # Reset any previous results
    for observable in sim_params.observables:
        observable.initialize(sim_params)

    times = np.arange(0, sim_params.T + sim_params.dt, sim_params.dt)

    # State must start in B form
    initial_state.normalize('B')
    args = [(i, initial_state, noise_model, sim_params, times, H) for i in range(sim_params.N)]

    max_workers = max(1, multiprocessing.cpu_count() - 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        if order == 2:
            futures = {executor.submit(run_trajectory_second_order, arg): arg[0] for arg in args}
        elif order == 1:
            futures = {executor.submit(run_trajectory_first_order, arg): arg[0] for arg in args}

        with tqdm(total=sim_params.N, desc="Running trajectories", ncols=80) as pbar:
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    result = future.result()
                    for obs_index, observable in enumerate(sim_params.observables):
                        observable.trajectories[i] = result[obs_index]
                except Exception as e:
                        print(f"\nTrajectory {i} failed with exception: {e}. Retrying...")
                        # Retry could be done here
                finally:
                    pbar.update(1)

    # Save average value of trajectories
    for observable in sim_params.observables:
        observable.results = np.mean(observable.trajectories, axis=0)
