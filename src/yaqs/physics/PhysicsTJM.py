import concurrent.futures
import copy
import multiprocessing
import numpy as np
from tqdm import tqdm

from yaqs.core.data_structures.networks import MPO, MPS
from yaqs.core.data_structures.noise_model import NoiseModel
from yaqs.core.data_structures.simulation_parameters import PhysicsSimParams
from yaqs.core.methods.dissipation import apply_dissipation
from yaqs.core.methods.dynamic_TDVP import dynamic_TDVP
from yaqs.core.methods.stochastic_process import stochastic_process


def initialize(state: MPS, noise_model: NoiseModel, sim_params: PhysicsSimParams) -> MPS:
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


def step_through(state: MPS, H: MPO, noise_model: NoiseModel, sim_params: PhysicsSimParams) -> MPS:
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


def sample(phi: MPS, H: MPO, noise_model: NoiseModel, sim_params: PhysicsSimParams, results: np.ndarray, j: int) -> MPS:
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
    psi = stochastic_process(psi, noise_model, sim_params.dt)
    if sim_params.sample_timesteps:
        for obs_index, observable in enumerate(sim_params.observables):
            results[obs_index, j] = copy.deepcopy(psi).measure(observable)
    else:
        for obs_index, observable in enumerate(sim_params.observables):
            results[obs_index, 0] = copy.deepcopy(psi).measure(observable)


def PhysicsTJM_2(args):
    """
    Run a single trajectory of the TJM.

    Args:
        args (tuple): Tuple containing index, initial state, noise model, simulation parameters, observables, sites, and Hamiltonian.

    Returns:
        list: Expectation values for the trajectory over time.
    """
    i, initial_state, noise_model, sim_params, H = args

    # Create deep copies of the shared inputs to avoid race conditions
    state = copy.deepcopy(initial_state)
    if sim_params.sample_timesteps:
        results = np.zeros((len(sim_params.observables), len(sim_params.times)))
    else:
        results = np.zeros((len(sim_params.observables), 1))

    if sim_params.sample_timesteps:
        for obs_index, observable in enumerate(sim_params.observables):
            results[obs_index, 0] = copy.deepcopy(state).measure(observable)

    phi = initialize(state, noise_model, sim_params)
    if sim_params.sample_timesteps:
        sample(phi, H, noise_model, sim_params, results, j=1)

    for j, _ in enumerate(sim_params.times[2:], start=2):
        phi = step_through(phi, H, noise_model, sim_params)
        if sim_params.sample_timesteps:
            sample(phi, H, noise_model, sim_params, results, j)
        elif j == len(sim_params.times)-1:
            sample(phi, H, noise_model, sim_params, results, j)

    return results


def PhysicsTJM_1(args):
    i, initial_state, noise_model, sim_params, H = args
    state = copy.deepcopy(initial_state)

    if sim_params.sample_timesteps:
        results = np.zeros((len(sim_params.observables), len(sim_params.times)))
    else:
        results = np.zeros((len(sim_params.observables), 1))

    if sim_params.sample_timesteps:
        for obs_index, observable in enumerate(sim_params.observables):
            results[obs_index, 0] = copy.deepcopy(state).measure(observable)

    for j, _ in enumerate(sim_params.times[1:], start=1):
        dynamic_TDVP(state, H, sim_params)
        if noise_model:
            apply_dissipation(state, noise_model, sim_params.dt)
            state = stochastic_process(state, noise_model, sim_params.dt)
        if sim_params.sample_timesteps:
            for obs_index, observable in enumerate(sim_params.observables):
                results[obs_index, j] = copy.deepcopy(state).measure(observable)
        elif j == len(sim_params.times)-1:
            for obs_index, observable in enumerate(sim_params.observables):
                results[obs_index, 0] = copy.deepcopy(state).measure(observable)

    return results


def run(initial_state: MPS, H: MPO, sim_params: PhysicsSimParams, noise_model: NoiseModel=None):
    """
    Perform the Tensor Jump Method (TJM) to simulate the noisy evolution of a quantum system.

    Args:
        initial_state (MPS): Initial state of the system.
        H (MPO): System Hamiltonian.
        sim_params (SimulationParams): Parameters needed to define all variables of the simulation.
        noise_model (NoiseModel): Noise model to apply to the system.

    Returns:
        None: Observables in SimulationParams are updated directly.
    """
    # Reset any previous results
    for observable in sim_params.observables:
        observable.initialize(sim_params)

    # Guarantee one trajectory if no noise model
    if not noise_model:
        sim_params.N = 1
        sim_params.order = 1

    # State must start in B form
    initial_state.normalize('B')
    args = [(i, initial_state, noise_model, sim_params, H) for i in range(sim_params.N)]

    max_workers = max(1, multiprocessing.cpu_count() - 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        if sim_params.order == 2:
            futures = {executor.submit(run_trajectory_second_order, arg): arg[0] for arg in args}
        elif sim_params.order == 1:
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

    sim_params.aggregate_trajectories()
