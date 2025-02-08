import concurrent.futures
import multiprocessing
from qiskit.circuit import QuantumCircuit
from tqdm import tqdm

from yaqs.core.data_structures.networks import MPO
from yaqs.core.data_structures.simulation_parameters import StrongSimParams, WeakSimParams

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from yaqs.core.data_structures.networks import MPS
    from yaqs.core.data_structures.noise_model import NoiseModel


def run(initial_state: 'MPS', operator, sim_params, noise_model: 'NoiseModel'=None):
    """
    Common simulation routine used by both circuit and Hamiltonian simulations.
    It normalizes the state, prepares trajectory arguments, runs the trajectories
    in parallel, and aggregates the results.
    
    The aggregation logic distinguishes between the circuit simulation types:
      - For QuantumCircuit simulations using WeakSimParams, each trajectory’s result
        is stored in sim_params.measurements and later aggregated via sim_params.aggregate_measurements().
      - Otherwise (for StrongSimParams or Hamiltonian simulations), each observable’s
        trajectory is stored and then averaged.
    """
    # For Hamiltonian simulations and for circuit simulations with StrongSimParams,
    # initialize observables. For WeakSimParams in the circuit case, no initialization needed.
    if isinstance(operator, MPO) or isinstance(sim_params, StrongSimParams):
        for observable in sim_params.observables:
            observable.initialize(sim_params)

    if isinstance(operator, QuantumCircuit):
        assert initial_state.length == operator.num_qubits, "State and circuit qubit counts do not match."
    
    if not noise_model or all(gamma == 0 for gamma in noise_model.strengths):
        sim_params.N = 1
    else:
        if isinstance(sim_params, WeakSimParams):
            sim_params.N = sim_params.shots
            sim_params.shots = 1

    # Normalize the state to the B form
    initial_state.normalize('B')

    # Prepare arguments for each trajectory
    args = [(i, initial_state, noise_model, sim_params, operator) for i in range(sim_params.N)]
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(sim_params.backend, arg): arg[0] for arg in args}
        with tqdm(total=sim_params.N, desc="Running trajectories", ncols=80) as pbar:
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    result = future.result()
                    # For circuit simulation using WeakSimParams, store in measurements;
                    # otherwise, update observable trajectories.
                    if isinstance(operator, QuantumCircuit) and isinstance(sim_params, WeakSimParams):
                        sim_params.measurements[i] = result
                    else:
                        for obs_index, observable in enumerate(sim_params.observables):
                            observable.trajectories[i] = result[obs_index]
                except Exception as e:
                    print(f"\nTrajectory {i} failed with exception: {e}.")
                finally:
                    pbar.update(1)

    if isinstance(operator, QuantumCircuit) and isinstance(sim_params, WeakSimParams):
        sim_params.aggregate_measurements()
    else:
        sim_params.aggregate_trajectories()
