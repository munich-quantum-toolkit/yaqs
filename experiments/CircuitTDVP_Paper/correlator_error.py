import pickle

from correlator.correlator_utils import generate_error_data
from mqt.yaqs.core.libraries.circuit_library import create_heisenberg_circuit, create_2d_ising_circuit


def run_correlator_test():
    # General Heisenberg parameters
    J = 1
    h = 1
    dt = 0.1
    num_qubits = 16

    min_bonds = [2, 4, 8, 16, 32]
    timesteps_list = [*range(1, 31)]

    # 1D Heisenberg model
    print("1D Heisenberg")
    
    results = generate_error_data(create_heisenberg_circuit, (num_qubits, J, J, J, h, dt),
                        min_bonds=min_bonds, timesteps=timesteps_list, periodic=False)
    filename = f"results/correlator_error/heisenberg_error_32.pickle"
    with open(filename, 'wb') as f:
        pickle.dump({
            'results': results,
        }, f)

    # 1D Perioidic Heisenberg model
    print("1D Periodic")
    results = generate_error_data(create_heisenberg_circuit, (num_qubits, J, J, J, h, dt),
                      min_bonds=min_bonds, timesteps=timesteps_list, periodic=True)
    filename = f"results/correlator_error/periodic_heisenberg_error_32.pickle"
    with open(filename, 'wb') as f:
        pickle.dump({
            'results': results,
        }, f)

    # 2D Ising model
    J = 1
    g = 1
    dt = 0.1
    num_rows = 4
    num_cols = 4

    print("2D Ising")
    results = generate_error_data(create_2d_ising_circuit, (num_rows, num_cols, J, g, dt),
                      min_bonds=min_bonds, timesteps=timesteps_list)
    filename = f"results/correlator_error/2d_ising_error_32.pickle"
    with open(filename, 'wb') as f:
        pickle.dump({
            'results': results,
        }, f)
