from __future__ import annotations

import numpy as np
import pickle

from bond_dimension.bond_dim_utils import generate_sim_data
from mqt.yaqs.core.libraries.circuit_library import create_heisenberg_circuit, create_2d_ising_circuit


def run_bond_dimension_test():
    max_bond = 512
    timesteps_list = [*range(1, 10001)]

    J = 1
    g = 1
    dt = 0.1
    num_qubits = 49

    # 1D Perioidic Heisenberg model
    print("1D Periodic")
    min_bond = 4
    results = generate_sim_data(
        make_circ=create_heisenberg_circuit,
        make_args=(num_qubits, J, J, J, g, dt),
        timesteps=timesteps_list,
        min_bond_dim=min_bond,
        periodic=True,
        break_on_exceed=True,
        bond_dim_limit=max_bond
    )
    filename = f"results/bond_dim_test/periodic_heisenberg_bonds.pickle"
    with open(filename, 'wb') as f:
        pickle.dump({
            'results': results,
        }, f)

    print("2D Ising")
    num_rows = int(np.sqrt(num_qubits))
    num_cols = int(np.sqrt(num_qubits))

    min_bond = 8
    results = generate_sim_data(
    make_circ=create_2d_ising_circuit,
    make_args=(num_rows, num_cols, J, g, dt),
    timesteps=timesteps_list,
    min_bond_dim=min_bond,
    break_on_exceed=True,
    bond_dim_limit=max_bond
)
    filename = f"results/bond_dim_test/2d_ising_bond.pickle"
    with open(filename, 'wb') as f:
        pickle.dump({
            'results': results,
        }, f)

    # 1D Heisenberg model
    print("1D Heisenberg")
    min_bond = 2

    results = generate_sim_data(
    make_circ=create_heisenberg_circuit,
    make_args=(num_qubits, J, J, J, g, dt),
    timesteps=timesteps_list,
    min_bond_dim=min_bond,
    periodic=False,
    break_on_exceed=True,
    bond_dim_limit=max_bond
    )

    filename = f"results/bond_dim_test/heisenberg_bonds.pickle"
    with open(filename, 'wb') as f:
        pickle.dump({
            'results': results,
        }, f)

if __name__ == "__main__":
    run_bond_dimension_test()
    