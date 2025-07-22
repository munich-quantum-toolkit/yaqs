from __future__ import annotations

import numpy as np
import pickle

from bond_dimension.bond_dim_utils import generate_sim_data_qaoa
from mqt.yaqs.core.libraries.circuit_library import create_two_local


def run_bond_dimension_test():
    max_bond = 512
    timesteps_list = [*range(1, 101)]

    num_qubits = 49

    # Linear
    print("Linear")
    entanglement = "pairwise"
    min_bond = 2
    results = generate_sim_data_qaoa(
        make_circ=create_two_local,
        make_args=(num_qubits, entanglement),
        timesteps=timesteps_list,
        min_bond_dim=min_bond,
        periodic=True,
        break_on_exceed=True,
        bond_dim_limit=max_bond
    )
    filename = f"results/two_local/qaoa_twolocal.pickle"
    with open(filename, 'wb') as f:
        pickle.dump({
            'results': results,
        }, f)

    # SCA
    # print("Circular")
    # entanglement = "circular"
    # min_bond = 4
    # results = generate_sim_data_two_local(
    #     make_circ=create_two_local,
    #     make_args=(num_qubits, entanglement),
    #     timesteps=timesteps_list,
    #     min_bond_dim=min_bond,
    #     periodic=True,
    #     break_on_exceed=True,
    #     bond_dim_limit=max_bond
    # )
    # filename = f"results/two_local/sca_twolocal.pickle"
    # with open(filename, 'wb') as f:
    #     pickle.dump({
    #         'results': results,
    #     }, f)

    # # Full
    # print("Full")
    # entanglement = "full"
    # min_bond = 4
    # results = generate_sim_data_two_local(
    #     make_circ=create_two_local,
    #     make_args=(num_qubits, entanglement),
    #     timesteps=timesteps_list,
    #     min_bond_dim=min_bond,
    #     periodic=True,
    #     break_on_exceed=True,
    #     bond_dim_limit=max_bond
    # )
    # filename = f"results/two_local/full_twolocal.pickle"
    # with open(filename, 'wb') as f:
    #     pickle.dump({
    #         'results': results,
    #     }, f)

if __name__ == "__main__":
    run_bond_dimension_test()
