from mqt.yaqs.core.data_structures.networks import MPS, MPO
from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs.core.data_structures.noise_model import NoiseModel

import numpy as np
import pickle


def tdvp_simulator(H_0, noise_model, state=None):
    L = H_0.length

    state = MPS(length=L)

    measurements = [Observable(Z(), [0])] + [Observable("entropy", [L//2, L//2+1])] + [Observable("max_bond")]
    sim_params = AnalogSimParams(observables=measurements,
                                elapsed_time=5,
                                dt=0.1,
                                num_traj=30,
                                threshold=1e-6,
                                trunc_mode="discarded_weight",
                                order=2,
                                sample_timesteps=False)

    simulator.run(state, H_0, sim_params, noise_model=noise_model)
    print("Obs Exp Val", sim_params.observables[0].results[-1])
    print("Entropy", sim_params.observables[1].results[-1])
    print("Max Bond", sim_params.observables[2].results[-1])
    return sim_params.observables

L_list = range(5, 105, 5)

for L in L_list:
    print(L)
    J = 1
    h = 1
    H_0 = MPO()
    H_0.init_ising(L, J, h)

    dps = np.logspace(-3, 0, 20)
    results1 = []
    results2 = []
    for j, dp in enumerate(dps):
        print(j+1, "of", len(dps))
        gamma = dp/0.1  # assuming dt=0.1
        # Unraveling 1
        noise_model = NoiseModel([
            {"name": name, "sites": [i], "strength": gamma} for i in range(L) for name in ["pauli_z", "pauli_x", "pauli_y"]
            ])
        cost = tdvp_simulator(H_0, noise_model)
        results1.append(cost)

        # Unraveling 2
        noise_model = NoiseModel([
            {"name": name, "sites": [i], "strength": gamma} for i in range(L) for name in ["measure_0", "measure_1", "measure_x_0", "measure_x_1", "measure_y_0", "measure_y_1"]
        ])
        gamma = 2*gamma
        cost = tdvp_simulator(H_0, noise_model)
        results2.append(cost)

    filename = f"u1t1_{L}.pickle"
    with open(filename, 'wb') as handle:
        pickle.dump(results1, handle)

    filename = f"u2t1_{L}.pickle"
    with open(filename, 'wb') as handle:
        pickle.dump(results2, handle)
