from mqt.yaqs.core.data_structures.networks import MPS, MPO
from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs.core.data_structures.noise_model import NoiseModel

import numpy as np
import pickle


def tdvp_simulator(H_0, dt, noise_model, state=None):
    L = H_0.length

    state = MPS(length=L)

    measurements = [Observable(Z(), [0])] + [Observable("entropy", [L//2, L//2+1])] + [Observable("max_bond")]
    sim_params = AnalogSimParams(observables=measurements,
                                elapsed_time=5,
                                dt=dt,
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

if __name__ == "__main__":
    L = 65
    J = 1
    h = 1
    H_0 = MPO()
    H_0.init_ising(L, J, h)
    dt_list = [0.005, 0.01, 0.02, 0.05, 0.1]
    for dt in dt_list:
        gamma_list = [0.5, 1, 2, 5, 10, 20, 50, 100]
        results1 = []
        results2 = []
        for j, gamma in enumerate(gamma_list):
            print(j+1, "of", len(gamma_list))

            if dt*gamma > 1:
                cost = None
            # Unraveling 1
            else:
                noise_model = NoiseModel([
                    {"name": name, "sites": [i], "strength": gamma} for i in range(L) for name in ["pauli_z", "pauli_x", "pauli_y"]
                    ])
                cost = tdvp_simulator(H_0, dt, noise_model)
            results1.append(cost)

            # # Unraveling 2
            # noise_model = NoiseModel([
            #     {"name": name, "sites": [i], "strength": 2*gamma} for i in range(L) for name in ["measure_0", "measure_1", "measure_x_0", "measure_x_1", "measure_y_0", "measure_y_1"]
            # ])
            # cost = tdvp_simulator(H_0, noise_model)
            # results2.append(cost)

        filename = f"u1_practical_{L}.pickle"
        with open(filename, 'wb') as handle:
            pickle.dump(results1, handle)

        # filename = f"u2t1_{L}.pickle"
        # with open(filename, 'wb') as handle:
        #     pickle.dump(results2, handle)
