from mqt.yaqs.core.data_structures.networks import MPS, MPO
from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs.core.data_structures.noise_model import NoiseModel

import numpy as np
import pickle
import time

def tdvp_simulator(H_0, dt, noise_model, state=None):
    L = H_0.length

    state = MPS(length=L)

    measurements = [Observable("max_bond")]
    sim_params = AnalogSimParams(observables=measurements,
                                elapsed_time=3,
                                dt=dt,
                                num_traj=10,
                                threshold=1e-6,
                                trunc_mode="discarded_weight",
                                order=2,
                                sample_timesteps=True)

    start_time = time.time()
    simulator.run(state, H_0, sim_params, noise_model=noise_model)
    elapsed_time = time.time() - start_time
    time_per_traj = elapsed_time

    # print("Obs Exp Val", sim_params.observables[0].results[-1])
    # print("Entropy", sim_params.observables[0].results[-1])
    print("Max Bond", sim_params.observables[0].results)
    print(time_per_traj)
    return [sim_params.observables, time_per_traj]

if __name__ == "__main__":
    L = 65
    J = 1
    h = 1
    H_0 = MPO()
    # H_0.init_ising(L, J, h)
    H_0.init_heisenberg(L, J, J, J, h)

    # 1000, 500, 400, 250, 200, 125, 100, 50, 25, 20, 10 steps
    # dt_list = [0.0025, 0.004, 0.005, 0.01, 0.0125, 0.02, 0.025, 0.04, 0.05, 0.1, 0.2, 0.25, 0.5]
    dt_list = [0.01, 0.0125, 0.02, 0.025, 0.04, 0.05, 0.1, 0.2, 0.25, 0.5]

    for k, dt in enumerate(dt_list):
        gamma_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]

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

        filename = f"walltime_heisenberg_{k}.pickle"
        with open(filename, 'wb') as handle:
            pickle.dump(results1, handle)

        # filename = f"u2t1_{L}.pickle"
        # with open(filename, 'wb') as handle:
        #     pickle.dump(results2, handle)
