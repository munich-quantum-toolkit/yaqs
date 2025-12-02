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
                                num_traj=10,
                                threshold=1e-6,
                                trunc_mode="discarded_weight",
                                order=2,
                                sample_timesteps=False)

    simulator.run(state, H_0, sim_params, noise_model=noise_model)
    print("Obs Exp Val", sim_params.observables[0].results[-1])
    print("Entropy", sim_params.observables[1].results[-1])
    print("Max Bond", sim_params.observables[2].results[-1])
    return sim_params.observables

