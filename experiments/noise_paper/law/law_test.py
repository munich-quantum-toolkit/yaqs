from mqt.yaqs.core.data_structures.networks import MPS, MPO
from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.data_structures.noise_model import NoiseModel

import numpy as np
import pickle


def tdvp_simulator(H_0, noise_model, *, T: float):
    """Run TDVP and return observables (max bond)."""
    L = H_0.length

    state = MPS(state="Neel", length=L)
    measurements = [Observable("max_bond")]

    sim_params = AnalogSimParams(
        observables=measurements,
        elapsed_time=T,
        dt=0.1,
        num_traj=30,
        threshold=1e-6,
        trunc_mode="discarded_weight",
        order=2,
        sample_timesteps=False,
    )

    simulator.run(state, H_0, sim_params, noise_model=noise_model)
    return sim_params.observables


if __name__ == "__main__":

    # --- parameters ---
    T_list = [2, 5, 8]
    L_list = [5, 10, 15, 20, 25, 30, 40, 60, 80]
    dps = [1e-3, 1e-2, 1e-1, 1e0]
    dt = 0.1

    for T in T_list:
        print(f"\n=== Running T = {T} ===")

        for L in L_list:
            print(f"L = {L}")

            # Build Hamiltonian once per L
            J = 1
            h = 1
            H_0 = MPO()
            H_0.init_ising(L, J, h)

            results_u1 = []
            results_u2 = []

            for j, dp in enumerate(dps):
                print(f"  dp {j+1}/{len(dps)} = {dp:.1e}")

                gamma = dp / dt

                # --- Unraveling A: Pauli jumps ---
                noise_u1 = NoiseModel([
                    {"name": name, "sites": [i], "strength": gamma}
                    for i in range(L)
                    for name in ["pauli_x", "pauli_y", "pauli_z"]
                ])
                obs_u1 = tdvp_simulator(H_0, noise_u1, T=T)
                results_u1.append(obs_u1)

                # --- Unraveling B: projective ---
                noise_u2 = NoiseModel([
                    {"name": name, "sites": [i], "strength": 2 * gamma}
                    for i in range(L)
                    for name in [
                        "measure_0", "measure_1",
                        "measure_x_0", "measure_x_1",
                        "measure_y_0", "measure_y_1",
                    ]
                ])
                obs_u2 = tdvp_simulator(H_0, noise_u2, T=T)
                results_u2.append(obs_u2)

            # --- Save ---
            with open(f"u1_law_T{T}_L{L}.pickle", "wb") as f:
                pickle.dump(results_u1, f)

            with open(f"u2_law_T{T}_L{L}.pickle", "wb") as f:
                pickle.dump(results_u2, f)
