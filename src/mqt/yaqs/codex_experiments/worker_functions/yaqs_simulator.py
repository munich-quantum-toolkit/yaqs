import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs import simulator

def run_yaqs(
    init_circuit,
    trotter_step,
    num_qubits: int,
    num_layers: int,
    nm: NoiseModel,
    *,
    parallel = True,
    num_traj = 1024,
    max_bond_dim = 256,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    # Build circuit: init once, then repeat Trotter steps
    circ = init_circuit.copy()
    circ.compose(trotter_step, qubits=range(num_qubits), inplace=True)
    circ.barrier(label="SAMPLE_OBSERVABLES")
    for _ in range(1, num_layers):
        circ.compose(trotter_step, qubits=range(num_qubits), inplace=True)
        circ.barrier(label="SAMPLE_OBSERVABLES")

    obs = [Observable(Z(), i) for i in range(num_qubits)]
    sim = StrongSimParams(observables=obs, num_traj=num_traj, max_bond_dim=max_bond_dim, sample_layers=True)
    sim.log_bond_dims = True
    state = MPS(num_qubits, state="zeros", pad=2)
    simulator.run(state, circ, sim, nm, parallel=parallel)
    # shape (Q, num_layers+?)
    res = np.stack([np.real(o.results) for o in sim.observables])
    # drop initial column and final aggregate → keep exactly num_layers points
    expvals = res[:, 1:-1]
    # YAQS stochastic variance across trajectories: per qubit, per layer, including initial (t=0)
    yaqs_var = np.zeros((num_qubits, num_layers + 1), dtype=float)
    for q, o in enumerate(sim.observables):
        traj = o.trajectories  # (num_traj, num_mid_measurements+2)
        assert traj is not None and traj.shape[1] >= 2
        # initial
        yaqs_var[q, 0] = float(np.var(np.real(traj[:, 0]), axis=0))
        # layers 1..num_layers (drop final aggregate)
        yaqs_var[q, 1:] = np.var(np.real(traj[:, 1:-1]), axis=0)
    return expvals, getattr(sim, "bond_dim_trajectories", None), yaqs_var



def build_noise_models(processes):
    # Always deep-copy; each NoiseModel gets its own process list.
    procs_std  = copy.deepcopy(processes)
    procs_proj = copy.deepcopy(processes)
    procs_2pt  = copy.deepcopy(processes)
    procs_gaus = copy.deepcopy(processes)

    # (1) standard (whatever your default is)
    noise_model_normal = NoiseModel(procs_std)

    # (2) projector unraveling: same Lindblad rate γ per process
    for p in procs_proj:
        p["unraveling"] = "projector"
    for p in procs_2pt:
        p["unraveling"] = "unitary_2pt"
    for p in procs_gaus:
        p["unraveling"] = "unitary_gauss"
        # strength unchanged
    noise_model_projector = NoiseModel(procs_proj)
    noise_model_unitary_2pt = NoiseModel(procs_2pt)
    noise_model_unitary_gauss = NoiseModel(procs_gaus, gauss_M=11)

    return (noise_model_normal,
            noise_model_projector,
            noise_model_unitary_2pt,
            noise_model_unitary_gauss)


