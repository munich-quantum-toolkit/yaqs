# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import copy
import operator

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from qiskit import transpile
from qiskit_aer import AerSimulator

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.libraries.circuit_library import (
    create_2d_heisenberg_circuit,
    create_2d_ising_circuit,
    create_heisenberg_circuit,
    create_ising_circuit,
)
from mqt.yaqs.core.libraries.gate_library import XX
from qiskit.quantum_info import SparsePauliOp, Statevector

def _mid_xx_operator(num_qubits):
    """Helper to build a SparsePauliOp for Z on the middle qubit."""
    mid = num_qubits // 2
    label = ["I"] * num_qubits
    label[mid] = "X"
    label[mid+1] = "X"
    return SparsePauliOp("".join(label))

def statevector_expectation(circ, init=None):
    circ2 = copy.deepcopy(circ)
    circ2.save_statevector()
    sim = AerSimulator(method="statevector")
    tc = transpile(circ2, sim)
    res = sim.run(tc, initial_statevector=init).result() if init is not None else sim.run(tc).result()
    sv = Statevector(res.get_statevector(tc))
    return sv, sv.expectation_value(_mid_xx_operator(tc.num_qubits)).real

def tebd_simulator(circ, initial_state=None):
    threshold = 1e-12
    circ2 = copy.deepcopy(circ)
    circ2.clear()
    if initial_state is not None:
        circ2.set_matrix_product_state(initial_state)
    circ2.append(circ, range(circ.num_qubits))
    circ2.save_matrix_product_state(label="final_mps")  
    op_xx = _mid_xx_operator(circ.num_qubits)
    mid = circ.num_qubits // 2
    circ2.save_expectation_value(op_xx, [*range(circ.num_qubits)], label="exp_xx")
    if circ.num_qubits < 20:
        circ2.save_statevector(label="final_state")

    sim = AerSimulator(
        method="matrix_product_state",
        # matrix_product_state_max_bond_dimension=max_bond,
        matrix_product_state_truncation_threshold=threshold,
    )
    tcirc = transpile(
        circ2,
        sim,
    )

    result = sim.run(tcirc).result()

    result = sim.run(tcirc).result()
    mps = result.data(0)["final_mps"]
    bonds = [lam[0].shape[0] for lam in mps[0][1::]]
    exp_xx = result.data(0)["exp_xx"]

    if circ.num_qubits < 20:
        statevector = result.data(0)["final_state"]
    else:
        statevector = None
    return bonds, exp_xx, statevector


def tdvp_simulator(circ, min_bond, initial_state=None):
    if initial_state is None:
        initial_state = MPS(length=circ.num_qubits)

    measurements = [Observable(XX(), [circ.num_qubits // 2, circ.num_qubits // 2+1])]
    sim_params = StrongSimParams(measurements, max_bond_dim=2**circ.num_qubits, min_bond_dim=min_bond, get_state=True, threshold=1e-12)

    # circ_flipped = copy.deepcopy(circ).reverse_bits()
    simulator.run(initial_state, circ, sim_params, noise_model=None)
    mps = sim_params.output_state

    bonds = [tensor.shape[1] for tensor in mps.tensors[1::]]
    exp_val = sim_params.observables[0].results[0]
    if circ.num_qubits < 20:
        statevector = mps.to_vec()
    else:
        statevector = None
    return bonds, exp_val, statevector

def benchmark(
    circ,
    *,
    min_bond_dim,
    bond_dim_limit=None,
    break_on_exceed=False
):
    """
    Run TEBD and TDVP on a given Qiskit QuantumCircuit in one shot.

    Args:
      circ: a qiskit QuantumCircuit
      min_bond_dim: minimum bond dimension for TDVP
      break_on_exceed: if True, will skip the second method if the first
                       exceeds bond_dim_limit (optional)
      bond_dim_limit: threshold for early stopping (optional)

    Returns:
      {
        "TEBD": [(bonds_tebd, exp_tebd)],
        "TDVP": [(bonds_tdvp, exp_tdvp)]
      }
    """
    results = {"TEBD": [], "TDVP": []}

    # TDVP
    bonds_tdvp, exp_tdvp, sv_tdvp = tdvp_simulator(
        circ,
        min_bond=min_bond_dim,
        initial_state=None
    )
    results["TDVP"].append((bonds_tdvp, exp_tdvp))

    # Optionally skip TEBD if TDVP already blew past the limit
    # do_tebd = True
    # if break_on_exceed and bond_dim_limit is not None:
    #     if max(bonds_tdvp, default=0) >= bond_dim_limit:
    #         do_tebd = False

    if circ.num_qubits < 20:
        sv, exp_sv = statevector_expectation(circ)

    # TEBD
    fidelity = None
    if circ.num_qubits < 64:
        bonds_tebd, exp_tebd, sv_tebd = tebd_simulator(
            circ,
            initial_state=None
        )
        if sv_tebd is not None:
            fidelity_tebd = np.abs(np.vdot(sv, sv_tebd))**2
            fidelity_tdvp = np.abs(np.vdot(sv, sv_tdvp))**2
            print("Fidelity:", fidelity_tebd, fidelity_tdvp)
    else:
        bonds_tebd = [np.nan] * len(bonds_tdvp)
        exp_tebd = None

    results["TEBD"].append((bonds_tebd, exp_tebd, fidelity))
    if exp_tebd:
        print("ERROR", np.abs(exp_tebd - exp_tdvp))
    # if exp_tebd and np.abs(exp_tebd - exp_tdvp) > 1e-1:
    #     return None
    
    print("TEBD max", max(bonds_tebd), "TEBD total", sum(bonds_tebd), "TDVP max", max(bonds_tdvp), "TDVP total", sum(bonds_tdvp))

    return results
