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
from mqt.yaqs.core.libraries.gate_library import XX, Z, ZZ
from qiskit.quantum_info import SparsePauliOp, Statevector

def _mid_xx_operator(num_qubits):
    """Helper to build a SparsePauliOp for Z on the middle qubit."""
    mid = num_qubits // 2
    label = ["I"] * num_qubits
    label[mid] = "X"
    label[mid+1] = "X"
    return SparsePauliOp("".join(label))

def tebd_simulator(circ, initial_state=None):
    threshold = 1e-9
    circ2 = copy.deepcopy(circ)
    circ2.clear()
    if initial_state is not None:
        circ2.set_matrix_product_state(initial_state)
    circ2.append(circ, range(circ.num_qubits))
    circ2.save_matrix_product_state(label="final_mps")  
    op_xx = _mid_xx_operator(circ.num_qubits)
    mid = circ.num_qubits // 2
    circ2.save_expectation_value(op_xx, [*range(circ.num_qubits)], label="exp_xx")

    sim = AerSimulator(
        method="matrix_product_state",
        # matrix_product_state_max_bond_dimension=max_bond,
        matrix_product_state_truncation_threshold=threshold,
    )
    tcirc = transpile(circ2, sim)

    result = sim.run(tcirc).result()

    result = sim.run(tcirc).result()
    mps = result.data(0)["final_mps"]
    bonds = [lam[0].shape[0] for lam in mps[0][1::]]
    exp_xx = result.data(0)["exp_xx"]
    return mps, bonds, exp_xx


def tdvp_simulator(circ, min_bond, initial_state=None):
    if initial_state is None:
        initial_state = MPS(length=circ.num_qubits)

    measurements = [Observable(XX(), [circ.num_qubits // 2, circ.num_qubits // 2+1])]
    sim_params = StrongSimParams(measurements, max_bond_dim=2**circ.num_qubits, min_bond_dim=min_bond, get_state=True)

    # circ_flipped = copy.deepcopy(circ).reverse_bits()
    simulator.run(initial_state, circ, sim_params, noise_model=None)
    mps = sim_params.output_state

    bonds = [tensor.shape[1] for tensor in mps.tensors[1::]]
    exp_val = sim_params.observables[0].results[0]
    return mps, bonds, exp_val

def generate_sim_data(
    make_circ, make_args,
    *,
    timesteps,
    min_bond_dim,
    periodic=False,
    break_on_exceed=False,
    bond_dim_limit=None
):
    """
    Generic driver computing TEBD/TDVP bond dims & expectation values
    for a fixed max_bond and threshold.

    make_circ(*make_args, nsteps, periodic=...) -> QuantumCircuit

    Returns:
      { "TEBD": [...], "TDVP": [...] }
    Each entry is a tuple:
      (timesteps, threshold, max_bond, bonds, exp_val)
    """
    results = {"TEBD": [], "TDVP": []}
    mps_tebd = None
    mps_tdvp = None

    calculate_tebd = calculate_tdvp = True

    for i, ts in enumerate(timesteps):
        print("Timesteps =", i)
        # incremental step count
        delta_ts = ts if i == 0 else ts - timesteps[i-1]
        circ_step = make_circ(*make_args, delta_ts, periodic=periodic)

        # TDVP
        if calculate_tdvp:
            mps_tdvp, bonds_tdvp, exp_tdvp = tdvp_simulator(
                circ_step,
                min_bond=min_bond_dim,
                initial_state=mps_tdvp
            )
        else:
            length = len(bonds_tdvp) if 'bonds_tdvp' in locals() else 0
            bonds_tdvp = [np.nan] * length
            exp_tdvp = None

        # TEBD
        if calculate_tebd:
            mps_tebd, bonds_tebd, exp_tebd = tebd_simulator(
                circ_step,
                initial_state=mps_tebd
            )
        else:
            length = len(bonds_tebd) if 'bonds_tebd' in locals() else 0
            bonds_tebd = [np.nan] * length
            exp_tebd = None

        # record results
        results["TDVP"].append((ts, bonds_tdvp, exp_tdvp))
        results["TEBD"].append((ts, bonds_tebd, exp_tebd))
        print("TEBD max", max(bonds_tebd), "TDVP max", max(bonds_tdvp))
        # optional early stop if bond dims exceed limit
        if break_on_exceed and bond_dim_limit is not None:
            if calculate_tdvp and max(bonds_tdvp, default=0) >= bond_dim_limit:
                calculate_tdvp = False
            if calculate_tebd and max(bonds_tebd, default=0) >= bond_dim_limit:
                calculate_tebd = False

    return results


def _mid_xx_operator_qaoa(num_qubits):
    """Helper to build a SparsePauliOp for Z on the middle qubit."""
    mid = num_qubits // 2
    label = ["I"] * num_qubits
    label[mid-1] = "X"
    label[mid] = "X"
    return SparsePauliOp("".join(label))

def tebd_simulator_qaoa(circ, initial_state=None):
    threshold = 1e-9
    circ2 = copy.deepcopy(circ)
    circ2.clear()
    if initial_state is not None:
        circ2.set_matrix_product_state(initial_state)
    circ2.append(circ, range(circ.num_qubits))
    circ2.save_matrix_product_state(label="final_mps")  
    op_xx = _mid_xx_operator_qaoa(circ.num_qubits)
    circ2.save_expectation_value(op_xx, [*range(circ.num_qubits)], label="exp_xx")

    sim = AerSimulator(
        method="matrix_product_state",
        # matrix_product_state_max_bond_dimension=max_bond,
        matrix_product_state_truncation_threshold=threshold,
    )
    tcirc = transpile(circ2, sim)

    result = sim.run(tcirc).result()

    result = sim.run(tcirc).result()
    mps = result.data(0)["final_mps"]
    bonds = [lam[0].shape[0] for lam in mps[0][1::]]
    exp_xx = result.data(0)["exp_xx"]
    return mps, bonds, exp_xx


def tdvp_simulator_qaoa(circ, min_bond, initial_state=None):
    if initial_state is None:
        initial_state = MPS(length=circ.num_qubits)

    measurements = [Observable(XX(), [circ.num_qubits // 2, circ.num_qubits // 2+1])]
    sim_params = StrongSimParams(measurements, max_bond_dim=2**circ.num_qubits, min_bond_dim=min_bond, get_state=True)

    simulator.run(initial_state, circ, sim_params, noise_model=None)
    mps = sim_params.output_state

    bonds = [tensor.shape[1] for tensor in mps.tensors[1::]]
    exp_val = sim_params.observables[0].results[0]
    return mps, bonds, exp_val

def generate_sim_data_qaoa(
    make_circ, make_args,
    *,
    timesteps,
    min_bond_dim,
    periodic=False,
    break_on_exceed=False,
    bond_dim_limit=None
):
    """
    Generic driver computing TEBD/TDVP bond dims & expectation values
    for a fixed max_bond and threshold.

    make_circ(*make_args, nsteps, periodic=...) -> QuantumCircuit

    Returns:
      { "TEBD": [...], "TDVP": [...] }
    Each entry is a tuple:
      (timesteps, threshold, max_bond, bonds, exp_val)
    """
    results = {"TEBD": [], "TDVP": []}
    mps_tebd = None
    mps_tdvp = None

    calculate_tebd = calculate_tdvp = True

    for i, ts in enumerate(timesteps):
        print("Timesteps =", i)
        # incremental step count
        delta_ts = ts if i == 0 else ts - timesteps[i-1]
        circ_step = make_circ(*make_args, delta_ts)

        # TDVP
        if calculate_tdvp:
            mps_tdvp, bonds_tdvp, exp_tdvp = tdvp_simulator_qaoa(
                circ_step,
                min_bond=min_bond_dim,
                initial_state=mps_tdvp
            )
        else:
            length = len(bonds_tdvp) if 'bonds_tdvp' in locals() else 0
            bonds_tdvp = [np.nan] * length
            exp_tdvp = None

        # TEBD
        if calculate_tebd:
            mps_tebd, bonds_tebd, exp_tebd = tebd_simulator_qaoa(
                circ_step,
                initial_state=mps_tebd
            )
        else:
            length = len(bonds_tebd) if 'bonds_tebd' in locals() else 0
            bonds_tebd = [np.nan] * length
            exp_tebd = None

        # record results
        results["TDVP"].append((ts, bonds_tdvp, exp_tdvp))
        results["TEBD"].append((ts, bonds_tebd, exp_tebd))
        print("TEBD max", max(bonds_tebd), "TDVP max", max(bonds_tdvp))
        # optional early stop if bond dims exceed limit
        if break_on_exceed and bond_dim_limit is not None:
            if calculate_tdvp and max(bonds_tdvp, default=0) >= bond_dim_limit:
                calculate_tdvp = False
            if calculate_tebd and max(bonds_tebd, default=0) >= bond_dim_limit:
                calculate_tebd = False

    return results


def generate_sim_data_supplemental(
    make_circ, make_args,
    *,
    timesteps,
    min_bond_dim,
    periodic=False,
    break_on_exceed=False,
    bond_dim_limit=None
):
    """
    Generic driver computing TEBD/TDVP bond dims & expectation values
    for a fixed max_bond and threshold.

    make_circ(*make_args, nsteps, periodic=...) -> QuantumCircuit

    Returns:
      { "TEBD": [...], "TDVP": [...] }
    Each entry is a tuple:
      (timesteps, threshold, max_bond, bonds, exp_val)
    """
    results = {"TEBD": [], "TDVP": []}
    mps_tebd = None
    mps_tdvp = None

    calculate_tebd = calculate_tdvp = True

    # delta_ts = ts if i == 0 else ts - timesteps[i-1]
    # incremental step count
    # delta_ts = ts if i == 0 else ts - timesteps[i-1]
    circ_step = make_circ(*make_args, timesteps=1, periodic=periodic)
    basis_gates = ['cx', 'rx', 'rz']
    circ_step = transpile(circ_step, basis_gates=basis_gates, optimization_level=3)

    for i, ts in enumerate(timesteps):
        print("Timesteps =", i)

        # print(circ_step)
        # TDVP
        if calculate_tdvp:
            mps_tdvp, bonds_tdvp, exp_tdvp = tdvp_simulator(
                circ_step,
                min_bond=min_bond_dim,
                initial_state=mps_tdvp
            )
        else:
            length = len(bonds_tdvp) if 'bonds_tdvp' in locals() else 0
            bonds_tdvp = [np.nan] * length
            exp_tdvp = None

        # TEBD
        if calculate_tebd:
            mps_tebd, bonds_tebd, exp_tebd = tebd_simulator(
                circ_step,
                initial_state=mps_tebd
            )
        else:
            length = len(bonds_tebd) if 'bonds_tebd' in locals() else 0
            bonds_tebd = [np.nan] * length
            exp_tebd = None

        # record results
        results["TDVP"].append((ts, bonds_tdvp, exp_tdvp))
        results["TEBD"].append((ts, bonds_tebd, exp_tebd))
        print("TEBD max", max(bonds_tebd), "TDVP max", max(bonds_tdvp))
        # optional early stop if bond dims exceed limit
        if break_on_exceed and bond_dim_limit is not None:
            if calculate_tdvp and max(bonds_tdvp, default=0) >= bond_dim_limit:
                calculate_tdvp = False
            if calculate_tebd and max(bonds_tebd, default=0) >= bond_dim_limit:
                calculate_tebd = False

    return results


