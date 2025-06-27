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
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
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


def _mid_z_operator(n):
    """Z⊗Z on the middle bond (as an XX-string in SparsePauliOp)."""
    m = n // 2
    label = ["I"] * n
    label[m] = label[m+1] = "X"
    return SparsePauliOp("".join(label))

def statevector_expectation(circ, init=None):
    circ2 = copy.deepcopy(circ)
    circ2.save_statevector()
    sim = AerSimulator(method="statevector")
    tc = transpile(circ2, sim)
    res = sim.run(tc, initial_statevector=init).result() if init is not None else sim.run(tc).result()
    sv = Statevector(res.get_statevector(tc))
    return sv, sv.expectation_value(_mid_z_operator(tc.num_qubits)).real

def tebd(circ, init=None, bond_dim=None, thresh=1e-13):
    circ2 = copy.deepcopy(circ); circ2.clear()
    if init is not None:
        circ2.set_matrix_product_state(init)
    circ2.append(circ, list(range(circ.num_qubits)))
    circ2.save_matrix_product_state(label="mps")
    circ2.save_statevector()
    sim = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=bond_dim,
        matrix_product_state_truncation_threshold=thresh,
    )
    tc = transpile(circ2, sim)
    r = sim.run(tc).result()
    sv = Statevector(r.get_statevector(tc))
    return r.data(0)["mps"], sv, sv.expectation_value(_mid_z_operator(circ.num_qubits)).real

def tdvp(circ, min_bond, init=None):
    if init == None:
        init = MPS(length=circ.num_qubits)
    meas = [Observable(XX(), [circ.num_qubits//2, circ.num_qubits//2+1])]
    params = StrongSimParams(meas, max_bond_dim=2**(circ.num_qubits//2),
                             min_bond_dim=min_bond, get_state=True)
    # circ_flipped = copy.deepcopy(circ)
    # circ_flipped.reverse_bits()
    simulator.run(init, circ, params, noise_model=None)
    mps_out = params.output_state

    return mps_out, mps_out.to_vec(), params.observables[0].results[0]

def _infidelity(a, b):
    return np.abs(1 - np.abs(np.vdot(a, b))**2)

def generate_error_data(make_circ, make_args, *,
                        min_bonds, timesteps, periodic=False):
    """
    Generic driver to compute TEBD/TDVP errors vs. an incrementally
    updated exact statevector.

    make_circ(*make_args, dt, nsteps, periodic=...) -> QuantumCircuit
    """
    results = {"TEBD": [], "TDVP": []}
    exact_sv = None
    exact_exp = None
    mps_tebd = None
    mps_tdvp = None

    exact_svs = []
    exact_exp_vals = []
    for j, mb in enumerate(min_bonds):
        print("Min bond dim", mb)
        for i, ts in enumerate(timesteps):
            print("  Timesteps =", ts)
            delta_ts = ts - (timesteps[i-1] if i else 0)

            # 1) build the small-step circuit
            circ_full = make_circ(*make_args, ts, periodic=periodic)
            circ_step = make_circ(*make_args, delta_ts, periodic=periodic)

            # 2) only on the first bond-dimension do we advance the exact SV
            if j == 0:
                exact_sv, exact_exp = statevector_expectation(circ_full, init=exact_sv)
                exact_svs.append(exact_sv)
                exact_exp_vals.append(exact_exp)
                mps_tebd, sv_tebd, exp_tebd = tebd(
                    circ_step,
                    init=mps_tebd,
                    bond_dim=2**(circ_step.num_qubits // 2)
                )
                results["TEBD"].append((
                    ts,
                    _infidelity(exact_sv, sv_tebd),
                    abs(exact_exp - exp_tebd)
                ))

            # 3) reset TDVP MPS at the start of each timestep sweep
            if i == 0:
                mps_tdvp = None

            # 4) run TDVP against the same small-step circuit
            mps_tdvp, sv_tdvp, exp_tdvp = tdvp(
                circ_step,
                min_bond=mb,
                init=mps_tdvp
            )
            results["TDVP"].append((
                mb,
                ts,
                _infidelity(exact_svs[i], sv_tdvp),
                abs(exact_exp_vals[i] - exp_tdvp)
            ))

    return results


def plot_error_vs_depth(results, bond_dims) -> None:
    # Create a 1xN figure (one subplot per threshold)
    _fig, axes = plt.subplots(2, 1, sharex=True)
    # Map bond dimensions to distinct colors
    color_map = {2: "lightsalmon", 4: "salmon", 8: "red", 16: "darkred", 32: "black"}
    # Use different markers for each simulator
    marker_map = {"TEBD": "^", "TDVP": "o"}

    axes[0].set_title("Infidelity")
    axes[1].set_title("Local Observable Error")
    for i, name in enumerate(["infidelity", "error"]):
        ax = axes[i]
        for j, bond_dim in enumerate(bond_dims):
            # Extract data for TEBD and TDVP
            if name == "infidelity":
                tdvp_data = [(depth, infid) for (bd, depth, infid, err) in results["TDVP"] if bd == bond_dim]
            if name == "error":
                tdvp_data = [(depth, err) for (bd, depth, infid, err) in results["TDVP"] if bd == bond_dim]

            # Sort by circuit depth
            # tebd_data.sort(key=operator.itemgetter(0))
            # tdvp_data.sort(key=operator.itemgetter(0))

            # Unpack for plotting
            # tebd_depths = [x[0] for x in tebd_data]
            # tebd_errors = [x[1] for x in tebd_data]
            tdvp_depths = [x[0] for x in tdvp_data]
            tdvp_errors = [x[1] for x in tdvp_data]

            # Plot main curves on the primary axis
            # ax.plot(
            #     tebd_depths,
            #     tebd_errors,
            #     label=f"TEBD" if i == 0 else "",
            #     color=color_map[bond_dim],
            #     marker=marker_map["TEBD"],
            #     linestyle="--",
            # )
            ax.plot(
                tdvp_depths,
                tdvp_errors,
                label=bond_dim if i == 0 else "",
                color=color_map[bond_dim],
                marker=marker_map["TDVP"],
                linestyle="-",
                linewidth=3
            )
        if name == "infidelity":
            tebd_data = [(depth, infid) for (depth, infid, err) in results["TEBD"]]
        elif name == "error":
            tebd_data = [(depth, err) for (depth, infid, err) in results["TEBD"]]
        tebd_data.sort(key=operator.itemgetter(0))
        tebd_depths = [x[0] for x in tebd_data]
        tebd_errors = [x[1] for x in tebd_data]
        ax.plot(
        tebd_depths,
        tebd_errors,
        label=f"TEBD" if i == 0 else "",
        color='k',
        marker=marker_map["TEBD"],
        linestyle="--",
        linewidth=3
        )
        ax.set_yscale("log")
        ax.set_ylim(1e-16, 5e-1)

        ax.grid(True)

    axes[1].set_ylabel("Circuit depth (Trotter steps)")
    axes[0].set_ylabel("Infidelity (log scale)")
    axes[1].set_ylabel("Local observable error (log scale)")

    axes[0].legend(loc="upper left", fontsize="small")
    axes[0].set_xlabel("Circuit depth (Trotter steps)")
    axes[0].set_xlabel(None)
    plt.tight_layout()
    plt.show()
