# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe
from qiskit.converters import circuit_to_dag

from ..core.data_structures.networks import MPO, MPS
from ..core.methods.dissipation import apply_dissipation
from ..core.methods.dynamic_TDVP import dynamic_TDVP
from ..core.methods.operations import measure
from ..core.methods.stochastic_process import stochastic_process
from .utils.dag_utils import convert_dag_to_tensor_algorithm

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from qiskit.circuit import QuantumCircuit
    from qiskit.dagcircuit import DAGCircuit, DAGOpNode

    from ..core.data_structures.noise_model import NoiseModel
    from ..core.data_structures.simulation_parameters import StrongSimParams, WeakSimParams
    from ..core.libraries.gate_library import BaseGate


def process_layer(dag: DAGCircuit) -> tuple[list[DAGOpNode], list[DAGOpNode], list[DAGOpNode]]:
    # Extract the current layer
    current_layer = dag.front_layer()

    # Prepare groups for even/odd two-qubit gates.
    single_qubit_nodes = []
    even_nodes = []
    odd_nodes = []

    # Separate the current layer into single-qubit and two-qubit gates.
    for node in current_layer:
        # Remove measurement and barrier nodes.
        if node.op.name in {"measure", "barrier"}:
            dag.remove_op_node(node)
            continue

        if len(node.qargs) == 1:
            single_qubit_nodes.append(node)
        elif len(node.qargs) == 2:
            # Group two-qubit gates by even/odd based on the lower qubit index.
            q0, q1 = node.qargs[0]._index, node.qargs[1]._index
            if min(q0, q1) % 2 == 0:
                even_nodes.append(node)
            else:
                odd_nodes.append(node)
        else:
            # TODO: Multi-qubit gates
            msg = "Only single- and two-qubit gates are currently supported."
            raise Exception(msg)

    return single_qubit_nodes, even_nodes, odd_nodes


def apply_single_qubit_gate(state: MPS, node: DAGOpNode) -> None:
    gate = convert_dag_to_tensor_algorithm(node)[0]
    state.tensors[gate.sites[0]] = oe.contract("ab, bcd->acd", gate.tensor, state.tensors[gate.sites[0]])


def construct_generator_MPO(gate: BaseGate, length: int) -> tuple[MPO, int, int]:
    tensors = []

    first_site = min(gate.sites)
    last_site = max(gate.sites)
    if gate.sites[0] < gate.sites[1]:
        first_gen = 0
        second_gen = 1
    else:
        first_gen = 1
        second_gen = 0

    first_site = gate.sites[first_gen]
    last_site = gate.sites[second_gen]
    for site in range(length):
        if site == first_site:
            W = np.zeros((1, 1, 2, 2), dtype=complex)
            W[0, 0] = gate.generator[first_gen]
            tensors.append(W)
        elif site == last_site:
            W = np.zeros((1, 1, 2, 2), dtype=complex)
            W[0, 0] = gate.generator[second_gen]
            tensors.append(W)
        else:
            W = np.zeros((1, 1, 2, 2), dtype=complex)
            W[0, 0] = np.eye(2)
            tensors.append(W)

    mpo = MPO()
    mpo.init_custom(tensors)
    return mpo, first_site, last_site


def apply_window(
    state: MPS, mpo: MPO, first_site: int, last_site: int, sim_params: StrongSimParams | WeakSimParams
) -> tuple[MPS, MPO, list[int]]:
    # Define a window for a local update.
    assert sim_params.window_size is not None
    window = [first_site - sim_params.window_size, last_site + sim_params.window_size]
    window[0] = max(window[0], 0)
    window[1] = min(window[1], state.length - 1)

    # Shift the orthogonality center for sites before the window.
    for i in range(window[0]):
        state.shift_orthogonality_center_right(i)

    short_mpo = MPO()
    short_mpo.init_custom(mpo.tensors[window[0] : window[1] + 1], transpose=False)
    assert window[1] - window[0] + 1 > 1, "MPS cannot be length 1"
    short_state = MPS(length=window[1] - window[0] + 1, tensors=state.tensors[window[0] : window[1] + 1])

    return short_state, short_mpo, window


def apply_two_qubit_gate(state: MPS, node: DAGOpNode, sim_params: StrongSimParams | WeakSimParams) -> None:
    gate = convert_dag_to_tensor_algorithm(node)[0]

    # Construct the MPO for the two-qubit gate.
    mpo, first_site, last_site = construct_generator_MPO(gate, state.length)

    if sim_params.window_size is not None:
        short_state, short_mpo, window = apply_window(state, mpo, first_site, last_site, sim_params)
        dynamic_TDVP(short_state, short_mpo, sim_params)
        # Replace the updated tensors back into the full state.
        for i in range(window[0], window[1] + 1):
            state.tensors[i] = short_state.tensors[i - window[0]]
    else:
        dynamic_TDVP(state, mpo, sim_params)


def CircuitTJM(
    args: tuple[int, MPS, NoiseModel | None, StrongSimParams | WeakSimParams, QuantumCircuit],
) -> NDArray[np.float64]:
    from ..core.data_structures.simulation_parameters import StrongSimParams, WeakSimParams

    _i, initial_state, noise_model, sim_params, circuit = args
    state = copy.deepcopy(initial_state)

    if isinstance(sim_params, StrongSimParams):
        results = np.zeros((len(sim_params.observables), 1))

    dag = circuit_to_dag(circuit)

    while dag.op_nodes():
        single_qubit_nodes, even_nodes, odd_nodes = process_layer(dag)

        for node in single_qubit_nodes:
            apply_single_qubit_gate(state, node)
            dag.remove_op_node(node)

        # Process two-qubit gates in even/odd sweeps.
        for group in [even_nodes, odd_nodes]:
            for node in group:
                apply_two_qubit_gate(state, node, sim_params)
                # Jump process occurs after each two-qubit gate
                apply_dissipation(state, noise_model, dt=1)
                state = stochastic_process(state, noise_model, dt=1)
                dag.remove_op_node(node)

    if isinstance(sim_params, WeakSimParams):
        if not noise_model or all(gamma == 0 for gamma in noise_model.strengths):
            # All shots can be done at once in noise-free model
            return measure(state, sim_params.shots)
        # Each shot is an individual trajectory
        return measure(state, shots=1)
    if isinstance(sim_params, StrongSimParams):
        temp_state = copy.deepcopy(state)
        last_site = 0
        for obs_index, observable in enumerate(sim_params.sorted_observables):
            if observable.site > last_site:
                for site in range(last_site, observable.site):
                    temp_state.shift_orthogonality_center_right(site)
                last_site = observable.site
            results[obs_index, 0] = temp_state.measure(observable)
        return results
    msg = f"CircuitTJM does not support sim_params of type: {type(sim_params)}"
    raise AssertionError(msg)
