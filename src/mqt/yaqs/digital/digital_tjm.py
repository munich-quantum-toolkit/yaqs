# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Digital Tensor Jump Method.

This module provides functions for simulating quantum circuits using the Tensor Jump Method (TJM). It includes
utilities for converting quantum circuits to DAG representations, processing gate layers, applying gates to
matrix product states (MPS) and constructing generator MPOs.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, cast

import numpy as np
import opt_einsum as oe
from qiskit.converters import circuit_to_dag

from ..core.data_structures.mpo import MPO
from ..core.data_structures.mpo_utils import resolve_lr_tensor
from ..core.data_structures.mps import MPS
from ..core.data_structures.noise_model import NoiseModel
from ..core.data_structures.simulation_parameters import (
    StrongSimParams,
    WeakSimParams,
)
from ..core.libraries.gate_library import BaseGate, GateLibrary
from ..core.methods.decompositions import merge_two_site, split_two_site
from ..core.methods.dissipation import apply_dissipation
from ..core.methods.stochastic_process import stochastic_process
from ..core.methods.tdvp.sweep_utils import get_min_keep, renorm_drift, uses_fixed_chi
from ..core.methods.tdvp.tdvp import evolve_window
from ..core.random_utils import make_trajectory_rng
from .utils.dag_utils import convert_dag_to_tensor_algorithm

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from qiskit.circuit import QuantumCircuit
    from qiskit.dagcircuit import DAGCircuit, DAGOpNode

    from ..core.data_structures.simulation_parameters import GateMode
    from ..core.methods.decompositions import TruncMode


def create_local_noise_model(noise_model: NoiseModel, first_site: int, last_site: int) -> NoiseModel:
    """Create local noise model.

    Create a local noise model from a global noise model for a given gate.

    Args:
        noise_model (NoiseModel): The global noise model.
        first_site (int): The first site of the gate.
        last_site (int): The last site of the gate.

    Returns:
        NoiseModel: The local noise model.
    """
    affected_sites = [first_site, last_site]

    local_processes = [
        process
        for process in noise_model.processes
        if process["sites"] == affected_sites or process["sites"] == [first_site] or process["sites"] == [last_site]
    ]
    return NoiseModel(local_processes)


def _is_terminal_measure(dag: DAGCircuit, node: DAGOpNode) -> bool:
    """Return whether a measure node has no later gates on its qubits.

    A measurement is terminal when no subsequent DAG operation acts on any qubit
    that the measurement targets.

    Args:
        dag: Circuit DAG containing ``node``.
        node: Measure operation node to classify.

    Returns:
        True if no later op nodes share qubits with ``node``; False otherwise.
    """
    measured = {dag.find_bit(q).index for q in node.qargs}
    topo = list(dag.topological_op_nodes())
    try:
        node_index = topo.index(node)
    except ValueError:
        return True
    for later in topo[node_index + 1 :]:
        later_qubits = {dag.find_bit(q).index for q in later.qargs}
        if later_qubits & measured:
            return False
    return True


def process_layer(dag: DAGCircuit) -> tuple[list[DAGOpNode], list[DAGOpNode], list[DAGOpNode], list[DAGOpNode]]:
    """Process quantum circuit layer before applying to MPS.

    Processes the current layer of a DAGCircuit and categorizes nodes into single-qubit, even-indexed two-qubit,
    and odd-indexed two-qubit gates.

    Args:
        dag (DAGCircuit): The directed acyclic graph representing the quantum circuit.

    Returns:
        tuple[list[DAGOpNode], list[DAGOpNode], list[DAGOpNode], list[DAGOpNode]]: A tuple containing four lists:
            - single_qubit_nodes: Nodes corresponding to single-qubit gates.
            - even_nodes: Nodes corresponding to two-qubit gates where the lower qubit index is even.
            - odd_nodes: Nodes corresponding to two-qubit gates where the lower qubit index is odd.
            - measure_barriers: Labelled barriers ("SAMPLE_OBSERVABLES") used as sampling points.

    Raises:
        NotImplementedError: If a node with more than two qubits is encountered.
        ValueError: If a non-terminal ``measure`` operation is encountered.
    """
    # Extract the current layer
    current_layer = dag.front_layer()

    # Prepare groups for even/odd two-qubit gates.
    single_qubit_nodes = []
    even_nodes = []
    odd_nodes = []
    measure_barriers = []

    # Separate the current layer into single-qubit and two-qubit gates.
    for node in current_layer:
        name = node.op.name

        # Drop terminal measurements during simulation. Unlike ``convert_dag_to_tensor_algorithm``,
        # which rejects ``measure`` when building a gate list, the live DAG path removes
        # measurement nodes so terminal Qiskit measurements do not block evolution.
        if name == "measure":
            if _is_terminal_measure(dag, node):
                dag.remove_op_node(node)
            else:
                msg = (
                    "Non-terminal measure operations are not supported during simulation; "
                    "removing them would ignore state collapse and classical dependencies."
                )
                raise ValueError(msg)
            continue

        # Keep ONLY barriers with label "SAMPLE_OBSERVABLES" (case-insensitive). Remove all other barriers.
        if name == "barrier":
            label = getattr(node.op, "label", None)
            if label is not None and str(label).upper() == "SAMPLE_OBSERVABLES":
                measure_barriers.append(node)
            else:
                dag.remove_op_node(node)
            continue

        if len(node.qargs) == 1:
            single_qubit_nodes.append(node)
        elif len(node.qargs) == 2:
            # Group two-qubit gates by even/odd based on the lower qubit index.
            q0, q1 = node.qargs[0]._index, node.qargs[1]._index  # noqa: SLF001
            if min(q0, q1) % 2 == 0:
                even_nodes.append(node)
            else:
                odd_nodes.append(node)
        else:
            raise NotImplementedError

    # Sort the nodes to minimize orthogonality center movement (zig-zag optimization)
    single_qubit_nodes.sort(key=lambda node: node.qargs[0]._index)  # noqa: SLF001
    even_nodes.sort(key=lambda node: min(node.qargs[0]._index, node.qargs[1]._index))  # noqa: SLF001
    odd_nodes.sort(key=lambda node: min(node.qargs[0]._index, node.qargs[1]._index))  # noqa: SLF001

    return single_qubit_nodes, even_nodes, odd_nodes, measure_barriers


def apply_single_qubit_gate(state: MPS, node: DAGOpNode) -> None:
    """Apply single qubit gate.

    This function applies a single-qubit gate to the MPS, used during circuit simulation.

    Parameters:
    state (MPS): The matrix product state (MPS) representing the quantum state.
    node (DAGOpNode): The directed acyclic graph (DAG) operation node representing the gate to be applied.
    """
    gate = convert_dag_to_tensor_algorithm(node)[0]
    site = gate.sites[0]
    state.tensors[site] = oe.contract("ab, bcd->acd", gate.tensor, state.tensors[site])
    if state.orthogonality_center is not None and state.orthogonality_center != site:
        state.set_center(None)


def construct_generator_mpo(gate: BaseGate, length: int) -> tuple[MPO, int, int]:
    """Construct Generator MPO.

    Constructs a Matrix Product Operator (MPO) representation of a generator for a given gate over a
    specified length.

    Args:
        gate: The gate containing the generator and the sites it acts on.
        length: The total number of sites in the system.

    Returns:
        A tuple containing the constructed MPO, the first site index, and the last site index.
    """
    tensors = []

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
            w = np.zeros((1, 1, 2, 2), dtype=complex)
            w[0, 0] = gate.generator[first_gen]
            tensors.append(w)
        elif site == last_site:
            w = np.zeros((1, 1, 2, 2), dtype=complex)
            w[0, 0] = gate.generator[second_gen]
            tensors.append(w)
        else:
            w = np.zeros((1, 1, 2, 2), dtype=complex)
            w[0, 0] = np.eye(2)
            tensors.append(w)

    mpo = MPO()
    mpo.custom(tensors)
    return mpo, first_site, last_site


def apply_window(state: MPS, mpo: MPO, first_site: int, last_site: int, window_size: int) -> tuple[MPS, MPO, list[int]]:
    """Apply Window.

    Apply a window to the given MPS and MPO for a local update.

    Args:
        state (MPS): The matrix product state (MPS) to be updated.
        mpo (MPO): The matrix product operator (MPO) to be applied.
        first_site (int): The index of the first site in the window.
        last_site (int): The index of the last site in the window.
        window_size: Number of sites on each side of first and last site

    Returns:
        tuple[MPS, MPO, list[int]]: A tuple containing the shortened MPS, the shortened MPO, and the window indices.
    """
    # Define a window for a local update.
    window = [first_site - window_size, last_site + window_size]
    window[0] = max(window[0], 0)
    window[1] = min(window[1], state.length - 1)

    # Shift the orthogonality center to the start of the window.
    if state.orthogonality_center is not None:
        state.shift_center_to(window[0])
    else:
        for i in range(window[0]):
            state.shift_orthogonality_center_right(i)

    short_mpo = MPO()
    short_mpo.custom(mpo.tensors[window[0] : window[1] + 1], transpose=False)
    assert window[1] - window[0] + 1 > 1, "MPS cannot be length 1"
    short_state = MPS(length=window[1] - window[0] + 1, tensors=state.tensors[window[0] : window[1] + 1])
    short_state.set_center(state.orthogonality_center)

    return short_state, short_mpo, window


def apply_two_qubit_gate_tdvp(
    state: MPS,
    gate: BaseGate,
    sim_params: StrongSimParams | WeakSimParams,
) -> tuple[int, int]:
    """Apply a two-qubit gate via generator MPO and TDVP.

    Long-range gates use local two-site TDVP (2TDVP) on a window-local MPS without
    post-sweep renormalization before grafting tensors back into the full chain.
    Nearest-neighbor gates in hybrid ``gate_mode="tdvp"`` use TEBD instead;
    callers should route via :func:`apply_two_qubit_gate`.

    Args:
        state: MPS updated in place.
        gate: Internal gate object from the gate library.
        sim_params: Truncation and Krylov settings for TDVP.

    Returns:
        ``(first_site, last_site)`` spanning the gate support in MPS order.

    Raises:
        ValueError: If ``sim_params.tdvp_mode`` is not ``"2site"``.

    """
    if sim_params.tdvp_mode != "2site":
        msg = f'apply_two_qubit_gate_tdvp only supports tdvp_mode="2site"; got {sim_params.tdvp_mode!r}.'
        raise ValueError(msg)
    mpo, first_site, last_site = construct_generator_mpo(gate, state.length)

    window_size = 1
    short_state, short_mpo, window = apply_window(state, mpo, first_site, last_site, window_size)

    evolve_window(short_state, short_mpo, sim_params)
    for i in range(window[0], window[1] + 1):
        state.tensors[i] = short_state.tensors[i - window[0]]
    if uses_fixed_chi(sim_params):
        renorm_drift(state, sim_params)
    if state.orthogonality_center is not None:
        state.set_center(window[0])

    return first_site, last_site


def apply_two_qubit_gate_tebd(
    state: MPS,
    gate: BaseGate,
    sim_params: StrongSimParams | WeakSimParams,
) -> tuple[int, int]:
    """Apply a two-qubit gate via TEBD/SVD, inserting adjacent SWAPs if needed.

    Args:
        state: MPS updated in place.
        gate: Internal gate object from the gate library.
        sim_params: Truncation settings shared with TDVP/MPS splitting.

    Returns:
        ``(left_site, right_site)`` spanning the gate support in MPS order.
    """

    def apply_swap(site_left: int) -> None:
        swap_gate = GateLibrary.swap()
        swap_gate.set_sites(site_left, site_left + 1)
        apply_two_qubit_gate_tebd(state, swap_gate, sim_params)

    site0, site1 = gate.sites[0], gate.sites[1]
    if abs(site0 - site1) != 1:
        left = min(site0, site1)
        right = max(site0, site1)

        for i in range(right - 1, left, -1):
            apply_swap(i)

        gate_adj = copy.deepcopy(gate)
        if site0 == left:
            gate_adj.set_sites(left, left + 1)
        else:
            gate_adj.set_sites(left + 1, left)
        apply_two_qubit_gate_tebd(state, gate_adj, sim_params)

        for i in range(left + 1, right):
            apply_swap(i)

        return left, right

    left_site = min(site0, site1)
    right_site = max(site0, site1)
    u_gate = resolve_lr_tensor(gate, left_site, right_site)

    left_tensor = state.tensors[left_site]
    right_tensor = state.tensors[right_site]
    d_left, d_right = left_tensor.shape[0], right_tensor.shape[0]

    merged = merge_two_site(left_tensor, right_tensor)
    theta_4 = merged.reshape(d_left, d_right, merged.shape[1], merged.shape[2])
    theta_new = np.asarray(oe.contract("ijkl,klab->ijab", u_gate, theta_4), dtype=np.complex128)
    merged_new = theta_new.reshape(d_left * d_right, merged.shape[1], merged.shape[2])

    new_left, new_right = split_two_site(
        merged_new,
        [d_left, d_right],
        svd_distribution="right",
        trunc_mode=cast("TruncMode", sim_params.trunc_mode),
        threshold=sim_params.svd_threshold,
        max_bond_dim=sim_params.max_bond_dim,
        min_keep=get_min_keep(sim_params),
    )
    state.tensors[left_site] = new_left
    state.tensors[right_site] = new_right
    state.update_center_after_split(left_site, right_site, "right")
    return left_site, right_site


def apply_long_range_gate_mpo(
    state: MPS,
    gate: BaseGate,
    sim_params: StrongSimParams | WeakSimParams,
) -> tuple[int, int]:
    """Apply a long-range two-qubit gate via :meth:`~mqt.yaqs.core.data_structures.mpo.MPO.multiply`.

    Args:
        state: MPS updated in place.
        gate: Two-qubit gate with sites and MPO data populated.
        sim_params: Truncation settings for the compression sweep.

    Returns:
        ``(first_site, last_site)`` spanning the gate support in MPS order.
    """
    site0, site1 = gate.sites[0], gate.sites[1]
    first_site = min(site0, site1)
    last_site = max(site0, site1)
    MPO.from_gate(gate, state.length).multiply(state, sim_params=sim_params, compress=True)
    return first_site, last_site


def apply_two_qubit_gate(state: MPS, node: DAGOpNode, sim_params: StrongSimParams | WeakSimParams) -> tuple[int, int]:
    """Apply a two-qubit gate using the configured two-qubit gate mode.

    Args:
        state: MPS updated in place.
        node: DAG node for the two-qubit gate.
        sim_params: Simulation parameters including ``gate_mode``.

    Returns:
        ``(first_site, last_site)`` for downstream local noise handling.

    Raises:
        ValueError: If ``gate_mode`` is unknown.
    """
    gate = convert_dag_to_tensor_algorithm(node)[0]
    site0, site1 = gate.sites[0], gate.sites[1]
    is_nearest_neighbor = abs(site0 - site1) == 1
    gate_mode: GateMode = getattr(sim_params, "gate_mode", "mpo")
    # Matrix-backed custom gates have no ``generator`` and bypass the TDVP window
    # path in ``tdvp`` / ``full-tdvp`` modes (TEBD for NN, MPO for LR).
    has_generator = getattr(gate, "generator", None) is not None

    if gate_mode == "full-tdvp":
        if has_generator:
            return apply_two_qubit_gate_tdvp(state, gate, sim_params)
        if is_nearest_neighbor:
            return apply_two_qubit_gate_tebd(state, gate, sim_params)
        return apply_long_range_gate_mpo(state, gate, sim_params)

    if gate_mode == "swaps":
        return apply_two_qubit_gate_tebd(state, gate, sim_params)

    if gate_mode == "tdvp":
        if is_nearest_neighbor:
            return apply_two_qubit_gate_tebd(state, gate, sim_params)
        if has_generator:
            return apply_two_qubit_gate_tdvp(state, gate, sim_params)
        return apply_long_range_gate_mpo(state, gate, sim_params)

    if gate_mode == "mpo":
        if is_nearest_neighbor:
            return apply_two_qubit_gate_tebd(state, gate, sim_params)
        return apply_long_range_gate_mpo(state, gate, sim_params)

    msg = f"Unknown gate_mode: {gate_mode!r}"
    raise ValueError(msg)


def digital_tjm(
    args: tuple[int, MPS, NoiseModel | None, StrongSimParams | WeakSimParams, QuantumCircuit],
) -> tuple[NDArray[np.float64] | dict[int, int], NDArray[np.float64] | None, MPS | None]:
    """Digital Tensor Jump Method.

    Simulates a quantum circuit using the Tensor Jump Method.

    Args:
        args: A tuple containing the following elements:
            - An index or identifier, primarily for parallelization
            - The initial state of the system represented as a Matrix Product MPS.
            - The noise model to be applied during the simulation, or None if no noise is to be applied.
            - Parameters for the simulation, either for strong or weak simulation.
            - The quantum circuit to be simulated.

    Returns:
        The results of the simulation.
        If StrongSimParams are used, the results are the measured observables.
        If WeakSimParams are used, the results are the measurement outcomes for each shot.
    """
    traj_idx, initial_state, noise_model, sim_params, circuit = args

    state = copy.deepcopy(initial_state)
    dag = circuit_to_dag(circuit)
    diagnostics: NDArray[np.float64] | None = None

    # Initialize results depending on simulation type
    if isinstance(sim_params, StrongSimParams):
        num_cols = (sim_params.num_mid_measurements + 2) if sim_params.sample_layers else 1
        diagnostics = np.zeros((3, num_cols), dtype=np.float64)
        if sim_params.sample_layers:
            results = np.zeros((len(sim_params.sorted_observables), sim_params.num_mid_measurements + 2))
            state.record_diagnostics(diagnostics, 0)
            state.evaluate_observables(sim_params, results, 0)
        else:
            results = np.zeros((len(sim_params.sorted_observables), 1))

    rng = make_trajectory_rng(traj_idx, base_seed=sim_params.random_seed)

    col_idx = 0
    while dag.op_nodes():
        single_qubit_nodes, even_nodes, odd_nodes, measure_barriers = process_layer(dag)

        for node in single_qubit_nodes:
            apply_single_qubit_gate(state, node)
            dag.remove_op_node(node)

        # Process two-qubit gates in even/odd sweeps.
        for _, group in [("even", even_nodes), ("odd", odd_nodes)]:
            for node in group:
                first_site, last_site = apply_two_qubit_gate(state, node, sim_params)

                if noise_model is None or all(proc["strength"] == 0 for proc in noise_model.processes):
                    # Normalizes state
                    state.normalize(form="B", decomposition="QR")
                else:
                    local_noise_model = create_local_noise_model(noise_model, first_site, last_site)
                    apply_dissipation(state, local_noise_model, dt=1, sim_params=sim_params)
                    state = stochastic_process(state, local_noise_model, dt=1, sim_params=sim_params, rng=rng)

                dag.remove_op_node(node)

        # Process measurement barriers (only when sampling layers in strong sim)
        if isinstance(sim_params, StrongSimParams) and sim_params.sample_layers:
            for measure_barrier in measure_barriers:
                dag.remove_op_node(measure_barrier)
                col_idx += 1
                assert diagnostics is not None
                state.record_diagnostics(diagnostics, col_idx)
                state.evaluate_observables(sim_params, results, col_idx)

    if isinstance(sim_params, WeakSimParams):
        per_call_shots = _per_call_shots(sim_params)
        if not noise_model or all(proc["strength"] == 0 for proc in noise_model.processes):
            counts = state.measure_shots(per_call_shots)
            final = state if sim_params.get_state else None
            return counts, None, final
        return state.measure_shots(shots=1), None, state if sim_params.get_state else None

    if state.orthogonality_center is None:
        state.normalize(form="B", decomposition="QR")

    assert isinstance(sim_params, StrongSimParams)
    assert diagnostics is not None
    final_col = results.shape[1] - 1
    state.record_diagnostics(diagnostics, final_col)
    state.evaluate_observables(sim_params, results, final_col)
    final = state if sim_params.get_state else None
    return results, diagnostics, final


def _per_call_shots(sim_params: WeakSimParams) -> int:
    """Return shots for this worker call (may differ from ``sim_params.shots`` when noisy).

    Returns:
        Number of shots for the current worker invocation.
    """
    try:
        from mqt.yaqs.simulator import WORKER_CTX  # noqa: PLC0415

        return int(WORKER_CTX["per_call_shots"])
    except KeyError:
        return sim_params.shots
