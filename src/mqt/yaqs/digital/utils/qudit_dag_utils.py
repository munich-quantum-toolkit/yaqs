# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""DAG utilities for qudit quantum circuits.

This module provides a directed acyclic graph (DAG) representation for qudit
circuits. Unlike the Qiskit-based :mod:`dag_utils`, this
DAG works natively with :class:`mqt.qudits.quantum_circuit.QuantumCircuit` and
tracks which energy *levels* each gate acts on — information that Qiskit's
DAGCircuit cannot represent.

Key classes and functions:
    - :class:`QuditOpNode`: DAG node holding the original gate object and its level footprint.
    - :class:`QuditDAG`: The DAG itself, built from a ``mqt.qudits.QuantumCircuit``.
    - :func:`circuit_to_dag`: Convert a ``mqt.qudits.QuantumCircuit`` to a :class:`QuditDAG`.
    - :func:`dag_to_circuit`: Reconstruct a ``mqt.qudits.QuantumCircuit`` from a :class:`QuditDAG`.
    - :func:`check_longest_gate`: Longest qudit span in the front layer (mirrors ``dag_utils``).
    - :func:`get_temporal_zone`: Extract gates that act only on a given qudit window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mqt.qudits.quantum_circuit import QuantumCircuit

if TYPE_CHECKING:
    from collections.abc import Generator

    from mqt.qudits.quantum_circuit.gate import Gate


def _levels_for_gate(gate: Gate, _qudit_idx: int, dimension: int) -> list[int]:
    """Return which energy levels *gate* acts on for a given qudit.

    MQT Qudit gates carry ``lev_a`` and ``lev_b`` (the two transition levels).
    When both are 0 the gate does not have a meaningful two-level transition
    (e.g. a full-unitary gate), so the full level range is returned.

    Args:
        gate: The MQT Qudit gate object.
        _qudit_idx: Index of the qudit within the full circuit (reserved for a
            future per-qudit override; not used today).
        dimension: Physical dimension of that qudit.

    Returns:
        Sorted list of level indices the gate touches.
    """
    lev_a: int = getattr(gate, "lev_a", 0)
    lev_b: int = getattr(gate, "lev_b", 0)

    if lev_a == lev_b:
        return list(range(dimension))

    return sorted({lev_a, lev_b})


class QuditDAGNode:
    """Base node in a :class:`QuditDAG`.

    Attributes:
        index: Unique integer index assigned at creation.
        dependencies: Set of predecessor nodes (direct data dependencies).
    """

    def __init__(self, index: int) -> None:
        """Initialize a base DAG node.

        Args:
            index: Unique integer index for this node.
        """
        self.index: int = index
        self.dependencies: set[QuditDAGNode] = set()

    def add_dependency(self, node: QuditDAGNode) -> None:
        """Register *node* as a direct predecessor."""
        self.dependencies.add(node)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f"Node({self.index})"


class QuditOpNode(QuditDAGNode):
    """DAG node representing a single quantum gate operation.

    Attributes:
        op_name: QASM tag of the gate (e.g. ``"cx"``, ``"rz"``).
        gate: Original MQT Qudit :class:`~mqt.qudits.quantum_circuit.gate.Gate`
            object.  Use ``node.gate.to_matrix()`` to obtain the unitary.
        target_qudits: List of qudit indices the gate acts on.
        dimensions: Physical dimension of each target qudit (same order).
        levels: Mapping ``{qudit_index: [level, ...]}`` of touched levels per qudit.
        params: Raw gate parameters (forwarded from the original gate).
    """

    def __init__(
        self,
        index: int,
        gate: Gate,
        target_qudits: list[int],
        dimensions: list[int],
        levels: dict[int, list[int]],
    ) -> None:
        """Initialize an operation node.

        Args:
            index: Unique integer index.
            gate: MQT Qudit gate object.
            target_qudits: Qudit indices the gate acts on.
            dimensions: Physical dimension of each target qudit.
            levels: Touched energy levels per qudit.
        """
        super().__init__(index)
        self.gate: Gate = gate
        self.op_name: str = getattr(gate, "qasm_tag", "unknown")
        self.target_qudits: list[int] = target_qudits
        self.dimensions: list[int] = dimensions
        self.levels: dict[int, list[int]] = levels
        self.params: object = getattr(gate, "_params", None)

    @property
    def qargs(self) -> list[int]:
        """Target qudit indices (alias for ``target_qudits``)."""
        return self.target_qudits

    @property
    def op(self) -> Gate:
        """The original gate object (alias for ``gate``)."""
        return self.gate

    def to_dict(self) -> dict:
        """Serialize this node to a plain dict.

        Returns:
            A dict with keys ``op``, ``targets``, ``levels``, ``dims``, and ``deps``.
        """
        return {
            "op": self.op_name,
            "targets": self.target_qudits,
            "levels": self.levels,
            "dims": self.dimensions,
            "deps": [n.index for n in self.dependencies],
        }

    def __repr__(self) -> str:
        """Return a concise string representation."""
        dep_ids = [n.index for n in self.dependencies]
        return (
            f"OpNode({self.index}: {self.op_name}, targets={self.target_qudits}, levels={self.levels}, deps={dep_ids})"
        )


class QuditDAG:
    """Directed acyclic graph for a qudit quantum circuit.

    The DAG is built from a ``mqt.qudits.QuantumCircuit``: each gate becomes a
    :class:`QuditOpNode` and edges encode data dependencies (two gates on the
    same qudit are connected in program order).

    Args:
        circuit: Source circuit.  Pass ``None`` together with *dimensions* to
            create an empty DAG (used internally by :meth:`copy_empty_like`).
        dimensions: Physical dimensions to use when *circuit* is ``None``.
    """

    def __init__(
        self,
        circuit: QuantumCircuit | None = None,
        dimensions: list[int] | None = None,
    ) -> None:
        """Initialize the DAG.

        Args:
            circuit: Source qudit circuit; ``None`` creates an empty DAG.
            dimensions: Physical dimensions when *circuit* is ``None``.
        """
        if circuit is not None:
            self.dimensions: list[int] = list(circuit.dimensions)
        else:
            self.dimensions = list(dimensions) if dimensions else []

        self.circuit: QuantumCircuit | None = circuit
        self.num_qudits: int = len(self.dimensions)
        self.nodes: list[QuditOpNode] = []

        if circuit is not None:
            self._build(circuit)

    def _build(self, circuit: QuantumCircuit) -> None:
        """Populate *nodes* and wire dependency edges."""
        last_node_per_qudit: list[QuditOpNode | None] = [None] * self.num_qudits

        for idx, instruction in enumerate(circuit.instructions):
            targets: list[int] = getattr(instruction, "target_qudits", [])
            if isinstance(targets, int):
                targets = [targets]

            dims = [self.dimensions[q] for q in targets]
            levels = {q: _levels_for_gate(instruction, q, self.dimensions[q]) for q in targets}

            node = QuditOpNode(idx, instruction, targets, dims, levels)

            for q in targets:
                prev = last_node_per_qudit[q]
                if prev is not None:
                    node.add_dependency(prev)
                last_node_per_qudit[q] = node

            self.nodes.append(node)

    def op_nodes(self) -> list[QuditOpNode]:
        """Return all remaining gate nodes."""
        return self.nodes

    def front_layer(self) -> list[QuditOpNode]:
        """Return nodes whose dependencies have all been removed."""
        remaining = set(self.nodes)
        return [n for n in self.nodes if not (n.dependencies & remaining)]

    def remove_op_node(self, node: QuditOpNode) -> None:
        """Remove *node* and clean up dependency references in successors."""
        if node not in self.nodes:
            return
        self.nodes.remove(node)
        for n in self.nodes:
            n.dependencies.discard(node)

    def apply_operation_back(
        self,
        gate: Gate,
        qargs: list[int],
    ) -> QuditOpNode:
        """Append *gate* at the back of the DAG and return the new node.

        Levels and dimensions are derived from the gate object and the stored
        *dimensions* of this DAG, so no information is lost.

        Args:
            gate: MQT Qudit gate to append.
            qargs: Qudit indices the gate acts on.

        Returns:
            The newly created :class:`QuditOpNode`.
        """
        new_idx = max((n.index for n in self.nodes), default=-1) + 1
        dims = [self.dimensions[q] for q in qargs]
        levels = {q: _levels_for_gate(gate, q, self.dimensions[q]) for q in qargs}
        node = QuditOpNode(new_idx, gate, qargs, dims, levels)

        for q in qargs:
            q_nodes = [n for n in self.nodes if q in n.target_qudits]
            if q_nodes:
                node.add_dependency(max(q_nodes, key=lambda n: n.index))

        self.nodes.append(node)
        return node

    def copy_empty_like(self) -> QuditDAG:
        """Return an empty DAG with the same qudit dimensions."""
        return QuditDAG(dimensions=self.dimensions)

    def layers(self) -> Generator[dict, None, None]:
        """Yield topological layers of nodes.

        Each yielded value is a dict ``{"graph": [QuditOpNode, ...], "index": int}``
        (mirrors the structure returned by Qiskit's ``DAGCircuit.layers()`` so that
        code using both can share the same iteration pattern).

        Nodes within a layer are mutually independent and can be applied in any
        order (or in parallel).

        Yields:
            dict: Layer dict with keys ``"graph"`` (list of nodes) and ``"index"`` (int).

        Raises:
            RuntimeError: If a dependency cycle is detected (should never occur for
                DAGs built by :func:`circuit_to_dag`).
        """
        remaining = list(self.nodes)
        removed: set[QuditOpNode] = set()
        layer_idx = 0

        while remaining:
            current: list[QuditOpNode] = [n for n in remaining if not (n.dependencies - removed)]
            if not current:
                msg = "QuditDAG contains a cycle — this should never happen."
                raise RuntimeError(msg)
            yield {"graph": current, "index": layer_idx}
            removed.update(current)
            for n in current:
                remaining.remove(n)
            layer_idx += 1

    def multigraph_layers(self) -> Generator[dict, None, None]:
        """Alias for :meth:`layers` (Qiskit API compatibility).

        Yields:
            dict: Layer dict with keys ``"graph"`` and ``"index"`` — see :meth:`layers`.
        """
        yield from self.layers()

    def to_dict(self) -> dict:
        """Serialize this DAG to a plain dict.

        Returns:
            A dict with keys ``dimensions`` and ``nodes``.
        """
        return {
            "dimensions": self.dimensions,
            "nodes": {node.index: node.to_dict() for node in self.nodes},
        }

    def get_edges(self) -> list[tuple[QuditDAGNode, QuditOpNode]]:
        """Return all (predecessor, successor) dependency pairs."""
        return [(dep, node) for node in self.nodes for dep in node.dependencies]

    def display(self) -> None:
        """Print DAG structure (no-op placeholder)."""


def circuit_to_dag(circuit: QuantumCircuit) -> QuditDAG:
    """Convert a ``mqt.qudits.QuantumCircuit`` to a :class:`QuditDAG`.

    Args:
        circuit: Source qudit circuit.

    Returns:
        The corresponding :class:`QuditDAG`.
    """
    return QuditDAG(circuit)


def dag_to_circuit(dag: QuditDAG) -> QuantumCircuit:
    """Reconstruct a ``mqt.qudits.QuantumCircuit`` from a :class:`QuditDAG`.

    The gates are emitted in topological order (layer by layer).  The
    original gate objects are reused, so no matrix information is lost.

    Args:
        dag: Source DAG.

    Returns:
        A new ``mqt.qudits.QuantumCircuit`` with the same gates in order.
    """
    qc = QuantumCircuit(dag.num_qudits, dag.dimensions)
    for layer in dag.layers():
        for node in layer["graph"]:
            qc.instructions.append(node.gate)
            qc.number_gates += 1
    return qc


def check_longest_gate(dag: QuditDAG) -> int:
    """Return the maximum qudit-index span of gates in the front layer.

    A span of 1 means single-qudit, 2 means nearest-neighbor two-qudit,
    anything larger indicates a long-range gate.  Mirrors the function of
    the same name in :mod:`dag_utils` for Qiskit circuits.

    Args:
        dag: The :class:`QuditDAG` to inspect.

    Returns:
        Largest qudit span found in the front layer (minimum 1).
    """
    largest = 1
    for node in dag.front_layer():
        if len(node.target_qudits) > 1:
            span = max(node.target_qudits) - min(node.target_qudits) + 1
            largest = max(largest, span)
    return largest


def get_temporal_zone(dag: QuditDAG, qudits: list[int]) -> QuditDAG:
    """Extract and remove the gates that act only within *qudits*.

    This mirrors :func:`dag_utils.get_temporal_zone` for Qiskit: it walks
    layers in order and collects gates whose qudit footprint is fully contained
    within the *active cone* (initially *qudits*).  When a gate partially
    overlaps the cone, the overlapping qudits are removed from the cone.
    Collected nodes are removed from *dag* in-place.

    Args:
        dag: Source :class:`QuditDAG` (modified in-place).
        qudits: Qudit indices that define the starting cone.

    Returns:
        A new :class:`QuditDAG` containing the extracted gates.
    """
    zone = dag.copy_empty_like()
    active: set[int] = set(range(min(qudits), max(qudits) + 1))

    for layer in list(dag.layers()):
        if not active:
            break
        for node in layer["graph"]:
            node_qudits = set(node.target_qudits)
            if node_qudits <= active:
                zone.nodes.append(node)
                dag.remove_op_node(node)
            elif node_qudits & active:
                active -= node_qudits & active

    return zone
