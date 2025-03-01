from __future__ import annotations
from qiskit.converters import circuit_to_dag
import time

from yaqs.core.data_structures.networks import MPO
from yaqs.circuits.utils.mpo_utils import iterate

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from qiskit.circuit.quantumcircuit import QuantumCircuit


def run(circuit1: QuantumCircuit, circuit2: QuantumCircuit, threshold: float=1e-13, fidelity: float=1-1e-13) -> dict:
    """
    Checks the equivalence of two quantum circuits using an MPO-based algorithm.

    Args:
        circuit1: The first quantum circuit.
        circuit2: The second quantum circuit.
        threshold: Threshold used for SVD truncation.
        fidelity: Fidelity threshold for determining equivalence.

    Returns:
        A dictionary containing:
            'equivalent': Boolean indicating if circuits are equivalent.
            'elapsed_time': Total runtime in seconds.
    """
    assert circuit1.num_qubits == circuit2.num_qubits, \
        "Circuits must have the same number of qubits."

    start_time = time.time()
    mpo = MPO()
    mpo.init_identity(circuit1.num_qubits)

    circuit1_dag = circuit_to_dag(circuit1)
    circuit2_dag = circuit_to_dag(circuit2)

    iterate(mpo, circuit1_dag, circuit2_dag, threshold)

    return {'equivalent': mpo.check_if_identity(fidelity), 'elapsed_time': time.time() - start_time}
