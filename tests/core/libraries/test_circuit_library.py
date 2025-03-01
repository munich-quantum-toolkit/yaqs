import pytest
from qiskit.circuit import QuantumCircuit

from mqt.yaqs.core.libraries.circuit_library import create_Ising_circuit, create_Heisenberg_circuit


def test_create_Ising_circuit_valid_even():
    # Use an even number of qubits.
    model = {'name': 'Ising', 'L': 4, 'g': 0.5, 'J': 1.0}
    dt = 0.1
    timesteps = 1
    circ = create_Ising_circuit(model, dt, timesteps)

    # Check that the output is a QuantumCircuit with the right number of qubits.
    assert isinstance(circ, QuantumCircuit)
    assert circ.num_qubits == model['L']

    # Count the gates by name.
    op_names = [instr.operation.name for instr in circ.data]
    rx_count = op_names.count("rx")
    rzz_count = op_names.count("rzz")

    assert rx_count == 4 * timesteps
    assert rzz_count == 3 * timesteps

def test_create_Ising_circuit_valid_odd():
    # Use an odd number of qubits.
    model = {'name': 'Ising', 'L': 5, 'g': 0.5, 'J': 1.0}
    dt = 0.1
    timesteps = 1
    circ = create_Ising_circuit(model, dt, timesteps)

    assert isinstance(circ, QuantumCircuit)
    assert circ.num_qubits == model['L']

    # For L=5:
    #   - The rx loop adds 5 gates.
    #   - The even-site loop: 5//2 = 2 iterations → 2*2 = 4 CX and 2 RZ.
    #   - The odd-site loop: range(1, 5//2) → 1 iteration → 2 CX and 1 RZ.
    #   - The extra clause (since 5 is odd and not 1) adds 2 CX and 1 RZ.
    # Total: 5 rx, (4+2+2)=8 CX, and (2+1+1)=4 RZ.
    op_names = [instr.operation.name for instr in circ.data]
    rx_count = op_names.count("rx")
    rzz_count = op_names.count("rzz")

    assert rx_count == 5
    assert rzz_count == 4


def test_create_Ising_circuit_invalid_model():
    # If the model name is not "Ising", an assertion error should occur.
    model = {'name': 'Heisenberg', 'L': 4, 'g': 0.5, 'J': 1.0}
    dt = 0.1
    timesteps = 1
    with pytest.raises(AssertionError):
        create_Ising_circuit(model, dt, timesteps)


def test_create_Heisenberg_circuit_valid_even():
    # Use an even number of qubits.
    model = {'name': 'Heisenberg', 'L': 4, 'Jx': 1.0, 'Jy': 1.0, 'Jz': 1.0, 'h': 0.5}
    dt = 0.1
    timesteps = 1
    circ = create_Heisenberg_circuit(model, dt, timesteps)

    assert isinstance(circ, QuantumCircuit)
    assert circ.num_qubits == model['L']

    # Check that the circuit contains the expected types of gates.
    op_names = [instr.operation.name for instr in circ.data]
    for gate in ["rz", "rzz", "rxx", "ryy"]:
        assert gate in op_names

def test_create_Heisenberg_circuit_valid_odd():
    # Use an odd number of qubits.
    model = {'name': 'Heisenberg', 'L': 5, 'Jx': 1.0, 'Jy': 1.0, 'Jz': 1.0, 'h': 0.5}
    dt = 0.1
    timesteps = 1
    circ = create_Heisenberg_circuit(model, dt, timesteps)

    assert isinstance(circ, QuantumCircuit)
    assert circ.num_qubits == model['L']

    op_names = [instr.operation.name for instr in circ.data]
    for gate in ["rz", "rzz", "rxx", "ryy"]:
        assert gate in op_names

def test_create_Heisenberg_circuit_invalid_model():
    # If the model name is not "Heisenberg", an assertion error should occur.
    model = {'name': 'Ising', 'L': 4, 'Jx': 1.0, 'Jy': 1.0, 'Jz': 1.0, 'h': 0.5}
    dt = 0.1
    timesteps = 1
    with pytest.raises(AssertionError):
        create_Heisenberg_circuit(model, dt, timesteps)
