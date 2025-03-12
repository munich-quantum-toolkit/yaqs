import pytest

import numpy as np
import scipy as sp

from qiskit.circuit import QuantumCircuit
from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
from qiskit_nature.second_q.hamiltonians.lattices import SquareLattice, BoundaryCondition
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit.quantum_info import Operator

from openfermion.hamiltonians import fermi_hubbard
from openfermion.linalg import get_sparse_operator

from yaqs.core.libraries.circuit_library import create_Ising_circuit, create_Heisenberg_circuit, create_2D_Fermi_Hubbard_circuit
from yaqs.circuits.reference_implementation.FH_reference import create_Fermi_Hubbard_model_qutip


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
    op_names = [instr.name for instr, _, _ in circ.data]
    rx_count = op_names.count("rx")
    cx_count = op_names.count("cx")
    rz_count = op_names.count("rz")
    
    # In each timestep:
    #   - There should be L rx gates.
    #   - For the even-site loop: for L=4, we have 4//2 = 2 iterations,
    #     each contributing two CX and one RZ (total 4 CX and 2 RZ).
    #   - For the odd-site loop: range(1,4//2) gives 1 iteration, adding 2 CX and 1 RZ.
    # Total per timestep: 4 rx, 6 CX, and 3 RZ.
    assert rx_count == 4 * timesteps
    assert cx_count == 6 * timesteps
    assert rz_count == 3 * timesteps

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
    op_names = [instr.name for instr, _, _ in circ.data]
    rx_count = op_names.count("rx")
    cx_count = op_names.count("cx")
    rz_count = op_names.count("rz")
    
    assert rx_count == 5
    assert cx_count == 8
    assert rz_count == 4

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
    op_names = [instr.name for instr, _, _ in circ.data]
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
    
    op_names = [instr.name for instr, _, _ in circ.data]
    for gate in ["rz", "rzz", "rxx", "ryy"]:
        assert gate in op_names

def test_create_Heisenberg_circuit_invalid_model():
    # If the model name is not "Heisenberg", an assertion error should occur.
    model = {'name': 'Ising', 'L': 4, 'Jx': 1.0, 'Jy': 1.0, 'Jz': 1.0, 'h': 0.5}
    dt = 0.1
    timesteps = 1
    with pytest.raises(AssertionError):
        create_Heisenberg_circuit(model, dt, timesteps)

def test_create_2D_Fermi_Hubbard_circuit_equal_qiskit():
    # Define the FH model parameters
    t = 1.0         # kinetic hopping
    mu = 0.5        # chemical potential
    u = 4.0         # onsite interaction
    Lx, Ly = 2, 2   # lattice dimensions
    timesteps = 1
    dt = 0.1
    num_trotter_steps = 10

    # yaqs implementation
    model = {'name': '2D_Fermi_Hubbard', 'Lx': Lx, 'Ly': Ly, 'mu': mu, 'u': u, 't': t, 'num_trotter_steps': num_trotter_steps}
    circuit = create_2D_Fermi_Hubbard_circuit(model, dt=dt, timesteps=timesteps)
    U_yaqs = Operator(circuit).to_matrix()

    # Qiskit implementation
    square_lattice = SquareLattice(rows=Lx, cols=Ly, boundary_condition=BoundaryCondition.OPEN)
    fh_hamiltonian = FermiHubbardModel(
        square_lattice.uniform_parameters(
            uniform_interaction=t,
            uniform_onsite_potential=mu,
        ),
        onsite_interaction=u,
    )
    mapper = JordanWignerMapper()
    qubit_jw_op = mapper.map(fh_hamiltonian.second_q_op())
    H_qiskit = qubit_jw_op.to_matrix()
    U_qiskit = sp.linalg.expm(-1j*dt*timesteps*H_qiskit)

    # Calculate error
    error = np.linalg.norm(U_qiskit - U_yaqs, 2)
    print("|U_qiskit - U_yaqs| = " + str(error))
    assert error <= 10e-3

def test_create_2D_Fermi_Hubbard_circuit_equal_qutip():
    # Define the FH model parameters
    t = 1.0         # kinetic hopping
    mu = 0.5        # chemical potential
    u = 4.0         # onsite interaction
    Lx, Ly = 2, 2   # lattice dimensions
    timesteps = 1
    dt = 0.1
    num_trotter_steps = 10

    # yaqs implementation
    model = {'name': '2D_Fermi_Hubbard', 'Lx': Lx, 'Ly': Ly, 'mu': -mu, 'u': u, 't': -t, 'num_trotter_steps': num_trotter_steps}
    circuit = create_2D_Fermi_Hubbard_circuit(model, dt=dt, timesteps=timesteps)
    U_yaqs = Operator(circuit).to_matrix()

    # QuTiP implementation
    H_qutip = create_Fermi_Hubbard_model_qutip(Lx, Ly, u, -t, mu)
    U_qutip = sp.linalg.expm(-1j*dt*timesteps*H_qutip.full())

    # Calculate error
    error = np.linalg.norm(U_qutip - U_yaqs, 2)
    print("|U_qutip - U_yaqs| = " + str(error))
    assert error <= 10e-3

def test_create_2D_Fermi_Hubbard_circuit_equal_openfermion():
    # Define the FH model parameters
    t = 1.0         # kinetic hopping
    mu = 0.5        # chemical potential
    u = 4.0         # onsite interaction
    Lx, Ly = 2, 2   # lattice dimensions
    timesteps = 1
    dt = 0.1
    num_trotter_steps = 10

    # yaqs implementation
    model = {'name': '2D_Fermi_Hubbard', 'Lx': Lx, 'Ly': Ly, 'mu': -mu, 'u': u, 't': -t, 'num_trotter_steps': num_trotter_steps}
    circuit = create_2D_Fermi_Hubbard_circuit(model, dt=dt, timesteps=timesteps)
    U_yaqs = Operator(circuit).to_matrix()

    # Openfermion implementation
    H_openfermion = get_sparse_operator(fermi_hubbard(Lx, Ly, tunneling=t, coulomb=u, chemical_potential=mu, periodic=False)).todense()
    U_openfermion = sp.linalg.expm(-1j*dt*timesteps*H_openfermion)

    # Calculate error
    error = np.linalg.norm(U_openfermion - U_yaqs, 2)
    print("|U_openfermion - U_yaqs| = " + str(error))
    assert error <= 10e-3