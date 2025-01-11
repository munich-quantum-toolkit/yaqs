import copy
import numpy as np
import qiskit.circuit
import qiskit.compiler
from qiskit.circuit.library.n_local import TwoLocal

from yaqs.circuits.equivalence_checking import equivalence_checker

# Define the initial circuit
num_qubits = 5
depth = num_qubits
circuit = qiskit.circuit.QuantumCircuit(num_qubits)

# Example: Two-Local Circuit
twolocal = TwoLocal(num_qubits, ['rx'], ['rzz'], entanglement='linear', reps=depth).decompose()        
num_pars = len(twolocal.parameters)
values = np.random.uniform(low=-np.pi, high=np.pi, size=num_pars)
circuit = copy.deepcopy(twolocal).assign_parameters(values)
circuit.measure_all()

# Transpilation to new basis gates
basis_gates = ['cz', 'rz', 'sx', 'x', 'id']
transpiled_circuit = qiskit.compiler.transpile(circuit, basis_gates=basis_gates, optimization_level=1)

# Define parameters for equivalence checking
threshold = 1e-6
fidelity = 1-1e-13
result = equivalence_checker.run(circuit, transpiled_circuit, threshold, fidelity)
print(result)
