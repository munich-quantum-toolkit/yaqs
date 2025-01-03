import copy
import numpy as np
import qiskit.circuit
import qiskit.compiler
from qiskit.circuit.library.n_local import TwoLocal

from yaqs.circuits.equivalence_checking.check_equivalence import run

num_qubits = 10
depth = num_qubits
circuit = qiskit.circuit.QuantumCircuit(num_qubits)
twolocal = TwoLocal(num_qubits, ['rx'], ['rzz'], entanglement='linear', reps=depth).decompose()        
num_pars = len(twolocal.parameters)
values = np.random.uniform(low=-np.pi, high=np.pi, size=num_pars)
circuit = copy.deepcopy(twolocal).assign_parameters(values)
circuit.measure_all()

basis_gates = ['cz', 'rz', 'sx', 'x', 'id']
threshold = 1e-6
fidelity = 1-1e-13

transpiled_circuit = qiskit.compiler.transpile(circuit, basis_gates=basis_gates, optimization_level=1)

result = run(copy.deepcopy(circuit), copy.deepcopy(transpiled_circuit), threshold, fidelity)
print(result)
