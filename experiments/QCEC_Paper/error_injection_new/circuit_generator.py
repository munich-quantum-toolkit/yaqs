import copy
import qiskit
import qiskit.qpy
from qiskit.circuit.library.n_local import TwoLocal
import numpy as np


num_qubits = 9
depth = num_qubits
starting_gates = ['h', 'x', 'cx', 'cz', 'swap', 'id', 'rz', 'rx', 'ry', 'rxx', 'ryy', 'rzz']

# Original
# basis_gates = ['h', 'x', 'cx', 'rz', 'id']
# IBM Heron
basis_gates = ['cz', 'rz', 'sx', 'x', 'id']
# Quantinuum H1-1, H1-2
# basis_gates = ['rx', 'ry', 'rz', 'rzz']

circuits = []
transpiled_circuits = []
samples = 10
for sample in range(samples):
    circuit = qiskit.circuit.QuantumCircuit(num_qubits)
    twolocal = TwoLocal(num_qubits, ['rz'], ['rxx'], entanglement='linear', reps=depth).decompose()        
    num_pars = len(twolocal.parameters)
    values = np.random.rand(num_pars)
    circuit = copy.deepcopy(twolocal).assign_parameters(values)
    transpiled_circuit = qiskit.compiler.transpile(circuit, basis_gates=basis_gates, optimization_level=1)

    circuits.append(circuit)
    transpiled_circuits.append(transpiled_circuit)

with open('error_circuits.qpy', 'wb') as file:
    qiskit.qpy.dump(circuits, file)

with open('error_circuits_transpiled.qpy', 'wb') as file:
    qiskit.qpy.dump(transpiled_circuits, file)

