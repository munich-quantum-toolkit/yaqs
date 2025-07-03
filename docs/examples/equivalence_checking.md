---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
  execution_timeout: 300
---

```{code-cell} ipython3
:tags: [remove-cell]
%config InlineBackend.figure_formats = ['svg']
```

# Quantum Circuit Equivalence Checking

This script demonstrates how to create a TwoLocal quantum circuit, assign random parameters,
measure the circuit, and then transpile it into a different basis. The original and transpiled
circuits are then compared for equivalence using the YAQS equivalence checker.

Define the number of qubits and circuit depth.

```{code-cell} ipython3
num_qubits = 5
depth = num_qubits
```

Create a TwoLocal circuit and decompose it.

```{code-cell} ipython3
from qiskit.circuit.library.n_local import TwoLocal

import numpy as np

circuit = TwoLocal(num_qubits, ["rx"], ["rzz"], entanglement="linear", reps=depth).decompose()
num_pars = len(circuit.parameters)
# Assign random parameters uniformly in [-pi, pi].
rng = np.random.default_rng()
values = rng.uniform(-np.pi, np.pi, size=num_pars)
circuit.assign_parameters(values, inplace=True)
circuit.measure_all()
circuit.draw(output="mpl")
```

Transpile the circuit to a new basis.

```{code-cell} ipython3
from qiskit import transpile

basis_gates = ["cz", "rz", "sx", "x", "id"]
transpiled_circuit = transpile(circuit, basis_gates=basis_gates, optimization_level=1)
transpiled_circuit.draw(output="mpl")
```

Define parameters for equivalence checking.

```{code-cell} ipython3
from mqt.yaqs.digital.equivalence_checker import run

threshold = 1e-6
fidelity = 1 - 1e-13
result = run(circuit, transpiled_circuit, threshold, fidelity)
print(f"Equivalence: {result}")
```
