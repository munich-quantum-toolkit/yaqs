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

# Weak Quantum Circuit Simulation (Shots)

This module demonstrates how to run a weak simulation using the YAQS simulator
with a TwoLocal circuit generated via Qiskit's circuit library. An MPS is initialized
in the $\ket{0}$ state, a noise model is applied, and weak simulation parameters are set.
After running the simulation, the measurement results (bitstring counts) are displayed
as a bar chart.

Create the circuit

```{code-cell} ipython3
from qiskit.circuit.library.n_local import TwoLocal

import numpy as np

num_qubits = 10
circuit = TwoLocal(num_qubits, ["rx"], ["rzz"], entanglement="linear", reps=num_qubits).decompose()
num_pars = len(circuit.parameters)
rng = np.random.default_rng()
values = rng.uniform(-np.pi, np.pi, size=num_pars)
circuit.assign_parameters(values, inplace=True)
circuit.measure_all()
circuit.draw(output="mpl")
```

Define the initial state

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.networks import MPS

state = MPS(num_qubits, state="zeros")
```

Define the noise model

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.noise_model import NoiseModel

gamma = 0.1
noise_model = NoiseModel(["relaxation", "dephasing"], [gamma, gamma])
```

Define the simulation parameters

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.simulation_parameters import WeakSimParams

shots = 1024
max_bond_dim = 4
threshold = 1e-6
window_size = 0
sim_params = WeakSimParams(shots, max_bond_dim, threshold, window_size)
```

Run the simulation

```{code-cell} ipython3
---
tags: [remove-output]
---
from mqt.yaqs import simulator

simulator.run(state, circuit, sim_params, noise_model)
```

Plot the measurement outcomes as a bar chart

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
import matplotlib.pyplot as plt

plt.bar(sim_params.results.keys(), sim_params.results.values())
plt.xlabel("Bitstring")
plt.ylabel("Counts")
plt.title("Measurement Results")
plt.show()
```
