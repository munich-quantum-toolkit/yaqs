---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
  execution_timeout: 600
---

```{code-cell} ipython3
:tags: [remove-cell]
%config InlineBackend.figure_formats = ['svg']
```

### Strong Circuit Simulation (Observable) with sampling

This example demonstrates how to use labelled barriers to sample observables at intermediate layers of a circuit using the digital Tensor Jump Method (TJM). Any `barrier` with label `"SAMPLE_OBSERVABLES"` (case-insensitive) is treated as a sampling point.

We reproduce a 3-qubit chain with repeated `rzz(0.5)` gates and noise consisting of per-qubit bit-flip (`pauli_x`) and nearest-neighbor `crosstalk_xx`, both with strength 0.01. We compare the simulated Z-expectations against a hardcoded Qiskit reference.

Define the circuit with SAMPLE_OBSERVABLES barriers

```{code-cell} ipython3
from qiskit.circuit import QuantumCircuit

num_qubits = 3
qc = QuantumCircuit(num_qubits)

# Five segments of entanglers with four labelled barriers in-between → 6 sampling points (initial + 4 mids + final)
qc.rzz(0.5, 0, 1)
qc.rzz(0.5, 1, 2)
qc.barrier(label="SAMPLE_OBSERVABLES")

qc.rzz(0.5, 0, 1)
qc.rzz(0.5, 1, 2)
qc.barrier(label="SAMPLE_OBSERVABLES")

qc.rzz(0.5, 0, 1)
qc.rzz(0.5, 1, 2)
qc.barrier(label="SAMPLE_OBSERVABLES")

qc.rzz(0.5, 0, 1)
qc.rzz(0.5, 1, 2)
qc.barrier(label="SAMPLE_OBSERVABLES")

qc.rzz(0.5, 0, 1)
qc.rzz(0.5, 1, 2)

qc.draw(output="mpl")
```

Define the noise model and initial state

```{code-cell} ipython3
import numpy as np

from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.networks import MPS

noise_factor = 0.01
processes = (
    [{"name": "pauli_x", "sites": [i], "strength": noise_factor} for i in range(num_qubits)]
    + [{"name": "crosstalk_xx", "sites": [i, i + 1], "strength": noise_factor} for i in range(num_qubits - 1)]
)
noise_model = NoiseModel(processes)

# Start in |000⟩; pad internal bonds slightly for numerical stability
state = MPS(num_qubits, state="zeros", pad=2)
```

Set up observables and simulation parameters with layer sampling enabled

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.libraries.gate_library import Z

observables = [Observable(Z(), i) for i in range(num_qubits)]

# Enable sampling at intermediate layers via labelled barriers
# Note: num_traj=1000 provides good agreement with reference; adjust if runtime is a concern
#       Typically, 100 trajectories gets you pretty far for prototyping.

num_traj = 1000
sim_params = StrongSimParams(observables, num_traj=num_traj, sample_layers=True)
```

Run the simulation

```{code-cell} ipython3
---
tags: [remove-output]
---
from mqt.yaqs import simulator

simulator.run(state, qc, sim_params, noise_model, parallel=False)
```

Compare against the hardcoded Qiskit reference

```{code-cell} ipython3
# Hardcoded reference (rows: qubit 0,1,2; columns: initial + 4 mids + final)
reference = np.array([
    [1.0, 0.9607894391523233, 0.9231163463866354, 0.8869204367171571, 0.8521437889662108, 0.8187307530779814],
    [1.0, 0.9231163463866359, 0.8521437889662113, 0.7866278610665535, 0.726149037073691, 0.6703200460356394],
    [1.0, 0.9607894391523233, 0.9231163463866354, 0.8869204367171571, 0.8521437889662108, 0.8187307530779814],
])

# YAQS results collected at initial + each SAMPLE_OBSERVABLES barrier + final
yaqs = np.vstack([np.real(obs.results) for obs in sim_params.observables])

diff = np.abs(yaqs - reference)
max_diff = float(diff.max())
print(f"max|YAQS - reference| = {max_diff:.4f}")
```

Visualize the trajectories per qubit

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
import matplotlib.pyplot as plt

layers = range(6)
labels = ["q0", "q1", "q2"]

fig, ax = plt.subplots(1, 1)
for q in range(num_qubits):
    ax.plot(layers, reference[q], "--", label=f"reference {labels[q]}")
    ax.plot(layers, yaqs[q], "o-", label=f"YAQS {labels[q]}")

ax.set_xlabel("Layer (initial + mid barriers + final)")
ax.set_ylabel(r"$\langle Z \rangle$")
ax.set_xticks(list(layers))
ax.legend(ncols=2)
plt.show()
```

Notes

- The sampling is triggered exclusively by `barrier(label="SAMPLE_OBSERVABLES")`. Other barriers and `measure` operations are ignored for sampling.
- Agreement with the reference improves with the number of Monte Carlo trajectories (`num_traj`).
