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

# Noisy Circuit Simulation

**Strong** digital simulation evolves a matrix-product state (MPS) through a Qiskit circuit while applying tensor-jump noise between gates. You choose which Pauli (or other) observables to measure and how many Monte Carlo trajectories to average.

| Workflow | Typical use | Key settings |
| -------- | ----------- | ------------ |
| **Final observables** | Noise scaling, benchmarking, device studies | {class}`~mqt.yaqs.core.data_structures.simulation_parameters.StrongSimParams` with observables evaluated after the last gate |
| **Mid-circuit observables** | Layer-wise diagnostics, depth-dependent calibration | `StrongSimParams(sample_layers=True)` plus `barrier(label="SAMPLE_OBSERVABLES")` markers in the circuit |
| **Shot-based readout** | Hardware-like bitstring statistics | {class}`~mqt.yaqs.core.data_structures.simulation_parameters.WeakSimParams` — see {doc}`weak_circuit_simulation` |

Circuits enter YAQS as {class}`qiskit.circuit.QuantumCircuit` objects (or OpenQASM strings). The initial state must use `representation="mps"` (the default for {class}`~mqt.yaqs.core.data_structures.state.State` presets). For accuracy presets, truncation knobs, and `random_seed`, see {doc}`simulation_parameters`. For bell-curve (Gaussian) noise strengths, see {doc}`realistic_noise_models`.

```{code-cell} ipython3
from mqt.yaqs import Simulator

sim = Simulator(show_progress=False)
```

## 1. Final observables and noise scaling

We build an Ising-style circuit, prepare $\ket{0}^{\otimes n}$, and sweep a global relaxation rate $\gamma$. Expectation values $\langle Z_i \rangle$ at the **end** of the circuit are collected into a heatmap.

### 1.1 Circuit and initial state

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit
from mqt.yaqs.core.libraries.gate_library import Z

num_qubits = 5
circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=10)
circuit.measure_all()
circuit.draw(output="mpl")

state = State(num_qubits, initial="zeros")
```

### 1.2 Simulation parameters

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams

sim_params = StrongSimParams(
    observables=[Observable(Z(), site) for site in range(num_qubits)],
    num_traj=100,
    max_bond_dim=4,
    svd_threshold=1e-6,
)
```

### 1.3 Noise sweep and heatmap

```{code-cell} ipython3
---
tags: [remove-output]
---
import numpy as np

from mqt.yaqs.core.data_structures.noise_model import NoiseModel

gammas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
heatmap = np.empty((num_qubits, len(gammas)))
for j, gamma in enumerate(gammas):
    noise_model = NoiseModel([
        {"name": "lowering", "sites": [i], "strength": gamma} for i in range(num_qubits)
    ])
    result = sim.run(state, circuit, sim_params, noise_model)
    for i in range(len(result.observables)):
        heatmap[i, j] = result.expectation_values[i][0]
```

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 3.5))
im = ax.imshow(heatmap, aspect="auto", vmin=0.5, vmax=1)
ax.set_ylabel("Qubit")
ax.set_xlabel(r"Relaxation rate $\gamma$")
ax.set_yticks(range(num_qubits))
ax.set_xticks(range(len(gammas)))
ax.set_xticklabels([f"$10^{{{int(np.log10(g))}}}$" for g in gammas])

fig.subplots_adjust(top=0.95, right=0.88)
cbar_ax = fig.add_axes(rect=(0.9, 0.11, 0.025, 0.8))
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.set_title(r"$\langle Z \rangle$")
plt.tight_layout()
plt.show()
```

## 2. Mid-circuit observables

Set `sample_layers=True` on {class}`~mqt.yaqs.core.data_structures.simulation_parameters.StrongSimParams` and insert barriers labelled `SAMPLE_OBSERVABLES` (case-insensitive) where you want measurements. YAQS records observables at the circuit start, after each labelled barrier, and after the final gate layer.

Other barriers and `measure` operations are ignored for this sampling schedule.

### 2.1 Circuit with sampling barriers

```{code-cell} ipython3
from qiskit.circuit import QuantumCircuit

layer_qubits = 3
qc = QuantumCircuit(layer_qubits)

for segment in range(5):
    qc.rzz(0.5, 0, 1)
    qc.rzz(0.5, 1, 2)
    if segment < 4:
        qc.barrier(label="SAMPLE_OBSERVABLES")

qc.draw(output="mpl")
```

Five entangler layers with four labelled barriers yield **six** sampling points: initial state, four mid-circuit checkpoints, and the final layer.

### 2.2 Noise model and parameters

```{code-cell} ipython3
import numpy as np

from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams

noise_factor = 0.01
layer_noise = NoiseModel(
    [{"name": "pauli_x", "sites": [i], "strength": noise_factor} for i in range(layer_qubits)]
    + [{"name": "crosstalk_xx", "sites": [i, i + 1], "strength": noise_factor} for i in range(layer_qubits - 1)]
)

layer_state = State(layer_qubits, initial="zeros", pad=2)
layer_observables = [Observable(Z(), i) for i in range(layer_qubits)]
layer_params = StrongSimParams(layer_observables, num_traj=1000, sample_layers=True)
```

Higher `num_traj` reduces Monte Carlo variance; `100` trajectories are often enough for prototyping.

### 2.3 Run and validate

```{code-cell} ipython3
---
tags: [remove-output]
---
layer_result = sim.run(layer_state, qc, layer_params, layer_noise)

# Reference expectations (rows: qubits; columns: initial + 4 barriers + final)
reference = np.array([
    [1.0, 0.9607894391523233, 0.9231163463866354, 0.8869204367171571, 0.8521437889662108, 0.8187307530779814],
    [1.0, 0.9231163463866359, 0.8521437889662113, 0.7866278610665535, 0.726149037073691, 0.6703200460356394],
    [1.0, 0.9607894391523233, 0.9231163463866354, 0.8869204367171571, 0.8521437889662108, 0.8187307530779814],
])

yaqs = np.vstack([np.real(v) for v in layer_result.expectation_values])
max_diff = float(np.abs(yaqs - reference).max())
print(f"max|YAQS − reference| = {max_diff:.4f}  (decreases with num_traj)")
```

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
import matplotlib.pyplot as plt

layers = range(yaqs.shape[1])
labels = [f"q{i}" for i in range(layer_qubits)]

fig, ax = plt.subplots(figsize=(7, 4))
for q in range(layer_qubits):
    ax.plot(layers, reference[q], "--", color=f"C{q}", alpha=0.6, label=f"reference {labels[q]}")
    ax.plot(layers, yaqs[q], "o-", color=f"C{q}", label=f"YAQS {labels[q]}")

ax.set_xlabel("Sampling point (initial → barriers → final)")
ax.set_ylabel(r"$\langle Z \rangle$")
ax.set_xticks(list(layers))
ax.legend(ncols=2, fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

## 3. Related topics

- {doc}`weak_circuit_simulation` — shot-based readout with {class}`~mqt.yaqs.core.data_structures.simulation_parameters.WeakSimParams`
- {doc}`custom_gates` — Qiskit gate translation and custom unitaries
- {doc}`realistic_noise_models` — Gaussian and other distributed noise strengths
- {doc}`equivalence_checking` — verify that two circuits implement the same unitary
