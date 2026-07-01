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

| Workflow                    | Typical use                                         | Key settings                                                                                                                 |
| --------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **Final observables**       | Noise scaling, benchmarking, device studies         | {class}`~mqt.yaqs.core.data_structures.simulation_parameters.StrongSimParams` with observables evaluated after the last gate |
| **Mid-circuit observables** | Layer-wise diagnostics, depth-dependent calibration | `StrongSimParams(sample_layers=True)` plus `barrier(label="SAMPLE_OBSERVABLES")` markers in the circuit                      |
| **Shot-based readout**      | Hardware-like bitstring statistics                  | {class}`~mqt.yaqs.core.data_structures.simulation_parameters.WeakSimParams` — see {doc}`weak_circuit_simulation`             |

Circuits enter YAQS as {class}`qiskit.circuit.QuantumCircuit` objects (or OpenQASM strings). The initial state must use `representation="mps"` (the default for {class}`~mqt.yaqs.core.data_structures.state.State` presets). For accuracy presets, truncation knobs, and `random_seed`, see {doc}`simulation_parameters`. For log-normal disorder on noise strengths, see {doc}`realistic_noise_models`.

```{code-cell} ipython3
from mqt.yaqs import Simulator

sim = Simulator(show_progress=False)
```

## 1. Final observables and noise scaling

We build an Ising-style circuit, prepare $\ket{0}^{\otimes n}$, and sweep a global relaxation rate $\gamma$. The plot below shows how each qubit's final $\langle Z_i \rangle$ moves toward $+1$ as relaxation dominates.

### 1.1 Circuit and initial state

```{code-cell} ipython3
from mqt.yaqs import State
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit

num_qubits = 5
circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=10)
circuit.measure_all()

state = State(num_qubits, initial="zeros")
```

### 1.2 Simulation parameters

```{code-cell} ipython3
from mqt.yaqs import Observable, StrongSimParams

sim_params = StrongSimParams(
    observables=[Observable("z", site) for site in range(num_qubits)],
    num_traj=100,
    max_bond_dim=4,
    svd_threshold=1e-6,
)
```

### 1.3 Noise sweep and plot

```{code-cell} ipython3
import numpy as np

from mqt.yaqs import NoiseModel

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

fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
colors = plt.cm.viridis(np.linspace(0.15, 0.85, num_qubits))
for i in range(num_qubits):
    ax.semilogx(gammas, heatmap[i], "o-", color=colors[i], linewidth=1.8, markersize=5, label=rf"$q_{i}$")
ax.set_xlabel(r"Relaxation rate $\gamma$")
ax.set_ylabel(r"$\langle Z_i \rangle$")
ax.set_ylim(-1.05, 1.05)
ax.legend(ncol=num_qubits, fontsize=8, loc="lower left", frameon=False)
ax.set_title("Final Z expectation vs amplitude-damping strength")
ax.grid(alpha=0.3, which="both")
plt.show()
```

## 2. Mid-circuit observables

(mid-circuit-observables)=

```{note}
This section uses `num_traj=64` during the documentation build. Increase `num_traj` locally for lower-variance layer curves.
```

Set `sample_layers=True` on {class}`~mqt.yaqs.core.data_structures.simulation_parameters.StrongSimParams` and insert barriers labelled `SAMPLE_OBSERVABLES` (case-insensitive) where you want measurements. YAQS records observables at the circuit start, after each labelled barrier, and after the final gate layer.

The example below starts from $\ket{+}^{\otimes n}$, applies a chain of $R_{ZZ}$ entanglers, and tracks how amplitude damping gradually drives each $\langle Z_i \rangle$ toward $+1$.

Other barriers and `measure` operations are ignored for this sampling schedule.

### 2.1 Circuit with sampling barriers

```{code-cell} ipython3
from qiskit.circuit import QuantumCircuit

layer_qubits = 5
qc = QuantumCircuit(layer_qubits)

for segment in range(6):
    for i in range(layer_qubits - 1):
        qc.rzz(0.7, i, i + 1)
    if segment < 5:
        qc.barrier(label="SAMPLE_OBSERVABLES")
```

Six entangler layers with five labelled barriers yield **seven** sampling points: initial state, five mid-circuit checkpoints, and the final layer.

### 2.2 Noise model and parameters

```{code-cell} ipython3
import numpy as np

from mqt.yaqs import Observable, StrongSimParams

noise_factor = 0.1
layer_noise = NoiseModel([
    {"name": "lowering", "sites": [i], "strength": noise_factor} for i in range(layer_qubits)
])

layer_state = State(layer_qubits, initial="x+", pad=16)
layer_observables = [Observable("z", i) for i in range(layer_qubits)]
layer_params = StrongSimParams(layer_observables, num_traj=64, sample_layers=True, max_bond_dim=12)
```

Higher `num_traj` reduces Monte Carlo variance; `100` trajectories are often enough for prototyping.

### 2.3 Run and plot

```{code-cell} ipython3
layer_result = sim.run(layer_state, qc, layer_params, layer_noise)

yaqs = np.vstack([np.real(v) for v in layer_result.expectation_values])
```

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
import matplotlib.pyplot as plt

depth = np.arange(yaqs.shape[1])
qubit_labels = [rf"$q_{i}$" for i in range(layer_qubits)]

fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
im = ax.imshow(
    yaqs,
    aspect="auto",
    origin="lower",
    vmin=-1,
    vmax=1,
    extent=(-0.5, yaqs.shape[1] - 0.5, -0.5, layer_qubits - 0.5),
)
ax.set_xlabel("Sampling index")
ax.set_ylabel("Qubit")
ax.set_xticks(depth)
ax.set_yticks(range(layer_qubits), qubit_labels)
ax.set_title(r"Mid-circuit $\langle Z \rangle$")
fig.colorbar(im, ax=ax, shrink=0.9, label=r"$\langle Z \rangle$")
plt.show()
```

## 3. OpenQASM inputs

Pass an OpenQASM 2 source string (or file path) directly to {meth}`~mqt.yaqs.Simulator.run` instead of building a {class}`qiskit.circuit.QuantumCircuit` in Python. Custom gate bodies declared in the program are translated like any other Qiskit operation.

```{code-cell} ipython3
from mqt.yaqs import State, WeakSimParams

qasm = """
OPENQASM 2.0;
include "qelib1.inc";

gate entangle a,b {
  h a;
  cx a,b;
}

qreg q[2];
entangle q[0], q[1];
"""

qasm_state = State(2, initial="zeros")
qasm_result = sim.run(
    qasm_state,
    qasm,
    WeakSimParams(shots=128, max_bond_dim=4),
)
```

OpenQASM 3 requires `pip install mqt-yaqs[qasm3]`. {class}`~mqt.yaqs.EquivalenceChecker` accepts the same path and string forms; see {doc}`equivalence_checking`.

## 4. Gate application modes

`StrongSimParams.gate_mode` (and `WeakSimParams.gate_mode`) selects how two-qubit gates are applied to the MPS. The default `"mpo"` uses extended gate MPOs for long-range pairs; `"tdvp"` uses a local TDVP window when an analytic generator is available. See {doc}`simulation_parameters` and {doc}`custom_gates` for the full matrix.

Below, a long-range `cx` on qubits 0 and 2 is simulated noiselessly with both modes:

```{code-cell} ipython3
from qiskit.circuit import QuantumCircuit

lr_qc = QuantumCircuit(3)
lr_qc.h(0)
lr_qc.cx(0, 2)

lr_state = State(3, initial="zeros")
z0_by_mode = {}
for mode in ("mpo", "tdvp"):
    mode_params = StrongSimParams(
        observables=[Observable("z", 0)],
        num_traj=1,
        gate_mode=mode,
        max_bond_dim=8,
    )
    mode_result = sim.run(lr_state, lr_qc, mode_params)
    z0_by_mode[mode] = float(mode_result.expectation_values[0][0])
```

## 5. Related topics

- {doc}`weak_circuit_simulation` — shot-based readout with {class}`~mqt.yaqs.core.data_structures.simulation_parameters.WeakSimParams`
- {doc}`custom_gates` — custom unitaries and gate translation
- {doc}`realistic_noise_models` — log-normal and other distributed noise strengths
- {doc}`equivalence_checking` — verify that two circuits implement the same unitary
- {doc}`quickstart` — minimal analog, digital, and equivalence-check workflows
