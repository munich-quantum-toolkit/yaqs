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

# Weak Circuit Simulation

**Weak** digital simulation samples computational-basis **shots** after a noisy circuit evolution, mimicking hardware readout statistics. Use {class}`~mqt.yaqs.WeakSimParams` and read bitstring counts from {attr}`~mqt.yaqs.Result.counts`.

For expectation-value simulation and mid-circuit observables, see {doc}`circuit_simulation`. For parameter presets and truncation settings, see {doc}`simulation_parameters`.

You can pass an OpenQASM file path or raw OpenQASM string to {meth}`~mqt.yaqs.Simulator.run` instead of building a {class}`qiskit.circuit.QuantumCircuit` in Python (OpenQASM 3 requires `pip install mqt-yaqs[qasm3]`).

## 1. Circuit

```{code-cell} ipython3
import numpy as np
from qiskit.circuit.library.n_local import TwoLocal

num_qubits = 5
circuit = TwoLocal(num_qubits, ["rx"], ["rzz"], entanglement="linear", reps=num_qubits).decompose()
rng = np.random.default_rng()
circuit.assign_parameters(rng.uniform(-np.pi, np.pi, size=len(circuit.parameters)), inplace=True)
circuit.measure_all()
circuit.draw(output="mpl")
```

## 2. Initial state and noise model

```{code-cell} ipython3
from mqt.yaqs import NoiseModel, State

state = State(num_qubits, initial="zeros")

gamma = 0.1
noise_model = NoiseModel([
    {"name": name, "sites": [i], "strength": gamma}
    for i in range(num_qubits)
    for name in ["lowering", "pauli_z"]
])
```

For bell-curve noise strengths, see {doc}`realistic_noise_models`.

## 3. Simulation parameters and run

`WeakSimParams` requires an explicit `shots` count (not covered by accuracy presets).

```{code-cell} ipython3
---
tags: [remove-output]
---
from mqt.yaqs import Simulator, WeakSimParams

sim_params = WeakSimParams(shots=1024, max_bond_dim=4, svd_threshold=1e-6)

sim = Simulator(show_progress=False)
result = sim.run(state, circuit, sim_params, noise_model)
```

## 4. Readout histogram

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(result.counts.keys(), result.counts.values(), color="tab:blue", alpha=0.8)
ax.set_xlabel("Bitstring")
ax.set_ylabel("Counts")
ax.set_title("Weak-simulation measurement outcomes")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
```

## Related topics

- {doc}`circuit_simulation` — strong simulation with final and mid-circuit observables
- {doc}`custom_gates` — gate translation from Qiskit
