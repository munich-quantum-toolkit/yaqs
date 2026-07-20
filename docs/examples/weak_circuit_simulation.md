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

# Weak Circuit Simulation

**Weak** digital simulation samples computational-basis **shots** after a noisy circuit evolution, mimicking hardware readout statistics. Use {class}`~mqt.yaqs.WeakSimParams` and read bitstring counts from {attr}`~mqt.yaqs.Result.counts`.

For expectation-value simulation and mid-circuit observables, see {doc}`strong_simulation`. For parameter presets and truncation settings, see {doc}`simulation_parameters`.

You can pass an OpenQASM file path or raw OpenQASM string to {meth}`~mqt.yaqs.Simulator.run` instead of building a {class}`qiskit.circuit.QuantumCircuit` in Python (OpenQASM 3 requires `pip install mqt-yaqs[qasm3]`).

## 1. Circuit

We use a shallow randomized ansatz—single-qubit $R_y$ rotations followed by a linear chain of $CZ$ gates—typical of variational benchmarks.

```{code-cell} ipython3
import numpy as np
from qiskit.circuit import QuantumCircuit

num_qubits = 6
circuit = QuantumCircuit(num_qubits)
rng = np.random.default_rng(42)
for i in range(num_qubits):
    circuit.ry(float(rng.uniform(0.6, 2.2)), i)
for i in range(num_qubits - 1):
    circuit.cz(i, i + 1)
circuit.measure_all()
```

## 2. Initial state and noise model

```{code-cell} ipython3
from mqt.yaqs import NoiseModel, State

state = State(num_qubits, initial="zeros")

gamma = 0.5
noise_model = NoiseModel([
    {"name": "lowering", "sites": [i], "strength": gamma} for i in range(num_qubits)
])
```

Amplitude damping relaxes each qubit toward $\ket{0}$. During circuit execution the noise channels compete with unitary spreading, so readout mass shifts toward the all-zeros bitstring compared with the noiseless run.

## 3. Simulation parameters and run

`WeakSimParams` requires an explicit `shots` count (not covered by accuracy presets). We run the **same** circuit twice: once without noise (ideal readout statistics) and once with on-site amplitude damping.

```{code-cell} ipython3
from mqt.yaqs import Simulator, WeakSimParams

sim_params = WeakSimParams(shots=1024, max_bond_dim=16, svd_threshold=1e-6, random_seed=7)

sim = Simulator(show_progress=False)
result_clean = sim.run(state, circuit, sim_params)
result_noisy = sim.run(state, circuit, sim_params, noise_model)
```

For log-normal disorder on relaxation rates, see {doc}`realistic_noise_models`.

## 4. Noiseless vs noisy readout histogram

Bitstrings are sorted lexicographically among **low Hamming-weight** outcomes (at most two excitations), where amplitude damping concentrates probability. `Result.counts` keys are integers (site 0 is the least-significant bit); see {doc}`simulation_parameters` for the encoding.

```{code-cell} ipython3
---
mystnb:
  image:
    width: 90%
    align: center
---
import matplotlib.pyplot as plt
import numpy as np

def format_bitstring(key: int, num_bits: int) -> str:
    """Format a little-endian integer outcome as a zero-padded bitstring."""
    return format(key, f"0{num_bits}b")

def hamming_weight(key: int) -> int:
    return key.bit_count()

# Low-weight outcomes (|0...0> and nearby strings) where T1 noise accumulates
keys = sorted(
    k
    for k in set(result_clean.counts) | set(result_noisy.counts)
    if hamming_weight(k) <= 2
)
bitstrings = [format_bitstring(k, num_qubits) for k in keys]
x = np.arange(len(keys))
width = 0.38

clean_vals = [result_clean.counts.get(k, 0) for k in keys]
noisy_vals = [result_noisy.counts.get(k, 0) for k in keys]

fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
ax.bar(x - width / 2, clean_vals, width, label="noiseless", color="black", alpha=0.75)
ax.bar(x + width / 2, noisy_vals, width, label="noisy (amplitude damping)", color="tab:orange", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(bitstrings, rotation=45, ha="right", fontsize=8)
ax.set_xlabel("Bitstring (Hamming weight $\\leq 2$)")
ax.set_ylabel("Counts")
ax.set_title(f"Weak simulation: relaxation drives readout toward $|0\\rangle^{{\\otimes {num_qubits}}}$")
ax.legend()
ax.grid(alpha=0.3, axis="y")
plt.show()
```

## Related topics

- {doc}`strong_simulation` — strong simulation with final and mid-circuit observables
- {doc}`custom_gates` — custom unitaries and gate translation
