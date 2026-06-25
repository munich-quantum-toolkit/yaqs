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

# Quickstart

This page runs three minimal workflows end-to-end plus a circuit equivalence check. Install the package first ({doc}`installation`), then copy the cells below.

Every example in this guide uses `Simulator(show_progress=False)` so progress bars do not clutter the built documentation.

## 1. Analog simulation

Tensor-network time evolution on an MPS. Attach a `NoiseModel` for open-system TJM trajectories; without noise, the simulator performs a single unitary trajectory regardless of `num_traj`:

```{code-cell} ipython3
---
tags: [remove-output]
---
from mqt.yaqs import AnalogSimParams, Hamiltonian, Observable, Simulator, State

L = 3
state = State(L, initial="zeros")
hamiltonian = Hamiltonian.ising(L, J=1.0, g=0.5)

params = AnalogSimParams(
    observables=[Observable("z", sites=0)],
    elapsed_time=0.5,
    dt=0.1,
    preset="fast",
    num_traj=8,
)

sim = Simulator(show_progress=False)
result = sim.run(state, hamiltonian, params)
```

## 2. Digital circuit simulation

Noisy or noise-free evolution through a Qiskit circuit on an MPS:

```{code-cell} ipython3
---
tags: [remove-output]
---
from qiskit.circuit import QuantumCircuit

from mqt.yaqs import Observable, StrongSimParams

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)

circuit_state = State(3, initial="zeros")
circuit_params = StrongSimParams(
    observables=[Observable("z", sites=i) for i in range(3)],
    preset="fast",
    num_traj=8,
)

circuit_result = sim.run(circuit_state, qc, circuit_params)
```

## 3. Equivalence checking

Verify that two circuits implement the same unitary (up to global phase) with {class}`~mqt.yaqs.EquivalenceChecker`:

```{code-cell} ipython3
---
tags: [remove-output]
---
from mqt.yaqs import EquivalenceChecker
from qiskit.circuit import QuantumCircuit

def bell_prep() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc

# Original vs. transpiled-style rewrite (same unitary on two qubits)
original = bell_prep()
rewritten = QuantumCircuit(2)
rewritten.h(0)
rewritten.cx(0, 1)

checker = EquivalenceChecker(representation="mpo", threshold=1e-6)
equiv = checker.check(original, rewritten)
```

For larger circuits, compiler passes, and OpenQASM inputs, see {doc}`equivalence_checking`.

## 4. Operational memory characterization

Primary memory metric via Hamiltonian characterize:

```{code-cell} ipython3
---
tags: [remove-output]
---
from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer

ham = Hamiltonian.ising(length=1, J=1.0, g=0.5)
params = AnalogSimParams(dt=0.1, max_bond_dim=12, order=1)
result = MemoryCharacterizer(parallel=False, show_progress=False).characterize(
    ham, params, k=1, cut=1, n_pasts=4, n_futures=4,
)
print(result.summary())
```

See {doc}`characterization` for the full funnel (surrogate predict, reference comb validation).

## 5. Where to go next

| Goal                                                | Start here                    |
| --------------------------------------------------- | ----------------------------- |
| Operational memory (S_V, V-matrix)                  | {doc}`characterization`       |
| Open-system dynamics, noise, time grids             | {doc}`analog_simulation`      |
| Bell-curve (log-normal) noise strengths             | {doc}`realistic_noise_models` |
| Circuit observables, mid-circuit sampling, OpenQASM | {doc}`circuit_simulation`     |
| Accuracy presets and truncation knobs               | {doc}`simulation_parameters`  |
| Check two circuits for equivalence                  | {doc}`equivalence_checking`   |

## Related topics

- {doc}`state_initialization` — `State` presets and representations
- {doc}`simulator_initialization` — parallelism, progress bars, `Result` fields
- {doc}`representation_comparison` — when to use MPS, MCWF, or Lindblad backends
- {doc}`equivalence_checking` — MPO backend, transpiler regression tests, OpenQASM
