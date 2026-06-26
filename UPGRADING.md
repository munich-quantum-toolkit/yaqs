# Upgrade Guide

This document describes breaking changes and how to upgrade. For a complete list of changes including minor and patch releases, please refer to the [changelog](CHANGELOG.md).

## [Unreleased]

The unreleased API refresh replaces free functions and deep module paths with a small set of
top-level types. The pieces fit together: construct physics objects and parameters, run through
`Simulator`, read everything from `Result`.

### Recommended migration (end-to-end)

**Before:**

```python
from mqt.yaqs import DEFAULT_MATRIX_MAX_QUBITS, simulator
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs.digital.equivalence_checker import run as check_equivalent

psi = MPS(4, state="zeros")
H = MPO.ising(4, J=1.0, g=0.5)
params = AnalogSimParams(
    observables=[Observable(Z(), sites=0), Observable("max_bond")],
    threshold=1e-8,
    solver="MCWF",
)

simulator.run(psi, H, params, noise_model, parallel=True)
print(params.observables[0].results)

equiv = check_equivalent(circuit1, circuit2, threshold=1e-6)
print(DEFAULT_MATRIX_MAX_QUBITS)
```

**After:**

```python
from mqt.yaqs import (
    AnalogSimParams,
    EquivalenceChecker,
    Hamiltonian,
    Observable,
    Simulator,
    State,
)

psi = State(4, initial="zeros", representation="vector")
H = Hamiltonian.ising(4, J=1.0, g=0.5)
params = AnalogSimParams(
    observables=[Observable("z", sites=0)],
    svd_threshold=1e-8,
)

sim = Simulator()
result = sim.run(psi, H, params, noise_model)
print(result.expectation_values[0])
print(result.max_bond)  # bond diagnostics; no longer an Observable

checker = EquivalenceChecker(threshold=1e-6, fidelity=1 - 1e-13)
equiv = checker.check(circuit1, circuit2)  # auto matrix cutover defaults to 7 qubits
```

### Breaking changes at a glance

| Area                     | Before                                                            | After                                                                                                                  |
| ------------------------ | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Entry point**          | `mqt.yaqs.simulator.run(...)`                                     | `Simulator(...).run(...)` → returns `Result`                                                                           |
| **Imports**              | `mqt.yaqs.core.data_structures.*`, `gate_library` for observables | `from mqt.yaqs import State, Hamiltonian, Observable, ...`                                                             |
| **State / Hamiltonian**  | Raw `MPS` / `MPO` passed to `run`                                 | `State` and `Hamiltonian` (set `representation` on `State`)                                                            |
| **Observables**          | `Observable(Z(), sites=0)` via `gate_library`                     | `Observable("z", sites=0)` — also `"entropy"`, `"schmidt_spectrum"`, etc.                                              |
| **Outputs**              | Written onto `*SimParams` / `Observable.results`                  | Read from `Result` (`expectation_values`, `counts`, `output_state`, …)                                                 |
| **SVD truncation**       | `threshold` on `*SimParams`                                       | `svd_threshold` (presets use the same key in `SIMULATION_PRESETS`)                                                     |
| **Execution UI**         | `show_progress` on `*SimParams`                                   | `show_progress` on `Simulator`; `num_threads` removed (unused)                                                         |
| **Equivalence checking** | `digital.equivalence_checker.run(...)`                            | `EquivalenceChecker(...).check(...)`                                                                                   |
| **Matrix auto cutover**  | `from mqt.yaqs import DEFAULT_MATRIX_MAX_QUBITS`                  | Default is **7** on `EquivalenceChecker(matrix_max_qubits=...)`; constant lives in `mqt.yaqs.equivalence_checker` only |
| **Bond diagnostics**     | `Observable("max_bond")` etc. in `observables`                    | `result.max_bond`, `result.total_bond`, `result.runtime_cost`                                                          |

### `Result` field map

`Simulator.run` no longer mutates the `*SimParams` you pass in. `result.sim_params` references
your original configuration unchanged.

| Old (`sim_params` / `Observable`)           | New (`result`)                 |
| ------------------------------------------- | ------------------------------ |
| `sim_params.observables[i].results`         | `result.expectation_values[i]` |
| `sim_params.output_state`                   | `result.output_state`          |
| `sim_params.noise_model`                    | `result.noise_model`           |
| `sim_params.results` (weak)                 | `result.counts`                |
| `sim_params.measurements`                   | `result.measurements`          |
| `sim_params.multi_time_observables_times`   | `result.multi_time_times`      |
| `sim_params.multi_time_observables_results` | `result.multi_time_results`    |

Removed from `*SimParams`: `noise_model`, `output_state`, `multi_time_observables_times`,
`multi_time_observables_results`, `measurements`, `results`, `aggregate_trajectories`,
`aggregate_measurements`. Observable _configuration_ (`observables`, `multi_time_observables`,
etc.) stays on `*SimParams`.

For MPS-backed analog and strong-digital runs, `result.runtime_cost`, `result.max_bond`, and
`result.total_bond` are filled automatically (aligned with `result.times` or the strong-sim layer
grid). MCWF, Lindblad, and weak digital runs leave these as `None`.

### MCWF / Lindblad operator ordering (dense backends)

MCWF (`State(..., representation="vector")`) and Lindblad (`representation="density_matrix"`)
embed jump operators and observables on the full Hilbert space using the same **site-0 LSB**
convention as MPS `to_vec`, Qiskit little-endian circuits, and the TJM (MPO) dissipation path.
Before this release, those dense embeddings used a different Kronecker-product order, so jump
probabilities, observables, and cross-solver comparisons could disagree with TJM even when the
`NoiseModel` definition looked identical.

**What changed:** `_embed_operator_sparse` / `_embed_observable_sparse` (and their dense
counterparts) now delegate to `state_utils.embed_*` helpers instead of building
`left ⊗ op ⊗ right` with reversed tensor-leg order.

**Why it matters:** MCWF, Lindblad, and TJM now agree on how a local operator on `sites=[i]` or
adjacent `sites=[i, i+1]` is placed in the full space. Regression tests compare TJM dissipative
norm loss to MCWF jump probabilities under lowering noise.

**What you need to do:**

- If you only pass standard `NoiseModel` processes (`sites`, built-in names, or matrices authored
  for the listed site order), **no change is required**—results may shift slightly because the
  previous ordering was incorrect.
- If you hand-built full-space jump operators or compared MCWF/Lindblad outputs to TJM using
  custom dense embeddings, rebuild those operators with
  `mqt.yaqs.core.data_structures.state_utils.embed_one_site_operator`,
  `embed_adjacent_two_site_operator`, or `embed_two_site_factors`, or pass the same local matrices
  through `NoiseModel` and let the solvers embed them.
- For adjacent two-site **matrix** processes, list sites in ascending order `[i, i+1]` with the
  local matrix written for that pair order. If you pass reversed sites `[i+1, i]`, the matrix is
  transposed automatically to match the `(i, i+1)` leg order.

### Characterization API

The legacy ``Characterizer`` workflow is replaced by :class:`~mqt.yaqs.memory_characterizer.MemoryCharacterizer`:

```python
from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer

mc = MemoryCharacterizer()
ham = Hamiltonian.ising(length=2, J=1.0, g=0.5)
params = AnalogSimParams(dt=0.1)

model = mc.train(ham, params, k=4)  # surrogate training
rho = mc.predict(model, rho0, sequence, k=4)  # fast dynamics
comb = mc.build_comb(ham, params, timesteps=[0.1])  # small-k reference
result = mc.characterize(ham, params, cut=2, k=4)  # operational memory S_V(c)
```

| Legacy intent | Replacement |
| ------------- | ----------- |
| Train surrogate | ``mc.train(...)`` |
| Predict dynamics | ``mc.predict(model, ...)`` or ``mc.predict(comb, ...)`` |
| Build reference comb | ``mc.build_comb(...)`` |
| Memory metrics | ``mc.characterize(target, cut=..., k=...)`` |

### Top-level public API

```python
from mqt.yaqs import (
    AnalogSimParams,
    EquivalenceChecker,
    Hamiltonian,
    MPO,
    MPS,
    NoiseModel,
    Observable,
    Result,
    SIMULATION_PRESETS,
    Simulator,
    State,
    StrongSimParams,
    WeakSimParams,
)
```

`Representation` is not exported at the top level (the name means different things on `State` vs
`Hamiltonian`). Custom gates and circuits still use `mqt.yaqs.core.libraries` when needed.

### Platform note

Starting with this release, x86 macOS is no longer tested in CI; we cannot guarantee that MQT YAQS
installs and runs correctly on those systems.

## [0.3.2]

### End of support for Python 3.9

Starting with this release, MQT YAQS no longer supports Python 3.9.
This is in line with the scheduled end of life of the version.
As a result, MQT YAQS is no longer tested under Python 3.9 and requires Python 3.10 or later.

<!-- Version links -->

[Unreleased]: https://github.com/munich-quantum-toolkit/yaqs/compare/v0.3.3...HEAD
[0.3.2]: https://github.com/munich-quantum-toolkit/yaqs/compare/v0.3.1...v0.3.2
