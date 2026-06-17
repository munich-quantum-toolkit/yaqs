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

sim = Simulator(show_progress=False)
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
