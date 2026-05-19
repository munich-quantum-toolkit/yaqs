# Upgrade Guide

This document describes breaking changes and how to upgrade. For a complete list of changes including minor and patch releases, please refer to the [changelog](CHANGELOG.md).

## [Unreleased]

### `simulator.run` becomes `Simulator(...).run(...)`

The free `mqt.yaqs.simulator.run` function has been replaced by a [`Simulator`](src/mqt/yaqs/simulator.py)
class. `Simulator` owns the execution-side configuration (parallel vs. serial execution, worker count,
progress reporting, multiprocessing context, retry policy); the physics inputs are passed to
[`Simulator.run`](src/mqt/yaqs/simulator.py). `Simulator.run` returns a
[`Result`](src/mqt/yaqs/core/data_structures/result.py) that holds all simulation outputs. The
`*SimParams` object you pass in is never mutated.

**Before:**

```python
from mqt.yaqs import simulator

simulator.run(state, op, sim_params, noise_model, parallel=True)
```

**After:**

```python
from mqt.yaqs import Simulator

sim = Simulator()
result = sim.run(state, op, sim_params, noise_model)
```

`show_progress` and `num_threads` were removed from `AnalogSimParams`, `StrongSimParams`, and
`WeakSimParams`. Pass `show_progress` to `Simulator` instead; `num_threads` was unused and has been
deleted.

### `digital.equivalence_checker.run` becomes `EquivalenceChecker(...).check(...)`

The free `mqt.yaqs.digital.equivalence_checker.run` function has been replaced by
[`EquivalenceChecker`](src/mqt/yaqs/equivalence_checker.py). `EquivalenceChecker` owns the
numerical thresholds (`threshold`, `fidelity`); the two circuits are passed to
[`EquivalenceChecker.check`](src/mqt/yaqs/equivalence_checker.py). The return value is unchanged:
a `dict` with keys `equivalent` and `elapsed_time`.

**Before:**

```python
from mqt.yaqs.digital.equivalence_checker import run

result = run(circuit1, circuit2, threshold=1e-6, fidelity=1 - 1e-13)
```

**After:**

```python
from mqt.yaqs import EquivalenceChecker

checker = EquivalenceChecker(threshold=1e-6, fidelity=1 - 1e-13)
result = checker.check(circuit1, circuit2)
```

### Read outputs from `Result`, not `*SimParams`

`Simulator.run` no longer writes outputs onto the `*SimParams` instance you pass in. Capture the
return value and read fields from `Result`. `result.sim_params` still references your original
configuration object (unchanged).

| Old (`sim_params`)                          | New (`result`)                 |
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
`aggregate_measurements`. Observable configuration (`observables`, `multi_time_observables`, etc.)
remains on `*SimParams`.

`Observable` no longer carries run outputs. After `Simulator.run`, read
`result.expectation_values[i]` (aggregated expectations), `result.trajectories[i]` (per-trajectory
data), and `result.times` (shared analog time grid). `result.observables[i]` is still the gate/sites
metadata for observable _i_.

### MPS bond diagnostics are automatic on `Result`

`runtime_cost`, `max_bond`, and `total_bond` are no longer configured as
[`Observable`](src/mqt/yaqs/core/data_structures/simulation_parameters.py) instances. For MPS-backed
analog and strong-digital runs, [`Simulator.run`](src/mqt/yaqs/simulator.py) fills
`result.runtime_cost`, `result.max_bond`, and `result.total_bond` (1D arrays aligned with
`result.times` or the strong-sim layer grid). MCWF, Lindblad, and weak digital runs leave these
fields as `None`.

**Before:**

```python
sim_params = AnalogSimParams(observables=[Observable(Z(), 0), Observable("max_bond")])
result = sim.run(state, H, sim_params)
max_bond_curve = result.expectation_values[-1]
```

**After:**

```python
sim_params = AnalogSimParams(observables=[Observable(Z(), 0)])
result = sim.run(state, H, sim_params)
max_bond_curve = result.max_bond
```

**Before:**

```python
sim = Simulator()
sim.run(state, op, sim_params, noise_model)
print(sim_params.observables[0].results)
```

**After:**

```python
sim = Simulator()
result = sim.run(state, op, sim_params, noise_model)
print(result.expectation_values[0])
```

### `simulator.run` uses `State` and `Hamiltonian`

Analog and circuit entry points no longer accept raw [`MPS`](src/mqt/yaqs/core/data_structures/mps.py) /
[`MPO`](src/mqt/yaqs/core/data_structures/mpo.py) objects. Use [`State`](src/mqt/yaqs/core/data_structures/state.py)
and [`Hamiltonian`](src/mqt/yaqs/core/data_structures/hamiltonian.py) instead.

**Before:**

```python
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.simulator import run

psi = MPS(4, state="zeros")
H = MPO.ising(4, J=1.0, g=0.5)
params = AnalogSimParams(..., solver="MCWF")
run(psi, H, params, noise_model)
```

**After:**

```python
from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.core.data_structures.state import State

psi = State(4, initial="zeros", representation="vector")
H = Hamiltonian.ising(4, J=1.0, g=0.5)
params = AnalogSimParams(...)
sim = Simulator()
result = sim.run(psi, H, params, noise_model)
```

### End of support for x86 macOS systems

Starting with this release, we can no longer guarantee support for x86 macOS systems.
x86 macOS systems are no longer tested in our CI and we can no longer guarantee that MQT YAQS installs and runs correctly on them.

## [0.3.2]

### End of support for Python 3.9

Starting with this release, MQT YAQS no longer supports Python 3.9.
This is in line with the scheduled end of life of the version.
As a result, MQT YAQS is no longer tested under Python 3.9 and requires Python 3.10 or later.

<!-- Version links -->

[Unreleased]: https://github.com/munich-quantum-toolkit/yaqs/compare/v0.3.3...HEAD
[0.3.2]: https://github.com/munich-quantum-toolkit/yaqs/compare/v0.3.1...v0.3.2
