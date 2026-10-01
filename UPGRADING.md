# Upgrade Guide

This document describes breaking changes and how to upgrade. For a complete list
of changes including minor and patch releases, please refer to the
[changelog](CHANGELOG.md).

## [Unreleased]

### Breaking: physical sites use one spatial ordering

YAQS now uses one public spatial dense-basis order. Site 0 is the
least-significant, fastest-varying subsystem. For qubits, this matches Qiskit's
statevector and operator order. The convention applies to:

- `State(vector=...)` and `State(density_matrix=...)`;
- `Hamiltonian(matrix=...)` and `Hamiltonian(sparse_matrix=...)`;
- `Hamiltonian.to_matrix()` and `Hamiltonian.to_sparse_matrix()`;
- `MPO.from_matrix()`, `MPO.to_matrix()`, and `MPO.to_sparse_matrix()`;
- dense operators returned by `EquivalenceChecker`;
- dense MCWF and Lindblad observable and jump-operator embeddings; and
- dense and MPS memory-characterization backends.

For example, `np.kron(I, X)` applies `X` to site 0 of a two-site system.
Previously, `MPO.to_matrix()` put site 0 in the leftmost Kronecker factor, and
`MPO.from_matrix()` expected that order. This also caused dense and sparse
Hamiltonians to change their physical site assignment when TJM converted them to
an MPO. One-site and site-reflection-symmetric operators are unchanged.
Asymmetric operators can produce different, now consistent, results.

Local observable and analog-noise matrices have a related rule: matrix tensor
factors follow the explicit `sites` list. YAQS permutes input and output matrix
legs when it embeds an operator into the full site-0-LSB basis. Named two-site
noise processes preserve the letter-to-site meaning when they normalize reversed
sites. Circuit gate matrices retain Qiskit's qarg and matrix convention. See the
{ref}`physical-site ordering guide <physical-site-ordering>` for the complete
mapping.

Code written against the unreleased development branch must replace
`MPO.to_matrix_mps_order()` with `MPO.to_matrix()`. To migrate a manual matrix
that put site 0 in the leftmost Kronecker factor, rebuild its Kronecker products
in descending site order or permute its output and input site axes.

Custom observable matrices already written in listed-site order need no change.
Remove any manual matrix-leg swap that compensated for the former dense MCWF or
Lindblad embedding. Custom adjacent noise matrices require ascending sites. To
convert one written for descending sites, reverse the site list and swap both
input and output matrix tensor-factor axes. Use per-site `factors` only for
non-adjacent noise. Results from asymmetric local noise and from memory
characterization can change because those paths previously acted on or extracted
the opposite end of the chain.

Process tensors use a separate convention. `MPOProcessTensor.to_matrix()` keeps
its final-output-first causal-leg order. `MPOProcessTensor.to_sparse_matrix()`
now uses that causal order too; it previously inherited the spatial MPO order.

### Breaking: observables are independent of gates

`Observable` now accepts a named Hermitian observable, a Hermitian matrix, or a
binary bitstring. Observable instances expose `name`, `matrix`, `sites`,
`interaction`, `kind`, and `bitstring` directly. They no longer accept
`BaseGate` instances or the `gate=` keyword, and they no longer expose a `.gate`
attribute. Named gate operations that are not Hermitian are no longer valid
observables.

### Added: direct probabilities for noisy equivalence checks

`EquivalenceChecker.check(..., noise_model=...)` interprets each resolved
process `strength` as a direct branch probability after an eligible two-qubit
gate. Probabilities on the same exact support must sum to at most one; distinct
supports are sampled independently. `Simulator` continues to interpret the same
field as a Lindblad rate.

### Breaking: `MPS.norm` returns the Euclidean norm, not its square

`MPS.norm(site=None)` previously returned `<psi|psi>`, the *squared* norm,
contradicting its name and docstring. It now returns `sqrt(<psi|psi>)`, so a
state scaled to `||psi|| = 3` reports `3.0` where it previously reported `9.0`.
The site-resolved branch changed the same way.

Where an algorithm needs the squared norm `<psi|psi> = ||psi||^2`, square the
result explicitly:

```python
from mqt.yaqs.core.data_structures.mps import MPS

state = MPS(3, state="zeros")
state.tensors[0] = state.tensors[0] * 3.0

state.norm()  # 3.0 -- was 9.0
state.norm() ** 2  # 9.0 -- the previous return value of norm()
```

Replace bare `state.norm()` with `state.norm() ** 2` wherever the value feeds a
probability, such as a quantum-jump weight `dt * gamma * ||L|psi>||^2` or the
Monte-Carlo wave-function factor `1 - <psi|psi>`. Code that only compares the
result against `1.0` to check normalization is unaffected, because
`sqrt(1) == 1`.

All jump-probability call sites inside YAQS were updated to square `norm()`
explicitly, so trajectory sampling remains numerically equivalent within
floating- point tolerance (`sqrt(x)**2` need not be bit-identical to `x`).

### Added: piecewise time-dependent Hamiltonians

Use `Hamiltonian.piecewise([(H, duration), ...])` to switch static Hamiltonians
on the analog `dt` grid. Each duration must be a positive multiple of `dt`, and
the durations must sum to `elapsed_time`. This currently supports a single MPS
state with TDVP. See the
[Hamiltonian guide](docs/examples/hamiltonians.md#time-dependent-hamiltonians).

### Breaking: analog durations must contain a whole number of fixed time steps

`AnalogSimParams` now requires a finite, positive `dt` and a finite,
non-negative `elapsed_time` that is an integer multiple of `dt`. Analog backends
execute only full `dt` steps; rejecting fractional grids prevents the reported
final timestamp from disagreeing with the physically evolved duration.

Choose an integer step count and derive the duration from it:

```python
from mqt.yaqs import AnalogSimParams

num_steps = 3
dt = 0.1
params = AnalogSimParams(elapsed_time=num_steps * dt, dt=dt)
```

Calls such as `AnalogSimParams(elapsed_time=0.25, dt=0.1)` now raise a
`ValueError`. Use a divisible step size or duration instead; fractional final
steps are not supported.

### Breaking: seeded stochastic RNG streams no longer use `base_seed + traj_idx`

When `random_seed` is set, trajectory jump draws, order-2 measurement-copy
jumps, and static noise-model disorder sampling each use distinct
`numpy.random.SeedSequence` coordinates (stream tags), instead of
`base_seed + traj_idx`.

This removes aliasing between pairs such as `(base_seed=0, traj_idx=1)` and
`(base_seed=1, traj_idx=0)`, and stops order-2 `sample()` calls from advancing
the trajectory jump stream.
**Bit-identical replay of older seeded runs is not preserved.** Re-record any
baselines that depend on exact sample paths.

### Breaking: unified circuit parameters as `DigitalSimParams`

Circuit simulation now uses a single {class}`~mqt.yaqs.DigitalSimParams` type.
Outputs are selected by which fields you set (`observables`, `shots`, and/or
`get_state`). The merged class permits observables and shots **simultaneously**;
simulation physics for equivalent single-output configurations is unchanged.

All `DigitalSimParams` constructor arguments are **keyword-only**. Migration
must use keyword arguments; positional calls are rejected (a bare positional
integer would be ambiguous among `shots`, `num_traj`, and `max_bond_dim`).

`num_traj` and `shots` remain independent:

- `num_traj` — noisy stochastic trajectories for observables/diagnostics.
- `shots` — total bitstring-sample budget.
- **Noisy combined runs** execute `num_traj` trajectories and distribute the
  total `shots` across them. `shots < num_traj` is supported (some trajectories
  get zero samples but still contribute observables).
- **Noiseless runs** use one trajectory; all `shots` are sampled from that final
  state.

```python
from mqt.yaqs import DigitalSimParams, Observable

# Observables (optionally with layer sampling)
DigitalSimParams(observables=[Observable("z", sites=0)], sample_layers=True)
DigitalSimParams(
    observables=[Observable("z", sites=0)],
    num_traj=7,
    max_bond_dim=64,
)

# Shot readout
DigitalSimParams(shots=1024)
DigitalSimParams(shots=1024, max_bond_dim=4)

# Observables and shots in one run (noisy: distribute shots across num_traj)
DigitalSimParams(
    observables=[Observable("z", sites=0)],
    shots=1024,
    num_traj=64,
)
```

| Area           | Notes                                   |
| -------------- | --------------------------------------- |
| Circuit params | Use `DigitalSimParams` only             |
| Observables    | `DigitalSimParams(observables=...)`     |
| Shots          | `DigitalSimParams(shots=N)`             |
| Constructor    | Keyword-only (`*` after `self`)         |
| Combined       | Set both `observables` and `shots`      |
| Example docs   | `circuit_observables` / `circuit_shots` |

### Qiskit 2.1 minimum

The minimum Qiskit version increases from **1.1.0 to 2.1.0**, dropping support
for all Qiskit 1.x releases and Qiskit 2.0. Upgrade Qiskit to 2.1.0 or newer.

### End of support for Python 3.10

Starting with this release, MQT YAQS no longer supports Python 3.10. As a
result, MQT YAQS is no longer tested under Python 3.10 and requires Python 3.11
or later.

### Restored testing on Python 3.13 and 3.14

MQT YAQS is again tested on Python 3.13 and 3.14. The optional Torch integration
requires Torch 2.6 or later on Python 3.13 and Torch 2.9 or later on Python
3.14.

## [0.6.0]

The unreleased API refresh replaces free functions and deep module paths with a
small set of top-level types. The pieces fit together: construct physics objects
and parameters, run through `Simulator`, read everything from `Result`.

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

`Simulator.run` no longer writes output data onto the `*SimParams` you pass in.
Before execution, it validates and normalizes mutable controls in place. For
`AnalogSimParams`, this also rebuilds `times` from the current `elapsed_time`
and `dt`. `result.sim_params` references the same normalized object.

| Old (`sim_params` / `Observable`)           | New (`result`)                 |
| ------------------------------------------- | ------------------------------ |
| `sim_params.observables[i].results`         | `result.expectation_values[i]` |
| `sim_params.output_state`                   | `result.output_state`          |
| `sim_params.noise_model`                    | `result.noise_model`           |
| `sim_params.results` (shot counts)          | `result.counts`                |
| `sim_params.measurements`                   | `result.measurements`          |
| `sim_params.multi_time_observables_times`   | `result.multi_time_times`      |
| `sim_params.multi_time_observables_results` | `result.multi_time_results`    |

Removed from `*SimParams`: `noise_model`, `output_state`,
`multi_time_observables_times`, `multi_time_observables_results`,
`measurements`, `results`, `aggregate_trajectories`, `aggregate_measurements`.
Observable *configuration* (`observables`, `multi_time_observables`, etc.) stays
on `*SimParams`.

For MPS-backed analog and digital-observable runs, `result.runtime_cost`,
`result.max_bond`, and `result.total_bond` are filled automatically (aligned
with `result.times` or the digital layer-sampling grid). MCWF, Lindblad, and
shot-only digital runs leave these as `None`.

### Top-level public API

```python
from mqt.yaqs import (
    AnalogSimParams,
    DigitalSimParams,
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
)
```

`Representation` is not exported at the top level (the name means different
things on `State` vs `Hamiltonian`). Custom gates and circuits still use
`mqt.yaqs.core.libraries` when needed.

### Platform note

Starting with this release, x86 macOS is no longer tested in CI; we cannot
guarantee that MQT YAQS installs and runs correctly on those systems.

## [0.3.2]

### End of support for Python 3.9

Starting with this release, MQT YAQS no longer supports Python 3.9. This is in
line with the scheduled end of life of the version. As a result, MQT YAQS is no
longer tested under Python 3.9 and requires Python 3.10 or later.

<!-- Version links -->

[Unreleased]: https://github.com/munich-quantum-toolkit/yaqs/compare/v0.3.3...HEAD
[0.3.2]: https://github.com/munich-quantum-toolkit/yaqs/compare/v0.3.1...v0.3.2
