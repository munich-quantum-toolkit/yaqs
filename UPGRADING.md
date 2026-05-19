# Upgrade Guide

This document describes breaking changes and how to upgrade. For a complete list of changes including minor and patch releases, please refer to the [changelog](CHANGELOG.md).

## [Unreleased]

### `simulator.run` becomes `Simulator(...).run(...)`

The free `mqt.yaqs.simulator.run` function has been replaced by a [`Simulator`](src/mqt/yaqs/simulator.py)
class. `Simulator` owns the execution-side configuration (parallel vs. serial execution, worker count,
progress reporting, multiprocessing context, retry policy); the physics inputs are passed to
[`Simulator.run`](src/mqt/yaqs/simulator.py). `Simulator.run` now returns a
[`Result`](src/mqt/yaqs/core/data_structures/result.py) wrapper around the populated simulation
parameters.

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
Simulator().run(psi, H, params, noise_model)
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
