# Upgrade Guide

This document describes breaking changes and how to upgrade. For a complete list of changes including minor and patch releases, please refer to the [changelog](CHANGELOG.md).

## [Unreleased]

### `simulator.run` uses `State` and `Hamiltonian`

Analog and circuit entry points no longer accept raw [`MPS`](src/mqt/yaqs/core/data_structures/networks.py) /
[`MPO`](src/mqt/yaqs/core/data_structures/networks.py) objects. Use [`State`](src/mqt/yaqs/core/data_structures/state.py)
and [`Hamiltonian`](src/mqt/yaqs/core/data_structures/hamiltonian.py) instead.

**Before:**

```python
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.simulator import run

psi = MPS(4, state="zeros")
H = MPO.ising(4, J=1.0, g=0.5)
params = AnalogSimParams(..., solver="MCWF")
run(psi, H, params, noise_model)
```

**After:**

```python
from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.simulator import run

psi = State(4, initial="zeros", representation="vector")
H = Hamiltonian.ising(4, J=1.0, g=0.5)
params = AnalogSimParams(...)
run(psi, H, params, noise_model)
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
