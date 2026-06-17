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

# Noisy Analog Simulation

This guide walks through an open-system **analog** simulation with the tensor jump method (TJM): build a Hamiltonian, attach a noise model, configure {class}`~mqt.yaqs.core.data_structures.simulation_parameters.AnalogSimParams`, and visualize time-resolved observables.

For Gaussian (bell-curve) noise strengths and static disorder, see {doc}`realistic_noise_models`. For execution options (parallelism, progress bars), see {doc}`simulator_initialization`.

## 1. Hamiltonian

Three equivalent ways to define the transverse-field Ising model $H = -J \sum_i Z_i Z_{i+1} - g \sum_i X_i$ on an open chain:

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
from mqt.yaqs.core.data_structures.mpo import MPO

L = 3
J = 1
g = 0.5

# Method 1: built-in constructor
H_0 = Hamiltonian.ising(L, J, g)

# Method 2: generic Pauli interface
H_0 = Hamiltonian.pauli(
    length=L,
    two_body=[(-J, "Z", "Z")],
    one_body=[(-g, "X")],
    bc="open",
)

# Method 3: explicit Pauli-string MPO
terms = [(-J, f"Z{i} Z{i+1}") for i in range(L - 1)] + [(-g, f"X{i}") for i in range(L)]
mpo = MPO()
mpo.from_pauli_sum(terms=terms, length=L)
H_0 = Hamiltonian.from_mpo(mpo)
```

## 2. Initial state and noise model

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.data_structures.noise_model import NoiseModel

state = State(L, initial="zeros")
# Alternative: State(L, initial="haar-random", pad=4)

gamma = 0.1
noise_model = NoiseModel([
    {"name": name, "sites": [i], "strength": gamma}
    for i in range(L)
    for name in ["lowering", "pauli_z"]
])
```

Pass a float for each `strength` here. For distribution-valued strengths (calibration spread), see {doc}`realistic_noise_models`.

## 3. Simulation parameters

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import X

sim_params = AnalogSimParams(
    observables=[Observable(X(), site) for site in range(L)],
    elapsed_time=10,
    dt=0.1,
    num_traj=100,
    max_bond_dim=4,
    svd_threshold=1e-6,
    order=2,
    sample_timesteps=True,
)
```

Optional `tdvp_sweeps` (default `1`) runs multiple symmetric TDVP substeps per physical step `dt`, improving unitary accuracy without changing the noise timestep.

## 4. Reproducible stochastic runs

With `num_traj > 1`, each {meth}`~mqt.yaqs.Simulator.run` call averages independent quantum-jump trajectories. Set {attr}`~mqt.yaqs.core.data_structures.simulation_parameters.AnalogSimParams.random_seed` to fix the pseudorandom stream across trajectories (and for distribution-valued noise strengths):

```{code-cell} ipython3
import copy

import numpy as np

from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable

repro_params = AnalogSimParams(
    observables=[Observable(X(), site) for site in range(L)],
    elapsed_time=1.0,
    dt=0.1,
    num_traj=50,
    max_bond_dim=4,
    svd_threshold=1e-6,
    order=2,
    sample_timesteps=True,
    random_seed=42,
)

sim = Simulator(parallel=True, show_progress=False)


def run_reproducible() -> list[np.ndarray]:
    st = copy.deepcopy(state)
    params = copy.deepcopy(repro_params)
    result = sim.run(st, H_0, params, copy.deepcopy(noise_model))
    return result.expectation_values


first_run = run_reproducible()
second_run = run_reproducible()
assert all(np.allclose(a, b) for a, b in zip(first_run, second_run, strict=True))
print("Both runs produced identical trajectory-averaged observables.")
```

The same `random_seed` field exists on {class}`~mqt.yaqs.core.data_structures.simulation_parameters.StrongSimParams` and {class}`~mqt.yaqs.core.data_structures.simulation_parameters.WeakSimParams`.

## 5. Run and visualize

```{code-cell} ipython3
---
tags: [remove-output]
---
result = sim.run(state, H_0, sim_params, noise_model)
```

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
import matplotlib.pyplot as plt

heatmap = result.expectation_values

fig, ax = plt.subplots(figsize=(5, 3))
im = ax.imshow(heatmap, aspect="auto", extent=(0, 10, L, 0), vmin=0, vmax=0.5)
ax.set_xlabel("Time")
ax.set_yticks([x - 0.5 for x in range(1, L + 1)], [str(x) for x in range(L)])
ax.set_ylabel("Site")

fig.subplots_adjust(top=0.95, right=0.88)
cbar_ax = fig.add_axes(rect=(0.9, 0.11, 0.025, 0.8))
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.set_title(r"$\langle X \rangle$")
plt.tight_layout()
plt.show()
```

## Related topics

- {doc}`representation_comparison` — MPS, state-vector, and density-matrix backends on the same problem
- {doc}`scheduled_jumps` — deterministic jumps at specified times
- {doc}`ensemble_evolution` — unitary ensemble correlations
