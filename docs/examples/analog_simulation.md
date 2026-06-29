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

# Noisy Analog Simulation

This guide walks through an open-system **analog** simulation with the tensor jump method (TJM): build a Hamiltonian, attach a noise model, configure {class}`~mqt.yaqs.core.data_structures.simulation_parameters.AnalogSimParams`, and visualize time-resolved observables.

For log-normal disorder on strengths and static calibration spread, see {doc}`realistic_noise_models`. For execution options (parallelism, progress bars), see {doc}`simulator_initialization`. To build Ising, Hubbard, Pauli-string, or hardware Hamiltonians, see {doc}`hamiltonians`.

## 1. Hamiltonian

```{code-cell} ipython3
from mqt.yaqs import Hamiltonian

L = 5
J, g = 1.0, 0.8
H_0 = Hamiltonian.ising(L, J, g)
```

See {doc}`hamiltonians` for Pauli sums, Fermi–Hubbard, Bose–Hubbard, and coupled-transmon factories.

## 2. Initial state and noise model

We prepare a Néel state $\ket{01010\ldots}$ and track staggered magnetization under a transverse-field Ising model with on-site amplitude damping. The alternating $\langle Z_i \rangle$ pattern at $t=0$ spreads and decays in a site-dependent way.

```{code-cell} ipython3
from mqt.yaqs import NoiseModel, State

state = State(L, initial="Neel")

gamma = 0.08
noise_model = NoiseModel([
    {"name": "lowering", "sites": [i], "strength": gamma} for i in range(L)
])
```

Pass a float for each `strength` here. For distribution-valued strengths (log-normal and other distributions), see {doc}`realistic_noise_models`.

## 3. Simulation parameters

```{code-cell} ipython3
from mqt.yaqs import AnalogSimParams, Observable

sim_params = AnalogSimParams(
    observables=[Observable("z", site) for site in range(L)],
    elapsed_time=6.0,
    dt=0.1,
    num_traj=20,
    max_bond_dim=16,
    svd_threshold=1e-6,
    order=2,
    sample_timesteps=True,
)
```

Optional `tdvp_sweeps` (default `1`) runs multiple symmetric TDVP substeps per physical step `dt`, improving unitary accuracy without changing the noise timestep.

**Evolution integrator:** analog simulations default to `EvolutionMode.TDVP` (two-site TDVP sweeps). `EvolutionMode.BUG` is available as an alternative on {class}`~mqt.yaqs.core.data_structures.simulation_parameters.AnalogSimParams` when you want the BUG integrator instead.

## 4. Reproducible stochastic runs

With `num_traj > 1`, each {meth}`~mqt.yaqs.Simulator.run` call averages independent quantum-jump trajectories. Set {attr}`~mqt.yaqs.core.data_structures.simulation_parameters.AnalogSimParams.random_seed` to fix the pseudorandom stream across trajectories (and for distribution-valued noise strengths):

```{code-cell} ipython3
---
tags: [remove-output]
---
import copy

import numpy as np

from mqt.yaqs import AnalogSimParams, Observable, Simulator

repro_params = AnalogSimParams(
    observables=[Observable("z", site) for site in range(L)],
    elapsed_time=1.0,
    dt=0.1,
    num_traj=16,
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

fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
im = ax.imshow(heatmap, aspect="auto", extent=(0, 6, L, 0), vmin=-1, vmax=1)
ax.set_xlabel("Time")
ax.set_yticks([x - 0.5 for x in range(1, L + 1)], [str(x) for x in range(L)])
ax.set_ylabel("Site")
fig.colorbar(im, ax=ax, shrink=0.9, label=r"$\langle Z \rangle$")
plt.show()
```

## Related topics

- {doc}`hamiltonians` — Pauli, Hubbard, and hardware Hamiltonians
- {doc}`representation_comparison` — MPS, MCWF, and Lindblad backends
- {doc}`scheduled_jumps` — deterministic jumps at specified times
- {doc}`ensemble_evolution` — unitary ensemble correlations
- {doc}`quickstart` — minimal first simulation
