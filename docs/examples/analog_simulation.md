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

This module demonstrates how to run a analog simulation using the YAQS simulator visualize the results.
In this example, an Ising Hamiltonian is built as a {class}`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian`, and a {class}`~mqt.yaqs.core.data_structures.state.State` is prepared in the $\ket{0}$ state.
A noise model is applied, and simulation parameters are defined for an analog simulation using the Tensor Jump Method (TJM).
After running the simulation, the expectation values of the $X$ observable are extracted and displayed as a heatmap.

Define the system Hamiltonian. We show 3 possible ways to define the Ising Hamiltonian as an example.

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
from mqt.yaqs.core.data_structures.mpo import MPO

L = 3
J = 1
g = 0.5

# Method 1: Pre-implemented Hamiltonians
H_0 = Hamiltonian.ising(L, J, g)

# Method 2: Same Ising Hamiltonian built via the generic Pauli interaction interface
H_0 = Hamiltonian.pauli(
    length=L,
    two_body=[(-J, "Z", "Z")],
    one_body=[(-g, "X")],
    bc="open",
)

# Method 3: Explicit Pauli-string expansion
terms = []

# -J Σ Z_i Z_{i+1}
for i in range(L - 1):
    terms.append((-J, f"Z{i} Z{i+1}"))

# -g Σ X_i
for i in range(L):
    terms.append((-g, f"X{i}"))

mpo = MPO()
mpo.from_pauli_sum(terms=terms, length=L)
H_0 = Hamiltonian.from_mpo(mpo)
```

Define the initial state

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.state import State

state = State(L, initial="zeros")
# Alternative: initialize an entangled random state with capped bond dimension.
# state = State(L, initial="haar-random", pad=4)
```

Define the noise model

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.noise_model import NoiseModel

gamma = 0.1
noise_model = NoiseModel([
    {"name": name, "sites": [i], "strength": gamma} for i in range(L) for name in ["lowering", "pauli_z"]
])
```

Define the simulation parameters

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, AnalogSimParams
from mqt.yaqs.core.libraries.gate_library import X

sim_params = AnalogSimParams(
    observables=[Observable(X(), site) for site in range(L)],
    elapsed_time=10,
    dt=0.1,
    num_traj=100,
    max_bond_dim=4,
    threshold=1e-6,
    order=2,
    sample_timesteps=True,
)
```

## Reproducible (deterministic) stochastic runs

Open-system simulations with `num_traj > 1` average over independent quantum-jump trajectories.
By default, each call to {meth}`~mqt.yaqs.Simulator.run` draws a new random jump sequence, so aggregated observables can differ slightly from run to run.

Set {attr}`~mqt.yaqs.core.data_structures.simulation_parameters.AnalogSimParams.random_seed` to fix the pseudorandom stream:

- Each trajectory uses `numpy.random.default_rng(random_seed + traj_idx)`, so parallel workers stay reproducible and independent.
- If the noise model has distribution-valued strengths, {meth}`~mqt.yaqs.Simulator.run` samples static disorder once using the same seed.

Leave `random_seed=None` (the default) for genuine Monte Carlo sampling in production.

The example below runs the same noisy setup twice; with a seed, the trajectory-averaged $\langle X \rangle$ curves match exactly.

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
    threshold=1e-6,
    order=2,
    sample_timesteps=True,
    random_seed=42,
)

sim = Simulator(parallel=True, show_progress=False)


def run_reproducible() -> list[np.ndarray]:
    st = copy.deepcopy(state)
    params = copy.deepcopy(repro_params)
    result = sim.run(st, H_0, params, copy.deepcopy(noise_model))
    return [np.asarray(obs.results) for obs in result.observables]


first_run = run_reproducible()
second_run = run_reproducible()
assert all(np.allclose(a, b) for a, b in zip(first_run, second_run, strict=True))
print("Both runs produced identical trajectory-averaged observables.")
```

The same `random_seed` field exists on {class}`~mqt.yaqs.core.data_structures.simulation_parameters.StrongSimParams` and {class}`~mqt.yaqs.core.data_structures.simulation_parameters.WeakSimParams` for noisy digital simulations.

After choosing a seed, you can tune `num_traj` for a fixed bias–variance trade-off instead of widening numerical tolerances to absorb run-to-run noise.

Run the simulation

```{code-cell} ipython3
---
tags: [remove-output]
---
from mqt.yaqs import Simulator

result = Simulator().run(state, H_0, sim_params, noise_model)
```

Plot the results

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
import matplotlib.pyplot as plt

heatmap = [observable.results for observable in result.observables]

fig, ax = plt.subplots(1, 1)
im = plt.imshow(heatmap, aspect="auto", extent=(0, 10, L, 0), vmin=0, vmax=0.5)
plt.xlabel("Site")
plt.yticks([x - 0.5 for x in list(range(1, L + 1))], [str(x) for x in range(1, L + 1)])
plt.ylabel("t")

fig.subplots_adjust(top=0.95, right=0.88)
cbar_ax = fig.add_axes(rect=(0.9, 0.11, 0.025, 0.8))
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.set_title("$\\langle X \\rangle$")

plt.show()
```
