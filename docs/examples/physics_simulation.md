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

# Noisy Hamiltonian Simulation

This module demonstrates how to run a Hamiltonian simulation using the YAQS simulator visualize the results.
In this example, an Ising Hamiltonian is initialized as an MPO, and an MPS state is prepared in the $\ket{0}$ state.
A noise model is applied, and simulation parameters are defined for a physics simulation using the Tensor Jump Method (TJM).
After running the simulation, the expectation values of the $X$ observable are extracted and displayed as a heatmap.

Define the system Hamiltonian

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.networks import MPO

L = 10
J = 1
g = 0.5
H_0 = MPO()
H_0.init_ising(L, J, g)
```

Define the initial state

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.networks import MPS

state = MPS(L, state="zeros")
```

Define the noise model

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.noise_model import NoiseModel

gamma = 0.1
noise_model = NoiseModel(["relaxation", "dephasing"], [gamma, gamma])
```

Define the simulation parameters

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams

elapsed_time = 10
dt = 0.1
sample_timesteps = True
num_traj = 100
max_bond_dim = 4
threshold = 1e-6
order = 2
measurements = [Observable("x", site) for site in range(L)]
sim_params = PhysicsSimParams(measurements, elapsed_time, dt, num_traj, max_bond_dim, threshold, order, sample_timesteps=sample_timesteps)
```

Run the simulation

```{code-cell} ipython3
---
tags: [remove-output]
---
from mqt.yaqs import simulator

simulator.run(state, H_0, sim_params, noise_model)
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

heatmap = [observable.results for observable in sim_params.observables]

fig, ax = plt.subplots(1, 1)
im = plt.imshow(heatmap, aspect="auto", extent=(0, elapsed_time, L, 0), vmin=0, vmax=0.5)
plt.xlabel("Site")
plt.yticks([x - 0.5 for x in list(range(1, L + 1))], [str(x) for x in range(1, L + 1)])
plt.ylabel("t")

fig.subplots_adjust(top=0.95, right=0.88)
cbar_ax = fig.add_axes(rect=(0.9, 0.11, 0.025, 0.8))
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.set_title("$\\langle X \\rangle$")

plt.show()
```
