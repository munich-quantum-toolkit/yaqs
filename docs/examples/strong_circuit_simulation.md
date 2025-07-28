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

# Strong Circuit Simulation (Observable)

This example demonstrates how to run a circuit simulation using the YAQS simulator.
An Ising circuit is created and an initial MPS is prepared in the $\ket{0}$ state.
A noise model is applied and simulation parameters (using StrongSimParams) are defined.
The simulation is run for a range of noise strengths (gamma values), and the expectation values of the $Z$ observable are recorded and displayed as a heatmap.

Define the circuit

```{code-cell} ipython3
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit

num_qubits = 10
circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=10)
circuit.measure_all()
circuit.draw(output="mpl")
```

Define the initial state

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.libraries.gate_library import Z

state = MPS(num_qubits, state="zeros")
```

Define the simulation parameters

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams

num_traj = 100
max_bond_dim = 4
threshold = 1e-6
measurements = [Observable(Z(), site) for site in range(num_qubits)]
sim_params = StrongSimParams(measurements, num_traj, max_bond_dim, threshold)
```

Run the simulations for a range of noise strengths

```{code-cell} ipython3
---
tags: [remove-output]
---
from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.noise_model import NoiseModel

import numpy as np

gammas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
heatmap = np.empty((num_qubits, len(gammas)))
for j, gamma in enumerate(gammas):
    # Define the noise model
    noise_model = NoiseModel([
        {"name": name, "sites": [i], "strength": gamma} for i in range(num_qubits) for name in ["relaxation"]
    ])
    simulator.run(state, circuit, sim_params, noise_model)
    for i, observable in enumerate(sim_params.observables):
        heatmap[i, j] = observable.results[0]
```

Display the results as a heatmap

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)
im = ax.imshow(heatmap, aspect="auto", vmin=0.5, vmax=1)
ax.set_ylabel("Site")
ax.set_xticks(range(len(gammas)))
formatted_gammas = [f"$10^{{{int(np.log10(g))}}}$" for g in gammas]
ax.set_xticklabels(formatted_gammas)

fig.subplots_adjust(top=0.95, right=0.88)
cbar_ax = fig.add_axes(rect=(0.9, 0.11, 0.025, 0.8))
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.ax.set_title("$\\langle Z \\rangle$")

plt.show()
```
