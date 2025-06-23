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

# Fermi Hubbard Simulation

This example demonstrates how to run a Hamiltonian simulation using the YAQS simulator with a 2D Fermi-Hubbard model.
An Fermi-Hubbard circuit is created and an initial MPS is prepared in the domain wall state.
The simulation parameters (using StrongSimParams) are defined and the simulation is run. Afterwards, the occupation probability $\braket{n_{i,\sigma}} = \frac{1}{2} \left( 1 - \braket{Z_{i,\sigma}} \right)$ is calculated for each site.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.libraries.circuit_library import create_2d_fermi_hubbard_circuit

from mqt.yaqs import simulator
```

Define the model

```{code-cell} ipython3
t = 1.0         # kinetic hopping
mu = 0.5        # chemical potential
u = 4.0         # onsite interaction
Lx, Ly = 2, 2   # lattice dimensions
```

Define the circuit for a total time of $T=10$

```{code-cell} ipython3
num_sites = Lx * Ly
num_qubits = 2 * num_sites
num_trotter_steps = 1
timesteps = 1000
dt = 0.01
total_time = dt * timesteps
print("Total time: ", total_time)
```

Set the initial state to the wall state

```{code-cell} ipython3
state = MPS(num_qubits, state='wall', pad=32)
```

Define the simulation parameters

```{code-cell} ipython3
N = 1
max_bond_dim = 16
threshold = 1e-6
window_size = 0
measurements = [Observable('z', site) for site in range(num_qubits)]
```

Run the simulation for the specified time

```{code-cell} ipython3
occupations = np.zeros((num_qubits, timesteps), dtype='complex')

for timestep in range(timesteps):
    print("Timestep: " + str(timestep))
    circuit = create_2d_fermi_hubbard_circuit(Lx=Lx, Ly=Ly, u=u, t=t, mu=mu,
                                              num_trotter_steps=num_trotter_steps,
                                              dt=dt, timesteps=1)

    sim_params = StrongSimParams(measurements, N, max_bond_dim, threshold, window_size, get_state=True)

    simulator.run(state, circuit, sim_params, None)

    for observable in sim_params.observables:
        index = observable.sites[0]
        occupations[index, timestep] = 0.5 * (1 - observable.results.item())

    state = MPS(num_qubits, sim_params.output_state.tensors, pad=32)
```

Plot the time evolution of the site occupations for the simulated time

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
plt.figure(figsize=(10, 5))
for i in range(num_qubits):
    site = i // 2
    spin = "↑" if i % 2 == 0 else '↓'
    plt.plot(occupations[i, :], label=f"Site {site} " + spin)

plt.xlabel("Time")
plt.ylabel("Occupation")
plt.title("2D Hubbard Model: Time Evolution of Site Occupations (mqt-yaqs)")
plt.legend()
plt.show()
```
