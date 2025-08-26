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

# Transmon-Resonator Chain Simulation

This example demonstrates how to run an analog simulation of a chain consisting of transmon qubits and resonators using YAQS.

An MPO Hamiltonian is initialized using a coupled transmon-resonator model. An MPS is prepared in a specific computational basis state. The system is evolved under a noise-free analog simulation using the Tensor Jump Method (TJM). Finally, expectation values for all computational basis states are collected and visualized.

## Define the system Hamiltonian

```{code-cell} ipython3
import numpy as np
from mqt.yaqs.core.data_structures.networks import MPO

length = 3 # Qubit - resonator - qubit
qubit_dim = 2
resonator_dim = 2
w_q = 4/(2*np.pi)
w_r = 4/(2*np.pi)
alpha = 0
g = 0.5/(2*np.pi)

H_0 = MPO()
H_0.init_coupled_transmon(
    length=length,
    qubit_dim=qubit_dim,
    resonator_dim=resonator_dim,
    qubit_freq=w_q,
    resonator_freq=w_r,
    anharmonicity=alpha,
    coupling=g  # T_swap = pi/2g
)
```

## Define the initial state

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.networks import MPS

# Initialize in state |100⟩: left qubit excited
state = MPS(length, state="basis", basis_string='100', physical_dimensions=[qubit_dim, resonator_dim, qubit_dim])
```

## Define the noise model

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.noise_model import NoiseModel

gamma = 0.0  # No noise for this simulation
noise_model = NoiseModel([
    {"name": name, "sites": [i], "strength": gamma}
    for i in range(length)
    for name in ["lowering"]
])
```

## Define the simulation parameters

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, AnalogSimParams

elapsed_time = np.pi/(2*g) # T_swap
dt = elapsed_time/100
sample_timesteps = True
num_traj = 1
max_bond_dim = 2**length
threshold = 0
order = 1

# Measure all computational basis states
bitstrings = ["000", "001", "010", "011", "100", "101", "110", "111"]
measurements = [Observable(bstr) for bstr in bitstrings]

sim_params = AnalogSimParams(
    measurements, elapsed_time, dt, num_traj, max_bond_dim, threshold, order,
    sample_timesteps=sample_timesteps
)
```

## Run the simulation

```{code-cell} ipython3
---
tags: [remove-output]
---
from mqt.yaqs import simulator

simulator.run(state, H_0, sim_params, noise_model)
```

## Plot the results

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---

import matplotlib.pyplot as plt

leakage = [1 for _ in measurements[0].results]
for measurement in measurements:
    leakage -= measurement.results
    plt.plot(measurement.results, label=measurement.gate.bitstring)
plt.plot(leakage, label="Leakage")

plt.xlabel("Timestep")
plt.ylabel("Probability")
plt.title("Population in Computational Basis States Over Time")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()
```
