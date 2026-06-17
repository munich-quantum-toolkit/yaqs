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

# Transmon-Resonator Chain Emulation

This example demonstrates how to run an analog simulation of a chain consisting of transmon qubits and resonators using YAQS.

A {class}`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian` is initialized using a coupled transmon-resonator model. A {class}`~mqt.yaqs.core.data_structures.state.State` is prepared in a specific computational basis state. The system is evolved under a noise-free analog simulation using the Tensor Jump Method (TJM). Finally, expectation values for all computational basis states are collected and visualized.

## Define the system Hamiltonian

```{code-cell} ipython3
import numpy as np
from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian

length = 3 # Qubit - resonator - qubit
qubit_dim = 2
resonator_dim = 2
w_q = 4/(2*np.pi)
w_r = 4/(2*np.pi)
alpha = 0
g = 2/(2*np.pi)

H_0 = Hamiltonian.coupled_transmon(
    length=length,
    qubit_dim=qubit_dim,
    resonator_dim=resonator_dim,
    qubit_freq=w_q,
    resonator_freq=w_r,
    anharmonicity=alpha,
    coupling=g,  # T_swap = pi/(2g)
)
```

## Define the initial state

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.state import State

# Initialize in state |100⟩: left qubit excited
state = State(
    length,
    initial="basis",
    basis_string="100",
    physical_dimensions=[qubit_dim, resonator_dim, qubit_dim],
)
```

## Define the simulation parameters

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, AnalogSimParams

T_swap = np.pi/(2*g)

# Measure all computational basis states
bitstrings = ["000", "001", "010", "011", "100", "101", "110", "111"]
sim_params = AnalogSimParams(
    observables=[Observable(bstr) for bstr in bitstrings], elapsed_time=T_swap, dt=T_swap/1000,
    sample_timesteps=True
)
```

## Run the simulation

This is a noise-free unitary evolution; omit `noise_model` (it defaults to `None`).

```{code-cell} ipython3
---
tags: [remove-output]
---
from mqt.yaqs import Simulator

sim = Simulator(show_progress=False)
result = sim.run(state, H_0, sim_params)
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
import numpy as np

pvm_observables = result.observables
leakage = np.ones_like(result.expectation_values[0])
for measurement, values in zip(pvm_observables, result.expectation_values, strict=True):
    leakage -= values
    plt.plot(values, label=measurement.gate.bitstring)
plt.plot(leakage, label="Leakage")

plt.xlabel("Timestep")
plt.ylabel("Probability")
plt.title("Population in Computational Basis States Over Time")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()
```

## Related topics

- {doc}`analog_simulation` — TJM time evolution
- {doc}`state_initialization` — custom `physical_dimensions` and basis states
- {doc}`simulation_parameters` — `sample_timesteps` and observables
