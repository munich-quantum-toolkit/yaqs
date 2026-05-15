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

# Scheduled Noise Jumps

This example demonstrates how to use **scheduled noise jumps** in YAQS.
Scheduled jumps allow you to apply specific operators at predetermined times during an analog simulation. This is useful for simulating controlled gates, sudden noise events, or time-dependent perturbations without needing a full time-dependent Hamiltonian.

In this notebook, we simulate a 10-site Ising chain and apply a scheduled Pauli-X flip to a specific site at $t=1.0$.

## 1. Setup

First, we define the Hamiltonian and the initial state. We'll use a standard transverse-field Ising model.

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.networks import MPO, MPS

L = 10
J = 1.0
g = 1.0

# Hamiltonian: H = -J Σ Z_i Z_{i+1} - g Σ X_i
hamiltonian = MPO.ising(length=L, J=J, g=g)

# Initial state: all zeros |00...0>
state = MPS(L, state="zeros")
```

## 2. Define the Scheduled Jump

We define a scheduled jump using a list of dictionaries in the `NoiseModel`. Each dictionary must specify:

- `time`: The time at which to apply the jump.
- `sites`: A list of site indices the jump acts on.
- `name`: The name of the jump operator (e.g., "x", "y", "z", "crosstalk_xx").

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.noise_model import NoiseModel

jump_time = 1.0
jump_site = 5 # Apply jump to the middle site

# Schedule a Pauli-X flip on site 5 at t=1.0
scheduled_jumps = [{"time": jump_time, "sites": [jump_site], "name": "x"}]
noise_model = NoiseModel(scheduled_jumps=scheduled_jumps)
```

## 3. Simulation Parameters

We measure the $Z$ expectation value on the first site ($Z_0$) to see how the jump at site 5 affects the dynamics through entanglement.

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import Z

z_obs = Observable(Z(), sites=0)

sim_params = AnalogSimParams(
    elapsed_time=5.0,
    dt=0.1,
    num_traj=1, # Jumps are deterministic, so 1 trajectory is sufficient
    observables=[z_obs],
    show_progress=False
)
```

## 4. Run Simulation

We run two simulations: one with the jump and a baseline without it.

```{code-cell} ipython3
---
tags: [remove-output]
---
from mqt.yaqs import simulator
import copy

# Baseline
state_baseline = copy.deepcopy(state)
sim_params_baseline = copy.deepcopy(sim_params)
simulator.run(state_baseline, hamiltonian, sim_params_baseline)

# With Jump
state_jump = copy.deepcopy(state)
sim_params_jump = copy.deepcopy(sim_params)
simulator.run(state_jump, hamiltonian, sim_params_jump, noise_model=noise_model)
```

## 5. Visualize Results

We plot the expectation value $\langle Z_0 \rangle$ over time.

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
import matplotlib.pyplot as plt

times = sim_params_jump.times
res_baseline = sim_params_baseline.observables[0].results
res_jump = sim_params_jump.observables[0].results

plt.figure(figsize=(8, 5))
plt.plot(times, res_baseline, label="Baseline (No Jump)", color="black", linestyle="--")
plt.plot(times, res_jump, label=f"Jump on site {jump_site}", color="tab:blue")
plt.axvline(x=jump_time, color='red', linestyle=':', label="Jump Time")

plt.xlabel("Time (t)")
plt.ylabel("$\langle Z_0 \\rangle$")
plt.title(f"Effect of a Scheduled Jump at $t={jump_time}$ on site {jump_site}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```
