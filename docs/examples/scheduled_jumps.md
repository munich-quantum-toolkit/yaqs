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

# Scheduled Jumps

This example demonstrates how to use **scheduled noise jumps** in YAQS.
Scheduled jumps allow you to apply specific operators at predetermined times during an analog simulation. This is useful for simulating controlled gates, sudden noise events, or time-dependent perturbations without needing a full time-dependent Hamiltonian.

In this example, we simulate a 10-site Ising chain and apply a scheduled Pauli-X flip to a specific site at $t=1.0$.

```{important}
Scheduled jump times must lie on the simulation time grid: choose `time` as a multiple of `dt` (for example `dt=0.1` → `0.0, 0.1, 0.2, …`).
```

## 1. Setup

First, we define the Hamiltonian and the initial state. We'll use a standard transverse-field Ising model.

```{code-cell} ipython3
from mqt.yaqs import Hamiltonian, State

L = 10
J = 1.0
g = 1.0

# Hamiltonian: H = -J Σ Z_i Z_{i+1} - g Σ X_i
hamiltonian = Hamiltonian.ising(length=L, J=J, g=g)

# Initial state: all zeros |00...0>
state = State(L, initial="zeros")
```

## 2. Define the Scheduled Jump

We define a scheduled jump using a list of dictionaries in the `NoiseModel`. Each dictionary must specify:

- `time`: The time at which to apply the jump.
- `sites`: A list of site indices the jump acts on.
- `name`: The name of the jump operator (e.g., `"x"`, `"y"`, `"z"`, `"crosstalk_xx"`), **or** any label when you pass a custom `matrix` (see below).

If `matrix` is omitted, `name` is resolved from {class}`~mqt.yaqs.core.libraries.noise_library.NoiseLibrary`. To apply a custom operator, add a `matrix` key with a local `d×d` NumPy array (`d=2` for qubits); `name` is then only an identifier. See {doc}`realistic_noise_models` § 6 for the full process-dict schema.

```{code-cell} ipython3
from mqt.yaqs import NoiseModel

jump_time = 1.0
jump_site = 5 # Apply jump to the middle site

# Schedule a Pauli-X flip on site 5 at t=1.0
scheduled_jumps = [{"time": jump_time, "sites": [jump_site], "name": "x"}]
noise_model = NoiseModel(scheduled_jumps=scheduled_jumps)
```

### Custom operator example

A $\pi/2 rotation about $Y$ can be scheduled explicitly instead of using a library name:

```{code-cell} ipython3
import numpy as np

ry_pi2 = np.array([[1, -1], [1, 1]], dtype=complex) / np.sqrt(2)

custom_jump = [{"time": jump_time, "sites": [jump_site], "name": "ry_pi2", "matrix": ry_pi2}]
custom_noise_model = NoiseModel(scheduled_jumps=custom_jump)
```

## 3. Simulation Parameters

We measure the $Z$ expectation value on the **jumped site** ($\langle Z_5 \rangle$). A Pauli-X jump flips that site's magnetization; measuring a distant site would show only a weak entanglement signal and can look like no jump occurred.

```{code-cell} ipython3
from mqt.yaqs import AnalogSimParams, Observable

z_obs = Observable("z", sites=jump_site)

sim_params = AnalogSimParams(
    elapsed_time=5.0,
    dt=0.1,
    num_traj=1, # Jumps are deterministic, so 1 trajectory is sufficient
    observables=[z_obs],
)
```

## 4. Run Simulation

We run two simulations: one with the jump and a baseline without it.

```{code-cell} ipython3
---
tags: [remove-output]
---
from mqt.yaqs import Simulator
import copy

sim = Simulator(show_progress=False)

# Baseline
state_baseline = copy.deepcopy(state)
sim_params_baseline = copy.deepcopy(sim_params)
result_baseline = sim.run(state_baseline, hamiltonian, sim_params_baseline)

# With Jump
state_jump = copy.deepcopy(state)
sim_params_jump = copy.deepcopy(sim_params)
result_jump = sim.run(state_jump, hamiltonian, sim_params_jump, noise_model=noise_model)
```

## 5. Visualize Results

We plot the expectation value $\langle Z_{\text{jump site}} \rangle$ over time.

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
import matplotlib.pyplot as plt

times = sim_params_jump.times
res_baseline = result_baseline.expectation_values[0]
res_jump = result_jump.expectation_values[0]

plt.figure(figsize=(8, 5))
plt.plot(times, res_baseline, label="Baseline (No Jump)", color="black", linestyle="--")
plt.plot(times, res_jump, label=f"Jump on site {jump_site}", color="tab:blue")
plt.axvline(x=jump_time, color='red', linestyle=':', label="Jump Time")

plt.xlabel("Time (t)")
plt.ylabel(f"$\\langle Z_{{{jump_site}}} \\rangle$")
plt.title(f"Effect of a Scheduled Jump at $t={jump_time}$ on site {jump_site}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Related topics

- {doc}`analog_simulation` — TJM workflow and noise models
- {doc}`realistic_noise_models` — built-in and custom jump operators, distributed strengths
- {doc}`simulation_parameters` — time grids and `dt` alignment
