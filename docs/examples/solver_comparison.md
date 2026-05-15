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

# Representation Comparison

It is often useful to compare different state representations to verify results or benchmark performance.
In this example, we compare MPS-based evolution (tensor network), dense state-vector evolution, and exact
density-matrix master-equation evolution on a small open-system benchmark.

```{code-cell} ipython3
import matplotlib.pyplot as plt
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, AnalogSimParams
from mqt.yaqs.simulator import run

# 1. Define System
L = 3
J = 1.0
g = 0.5
H = MPO.ising(L, J, g)
psi_0 = MPS(L, state="zeros")

# Noise: Dephasing on all sites
gamma = 0.2
noise_processes = [{"name": "pauli_z", "sites": [i], "strength": gamma} for i in range(L)]
noise = NoiseModel(processes=noise_processes)

# Observable: <X_0>
obs = Observable("x", sites=[0])

# 2. Simulation Parameters
t_max = 5.0
dt = 0.05

# 3. density_matrix (exact ensemble average)
print("Running density_matrix...")
params_rho = AnalogSimParams(
    observables=[obs],
    elapsed_time=t_max,
    dt=dt,
    representation="density_matrix",
)
run(psi_0, H, params_rho, noise)
res_rho = obs.results.flatten()
times = params_rho.times

# 4. vector (stochastic trajectories)
print("Running vector...")
params_vector = AnalogSimParams(
    observables=[obs],
    elapsed_time=t_max,
    dt=dt,
    representation="vector",
    num_traj=500,
)
run(psi_0, H, params_vector, noise)
res_vector = obs.results.flatten()

# 5. mps (default, stochastic trajectories)
print("Running mps...")
params_mps = AnalogSimParams(
    observables=[obs],
    elapsed_time=t_max,
    dt=dt,
    representation="mps",
    num_traj=500,
    max_bond_dim=16,
)
run(psi_0, H, params_mps, noise)
res_mps = obs.results.flatten()

# 6. Plot Comparison
plt.figure()
plt.plot(times, res_rho, label="density_matrix (exact)", linewidth=2, color="black")
plt.plot(times, res_vector, label="vector (500 traj)", linestyle="--", linewidth=2)
plt.plot(times, res_mps, label="mps (500 traj)", linestyle=":", linewidth=2)
plt.xlabel("Time")
plt.ylabel("$\\langle X_0 \\rangle$")
plt.legend()
plt.title("Representation Comparison")
plt.show()
```

> **Note on Statistical Uncertainty**:
> The results for `vector` and `mps` are averaged over 500 trajectories. As these use stochastic
> open-system evolution when a noise model is present, the plotted curves represent the mean value
> and carry statistical uncertainty (standard error $\propto 1/\sqrt{N_{traj}}$).
> For production-quality results, we recommend increasing `num_traj` (e.g., to 1000 or more) or plotting
> confidence intervals to visualize the uncertainty. The `density_matrix` representation, in contrast,
> returns the deterministic ensemble average directly.

## Noiseless cross-check

With no noise, `representation="density_matrix"` evolves $\rho$ under the Hamiltonian only
(`noise_model=None`). The same holds for `mps` and `vector` (a single deterministic trajectory).

```{code-cell} ipython3
obs_mps = Observable("z", sites=[0])
obs_rho = Observable("z", sites=[0])
params_mps_unitary = AnalogSimParams(
    observables=[obs_mps],
    elapsed_time=1.0,
    dt=0.1,
    representation="mps",
    max_bond_dim=16,
    show_progress=False,
)
params_rho_unitary = AnalogSimParams(
    observables=[obs_rho],
    elapsed_time=1.0,
    dt=0.1,
    representation="density_matrix",
    show_progress=False,
)
run(psi_0, H, params_mps_unitary, None)
z_mps = obs_mps.results[-1]
run(psi_0, H, params_rho_unitary, None)
z_rho = obs_rho.results[-1]
print(f"Noiseless <Z_0> at t=1: mps={z_mps:.6f}, density_matrix={z_rho:.6f}")
```
