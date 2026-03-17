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

# Solver Comparison

It is often useful to compare different solvers to verify results or benchmark performance.
In this example, we compare the Tensor Jump Method (TJM) against the dense operator-based Monte Carlo Wavefunction (MCWF) solver and the exact Lindblad master equation solver.

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

# 3. Solver 1: Lindblad (Exact)
print("Running Lindblad...")
params_lindblad = AnalogSimParams(
    observables=[obs],
    elapsed_time=t_max,
    dt=dt,
    solver="Lindblad"
)
run(psi_0, H, params_lindblad, noise)
res_lindblad = obs.results.flatten()
times = params_lindblad.times

# 4. Solver 2: MCWF (Stochastic)
print("Running MCWF...")
params_mcwf = AnalogSimParams(
    observables=[obs],
    elapsed_time=t_max,
    dt=dt,
    solver="MCWF",
    num_traj=500
)
run(psi_0, H, params_mcwf, noise)
res_mcwf = obs.results.flatten()

# 5. Solver 3: TJM (Default Stochastic)
print("Running TJM...")
params_tjm = AnalogSimParams(
    observables=[obs],
    elapsed_time=t_max,
    dt=dt,
    solver="TJM",
    num_traj=500,
    max_bond_dim=16
)
run(psi_0, H, params_tjm, noise)
res_tjm = obs.results.flatten()

# 6. Plot Comparison
plt.figure()
plt.plot(times, res_lindblad, label="Lindblad (Exact)", linewidth=2, color="black")
plt.plot(times, res_mcwf, label="MCWF (500 traj)", linestyle="--", linewidth=2)
plt.plot(times, res_tjm, label="TJM (500 traj)", linestyle=":", linewidth=2)
plt.xlabel("Time")
plt.ylabel("$\\langle X_0 \\rangle$")
plt.legend()
plt.title("Solver Comparison")
plt.show()
```

> **Note on Statistical Uncertainty**:
> The results for `MCWF` and `TJM` are averaged over 500 trajectories. As these are stochastic methods, the plotted curves represent the mean value and carry statistical uncertainty (standard error $\propto 1/\sqrt{N_{traj}}$).
> For production-quality results, we recommend increasing `num_traj` (e.g., to 1000 or more) or plotting confidence intervals to visualize the uncertainty. The exact Lindblad solution, in contrast, is deterministic.
