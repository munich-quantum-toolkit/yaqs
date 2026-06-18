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

YAQS supports multiple state **representations** for analog evolution. Each path targets a different scaling regime; the table below summarizes when each is appropriate.

For how to set `representation` on {class}`~mqt.yaqs.core.data_structures.state.State`, see {doc}`state_initialization`.

## Choosing a representation

| Path               | When to use                                             | Notes                                                                                                  |
| ------------------ | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `"mps"` (default)  | Larger systems and tensor-network-friendly Hamiltonians | TJM trajectories; tune `num_traj`, `max_bond_dim`, and {doc}`accuracy presets <simulation_parameters>` |
| `"vector"`         | MCWF / state-vector quantum trajectories                | Exponential memory in qubits; single-trajectory wavefunction dynamics                                  |
| `"density_matrix"` | Lindblad master-equation evolution                      | Exponential memory; deterministic ensemble average without trajectory sampling                         |

Practical guidance:

- Start with `preset="balanced"` (or `"fast"` while exploring) on {class}`~mqt.yaqs.core.data_structures.simulation_parameters.AnalogSimParams` and increase `num_traj` until observables stabilize.
- Tighten `max_bond_dim` / `svd_threshold` when entanglement growth demands it.
- For trade-offs between unravellings and trajectory cost, see {cite:p}`sander2026_computationalregimes` ({doc}`references`).

The sections below run the **same** noisy benchmark on all three paths so you can validate agreement on small systems.

## 1. Noisy open-system benchmark

```{code-cell} ipython3
import matplotlib.pyplot as plt

from mqt.yaqs import AnalogSimParams, Hamiltonian, NoiseModel, Observable, Simulator, State

sim = Simulator(show_progress=False)

L = 3
H = Hamiltonian.ising(L, J=1.0, g=0.5)
noise = NoiseModel([{"name": "pauli_z", "sites": [i], "strength": 0.2} for i in range(L)])
obs = Observable("x", sites=[0])

t_max = 5.0
dt = 0.05

print("Running density_matrix...")
params_rho = AnalogSimParams(observables=[obs], elapsed_time=t_max, dt=dt)
result_rho = sim.run(State(L, initial="zeros", representation="density_matrix"), H, params_rho, noise)
res_rho = result_rho.expectation_values[0].flatten()
times = params_rho.times

print("Running vector...")
params_vector = AnalogSimParams(observables=[obs], elapsed_time=t_max, dt=dt, num_traj=500)
result_vector = sim.run(State(L, initial="zeros", representation="vector"), H, params_vector, noise)
res_vector = result_vector.expectation_values[0].flatten()

print("Running mps...")
params_mps = AnalogSimParams(observables=[obs], elapsed_time=t_max, dt=dt, num_traj=500, max_bond_dim=16)
result_mps = sim.run(State(L, initial="zeros", representation="mps"), H, params_mps, noise)
res_mps = result_mps.expectation_values[0].flatten()
```

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(times, res_rho, label="density_matrix (exact)", linewidth=2, color="black")
ax.plot(times, res_vector, label="vector (500 traj)", linestyle="--")
ax.plot(times, res_mps, label="mps (500 traj)", linestyle=":")
ax.set_xlabel("Time")
ax.set_ylabel(r"$\langle X_0 \rangle$")
ax.legend()
ax.set_title("Open-system evolution across representations")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

```{note}
`vector` and `mps` curves are Monte Carlo means over 500 trajectories; statistical error scales as $1/\sqrt{N_{\mathrm{traj}}}$. The `density_matrix` path returns the deterministic ensemble average directly.
```

## 2. Noiseless cross-check

With `noise_model=None`, all three representations should agree on unitary observables (single trajectory for `mps` and `vector`):

```{code-cell} ipython3
obs_z = Observable("z", sites=[0])
params_mps_u = AnalogSimParams(observables=[obs_z], elapsed_time=1.0, dt=0.1, max_bond_dim=16)
params_rho_u = AnalogSimParams(observables=[obs_z], elapsed_time=1.0, dt=0.1)

z_mps = sim.run(State(L, initial="zeros", representation="mps"), H, params_mps_u, None).expectation_values[0][-1]
z_vec = sim.run(State(L, initial="zeros", representation="vector"), H, params_mps_u, None).expectation_values[0][-1]
z_rho = sim.run(State(L, initial="zeros", representation="density_matrix"), H, params_rho_u, None).expectation_values[0][-1]
print(f"Noiseless ⟨Z₀⟩ at t=1: mps={z_mps:.6f}, vector={z_vec:.6f}, density_matrix={z_rho:.6f}")
```

## Related topics

- {doc}`analog_simulation` — TJM workflow with MPS
- {doc}`state_initialization` — choosing a representation
- {doc}`simulation_parameters` — presets, `num_traj`, and truncation
- {doc}`quickstart` — minimal first simulation
