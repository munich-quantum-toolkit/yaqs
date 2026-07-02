---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
  execution_timeout: 900
---

```{code-cell} ipython3
:tags: [remove-cell]
%config InlineBackend.figure_formats = ['svg']
```

# Lindblad Noise-Rate Learning

This tutorial fits unknown Lindblad jump rates by matching **simulated observable time series** to a reference trajectory.
YAQS uses the tensor jump method (TJM) as the forward model and **CMA-ES** as the default optimizer.

Install the optional dependency with `pip install mqt.yaqs[noise]` (pulls in `cma`).
The entry point is {class}`~mqt.yaqs.noise_characterizer.NoiseCharacterizer`.

```{note}
Rates are not always uniquely identifiable from a fixed observable set.
Judge a fit by **trajectory overlap** and the cost $J$, not only by comparing learned $\gamma$ to planted values.
```

## 1. Problem setup

Single-site transverse-field Ising qubit with a stronger drive ($g=2$).
From $|0\rangle$, $\langle Y\rangle$ and $\langle Z\rangle$ carry the dissipation signal (while $\langle X\rangle$ stays near zero).
We plant a known `reference_model` for this benchmark; in an experiment you would supply measured expectations instead.

```{code-cell} ipython3
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", message=".*special injected samples.*")

from mqt.yaqs import AnalogSimParams, CompactNoiseModel, Hamiltonian, NoiseCharacterizer, Observable, State
from mqt.yaqs.core.libraries.gate_library import X, Y, Z

n_sites = 1
j_coupling = 1.0
transverse_field = 2.0
gamma_ref = 0.08
gamma_init = 0.35
cma_seed = 42

hamiltonian = Hamiltonian.ising(n_sites, J=j_coupling, g=transverse_field)
init_state = State(n_sites, initial="zeros")

observables = [Observable(X(), 0), Observable(Y(), 0), Observable(Z(), 0)]

sim_params = AnalogSimParams(
    observables=observables,
    elapsed_time=0.8,
    dt=0.1,
    num_traj=16,
    max_bond_dim=8,
    order=1,
    sample_timesteps=True,
    random_seed=21,
)

reference_model = CompactNoiseModel([
    {"name": "pauli_x", "sites": [0], "strength": gamma_ref},
    {"name": "pauli_y", "sites": [0], "strength": gamma_ref},
    {"name": "pauli_z", "sites": [0], "strength": gamma_ref},
])

init_guess = CompactNoiseModel([
    {"name": "pauli_x", "sites": [0], "strength": gamma_init},
    {"name": "pauli_y", "sites": [0], "strength": gamma_init},
    {"name": "pauli_z", "sites": [0], "strength": gamma_init},
])

rate_bounds_low = np.zeros(3)
rate_bounds_high = np.full(3, 0.5)
pauli_labels = ["X", "Y", "Z"]
gamma_reference = np.full(len(pauli_labels), gamma_ref)
```

Each compact process fixes the **topology** (which sites and jump operators) while the characterizer optimizes one **strength** per process.
Use a large enough `num_traj` in `AnalogSimParams` — the optimizer reuses that ensemble size on every loss evaluation.

## 2. Reference trajectory and characterizer

{meth}`~mqt.yaqs.noise_characterizer.NoiseCharacterizer.from_reference` simulates the reference once and wires a {class}`~mqt.yaqs.characterization.noise.shared.loss.TrajectoryLoss` for subsequent optimizer evaluations.

```{code-cell} ipython3
characterizer = NoiseCharacterizer.from_reference(
    sim_params=sim_params,
    hamiltonian=hamiltonian,
    init_state=init_state,
    reference_model=reference_model,
    init_guess=init_guess,
    observables=observables,
)

ref_traj = characterizer.loss.ref_traj_array.copy()
times = characterizer.propagator.times
```

## 3. Optimize with CMA-ES

Pass box constraints with one bound per compact process (same order as `init_guess.compact_processes`).
Set `seed` for reproducible CMA-ES runs; additional keyword arguments (`sigma0`, `popsize`, `max_iter`, …) are forwarded to the wrapper.

```{code-cell} ipython3
result = characterizer.optimize(
    x_low=rate_bounds_low,
    x_up=rate_bounds_high,
    sigma0=0.05,
    popsize=8,
    max_iter=40,
    seed=cma_seed,
)

sqrt_j_before = float(np.sqrt(result.loss_history[0]))
sqrt_j_after = float(np.sqrt(result.best_loss))
gamma_learned = result.best_parameters.copy()

print(f"√J: {sqrt_j_before:.3f} → {sqrt_j_after:.2e}")
```

## 4. Validate the fit

Re-simulate with the optimized model and compare both the **dynamics** (primary metric) and the **learned rates** (secondary, often non-unique).

```{code-cell} ipython3
characterizer.propagator.run(result.optimal_model)
fit_traj = characterizer.propagator.obs_array.copy()

residual = fit_traj - ref_traj
traj_rmse = float(np.sqrt(np.mean(residual**2)))
max_abs_err = float(np.max(np.abs(residual)))

print(f"trajectory RMSE (optimized vs reference): {traj_rmse:.2e}")
print(f"max |optimized − reference|:             {max_abs_err:.2e}")

fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]})

x_pos = np.arange(len(pauli_labels))
bar_width = 0.35
axes[0].bar(x_pos - bar_width / 2, gamma_reference, bar_width, label="reference", color="0.35")
axes[0].bar(x_pos + bar_width / 2, gamma_learned, bar_width, label="optimized", color="C0")
axes[0].set_xticks(x_pos, pauli_labels)
axes[0].set_ylabel(r"$\gamma$")
axes[0].set_title("Learned rates vs. reference")
axes[0].legend(loc="upper right", fontsize=8)

fit_obs = [(1, r"$\langle Y\rangle$"), (2, r"$\langle Z\rangle$")]
for ax, (obs_idx, ylabel) in zip(axes[1:], fit_obs, strict=True):
    ax.plot(times, fit_traj[obs_idx], color="C0", lw=2.5, label="optimized", zorder=1)
    ax.plot(times, ref_traj[obs_idx], color="0.2", ls=":", lw=2.5, label="reference", zorder=2)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-1.05, 1.05)
    panel_rmse = float(np.sqrt(np.mean(residual[obs_idx] ** 2)))
    ax.text(0.03, 0.06, rf"RMSE={panel_rmse:.1e}", transform=ax.transAxes, fontsize=8)
    ax.legend(loc="upper right", fontsize=8)

fig.suptitle("Optimized trajectories overlap the reference", y=1.05, fontsize=11)
fig.tight_layout()
```

The rate bar chart may still deviate from the planted $\gamma$ values even when the dynamics agree.

## Workflow summary

| Step                                                          | API                                                     |
| ------------------------------------------------------------- | ------------------------------------------------------- |
| 1. Fix Hamiltonian, initial state, observables, TJM settings  | `Hamiltonian`, `State`, `AnalogSimParams`, `Observable` |
| 2. Declare jump-operator topology and initial rate guess      | `CompactNoiseModel`                                     |
| 3. Provide reference expectations (simulated or experimental) | `NoiseCharacterizer.from_reference`                     |
| 4. Run derivative-free optimization                           | `NoiseCharacterizer.optimize`                           |
| 5. Validate dynamics and rates                                | trajectory RMSE, overlays, and rate comparison          |

Lower-level building blocks live under `mqt.yaqs.characterization.noise` (`Propagator`, `TrajectoryLoss`) if you need custom optimizers or training loops.

## See also

- {doc}`analog_simulation` — TJM open-system simulation
- {doc}`realistic_noise_models` — `NoiseModel` processes, crosstalk, and custom jump operators
- {doc}`characterization` — non-Markovian **memory** characterization (orthogonal to rate learning here)
