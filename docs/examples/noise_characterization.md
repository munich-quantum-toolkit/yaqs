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

# Noise Digital Twin from Experimental Trajectories

Build a **digital twin** of an open quantum system: learn unknown Lindblad jump rates from observable time series, validate the fit on the measured traces, then deploy the learned model in {class}`~mqt.yaqs.Simulator` to predict **held-out** observables.

Install the optional dependency with `pip install mqt.yaqs[noise]` (pulls in `cma`).
The entry point is {class}`~mqt.yaqs.noise_characterizer.NoiseCharacterizer`.

```{note}
Rates are not always uniquely identifiable from a sparse observable set.
Judge a fit by **trajectory overlap** first; rate bars are secondary validation.
```

```{note}
**Forward backends:** `representation="auto"` (default) prefers deterministic Lindblad on small chains, then MCWF (`"vector"`), then TJM (`"mps"`). See {doc}`representation_comparison` for cross-backend validation.
```

## 1. Minimal fit

Three-site transverse-field Ising chain with homogeneous Pauli noise. Pass `reference_model=` to simulate target trajectories internally (benchmark shortcut); for lab data use `ref_expectations=` instead (section 3).

```{code-cell} ipython3
import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", message=".*special injected samples.*")

from mqt.yaqs import AnalogSimParams, Hamiltonian, NoiseCharacterizer, NoiseModel, Observable, Simulator, State

n_sites = 3
j_coupling = 1.0
transverse_field = 2.0
gamma_true = 0.08
gamma_init = 0.35
cma_seed = 42
sites = list(range(n_sites))

hamiltonian = Hamiltonian.ising(n_sites, J=j_coupling, g=transverse_field)
init_state = State(n_sites, initial="zeros")

fitting_observables = [
    Observable("y", 0),
    Observable("z", 0),
    Observable("y", 1),
]
prediction_observables = [
    Observable("x", 0),
    Observable("x", 1),
    Observable("x", 2),
    Observable("z", 2),
]

sim_params = AnalogSimParams(
    observables=fitting_observables,
    elapsed_time=0.8,
    dt=0.1,
    order=1,
    sample_timesteps=True,
)

reference_model = NoiseModel(
    [{"name": "pauli_x", "sites": [s], "strength": gamma_true} for s in sites]
    + [{"name": "pauli_y", "sites": [s], "strength": gamma_true} for s in sites]
    + [{"name": "pauli_z", "sites": [s], "strength": gamma_true} for s in sites]
)

init_guess = NoiseModel(
    [{"name": "pauli_x", "sites": [s], "strength": gamma_init} for s in sites]
    + [{"name": "pauli_y", "sites": [s], "strength": gamma_init} for s in sites]
    + [{"name": "pauli_z", "sites": [s], "strength": gamma_init} for s in sites]
)

rate_bounds_low = np.zeros(len(init_guess.processes))
rate_bounds_high = np.full(len(init_guess.processes), 0.5)
pauli_labels = ["X", "Y", "Z"]

nc = NoiseCharacterizer(show_progress=False)
result = nc.characterize(
    hamiltonian,
    sim_params,
    init_state=init_state,
    init_guess=init_guess,
    observables=fitting_observables,
    reference_model=reference_model,
    x_low=rate_bounds_low,
    x_up=rate_bounds_high,
    sigma0=0.05,
    popsize=8,
    max_iter=40,
    seed=cma_seed,
)

gamma_learned = np.array([
    result.best_parameters[0:n_sites].mean(),
    result.best_parameters[n_sites : 2 * n_sites].mean(),
    result.best_parameters[2 * n_sites : 3 * n_sites].mean(),
])
times = result.times
print(f"√J: {result.sqrt_loss_before():.3f} → {result.sqrt_loss_after():.2e}")
print(f"fitting trajectory RMSE: {result.trajectory_rmse():.2e}")
```

## 2. Validate fitted dynamics and rates

```{code-cell} ipython3
gamma_reference = np.full(len(pauli_labels), gamma_true)
ref_traj = result.ref_traj
fit_traj = result.fit_traj

fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), gridspec_kw={"width_ratios": [1.1, 1.0, 1.0]})

x_pos = np.arange(len(pauli_labels))
bar_width = 0.35
axes[0].bar(x_pos - bar_width / 2, gamma_reference, bar_width, label=r"$\gamma_{\mathrm{true}}$", color="0.35")
axes[0].bar(x_pos + bar_width / 2, gamma_learned, bar_width, label="learned twin", color="C0")
axes[0].set_xticks(x_pos, pauli_labels)
axes[0].set_ylabel(r"$\gamma$")
axes[0].set_title("Learned rates vs. hidden truth")
axes[0].legend(loc="upper right", fontsize=8)

fit_panels = [(0, r"$\langle Y_0\rangle$"), (1, r"$\langle Z_0\rangle$")]
for ax, (obs_idx, ylabel) in zip(axes[1:], fit_panels, strict=True):
    ax.plot(times, fit_traj[obs_idx], color="C0", lw=2.5, label="twin", zorder=1)
    ax.plot(times, ref_traj[obs_idx], color="0.2", ls=":", lw=2.5, label="experiment", zorder=2)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-1.05, 1.05)
    panel_rmse = float(np.sqrt(np.mean((fit_traj[obs_idx] - ref_traj[obs_idx]) ** 2)))
    ax.text(0.03, 0.06, rf"RMSE={panel_rmse:.1e}", transform=ax.transAxes, fontsize=8)
    ax.legend(loc="upper right", fontsize=8)

fig.suptitle("Twin reproduces the experimental fitting observables", y=1.05, fontsize=11)
fig.tight_layout()
```

## 3. Experimental data

When trajectories come from the lab (or an external simulator), pass them as `ref_expectations` with shape `(n_obs, n_times)` matching `observables` and `sim_params.times`. Below we reuse the reference trajectories from section 1 as a stand-in for measured data.

```{code-cell} ipython3
experimental_data = np.asarray(result.ref_traj, dtype=float)

lab_result = NoiseCharacterizer(show_progress=False).characterize(
    hamiltonian,
    sim_params,
    init_state=init_state,
    init_guess=init_guess,
    observables=fitting_observables,
    ref_expectations=experimental_data,
    x_low=rate_bounds_low,
    x_up=rate_bounds_high,
    sigma0=0.05,
    popsize=8,
    max_iter=40,
    seed=cma_seed,
)
print(f"lab-data fit RMSE: {lab_result.trajectory_rmse():.2e}")
```

## 4. Predict held-out observables with the twin

Plug `result.optimal_model` into {class}`~mqt.yaqs.Simulator` and compare to the hidden reference on observables **not** used during fitting.

```{code-cell} ipython3
pred_params = AnalogSimParams(
    observables=prediction_observables,
    elapsed_time=sim_params.elapsed_time,
    dt=sim_params.dt,
    order=sim_params.order,
    sample_timesteps=True,
)
simulator = Simulator(show_progress=False)

twin_result = simulator.run(init_state, hamiltonian, pred_params, result.optimal_model)
truth_result = simulator.run(init_state, hamiltonian, pred_params, reference_model)
twin_traj = np.asarray(twin_result.expectation_values, dtype=float)
truth_traj = np.asarray(truth_result.expectation_values, dtype=float)

fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
holdout_panels = [(0, r"$\langle X_0\rangle$"), (3, r"$\langle Z_2\rangle$")]
for ax, (obs_idx, ylabel) in zip(axes, holdout_panels, strict=True):
    ax.plot(times, twin_traj[obs_idx], color="C0", lw=2.5, label="twin", zorder=1)
    ax.plot(times, truth_traj[obs_idx], color="0.2", ls=":", lw=2.5, label="reference", zorder=2)
    ax.set_xlabel("time")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(loc="upper right", fontsize=8)

fig.suptitle("Twin predicts observables outside the fitting set", y=1.05, fontsize=11)
fig.tight_layout()
```

## 5. Stochastic experimental data (MCWF)

The same workflow works with trajectory-averaged MCWF data. Increase `num_traj` until observables stabilize; the objective becomes stochastic.

```{code-cell} ipython3
mcwf_sim_params = AnalogSimParams(
    observables=fitting_observables,
    elapsed_time=0.8,
    dt=0.1,
    order=1,
    num_traj=32,
    sample_timesteps=True,
)

mcwf_result = NoiseCharacterizer(show_progress=False, representation="vector").characterize(
    hamiltonian,
    mcwf_sim_params,
    init_state=init_state,
    init_guess=init_guess,
    observables=fitting_observables,
    reference_model=reference_model,
    x_low=rate_bounds_low,
    x_up=rate_bounds_high,
    sigma0=0.05,
    popsize=8,
    max_iter=20,
    seed=cma_seed,
)

fig, ax = plt.subplots(figsize=(4.5, 2.8))
obs_idx = 1
ax.plot(times, mcwf_result.fit_traj[obs_idx], color="C0", lw=2.5, label="MCWF twin", zorder=1)
ax.plot(times, mcwf_result.ref_traj[obs_idx], color="0.2", ls=":", lw=2.5, label="experiment", zorder=2)
ax.set_xlabel("time")
ax.set_ylabel(r"$\langle Z_0\rangle$")
ax.set_ylim(-1.05, 1.05)
ax.legend(loc="upper right", fontsize=8)
ax.set_title(f"MCWF fit: √J → {mcwf_result.sqrt_loss_after():.2e}")
fig.tight_layout()
```

## Workflow summary

| Step | Action                                                              |
| ---- | ------------------------------------------------------------------- |
| 1    | Collect experimental trajectories on a fitting observable set       |
| 2    | `NoiseCharacterizer.characterize(..., ref_expectations=...)`        |
| 3    | Compare learned rates and fitted-observable dynamics to reference   |
| 4    | `Simulator.run` with `result.optimal_model` on held-out observables |

For benchmarks, pass `reference_model=` instead of `ref_expectations=`. Library authors wiring custom optimizers can use `build_trajectory_loss()` and `cma_opt()` from `mqt.yaqs.characterization.noise.trajectory_matching`.

## See also

- {doc}`representation_comparison` — Lindblad vs MCWF vs TJM on the same benchmark
- {doc}`analog_simulation` — open-system simulation overview
- {doc}`characterization` — non-Markovian **memory** characterization (the memory twin submodule)
