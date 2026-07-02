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

# Realistic Noise Models

YAQS ships a library of physically motivated jump operators—relaxation (`lowering`), excitation (`raising`), single-qubit Pauli channels, and nearest-neighbor crosstalk (`crosstalk_xx`, `crosstalk_zz`, …)—that you assemble into a {class}`~mqt.yaqs.core.data_structures.noise_model.NoiseModel`.

For hardware with **static disorder** (calibration drift, fabrication spread), each process strength can be a **distribution** instead of a fixed float. YAQS samples one concrete strength per process when {meth}`~mqt.yaqs.Simulator.run` starts; all trajectories in that run share the same sampled disorder. The realized model is stored on {attr}`~mqt.yaqs.Result.noise_model`.

This page shows:

1. A typical multi-channel noise model for an analog chain.
2. **Log-normal disorder on strengths** (recommended when rates span orders of magnitude) and other built-in distributions.
3. How sampled disorder changes open-system dynamics compared to a median-strength baseline.
4. **Custom jump operators** via an explicit `matrix` (not only built-in library names).

## 1. Built-in noise processes

Each process is a dictionary with `name`, `sites`, and `strength`. YAQS fills in the operator `matrix` (or per-site `factors` for long-range crosstalk) from {class}`~mqt.yaqs.core.libraries.noise_library.NoiseLibrary`.

```{code-cell} ipython3
from mqt.yaqs import NoiseModel

L = 4
processes = [
    {"name": "lowering", "sites": [i], "strength": 0.05} for i in range(L)
] + [
    {"name": "pauli_z", "sites": [i], "strength": 0.02} for i in range(L)
] + [
    {"name": "crosstalk_xx", "sites": [i, i + 1], "strength": 0.01} for i in range(L - 1)
]

noise_model = NoiseModel(processes)
```

## 2. Log-normal disorder on strengths

When calibration rates vary across devices or qubits, strengths often span **several orders of magnitude**. A **log-normal** distribution is usually more realistic than a symmetric Gaussian on the rate itself.

Replace a scalar `strength` with a dict. For log-normal sampling, `mean` and `std` are the parameters of the underlying normal distribution on $\log\gamma$:

```{code-cell} ipython3
bell_curve_strength = {"distribution": "lognormal", "mean": -2.3, "std": 0.5}

disordered_processes = [
    {
        "name": "pauli_z",
        "sites": [i],
        "strength": bell_curve_strength,
    }
    for i in range(L)
]

disordered_model = NoiseModel(disordered_processes)
```

Other supported distributions:

| `distribution`       | Parameters    | Use when                                                                                  |
| -------------------- | ------------- | ----------------------------------------------------------------------------------------- |
| `"lognormal"`        | `mean`, `std` | **Default choice** for positive rates spanning magnitudes (`mean`/`std` on $\log\gamma$). |
| `"normal"`           | `mean`, `std` | Symmetric spread around a target rate; negatives are clamped to `0`.                      |
| `"truncated_normal"` | `mean`, `std` | Same shape as normal but sampled only for non-negative strengths.                         |

Sample many independent disorder realizations and plot the bell curve on a log scale:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
samples = [disordered_model.sample(rng=rng).processes[0]["strength"] for _ in range(5000)]

mu = bell_curve_strength["mean"]
sigma = bell_curve_strength["std"]
x = np.logspace(np.log10(min(samples)), np.log10(max(samples)), 200)
pdf = stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu))

fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
ax.hist(samples, bins=40, density=True, alpha=0.7, color="tab:blue", label="sampled strengths")
ax.plot(x, pdf, color="black", lw=1.5, label="log-normal pdf")
ax.set_xscale("log")

# Sparse decade ticks with plain decimal labels (avoids crowded sci-notation on log axes)
lo, hi = float(min(samples)), float(max(samples))
tick_decades = np.arange(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
tick_candidates = np.concatenate([np.array([1, 2, 5]) * 10.0**e for e in tick_decades])
ticks = tick_candidates[(tick_candidates >= lo * 0.9) & (tick_candidates <= hi * 1.1)]
if len(ticks) > 6:
    ticks = ticks[np.linspace(0, len(ticks) - 1, 6, dtype=int)]
ax.set_xticks(ticks)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_locator(mticker.NullLocator())

ax.set_xlabel("sampled dephasing strength")
ax.set_ylabel("density")
ax.set_title("Log-normal disorder (median ≈ {:.3f})".format(np.exp(mu)))
ax.legend()
ax.grid(alpha=0.3, which="both")
plt.show()
```

## 3. Disorder in an analog simulation

We evolve a short Ising chain from a Néel product state and compare:

- **Baseline:** every site uses the log-normal **median** $\exp(\text{mean})$ as a fixed strength.
- **Disordered:** strengths are drawn from the log-normal once at the start of each run.
- **Ensemble band:** several independent disorder draws (different `random_seed`) to show typical spread.

```{code-cell} ipython3
from mqt.yaqs import AnalogSimParams, Hamiltonian, Observable, Simulator, State

# Wider log-normal spread for a visible disorder effect in dynamics
dyn_strength = {"distribution": "lognormal", "mean": -0.7, "std": 1.0}
dyn_disordered = NoiseModel([
    {"name": "pauli_z", "sites": [i], "strength": dyn_strength} for i in range(L)
])

hamiltonian = Hamiltonian.ising(length=L, J=1.0, g=0.5)
state = State(L, initial="Neel")
z_obs = Observable("z", sites=0)

sim_params = AnalogSimParams(
    observables=[z_obs],
    elapsed_time=8.0,
    dt=0.1,
    num_traj=32,
    max_bond_dim=24,
    random_seed=7,
)

median_strength = float(np.exp(dyn_strength["mean"]))
baseline_model = NoiseModel([
    {"name": "pauli_z", "sites": [i], "strength": median_strength} for i in range(L)
])

sim = Simulator(show_progress=False)
result_baseline = sim.run(state, hamiltonian, sim_params, baseline_model)
result_disordered = sim.run(state, hamiltonian, sim_params, dyn_disordered)

# Ensemble of disorder realizations for a shaded band (keep small for doc build time)
ensemble_curves = []
for seed in range(8, 12):
    params_i = AnalogSimParams(
        observables=[z_obs],
        elapsed_time=8.0,
        dt=0.1,
        num_traj=16,
        max_bond_dim=24,
        random_seed=seed,
    )
    res_i = sim.run(state, hamiltonian, params_i, dyn_disordered)
    ensemble_curves.append(res_i.expectation_values[0])
ensemble_curves = np.asarray(ensemble_curves)
```

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

times = sim_params.times
baseline_curve = result_baseline.expectation_values[0]
disordered_curve = result_disordered.expectation_values[0]

fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
ax.fill_between(
    times,
    ensemble_curves.min(axis=0),
    ensemble_curves.max(axis=0),
    color="tab:orange",
    alpha=0.25,
    label="disordered ensemble (4 seeds)",
)
ax.plot(times, baseline_curve, label="fixed median strength", color="black", linestyle="--", lw=2)
ax.plot(times, disordered_curve, label="one disordered sample", color="tab:orange", lw=1.5)
ax.set_xlabel("time")
ax.set_ylabel(r"$\langle Z_0 \rangle$")
ax.set_title("Log-normal static disorder shifts open-system decay")
ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
ax.legend()
ax.grid(alpha=0.3)
plt.show()
```

Re-running with the same `random_seed` reproduces the same sampled strengths and trajectory-averaged curve. Leave `random_seed=None` for fresh disorder draws in production Monte Carlo studies.

## 4. Disorder on a noisy circuit

The same distribution syntax works in digital simulation. Below, bit-flip rates on each qubit follow independent log-normal draws; one sample is drawn per `Simulator.run` call.

```{code-cell} ipython3
from mqt.yaqs import Observable, StrongSimParams
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit

num_qubits = 3
circuit = create_ising_circuit(L=num_qubits, J=1.0, g=0.5, dt=0.1, timesteps=5)

circuit_noise = NoiseModel([
    {
        "name": "pauli_x",
        "sites": [i],
        "strength": {"distribution": "lognormal", "mean": -3.0, "std": 0.4},
    }
    for i in range(num_qubits)
])

circuit_params = StrongSimParams(
    observables=[Observable("z", site) for site in range(num_qubits)],
    num_traj=32,
    max_bond_dim=8,
    random_seed=11,
)

circuit_result = sim.run(State(num_qubits, initial="zeros"), circuit, circuit_params, circuit_noise)
```

## 5. Long-range crosstalk

Non-adjacent pairs use the `longrange_crosstalk_{ab}` naming convention; YAQS attaches per-site Pauli factors automatically:

```{code-cell} ipython3
lr_model = NoiseModel([
    {"name": "longrange_crosstalk_xy", "sites": [0, 2], "strength": 0.05},
])
sampled = lr_model.sample(rng=0)
```

## 6. Custom jump operators

Every noise process is a dictionary. Besides the built-in {class}`~mqt.yaqs.core.libraries.noise_library.NoiseLibrary` names (`lowering`, `pauli_x`, `crosstalk_xx`, …), you can supply your own operator as a NumPy array:

| Key        | Required | Description                                                                                                                        |
| ---------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `name`     | yes      | Label for the process. When `matrix` is omitted, must match a `NoiseLibrary` entry. When `matrix` is provided, any string is fine. |
| `sites`    | yes      | Site indices the jump acts on (one site for single-qubit channels).                                                                |
| `strength` | yes      | Rate $\gamma$ in Lindblad form; YAQS uses jump operators $L_k = \sqrt{\gamma}\,L$.                                                 |
| `matrix`   | no       | Local operator $L$ as a `d×d` array (`d=2` for qubits). If omitted, YAQS looks up `name` in `NoiseLibrary`.                        |

YAQS does not check complete positivity; supply physically meaningful jump operators. The same `matrix` override works for **scheduled jumps** (see {doc}`scheduled_jumps`) and for all backends—TJM (`mps`), MCWF (`vector`), Lindblad (`density_matrix`), and noisy circuits.

### Amplitude damping with an explicit $\sigma_-$

The built-in `lowering` operator is $\sigma_- = |0\rangle\langle 1|$. You can pass the same matrix explicitly and mix custom and library processes in one model:

```{code-cell} ipython3
import numpy as np

sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)

custom_model = NoiseModel([
    {"name": "t1_explicit", "sites": [0], "strength": 0.1, "matrix": sigma_minus},
    {"name": "pauli_z", "sites": [1], "strength": 0.05},
])
```

Run a short analog simulation—the custom operator is used wherever `NoiseModel.processes` is consumed:

```{code-cell} ipython3
from mqt.yaqs import AnalogSimParams, Hamiltonian, Observable, Simulator, State

L2 = 2
hamiltonian = Hamiltonian.ising(length=L2, J=1.0, g=0.5)
state = State(L2, initial="basis", basis_string="10")

sim_params = AnalogSimParams(
    observables=[Observable("z", sites=0), Observable("z", sites=1)],
    elapsed_time=1.0,
    dt=0.1,
    num_traj=32,
    max_bond_dim=8,
    random_seed=3,
)

result = Simulator(show_progress=False).run(state, hamiltonian, sim_params, custom_model)
```

For $d>2$ local Hilbert spaces (e.g. transmon leakage), pass a `d×d` `matrix` matching the site's physical dimension—see {doc}`transmon_emulation`.

## Related topics

- {doc}`analog_simulation` — TJM workflow with static noise strengths
- {doc}`strong_simulation` — strong digital simulation
- {doc}`scheduled_jumps` — deterministic jumps at fixed times (library or custom `matrix`)
- {doc}`representation_comparison` — MCWF and Lindblad backends with the same `NoiseModel`
- {doc}`simulation_parameters` — presets and `random_seed` for reproducible trajectories
- {doc}`quickstart` — minimal first simulation
