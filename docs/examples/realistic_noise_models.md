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
2. **Gaussian (bell-curve) strengths** and the other built-in distributions.
3. How sampled disorder changes open-system dynamics compared to a mean-strength baseline.
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
print(f"{len(noise_model.processes)} processes:")
for proc in noise_model.processes:
    print(f"  {proc['name']:16s} sites={proc['sites']}  strength={proc['strength']}")
```

## 2. Bell-curve (Gaussian) disorder on strengths

Replace a scalar `strength` with a dict describing the distribution. For a **normal** (Gaussian) bell curve, set `distribution` to `"normal"` and provide `mean` and `std`:

```{code-cell} ipython3
bell_curve_strength = {"distribution": "normal", "mean": 0.08, "std": 0.02}

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

| `distribution`       | Parameters    | Use when                                                                                 |
| -------------------- | ------------- | ---------------------------------------------------------------------------------------- |
| `"normal"`           | `mean`, `std` | Symmetric spread around a target rate; negatives are clamped to `0`.                     |
| `"truncated_normal"` | `mean`, `std` | Same shape as normal but sampled only for non-negative strengths.                        |
| `"lognormal"`        | `mean`, `std` | Log-normal rates (strictly positive; `mean`/`std` are the underlying normal parameters). |

Sample many independent disorder realizations and plot the bell curve:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)
samples = [disordered_model.sample(rng=rng).processes[0]["strength"] for _ in range(5000)]

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.hist(samples, bins=40, density=True, alpha=0.7, color="tab:blue", label="sampled strengths")
x = np.linspace(0, max(samples), 200)
pdf = (
    1.0 / (bell_curve_strength["std"] * np.sqrt(2 * np.pi))
    * np.exp(-0.5 * ((x - bell_curve_strength["mean"]) / bell_curve_strength["std"]) ** 2)
)
ax.plot(x, pdf, color="black", lw=1.5, label="Gaussian pdf (negatives clamped)")
ax.set_xlabel("sampled dephasing strength")
ax.set_ylabel("density")
ax.set_title("Bell-curve disorder from a normal strength distribution")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

## 3. Disorder in an analog simulation

We evolve a short Ising chain and compare:

- **Baseline:** every site uses the distribution mean (`0.08`) as a fixed strength.
- **Disordered:** strengths are drawn from the bell curve once at the start of each run.

```{code-cell} ipython3
from mqt.yaqs import AnalogSimParams, Hamiltonian, Observable, Simulator, State

hamiltonian = Hamiltonian.ising(length=L, J=1.0, g=0.5)
state = State(L, initial="zeros")
z_obs = Observable("z", sites=0)

sim_params = AnalogSimParams(
    observables=[z_obs],
    elapsed_time=2.0,
    dt=0.1,
    num_traj=64,
    max_bond_dim=16,
    random_seed=7,
)

baseline_model = NoiseModel([
    {"name": "pauli_z", "sites": [i], "strength": bell_curve_strength["mean"]} for i in range(L)
])

sim = Simulator(show_progress=False)
result_baseline = sim.run(state, hamiltonian, sim_params, baseline_model)
result_disordered = sim.run(state, hamiltonian, sim_params, disordered_model)

print("Sampled strengths from the disordered run:")
for proc in result_disordered.noise_model.processes:
    print(f"  site {proc['sites'][0]}: {proc['strength']:.4f}")
```

```{code-cell} ipython3
---
mystnb:
  image:
    width: 80%
    align: center
---
times = sim_params.times
baseline_curve = result_baseline.expectation_values[0]
disordered_curve = result_disordered.expectation_values[0]

plt.figure(figsize=(7, 4))
plt.plot(times, baseline_curve, label="fixed mean strength", color="black", linestyle="--")
plt.plot(times, disordered_curve, label="bell-curve disorder (one sample / run)", color="tab:orange")
plt.xlabel("time")
plt.ylabel(r"$\langle Z_0 \rangle$")
plt.title("Static disorder shifts open-system decay")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

Re-running with the same `random_seed` reproduces the same sampled strengths and trajectory-averaged curve. Leave `random_seed=None` for fresh disorder draws in production Monte Carlo studies.

## 4. Disorder on a noisy circuit

The same distribution syntax works in digital simulation. Below, bit-flip rates on each qubit follow independent bell curves; one sample is drawn per `Simulator.run` call.

```{code-cell} ipython3
from mqt.yaqs import Observable, StrongSimParams
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit

num_qubits = 3
circuit = create_ising_circuit(L=num_qubits, J=1.0, g=0.5, dt=0.1, timesteps=5)

circuit_noise = NoiseModel([
    {
        "name": "pauli_x",
        "sites": [i],
        "strength": {"distribution": "truncated_normal", "mean": 0.05, "std": 0.02},
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
sampled = [proc["strength"] for proc in circuit_result.noise_model.processes]
print(f"truncated-normal bit-flip rates: {[f'{s:.4f}' for s in sampled]}")
print(f"final <Z_0>: {circuit_result.expectation_values[0][0]:.4f}")
```

## 5. Long-range crosstalk

Non-adjacent pairs use the `longrange_crosstalk_{ab}` naming convention; YAQS attaches per-site Pauli factors automatically:

```{code-cell} ipython3
lr_model = NoiseModel([
    {"name": "longrange_crosstalk_xy", "sites": [0, 2], "strength": 0.05},
])
sampled = lr_model.sample(rng=0)
print("sites:", sampled.processes[0]["sites"])
print("strength:", sampled.processes[0]["strength"])
print("factors:", [f.shape for f in sampled.processes[0]["factors"]])
```

## 6. Custom jump operators

Every noise process is a dictionary. Besides the built-in {class}`~mqt.yaqs.core.libraries.noise_library.NoiseLibrary` names (`lowering`, `pauli_x`, `crosstalk_xx`, …), you can supply your own operator as a NumPy array:

| Key | Required | Description |
| --- | --- | --- |
| `name` | yes | Label for the process. When `matrix` is omitted, must match a `NoiseLibrary` entry. When `matrix` is provided, any string is fine. |
| `sites` | yes | Site indices the jump acts on (one site for single-qubit channels). |
| `strength` | yes | Rate $\gamma$ in Lindblad form; YAQS uses jump operators $L_k = \sqrt{\gamma}\,L$. |
| `matrix` | no | Local operator $L$ as a `d×d` array (`d=2` for qubits). If omitted, YAQS looks up `name` in `NoiseLibrary`. |

YAQS does not check complete positivity; supply physically meaningful jump operators. The same `matrix` override works for **scheduled jumps** (see {doc}`scheduled_jumps`) and for all backends—TJM (`mps`), MCWF (`vector`), Lindblad (`density_matrix`), and noisy circuits.

### Amplitude damping with an explicit $\sigma_-$

The built-in `lowering` operator is $\sigma_- = |0\rangle\langle 1|$. You can pass the same matrix explicitly and mix custom and library processes in one model:

```{code-cell} ipython3
import numpy as np

sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)

custom_model = NoiseModel([
    {"name": "t1_explicit", "sites": [0], "strength": 0.1, "matrix": sigma_minus},
    {"name": "pauli_z", "sites": [1], "strength": 0.05},  # library lookup
])

for proc in custom_model.processes:
    print(f"{proc['name']:14s}  matrix shape={proc['matrix'].shape}")
```

Run a short analog simulation—the custom operator is used wherever `NoiseModel.processes` is consumed:

```{code-cell} ipython3
from mqt.yaqs import AnalogSimParams, Hamiltonian, Observable, Simulator, State

L = 2
hamiltonian = Hamiltonian.ising(length=L, J=1.0, g=0.5)
state = State(L, initial="one")  # |01>: site 0 can relax toward |00>

sim_params = AnalogSimParams(
    observables=[Observable("z", sites=0), Observable("z", sites=1)],
    elapsed_time=1.0,
    dt=0.1,
    num_traj=32,
    max_bond_dim=8,
    random_seed=3,
)

result = Simulator(show_progress=False).run(state, hamiltonian, sim_params, custom_model)
print(f"final <Z_0> = {result.expectation_values[0][-1]:.4f}")
print(f"final <Z_1> = {result.expectation_values[1][-1]:.4f}")
```

For $d>2$ local Hilbert spaces (e.g. transmon leakage), pass a `d×d` `matrix` matching the site's physical dimension—see {doc}`transmon_emulation`.

## Related topics

- {doc}`analog_simulation` — TJM workflow with static noise strengths
- {doc}`circuit_simulation` — noisy digital simulation
- {doc}`scheduled_jumps` — deterministic jumps at fixed times (library or custom `matrix`)
- {doc}`representation_comparison` — MCWF and Lindblad backends with the same `NoiseModel`
- {doc}`simulation_parameters` — presets and `random_seed` for reproducible trajectories
- {doc}`quickstart` — minimal first simulation
