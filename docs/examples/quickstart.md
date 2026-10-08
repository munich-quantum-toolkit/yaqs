---
file_format: mystnb
kernelspec:
  name: python3
language_info:
  name: python
mystnb:
  number_source_lines: true
  execution_timeout: 180
---

# Quickstart

Explore the main YAQS workflows with executable examples and their results.
After {doc}`installing YAQS <../installation>`, run the cells in a notebook. For
a standalone script, put the execution code inside an
`if __name__ == "__main__":` guard; see {doc}`simulator_initialization`.

The examples use `show_progress=False` to keep the documentation quiet. Omit
this argument to see progress. Times and rates use units consistent with each
Hamiltonian, with $\hbar=1$. Expand the plotting cells to reuse the figures.

```{code-cell} python
:tags: [hide-input]
import matplotlib.pyplot as plt
import numpy as np
from matplotlib_inline.backend_inline import set_matplotlib_formats

set_matplotlib_formats("svg")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "lines.linewidth": 1.6,
    "lines.markersize": 4,
    "figure.figsize": (6.6, 3.0),
    "figure.constrained_layout.use": True,
    "savefig.dpi": 180,
})
```

## Large-scale analog dynamics

Follow one excitation as it spreads through a **50-site XY spin chain**. YAQS
uses an MPS rather than storing all $2^{50}$ state amplitudes.

```{code-cell} python
from mqt.yaqs import AnalogSimParams, Hamiltonian, Observable, Simulator, State

length = 50
center = length // 2
basis = "0" * center + "1" + "0" * (length - center - 1)
state = State(length, initial="basis", basis_string=basis)
hamiltonian = Hamiltonian.heisenberg(length, Jx=0.5, Jy=0.5, Jz=0.0)
params = AnalogSimParams(
    observables=[Observable("z", site) for site in range(length)],
    elapsed_time=6.0,
    dt=0.2,
    preset="fast",
)

simulator = Simulator(show_progress=False)
analog = simulator.run(state, hamiltonian, params)
```

```{code-cell} python
:tags: [hide-input]
from matplotlib.colors import PowerNorm

occupation = (1 - np.asarray(analog.expectation_values).real) / 2
sites = np.arange(length)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), width_ratios=[1.2, 1])
image = axes[0].pcolormesh(
    analog.times, sites, occupation, shading="auto", cmap="cividis",
    norm=PowerNorm(0.5, vmin=0, vmax=1), rasterized=True,
)
fig.colorbar(image, ax=axes[0], label=r"$\langle n_i\rangle$", ticks=[0, 0.1, 0.5, 1])
axes[0].set(xlabel=r"Time $t$", ylabel=r"Site $i$", xlim=(0, 6), ylim=(-0.5, length - 0.5))
axes[0].set_title("(a) Excitation transport", loc="left", fontsize=11)
for time, color, marker in zip((2, 4, 6), ("#0072B2", "#D55E00", "#009E73"), ("o", "s", "^"), strict=True):
    index = np.argmin(np.abs(analog.times - time))
    axes[1].plot(sites, occupation[:, index], color=color, marker=marker, markevery=3, label=rf"$t={time}$")
axes[1].set(xlabel=r"Site $i$", ylabel=r"Occupation $\langle n_i\rangle$", xlim=(0, length - 1), ylim=(0, 0.22))
axes[1].set_title("(b) Spatial profiles", loc="left", fontsize=11)
axes[1].legend()
plt.show()
```

The occupation $n_i=(1-Z_i)/2$ shows propagation and interference. The color
scale emphasizes small occupations; the total excitation remains one. This
low-excitation example stays inexpensive because its entanglement is limited.
For noise, larger trajectory budgets, and convergence checks, see
{doc}`analog_simulation` and {doc}`simulation_parameters`.

## Noisy circuit readout

Prepare an eight-qubit GHZ circuit and compare ideal and damped readout with 256
shots per run.

```{code-cell} python
from qiskit import QuantumCircuit

from mqt.yaqs import DigitalSimParams, NoiseModel, Simulator, State

num_qubits = 8
circuit = QuantumCircuit(num_qubits)
circuit.h(0)
for site in range(num_qubits - 1):
    circuit.cx(site, site + 1)
circuit.measure_all()

state = State(num_qubits, initial="zeros")
params = DigitalSimParams(shots=256, preset="fast", random_seed=7)
noise = NoiseModel([
    {"name": "lowering", "sites": [site], "strength": 0.3} for site in range(num_qubits)
])

simulator = Simulator(show_progress=False)
ideal = simulator.run(state, circuit, params)
damped = simulator.run(state, circuit, params, noise)
```

```{code-cell} python
:tags: [hide-input]
excitation_number = np.arange(num_qubits + 1)
fig, ax = plt.subplots(figsize=(5.4, 2.9))
for result, offset, color, label in (
    (ideal, -0.18, "#0072B2", "Ideal"),
    (damped, 0.18, "#D55E00", "Damped"),
):
    probability = np.zeros(num_qubits + 1)
    for outcome, count in result.counts.items():
        probability[outcome.bit_count()] += count / params.shots
    ax.bar(excitation_number + offset, probability, width=0.36, color=color,
           edgecolor="white", linewidth=0.6, label=label)
ax.set(xlabel="Number of excited qubits", ylabel="Measured probability", xticks=excitation_number, ylim=(0, 0.7))
ax.legend()
plt.show()
```

The ideal populations lie at zero and eight excitations. Damping shifts weight
toward lower excitation numbers. This histogram summarizes readout populations;
see {doc}`circuit_shots` for bitstring counts and {doc}`circuit_observables` for
expectation values and OpenQASM input.

## Circuit equivalence

Verify a transpiled circuit, then measure how an extra $Z$ rotation changes its
agreement with the original.

```{code-cell} python
import numpy as np
from qiskit import QuantumCircuit, transpile

from mqt.yaqs import EquivalenceChecker

circuit = QuantumCircuit(4)
circuit.h(0)
for site in range(3):
    circuit.cx(site, site + 1)
decomposed = transpile(circuit, basis_gates=["rz", "sx", "x", "cx"])

checker = EquivalenceChecker()
print("Equivalent:", checker.check(circuit, decomposed)["equivalent"])
angles = np.linspace(0, np.pi, 17)
overlaps = []
for angle in angles:
    perturbed = decomposed.copy()
    perturbed.rz(float(angle), 0)
    overlaps.append(checker.check(circuit, perturbed)["fidelity"])
```

```{code-cell} python
:tags: [hide-input]
fig, ax = plt.subplots(figsize=(4.6, 2.9))
ax.plot(angles, overlaps, "o", color="#0072B2", label="YAQS")
ax.plot(angles, np.abs(np.cos(angles / 2)), "--", color="0.3", label=r"Analytic $|\cos(\theta/2)|$")
ax.set(xlabel=r"Added rotation $\theta$ (rad)", ylabel="Normalized operator overlap",
       xlim=(0, np.pi), ylim=(0, 1.05), xticks=[0, np.pi / 2, np.pi],
       xticklabels=["0", r"$\pi/2$", r"$\pi$"])
ax.legend()
plt.show()
```

An overlap of one indicates agreement up to a global phase. The added rotation
produces a controlled difference with a known analytic overlap. See
{doc}`equivalence_checking` for noise and accuracy controls.

## Environmental memory

Compare the response modes of a qubit with and without coupling to a two-spin
environment, using the same probe grid.

```{code-cell} python
import numpy as np

from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer

params = AnalogSimParams(elapsed_time=0.5, dt=0.5, preset="fast")
characterizer = MemoryCharacterizer(show_progress=False)
memories = []
for coupling in (0.0, 1.0):
    hamiltonian = Hamiltonian.ising(3, J=coupling, g=1.0)
    memories.append(characterizer.characterize(
        hamiltonian, params, num_interventions=4, cut=2, preset="quick",
        rng=np.random.default_rng(7),
    ))
```

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0))
for memory, color, marker, label in zip(
    memories, ("#D55E00", "#0072B2"), ("s", "o"), ("Uncoupled", "Coupled"), strict=True,
):
    spectrum = memory.singular_values(2)
    weights = spectrum**2 / np.sum(spectrum**2)
    axes[0].semilogy(np.arange(1, len(weights) + 1), weights, marker=marker, color=color, label=label)
axes[0].set(xlabel="Mode index", ylabel=r"Resolved mode weight $p_k$", ylim=(1e-10, 2))
axes[0].set_title("(a) Memory spectrum", loc="left", fontsize=11)
axes[0].legend()
image = axes[1].imshow(np.abs(memories[1].response_matrix(2)), aspect="auto", cmap="cividis", origin="lower")
axes[1].set(xlabel="Past probe index", ylabel="Future response row")
axes[1].set_title("(b) Coupled response", loc="left", fontsize=11)
fig.colorbar(image, ax=axes[1], label=r"$|V_{\mu j}|$")
plt.show()
```

The weights $p_k=s_k^2/\sum_j s_j^2$ describe memory resolved by the sampled
probes. Without coupling, this example has one resolved mode; coupling reveals
additional modes. These weights are not the environment's state populations. See
{doc}`characterization` for probe choices and interpretation.

## Noise characterization

Learn a dephasing rate from synthetic dynamics and compare the fitted
trajectories with the reference.

```{code-cell} python
import numpy as np

from mqt.yaqs import AnalogSimParams, Hamiltonian, NoiseCharacterizer, NoiseModel, Observable, State

hamiltonian = Hamiltonian.ising(2, J=1.0, g=1.0)
observables = [Observable("z", 0), Observable("y", 0)]
params = AnalogSimParams(observables=observables, elapsed_time=3.0, dt=0.1, preset="fast")
reference = NoiseModel([{"name": "pauli_z", "sites": [0], "strength": 0.18}])
guess = NoiseModel([{"name": "pauli_z", "sites": [0], "strength": 0.6}])

characterizer = NoiseCharacterizer(show_progress=False)
fit = characterizer.characterize(
    hamiltonian,
    params,
    init_state=State(2, initial="zeros"),
    init_guess=guess,
    observables=observables,
    reference_model=reference,
    x_low=np.array([0.0]),
    x_up=np.array([1.0]),
    max_iter=20,
)
print("Fitted rate:", fit.best_parameters)
```

```{code-cell} python
:tags: [hide-input]
fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), width_ratios=[1.5, 1])
for index, (color, label) in enumerate((("#0072B2", r"$\langle Z_0\rangle$"), ("#D55E00", r"$\langle Y_0\rangle$"))):
    axes[0].plot(fit.times, fit.fit_traj[index], color=color, label=label)
    axes[0].plot(fit.times[::2], fit.ref_traj[index, ::2], "o", color=color, markerfacecolor="white")
axes[0].plot([], [], "o", color="0.3", markerfacecolor="white", label="Reference")
axes[0].set(xlabel=r"Time $t$", ylabel="Expectation value", ylim=(-1.05, 1.05))
axes[0].set_title("(a) Fitted dynamics", loc="left", fontsize=11)
axes[0].legend()
axes[1].bar([0, 1], [0.6, fit.best_parameters[0]], color=["#D55E00", "#0072B2"], width=0.55)
axes[1].axhline(0.18, color="0.3", linestyle="--", linewidth=1.1, label="Reference")
axes[1].set(xticks=[0, 1], xticklabels=["Initial guess", "Fit"], ylabel=r"Dephasing rate $\gamma$", ylim=(0, 0.7))
axes[1].set_title("(b) Recovered rate", loc="left", fontsize=11)
axes[1].legend()
plt.show()
```

For measured data, supply `ref_expectations` instead of `reference_model`. See
{doc}`digital_twin` for data preparation and validation of the fitted model.

## Surrogate prediction

Train a model on control sequences, then compare its predictions on
**new sequences** with Hamiltonian calculations. Install the `torch` extra
first: `uv pip install "mqt.yaqs[torch]"`.

```{code-cell} python
import torch

from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer

torch.manual_seed(7)
hamiltonian = Hamiltonian.ising(3, J=1.0, g=0.7)
params = AnalogSimParams(elapsed_time=0.2, dt=0.2, preset="fast")
characterizer = MemoryCharacterizer(show_progress=False)
model = characterizer.train(
    hamiltonian, params, num_interventions=2, n=160, seed=7,
    intervention_style="measure_prepare",
)
held_out = characterizer.sample(
    hamiltonian, params, num_interventions=2, n=48, seed=99,
    intervention_style="measure_prepare",
)
features, rho0, target = held_out.tensors
prediction = model.predict(features.numpy(), rho0.numpy(), return_numpy=True)
```

```{code-cell} python
:tags: [hide-input]
# Packed entries 0 and 6 are the two diagonal populations.
z_reference = target.numpy()[:, -1, 0] - target.numpy()[:, -1, 6]
z_prediction = prediction[:, -1, 0] - prediction[:, -1, 6]
rmse = np.sqrt(np.mean((z_prediction - z_reference)**2))
fig, ax = plt.subplots(figsize=(3.8, 3.5))
ax.plot([-1, 1], [-1, 1], "--", color="0.3", linewidth=1.0, label="Exact agreement")
ax.scatter(z_reference, z_prediction, s=25, color="#0072B2", alpha=0.8, edgecolor="white", linewidth=0.4)
ax.text(0.06, 0.94, f"RMSE = {rmse:.3f}", transform=ax.transAxes, va="top")
ax.set(xlabel=r"Hamiltonian reference $\langle Z\rangle$", ylabel=r"Surrogate prediction $\langle Z\rangle$",
       xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), xticks=[-1, 0, 1], yticks=[-1, 0, 1])
ax.set_aspect("equal")
plt.show()
```

The scatter tests observable prediction on 48 held-out sequences. It does not
certify every predicted density matrix or other control settings. See
{doc}`memory_surrogate` for broader accuracy checks and prediction through
{meth}`~mqt.yaqs.MemoryCharacterizer.predict`.

## Next steps

| Task                                   | Guide                            |
| -------------------------------------- | -------------------------------- |
| Choose accuracy settings               | {doc}`simulation_parameters`     |
| Choose states and simulation backends  | {doc}`state_initialization`      |
| Combine analog evolution and circuits  | {doc}`digital_analog_simulation` |
| Check whether two circuits agree       | {doc}`equivalence_checking`      |
| Learn noise models from dynamics       | {doc}`digital_twin`              |
| Study memory in a system's environment | {doc}`characterization`          |
| Train models for dynamics with memory  | {doc}`memory_surrogate`          |
