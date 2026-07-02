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

# Quickstart

This page runs minimal workflows end-to-end: analog and digital simulation, equivalence checking, environmental memory (characterize from the response matrix, then train a surrogate to predict probe density matrices under a control sequence), and Markovian noise digital-twin fitting. Install the package first ({doc}`installation`), then copy the cells below.

Every example in this guide uses `show_progress=False` on `Simulator`, `MemoryCharacterizer`, and `NoiseCharacterizer` so tqdm progress bars do not clutter the documentation; figures below each cell show the main results.

## 1. Analog simulation

Néel-initialized transverse-field Ising chain with on-site damping. Staggered $\langle Z_i \rangle$ spreads and decays in a site-dependent way under open-system evolution:

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np

from mqt.yaqs import AnalogSimParams, Hamiltonian, NoiseModel, Observable, Simulator, State

L = 5
state = State(L, initial="Neel")
hamiltonian = Hamiltonian.ising(L, J=1.0, g=0.8)
noise_model = NoiseModel([
    {"name": "lowering", "sites": [i], "strength": 0.06} for i in range(L)
])

params = AnalogSimParams(
    observables=[Observable("z", site) for site in range(L)],
    elapsed_time=4.0,
    dt=0.1,
    num_traj=16,
    max_bond_dim=16,
    order=2,
    sample_timesteps=True,
)

sim = Simulator(show_progress=False)
result = sim.run(state, hamiltonian, params, noise_model)

heatmap = np.vstack([np.real(v) for v in result.expectation_values])
fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
im = ax.imshow(heatmap, aspect="auto", extent=(0, 4.0, L, 0), vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xlabel("time")
ax.set_yticks([x - 0.5 for x in range(1, L + 1)], [str(x) for x in range(L)])
ax.set_ylabel("site")
fig.colorbar(im, ax=ax, shrink=0.9, label=r"$\langle Z \rangle$")
ax.set_title("Staggered magnetization under damping")
```

## 2. Strong simulation

Evolve a short Trotterized Ising circuit and compare final $\langle Z_i\rangle$ without noise and with an optional {class}`~mqt.yaqs.NoiseModel`. See {doc}`strong_simulation` for noise sweeps, mid-circuit sampling, and gate modes.

```{code-cell} ipython3
from mqt.yaqs import NoiseModel, Observable, StrongSimParams
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit

num_qubits = 3
qc = create_ising_circuit(L=num_qubits, J=1.0, g=0.8, dt=0.1, timesteps=6)
circuit_state = State(num_qubits, initial="zeros")
circuit_params = StrongSimParams(
    observables=[Observable("z", site) for site in range(num_qubits)],
    preset="fast",
    num_traj=32,
)
noise_model = NoiseModel([
    {"name": "lowering", "sites": [site], "strength": 0.05} for site in range(num_qubits)
])

clean_result = sim.run(circuit_state, qc, circuit_params)
noisy_result = sim.run(State(num_qubits, initial="zeros"), qc, circuit_params, noise_model)
clean_z = np.array([float(np.real(v[0])) for v in clean_result.expectation_values])
noisy_z = np.array([float(np.real(v[0])) for v in noisy_result.expectation_values])

fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")
x = np.arange(num_qubits)
bar_width = 0.35
ax.bar(x - bar_width / 2, clean_z, bar_width, label="unitary", color="0.55")
ax.bar(x + bar_width / 2, noisy_z, bar_width, label="with damping", color="C0")
ax.set_xticks(x, [rf"$\langle Z_{i}\rangle$" for i in range(num_qubits)])
ax.set_ylim(-1.05, 1.05)
ax.set_ylabel("expectation value")
ax.set_title("Digital Ising circuit: optional open-system noise")
ax.legend(frameon=False)
```

## 3. Equivalence checking

Verify that a native GHZ circuit matches its transpiled decomposition (different gate basis, same unitary) with {class}`~mqt.yaqs.EquivalenceChecker`:

```{code-cell} ipython3
from qiskit import transpile
from qiskit.circuit import QuantumCircuit

from mqt.yaqs import EquivalenceChecker

ghz_native = QuantumCircuit(3)
ghz_native.h(0)
ghz_native.cx(0, 1)
ghz_native.cx(1, 2)

ghz_transpiled = transpile(
    ghz_native,
    basis_gates=["rz", "sx", "x", "cx"],
    optimization_level=1,
)

checker = EquivalenceChecker(representation="mpo", threshold=1e-6)
equiv = checker.check(ghz_native, ghz_transpiled)
print(f"equivalent: {equiv['equivalent']}")
print(f"fidelity: {equiv['fidelity']:.4e}")
print(f"center-cut operator entropy: {equiv['center_cut_entanglement_entropy']:.4f}")
print(f"global operator entropy: {equiv['global_entanglement_entropy']:.4f}")

fig, ax = plt.subplots(figsize=(4.5, 3))
ax.semilogy(equiv["schmidt_values"], "o-")
ax.set_xlabel("Schmidt index")
ax.set_ylabel("singular value")
ax.set_title("Composed operator $W = U_2^\\dagger U_1$")
fig.tight_layout()
```

For larger circuits, compiler passes, and OpenQASM inputs, see {doc}`equivalence_checking`.

## 4. Characterize environmental memory

Probe a probe qubit coupled to a short chain at an interior temporal cut. The memory spectrum and response matrix show how many independent past branches remain visible at the cut:

```{code-cell} ipython3
import numpy as np

from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer
from mqt.yaqs.characterization.memory.shared.utils import make_zero_psi

length = 4
ham = Hamiltonian.ising(length=length, J=1.0, g=1.0)
params = AnalogSimParams(dt=0.1, max_bond_dim=16, order=1)
mc = MemoryCharacterizer(show_progress=False)

cut, num_interventions = 4, 6
result = mc.characterize(
    ham,
    params,
    num_interventions=num_interventions,
    cut=cut,
    n_pasts=6,
    n_futures=6,
    initial_psi=make_zero_psi(length),
    rng=np.random.default_rng(0),
)
sv = result.singular_values(cut)
v = result.response_matrix(cut)

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].semilogy(sv, "o-")
axes[0].set_xlabel("mode index")
axes[0].set_ylabel("singular value")
axes[0].set_title(rf"Memory spectrum: $S_V(c={cut})={result.entropy(cut):.2f}$")

im = axes[1].imshow(np.abs(v), aspect="auto", cmap="viridis")
axes[1].set_title(rf"$|\widetilde{{V}}(c)|$, $R(c)={result.modes(cut):.1f}$")
axes[1].set_xlabel("future probe")
axes[1].set_ylabel("past probe")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
fig.tight_layout()
```

## 5. Fit a Markovian noise digital twin

Learn compact Lindblad jump rates from observable trajectories with {class}`~mqt.yaqs.noise_characterizer.NoiseCharacterizer`. Install the optional CMA-ES backend with `pip install mqt.yaqs[noise]`.

```{code-cell} ipython3
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*special injected samples.*")

from mqt.yaqs import AnalogSimParams, CompactNoiseModel, Hamiltonian, NoiseCharacterizer, Observable, State

n_sites = 3
sites = list(range(n_sites))
hamiltonian = Hamiltonian.ising(n_sites, J=1.0, g=2.0)
init_state = State(n_sites, initial="zeros")
fitting_observables = [Observable("y", 0), Observable("z", 0), Observable("y", 1)]
sim_params = AnalogSimParams(
    observables=fitting_observables,
    elapsed_time=0.8,
    dt=0.1,
    order=1,
    sample_timesteps=True,
)
reference_model = CompactNoiseModel([
    {"name": "pauli_x", "sites": sites, "strength": 0.08},
    {"name": "pauli_y", "sites": sites, "strength": 0.08},
    {"name": "pauli_z", "sites": sites, "strength": 0.08},
])
init_guess = CompactNoiseModel([
    {"name": "pauli_x", "sites": sites, "strength": 0.35},
    {"name": "pauli_y", "sites": sites, "strength": 0.35},
    {"name": "pauli_z", "sites": sites, "strength": 0.35},
])

result = NoiseCharacterizer(show_progress=False).characterize(
    hamiltonian,
    sim_params,
    init_state=init_state,
    init_guess=init_guess,
    observables=fitting_observables,
    reference_model=reference_model,
    x_low=np.zeros(3),
    x_up=np.full(3, 0.5),
    sigma0=0.05,
    popsize=8,
    max_iter=20,
    seed=42,
)

times = result.times
obs_labels = [r"$\langle Y_0\rangle$", r"$\langle Z_0\rangle$", r"$\langle Y_1\rangle$"]
fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), layout="constrained", sharey=True)
for ax, traj, title in zip(axes, [result.fit_traj, result.ref_traj], ["learned twin", "reference"], strict=True):
    im = ax.imshow(
        traj,
        aspect="auto",
        extent=(times[0], times[-1], len(obs_labels), 0),
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
    )
    ax.set_yticks([i + 0.5 for i in range(len(obs_labels))], obs_labels)
    ax.set_xlabel("time")
    ax.set_title(title)
fig.colorbar(im, ax=axes, shrink=0.9, label="expectation")
fig.suptitle(rf"Twin fit: RMSE={result.trajectory_rmse():.2e}", y=1.02)
```

See {doc}`noise_characterization` for experimental-data workflows, held-out prediction, and MCWF fitting.

## 6. Train a surrogate and predict under controls

Train a causal surrogate with {class}`~mqt.yaqs.memory_characterizer.MemoryCharacterizer`, then predict the probe-qubit state after one or more control legs.
Pass an explicit per-leg list to compare different sequences on the same trained model.
Surrogate training requires PyTorch (`pip install mqt.yaqs[torch]`).

```{code-cell} ipython3
rho0 = np.eye(2, dtype=np.complex128) / 2.0
ham_sure = Hamiltonian.ising(length=2, J=1.0, g=1.0)

model = mc.train(
    ham_sure,
    params,
    num_interventions=1,
    n=32,
    train_kwargs={"epochs": 30, "batch_size": 8},
    model_kwargs={"d_model": 32, "nhead": 4, "num_layers": 1, "dim_ff": 64},
)

hadamard = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
control_sequences = {
    r"$\mathrm{H}$": [{"unitary": hadamard}],
    r"$\mathrm{X}$": [{"unitary": pauli_x}],
}

pauli_ops = {
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}
expectations = {
    label: [
        float(np.trace(op @ mc.predict(model, rho0, controls, num_interventions=1)).real)
        for op in pauli_ops.values()
    ]
    for label, controls in control_sequences.items()
}

pauli_names = list(pauli_ops)
x = np.arange(len(pauli_names))
width = 0.35

fig, ax = plt.subplots(figsize=(5, 3.5))
for offset, (label, values) in zip((-width / 2, width / 2), expectations.items()):
    ax.bar(x + offset, values, width, label=f"control {label}")
ax.set_xticks(x, pauli_names)
ax.set_ylabel("expectation value")
ax.set_title("Probe Pauli expectations for two control sequences")
ax.legend(frameon=False)
fig.tight_layout()
```

`predict` also accepts a style string (for example `"haar"`) or a per-leg list mixing unitaries and measure–prepare slots. See {doc}`characterization` for environmental memory probing and {doc}`memory_surrogate` for held-out accuracy checks and exact-reference validation.

## 7. Where to go next

| Goal                                                 | Start here                    |
| ---------------------------------------------------- | ----------------------------- |
| Environmental memory probing                         | {doc}`characterization`       |
| Markovian noise digital-twin fitting                 | {doc}`noise_characterization` |
| Surrogate training, prediction, and exact validation | {doc}`memory_surrogate`       |
| Open-system dynamics, noise, time grids              | {doc}`analog_simulation`      |
| Bell-curve (log-normal) noise strengths              | {doc}`realistic_noise_models` |
| Strong simulation, mid-circuit sampling, OpenQASM    | {doc}`strong_simulation`      |
| Accuracy presets and truncation knobs                | {doc}`simulation_parameters`  |
| Check two circuits for equivalence                   | {doc}`equivalence_checking`   |

## Related topics

- {doc}`state_initialization` — `State` presets and representations
- {doc}`simulator_initialization` — parallelism, progress bars, `Result` fields
- {doc}`representation_comparison` — when to use MPS, MCWF, or Lindblad backends
- {doc}`equivalence_checking` — MPO backend, transpiler regression tests, OpenQASM
