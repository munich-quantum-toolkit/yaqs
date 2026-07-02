---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
  execution_timeout: 600
---

```{code-cell} ipython3
:tags: [remove-cell]
%config InlineBackend.figure_formats = ['svg']
```

# Quickstart

This page runs minimal workflows end-to-end: analog and digital simulation, equivalence checking, environmental memory (characterize from the response matrix, then train a surrogate to predict probe density matrices under a control sequence), and Markovian noise digital-twin fitting. Install the package first ({doc}`installation`), then copy the cells below.

Every example in this guide uses `show_progress=False` on `Simulator`, `MemoryCharacterizer`, and `NoiseCharacterizer` so tqdm progress bars do not clutter the documentation; figures below each cell show the main results.

## 1. Analog simulation

Tensor-network time evolution on an MPS. Attach a `NoiseModel` for open-system TJM trajectories; without noise, the simulator performs a single unitary trajectory regardless of `num_traj`:

```{code-cell} ipython3
import matplotlib.pyplot as plt

from mqt.yaqs import AnalogSimParams, Hamiltonian, Observable, Simulator, State

L = 3
state = State(L, initial="zeros")
hamiltonian = Hamiltonian.ising(L, J=1.0, g=0.5)

params = AnalogSimParams(
    observables=[Observable("z", sites=0)],
    elapsed_time=0.5,
    dt=0.1,
    preset="fast",
    num_traj=8,
)

sim = Simulator(show_progress=False)
result = sim.run(state, hamiltonian, params)

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(result.times, result.expectation_values[0].real, "o-")
ax.set_xlabel("time")
ax.set_ylabel(r"$\langle Z_0 \rangle$")
ax.set_title("Single-site magnetization")
fig.tight_layout()
```

## 2. Digital circuit simulation

Noisy or noise-free evolution through a Qiskit circuit on an MPS:

```{code-cell} ipython3
from qiskit.circuit import QuantumCircuit

from mqt.yaqs import Observable, StrongSimParams

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)

circuit_state = State(3, initial="zeros")
circuit_params = StrongSimParams(
    observables=[Observable("z", sites=i) for i in range(3)],
    preset="fast",
    num_traj=8,
)

circuit_result = sim.run(circuit_state, qc, circuit_params)
```

## 3. Equivalence checking

Verify that two circuits implement the same unitary (up to global phase) with {class}`~mqt.yaqs.EquivalenceChecker`:

```{code-cell} ipython3
from mqt.yaqs import EquivalenceChecker
from qiskit.circuit import QuantumCircuit

def bell_prep() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc

# Original vs. transpiled-style rewrite (same unitary on two qubits)
original = bell_prep()
rewritten = QuantumCircuit(2)
rewritten.h(0)
rewritten.cx(0, 1)

checker = EquivalenceChecker(representation="mpo", threshold=1e-6)
equiv = checker.check(original, rewritten)
```

For larger circuits, compiler passes, and OpenQASM inputs, see {doc}`equivalence_checking`.

## 4. Characterize environmental memory

Probe a probe qubit coupled to a short chain at an interior temporal cut and inspect the memory mode spectrum:

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
fig, ax = plt.subplots(figsize=(5, 3))
ax.semilogy(sv, "o-")
ax.set_xlabel("mode index")
ax.set_ylabel("singular value")
ax.set_title(rf"$S_V(c={cut})={result.entropy(cut):.3f}$, $R(c)={result.modes(cut):.2f}$")
fig.tight_layout()
```

## 5. Fit a Markovian noise digital twin

Learn compact Lindblad jump rates from observable trajectories with {class}`~mqt.yaqs.noise_characterizer.NoiseCharacterizer`. Install the optional CMA-ES backend with `pip install mqt.yaqs[noise]`.

```{code-cell} ipython3
import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*special injected samples.*")

from mqt.yaqs import AnalogSimParams, CompactNoiseModel, Hamiltonian, NoiseCharacterizer, Observable, State
from mqt.yaqs.core.libraries.gate_library import Y, Z

n_sites = 3
sites = list(range(n_sites))
hamiltonian = Hamiltonian.ising(n_sites, J=1.0, g=2.0)
init_state = State(n_sites, initial="zeros")
fitting_observables = [Observable(Y(), 0), Observable(Z(), 0), Observable(Y(), 1)]
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
print(f"√J after fit: {result.sqrt_loss_after():.2e}")
print(f"trajectory RMSE: {result.trajectory_rmse():.2e}")
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
| Circuit observables, mid-circuit sampling, OpenQASM  | {doc}`circuit_simulation`     |
| Accuracy presets and truncation knobs                | {doc}`simulation_parameters`  |
| Check two circuits for equivalence                   | {doc}`equivalence_checking`   |

## Related topics

- {doc}`state_initialization` — `State` presets and representations
- {doc}`simulator_initialization` — parallelism, progress bars, `Result` fields
- {doc}`representation_comparison` — when to use MPS, MCWF, or Lindblad backends
- {doc}`equivalence_checking` — MPO backend, transpiler regression tests, OpenQASM
