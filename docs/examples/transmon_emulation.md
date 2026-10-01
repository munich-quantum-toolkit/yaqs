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

# Transmon-Resonator Chain Emulation

This example simulates a **qubit–resonator–qubit** chain with
{meth}`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian.coupled_transmon`
(dipole coupling per
{meth}`~mqt.yaqs.core.data_structures.mpo.MPO.coupled_transmon`).

We prepare $|100\rangle$ (left transmon excited) and evolve for one resonant
swap period $T_{\mathrm{swap}} = \pi/(\sqrt{2}\,g)$. The same evolution is run
**twice**:

1. **Noiseless** — unitary analog simulation (TDVP on the MPO).
2. **Noisy** — open-system simulation with relaxation and dephasing on the qubit
   sites (TJM trajectories).

Local projectors track the $|1\rangle$ population on each transmon and the
$|2\rangle$ population on every site. Binary bitstring observables and shot
counts are restricted to all-qubit states, so this non-qubit example uses local
three-level observables instead. The sum of the local $|2\rangle$ populations is
the expected number of sites in the leakage level.

## 1. Hamiltonian and initial state

```{code-cell} ipython3
import numpy as np
from mqt.yaqs import Hamiltonian, State

length = 3  # qubit – resonator – qubit
qubit_dim = 3
resonator_dim = 3
w_q = 4 / (2 * np.pi)
w_r = 4 / (2 * np.pi)
alpha = -0.3 / (2 * np.pi)
g = 0.2 / (2 * np.pi)

H_0 = Hamiltonian.coupled_transmon(
    length=length,
    qubit_dim=qubit_dim,
    resonator_dim=resonator_dim,
    qubit_freq=w_q,
    resonator_freq=w_r,
    anharmonicity=alpha,
    coupling=g,
)

T_swap = np.pi / (np.sqrt(2) * g)
dt = T_swap / 100

# |100⟩: left qubit (site 0) in |1⟩
state = State(
    length,
    initial="basis",
    basis_string="100",
    physical_dimensions=[qubit_dim, resonator_dim, qubit_dim],
)
```

## 2. Observables and shared parameters

```{code-cell} ipython3
from mqt.yaqs import AnalogSimParams, Observable

projector_1 = np.diag([0.0, 1.0, 0.0])
projector_2 = np.diag([0.0, 0.0, 1.0])
population_observables = [
    Observable(projector_1, sites=0),
    Observable(projector_1, sites=2),
    *(Observable(projector_2, sites=site) for site in range(length)),
]

sim_params = AnalogSimParams(
    observables=population_observables,
    elapsed_time=T_swap,
    dt=dt,
    sample_timesteps=True,
)


def population_curve(result, observable_index: int) -> np.ndarray:
    values = result.expectation_values[observable_index]
    if values is None:
        msg = f"observable {observable_index} has no values"
        raise ValueError(msg)
    return np.asarray(values, dtype=float)


def leakage_curve(result) -> np.ndarray:
    return sum((population_curve(result, index) for index in range(2, 5)), start=np.zeros(len(result.times)))
```

## 3. Noiseless SWAP

```{code-cell} ipython3
import copy

from mqt.yaqs import Simulator

sim = Simulator(show_progress=False)
result_clean = sim.run(copy.deepcopy(state), H_0, copy.deepcopy(sim_params))
```

```{code-cell} ipython3
left_clean = population_curve(result_clean, 0)
right_clean = population_curve(result_clean, 1)
times = sim_params.times
```

## 4. Noisy SWAP

Relaxation and dephasing on transmon sites (even indices). Built-in `lowering`
and `pauli_z` processes are 2×2; for `qubit_dim = 3` we pass explicit jump
matrices ({class}`~mqt.yaqs.core.libraries.gate_library.Destroy` and a
computational-subspace dephasing operator). For log-normal and other distributed
noise strengths, see {doc}`realistic_noise_models`.

```{code-cell} ipython3
from mqt.yaqs import NoiseModel
from mqt.yaqs.core.libraries.gate_library import Destroy

relax = Destroy(qubit_dim).matrix
dephase = np.diag([1.0, -1.0, 1.0]).astype(complex)  # |2⟩ unaffected

noise_model = NoiseModel(
    [{"name": "t1", "sites": [i], "strength": 0.03, "matrix": relax} for i in (0, 2)]
    + [{"name": "dephase", "sites": [i], "strength": 0.02, "matrix": dephase} for i in (0, 2)]
)

noisy_params = AnalogSimParams(
    observables=population_observables,
    elapsed_time=T_swap,
    dt=dt,
    sample_timesteps=True,
    num_traj=32,
    random_seed=7,
)

result_noisy = sim.run(copy.deepcopy(state), H_0, noisy_params, noise_model)
```

```{code-cell} ipython3
left_noisy = population_curve(result_noisy, 0)
right_noisy = population_curve(result_noisy, 1)
```

## 5. Comparison plot

```{code-cell} ipython3
---
mystnb:
  image:
    width: 90%
    align: center
---
import matplotlib.pyplot as plt

fig, (ax_pop, ax_leak) = plt.subplots(1, 2, figsize=(9, 3.5))

ax_pop.plot(times, right_clean, "-", color="tab:blue", label=r"noiseless right $P(|1\rangle)$")
ax_pop.plot(times, left_clean, "-", color="tab:orange", label=r"noiseless left $P(|1\rangle)$")
ax_pop.plot(times, right_noisy, "--", color="tab:blue", label=r"noisy right $P(|1\rangle)$")
ax_pop.plot(times, left_noisy, "--", color="tab:orange", label=r"noisy left $P(|1\rangle)$")
ax_pop.axvline(T_swap, color="gray", linestyle=":", alpha=0.6, label=r"$T_{\mathrm{swap}}$")
ax_pop.set_xlabel("time")
ax_pop.set_ylabel("probability")
ax_pop.set_title("SWAP populations: noiseless vs noisy")
ax_pop.legend(fontsize=8)
ax_pop.grid(alpha=0.3)

leak_clean = leakage_curve(result_clean)
leak_noisy = leakage_curve(result_noisy)
ax_leak.plot(times, leak_clean, "-", color="tab:green", label="noiseless leakage")
ax_leak.plot(times, leak_noisy, "--", color="tab:red", label="noisy leakage")
ax_leak.set_xlabel("time")
ax_leak.set_ylabel(r"summed $|2\rangle$ population")
ax_leak.set_title("Occupation of the leakage level")
ax_leak.legend(fontsize=8)
ax_leak.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

## Related topics

- {doc}`analog_simulation` — analog time evolution and noise models
- {doc}`realistic_noise_models` — distributed noise strengths
- {doc}`state_initialization` — custom `physical_dimensions` and basis states
- {doc}`simulation_parameters` — `sample_timesteps`, `num_traj`, and observables
