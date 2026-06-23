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

This example simulates a **qubit–resonator–qubit** chain with {meth}`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian.coupled_transmon` (dipole coupling per {meth}`~mqt.yaqs.core.data_structures.mpo.MPO.coupled_transmon`).

We prepare $|100\rangle$ (left transmon excited) and evolve for one resonant swap period $T_{\mathrm{swap}} = \pi/(\sqrt{2}\,g)$. The same evolution is run **twice**:

1. **Noiseless** — unitary analog simulation (TDVP on the MPO).
2. **Noisy** — open-system simulation with relaxation and dephasing on the qubit sites (TJM trajectories).

PVM observables track probabilities for bitstrings using only local indices $0$ and $1$ per site. With `qubit_dim = resonator_dim = 3`, population in the $|2\rangle$ level appears as **leakage** (not counted in those bitstrings).

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

all_bitstrings = ["000", "001", "010", "011", "100", "101", "110", "111"]

sim_params = AnalogSimParams(
    observables=[Observable(bstr) for bstr in all_bitstrings],
    elapsed_time=T_swap,
    dt=dt,
    sample_timesteps=True,
)


def pvm_curve(result, bitstring: str) -> np.ndarray:
    for obs, vals in zip(result.observables, result.expectation_values, strict=True):
        if obs.gate.bitstring == bitstring:
            return np.asarray(vals, dtype=float)
    msg = f"bitstring {bitstring!r} not in observables"
    raise ValueError(msg)


def leakage_at_t(result, t_idx: int) -> float:
    leak = 1.0
    for obs, vals in zip(result.observables, result.expectation_values, strict=True):
        if obs.gate.bitstring in all_bitstrings:
            leak -= float(vals[t_idx])
    return leak
```

## 3. Noiseless SWAP

```{code-cell} ipython3
---
tags: [remove-output]
---
import copy

from mqt.yaqs import Simulator

sim = Simulator(show_progress=False)
result_clean = sim.run(copy.deepcopy(state), H_0, copy.deepcopy(sim_params))
```

```{code-cell} ipython3
---
tags: [remove-output]
---
p100_clean = pvm_curve(result_clean, "100")
p001_clean = pvm_curve(result_clean, "001")
times = sim_params.times
```

## 4. Noisy SWAP

Relaxation and dephasing on transmon sites (even indices). Built-in `lowering` and `pauli_z` processes are 2×2; for `qubit_dim = 3` we pass explicit jump matrices ({class}`~mqt.yaqs.core.libraries.gate_library.Destroy` and a computational-subspace dephasing operator). For log-normal and other distributed noise strengths, see {doc}`realistic_noise_models`.

```{code-cell} ipython3
---
tags: [remove-output]
---
from mqt.yaqs import NoiseModel
from mqt.yaqs.core.libraries.gate_library import Destroy

relax = Destroy(qubit_dim).matrix
dephase = np.diag([1.0, -1.0, 1.0]).astype(complex)  # |2⟩ unaffected

noise_model = NoiseModel(
    [{"name": "t1", "sites": [i], "strength": 0.03, "matrix": relax} for i in (0, 2)]
    + [{"name": "dephase", "sites": [i], "strength": 0.02, "matrix": dephase} for i in (0, 2)]
)

noisy_params = AnalogSimParams(
    observables=[Observable(bstr) for bstr in all_bitstrings],
    elapsed_time=T_swap,
    dt=dt,
    sample_timesteps=True,
    num_traj=32,
    random_seed=7,
)

result_noisy = sim.run(copy.deepcopy(state), H_0, noisy_params, noise_model)
```

```{code-cell} ipython3
---
tags: [remove-output]
---
p100_noisy = pvm_curve(result_noisy, "100")
p001_noisy = pvm_curve(result_noisy, "001")
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

ax_pop.plot(times, p001_clean, "-", color="tab:blue", label=r"noiseless $P(|001\rangle)$")
ax_pop.plot(times, p100_clean, "-", color="tab:orange", label=r"noiseless $P(|100\rangle)$")
ax_pop.plot(times, p001_noisy, "--", color="tab:blue", label=r"noisy $P(|001\rangle)$")
ax_pop.plot(times, p100_noisy, "--", color="tab:orange", label=r"noisy $P(|100\rangle)$")
ax_pop.axvline(T_swap, color="gray", linestyle=":", alpha=0.6, label=r"$T_{\mathrm{swap}}$")
ax_pop.set_xlabel("time")
ax_pop.set_ylabel("probability")
ax_pop.set_title("SWAP populations: noiseless vs noisy")
ax_pop.legend(fontsize=8)
ax_pop.grid(alpha=0.3)

leak_clean = [leakage_at_t(result_clean, i) for i in range(len(times))]
leak_noisy = [leakage_at_t(result_noisy, i) for i in range(len(times))]
ax_leak.plot(times, leak_clean, "-", color="tab:green", label="noiseless leakage")
ax_leak.plot(times, leak_noisy, "--", color="tab:red", label="noisy leakage")
ax_leak.set_xlabel("time")
ax_leak.set_ylabel("leakage")
ax_leak.set_title("Population outside 0/1 subspace per site")
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
