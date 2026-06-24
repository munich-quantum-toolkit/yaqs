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

# Process Tensor Tomography

```{note}
This page runs several trajectory-heavy tomography steps and may take one to two minutes during a documentation build (`execution_timeout: 600`).
```

Process Tensor Tomography (PTT) characterises multi-time correlations in an open quantum system
by reconstructing a _process tensor_, a generalisation of the quantum channel concept to several
time steps.
The cells below use **noise-free** unitary evolution for a quick demonstration; attach a
{class}`~mqt.yaqs.core.data_structures.noise_model.NoiseModel` in production runs to characterise
open-system channels.
Given a {class}`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian` and a set of state preparations and measurement projections, YAQS runs the simulation in parallel,
applies these interventions at each time point, and reconstructs a **process tensor comb** that can be used to:

- compute the **Quantum Mutual Information** of the process (how much information the environment retains),
- **predict the final state** for arbitrary held-out input and intervention sequences without re-running the simulator.

## 1. Define the system

```{code-cell} ipython3
from mqt.yaqs import AnalogSimParams, Hamiltonian

num_sites = 2
hamiltonian = Hamiltonian.ising(num_sites, J=1.0, g=0.5)
operator = hamiltonian.mpo

sim_params = AnalogSimParams(
    dt=0.1,
    max_bond_dim=16,
    order=1,
)
```

## 2. Single-step tomography

```{code-cell} ipython3
---
tags: [remove-output]
---
from mqt.yaqs import construct_process_tensor

comb_single = construct_process_tensor(
    operator,
    sim_params,
    timesteps=[0.1],
    num_trajectories=100,
    return_type="dense",
)

print(f"Comb Choi matrix shape: {comb_single.to_matrix().shape}")
```

## 3. Multi-step tomography

```{code-cell} ipython3
---
tags: [remove-output]
---
from mqt.yaqs import construct_process_tensor

comb_two = construct_process_tensor(
    operator,
    sim_params,
    timesteps=[0.1, 0.1],
    num_trajectories=100,
    return_type="dense",
)

print(f"Comb Choi matrix shape: {comb_two.to_matrix().shape}")
```

## 4. Predicting held-out states

```{code-cell} ipython3
---
tags: [remove-output]
---
import numpy as np

rng = np.random.default_rng(0)

def _random_rho(rng: np.random.Generator) -> np.ndarray:
    psi = rng.standard_normal(2) + 1j * rng.standard_normal(2)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    return 0.7 * rho + 0.3 * 0.5 * np.eye(2, dtype=complex)

rho_0 = _random_rho(rng)

def initial_prep(rho: np.ndarray) -> np.ndarray:
    return rho_0

def x_gate_intervention(rho: np.ndarray) -> np.ndarray:
    x_mat = np.array([[0, 1], [1, 0]], dtype=complex)
    return x_mat @ rho @ x_mat.conj().T

rho_pred = comb_two.predict([initial_prep, x_gate_intervention])
print("Predicted output density matrix:")
print(np.round(rho_pred, 4))
```

## Related topics

- {doc}`analog_simulation` — open-system dynamics underlying the process tensor
- {doc}`realistic_noise_models` — noise models for non-Markovian environments
- {doc}`simulation_parameters` — `num_traj` and truncation settings
