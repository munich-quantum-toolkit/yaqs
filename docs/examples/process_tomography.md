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

Process tensor tomography reconstructs a **multi-time comb** that characterises non-Markovian memory in an open quantum process.
Given a Hamiltonian and a schedule of interventions, YAQS runs parallel trajectories and assembles a comb you can use to:

- **predict** held-out final states for new intervention sequences without re-running the simulator,
- estimate **operational memory** metrics (bond entropy :math:`S_V(c)`, rank, singular spectrum) via split-cut probing — see {doc}`operational_memory`.

The cells below use **noise-free** unitary evolution for a quick demonstration; attach a
{class}`~mqt.yaqs.core.data_structures.noise_model.NoiseModel` in production runs.

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

## 2. Dense comb (small `k`)

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
    parallel=False,
    return_type="dense",
)

print(f"Comb Choi matrix shape: {comb_single.to_matrix().shape}")
```

## 3. Multi-step dense comb

```{code-cell} ipython3
---
tags: [remove-output]
---
comb_two = construct_process_tensor(
    operator,
    sim_params,
    timesteps=[0.1, 0.1],
    num_trajectories=100,
    parallel=False,
    return_type="dense",
)

print(f"Comb Choi matrix shape: {comb_two.to_matrix().shape}")
```

## 4. MPO comb (compressed)

For slightly larger `k`, request an {class}`~mqt.yaqs.characterization.memory.combs.tomography.combs.MPOComb` and convert to dense when needed.

```{code-cell} ipython3
---
tags: [remove-output]
---
comb_mpo = construct_process_tensor(
    operator,
    sim_params,
    timesteps=[0.1],
    num_trajectories=80,
    parallel=False,
    return_type="mpo",
    compress_every=1,
)

print(type(comb_mpo).__name__, comb_mpo.to_dense().to_matrix().shape)
```

## 5. Predicting held-out states

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

## 6. Operational memory diagnostics

Bond entropy :math:`S_V(c)`, the singular spectrum, and operational rank are available on any comb via split-cut probing.
For a full walkthrough of the V-matrix pipeline, return dictionary, and parameter sweeps, see {doc}`operational_memory`.

```{code-cell} ipython3
---
tags: [remove-output]
---
ent = comb_two.entropy(cut=2, n_pasts=8, n_futures=8)
print(f"S_V(2) = {ent:.4f} nats")
```

## Related topics

- {doc}`operational_memory` — V-matrix diagnostics, `probe_process`, sweeps
- {doc}`process_tensor_surrogates` — learned comb for larger `k`
- {doc}`analog_simulation` — open-system dynamics underlying the process tensor
- {doc}`realistic_noise_models` — noise models for non-Markovian environments
