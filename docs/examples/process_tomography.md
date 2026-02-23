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

Process Tensor Tomography (PTT) characterises multi-time correlations in an open quantum system
by reconstructing a _process tensor_, a generalisation of the quantum channel concept to several
time steps.
Given an MPO Hamiltonian and a set of state preparations, YAQS runs the simulation in parallel,
injects probe states at each intervention point, and assembles the result into a {class}`ProcessTensor`
object that can be used to:

- compute the **Holevo information** of the channel (how much information the environment retains),
- **predict the final state** for arbitrary held-out input sequences without re-running the simulator.

## 1. Define the system

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

# Two-site Ising chain: H = -J Σ ZZ - g Σ X
num_sites = 2
operator = MPO.ising(num_sites, J=1.0, g=0.5)

sim_params = AnalogSimParams(
    dt=0.1,
    max_bond_dim=16,
    order=1,
)
```

## 2. Single-step tomography

Run tomography for a single evolution segment of length `t = 0.1`.
100 trajectories per preparation sequence are sufficient for a quick demonstration.

```{code-cell} ipython3
---
tags: [remove-output]
---
from mqt.yaqs.characterization.tomography.tomography import run

pt_single = run(
    operator,
    sim_params,
    timesteps=[0.1],
    num_trajectories=100,
)

print(f"Process tensor shape: {pt_single.tensor.shape}")  # (4, 6)
```

The tensor has shape `(4, N)` where `4` encodes the vectorised output density matrix
and `N = 6` is the number of Pauli frame states used as input probes.

## 3. Holevo information

The **Holevo information** $\chi$ quantifies how much (classical) information the channel
can transmit per use:

```{code-cell} ipython3
chi = pt_single.holevo_information(base=2)
print(f"Holevo information (single step): {chi:.4f} bits")
```

A value near 1 means the channel is almost perfectly distinguishing the six input states;
a value near 0 indicates a fully depolarising channel.

## 4. Multi-step tomography and conditional Holevo information

For two successive evolution segments we can ask: _how much information injected at step 0
survives to the output after step 1?_

```{code-cell} ipython3
---
tags: [remove-output]
---
pt_two = run(
    operator,
    sim_params,
    timesteps=[0.1, 0.1],       # two segments of dt each
    num_trajectories=100,
    measurement_bases="Z",      # project in Z basis at the intervention point
)

print(f"Process tensor shape: {pt_two.tensor.shape}")  # (4, 6, 6)
```

```{code-cell} ipython3
# Fix step=1, probe all step-0 inputs → measures how much step-0 info reaches the output
chi_cond = pt_two.holevo_information_conditional(fixed_step=1, fixed_idx=0, base=2)
print(f"Conditional Holevo χ (fix step=1): {chi_cond:.4f} bits")
```

## 5. Predicting held-out states

Once the process tensor is available, you can predict the output for _any_ input density matrix
sequence without additional simulation runs.
The prediction uses a dual-frame contraction — an efficient linear-algebraic operation:

```{code-cell} ipython3
import numpy as np
from mqt.yaqs.characterization.tomography.tomography import (
    _calculate_dual_frame,  # noqa: PLC2701
    _get_basis_states,       # noqa: PLC2701
)

# Build the dual frame from the same basis used during tomography
basis_set = _get_basis_states()
duals = _calculate_dual_frame([b[2] for b in basis_set])

# Choose two arbitrary mixed input states
rng = np.random.default_rng(0)

def _random_rho(rng: np.random.Generator) -> np.ndarray:
    """Sample a random 2×2 density matrix."""
    psi = rng.standard_normal(2) + 1j * rng.standard_normal(2)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    # Mix with identity to make it a proper mixed state
    return 0.7 * rho + 0.3 * 0.5 * np.eye(2, dtype=complex)

rho_0 = _random_rho(rng)
rho_1 = _random_rho(rng)

# Predict final state — no simulator call needed
rho_pred = pt_two.predict_final_state([rho_0, rho_1], duals)
print("Predicted output density matrix:")
print(np.round(rho_pred, 4))
```

The result `rho_pred` is a `(2, 2)` density matrix giving the expected output state at the final
time step given those two successive input states.
