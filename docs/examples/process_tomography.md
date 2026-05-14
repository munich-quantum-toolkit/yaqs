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
Given an MPO Hamiltonian and a set of state preparations and measurement projections, YAQS runs the simulation in parallel,
applies these interventions at each time point, and assembles the result into a {class}`ProcessTensor`
object that can be used to:

- compute the **Quantum Mutual Information** of the process (how much information the environment retains),
- **predict the final state** for arbitrary held-out input and intervention sequences without re-running the simulator.

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

## 3. Quantum Mutual Information

The **Quantum Mutual Information** quantifies how much information is preserved by the channel between the input state ensemble and the final output:

```{code-cell} ipython3
qmi = pt_single.quantum_mutual_information(base=2)
print(f"Quantum Mutual Information (single step): {qmi:.4f} bits")
```

For unitary channels, this value approaches the entropy of the average input state (~0.907 bits for the standard 4-state Pauli frame).
A value near 0 indicates a fully depolarising channel that destroys all quantum and classical information.

## 4. Multi-step tomography

For two successive evolution segments, we can reconstruct the temporal correlation map across an intermediate time step:

```{code-cell} ipython3
---
tags: [remove-output]
---
pt_two = run(
    operator,
    sim_params,
    timesteps=[0.1, 0.1],       # two segments of dt each
    num_trajectories=100,
)

print(f"Process tensor shape: {pt_two.tensor.shape}")  # (4, 16, 16)
```

````

## 5. Predicting held-out states

Once the process tensor is available, you can predict the output for _any_ initial density matrix
and any _arbitrary local interventions_ applied between time steps without additional simulation runs.
The prediction uses a dual-frame polynomial sum — an efficient linear-algebraic operation:

```{code-cell} ipython3
import numpy as np

# Choose an arbitrary mixed input state
rng = np.random.default_rng(0)

def _random_rho(rng: np.random.Generator) -> np.ndarray:
    """Sample a random 2×2 density matrix."""
    psi = rng.standard_normal(2) + 1j * rng.standard_normal(2)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    # Mix with identity to make it a proper mixed state
    return 0.7 * rho + 0.3 * 0.5 * np.eye(2, dtype=complex)

rho_0 = _random_rho(rng)

# The first intervention is the preparation of the initial state at t=0
def initial_prep(rho: np.ndarray) -> np.ndarray:
    return rho_0

# Define an arbitrary CPTP intervention map applied at the intermediate timestep
def x_gate_intervention(rho: np.ndarray) -> np.ndarray:
    x_mat = np.array([[0, 1], [1, 0]], dtype=complex)
    return x_mat @ rho @ x_mat.conj().T

# Predict final state — no simulator call needed!
rho_pred = pt_two.predict_final_state(
    interventions=[initial_prep, x_gate_intervention]
)
print("Predicted output density matrix:")
print(np.round(rho_pred, 4))
````

The result `rho_pred` is a `(2, 2)` density matrix giving the expected output state at the final
time step given the initial state and the local unitary intervention applied to the system between the two segments.
