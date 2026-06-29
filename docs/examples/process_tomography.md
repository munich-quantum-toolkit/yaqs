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
applies these interventions at each time point, and assembles the result into a {class}`ProcessTensor`
object that can be used to:

- compute the **Quantum Mutual Information** of the process (how much information the environment retains),
- **predict the final state** for arbitrary held-out input and intervention sequences without re-running the simulator.

## 1. Define the system

```{code-cell} ipython3
from mqt.yaqs import AnalogSimParams, Hamiltonian

# Two-site Ising chain: H = -J Σ ZZ - g Σ X
num_sites = 2
operator = Hamiltonian.ising(num_sites, J=1.0, g=0.5)

sim_params = AnalogSimParams(
    dt=0.1,
    max_bond_dim=16,
    order=1,
)
```

## 2. Single-step tomography

Run tomography for a single evolution segment of length `t = 0.1`.
A few dozen trajectories per preparation sequence are enough for this demonstration.

```{code-cell} ipython3
---
tags: [remove-output]
---
from mqt.yaqs.characterization.tomography.tomography import run

pt_single = run(
    operator,
    sim_params,
    timesteps=[0.1],
    num_trajectories=40,
)
```

The tensor has shape `(4, N)` where `4` encodes the vectorised output density matrix
and `N = 16` is the number of input probes in the Pauli/Liouville frame for this two-qubit chain.

## 3. Quantum Mutual Information

The **Quantum Mutual Information** quantifies how much information is preserved by the channel between the input state ensemble and the final output:

```{code-cell} ipython3
---
tags: [remove-output]
---
qmi = pt_single.quantum_mutual_information(base=2)
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
    num_trajectories=40,
)
```

## 5. Predicting held-out states

Once the process tensor is available, you can predict the output for _any_ initial density matrix
and any _arbitrary local interventions_ applied between time steps without additional simulation runs.
The prediction uses a dual-frame polynomial sum — an efficient linear-algebraic operation:

```{code-cell} ipython3
---
tags: [remove-output]
---
import numpy as np

# Choose an arbitrary mixed input state
rng = np.random.default_rng(0)

def _random_rho(rng: np.random.Generator) -> np.ndarray:
    """Sample a random 2×2 density matrix."""
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

rho_pred = pt_two.predict_final_state(
    interventions=[initial_prep, x_gate_intervention]
)
```

The result `rho_pred` is a `(2, 2)` density matrix giving the expected output state at the final
time step given the initial state and the local unitary intervention applied to the system between the two segments.

## Related topics

- {doc}`analog_simulation` — open-system dynamics underlying the process tensor
- {doc}`realistic_noise_models` — noise models for non-Markovian environments
- {doc}`simulation_parameters` — `num_traj` and truncation settings
