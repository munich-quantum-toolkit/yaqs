---
file_format: mystnb
kernelspec:
  name: python3
language_info:
  name: python
mystnb:
  number_source_lines: true
  execution_timeout: 300
---

# Initializing Quantum States

The initial state sets the starting point for a YAQS simulation. Use `State` to
choose a named preparation or supply your own data. The default matrix product
state (MPS) representation supports analog and circuit simulation without
storing a full state vector.

## Choose a product state

Specify the number of sites and, optionally, a preset. Sites are qubits unless
you provide other local dimensions:

```{code-cell} python
from mqt.yaqs import State

zeros = State(20)
polarized = State(20, initial="x+")
```

Here, `zeros` puts every site in $|0\rangle$, while `polarized` puts every site
in $(|0\rangle + |1\rangle)/\sqrt{2}$. Both are product states: the sites have
no initial entanglement.

| `initial`           | Preparation                                                                                      |
| ------------------- | ------------------------------------------------------------------------------------------------ |
| `"zeros"` (default) | Every site in $\lvert 0\rangle$.                                                                 |
| `"ones"`            | Every site in $\lvert 1\rangle$.                                                                 |
| `"x+"`, `"x-"`      | Every site in $(\lvert 0\rangle \pm \lvert 1\rangle)/\sqrt{2}$.                                  |
| `"y+"`, `"y-"`      | Every site in $(\lvert 0\rangle \pm i\lvert 1\rangle)/\sqrt{2}$.                                 |
| `"Neel"`            | Alternating levels, starting with site 0 in $\lvert 1\rangle$: `1010…` in site order.            |
| `"wall"`            | The first `length // 2` sites in $\lvert 0\rangle$ and the remaining sites in $\lvert 1\rangle$. |
| `"basis"`           | One computational-basis configuration, supplied with `basis_string`.                             |
| `"random"`          | A random product state; see the random-state examples below.                                     |

## Place an excitation and check site order

The {doc}`analog_simulation` guide follows an excitation that starts near the
center of a chain. Prepare that state by giving one basis digit per site:

```{code-cell} python
length = 20
center = length // 2
basis = "0" * center + "1" + "0" * (length - center - 1)
localized = State(length, initial="basis", basis_string=basis)
```

Character `i` in `basis_string` selects site `i`, starting with site 0 on the
left. A measurement bitstring instead displays site 0 on the right:

```{code-cell} python
site_zero = State(3, initial="basis", basis_string="100")
print(site_zero.mps.to_vec())
```

This state has site 0 in $|1\rangle$ and the other sites in $|0\rangle$. Its
dense vector has its only nonzero entry at index 1, and its readout bitstring is
`"001"`. Dense vectors and the rows and columns of density matrices use site 0
as the fastest-varying subsystem. For qubits, this matches Qiskit's ordering.

## Choose a representation

Set `representation` when constructing a preset state, for example
`State(4, initial="x+", representation="vector")`. YAQS selects the analog
backend from this choice:

| `representation`   | State storage and supported use                                                    |
| ------------------ | ---------------------------------------------------------------------------------- |
| `"mps"` (default)  | Tensor network for analog evolution, circuits, and unitary ensembles.              |
| `"vector"`         | Dense pure state for analog evolution with Monte Carlo wave-function trajectories. |
| `"density_matrix"` | Dense pure or mixed state for analog Lindblad evolution.                           |

For $N$ qubits, a dense vector has $2^N$ entries and a density matrix has $4^N$
entries. Keep dense calculations small; circuit simulation requires `"mps"`. See
{doc}`representation_comparison` for a comparison on the same physical model.

## Prepare random states

Use `"random"` for an unentangled state with real, nonnegative random local
amplitudes. To allow initial entanglement, use `"haar-random"` and choose an
initial maximum bond dimension with `pad`:

```{code-cell} python
random_product = State(6, initial="random")
random_mps = State(6, initial="haar-random", pad=4)
```

`"haar-random"` builds an MPS from random isometries. It does not sample
uniformly from all pure states in the full Hilbert space. The bond dimensions
are limited by `pad` and the sizes of the neighboring subsystems. Omitting `pad`
gives a maximum bond dimension of one, so the state is then a product state.
This initial choice is separate from `max_bond_dim` during evolution.

For a reproducible random product state in a dense representation, pass `seed`:

```{code-cell} python
repeatable = State(4, initial="random", representation="vector", seed=7)
```

The same seed reproduces the same initial vector. It also works with
`representation="density_matrix"`, which forms the corresponding pure-state
density matrix. Currently, random MPS initialization and `"haar-random"` ignore
`seed`. To reuse those preparations, construct the state once and pass the same
object to successive runs. The simulation parameter `random_seed` controls
stochastic evolution; it does not seed state preparation.

## Supply a vector or a mixed state

Pass exactly one of `vector`, `density_matrix`, or `tensors` for manual data.
The representation is inferred, so omit `representation`. Do not combine manual
data with preset options such as `initial`, `basis_string`, `seed`, or `pad`.

The vector below prepares the entangled Bell state
$(|00\rangle + |11\rangle)/\sqrt{2}$:

```{code-cell} python
import numpy as np

bell = State(vector=np.array([1, 0, 0, 1], dtype=complex))
print(bell.vector)
```

YAQS copies and normalizes the vector. The input must be a finite, nonzero,
one-dimensional array. Without explicit local dimensions, its size must be a
power of two; YAQS infers the number of qubits.

A mixed state describes a statistical preparation. This example puts the system
in $|00\rangle$ with probability 0.6 and $|11\rangle$ with probability 0.4:

```{code-cell} python
rho = np.diag([0.6, 0.0, 0.0, 0.4]).astype(complex)
mixed = State(density_matrix=rho)
```

YAQS copies the matrix and normalizes its trace. The input must be finite,
square, Hermitian, positive semidefinite, and have positive real trace. Manual
vectors and density matrices select their dense analog backends. They cannot
serve as circuit inputs; use a preset or MPS tensors instead.

## Use other local dimensions

Set `physical_dimensions` to an integer for a uniform chain or a list for
different dimensions at each site:

```{code-cell} python
qutrits = State(3, physical_dimensions=3)
qubit_qutrit = State(
    2, initial="basis", basis_string="12", physical_dimensions=[2, 3]
)
```

The second state has a qubit at site 0 in level 1 and a qutrit at site 1 in
level 2. Presets such as `"ones"` still use level 1, not the highest local
level. For manual dense data, local dimensions must multiply to the vector
length or matrix dimension. Bitstring probabilities and shot measurements
require qubits; digital gates currently require qubit target sites. See
{doc}`transmon_emulation` and {doc}`trapped_ion` for device examples.

## Advanced MPS preparation

:::{dropdown} Supply MPS tensors or wrap an existing MPS

Use `tensors=` for custom open-boundary MPS cores. Each core has axes
`(physical, left, right)`. Neighboring bond dimensions must match, exterior
bonds must have dimension one, and all entries must be finite. This example
prepares the same Bell state as the vector above, now in MPS form:

```python
left = (np.eye(2) / np.sqrt(2)).reshape(2, 1, 2)
right = np.eye(2).reshape(2, 2, 1)
bell_mps = State(tensors=[left, right])
wrapped = State.from_mps(bell_mps.mps)
```

`State(tensors=...)` infers the site count and normalizes the MPS. For other
physical dimensions, supply matching `physical_dimensions`.

When you already have an {class}`~mqt.yaqs.MPS`, use
{meth}`~mqt.yaqs.State.from_mps` to wrap it. This method references the same MPS
without copying or normalizing it. Changes through either reference affect the
same state. For an independent normalized state, supply copies of its tensors
through `State(tensors=...)` and preserve its physical dimensions.

:::

:::{dropdown} Pad the initial MPS bonds

For product presets, `pad` adds zero entries to enlarge the initial MPS bonds
without changing the physical state. For `"haar-random"`, it instead sets the
maximum initial bond dimension used during random construction.

Padding allocates initial MPS space; it does not set the bond limit during
evolution. That limit is `max_bond_dim` in the simulation parameters. Dense
product-state construction does not use MPS padding.

:::

Once the initial state is prepared, pass it to `Simulator.run` with the model,
measurements, and optional noise. The {doc}`simulator_initialization` guide
shows how to run and reuse a simulator, while {doc}`simulation_parameters`
describes accuracy and sampling choices. Full constructor details are in the
{class}`~mqt.yaqs.State` API reference.
