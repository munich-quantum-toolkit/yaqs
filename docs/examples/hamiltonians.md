---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
  execution_timeout: 120
---

```{code-cell} ipython3
:tags: [remove-cell]
%config InlineBackend.figure_formats = ['svg']
```

# Building Hamiltonians

Analog simulations take a
{class}`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian` as the operator
argument to {meth}`~mqt.yaqs.Simulator.run`. Most models are built as
**matrix product operators (MPOs)** under the hood; the `Hamiltonian` wrapper
materialises once at construction and can be reused across parameter sweeps.

**Backend selection** is driven only by
{class}`~mqt.yaqs.core.data_structures.state.State` representation — not by how
the Hamiltonian was constructed:

| `State.representation` | Backend  | Form materialized at `run` |
| ---------------------- | -------- | -------------------------- |
| `"mps"` (default)      | TJM      | MPO                        |
| `"vector"`             | MCWF     | sparse                     |
| `"density_matrix"`     | Lindblad | sparse                     |

Hamiltonian inputs (`tensors`, `matrix`, `sparse_matrix`, or a preset such as
`Hamiltonian.ising(...)`) are **source data**, not backend choices. The same
`Hamiltonian` instance works with all three state representations;
`Simulator.run` converts and caches the required MPO or sparse form.

```{warning}
Selecting ``State.representation="mps"`` (TJM) calls
{meth}`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian.ensure_mpo`. For a
``sparse_matrix`` source this densifies the operator before MPO factorization,
allocating the full Hilbert-space matrix and risking out-of-memory failures on
large systems. Prefer an MPO preset, ``Hamiltonian.from_mpo(...)``, or
``tensors=`` when targeting TJM.
```

This page covers the factory methods in the library. For open-system evolution
after the Hamiltonian is defined, see {doc}`analog_simulation`. For choosing a
state representation, see {doc}`state_initialization` and
{doc}`representation_comparison`.

## `Hamiltonian` versus `MPO`

| Layer                                                           | Role                                              |
| --------------------------------------------------------------- | ------------------------------------------------- |
| {class}`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian` | User-facing type passed to `Simulator.run`        |
| {class}`~mqt.yaqs.core.data_structures.mpo.MPO`                 | Tensor-network operator; built by factories below |

Typical patterns:

- **Preset classmethods** — `Hamiltonian.ising(...)`, `Hamiltonian.pauli(...)`,
  etc. (no `representation=` argument).
- **Wrap an MPO** — `Hamiltonian.from_mpo(mpo)` after `MPO.bose_hubbard(...)` or
  a custom build.
- **Manual data** — `Hamiltonian(tensors=...)`, `Hamiltonian(matrix=...)`, or
  `Hamiltonian(sparse_matrix=...)`. Any of these can drive TJM, MCWF, or
  Lindblad once paired with the matching `State.representation`.
- **Time-dependent terms** — `Hamiltonian(length=..., parameterized_terms=...)`
  pairs an operator factory with each scalar or array-valued time schedule.

Access the materialized MPO with `H.mpo` (after a TJM run or `H.ensure_mpo()`)
and the sparse form with `H.sparse_matrix` (after an MCWF / Lindblad run or
`H.ensure_sparse()`). Both can coexist on one instance.

## Time-dependent Hamiltonians

YAQS allows time-dependent Hamiltonians of the form

```{math}
H(t)=\sum_k H_k\!\left(p_{k,1}(t),p_{k,2}(t),\ldots,p_{k,m_k}(t)\right)
```

Each time-dependent Hamiltonian term $H_k$ must be provided as a
`(factory, schedule)` pair. The term may depend on one or more parameter
functions $p_{k,1}(t), p_{k,2}(t), \ldots$. At the midpoint $t_i$ of every
timestep, the schedule evaluates these parameters, and the factory uses their
current values to construct the corresponding static Hamiltonian term
$H_k(p_{k,1}(t_i),p_{k,2}(t_i),\ldots)$ for that substep. For a term with one
parameter, the schedule may return that value directly as a scalar. For a term
with several parameters, it returns their values together as a tuple or numeric
array. The following example describes a drive whose amplitude increases
linearly in time:

```python
from mqt.yaqs import AnalogSimParams, Hamiltonian, Observable, Simulator, State


def x_drive(amplitude: object) -> Hamiltonian:
    return Hamiltonian.pauli(length=1, one_body=[(float(amplitude), "X")])


def drive_amplitude(time: float) -> float:
    return 1.0 + time


drive = Hamiltonian(
    length=1,
    parameterized_terms=[(x_drive, drive_amplitude)],
)
params = AnalogSimParams(
    observables=[Observable("z", 0)],
    elapsed_time=0.25,
    dt=0.1,
    tdvp_sweeps=2,
    sample_timesteps=True,
)
result = Simulator(parallel=False).run(State(1, initial="zeros"), drive, params)
```

YAQS evaluates each schedule at the midpoint of every TDVP substep and uses the
factory to construct the Hamiltonian term at that instant. Lambda expressions
are not required: any callable taking the time and returning the term's current
physical parameters is accepted. Named functions are generally clearer and
easier to reuse.

Each factory may depend arbitrarily and nonlinearly on its parameter bundle, but
must return a static `Hamiltonian` or `MPO` of the declared `length`. Multiple
factory-schedule pairs are additive terms. Schedule values must be finite
numeric scalars or arrays, and resolved operators must have compatible physical
legs and be Hermitian.

Parameterized Hamiltonians currently support a single MPS `State` with TDVP,
including order-1 and order-2 TJM and `SimulationProgram` analog tuples. MCWF,
Lindblad, BUG, deterministic `list[State]` ensembles, and multi-time observables
reject them before execution. When running parallel trajectories with
`Simulator(parallel=True)`, factories must be pickleable so that Python can send
them to worker processes. Define them as named functions at module level rather
than as lambdas or nested functions. Schedules do not have this restriction.

## Built-in models (quick reference)

| Model                       | Entry point                                                                                                                           | Local dimension per site              |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| Transverse-field Ising      | {meth}`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian.ising`                                                                  | 2 (qubits)                            |
| Heisenberg                  | {meth}`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian.heisenberg`                                                             | 2                                     |
| Generic Pauli sums          | {meth}`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian.pauli` or {meth}`~mqt.yaqs.core.data_structures.mpo.MPO.from_pauli_sum` | 2                                     |
| 1D Fermi–Hubbard            | {meth}`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian.fermi_hubbard_1d`                                                       | 4 (fermionic) or 2 (Jordan–Wigner)    |
| Bose–Hubbard                | {meth}`~mqt.yaqs.core.data_structures.mpo.MPO.bose_hubbard` → `Hamiltonian.from_mpo`                                                  | `local_dim` (boson occupation cutoff) |
| Coupled transmon chain      | {meth}`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian.coupled_transmon`                                                       | alternating qubit / resonator dims    |
| Trapped ion (position grid) | {meth}`~mqt.yaqs.core.data_structures.mpo.MPO.trapped_ion` → `Hamiltonian.from_mpo`                                                   | grid points per ion (1–2 ions)        |

Open (`bc="open"`) and periodic (`bc="periodic"`) boundaries are supported on
the Pauli builders.

## Pauli-string Hamiltonians

### Ising model (shortcut)

The transverse-field Ising Hamiltonian on an open chain is

```{math}
H = -J \sum_i Z_i Z_{i+1} - g \sum_i X_i.
```

```{code-cell} ipython3
from mqt.yaqs import Hamiltonian

L = 4
J, g = 1.0, 0.5
H_ising = Hamiltonian.ising(L, J, g)
```

### Structured one- and two-body Pauli terms

{meth}`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian.pauli` expands
nearest-neighbour `two_body` and on-site `one_body` lists into Pauli strings
automatically:

```{code-cell} ipython3
H_ising = Hamiltonian.pauli(
    length=L,
    two_body=[(-J, "Z", "Z")],
    one_body=[(-g, "X")],
    bc="open",
)
```

The Heisenberg model is available as a one-liner as well:

```{code-cell} ipython3
H_heisenberg = Hamiltonian.heisenberg(L, Jx=1.0, Jy=1.0, Jz=1.0, h=0.2)
```

### Explicit Pauli strings (`from_pauli_sum`)

For **arbitrary** Pauli strings—including long-range couplings—pass
`(coefficient, spec)` pairs to
{meth}`~mqt.yaqs.core.data_structures.mpo.MPO.from_pauli_sum`. Each `spec` lists
operators with **site indices**, e.g. `"Z0 Z3"` or `"X2"`:

```{code-cell} ipython3
from mqt.yaqs import MPO

terms = [(-J, f"Z{i} Z{i+1}") for i in range(L - 1)] + [(-g, f"X{i}") for i in range(L)]
mpo = MPO()
mpo.from_pauli_sum(terms=terms, length=L)
H_custom = Hamiltonian.from_mpo(mpo)
```

Long-range terms are ordinary entries in `terms`:

```python
terms.append((0.1, "Z0 Z3"))  # Z on sites 0 and 3
```

Pauli labels are `I`, `X`, `Y`, `Z` (case-insensitive). Only
`physical_dimension=2` is supported for this builder.

## Fermi–Hubbard (1D)

{meth}`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian.fermi_hubbard_1d`
implements

```{math}
H = -t \sum_{i,\sigma} \left(c^\dagger_{i,\sigma} c_{i+1,\sigma} + \mathrm{h.c.}\right)
+ U \sum_i n_{i,\uparrow} n_{i,\downarrow}
```

(open boundaries, no chemical potential).

### Fermionic sites (default)

One **physical site** has local dimension 4 with basis
$|0\rangle, |\!\downarrow\rangle, |\!\uparrow\rangle, |\!\uparrow\downarrow\rangle$.
Ladder operators act on the composite ↑/↓ space per site (not a Jordan–Wigner
qubit chain across sites).

```{code-cell} ipython3
num_sites = 4
t, u = 1.0, 0.5

H_fermi = Hamiltonian.fermi_hubbard_1d(num_sites, t=t, u=u)
```

Pair with {class}`~mqt.yaqs.core.data_structures.state.State` using
`physical_dimensions=[4] * num_sites` when building product Fock states (see
{doc}`state_initialization`).

### Jordan–Wigner Pauli chain

Pass `jordan_wigner=True` for a qubit chain in the order 1↑, 1↓, 2↑, 2↓, … Here
`length` is the number of **spin orbitals** (must be even):

```{code-cell} ipython3
num_orbitals = 2 * num_sites
H_jw = Hamiltonian.fermi_hubbard_1d(num_orbitals, t=t, u=u, jordan_wigner=True)
```

Use this mode when you need Pauli-string semantics with full JW signs between
orbitals.

```{note}
The analog MPO factories omit a chemical potential $\mu$. For a **digital**
Trotter circuit with $\mu$, see
{func}`~mqt.yaqs.core.libraries.circuit_library.create_1d_fermi_hubbard_circuit`
and {doc}`circuit_observables`.
```

Correctness of the fermionic and JW MPOs is covered by `test_fermi_hubbard_1d_*`
in the package test suite.

## Bose–Hubbard

The Bose–Hubbard model

```{math}
H = \sum_i \left(\omega\, n_i + \frac{U}{2}\, n_i(n_i-1)\right)
- J \sum_i \left(a^\dagger_i a_{i+1} + \mathrm{h.c.}\right)
```

is available on {meth}`~mqt.yaqs.core.data_structures.mpo.MPO.bose_hubbard`.
Wrap the MPO for analog simulation:

```{code-cell} ipython3
from mqt.yaqs import Hamiltonian, MPO

local_dim = 3  # occupations 0, 1, …, local_dim - 1
H_bh = Hamiltonian.from_mpo(
    MPO.bose_hubbard(
        length=3,
        local_dim=local_dim,
        omega=1.0,
        hopping_j=0.2,
        hubbard_u=0.5,
    )
)
```

Initial states must respect the boson dimension, e.g.
`State(length, initial="zeros", physical_dimensions=[local_dim] * length)`.

## Coupled transmon–resonator chains

{meth}`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian.coupled_transmon`
builds an alternating chain of transmon qubits and resonators with local
dimensions `qubit_dim` and `resonator_dim`:

```{code-cell} ipython3
H_transmon = Hamiltonian.coupled_transmon(
    length=3,
    qubit_dim=3,
    resonator_dim=5,
    qubit_freq=5.0,
    resonator_freq=7.0,
    anharmonicity=-0.3,
    coupling=0.1,
)
```

A full SWAP-style open-system example is in {doc}`transmon_emulation`.

## Trapped-ion position grid

{meth}`~mqt.yaqs.core.data_structures.mpo.MPO.trapped_ion` builds a **static**
Hamiltonian for one or two ions on a uniform position grid. Each ion is one MPO
site with local dimension equal to the number of grid points. The local terms
are a harmonic trap plus a centered finite-difference kinetic energy; for two
ions, a softened Coulomb repulsion is compressed into MPO channels (optional SVD
truncation via `coulomb_cutoff` or `max_bond_dim`).

```{math}
H = \sum_i \left[-\frac{\hbar^2}{2m_i}\frac{d^2}{dx_i^2} + \tfrac{1}{2} m_i \omega^2 (x_i - q)^2\right]
+ \frac{g}{\sqrt{(x_1-x_2)^2 + a^2}}
```

(the Coulomb term applies only when two masses are supplied).

```{code-cell} ipython3
from mqt.yaqs import Hamiltonian, MPO
import numpy as np

positions = np.linspace(-6.0, 6.0, 25)
H_ion = Hamiltonian.from_mpo(
    MPO.trapped_ion(
        positions,
        masses=[1.0],
        omega=1.0,
        trap_center=0.0,
    )
)

# Two ions with softened Coulomb repulsion on the same grid spacing
H_pair = Hamiltonian.from_mpo(
    MPO.trapped_ion(
        positions,
        masses=[1.0, 1.0],
        omega=1.0,
        coulomb_strength=1.0,
        softening_length=float(positions[1] - positions[0]),
    )
)
```

Pair with {class}`~mqt.yaqs.core.data_structures.state.State` using
`physical_dimensions=[len(positions)]` per ion site. A wavepacket reflection
benchmark is in {doc}`trapped_ion`.

```{note}
YAQS applies $\exp(-\mathrm{i}\,\Delta t\, H)$ during evolution. When using SI
units, pass energies and times in consistent units or rescale $H/\hbar$
explicitly (see the factory docstring).
```

## Manual Hamiltonians

For imported MPO cores or small-system dense/sparse operators:

```python
# MPO tensor cores (rank-4 per site, already in MPO layout) — preferred for TJM
H = Hamiltonian(tensors=my_cores)

# Dense or sparse matrix — YAQS converts to MPO or sparse as needed at run time.
# sparse_matrix → TJM densifies; prefer tensors=/from_mpo/presets for large systems.
H = Hamiltonian(matrix=dense_h, physical_dimension=2)
H = Hamiltonian(sparse_matrix=sparse_h, physical_dimension=2)
```

## Related topics

- {doc}`analog_simulation` — TJM evolution, noise, and observables
- {doc}`transmon_emulation` — multi-level transmon physics
- {doc}`trapped_ion` — position-grid wavepacket dynamics
- {doc}`state_initialization` — `physical_dimensions` and representations
- {doc}`simulation_parameters` — truncation presets for MPO evolution
