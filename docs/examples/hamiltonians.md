---
file_format: mystnb
kernelspec:
  name: python3
language_info:
  name: python
mystnb:
  number_source_lines: true
  execution_timeout: 120
---

# Building Hamiltonians

A Hamiltonian defines the energies and interactions in an analog simulation.
Build one with a named model, a sum of Pauli terms, or your own operator data,
then pass it to `Simulator.run`. Match its site count and local dimensions to
those of the initial `State`.

## Choose a built-in model

Most builders create a matrix product operator (MPO), which stores the operator
as a tensor network. Use the `Hamiltonian` methods directly when available; wrap
an MPO with `Hamiltonian.from_mpo` for the other models.

| Model                                             | Constructor                                    | Site layout                                           |
| ------------------------------------------------- | ---------------------------------------------- | ----------------------------------------------------- |
| Transverse-field Ising                            | {meth}`~mqt.yaqs.Hamiltonian.ising`            | Qubits.                                               |
| Heisenberg or XY                                  | {meth}`~mqt.yaqs.Hamiltonian.heisenberg`       | Qubits.                                               |
| On-site and nearest-neighbor Pauli terms          | {meth}`~mqt.yaqs.Hamiltonian.pauli`            | Qubits.                                               |
| Indexed Pauli strings, including long-range terms | {meth}`~mqt.yaqs.MPO.from_pauli_sum`           | Qubits; wrap the MPO.                                 |
| 1D Fermi–Hubbard                                  | {meth}`~mqt.yaqs.Hamiltonian.fermi_hubbard_1d` | Dimension-four sites, or a Jordan–Wigner qubit chain. |
| Bose–Hubbard                                      | {meth}`~mqt.yaqs.MPO.bose_hubbard`             | Truncated boson occupation; wrap the MPO.             |
| Coupled transmons and resonators                  | {meth}`~mqt.yaqs.Hamiltonian.coupled_transmon` | Alternating transmon and resonator dimensions.        |
| Trapped ion position grid                         | {meth}`~mqt.yaqs.MPO.trapped_ion`              | One grid per ion, for one or two ions; wrap the MPO.  |

YAQS evolves with $\exp(-itH)$, using $\hbar=1$. Hamiltonian coefficients and
times must use consistent units. For energies in SI units, divide the operator
by $\hbar$ before evolving with time in seconds.

## Build a spin chain

For an open chain, the Ising shortcut constructs

$$
H_{\mathrm{Ising}}=-J\sum_{i=0}^{L-2}Z_iZ_{i+1}-g\sum_{i=0}^{L-1}X_i.
$$

```{code-cell} python
from mqt.yaqs import Hamiltonian, State

length = 4
hamiltonian = Hamiltonian.ising(length, J=1.0, g=0.5)
state = State(length, initial="zeros")
```

The Heisenberg shortcut uses the sign convention

$$
H_{\mathrm{Heisenberg}}=-\sum_{i=0}^{L-2}
\left(J_xX_iX_{i+1}+J_yY_iY_{i+1}+J_zZ_iZ_{i+1}\right)
-h\sum_{i=0}^{L-1}Z_i.
$$

Here, $X$, $Y$, and $Z$ are Pauli matrices with eigenvalues $\pm1$, not spin
operators with eigenvalues $\pm1/2$. Setting `Jz=0` gives the XY model used in
{doc}`analog_simulation`:

```{code-cell} python
xy = Hamiltonian.heisenberg(length, Jx=0.5, Jy=0.5, Jz=0.0)
```

These builders and `Hamiltonian.pauli` use open boundaries by default. Set
`bc="periodic"` to include the bond from the last site to site 0.

## Specify your own Pauli terms

`Hamiltonian.pauli` repeats each `two_body` term over neighboring sites and each
`one_body` term over all sites. Coefficients enter with the sign you supply. For
example, the following builds the same XY Hamiltonian as above:

```{code-cell} python
xy_from_terms = Hamiltonian.pauli(
    length=length,
    two_body=[(-0.5, "X", "X"), (-0.5, "Y", "Y")],
)
```

Add an on-site field with an entry such as `one_body=[(-0.2, "Z")]`. These
structured builders require finite real coefficients and qubit sites.

For site-dependent fields, separated sites, or longer strings, build an MPO from
explicit `(coefficient, string)` pairs:

```{code-cell} python
from mqt.yaqs import MPO

mpo = MPO()
mpo.from_pauli_sum(
    terms=[(0.4, "Z0 Z3"), (0.2, "Y1"), (0.1, "")],
    length=length,
)
custom = Hamiltonian.from_mpo(mpo)
```

This operator is $0.4Z_0Z_3+0.2Y_1+0.1I$. Site indices start at zero; omitted
sites act as identities, and an empty string denotes the identity operator.
Labels `I`, `X`, `Y`, and `Z` are case-insensitive. Use real coefficients for
Hermitian Pauli terms.

## Use other local dimensions

Hubbard and device models need an initial state with the same local dimensions
as the Hamiltonian. For a uniform layout, use `physical_dimensions=local_dim`;
for different dimensions, supply a list in site order. The
{doc}`state_initialization` guide explains these preparations.

The coupled-transmon builder alternates transmons and resonators, starting with
a transmon. Its `length` counts both kinds of sites. The trapped ion builder
uses one site per ion, with a local dimension equal to the number of grid
points. See {doc}`transmon_emulation` and {doc}`trapped_ion` for worked device
examples, noise, and the relevant units.

:::{dropdown} Fermi–Hubbard: physical sites and Jordan–Wigner orbitals

The default builder uses dimension-four sites with local basis $|0\rangle$,
$|\!\downarrow\rangle$, $|\!\uparrow\rangle$, $|\!\uparrow\downarrow\rangle$:

```python
num_sites = 3
fermi = Hamiltonian.fermi_hubbard_1d(num_sites, t=1.0, u=0.5)
fermi_state = State(num_sites, physical_dimensions=4)
```

This mode uses ladder operators on composite sites. For a Pauli-chain model with
full Jordan–Wigner signs between spin orbitals, set `jordan_wigner=True`:

```python
jw = Hamiltonian.fermi_hubbard_1d(2 * num_sites, t=1.0, u=0.5, jordan_wigner=True)
jw_state = State(2 * num_sites)
```

The two modes use different hopping-sign conventions. Match both the state and
operator basis when comparing them.

Here, `length` counts spin orbitals and must be even and at least two. Site
order is $1\uparrow,1\downarrow,2\uparrow,2\downarrow,\ldots$. Both builders use
open boundaries and omit a chemical-potential term. The
{func}`~mqt.yaqs.core.libraries.circuit_library.create_1d_fermi_hubbard_circuit`
provides a digital Trotter circuit with a chemical-potential option.

:::

:::{dropdown} Bose–Hubbard and the occupation cutoff

`local_dim` retains occupations from zero through `local_dim - 1` at each site.
The builder includes an on-site frequency, an interaction $U n_i(n_i-1)/2$, and
nearest-neighbor hopping with coefficient $-J$:

```python
local_dim = 3
bose = Hamiltonian.from_mpo(
    MPO.bose_hubbard(
        length=3,
        local_dim=local_dim,
        omega=1.0,
        hopping_j=0.2,
        hubbard_u=0.5,
    )
)
bose_state = State(3, physical_dimensions=local_dim)
```

Choose a large enough occupation cutoff for your preparation and dynamics. Check
convergence by increasing `local_dim` when higher occupations matter.

:::

## Supply a dense or sparse matrix

For a small custom operator, pass exactly one of `matrix`, `sparse_matrix`, or
`tensors`. Dense and sparse matrices must be finite, square, and Hermitian. The
manual constructor uses a uniform `physical_dimension`, defaulting to two; it
infers `length` from the matrix size when omitted.

(physical-site-ordering)=

### Physical-site ordering

Dense and sparse matrices use site 0 as the least-significant, fastest-varying
subsystem, matching Qiskit's qubit ordering. An operator acting on site 0 is
therefore the rightmost Kronecker factor:

```{code-cell} python
import numpy as np
from scipy.sparse import csr_matrix

identity = np.eye(2, dtype=complex)
pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
x_on_site_0 = np.kron(identity, pauli_x)

dense = Hamiltonian(matrix=x_on_site_0)
sparse = Hamiltonian(sparse_matrix=csr_matrix(x_on_site_0))
```

For local dimensions $d_0,d_1,\ldots$, the flat index for basis digits
$(s_0,s_1,\ldots)$ is $s_0+d_0s_1+d_0d_1s_2+\cdots$. This order applies to full
spatial state vectors and operator matrices. Local observable and noise matrices
use the order of their explicit site list: for example,
`Observable(np.kron(X, Z), sites=[0, 1])` means $X_0Z_1$. Custom adjacent
two-site noise matrices require ascending site lists. Circuit matrices follow
Qiskit's gate convention.

The initial state's representation selects the analog backend, independently of
the Hamiltonian's source data. See {doc}`representation_comparison` for the
supported choices.

```{warning}
Converting a sparse matrix to an MPO densifies the full operator. A sparse
source therefore does not avoid the full matrix allocation when you run with an
MPS state. For large MPS simulations, build an MPO directly with a preset, Pauli
terms, or custom tensors.
```

## Time-dependent Hamiltonians

Use `Hamiltonian.piecewise` for an analog quench: evolve under one static
Hamiltonian and then another, on a shared time grid. Every duration must be a
positive integer multiple of `dt`, and their sum must equal `elapsed_time`. All
pieces must have matching site counts and local dimensions:

```{code-cell} python
from mqt.yaqs import AnalogSimParams, Observable, Simulator

quench = Hamiltonian.piecewise([
    (hamiltonian, 0.2),
    (Hamiltonian.ising(length, J=1.0, g=2.0), 0.2),
])
params = AnalogSimParams(
    observables=[Observable("z", 0)], elapsed_time=quench.duration, dt=0.05
)

sim = Simulator(show_progress=False)
result = sim.run(state, quench, params)
```

The result contains nine sample times, including time zero, on one continuous
timeline. This path supports a single MPS state and the default TDVP evolution;
it also accepts a shared noise model. It does not support dense state
representations, BUG evolution, or list-of-state ensembles.

For digital gates, different time steps, or segment-specific noise, use a
`SimulationProgram`; see {doc}`digital_analog_simulation`. A piecewise
Hamiltonian has no single static MPO or matrix. Select a static entry from
`quench.pieces` when you need its operator.

## Advanced operator use

:::{dropdown} Custom MPO tensors and construction accuracy

Manual cores use `(left, right, physical_out, physical_in)` axes in ascending
site order. Neighboring bonds must match, exterior bonds must have dimension
one, and entries must be finite. The manual `Hamiltonian` constructor requires
uniform local dimensions. For example, these bond-one cores build the same $X_0$
operator as the matrix example:

```python
tensor_hamiltonian = Hamiltonian(
    tensors=[
        pauli_x.reshape(1, 1, 2, 2),
        identity.reshape(1, 1, 2, 2),
    ]
)
```

When you already have an MPO, use `Hamiltonian.from_mpo`. It references the same
MPO, checks its structure, and takes its local dimensions from the cores.
Wrapped MPOs and tensor inputs must represent a globally Hermitian operator;
YAQS does not perform a full-matrix Hermiticity check for these inputs.

Pauli builders expose `tol`, `max_bond_dim`, and `n_sweeps` for operator
compression. Dense-to-MPO conversion also uses an SVD cutoff, so the cached MPO
can approximate the source matrix. Set conversion options explicitly with
`MPO.from_matrix(matrix, d=2, cutoff=..., max_bond=...)` and wrap the result
when you need control over this approximation. These choices concern the
operator itself; simulation-parameter presets control the evolving state.

Static Hamiltonians cache converted forms. Call `ensure_mpo()` before reading
`.mpo`, or `ensure_sparse()` before reading `.sparse_matrix` if the form is not
already available. `to_matrix()` and `to_sparse_matrix()` return full operators
for small-system inspection. Create a new Hamiltonian when changing the
operator; cached forms do not track edits to the source data.

:::

:::{dropdown} Energy and long-range correlations of an MPS

Contract an MPS with a static Hamiltonian's MPO to obtain its energy. For an
arbitrary nonzero state, divide the raw contraction by the squared norm:

```python
hamiltonian.ensure_mpo()
norm_squared = state.mps.norm() ** 2
energy = state.mps.expect_mpo(hamiltonian.mpo) / norm_squared
```

`expect_mpo` returns the raw complex contraction without normalizing the state.
It uses the cached MPO, including any construction approximation. A Hermitian
energy is real up to numerical error.

The same method evaluates a full-chain Pauli product on separated sites:

```python
correlation = MPO()
correlation.from_pauli_sum(terms=[(1.0, "Z0 Z3")], length=length, n_sweeps=0)
zz = state.mps.expect_mpo(correlation) / norm_squared
z0 = state.mps.expect(Observable("z", 0)) / norm_squared
z3 = state.mps.expect(Observable("z", 3)) / norm_squared
connected = zz - z0 * z3
```

Here, `zz` is $\langle Z_0Z_3\rangle$ and `connected` subtracts
$\langle Z_0\rangle\langle Z_3\rangle$. Omitted sites act as identities. Use
`MPO.from_local_ops` instead when you already have one local matrix per site,
including identities between separated factors.

:::

With the operator and initial state prepared, continue with
{doc}`analog_simulation` for noisy dynamics or {doc}`simulation_parameters` for
accuracy and sampling choices. The {class}`~mqt.yaqs.Hamiltonian` API reference
lists the full constructor signatures.
