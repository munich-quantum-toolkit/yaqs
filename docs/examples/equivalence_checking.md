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

# Equivalence Checking

YAQS can test whether two quantum circuits implement the same unitary map, up to a **global
phase** and numerical tolerance. The public API is {class}`~mqt.yaqs.EquivalenceChecker`, which
forms the composed operator $W = U_2^\dagger U_1$ from the two circuits and checks whether $W$
is close to the identity.

For most workflows—comparing a high-level circuit to a transpiled variant, regression tests on
compiled circuits, or checking compiler passes—the **MPO backend** (`representation="mpo"`) is
the intended tool. It scales to larger qubit counts via tensor-network updates and SVD
truncation controlled by `threshold`. The **matrix backend** (`representation="matrix"`) is a
dense, tensorized reference useful on very small circuits; both backends target the same
equivalence criterion.

## Choosing a backend

| Backend                 | When to use                                                                      | Scaling                                                                             | Numerical knobs                          |
| ----------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------- |
| **`mpo`** (recommended) | Default for real circuits; long-range gates; anything beyond a handful of qubits | Polynomial in qubits for many structured circuits; memory grows with bond dimension | `threshold` (SVD truncation), `fidelity` |
| **`matrix`**            | Small-circuit checks, debugging, cross-checking the MPO path                     | Exponential in qubits ($4^n$ complex numbers for the dense operator tensor)         | `fidelity` only                          |
| **`auto`**              | Convenience: picks matrix for `num_qubits <= matrix_max_qubits`, otherwise MPO   | Same as the selected backend                                                        | Both when MPO is selected                |

```{note}
`representation="auto"` remains the constructor default, but **you should pass
`representation="mpo"` explicitly** when equivalence checking is part of a pipeline you care
about. Auto only avoids thinking about backend choice on tiny circuits; it does not change
the fact that MPO is the primary algorithm in YAQS.
```

With the default cutover of **7** qubits (`matrix_max_qubits` on {class}`~mqt.yaqs.EquivalenceChecker`), auto uses the
matrix backend only for circuits with **at most seven qubits**. From eight qubits upward, auto
selects MPO. Override the cutover with `matrix_max_qubits` if needed.

## What “equivalent” means

Two circuits $C_1$ and $C_2$ on $n$ qubits are reported as equivalent when their unitaries
$U_1$ and $U_2$ satisfy

$$
U_2^\dagger U_1 \approx e^{i\phi}\, I
$$

for some global phase $\phi$, within `fidelity`. On the **matrix** path, only **final**
measurements are stripped before building $U$; mid-circuit measurements raise an error.
Barriers are ignored on the matrix path. The **MPO** backend walks circuit DAGs directly
(measurements and barriers are skipped during zone extraction); mid-circuit measurements
are not supported for unitary equivalence on either backend. See
{cite:p}`sander2025_EquivalenceChecking` for the underlying MPO method.

`check` returns a dictionary:

| Key                               | Type                | Meaning                                                                   |
| --------------------------------- | ------------------- | ------------------------------------------------------------------------- |
| `equivalent`                      | `bool`              | Whether the circuits pass the identity test                               |
| `fidelity`                        | `float`             | Measured normalized overlap of $W=U_2^\dagger U_1$ with the identity      |
| `elapsed_time`                    | `float`             | Wall time in seconds                                                      |
| `representation`                  | `str`               | `"matrix"` or `"mpo"` — which backend ran                                 |
| `matrix`                          | `ndarray` or `None` | Dense composed operator $W$ as a $(2^n, 2^n)$ matrix; matrix backend only |
| `mpo`                             | `MPO` or `None`     | Composed operator on the MPO backend; `None` on matrix                    |
| `schmidt_values`                  | `ndarray` or `None` | Center-cut operator Schmidt values (`length // 2`); MPO backend only      |
| `center_cut_entanglement_entropy` | `float` or `None`   | Operator entanglement entropy at `length // 2`; MPO backend only          |
| `global_entanglement_entropy`     | `float` or `None`   | Sum of operator entanglement entropies over internal bonds; MPO only      |

## Parameters

{class}`~mqt.yaqs.EquivalenceChecker` stores settings on the instance; circuits are passed to
{meth}`~mqt.yaqs.EquivalenceChecker.check` each time.

- **`threshold`** (default `1e-13`): singular-value cutoff during MPO updates. Smaller values
  retain more bond dimension and are stricter; larger values speed up checks at the cost of
  accuracy.
- **`fidelity`** (default `1 - 1e-13`): minimum normalized overlap between $W$ and the
  identity (global phase removed). Used by **both** backends.
- **`representation`**: `"mpo"`, `"matrix"`, or `"auto"`.
- **`matrix_max_qubits`** (default **7**): only affects
  `"auto"`.
- **`parallel`** (default `True`): when enabled, checkerboard **MPO** pair updates run in a
  **thread pool** from 12 qubits upward (ignored for the matrix backend and below the cutoff).
- **`max_workers`** (default `None`): cap on worker threads when `parallel=True` (defaults to
  the machine CPU count via {func}`~mqt.yaqs.core.parallel_utils.available_cpus`).
- **`mp_context`**: reserved for a future process-pool mode; MPO parallelism uses threads today.

```{code-cell} ipython3
---
tags: [remove-output]
---
from mqt.yaqs import EquivalenceChecker

# Recommended: MPO for the circuits you care about
mpo_checker = EquivalenceChecker(
    representation="mpo",
    threshold=1e-6,
    fidelity=1 - 1e-13,
)

# Auto: matrix if num_qubits <= 7, else MPO
auto_checker = EquivalenceChecker(representation="auto")
```

## Loading from OpenQASM

{meth}`~mqt.yaqs.EquivalenceChecker.check` accepts OpenQASM 2 and OpenQASM 3 inputs directly —
no need to call Qiskit's loaders first. Pass a filesystem path, a `pathlib.Path`, or a raw
OpenQASM string (when the first substantive line declares `OPENQASM`):

```python
checker = EquivalenceChecker(representation="mpo")

# File paths (preferred when the program uses include directives)
result = checker.check("original.qasm", "transpiled.qasm")

# Raw source strings
result = checker.check(qasm_source_a, qasm_source_b)
```

OpenQASM 3 requires the optional package `qiskit-qasm3-import`
(`pip install mqt-yaqs[qasm3]`). The same path and string forms work with
{meth}`~mqt.yaqs.Simulator.run` for circuit simulation.

## Example: compare original and transpiled circuits

The workflow below builds a parameterized circuit, transpiles it to another gate set, and
checks equivalence with the **MPO backend**. This matches typical compiler-verification use
cases.

Define the number of qubits and circuit depth.

```{code-cell} ipython3
num_qubits = 5
depth = num_qubits
```

Create a TwoLocal circuit and decompose it.

```{code-cell} ipython3
---
tags: [remove-output]
---
from qiskit.circuit.library.n_local import TwoLocal

import numpy as np

circuit = TwoLocal(num_qubits, ["rx"], ["rzz"], entanglement="linear", reps=depth).decompose()
num_pars = len(circuit.parameters)
rng = np.random.default_rng()
values = rng.uniform(-np.pi, np.pi, size=num_pars)
circuit.assign_parameters(values, inplace=True)
circuit.measure_all()
```

Transpile the circuit to a new basis.

```{code-cell} ipython3
---
tags: [remove-output]
---
from qiskit import transpile

basis_gates = ["cz", "rz", "sx", "x", "id"]
transpiled_circuit = transpile(circuit, basis_gates=basis_gates, optimization_level=1)
```

Run equivalence checking with the MPO backend.

```{code-cell} ipython3
---
tags: [remove-output]
---
from mqt.yaqs import EquivalenceChecker

checker = EquivalenceChecker(representation="mpo", threshold=1e-6, fidelity=1 - 1e-13)
result = checker.check(circuit, transpiled_circuit)
```

The same pair with `representation="auto"` on this five-qubit example selects the matrix
backend because $5 \leq 7$. For a consistent pipeline, keep `representation="mpo"` as above.

```{code-cell} ipython3
---
tags: [remove-output]
---
auto_result = EquivalenceChecker(representation="auto").check(circuit, transpiled_circuit)
```

## Matrix backend (small circuits)

The matrix backend builds $W = U_2^\dagger U_1$ as a tensor with $2n$ indices of dimension 2
and applies local gate contractions. It uses the same trace-based identity test as the MPO
path. Memory and time grow as $\mathcal{O}(4^n)$, so this backend is practical only for very
small $n$.

Use it when:

- You want a dense reference on at most a few qubits.
- You are debugging the equivalence machinery itself.

```python
small_checker = EquivalenceChecker(representation="matrix", fidelity=1 - 1e-13)
```

Forcing `representation="matrix"` on large circuits is allowed but can exhaust memory; prefer
MPO instead.

## Parallel execution

Set `parallel=True` on {class}`~mqt.yaqs.EquivalenceChecker` to speed up **MPO** checks on circuits
where many independent updates can run at once. This is the default; below 12 qubits the
implementation keeps the serial path even when `parallel=True`, because thread overhead would
dominate. The matrix backend is always serial.

Within each checkerboard sweep, disjoint nearest-neighbor pairs update different MPO site
tensors and can be computed in parallel in a shared thread pool (one pool per `iterate()` call).
Temporal zones are still extracted from the DAGs serially; only the tensor contraction and SVD
step runs concurrently. Long-range gate handling stays serial in this version.

```{code-cell} ipython3
wide_checker = EquivalenceChecker(
    representation="mpo",
    max_workers=4,
)
```

Expect the largest gains on **wide** nearest-neighbor circuits (typically **12+ qubits**) where
each sweep has several disjoint pairs. Below 12 qubits the implementation keeps the serial path
even when `parallel=True`, because thread overhead would dominate.

## Performance notes

Internal benchmarks (`benchmarks/bench_equivalence_matrix_vs_mpo.py`) on random
`EfficientSU2` circuits show the matrix backend winning only at very small qubit counts; MPO
is faster from roughly eight qubits upward on those workloads. That aligns with the default
auto cutover at seven qubits: auto uses matrix only where it is still affordable, and MPO for
everything larger.

## Related topics

- {doc}`custom_gates` — Qiskit translation, matrix fallback, and TDVP generators
- {doc}`simulator_initialization` — running simulations with {class}`~mqt.yaqs.Simulator`
- {doc}`simulation_parameters` — presets and truncation for **simulation** (separate from
  equivalence `threshold`)
