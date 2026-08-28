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

YAQS can test whether two quantum circuits implement the same unitary map, up to
a **global phase** and numerical tolerance. The public API is
{class}`~mqt.yaqs.EquivalenceChecker`, which forms the composed operator
$W = U_1 U_2^\dagger$ from the two circuits and checks whether $W$ is close to
the identity.

For most workflows—comparing a high-level circuit to a transpiled variant,
regression tests on compiled circuits, or checking compiler passes—the
**MPO backend** (`representation="mpo"`) is the intended tool. It scales to
larger qubit counts via tensor-network updates and SVD truncation controlled by
`threshold`. The **matrix backend** (`representation="matrix"`) is a dense,
tensorized reference useful on very small circuits; both backends target the
same equivalence criterion.

## Choosing a backend

| Backend                 | When to use                                                                      | Scaling                                                                             | Numerical knobs                          |
| ----------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------- |
| **`mpo`** (recommended) | Default for real circuits; long-range gates; anything beyond a handful of qubits | Polynomial in qubits for many structured circuits; memory grows with bond dimension | `threshold` (SVD truncation), `fidelity` |
| **`matrix`**            | Small-circuit checks, debugging, cross-checking the MPO path                     | Exponential in qubits ($4^n$ complex numbers for the dense operator tensor)         | `fidelity` only                          |
| **`auto`**              | Convenience: picks matrix for `num_qubits <= matrix_max_qubits`, otherwise MPO   | Same as the selected backend                                                        | Both when MPO is selected                |

```{note}
`representation="auto"` remains the constructor default, but
**you should pass `representation="mpo"` explicitly** when equivalence checking
is part of a pipeline you care about. Auto only avoids thinking about backend
choice on tiny circuits; it does not change the fact that MPO is the primary
algorithm in YAQS.
```

With the default cutover of **7** qubits (`matrix_max_qubits` on
{class}`~mqt.yaqs.EquivalenceChecker`), auto uses the matrix backend only for
circuits with **at most seven qubits**. From eight qubits upward, auto selects
MPO. Override the cutover with `matrix_max_qubits` if needed.

## What “equivalent” means

Two circuits $C_1$ and $C_2$ on $n$ qubits are reported as equivalent when their
unitaries $U_1$ and $U_2$ satisfy

```{math}
U_1 U_2^\dagger \approx e^{i\phi}\, I
```

for some global phase $\phi$, within `fidelity`. On the **matrix** path, only
**final** measurements are stripped before building $U$; mid-circuit
measurements raise an error. Barriers are ignored on the matrix path. The
**MPO** backend walks circuit DAGs directly (measurements and barriers are
skipped during zone extraction); mid-circuit measurements are not supported for
unitary equivalence on either backend. Gates on more than two qubits (for
example `ccx`) are supported on the matrix backend only; the MPO backend rejects
them with a `ValueError`. Unknown unitaries translate via the matrix fallback,
which supports at most eight qubits (see {doc}`custom_gates`). See
{cite:p}`sander2025_EquivalenceChecking` for the underlying MPO method.

A noiseless `check` returns a dictionary:

| Key                               | Type                | Meaning                                                                   |
| --------------------------------- | ------------------- | ------------------------------------------------------------------------- |
| `equivalent`                      | `bool`              | Whether the circuits pass the identity test                               |
| `fidelity`                        | `float`             | Measured normalized overlap of $W=U_1U_2^\dagger$ with the identity       |
| `elapsed_time`                    | `float`             | Wall time in seconds                                                      |
| `representation`                  | `str`               | `"matrix"` or `"mpo"` — which backend ran                                 |
| `matrix`                          | `ndarray` or `None` | Dense composed operator $W$ as a $(2^n, 2^n)$ matrix; matrix backend only |
| `mpo`                             | `MPO` or `None`     | Composed operator on the MPO backend; `None` on matrix                    |
| `schmidt_values`                  | `ndarray` or `None` | Center-cut operator Schmidt values (`length // 2`); MPO backend only      |
| `center_cut_entanglement_entropy` | `float` or `None`   | Operator entanglement entropy at `length // 2`; MPO backend only          |
| `global_entanglement_entropy`     | `float` or `None`   | Sum of operator entanglement entropies over internal bonds; MPO only      |

## Parameters

{class}`~mqt.yaqs.EquivalenceChecker` stores settings on the instance; circuits
are passed to {meth}`~mqt.yaqs.EquivalenceChecker.check` each time.

- **`threshold`** (default `1e-13`): singular-value cutoff during MPO updates.
  Smaller values retain more bond dimension and are stricter; larger values
  speed up checks at the cost of accuracy.
- **`fidelity`** (default `1 - 1e-13`): minimum normalized overlap between $W$
  and the identity (global phase removed). It must be finite and between `0` and
  `1`. Used by **both** backends. A noisy ensemble squares this value for its
  process-fidelity threshold.
- **`representation`**: `"mpo"`, `"matrix"`, or `"auto"`.
- **`matrix_max_qubits`** (default **7**): only affects `"auto"`.
- **`parallel`** (default `True`): enables checkerboard **MPO** pair updates in
  a **thread pool** from 12 qubits upward and allows noisy trajectories to use a
  **process pool**. Set it to `False` to keep either path serial.
- **`max_workers`** (default `None`): cap on worker threads when `parallel=True`
  (noiseless MPO checks), and on worker processes for noisy ensembles. Worker
  counts also respect the available CPUs and number of trajectories.
- **`mp_context`**: start method for noisy-ensemble process pools (`"auto"`,
  `"fork"`, `"spawn"`) when one is created. Noiseless MPO zone parallelism
  inside `iterate()` still uses in-process threads.

```{code-cell} ipython3
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

{meth}`~mqt.yaqs.EquivalenceChecker.check` accepts OpenQASM 2 and OpenQASM 3
inputs directly — no need to call Qiskit's loaders first. Pass a filesystem
path, a `pathlib.Path`, or a raw OpenQASM string (when the first substantive
line declares `OPENQASM`):

```python
checker = EquivalenceChecker(representation="mpo")

# File paths (preferred when the program uses include directives)
result = checker.check("original.qasm", "transpiled.qasm")

# Raw source strings
result = checker.check(qasm_source_a, qasm_source_b)
```

OpenQASM 3 requires the optional package `qiskit-qasm3-import`
(`uv pip install mqt-yaqs[qasm3]`). The same path and string forms work with
{meth}`~mqt.yaqs.Simulator.run` for circuit simulation.

## Example: compare original and transpiled circuits

The workflow below builds a parameterized circuit, transpiles it to another gate
set, and checks equivalence with the **MPO backend**. This matches typical
compiler-verification use cases.

Define the number of qubits and circuit depth.

```{code-cell} ipython3
num_qubits = 5
depth = num_qubits
```

Create a TwoLocal circuit and decompose it.

```{code-cell} ipython3
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
from qiskit import transpile

basis_gates = ["cz", "rz", "sx", "x", "id"]
transpiled_circuit = transpile(circuit, basis_gates=basis_gates, optimization_level=1)
```

Run equivalence checking with the MPO backend.

```{code-cell} ipython3
from mqt.yaqs import EquivalenceChecker

checker = EquivalenceChecker(representation="mpo", threshold=1e-6, fidelity=1 - 1e-13)
result = checker.check(circuit, transpiled_circuit)
```

The same pair with `representation="auto"` on this five-qubit example selects
the matrix backend because $5 \leq 7$. For a consistent pipeline, keep
`representation="mpo"` as above.

```{code-cell} ipython3
auto_result = EquivalenceChecker(representation="auto").check(circuit, transpiled_circuit)
```

## Matrix backend (small circuits)

The matrix backend builds $W = U_1 U_2^\dagger$ as a tensor with $2n$ indices of
dimension 2 and applies local gate contractions. It uses the same trace-based
identity test as the MPO path. Memory and time grow as $\mathcal{O}(4^n)$, so
this backend is practical only for very small $n$.

Use it when:

- You want a dense reference on at most a few qubits.
- You are debugging the equivalence machinery itself.

```python
small_checker = EquivalenceChecker(representation="matrix", fidelity=1 - 1e-13)
```

Forcing `representation="matrix"` on large circuits is allowed but can exhaust
memory; prefer MPO instead.

## Parallel execution

Set `parallel=True` on {class}`~mqt.yaqs.EquivalenceChecker` to speed up **MPO**
checks on circuits where many independent updates can run at once. This is the
default; below 12 qubits the implementation keeps a single noiseless check
serial even when `parallel=True`, because thread overhead would dominate. A
single matrix check is also serial. When a noise model is supplied, independent
matrix or MPO trajectories can instead run across processes.

Within each checkerboard sweep, disjoint nearest-neighbor pairs update different
MPO site tensors and can be computed in parallel in a shared thread pool (one
pool per `iterate()` call). Temporal zones are still extracted from the DAGs
serially; only the tensor contraction and SVD step runs concurrently. Long-range
gate handling stays serial in this version.

```{code-cell} ipython3
wide_checker = EquivalenceChecker(
    representation="mpo",
    max_workers=4,
)
```

(equivalence-noise-model)=

## Comparing with a noise model

Passing a {class}`~mqt.yaqs.NoiseModel` asks how close a
**compiled, hardware-like** circuit remains to an **ideal** specification under
sampled noise. Each trajectory materializes a stochastic realization of the
noise model on the compiled circuit. This gives a Monte Carlo comparison rather
than an exact noisy-channel equivalence certificate.

The current circuit-sampling path accepts processes that normalize to one-site
Pauli operators or two-site Pauli products. Other process types and scheduled
jumps are rejected; those remain available through the simulator.

Noise is sampled onto the **second** circuit argument only. A supported
two-qubit unitary gate is a noise opportunity; single-qubit gates, gates on
three or more qubits, barriers, and measurements are not. A process is eligible
when its complete site support is contained in the gate support. The selected
equivalence backend must also support the original gate.

Within `EquivalenceChecker`, each resolved `strength` is a dimensionless branch
probability $p_i$ at every eligible gate. Processes with the same exact site
support form one categorical draw: process $i$ occurs with probability $p_i$,
and no process from that support occurs with probability $1-\sum_i p_i$.
Consequently, each same-support sum must be at most one. Different exact
supports are sampled independently, so multiple errors may follow one gate. This
also applies to overlapping supports such as `[0]` and `[0, 1]`, which may both
be selected in one trajectory.

For an isotropic Pauli error with total probability $p$ on each gate qubit,
assign `strength=p/3` to X, Y, and Z on that one-qubit support. The identity
then has probability $1-p$ on each qubit, and the per-qubit draws are
independent.

Writing $U_{\mathrm{ideal}}$ for the first circuit and $U_{\mathrm{noisy},r}$
for trajectory $r$ of the second, the relative operator has the order

```{math}
Q_r = U_{\mathrm{ideal}} U_{\mathrm{noisy},r}^\dagger.
```

If $a_r = |\operatorname{Tr}(Q_r)| / d$ is the normalized root overlap, the
sampled channel's process fidelity $F_{\mathrm{pro}}=\mathbb E[a_r^2]$ is
estimated by

```{math}
\widehat F_{\mathrm{pro}} = \frac{1}{N}\sum_{r=1}^{N} a_r^2.
```

For a noisy result, `fidelity` is this Monte Carlo sample mean. For $N>1$, the
reported sampling uncertainty is

```{math}
\mathtt{fidelity\_error}
= \sqrt{\frac{1}{N(N-1)}\sum_{r=1}^{N}
\left(a_r^2-\widehat F_{\mathrm{pro}}\right)^2}.
```

For one trajectory, `fidelity_error` is `None` because sampling uncertainty
cannot be estimated.

Apply noise to the transpiled circuit from the earlier example. The noiseless
pair remains equivalent, while the noisy run estimates its process fidelity:

```{code-cell} ipython3
from mqt.yaqs import NoiseModel

checker = EquivalenceChecker(representation="mpo", threshold=1e-6)
noiseless = checker.check(circuit, transpiled_circuit)
noise = NoiseModel([
    {"name": "pauli_x", "sites": [qubit], "strength": 0.02} for qubit in range(num_qubits)
])
noisy = checker.check(
    circuit,
    transpiled_circuit,
    noise_model=noise,
    num_traj=24,
    random_seed=0,
)

print(f"noiseless: equivalent={noiseless['equivalent']}, fidelity={noiseless['fidelity']:.6f}")
print(
    "noisy:     "
    f"sample threshold passed={noisy['equivalent']}, "
    f"process fidelity={noisy['fidelity']:.4f} "
    f"+/- {noisy['fidelity_error']:.4f}"
)
print(f"trajectories: {noisy['num_traj']}")
```

Here `strength=0.02` means a 2% X-error probability on that qubit after each
eligible two-qubit gate containing it. It is neither a Lindblad rate nor a 2%
error probability for the complete circuit.

A noisy `check` returns
{class}`~mqt.yaqs.equivalence_checker.EquivalenceEnsembleResult` with the same
primary `equivalent` and `fidelity` keys as a noiseless check, plus
`fidelity_error`, `num_traj`, and `trajectories`. Noisy `equivalent` compares the
sample mean with `checker.fidelity**2`; it is not an exact certificate. On the
MPO backend, the ensemble also averages operator entanglement and concatenates
the center-cut Schmidt spectra in `schmidt_values`.

Distribution-valued strengths are resolved once per `check` call, so every
trajectory uses the same resolved probabilities. The checker then validates the
same-support sums; an out-of-range draw raises `ValueError`. A nonnegative
`random_seed` makes the ordered trajectory results reproducible independently
of process scheduling.

## Performance notes

Internal benchmarks (`benchmarks/bench_equivalence_matrix_vs_mpo.py`) on random
`EfficientSU2` circuits show the matrix backend winning only at very small qubit
counts; MPO is faster from roughly eight qubits upward on those workloads. That
aligns with the default auto cutover at seven qubits: auto uses matrix only
where it is still affordable, and MPO for everything larger.

## Related topics

- {doc}`realistic_noise_models` — Pauli and dissipative process names, disorder
- {doc}`custom_gates` — Qiskit translation, matrix fallback, and TDVP generators
- {doc}`simulator_initialization` — running simulations with
  {class}`~mqt.yaqs.Simulator`
- {doc}`simulation_parameters` — presets and truncation for **simulation**
  (separate from equivalence `threshold`)
