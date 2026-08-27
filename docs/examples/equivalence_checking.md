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
$W = U_2^\dagger U_1$ from the two circuits and checks whether $W$ is close to
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
U_2^\dagger U_1 \approx e^{i\phi}\, I
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
| `fidelity`                        | `float`             | Measured normalized overlap of $W=U_2^\dagger U_1$ with the identity      |
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
- **`parallel`** (default `True`): when enabled, checkerboard **MPO** pair
  updates run in a **thread pool** from 12 qubits upward (ignored for the matrix
  backend, below the cutoff, and when a noise model is set).
- **`max_workers`** (default `None`): cap on worker threads when `parallel=True`
  (noiseless MPO checks), and on worker processes for noisy ensembles. When
  unset, noiseless MPO zone threads use
  `min(available_cpus(), number_of_work_items)`, and noisy trajectory processes
  use `max(1, available_cpus() - 1)`, where
  {func}`~mqt.yaqs.core.parallel_utils.available_cpus` respects
  `YAQS_MAX_WORKERS`, returns `1` under `PYTEST_XDIST_WORKER`, reads Slurm CPU
  limits when set, and falls back to CPU affinity or `os.cpu_count()` on the
  host.
- **`mp_context`**: start method for noisy-ensemble process pools (`"auto"`,
  `"fork"`, `"spawn"`). Noiseless MPO zone parallelism inside `iterate()` still
  uses in-process threads.

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

The matrix backend builds $W = U_2^\dagger U_1$ as a tensor with $2n$ indices of
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
default; below 12 qubits the implementation keeps the serial path even when
`parallel=True`, because thread overhead would dominate. The matrix backend is
always serial.

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

Expect the largest gains on **wide** nearest-neighbor circuits (typically
**12+ qubits**) where each sweep has several disjoint pairs. Below 12 qubits the
implementation keeps the serial path even when `parallel=True`, because thread
overhead would dominate.

## Noisy ensembles

A typical use of noisy equivalence checking is to ask how close a
**compiled, hardware-like** circuit still is to an **ideal** specification once
Pauli noise acts on the compiled gates. Pass a Pauli
{class}`~mqt.yaqs.NoiseModel` to {meth}`~mqt.yaqs.EquivalenceChecker.check`.
Each trajectory samples an explicit circuit $\widetilde G_r$ (local Pauli errors
after gates on two or more qubits on the **second** argument only) and runs the
same relative-operator check $Q_r = \widetilde G_r^\dagger G$ used in the
noiseless case. If $a_r = |\operatorname{Tr}(Q_r)| / d$ is a trajectory's
normalized root overlap, the random-unitary channel process fidelity is

```{math}
\widehat F_{\mathrm{pro}} = \frac{1}{N}\sum_{r=1}^{N} a_r^2.
```

For a noisy result, `fidelity` is this Monte Carlo sample mean and
`fidelity_error` is its empirical standard error.

Start from a small ideal circuit and transpile it to a device-style basis:

```{code-cell} ipython3
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit.circuit import QuantumCircuit

from mqt.yaqs import EquivalenceChecker, NoiseModel

ideal = QuantumCircuit(4)
for qubit in range(4):
    ideal.ry(0.4 * (qubit + 1), qubit)
for qubit in range(3):
    ideal.cx(qubit, qubit + 1)

compiled = transpile(
    ideal,
    basis_gates=["rz", "sx", "x", "cx"],
    optimization_level=1,
)
print(f"gates: ideal {ideal.size()}, compiled {compiled.size()}")
```

Without noise the two circuits implement the same unitary (up to global phase).
With Pauli-X noise on the compiled circuit, the estimated process fidelity drops
and the per-trajectory values show how often a sampled error knocks $Q_r$ off
the identity:

```{code-cell} ipython3
checker = EquivalenceChecker(representation="mpo", threshold=1e-6)
noiseless = checker.check(ideal, compiled)
noise = NoiseModel([
    {"name": "pauli_x", "sites": [qubit], "strength": 0.02} for qubit in range(4)
])
noisy = checker.check(
    ideal,
    compiled,
    noise_model=noise,
    num_traj=24,
    random_seed=0,
)
traj_process_fidelities = [traj["fidelity"] ** 2 for traj in noisy["trajectories"]]

print(f"noiseless: equivalent={noiseless['equivalent']}, fidelity={noiseless['fidelity']:.6f}")
print(
    "noisy:     "
    f"equivalent={noisy['equivalent']}, "
    f"process fidelity={noisy['fidelity']:.4f} "
    f"+/- {noisy['fidelity_error']:.4f}"
)
print(f"trajectories: {noisy['num_traj']}")

fig, ax = plt.subplots(figsize=(5.5, 3.2), layout="constrained")
ax.hist(traj_process_fidelities, bins=12, range=(0.0, 1.0), color="C0", alpha=0.85)
ax.axvline(noiseless["fidelity"] ** 2, color="0.35", ls="--", label="noiseless compiled")
ax.axvline(noisy["fidelity"], color="C1", ls="-", label="noisy sample mean")
ax.set_xlabel("process-fidelity sample $a_r^2$")
ax.set_ylabel("trajectories")
ax.set_title("Ideal vs compiled circuit with Pauli-X noise")
ax.legend(frameon=False)
```

A noisy `check` returns
{class}`~mqt.yaqs.equivalence_checker.EquivalenceEnsembleResult` with the same
primary `equivalent` and `fidelity` keys as a noiseless check, plus
`fidelity_error`, `num_traj`, and `trajectories`. Here `fidelity` is the mean of
the squared trajectory overlaps, and `fidelity_error` is its Monte Carlo
standard error; the latter is `None` for one trajectory. Noisy `equivalent` only
means that this observed mean is at least `checker.fidelity**2`. It is not an
equivalence certificate or a confidence-level decision. Compare noisy and
noiseless fidelities by squaring the noiseless value. On the MPO backend the
ensemble also averages operator entanglement. Distribution-valued strengths are
resolved once per `check` call.

Non-Pauli processes such as `raising`/`lowering` and scheduled jumps are
rejected; those remain on the TJM simulation path. Independent trajectories run
in a process pool with serial MPO updates inside each worker (`max_workers` caps
the pool). Checkerboard zone threads are not used when a noise model is set.

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
