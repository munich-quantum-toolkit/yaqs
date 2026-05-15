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

# Initializing quantum states

YAQS separates **how you specify** an initial condition from **how the simulator represents** it during evolution.

- **`State`** ([`mqt.yaqs.core.data_structures.state.State`](mqt.yaqs.core.data_structures.state.State)) is the user-facing object passed to [`run`](mqt.yaqs.simulator.run) for analog and circuit simulations.
- **`MPS`** ([`mqt.yaqs.core.data_structures.networks.MPS`](mqt.yaqs.core.data_structures.networks.MPS)) is the tensor-network data structure used internally for MPS-based backends and for low-level algorithms.

For analog Hamiltonian evolution, set `AnalogSimParams.representation` to `"mps"`, `"vector"`, or `"density_matrix"`. `run` calls [`State.encode`](mqt.yaqs.core.data_structures.state.State.encode) with that choice before dispatching to the appropriate backend.

## `State` versus `MPS`

Use `State` at the API boundary and `MPS` when you work directly with tensor networks (custom algorithms, tests, or wrapping an existing MPS).

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.state import State

# Preset product state (same names as MPS(..., state=...))
psi = State(4, initial="x+")

# Wrap an existing MPS
mps = MPS(4, state="zeros")
wrapped = State.from_mps(mps)
assert wrapped.mps is mps
```

Circuit simulation still requires an MPS internally: `run` encodes the state as `"mps"` and uses [`State.mps`](mqt.yaqs.core.data_structures.state.State.mps).

## Preset product states

If you do not pass `tensors`, `vector`, or `density_matrix`, `State` builds from `initial=` using the same presets as `MPS` (`"zeros"`, `"ones"`, `"x+"`, `"Neel"`, `"wall"`, `"basis"`, `"random"`, …).

```{code-cell} ipython3
import numpy as np
from mqt.yaqs.core.data_structures.state import State

neel = State(4, initial="Neel")
neel.encode("vector")
# Site 0 is the least significant index in the dense vector (matches MPS.to_vec()).
print(np.round(np.abs(neel.vector), 3))
```

For product presets, `encode("vector")` and `encode("density_matrix")` can build the dense object **without** constructing an MPS (faster for small systems used with MCWF or Lindblad).

Entangled presets such as `"haar-random"` still go through an MPS when you request a dense representation.

```{code-cell} ipython3
entangled = State(4, initial="haar-random", pad=4)
entangled.encode("vector")
# Entangled presets materialize an MPS internally when building the dense vector.
_ = entangled.mps  # succeeds: MPS was built
```

Use `seed=` for reproducible `"random"` presets when encoding to a vector.

```{code-cell} ipython3
a = State(3, initial="random", seed=7)
b = State(3, initial="random", seed=7)
a.encode("vector")
b.encode("vector")
np.testing.assert_allclose(a.vector, b.vector)
```

## Manual initialization

Exactly one of `tensors`, `vector`, or `density_matrix` may be supplied (in that priority order in the signature). Each is **encoded automatically** at construction. Preset-only options (`initial`, `pad`, `basis_string`, `seed`) must not be passed with manual data.

### MPS cores (`tensors=`)

```{code-cell} ipython3
mps_ref = MPS(3, state="zeros")
spec = State(tensors=list(mps_ref.tensors))
np.testing.assert_allclose(spec.mps.to_vec(), mps_ref.to_vec(), atol=1e-10)
```

### Dense state vector (`vector=`)

`length` can be inferred when the Hilbert-space dimension is a power of two.

```{code-cell} ipython3
vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)  # |00>
spec = State(vector=vec)
print(spec.length, np.linalg.norm(spec.vector))
```

### Density matrix (`density_matrix=`)

```{code-cell} ipython3
rho = np.diag([1.0, 0.0, 0.0, 0.0]).astype(np.complex128)
spec = State(density_matrix=rho)
print(spec.density_matrix.shape, np.trace(spec.density_matrix))
```

You cannot build an MPS from a `State` that was created only with `vector=` or `density_matrix=`; use `tensors=` or a preset instead.

## `encode()` and inspection

Preset-only `State(length, initial=...)` objects are not encoded until you call `encode(representation)` or pass them to `run` (which encodes from `AnalogSimParams.representation`).

| `representation` | Property | When available |
|--------------------|----------|----------------|
| `"mps"` | `.mps` | After `tensors=` init, `encode("mps")`, or `run` |
| `"vector"` | `.vector` | After `vector=` init, `encode("vector")`, or `run` |
| `"density_matrix"` | `.density_matrix` | After `density_matrix=` init, `encode("density_matrix")`, or `run` |

`encode` is idempotent: calling it again with the same argument is a no-op.

```{code-cell} ipython3
spec = State(3, initial="zeros")
spec.encode("vector")
v1 = spec.vector.copy()
spec.encode("vector")
np.testing.assert_allclose(spec.vector, v1)
```

Compare a preset dense vector to the MPS reference:

```{code-cell} ipython3
spec = State(3, initial="x+")
spec.encode("vector")
ref = MPS(3, state="x+").to_vec()
ref /= np.linalg.norm(ref)
np.testing.assert_allclose(spec.vector, ref, atol=1e-10)
```

## Analog simulation and `run()`

Pass a `State` and set `AnalogSimParams.representation`. `run` encodes automatically.

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.simulator import run

L = 3
H = MPO.ising(L, J=1.0, g=0.5)
obs = Observable("z", sites=[0])

state = State(L, initial="zeros")
params = AnalogSimParams(
    observables=[obs],
    elapsed_time=0.2,
    dt=0.05,
    representation="vector",
    show_progress=False,
)
run(state, H, params, noise_model=None)
print("Z_0 at final time:", obs.results[-1])
```

The same preset works with `representation="density_matrix"` (exact master equation on small systems) or `"mps"` (tensor jump method). See {doc}`solver_comparison` for a side-by-side benchmark.

```{code-cell} ipython3
state_dm = State(L, initial="zeros")
params_dm = AnalogSimParams(
    observables=[Observable("z", sites=[0])],
    elapsed_time=0.2,
    dt=0.05,
    representation="density_matrix",
    show_progress=False,
)
run(state_dm, H, params_dm, None)
print("density_matrix Z_0:", params_dm.observables[0].results[-1])
```

## Practical limits

- **System size**: `vector` and `density_matrix` scale as $2^N$ and $2^{2N}$ in memory; warnings are emitted for large `N`. Prefer `representation="mps"` for longer chains.
- **Entangled presets**: `"haar-random"` needs an MPS to form dense data.
- **Circuits**: digital simulation always uses an MPS; pass `State` presets or `tensors=`, not raw `vector=` / `density_matrix=`, unless you first `encode("mps")` from tensors.
- **Ensemble runs**: a `list[State]` for deterministic unitary ensembles is evolved in MPS form; each member should be encodable as `"mps"`.
- **Stochastic output state**: `get_state=True` with noisy open-system trajectories is not supported (same as before).

For algorithm-level use of MPS/MPO directly (without `State`), see {doc}`analog_simulation` and the API reference for [`MPS`](mqt.yaqs.core.data_structures.networks.MPS).
