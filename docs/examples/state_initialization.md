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

YAQS separates **what you specify** (a [`State`](mqt.yaqs.core.data_structures.state.State)) from **how evolution runs** ([`AnalogSimParams`](mqt.yaqs.core.data_structures.simulation_parameters.AnalogSimParams), Hamiltonian, noise).

| Layer | Role |
|-------|------|
| **`State`** | User-facing initial condition: length, preset name, optional raw data, and **which representation** to evolve in (`"mps"`, `"vector"`, or `"density_matrix"`). |
| **`MPS`** | Internal tensor network; built when needed for TJM/circuits or when dense data must come from an MPS. |

**Representation lives on `State`, not on `AnalogSimParams`.** For analog Hamiltonian evolution, set `State(..., representation=...)` (default `"mps"` → tensor jump method). [`run`](mqt.yaqs.simulator.run) calls [`State.encode`](mqt.yaqs.core.data_structures.state.State.encode) from that field before dispatching to TJM, MCWF, or Lindblad.

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.state import State

preset = State(4, initial="x+")
print("preset representation:", preset.representation)  # default "mps"

mcwf_state = State(4, initial="zeros", representation="vector")
print("MCWF representation:", mcwf_state.representation)
```

## `State` versus `MPS`

Use `State` in [`run`](mqt.yaqs.simulator.run). Use [`MPS`](mqt.yaqs.core.data_structures.networks.MPS) directly only for low-level tensor-network code or to wrap an existing MPS.

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.networks import MPS

mps = MPS(4, state="zeros")
wrapped = State.from_mps(mps)
assert wrapped.representation == "mps"
assert wrapped.mps is mps
```

**Circuit simulation** still evolves an MPS internally: `run` always encodes the state as `"mps"` for digital backends, regardless of `State.representation` on analog-style presets.

## How `representation` is chosen

| How you build `State` | `representation` | Encoded at `__init__`? |
|-----------------------|--------------------|-------------------------|
| Preset only (`length`, `initial=`, …) | Default `"mps"`; override with `representation="vector"` or `"density_matrix"` | No (lazy until `encode()` / `run`) |
| `tensors=` (MPS cores) | Inferred `"mps"` — do **not** pass `representation=` | Yes |
| `vector=` | Inferred `"vector"` | Yes |
| `density_matrix=` | Inferred `"density_matrix"` | Yes |

```{code-cell} ipython3
import numpy as np

vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
from_vector = State(vector=vec)
print(from_vector.representation, from_vector._encoded_as)

# representation= is only for preset states
lindblad_ready = State(2, initial="zeros", representation="density_matrix")
print(lindblad_ready.representation)
```

## Preset product states

Presets match `MPS(..., state=...)` names: `"zeros"`, `"ones"`, `"x+"`, `"Neel"`, `"wall"`, `"basis"`, `"random"`, etc.

```{code-cell} ipython3
neel = State(4, initial="Neel")
neel.encode("vector")  # or: State(4, initial="Neel", representation="vector") then run(...)
print(np.round(np.abs(neel.vector), 3))
```

For **product** presets, `encode("vector")` / `encode("density_matrix")` can build dense data **without** keeping an MPS in memory (useful for small systems with MCWF or Lindblad).

**Entangled** presets (e.g. `"haar-random"`) still build an MPS when you ask for a dense representation.

```{code-cell} ipython3
entangled = State(4, initial="haar-random", pad=4)
entangled.encode("vector")
assert entangled.mps is not None  # MPS was materialized on this path
```

Reproducible `"random"` presets: pass `seed=` on `State`.

```{code-cell} ipython3
a = State(3, initial="random", seed=7)
b = State(3, initial="random", seed=7)
a.encode("vector")
b.encode("vector")
np.testing.assert_allclose(a.vector, b.vector)
```

## Manual initialization

Pass **exactly one** of `tensors`, `vector`, or `density_matrix`. Representation is **inferred**; do not pass `representation=`. Preset-only kwargs (`initial`, `pad`, `basis_string`, `seed`) cannot be combined with manual data.

### MPS cores (`tensors=`)

```{code-cell} ipython3
mps_ref = MPS(3, state="zeros")
spec = State(tensors=list(mps_ref.tensors))
print(spec.representation)
np.testing.assert_allclose(spec.mps.to_vec(), mps_ref.to_vec(), atol=1e-10)
```

### Dense state vector (`vector=`)

`length` is inferred when the Hilbert-space dimension is a power of two.

```{code-cell} ipython3
vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)  # |00>
spec = State(vector=vec)
print(spec.length, spec.representation, np.linalg.norm(spec.vector))
```

### Density matrix (`density_matrix=`)

```{code-cell} ipython3
rho = np.diag([1.0, 0.0, 0.0, 0.0]).astype(np.complex128)
spec = State(density_matrix=rho)
print(spec.representation, spec.density_matrix.shape, np.trace(spec.density_matrix))
```

A `State` created only with `vector=` or `density_matrix=` cannot be turned into an MPS via `encode("mps")`; use `tensors=` or a preset instead.

## `encode()` and inspection

- After manual init, data is already encoded; properties `.mps`, `.vector`, or `.density_matrix` are ready.
- Preset-only states encode on first `encode()` or when passed to `run`.
- `encode()` with **no argument** uses [`State.representation`](mqt.yaqs.core.data_structures.state.State.representation).

| `representation` | Property | Backend (analog) |
|------------------|----------|------------------|
| `"mps"` (default) | `.mps` | TJM (`analog_tjm_1` / `analog_tjm_2`) |
| `"vector"` | `.vector` | MCWF |
| `"density_matrix"` | `.density_matrix` | Lindblad (small systems) |

`encode` is idempotent for the same representation.

```{code-cell} ipython3
spec = State(3, initial="zeros", representation="vector")
spec.encode()  # uses spec.representation
v1 = spec.vector.copy()
spec.encode()
np.testing.assert_allclose(spec.vector, v1)
```

Dense product preset vs MPS reference:

```{code-cell} ipython3
spec = State(3, initial="x+")
spec.encode("vector")
ref = MPS(3, state="x+").to_vec()
ref /= np.linalg.norm(ref)
np.testing.assert_allclose(spec.vector, ref, atol=1e-10)
```

## Analog simulation with `run()`

Set **`representation` on `State`**, not on `AnalogSimParams`. `run` calls `state.encode()` internally.

### Default: MPS / TJM

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.simulator import run

L = 3
H = MPO.ising(L, J=1.0, g=0.5)
obs = Observable("z", sites=[0])

state_mps = State(L, initial="zeros")  # representation="mps" by default
params = AnalogSimParams(
    observables=[obs],
    elapsed_time=0.2,
    dt=0.05,
    show_progress=False,
)
run(state_mps, H, params, noise_model=None)
print("TJM Z_0:", obs.results[-1])
```

### MCWF (`representation="vector"`)

Product presets can skip materializing an MPS when `run` encodes to vector.

```{code-cell} ipython3
state_vec = State(L, initial="zeros", representation="vector")
obs_vec = Observable("z", sites=[0])
params_vec = AnalogSimParams(
    observables=[obs_vec],
    elapsed_time=0.2,
    dt=0.05,
    show_progress=False,
)
run(state_vec, H, params_vec, None)
print("MCWF Z_0:", obs_vec.results[-1])
```

### Lindblad (`representation="density_matrix"`)

```{code-cell} ipython3
state_dm = State(L, initial="zeros", representation="density_matrix")
obs_dm = Observable("z", sites=[0])
params_dm = AnalogSimParams(
    observables=[obs_dm],
    elapsed_time=0.2,
    dt=0.05,
    show_progress=False,
)
run(state_dm, H, params_dm, None)
print("Lindblad Z_0:", obs_dm.results[-1])
```

See {doc}`solver_comparison` for a side-by-side comparison of the three representations on the same Hamiltonian.

### Passing dense data directly

If you already have $|\psi\rangle$ or $\rho$, pass `vector=` or `density_matrix=` — representation is inferred and no MPS is required for `run`:

```{code-cell} ipython3
psi = np.zeros(2**L, dtype=np.complex128)
psi[0] = 1.0
state_from_vec = State(vector=psi)
run(state_from_vec, H, params_vec, None)
print("From vector=, MCWF Z_0:", obs_vec.results[-1])
```

## Practical limits

- **Memory**: `vector` scales as $2^N$; `density_matrix` as $2^{2N}$. Prefer `representation="mps"` for longer chains.
- **Entangled presets**: `"haar-random"` needs an MPS to form dense data.
- **Circuits**: digital `run` always uses an MPS; use presets or `tensors=`, not raw `vector=` / `density_matrix=`, unless you only care about analog paths.
- **Ensemble runs**: `list[State]` for deterministic unitary ensembles requires each member with `representation="mps"`.
- **`get_state`**: not supported with `representation="density_matrix"` or with stochastic noise (unchanged).

For MPO/TJM details without `State`, see {doc}`analog_simulation` and the [`MPS`](mqt.yaqs.core.data_structures.networks.MPS) API reference.
