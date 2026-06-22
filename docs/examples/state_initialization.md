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

# Initializing Quantum States

YAQS separates **what you specify** (a [`State`](mqt.yaqs.core.data_structures.state.State)) from **how evolution runs** ([`AnalogSimParams`](mqt.yaqs.core.data_structures.simulation_parameters.AnalogSimParams), Hamiltonian, noise).

| Layer       | Role                                                                                                                                                           |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`State`** | User-facing initial condition: length, preset name, optional raw data, and **which representation** to evolve in (`"mps"`, `"vector"`, or `"density_matrix"`). |
| **`MPS`**   | Internal tensor network; used by the simulator when needed. Prefer [`State`](mqt.yaqs.core.data_structures.state.State) in application code.                   |

**Workflow:** build a [`State`](mqt.yaqs.core.data_structures.state.State) and a [`Hamiltonian`](mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian) once (both materialize at construction), then pass them to [`Simulator.run`](mqt.yaqs.Simulator) — including in parameter loops.

```{code-cell} ipython3
from mqt.yaqs import State

preset = State(4, initial="x+")
print("preset representation:", preset.representation)  # default "mps"

mcwf_state = State(4, initial="zeros", representation="vector")
print("MCWF representation:", mcwf_state.representation)
```

## `State` versus `MPS`

Use `State` in [`Simulator.run`](mqt.yaqs.Simulator). Use [`MPS`](mqt.yaqs.core.data_structures.mps.MPS) directly only for low-level tensor-network code, or wrap an existing MPS with [`State.from_mps`](mqt.yaqs.core.data_structures.state.State.from_mps).

**Circuit simulation** requires `representation="mps"` (the preset default). `Simulator.run` with `StrongSimParams` / `WeakSimParams` rejects vector and density-matrix states.

## How `representation` is chosen

| How you build `State`                 | `representation`                                                               |
| ------------------------------------- | ------------------------------------------------------------------------------ |
| Preset only (`length`, `initial=`, …) | Default `"mps"`; override with `representation="vector"` or `"density_matrix"` |
| `tensors=` (MPS cores)                | Inferred `"mps"` — do **not** pass `representation=`                           |
| `vector=`                             | Inferred `"vector"`                                                            |
| `density_matrix=`                     | Inferred `"density_matrix"`                                                    |

```{code-cell} ipython3
import numpy as np

vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
from_vector = State(vector=vec)
print(from_vector.representation)

lindblad_ready = State(2, initial="zeros", representation="density_matrix")
print(lindblad_ready.representation)
```

## Preset product states

Presets match `MPS(..., state=...)` names: `"zeros"`, `"ones"`, `"x+"`, `"Neel"`, `"wall"`, `"basis"`, `"random"`, etc.

For **MCWF** or **Lindblad**, set `representation` on the `State` and call `Simulator.run` — no extra steps:

```{code-cell} ipython3
neel_mcwf = State(4, initial="Neel", representation="vector")
print(neel_mcwf.representation)
```

Product presets can evolve in dense form without ever building an MPS in memory. **Entangled** presets (e.g. `"haar-random"`) may still require an internal MPS when you choose a dense representation.

Reproducible `"random"` presets: pass `seed=` on `State`.

```{code-cell} ipython3
a = State(3, initial="random", seed=7, representation="vector")
b = State(3, initial="random", seed=7, representation="vector")
# Same specification; run() will evolve both consistently.
```

## Manual initialization

Pass **exactly one** of `tensors`, `vector`, or `density_matrix`. Representation is **inferred**; do not pass `representation=`. Preset-only kwargs (`initial`, `pad`, `basis_string`, `seed`) cannot be combined with manual data.

### MPS cores (`tensors=`)

```{code-cell} ipython3
from mqt.yaqs import MPS

mps_ref = MPS(3, state="zeros")
spec = State(tensors=list(mps_ref.tensors))
print(spec.representation)
```

### Dense state vector (`vector=`)

`length` is inferred when the Hilbert-space dimension is a power of two.

```{code-cell} ipython3
vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)  # |00>
spec = State(vector=vec)
print(spec.length, spec.representation)
```

### Density matrix (`density_matrix=`)

```{code-cell} ipython3
rho = np.diag([1.0, 0.0, 0.0, 0.0]).astype(np.complex128)
spec = State(density_matrix=rho)
print(spec.representation)
```

A `State` created only with `vector=` or `density_matrix=` cannot be used for circuit simulation; use `tensors=` or a preset with `representation="mps"` instead.

## Representation and backends (analog)

Set **`representation` on `State`**, not on `AnalogSimParams`. [`Simulator.run`](mqt.yaqs.Simulator) materializes the correct internal form and dispatches:

| `representation`   | Backend (analog)                      |
| ------------------ | ------------------------------------- |
| `"mps"` (default)  | TJM (`analog_tjm_1` / `analog_tjm_2`) |
| `"vector"`         | MCWF                                  |
| `"density_matrix"` | Lindblad (small systems)              |

### Default: MPS / TJM

```{code-cell} ipython3
from mqt.yaqs import Hamiltonian, MPS, Simulator, AnalogSimParams, Observable

sim = Simulator(show_progress=False)

L = 3
H = Hamiltonian.ising(L, J=1.0, g=0.5)
obs = Observable("z", sites=[0])

state_mps = State(L, initial="zeros")  # representation="mps" by default
params = AnalogSimParams(
    observables=[obs],
    elapsed_time=0.2,
    dt=0.05,
)
result = sim.run(state_mps, H, params, noise_model=None)
print("TJM Z_0:", result.expectation_values[0][-1])
```

### MCWF (`representation="vector"`)

For guidance on choosing a representation, see {doc}`representation_comparison`.

```{code-cell} ipython3
state_vec = State(L, initial="zeros", representation="vector")
obs_vec = Observable("z", sites=[0])
params_vec = AnalogSimParams(
    observables=[obs_vec],
    elapsed_time=0.2,
    dt=0.05,
)
result = sim.run(state_vec, H, params_vec, None)
print("MCWF Z_0:", result.expectation_values[0][-1])
```

### Lindblad (`representation="density_matrix"`)

For guidance on choosing a representation, see {doc}`representation_comparison`.

```{code-cell} ipython3
state_dm = State(L, initial="zeros", representation="density_matrix")
obs_dm = Observable("z", sites=[0])
params_dm = AnalogSimParams(
    observables=[obs_dm],
    elapsed_time=0.2,
    dt=0.05,
)
result = sim.run(state_dm, H, params_dm, None)
print("Lindblad Z_0:", result.expectation_values[0][-1])
```

See {doc}`representation_comparison` for a side-by-side comparison of the three representations on the same Hamiltonian.

### Passing dense data directly

If you already have $|\psi\rangle$ or $\rho$, pass `vector=` or `density_matrix=` — representation is inferred:

```{code-cell} ipython3
psi = np.zeros(2**L, dtype=np.complex128)
psi[0] = 1.0
state_from_vec = State(vector=psi)
result = sim.run(state_from_vec, H, params_vec, None)
print("From vector=, MCWF Z_0:", result.expectation_values[0][-1])
```

## Practical limits

- **Memory**: dense `vector` scales as $2^N$; `density_matrix` as $2^{2N}$. Prefer `representation="mps"` for longer chains.
- **Entangled presets**: `"haar-random"` may need an internal MPS for dense representations.
- **Circuits**: use `State(..., representation="mps")` (default); `vector=` / `density_matrix=` states cannot run circuits.
- **Ensemble runs**: `list[State]` for deterministic unitary ensembles requires each member with `representation="mps"`.
- **`get_state`**: when supported, `result.output_state` is a [`State`](mqt.yaqs.core.data_structures.state.State). Use `.mps` for MPS runs, `.vector` for MCWF, or `.density_matrix` for Lindblad. Not supported with stochastic noise on `mps` or `vector` representations (use `density_matrix` for the exact ensemble average).

For MPO/TJM details without `State`, see {doc}`analog_simulation` and the [`MPS`](mqt.yaqs.core.data_structures.mps.MPS) API reference.

## Related topics

- {doc}`quickstart` — minimal first simulation
- {doc}`representation_comparison` — MPS, MCWF, and Lindblad backends
- {doc}`analog_simulation` — TJM evolution workflow
- {doc}`simulation_parameters` — presets and trajectory settings
