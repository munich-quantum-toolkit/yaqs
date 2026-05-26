---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
  execution_timeout: 300
---

# Configuring Simulation Parameters

YAQS separates **what you evolve** ({class}`~mqt.yaqs.core.data_structures.state.State`, circuits, Hamiltonians) from **how you truncate and sample** via parameter objects passed to {meth}`~mqt.yaqs.Simulator.run`:

| Class                                                                         | Use when                                                                                     |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| {class}`~mqt.yaqs.core.data_structures.simulation_parameters.AnalogSimParams` | Open-system or unitary time evolution (TDVP / BUG, MCWF trajectories, Lindblad-style paths). |
| {class}`~mqt.yaqs.core.data_structures.simulation_parameters.StrongSimParams` | Noisy **strong** digital simulation (per-trajectory MPS evolution with observables).         |
| {class}`~mqt.yaqs.core.data_structures.simulation_parameters.WeakSimParams`   | Noisy **weak** digital simulation (shot-based sampling; you set `shots` explicitly).         |

This page shows how to construct each class. For {class}`~mqt.yaqs.Simulator` execution options (parallelism, progress bars), see {doc}`simulator_initialization`.

## Start with a preset

You do **not** need to tune every numerical knob before running a simulation. Pick a **preset** and let it fill in the truncation and sampling settings you may be unfamiliar with (`svd_threshold`, `max_bond_dim`, `num_traj` on analog/strong runs, and `krylov_tol`).

All three `*SimParams` classes accept a keyword-only `preset` argument (default `"balanced"`):

| `preset`               | `svd_threshold` | `max_bond_dim` | `num_traj` (analog / strong) | `krylov_tol` |
| ---------------------- | --------------- | -------------- | ---------------------------- | ------------ |
| `"fast"`               | `1e-3`          | `16`           | `128`                        | `1e-3`       |
| `"balanced"` (default) | `1e-6`          | `128`          | `256`                        | `1e-4`       |
| `"accurate"`           | `1e-9`          | `4096`         | `1024`                       | `1e-6`       |
| `"exact"`              | `1e-13`         | `None`         | `1024`                       | `1e-12`      |

- **`"fast"`** — qualitative exploration and quick tests; not intended for strict dense comparisons.
- **`"balanced"`** — recommended default for exploratory work.
- **`"accurate"`** — high-quality production settings.
- **`"exact"`** — strict reference/debug preset with minimal internal numerical relaxation. Stochastic trajectory sampling, finite time steps, and model error still apply; this is not mathematically exact.

`svd_threshold` controls **tensor-network SVD truncation** (bond truncation). `krylov_tol` controls the **adaptive Krylov/Lanczos matrix exponential** inside TDVP updates. These are independent: tightening one does not change the other. `min_bond_dim` (default `2`) and `trunc_mode` (default `"discarded_weight"`) are unchanged across presets. The chosen preset name is stored on the object as `params.preset`.

## Override only what you need

**Explicit constructor arguments override the preset; everything you omit keeps the preset value.**

That is the intended workflow when you know _some_ settings but not all:

1. Choose the closest preset (`"fast"`, `"balanced"`, `"accurate"`, or `"exact"`).
2. Pass **only** the fields you want to change.
3. Leave the rest unset — they stay at the preset defaults.

Overridable preset fields:

| Argument        | What it controls                                   |
| --------------- | -------------------------------------------------- |
| `svd_threshold` | SVD bond truncation during MPS/MPO updates         |
| `max_bond_dim`  | Hard cap on bond dimension (`None` = no cap)       |
| `num_traj`      | Trajectory count (analog / strong only)            |
| `krylov_tol`    | Adaptive Krylov/Lanczos matrix exponential in TDVP |

`WeakSimParams` always requires `shots` separately; `shots` is **not** part of any preset.

If you omit an overridable argument, the preset supplies it. If you pass a value explicitly, **that value wins** for that field only — the other preset fields are unchanged. For `max_bond_dim`, omit the argument to keep the preset cap; pass `None` explicitly to remove the cap.

## Recommended usage

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.simulation_parameters import (
    SIMULATION_PRESETS,
    AnalogSimParams,
    Observable,
    StrongSimParams,
    WeakSimParams,
)
from mqt.yaqs.core.libraries.gate_library import Z


def _trunc_summary(params: AnalogSimParams | StrongSimParams | WeakSimParams) -> dict[str, object]:
    """Collect preset-related fields for display."""
    out: dict[str, object] = {
        "preset": params.preset,
        "svd_threshold": params.svd_threshold,
        "max_bond_dim": params.max_bond_dim,
        "krylov_tol": params.krylov_tol,
    }
    if isinstance(params, WeakSimParams):
        out["shots"] = params.shots
    else:
        out["num_traj"] = params.num_traj
    return out
```

Pick a preset — no other truncation arguments required:

```{code-cell} ipython3
# Default: balanced preset fills in all truncation settings
analog_params = AnalogSimParams()
print("default", _trunc_summary(analog_params))

for name in ("fast", "balanced", "accurate", "exact"):
    params = AnalogSimParams(preset=name)
    print(name, _trunc_summary(params))
```

Override **one** field; the rest stay from `"balanced"`:

```{code-cell} ipython3
balanced = AnalogSimParams(preset="balanced")
tighter_krylov = AnalogSimParams(preset="balanced", krylov_tol=1e-8)

print("balanced preset only", _trunc_summary(balanced))
print("override krylov_tol only", _trunc_summary(tighter_krylov))
# svd_threshold, max_bond_dim, and num_traj still come from "balanced"
assert tighter_krylov.svd_threshold == balanced.svd_threshold
assert tighter_krylov.max_bond_dim == balanced.max_bond_dim
assert tighter_krylov.num_traj == balanced.num_traj
```

Override **several** fields when you know exactly what you want; the remaining preset fields still apply:

```{code-cell} ipython3
custom_params = AnalogSimParams(
    preset="fast",  # start from fast defaults for everything else
    max_bond_dim=512,
    num_traj=32,
)
_trunc_summary(custom_params)
```

Weak simulation: set `shots` yourself, use a preset for truncation:

```{code-cell} ipython3
weak_params = WeakSimParams(
    shots=1024,
    preset="fast",
)
_trunc_summary(weak_params)
```

## `AnalogSimParams`

Besides the preset (and any overrides), you typically set the time grid (`elapsed_time`, `dt`), observables, and whether to record intermediate times (`sample_timesteps`).

```{code-cell} ipython3
L = 4
observables = [Observable(Z(), site) for site in range(L)]

analog = AnalogSimParams(
    observables=observables,
    elapsed_time=0.2,
    dt=0.05,
    preset="accurate",
)
_trunc_summary(analog)
```

Need a smaller bond cap for a quick test, but keep the rest of `"accurate"`?

```{code-cell} ipython3
analog_quick = AnalogSimParams(
    observables=observables,
    elapsed_time=0.2,
    dt=0.05,
    preset="accurate",
    max_bond_dim=256,
)
_trunc_summary(analog_quick)
```

Pass the resulting object to {meth}`~mqt.yaqs.Simulator.run` together with a {class}`~mqt.yaqs.core.data_structures.state.State` and {class}`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian` (see {doc}`analog_simulation`).

## `StrongSimParams`

Used for noisy strong circuit simulation. Provide observables and optionally enable layer sampling (see {doc}`strong_circuit_simulation`).

```{code-cell} ipython3
strong = StrongSimParams(
    observables=[Observable(Z(), 0)],
    preset="accurate",
)
_trunc_summary(strong)
```

## `WeakSimParams`

Used for noisy weak simulation. **`shots` is always required** and is not part of the preset.

```{code-cell} ipython3
weak_balanced = WeakSimParams(shots=1000)
weak_exact = WeakSimParams(shots=1000, preset="exact")
print("balanced", _trunc_summary(weak_balanced))
print("exact", _trunc_summary(weak_exact))
```

See {doc}`weak_circuit_simulation` for a full example with measurement histograms.

## Reference: preset table in code

The built-in values are defined in {data}`~mqt.yaqs.core.data_structures.simulation_parameters.SIMULATION_PRESETS`:

```{code-cell} ipython3
SIMULATION_PRESETS
```
