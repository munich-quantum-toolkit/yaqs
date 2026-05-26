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

## Accuracy presets

All three classes accept a keyword-only `accuracy` argument (default `"balanced"`) that sets SVD truncation and, for analog/strong simulations, trajectory counts. It also controls `krylov_tol`, the tolerance for the adaptive Krylov/Lanczos matrix exponential used in TDVP updates:

| `accuracy`             | `threshold` | `max_bond_dim` | `num_traj` (analog / strong) | `krylov_tol` |
| ---------------------- | ----------- | -------------- | ---------------------------- | ------------ |
| `"fast"`               | `1e-3`      | `16`           | `64`                         | `1e-3`       |
| `"balanced"` (default) | `1e-6`      | `128`          | `256`                        | `1e-4`       |
| `"accurate"`           | `1e-9`      | `4096`         | `1024`                       | `1e-6`       |

- **`"fast"`** — quick tests, examples, and CI-style runs.
- **`"balanced"`** — default tradeoff for exploratory work.
- **`"accurate"`** — strictest built-in preset (`threshold=1e-9`, `max_bond_dim=4096`, `num_traj=1024`, `krylov_tol=1e-6`).
- **Overrides** — pass `threshold`, `max_bond_dim`, and/or `num_traj` explicitly; any non-`None` value wins over the preset.

`threshold` controls **SVD truncation** (bond truncation), while `krylov_tol` controls the **adaptive Krylov/Lanczos matrix exponential** inside TDVP updates. `min_bond_dim` (default `2`) and `trunc_mode` (default `"discarded_weight"`) are unchanged across presets. The chosen preset is stored on the object as `params.accuracy`.

## Recommended usage

```{code-cell} ipython3
from mqt.yaqs.core.data_structures.simulation_parameters import (
    ACCURACY_PRESETS,
    AnalogSimParams,
    Observable,
    StrongSimParams,
    WeakSimParams,
)
from mqt.yaqs.core.libraries.gate_library import Z


def _trunc_summary(params: AnalogSimParams | StrongSimParams | WeakSimParams) -> dict[str, object]:
    """Collect accuracy-related fields for display."""
    out: dict[str, object] = {
        "accuracy": params.accuracy,
        "threshold": params.threshold,
        "max_bond_dim": params.max_bond_dim,
        "krylov_tol": params.krylov_tol,
    }
    if isinstance(params, WeakSimParams):
        out["shots"] = params.shots
    else:
        out["num_traj"] = params.num_traj
    return out
```

```{code-cell} ipython3
# Default: balanced preset
analog_params = AnalogSimParams()
print("default", _trunc_summary(analog_params))

# Fast preset for quick tests, examples, and CI-style runs
fast_params = AnalogSimParams(accuracy="fast")
print("fast", _trunc_summary(fast_params))

# Balanced preset for normal exploratory use
balanced_params = StrongSimParams(accuracy="balanced")
print("balanced", _trunc_summary(balanced_params))

# Accurate preset for stricter built-in settings
accurate_params = StrongSimParams(accuracy="accurate")
print("accurate", _trunc_summary(accurate_params))
```

Explicit numerical values override the preset (advanced control):

```{code-cell} ipython3
custom_params = StrongSimParams(
    accuracy="balanced",
    threshold=1e-8,
    krylov_tol=1e-12,
    max_bond_dim=512,
    num_traj=512,
)
_trunc_summary(custom_params)
```

Weak simulation: `shots` remain explicit and are **not** controlled by `accuracy`:

```{code-cell} ipython3
weak_params = WeakSimParams(
    shots=1024,
    accuracy="fast",
)
_trunc_summary(weak_params)
```

## `AnalogSimParams`

Besides truncation settings, you typically set the time grid (`elapsed_time`, `dt`), observables, and whether to record intermediate times (`sample_timesteps`).

```{code-cell} ipython3
L = 4
observables = [Observable(Z(), site) for site in range(L)]

analog = AnalogSimParams(
    observables=observables,
    elapsed_time=0.2,
    dt=0.05,
    accuracy="accurate",
)
_trunc_summary(analog)
```

Pass the resulting object to {meth}`~mqt.yaqs.Simulator.run` together with a {class}`~mqt.yaqs.core.data_structures.state.State` and {class}`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian` (see {doc}`analog_simulation`).

## `StrongSimParams`

Used for noisy strong circuit simulation. Provide observables and optionally enable layer sampling (see {doc}`strong_circuit_simulation`).

```{code-cell} ipython3
strong = StrongSimParams(
    observables=[Observable(Z(), 0)],
    accuracy="accurate",
)
_trunc_summary(strong)
```

## `WeakSimParams`

Used for noisy weak simulation. **`shots` is always required** and is not part of the accuracy preset.

```{code-cell} ipython3
weak_balanced = WeakSimParams(shots=1000)
weak_accurate = WeakSimParams(shots=1000, accuracy="accurate")
print("balanced", _trunc_summary(weak_balanced))
print("accurate", _trunc_summary(weak_accurate))
```

See {doc}`weak_circuit_simulation` for a full example with measurement histograms.

## Reference: preset table in code

The built-in values are defined in {data}`~mqt.yaqs.core.data_structures.simulation_parameters.ACCURACY_PRESETS`:

```{code-cell} ipython3
ACCURACY_PRESETS
```
