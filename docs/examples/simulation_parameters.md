---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
  execution_timeout: 300
---

# Configuring Simulation Parameters

YAQS separates **what you evolve** ({class}`~mqt.yaqs.State`, circuits, Hamiltonians) from **how you truncate and sample** via parameter objects passed to {meth}`~mqt.yaqs.Simulator.run`:

| Class                              | Use when                                                                                     |
| ---------------------------------- | -------------------------------------------------------------------------------------------- |
| {class}`~mqt.yaqs.AnalogSimParams` | Open-system or unitary time evolution (TDVP / BUG, MCWF trajectories, Lindblad-style paths). |
| {class}`~mqt.yaqs.StrongSimParams` | Noisy **strong** digital simulation (per-trajectory MPS evolution with observables).         |
| {class}`~mqt.yaqs.WeakSimParams`   | Noisy **weak** digital simulation (shot-based sampling; you set `shots` explicitly).         |

This page shows how to construct each class. For {class}`~mqt.yaqs.Simulator` execution options (parallelism, progress bars), see {doc}`simulator_initialization`.

## Observable string names

{class}`~mqt.yaqs.Observable` accepts a **string gate name** as its first argument. YAQS resolves the name to the corresponding operator internally — you do not import gate classes for standard measurements.

| String                         | Meaning                                                        | Example                                     |
| ------------------------------ | -------------------------------------------------------------- | ------------------------------------------- |
| `"x"`, `"y"`, `"z"`            | Single-qubit Pauli operators                                   | `Observable("z", sites=0)`                  |
| `"h"`, `"s"`, `"t"`, `"rx"`, … | Other single-qubit gates from the built-in library             | `Observable("h", sites=0)`                  |
| `"xx"`, `"yy"`, `"zz"`         | Two-qubit Pauli strings                                        | `Observable("zz", sites=[0, 1])`            |
| `"entropy"`                    | Bipartite entanglement entropy across a cut                    | `Observable("entropy", sites=cut)`          |
| `"schmidt_spectrum"`           | Schmidt spectrum across a cut                                  | `Observable("schmidt_spectrum", sites=cut)` |
| bitstring / `"pvm"`            | Projection-valued measurement onto a computational basis state | see {doc}`circuit_simulation`               |

For custom unitaries and circuit gates, use {doc}`custom_gates` — those workflows still use `GateLibrary` or Qiskit circuits directly.

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

`svd_threshold` controls **tensor-network SVD truncation** (bond truncation). `krylov_tol` controls the **adaptive Krylov/Lanczos matrix exponential** inside TDVP updates. These are independent: tightening one does not change the other. `trunc_mode` (default `"discarded_weight"`) is unchanged across presets. The chosen preset name is stored on the object as `params.preset`.

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
from mqt.yaqs import (
    SIMULATION_PRESETS,
    AnalogSimParams,
    Observable,
    StrongSimParams,
    WeakSimParams,
)


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

for name in ("fast", "balanced", "accurate", "exact"):
    _trunc_summary(AnalogSimParams(preset=name))
```

Override **one** field; the rest stay from `"balanced"`:

```{code-cell} ipython3
balanced = AnalogSimParams(preset="balanced")
tighter_krylov = AnalogSimParams(preset="balanced", krylov_tol=1e-8)
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
observables = [Observable("z", site) for site in range(L)]

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

Pass the resulting object to {meth}`~mqt.yaqs.Simulator.run` together with a {class}`~mqt.yaqs.State` and {class}`~mqt.yaqs.Hamiltonian` (see {doc}`analog_simulation`).

## `StrongSimParams`

Used for noisy strong circuit simulation. Provide observables and optionally enable layer sampling (see {doc}`circuit_simulation`).

### Two-qubit gate mode (`gate_mode`)

Digital circuit simulation on an MPS defaults to **`gate_mode="mpo"`** (generic MPO--MPS application): nearest-neighbor gates use the same local TEBD/SVD path as `swaps`, and long-range gates contract an extended gate MPO site-wise (library leg ordering, MPS virtual index before MPO virtual index) followed by compression with `svd_threshold` and `max_bond_dim`. Other modes differ only in how two-qubit gates are applied:

- **`swaps`** — TEBD/SVD for every two-qubit gate; long-range gates are routed with adjacent SWAP insertion before and after the local update.
- **`tdvp`** — TEBD/SVD on nearest-neighbor gates; long-range gates use the generator MPO + **two-site TDVP (2TDVP)** on a local window.
- **`full-tdvp`** — TDVP (generator MPO + 2TDVP on a local window) on every two-qubit gate.

Matrix-backed custom gates (from Qiskit `UnitaryGate` or other unknown 1-/2-qubit unitaries) have no
analytic generator. In `gate_mode="tdvp"` or `"full-tdvp"`, those gates use TEBD on nearest-neighbor
pairs and the MPO path on long-range pairs instead of the TDVP generator window. See {doc}`custom_gates`
for the full gate translation and custom-gate workflow.

Long-range gates in `gate_mode="tdvp"` apply 2TDVP on the gate support window via `evolve_window`.

Use **`tdvp_sweeps`** (default `1`) to split each TDVP evolution step into multiple substeps of equal total time. Values greater than `1` are opt-in and may improve accuracy on some circuits. The setting applies to all TDVP kernels on `AnalogSimParams`, `StrongSimParams`, and `WeakSimParams`.

Use **`tdvp_mode`** to select the TDVP integrator: `"1site"` (1TDVP), `"2site"` (2TDVP), or `"dynamic"` (adaptive single/two-site updates). The default is **`"2site"`** (2TDVP) on `AnalogSimParams`, `StrongSimParams`, and `WeakSimParams`. Pass `"dynamic"` explicitly for adaptive 1/2-site switching during analog evolution.

Substep geometry: each substep is **symmetric** (left-to-right then right-to-left) at evolution time `step_time / tdvp_sweeps` for analog (`dt`) and digital gates. The total generator time applied to one digital gate remains `1` across all substeps. Noise and dissipation after TDVP still use the full physical step `dt` in analog simulation.

```{code-cell} ipython3
strong = StrongSimParams(
    observables=[Observable("z", 0)],
    gate_mode="tdvp",
    tdvp_sweeps=2,
    preset="accurate",
)
_trunc_summary(strong)
```

```{code-cell} ipython3
strong_default = StrongSimParams(
    observables=[Observable("z", 0)],
    preset="accurate",
)
_trunc_summary(strong_default)
```

## `WeakSimParams`

Used for noisy weak simulation. **`shots` is always required** and is not part of the preset.

YAQS stores weak-simulation measurement histograms in `Result.counts` as a `dict[int, int]`. The integer key encodes the measured bitstring with **site 0 as the least-significant bit** (little-endian). This matches Qiskit’s default convention if you interpret Qiskit bitstrings (`c_{n-1}...c_0`) via `int(bitstring, 2)`.

```{code-cell} ipython3
weak_balanced = WeakSimParams(shots=1000)
weak_exact = WeakSimParams(shots=1000, preset="exact")
```

See {doc}`weak_circuit_simulation` for a full example with measurement histograms.

## Reference: preset table in code

The built-in values are defined in {data}`~mqt.yaqs.SIMULATION_PRESETS`:

```{code-cell} ipython3
SIMULATION_PRESETS
```

## Related topics

- {doc}`quickstart` — minimal first simulation
- {doc}`analog_simulation` — analog parameters in context
- {doc}`circuit_simulation` — `StrongSimParams`, `gate_mode`, and layer sampling
- {doc}`weak_circuit_simulation` — `WeakSimParams` and shot readout
