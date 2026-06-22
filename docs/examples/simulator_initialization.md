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

# Configuring the Simulator

YAQS draws a sharp line between **what** you simulate and **how** it runs:

| Layer                                                                                                                                                                                                | Role                                                                                                                                                           |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| {class}`~mqt.yaqs.State`, {class}`~mqt.yaqs.Hamiltonian`, {class}`~mqt.yaqs.AnalogSimParams` / {class}`StrongSimParams <mqt.yaqs.StrongSimParams>` / {class}`WeakSimParams <mqt.yaqs.WeakSimParams>` | The physics: initial state, operator, time grid, observables, trajectory count, truncation, noise.                                                             |
| {class}`~mqt.yaqs.Simulator`                                                                                                                                                                         | The execution: parallel vs. serial trajectories, worker count, progress reporting, multiprocessing start method, and retry policy for transient worker errors. |

This page walks through every option on the {class}`~mqt.yaqs.Simulator` class so you can tune execution without touching the physics.

```{code-cell} ipython3
from mqt.yaqs import AnalogSimParams, Hamiltonian, Observable, Result, Simulator, State
```

A small reusable analog problem we will simulate throughout:

```{code-cell} ipython3
L = 4
H = Hamiltonian.ising(L, J=1.0, g=0.5)


def make_params(num_traj: int = 8) -> AnalogSimParams:
    """A short Ising evolution measuring `<Z>` on every site."""
    return AnalogSimParams(
        observables=[Observable("z", site) for site in range(L)],
        elapsed_time=0.2,
        dt=0.05,
        num_traj=num_traj,
        max_bond_dim=8,
        svd_threshold=1e-9,
        sample_timesteps=False,
        random_seed=0,
    )


state = State(L, initial="zeros")
```

## Quick start: defaults

Calling `Simulator()` with no arguments gives you parallel execution across most of your CPU cores, a `tqdm` progress bar, an `"auto"` multiprocessing context, and a generous retry policy.

```{code-cell} ipython3
sim = Simulator()
print("parallel:        ", sim.parallel)
print("max_workers:     ", sim.max_workers)
print("show_progress:   ", sim.show_progress)
print("mp_context:      ", sim.mp_context)
print("max_retries:     ", sim.max_retries)
print("retry_exceptions:", sim.retry_exceptions)
```

Every option is keyword-only, so you can override one without specifying the others:

```{code-cell} ipython3
quiet_sim = Simulator(show_progress=False)
```

## Reusing one `Simulator` across runs

A `Simulator` instance is **stateless** with respect to the physics; the same instance can drive arbitrarily many {meth}`~mqt.yaqs.Simulator.run` calls. This is the recommended pattern in scripts and notebooks because it keeps execution configuration in one place.

```{code-cell} ipython3
sim = Simulator(show_progress=False)

for noise_strength in (0.0, 0.05, 0.1):
    params = make_params()
    # ... build a NoiseModel from `noise_strength` here in real code ...
    result = sim.run(state, H, params, noise_model=None)
    print(f"gamma={noise_strength}: <Z_0>={float(result.expectation_values[0][0]):+.4f}")
```

Each call constructs a short-lived `ProcessPoolExecutor` when `parallel=True`; pools are not persisted across {meth}`~mqt.yaqs.Simulator.run` calls, so you can safely change `sim.max_workers` (or replace `sim` entirely) between calls.

## `parallel`: process-pool vs. in-process execution

`parallel=True` (the default) runs trajectories in worker processes via `concurrent.futures.ProcessPoolExecutor`. `parallel=False` runs every trajectory in the calling process, which is useful for:

- Debugging (full tracebacks, no pickling, breakpoints work).
- Very small jobs where the pool startup cost dominates.
- Notebook cells where you want to share state with the caller.

Both modes produce identical results for a fixed `random_seed`:

```{code-cell} ipython3
import numpy as np

params_serial = make_params()
sim_serial = Simulator(parallel=False, show_progress=False)
result_serial = sim_serial.run(state, H, params_serial)

params_parallel = make_params()
sim_parallel = Simulator(parallel=True, max_workers=2, show_progress=False)
result_parallel = sim_parallel.run(state, H, params_parallel)

for vals_s, vals_p in zip(result_serial.expectation_values, result_parallel.expectation_values, strict=True):
    np.testing.assert_allclose(vals_s, vals_p, atol=1e-10)
print("Serial and parallel results match.")
```

```{note}
For runs with `num_traj == 1` (e.g. noise-free analog/circuit dynamics, Lindblad), the simulator automatically takes the in-process path even with `parallel=True`. The pool is only spun up when there is more than one trajectory to dispatch.
```

## `max_workers` and how the default is chosen

When `max_workers` is left as `None`, the simulator picks `max(1, available_cpus() - 1)` to leave one core free for the parent process and the OS:

```{code-cell} ipython3
from mqt.yaqs.simulator import available_cpus

print("available CPUs YAQS would use:", available_cpus())
print("default max_workers:          ", Simulator().max_workers)
```

{func}`~mqt.yaqs.parallel_utils.available_cpus` (re-exported as {func}`~mqt.yaqs.simulator.available_cpus`) is deliberately cgroup- and scheduler-aware. In priority order it honours:

1. `YAQS_MAX_WORKERS` (explicit user override; positive integer).
2. `PYTEST_XDIST_WORKER` (returns `1` to avoid nested parallelism in tests).
3. SLURM hints — `SLURM_CPUS_PER_TASK` then `SLURM_CPUS_ON_NODE`.
4. Linux `os.sched_getaffinity(0)` (respects `taskset`, containers, cgroups).
5. `os.cpu_count()` as a final fallback.

Override the resolution either by setting the environment variable…

```python
# In a shell: export YAQS_MAX_WORKERS=4
```

…or by passing `max_workers` explicitly:

```{code-cell} ipython3
sim_four = Simulator(max_workers=4, show_progress=False)
print("fixed max_workers:", sim_four.max_workers)
```

## `show_progress`: tqdm bars

`show_progress=True` (default) shows a `tqdm` bar labelled "Running trajectories" (or "Running unitary ensemble" for the deterministic ensemble path). Set `show_progress=False` to silence it — useful in test suites, batch scripts, and CI logs:

```{code-cell} ipython3
silent = Simulator(show_progress=False)
silent.run(state, H, make_params(num_traj=4))
```

The bar is suppressed regardless of `parallel`, so the same flag also silences serial runs.

## `mp_context`: multiprocessing start method

`mp_context` controls how worker processes are spawned. The default `"auto"` picks the best option per OS:

| Value     | Behaviour                                                                                                       |
| --------- | --------------------------------------------------------------------------------------------------------------- |
| `"auto"`  | `"fork"` on Linux, `"spawn"` everywhere else.                                                                   |
| `"fork"`  | Fastest worker startup; reuses Python state from the parent. Safe in YAQS because BLAS/OpenMP pools are capped. |
| `"spawn"` | Fresh interpreter per worker. Required on Windows/macOS; slower startup but more isolated.                      |

```{code-cell} ipython3
from mqt.yaqs.parallel_utils import get_parallel_context

for choice in ("auto", "fork", "spawn"):
    try:
        ctx = get_parallel_context(choice)
    except ValueError as exc:
        print(f"{choice}: not available on this platform ({exc})")
    else:
        print(f"{choice}: start_method={ctx.get_start_method()}")
```

If you mix YAQS with GPU libraries or anything that does not survive `fork()`, force `mp_context="spawn"`.

## `max_retries` and `retry_exceptions`

Long parallel runs occasionally encounter transient worker failures: a worker is cancelled by the OS, a `TimeoutError` is raised, or a transient `OSError` (e.g. a temporary file system hiccup) propagates out of a backend. By default, the simulator retries each failing trajectory up to **10 times** for the following exception types:

- `concurrent.futures.CancelledError`
- `TimeoutError`
- `OSError`

```{code-cell} ipython3
print("default max_retries:    ", Simulator().max_retries)
print("default retry_exceptions:", Simulator().retry_exceptions)
```

Tighten the policy for fail-fast development (e.g. when bisecting an error):

```{code-cell} ipython3
strict = Simulator(max_retries=0, show_progress=False)
```

Or broaden it for unreliable environments:

```{code-cell} ipython3
import concurrent.futures

resilient = Simulator(
    max_retries=20,
    retry_exceptions=(concurrent.futures.CancelledError, TimeoutError, OSError, ConnectionError),
    show_progress=False,
)
```

Permanent errors (e.g. `ValueError` from your physics setup, `AssertionError` from invariants) are not retried — they propagate after the first failure regardless of `max_retries`.

## Inspecting the return value: `Result`

{meth}`~mqt.yaqs.Simulator.run` returns a {class}`~mqt.yaqs.Result` that holds every simulation output through a small, stable surface. The {class}`~mqt.yaqs.core.data_structures.simulation_parameters.AnalogSimParams` you passed in is referenced unchanged at `result.sim_params`:

```{code-cell} ipython3
sim = Simulator(show_progress=False)
params = make_params()
result = sim.run(state, H, params)

print("type:                 ", type(result).__name__)
print("len(observables):     ", len(result.observables))
print("len(expectation_values):", len(result.expectation_values))
print("len(trajectories):    ", len(result.trajectories))
print("times:                ", result.times)
print("noise_model:          ", result.noise_model)
print("output_state:         ", result.output_state)
print("counts (weak only):   ", result.counts)
print("multi_time_times:     ", result.multi_time_times)
print("multi_time_results:   ", result.multi_time_results)
print("runtime_cost:         ", result.runtime_cost)
print("max_bond:             ", result.max_bond)
print("total_bond:           ", result.total_bond)
```

The properties that don't apply to your simulation kind return `None` (or an empty list for `observables` in weak simulations), so you can branch on them safely. The full set is:

| Property                                 | Populated for                                                                                                                                                        |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `observables`                            | Analog and strong digital runs. Empty list for weak digital.                                                                                                         |
| `expectation_values`                     | Aggregated expectation per observable (parallel to `observables`).                                                                                                   |
| `trajectories`                           | Per-trajectory data per observable (parallel to `observables`).                                                                                                      |
| `times`                                  | Shared analog time grid; `None` for digital circuits.                                                                                                                |
| `runtime_cost`                           | MPS-backed analog and strong digital runs (contraction-cost heuristic over time).                                                                                    |
| `max_bond`                               | MPS-backed analog and strong digital runs (maximum bond dimension over time).                                                                                        |
| `total_bond`                             | MPS-backed analog and strong digital runs (sum of internal bond dimensions).                                                                                         |
| `noise_model`                            | Any run that was given a `NoiseModel`; otherwise `None`.                                                                                                             |
| `output_state`                           | Runs with `get_state=True` on `AnalogSimParams` or `StrongSimParams`. For Lindblad (`density_matrix`), noisy runs are supported; for `mps`/`vector`, noiseless only. |
| `multi_time_times`, `multi_time_results` | Analog deterministic ensembles with `multi_time_observables` set.                                                                                                    |
| `counts`                                 | Weak digital simulations (the `dict[int, int]` of aggregated measurement outcomes).                                                                                  |

`Result` (and its wrapped `sim_params`) is pickleable, so you can checkpoint and resume analysis from disk:

```{code-cell} ipython3
import pickle

blob = pickle.dumps(result)
restored: Result = pickle.loads(blob)  # noqa: S301
print("restored observable count:", len(restored.observables))
```

## Choosing settings for common scenarios

| Scenario                                          | Recommended `Simulator(...)`                                           |
| ------------------------------------------------- | ---------------------------------------------------------------------- |
| Quick local run, want to see a progress bar       | `Simulator()`                                                          |
| Notebook / docs build / CI logs                   | `Simulator(show_progress=False)`                                       |
| Debugging a physics setup                         | `Simulator(parallel=False, show_progress=False)`                       |
| Single-process benchmark, all cores in the worker | `Simulator(parallel=False)` and let BLAS/OpenMP use all threads        |
| Fixed core budget (e.g. SLURM job step)           | `YAQS_MAX_WORKERS=N` in the environment, or `Simulator(max_workers=N)` |
| Mixing with GPU / non-fork-safe code              | `Simulator(mp_context="spawn")`                                        |
| Long unattended run on a flaky cluster            | `Simulator(max_retries=20)` with broadened `retry_exceptions`          |

For physics-side settings (`num_traj`, `max_bond_dim`, `svd_threshold`, `random_seed`, `sample_timesteps`, observables, noise), see {doc}`analog_simulation`, {doc}`representation_comparison`, and {doc}`state_initialization`.

## Related topics

- {doc}`quickstart` — minimal first simulation
- {doc}`simulation_parameters` — physics-side presets and truncation
- {doc}`analog_simulation` — TJM workflow with noise
- {doc}`circuit_simulation` — digital simulation entry point
