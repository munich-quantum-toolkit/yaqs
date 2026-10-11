---
file_format: mystnb
kernelspec:
  name: python3
language_info:
  name: python
mystnb:
  number_source_lines: true
  execution_timeout: 300
---

# Configuring the Simulator

`Simulator` controls how a calculation runs: parallel workers, progress bars,
and worker-error handling. Construct one instance and reuse it for successive
runs. The state, Hamiltonian or circuit, measurements, and noise are supplied
with each call; their settings are described in {doc}`simulation_parameters`.

## Run a small noisy simulation

This four-site Ising chain starts with every spin in $|1\rangle$. Local
relaxation acts during the evolution, and eight trajectories contribute to the
mean Pauli $Z$ expectations:

```{code-cell} python
from mqt.yaqs import AnalogSimParams, Hamiltonian, NoiseModel, Observable, Simulator, State

length = 4
state = State(length, initial="ones")
hamiltonian = Hamiltonian.ising(length, J=1.0, g=0.5)
params = AnalogSimParams(
    observables=[Observable("z", site) for site in range(length)],
    elapsed_time=0.4,
    dt=0.05,
    num_traj=8,
    random_seed=7,
)
noise = NoiseModel([
    {"name": "lowering", "sites": [site], "strength": 0.4}
    for site in range(length)
])

sim = Simulator(show_progress=False)
result = sim.run(state, hamiltonian, params, noise_model=noise)
```

Parallel execution remains enabled. The documentation suppresses progress bars
with `show_progress=False`; use `Simulator()` to see progress in your own runs.
The small trajectory budget illustrates execution and result access, rather than
sampling convergence.

## Read the result and reuse the simulator

`run` returns a `Result`. Observable order matches the supplied list:

```{code-cell} python
times = result.times
z0_mean = result.expectation_values[0]
z0_trajectories = result.trajectories[0]
```

Here, `times` and `z0_mean` have nine entries, including time zero.
`z0_trajectories` has shape `(8, 9)`: one row per trajectory and one column per
time. `z0_mean` averages those rows. Circuit runs can also return readout counts
in `result.counts`; see {doc}`circuit_shots`.

Reuse the same simulator and inputs for a noiseless reference:

```{code-cell} python
reference = sim.run(state, hamiltonian, params)
```

Each run starts from the supplied initial state. It does not continue from the
previous result. Sequential calls share execution settings, but they do not keep
a worker pool alive. YAQS creates a pool only when the calculation has multiple
independent jobs and more than one worker is available.

## Choose the common execution controls

All constructor options are keyword-only. Leave the defaults in place unless you
need a specific execution budget or a quieter run.

| Option          | Default   | When to change it                                                                       |
| --------------- | --------- | --------------------------------------------------------------------------------------- |
| `show_progress` | `True`    | Set `False` to suppress trajectory and readout bars in documentation or logs.           |
| `max_workers`   | Automatic | Set a positive integer to cap worker processes, for example `Simulator(max_workers=4)`. |
| `parallel`      | `True`    | Set `False` to debug in the calling process or avoid process startup for a small job.   |

A pool requires `parallel=True`, more than one independent job, and
`max_workers > 1`. Noiseless single-state evolution and density-matrix evolution
run in the calling process. `max_workers=1` also keeps execution in that
process, even when parallel execution is enabled, and limits numerical threads
to one. Worker processes limit their own numerical threads to avoid multiplying
thread pools across CPUs.

You can change settings between calls, for example `sim.max_workers = 2` or
`sim.show_progress = True`. Setting `sim.max_workers = None` restores automatic
worker selection.

## Run from a Python script

Worker processes start through `forkserver` on Linux and `spawn` on Windows and
macOS by default. These methods need a script entry-point guard so importing the
script in a child process does not start the simulation again.

Keep the imports and input definitions from the first example. Replace its
simulator creation and run calls with this block, and keep subsequent run calls
inside the guard:

```python
if __name__ == "__main__":
    sim = Simulator()
    result = sim.run(state, hamiltonian, params, noise_model=noise)
    reference = sim.run(state, hamiltonian, params)
```

Run the file with `python your_script.py`. In a notebook, execute the cells in
order without adding this guard. If process startup is the cause of a debugging
problem, `parallel=False` lets you inspect the calculation in the notebook's
process.

## Advanced execution options

:::{dropdown} Automatic worker budgets and numerical threads

With `max_workers=None`, the simulator uses `max(1, available_cpus() - 1)`. CPU
discovery takes the first valid hint from:

1. `YAQS_MAX_WORKERS`.
2. A pytest-xdist worker, which reports one CPU to avoid nested pools.
3. `SLURM_CPUS_PER_TASK`, then `SLURM_CPUS_ON_NODE`.
4. Process CPU affinity, when available.
5. The operating system's CPU count.

Invalid or non-positive environment hints are ignored. The default worker policy
then leaves one reported CPU free. Thus `YAQS_MAX_WORKERS=4` normally resolves
to three workers; `Simulator(max_workers=4)` explicitly permits four. Use the
constructor argument when you need an exact process cap. CPU affinity can
restrict the available cores, but this discovery does not read every container
CPU-quota setting.

A worker cap bounds process count, not total memory. Each process needs its own
simulation state. Reduce the cap when several concurrent trajectories exceed
your memory budget. Numerical libraries are capped inside workers; turning off
process parallelism does not promise unrestricted BLAS threading.

:::

:::{dropdown} Multiprocessing start methods

`mp_context="auto"` selects `"forkserver"` on Linux and `"spawn"` elsewhere. For
an explicit selection, the public options are `"spawn"` and `"fork"` on
platforms that support them. There is no fallback for an unavailable method.
Keep `"auto"` unless the environment requires a specific method.

`"spawn"` starts a fresh interpreter for each worker. It can also be useful when
combining YAQS with libraries that need fresh process initialization. `"fork"`
copies the application's process and can be unsafe when the parent has active
threads. Thread limits do not remove that risk. The automatic Linux context
starts workers from a separate server process; its first pool has additional
startup cost.

Worker inputs must be pickleable. Pools are scoped to individual runs, although
the Python forkserver helper can remain alive between calls. These execution
choices do not change the requested simulation model.

:::

:::{dropdown} Retries for worker errors

`max_retries=10` allows up to ten additional attempts for a failed job in a
process pool. The default `retry_exceptions` tuple contains
`concurrent.futures.CancelledError`, `TimeoutError`, and `OSError`.

Only matching exceptions raised when retrieving a worker result trigger a retry.
After the retry budget is exhausted, the error propagates. Other exceptions,
including `ValueError`, propagate immediately under the default policy. Set
`max_retries=0` to propagate the first worker failure, or supply a tuple of
exception classes for a specific transient failure in your environment.

Retries do not impose a timeout, apply to in-process execution, or restart a
broken pool. Pool startup and submission failures are outside this retry policy.
Do not increase the retry budget to handle a repeatable error in the model.

:::

:::{dropdown} Other result fields and configuration references

Outputs that do not apply to a run remain `None` or empty. The API reference for
{class}`~mqt.yaqs.Result` describes the full result structure:

| Fields                                              | Use                                                                                                                          |
| --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `observables`, `expectation_values`, `trajectories` | Requested observables, their means, and individual realization data in the supplied order.                                   |
| `times`                                             | Analog sample times, or the stitched timeline of a program with observables. Standalone circuits have no physical time axis. |
| `counts`, `measurements`                            | Total circuit readout counts and per-trajectory histograms when shots are requested.                                         |
| `output_state`                                      | A final state when `get_state=True` is supported; see {doc}`simulation_parameters`.                                          |
| `max_bond`, `total_bond`, `runtime_cost`            | MPS bond diagnostics and an estimated contraction cost, when recorded.                                                       |
| `multi_time_times`, `multi_time_results`            | Complex two-time correlations for unitary ensembles; see {doc}`ensemble_evolution`.                                          |
| `segment_results`                                   | Individual results from a `SimulationProgram`; see {doc}`digital_analog_simulation`.                                         |
| `sim_params`, `noise_model`                         | The validated simulation parameters and the noise model used by the run.                                                     |

For a standalone run, `result.sim_params` references the supplied parameter
object. Validation normalizes that object and rebuilds its analog time grid; it
is not an immutable snapshot. Construct a new parameter object when you need to
retain a separate configuration. State and Hamiltonian wrappers may also
populate cached representations, while evolution uses copies of their input
states.

A program's top-level `sim_params` is `None`; read segment parameters through
`result.segment_results`. Program settings and trajectory budgets are explained
in {doc}`digital_analog_simulation`.

For temporary storage, pickle can save results for later analysis with matching
Python, YAQS, and dependency versions. This stores the result; it does not
resume an interrupted simulation. Only load pickle files from a trusted source.

:::

## Related guides

- {doc}`simulation_parameters` — accuracy, sampling budgets, and output
  requests.
- {doc}`analog_simulation` — noisy analog dynamics.
- {doc}`circuit_observables` — circuit expectations and checkpoints.
- {doc}`digital_analog_simulation` — analog-digital programs and segment
  results.
- {doc}`representation_comparison` — choosing the analog state representation.

Full constructor and `run` signatures are in the API reference for
{class}`~mqt.yaqs.Simulator`.
