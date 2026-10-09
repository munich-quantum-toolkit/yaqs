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

# Configuring Simulation Parameters

Simulation parameters choose what to measure, when to record it, and the
numerical accuracy. Start with a preset, then change the settings that matter
for your calculation.

| Class              | Use for                                                                |
| ------------------ | ---------------------------------------------------------------------- |
| `AnalogSimParams`  | Hamiltonian evolution, including noisy dynamics and unitary ensembles. |
| `DigitalSimParams` | Circuit expectation values, shot counts, or a final state.             |

Pass the parameter object to `Simulator.run` with the initial state and
Hamiltonian or circuit. Supply noise through `noise_model` in that call. The
state representation selects the analog backend; see
{doc}`representation_comparison`. Parallel execution and progress controls
belong to `Simulator`, as described in {doc}`simulator_initialization`.

## Choose a preset

Both classes default to `preset="balanced"`. A preset supplies four settings:

| Preset       | `svd_threshold` | `max_bond_dim` | `num_traj` | `krylov_tol` |
| ------------ | --------------- | -------------- | ---------- | ------------ |
| `"fast"`     | `1e-3`          | `16`           | `128`      | `1e-3`       |
| `"balanced"` | `1e-6`          | `128`          | `256`      | `1e-4`       |
| `"accurate"` | `1e-9`          | `4096`         | `1024`     | `1e-6`       |
| `"exact"`    | `1e-13`         | `None`         | `1024`     | `1e-12`      |

Use `"fast"` to explore a model and `"balanced"` as a starting point for
accuracy checks. `"accurate"` tightens the numerical settings and increases the
trajectory budget, at greater cost. `"exact"` provides strict reference settings
and removes the bond cap; finite timesteps, truncation, and sampling error still
apply. No preset establishes convergence for every model.

Explicit constructor arguments replace only the corresponding preset values:

```{code-cell} python
from mqt.yaqs import AnalogSimParams, DigitalSimParams, Observable

custom_params = AnalogSimParams(
    preset="accurate",
    max_bond_dim=256,
    num_traj=64,
)
```

Here, `svd_threshold` and `krylov_tol` retain their `"accurate"` values. Omit
`max_bond_dim` to keep the preset cap; pass `None` explicitly to remove it.
Choose the preset when constructing the object. Assigning a new value to
`params.preset` later does not reset the other fields. `shots` is always an
explicit budget and is not part of a preset.

## Set analog measurements and times

For a four-site chain, record the local Pauli $Z$ expectations over two time
units:

```{code-cell} python
analog_params = AnalogSimParams(
    observables=[Observable("z", site) for site in range(4)],
    elapsed_time=2.0,
    dt=0.05,
)
```

The default `sample_timesteps=True` records 41 samples, including time zero and
the final time. After the run, `result.times` holds the sampled times and
`result.expectation_values[i]` holds the values for the $i$th supplied
observable. Set `sample_timesteps=False` to keep only the final sample;
evolution still uses the same `dt`.

The timestep must be positive, and the duration must be non-negative and an
integer multiple of `dt`. For a computed grid, set
`elapsed_time = num_steps * dt` to keep the duration consistent with the step
count. Invalid grids raise `ValueError`.

For noisy trajectory evolution, `num_traj` sets the number of realizations to
average. A noiseless single-state run uses one realization. A density-matrix
backend evolves the ensemble directly, while a supplied `list[State]` uses the
list length as its ensemble size. See {doc}`analog_simulation` for a noisy
walkthrough and {doc}`ensemble_evolution` for unitary ensemble averages.

## Choose circuit outputs

Request observables, shots, or both. At least one output must be requested in a
standalone run; `get_state=True` is another option for supported noiseless runs.

```{code-cell} python
observable_params = DigitalSimParams(observables=[Observable("z", 0)])
shot_params = DigitalSimParams(shots=1024)
combined_params = DigitalSimParams(
    observables=[Observable("z", 0)],
    shots=1024,
    num_traj=64,
)
```

Observables are recorded at the circuit's end by default. With
`sample_layers=True`, YAQS also records the initial state and checkpoints marked
by `circuit.barrier(label="SAMPLE_OBSERVABLES")`. It does not sample every
circuit layer automatically. See {doc}`circuit_observables` for checkpoint
placement and the resulting sample axis.

`shots` is the total readout budget. `num_traj` controls the noisy observable
ensemble, so the two settings have different roles:

| Run                     | How YAQS uses the budgets                                                       |
| ----------------------- | ------------------------------------------------------------------------------- |
| Noiseless               | Evolve once and sample all requested shots from the final state.                |
| Noisy, with observables | Average `num_traj` trajectories. Distribute any requested shots across them.    |
| Noisy, shots only       | Run one single-shot trajectory per shot; `num_traj` does not control this path. |

A combined run supports `shots < num_traj`: every trajectory contributes to the
observable mean, while some receive no readout samples. Counts appear in
`result.counts` as integer keys and integer counts. Site 0 is the
least-significant bit, matching Qiskit's `int(bitstring, 2)` convention. See
{doc}`circuit_shots` for histogram plotting.

## Choose observables and diagnostics

Named operators avoid imports from gate libraries. The Pauli names below refer
to $\sigma^\alpha$, rather than spin operators $S^\alpha=\sigma^\alpha/2$.

| Request                           | Example                                                                |
| --------------------------------- | ---------------------------------------------------------------------- |
| Single-site Pauli operator        | `Observable("z", 0)`; also `"x"` and `"y"`                             |
| Identity or basis projector       | `Observable("id", 0)`, `Observable("p0", 0)`, or `Observable("p1", 0)` |
| Two-site Pauli operator           | `Observable("zz", [0, 1])`; also `"xx"` and `"yy"`                     |
| Position on a supplied grid       | `Observable("position", 0, positions=grid)`                            |
| Probability of a full basis state | `Observable("0101")` for a four-qubit system                           |
| Custom Hermitian operator         | `Observable(matrix, sites=0)` or `Observable(matrix, sites=[0, 1])`    |

Two-site local operators support adjacent sites and the periodic end-to-end bond
on qubit chains. Matrix factors follow the supplied site order. Bitstring
probability requests cannot share an observable list with ordinary operators or
entanglement diagnostics; shot counts can accompany ordinary observables.

For MPS entanglement across the bond between sites 1 and 2, request the adjacent
pair:

```{code-cell} python
diagnostic_params = AnalogSimParams(
    observables=[
        Observable("entropy", sites=[1, 2]),
        Observable("schmidt_spectrum", sites=[1, 2]),
    ],
    elapsed_time=2.0,
    dt=0.05,
)
```

These diagnostics work at sampled times or circuit checkpoints, including noisy
MPS runs. Spectra contain descending Schmidt coefficients, with up to 500
entries and `NaN` padding. With noise, YAQS averages the pure-trajectory
entropies and coefficients; these are not the entropy or spectrum of the
ensemble's mixed density matrix. Missing ranks contribute zero to coefficient
means, while ranks absent from every trajectory remain `NaN`.

## Refine accuracy and retain results

Change one source of error at a time and compare the observable that matters for
your calculation:

| Setting         | When to change it                                                                                          |
| --------------- | ---------------------------------------------------------------------------------------------------------- |
| `dt`            | Reduce it at fixed analog duration to check time-step error.                                               |
| `num_traj`      | Increase it to reduce uncertainty in noisy trajectory averages.                                            |
| `max_bond_dim`  | Increase the MPS bond cap if it limits the evolving state. Larger bonds cost memory and time.              |
| `svd_threshold` | Reduce it to retain more information during tensor truncation.                                             |
| `krylov_tol`    | Reduce it to tighten local matrix-exponential solves in TDVP or BUG. This does not tighten SVD truncation. |

Keep an explicit trajectory budget when comparing presets, since a preset
changes that budget along with the numerical settings. Tensor truncation and
integrator controls concern MPS evolution; dense backends use different
numerical methods. See {doc}`representation_comparison` before changing them.

Set `random_seed` to a non-negative integer to repeat jump decisions and sampled
static disorder for the same input and configuration. It does not seed random
state initialization or measurement-shot sampling.

Set `get_state=True` to retain a supported final state in `result.output_state`.
Noisy MPS, statevector, and circuit runs cannot return one final pure state for
the trajectory ensemble. Noisy density-matrix evolution can return its final
mixed state. Unitary list-of-state ensembles do not return a final ensemble
state.

## Advanced numerical options

:::{dropdown} Analog integrators and two-time correlations

MPS analog evolution defaults to `evolution_mode=EvolutionMode.TDVP`. Import
`EvolutionMode` from `mqt.yaqs` and select `EvolutionMode.BUG` to use the BUG
integrator. See {doc}`analog_simulation` for the workflow.

`order=1` or `order=2` selects the TJM splitting order for noisy MPS evolution.
This is separate from `tdvp_mode`, which controls the TDVP state updates:

- `"2site"` (default) allows bond growth through two-site updates.
- `"1site"` uses single-site updates with fixed bond dimensions.
- `"dynamic"` switches between single-site and two-site updates.

`tdvp_sweeps` defaults to 1. Increasing it subdivides each TDVP evolution step
into symmetric substeps with the same total evolution time. In noisy analog
runs, noise still acts on the full physical timestep `dt`; extra TDVP substeps
do not refine that noise timestep.

For two-time correlations, pass `multi_time_observables=[(A, B)]` with a
noiseless MPS `list[State]` and a static Hamiltonian. `B` acts at time zero and
`A` at the later time. The complex results appear in `multi_time_results`, with
one row per pair. See {doc}`ensemble_evolution` for the definition and example.

:::

:::{dropdown} Circuit gate-application modes

Keep `gate_mode="mpo"` unless you need to compare another method. It applies
gates directly and compresses the result using the configured tensor truncation.
The available modes are:

| Mode              | Two-qubit gates                                                                     |
| ----------------- | ----------------------------------------------------------------------------------- |
| `"mpo"` (default) | Direct local updates for neighbors; an extended gate MPO for separated sites.       |
| `"swaps"`         | Route separated sites together with SWAPs, apply the gate, and restore their order. |
| `"tdvp"`          | Direct local updates for neighbors; generator-based TDVP for separated sites.       |
| `"full-tdvp"`     | Generator-based TDVP for both neighboring and separated sites.                      |

TDVP gate paths are variational approximations and can miss the bond growth an
entangling gate requires. Additional `tdvp_sweeps` may help, but do not
guarantee exact gate application. Use `"mpo"`, or `"swaps"` for two-qubit gates,
for direct application up to the configured truncation. Digital generator-based
TDVP requires `tdvp_mode="2site"`.

Matrix-backed custom gates without an analytic generator use direct local
updates for neighbors and the MPO path for separated sites, including in TDVP
modes. Gates on three or more qubits use an MPO, except supported product-form
generators such as `ccx` and `ccz` in TDVP modes. See
{ref}`circuit-custom-gates` for custom gate inputs.

:::

:::{dropdown} SVD truncation modes

`trunc_mode="discarded_weight"` is the default. `svd_threshold` limits the sum
of discarded squared singular values. The other modes interpret the threshold as
a fraction of total squared weight (`"relative_discarded_weight"`), a ratio to
the largest singular value (`"relative"`), or an absolute singular-value cutoff
(`"hard_cutoff"`). `max_bond_dim` can force further truncation in every mode.
Presets do not change `trunc_mode`.

:::

## Related guides

- {doc}`quickstart` — simulation and characterization workflows.
- {doc}`analog_simulation` — noisy spin dynamics and convergence checks.
- {doc}`circuit_observables` — circuit dynamics and sampling checkpoints.
- {doc}`circuit_shots` — noisy readout distributions.
- {doc}`simulator_initialization` — execution controls and result fields.

Full constructor signatures are in the API reference for
{class}`~mqt.yaqs.AnalogSimParams` and {class}`~mqt.yaqs.DigitalSimParams`.
