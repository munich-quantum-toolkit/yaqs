# Upgrade Guide

This document describes breaking changes and how to upgrade. For a complete list of changes including minor and patch releases, please refer to the [changelog](CHANGELOG.md).

## [Unreleased]

### Resolve width-complete WP22 implementations

WP22B adds a fail-closed repository implementation catalog for the tiny-budget
smoke universe, the exact paired q6/q12 pilot, the nine-method q6 screen, and
dormant q6 confirmation. Catalog resolution returns a typed implementation
artifact together with its concrete repository runner adapter; confirmation
reuses the already screened configuration and cannot introduce a new treatment.

The q6 layerwise, noiseless, fixed-depth, Adam, and SPSA defaults retain their
existing identities. Pass `qubit_count=12` only for the secondary pilot
projection: its circuit topology and parameter counts are width-derived while
the layer schedule, gate ordering, optimizer settings, and treatment semantics
remain unchanged. q12 entries stay descriptive-only and cannot resolve for
screening, promotion, or confirmation. WP22C through WP22F remain responsible
for scheduled execution, custody, production evidence, and the final seal.

The `energy_adapt_vqe` smoke route checks only the existing target-bound
analytic TFIM implementation, with zero noisy training trajectories. It is
non-promotional and cannot be used as evidence for the preregistered noisy
Energy-ADAPT candidate; that treatment remains unsupported unless a separately
reviewed noisy energy executor is added later.

### Freeze the prospective WP22 execution protocol

WP22A adds strict, checksum-bearing records for noisy-training schedules,
trajectory membership, execution-seed derivation, checkpoint selection, fresh
evaluation, pilot diagnostics, operator growth, resource budgets, and scoped
publication-to-implementation bindings. The reviewed operational amendment is
checked in separately and anchored to both the immutable WP15
preregistration and the prospective WP22 implementation-plan commit.

Use these records to review and seal scientific choices; they do not authorize
a numerical paper claim. WP22B must still provide width-complete executable
catalogs, WP22C must execute schedules with exact restart behavior, and WP22D
through WP22F must run the pilot, screening, promotion, confirmation seal, and
primary analysis. Phase I and WP15 through WP21 identities are unchanged.

### Run reproducible top-down pruning pipelines

Use the WP21 top-down pipeline builders for `topdown_random`,
`topdown_magnitude`, `topdown_impact_one_shot`, and
`topdown_impact_iterative`. Impact scores use the generalized
gate-occurrence parameter-shift derivative, so parameters shared by several
gate occurrences are differentiated as the sum of the occurrence shifts.
Primary native two-qubit budget claims prune compiler-derived entangler groups,
then deterministically remap retained parameters and resolve equal scores by the
sealed tie rule.

Every impact-scoring round binds its exact objective, pre-pruning circuit, and
fixed trajectory maps. Iterative impact pipelines alternate one pruning round
with a relaxation stage and require at least two pruning rounds, so the
iterative identity cannot collapse to a one-shot treatment. Those stage
boundaries can be verified and resumed through the Phase II artifact store.
Because recompilation can produce non-monotonic native counts, select only
observed reachable strata at or below the cap; an unavailable stratum produces
typed infeasibility evidence instead of an invented exact resource match.

New stage metadata uses the v3 schema to distinguish the circuit sampled for
fixed maps from the stage output circuit. Existing checksum-sealed v2 metadata
remains verifiable and reopenable without rewriting its stored bytes.

Noisy fine-tuning is an optional stage after pruning, not part of the pruning
method identity, and final testing always uses fresh streams separated from
training, scoring, relaxation, and checkpoint selection. These pipelines are
not reconstructions of the historical one-shot magnitude-pruned CSV: that
archive remains historical evidence with its original semantics. WP22 owns the
common training runner, artifact-level pilot orchestration, screening, and
final seal; WP21 alone makes no numerical or promotion claim.

### Compare noisy fine-tuning with fair Phase II controls

Use the builders in `benchmarks.state_preparation.phase2.fair_controls` to
construct the noiseless-final-stage, direct fixed-depth noisy, independent
fixed-CRN, independently resampled, modern cross-CRN, Phase-I-fixture, and
unpruned-depth controls. The modern cross-CRN control is distinct from the exact
WP19 legacy cross-CRN reproduction. The matched layerwise controls preserve the
corrected WP19 topology and stage budgets; secondary unpruned controls carry
compiler-derived match, residual, or excess evidence and cannot be presented
as exact resource matches when they miss the cap. Execute the registered direct
q6/depth-four baseline with
`FixedDepthBMPDStageRunner`; the modern layerwise controls use
`LayerwiseBMPDStageRunner`. Phase-I-fixture and unpruned descriptors remain
explicitly secondary inputs for the WP22 orchestration layer and cannot enter
sealed screening or confirmation roles.

`ParameterShiftAdamStageAdapter` implements an exact two-evaluation Pauli
parameter-shift gradient followed by bias-corrected Adam, and
`SPSAStageAdapter` implements one-based Spall schedules with a fresh paired
Rademacher perturbation stream per update. Resolve their fixed or layerwise
pipeline templates and run them through `BMPDCompetitorStageRunner`, whose
`FixedRateNoisyCompetitorObjective` binds an authorized target, the standard
noise providers, and every sampled map. Generic callbacks are retained only
for numerical adapter tests and cannot produce publishable stage evidence.
Completed target-bound executions convert to ordinary Phase II checkpoints
without being mislabeled as resumable Krotov states. SPSA's iteration-zero
monitor and first update share the first CRN window; each subsequent update
uses its own newly sampled window.

For family-wide noisy operator growth, use
`run_standard_fixed_rate_noisy_operator_growth`; it constructs a
`StandardFixedRateNoisyOperatorGrowthEvaluator` bound to an authorized target
and returns a self-verifying result with the exact pool, growth policy,
objective requests, resources, and trajectory provenance. The analytic
`adapt_style_state_preparation` entry point is a reference calculation and is
not promotion eligible. `target_bound_energy_adapt_vqe` derives the genuine
TFIM Hamiltonian from an authorized TFIM target specification and rejects
non-TFIM targets. The generic reference-only `energy_adapt_vqe` returns typed
zero-work not-applicable evidence for non-TFIM families; the TFIM comparator
remains ineligible for family-wide promotion. WP22 must wrap standalone
operator-growth results in its runner/artifact orchestration before screening;
WP20 does not fabricate a Krotov resume state for them.

Use `measure_circuit_resources` and `WP20WorkLedger` for the common logical and
Quantinuum-native counting rules and complete optimizer/trajectory work.
`select_reachable_resource_stratum` reports residual gaps for discrete paths,
`deterministic_pareto_frontier` constructs the frozen two-resource frontier,
and the paired-block utilities prohibit shared initialization, optimizer,
training-trajectory, and checkpoint-selection streams. Event-level final-test
coupling additionally requires two complete matching `PipelineEvaluationConfig`
protocols and full native-event alignment. Algorithmic work is stored in the
WP20 ledger. The WP18 execution boundary attaches wall time and the
`tracemalloc` Python-allocation peak to staged runs; standalone operator-growth
results receive those runtime measurements only after WP22 supplies their
runner/artifact wrapper. Neither boundary claims a process-wide memory peak.

### Run bottom-up layerwise BMPD with noisy fine-tuning

Use `build_layerwise_bmpd_crn_legacy_v1_template` only for the isolated
historical reproduction. It fixes q8 depths 1 through 4, four 100-iteration
noiseless growth stages, RandomState initialization, the historical reused
optimizer seeds, and a 200-iteration three-trajectory cross-CRN fine-tune under
`ibm_inspired_pauli_legacy_v1`. `resolve_layerwise_bmpd_crn_legacy_v1_pipeline`
binds one of the five legacy target seeds and `LayerwiseBMPDStageRunner` executes
the resulting stages through the Phase II artifact path.

The checked-in `legacy_tfim_targets_v1.json` vectors are WP19 reconstructions,
not recovered archived vectors. Load them with `load_legacy_target_collection`;
the records preserve the exact historical generator and current reconstruction
runtime, disclose missing archival provenance, and validate a fresh eigensolver
run phase-invariantly at the declared tolerance.

For new experiments, use `build_layerwise_bmpd_crn_v2_template` and supply both
training and checkpoint-validation trajectory counts explicitly from frozen
pilot evidence. The corrected profile uses standard fixed-rate noise,
independent updates, disjoint derived streams, a separately fixed validation
ensemble, checkpoints at iteration zero/every ten/final, and earliest-iteration
tie breaking. A changed count, refresh policy, update rule, cadence, or depth
schedule is a different configuration and must not retain the v2 identity.

`derive_legacy_krotov_trajectory_seed` and the sampler's
`legacy_linear_seed=True` option exist solely for this reproduction. New code
must keep the default hash-derived trajectory seeds.

Legacy seed arithmetic and legacy compact-map replay are separate policies.
Only the historical fixed-CRN fine-tuning stage replays compact Pauli maps
without per-gate normalization; the historical independent evaluation retains
normalized trajectories, and corrected methods remain on the modern replay
path even if they use the same noise condition.

The five-target q8 job is deliberately opt-in and can be started from the
repository root:

```console
uv run python -m benchmarks.state_preparation.phase2.run_historical_reproduction \
  --output-root output/wp19_historical_reproduction \
  --execute-expensive
```

Add `--resume` after an interruption; `--overwrite` instead replaces only its
managed artifacts. The command exits with status `0` only when all five rows
reproduce within tolerance, `1` for a complete but discrepant comparison, and
`2` for a target, setup, or job-level failure.

The job holds one output-root lock for its complete lifetime and seals one
launch manifest containing the exact tracked implementation, lockfile, and
study-input bytes. It rechecks that snapshot before every target and before
publishing the report; every row records its target-specific WP18 runtime
fingerprint, and the report binds both the source manifest and runtime.

### Persist and resume Phase II staged pipelines

Use `benchmarks.state_preparation.phase2.capture_resumability_fingerprint` to
seal the starting Git commit, complete pipeline prefix, resolved dependency
versions, tracked execution sources, lockfiles, and study inputs. Pass a
dedicated generated-output directory separately; output paths are rejected as
fingerprint inputs so a run does not invalidate itself as it writes evidence.

Create `Phase2ArtifactStore(output_dir, pipeline, fingerprint)` for a new run
and pass it to `Phase2PipelineExecutor`. A stage callback receives only the
resolved stage and predecessor parameters, and each successful callback is
persisted before the next stage begins. Reopen the same store with
`resume=True`; it verifies the canonical stage prefix, referenced checkpoints,
traces, metadata, fixed maps, source parameters, and provenance, then skips
every completed stage. Existing output is never reused implicitly, and
`overwrite=True` removes only the store's known versioned outputs.
Mutations are serialized across processes and compare the retained manifest
before writing. If another handle advances the store, reopen it before
continuing instead of retrying through the stale object.

Standalone WP17 execution may still use raw arrays or MPS targets and a custom
initial state. Publishing genuine WP17 evidence into the Phase II store is the
stricter scientific boundary: it requires the sealed objective binding to name
the pipeline's authorized `MaterializedTarget` and the computational-zero
initial state before any artifact is written.

For a first stage using `load_checkpoint`, provide a relative external-checkpoint
path rooted at the process's launch directory and an `ExternalCheckpointRef`
derived from the producer result. The store verifies the source before creating
or overwriting output, copies its exact bytes into the managed checkpoint area,
and uses that sealed copy for all later resumes. The source may therefore move
or disappear after initialization, and the executor always hands the selected
validation checkpoint—not merely the producer's last iterate—to the stage.

Pipeline and complete stage-prefix mismatches always reject resume. For an
otherwise identical pipeline, runtime-fingerprint drift also rejects scientific
resume, but exploratory work may continue by constructing a
`NonScientificResumeOverride` for the exact stored/current fingerprint pair and
supplying it as `resume_override`; the override is checksum sealed and retained
in the artifact history instead of disguising the drift.

After `store.pipeline_result` becomes available, construct
`ParallelPhase2Evaluator(store, deserialize_circuit)` for final-test fan-out.
For each attempt with pending rows, the materialization callback returns a
`MaterializedCircuitPayload` whose deterministic bytes identify the runtime
circuit. Those bytes are checksum verified, reconstructed through the trusted
decoder, and persisted or matched against the existing circuit artifact before
the attempt is committed. A later resume may rematerialize the circuit when
rows remain pending, but successful rows are skipped. Each evaluation callback
returns a `PipelineEvaluationMeasurement` with row-local fidelities, map
ensembles, work, provider identity, time, and memory. Canonical JSONL is the
authority for the derived CSV; the required checksum-sealed manifest is the
commit baseline used to detect ledger rollback relative to the retained
manifest on resume. Detecting a consistent rollback of the entire store would
require an external monotonic anchor.

Low-level integrations can use `StageParameterCheckpoint` and
`create_phase2_trajectory_sidecar`/`read_phase2_trajectory_sidecar` directly.
Both codecs use bounded deterministic NPZ envelopes and verify scientific
metadata as well as exact bytes; do not replace them with unvalidated
`numpy.load` calls.

### Run fixed-rate noisy Krotov as a Phase II training stage

Use `benchmarks.state_preparation.phase2.NoisyKrotovCircuitBinding` and
`execute_fixed_rate_krotov_stage` for the new benchmark-grade path. The adapter
accepts a resolved `TrainingStageConfig`, logical parameterized circuit, target,
and initial parameters; it intentionally accepts no final-test configuration.
It supports all ten standard fixed-rate profiles, the frozen
`ibm_inspired_pauli_legacy_v1` physical-noise profile, independent pathwise
updates, separately labelled cross dense-sum updates, fixed/resampled/refreshed
trajectory ensembles, and independently seeded checkpoint validation.
Checkpoint validation includes iteration zero. When resuming a chunk, pass the
previous execution's `resume_state` together with the required replay ensembles;
it binds the target, initial state, final parameters, prior best checkpoint, and
cumulative work so foreign or incomplete resumes are rejected.
Cross updates expose their dense `R**2` trajectory-pair count separately because
the strict Phase II normalized-work schema has no cross-pair counter.

Use `KrotovFixedMapEnsemble` to serialize, checksum, verify, and replay exact
trajectory maps. Its logical ensemble identifier depends only on the resolved
random-stream seed and explicit stage/ensemble/trajectory/refresh coordinates;
the separate content checksum also binds the stage, circuit, provider, and map
bytes. Existing Phase I Krotov methods and their historical seed behavior are
unchanged.

### Run Krotov state preparation through the method adapter

Use `benchmarks.state_preparation.KrotovStatePreparationMethod` to construct
the shared BMPD ansatz, initialize parameters, run noiseless full-batch Krotov
optimization, extract final parameters and training fidelity, and perform
noiseless evaluation. The adapter accepts the typed benchmark configuration
records and returns complete normalized optimizer metadata.

Random initialization uses a dedicated NumPy generator seeded by
`InitializationConfig`; it never reads or modifies NumPy's global random
state. Warm starts verify their declared SHA-256 checksum before decoding and
accept versioned Krotov NPZ checkpoints or legacy numeric NPY vectors.

Use `state_preparation_training_id` rather than `BenchmarkConfig.run_id` to
cache optimization artifacts. The training identifier excludes test noise and
evaluation policy so the exact same trained parameters can be reused across
every test configuration. `train_state_preparation_method` performs that
method-generic training boundary once and returns a
`StatePreparationTrainingArtifact` with immutable parameter bytes, detached
optimizer metadata, training fidelity, and a serialized checkpoint for
evaluation fan-out. Pass the validated `TargetCollection` matching the
configuration; the helper resolves the record itself and verifies the fixture
format and checksum before computing. Failures carry their reporting phase and
original exception in `StatePreparationTrainingError`.

Checkpoints contain parameters rather than target-specific optimizer traces.
They are bound to the complete data-free logical circuit, use explicit
little-endian numeric encodings, and reject incompatible layouts before use.
Training artifacts are factory-created so arbitrary checkpoint bytes cannot be
attached to otherwise valid parameters.
Checkpoint serialization returns bytes; atomic file writes and result-stream
persistence remain the reporting layer's responsibility.

### Collect state-preparation circuit statistics centrally

Use `benchmarks.state_preparation.collect_circuit_statistics` instead of
counting gates in benchmark scripts. The collector records configured BMPD
depth, brickwall layer count, trainable parameter count, dependency depth, and
one- and two-qubit gate counts for both the logical and native circuits, with
optional counts by gate name.

For standard-noise rows, select the logical evaluated representation. For
Ballarin rows, pass the final `BallarinCircuitMaterialization` and select the
native representation. The latter reports counts after threshold pruning and
safe basis-change cancellation while retaining the logical counts as extended
metadata.

### Materialize Ballarin circuits before final evaluation

Use `benchmarks.state_preparation.materialize_ballarin_circuit` after
`compile_quantinuum_native` and noiseless optimization. The materializer binds
every native angle, canonicalizes `RZZ` angles to `[-pi, pi)`, removes rotations
whose canonical magnitude is at most `1e-4` together with any corresponding
compiler basis-change round trip, and cancels only exact inverse
compilation-only basis changes.

The returned `BallarinCircuitMaterialization` preserves stable native gate IDs
and exposes compact final indices plus explicit pruning and cancellation
provenance. Its authoritative `FrozenNativeCircuit` is a fully bound,
zero-parameter executable snapshot. Use the same snapshot for final noiseless
and noisy evaluation; call `to_parameterized_circuit()` only when a detached
mutable copy is required.

Construct the matching post-gate product-Pauli channel with
`create_ballarin_noise_provider`. It acts only after retained native `RZZ`
gates, uses the canonical angle magnitude for the Ballarin calibration, and
does not compose with the common one- or two-qubit benchmark rates.

### Compile state-preparation circuits to the Quantinuum-native basis

Use `benchmarks.state_preparation.compile_quantinuum_native` to preserve
one-qubit gate operations and native `RZZ` rotations while rewriting every
logical `RXX` or `RYY` into noiseless basis changes around exactly one
angle-preserving `RZZ`. All compiled one-qubit gates are marked noiseless under
the Ballarin convention. The returned immutable mapping records retain source
indices, parameterization, native indices, and the complete basis-change group
required for later pruning.

Unsupported two-qubit gates are rejected instead of being passed through.
Compilation neither resolves nor canonicalizes angles, and it preserves the
logical circuit's parameter-vector size, including unused indices.
Mapping indices describe the pre-pruning circuit; downstream materialization
should build a new circuit and mapping together while retaining each gate's
stable `native_gate_id`.

### Construct standard benchmark noise through the gate-local registry

Use `benchmarks.state_preparation.create_standard_noise_provider` for the ten
standard dephasing and depolarizing benchmark identifiers. The provider builds
a fresh local TJM model from each gate's support, applies the full benchmark
strength to every jump operator, and tags realized maps with the selected
identifier.

Pass the explicit `NoiseConfig.tjm_dt` to `KrotovTJMOptions` and use
`apply_noise_to="all"`; the provider itself implements the identifier's
one-qubit versus two-qubit placement. Do not construct one global model or
combine a standard provider with `ballarin_coupled`, which remains a separate
exact product-Pauli channel.

### Sample independent product-Pauli benchmark noise locally

State-preparation benchmarks can now use
`benchmarks.state_preparation.sample_product_pauli_channel` to sample two
independent local Pauli distributions. The helper consumes one trajectory-local
random draw per site and returns only the realized non-identity
`LocalOperator`s, in call order.

Apply the returned one-site operators sequentially, or wrap them in a
`RandomUnitaryInstruction` for a gate-local provider. Do not combine product
outcomes into a weighted two-site matrix: the helper already samples the
channel probabilities exactly, and its matrices are bare unitary Paulis.

### Use gate-local providers for context-dependent circuit noise

Noisy Krotov evaluation and training functions now accept an optional
`noise_provider` keyword. Providers receive immutable gate metadata and the
trajectory-local random-number generator, and may return either a local
`NoiseModel` or a realized `RandomUnitaryInstruction`. A deliberate mixed
channel can return `CompositeGateNoiseInstruction`, whose tagged components run
in tuple order. Existing positional global `NoiseModel` calls remain unchanged.

Passing both a global model and a provider is rejected. Move the relevant local
model into an explicit composite provider when mixed mechanisms are required.

### Use the validated loader for state-preparation benchmark targets

Code consuming `benchmarks/state_preparation_target_states.json` should use
`benchmarks.state_preparation.load_target_collection`, `load_target`, or `iter_targets` instead of
parsing the fixture directly. The loader validates the versioned fixture and target metadata,
returns immutable records, and exposes the raw-file SHA-256 checksum required for reproducible run
provenance.

```python
from benchmarks.state_preparation import load_target

target = load_target(6, "tfim_critical")
state_vector = target.state_vector_copy()
```

### Run state-preparation matrices through the reproducible CLI

Use `python -m benchmarks.state_preparation.runner` instead of assembling the
benchmark matrix manually. The runner freezes smoke, minimum, and full presets,
prints the resolved matrix, trains each shared checkpoint once, evaluates every
test-noise cell independently, and writes schema-validated JSONL, CSV,
checkpoints, and a manifest.

```bash
uv run python -m benchmarks.state_preparation.runner --preset smoke
uv run python -m benchmarks.state_preparation.runner --preset minimum
uv run python -m benchmarks.state_preparation.runner --preset full --resume
```

The minimum preset is defined by noiseless and Ballarin evaluation plus
`dephasing_1s_all`, `dephasing_2s_2q`, `depolarizing_1s_all`, and
`depolarizing_2s_2q`. Earlier development snapshots of the unreleased runner
selected `1s_1q` and `1s_2q` instead; explicitly pass those `--noise-id`
options if that noncanonical matrix must be reproduced.

Existing output is never replaced implicitly. Pass `--resume` to validate and
continue it, or `--overwrite` to start a replacement stream. The canonical
result file is `results.jsonl`; use
`benchmarks.state_preparation.read_jsonl_records` rather than parsing its
versioned records ad hoc.

### `simulator.run` becomes `Simulator(...).run(...)`

The free `mqt.yaqs.simulator.run` function has been replaced by a `Simulator` class.
`Simulator` owns the execution-side configuration (parallel vs. serial execution, worker count,
progress reporting, multiprocessing context, retry policy); the physics inputs are passed to
`Simulator.run`. `Simulator.run` returns a `Result` (from
`mqt.yaqs.core.data_structures.result`) that holds all simulation outputs. The `*SimParams`
object you pass in is never mutated.

**Before:**

```python
from mqt.yaqs import simulator

simulator.run(state, op, sim_params, noise_model, parallel=True)
```

**After:**

```python
from mqt.yaqs import Simulator

sim = Simulator()
result = sim.run(state, op, sim_params, noise_model)
```

`show_progress` and `num_threads` were removed from `AnalogSimParams`, `StrongSimParams`, and
`WeakSimParams`. Pass `show_progress` to `Simulator` instead; `num_threads` was unused and has been
deleted.

### `threshold` renamed to `svd_threshold` on `*SimParams`

The SVD bond-truncation setting on `AnalogSimParams`, `StrongSimParams`, and `WeakSimParams` is now
`svd_threshold` (attribute and constructor keyword). Simulation presets use the key `svd_threshold`
in `SIMULATION_PRESETS`. This distinguishes SVD truncation from `krylov_tol` (Krylov/Lanczos matrix
exponential) and from unrelated `threshold` parameters elsewhere (for example
`EquivalenceChecker(threshold=...)`).

**Before:**

```python
params = AnalogSimParams(threshold=1e-8)
value = params.threshold
```

**After:**

```python
params = AnalogSimParams(svd_threshold=1e-8)
value = params.svd_threshold
```

### `digital.equivalence_checker.run` becomes `EquivalenceChecker(...).check(...)`

The free `mqt.yaqs.digital.equivalence_checker.run` function has been replaced by
`EquivalenceChecker` (now exposed at `mqt.yaqs.EquivalenceChecker`). `EquivalenceChecker` owns the
numerical thresholds (`threshold`, `fidelity`); the two circuits are passed to
`EquivalenceChecker.check`. The return value is unchanged: a `dict` with keys `equivalent` and
`elapsed_time`.

**Before:**

```python
from mqt.yaqs.digital.equivalence_checker import run

result = run(circuit1, circuit2, threshold=1e-6, fidelity=1 - 1e-13)
```

**After:**

```python
from mqt.yaqs import EquivalenceChecker

checker = EquivalenceChecker(threshold=1e-6, fidelity=1 - 1e-13)
result = checker.check(circuit1, circuit2)
```

### Read outputs from `Result`, not `*SimParams`

`Simulator.run` no longer writes outputs onto the `*SimParams` instance you pass in. Capture the
return value and read fields from `Result`. `result.sim_params` still references your original
configuration object (unchanged).

| Old (`sim_params`)                          | New (`result`)                 |
| ------------------------------------------- | ------------------------------ |
| `sim_params.observables[i].results`         | `result.expectation_values[i]` |
| `sim_params.output_state`                   | `result.output_state`          |
| `sim_params.noise_model`                    | `result.noise_model`           |
| `sim_params.results` (weak)                 | `result.counts`                |
| `sim_params.measurements`                   | `result.measurements`          |
| `sim_params.multi_time_observables_times`   | `result.multi_time_times`      |
| `sim_params.multi_time_observables_results` | `result.multi_time_results`    |

Removed from `*SimParams`: `noise_model`, `output_state`, `multi_time_observables_times`,
`multi_time_observables_results`, `measurements`, `results`, `aggregate_trajectories`,
`aggregate_measurements`. Observable configuration (`observables`, `multi_time_observables`, etc.)
remains on `*SimParams`.

`Observable` no longer carries run outputs. After `Simulator.run`, read
`result.expectation_values[i]` (aggregated expectations), `result.trajectories[i]` (per-trajectory
data), and `result.times` (shared analog time grid). `result.observables[i]` is still the gate/sites
metadata for observable _i_.

### MPS bond diagnostics are automatic on `Result`

`runtime_cost`, `max_bond`, and `total_bond` are no longer configured as `Observable` instances.
For MPS-backed analog and strong-digital runs, `Simulator.run` fills `result.runtime_cost`,
`result.max_bond`, and `result.total_bond` (1D arrays aligned with `result.times` or the
strong-sim layer grid). MCWF, Lindblad, and weak digital runs leave these fields as `None`.

**Before:**

```python
sim_params = AnalogSimParams(observables=[Observable(Z(), 0), Observable("max_bond")])
result = sim.run(state, H, sim_params)
max_bond_curve = result.expectation_values[-1]
```

**After:**

```python
sim_params = AnalogSimParams(observables=[Observable(Z(), 0)])
result = sim.run(state, H, sim_params)
max_bond_curve = result.max_bond
```

**Before:**

```python
sim = Simulator()
sim.run(state, op, sim_params, noise_model)
print(sim_params.observables[0].results)
```

**After:**

```python
sim = Simulator()
result = sim.run(state, op, sim_params, noise_model)
print(result.expectation_values[0])
```

### `simulator.run` uses `State` and `Hamiltonian`

Analog and circuit entry points no longer accept raw `MPS` / `MPO` objects. Use `State` (from
`mqt.yaqs.core.data_structures.state`) and `Hamiltonian` (from
`mqt.yaqs.core.data_structures.hamiltonian`) instead.

**Before:**

```python
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.simulator import run

psi = MPS(4, state="zeros")
H = MPO.ising(4, J=1.0, g=0.5)
params = AnalogSimParams(..., solver="MCWF")
run(psi, H, params, noise_model)
```

**After:**

```python
from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.core.data_structures.state import State

psi = State(4, initial="zeros", representation="vector")
H = Hamiltonian.ising(4, J=1.0, g=0.5)
params = AnalogSimParams(...)
sim = Simulator()
result = sim.run(psi, H, params, noise_model)
```

### End of support for x86 macOS systems

Starting with this release, we can no longer guarantee support for x86 macOS systems.
x86 macOS systems are no longer tested in our CI and we can no longer guarantee that MQT YAQS installs and runs correctly on them.

## [0.3.2]

### End of support for Python 3.9

Starting with this release, MQT YAQS no longer supports Python 3.9.
This is in line with the scheduled end of life of the version.
As a result, MQT YAQS is no longer tested under Python 3.9 and requires Python 3.10 or later.

<!-- Version links -->

[Unreleased]: https://github.com/munich-quantum-toolkit/yaqs/compare/v0.3.3...HEAD
[0.3.2]: https://github.com/munich-quantum-toolkit/yaqs/compare/v0.3.1...v0.3.2
