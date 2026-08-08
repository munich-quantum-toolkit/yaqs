# State-Preparation Benchmark Implementation Plan

This document describes the implementation work required to turn the benchmark
definition in
[`state_preparation_benchmarks.md`](state_preparation_benchmarks.md) into a
reproducible, end-to-end experiment suite.

Work packages 0-14 define the completed Phase I benchmark infrastructure.
Phase II, specified after the Phase I definition of done, adds versioned noisy
fine-tuning pipelines, historically faithful reproduction, related methods, and
publication-grade evidence without changing Phase I run semantics.

Phase I began from existing target-state fixtures, a brickwall ansatz, a
noiseless Krotov state-preparation optimizer, and generic trajectory-based noise
machinery. Its completed work added benchmark orchestration, exact gate-local
noise models, Quantinuum-native compilation, final-circuit accounting,
independent evaluation, and standardized reporting.

## Design decisions

### Phase I benchmark workflow

For each combination of method, target, qubit count, ansatz configuration, and
initialization seed:

1. Optimize the shared brickwall ansatz noiselessly.
2. Evaluate the final circuit without noise.
3. Evaluate the same trained parameters with fresh trajectories under every
   selected noise configuration.
4. Emit one result row per noise configuration.

Noisy optimization is not part of the canonical Phase I implementation because
that benchmark matrix specifies noiseless optimization followed by noisy
evaluation. Phase II adds it through separate versioned schemas, method
identities, presets, and result streams so existing Phase I configurations and
run identifiers remain unchanged.

### Ballarin product channel

The production implementation shall preserve the channel's product structure.
After each retained native `RZZ` gate on sites `i` and `j`, sample one local
Pauli independently on each site:

```text
P(I) = 1 - 3r
P(X) = r
P(Y) = r
P(Z) = r
```

Apply each sampled non-identity Pauli as a one-site operator. A trajectory may
therefore contain zero, one, or two local errors after a gate.

This is equivalent to one categorical draw from the 16 joint branches, but it is
preferable because it:

- directly expresses the product channel;
- avoids constructing 15 non-identity process records;
- avoids two-site SVDs for product operators such as `XX`;
- keeps channel probabilities independent of MPS truncation and norm drift; and
- fits the existing `KrotovNoiseMap.operators` sequence.

The explicit 16-branch distribution and a rate-calibrated one-step TJM model
shall be retained only as independent correctness oracles in tests.

### Two-site depolarizing set

Freeze two-site depolarizing noise as the nine strictly two-sided Pauli
products:

```text
XX  XY  XZ
YX  YY  YZ
ZX  ZY  ZZ
```

This keeps the support axes unambiguous:

- `single_site` contains operators with one non-identity factor;
- `two_site` contains operators with non-identity factors on both qubits; and
- a combined configuration contains both sets.

### Standard noise-strength convention

Use the benchmark's strength as the strength of each YAQS jump operator:

- `6.4e-4` for every applicable process after a one-qubit gate;
- `5.1e-3` for every applicable process after a two-qubit gate.

Freeze the circuit-TJM effective step to `dt = 1.0` for the standard benchmark
presets. A different step size defines a different simulator parameterization
and must therefore be an explicit configuration value rather than a hidden
default.

The run metadata must record this per-operator interpretation, the TJM step
size, the complete operator set, and any conversion performed by another
method.

### Exactness terminology

An exact trajectory sampler produces trajectories with exactly the channel's
probability law. A finite ensemble still introduces Monte Carlo uncertainty.
MPS truncation remains a separate approximation to the state representation.

## Test and change-control policy

The implementation shall be divided into small, behaviorally cohesive batches.
Every code change must include automated tests.

### Baseline

Before implementation begins:

```bash
git status --short
git diff --stat
uv sync
uv run pytest
uvx nox -s lint
uv run python benchmarks/generate_state_preparation_targets.py --check
```

Record all pre-existing failures before editing. The current worktree may
contain unrelated changes, which must not be reformatted, reverted, staged, or
committed with benchmark work.

### Before each batch

Run the tests covering the code about to change. For example:

```bash
uv run pytest -q tests/optimization/test_krotov.py
```

or:

```bash
uv run pytest -q tests/benchmarks/test_state_preparation_noise.py
```

### After each batch

1. Run the same targeted tests.
2. Run all tests for affected subsystems.
3. Run the mandatory lint suite:

   ```bash
   uvx nox -s lint
   ```

4. Run the full test suite whenever core simulation behavior or a public
   interface changes:

   ```bash
   uv run pytest
   ```

### Major milestones and final validation

```bash
uvx nox -s tests
uvx nox -s minimums
uv run python benchmarks/generate_state_preparation_targets.py --check
```

The full 12-qubit experiment matrix is not a CI test. CI shall run small exact
integration tests and one bounded 6-qubit smoke benchmark.

## Proposed code organization

Benchmark-specific functionality:

```text
benchmarks/state_preparation/
    __init__.py
    schema.py
    targets.py
    configuration.py
    noise.py
    circuits.py
    methods.py
    reporting.py
    runner.py
```

Reusable optimization functionality:

```text
src/mqt/yaqs/optimization/gate_noise.py
src/mqt/yaqs/optimization/parameterized_circuit.py
src/mqt/yaqs/optimization/krotov.py
```

Tests:

```text
tests/benchmarks/
    test_state_preparation_schema.py
    test_state_preparation_targets.py
    test_state_preparation_noise.py
    test_state_preparation_reporting.py
    test_state_preparation_runner.py

tests/optimization/
    test_gate_noise.py
    test_native_compilation.py
    test_krotov.py
```

The exact module boundaries may be adjusted to match neighboring code, but
benchmark orchestration should remain outside the public YAQS package. Only
reusable simulation and optimization abstractions should be added under `src/`.

## Work package 0: Establish the complete baseline

### Work

- Run the complete baseline commands.
- Inspect existing modifications in MPS and Krotov code because later work
  overlaps those modules.
- Confirm that coverage tooling is available.
- Record current test-suite runtimes.
- Identify generated benchmark outputs that should be ignored.
- Confirm that the target JSON is fresh.

### Acceptance criteria

- The initial repository state and any pre-existing failures are recorded.
- No repository content changes.
- Existing user changes are preserved.

## Work package 1: Typed configuration and result schema

### Data structures

Add typed records for:

- `BenchmarkConfig`
- `TargetSelection`
- `AnsatzConfig`
- `InitializationConfig`
- `OptimizerConfig`
- `EvaluationConfig`
- `NoiseConfig`
- `CircuitStatistics`
- `BenchmarkResult`
- `BenchmarkFailure`

Keep initialization in its own record because its rule, random seed, scale, or
warm-start checksum independently changes the trained artifact and stable run
identity. A warm-start filesystem path is provenance, but only the content
checksum belongs in the scientific run key.

Treat these records as fully resolved run cells rather than permissive input
templates. In particular:

- bind every target identifier to the fixture's exact format and generation
  seed;
- require distinct resolved seeds for initialization, optimizer ordering,
  training, and testing;
- require an explicit positive `tjm_dt` for standard TJM noise (the canonical
  presets use `1.0`, while another explicit value is a distinct
  parameterization); and
- reject noiseless trajectory-sidecar or confidence-interval requests and
  confidence intervals with fewer than two samples.

### Required result fields

Represent every field in the benchmark reporting template:

- method;
- qubit count;
- target identifier;
- noise identifier;
- target-generation seed;
- ansatz description;
- layer count;
- parameter count;
- evaluated circuit depth;
- one- and two-qubit gate counts;
- optimizer budget;
- training trajectories or shots;
- training fidelity;
- test noiseless fidelity;
- test noisy fidelity;
- test trajectories or shots;
- wall time; and
- notes.

### Additional provenance

Add:

- schema version;
- stable run identifier;
- initialization seed and rule;
- training-noise identifier;
- optimizer and trajectory seeds;
- test seed;
- complete optimizer hyperparameters;
- MPS truncation settings;
- logical and native gate counts;
- pruned native-gate count;
- logical pre-compilation, native pre-pruning, and final materialized-circuit
  noiseless fidelities;
- trajectory uncertainty and an optional `normal_clipped` confidence interval;
- optimization and evaluation wall times;
- YAQS, Python, NumPy, and SciPy versions;
- Git commit and dirty-tree flag;
- result status;
- mandatory parameter-checkpoint path; and
- mandatory checkpoint checksum.

Do not overload the benchmark's target-generation `seed` field with any runtime
seed. The confidence level, confidence-interval method, trajectory-sidecar
storage policy, and artifact path spelling are post-processing or output
policies and do not change the scientific run identifier.

### Tests

- Valid minimal and complete results.
- Missing required fields.
- Unknown target and noise identifiers.
- Fidelity range validation.
- Rejection of NaN and infinite values.
- JSON round-trip.
- CSV flattening and stable column ordering.
- Schema-version validation.
- Failure-record serialization.
- Deterministic run-identifier construction.
- Exact target-seed and fixture-format validation.
- Cross-record parameter, gate-count, pruning, and fidelity consistency.
- Strict derived-field type checking, including rejection of Boolean aliases
  for integer counts.
- Integer-spelled JSON numbers accepted and normalized for real-valued fields.
- Required successful checkpoints and Ballarin pre-pruning diagnostics.
- Confidence-interval method and sample-budget validation.

### Acceptance criteria

Every planned run can be represented losslessly and deterministically, and no
accepted configuration is structurally incapable of producing a valid success
record.

## Work package 2: Validated target loader

### API

Provide an API resembling:

```python
load_target_collection(path: Path | None = None) -> TargetCollection
load_target(num_qubits: int, target_id: str) -> TargetRecord
iter_targets(...) -> Iterator[TargetRecord]
```

### Validation

- Supported format version.
- Exactly one record for each `(num_qubits, target_id)` key.
- Supported qubit count and target identifier.
- State-vector length equal to `2**num_qubits`.
- Finite complex amplitudes.
- Unit norm within the documented tolerance.
- Valid target seed and parameters.
- Immutable arrays or defensive copies.

### Tests

- Load and look up all 18 checked-in targets.
- Filter by target and qubit count.
- Reject duplicate or missing records.
- Reject malformed complex pairs.
- Reject incorrect dimensions.
- Reject non-finite and unnormalized vectors.
- Preserve all target metadata.

### Acceptance criteria

No runner or method adapter parses target JSON directly.

## Work package 3: Gate-local noise-provider interface

This is the central reusable simulator change.

### Gate context

Each post-gate request must expose:

- gate index;
- gate name;
- gate sites;
- gate arity;
- resolved angle, when applicable;
- logical and native gate identifiers; and
- parameter index, when applicable.

### Provider output

Support two gate-local noise instructions:

1. a gate-local `NoiseModel` evaluated by existing TJM machinery; and
2. a state-independent random-unitary channel producing zero or more local
   operators.

A small tagged union or protocol is sufficient. Do not build a general
density-matrix or arbitrary Kraus framework unless another requirement needs
it.

### Compatibility and validation

- Existing global `NoiseModel` calls remain unchanged.
- Reject simultaneous global and gate-local noise unless an explicit composite
  provider is used.
- Validate that provider output acts only on the current gate's support.
- Ensure Ballarin cannot accidentally compose with common-rate noise.
- Reject unsupported state-dependent categorical channels.

### Realized-map diagnostics

Extend `KrotovNoiseMap` with optional, backward-compatible metadata:

- channel identifier;
- outcome labels;
- source gate index;
- resolved native angle; and
- identity or non-identity indication.

Its existing operator sequence remains the physical replay representation.

### Tests

- Regression tests for the current global-noise path.
- Provider invocation after the correct gates.
- Correct gate context and resolved angle.
- Excluded gates receive no request.
- Providers producing zero, one, or two operators.
- Invalid support and conflicting providers.
- Fixed-map forward replay.
- Adjoint pullback in reverse operator order.
- Fixed-seed reproducibility.
- Zero-noise equivalence.
- Existing fixed-map finite-difference tests.

### Acceptance criteria

Noise may depend on gate name, arity, index, sites, and current angle without
changing existing callers.

## Work package 4: Independent product-Pauli sampler

### Production helpers

Add helpers resembling:

```python
sample_local_pauli(distribution, site, rng) -> LocalOperator | None
sample_product_pauli_channel(
    first_site,
    second_site,
    first_distribution,
    second_distribution,
    rng,
) -> tuple[LocalOperator, ...]
```

The product sampler calls the local sampler independently for both sites and
returns only non-identity operators.

### Prohibited production behavior

- Do not materialize product Paulis as two-site matrices.
- Do not invoke a two-site SVD for `XX`, `XY`, or another product branch.
- Do not sample the 15 composite non-identity events independently.
- Do not embed square-root probabilities in operators and weight them again.

### Independent correctness oracles

Implement internal or test-only references for:

1. the explicit 16-branch product distribution; and
2. the rate-calibrated one-step TJM distribution with truncation disabled.

For a combined 15-branch model:

```math
P_{II} = (1 - 3r)^2,
```

```math
\Gamma = -\frac{\log P_{II}}{\Delta t},
```

```math
\gamma_j = \Gamma\frac{P_j}{1-P_{II}}.
```

For two independent local TJM calls:

```math
\Gamma_{\mathrm{local}}
= -\frac{\log(1-3r)}{\Delta t},
```

```math
\gamma_X = \gamma_Y = \gamma_Z
= \frac{\Gamma_{\mathrm{local}}}{3}.
```

These calibrated rates are simulator-specific reference weights, not a claim
that the finite-time continuous Lindblad semigroup equals the discrete channel.
Use this oracle only for normalized input states, unitary Pauli branches,
truncation-free operator application, and an otherwise empty local noise
invocation. Handle `P_II = 1` as the trivial channel and do not use the
logarithmic construction at `P_II = 0`. These restrictions are another reason
not to use rate conversion in the production Ballarin path.

### Tests

#### Probability algebra

- Bit-flip product probabilities.
- All 16 Ballarin branch probabilities.
- Six one-sided branches equal to `r * (1 - 3r)`.
- Nine two-sided branches equal to `r**2`.
- Normalization.
- Correct local marginals.
- Factorization of joint probabilities.

#### Validation

- Identity-only distribution.
- `r = 0`.
- Maximum valid local distribution.
- Negative probabilities.
- Probabilities exceeding one.
- Non-normalized input.

#### Sampling

- Fixed-seed reproducibility.
- Independent local draws.
- Zero, one, and two non-identity outcomes.
- Deterministic random-number boundary checks.
- Seeded frequency checks with statistically justified tolerances.

#### MPS behavior

- Two local `X` applications equal dense `X` tensor `X`.
- No two-site SVD occurs.
- Bond dimensions remain unchanged under local Pauli errors.
- Product-channel probabilities are independent of MPS norm and truncation.

#### Cross-validation

With truncation disabled, compare:

- the production independent sampler;
- exhaustive 16-branch density evolution;
- the calibrated one-step TJM model; and
- a dense Kraus reference.

### Acceptance criteria

Product-channel trajectories have the exact channel law without composite
two-site noise operations.

## Work package 5: Standard noise registry

### Identifiers

Implement:

```text
dephasing_1s_1q
dephasing_1s_2q
dephasing_1s_all
dephasing_2s_2q
dephasing_1s2s_all

depolarizing_1s_1q
depolarizing_1s_2q
depolarizing_1s_all
depolarizing_2s_2q
depolarizing_1s2s_all
```

### Per-gate construction

Build local process lists dynamically for each gate instead of filtering one
global model. Examples:

- `dephasing_1s_1q`
  - after a one-qubit gate: `Z` on that site at `6.4e-4`;
  - after a two-qubit gate: no process.
- `dephasing_1s_all`
  - after a one-qubit gate: `Z` at `6.4e-4`;
  - after a two-qubit gate: `Z` on each participating site at `5.1e-3`.
- `dephasing_1s2s_all`
  - after a one-qubit gate: local `Z` at `6.4e-4`;
  - after a two-qubit gate: local `Z` on both sites and `ZZ`, each at
    `5.1e-3`.

Depolarizing configurations use:

- `X`, `Y`, and `Z` for each single-site component; and
- the nine frozen products for each two-site component.

### Tests

Use a toy circuit alternating one- and two-qubit gates. For every identifier,
assert:

- exact noisy gate indices;
- exact process names and sites;
- exact strengths;
- exact process counts;
- no incorrect placement;
- correct mixed-rate behavior;
- correct combined support;
- registry lookup and serialization;
- unknown-identifier rejection; and
- no accidental composition with Ballarin noise.

### Acceptance criteria

Every recommended configuration in the benchmark definition has a validated
factory.

## Work package 6: Quantinuum-native compiler

### Rewrites

- Preserve native `RZZ(theta)`.
- Rewrite `RXX(theta)` into verified noiseless basis changes and one
  `RZZ(theta)`.
- Rewrite `RYY(theta)` in the same manner.
- Preserve parameter index, scale, offset, and sign.
- Mark compilation basis changes as noiseless.
- Reject unsupported two-qubit gates unless an explicit native decomposition
  is implemented.

Derive and numerically verify basis-change identities under YAQS site ordering
and Qiskit angle conventions. Do not rely on untested symbolic assumptions.

### Logical-to-native mapping

Retain:

- source logical gate index;
- source parameter index;
- native gate index;
- native angle expression; and
- compilation basis-change relationship.

### Tests

- Dense equivalence for random signed angles.
- Zero, half-turn, and near-boundary angles.
- Whole-circuit equivalence up to global phase.
- Both site orderings.
- Parameter-index preservation.
- Analytic derivatives against finite differences.
- Exactly one native `RZZ` per logical `RXX`, `RYY`, or `RZZ`.
- No Ballarin noise on basis changes.
- Clear failure for unsupported gates.

### Acceptance criteria

The shared logical ansatz can be converted into an equivalent, traceable native
circuit.

## Work package 7: Canonicalization, pruning, and Ballarin provider

### Final circuit materialization

After noiseless optimization:

1. Resolve all native angles.
2. Canonicalize every `RZZ` angle to `theta_canonical` in `[-pi, pi)`.
3. Set `a = abs(theta_canonical)`.
4. Remove rotations with `a <= 1e-4`.
5. Omit the corresponding basis-change round trip when a compiled `RXX` or
   `RYY` rotation is removed.
6. Perform only mathematically safe basis-change cancellations.
7. Freeze the circuit used for final evaluation and counting.

The Ballarin noiseless and noisy test fidelities must use the same
compiled-and-pruned ideal circuit. Preserve pre-compilation and pre-pruning
fidelities as diagnostics.

### Ballarin provider

For each retained native rotation, use the canonical magnitude `a` to compute:

```math
\epsilon(a)
= 2.1 \times 10^{-4}
+ 1.43 \times 10^{-3} a,
```

```math
r(a)
= \frac{1}{3}
  \left(
    1-\sqrt{1-\frac{5}{4}\epsilon(a)}
  \right).
```

Then invoke the independent product-Pauli sampler. Do not apply any common-rate
noise in this configuration.

### Tests

- Canonicalization around `-pi`, `pi`, `3*pi`, and `-3*pi`.
- Equal error strengths for opposite signed angles.
- Values below, at, and above the pruning threshold.
- Formula values at representative angles.
- Square-root domain validation.
- The consistency identity:

  ```math
  \epsilon
  = \frac{4}{5}
    \left[
      1-(1-3r)^2
    \right].
  ```

- No channel for a pruned rotation.
- Exactly one product-channel application for a retained rotation.
- No common-rate noise.
- Dense compiled-and-pruned state reference.
- Removal of redundant basis changes.
- Correct native counts after pruning.

### Acceptance criteria

`ballarin_coupled` implements the benchmark convention exactly.

## Work package 8: Circuit statistics

### Statistics

Collect:

- configured BMPD depth;
- ansatz layer count;
- logical circuit depth;
- native circuit depth;
- trainable parameter count;
- logical one- and two-qubit gate counts;
- final evaluated one- and two-qubit gate counts;
- native `RZZ` count;
- pruned rotation count; and
- optional counts by gate name.

Because BMPD depth produces twice as many brickwall layers, report both
configured depth and final layer count explicitly.

### Row-counting rule

- Standard-noise rows report the final logical primitive circuit counts.
- Ballarin rows report final compiled-and-pruned native counts.
- Extended metadata preserves both logical and native counts in all rows.

### Tests

- Closed-form small-circuit counts.
- Even and odd qubit counts.
- Zero depth.
- Initial product layer enabled and disabled.
- Independent Qiskit depth comparison.
- Native compilation counts.
- Threshold-pruning counts.
- Safe basis-change cancellation.
- Shared parameters and removed gates.

### Acceptance criteria

No benchmark script counts gates manually.

## Work package 9: Method adapter

### Interface

A method adapter must provide:

- method name and version;
- ansatz construction;
- parameter initialization;
- noiseless optimization;
- final parameter extraction;
- training-fidelity extraction;
- noiseless evaluation;
- optimizer metadata; and
- checkpoint serialization.

Implement `KrotovStatePreparationMethod` first.

### Optimization reuse

Train once per:

```text
method
+ target
+ qubit count
+ layer configuration
+ initialization
+ optimizer configuration
```

Reuse the trained parameters for every noisy test configuration.

### Initialization metadata

Require:

- rule;
- seed;
- distribution or warm-start source;
- distribution scale; and
- warm-start checksum, when applicable.

Do not use implicit global random state.

### Tests

- Fake adapter contract.
- Deterministic Krotov initialization.
- Zero- and one-iteration optimization.
- Complete optimizer metadata.
- Warm-start validation.
- Checkpoint round-trip.
- Failure conversion.
- Reuse of identical trained parameters across evaluations.

### Acceptance criteria

A second state-preparation method can be added without modifying target,
noise, reporting, or runner modules.

## Work package 10: Independent test evaluation

### Seed domains

Derive stable, disjoint seeds for:

- parameter initialization;
- optimizer ordering;
- training trajectories, when used;
- test trajectories; and
- repeated test evaluation.

Use `numpy.random.SeedSequence` or an equivalent stable construction. Do not
use Python's randomized `hash()`.

### Evaluation

For each trained artifact:

1. Record final training fidelity.
2. Evaluate the final logical circuit noiselessly.
3. Materialize the circuit required by the noise configuration.
4. Evaluate that materialized circuit noiselessly.
5. Evaluate it with fresh noisy trajectories.
6. Optionally store trajectory-level fidelities in a sidecar.
7. Store the mean and uncertainty.

### Additional statistical fields

Record:

- trajectory-fidelity standard deviation;
- standard error;
- optional confidence interval; and
- count of sampled non-identity events.

### Tests

- Disjoint train and test streams.
- Exact reproducibility of a fixed run.
- A changed test seed does not change optimization.
- Correct trajectory budget.
- Aggregated mean equals the sidecar mean.
- No training map is reused for testing.
- Noiseless fidelity is seed independent.
- Ballarin outcomes refresh for each independent evaluation.

### Acceptance criteria

Training and test ensembles cannot be shared accidentally.

## Work package 11: Reporting, checkpointing, and resumability

### Outputs

- JSON Lines as the canonical result stream.
- CSV as a derived convenience table.
- A run manifest.
- NPZ parameter checkpoints.
- Optional compressed trajectory sidecars.

### Stable run key

Include:

- method;
- target;
- qubit count;
- layer configuration;
- initialization;
- optimizer configuration;
- training-noise configuration;
- test-noise configuration; and
- evaluation budget.

### Reliability

- Atomic writes.
- Duplicate-run prevention.
- Resume completed runs.
- Compare stored Git and software provenance before treating an existing
  scientific run identifier as reusable; require an explicit override when the
  implementation fingerprint differs.
- Preserve successful rows after later failures.
- Explicit overwrite option.
- Validation before writing.
- Checkpoint checksum verification.
- Failure rows instead of silent zero-fidelity substitution.

### Wall-time semantics

Store:

- `optimization_wall_time_seconds`;
- `evaluation_wall_time_seconds`; and
- `wall_time_seconds`.

For an individual result row, total wall time is the optimization time plus
that row's evaluation time. Separate values avoid ambiguity when one
optimization is shared by many rows.

### Tests

- JSON Lines and CSV consistency.
- Atomic writes.
- Interrupted-run recovery.
- Duplicate prevention.
- Resume and overwrite behavior.
- Failure-row preservation.
- Checkpoint checksums.
- Stable run identifiers.
- Dirty Git state.
- Temporary output isolation.

### Acceptance criteria

A long-running 12-qubit sweep can be interrupted and resumed safely.

## Work package 12: CLI and presets

### Entry point

```bash
uv run python -m benchmarks.state_preparation.runner
```

### Options

- JSON configuration file;
- `--preset`;
- `--num-qubits`;
- `--target-id`;
- `--noise-id`;
- `--method`;
- `--num-layers`;
- `--initialization-seed`;
- `--optimizer-iterations`;
- `--train-trajectories`;
- `--test-trajectories`;
- `--output-dir`;
- `--resume`;
- `--overwrite`;
- `--dry-run`; and
- `--fail-fast`.

Use JSON for the first configuration format to avoid adding a parser
dependency.

### Presets

- `smoke`: one bounded 6-qubit case with every noise implementation.
- `minimum`: all 18 targets with noiseless, Ballarin, and the two minimum
  configurations from each standard noise family.
- `full`: all 18 targets and all 12 noise identifiers.

Per method, layer choice, and initialization seed:

- minimum preset: `18 * 6 = 108` result rows;
- full preset: `18 * 12 = 216` result rows.

### Execution

- Print the fully resolved matrix before starting.
- Train once and fan out evaluations.
- Default to sequential execution for reproducibility and memory safety.
- Add parallel execution only after resource measurements.
- Write progress and results without unsafe interleaving.

### Tests

- CLI parsing and configuration precedence.
- Minimum and full matrix cardinality.
- Filters and unknown identifiers.
- No output mutation in dry-run mode.
- Smoke run in a temporary directory.
- Partial-run resume.
- Deterministic output.
- Failure and fail-fast behavior.

### Acceptance criteria

The full experiment suite requires no source-code editing to configure or
launch.

## Work package 13: End-to-end numerical validation

### Small exact integrations

Use two- and three-qubit systems to exercise:

- target loading;
- brickwall construction;
- noiseless optimization;
- native compilation;
- pruning;
- every noise identifier;
- fresh evaluation;
- circuit statistics;
- serialization; and
- resume behavior.

### Exact density references

Compare with dense density-matrix evolution for:

- bit-flip product noise;
- Ballarin product depolarizing noise;
- single-site dephasing;
- correlated `ZZ`; and
- the nine-operator two-site depolarizing set.

For Ballarin, use exhaustive branch summation:

```math
\rho'
= \sum_{A,B}
  p_Ap_B
  (A_iB_j)
  \rho
  (A_iB_j)^\dagger.
```

Test trajectory convergence separately from the exact enumerated channel.

### Benchmark smoke test

Run one 6-qubit target with:

- a small brickwall configuration;
- a minimal optimization budget;
- every noise identifier; and
- a small, nonzero test-trajectory count.

### Coverage objectives

- At least 95% line and branch coverage for new benchmark modules.
- At least 90% branch coverage for new core noise and compiler paths.
- Every noise identifier exercised.
- Every practical validation branch covered.
- No flaky statistical assertions.
- Dense exact references for all channel families.
- Parameterized tests over angles, sites, seeds, and gate types.

### Acceptance criteria

The smoke benchmark covers the complete path from target fixture to final
result rows.

## Work package 14: Documentation and release notes

Update the benchmark definition to freeze:

- the nine two-site depolarizing operators;
- the per-operator strength convention;
- direct independent Ballarin sampling;
- exact channel sampling versus finite-trajectory estimation;
- angle canonicalization and pruning;
- native gate-count rules;
- seed derivation;
- minimum and full presets;
- train/test separation; and
- the result schema.

Also update:

- `CHANGELOG.md`;
- `UPGRADING.md` if a public API or existing behavior changes;
- benchmark usage documentation;
- example CLI commands; and
- guidance for interpreting sampling uncertainty.

### Acceptance criteria

A new contributor can reproduce the smoke and full configurations using only
the documentation.

## Deferred extension: exact Ballarin noise-aware optimization

Fixed-trajectory replay is not an exact gradient of an angle-dependent noisy
objective. If:

```math
L(\theta)
= \sum_z
  P_\theta(z)
  L_z(\theta),
```

then:

```math
\frac{\partial L}{\partial\theta}
= \sum_z
  P_\theta(z)
  \frac{\partial L_z}{\partial\theta}
+ \sum_z
  \frac{\partial P_\theta(z)}{\partial\theta}
  L_z(\theta).
```

The current pathwise Krotov pullback contains the first contribution but not
the derivative of the branch probabilities. Canonicalization, `abs(theta)`,
and threshold pruning also introduce nondifferentiable points.

If exact Ballarin noise-aware training becomes a benchmark requirement, add a
separate implementation based on one of:

1. density-matrix or locally purified tensor-network differentiation;
2. an analytic adjoint-channel derivative; or
3. a score-function estimator:

   ```math
   \mathbb{E}
   \left[
     \partial_\theta L_z
     + L_z\partial_\theta\log P_\theta(z)
   \right].
   ```

That work requires independent bias, variance, and finite-difference
validation. It is not required for noiseless training followed by fresh noisy
evaluation or for Phase II fine-tuning under angle-independent fixed-rate Pauli
noise. Ballarin remains evaluation-only throughout Phase II.

## Phase I definition of done

The implementation is complete when:

- all 18 target states are available through a validated loader;
- the shared brickwall ansatz is used consistently;
- all 12 noise identifiers are implemented;
- mixed one- and two-qubit rates work correctly;
- Ballarin uses two independent local Pauli draws;
- no production Ballarin path uses unnecessary two-site product-operator
  SVDs;
- native compilation is numerically verified;
- signed angles are canonicalized;
- native rotations at or below `1e-4` are removed;
- final logical and native circuit statistics are reported;
- training and testing use disjoint random streams;
- every result satisfies the reporting schema;
- runs are reproducible, resumable, and checkpointed;
- the minimum dry run resolves to 108 rows per method, layer choice, and seed;
- the full dry run resolves to 216 rows per method, layer choice, and seed;
- new code meets the coverage objectives; and
- full tests, nox tests, lint, target freshness checking, and the end-to-end
  smoke run pass.

## Phase II: Publication-grade noisy fine-tuning and method comparison

Work packages 0-14 and the preceding
[Phase I definition of done](#phase-i-definition-of-done) form the immutable
Phase I baseline.
The packages below are additive. They must not change the executable meaning,
serialization, preset cardinality, or stable run identifiers of any Phase I
configuration. Phase II clarifications elsewhere in this document do not reopen
the completed Phase I package bodies.

### Historical motivation

Before Phase I, the highest mean noisy fidelity among the three methods in the
shared `rigorous_benchmark_5states.csv` protocol came from a bottom-up layerwise
BMPD/Krotov procedure that:

1. grew the BMPD ansatz through configured depths `1 -> 2 -> 3 -> 4`;
2. relaxed each depth noiselessly;
3. warm-started the next depth by copying existing parameters and initializing
   only new parameters near zero;
4. fine-tuned the final circuit under a fixed-rate Pauli noise model using
   three cross-trajectory CRN trajectories; and
5. evaluated the final parameters with 500 separately sampled trajectories.

The historical CSV labels this procedure `ADAPT-VQE`, but it is not an
operator-pool ADAPT-VQE implementation. Phase II calls the method family
`layerwise_bmpd_crn`, with the exact historical implementation identified as
`layerwise_bmpd_crn_legacy_v1` and the corrected study implementation as
`layerwise_bmpd_crn_v2`.

For the five eight-qubit disordered-TFIM targets with historical seeds
`100`, `200`, `300`, `400`, and `500`, the archived simulated noisy fidelities
were:

```text
0.7714404528882339
0.8133339085571137
0.8112718858400323
0.7482008490029097
0.8082234870125117
```

Their mean is `0.7904941166601602`. These numbers are regression references for
historical reproduction, not success thresholds for the new scientific study.
The old experiment used one optimizer run per target, unequal method budgets,
only three fixed training trajectories, no retained test uncertainty, a default
test seed, and a hardware-inspired simulation rather than QPU execution.

### Phase II scientific questions

The full Phase II program investigates:

1. Does noisy fine-tuning improve fresh-test noisy fidelity relative to
   noiseless training under the preregistered primary resource constraint?
2. Does fixed-trajectory CRN generalize better than resampled or periodically
   refreshed trajectory training?
3. Does accurately named bottom-up layerwise growth provide a more favorable
   fidelity-resource Pareto frontier than fixed-depth training, genuine
   operator-pool growth, or top-down pruning?
4. Which gains persist across target families, initialization seeds, noise
   strengths, and qubit counts, and what is each method's failure probability?

The locked confirmatory study is deliberately narrower. It tests noisy versus
noiseless fine-tuning and, when they are distinct, the promoted method versus
`layerwise_bmpd_crn_v2` under the one primary resource and noise condition.
Family-stratified effects and failure probability are confirmatory only to the
extent frozen in the final seal. Broad comparisons among every CRN policy,
optimizer, growth/pruning family, noise strength, qubit count, and Pareto stratum
remain screening or exploratory analyses unless an exact contrast is included,
powered, and multiplicity-controlled in that seal.

The preregistered protocol must freeze exactly one primary resource constraint,
such as a native two-qubit gate-count budget with a normalized-compute cap.
Native depth, additional gate-count strata, and the complete Pareto frontier are
secondary analyses rather than simultaneous matching requirements.

The protocol must predeclare:

- the primary noise condition;
- target-family weights;
- a corrected `layerwise_bmpd_crn_v2` comparator;
- the contrast between noisy and noiseless fine-tuning;
- the contrast between the promoted method and
  `layerwise_bmpd_crn_v2` when they are distinct;
- superiority or non-inferiority margins;
- failure handling, including a failure-rate endpoint and an
  intention-to-treat sensitivity rule when no circuit is produced;
- cluster-aware sample-size or precision targets;
- multiplicity control for the primary contrasts; and
- the aggregation rule for secondary noise identifiers.

The primary endpoint is fresh-test noisy fidelity under that resource
constraint. Secondary outcomes are noiseless fidelity, failure probability,
native circuit statistics, trajectory-equivalent compute, wall time, and peak
memory. The final protocol must distinguish Monte Carlo uncertainty within one
trained artifact from variability across targets and independent optimization
seeds.

### Phase II guardrails

- Preserve Phase I schemas, method versions, CLI behavior, and run identifiers.
- Treat the archived scripts and CSVs as legacy evidence, never as
  confirmatory data.
- Use the `layerwise_bmpd_crn` family, not `ADAPT-VQE`, for staged growth of
  the fixed brickwall ansatz.
- Use the name `ADAPT-VQE` only for an implementation that selects operators
  from a frozen pool using the documented ADAPT rule.
- Describe IBM-, Google-, or Quantinuum-derived parameters as
  hardware-inspired simulation unless corresponding QPU data are collected.
- Never select hyperparameters, checkpoints, or methods using final test
  trajectories.
- Use distinct training, checkpoint-validation, screening-selection, and
  confirmatory target or random-stream domains.
- Never discard failed targets or seeds from primary summaries.
- Compare native counts in the same compilation basis and report normalized
  computational work in addition to optimizer iterations.
- Freeze the native basis, device connectivity, routing and SWAP policy,
  compilation optimization level, and logical-versus-native noise placement
  before screening begins.
- Treat fixed-CRN training fidelity as performance on a finite sample-average
  objective, not as fresh expected noisy fidelity.
- Keep `ballarin_coupled` evaluation-only. Any training configuration that
  requests Ballarin noise must fail validation with a clear error.
- Do not promote a noisy gradient as exact outside the conditions established
  by the small-system validation package.

The intended dependency order is:

```text
WP15 -> WP16 -> WP17 -> WP18
                         | \
                         |  +-> WP19 -> WP20 --+
                         +----> WP21 ----------+-> WP22 -> WP23 -> WP24
```

## Work package 15: Protocol freeze and legacy-evidence audit

### Objective

Freeze the Phase II scientific protocol and record exactly what the historical
experiments did before changing benchmark code.

### Work

- Audit the pre-Phase-I commit, environment lock, experiment scripts, result
  CSVs, figures, manuscript numbers, target generators, noise construction,
  seeds, pruning rule, trajectory modes, and optimizer budgets.
- Map every legacy result used by the manuscript to its producing code and
  classify it as reproduced, discrepant, or unreproducible.
- Record explicitly that:
  - the archived bottom-up procedure is staged brickwall growth, not ADAPT-VQE;
  - the archived five-target top-down CSV used one-shot magnitude pruning, not
    iterative impact pruning;
  - vendor-labelled noise was simulated;
  - the nominally unseeded 500-trajectory evaluation deterministically used
    trajectory seeds `0` through `499` through the library's default seed; and
  - the historical methods used different optimization budgets.
- Freeze method names, training objectives, data-role boundaries, primary
  resource and noise conditions, Phase II target-family parameter
  distributions, target-family weights, resource metrics, failure handling,
  screening rules, promotion rules, confirmatory analysis, multiplicity policy,
  and permissible blinded cluster-level sample-size re-estimation. The fixed
  per-cell trajectory count must never use outcome-dependent optional stopping.
- Freeze the target-generator random-number algorithm, seed derivation and draw
  order, numerical and gauge conventions, degeneracy tolerances, and exact
  family/stratum/qubit allocation policy before WP16 implements the generators.
- Create an initial preregistration seal containing the scientific questions,
  allowed candidate family, development/holdout construction rules, analysis
  template, and mechanical promotion rule.
- Define checksum-sealed screening-manifest and screening-evidence schemas. The
  manifest must enumerate the complete candidate-configuration and
  family/stratum/target/optimization-seed cell universe before screening. The
  evidence ledger must contain exactly one source-addressed outcome for every
  manifest candidate/cell pair. Promotion must be recomputable from those two
  artifacts; caller-supplied summaries or opaque, unchecked manifest checksums
  cannot authorize confirmation.
- Define a separate final-confirmation-seal schema that WP22 must instantiate
  after the pilot and screening packages finish. It contains the promoted
  method, primary comparators, the externally held target-manifest checksum and
  family counts, sample size, fixed test-trajectory budget, hyperparameters,
  resource budget, and analysis checksum.
- Bind every primary contrast to typed configuration references. The corrected
  v2 comparator must use the exact screened baseline configuration, and the
  noiseless control must carry a checksum of the WP16 matching projection that
  proves it differs from its noisy counterpart only in preregistered treatment
  fields.
- Separate the checksum of the natural-language analysis template from the
  checksum of the frozen executable analysis-source manifest.
- Require the final seal to reference a checksum-sealed, pilot-derived
  sample-size design with a frozen calculation method, nuisance inputs,
  contrast set, family/stratum allocation, minimum and maximum target and
  optimization-seed counts, achieved power and precision, and the bounded
  nuisance-only re-estimation history. Denormalized counts in the final seal
  must agree exactly with that design.
- Require the initial preregistration and the instantiated final confirmation
  seal to use canonical JSON with content checksums.
- Define a mechanical promotion rule that may select at most one noisy-training
  configuration after screening. A null screen may promote the corrected v2
  baseline only when the baseline itself passes every resource, failure, data
  separation, and protocol-integrity requirement; a compromised baseline
  aborts promotion.
- Keep legacy reproduction artifacts in a separate namespace that cannot be
  consumed accidentally by confirmatory analysis.
- Prohibit study-team access to confirmatory identifiers, seeds, target vectors,
  or outputs before the final confirmation seal is written. An independent
  external custodian retains the seed-bearing manifest and exposes only its
  checksum until the final seal and execution fingerprint have been verified.
  The in-process authorization object is an accidental-access guard, not a
  cryptographic blinding mechanism.

### Evidence and tests

- JSON schema and round-trip tests for the study protocol.
- Stable preregistration- and confirmation-checksum tests.
- Synthetic-data tests for the complete-manifest, raw-evidence recomputation,
  mechanical promotion, fatal-integrity, comparator-pairing, minimum sample
  size, and failure-inclusion rules.
- Claim-to-artifact audit tests that reject missing source paths, commits,
  configurations, or checksums.
- Golden tests proving that all Phase I schemas and stable run identifiers are
  unchanged.

### Acceptance criteria

The historical evidence is traceable without inventing missing provenance, the
screening decision rule and final-seal schema are locked before screening, and
WP22 cannot authorize confirmation until the complete screening universe, raw
screening evidence, promoted method, matched primary contrasts, executable
analysis source, sample-size design, and confirmatory design have been
separately checksum-locked and cross-verified.

## Work package 16: Versioned target-population and staged-training schemas

### Dependencies

Phase I schemas and the initial protocol seal from WP15.

### Objective

Generate independent Phase II target populations and represent noiseless
growth, noisy fine-tuning, checkpoint validation, pruning, and checkpoint
selection as lossless deterministic identities without relaxing the Phase I
schemas.

### Data structures

Add versioned records resembling:

```python
TargetPopulationConfig
TargetInstanceSpec
TargetPopulationManifest
TrainingStageConfig
TrainingPipelineConfig
TrainingStageResult
TrainingPipelineResult
PipelineEvaluationConfig
PipelineBenchmarkResult
PipelineBenchmarkFailure
```

The Phase II target-population configuration must freeze:

- a versioned family definition and parameter distribution for Gaussian
  amplitude, TFIM ground-state, dense Haar-random, and random-MPS targets;
- Gaussian mean, width, support, and boundary constraints;
- TFIM coupling and field distributions, boundary conditions, and any
  ferromagnetic, critical, or paramagnetic regime strata;
- Haar real/imaginary draw conventions;
- MPS tensor distribution, bond-dimension strata, and canonicalization;
- qubit-count strata, instance count per family and stratum, and instance seed
  domain;
- endianness, global-phase convention, floating-point precision, eigensolver
  policy, and generator version, including a spectrum-only manifest solver, an
  authorized state-vector solver, and their exact agreement tolerances; and
- externally custodied manifest, authorized materialization, and checksum
  policies.

The generator must give every Phase II instance a unique stable identifier
derived from its population configuration, family, stratum, qubit count, and
instance seed. It must support writing a seed-bearing manifest of identifiers
and parameter records without materializing state vectors, committing its
canonical checksum publicly, and withholding the manifest under external
custody until authorization. Gaussian and uniform-TFIM Phase I fixtures must
never be reused as if a new seed made them independent; their Phase II family
definitions must sample the sealed physical parameters. The canonical 18 Phase
I fixtures and the five legacy reproduction fixtures remain in separate
immutable namespaces.

Before any Phase II target manifest or benchmark data are generated, WP16
corrects the initially ambiguous TFIM `eigensolver` seal. Target-generator
v2 uses dense-Hermitian `numpy.linalg.eigvalsh` only to seal manifest spectra
and rejection decisions, reserves dense-Hermitian `numpy.linalg.eigh` for
authorized state-vector materialization, and freezes both relative and absolute
cross-solver agreement tolerances at `1e-13`. The corrected preregistration,
target-policy checksum, and generator version replace the unexecuted v1 target
policy; no v1 Phase II target data may be generated or promoted.

Each stage configuration must resolve and validate:

- stage identifier and kind;
- input topology and output topology;
- parameter-transfer or initialization rule;
- initialization and optimizer seeds;
- optimizer kind, hyperparameters, and iteration budget;
- training-noise identifier and optional frozen strength scale;
- TJM `dt`;
- trajectory count;
- trajectory update (`independent` or `cross`);
- sampling policy (`none`, `resampled`, `crn_fixed`, or `crn_refresh`);
- CRN refresh interval, when applicable;
- checkpoint-validation trajectory count, seed, and cadence;
- checkpoint-selection rule;
- truncation settings; and
- an optional input checkpoint content checksum and compatible pipeline prefix.

Each stage result must record:

- the resolved stage configuration checksum;
- produced checkpoint path and checksum;
- training and checkpoint-validation summaries;
- CRN-map ensemble identities or checksums;
- optimizer trace and diagnostic-sidecar paths and checksums;
- stage wall time and peak memory; and
- normalized computational-work counters.

The complete pipeline must resolve:

- ordered stages;
- target and ansatz family;
- method identity and version;
- disjoint initialization, optimizer, training, checkpoint-validation,
  screening-selection, and confirmatory-test seed domains;
- the final materialization and test policy; and
- development, checkpoint-validation, screening-selection, secondary-benchmark,
  or confirmatory data role.

The Phase II evaluation records must resolve:

- pipeline training identifier;
- materialized-circuit identifier and checksum;
- test-noise identifier;
- evaluation seed, repetition, and trajectory budget;
- confidence-interval or fixed-sample policy;
- sidecar storage policy; and
- result or structured-failure status.

The stable evaluation-row identifier is derived from the training identifier,
materialized circuit, test noise, evaluation policy, evaluation seed, and
repetition. Output path spelling is never part of a scientific identity.

### Compatibility and identity

- Keep `BenchmarkConfig`, `KrotovStatePreparationMethod` version 1, and all
  Phase I result schemas immutable.
- Use distinct pipeline configuration and result schema identifiers.
- Include every training stage and training-noise choice in the stable training
  identity.
- Exclude final test-noise fan-out and output-path spelling from the training
  identity.
- A changed checkpoint-validation configuration, CRN policy, trajectory budget,
  refresh schedule, pruning rule, or checkpoint-selection rule must produce a
  different training identity.
- Checkpoint-validation configuration belongs in the training identity because
  it may select the trained artifact. Validation outcomes, generated output
  paths, and every final-test field do not.
- Reject reuse of a checkpoint whose pipeline prefix, provenance, or checksum
  differs.

### Evidence and tests

- Minimal and complete staged pipelines.
- Deterministic Phase II target generation for every family and stratum.
- Unique instance identifiers across development, screening-selection, and
  confirmatory populations.
- Blinded-manifest generation without state-vector materialization.
- Rejection of any target-vector eigensolver or phase-normalization call during
  blinded-manifest generation and before confirmatory authorization.
- Manifest/vector/checksum agreement after authorized materialization.
- Rejection of Phase I or legacy identifiers in a Phase II holdout manifest.
- JSON and CSV round-trips with stable ordering.
- v1 golden serialization and run-identifier regression tests.
- Cross-stage topology and parameter-count consistency.
- Missing, duplicate, or reordered stage identifiers.
- Invalid parameter transfers and seed collisions.
- Invalid CRN refresh settings.
- Noisy stages without positive trajectories or training seeds.
- Noiseless stages with trajectory settings.
- Checkpoint-validation configuration included in training identity.
- Checkpoint-validation outcomes and final-test fields excluded from training
  identity.
- Required stage checkpoints and checksums.
- Stable evaluation-row identifiers and repeated-evaluation identities.
- Success and failure serialization for Phase II evaluations.
- Explicit rejection of Ballarin training.

### Acceptance criteria

Independent Phase II target instances and every proposed noisy-training method
can be represented without special-case runner state, while all Phase I
configurations remain byte-for-byte stable.

## Work package 17: Benchmark-grade fixed-rate noisy Krotov stage

### Dependencies

The Phase II staged-training and evaluation schemas from WP16.

### Objective

Expose the existing noisy Krotov implementation through a new benchmark adapter
with explicit scientific semantics, seed separation, and exact small-system
validation.

### Work

- Implement a new noisy training-stage adapter rather than changing the
  immutable Phase I `KrotovStatePreparationMethod`.
- Translate each stage into:
  - `KrotovOptions`;
  - `KrotovTJMOptions`;
  - the requested standard gate-local noise provider; and
  - the resolved initial state and parameter vector.
- Initially support the ten fixed-rate standard dephasing/depolarizing
  configurations and the frozen historical reproduction profile.
- Add an immutable scaled-provider factory for noise-strength studies. The
  resolved base identifier and scale belong in the stage identity; never mutate
  the frozen Phase I registry.
- Support:
  - independent-trajectory updates;
  - cross-trajectory updates as a separately named heuristic update rule;
  - resampled trajectories;
  - one fixed CRN ensemble; and
  - periodically refreshed CRN ensembles.
- Extend the reusable Krotov layer with an explicit, serializable fixed-map
  ensemble abstraction supporting:
  - ensemble identity and checksum;
  - stage, ensemble, trajectory, and refresh indices;
  - caller-supplied replay;
  - refresh schedules;
  - schedule-continuous global iteration offsets;
  - non-identity-event diagnostics; and
  - deterministic serialization.
- Make sampled-map identity depend only on the resolved training seed and
  explicit stage/ensemble/trajectory/refresh indices. Optimizer-ordering seeds
  must not alter trajectory maps.
- Derive stage and trajectory seeds from the reserved training domains rather
  than a shared generator or Python `hash()`.
- Record the checksums or deterministic identities of fixed CRN map ensembles.
- Record per-iteration monitoring loss and fidelity, checkpoint-validation
  fidelity, update signal, gradient norm only where established, update norm,
  trajectory count, non-identity events, and cumulative computational work.
- For cross-trajectory mode, record the actual dense-sum update diagnostic
  separately from independent mean trajectory fidelity. Do not label either as
  the optimized scalar objective unless a scalar surrogate and derivative are
  derived and validated.
- Freeze whether each supported training and test profile acts on logical or
  compiled native gates. Use the compiler basis, connectivity, routing, and
  counting policy sealed in WP15.
- Ensure a final test configuration cannot affect parameter initialization,
  training maps, checkpoint-validation maps, checkpoint selection, or final
  parameters.

### Exact and numerical validation

On two- to four-qubit systems with truncation disabled:

- compare frozen-map pathwise gradients with central finite differences of the
  same frozen-map sample-average objective;
- enumerate the exact circuit-TJM discretization, including drift,
  at-most-one-jump sampling, `dt`, normalization, gate placement, and branch
  probabilities;
- compare the independent finite-trajectory estimator with exhaustive
  circuit-TJM evolution and its full expected-objective derivative, including
  probability-derivative terms when present;
- validate the cross-trajectory update expression against its explicit dense
  double sum without assuming that it is the gradient of returned mean
  fidelity;
- test resampled estimator convergence;
- measure the generalization gap between a fixed CRN training ensemble and
  fresh checkpoint-validation ensembles; and
- repeat selected checks with representative MPS truncation to quantify bias.

The frozen-map derivative may be called a sample-average pathwise gradient only
under the validated conditions. The independent estimator may be called an
estimator of the expected discrete-channel gradient only after exhaustive
branch-sum agreement and bias/variance characterization. Cross-trajectory
updates remain heuristic unless a scalar objective and matching derivative are
independently established.

### Tests

- Zero-noise equivalence with Phase I noiseless Krotov.
- Fixed-seed and execution-order reproducibility.
- Training-map identity independent of the optimizer-ordering seed.
- CRN providers sampled exactly once per fixed ensemble.
- Fixed-map serialization, checksum verification, and caller-supplied replay.
- Correct resampling and refresh cadence.
- Immutable scaled-provider construction and identity.
- All ten standard noise providers.
- Training-noise placement, `dt`, strengths, and process counts.
- Logical/native noise-placement and routing-policy consistency.
- Disjoint training, checkpoint-validation, screening-selection, and
  confirmatory-test maps.
- Changed test settings leave the checkpoint checksum unchanged.
- Training failures become structured stage failures.
- Ballarin training is rejected before any output mutation.

### Acceptance criteria

Fixed-rate noisy Krotov training is reproducible, resumable, independently
validated, and cannot be confused with exact angle-dependent Ballarin training.

## Work package 18: Multi-stage artifacts, checkpointing, and resumability

### Dependencies

The Phase II schemas from WP16 and noisy Krotov stage from WP17.

### Objective

Execute long layerwise and pruning pipelines safely without repeating completed
stages or losing partial scientific evidence.

### Implementation status

**Implemented.** The Phase II package now provides bounded, checksum-verified
parameter-checkpoint and trajectory-sidecar codecs, an atomic
`Phase2ArtifactStore`, and a `Phase2PipelineExecutor` that commits only a
verified contiguous stage prefix. Every stage artifact binds its source,
selected, and final parameter states to topology, optimizer trace, map
ensembles, work, resource measurements, provenance, and the stage-prefix
identity; reopening verifies those cross-links before any completed stage is
reused.
Cross-process writer locking and a retained-manifest compare-and-swap reject
stale handles before they can rewrite canonical evidence.

Standalone WP17 calls retain raw-array and MPS target support, but publishing
genuine WP17 evidence is stricter: the sealed objective must bind the
pipeline's authorized materialized Phase II target and computational-zero
initial state before any artifact is written.

Producer-backed external checkpoints are verified before any output mutation,
sealed into a deterministic managed path, and replayed from that immutable
copy. This keeps path spelling outside scientific identity while ensuring that
the validation-selected parameter vector—not an unselected final iterate—is
the exact input to the resumed pipeline.

Explicit `ResumabilityFingerprint` records cover the starting commit, tracked
execution sources, lockfiles, sealed inputs, dependency versions, and complete
configured pipeline prefix while rejecting generated output as an input.
Pipeline and complete stage-prefix mismatches always reject resume. For an
otherwise identical pipeline, runtime-fingerprint drift stops scientific resume
unless a checksum-sealed `NonScientificResumeOverride` records the exact
mismatch pair.

For each pending fan-out attempt, the separate `ParallelPhase2Evaluator`
materializes a selected pipeline checkpoint once, checksum-verifies its
deterministic circuit bytes, reconstructs the runtime circuit through a trusted
decoder, and evaluates pending rows concurrently. A later resume may
rematerialize the circuit if rows remain pending, but successful rows are not
replayed. Materialization and evaluation attempts, structured failures, fixed
maps, optional trajectory fidelities, and derived CSV remain linked to the
complete training artifact. The required checksum-sealed manifest records the
committed baseline used to detect ledger rollback relative to the retained
manifest when the store is reopened; detecting a consistent rollback of the
complete store requires an external monotonic anchor.

### Work

- Execute stages sequentially and emit an immutable artifact after every stage.
- Store:
  - bound parameters;
  - circuit topology and statistics;
  - optimizer trace;
  - training and checkpoint-validation summaries;
  - fixed-map identities or checksums;
  - normalized work counters;
  - software and Git provenance;
  - wall time and peak memory; and
  - checkpoint and sidecar checksums.
- Resume from the latest verified stage checkpoint.
- Preserve completed stages and structured failures after interruption.
- Build the resumability fingerprint from an explicit manifest of tracked
  execution source, lockfiles, sealed configuration inputs, and the starting
  commit. Prefer an output root outside the source tree and explicitly exclude
  the configured output root and its generated JSONL, checkpoints, manifests,
  and sidecars from the fingerprint.
- Reject resume outright when the stored and requested pipeline configuration
  or complete stage prefix differs.
- For an otherwise identical pipeline, reject runtime-fingerprint drift in the
  method implementation, study protocol, dependency versions, lockfiles,
  starting commit, or tracked execution sources unless an explicit
  non-scientific override is recorded.
- Support validation-based checkpoint selection without exposing final test
  trajectories.
- Add a parallel Phase II evaluator and result store for pipeline artifacts
  rather than widening Phase I result types.
- Store canonical Phase II JSONL rows, a derived CSV, a manifest, parameter and
  stage checkpoints, fixed-map artifacts, and optional compressed trajectory
  sidecars.
- Link every final test row and structured failure to the complete
  training-pipeline artifact through stable identifiers and checksums.

### Tests

- Interruption and resume after every stage kind.
- Atomic writes and checksum failures.
- No replay of a successfully completed stage.
- Provenance and protocol mismatches.
- Generated outputs do not invalidate their own resume fingerprint.
- Tracked source, lockfile, or sealed-input changes do invalidate resume.
- Partial-stage cleanup without deleting completed artifacts.
- Stable pipeline and stage identifiers.
- Stable Phase II evaluation-row identifiers.
- Phase II JSONL/CSV/manifest consistency.
- Failure preservation.
- No training, checkpoint-validation, screening-selection, or confirmatory-test
  map reuse.
- Total wall time equals the sum of stage, materialization, and row-specific
  evaluation times under the documented convention.

### Acceptance criteria

A multi-hour noisy layerwise or pruning run may be interrupted and resumed
without changing its scientific identity or silently reusing invalid evidence.

## Work package 19: Historical bottom-up layerwise BMPD-CRN reproduction

### Dependencies

The staged pipeline executor and Phase II artifact store from WP18.

### Objective

Reproduce the highest mean among the three methods in the archived shared
five-target CSV protocol under its exact historical semantics, then provide a
corrected publication-grade version of the same method.

### Implementation status

**Implemented.** WP19 now provides the exact five-stage
`layerwise_bmpd_crn_legacy_v1` compatibility profile and the separately
identified `layerwise_bmpd_crn_v2` profile. The legacy path preserves the
audited RandomState initialization, prefix growth, reused `30 * target_seed`
optimizer stream, `40 * target_seed` final optimizer/fixed-CRN stream, cross
update, historical fixed-rate logical TJM profile, and evaluation seed range
`0` through `499`. The corrected path requires pilot-frozen training and
checkpoint-validation counts explicitly and uses independent updates, standard
noise, disjoint derived streams, a separately fixed validation ensemble,
iteration-zero/every-ten/final validation, and earliest-iteration tie breaking.
Legacy fixed-CRN training also preserves the archived compact Pauli-map replay
without per-gate replay normalization, while independent evaluation retains
the normalized sampled-trajectory behavior; these policies are separately
identified and cannot be selected by the noise profile alone.

The five checked-in q8 target vectors are explicitly labelled WP19
reconstructed references because the audit established that no archived target
vectors, generator draws, eigensolver outputs, or exact historical runtime were
retained. Their sealed collection records the commit-addressed generator,
RandomState draw order, dense eigensolver convention, current reconstruction
runtime, null archived-vector checksums, and missing provenance; regeneration
is compared phase-invariantly at the declared tolerance.

An explicit opt-in pinned job executes all five rows through the WP18 store
under one output-root lock and one checksum-sealed launch snapshot of tracked
implementation, lockfile, and study-input bytes. It revalidates that snapshot
between targets and before publication, then builds a report that binds the
shared manifest/runtime and every row's target-specific WP18 fingerprint.
Ordinary CI covers schemas, identity isolation, target regeneration, transfer
continuity, seed/noise semantics, v2 selection policy, artifact binding, and
report arithmetic without running the expensive q8 optimization. This
implementation does not claim a new numerical reproduction until that opt-in
job has completed; any resulting difference is retained as a discrepancy
rather than replaced by the archived CSV value.

### Historical fixtures and profile

- Add a separate legacy target collection containing the five eight-qubit
  disordered-TFIM ground states with seeds `100`, `200`, `300`, `400`, and
  `500`.
- Freeze the historical generator:
  - `numpy.random.RandomState(seed)`;
  - nearest-neighbor couplings and transverse fields drawn from
    `Uniform(0.8, 1.2)`;
  - exact dense ground-state diagonalization;
  - the historical lack of an explicit global-phase convention; and
  - stored couplings, fields, energy, NumPy/SciPy versions, BLAS/LAPACK build
    provenance, platform, and archived state checksum.
- Because the audit found that archived vectors were not retained, store
  explicitly labelled WP19 reconstructed references with null archived-vector
  checksums. Validate fresh regeneration phase-invariantly within a declared
  tolerance rather than requiring bitwise eigensolver portability, and never
  pretend that the historical path imposed a global-phase convention.
- Freeze a profile named `ibm_inspired_pauli_legacy_v1`, not `IBM`:
  - local `X`, `Y`, and `Z` processes, each with strength `3e-4 / 3`;
  - nearest-neighbor `XX` and `ZZ` processes, each with strength `3e-3 / 2`;
  - legacy gate-support filtering; and
  - `tjm_dt = 1.0`.

### Historical pipeline

Implement `layerwise_bmpd_crn_legacy_v1` with:

1. BMPD depths `1`, `2`, `3`, and `4`;
2. 100 noiseless Krotov iterations per depth;
3. depth-one initialization from a normal distribution with scale `0.05`;
4. exact prefix transfer of existing parameters;
5. newly added parameters initialized from a normal distribution with scale
   `0.001`;
6. a constant noiseless step size of `1.0`;
7. 200 final noisy iterations with step size `0.2`, exponential decay `0.01`,
   three cross trajectories, and fixed CRN maps; and
8. a 500-trajectory legacy evaluation reproducing the archived seed semantics.

Implement a separate `layerwise_bmpd_crn_v2` profile with the following complete
default algorithm:

1. use the same depths, 100-iteration noiseless stages, parameter-transfer
   mapping, initialization scales, and noiseless update schedule as the legacy
   profile;
2. use the standard fixed-rate training-noise condition sealed by WP15 rather
   than inheriting the legacy profile implicitly;
3. run a 200-iteration final fine-tuning stage with the legacy step-size and
   decay schedule;
4. use independent-trajectory updates and one fixed CRN training ensemble,
   with the pilot-derived trajectory count frozen before screening;
5. evaluate checkpoint candidates at iteration zero, every ten iterations, and
   the final iteration on one separately fixed checkpoint-validation ensemble;
6. select the candidate with highest checkpoint-validation mean fidelity,
   breaking exact ties in favor of the earliest candidate;
7. use disjoint derived initialization, optimizer-ordering, training,
   checkpoint-validation, screening-selection, and confirmatory-test seed
   domains; and
8. retain the complete optimizer trace, work ledger, map checksums, and
   per-trajectory sidecars.

Any screened continuation, refresh, rolling-ensemble, cross-update, altered
trajectory count, altered depth schedule, or altered checkpoint rule receives a
different method/configuration identity and is not silently called
`layerwise_bmpd_crn_v2`. The final test set never selects a checkpoint.

### Tests and reproduction evidence

- Phase-invariant legacy target regeneration within a declared tolerance.
- Stage topology and parameter-transfer mapping.
- Deterministic initialization of newly added layers.
- Fidelity continuity before and after a zero-initialized stage transfer.
- Exact historical noise placement and strengths.
- Historical and corrected seed-domain behavior.
- v2 independent fixed-CRN update, checkpoint cadence, deterministic tie rule,
  and complete work-ledger behavior.
- Distinct legacy-v1 and corrected-v2 method/configuration identities.
- Small structural golden tests suitable for ordinary CI.
- An opt-in pinned historical reproduction job that emits the five legacy rows;
  the expensive eight-qubit training and 500-trajectory evaluation are not
  ordinary CI tests.
- A numerical comparison report against the archived reference vector and
  mean, with a documented tolerance justified by dependency or core changes.
- Any discrepancy is preserved and explained; reference values must never be
  copied into generated output.

### Acceptance criteria

The old result is either reproduced from executable artifacts or transparently
classified as discrepant. The corrected method produces independently tested
rows and is never labelled ADAPT-VQE.

## Work package 20: Fair controls and familiar competitor methods

### Dependencies

The versioned noisy-stage and pipeline infrastructure from WP16-WP18. Historical
parameter-transfer behavior may reuse the tested implementation from WP19.

### Objective

Determine whether noisy fine-tuning, layerwise growth, CRN, or the Krotov update
itself creates the observed advantage.

### Implementation status

**Implemented.** WP20 now provides first-class templates for the matched
noiseless, fixed-depth noisy, independent fixed-CRN, independently resampled,
modern cross-CRN, Phase-I-fixture, and unpruned-depth controls. The direct
q6/depth-four and modern layerwise primary controls have target-bound runners;
Phase-I-fixture and unpruned descriptors remain explicitly secondary and are
blocked from sealed promotion roles. The exact WP19 legacy profile remains the
separate historical cross-CRN reproduction control. Exact parameter-shift Adam
and fresh-objective SPSA use distinct fixed and layerwise identities,
authorized-target objectives, complete parameter traces, domain-separated
requests, atomic work-budget stopping, and optimizer-specific,
checksum-validated Phase II stage evidence rather than fabricated Krotov resume
states.

The promotion-capable family-wide operator-growth comparator uses a
target-bound standard fixed-rate noisy evaluator and records the exact projector
objective, pool, growth specification, requests, and trajectory provenance.
Its circuit and resources are mechanically reconstructed and verified from
that evidence rather than accepted as independently persisted claims. Its
analytic entry point is explicitly promotion-ineligible. Target-bound genuine
energy ADAPT-VQE accepts authorized TFIM specifications only and rejects
non-TFIM targets; the generic reference-only `energy_adapt_vqe` returns a
zero-work structural not-applicable result for non-TFIM families. Shared native
compilation/counting records, detailed work ledgers, reachable resource strata,
deterministic Pareto frontiers, method-wide random-stream isolation, and
full-protocol event-level test coupling make unequal work or resource gaps
explicit. WP22 owns the common runner/artifact wrapper for the standalone
operator-growth result before screening; WP20 does not claim a winning method
before the WP22-WP24 locked runs.

### Required controls

Implement first-class pipelines for:

- Phase I noiseless training followed by noisy testing;
- fixed-depth BMPD with direct fixed-rate noisy Krotov training;
- bottom-up layerwise BMPD without noisy fine-tuning;
- bottom-up layerwise BMPD with independent fixed-CRN fine-tuning;
- bottom-up layerwise BMPD with independent resampling;
- bottom-up layerwise BMPD with modern cross-CRN fine-tuning, plus the exact
  WP19 historical cross-CRN profile as a separately labelled reproduction
  comparator; and
- an unpruned deep circuit evaluated at matched and unmatched native budgets.

### Familiar competitor methods

Add adapters for:

- parameter-shift Adam on the same fixed and layerwise ansatz;
- SPSA on the same fresh noisy objective;
- `run_standard_fixed_rate_noisy_operator_growth`, using a target-bound noisy
  projector/fidelity cost and a frozen operator pool for target families
  without a Hamiltonian objective, with `adapt_style_state_preparation`
  retained only as an analytic, promotion-ineligible reference; and
- genuine energy-based ADAPT-VQE on the TFIM subset only, using the frozen TFIM
  Hamiltonian, operator pool, gradient-selection rule, reoptimization rule, and
  stopping criterion.

Each operator-growth pool must state its one- and two-qubit generators, site
ordering, duplicate policy, symmetry restrictions, cost function, and native
compilation. Fidelity/projector-based growth must never use the ADAPT-VQE label.
`run_standard_fixed_rate_noisy_operator_growth` is the family-wide
operator-growth comparator; `adapt_style_state_preparation` is an analytic
reference only. Genuine energy-based ADAPT-VQE is a TFIM-subset exploratory
analysis and is ineligible for family-wide promotion unless a separate
TFIM-only estimand and promotion rule were sealed initially. The target-bound
entry point rejects non-TFIM targets, while generic reference-only
`energy_adapt_vqe` records structurally inapplicable non-TFIM cells as not
applicable rather than optimizer failures.

### Resource matching

For every method report:

- native one- and two-qubit gate counts and depth;
- trainable parameter count;
- forward and backward circuit evaluations;
- trajectory-gate applications;
- total sampled trajectories;
- objective and gradient calls;
- cross-trajectory pairings, whose work scales as `R**2`;
- wall time; and
- the `tracemalloc` Python-allocation peak.

WP20 records the algorithmic ledger for standalone operator-growth results but
does not invent runtime measurements for them. WP22 must attach wall time and
the `tracemalloc` Python-allocation peak through the common runner/artifact
wrapper before those methods enter screening; these fields are not claims about
process-wide peak memory.

Run both:

1. fixed-resource comparisons; and
2. a frozen Pareto sweep over native two-qubit gates and normalized compute.

For discrete growth or pruning paths, use sealed reachable strata at or below
the resource limit, report the residual resource gap, and do not describe
unequal compiled counts as exactly matched.

All methods must use the device graph, routing and SWAP rules, compiler
optimization policy, native basis, and logical/native noise-placement convention
sealed by WP15. The broad optimizer and operator-growth collection is
exploratory screening material. Locked confirmation later includes only the
promoted method and two or three primary comparators.

### Tests

- Method adapter contract and deterministic initialization.
- Projector-cost operator growth and genuine TFIM ADAPT-VQE selection on
  analytically solvable small systems.
- Promotion eligibility and not-applicable handling for TFIM-only methods.
- Adam and SPSA update references.
- Equal-budget stopping.
- Same native compilation and counting rules across methods.
- No parameter-count-as-gate-count substitution.
- Pairing by target and optimization-seed block without sharing initialization,
  optimizer, training-trajectory, or checkpoint-selection randomness.
- Event-level test coupling used only where complete final-test protocols and
  stable native gate identifiers align; otherwise independent Monte Carlo
  streams are recorded explicitly.
- Complete work-ledger accounting.

### Acceptance criteria

Every named competitor implements the algorithm its name denotes, and no method
can win a fixed-budget comparison by receiving unreported additional work.

## Work package 21: Reproducible top-down pruning competitors

### Dependencies

The staged artifact engine from WP18 and native resource rules sealed in WP15.

### Objective

Turn the manuscript's intended top-down methods into first-class pipelines and
separate the effects of pruning score, iteration, and noisy fine-tuning.

### Implementation status

**Implemented.** WP21 provides first-class `topdown_random`,
`topdown_magnitude`, `topdown_impact_one_shot`, and
`topdown_impact_iterative` pipelines. Impact scoring uses a generalized
gate-occurrence parameter shift for parameters shared by several gates, and
primary native-budget claims remove compiler-derived entangler groups.
Deterministic score ties and retained-parameter remapping preserve gate
semantics. Every scoring round seals its objective, pre-pruning circuit, and
sampled-map provenance; iterative impact alternates pruning and relaxation at
artifact-verifiable resume boundaries and requires at least two pruning rounds.

Compiled rounds expose only observed reachable native-resource strata and
return typed infeasibility when no stage meets a requested cap. Optional
fixed-rate noisy fine-tuning is a separate pipeline stage, and final testing
uses fresh randomness isolated from scoring, relaxation, fine-tuning, and
checkpoint selection. These methods are distinct from the archived one-shot
magnitude-pruned CSV and make no numerical result claim. WP22 owns the common
training runner and artifact-level pilot, screening, and final-seal
orchestration.

### Methods

Implement:

- `topdown_random`;
- `topdown_magnitude`;
- `topdown_impact_one_shot`; and
- `topdown_impact_iterative`.

Trajectory sampling and noisy fine-tuning are pipeline-stage configurations,
not part of the pruning-method identity.

The iterative impact method must:

1. grow or train the frozen deep starting circuit under the configured
   protocol;
2. freeze the scoring objective `F`, its data role, and whether it is noiseless
   fidelity, a fixed-map sample-average fidelity, or another explicitly derived
   scalar;
3. compute the documented score

   ```math
   I_i = \left|\theta_i \frac{\partial F}{\partial \theta_i}\right|;
   ```

4. define the pruning unit as a parameter, gate, shared-parameter group, or
   compiled entangler group;
5. use compiled entangler groups for primary native two-qubit gate-count budget
   claims, including basis changes, routing, and SWAPs;
6. remove a frozen number or fraction of the least-impactful units;
7. apply deterministic tie-breaking;
8. rebuild parameter indices without changing retained gate semantics;
9. relax the circuit between pruning rounds;
10. recompile every retained round with the frozen native compiler and record
    the resulting routed circuit and resource counts;
11. for a native two-qubit budget, select deterministically the attempted stage
    with the largest reachable count not exceeding the sealed budget, or emit a
    structured infeasible-budget result when none exists; and
12. optionally apply the same fixed-rate noisy fine-tuning stage as the
    bottom-up method.

Because routing and optimization may make compiled counts non-monotonic,
confirmation may instead seal only budget strata shown reachable by the pilot.
No method may claim an arbitrary exact compiled count that its pruning path did
not produce.

Do not import production behavior from the legacy experiment scripts.

### Tests

- Parameter-shift impact against analytic and central finite-difference
  gradients.
- Generalized derivatives for parameters shared by multiple gates.
- Shared-parameter scoring and removal.
- Entangler-group pruning and complete compiled-cost accounting.
- Deterministic ties.
- Exact retained-parameter schedules and reachable native-gate constraints.
- Non-monotonic post-routing counts and infeasible-budget handling.
- State equivalence after parameter remapping.
- Iterative relaxation and resumability.
- Random-pruning repetitions and seeds.
- Clear distinction between magnitude, one-shot impact, and iterative impact.
- Fresh noisy testing after pruning and fine-tuning.

### Acceptance criteria

The method called iterative impact pruning actually performs iterative
gradient-impact pruning, and its result rows cannot be confused with the
historical magnitude-pruned CSV.

## Work package 22: Training runner, pilot, strategy screening, and final seal

### Dependencies

The corrected historical method from WP19, exploratory controls from WP20, and
top-down competitors from WP21, all using the WP18 artifact store.

### Objective

Provide safe orchestration, calibrate a feasible evidence budget, screen a
small predeclared strategy family, and create the final confirmation seal
without materializing confirmatory targets.

### Prospective replanning status

**Replanned before execution.** The original monolithic WP22 was decomposed into
WP22A through WP22F. WP22G below is a second prospective custody hardening
addendum introduced before any held confirmatory target is revealed. No WP22
numerical pilot, screening, promotion, or final-seal evidence has been
generated. The initial WP15 preregistration remains immutable; these reviewed
plan revisions fill operational details which the preregistration did not
specify. The final seal must bind the governing documents and the commits that
introduced them. If they ever conflict, the preregistration governs and
execution aborts rather than weakening its protections.

Each subpackage receives a separate implementation commit and review. WP22 is
not operationally complete until WP22G passes. Planning-only records, callback
seams, or synthetic fixtures do not satisfy an execution acceptance criterion.

### Frozen population and method scope

- `training-smoke` runs every repository implementation kind on one applicable
  q6 development target with explicitly separate tiny-budget artifacts. It
  includes the nine family-wide method identities and the TFIM-only exploratory
  method, but produces no paper evidence.
- The primary q6 pilot uses all 48 development targets, the exact ordered five
  optimization seeds derived from the preregistration checksum, and exactly
  three configurations: `layerwise_bmpd_crn_v2`,
  `layerwise_bmpd_noiseless`, and `fixed_depth_bmpd_crn`. The fixed-depth method
  is a representative candidate for variance calibration and is not thereby
  promoted. The Cartesian population therefore contains exactly 720 jobs.
- The secondary q12 pilot uses the same three treatment projections, five
  optimization seeds, and the preregistered 24 targets, six per family. Its
  target manifest retains `screening_selection/secondary_q12` custody while
  jobs and outputs use `secondary_benchmark`. q12 observations are excluded
  mechanically from pilot inference, sample-size calculation, screening,
  promotion, and the final seal. The secondary population contains exactly 360
  jobs.
- The q6 paper screen contains exactly the nine preregistered family-wide
  methods, 48 screening-selection targets, and three checksum-derived
  optimization seeds: 1,296 expected method cells. The TFIM-only exploratory
  method is reported separately and cannot enter family-wide promotion.
- A checked-in execution profile freezes exactly one publication configuration
  per screened method before any screening-selection result is accessed. The
  first paper screen is nonadaptive and has no interaction configuration.
  Continuation, rolling, mixture, and multistart experiments use development
  targets only and cannot enter promotion. Any later interaction study requires
  a new prospectively sealed candidate universe; screening outcomes may never
  construct a later candidate.

### Frozen schedule semantics

- Training schedules execute at the optimizer-update boundary with zero-based,
  inclusive update indices. Noise strength, trajectory count, mixture
  allocation, and ensemble membership are resolved before the corresponding
  update.
- The common production budget is 200 terminal optimization updates, eight
  noisy-training trajectories, 256 fixed checkpoint-validation trajectories,
  and validation at update zero, every ten updates, and the terminal update.
  Best-validation selection uses the earliest-update tie rule. Primary methods
  do not stop early; a prospective atomic work cap may stop only before the next
  complete update and yields structured failure evidence.
- Krotov, Adam, and SPSA optimizer state is preserved across continuation,
  curriculum, and CRN-refresh boundaries. A schedule may not be approximated by
  restarting independently trained stages.
- Direct matched-noise training uses strength `1.0` for all 200 updates.
  Continuation uses strength zero at update zero, increases linearly to `1.0`
  at update 49, and remains at `1.0` through update 199. The trajectory
  curriculum uses counts two for updates 0 through 49, four for updates 50
  through 99, and eight for updates 100 through 199.
- Fixed CRN retains member identities; resampling creates a new ensemble every
  update; periodic refresh changes the full ensemble when the update index is a
  positive multiple of 20; and rolling refresh uses the same interval,
  deterministically retains exactly one half of the previous ensemble, and
  fills the remaining positions with new members.
- A trajectory-count increase preserves eligible existing members and appends
  newly derived members. A decrease uses the same checksum-ranked deterministic
  selection rule as rolling retention. Every transition records its predecessor
  checksum.
- Frozen-mixture counts use largest-remainder allocation from the declared
  component weights. Fractional ties follow component declaration order. The
  frozen mixture is exactly one half `depolarizing_1s_all` and one half
  `dephasing_1s_all`, both at the scheduled strength. Component means are
  evaluated separately and combined with exact weights one half.
  Component-local seed domains, labels, and memberships are persisted and
  cannot change after resume.
- Every multistart is a complete independent run with isolated initialization,
  optimizer, and training-trajectory streams. The exploratory multistart count
  is exactly three. Selection uses only checkpoint-validation fidelity, with
  earliest checkpoint and then lowest start index resolving exact ties. All
  starts count toward work, and a multistart is ineligible for promotion unless
  its complete work fits a separately sealed cap.
- Early stopping and checkpoint selection consume only the fixed inner
  checkpoint-validation stream. Fresh pilot, screening, and confirmatory values
  are unavailable to the schedule engine.
- A resumed execution must reconstruct byte-identical update programs,
  trajectory memberships, optimizer state, checkpoint selection, and result
  checksums. An unsupported method/schedule composition is rejected instead of
  being approximated.

### Frozen pilot and fresh-evaluation semantics

- A typed `FreshEvaluationPolicy` seals the data role, complete standard-noise
  condition, fixed trajectory count, seed derivation, provider and map versions,
  truncation policy, worker count, sidecar format, and failure treatment.
- Each primary-q6 pilot job stores one ordered 1,024-trajectory fresh-evaluation
  batch. Trajectory convergence is reported on the nested prefixes 64, 128,
  256, 512, and 1,024; execution always generates all 1,024 trajectories, so
  the prefixes cannot create optional stopping.
- The q6 update-noise diagnostic is evaluated only for successful jobs at the
  inner-validation-selected checkpoint. It stores exactly 32 independent
  single-trajectory pathwise update vectors under the primary noise condition
  and the `pilot_evaluation` seed domain. Evidence includes every complete
  vector, parameter ordering, checkpoint checksum, estimator and provider
  checksums, and all 32 seeds. It derives unbiased coordinate variances and
  their arithmetic mean and maximum. The endpoint is named
  `post_training_primary_noise_pathwise_update_variance`; it is descriptive,
  cannot affect promotion or confirmation, and is not described as the exact
  expected-channel gradient. The noiseless comparator receives the same noisy
  diagnostic at its final parameters rather than a fabricated zero.
- Each q12 pilot job stores 256 fresh trajectories plus runtime, work, memory,
  resource, and failure evidence. It has no gradient diagnostic and no
  inferential convergence endpoint.
- The pilot-derived `SampleSizeDesign` selects the confirmatory target count,
  allowed optimization-seed count, and fixed test-trajectory count using the
  immutable WP15 rule. `paper-screen` copies that fixed trajectory count exactly
  for its outer evaluation. A missing design, changed count, or adaptive
  trajectory stopping aborts before output creation.
- The outer trajectory count uses a bounded-data rule rather than a
  normality-dependent variance interval. It applies the one-sided
  standard-deviation inequality in Theorem 10 of
  [Maurer and Pontil (2009)](https://arxiv.org/abs/0907.3740) with a union bound
  over the planned jobs. For q6 pilot job `j`, let `s_j^2` be the unbiased
  variance of its 1,024 fidelities and let `M = 720`. Compute

  ```math
  V_j^+ = \min\left\{\frac14,
  \left(\sqrt{s_j^2} +
  \sqrt{\frac{2\log(M/0.05)}{1023}}\right)^2\right\},
  \qquad V^+ = \max_j V_j^+.
  ```

  Then set

  ```math
  N_{\mathrm{outer}} = 2^{\left\lceil\log_2\left(
  \max\left\{256,
  \left\lceil V^+/0.005^2\right\rceil\right\}\right)\right\rceil}.
  ```

  Counts above 16,384 are infeasible and abort before screening. The same fixed
  count is used for every q6 screening cell and becomes the design's fixed
  confirmatory trajectory count; there is no trajectory optional stopping.

- Wall time uses a monotonic clock and memory uses the existing WP18 Python
  allocation measurement. Neither is described as process-wide peak memory.

### Frozen operator-growth policy

The target-bound noisy operator-growth configuration runs only at q6. Its
site-major `RX/RY/RZ` and edge-major `RXX/RYY/RZZ` projector pool is sampled
without replacement and without a symmetry restriction. It minimizes
pure-state projector infidelity, selects the largest absolute generalized
parameter-shift score at an appended zero parameter, and resolves ties by the
lowest pool index.

The binding seals the complete pool, growth and stopping rules, Adam settings,
preregistered primary training-noise condition, trajectory seed derivation,
resource policy, and fresh-evaluation policy. It uses tolerance `1e-10`, at
most 16 selected operators, native cap 12 per chain edge, 100 full-parameter
Adam reoptimization updates after each append, learning rate `0.08`, betas
`0.9` and `0.999`, and epsilon `1e-8`. Selection and reoptimization use one
fixed eight-trajectory primary-noise CRN ensemble. Every completed prefix is
evaluated on the separate fixed 256-trajectory checkpoint-validation ensemble;
the greatest validation fidelity wins, with the earliest growth step breaking
ties. Outer screening uses the common `N_outer` and a fresh
`screening_selection` stream. Unsupported continuation, rolling, mixture, or
width combinations fail explicitly. Smoke and screening artifacts have
role-specific wrappers; only a complete q6 screening artifact may produce
promotion evidence.

### WP22A: Execution-protocol closure and typed bindings

**Status: implemented.**

The implementation provides the checked-in operational amendment, the frozen
training-policy universe, deterministic membership schemas, a sealed
execution-seed policy suite, exact pilot and operator-growth policies, and
fail-closed scoped binding/profile records. It does not provide the WP22B
executable catalogs or any WP22 numerical evidence.

Add a checksum-sealed operational-protocol amendment referencing the immutable
initial preregistration, and strict canonical artifacts for:

- `FreshEvaluationPolicy` and `PilotDiagnosticPolicy`;
- a complete `OperatorGrowthExecutionSpec`;
- a `ScopedImplementationBinding` keyed by preset, publication-candidate
  checksum, and target scope; and
- a `TrainingExecutionProfile` containing the complete, unique binding and
  policy universe for one preset.

A scoped binding keeps the publication candidate identity separate from its
actual width- and preset-specific implementation. It includes manifest and
execution roles, the materialized implementation artifact, exact strategy
schedule, controlled stage, resource and evaluation policies, budget profile,
and a checksum-covered q6/q12 treatment projection. No target vector or secret
role entropy is materialized by WP22A.

#### WP22A tests and acceptance criteria

- Strict JSON round trips and nested checksum-substitution rejection.
- Exactly one binding for every `(preset, candidate, target scope)` key.
- Distinct q6, q12, and smoke implementation checksums without changing the
  publication candidate identity.
- Rejection of a q12 primary role, a q12 screening/promotion binding, an
  incomplete operator specification, or an evaluation policy without a fixed
  count.
- Exact enforcement of the 200-update training budget, training and validation
  counts, diagnostic design, nonadaptive 1,296-cell screen, bounded outer-count
  rule, and complete operator-growth policy.
- No changes to Phase I or WP15 through WP21 identities.

WP22A is complete only when every scientific choice needed by later profiles is
represented by a typed checksum-bearing artifact rather than an identifier or
free-form callback argument.

### WP22B: Width-complete repository implementation catalog

**Status: implemented.**

The implementation preserves the literal q6 template identities while adding
non-retuned q12 projections, width-aware direct-control execution, and a
fail-closed repository catalog whose entries resolve concrete runner adapters.
The catalog contains the exact smoke, pilot, and screening universes; dormant
confirmation reuses an eligible q6 screening entry and creates no new
configuration. The tenth, TFIM-only Energy-ADAPT smoke route is explicitly an
analytic API preflight: the preregistered noisy Energy-ADAPT treatment remains
unsupported and cannot enter pilot, screening, promotion, or evidence. WP22B
produces no pilot, screening, or confirmation result.

Generalize the q6-only layerwise, noiseless, fixed-depth, Adam, and SPSA
builders and the fixed-depth runner where required. q6 behavior and checksums
must remain unchanged. q12 variants use the same layer count, gate ordering,
per-edge construction, optimizer hyperparameters, and treatment semantics
without retuning; topology, parameter counts, compilation, and resource
evidence are derived at q12.

Provide repository-owned binding catalogs for:

- tiny-budget smoke coverage of every implementation and stage-runner family;
- the exact three q6 and q12 pilot configurations;
- all nine q6 family-wide screening configurations; and
- dormant confirmation resolution, which introduces no new configuration.

#### WP22B tests and acceptance criteria

- Every catalog entry contains an executable implementation artifact, not only
  its checksum.
- Real one-update/one-trajectory artifact tests cover every runner family and
  both widths where applicable.
- All nine family-wide methods resolve at q6; exactly the three frozen methods
  resolve at q12.
- Missing, duplicate, cross-preset, primary-q6-for-q12, and unsupported
  operator-growth bindings fail before mutation.
- q12 artifacts cannot enter primary pilot summaries, screening manifests,
  promotion decisions, or final-seal inputs.

### WP22C: Schedule-aware optimizer engine and exact restart

**Status: implemented.**

The implementation closes each scientific program over its complete
`ExecutableScopedBinding`, compiles all schedule memberships, and keeps
checkpoint-validation fields out of the pre-update training interface. Typed
Krotov, parameter-shift Adam, SPSA, and operator-growth payloads retain every
restart field and are checked against the binding-owned dimension,
initialization, hyperparameters, parameter-shift scales, cross-trajectory
mode, and growth specification. Post-update validation is sealed to a
recoverable parameter artifact, complete multistart selection, and exact
request/result/work receipt chain. Resume revalidates that chain and the
prospective job-wide compute cap before invoking a numerical callback. Target
and raw numerical-evidence custody remain assigned to WP22D and WP22E.

Compile each binding, schedule, and job seed set into a checksum-sealed
`ScheduledExecutionProgram` with one `ScheduledUpdatePolicy` per optimizer
update. Persist `ScheduledOptimizerState`, exact trajectory and mixture
membership, validation-tracker state, and complete multistart evidence. Apply
the schedule only to the binding's declared controlled stage; structural growth
and pruning stages keep their own sealed semantics.

#### WP22C tests and acceptance criteria

- Golden update traces cover continuation, curricula, fixed CRN, resampling,
  periodic refresh, rolling retention, frozen mixtures, phase boundaries,
  checkpoint selection, and multistart selection.
- Interrupted/resumed and uninterrupted executions are byte-identical.
- Constant schedules reproduce the existing WP17 through WP21 numerical
  behavior at fixed seeds.
- Optimizer state cannot reset at a schedule boundary and final-test evidence is
  inaccessible to the engine.
- Unsupported schedules abort rather than silently changing their meaning.

### WP22D: Execution context, target custody, source fingerprint, and CLI

**Status: implemented.**

The implementation adds a redacted, non-serializable external-entropy keyring
and complete execution context, carries every profile, binding,
implementation, schedule, evaluation, target, program, and source fingerprint
into jobs and plans, and validates the whole plan before output mutation. The
strict opt-in CLI supports all five presets, JSON/CLI assertions, dry-run,
resume, overwrite, fail-fast, the source-locked `factory(context)` seam, and
the repository-owned default registry. Exact 10-job smoke, 1,080-job paired
q6/q12 pilot, and 1,296-job q6 screen fan-outs are enforced; q12 custody stays
separate, attempt histories are append-only, and confirmation treats the first
terminal attempt as authoritative. WP22E remains responsible for typed raw
numerical artifacts and the concrete production executors.

Add a non-serializable `ExternalEntropyKeyring` and a typed
`TrainingExecutionContext` containing the plan, execution profile,
preregistration, candidates, schedules, scoped bindings, target configurations
and manifests, authorized materializations, screening manifest and cells,
required sample-size design, execution-source manifest, and resumability
fingerprints. Secret entropy is excluded from JSON, checksums, representations,
logs, errors, and outcomes.

Extend each job and plan with the execution-profile, scoped-binding,
implementation, evaluation-policy, target-configuration, and source-fingerprint
checksums. Complete preflight verifies all of them, target authorization,
schedule compilation, outer trajectory counts, and source bytes before the
output root exists.

Keep the separate opt-in entry point:

```bash
uv run python -m benchmarks.state_preparation.training_runner
```

It supports strict JSON plus CLI assertions, dry-run, resume, overwrite,
fail-fast, the five Phase II presets, and explicit historical reproduction. The
ordinary path uses the repository-owned executor registry. A `factory(context)`
extension seam is permitted for tests and external development but cannot
replace sealed scientific artifacts in a paper preset.

#### WP22D tests and acceptance criteria

- CLI/JSON precedence, preset cardinalities, canonical ordering, and dry-run
  no-mutation behavior.
- Missing or wrong entropy, target config, binding, sample-size design, source
  file, schedule, or evaluation policy fails with zero output mutation.
- No entropy appears in plans, tracebacks, logs, or persisted artifacts.
- q12 target custody remains `screening_selection/secondary_q12` while its jobs
  and outputs remain `secondary_benchmark`.
- Attempt histories are append-only; the first terminal confirmatory attempt is
  authoritative and cannot be overwritten.

### WP22E: Repository-owned production executors and evidence

**Status: implemented.**

The repository-owned registry now resolves every exact context-owned pipeline
or operator-growth job, executes the real scheduled numerical callbacks, and
publishes append-only source-addressed results. Terminal manifests close over
schedule snapshots, fixed-map ensembles, fresh raw trajectory sidecars, q6
pathwise pilot diagnostics, runtime/resources, normalized work, and structured
failures; typed reopening verifies every byte, member, alias, and nested map
reference. Operator-growth screening executes two independently validated
100-update prefixes, retains the full active structure, and performs its fresh
evaluation on the validation-selected prefix. Each trajectory's fidelity and
fixed replay map arise from one numerical pass, and the reference, manifest,
and evidence independently close over the direct execution-source manifest.
The default runner path covers all ten smoke families, while a request-bound
synthetic confirmation executor tests the dormant path without opening a held
manifest. WP22F remains responsible for pilot inference, full screening replay,
promotion, and final sealing.

Implement `create_default_training_executor_registry(context)`. The Phase II
executor resolves the scoped implementation, authorized target, artifact store,
runner, schedule engine, circuit materializer, and fresh evaluator. It persists
fixed maps, raw trajectory sidecars, runtime, resources, normalized work,
failures, and source-addressed results. The operator-growth executor constructs
the exact request from its complete spec and produces the same role-correct raw
evidence. The historical delegate remains unchanged.

Implement the dormant confirmation executor through the same production paths.
It accepts only a complete `ConfirmExecutionRequest` and synthetic fixtures in
WP22; it may not open the real held manifest.

#### WP22E tests and acceptance criteria

- A real one-target CLI smoke run covers every implementation and runner family
  without a custom factory or source edit.
- Pilot and screening executors return typed source-addressed artifacts rather
  than caller-supplied summary checksums.
- Resume reopens the exact artifact and never fabricates a second authoritative
  attempt.
- Operator-growth smoke and screening use the same sealed numerical callbacks
  with role-specific custody.
- Ballarin training and unsupported method/schedule pairs fail before
  optimization.
- Synthetic confirmation dispatch proves the production path is complete
  without materializing a confirmatory target.

### WP22F: Pilot, screening, promotion, source lock, and final seal

**Status: implemented; depends on WP22A through WP22E.**

Integrate production execution with the pilot, sample-size, screening,
promotion, result-custody, source-lock, dormant confirmation, and frozen
primary-analysis layers:

1. execute or replay the exact q6/q12 pilot profile;
2. derive nuisance summaries and sample size from q6 raw evidence only;
3. archive q12 solely as secondary scaling evidence;
4. instantiate the q6 screen using the design's fixed trajectory count;
5. execute or replay the complete 1,296-cell screening universe;
6. reconstruct promotion only from authoritative first-attempt raw evidence;
7. freeze execution and analysis source manifests in a clean commit; and
8. create the final seal without opening or materializing confirmatory targets.

Promotion follows WP15 mechanically, promotes at most one configuration, and
falls back to `layerwise_bmpd_crn_v2`. Comparator identities are de-duplicated;
a promoted baseline cannot create a self-contrast.

#### WP22F tests and acceptance criteria

- End-to-end fixture coverage from execution profile through pilot evidence,
  sample-size design, screen plan, complete source records, promotion, and seal.
- Exact raw-custody rejection of missing, duplicate, self-authenticated,
  wrong-role, wrong-count, wrong-source, or non-first-attempt evidence.
- q12 mutations cannot affect inference, sample size, promotion, or seal bytes.
- Source locking covers all tracked state-preparation and YAQS execution code,
  environment locks, operational profiles, preregistration, and primary
  analysis.
- Full WP15 through WP22 tests, the complete project test suite, documentation,
  and lint pass.

The implementation reopens the authoritative production attempts, derives q6
pilot inference and the q6-calibrated compute cap, replays all 1,296 screening
cells, and mechanically rebuilds promotion before sealing. A separate aggregate
manifest binds every final configuration to its exact screened schedule,
implementation, scoped binding, and executable binding while preserving the
preregistered final-confirmation schema v1. Confirmation requests use schema v2
to carry both that aggregate root and the exact configuration-specific
execution identity.

WP22 is complete only when historical reproduction, smoke, pilot, and screening
need no source editing; screening promotes at most one fully specified candidate
without materializing confirmation; and all execution and primary-analysis code
is committed and fingerprinted before the immutable, pilot-feasible final seal
is written.

### WP22G: Prospective pre-reveal runner and output-custody hardening

**Status: implemented prospectively; depends on WP22A through WP22F and must be
reviewed before WP23 unblinding.**

Harden only the operational boundary of the already frozen `paper-confirm`
route. This subpackage adds no training, evaluation, target-generation, or
analysis behavior and changes no sealed method, schedule, count, seed, noise,
resource, or statistical choice.

- Apply a strict allowlist before preregistration, external entropy, or held
  target paths are opened. Only sealed custody-artifact locations, exact
  checksum assertions that are subsequently verified, `resume`, `dry_run`, the
  expensive-execution opt-in, repository identity, and output custody controls
  are admitted.
- Reject method and pipeline assertions, stage depths or budgets, training
  noise or strength, update and sampling policies, training or validation
  counts, CRN and checkpoint controls, resource overrides, pilot seeds,
  overwrite/fail-fast controls, resumability fingerprints, and caller-selected
  executor or execution-profile routes.
- Require an explicit CLI `--output` for `paper-confirm`; the output root must
  be disjoint from and outside `repository_root`, contain no symlink component,
  and contain no development or screening role tree. No repository-relative
  confirmation output default is admissible. Its writable parent must share the
  output filesystem and holds only non-authoritative crash staging and the
  whole-run lock; temporary files never enter the scientific output inventory.
- Preserve the earlier programmatic context/executor-injection rejection order
  and the explicitly opted-in dry-run semantics.
- Require a checksum-sealed prior-target exposure inventory that closes the
  legacy, Phase-I, pilot, and screening identifier/instance-seed universe and
  binds the exact pilot/screen plans, production-calibration custody, and
  governed execution source. Reject every confirmation ID or instance-seed
  collision before reading held entropy or materializing a target.
- Read held entropy only from an owner-private, single-link regular file through
  a pinned nofollow descriptor with stable pre/open/post identity and exact
  length checks.
- Bind every real executor to the one canonical output root, plan job directory,
  and canonical plan position. A whole-plan marker and all-unattempted locked
  study snapshot must exist before direct production dispatch can resolve or
  materialize a held target.
- Close the exact output tree with content-addressed append-only study
  snapshots. Each snapshot includes all 576 plan rows, the contiguous terminal
  prefix, authenticated first-attempt production references, fixed trajectory
  counts, 288 explicit independent-stream pairability records, typed
  resource-limit proof/status, source/target/exposure roots, and exact
  non-snapshot byte receipts.
- Preserve truncated or zero-byte known-path members as opaque immutable partial
  custody and recover an interrupted production attempt into its same
  first-attempt structured failure. Never overwrite scientific bytes or create
  attempt two. Authenticate every existing outer/production result before skip.
- Emit the checksum-sealed snapshot-head reference for independent/WORM custody;
  require its external custody path on the first invocation, publish snapshot
  zero there before dispatch, advance it after every terminal cell, and require
  the retained authenticated chain member on every resume to prevent rollback
  or alternate append-only branches. `fail_fast` remains forbidden.

#### WP22G tests and acceptance criteria

- Sentinels prove every forbidden knob and unsafe output layout fails before
  preregistration, entropy-key, or held-target access.
- Config-only output, repository-contained or repository-containing output,
  symlinked paths, mixed role trees, and non-directory role paths fail closed.
- Existing programmatic-injection ordering, opt-in enforcement, and dry-run
  non-dispatch behavior remain unchanged.
- Collision fixtures prove exposed IDs/seeds fail before entropy or target
  materialization; secure-file tests cover symlink, hardlink, mode, length, and
  descriptor/path identity races.
- Exact-store tests cover snapshot-zero-before-results, canonical directory
  binding, foreign/special/link rejection, fixed-count custody, contiguous
  terminal prefixes, resource-stop suffix rejection, crash recovery, immutable
  resume, external-head rollback rejection, and no second attempt.
- The primary WP23 execution design is byte-for-byte unchanged.

The previously described tiered secondary sweep over remaining standard-noise
conditions, noiseless testing, and Ballarin evaluation is not implemented by
WP22G or WP23. It is deferred to WP24 as separately versioned exploratory work,
must use independent artifact identities, and cannot alter the sealed primary
analysis or claims. Ballarin remains evaluation-only.

### Boundary with WP23 and WP24

WP22F may create the immutable final seal and test confirmation custody with
synthetic fixtures or stop the real, source-locked route before numerical
training. WP22G hardens the pre-reveal CLI and output-custody boundary without
changing that sealed scientific route. Neither may expose the held confirmatory
population. WP23 alone reveals the externally custodied confirmatory manifest
and runs the already frozen executor; it adds no training, evaluation, or
analysis behavior.

WP22 may freeze and synthetically test primary-analysis source, but it does not
analyze real confirmatory outcomes, generate paper figures, formulate claims, or
build the archival publication bundle. Those remain WP24.

## Work package 23: Locked confirmatory execution

### Dependencies

The Phase II runner and final confirmation seal from WP22.

### Objective

Execute the sealed primary experiment without target leakage, post-unblinding
tuning, invalid trajectory-level replication, or uncontrolled scope growth.
This package executes frozen code; it does not add or alter training,
confirmation, target-generation, or primary-analysis behavior.

### Confirmatory preset

Execute the already implemented `paper-confirm` path without changing its
source. It must refuse to run if:

- the preregistration, final-seal, promoted-method, or comparator checksum
  differs;
- the sealed tracked execution-source, lockfile, or configuration fingerprint
  differs;
- a run attempts to change a hyperparameter, target, seed, resource limit,
  trajectory budget, primary noise condition, or analysis rule;
- development or screening outputs share its output directory;
- any confirmatory target identifier or seed occurred in legacy, development,
  checkpoint-validation, or screening-selection data; or
- Ballarin is requested as training noise.

### Confirmatory population and methods

- Use a newly generated, family-stratified, checksum-sealed target collection
  as the primary confirmatory population; its untouched instance identifiers
  and seeds remain under external custody until authorization.
- Reveal the WP16 manifest and materialize its target vectors only after the
  final seal and execution fingerprint have been verified.
- Treat the already exposed canonical 18-target matrix as a secondary benchmark,
  not as held-out confirmation.
- Confirm only the promoted method, `layerwise_bmpd_crn_v2`, the matched
  noiseless-training control, and at most one additional comparator sealed by
  WP22, after de-duplicating identical method/configuration checksums.
- Use the cluster-aware optimization-seed count and common fixed test-trajectory
  count selected by the pilot.
- Use the fixed per-cell test-trajectory count selected by the pilot and sealed
  before confirmation. The historical reproduction retains its separate
  500-trajectory evaluation.
- Pair comparisons by target and optimization-seed block. Event-level random
  coupling is allowed only when stable native gate identifiers align; otherwise
  record independent Monte Carlo streams.
- Use the sealed primary fixed-rate noise condition for the powered comparison.
- Preserve every failure row and partial artifact and apply the sealed
  intention-to-treat and failure-rate rules.
- Default to sequential execution until pilot measurements justify safe,
  deterministic parallelism.

### Tests

- `paper-confirm` cardinality and deterministic ordering.
- Dry-run output and no mutation.
- Protocol, method, comparator, target, seed, budget, and analysis checksum
  enforcement.
- Rejection of exposed target identifiers and seeds.
- Fixed trajectory-count accounting.
- Target/optimization-block pairing and explicit event-pairability metadata.
- Resource-limit stopping and structured incomplete-study status.
- Resume without changing the sealed design.
- No Ballarin training; WP23 adds no secondary Ballarin evaluation route.

### Acceptance criteria

The complete confirmatory manifest contains all planned results or structured
failures, uses untouched target instances, and can be analyzed without any
post-unblinding method or hyperparameter choice.

## Work package 24: Statistical analysis, documentation, and archival release

### Dependencies

The complete locked study manifest and artifacts from WP23.

### Objective

Convert canonical Phase II artifacts into defensible claims and a
one-command-reproducible publication package.

### Statistical analysis

- Execute the frozen primary-analysis code and checksum from the final seal
  without post-unblinding edits. Version any additional analysis separately and
  label it exploratory.
- Run any tiered secondary sweep over remaining standard-noise conditions,
  noiseless testing, or Ballarin evaluation only as a separately versioned
  exploratory WP24 protocol with distinct artifacts and no effect on the sealed
  primary analysis. Ballarin remains evaluation-only.
- Analyze target and optimization-seed blocks rather than treating trajectories
  as independent algorithm repetitions.
- Separate:
  - within-artifact Monte Carlo uncertainty;
  - across-optimization-seed variability;
  - across-target variability; and
  - failure probability.
- Use paired confidence intervals and effect sizes for the frozen primary
  comparisons.
- Apply the sealed multiplicity policy to secondary method comparisons.
- Report unconditional results with failures included under the sealed
  failure-rate and intention-to-treat rules.
- Label the following as exploratory sensitivity analyses unless they were
  powered and included in the final seal:
  - training and test trajectory budgets;
  - fixed versus refreshed CRN;
  - independent versus cross updates;
  - noise strength;
  - MPS truncation;
  - target family;
  - qubit count; and
  - native gate and compute budget.
- A null or negative result completes the package; manuscript claims must follow
  the locked thresholds.

### Figures and artifacts

- Generate every table and figure from canonical JSONL records, manifests,
  checkpoints, and sidecars.
- Ban hard-coded result values and absolute local paths.
- Emit a machine-readable claim-to-run-ID index.
- Preserve failed runs and outliers in both data and plots.
- Provide one command that regenerates all tables and figures from the archived
  result bundle.

### Documentation and release

- Update the benchmark definition with Phase II method, objective, seed, noise,
  budget, validation, and statistical conventions.
- Document the historical reproduction separately from corrected evidence.
- Update `CHANGELOG.md` and `UPGRADING.md` for new public optimization APIs.
- Reconcile manuscript method names and claims with executable pipeline
  identifiers.
- Call simulated vendor-derived profiles hardware-inspired unless QPU evidence
  is added.
- Publish:
  - the clean tagged source revision;
  - locked dependency metadata;
  - protocol and study-design checksums;
  - canonical raw and derived data;
  - checkpoints and manifests;
  - figure-generation code; and
  - an archival DOI or equivalent immutable deposit.

### Acceptance criteria

An independent contributor can regenerate every manuscript number from an
immutable run identifier without using the legacy experiment scripts.

## Phase II definition of done

Phase II is complete when:

- every Phase I golden schema, run identifier, and preset remains unchanged;
- versioned Phase II target-population manifests generate disjoint,
  checksum-verifiable development, screening-selection, and confirmatory
  instances without rebranding deterministic Phase I fixtures as new samples;
- the historical five-target layerwise result is reproduced or transparently
  classified as discrepant;
- `layerwise_bmpd_crn_legacy_v1` and `layerwise_bmpd_crn_v2` are implemented
  and never mislabeled ADAPT-VQE;
- fixed-rate noisy Krotov supports independent, cross, resampled, fixed-CRN,
  and refreshed-CRN policies with explicit identities;
- training, checkpoint-validation, screening-selection, and confirmatory-test
  seed domains and map ensembles are disjoint;
- exact small-system validation characterizes the sample-average,
  expected-channel, normalization, discretization, and MPS-truncation
  approximations before any noisy update is described as a gradient;
- Ballarin training is rejected and Ballarin testing remains supported;
- all multi-stage checkpoints and resumes are checksum- and
  provenance-verified, and generated outputs do not invalidate their own
  source fingerprint;
- target-bound noisy projector-cost operator growth, TFIM-only genuine
  ADAPT-VQE, Adam, SPSA, and fixed-depth controls follow the shared resource
  ledger and native counting rules, while `adapt_style_state_preparation`
  remains an analytic reference only;
- iterative impact-pruning rows come from the actual iterative impact method;
- native-budget claims use reachable compiled strata and preserve structured
  infeasible-budget outcomes;
- checkpoint selection and method screening use their separate development and
  screening-selection domains and promote at most one sealed configuration;
- confirmation evaluates only the promoted method and sealed primary
  comparators on the untouched holdout under the frozen target, seed, fixed
  trajectory count, budget, and analysis protocol;
- the confirmation executor and primary-analysis code are implemented, tested,
  and fingerprinted before the final seal is written;
- failures remain in unconditional summaries;
- Monte Carlo and algorithmic uncertainty are reported separately;
- every figure and manuscript number is generated from canonical artifacts;
- full tests, minimum-version tests, lint, target checks, documentation, and
  end-to-end Phase I and Phase II smoke runs pass; and
- the tagged source and immutable evidence archive reproduce the publication
  artifacts.
