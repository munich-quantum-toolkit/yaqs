# State-Preparation Benchmark Implementation Plan

This document describes the implementation work required to turn the benchmark
definition in
[`state_preparation_benchmarks.md`](state_preparation_benchmarks.md) into a
reproducible, end-to-end experiment suite.

The target-state fixtures, brickwall ansatz, noiseless Krotov state-preparation
optimizer, and generic trajectory-based noise machinery already exist. The
remaining work consists of benchmark orchestration, exact gate-local noise
models, Quantinuum-native compilation, final-circuit accounting, independent
evaluation, and standardized reporting.

## Design decisions

### Benchmark workflow

For each combination of method, target, qubit count, ansatz configuration, and
initialization seed:

1. Optimize the shared brickwall ansatz noiselessly.
2. Evaluate the final circuit without noise.
3. Evaluate the same trained parameters with fresh trajectories under every
   selected noise configuration.
4. Emit one result row per noise configuration.

Noisy optimization may be added by a method adapter later. It is not part of the
canonical first implementation because the benchmark matrix currently specifies
noiseless optimization followed by noisy evaluation.

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
- `OptimizerConfig`
- `EvaluationConfig`
- `NoiseConfig`
- `CircuitStatistics`
- `BenchmarkResult`
- `BenchmarkFailure`

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
- optimization and evaluation wall times;
- YAQS, Python, NumPy, and SciPy versions;
- Git commit and dirty-tree flag;
- result status;
- parameter-checkpoint path; and
- checkpoint checksum.

Do not overload the benchmark's target-generation `seed` field with any runtime
seed.

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

### Acceptance criteria

Every planned run can be represented losslessly and deterministically.

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
evaluation.

## Definition of done

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
