# State-Preparation Benchmark Definition

This document defines the benchmark suite for comparing quantum state-preparation
methods against the same targets, ansatz family, and noise assumptions.

## Scope

- Qubit counts: `6` and `12`.
- Ansatz: brickwall circuit.
- Target states: 9 target-state families per qubit count.
- Noise families:
  1. Ballarin/Quantinuum angle-dependent local depolarizing noise from
     arXiv:2511.15674.
  2. Dephasing noise.
  3. Depolarizing noise.
- Primary metric: target-state fidelity after optimization.
- Secondary metrics: circuit depth, number of one-qubit gates, number of
  two-qubit gates, number of trainable parameters, optimization wall time, and
  number of noise trajectories or shots used for evaluation.

All methods should report both the noiseless fidelity and the noisy fidelity
under each selected noise configuration.

## Shared Circuit Ansatz

All benchmark methods should use the same brickwall ansatz family.

Required run metadata:

- `num_qubits`: `6` or `12`.
- `num_layers`: method-specific, but must be reported.
- `num_parameters`: total number of trainable parameters.
- `num_1q_gates`: total number of single-qubit gates in the final circuit.
- `num_2q_gates`: total number of two-qubit gates in the final circuit.
- `initialization`: random seed or warm-start rule.
- `optimizer`: optimizer name and hyperparameters.

## Noise Models

### Common Rates

Use the following rates for dephasing and depolarizing noise:

| Gate context                                   |            Strength |
| ---------------------------------------------- | ------------------: |
| Noise applied after single-qubit gates         | `0.064%` (`6.4e-4`) |
| Jump operators applied after multi-qubit gates |  `0.51%` (`5.1e-3`) |

Each listed strength is the strength of **each jump operator**, not a total
channel strength divided among the operators. For example, single-site
depolarizing noise after a one-qubit gate has three processes (`X`, `Y`, and
`Z`), each with strength `6.4e-4`; the nine-operator two-site depolarizing
channel has nine processes, each with strength `5.1e-3`. YAQS passes these
values to circuit TJM as Lindblad-process rates with the configured `tjm_dt`.
An implementation using another parameterization must document its conversion
from this per-operator convention.

### N1: Ballarin/Quantinuum Angle-Dependent Local Depolarizing Noise

Source: Marco Ballarin, Juan Jose Garcia-Ripoll, David Hayes, and Michael
Lubasch, "Efficient quantum state preparation of multivariate functions using
tensor networks", arXiv:2511.15674, Eqs. (8)-(9).

Definition:

- Express each two-qubit entangler in the Quantinuum-native form
  `RZZ(theta) = exp(-i theta (Z ⊗ Z) / 2)`. The shared ansatz's `RXX` and
  `RYY` gates are compiled using noiseless single-qubit basis changes and one
  `RZZ` gate with the same angle. Any other two-qubit gate must be compiled to
  an explicit `RZZ` sequence, and the resulting angles and gate count must be
  reported.
- Angles use the YAQS/Qiskit radian convention. Canonicalize each native angle
  to `[-pi, pi)` and set `a = abs(theta)`. Using the magnitude is the benchmark
  convention that extends the paper's nonnegative calibration fit to signed
  parameterized rotations.
- Remove native two-qubit rotations with `a <= 1e-4` before counting gates or
  applying noise, matching the pruning rule used by Ballarin et al.
- Treat every single-qubit gate, including compilation basis changes, as
  noiseless.
- Immediately after each retained ideal `RZZ(theta)` on qubits `i` and `j`,
  apply the product channel `K_i(a) ⊗ K_j(a)`, where

```{math}
\epsilon(a) = 2.1 \times 10^{-4} + 1.43 \times 10^{-3} a,
```

```{math}
r(a) = \frac{1}{3}\left(1-\sqrt{1-\frac{5}{4}\epsilon(a)}\right),
```

```{math}
K_q(a)(\rho) =
  [1-3r(a)]\rho
  + r(a)\left(X_q\rho X_q + Y_q\rho Y_q + Z_q\rho Z_q\right).
```

An exact Kraus representation of each local channel is
`{sqrt(1-3r) I, sqrt(r) X, sqrt(r) Y, sqrt(r) Z}`. The two-qubit
channel therefore has the 16 tensor-product Kraus operators formed from the
two local sets. In a trajectory implementation, independently sample
`I`, `X`, `Y`, or `Z` on each participating qubit with probabilities
`1-3r`, `r`, `r`, and `r`, respectively, and apply the sampled Pauli pair
after the ideal gate. This is a product of two local depolarizing channels,
not a correlated two-qubit Pauli channel.

The implementation samples this categorical channel directly: one draw is
made for qubit `i` and one independent draw for qubit `j`. It does not
approximate the 16 branch probabilities with TJM rates or combine the sampled
operators into a weighted matrix. This makes each trajectory an exact sample
from the specified channel. A finite ensemble average is still a Monte Carlo
estimate of the noisy fidelity and therefore has sampling uncertainty.

Post-gate placement and the use of `abs(theta)` for signed rotations are
explicit conventions of this benchmark; the paper does not specify those two
cases. Do not additionally apply the `6.4e-4` or `5.1e-3` common rates in this
configuration.

Benchmark identifier:

```text
noise=ballarin_coupled
```

The existing identifier is retained for compatibility; `coupled` means that
the two local channels are attached to the same native gate, not that their
Pauli samples are correlated.

### N2: Dephasing Noise

Dephasing should be benchmarked with explicit choices along two axes:

- Site support:
  - `single_site`: local `Z`-type dephasing on each affected qubit.
  - `two_site`: correlated `ZZ`-type dephasing on the two qubits of a
    multi-qubit gate.
- Gate placement:
  - `single_qubit_gates`: apply noise only after single-qubit gates.
  - `multi_qubit_gates`: apply noise only after multi-qubit gates.
  - `all_gates`: apply noise after single-qubit and multi-qubit gates.

Recommended configurations:

| Identifier           | Site support                  | Gate placement         | Strengths                                                                        |
| -------------------- | ----------------------------- | ---------------------- | -------------------------------------------------------------------------------- |
| `dephasing_1s_1q`    | `single_site`                 | `single_qubit_gates`   | `6.4e-4` after 1q gates                                                          |
| `dephasing_1s_2q`    | `single_site`                 | `multi_qubit_gates`    | `5.1e-3` after multi-qubit gates                                                 |
| `dephasing_1s_all`   | `single_site`                 | `all_gates`            | `6.4e-4` after 1q gates, `5.1e-3` after multi-qubit gates                        |
| `dephasing_2s_2q`    | `two_site`                    | multi-qubit gates only | `5.1e-3` after multi-qubit gates                                                 |
| `dephasing_1s2s_all` | `single_site` plus `two_site` | `all_gates`            | `6.4e-4` after 1q gates, `5.1e-3` for all jump operators after multi-qubit gates |

If the full benchmark matrix is too large, the minimum dephasing set is
`dephasing_1s_all` and `dephasing_2s_2q`.

### N3: Depolarizing Noise

Depolarizing should use the same support and placement axes as dephasing.

- Single-site depolarizing: local `X`, `Y`, and `Z` Pauli jump operators on each
  affected qubit.
- Two-site depolarizing/correlated Pauli noise: correlated two-qubit Pauli jump
  operators on the two qubits of a multi-qubit gate. The frozen operator set is
  exactly

```text
XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ
```

Each of these nine operators has the full applicable per-operator strength.
Identity factors are excluded: operators such as `X tensor I`, `I tensor Z`,
and `I tensor I` belong to neither the two-site set nor its rate budget. The
`1s2s` configurations add the six single-site processes (three Paulis on each
gate qubit) to these nine two-site processes.

Recommended configurations:

| Identifier              | Site support                  | Gate placement         | Strengths                                                                        |
| ----------------------- | ----------------------------- | ---------------------- | -------------------------------------------------------------------------------- |
| `depolarizing_1s_1q`    | `single_site`                 | `single_qubit_gates`   | `6.4e-4` after 1q gates                                                          |
| `depolarizing_1s_2q`    | `single_site`                 | `multi_qubit_gates`    | `5.1e-3` after multi-qubit gates                                                 |
| `depolarizing_1s_all`   | `single_site`                 | `all_gates`            | `6.4e-4` after 1q gates, `5.1e-3` after multi-qubit gates                        |
| `depolarizing_2s_2q`    | `two_site`                    | multi-qubit gates only | `5.1e-3` after multi-qubit gates                                                 |
| `depolarizing_1s2s_all` | `single_site` plus `two_site` | `all_gates`            | `6.4e-4` after 1q gates, `5.1e-3` for all jump operators after multi-qubit gates |

If the full benchmark matrix is too large, the minimum depolarizing set is
`depolarizing_1s_all` and `depolarizing_2s_2q`.

### Exact Channels and Finite-Trajectory Estimates

The channel definitions above are exact. Direct Ballarin sampling selects an
exact product-channel branch, while standard dephasing and depolarizing models
select circuit-TJM no-jump or jump trajectories from their specified process
sets. Reported noisy fidelities are arithmetic means over the configured
number of fresh test trajectories; they are not exact density-matrix
fidelities.

For at least two noisy test trajectories, report the sample standard deviation
(`ddof=1`), its standard error `s / sqrt(N)`, and the configured confidence
interval. The bundled runner uses a clipped normal 95% interval:

```{math}
\left[
  \max(0, \bar{F} - z_{0.975}s/\sqrt{N}),
  \min(1, \bar{F} + z_{0.975}s/\sqrt{N})
\right].
```

With one trajectory, the mean is valid but sampling uncertainty is undefined.
Increasing `--test-trajectories` reduces Monte Carlo uncertainty but does not
change the exact channel being sampled. Compare methods using the same test
budget and resolved seeds, and interpret overlapping intervals as uncertainty
in the finite-trajectory estimate rather than equivalence of the methods.

## Target States

Each target state is generated for both `n = 6` and `n = 12` qubits.
The final state vectors are stored in
`benchmarks/state_preparation_target_states.json` and can be regenerated with
`benchmarks/generate_state_preparation_targets.py`.

The JSON file encodes each complex amplitude as `[real, imaginary]`. Amplitude
index `k` uses little-endian computational-basis ordering, i.e., qubit `i` is
the `i`-th bit of `k`. The global phase is fixed so the largest-magnitude
amplitude is positive real.

The JSON records the NumPy and SciPy versions used to generate it as provenance.
The generator's `--check` mode ignores the exact version strings, compares all
benchmark-defining metadata exactly, and uses tight numerical tolerances only
for state amplitudes, norms, and TFIM ground-state energies.

### T1: Gaussian Amplitude State

Identifier:

```text
target=gaussian_mu0p5_sigma0p1
```

Definition:

- Grid: the endpoint-excluded quantics grid on `[0, 1)` from arXiv:2602.12042,
  Section 3.2.1. In the JSON little-endian basis convention, amplitude index
  `k` has qubit `i` equal to bit `i` of `k`, and qubit `i` has quantics weight
  `2^{-(i+1)}`:

```{math}
x_k = \sum_{i=0}^{n-1} \operatorname{bit}_i(k) 2^{-(i+1)}.
```

- Mean: `mu = 0.5`.
- Standard deviation: `sigma = 0.1`.
- The target encodes the classical Gaussian probability density `f(x)` as
  amplitudes `psi(x) = sqrt(f(x))`. The unnormalized amplitudes are therefore

```{math}
\psi(x) \propto \exp\left(-\frac{(x - 0.5)^2}{4(0.1)^2}\right).
```

- Normalize the amplitude vector to unit 2-norm.

### T2-T4: Transverse-Field Ising Model Ground States

Use ground states of the transverse-field Ising model (TFIM)

```{math}
H = -J \sum_i Z_i Z_{i+1} - h \sum_i X_i.
```

Use the standard uniform open-chain 1D TFIM with `J = 1.0` and uniform
transverse field `h`. In the little-endian state-vector convention, site `i` is
qubit `i` and bit `i` of the computational-basis index. The benchmark includes
one state in each regime:

| Identifier      | Regime        | Condition     | Eigensolver base seed |
| --------------- | ------------- | ------------- | --------------------- |
| `tfim_ferro`    | ferromagnetic | `h / J = 0.5` | `1729`                |
| `tfim_critical` | critical      | `h / J = 1.0` | `2718`                |
| `tfim_para`     | paramagnetic  | `h / J = 1.5` | `3141`                |

For an `n`-qubit target, the deterministic eigensolver initial-vector seed is
`base_seed + 10000 * n`. The base seed is not a physical disorder seed. The
generated JSON records the resulting initial-vector seed, uniform `J_i`,
uniform `h_i`, and ground-state energy for each qubit count.

### T5-T7: Dense Haar-Random States

Generate three complete random dense states for each qubit count.

| Identifier      | Description                           | Seed   |
| --------------- | ------------------------------------- | ------ |
| `haar_random_1` | Dense normalized complex random state | `4001` |
| `haar_random_2` | Dense normalized complex random state | `4002` |
| `haar_random_3` | Dense normalized complex random state | `4003` |

Recommended generation rule:

1. Draw real and imaginary components independently from a standard normal
   distribution.
2. Normalize the resulting complex vector to unit 2-norm.
3. Store the seed and generator version in the run metadata.

### T8-T9: Random MPS States

Generate two random MPS states for each qubit count.

| Identifier         | Bond dimension | Seed   |
| ------------------ | -------------: | ------ |
| `random_mps_bond2` |            `2` | `5002` |
| `random_mps_bond3` |            `3` | `5003` |

Recommended generation rule:

1. Draw real random MPS tensors with independent standard normal entries,
   matching Quimb's `qtn.MPS_rand_state(L=n, bond_dim=..., normalize=True)`
   default tensor distribution.
2. Normalize the resulting state.
3. Use the same generation rule for both `n = 6` and `n = 12`.
4. Report the open-boundary bond dimensions. Internal bonds use the requested
   bond dimension throughout, matching Quimb's convention.

## Benchmark Matrix

The base target matrix contains:

```text
2 qubit counts x 9 target states = 18 target instances
```

For each target instance, run:

1. Noiseless optimization and evaluation.
2. Ballarin-style coupled-noise evaluation.
3. Selected dephasing configurations.
4. Selected depolarizing configurations.

Each reported result must include the exact noise identifier, target identifier,
qubit count, circuit size, optimizer settings, training budget, and test
evaluation budget.

### Train/Test Separation

The bundled Phase I Krotov benchmark trains each target, ansatz,
initialization, and optimizer configuration once without noise. It then
evaluates the identical checkpoint under every selected test-noise
configuration. Test noise and test trajectory settings are excluded from the
stable training identity, so adding a test cell must not retrain or alter the
parameters.

`train_fidelity` describes optimization on the training objective.
`logical_test_noiseless_fidelity` is a fresh logical-circuit evaluation of the
final checkpoint. `test_noiseless_fidelity` and `test_noisy_fidelity` use the
actual evaluated representation: the logical circuit for noiseless and
standard-noise rows, and the final materialized native circuit for Ballarin
rows. No test trajectory may be reused for optimization.

### Phase II Staged Training and Resumability

Phase II supports ordered `optimize`, `grow`, and `prune` stages, including
fixed-rate noisy Krotov optimization with resampled, fixed-CRN, or refreshed-CRN
training maps. Checkpoint-validation settings and outcomes are part of the
training protocol and may select an earlier stage iterate; final-test settings
remain outside the training identity and are never passed to a stage callback.

Construct a `Phase2ArtifactStore` from a fully resolved
`TrainingPipelineConfig` and an explicit `ResumabilityFingerprint`, then use
`Phase2PipelineExecutor` to execute only its unfinished suffix. A successful
stage is committed as an immutable contiguous-prefix artifact containing the
selected and final parameters, circuit binding and statistics, optimizer
trace, training and validation summaries, exact fixed-map references,
normalized work, wall time, peak memory, and checksummed provenance. On
reopening with `resume=True`, the store verifies every ledger row and referenced
artifact before exposing the latest selected checkpoint.
Store mutations use a cross-process writer lock and retained-manifest
compare-and-swap. A handle whose baseline was advanced by another writer must
be reopened, preventing stale whole-ledger rewrites from discarding evidence.

When the first stage consumes an external checkpoint, the store preflights the
producer-backed reference before mutating output and seals the exact bytes in
its own managed checkpoint area. Later resumes use that store-local copy even
if the source path has moved or disappeared, and the executor supplies the
producer's validation-selected parameters rather than its last iterate.

The resumability fingerprint must enumerate the tracked execution sources,
dependency lockfiles, sealed study inputs, dependency versions, starting Git
commit, and complete configured stage prefix. Generated output is explicitly
excluded, so writing checkpoints or result rows cannot invalidate the run;
changing scientific source or sealed input does invalidate it. A pipeline or
complete stage-prefix mismatch always rejects resume. With an otherwise
identical pipeline, runtime-fingerprint drift may be continued only as
explicitly labelled non-scientific work through a checksum-sealed
`NonScientificResumeOverride` that identifies both fingerprints.

After the complete training artifact exists,
`ParallelPhase2Evaluator(store, deserialize_circuit)` materializes its selected
final checkpoint once per pending fan-out attempt and evaluates pending
final-test rows concurrently. The deterministic circuit bytes are checksum
verified and reconstructed through the trusted decoder; a later resume may
rematerialize them when rows remain pending. Canonical rows are published in
requested configuration order, already successful rows are not replayed on
resume, and training, checkpoint-validation, screening-selection, and
confirmatory map roles are checked for isolation. Materialization work is
counted once per attempt; row-specific evaluation time is counted once per row,
and total wall time is the sum of all stage attempts, materialization attempts,
and evaluation attempts recorded by the store.

### Phase II Layerwise Noisy Fine-Tuning

WP19 defines two deliberately distinct bottom-up BMPD profiles. The historical
`layerwise_bmpd_crn_legacy_v1` profile is an isolated q8 reproduction: it grows
depths `1 -> 2 -> 3 -> 4`, performs 100 noiseless Krotov updates at every depth,
copies each trained prefix, initializes only the appended tail, and performs a
200-update noisy fine-tune with three fixed cross-trajectory CRN paths. Its
logical TJM simulation profile is `ibm_inspired_pauli_legacy_v1`; the name is
hardware-inspired provenance and does not indicate IBM hardware execution.
The profile is fixed brickwall growth and must not be labelled ADAPT-VQE.

The legacy initialization seeds are `20 * target_seed` at depth one and
`20 * target_seed + depth` for appended depths. All noiseless stages reuse
`30 * target_seed`; final optimization and fixed training maps use
`40 * target_seed`. The historical fixed-map seed formula is
`1_000_003 * (40 * target_seed) + trajectory_index`. The 500-trajectory
evaluation uses effective seeds `0` through `499`. These compatibility rules
are reserved for the five legacy targets and cannot be selected by a Phase II
screening or confirmation pipeline.

The archived fixed-CRN training path sampled normalized trajectories but
stored compact Pauli maps whose replay did not normalize after each gate; WP19
preserves that behavior only for the legacy fine-tuning stage. The historical
`use_crn=False` evaluation instead scored the normalized sampled trajectories,
so its seed-compatible map replay keeps normalization enabled. Seed derivation
and compact-replay policy are separate controls to prevent this legacy behavior
from leaking into corrected methods that use the same noise condition.

The target collection for seeds `100`, `200`, `300`, `400`, and `500` contains
WP19 reconstructed references, not archived state vectors. The original source
did not retain target vectors, coupling/field draws, eigensolver outputs, or a
complete runtime fingerprint. The sealed collection therefore stores null
archived checksums, the commit-addressed generator semantics, current
NumPy/SciPy/BLAS/LAPACK/platform provenance, and explicit missing-provenance
notes. Fresh regeneration is compared after global-phase alignment with the
declared `1e-10` absolute and relative tolerances.

The publication profile `layerwise_bmpd_crn_v2` retains the depth and update
budgets but uses the standard `depolarizing_1s_all` condition, independent
trajectory updates, hash-derived disjoint seed domains, and separate fixed CRN
ensembles for training and checkpoint validation. Validation candidates are
iteration zero, every ten updates, and the final update; the highest validation
mean wins, with exact ties resolved to the earliest iteration. Training and
validation trajectory counts have no defaults and must be supplied from frozen
pilot evidence before screening. Final-test trajectories never select a
checkpoint.

The opt-in historical job is intentionally outside ordinary CI because it runs
five q8 pipelines and 500 evaluation trajectories per target. Its canonical
comparison report is computed only from emitted evaluation rows or retained
training/orchestration failures and links each value to the archived CSV audit;
missing or discrepant rows remain visible and archived values are never copied
into computed fields.

One exclusive output-root lock covers preparation through final report
publication. A launch manifest seals the exact tracked implementation,
lockfiles, and study inputs; the runner rechecks it before every target and
again before publication. Each outcome carries its target-specific WP18
runtime fingerprint, while the report checksum-binds the shared manifest and
thread-pinned runtime, preventing a five-row result from mixing scientific
source snapshots.

Run the pinned reproduction explicitly from the repository root:

```console
uv run python -m benchmarks.state_preparation.phase2.run_historical_reproduction \
  --output-root output/wp19_historical_reproduction \
  --execute-expensive
```

Use `--resume` to verify and continue that same artifact root after an
interruption. Use `--overwrite` only to replace the runner's managed artifacts;
the two modes are mutually exclusive. Exit status `0` means reproduced within
tolerance, `1` means a complete scientific discrepancy, and `2` means at least
one target, setup step, or job-level check failed.

### Phase II Fair Controls and Familiar Competitors

WP20 isolates the effects of noisy fine-tuning, layerwise growth, CRN policy,
and optimizer choice. The control templates cover noiseless final training,
direct fixed-depth noisy Krotov, independent fixed CRN, per-update resampling,
modern cross CRN distinct from the exact WP19 legacy cross-CRN reproduction,
Phase-I-style noiseless training with fresh noisy testing, and matched or
explicitly unmatched unpruned depths. Required iteration and trajectory budgets
are identity-bearing; compiled circuits that miss a cap retain their residual
gap or resource excess and are never described as exact matches.
`FixedDepthBMPDStageRunner` executes the registered direct q6/depth-four
baseline, while `LayerwiseBMPDStageRunner` executes the matched modern
layerwise controls. The Phase-I-fixture and unpruned-depth descriptors are
secondary controls and are excluded from sealed promotion roles.

Exact parameter-shift Adam and SPSA use the same fixed or layerwise BMPD
ansatz. Parameter-shift pairs share their declared objective stream and use the
Pauli two-evaluation rule; SPSA uses a fresh paired objective and Rademacher
direction at every update. Both methods record every objective and gradient
call, circuit evaluation, trajectory-gate application, validation trajectory,
and stopping decision. Their Phase II checkpoints carry optimizer-specific
execution evidence and are not represented as resumable Krotov states.
Publishable runs use `BMPDCompetitorStageRunner` and its authorized-target
`FixedRateNoisyCompetitorObjective`; arbitrary callbacks cannot cross the
artifact boundary. The SPSA iteration-zero monitor shares the first update's
CRN window, and every later update receives a distinct resampled window.

`run_standard_fixed_rate_noisy_operator_growth` is the promotion-capable
family-wide operator-growth comparator and minimizes pure-state projector
infidelity with a frozen ordered pool and target-bound fixed-rate TJM
evaluator. The analytic `adapt_style_state_preparation` entry point is only a
non-promotion reference and is not ADAPT-VQE. The separate
`target_bound_energy_adapt_vqe` method derives the frozen TFIM Hamiltonian from
the authorized TFIM target specification and rejects non-TFIM targets. The
generic reference-only `energy_adapt_vqe` returns structural not-applicable
outcomes for non-TFIM cells; those outcomes are not optimizer failures. The TFIM
analysis is not eligible for family-wide promotion. Operator-growth results are
strict standalone WP20 evidence; WP22 supplies their pipeline runner and
artifact-store orchestration before any screening run.

All comparisons use the same logical-to-native compiler and dependency-depth
counting. The detailed work ledger includes forward and backward evaluations,
all trajectory roles, and cross-trajectory pairings. The WP18 execution
boundary attaches wall time and the `tracemalloc` Python-allocation peak to
staged runs. Standalone operator-growth results receive those runtime metrics
only from the WP22 wrapper; neither boundary reports process-wide peak memory.
Fixed-resource selection uses only reachable strata at or below both caps, and
the Pareto analysis reports native two-qubit gates against normalized compute.
Methods are paired by target and optimization-seed block without sharing
initialization, optimizer, training-trajectory, or checkpoint-selection
randomness. Final-test streams are coupled event by event only when their
complete evaluation protocols and stable native-event signatures align.

### Phase II Top-Down Pruning Competitors

WP21 provides first-class `topdown_random`, `topdown_magnitude`,
`topdown_impact_one_shot`, and `topdown_impact_iterative` pipelines. Magnitude
and one-shot impact score the frozen starting circuit once, while iterative
impact requires at least two pruning rounds and alternates them with explicit
relaxation stages. Impact uses the generalized gate-occurrence parameter-shift
derivative: a parameter shared by multiple gates receives the sum of the
occurrence-level derivatives before the score `|theta_i dF/dtheta_i|` is
formed.

Primary native two-qubit resource claims remove compiler-derived entangler
groups, including their routed native consequences, instead of treating a
logical parameter as a native gate. Score ties, group order, and retained
parameter remapping are deterministic, and retained gates preserve their
original semantics. Each scoring round seals the exact objective, input-circuit
binding, and sampled-map provenance; iterative pruning and relaxation are
separate artifact stages, providing verification and resume boundaries.

Every compiled round contributes only an observed reachable native-resource
stratum. If no attempted circuit lies at or below a requested cap, the pipeline
returns a typed infeasible result; it never reports an unobserved exact match.
The same fixed-rate noisy fine-tuning used by the bottom-up method may be added
after pruning, but it is a pipeline-stage choice rather than a pruning identity.
Final tests use fresh streams isolated from pruning scores, relaxation,
fine-tuning, and checkpoint selection.

These competitors are distinct from the archived one-shot magnitude-pruned
CSV, which remains historical evidence and does not define current production
behavior. WP22 owns the common training runner, pilot orchestration, screening,
and final seal. WP21 therefore records method and resource evidence, not a
numerical paper result or promotion decision.

### Phase II operational-protocol closure

WP22A freezes the prospective study choices needed by later execution work. A
checked-in operational amendment is independently checksum-anchored to the
immutable WP15 preregistration and the reviewed WP22 plan. Strict canonical
records describe fresh evaluation, the q6 pathwise-update diagnostic, bounded
outer-trajectory sizing, q6-only projector operator growth, scoped
publication-to-implementation bindings, budgets, and q6/q12 treatment roles.
They contain neither target vectors nor secret role entropy.

`TrainingStrategySchedule` covers direct matched-noise and noiseless controls,
noise continuation, trajectory curricula, fixed, periodic, rolling, and
resampled CRN policies, a frozen two-noise mixture, and three-start exploration.
Trajectory membership and component-local seed domains are deterministic and
checksum-linked across updates. A sealed execution-seed suite fixes every seed
preimage and sharing scope used by smoke, pilot, screening, confirmation,
diagnostics, stages, and schedules; persisted schedule member seeds are the
sampler-consumed seeds. Checkpoint selection uses only the fixed inner validation
stream; optimizer state must be preserved across schedule boundaries, and
unsupported compositions are rejected rather than approximated.

The production policy remains 200 optimizer updates, at most eight noisy
training trajectories, 256 checkpoint-validation trajectories, and validation
at update zero, every ten updates, and update 199. The q6 pilot policy fixes
1,024 fresh trajectories with nested reporting prefixes and 32 independent
pathwise update vectors; q12 fixes 256 fresh trajectories and no gradient
diagnostic. q12 is secondary descriptive evidence only. The first q6 screen is
the nonadaptive nine-method, 48-target, three-seed population of 1,296 cells.

WP22A generates no pilot, screening, promotion, or confirmation result. WP22B
provides the width-complete repository implementation catalog: q6 identities
remain frozen, q12 pilot circuits are derived without retuning, every smoke and
screen entry resolves a concrete repository runner adapter, and dormant
confirmation reuses a screened q6 configuration. Exact schedule execution and
restart are implemented by WP22C: programs retain the full executable binding,
compile every optimizer-update membership, expose validation only after an
update, preserve typed optimizer state across phase boundaries, and revalidate
the complete receipt/work chain before resume. Recoverable validation
checkpoints and complete multistart evidence use no final-test input.

WP22C creates no numerical study evidence. Target custody, execution-source
fingerprints, raw map and trajectory artifacts, screening, final sealing, and
primary analysis remain the explicit responsibilities of WP22D through WP22F.

The TFIM-only `energy_adapt_vqe` smoke cell is an analytic API preflight with
zero noisy training trajectories. It is not evidence for the preregistered
noisy Energy-ADAPT treatment, which remains unsupported, and it is excluded
from the pilot, family-wide screen, promotion, and confirmation catalogs.

### Native Gate-Count Rules

Noiseless and standard-noise rows report gate counts and depth for the logical
brickwall circuit. Ballarin rows report the final Quantinuum-native circuit
after binding, angle canonicalization, `RZZ` threshold pruning, and safe
cancellation of exact inverse compilation-only basis changes. A pruned `RXX`
or `RYY` removes its central native `RZZ` and the associated basis-change round
trip; retained basis changes count as one-qubit gates. The pre-pruning native
counts and number of pruned rotations remain available in the nested
`circuit_statistics` record. The logical parameter count and configured
brickwall layer count never change during native materialization.

### Seed Derivation

Target-generation seeds are frozen in the target fixture. For a runner
initialization seed `s`, parameter initialization uses `s`, optimizer ordering
uses `(s + 1) mod 2^64`, and the independent noisy test evaluation uses
`(s + 2) mod 2^64`. The noiseless training configuration has no trajectory
seed.

Each test trajectory is derived from the resolved test seed with stable
`SeedSequence` domain tags, the repeated-evaluation index, and the trajectory
index. Together with the runner's explicit seed offsets, this separates the
initialization, optimizer, repeated-test, and test-trajectory streams. The
public seed-domain registry also reserves distinct training-trajectory and
checkpoint-validation domains for Phase II noisy training. The Phase I runner
does not consume them. Results are reproducible independently of execution
order; do not replace the derivation with a shared mutable random generator.

## Presets and Command-Line Usage

Install the project from the repository root:

```bash
uv sync
```

The three frozen presets are:

| Preset    | Targets              | Noise identifiers                                                                                       | Optimization/test budget              | Result rows |
| --------- | -------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------- | ----------: |
| `smoke`   | First 6-qubit target | All 12                                                                                                  | 0 iterations, 2 test trajectories     |          12 |
| `minimum` | All 18               | Noiseless, Ballarin, `dephasing_1s_all`, `dephasing_2s_2q`, `depolarizing_1s_all`, `depolarizing_2s_2q` | 100 iterations, 100 test trajectories |         108 |
| `full`    | All 18               | All 12                                                                                                  | 100 iterations, 100 test trajectories |         216 |

These cardinalities are per method, layer choice, and initialization seed.
Inspect any fully resolved matrix without creating output:

```bash
uv run python -m benchmarks.state_preparation.runner \
  --preset smoke \
  --dry-run
```

Run the bounded end-to-end configuration:

```bash
uv run python -m benchmarks.state_preparation.runner \
  --preset smoke \
  --output-dir state_preparation_results/smoke
```

Run or resume the canonical minimum and full configurations:

```bash
uv run python -m benchmarks.state_preparation.runner \
  --preset minimum \
  --output-dir state_preparation_results/minimum

uv run python -m benchmarks.state_preparation.runner \
  --preset full \
  --output-dir state_preparation_results/full

uv run python -m benchmarks.state_preparation.runner \
  --preset full \
  --output-dir state_preparation_results/full \
  --resume
```

Existing output requires an explicit `--resume` or `--overwrite`. Resume
validates the manifest and artifacts, skips successful run IDs, and retries
failed or missing cells. `--overwrite` starts a replacement result stream.
Use `--fail-fast` to stop after the first failed cell. Repeated
`--num-qubits`, `--target-id`, `--noise-id`, `--method`, `--num-layers`, and
`--initialization-seed` options filter or expand a preset.

A JSON configuration can carry the same values. Command-line options override
the JSON file, which overrides the selected preset:

```json
{
  "format": "yaqs.state_preparation.runner_config.v1",
  "preset": "minimum",
  "num_qubits": [6],
  "target_ids": ["tfim_critical"],
  "test_trajectories": 500,
  "output_dir": "state_preparation_results/critical-6q"
}
```

```bash
uv run python -m benchmarks.state_preparation.runner \
  --config benchmark.json
```

## Result Schema and Artifacts

`results.jsonl` is the canonical append-safe stream. Every line is either a
successful result or a structured failure using schema
`yaqs.state_preparation.result.v1`; `status` discriminates the two forms.
`results.csv` is a derived flattened view. `manifest.json` records completed
and failed run IDs, provenance history, schema versions, and artifact
directories. Parameter checkpoints are stored under `checkpoints/`. Optional
per-trajectory fidelity sidecars use checksum-verified NPZ files under
`trajectories/`.

Stable `run_id` values hash the complete canonical configuration. Stable
training IDs omit test-only noise and evaluation policy to support train-once
fan-out. All paths in result rows are relative to the output directory, and
every checkpoint or sidecar has a SHA-256 checksum.

Successful rows expose these reporting groups:

| Group                | Fields                                                                                                                                                                                                                   |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Identity             | `schema_version`, `status`, `run_id`, `method`, `num_qubits`, `target_id`, `noise_id`, `seed`                                                                                                                            |
| Circuit              | `ansatz`, `num_layers`, `num_parameters`, `circuit_depth`, `num_1q_gates`, `num_2q_gates`, nested `circuit_statistics`                                                                                                   |
| Training             | `optimizer_budget`, `train_trajectories_or_shots`, `train_fidelity`, `optimization_wall_time_seconds`                                                                                                                    |
| Independent test     | `logical_test_noiseless_fidelity`, `native_pre_pruning_noiseless_fidelity`, `test_noiseless_fidelity`, `test_noisy_fidelity`, `test_trajectories_or_shots`, `sampled_nonidentity_events`, `evaluation_wall_time_seconds` |
| Uncertainty          | `noisy_fidelity_standard_deviation`, `noisy_fidelity_standard_error`, `confidence_interval_lower`, `confidence_interval_upper`                                                                                           |
| Provenance/artifacts | `software_versions`, `git_commit`, `git_dirty`, `git_diff_checksum`, checkpoint and trajectory paths/checksums, `wall_time_seconds`, `notes`, nested `config`                                                            |

Failure rows retain the complete nested configuration and provenance while
recording `failure_phase`, exception type/message, retryability, traceback,
and wall time. Consumers should parse the canonical JSONL through
`read_jsonl_records` or the derived CSV through `read_csv_records` instead of
depending on column order.

Phase II deliberately uses a separate store and schemas. Its authoritative
files include `stage_results.jsonl`, `stage_failures.jsonl`,
`materializations.jsonl`, `materialization_attempts.jsonl`, `results.jsonl`,
`evaluation_failures.jsonl`, and `evaluation_evidence.jsonl`; `results.csv` is a
rebuildable view. The required checksum-sealed `manifest.json` records the last
committed ledger prefixes, stream checksums, and artifact inventory so removal
or ledger rollback relative to the retained manifest is detectable on resume.
A consistent rollback of the complete store requires an external monotonic
anchor to detect. Versioned parameter checkpoints,
optimizer traces, stage metadata, fixed maps, deterministic circuit bytes, and
optional trajectory-fidelity sidecars live in dedicated subdirectories. Final
evaluation successes and failures resolve the complete pipeline result and
runtime fingerprint through stable identifiers and checksums; stage failures
instead resolve their exact configured and completed predecessor prefix.
Standalone noisy-Krotov calls may use raw arrays or MPS targets, while Phase II
artifact publication additionally requires the sealed objective to match the
pipeline's authorized materialized target and computational-zero initial state.
