# YAQS 1.0 Release TODO

## Goal and scope

Release the existing YAQS feature set with correct numerical behavior, a clear
stable API, and working documentation. Add no new features before 1.0. HDF5
persistence is outside this release.

Complete correctness repairs and validation first. Then Aaron reviews the code
and reads and updates the README and documentation. Resolve findings from those
reviews before validating and publishing the release candidate.

Chunks 1 through 5 contain the required software-release work. Optional cleanup
and SciPost paper work have separate sections below.

## Current status

The review on 2026-10-07 covered local `remove-legacy` at `9b066659` and GitHub
`main` at `e22c3ef4`. The local branch was merged through PR #609; later changes
on `main` include dependency and tooling updates.

- [x] Establish one public spatial ordering: site 0 is the least-significant
      subsystem. Cover asymmetric states, operators, local noise, observables,
      mixed physical dimensions, and memory-characterization conversions.
- [x] Use uncapped direct process-tensor construction by default. Warn that
      finite branch caps are experimental and document exponential cost.
- [x] Validate process-tensor shape, conditional-state positivity, Bloch bounds,
      and information metrics. Cover dense and analytic references.
- [x] Harden public simulation controls, mutable time grids, Hamiltonian inputs,
      real-valued results, state inputs, tensor shapes, and cross-object
      physical dimensions. Test public errors under normal Python and
      `python -O`.
- [x] Reject unsupported dense-representation diagnostics and multi-time output
      combinations. Preserve sampled and final Schmidt spectra for MPS runs,
      including noisy trajectories and direct MPS spectrum evaluation.
- [x] Remove legacy solver fallbacks and preserve explicit backend selection.
- [x] Separate local observables from gate classes and support explicit
      final-state full-chain expectations through `MPS.expect_mpo()`.
- [x] Pin the current top-level exports with `tests/test_public_api.py`.
- [x] Make core-only imports silent and include `py.typed` in the wheel.
- [x] Add serial/parallel reproducibility, explicit process-pool, spawn-process,
      and JIT-enabled tests. Keep slow rate recovery in the manual release tier.
- [x] Pass the local Python 3.14 serial suite: 3,288 passed, 3 skipped, 3
      deselected, and 5 xfailed. Pass the manual rate-recovery test separately.
- [x] Pass current-main CI on Ubuntu, Ubuntu ARM, macOS, and Windows, including
      Linux and Windows JIT checks and the clean-wheel check.
- [x] Build and inspect the current-main sdist and wheel.

These checks apply to the named commits. They do not replace validation of the
final candidate. Strict documentation has known content defects. Fresh clean
notebook execution and external link checking remain release requirements.

## Working rules

- Keep each repair small enough for a focused review.
- Add or update behavioral regressions for every code change in the test tree
  owned by the affected component. Test the supported public contract.
- Run targeted tests during development and `uvx nox -s lint` after each batch
  of changes. Resolve substantive failures before moving to the next chunk.
- Update `CHANGELOG.md` and `UPGRADING.md` for user-facing or breaking changes.
  Include required PR references and author links, and disclose AI assistance in
  any authorized pull request.
- Preserve independent numerical references and slow scientific regressions.
  Remove only genuine duplication or unnecessary cost.
- Preserve Python 3.11 through 3.14 testing on every supported CI platform.
- Fix public-boundary validation and numerical defects without adding repeated
  whole-network checks to numerical loops or rewriting the package structure.
- Keep unsupported, approximate, and experimental combinations explicit.
- Do not modify template-managed files directly. Address template changes
  upstream or state package-specific behavior in repository-owned documentation.

## Chunk 1: Correctness repairs and validation

Complete this chunk before Aaron's code and documentation reviews.

### 1.1 Make Schmidt-spectrum output agree with the supported contract

Spectrum observables retain trajectory, sample, and coefficient axes. MPS analog
and digital runs, deterministic list ensembles, and simulation programs support
sampled and final-only spectra, including noisy pure trajectories.

- [x] Define result shapes: `(num_traj, num_samples, 500)` for trajectories and
      `(num_samples, 500)` for means. Preserve descending coefficients and `NaN`
      padding from direct `MPS.get_schmidt_spectrum()`.
- [x] Repair worker buffers and result storage without changing scalar result
      shapes or observable ordering.
- [x] Average trajectory coefficients with missing ranks counted as zero.
      Preserve padding when every trajectory lacks a coefficient. Explain that
      this mean is not a Schmidt spectrum of the mixed state.
- [x] Stitch program mean spectra along the sample axis and retain individual
      trajectories in segment results.
- [x] Cover both analog orders, digital checkpoints, changing ranks, mixed
      observable ordering, sampled and final-only output, deterministic
      ensembles, and noisy serial/parallel execution. Preserve direct MPS tests
      and independent dense SVD and analytic trajectory references.
- [x] Update observable and result documentation, examples, and release notes.
      Remove the obsolete spectrum-concatenation branch and its tests.

Acceptance: scheduled spectra agree with independent references, preserve their
axes across execution paths, and work for noisy trajectories without requesting
a representative final state.

### 1.2 Honor noise-optimizer controls and report losses accurately

Noise fitting applies the validated optimizer limits and reports the loss of the
supplied initial model separately from candidate history.

- [x] Pass `max_iter` to bounded scalar search. Document CMA-ES generations,
      SciPy's evaluation stopping limit, and the two-evaluation startup case.
- [x] Evaluate the initial model before optimization and store `initial_loss`.
      Make `sqrt_loss_before()` use this baseline. Keep candidate history
      separate from baseline and final-fit evaluations.
- [x] Add public characterization regressions with one and two fitted parameters
      and small iteration limits. Cover scalar and CMA-ES dispatch.
- [x] Compare the baseline with independent analytic Pauli-noise trajectories.
      Preserve fitted-rate and dynamics recovery coverage.
- [x] Update result and optimizer docstrings, the digital-twin example, and
      release notes.

Acceptance: optimizer controls affect execution, and the before-optimization
loss describes the supplied initial model. Fitted-rate recovery still passes.

### 1.3 Honor quiet simulation during shot readout

Simulator progress controls cover trajectory and shot-readout bars, including
program segments. Progress remains visible by default. Simulator shot readout
runs within each trajectory and does not create another process pool.

- [x] Make shot readout respect `show_progress=False` through `Simulator.run`.
- [x] Document execution controls: `parallel` and `max_workers` govern
      trajectory pools, and simulator shots run serially within each trajectory.
      Direct `MPS.measure_shots()` retains parallel sampling.
- [x] Remove nested shot pools from combined noisy observable-and-shot runs.
      `parallel=False` keeps simulator readout in the current process.
- [x] Add regressions for quiet and visible readout, direct MPS sampling,
      program segments, worker limits, and shot counts. Preserve seeded
      trajectory tests.

Acceptance: documentation runs can suppress all progress bars, default progress
remains visible, and shot readout cannot bypass simulator worker controls.

### 1.4 Verify existing workflows after the repairs

Use existing tests and references first. Add tests for uncovered supported
contracts or concrete regressions, not for a new feature matrix.

- [x] Run targeted regressions for the repairs above and public validation.
- [x] Check asymmetric analog evolution across MPS/TJM, vector/MCWF, and
      density-matrix/Lindblad paths against independent dense references.
- [x] Check noisy analog dynamics against analytic or Lindblad references.
      Preserve jump-probability and trajectory-convergence checks.
- [x] Check default-MPO digital evolution against Qiskit, including long-range
      and multi-qubit gates, observable ordering, and shot counts.
- [x] Check list ensembles, piecewise evolution, and mixed programs against
      existing independent or exact small-system references.
- [x] Check memory characterization, default direct process tensors, conditional
      responses, information metrics, and noise fitting against existing
      references.
- [x] Re-run the full serial suite with numerical-library thread limits.
- [x] Pass the supported OS/Python CI matrix and minimum-dependency tests.
      Review warnings from minimum-dependency runs, not only their exit status.
- [x] Pass Linux and Windows JIT tests with compilation enabled and no coverage.
- [x] Pass `uvx nox -s release-tests` and `uvx nox -s lint`.
- [x] Record commit, environment, commands, outcomes, and accepted limitations.
      Account for every skip and xfail. The five circuit-TDVP rank-growth xfails
      may remain only with accurate documentation and a supported default route.
- [x] Replace automatic Linux `fork` with `forkserver`. Preserve explicit start
      methods and real process-pool coverage.

Acceptance: no advertised stable workflow has an unresolved failure. Relevant
independent references support numerical agreement.

Validation on 2026-10-08 covers `pool-fix` at `7d1e5c3a` on Linux with Python
3.14.2. The full serial suite passed with 3,378 passed, 3 skipped, 3 deselected,
and 5 xfailed in 135.58 seconds. The skips are invalid site/chain combinations.
The five xfails cover documented rank-growth limits in the optional circuit-TDVP
paths; the default MPO route passes. The three deselected tests passed
separately: one manual rate-recovery test and two JIT tests. Lint passed.

The serial run used numerical thread limits of one, `YAQS_MAX_WORKERS=2`,
`NUMBA_DISABLE_JIT=0`, and pytest
`-n 0 -p no:cacheprovider -m 'not release and not jit' --durations=30`.
Dedicated integration tests retain explicit process pools. Nox ran
`release-tests` and `jit-tests` without coverage. Environment versions,
commands, XML results, logs, and accepted limitations are recorded in
`/tmp/yaqs-1-4-validation-7d1e5c3a/`.

[CI run 37695876971](https://github.com/munich-quantum-toolkit/yaqs/actions/runs/37695876971)
passed for the same commit. Python 3.11 through 3.14 passed on Ubuntu x86,
Ubuntu ARM, macOS, and Windows. Both Ubuntu minimum-dependency runs passed, as
did Linux and Windows JIT tests and the Linux wheel check.

The earlier Python 3.14 minimum-dependency CI run emitted 14 warnings about
`fork` from multi-threaded processes. Automatic Linux context selection now uses
`forkserver`. Explicit start methods remain supported.

Local follow-up validation on 2026-10-08 covers `7d1e5c3a` plus the working-tree
repair. The full Python 3.14 suite passed with 3,382 passed, 3 skipped, 3
deselected, and 5 xfailed in 141.86 seconds. The minimum-dependency suite passed
with 3,382 passed, 3 skipped, and 5 xfailed in 46.00 seconds. Both runs treated
`multiprocessing.popen_fork` deprecation warnings as errors; neither reported
fork warnings. Minimum dependencies include Numba 0.63.0, NumPy 2.3.2, SciPy
1.16.1, and Torch 2.9.0 CPU. All optional tests remain included.

The 36 targeted checks passed on Python 3.11 and 3.14. They include a live
parent thread with two real workers, serial/parallel shot counts, seeded
noise-fitting trajectories, and automatic and explicit-spawn equivalence
checking. Lint passed. Commands, source checksums, environments, timings, and
logs are in `/tmp/yaqs-forkserver-validation-7d1e5c3a/`. The supported OS/Python
CI matrix must validate the updated branch before merge.

### 1.5 Continue trajectory RNG streams across MCWF memory segments

MCWF memory characterization uses one continuous RNG stream per trajectory
across evolution segments. Independent noisy-channel references protect
conditional process-tensor responses and joint weights.

- [x] Let consecutive MCWF segments reuse the worker-owned trajectory RNG.
- [x] Forward that RNG through the memory-characterization backend.
- [x] Compare noisy dense process-tensor conditional responses and joint weights
      with an analytic channel across two nonzero evolution slots.
- [x] Preserve single-segment reproducibility and the existing TJM RNG contract.

Acceptance: seeded MCWF process-tensor predictions agree with independent noisy
channel references within justified sampling error.

## Chunk 2: Aaron's code review and API freeze

Start after chunk 1 passes. Review substantive correctness and maintainability.
Avoid broad refactoring for appearance or module size.

- [ ] Review simulator dispatch, state handoff, parameter mutation, result
      allocation, time grids, observable ordering, and output aggregation.
- [ ] Review MPS/MPO conversions, site ordering, physical dimensions,
      orthogonality-center tracking, normalization, and truncation contracts.
- [ ] Review analog integration, dissipation, jump selection, and supported
      MCWF, Lindblad, ensemble, and piecewise paths.
- [ ] Review digital gates, long-range operations, measurement, noise
      application, and equivalence-checking results and approximation claims.
- [ ] Review memory and noise characterization, process-tensor physicality,
      conditional responses, information metrics, and optimizer contracts.
- [ ] Review public validation, exception behavior, RNG streams, optional
      imports, parallel execution, and optimized-Python behavior.
- [ ] Review automated coverage against these contracts. Preserve independent
      references and distinguish unit tests from scientific reproduction.
- [ ] Record and resolve major findings with focused tests. Re-run affected
      checks before accepting each repair.
- [ ] Define stable import paths for existing facades, result types, and
      process-tensor types. Use existing paths where practical; do not expand
      the public surface merely to flatten the package.
- [ ] Pin the final supported boundary in public API tests. Keep workers,
      encoders, backend helpers, and other implementation details outside it.
- [ ] Confirm that supported combinations work or have clear errors, and every
      approximation or experimental path has a stated limitation.
- [ ] Complete human review of the code and materially AI-assisted changes.

Acceptance: Aaron understands the major numerical and public contracts and has
no unresolved major finding. The stable API can carry compatibility promises
throughout 1.x.

## Chunk 3: Aaron's README and documentation review

### 3.1 Read and update the user-facing text

- [ ] Read and update the README: installation, first-use examples, feature
      scope, limitations, support, and software and method citations.
- [ ] Read the installation and all user guides. Check each guide against the
      final implementation and supported imports.
- [ ] Review the API reference for accurate signatures, result fields, return
      types, defaults, units, ordering, shapes, and error behavior.
- [ ] Add one supported-combinations table for analog representations, digital
      MPS simulation, list ensembles, piecewise and mixed programs, local and
      bitstring observables, shot readout, and characterization backends.
- [ ] State process-tensor exponential cost, operations that densify, finite-cap
      experimental status, and circuit-TDVP rank-growth limits. Keep the default
      circuit MPO route and explicit `MPS.expect_mpo()` contractions clear.
- [ ] Correct `UPGRADING.md`: `Observable` exposes `.type`, not `.kind`.
- [ ] Document existing `Result` fields, requested versus executed trajectory
      counts, time and observable axes, shot totals, diagnostics, final states,
      multi-time outputs, and nested program results. Add no result API
      features.
- [ ] Describe pickle as trusted, same-version, temporary checkpoint storage. Do
      not promise portable or versioned result persistence.
- [ ] State the tested Python range as 3.11 through 3.14 and confirm support
      ownership and the maintenance policy.
- [ ] Use consistent terms and precise prose. Separate method-paper citations
      from the software citation and bound numerical claims by their evidence.
- [ ] Publish existing evidence for the claimed equivalence-checking eight-qubit
      crossover, including the referenced benchmark script, or remove the claim.
      Describe the configured automatic cutoff as a heuristic where appropriate.
- [ ] Read and finalize release notes for all user-facing and breaking changes,
      with PR references and all contributing authors.

### 3.2 Verify examples and the documentation build

- [ ] Use supported imports in first-use and ordinary user examples. Mark any
      intentionally documented low-level API clearly.
- [ ] Give stochastic examples fixed seeds where reproducibility matters. Use
      documented valid seeds, including zero where supported.
- [ ] Add tolerances and independent reference assertions to the small
      representation-comparison example. Verify other release examples' expected
      outputs without duplicating the numerical test suite.
- [ ] Fix the unknown Mermaid directive in `docs/index.md` and malformed
      bibliography directive in `docs/references.md`.
- [ ] Fix document, method, citation, and included-file references, including
      first-use links and the `State.from_mps` reference.
- [ ] Curate AutoAPI to remove duplicate objects and unresolved targets and
      distinguish stable API from implementation details.
- [ ] Add a fast strict documentation check to CI. Pass a fresh
      `sphinx-build -E -a -n -T -W --keep-going` build without blanket warning
      suppression. Keep necessary external-reference exceptions narrow.
- [ ] Execute every documentation notebook and README workflow in a clean
      environment using a wheel built from the reviewed checkout. Cover optional
      examples with declared extras and keep the source tree off the import
      path. Repeat these checks on the tagged release candidate in chunk 5.
- [ ] Run `uvx nox -s docs -- -b linkcheck`. Resolve broken project links and
      record necessary exceptions for unavailable external sites.
- [ ] Inspect rendered desktop and narrow-screen pages: code blocks, diagrams,
      figures, tables, navigation, API links, and included release notes.
- [ ] Confirm that Read the Docs builds the reviewed documentation successfully.
- [ ] Complete Aaron's own README and documentation review and resolve findings.

## Chunk 4: Compatibility policy and release metadata

- [ ] Commit to compatibility for the documented stable API throughout 1.x.
      Remove the changelog exception permitting breaking minor releases, or
      limit that exception explicitly to pre-1.0 versions.
- [ ] Finalize the 1.0 changelog and migration instructions from the last
      release.
- [ ] Change the final package development-status classifier from Beta to the
      appropriate stable-release status.
- [ ] Add `CITATION.cff` for the software release and align citation
      instructions.
- [ ] Verify authors, maintainers, license, supported Python versions, extras,
      package metadata, and repository, changelog, and release URLs.
- [ ] Plan the software archive and DOI or release identifier. Add the final
      identifier to citation and release metadata when it becomes available.
- [ ] Review GitHub issues against the frozen scope. Record that issue #416's
      original gate-coupling problem is resolved; keep issue #35 and other
      new-feature requests outside the 1.0 milestone.
- [ ] Require optional runner upgrades or other infrastructure changes only when
      they fix a demonstrated release failure.

## Chunk 5: Release candidate and final gate

Mark these complete for the exact candidate, even where an earlier commit passed
an equivalent check. Re-run affected checks after any candidate change.

- [ ] Publish or distribute `1.0.0rc1` after correctness and human reviews pass.
- [ ] Build the sdist and wheel from the candidate tag with its actual version.
- [ ] Inspect artifact contents: version, public modules, `py.typed`, license,
      metadata, and required source-distribution files.
- [ ] Install and exercise the wheel and sdist in clean environments. Verify a
      silent core-only import and documented workflows outside the source tree.
- [ ] Run current dependencies on Python 3.11, 3.12, 3.13, and 3.14 on Ubuntu,
      Ubuntu ARM, macOS, and Windows. Preserve the current coverage policy.
- [ ] Run minimum dependencies on Ubuntu with Python 3.11 and 3.14.
- [ ] Pass JIT-enabled checks on Ubuntu and Windows, optional-dependency checks,
      the clean-wheel check, and the manual scientific release tests.
- [ ] Run the full serial suite and `uvx nox -s lint` on the final source.
- [ ] Pass strict documentation, installed-wheel examples, all executable
      notebooks, external link checking, and Read the Docs.
- [ ] Review remaining skips, xfails, warnings, approximations, and experimental
      paths. Confirm their technical reasons and accurate user documentation.
- [ ] Record source commit and tag, dependency versions, commands, test and
      documentation results, and artifact checksums in the release evidence.
- [ ] Complete Aaron's final review of artifacts and user-facing material.
      Resolve every remaining software release blocker.
- [ ] Tag and publish 1.0, archive the exact published source and artifacts, and
      complete citation metadata with the archive identifier.

For memory and process-tensor checks, use serial execution and capped numerical
threads when needed:

```bash
OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
NUMBA_NUM_THREADS=1 \
uv run pytest -n 0 -p no:cacheprovider tests/characterization/memory tests/test_memory_characterizer.py
```

## Optional cleanup

These tasks do not block 1.0 unless they expose a correctness, reliability, or
resource-use defect.

- [x] Consolidate duplicate analog and digital golden tests while preserving
      independent references, observable ordering, and actual pool coverage.
- [x] Reduce redundant shot sampling or flaky random assertions using tests of
      the intended probability or measurement contract.
- [ ] Shorten the quickstart and move advanced material into focused guides
      where that improves first use.
- [ ] Reduce optional documentation dependency cost if the build is needlessly
      expensive. Preserve coverage of supported optional paths.
- [x] Record slowest-test durations and investigate avoidable memory use. Keep
      scientific regression tests whose cost protects meaningful behavior.

## SciPost paper evidence

These tasks are required for the paper and its numerical claims. They do not
block the software release unless the release makes the same unsupported claim.
Archive data without adding package persistence or uncertainty APIs.

- [ ] Select paper examples, benchmarks, figures, and claims supported by the
      frozen existing feature set.
- [ ] Archive scripts, inputs, probe settings, intervention schedules, seeds,
      raw outputs, and plotting scripts for every quantitative claim.
- [ ] Record the YAQS commit and tag, dependency lock or container, Python
      version, OS, hardware, commands, metrics, and tolerances.
- [ ] Add checksums for each paper input, result, and figure. Store numerical
      arrays with readable metadata rather than relying on saved Python objects.
- [ ] Reproduce every paper figure and numerical table in a clean environment
      using archived inputs and the published software artifact.
- [ ] Report actual trajectory counts and statistical errors from archived raw
      trajectories in the paper analysis.
- [ ] Complete and archive the response-matrix campaign if the paper uses it.
      Keep unresolved scientific discrepancies explicit.
- [ ] Finalize the paper's software citation and archive links separately from
      citations to the underlying methods.

## Deferred features and refactoring

The following work is outside 1.0:

- HDF5 persistence, a versioned result-file schema, or a new save/load API.
- Early stopping for stochastic trajectories.
- A circuit statevector backend.
- Neural-network noise characterization.
- A new process-tensor compression algorithm.
- Scheduled or trajectory-averaged `Observable(MPO)` support.
- Converting local observables, diagnostics, or bitstrings into MPOs.
- A new uncertainty-reporting `Result` API.
- Multi-level or qudit simulation features.
- Trajectory visualization.
- Broad rewrites based only on module size and a package-wide validation
  reorganization.
- Compatibility adapters for abandoned pre-1.0 APIs.
