# Noisy Krotov experiment handoff and scope reset

- **Status date:** 2026-08-10
- **Repository:** MQT YAQS
- **Branch:** `Noisy-State-Preparation`
- **Frozen reference commit:** `9779fdfc1816d59b9d1d1133509350430c8b6bd1`

## Purpose

This document is the handoff for the next agent. The immediate goal is **not**
to extend the Phase-II governance framework. It is to run a lean, scientifically
useful sequence of noisy state-preparation experiments using bottom-up
layerwise Krotov training.

The narrow noisy-Krotov capability was complete by WP19. WP20 through WP22H
added broad comparator coverage and a blinded, preregistered, crash-hardened
pilot/screen/confirmation system. That infrastructure is retained, but it is
deferred until small numerical runs demonstrate both a useful signal and a
feasible runtime.

## Executive decision

1. Freeze the existing WP15-WP22H implementation. Do not delete it, but do not
   add WP23/WP24 behavior now.
2. Use a short numerical ladder:
   - verified two-qubit smoke;
   - one q8 target as a runtime and fidelity calibration;
   - the exact five-target WP19 historical reproduction;
   - a compact Krotov-only corrected comparison.
3. Decide whether the large paper protocol is warranted only after reviewing
   those results.
4. Prefer a Git tag, pinned environment, declarative run configuration,
   checksummed outputs, raw traces, and independent test trajectories over more
   ceremony code for the development study.

## Why the scope is being reset

The user's original objective was noisy state preparation with Krotov. The
current repository contains about 205,995 tracked Python lines. Phase II alone
contains about 93,217 implementation lines plus roughly 47,828 dedicated
benchmark-test lines. WP15 through WP22H added about 139,321 lines, and almost
98,000 lines were added after the narrow WP19 reproduction capability existed.

That growth is not mostly Krotov mathematics. It is primarily study design,
multi-method orchestration, source locking, held-target custody, immutable
artifact inventories, crash recovery, and tests. Some tightening was justified:
the legacy audit found method-label, CRN-claim, provenance, and budget-equivalence
problems. However, the response became disproportionate to the task of obtaining
the first reliable noisy-Krotov results.

For a strong journal paper, prioritize a clear hypothesis, fair compute,
multiple targets and optimization seeds, fresh held-out noisy evaluation,
uncertainty, raw traces, and reproducible commands. The size of the custody
framework is not itself a scientific contribution.

## Current verified readiness

### Core smoke

The core optimizer works at the frozen reference commit. The following command
was verified locally:

```bash
uv run python experiments/noisy_krotov_fast.py
```

Observed runtime was about 2.08 seconds. The five-update fixed-cross-CRN noisy
trace increased from approximately `0.13475848` to `0.14124184`. This is an
environment and code-path smoke test only; it is not paper evidence because it
reports training-map fidelity, has no fresh final test, and writes no governed
result artifact.

The focused core checks were also verified:

```bash
uv run pytest -q \
  tests/optimization/test_krotov.py::test_noisy_state_preparation_crn_monotonic_descent \
  tests/optimization/test_krotov.py::test_noisy_state_preparation_training_does_not_mutate_noise_model
```

They passed in about 2.19 seconds.

### Exact historical reproduction

The maintained WP19 route is ready but has not been numerically executed to
completion. It runs only the desired bottom-up Krotov family on five q8 targets:

- depths 1, 2, 3, and 4;
- 100 noiseless growth updates at each depth;
- 200 final noisy fixed-cross-CRN updates;
- 3 noisy training trajectories;
- 500 fresh noisy evaluation trajectories for every target;
- target seeds 100, 200, 300, 400, and 500;
- serial, resumable execution with retained artifacts.

Run it from an exactly clean worktree with an output root outside the repository:

```bash
uv run python -m benchmarks.state_preparation.phase2.run_historical_reproduction \
  --output-root /absolute/external/path/wp19-historical-reproduction \
  --execute-expensive
```

Resume the same immutable run after interruption with:

```bash
uv run python -m benchmarks.state_preparation.phase2.run_historical_reproduction \
  --output-root /absolute/external/path/wp19-historical-reproduction \
  --execute-expensive \
  --resume
```

The full job performs 3,000 optimizer updates plus 2,500 fresh evaluation
trajectories and should be treated as a several-hour or overnight job until a
real timing cell provides a better estimate.

Its process exit status is scientifically meaningful:

- `0`: the five-target report classified the run as reproduced;
- `1`: the run completed but produced a numerical discrepancy;
- `2`: execution or artifact verification failed.

An agent or shell wrapper must preserve and inspect a completed exit-1 report;
it must not treat that status as permission to delete or blindly rerun the
artifacts.

### Existing exploratory data

Tracked exploratory traces already exist, including Gaussian SMPD runs, q4
diagnostics, and `experiments/results/layerwise_8q_noisy.csv`. They are useful
for debugging and historical context, but they are not a checksum-sealed WP19
reproduction and must not be presented as new paper evidence.

Do not use `experiments/noisy_krotov_exploration.py` as scientific support for a
CRN claim. The retained CSV contradicts the old manuscript's claimed CRN
performance. Do not describe the historical bottom-up method as ADAPT-VQE: it
grows a predetermined brickwall ansatz and performs no operator-pool selection.

## Immediate work plan

### Step 0: preserve the repository

Before editing or running:

```bash
git status --short
git rev-parse HEAD
```

The shared workspace currently contains a user modification to `.gitignore` and
unrelated untracked experiment/manuscript/output files. Preserve all of them.
Do not use `git add -A`, destructive cleanup, `git reset --hard`, or broad
restore commands.

For implementation work, branch from the frozen reference commit. After the
small calibration runner is reviewed and committed, run it from a separate clean
worktree pinned to that **new calibration-runner commit**; a runner added after
`9779fdfc` cannot be executed from `9779fdfc` itself. Record that exact commit in
the exploratory result.

Run the unchanged five-target WP19 historical reproduction separately from a
clean worktree pinned to `9779fdfc`, so its source identity remains the frozen
reference described above. Do not "clean" the user's current workspace to
satisfy either source-custody requirement.

### Step 1: add the missing middle-sized calibration path

There is currently a usability gap between the two-second smoke and the
multi-hour five-target job. Add one deliberately small, explicitly exploratory
single-target q8 runner.

Preferred location:

```text
experiments/noisy_krotov_single_target_calibration.py
```

Requirements:

- Reuse the existing Krotov, legacy-target, layerwise-pipeline, noise, and fresh
  evaluator implementations. Do not duplicate optimizer mathematics.
- Default to one audited q8 target seed, preferably seed 100.
- Execute the exact historical stage sequence by default so its timing predicts
  the full WP19 job.
- Keep training and final evaluation RNG domains disjoint.
- Preserve the depth-4 parameters immediately before noisy fine-tuning as well
  as the post-fine-tuning parameters.
- Sample one fresh held-out trajectory ensemble that is disjoint from every
  training map, then evaluate the pre- and post-fine-tuning parameters on that
  same paired ensemble. Never use the fixed training CRN maps as final evidence.
- Record canonical JSON or CSV containing configuration, Git commit, dependency
  versions, target identity, stage traces, final parameters or their artifact
  path/checksum, paired pre/post per-trajectory fidelities and differences,
  mean/std/error summaries, normalized work, wall time, and terminal status.
- Write only beneath an explicit output directory outside the repository.
- Refuse overwrite; allow exact resume only if it can reuse the existing Phase-II
  store without substantial new machinery.
- Clearly label every artifact `exploratory` and `not confirmatory evidence`.
- Add one focused test that uses a tiny/bounded configuration and proves fresh
  evaluation and deterministic configuration/output behavior.

Guardrails:

- One runner module and one focused test are the intended scope.
- Target roughly 300-500 new lines in total. If the implementation starts
  requiring a new schema family, registry, custody layer, or work package, stop
  and simplify.
- Do not modify the sealed five-target WP19 scientific identity merely to make
  calibration convenient.
- Do not wire this runner into WP22H or `paper-confirm`.

Before committing, run its focused tests plus the two core tests listed above.
Follow repository lint guidance on the explicitly changed files. Repository-wide
lint currently has unrelated baseline diagnostics and previously modified
unrelated files when run with auto-fixing hooks; use a clean disposable clone if
a global auto-fixing lint run is required.

### Step 2: run one q8 timing/fidelity calibration

Run the new single-target calibration from the clean worktree. Retain:

- command line and stdout/stderr log;
- exact Git commit and environment lock;
- wall time and peak memory if available;
- every stage's training trace;
- final parameters;
- raw fresh-evaluation trajectory fidelities;
- output checksums.

Review before proceeding:

1. Did every stage terminate without NaN, corruption, or silent failure?
2. Did the paired fresh held-out post-minus-pre fidelity change support an
   improvement from noisy fine-tuning rather than only an improvement on the
   training maps?
3. Is the runtime consistent with completing five targets overnight or within an
   agreed compute window?
4. Are resume and output artifacts understandable without reading internal code?

Do not launch the full WP22 pilot based solely on this timing cell.

### Step 3: execute the exact five-target WP19 reproduction

If the calibration is healthy, run the maintained command above without changing
its configuration. Preserve its four top-level outputs:

- `historical_reproduction_report.json`;
- `historical_reproduction_runtime.json`;
- `historical_reproduction_source_manifest.json`;
- `targets/seed_{100,200,300,400,500}/`.

The report must distinguish:

- exact/tolerance reproduction of the archived five fidelities;
- a complete numerical discrepancy;
- execution failure.

Do not tune hyperparameters after inspecting one target and then describe the
result as the historical reproduction. Any changed method is a new exploratory
method with a new identity.

### Step 4: run a compact corrected Krotov-only study

Only after the WP19 result and runtime are reviewed, define a compact modern
study. Its target/noise scope is explicit: use the existing public
**development-role `primary_q6` population policy** and the Phase-II primary
fixed-rate noise condition `depolarizing_1s_all` at strength scale `1.0`,
`tjm_dt=1.0`, on logical parameterized gates. The development targets are not
held confirmatory targets. Generate and retain their public configuration,
manifest, and external development entropy through the existing target APIs;
do not reuse legacy q8 targets under a relabeled modern method.

The modern builders support q6 (and their defined q12 projections), not q8. If a
new q8 modern adaptation is desired, stop and obtain explicit scientific-scope
approval; give it a distinct exploratory method identity rather than silently
changing `layerwise_bmpd_crn_v2`.

Within that q6 scope, compare these Krotov conditions:

1. corrected layerwise fixed-CRN training (`layerwise_bmpd_crn_v2`);
2. matched layerwise noiseless training (`layerwise_bmpd_noiseless`);
3. layerwise noisy training resampled at each update
   (`layerwise_bmpd_resampled`);
4. optional corrected cross-CRN ablation (`layerwise_bmpd_cross_crn`).

Use the existing builders in:

- `benchmarks/state_preparation/phase2/layerwise_bmpd.py`;
- `benchmarks/state_preparation/phase2/fair_controls.py`;
- `benchmarks/state_preparation/phase2/noisy_krotov.py`.

Declare the development training-trajectory, checkpoint-validation, optimization
seed, and update counts once in a small run configuration informed by the q8
timing calibration. Use the same applicable counts across matched q6 conditions;
do not tune them separately by target or after inspecting test results.

Do not add Adam, SPSA, pruning, operator growth, q12 scaling, or nine-method
screening at this stage. They can be considered later if the Krotov result is
promising and the target journal genuinely requires them.

Minimum design principles:

- Same targets, initialization policy, ansatz growth, update budget, and
  evaluation noise across conditions.
- At least three optimization seeds per target for development; increase only
  after measuring between-seed variability.
- Multiple target instances, not repeated trajectories from one trained circuit,
  are the primary independent replication units.
- Training, checkpoint validation, and final test seeds must be disjoint.
- Select checkpoints using validation only. Report final test performance once
  for the selected checkpoint.
- Match or explicitly report normalized trajectory-equivalent work.
- Retain failures and use an intention-to-treat summary rather than silently
  dropping difficult targets.
- Save raw trajectory results, stage traces, selected checkpoint identities,
  final parameters, resource measurements, and wall time.

Primary development question:

> Does noisy layerwise Krotov fine-tuning improve fresh held-out fidelity under
> the target noise model relative to matched noiseless layerwise training at a
> comparable compute budget?

Secondary question:

> How do fixed-CRN and per-update resampling affect convergence stability and
> held-out fidelity?

Do not formulate a final confirmatory claim or choose an effect threshold until
the development results and achievable sample size are understood.

### Step 5: paper-scope decision gate

After the compact study, prepare a short decision memo containing:

- effect estimates and uncertainty across targets and optimization seeds;
- failures and sensitivity to target/noise strength;
- runtime and projected total compute;
- whether corrected noisy fine-tuning beats the matched noiseless condition;
- whether the contribution is algorithmic, empirical, or primarily a
  reproducibility correction;
- the smallest set of external baselines needed for the intended journal.

Then choose one of two paths:

**Lean paper path:** freeze a compact Krotov protocol, add only two to four strong
baselines/ablations, broaden target families/scales in proportion to the signal,
and publish raw data plus a reproducible run manifest.

**Full blinded path:** only if explicitly desired and computationally justified,
resume the existing WP22H ceremony, 1,080-job pilot, 1,296-cell screen, WP23 held
confirmation, and WP24 analysis. The frozen pilot plus screen implies at least
roughly 17 million trajectory-path evaluations before additional optimizer
overhead, so it is not an appropriate first timing experiment.

## Files and APIs to read first

- Core optimizer: `src/mqt/yaqs/optimization/krotov.py`
- Tiny smoke: `experiments/noisy_krotov_fast.py`
- Benchmark noisy stage: `benchmarks/state_preparation/phase2/noisy_krotov.py`
- Layerwise profiles and runner:
  `benchmarks/state_preparation/phase2/layerwise_bmpd.py`
- Matched Krotov controls: `benchmarks/state_preparation/phase2/fair_controls.py`
- Historical job:
  `benchmarks/state_preparation/phase2/run_historical_reproduction.py`
- Historical targets: `benchmarks/state_preparation/phase2/legacy_targets.py`
- Legacy evidence audit:
  `benchmarks/state_preparation/phase2/data/legacy_evidence_audit_v1.json`
- Existing benchmark runbook: `benchmarks/state_preparation_benchmarks.md`
- Full implementation plan: `benchmarks/state_preparation_implementation_plan.md`

## Definition of success for the next agent

The next handoff should contain:

1. a small reviewed single-target calibration runner, or a documented reason why
   no new runner was needed;
2. its focused green tests and exact commands;
3. one completed q8 calibration with raw fresh-evaluation results and measured
   runtime;
4. a recommendation to launch, revise, or stop before the five-target WP19 run;
5. no unrelated file changes and no expansion of the WP22/WP23 framework.

The first useful numerical result is the priority. More infrastructure is not.
