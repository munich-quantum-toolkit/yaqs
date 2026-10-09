# YAQS 1.0 Documentation TODO

## Goal and scope

Make the documentation a concise guide to the workflows and options available in
YAQS. Restore reliable Read the Docs builds within the 15-minute limit. Document
the existing feature set; add no library features for this work.

Work through the chunks in small, reviewable batches. Use the workflow-guide
order below for the detailed examples. Preserve supported workflows, scientific
interpretation, and existing page URLs. Keep detailed method explanations and
implementation details in advanced sections or the API reference. Leave
template-managed files unchanged; contribute required template fixes upstream.

## Build baseline

The review on 2026-10-08 covered documentation at `8684385e` and existing local
notebook caches.

- [Read the Docs build 35013130](https://app.readthedocs.org/projects/mqt-yaqs/builds/35013130/)
  at `d6e8add6` spent 116 seconds installing dependencies and reached the
  900-second timeout during Sphinx. The unfinished Sphinx command did not retain
  per-notebook timings.
- The baseline documentation install included all extras, the CUDA-enabled Torch
  stack, and development dependencies.
- `sphinx_llm.txt` started a second Sphinx build for Markdown. HTML and Markdown
  used separate notebook caches. Both caches contain executions of identical
  example code.
- An isolated HTML render at `8684385e`, with notebook execution and Markdown
  generation disabled, took 39 seconds locally. Strict checks failed on
  documentation warnings; this was not a successful full documentation build.

Historical local cache timings, mostly from Python 3.12:

| Notebook               | HTML execution | Markdown execution |
| ---------------------- | -------------: | -----------------: |
| Quickstart             |          176 s |              173 s |
| Noise characterization |           27 s |               29 s |
| Memory surrogate       |           15 s |               17 s |
| Environmental memory   |           13 s |               14 s |
| All 19 notebooks       |          322 s |              323 s |

These are execution totals from separate historical runs, not elapsed time for
one build or current RTD timings. Code hashes match 17 of the 19 current
notebooks, but library versions and execution environments have changed. Fresh
profiling must establish the remaining bottlenecks.

## Chunk 1: Restore a fast documentation build

- [x] Make Markdown generation reuse the HTML build's parsed notebooks and
      outputs. Start with `llms_txt_build_parallel=False` and an explicit shared
      `nb_execution_cache_path` in `docs/conf.py`. Preserve the MQT LLM files,
      including an explicit `llms_txt_full_build=True`.
- [x] Align local Nox and RTD documentation installation. Omit development and
      test dependencies, use CPU-only Torch, and retain dependencies needed by
      the OpenQASM 3 and surrogate examples.
- [x] Bound numerical threads and process workers in the documentation build
      environment. Preserve ordinary examples' default parallel execution and
      suppress progress bars in rendered documentation.
- [x] Measure a cold build. Record the commit, environment, dependency-install
      time, per-notebook execution time, HTML and Markdown generation time, and
      total build time. Confirm that each notebook executes only once.
- [x] Reduce example workloads only where fresh timings justify it. Preserve
      meaningful outputs and coverage of supported optional paths. Keep
      expensive numerical validation in the appropriate test or
      release-validation tier. The local cold build met the target without
      reducing workloads.
- [x] Remove `htmlzip` if the downloadable archive is not needed.

Acceptance: a cold RTD build completes in roughly 10 minutes or less, leaving
headroom below the 15-minute limit. Required examples and optional paths remain
validated. Cached builds alone do not satisfy this check.

### Chunk 1 validation: 2026-10-08

The final local cold build used `8684385e` plus the configuration changes above,
Python 3.14.2, a fresh environment, and empty uv, notebook, Numba, and Sphinx
caches. The build completed in **9 minutes 50 seconds**.

| Phase                              |  Time |
| ---------------------------------- | ----: |
| Dependency installation            | 200 s |
| HTML parsing, execution, rendering | 372 s |
| Markdown generation                |   3 s |
| LLM file combination and shutdown  |  15 s |
| Total, including environment setup | 590 s |

Notebook execution took 335 seconds within the HTML phase. All 19 notebooks
executed once; Markdown executed none. The slowest guides were circuit
observables (75 s), noise characterization (44 s), and analog simulation (41 s).
No workload reductions were needed. The build retained 33 SVG plot outputs,
per-page Markdown, `llms.txt`, and `llms-full.txt`. No notebook produced an
execution error or progress bar.

The environment contained CPU-only Torch and the OpenQASM 3 importer, with no
CUDA, Triton, development, or test packages. The strict integration test passed
with these resolved documentation dependencies. Full lint and the new-file hooks
passed.

The full build did not use `-W`. The parent build reported 1,443 warnings and
the Markdown child reported 1,204 warnings; chunk 4 must resolve these. Cold RTD
validation remains pending after publication of the changes.

Local evidence, including source hashes, package versions, full notebook
timings, logs, and generated pages:
[/tmp/yaqs-docs-chunk1-final-8684385e/report.md](/tmp/yaqs-docs-chunk1-final-8684385e/report.md).

## Chunk 2: Shorten first-use and configuration guides

- [x] Keep `quickstart.md` as a compact tour of the main use cases. Showcase
      useful scales within the documentation budget, with scientifically
      meaningful plots and a consistent journal style. Keep analog, digital,
      analog-digital, equivalence, memory, noise fitting, and surrogate
      examples. Fold plotting code, link to dedicated guides, and omit
      unnecessary advanced settings.
- [ ] Simplify `simulation_parameters.md`: one preset table, one override
      example, and short analog and digital recipes. Explain when to change a
      setting and what the change affects. Remove repeated override rules and
      move gate-update mechanics into an advanced section.
- [ ] Simplify `simulator_initialization.md`: common controls and one reusable
      example first. Move CPU-discovery details, process internals, and retry
      customization into an advanced section. Move the result catalogue into the
      results guide in chunk 3.
- [ ] Start `custom_gates.md` with the common task of supplying a custom
      unitary. Put DAG translation, `BaseGate` fields, manual gate construction,
      and generator details later in the page.
- [ ] Start `hamiltonians.md` with the built-in model catalogue and construction
      examples. Move energy and correlation contractions after model selection.
- [ ] Consolidate backend-selection explanations. Use one main representation
      guide and link to it from state, Hamiltonian, and workflow pages.
- [ ] Extend the quickstart examples into the dedicated workflow guides using
      the checklist below. Replace competing introductory examples with one main
      worked example per guide. Preserve distinct supported workflows in focused
      later sections or linked advanced guides.
- [ ] Preserve units, time grids, spatial ordering, shot and trajectory budgets,
      accuracy tradeoffs, supported restrictions, and the meaning of scientific
      diagnostics. Move deeper explanations rather than remove needed context.

Acceptance: each guide leads with its purpose, a small working example, the main
user choices, and how to read the output. Detailed signatures remain in the API
reference. All existing capabilities remain discoverable.

### Workflow guides: extend the quickstart examples

Keep the quickstart as a compact tour. Each detailed guide should stand alone,
use the corresponding quickstart model and terminology, and explain each step in
the order a user performs it. Work through one guide at a time in the order
below. Extend the scientific question and user choices before increasing system
size, trajectories, sweeps, or training cost.

Follow [PAPER_WRITING.md](PAPER_WRITING.md) for prose and structure. Begin with
the physical question, narrow to the setup and technical steps, then return to
what the results mean and where the conclusion stops. Connect paragraphs through
the questions the reader needs answered. Keep the instructional steps, but avoid
clipped prose, filler, and repeated claims.

Shared structure for each guide:

1. State the task, expected result, and prerequisites, including optional
   extras.
2. Build the model, initial state, and noise or controls. Explain their physical
   meaning, units, site ordering, and the supported input forms used here.
3. Choose accuracy and sampling settings. Explain the few settings that matter
   for this task and link to the setup guides for the full options.
4. Initialize the public interface, run the calculation, and extract the output.
   Explain the relevant result fields, array axes, and ordering beside the code.
5. Read the figure, add one useful extension or reference check, and explain the
   limits of the conclusion. End with a short options summary and related
   guides.

Keep working code visible and plotting code folded. Use the quickstart's journal
figure style, clear panel labels, units, and shared scales for comparisons.
Explain decisions and interpretation; keep algorithm derivations, full
signatures, and internal helpers in advanced sections or the API reference.
Preserve default parallel execution, show separate initialization and run calls,
and suppress progress only for the documentation. Explain the script entry-point
guard where relevant. Use supported public imports.

Where supported, extend each walkthrough with a small noise-strength comparison:
include a noiseless baseline and several clearly different strengths, such as
weak, intermediate, and strong noise. Keep the Hamiltonian or circuit, initial
state, time grid, and accuracy settings fixed. Use comparable sampling budgets,
label rates or probabilities correctly, and explain sampling uncertainty. Use
shared axes and color scales so each figure shows the physical change rather
than a change in normalization. Choose strengths after checking that the effect
is visible and the runtime is reasonable. Distinguish added Markovian noise from
the effects of coupling to an explicit environment.

#### 1. Analog simulation — `docs/examples/analog_simulation.md`

- [x] Expand the 20-site excitation-transport example: construct the XY
      Hamiltonian, prepare the localized excitation, define relaxation, select
      observables and time grid, and run noiseless and noisy simulations.
- [x] Explain occupation heatmaps and total excitation, including boundary
      reflections, relaxation, and trajectory fluctuations. Keep the analytic
      decay comparison and distinguish an ensemble expectation from a finite
      trajectory estimate.
- [x] Finish with a row or grid of occupation heatmaps for zero, weak,
      intermediate, and strong relaxation, with shared time and site axes and
      one color scale. Reuse the baseline runs. Compare total excitation on a
      companion plot and explain which transport features relaxation suppresses.
      Explain how to choose trajectory count, accuracy preset, and system size.
      Preserve distinct existing capabilities through later sections or links.

The comparison uses relaxation rates 0, 0.5, 1.5, and 4 with one shared color
scale. The nonzero rates span lifetimes from 2 to 1/4, giving visible
differences in how much excitation survives during transport. The guide follows
the physical question through the setup to the interpretation and limits of the
result.

Validation on 2026-10-09: all nine guide cells executed in 67 seconds in the
existing Python 3.12 documentation environment with capped threads and two
workers. The noiseless/strong-relaxation pair took 22 seconds; the additional
weak/intermediate runs took 43 seconds. Independent single-excitation dynamics
gave a maximum coherent occupation error of 0.0051. Checks passed for result
axes, trajectory averages, excitation conservation, conditional spatial
profiles, and analytic decay within finite-sample uncertainty at all four rates.

The isolated strict HTML build passed, and both figures were inspected. Plotting
cells were folded, Markdown reused execution, and LLM files remained available.
Two SVG MIME-priority warnings remain in the Markdown child for chunk 4. Linked
guide paths were checked but their contents were stubbed in this isolated build;
full-site and final cold-build validation remain pending.

Preview, raw trajectory data, CSV references, figures, timings, and logs:
[/tmp/yaqs-analog-guide/report.md](/tmp/yaqs-analog-guide/report.md).

#### 2. Circuit measurements — `docs/examples/circuit_shots.md`

- [x] Expand the 16-qubit circuit example: build the circuit, prepare the input,
      define damping, set shots and accuracy, initialize the simulator, and
      collect noiseless and noisy counts.
- [x] Explain outcome encoding and how counts become probabilities and grouped
      excitation-number histograms. Distinguish shots from trajectories and
      explain finite-sample fluctuations without promising identical histograms.
- [x] Compare grouped readout histograms at zero and several damping strengths,
      keeping the circuit and shot budget fixed. Explain the shift in excitation
      number and finite-sample fluctuations. Keep individual outcomes
      discoverable alongside the grouped histogram. Base
      `circuit_observables.md` on the analog XY transport example, reconstruct
      its dynamics with exchange gates and mid-circuit observable checkpoints,
      and compare noiseless and noisy results. Explain how circuit noise
      strengths relate to the represented time step. Retain OpenQASM inputs and
      gate-application choices in focused later sections.

The shot guide uses a 16-qubit graph state and rates 0, 0.1, 0.5, and 1.5,
comparing complete excitation-number histograms with a shared sampled baseline.
The observable guide uses the analog example's 20-site XY chain, localized
excitation, time grid, and rates 0, 0.5, 1.5, and 4. Symmetric Trotter steps
reconstruct the transport heatmaps. Per-site gate counts set noise strengths so
each step accumulates the intended relaxation exposure. A companion figure
compares analog and digital profiles, Trotter-step refinement, and excitation
survival. Sixteen noisy trajectories keep the observable example within the
documentation budget. Plotting code is folded.

Validation on 2026-10-09 used the existing Python 3.12 documentation environment
with capped threads and two workers. The six shot-guide cells took 77 seconds;
checks covered Qiskit probabilities, the exact noiseless binomial distribution,
weak-noise marginals within sampling uncertainty, bit encoding, and count
totals. The nine observable-guide cells took 104 seconds. Against independent
exact single-excitation dynamics, the maximum coherent occupation error fell
from 0.0070 to 0.0017 when the circuit step halved. The analog reference error
was 0.0051. Noisy outputs matched an independent finite-circuit channel
reference within sampling uncertainty, with conditional spatial-profile errors
below 0.000001. Trajectory aggregation, checkpoint axes, excitation conservation
or loss, OpenQASM counts, and both gate modes also passed checks.

The isolated strict HTML builds passed and all three figures were inspected.
Markdown reused execution, and LLM files remained available. SVG MIME-priority
warnings remain in the Markdown children for chunk 4. Linked guide paths were
checked but their contents were stubbed; full-site and final cold-build
validation remain pending.

Previews, raw data, CSV files, figures, timings, and logs:

- Shots:
  [/tmp/yaqs-circuit-guides/report.md](/tmp/yaqs-circuit-guides/report.md).
- Observables:
  [/tmp/yaqs-digital-xy-guide/report.md](/tmp/yaqs-digital-xy-guide/report.md).

#### 3. Circuit verification — `docs/examples/equivalence_checking.md`

- [x] Expand the quickstart comparison into hardware-constrained compilation, a
      deliberate rotation-angle bug, and a noise-strength sweep. Rename the
      guide and sidebar entry to Circuit Verification.
- [x] Explain checker setup, output-layout alignment, returned overlap, decision
      thresholds, and sampling uncertainty. Compare correct and faulty
      compilations across noiseless, weak, and stronger Pauli noise on shared
      axes. Distinguish the offline target and assumed noise from measured
      device data.
- [x] Keep backend selection, OpenQASM inputs, and execution controls as short
      option sections after the worked example. Validate against independent
      unitary and noisy-channel references and remove unsupported performance
      claims.

The guide compiles a four-qubit circuit for a nearest-neighbor target using
native `rz`, `sx`, `x`, and `cx` gates. Routing increases the controlled-X count
from three to nine. The reference includes the final output permutation; without
this alignment the overlap is 0.25, while the aligned comparison gives one. The
compiled circuit keeps its physical gate sequence for the noise sweep. A native
rotation-angle error follows the exact cosine overlap. Correct and faulty
compilations then use six Pauli-error probabilities and 256 trajectories per
point. Plotting code is folded, and default backend selection and parallel
execution remain enabled.

Validation on 2026-10-09 used the existing Python 3.12 documentation environment
with capped numerical threads and two workers. All five code cells executed in
about 10 seconds, including independent checks. Qiskit's layout-aware unitary
verified the compilation; dense operators checked every angle point. Exact
Qiskit channels summed the Pauli branches at every eligible gate for both
implementations. Sampled fidelities differed from these references by less than
1.6 standard errors. Additional checks covered native connectivity, retained
trajectory aggregation and error estimates, reproducibility across worker
counts, and agreement between matrix and MPO backends.

The isolated strict HTML build passed and the figure was inspected. Markdown
reused execution and LLM files remained available. SVG MIME-priority and
bibliography-node warnings remain in the Markdown child for chunk 4. Linked
guide paths were checked but their contents were stubbed; full-site and final
cold-build validation remain pending.

Preview, raw data, exact references, CSV files, figure exports, timings, and
logs:
[/tmp/yaqs-verification-guide/report.md](/tmp/yaqs-verification-guide/report.md).

#### 4. Environmental memory — `docs/examples/characterization.md`

- [x] Expand the coupling sweep: define the system and environment, configure
      the probing schedule and cut, run characterization, and extract spectra
      and entropy from the results.
- [x] Explain normalized spectral weights, entropy, and what they reveal about
      the response to the chosen probes. Distinguish these diagnostics from
      environment populations and mixed-state Schmidt spectra. Explain the
      nonmonotonic coupling result without claiming a universal memory measure.
- [x] Explain the main sampling and intervention choices. Keep conditioned reset
      delay, response-mode inspection, and process-tensor diagnostics as focused
      sections, preserving the `memory-theory` and `reset-delay` anchors.
- [x] Add a small dephasing comparison through dense process-tensor tomography,
      then characterize the reconstructed tensors. Explain that the direct
      Hamiltonian path does not accept a `NoiseModel`, and distinguish coupling
      changes from added Markovian noise. Validate the reconstruction and limit
      claims about small noisy spectral weights.

The main example uses three Ising spins, with site 0 as the probe and two spins
as its environment. Thirteen couplings share one 8-by-8 probe grid, four
interventions, and cut 2. The guide explains the five evolution intervals,
outcome-probability weighting, response-matrix axes, retained singular values,
entropy, and effective modes. A red spectrum sweep and entropy plot reproduce
the quickstart's nonmonotonic result. Seven explicit reset delays show
conditioned persistence without implying an all-outcome memory length.

The added-noise example uses two spins and one intervention to keep dense
tomography small. Each noisy reconstruction uses 16 sequences and 512
trajectories per sequence, with integration step 0.025. Dephasing rates 0, 1,
and 4 concentrate the response in the leading mode. A separate temporal-entropy
calculation explains its distinction from probe-response entropy. Public
imports, automatic representation selection, and default parallel execution
remain in the examples. Plotting code is folded.

Validation on 2026-10-09 used the existing Python 3.12 documentation environment
with capped numerical threads and two workers. All nine cells executed in 45
seconds, including independent checks. Explicit spin-Hamiltonian exponentials
and selected-branch density-matrix evolution reproduced every response-matrix
entry in the coupling and delay sweeps within 0.00000005. Checks also covered
probe reuse, SVD reconstruction, tail weight, entropy, effective modes, and the
uncoupled rank-one limit.

An independent continuous-time Lindblad generator reproduced the qualitative
noise effect. The noisy response matrices differed from this reference by at
most 0.022 and 0.031, including finite-step and sampling error. The strongest
noise's small entropy remains sensitive to the sampling floor; the guide does
not interpret every retained tail mode as physical memory. Reconstructed tensors
passed explicit Hermiticity, positivity, and causal-normalization checks.
Noiseless dense and uncapped direct-MPO tensors agreed, including response
matrices and temporal entropy.

The isolated strict HTML build passed and all three figures were inspected.
Markdown reused execution and LLM files remained available. Three SVG
MIME-priority warnings remain in the Markdown child for chunk 4. Linked guide
paths were checked but their contents were stubbed; full-site and final
cold-build validation remain pending.

Preview, raw matrices, exact references, CSV files, figure exports, timings, and
logs: [/tmp/yaqs-memory-guide/report.md](/tmp/yaqs-memory-guide/report.md).

#### 5. Noise characterization — `docs/examples/digital_twin.md`

- [x] Expand the four-site transport example: generate synthetic dynamics,
      select endpoint observations, define candidate relaxation and dephasing
      channels, choose initial guesses and bounds, and fit the rates.
- [x] Explain observation axes, time alignment, observable selection, and
      parameter order and meaning. State that channel types and locations are
      assumed known; fitting strengths does not discover an arbitrary model.
- [x] Rerun the fitted model, compare dynamics on shared heatmap scales, and
      validate withheld interior observables. Explain how measured data and
      sampling uncertainty replace synthetic input. Keep stochastic fitting and
      optimizer controls as short options after the worked example.
- [x] Show zero, weaker, reference, and stronger noise with matched occupation
      heatmaps. Fit only the reference case and reuse its fitted dynamics for
      validation. Explain the distinct physical effects of relaxation and
      dephasing without attributing their combined sweep to one channel alone.

The guide fits two local Lindblad rates in a four-spin XY chain from endpoint Z
traces. It explains the initial excitation, observation grid, jump operators,
rate bounds, mean-squared objective, and fitted result. Two figures show the
noise-strength comparison, reference and fitted transport, and predictions at
the withheld interior sites. All heatmaps share a square-root color scale. Only
one optimization runs. Public imports, automatic fitting-backend selection, and
default parallel settings remain. Plotting code is folded.

Measured-data guidance covers observable and time ordering, converting
occupations to Z expectations, equal objective weights, finite-shot uncertainty,
model assumptions, and limits on identifiability and extrapolation.
Forward-model and optimizer options explain stochastic sampling, random seeds,
scalar search, and result fields without further fits or promises of future
features.

Validation on 2026-10-09 used the existing Python 3.12 documentation environment
with capped numerical threads and two workers. All eight cells executed in 14
seconds, including independent checks; the fit took nine seconds. Fitted rates
were 0.3499 and 0.1199 for synthetic rates 0.35 and 0.12. Endpoint Z-trace RMSE
fell from 0.069 to 0.00011, and withheld interior RMSE was 0.000071.

An independent five-state vacuum-plus-single-excitation master equation,
integrated with SciPy DOP853, reproduced every site and time sample at all four
noise scales and for the fitted rerun within 0.000000000012. Checks also covered
trace, positivity, excitation conservation or loss, initial-state placement,
observation ordering, fit versus rerun consistency, process ordering, unchanged
initial guesses, and optimizer losses. Separate channel references confirmed
that dephasing preserves population while changing transport. A local endpoint
sensitivity check found distinct rate signatures; this does not establish global
identifiability or experimental confidence intervals.

The isolated strict HTML build passed and both figures were inspected. Markdown
reused execution and LLM files remained available. Two SVG MIME-priority
warnings remain in the Markdown child for chunk 4. Linked guide paths were
checked but their contents were stubbed; full-site and final cold-build
validation remain pending.

Preview, raw dynamics, independent references, CSV files, figure exports, loss
history, timings, and logs:
[/tmp/yaqs-noise-guide/report.md](/tmp/yaqs-noise-guide/report.md).

#### 6. Experimental surrogate models — `docs/examples/memory_surrogate.md`

- [x] Expand the random-control training and unseen pulse-angle sweep through
      public `MemoryCharacterizer.sample`, `train`, and `predict` calls. Explain
      the environment preparation, intervention schedule, training set,
      checkpoint-selection validation set, and chosen prediction sequences.
- [x] Explain the final-state Bloch-plane plot and coherence comparison with
      free evolution. Add an independent small-system reference comparison;
      report prediction errors and check trace, Hermiticity, and positivity
      without concealing errors through clipping or projection.
- [x] Keep the experimental and publication-status note. Explain that random
      validation accuracy does not certify chosen controls or longer horizons.
      Keep this example within the validated two-intervention horizon; reliable
      long-protocol generalization is separate work, not a documentation
      promise.
- [x] If training and validation cost permit, compare the control response at
      weak and stronger system-environment coupling. Train and validate a
      separate model for each Hamiltonian; the current model does not take
      coupling strength as a prediction input. The public training path does not
      accept a `NoiseModel`, so describe this as an environmental-coupling
      comparison. Keep added-noise training outside the documentation scope.

The guide now trains separate models at $J=0.3$ and $J=1$ using 4,096 random
unitary sequences and 256 checkpoint-selection sequences per Hamiltonian. The
schedule contains two interventions and two evolution intervals of 0.6. Both
models predict the same 61-angle pulse sweep from a probe in $|+\rangle$ and an
environment in $|0\rangle$. The worked example uses only public YAQS imports,
retains default parallel data generation, and suppresses documentation progress
bars. It replaces the repeated zero-evolution training examples with one
training loop, a control-response plot, and a coupling/reference comparison.

All 7 production code cells matched the executed notebook. In the existing
Python 3.12 documentation environment, capped to one numerical thread and two
workers, execution took 171.5 seconds including independent checks; the
two-model sampling and training cell took 167.2 seconds. The direct reference
builds the Hamiltonian from Pauli matrices and evolves the joint state with
SciPy's matrix exponential. An explicit index sum independently checks the
partial trace. Checks also cover site ordering, schedule, both intervention
outputs, periodic pulse endpoints, array shapes, finite values, and unchanged
probe preparation.

For $J=0.3$ and $J=1$, coherence RMSE was 0.0179 and 0.0339, respectively. The
maximum half-trace-norm matrix errors were 0.0513 and 0.0593. All returned
matrices had positive eigenvalues in these sweeps, with maximum trace errors
0.0062 and 0.0092. The facade makes estimates Hermitian; it does not enforce
normalization or positivity. The guide checks and reports these properties
without clipping, renormalization, or projection. It retains the experimental
and unpublished status, the fixed environment and schedule, and the limits of
generalization beyond the tested unitary controls and horizon.

The isolated strict HTML build passed and both SVG figures were inspected.
Markdown reused notebook execution and LLM files remained available. The 2 SVG
MIME-priority warnings in the Markdown child remain tracked for chunk 4. The
two-model comparison adds training cost; its place in the total 15-minute budget
must be checked in the final full-site cold build. These isolated runs use
existing dependencies and compilation caches, and linked page contents were
stubbed. They do not establish cold RTD build time.

Preview, raw predictions, independent reference states, CSV data, trained state
dictionaries, figure exports, timings, and logs:
[/tmp/yaqs-surrogate-guide/report.md](/tmp/yaqs-surrogate-guide/report.md).

#### 7. Analog-digital simulation — `docs/examples/digital_analog_simulation.md`

- [x] Replace the one-qubit example with a 20-site XY excitation echo. Prepare
      the excitation with a circuit, alternate analog intervals and staggered
      phase pulses, and explain how the final pulse restores the phase frame.
- [x] Explain program-wide settings, local segment outputs, instantaneous gate
      timestamps, and occupation extraction across repeated analog boundaries.
      Retain supported program options in a short final section.
- [x] Compare the noiseless echo with uniform relaxation at rate 0.5 and local
      Pauli-Z dephasing at rates 0.05 and 0.2. Use shared heatmap scales and
      pointwise standard errors. Explain the difference between excitation loss
      and loss of refocusing without equating the channel strengths.
- [x] Execute all cells, inspect the figures, and compare the plotted dynamics
      with an independent Lindblad reference. Keep the final full-site and cold
      RTD build checks pending.

All 7 production cells matched the executed notebook. Execution took 79.3
seconds in the existing Python 3.12 documentation environment, with one
numerical thread and two workers. The three noisy simulations took 75.2 seconds
together. The independent reference constructs the nearest-neighbor hopping
matrix and Lindblad generator in the vacuum and single-excitation sector, then
applies the phase pulses explicitly. It uses no YAQS propagation or Hamiltonian
helpers.

The noiseless return was 0.99994, against the exact value 1. The maximum
occupation error over all coherent site/time samples was 0.0051. With
relaxation, the final population was 0.21875 against the exact value 0.22313.
The two dephasing returns were 0.650 and 0.341, against reference values 0.719
and 0.350; their estimated standard errors were 0.068 and 0.059. All noisy
profiles passed the recorded finite-ensemble bounds. Checks also covered
input-state preservation, observable ordering, trajectory means, segment
continuity, timelines, and population conservation under dephasing. Population
uncertainty was calculated after summing sites within each trajectory.

The isolated strict HTML build passed and both SVG figures were inspected.
Markdown reused execution and LLM files remained available. Three MIME-priority
warnings remain in the Markdown child: two figure outputs and one text output.
These remain tracked for chunk 4. Linked page paths were checked, but their
contents were stubbed. The run used existing dependencies and warm compilation
caches, so it does not establish full-site or cold RTD build time.

Preview, raw trajectories, independent references, CSV data, figure exports,
source hash, versions, timings, and logs:
[/tmp/yaqs-hybrid-guide/report.md](/tmp/yaqs-hybrid-guide/report.md).

#### Preserve other workflows and validate each guide

- [ ] Keep analog-digital programs, custom gates, hardware models, scheduled
      jumps, and ensembles discoverable. Reuse setup and terminology where
      useful; retain their distinct worked examples rather than force them into
      a quickstart example that does not cover their purpose.
- [ ] Execute each revised guide independently and inspect its figures. Check
      public imports, result interpretation, meaningful numerical references,
      links, and lint before Aaron reviews the guide.
- [ ] Record per-guide execution time and cumulative cold-build cost. Quickstart
      and detailed notebooks execute separately; matching code snippets alone do
      not share execution. Reuse expensive fits and trained models within each
      guide, and avoid extra runs merely to produce another plot.
- [ ] Remove superseded introductory examples and repeated option catalogues
      only after checking that all distinct supported capabilities remain
      documented. Update quickstart links and the homepage task table as needed.

Acceptance: each main quickstart example has a clear, independently runnable
walkthrough with an explained extension or validation. Include a meaningful
noise-strength comparison wherever the public workflow and build budget allow
it; document the supported alternative where they do not. The guide teaches
users how to adapt the workflow, retains its scientific limits, and fits the
final cold-build budget. Complete the full-site validation in chunk 4 after the
guide reviews.

### Quickstart validation: 2026-10-08

The six workflows executed in 136 seconds in the existing Python 3.12
documentation environment. Noiseless and noisy transport on 20 sites took 23
seconds; 16-qubit readout took 26 seconds; endpoint noise fitting and its
simulation rerun took 9 seconds; surrogate training and its pulse-angle sweep
took 76 seconds. Equivalence and memory sweeps together took 3 seconds. The
isolated strict HTML build took 142 seconds.

Numerical checks passed for excitation conservation, circuit sampling and its
damping shift, analytic equivalence overlaps, memory spectra, and fitted noise
rates. Earlier independent transport and withheld-site noise checks are in the
comparison report linked below; those five example workflows are unchanged.

The surrogate uses only public `MemoryCharacterizer.sample`, `train`, and
`predict` calls. It trains on 4,096 random unitary sequences and selects a
checkpoint using 256 random validation sequences. One model predicts 61 chosen
Z-pulse angles, applied between two evolution intervals. The figure shows final
coherence against pulse angle and the predicted free-evolution baseline.

Private matrix-exponential references give coherence RMSE 0.034 and maximum
error 0.078 across the sweep. Complex density-matrix entry RMSE is 0.032. The
predictions are unmodified, Hermitian, and positive in this example, with trace
errors below 0.005. The zero and full-turn pulses give the same prediction. The
smaller training budget failed fresh sweep checks. This validates the shown
short-horizon coherence sweep, not arbitrary controls, long horizons, or exact
density-matrix reconstruction. Independently check the surrogate guide's
chosen-control examples during its review.

All 15 production code cells matched the executed notebook. Six SVG figures
rendered, plotting and training cells were folded, and LLM files remained
available. The surrogate figure shows final probe states in the Bloch plane,
colored by pulse angle, beside a coherence sweep with shaded gains and losses. A
section note marks surrogate modeling as experimental and not yet supported by a
published YAQS paper. The updated figure was inspected. Other figures were
inspected in the earlier comparison build. The Markdown child retains six SVG
MIME-priority warnings for chunk 4. This used warm numerical caches; final
full-site and cold RTD validation remain pending.

Current plot, pulse-sweep data, package versions, timings, and logs:
[/tmp/yaqs-quickstart-pulse-sweep/report.md](/tmp/yaqs-quickstart-pulse-sweep/report.md).

Earlier independent checks and longer-horizon training trials:
[/tmp/yaqs-quickstart-generalization/report.md](/tmp/yaqs-quickstart-generalization/report.md).

### Analog-digital quickstart addition: 2026-10-09

The quickstart now includes the 20-site XY excitation echo with three occupation
heatmaps: free evolution, refocusing, and refocusing with Pauli-Z dephasing at
rate 0.2. All panels share one scale. The setup uses public imports, separate
simulator initialization, default parallel execution, and 32 noisy trajectories.
Plotting code is folded, and the section links to the detailed program guide.

All 17 production cells matched the executed notebook. The complete quickstart
executed in 160.9 seconds; the new simulation cell took 15.8 seconds. Numerical
checks passed for all seven workflows. The noiseless return was 0.99994, and the
dephased return was 0.341 against an independent Lindblad reference of 0.350,
with estimated standard error 0.059. Checks covered every new heatmap sample,
the plotted arrays, input preservation, segment continuity, observable order,
trajectory means, and excitation conservation on each trajectory.

The isolated strict HTML build passed. The new figure was inspected, all seven
SVG figures rendered, and Markdown reused execution. LLM files remain available.
The Markdown child retains seven SVG MIME-priority warnings for chunk 4. This
run used the existing Python 3.12 documentation environment with one numerical
thread, two workers, and warm compilation caches. Linked page contents were
stubbed; full-site and cold RTD validation remain pending.

Preview, numerical checks, raw echo trajectories, independent references, CSV
data, figure exports, source hash, versions, timings, and logs:
[/tmp/yaqs-quickstart-echo/report.md](/tmp/yaqs-quickstart-echo/report.md).

## Chunk 3: Fill practical gaps and correct claims

- [ ] Add one supported-combinations overview. Consolidate representation,
      noise, circuit, diagnostic, ensemble, piecewise-program, and
      characterization restrictions. Link to existing guides for details.
- [ ] Add a short results guide covering observable ordering, array axes, time
      grids, counts, requested versus executed trajectory counts, diagnostics,
      spectra, final states, and program segments. Explain that averaged noisy
      Schmidt spectra describe pure trajectories, not a mixed-state spectrum.
- [ ] Explain when outputs are populated and which combinations are supported.
      Keep result-field details accurate without adding a persistence API.
- [ ] Document pickle as trusted, temporary, same-version checkpoint storage.
      Avoid promises of portable or versioned persistence.
- [ ] Add a complete parallel script example with an
      `if __name__ == "__main__":` guard. Explain notebook execution separately
      and keep automatic process-context guidance current.
- [ ] Correct reproducibility claims, including the limits of `State(seed=...)`.
      Use supported seeds where examples need repeatable stochastic results.
- [ ] Correct claims about configuration mutation and automatic backend
      selection against the implementation.
- [ ] Supply existing evidence for the equivalence-performance crossover claim,
      including the referenced benchmark script, or remove the claim. Describe
      the configured automatic cutoff as a heuristic.
- [ ] Use supported public imports in ordinary examples. Mark intentionally
      documented low-level interfaces clearly without exporting extra helpers
      solely for documentation.

Acceptance: users can choose a supported workflow, interpret its output, and
understand relevant limitations. No major feature needs another broad tutorial.

## Chunk 4: Simplify navigation and complete validation

- [ ] Align the documentation homepage's title and introduction with the README.
      Remove the unsupported "under a minute" quickstart promise. Shorten the
      21-row learning-path table to the main user tasks. Describe executable and
      static examples accurately.
- [x] Regroup sidebar navigation while preserving existing page URLs:

| Group                             | Contents                                                                    |
| --------------------------------- | --------------------------------------------------------------------------- |
| Start here                        | Installation and quickstart                                                 |
| Simulation setup                  | States, Hamiltonians, noise, representations, presets, execution, results   |
| Simulation workflows              | Analog, circuits, shots, analog-digital programs                            |
| Characterization and verification | Memory, noise fitting, circuit equivalence                                  |
| Advanced examples                 | Ensembles, scheduled jumps, custom gates, hardware, experimental surrogates |
| Reference and contributing        | API, citations, changelog, upgrading, development, support                  |

- [ ] Curate API navigation around the stable public interface. Remove duplicate
      object indexing and unresolved targets. Keep implementation helpers from
      overwhelming the public reference and preserve needed canonical links.
- [ ] Fix the unsupported Mermaid directive, notebook metadata and lexer
      warnings, document and method references, and remaining citation or
      included-file references. The bibliography directive was repaired in the
      README/reference update; verify the merged version rather than repeat it.
- [ ] Add a fast strict documentation check to CI. Pass a fresh
      `sphinx-build -E -a -n -T -W --keep-going` check without blanket warning
      suppression. Scope necessary external-reference exceptions narrowly.
- [ ] Execute all documentation notebooks in a clean environment, including
      optional examples. Validate supported imports and meaningful outputs
      without duplicating the numerical test suite. Record execution timings.
- [ ] Run `uvx nox -s docs -- -b linkcheck`. Fix broken project links and record
      necessary exceptions for unavailable external sites.
- [ ] Inspect rendered desktop and narrow-screen pages: navigation, code,
      figures, tables, diagrams, API links, citations, and release notes.
- [ ] Run `uvx nox -s lint` after each batch of changes.
- [ ] Confirm that RTD builds the final reviewed documentation successfully.
- [ ] Complete Aaron's documentation review and resolve substantive findings.

Acceptance: users can find each supported capability through the sidebar and
task links. Strict checks, example execution, link checking, rendered-page
inspection, and a cold RTD build pass for the reviewed commit.

### Navigation validation: 2026-10-09

The sidebar uses the six groups above with short labels. All 29 existing page
targets remain present once, and page URLs are unchanged. The results guide can
join Simulation setup when chunk 3 adds it.

A full HTML render with notebook execution disabled succeeded. Rendered sidebar
checks passed on the homepage, quickstart, equivalence guide, and API root: each
page retains all six groups in order and links to all existing targets. This
check omitted external inventories and did not use `-W`; the build retained 477
documentation warnings. Full strict validation and responsive browser inspection
remain pending. Full lint and the planning-file hooks passed.

Navigation preview and evidence:
[/tmp/yaqs-docs-navigation/html/index.html](/tmp/yaqs-docs-navigation/html/index.html),
[/tmp/yaqs-docs-navigation/record.json](/tmp/yaqs-docs-navigation/record.json),
and
[/tmp/yaqs-docs-navigation/build.log](/tmp/yaqs-docs-navigation/build.log).
