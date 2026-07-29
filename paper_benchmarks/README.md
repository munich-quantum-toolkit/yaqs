# paper_benchmarks — final PRA numerical results

Complete, validated numerical dataset and publication-ready figures for the
revised Physical Review A manuscript on gate-local (projected-dynamics) TDVP
gate application. The pipeline **reuses the corrected post-repair datasets**
(repair protocol `compress_rightcanon_ltr+var_multistart+tdvp_n1_v1`) and
generates only the small amount of missing data; no new large campaigns were
run.

`raw/` and `raw_new/` are intentionally **not** in git (GitHub file-size
limits; regenerable). Keep `processed/`, `figures/`, and `configs/` as the
published artifacts; rebuild raw inputs with the stage-3 scripts below (and
optional suites such as `generate_dt001_circuits.py`).

## Directory layout

| Path | Contents |
| --- | --- |
| `configs/` | locked configuration snapshots (corrected campaigns + extension) |
| `raw/` | reused corrected raw data (**gitignored**; regenerate / copy locally) |
| `raw_new/` | new pipeline raw data (**gitignored**; regenerate via scripts below) |
| `processed/` | tidy analysis CSVs (single source for every figure) |
| `figures/` | main + supplemental figures (vector PDF + PNG preview) + captions (`.tex`) |
| `tables/` | LaTeX validation tables and macro files |
| `logs/` | per-stage logs and validation JSONs, PDF render checks |
| `scripts/` | the entire pipeline (stages below) |
| `tests/` | pytest checks (`uv run pytest paper_benchmarks/tests -q`) |
| `data_manifest.json` | provenance manifest (source, sha256, script, commit, flags) |
| `validation_report.{json,md}` | consolidated validation results |

## Pipeline stages (documented commands)

```bash
# 1. provenance manifest (audit)
uv run python paper_benchmarks/scripts/make_manifest.py

# 2. validations (must pass before any generation)
uv run python paper_benchmarks/scripts/validate_dense.py      # spec 3.1
uv run python paper_benchmarks/scripts/validate_locality.py   # spec 3.2

# 3. generate missing data only (deterministic, checkpointed, resume-safe)
PB_WORKERS=8 uv run python paper_benchmarks/scripts/generate_single_gate_ext.py
uv run python paper_benchmarks/scripts/generate_heisenberg_traj.py
uv run python paper_benchmarks/scripts/generate_1d_circuits.py

# 4. aggregate tidy processed CSVs (with consistency assertions)
uv run --with pandas python paper_benchmarks/scripts/aggregate.py

# 5. figures (main + supplemental)
uv run --with matplotlib python paper_benchmarks/scripts/plot_fig1.py
uv run --with pandas --with matplotlib python paper_benchmarks/scripts/plot_fig2.py
uv run --with pandas --with matplotlib python paper_benchmarks/scripts/plot_fig3.py
uv run --with pandas --with matplotlib python paper_benchmarks/scripts/plot_supp.py

# 6. final validation report, LaTeX macros/tables, PDF QA
uv run --with pandas python paper_benchmarks/scripts/validate_outputs.py

# tests
uv run pytest paper_benchmarks/tests -q
```

## Data provenance (summary)

Reused unmodified (checksummed in `data_manifest.json`, write-protected):

- `raw/single_gate_corrected/` — corrected single-gate campaign
  (`experiments/single_gate/regenerate.py`): RZZ, seed 11, sites (2,9),
  L=12, chi in {8,12,16}, TDVP n=1, SVD `discarded_weight` @ 1e-13,
  Krylov tol 1e-12, gate-split hard cutoff 1e-14.
- `raw/circuits_corrected/` — corrected 4x4 TFIM + Heisenberg fixed-chi
  benchmark (`experiments/fixed_resources/generate_corrected.py`): dense
  exact reference of the identical 2nd-order Trotter circuit (dt=0.1,
  30 steps, snake ordering, open BC), hybrid TDVP n=2 on long-range gates
  only, validated by a chi=256 control and a deterministic repeat.
- `raw/single_gate_validation/`, `raw/svd_diagnostic/` — audit artifacts
  (supplemental statements only).

Generated here (`raw_new/`, not committed; small jobs, ~4 min wall total —
plus optional `generate_dt001_circuits.py` / chi=8 suites which are longer):

- single-gate extension: RXX/RYY (all seeds) + RZZ (seeds 22, 33) on the
  identical corrected angle grid, chi in {8,12,16}, six methods, plus
  theta=0 identity rows; 4,059 rows in 46 s (8 workers).
- exact-limit substep study: x=1/4, chi=32 nonbinding, n in {1,...,256},
  all three gates.
- full 30-step Heisenberg chi=32 trajectories (three methods, ~3.5 min);
  the corrected campaign early-stopped these after the epsilon crossing.
- 1D chain circuits (`raw_new/circuits_1d/`, ~3.5 min): 16-site 1D TFIM
  (J=g=1, from |0...0>) and 1D XXX Heisenberg (J=h=1, from Neel), dt=0.1,
  30 second-order Trotter steps, chi=32, with dense exact references. The
  TDVP method here is `full_tdvp` (gate_mode `full-tdvp`): every two-qubit
  gate goes through the gate-local TDVP window update with n=2 substeps —
  including nearest-neighbour gates, which the 2D hybrid applies directly.
  Since all 1D gates are nearest-neighbour, TEBD+SWAP and MPO zip-up reduce
  to the identical direct SVD update (their trajectories coincide).

Excluded (documented in `data_manifest.json`): all `archive/pre_repair_*`
data, the pre-repair `save/` campaigns (including the 4,833-row
`long_range_gate_paper/trials.csv`, generated before the repairs landed),
`experiments/resource_frontier/` (not regenerated; framing dropped), and the
SVD-cutoff diagnostic campaign (internal audit only).

## Figures

- `fig1_gate_locality` — schematic: gate window, exact support of the
  fixed-rank TDVP vector field, one-site halo, frozen tensors, direct
  NN-gate note.
- `fig2_single_gate` — (a-c) infidelity vs angle per gate at chi_max=8
  (median over 3 seeds, min-max band, 4 methods, empirical theta^2 guide,
  special angles as open markers); (d) substep convergence at the
  nonbinding-cap exact-limit configuration.
- `fig3_circuits` — 2x2 circuit assessment at chi_max=32 with **full**
  gate-local TDVP on every two-qubit gate (including nearest-neighbour):
  top row 1D TFIM / 1D XXX Heisenberg, bottom row 2D 4x4 TFIM / Heisenberg.
  See `HYBRID_VS_FULL_TDVP.md` for the hybrid→full difference analysis.
- `figS1_angle_chi_grid`, `figS2_substeps`, `figS3_circuit_resources` —
  full parameter grids, complete substep data (incl. phase-aligned
  self-convergence), resource diagnostics for all four circuits (1D + 2D).
  Diagnostic overlay `fig_hybrid_vs_full_tdvp` compares hybrid vs full on 2D.

Method identity in every figure (Okabe-Ito, colorblind-safe): hybrid
gate-local TDVP = vermilion circles solid; TEBD+SWAP = blue triangles
dashed; MPO zip-up = green squares dash-dot; variational MPO = purple
diamonds dotted; no-update = gray dotted. Captions live next to the figures
as `.tex` files; numeric macros in `tables/result_macros.tex` and locked
parameters in `tables/benchmark_parameters.tex`.

## Determinism and caveats

- All generation runs pin BLAS to one thread per worker and checkpoint by
  task hash; reruns skip completed work. A deterministic-repeat check of the
  Heisenberg step-1 rows reproduces stored values to <1e-13 (see
  `validation_report.md`).
- The TEBD+SWAP Heisenberg step-1 value differs from the corrected-campaign
  row by 1.4% (roundoff amplified by severe routing truncation; both values
  far above epsilon, no conclusion affected). Flagged in the validation
  report.
- No cross-method runtime figure is published: reused and new timings come
  from different sessions. Runtime columns are retained in the processed
  CSVs for completeness.
