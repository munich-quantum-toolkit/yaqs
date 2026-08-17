# BUG MPS paper benchmark scripts

This directory contains the scripts used to generate the uncompressed
dense-reference check and the 16-site BUG versus 2-TDVP comparisons in the
accompanying manuscript. The enclosing Git commit pins the benchmark scripts
and the exact YAQS implementation in one source tree. This branch is a paper
artifact; it is not intended to be merged into `main` or published as a YAQS
release.

Numerical result files are deliberately not included. Running the scripts
creates them locally in the corresponding benchmark directory.

## Contents

- `six_site_dense_reference_2026-08-17/run_benchmark.py` runs the asymmetric
  six-site, uncompressed dense-reference refinement study and records all
  ordering, reflection, endpoint, and input-preservation checks.
- `six_site_dense_reference_2026-08-17/validate_results.py` checks the complete
  saved output against the manuscript table and structural diagnostics.
- `l16_matched_optimized_2026-08-12/run_benchmark.py` runs the matched-parameter
  TFIM and Haldane--Shastry benchmarks.
- `l16_matched_optimized_2026-08-12/summarize.py` derives the compact summary
  table from `raw_results.json`.
- `l16_matched_optimized_2026-08-12/validate_results.py` validates the matched
  benchmark output.
- `l16_matched_optimized_2026-08-12/export_julia_hs_mpo.jl` exports the Julia
  Haldane--Shastry MPO for the optional cross-implementation tensor check.
- `l16_tradeoff_caps_2026-08-12/run_experiments.py` runs the runtime--accuracy
  and active-cap studies.
- `l16_tradeoff_caps_2026-08-12/validate_and_export.py` validates those runs and
  exports their derived tables.
- `l16_tradeoff_caps_2026-08-12/make_figure.py` creates the runtime--accuracy
  figure.
- `l16_tradeoff_caps_2026-08-12/build_workbook.mjs` creates the optional
  workbook export.

## Reproduction

Create the YAQS environment from the repository root:

```bash
uv sync
```

Run and validate the complete six-site dense-reference refinement table:

```bash
uv run python paper/bug-mps-benchmarks/six_site_dense_reference_2026-08-17/run_benchmark.py
uv run python paper/bug-mps-benchmarks/six_site_dense_reference_2026-08-17/validate_results.py
```

Run the complete matched-parameter benchmark and produce its table and
validation report:

```bash
uv run python paper/bug-mps-benchmarks/l16_matched_optimized_2026-08-12/run_benchmark.py
uv run python paper/bug-mps-benchmarks/l16_matched_optimized_2026-08-12/summarize.py
uv run python paper/bug-mps-benchmarks/l16_matched_optimized_2026-08-12/validate_results.py
```

Run the complete trade-off and bond-cap study from scratch, then validate it
and regenerate the figure:

```bash
uv run python paper/bug-mps-benchmarks/l16_tradeoff_caps_2026-08-12/run_experiments.py --stage all
uv run python paper/bug-mps-benchmarks/l16_tradeoff_caps_2026-08-12/validate_and_export.py
uv run --with matplotlib python paper/bug-mps-benchmarks/l16_tradeoff_caps_2026-08-12/make_figure.py
```

All runners expose smaller selectable grids through `--help`. The benchmark
configuration used Python 3.12.11, NumPy 2.4.6, and SciPy 1.18.0 on an
eight-core Apple M1 Pro. Construction, warm-up, padding, and reference-state
calculation are excluded from the reported timings by the runners.
