# Running the Unraveling vs. MPS Benchmarks

This guide explains how to reproduce the multi-circuit, multi-noise-strength
benchmark campaign that compares YAQS' trajectory unravelings with Qiskit's
matrix product state (MPS) simulator. The workflow mirrors the configuration
implemented in `combined_unraveling_mps_benchmark.py` and the downstream
aggregation in `analyze_unraveling_mps_results.py`.

## 1. Prerequisites

* Install the project dependencies via [`uv`](https://github.com/astral-sh/uv)
  (the repository already ships with an `uv.lock`). From the repository root:

  ```bash
  uv sync
  ```

* All benchmark inputs and outputs live under
  `src/mqt/yaqs/codex_experiments`. The scripts assume they are executed from
  the repository root so relative paths resolve correctly.

## 2. Run the benchmark sweep

Execute the combined benchmark script. It iterates over the pre-defined
`BenchmarkConfig` presets, evaluates each circuit across several noise strengths,
and stores one pickle per circuit/noise pair under
`src/mqt/yaqs/codex_experiments/results/`.

```bash
uv run python src/mqt/yaqs/codex_experiments/experiments/combined_unraveling_mps_benchmark.py
```

The script prints progress for each circuit/noise configuration and persists the
following data per pickle file:

* expectation-value trajectories for the exact solver, Qiskit MPS, and the
  four YAQS unraveling variants (standard TJM, projector, unitary 2-point, and
  unitary Gaussian),
* average bond dimensions for each method, and
* trajectory variances and absolute errors versus the exact reference.

## 3. Aggregate and compare the results

After the benchmark sweep finishes, run the analysis helper to load all pickles,
compute summary statistics, and highlight the cases where YAQS' standard TJM
unraveling matches the Qiskit MPS accuracy while using a lower bond dimension.

```bash
uv run python src/mqt/yaqs/codex_experiments/experiments/analyze_unraveling_mps_results.py
```

The analysis script writes its aggregated table to the console and saves an
`analysis_summary.pkl` artifact next to the individual benchmark pickles. This
file can be reused when generating publication-ready plots.

## 4. Next steps

You can inspect the pickled dictionaries directly (for example with a short
Python REPL session) to develop custom plots or filtering logic. Each payload is
keyed by method label and contains both the raw trajectory data and the computed
statistics that were used to surface the TJM-vs-MPS comparisons.
