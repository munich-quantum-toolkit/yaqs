# TDVP claims benchmark

## Executive summary

The main supported claim is fixed-χ observable accuracy improvement in sparse long-range/routing-limited regimes.

The benchmark does not attempt to show universal TDVP superiority.

TEBD remains the preferred nearest-neighbor update.

Runtime advantage is not assumed; wall-clock is reported honestly.

The Floquet application is exploratory unless it clearly shows improved fixed-χ collision-entropy accuracy.

- Fixed-χ advantage subset: 412 cases; TDVP wins (2× error) in 284.
- Padding: required=235, helpful=9 over 468 pairs.
- NN control median TDVP/TEBD error ratio: 8.11

## Validation and dispatch checks

- Pre-flight validation runs before the benchmark loop.

## Padding diagnostic

- periodic_1d padding pairs: 234
- Does padding fix periodic gates? See `padding_diagnostic.csv`.
- Is padded4 always better? Expected: no.

## Sparse long-range fixed-χ advantage

- sparse comparisons: 468

## Recovery χ

- Recovered within tested χ: 450/592

## Nearest-neighbor control

- Cases with TDVP ≥ TEBD error: 209/234

## Heisenberg Floquet mini-application

- Floquet patch rows: 648 (exploratory).

## Failure/scope cases

- dense_long_range_failure rows: 0

## Recommended paper figures

- `plots/sparse_fixed_chi_error_vs_chi.png`
- `plots/recovery_chi.png`
- `plots/padding_periodic_diagnostic.png`
- `plots/nn_control_tebd_vs_tdvp.png`

---

Total runs: 2545.
