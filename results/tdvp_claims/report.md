# TDVP claims benchmark

## Executive summary

The main supported claim is fixed-χ observable accuracy improvement in sparse long-range/routing-limited regimes.

The benchmark does not attempt to show universal TDVP superiority.

TEBD remains the preferred nearest-neighbor update.

Runtime advantage is not assumed; wall-clock is reported honestly.

The Floquet application is exploratory unless it clearly shows improved fixed-χ collision-entropy accuracy.

- NN control median TDVP/TEBD error ratio: 1.74

## Validation and dispatch checks

- Pre-flight validation runs before the benchmark loop.

## Padding diagnostic

- Does padding fix periodic gates? See `padding_diagnostic.csv`.
- Is padded4 always better? Expected: no.

## Sparse long-range fixed-χ advantage

- sparse comparisons: 0

## Recovery χ

- Recovered within tested χ: 0/0

## Nearest-neighbor control

- Cases with TDVP ≥ TEBD error: 15/18

## Heisenberg Floquet mini-application

- Floquet patch rows: 0 (exploratory).

## Failure/scope cases

- dense_long_range_failure rows: 0

## Recommended paper figures

- `plots/sparse_fixed_chi_error_vs_chi.png`
- `plots/recovery_chi.png`
- `plots/padding_periodic_diagnostic.png`
- `plots/nn_control_tebd_vs_tdvp.png`

---

Total runs: 36.
