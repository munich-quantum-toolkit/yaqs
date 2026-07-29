# SVD-cutoff diagnostic report

## Cutoff semantics

- Production mode: **`discarded_weight`**.
- τ meaning: Cumulative discarded squared singular-value weight; retained_rank=min(keep_cutoff, chi_max), min_keep=1.
- Gate-library `split_tensor` hard cutoff held fixed at `1e-14`.
- Corrected benchmark used `svd_threshold=1e-13`; diagnostic reference τ=1e-14 was **re-run** (not reused).
- Krylov tol, TDVP n=2, χmax, Δt, and circuit held fixed.

## Summary table (ε=1e-2)

| method | χmax | τ | Tε | peak χ | peak params | runtime [s] | Σ disc. wt | f_cut | f_χ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hybrid_tdvp | 16 | 1e-14 | 0.90 | 16 | 4776 | — | — | 0.26 | 0.40 |
| hybrid_tdvp | 16 | 1e-12 | 0.90 | 16 | 4776 | — | — | 0.28 | 0.38 |
| hybrid_tdvp | 16 | 1e-09 | 0.90 | 16 | 4776 | — | — | 0.33 | 0.35 |
| hybrid_tdvp | 16 | 1e-06 | 0.30 | 16 | 4776 | — | — | 0.75 | 0.04 |
| hybrid_tdvp | 16 | 1e-03 | 0.30 | 3 | 162 | — | — | 0.95 | 0.00 |
| hybrid_tdvp | 32 | 1e-14 | 1.20 | 32 | 15016 | — | — | 0.26 | 0.27 |
| hybrid_tdvp | 32 | 1e-12 | 1.30 | 32 | 15016 | — | — | 0.28 | 0.25 |
| hybrid_tdvp | 32 | 1e-09 | 1.20 | 32 | 15016 | — | — | 0.35 | 0.22 |
| hybrid_tdvp | 32 | 1e-06 | 0.30 | 32 | 15016 | — | — | 0.77 | 0.01 |
| hybrid_tdvp | 32 | 1e-03 | 0.30 | 3 | 162 | — | — | 0.95 | 0.00 |
| mpo_zipup | 16 | 1e-14 | 1.00 | 16 | 4776 | — | — | 0.37 | 0.18 |
| mpo_zipup | 16 | 1e-12 | 1.00 | 16 | 4776 | — | — | 0.38 | 0.18 |
| mpo_zipup | 16 | 1e-09 | 1.00 | 16 | 4776 | — | — | 0.40 | 0.17 |
| mpo_zipup | 16 | 1e-06 | 1.10 | 16 | 4776 | — | — | 0.46 | 0.14 |
| mpo_zipup | 16 | 1e-03 | 0.20 | 2 | 96 | — | — | 0.94 | 0.00 |
| mpo_zipup | 32 | 1e-14 | 1.50 | 32 | 15016 | — | — | 0.33 | 0.12 |
| mpo_zipup | 32 | 1e-12 | 1.50 | 32 | 15016 | — | — | 0.35 | 0.12 |
| mpo_zipup | 32 | 1e-09 | 1.50 | 32 | 15016 | — | — | 0.38 | 0.11 |
| mpo_zipup | 32 | 1e-06 | 1.50 | 32 | 15016 | — | — | 0.51 | 0.09 |
| mpo_zipup | 32 | 1e-03 | 0.20 | 2 | 96 | — | — | 0.94 | 0.00 |
| tebd_swap | 16 | 1e-14 | 0.20 | 16 | 4776 | — | — | 0.18 | 0.26 |
| tebd_swap | 16 | 1e-12 | 0.20 | 16 | 4776 | — | — | 0.18 | 0.26 |
| tebd_swap | 16 | 1e-09 | 0.20 | 16 | 4776 | — | — | 0.18 | 0.26 |
| tebd_swap | 16 | 1e-06 | 0.20 | 16 | 4776 | — | — | 0.18 | 0.26 |
| tebd_swap | 16 | 1e-03 | 0.20 | 16 | 4530 | — | — | 0.35 | 0.19 |
| tebd_swap | 32 | 1e-14 | 0.20 | 32 | 15016 | — | — | 0.18 | 0.18 |
| tebd_swap | 32 | 1e-12 | 0.20 | 32 | 15016 | — | — | 0.18 | 0.18 |
| tebd_swap | 32 | 1e-09 | 0.20 | 32 | 15016 | — | — | 0.18 | 0.18 |
| tebd_swap | 32 | 1e-06 | 0.20 | 32 | 15016 | — | — | 0.19 | 0.18 |
| tebd_swap | 32 | 1e-03 | 0.20 | 32 | 14088 | — | — | 0.37 | 0.13 |

## Interpretation

### Are 1e-12 and 1e-9 numerically indistinguishable from 1e-14?

**Yes** for this TFIM scope: horizons and peak parameter counts match the τ=1e-14 reference through τ=1e-9 (within ≤ one sampled step in Tε and exact param-count equality).

### At what τ does the cutoff begin reducing actual bond dimensions?

- TDVP χ=16: first reduction at τ=0.001.
- TDVP χ=32: first reduction at τ=0.001.
- TEBD+SWAP χ=16: no reduction vs τ=1e-14 on this grid.
- TEBD+SWAP χ=32: no reduction vs τ=1e-14 on this grid.
- MPO zip-up χ=16: first reduction at τ=0.001.
- MPO zip-up χ=32: first reduction at τ=0.001.

### Does reduced memory compensate for any loss of horizon?

- TDVP χ=32: τ=1e-14 → Tε=1.20, params=15016; τ=1e-3 → Tε=0.30, params=162 (ΔTε=-0.90, params×0.01).
- MPO zip-up χ=32: τ=1e-14 → Tε=1.50, params=15016; τ=1e-3 → Tε=0.20, params=96 (ΔTε=-1.30, params×0.01).
If Tε drops while params shrink, memory savings do **not** compensate for horizon loss under a fixed-ε reliability criterion.

### Does the TDVP-versus-zip-up ranking change (also by parameter count)?

- τ=1e-14: MPO zip-up(1.50) > TDVP(1.20) > TEBD+SWAP(0.20)
- τ=1e-12: MPO zip-up(1.50) > TDVP(1.30) > TEBD+SWAP(0.20)
- τ=1e-09: MPO zip-up(1.50) > TDVP(1.20) > TEBD+SWAP(0.20)
- τ=1e-06: MPO zip-up(1.50) > TDVP(0.30) > TEBD+SWAP(0.20)
- τ=0.001: TDVP(0.30) > TEBD+SWAP(0.20) > MPO zip-up(0.20)

- τ=1e-14, χ=32 by params: TDVP params=15016 (Tε=1.20), zip-up params=15016 (Tε=1.50).
- τ=1e-12, χ=32 by params: TDVP params=15016 (Tε=1.30), zip-up params=15016 (Tε=1.50).
- τ=1e-09, χ=32 by params: TDVP params=15016 (Tε=1.20), zip-up params=15016 (Tε=1.50).
- τ=1e-06, χ=32 by params: TDVP params=15016 (Tε=0.30), zip-up params=15016 (Tε=1.50).
- τ=0.001, χ=32 by params: TDVP params=162 (Tε=0.30), zip-up params=96 (Tε=0.20).

### Are runs primarily χ-limited or cutoff-limited?

- TDVP: mean fraction cutoff-limited=0.52, χ-limited=0.19.
- TEBD+SWAP: mean fraction cutoff-limited=0.22, χ-limited=0.21.
- MPO zip-up: mean fraction cutoff-limited=0.51, χ-limited=0.11.

## Stopping rule

Results are unchanged through τ=1e-9 and only change at looser thresholds (1e-6 / 1e-3). **Stop here**; do not search for a τ that restores a TDVP advantage.

## Outputs

- `svd_cutoff_trajectories.csv`
- `svd_truncation_events.csv`
- `svd_cutoff_summary.csv`
- `representative_spectra.npz`
- `figure_svd_cutoff_diagnostic.{pdf,png}`
- `cutoff_semantics.json`

