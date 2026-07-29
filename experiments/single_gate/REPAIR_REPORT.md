# Single-gate repair report (corrected vs archived claims)

Archive: `experiments/single_gate/archive/pre_repair_20260723T135839Z/`  
Protocol meta: `compress_rightcanon_ltr+var_multistart+tdvp_n1_v1`  
Main TDVP curves use **fractional-time substeps** `n=1` (each substep = one symmetric LTR+RTL 2-site sweep).

## Repairs

1. **`MPS.compress`**: right-canonicalize with QR (no χ cap), then LTR truncated SVD; restore center. Fixes the non-canonical product-state truncation that produced the θ-independent ~5×10⁻² plateau.
2. **`split_tensor` hard cutoff**: `1e-6 → 1e-14` so tiny-angle RZZ MPOs do not spuriously collapse to bond-1.
3. **Variational MPO**: multi-start (input / corrected zip-up); projected two-site updates with compatible virtual dims; no silent `except`; residual non-increasing; χ=16 must be exact.

Regression: `tests/experiments/test_single_gate_repairs.py` (11 passed).

## Old vs corrected numerical claims (seed 11)

| Claim / quantity | Old (archived) | Corrected | Verdict |
|---|---:|---:|---|
| χ=8, x=10⁻⁴, MPO zip-up | 5.40×10⁻² | 6.25×10⁻⁸ | Plateau was compress artifact |
| χ=8, x=10⁻⁴, variational MPO | 5.40×10⁻² | 6.25×10⁻⁸ | Was zip-up no-op |
| χ=8, x=10⁻⁴, TDVP | 6.25×10⁻⁸ (n=64) | 6.25×10⁻⁸ (n=1) | Unchanged at weak θ |
| χ=8, x=10⁻⁴, no-update | (not plotted) | 9.87×10⁻⁸ | Analytic identity holds |
| χ=8, x=10⁻⁴, TT-SVD candidate | — | 6.25×10⁻⁸ | Matches repaired zip-up |
| χ=8, x=0.1, TDVP | 3.28×10⁻² (n=64) | 1.95×10⁻² (n=1) | n=1 better under compression |
| χ=8, x=0.1, zip-up / variational | 5.40×10⁻² | 1.55×10⁻² | Beats TDVP n=1 |
| χ=16, x=0.1, non-TDVP | ~0 | ~0 | Still exact |
| χ=16, x=0.1, TDVP n=1 | — | ~10⁻¹⁶ | Exact at full χ |
| χ=16 substeps≥16 | jump to ~10⁻¹⁰ | same | Accumulated Krylov/projector noise |

## Defensible conclusions (from corrected raw data)

- One-substep local TDVP is a stable compressed weak-gate update and **improves over omitting the gate** (beats no-update on the audited angles).
- At infinitesimal angles under χ=8, TDVP, repaired zip-up, variational MPO, and the independent TT-SVD candidate **agree** (~6×10⁻⁸ at x=10⁻⁴).
- Under active compression, **increasing TDVP subdivision is not beneficial** (e.g. χ=8, x=0.1: n=1 → 1.95×10⁻² vs n=64 → 3.28×10⁻²; at x=1 the gap is larger).
- After repair, **variational MPO / zip-up can outperform TDVP** at moderate angles (χ=8, x=0.1). Do **not** claim TDVP beats properly optimized variational compression.
- TEBD+SWAP remains dominated by SWAP-routing truncation at χ<16 (already visible at θ=0).

## Terminology

`tdvp_sweeps` / plotted “TDVP substeps **n**” = number of **equal fractional-time substeps**. Each substep runs one full symmetric LTR+RTL 2-site sweep at time `1/n`.

## Circuit baseline

Do **not** flip all circuit benchmarks to n=1 from this single-gate result alone. See
`output/circuit_subdivision_check/circuit_subdivision_check.md` and
`affected_benchmark_inventory.md`. Prior Heisenberg evidence suggests n=1 may not
be a valid universal circuit setting.

## Figures

- Main: `experiments/figures/figure_single_gate_main_text.{pdf,png}` — TDVP n=1, repaired TEBD/MPO/variational, no-update baseline (grey dotted); panel (d) subdivision at x=0.1.
- TT-SVD comparison retained in CSV as `ttsvd_candidate` (Supplementary / diagnostic; not overcrowding the main figure).
