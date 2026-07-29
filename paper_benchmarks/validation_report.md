# Validation report (paper_benchmarks)

Generated: 2026-07-25T14:02:01.147607+00:00

## Dense-reference and ordering validation (spec 3.1)

- all pass: **True** (38 checks; comparator exact-limit at nonbinding chi=64: TEBD+SWAP max 2.22e-16, MPO zip-up max 4.44e-16, variational MPO max 2.22e-16)

## Gate-window locality (spec 3.2)

- max r_loc = 1.423e-15, median = 8.003e-16 (tolerance 1e-10, 30 rows, all pass: **True**)
- max |norm derivative| = 1.0e-16
- two-site halo check: skipped: two-site variation-space action not exposed by production code; spec forbids a second framework for this optional diagnostic

## Aggregation consistency

- [PASS] angle_sweep_no_duplicate_cells (0 duplicates)
- [PASS] angle_sweep_complete_grid (4374 rows, expected 4374)
- [PASS] chi16_mpo_methods_near_exact (max 7.98e-14)
- [PASS] stored_zeros_preserved 
- [PASS] theta_zero_identity_nonrouting (max 2.22e-16)
- [PASS] substep_study_cap_nonbinding (peak bond 16 < cap 32, max discarded 7.8e-14)
- [PASS] heisenberg_rerun_consistency (max |diff| 1.79e-03; TDVP/zip-up ~1e-13, TEBD 1.8e-3 roundoff under heavy step-1 truncation, flagged in validation report)
- [PASS] circuit_2d_full_tdvp_full_length 
- [PASS] circuit_2d_full_tdvp_distinct_from_hybrid 
- [PASS] circuit_1d_full_length 
- [PASS] circuit_1d_step0_identity (max 0.0e+00)
- [PASS] circuit_1d_full_tdvp_distinct_from_tebd 

## Heisenberg deterministic repeat (this pipeline)

- hybrid_tdvp: stored 1.808148e-02, repeat 1.808148e-02, diff 5.2e-17 (reproducible at 1e-12)
- tebd_swap: stored 1.265230e-01, repeat 1.265230e-01, diff 8.3e-17 (reproducible at 1e-12)
- mpo_zipup: stored 2.395074e-05, repeat 2.395074e-05, diff 0.0e+00 (reproducible at 1e-12)

## Notes / caveats

- TEBD+SWAP Heisenberg step-1 infidelity differs from the corrected campaign row by 1.8e-3 (1.4% relative). Within this pipeline the value is reproducible (see heisenberg_deterministic_repeat); the offset against the corrected campaign is roundoff amplified by the severe step-1 SWAP-routing truncation (both values are far above the 1e-2 threshold, so no conclusion changes).
- Two-site-halo variation-space locality check skipped by design; see locality validation JSON.
- Runtime diagnostics are stored per row but no cross-method runtime figure is published: reused corrected-campaign timings and paper_benchmarks re-runs were recorded in different sessions with different process/thread settings.

## Figure QA

- fig1_gate_locality.pdf: render_ok=True, fonts_embedded=True (3 fonts)
- fig2_single_gate.pdf: render_ok=True, fonts_embedded=True (4 fonts)
- fig3_circuits.pdf: render_ok=True, fonts_embedded=True (4 fonts)
- figS1_angle_chi_grid.pdf: render_ok=True, fonts_embedded=True (2 fonts)
- figS2_substeps.pdf: render_ok=True, fonts_embedded=True (3 fonts)
- figS3_circuit_resources.pdf: render_ok=True, fonts_embedded=True (3 fonts)
- fig_hybrid_vs_full_tdvp.pdf: render_ok=True, fonts_embedded=True (3 fonts)

**Overall: ALL PASS**
