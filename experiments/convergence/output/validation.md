# TFIM TDVP substep convergence audit

## Configuration
- Benchmark hash: `60c5f6e09b155e7b`
- Identical 4×4 TFIM Strange circuit / gate_runtime TDVP as fixed_resources & resource_frontier
- χmax ∈ [16, 32, 64]; substeps run: [1, 2, 4, 8, 16]
- Δt=0.1 through t=1.5 (15 steps); ε=10⁻²
- Physical gate fixed: n substeps via existing `tdvp_sweeps` (angle τ/n each)
- Probe χ for ladder extension: 32

## nε by (χ, n)

- χ=16, n=1: nε=9
- χ=16, n=2: nε=9
- χ=16, n=4: nε=9
- χ=16, n=8: nε=9
- χ=16, n=16: nε=9
- χ=32, n=1: nε=12
- χ=32, n=2: nε=13
- χ=32, n=4: nε=13
- χ=32, n=8: nε=13
- χ=32, n=16: nε=13
- χ=64, n=1: nε=15
- χ=64, n=2: nε=15
- χ=64, n=4: nε=15
- χ=64, n=8: nε=15
- χ=64, n=16: nε=15

## Convergence tests

- χ=16 n2_vs_n4: converged=0 — infidelity disagree at step 8: 2.983270e-03 vs 2.819013e-03
- χ=16 n4_vs_n8: converged=0 — infidelity disagree at step 3: 4.294433e-04 vs 5.528999e-04
- χ=16 n8_vs_n16: converged=0 — infidelity disagree at step 7: 2.489850e-03 vs 2.059919e-03
- χ=16 n1_vs_reference: converged=0 — No converged reference (infidelity disagree at step 7: 2.489850e-03 vs 2.059919e-03) [FAIL]
- χ=32 n2_vs_n4: converged=0 — infidelity disagree at step 10: 1.821203e-03 vs 1.668873e-03
- χ=32 n4_vs_n8: converged=0 — infidelity disagree at step 3: 4.295207e-04 vs 5.528172e-04
- χ=32 n8_vs_n16: converged=1 — converged (nε=13)
- χ=32 n1_vs_n8: converged=0 — FAIL: nε mismatch: 12 vs 13 [FAIL]
- χ=64 n2_vs_n4: converged=0 — infidelity disagree at step 12: 1.393175e-03 vs 1.286001e-03
- χ=64 n4_vs_n8: converged=0 — infidelity disagree at step 3: 4.295207e-04 vs 5.528172e-04
- χ=64 n8_vs_n16: converged=0 — infidelity disagree at step 10: 6.699991e-04 vs 7.720098e-04
- χ=64 n1_vs_reference: converged=0 — No converged reference (infidelity disagree at step 10: 6.699991e-04 vs 7.720098e-04) [FAIL]

## Substep difference D_n(t)=|I_n(t)−I_2n(t)|

See `tfim_tdvp_substeps_D.csv` and `tfim_tdvp_substeps_D_ratios.csv` (ratio label `D8_over_D4`).
- χ=16 D₈/D₄ on n=8 vs n=16 window: median=0.6028474229567196, mean=0.9731496904432786, max=4.799954836095086, fraction<1=0.8, decreasing=True
- χ=16 D₈/D₄ over all steps: median=0.7971024434063269, mean=3.2782127469317546, decreasing=True
- χ=32 D₈/D₄ on n=8 vs n=16 window: median=0.6732801729169762, mean=1.3130218569520296, max=7.550865209900928, fraction<1=0.7857142857142857, decreasing=True
- χ=32 D₈/D₄ over all steps: median=0.6732801729169762, mean=1.2088255737867937, decreasing=True
- χ=64 D₈/D₄ on n=8 vs n=16 window: median=0.9606323311540305, mean=2.099639316455994, max=5.150867025653262, fraction<1=0.5333333333333333, decreasing=True
- χ=64 D₈/D₄ over all steps: median=0.8588283853786118, mean=1.9684118591774942, decreasing=True

## Verdict on production n=1

**FAIL:** n=1 does not agree with a converged reference for at least one χ.
- Per-χ n=1 PASS: {16: False, 32: False, 64: False}
- Converged reference n per χ: {16: None, 32: 8, 64: None}
- Smallest substep count converged for all three bond dimensions: None
- Convergence-selected production substeps (n=8 only if n=8 vs n=16 for **all** χ): None

### Ladder-extension outcome
- Probe χ=32: n=8 vs n=16 PASS; D₈/D₄ median < 1 → continued to other χ (no implementation halt).
- χ=16 and/or χ=64 did **not** all pass n=8 vs n=16 under the preregistered criteria, so n=8 is **not** yet the all-χ production count.
- Per protocol, n=32 is added only when the **probe** χ fails n=8 vs n=16; that did not occur, so n=32 was not run.
- fixed_resources / resource_frontier were **not** regenerated.

Do **not** automatically regenerate other experiments. If adopting a convergence-selected production count once all χ agree, replace:
- All hybrid_tdvp TFIM runs in `experiments/fixed_resources`.
- All hybrid_tdvp TFIM runs in `experiments/resource_frontier` (raw_runs, TDVP Pmax frontier points, three-repeat TDVP timings) and recompute medians/IQR.

## Outputs
- `tfim_tdvp_substeps.csv`
- `tfim_tdvp_substeps_summary.csv`
- `tfim_tdvp_substeps_D.csv`
- `tfim_tdvp_substeps_D_ratios.csv`
- `tfim_tdvp_substeps.pdf` / `.png`
- `validation.md`
