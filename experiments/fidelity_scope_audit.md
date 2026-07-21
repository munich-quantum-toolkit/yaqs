# Fidelity definition scope audit

Audit of how state fidelity / infidelity is computed across YAQS experiment
pipelines, following the single-gate correction to normalized state fidelity

\[
F_{\mathrm{norm}}=\frac{|\langle\psi_{\mathrm{exact}}|\psi_{\mathrm{approx}}\rangle|^2}
{\langle\psi_{\mathrm{exact}}|\psi_{\mathrm{exact}}\rangle\,
\langle\psi_{\mathrm{approx}}|\psi_{\mathrm{approx}}\rangle}.
\]

This document does **not** authorize or perform reruns of expensive benchmarks.

## Shared YAQS library (`src/mqt/yaqs`)

- No shared helper computes MPS/statevector fidelity as
  `|⟨ψ|φ⟩|²` without norms for digital-gate benchmarks.
- Related APIs (`compute_identity_fidelity` on operators / equivalence checker)
  measure operator–identity overlap, not approximate-state fidelity.
- **Action:** none for this bug; the corrected helper lives in
  `experiments/single_gate/gate_runtime.py`
  (`normalized_state_fidelity`).

## `experiments/single_gate`

| Question | Finding |
| --- | --- |
| Were approximate states already normalized before fidelity? | **No.** MPS outputs were left as produced; `norm_after` recorded the Euclidean L2 norm `‖ψ‖`. |
| Was the incorrect raw `|⟨e\|a⟩|²` metric used? | **Yes** (pre-correction). Canonical `fidelity` / `infidelity` columns stored raw overlap squared and `1 −` that. |
| Do stored norms permit post-processing? | **Yes.** `norm_before` / `norm_after` are `‖ψ‖`, and raw overlap squared was retained as `overlap_squared_raw` after migration. Backfill: `F_norm = overlap_squared_raw / (norm_before² · norm_after²)`. All 469 rows migrated without resimulation. |
| Could reliability horizons / resource frontiers change? | N/A for this experiment (no reliability horizon). Selected `χ_max` **unchanged** (8 / 12 / 16). Absolute TEBD/MPO infidelities at compressed χ drop substantially (max `|I_raw − I_norm| ≈ 0.263`); method **ordering** at fixed χ is unchanged at the plotted angles. Figure and CSVs regenerated under `fidelity_definition=normalized_state_fidelity_v2`. |

## `experiments/fixed_resources`

| Question | Finding |
| --- | --- |
| Were approximate states already normalized before fidelity? | **Yes, for evaluation only.** `trajectory.compute_metrics` divides exact and approx vectors by their L2 norms before `\|⟨·\|·⟩\|²`. The MPS itself is not renormalized in the simulation path. |
| Was the incorrect raw metric used? | **No.** Stored `fidelity` / `infidelity` are already equivalent to `F_norm` / `I_norm`. |
| Do stored norms permit post-processing? | `state_norm` (= `‖approx‖`) is stored; exact-state norm is not always persisted, but correction is unnecessary because evaluation already normalizes both sides. Observed `state_norm` range in `trajectories.csv` includes strong loss (down to ~0.06), so a raw-overlap metric would have been badly wrong—but that metric was not used. |
| Could reliability horizons change? | **Unlikely from a fidelity-definition bug**—horizons already use the normalized metric. Re-running is not indicated for this issue. |

## `experiments/resource_frontier`

| Question | Finding |
| --- | --- |
| Were approximate states already normalized before fidelity? | **Yes (evaluation).** Workers import `compute_metrics` from `fixed_resources/trajectory.py` (via path setup) and use the same normalized overlap. |
| Was the incorrect raw metric used? | **No.** |
| Do stored norms permit post-processing? | `state_norm` is stored (observed range ~0.97–1.00 in `raw_runs.csv`). Post-processing not required. |
| Could resource frontiers / reliability crossings change? | **Not from this definition bug.** Frontiers already sit on normalized infidelity. No automatic rerun. |

## `experiments/convergence`

| Question | Finding |
| --- | --- |
| Were approximate states already normalized before fidelity? | **Yes (evaluation).** Trajectory metrics use `fixed_resources.trajectory.compute_metrics`. Pairwise analysis (`analyze.pairwise_state_infidelity`) also normalizes both vectors before overlap. |
| Was the incorrect raw metric used? | **No** for stored step infidelity or pairwise checks. |
| Do stored norms permit post-processing? | `state_norm` stored; TDVP runs stay near unit norm (~1±1e-6 in `tfim_tdvp_substeps.csv`), so raw vs normalized would nearly coincide anyway. |
| Could reliability / convergence conclusions change? | **Not from this definition bug.** Existing convergence failures are about Trotter-substep disagreement, not norm-loss mis-definition. No automatic rerun. |

## Summary

| Experiment | Incorrect raw metric? | Already normalized at eval? | Post-process possible? | Rerun recommended for this bug? |
| --- | --- | --- | --- | --- |
| `single_gate` | Yes (fixed) | No | Yes (done) | Done (backfill + figure) |
| `fixed_resources` | No | Yes | N/A | No |
| `resource_frontier` | No | Yes | N/A | No |
| `convergence` | No | Yes | N/A | No |
| Shared YAQS fidelity helper | N/A | N/A | N/A | No |

Scientific conclusions of the single-gate main-text figure are **quantitatively** affected for compressed-χ TEBD/MPO (lower absolute infidelity) but **not** overturned: χ selection, TDVP small-angle scaling at χ=8, and full-χ exactness of non-TDVP methods are unchanged. Other expensive experiments were therefore **not** regenerated.
