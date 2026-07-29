# Circuit suite at χ_max = 8, dt = 0.1 — report

**Wall time:** ~48 s for all 12 trajectories.  
**Setup:** same models/methods/schedules as production (`dt=0.1`, 30 steps, full TDVP on all 2q gates), bond cap lowered from 32 → **8**. Exact references reused unchanged.

Data: `raw_new/circuits_chi8/`, `processed/circuit_trajectories_chi8.csv`, `processed/chi8_vs_chi32_summary.csv`  
Figures: `fig3_circuits_chi8.*`, `figS3_circuit_resources_chi8.*`, `fig_chi8_vs_chi32.*`

---

## Bottom line

Lowering the cap makes **everyone worse**, as expected. Method **rankings are mostly unchanged**:

- **1D:** TDVP still best (vs TEBD = zip).
- **2D TFIM:** zip ≲ TDVP ≪ TEBD (same order as χ=32).
- **2D Heisenberg:** at χ=8, TDVP slightly edges zip on *mean* \(1-F\) (both already failed); at χ=32 zip was clearly better. Not a usable regime either way.

Absolute errors jump by ~1–3 orders of magnitude vs χ=32 on the milder models.

---

## Accuracy

| Model | Best @χ=8 (mean \(1-F\)) | Best @χ=32 | Final \(1-F\) TDVP χ=8 → 32 |
| --- | --- | --- | ---: |
| 1D TFIM | **TDVP** (0.013) | TDVP | 0.16 → \(6\times10^{-5}\) |
| 1D XXX | **TDVP** (0.65) | TDVP | 0.99 → 0.59 |
| 2D TFIM | zip (0.124) ≲ TDVP (0.129) | zip ≲ TDVP | 0.28 → 0.054 |
| 2D Heis | TDVP (0.83) ≲ zip (0.85) | zip < TDVP | 0.96 → 0.73 |

ε-horizons shrink under the tighter cap (e.g. 1D TFIM TDVP: never crossed @32 → crosses at \(t=2.6\) @8; 2D TFIM: \(1.6\) → \(0.7\)).

During evolution at χ=8, 1D TFIM still shows the delayed crossover: TEBD/zip better only at the earliest steps, then TDVP leads.

---

## Runtime / peak parameters

- **Peak \(P\):** all methods hit the same final peak (1448 at χ=8; 15016 at χ=32). No differential resource win.
- **Runtime:** TDVP still much slower (≈14–34× vs TEBD/zip at χ=8), but absolute times are small (TFIM TDVP ~6–16 s; 2D Heis TDVP ~45 s).

---

## Takeaway

χ=8 is a harsher fixed-resource probe, not a different methods story. TDVP’s 1D accuracy advantage survives the lower cap; in 2D it remains competitive with zip and far ahead of TEBD, without becoming the clear winner.
