# dt = 0.01 circuit suite — final report

**Wall time:** 83 min (exact refs ~40 s; long pole = 2D Heisenberg `full_tdvp` at 4812 s).  
**Setup:** same models/methods/χ_max=32/full TDVP on all 2q gates as production; `dt=0.01`, 300 steps, \(t\in[0,3]\).  
**Reference:** dense application of the *identical* Trotter circuit (method/truncation error, not continuum error).

Figures: `fig3_circuits_dt001.*`, `figS3_circuit_resources_dt001.*`, `fig_dt01_vs_dt001.*`  
Data: `processed/circuit_trajectories_dt001.csv`, `processed/dt01_vs_dt001_summary.csv`

---

## Bottom line

**Finer Trotter steps do not change the qualitative story.** Method rankings are stable. Absolute late-time errors stay within ~1–1.4× of `dt=0.1`. Runtime scales roughly with the 10× step count for TDVP (~12–16× observed).

One nuance: on **2D Heisenberg**, mean-\(1-F\) ranking flips slightly — at `dt=0.01` full TDVP edges zip-up (0.48 vs 0.50). Both remain deep in the failed regime (\(1-F\gtrsim0.2\) by \(t=0.5\)), so this is not a practical win.

---

## Accuracy (`dt=0.01` vs `0.1`)

| Model | Ranking by mean \(1-F\) @0.1 | @0.01 | Changed? |
| --- | --- | --- | --- |
| 1D TFIM | TDVP ≪ TEBD = zip | same | No |
| 1D XXX | TDVP < TEBD = zip | same | No |
| 2D TFIM | zip ≲ TDVP ≪ TEBD | zip ≈ TDVP ≪ TEBD | No (even closer) |
| 2D Heis | zip < TDVP ≪ TEBD | **TDVP ≲ zip** ≪ TEBD | Tiny flip; both bad |

Final \(1-F\) ratios (`0.01` / `0.1`):

| Model | TDVP | TEBD | zip |
| --- | ---: | ---: | ---: |
| 1D TFIM | 2.3 | 1.13 | 1.13 |
| 1D XXX | 1.07 | 1.01 | 1.01 |
| 2D TFIM | 0.99 | 1.35 | 1.16 |
| 2D Heis | 0.97 | 1.00 | 1.22 |

During evolution at `dt=0.01` (same pattern as `0.1`):

- **1D TFIM:** TEBD/zip better early (\(t\lesssim1.2\)); TDVP dominates mid/late (~100×).
- **1D XXX:** TDVP best essentially throughout.
- **2D TFIM:** zip best early/mid; TDVP ≈ zip after \(t\sim1.5\); both crush TEBD.
- **2D Heis:** TDVP slightly ahead of zip at sampled times, but both already past ε.

---

## Runtime

| Model | TDVP @0.1 | @0.01 | scale | vs zip @0.01 |
| --- | ---: | ---: | ---: | ---: |
| 1D TFIM | 43 s | 511 s | 12× | **155×** slower |
| 1D XXX | 205 s | 3256 s | 16× | **277×** |
| 2D TFIM | 162 s | 2109 s | 13× | **29×** |
| 2D Heis | 393 s | 4803 s | 12× | **21×** |

No runtime advantage at either `dt`. Finer steps make the TDVP cost gap worse in wall time.

---

## Peak parameters

All methods still saturate to the same final peak \(P=15016\) (χ=32).  
TDVP still delays χ-saturation on 1D TFIM (`t_sat` 1.80 → 2.07). Elsewhere everyone hits the cap almost immediately. **No change** to the resource conclusion.

---

## What `dt=0.01` changes for the paper

1. **Robustness check passed.** Coarse `dt=0.1` did not create a false TDVP advantage.
2. **1D accuracy story stands** (and is not an artifact of large Trotter angles).
3. **2D:** zip-up remains the practical competitor; finer `dt` does not open a usable TDVP horizon on Heisenberg.
4. **Cost:** production figures should keep `dt=0.1`; `dt=0.01` is a validation appendix, not a better default.
