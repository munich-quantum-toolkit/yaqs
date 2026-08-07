# Fixed-horizon tolerance summary

This is a pure reanalysis of the retained `n=15` cap sweep. No simulations were run.
Each reported point is the **first-passing tested** cap, not a minimum over untested caps
or a globally optimal accuracy--cost point.

| $\epsilon$ | Method | Selection | $E_\star$ | $P_{\max}$ | Median runtime (s) | Previous tested failure |
|---:|:---|:---|---:|---:|---:|:---|
| 5e-3 | TDVP | **No pass on tested grid** (best: $\chi=32$) | 0.00735703 (best) | -- | -- | -- |
| 5e-3 | MPO | $\chi=128$ (first-passing tested) | 0.00490215 | 430,760 | 15.0944 | $\chi=112$, $E_\star=0.00688865$ |
| 5e-3 | TEBD+SWAP | $\chi=192$ (first-passing tested) | 0.00440464 | 141,992 | 12.8141 | $\chi=176$, $E_\star=0.0208594$ |
| 1e-2 | TDVP | $\chi=28$ (first-passing tested) | 0.00952417 | 11,880 | 21.5427 | $\chi=26$, $E_\star=0.0111032$ |
| 1e-2 | MPO | $\chi=96$ (first-passing tested) | 0.00930666 | 283,304 | 10.5545 | $\chi=80$, $E_\star=0.0136913$ |
| 1e-2 | TEBD+SWAP | $\chi=192$ (first-passing tested) | 0.00440464 | 141,992 | 12.8141 | $\chi=176$, $E_\star=0.0208594$ |
| 2e-2 | TDVP | $\chi=24$ (first-passing tested) | 0.012494 | 9,128 | 14.7193 | $\chi=16$, $E_\star=0.0301146$ |
| 2e-2 | MPO | $\chi=80$ (first-passing tested) | 0.0136913 | 221,864 | 8.59794 | $\chi=64$, $E_\star=0.0444647$ |
| 2e-2 | TEBD+SWAP | $\chi=192$ (first-passing tested) | 0.00440464 | 141,992 | 12.8141 | $\chi=176$, $E_\star=0.0208594$ |

`E_star` is the maximum normalized infidelity through step 15. `P_max` is the
peak recorded MPS tensor-entry count. Runtimes are medians over the retained
repeat count recorded in the CSV.

## Source provenance

- `combined_cap_sweep.csv` SHA-256: `a3510a5d152da66791a2dfb2d56b1f3c77ac29c781c0e26d3778574fa583990c`
- `cap_timing_summary.csv` SHA-256: `43e926a5f6e862f831b81bf0c1c54d93abd17b3105c64ce663fc180c49760150`
- Fixed tolerances: `5e-3, 1e-2, 2e-2`
