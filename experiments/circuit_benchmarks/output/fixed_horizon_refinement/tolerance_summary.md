# Fixed-horizon tolerance summary

This is a pure reanalysis of the retained `n=15` cap sweep. No simulations were run.
Each reported point is the **first-passing tested** cap, not a minimum over untested caps
or a globally optimal accuracy--cost point.

| $\epsilon$ | Method | Selection | $E_\star$ | $P_{\max}$ | Median runtime (s) | Previous tested failure |
|---:|:---|:---|---:|---:|---:|:---|
| 5e-3 | TDVP | **No pass on tested grid** (best: $\chi=32$) | 0.00735703 (best) | -- | -- | -- |
| 5e-3 | MPO | **No pass on tested grid** (best: $\chi=32$) | 0.00547103 (best) | -- | -- | -- |
| 5e-3 | TEBD+SWAP | **No pass on tested grid** (best: $\chi=32$) | 0.00935391 (best) | -- | -- | -- |
| 1e-2 | TDVP | $\chi=28$ (first-passing tested) | 0.00952417 | 11,880 | 19.7398 | $\chi=26$, $E_\star=0.0111032$ |
| 1e-2 | MPO | $\chi=26$ (first-passing tested) | 0.00864811 | 36,456 | 2.59617 | $\chi=24$, $E_\star=0.0103964$ |
| 1e-2 | TEBD+SWAP | $\chi=32$ (first-passing tested) | 0.00935391 | 15,016 | 2.2321 | $\chi=30$, $E_\star=0.0108425$ |
| 2e-2 | TDVP | $\chi=24$ (first-passing tested) | 0.012494 | 9,128 | 14.3772 | $\chi=16$, $E_\star=0.0301146$ |
| 2e-2 | MPO | $\chi=24$ (first-passing tested) | 0.0103964 | 31,400 | 2.42869 | $\chi=16$, $E_\star=0.0273462$ |
| 2e-2 | TEBD+SWAP | $\chi=24$ (first-passing tested) | 0.0171784 | 9,128 | 1.81657 | $\chi=16$, $E_\star=0.0374606$ |

`E_star` is the maximum normalized infidelity through step 15. `P_max` is the
peak recorded MPS tensor-entry count. Runtimes are medians over the retained
repeat count recorded in the CSV.

## Source provenance

- `combined_cap_sweep.csv` SHA-256: `054ed462772b32d43bfd8915081b905de4fd8cf3f6a4df6bed13f6753bc1ead5`
- `cap_timing_summary.csv` SHA-256: `0fa4453720180dd1539ff7060de7d6651904f3364810d2ce7703ad1e4b213404`
- Fixed tolerances: `5e-3, 1e-2, 2e-2`
