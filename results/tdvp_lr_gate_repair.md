# TDVP LR Pauli gate repair diagnostic

## Summary

| circuit | target | plain TDVP | enrichment | best repair fid | strict pass |
|---|---|---:|---:|---:|---|
| `ising2d_3x3_h1_dt0.1_L4_plus` | h1_L4_plus_rzz_4_7 | 9.649e-03 | 9.166e-13 | 1.704e-04 | False |
| `ising2d_3x3_h0.5_dt0.1_L4_plus` | h05_L4_plus_rzz_2_5 | 7.530e-08 | 2.268e-12 | 7.056e-08 | False |
| `ising2d_3x3_h1_dt0.1_L4_random_product` | random_rzz_4_7 | 8.393e-03 | 9.166e-13 | 2.144e-04 | False |
| `ising2d_3x3_h2_dt0.1_L2_random_product` | h2_L2_random_rzz_4_7 | 8.497e-03 | 1.787e-12 | 2.026e-04 | False |
| `ising2d_3x3_h0.5_dt0.1_L2_all_zero` | control_rzz_2_5 | 1.008e-08 | 9.548e-13 | 1.008e-08 | False |

## Single-gate repair grid

### h1_L4_plus_rzz_4_7

- Best: sweeps=8, substeps=1, window=1, full_sweep=True, fid=1.704e-04, wall=0.227s, max_bond=15

### h05_L4_plus_rzz_2_5

- Best: sweeps=1, substeps=2, window=1, full_sweep=False, fid=7.056e-08, wall=0.039s, max_bond=16

### random_rzz_4_7

- Best: sweeps=8, substeps=1, window=1, full_sweep=True, fid=2.144e-04, wall=0.140s, max_bond=15

### h2_L2_random_rzz_4_7

- Best: sweeps=8, substeps=1, window=1, full_sweep=True, fid=2.026e-04, wall=0.186s, max_bond=15

### control_rzz_2_5

- Best: sweeps=1, substeps=1, window=4, full_sweep=False, fid=1.008e-08, wall=0.018s, max_bond=8

## Sweep effects

(See CSV for full grids; mini-tables per target in CSV `section=grid`.)

## Full-circuit validation

| circuit | policy | fidelity | obs_max | max_bond | wall_s |
|---|---|---:|---:|---:|---:|
| `ising2d_3x3_h0.5_dt0.1_L2_all_zero` | tebd_swaps | 0.000e+00 | 6.300e-04 | 8 | 0.05 |
| `ising2d_3x3_h0.5_dt0.1_L2_all_zero` | all_enrichment | 9.653e-12 | 6.300e-04 | 5 | 0.15 |
| `ising2d_3x3_h0.5_dt0.1_L2_all_zero` | current_router | 5.073e-07 | 6.248e-04 | 5 | 0.19 |
| `ising2d_3x3_h0.5_dt0.1_L2_all_zero` | repaired_tdvp | 5.048e-07 | 6.246e-04 | 5 | 0.28 |
| `ising2d_3x3_h0.5_dt0.1_L4_plus` | tebd_swaps | 0.000e+00 | 1.885e-01 | 16 | 0.14 |
| `ising2d_3x3_h0.5_dt0.1_L4_plus` | all_enrichment | 4.328e-12 | 1.885e-01 | 16 | 0.29 |
| `ising2d_3x3_h0.5_dt0.1_L4_plus` | current_router | 1.795e-02 | 2.172e-01 | 16 | 0.47 |
| `ising2d_3x3_h0.5_dt0.1_L4_plus` | repaired_tdvp | 7.204e-03 | 2.127e-01 | 16 | 0.61 |
| `ising2d_3x3_h1_dt0.1_L4_plus` | tebd_swaps | 0.000e+00 | 1.614e-01 | 16 | 0.13 |
| `ising2d_3x3_h1_dt0.1_L4_plus` | all_enrichment | 6.339e-13 | 1.614e-01 | 16 | 0.31 |
| `ising2d_3x3_h1_dt0.1_L4_plus` | current_router | 4.489e-02 | 1.911e-01 | 16 | 0.41 |
| `ising2d_3x3_h1_dt0.1_L4_random_product` | tebd_swaps | 0.000e+00 | 4.457e-01 | 16 | 0.15 |
| `ising2d_3x3_h1_dt0.1_L4_random_product` | all_enrichment | 7.339e-13 | 4.457e-01 | 16 | 0.29 |
| `ising2d_3x3_h1_dt0.1_L4_random_product` | current_router | 4.104e-02 | 4.566e-01 | 16 | 0.46 |
| `ising2d_3x3_h2_dt0.1_L2_random_product` | tebd_swaps | 0.000e+00 | 4.906e-01 | 16 | 0.05 |
| `ising2d_3x3_h2_dt0.1_L2_random_product` | all_enrichment | 1.675e-12 | 4.906e-01 | 16 | 0.13 |
| `ising2d_3x3_h2_dt0.1_L2_random_product` | current_router | 8.497e-03 | 4.913e-01 | 16 | 0.20 |

## Conclusion

- H1 (more sweeps): partially supported — sweeps help but may not reach 1e-6 alone.
- H2 (substepping): partially supported.
- H3 (larger window): partially supported.
- H4 (full sweep): partially supported.
- All-enrichment remains exact; repaired TDVP does not match — keep enrichment for production.
- H1 (more sweeps): not supported — more sweeps did not beat plain TDVP.
- H3 (larger window): not supported.
- H4 (full sweep): not supported.
- Practical single-gate repair (<=1e-6) may be achievable; check full-circuit validation.
- H2 (substepping): not supported.
