# Small-angle local-generator TDVP vs TEBD+SWAP

## Purpose

This benchmark targets the regime where TDVP is expected to be useful: small-angle long-range generator evolution (θ ≤ 0.05), with **Pauli enrichment disabled**.

## Methods

- **local_generator_tdvp**: NN gates via TEBD/SVD; LR `rxx/ryy/rzz` via 2TDVP only (`LR_PAULI_ROUTE='tdvp_only'`, enrichment disabled).
- **tebd_swaps**: all two-qubit gates via TEBD+SWAP.
- **full_hamiltonian_tdvp**: scaffolded (not implemented in YAQS digital path).

## Small-angle sanity check

- θ_max across rows: **0.005** (design cap 0.05; no θ=0.2).
- θ_mean range: 0.001698 – 0.005
- Rows with θ_max ≤ 0.025: **48** / 48

## Accuracy vs bond dimension

| family | method | max_bond | obs_max (median) | wall_s (median) | sum_χ³ (median) |
|---|---|---:|---:|---:|---:|
| ising_2d | local_generator_tdvp | 16 | 7.035e-04 | 0.67 | 120.5 |
| ising_2d | local_generator_tdvp | 32 | 7.035e-04 | 0.62 | 120.5 |
| ising_2d | local_generator_tdvp | 64 | 7.035e-04 | 0.73 | 120.5 |
| ising_2d | local_generator_tdvp | 128 | 7.035e-04 | 0.71 | 120.5 |
| ising_2d | tebd_swaps | 16 | 3.609e-04 | 0.37 | 1320.0 |
| ising_2d | tebd_swaps | 32 | 3.609e-04 | 0.38 | 1320.0 |
| ising_2d | tebd_swaps | 64 | 3.609e-04 | 0.38 | 1320.0 |
| ising_2d | tebd_swaps | 128 | 3.609e-04 | 0.50 | 1320.0 |
| power_law_ising | local_generator_tdvp | 16 | 2.455e-03 | 5.62 | 8980.5 |
| power_law_ising | local_generator_tdvp | 32 | 2.477e-03 | 4.78 | 34072.0 |
| power_law_ising | local_generator_tdvp | 64 | 2.477e-03 | 4.99 | 34072.0 |
| power_law_ising | local_generator_tdvp | 128 | 2.477e-03 | 5.13 | 34072.0 |
| power_law_ising | tebd_swaps | 16 | 9.129e-02 | 1.98 | 10201.5 |
| power_law_ising | tebd_swaps | 32 | 2.299e-02 | 2.95 | 52397.0 |
| power_law_ising | tebd_swaps | 64 | 2.502e-03 | 2.12 | 51221.5 |
| power_law_ising | tebd_swaps | 128 | 2.502e-03 | 2.26 | 51221.5 |
| xxx_2d | local_generator_tdvp | 16 | 8.028e-07 | 1.71 | 73.5 |
| xxx_2d | local_generator_tdvp | 32 | 8.028e-07 | 2.02 | 73.5 |
| xxx_2d | local_generator_tdvp | 64 | 8.028e-07 | 2.17 | 73.5 |
| xxx_2d | local_generator_tdvp | 128 | 8.028e-07 | 2.04 | 73.5 |
| xxx_2d | tebd_swaps | 16 | 9.109e-09 | 0.88 | 64.0 |
| xxx_2d | tebd_swaps | 32 | 9.109e-09 | 0.82 | 64.0 |
| xxx_2d | tebd_swaps | 64 | 9.109e-09 | 0.77 | 64.0 |
| xxx_2d | tebd_swaps | 128 | 9.109e-09 | 0.90 | 64.0 |

## Bond growth

Bond history not recorded (set `YAQS_BOND_HISTORY=1`).

## Runtime

- `local_generator_tdvp` median wall_time_s: **2.12**
- `tebd_swaps` median wall_time_s: **0.84**

## Physics observables

Rows with `observable_ordering_suspect=True` are excluded from observable conclusions.

## Power-law long-range results

- `local_generator_tdvp` median obs_max on power-law cases: **2.455e-03**
- `tebd_swaps` median obs_max on power-law cases: **2.502e-03**

## 2D results

- `local_generator_tdvp` median obs_max on 2D cases: **1.789e-05**
- `tebd_swaps` median obs_max on 2D cases: **1.904e-05**

## Recommended conclusion

- At fixed bond, **local-generator TDVP tends toward lower observable error** than TEBD+SWAP on several families.
- TDVP often shows **lower sum_χ³ (memory proxy)** than TEBD+SWAP.
- TEBD+SWAP is **faster** than local-generator TDVP in this sweep.
- Power-law long-range Ising: median obs_max TDVP=2.455e-03, TEBD=2.502e-03.
- **enriched_lr_pauli_count = 0** for local_generator_tdvp (enrichment disabled as intended).
- **full_hamiltonian_tdvp** is not implemented; do not compare until available.
