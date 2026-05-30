# Fixed-bond advantage: local-generator TDVP vs TEBD+SWAP

**Stage 0** (`YAQS_FIXED_CHI_STAGE`). `LR_PAULI_ROUTE='tdvp_only'`; Pauli enrichment disabled for TDVP path.

## Claim tested

At fixed χ, local-generator TDVP can be more accurate than TEBD+SWAP for small-angle long-range Hamiltonian evolution, because SWAP networks inflate intermediate entanglement.

## Method

- **local_generator_tdvp**: NN → TEBD; LR Pauli → 2TDVP only.
- **tebd_swaps**: all 2q → TEBD+SWAP.
- Reference: Qiskit (n≤14), strict TEBD (χ=∞), optional exact Pauli enrichment.

## Sanity checks (Qiskit reference)

- χ=16: median energy_density err TDVP=1.054e-02, TEBD=8.787e-03
- χ=24: median energy_density err TDVP=1.054e-02, TEBD=6.572e-03
- χ=32: median energy_density err TDVP=1.054e-02, TEBD=9.628e-03

## Same-χ accuracy comparison

| χ | TDVP median obs_max | TEBD median obs_max | TDVP wins (count) |
|---:|---:|---:|---:|
| 8 | 4.744e-01 | 5.192e-01 | 2 |
| 12 | 4.743e-01 | 2.927e-01 | 2 |
| 16 | 4.741e-01 | 3.954e-01 | 2 |
| 24 | 4.741e-01 | 4.426e-01 | 2 |
| 32 | 4.741e-01 | 4.333e-01 | 2 |
| 48 | 4.741e-01 | 4.114e-01 | 2 |
| 64 | 4.741e-01 | 4.740e-01 | 2 |

## TEBD recovery χ

Smallest TEBD χ with error ≤ TDVP at χ∈{16,24,32}. See `fixed_chi_tdvp_advantage_summary.csv`.

Cases with recovery ratio ≥ 2 (observable):

- `pl_s0_n12_a1.5_R5_h0.5_dt0.00125_L10_plus` χ_TDVP=16: recovery_χ=64 (ratio=4.0)
- `pl_s0_n12_a1.5_R5_h0.5_dt0.00125_L10_plus` χ_TDVP=24: recovery_χ=64 (ratio=2.7)
- `pl_s0_n12_a1.5_R5_h0.5_dt0.00125_L10_plus` χ_TDVP=32: recovery_χ=64 (ratio=2.0)
- `pl_s0_n12_a1.5_R5_h0.5_dt0.00125_L20_plus` χ_TDVP=16: recovery_χ=64 (ratio=4.0)
- `pl_s0_n12_a1.5_R5_h0.5_dt0.00125_L20_plus` χ_TDVP=24: recovery_χ=64 (ratio=2.7)
- `pl_s0_n12_a1.5_R5_h0.5_dt0.00125_L20_plus` χ_TDVP=32: recovery_χ=64 (ratio=2.0)

## Bond growth and SWAP overhead

- `local_generator_tdvp`: median sum_χ³=41900, median max_χ_obs=24
- `tebd_swaps`: median sum_χ³=50832, median max_χ_obs=24

## Runtime

- `local_generator_tdvp` median wall_time_s: 9.05
- `tebd_swaps` median wall_time_s: 3.80

## Best examples / counterexamples


## Conclusion

- Same-χ pairs: TDVP lower obs error in **14** cases; TEBD lower in **14**.
- Recovery ratio ≥ 2 (TEBD needs ≥2× χ to match TDVP): **6** summary rows.
- `enriched_lr_pauli_count = 0` for local_generator_tdvp ✓

**Not supported in this stage:** TEBD+SWAP is as accurate or better at the same χ on most cases tested.
