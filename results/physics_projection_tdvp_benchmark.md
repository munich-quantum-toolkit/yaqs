# Physics benchmark: projection-aware generator TDVP

## Method

Routing rule: TDVP if projection defect \(d = 1 - \min(\mathrm{projected\_ratio}, 1)\) is <= epsilon; exact Pauli-product enrichment otherwise.

Practical pass: pauli_obs_max_error <= 1e-3.

Strict pass: fidelity_error <= 1e-10 AND pauli_obs_max_error <= 1e-10 (when reference available).

## Summary (grouped)

family | geometry | boundary | init | eps | n | dt | layers | lr_pauli | tdvp_frac | obs_max | fid_err

---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:

ising_1d_periodic | 1d | periodic | all_zero | 1e-04 | 8 | 0.1 | 1 | 1 | 1.000 | 2.998e-15 | 0.000e+00
ising_1d_periodic | 1d | periodic | all_zero | 1e-06 | 8 | 0.1 | 1 | 1 | 1.000 | 2.998e-15 | 0.000e+00
ising_1d_periodic | 1d | periodic | neel | 1e-04 | 8 | 0.1 | 1 | 1 | 1.000 | 4.108e-15 | 0.000e+00
ising_1d_periodic | 1d | periodic | neel | 1e-06 | 8 | 0.1 | 1 | 1 | 1.000 | 4.108e-15 | 0.000e+00
ising_1d_periodic | 1d | periodic | plus | 1e-04 | 8 | 0.1 | 1 | 1 | 0.000 | 1.258e-15 | 0.000e+00
ising_1d_periodic | 1d | periodic | plus | 1e-06 | 8 | 0.1 | 1 | 1 | 0.000 | 1.258e-15 | 0.000e+00
ising_1d_periodic | 1d | periodic | random_product | 1e-04 | 8 | 0.1 | 1 | 1 | 0.000 | 2.220e-15 | 0.000e+00
ising_1d_periodic | 1d | periodic | random_product | 1e-06 | 8 | 0.1 | 1 | 1 | 0.000 | 2.220e-15 | 0.000e+00
xxx_1d_periodic | 1d | periodic | all_zero | 1e-04 | 8 | 0.1 | 1 | 3 | 0.667 | 3.331e-15 | 0.000e+00
xxx_1d_periodic | 1d | periodic | all_zero | 1e-06 | 8 | 0.1 | 1 | 3 | 0.667 | 3.331e-15 | 0.000e+00
xxx_1d_periodic | 1d | periodic | neel | 1e-04 | 8 | 0.1 | 1 | 3 | 0.000 | 3.775e-15 | 0.000e+00
xxx_1d_periodic | 1d | periodic | neel | 1e-06 | 8 | 0.1 | 1 | 3 | 0.000 | 3.775e-15 | 0.000e+00
xxx_1d_periodic | 1d | periodic | plus | 1e-04 | 8 | 0.1 | 1 | 3 | 0.667 | 5.379e-14 | 0.000e+00
xxx_1d_periodic | 1d | periodic | plus | 1e-06 | 8 | 0.1 | 1 | 3 | 0.667 | 5.379e-14 | 0.000e+00
xxx_1d_periodic | 1d | periodic | random_product | 1e-04 | 8 | 0.1 | 1 | 3 | 0.000 | 1.323e-14 | 0.000e+00
xxx_1d_periodic | 1d | periodic | random_product | 1e-06 | 8 | 0.1 | 1 | 3 | 0.000 | 1.323e-14 | 0.000e+00
ising_1d_periodic | 1d | periodic | all_zero | 1e-04 | 8 | 0.1 | 4 | 4 | 0.500 | 4.568e-08 | 3.479e-12
ising_1d_periodic | 1d | periodic | all_zero | 1e-06 | 8 | 0.1 | 4 | 4 | 0.250 | 1.138e-08 | 8.389e-13
ising_1d_periodic | 1d | periodic | neel | 1e-04 | 8 | 0.1 | 4 | 4 | 0.500 | 4.568e-08 | 3.478e-12
ising_1d_periodic | 1d | periodic | neel | 1e-06 | 8 | 0.1 | 4 | 4 | 0.250 | 1.138e-08 | 8.393e-13
ising_1d_periodic | 1d | periodic | plus | 1e-04 | 8 | 0.1 | 4 | 4 | 0.000 | 2.036e-10 | 2.816e-13
ising_1d_periodic | 1d | periodic | plus | 1e-06 | 8 | 0.1 | 4 | 4 | 0.000 | 2.036e-10 | 2.816e-13
ising_1d_periodic | 1d | periodic | random_product | 1e-04 | 8 | 0.1 | 4 | 4 | 0.000 | 5.575e-08 | 8.094e-13
ising_1d_periodic | 1d | periodic | random_product | 1e-06 | 8 | 0.1 | 4 | 4 | 0.000 | 5.575e-08 | 8.094e-13
xxx_1d_periodic | 1d | periodic | all_zero | 1e-04 | 8 | 0.1 | 4 | 12 | 0.667 | 4.361e-14 | 0.000e+00
xxx_1d_periodic | 1d | periodic | all_zero | 1e-06 | 8 | 0.1 | 4 | 12 | 0.667 | 4.361e-14 | 0.000e+00
xxx_1d_periodic | 1d | periodic | neel | 1e-04 | 8 | 0.1 | 4 | 12 | 0.000 | 1.837e-14 | 0.000e+00
xxx_1d_periodic | 1d | periodic | neel | 1e-06 | 8 | 0.1 | 4 | 12 | 0.000 | 1.837e-14 | 0.000e+00
xxx_1d_periodic | 1d | periodic | plus | 1e-04 | 8 | 0.1 | 4 | 12 | 0.667 | 6.882e-14 | 0.000e+00
xxx_1d_periodic | 1d | periodic | plus | 1e-06 | 8 | 0.1 | 4 | 12 | 0.667 | 6.882e-14 | 0.000e+00
xxx_1d_periodic | 1d | periodic | random_product | 1e-04 | 8 | 0.1 | 4 | 12 | 0.000 | 1.613e-14 | 0.000e+00
xxx_1d_periodic | 1d | periodic | random_product | 1e-06 | 8 | 0.1 | 4 | 12 | 0.000 | 1.613e-14 | 0.000e+00
ising_1d_periodic | 1d | periodic | all_zero | 1e-04 | 8 | 0.1 | 8 | 8 | 0.250 | 7.137e-07 | 6.474e-12
ising_1d_periodic | 1d | periodic | all_zero | 1e-06 | 8 | 0.1 | 8 | 8 | 0.125 | 3.743e-07 | 3.380e-12
ising_1d_periodic | 1d | periodic | neel | 1e-04 | 8 | 0.1 | 8 | 8 | 0.250 | 7.137e-07 | 6.475e-12
ising_1d_periodic | 1d | periodic | neel | 1e-06 | 8 | 0.1 | 8 | 8 | 0.125 | 3.743e-07 | 3.380e-12
ising_1d_periodic | 1d | periodic | plus | 1e-04 | 8 | 0.1 | 8 | 8 | 0.000 | 5.112e-09 | 2.751e-13
ising_1d_periodic | 1d | periodic | plus | 1e-06 | 8 | 0.1 | 8 | 8 | 0.000 | 5.112e-09 | 2.751e-13
ising_1d_periodic | 1d | periodic | random_product | 1e-04 | 8 | 0.1 | 8 | 8 | 0.000 | 6.505e-08 | 8.025e-13
ising_1d_periodic | 1d | periodic | random_product | 1e-06 | 8 | 0.1 | 8 | 8 | 0.000 | 6.505e-08 | 8.025e-13
xxx_1d_periodic | 1d | periodic | all_zero | 1e-04 | 8 | 0.1 | 8 | 24 | 0.667 | 2.272e-14 | 0.000e+00
xxx_1d_periodic | 1d | periodic | all_zero | 1e-06 | 8 | 0.1 | 8 | 24 | 0.667 | 2.272e-14 | 0.000e+00
xxx_1d_periodic | 1d | periodic | neel | 1e-04 | 8 | 0.1 | 8 | 24 | 0.000 | 1.654e-14 | 0.000e+00
xxx_1d_periodic | 1d | periodic | neel | 1e-06 | 8 | 0.1 | 8 | 24 | 0.000 | 1.654e-14 | 0.000e+00
xxx_1d_periodic | 1d | periodic | plus | 1e-04 | 8 | 0.1 | 8 | 24 | 0.667 | 6.943e-14 | 0.000e+00
xxx_1d_periodic | 1d | periodic | plus | 1e-06 | 8 | 0.1 | 8 | 24 | 0.667 | 6.943e-14 | 0.000e+00
xxx_1d_periodic | 1d | periodic | random_product | 1e-04 | 8 | 0.1 | 8 | 24 | 0.000 | 2.029e-14 | 0.000e+00
xxx_1d_periodic | 1d | periodic | random_product | 1e-06 | 8 | 0.1 | 8 | 24 | 0.000 | 2.029e-14 | 0.000e+00
ising_1d_periodic | 1d | periodic | all_zero | 1e-04 | 12 | 0.1 | 1 | 1 | 1.000 | 1.865e-14 | 0.000e+00
ising_1d_periodic | 1d | periodic | all_zero | 1e-06 | 12 | 0.1 | 1 | 1 | 1.000 | 1.865e-14 | 0.000e+00
ising_1d_periodic | 1d | periodic | neel | 1e-04 | 12 | 0.1 | 1 | 1 | 1.000 | 7.994e-15 | 0.000e+00
ising_1d_periodic | 1d | periodic | neel | 1e-06 | 12 | 0.1 | 1 | 1 | 1.000 | 7.994e-15 | 0.000e+00
ising_1d_periodic | 1d | periodic | plus | 1e-04 | 12 | 0.1 | 1 | 1 | 0.000 | 4.219e-15 | 0.000e+00
ising_1d_periodic | 1d | periodic | plus | 1e-06 | 12 | 0.1 | 1 | 1 | 0.000 | 4.219e-15 | 0.000e+00
ising_1d_periodic | 1d | periodic | random_product | 1e-04 | 12 | 0.1 | 1 | 1 | 0.000 | 3.109e-15 | 0.000e+00
ising_1d_periodic | 1d | periodic | random_product | 1e-06 | 12 | 0.1 | 1 | 1 | 0.000 | 3.109e-15 | 0.000e+00
xxx_1d_periodic | 1d | periodic | all_zero | 1e-04 | 12 | 0.1 | 1 | 3 | 0.667 | 4.774e-15 | 0.000e+00
xxx_1d_periodic | 1d | periodic | all_zero | 1e-06 | 12 | 0.1 | 1 | 3 | 0.667 | 4.774e-15 | 0.000e+00
xxx_1d_periodic | 1d | periodic | neel | 1e-04 | 12 | 0.1 | 1 | 3 | 0.000 | 2.193e-14 | 0.000e+00
xxx_1d_periodic | 1d | periodic | neel | 1e-06 | 12 | 0.1 | 1 | 3 | 0.000 | 2.193e-14 | 0.000e+00
xxx_1d_periodic | 1d | periodic | plus | 1e-04 | 12 | 0.1 | 1 | 3 | 0.667 | 6.451e-13 | 0.000e+00
xxx_1d_periodic | 1d | periodic | plus | 1e-06 | 12 | 0.1 | 1 | 3 | 0.667 | 6.451e-13 | 0.000e+00
xxx_1d_periodic | 1d | periodic | random_product | 1e-04 | 12 | 0.1 | 1 | 3 | 0.000 | 3.861e-14 | 0.000e+00
xxx_1d_periodic | 1d | periodic | random_product | 1e-06 | 12 | 0.1 | 1 | 3 | 0.000 | 3.861e-14 | 0.000e+00
ising_1d_periodic | 1d | periodic | all_zero | 1e-04 | 12 | 0.1 | 4 | 4 | 0.500 | 4.834e-08 | 6.394e-12
ising_1d_periodic | 1d | periodic | all_zero | 1e-06 | 12 | 0.1 | 4 | 4 | 0.250 | 1.394e-08 | 4.694e-12
ising_1d_periodic | 1d | periodic | neel | 1e-04 | 12 | 0.1 | 4 | 4 | 0.500 | 4.834e-08 | 6.395e-12
ising_1d_periodic | 1d | periodic | neel | 1e-06 | 12 | 0.1 | 4 | 4 | 0.250 | 1.394e-08 | 4.694e-12
ising_1d_periodic | 1d | periodic | plus | 1e-04 | 12 | 0.1 | 4 | 4 | 0.000 | 1.786e-07 | 2.141e-12
ising_1d_periodic | 1d | periodic | plus | 1e-06 | 12 | 0.1 | 4 | 4 | 0.000 | 1.786e-07 | 2.141e-12
ising_1d_periodic | 1d | periodic | random_product | 1e-04 | 12 | 0.1 | 4 | 4 | 0.000 | 5.268e-08 | 4.104e-12
ising_1d_periodic | 1d | periodic | random_product | 1e-06 | 12 | 0.1 | 4 | 4 | 0.000 | 5.268e-08 | 4.104e-12
xxx_1d_periodic | 1d | periodic | all_zero | 1e-04 | 12 | 0.1 | 4 | 12 | 0.667 | 1.676e-14 | 0.000e+00
xxx_1d_periodic | 1d | periodic | all_zero | 1e-06 | 12 | 0.1 | 4 | 12 | 0.667 | 1.676e-14 | 0.000e+00
xxx_1d_periodic | 1d | periodic | neel | 1e-04 | 12 | 0.1 | 4 | 12 | 0.000 | 1.291e-07 | 3.745e-12
xxx_1d_periodic | 1d | periodic | neel | 1e-06 | 12 | 0.1 | 4 | 12 | 0.000 | 1.291e-07 | 3.745e-12
xxx_1d_periodic | 1d | periodic | plus | 1e-04 | 12 | 0.1 | 4 | 12 | 0.667 | 1.145e-13 | 0.000e+00
xxx_1d_periodic | 1d | periodic | plus | 1e-06 | 12 | 0.1 | 4 | 12 | 0.667 | 1.145e-13 | 0.000e+00
xxx_1d_periodic | 1d | periodic | random_product | 1e-04 | 12 | 0.1 | 4 | 12 | 0.000 | 7.857e-09 | 1.005e-11
xxx_1d_periodic | 1d | periodic | random_product | 1e-06 | 12 | 0.1 | 4 | 12 | 0.000 | 7.857e-09 | 1.005e-11
ising_1d_periodic | 1d | periodic | all_zero | 1e-04 | 12 | 0.1 | 8 | 8 | 0.250 | 8.236e-07 | 2.101e-11
ising_1d_periodic | 1d | periodic | all_zero | 1e-06 | 12 | 0.1 | 8 | 8 | 0.125 | 3.499e-07 | 1.852e-11
ising_1d_periodic | 1d | periodic | neel | 1e-04 | 12 | 0.1 | 8 | 8 | 0.250 | 8.236e-07 | 2.101e-11
ising_1d_periodic | 1d | periodic | neel | 1e-06 | 12 | 0.1 | 8 | 8 | 0.125 | 3.499e-07 | 1.852e-11
ising_1d_periodic | 1d | periodic | plus | 1e-04 | 12 | 0.1 | 8 | 8 | 0.000 | 1.712e-07 | 1.098e-11
ising_1d_periodic | 1d | periodic | plus | 1e-06 | 12 | 0.1 | 8 | 8 | 0.000 | 1.712e-07 | 1.098e-11
ising_1d_periodic | 1d | periodic | random_product | 1e-04 | 12 | 0.1 | 8 | 8 | 0.000 | 1.405e-07 | 1.153e-11
ising_1d_periodic | 1d | periodic | random_product | 1e-06 | 12 | 0.1 | 8 | 8 | 0.000 | 1.405e-07 | 1.153e-11
xxx_1d_periodic | 1d | periodic | all_zero | 1e-04 | 12 | 0.1 | 8 | 24 | 0.667 | 2.145e-13 | 0.000e+00
xxx_1d_periodic | 1d | periodic | all_zero | 1e-06 | 12 | 0.1 | 8 | 24 | 0.667 | 2.145e-13 | 0.000e+00
xxx_1d_periodic | 1d | periodic | neel | 1e-04 | 12 | 0.1 | 8 | 24 | 0.000 | 1.890e-07 | 3.730e-12
xxx_1d_periodic | 1d | periodic | neel | 1e-06 | 12 | 0.1 | 8 | 24 | 0.000 | 1.890e-07 | 3.730e-12
xxx_1d_periodic | 1d | periodic | plus | 1e-04 | 12 | 0.1 | 8 | 24 | 0.667 | 1.478e-13 | 0.000e+00
xxx_1d_periodic | 1d | periodic | plus | 1e-06 | 12 | 0.1 | 8 | 24 | 0.667 | 1.478e-13 | 0.000e+00
xxx_1d_periodic | 1d | periodic | random_product | 1e-04 | 12 | 0.1 | 8 | 24 | 0.000 | 5.492e-08 | 1.373e-11
xxx_1d_periodic | 1d | periodic | random_product | 1e-06 | 12 | 0.1 | 8 | 24 | 0.000 | 5.492e-08 | 1.373e-11

## TDVP vs enrichment usage

family | geometry | boundary | init | eps | lr_pauli | tdvp | enriched | tdvp_frac

---|---|---|---|---:|---:|---:|---:|---:

ising_1d_periodic | 1d | periodic | all_zero | 1e-04 | 1 | 1 | 0 | 1.000
ising_1d_periodic | 1d | periodic | all_zero | 1e-06 | 1 | 1 | 0 | 1.000
ising_1d_periodic | 1d | periodic | neel | 1e-04 | 1 | 1 | 0 | 1.000
ising_1d_periodic | 1d | periodic | neel | 1e-06 | 1 | 1 | 0 | 1.000
ising_1d_periodic | 1d | periodic | plus | 1e-04 | 1 | 0 | 1 | 0.000
ising_1d_periodic | 1d | periodic | plus | 1e-06 | 1 | 0 | 1 | 0.000
ising_1d_periodic | 1d | periodic | random_product | 1e-04 | 1 | 0 | 1 | 0.000
ising_1d_periodic | 1d | periodic | random_product | 1e-06 | 1 | 0 | 1 | 0.000
xxx_1d_periodic | 1d | periodic | all_zero | 1e-04 | 3 | 2 | 1 | 0.667
xxx_1d_periodic | 1d | periodic | all_zero | 1e-06 | 3 | 2 | 1 | 0.667
xxx_1d_periodic | 1d | periodic | neel | 1e-04 | 3 | 0 | 3 | 0.000
xxx_1d_periodic | 1d | periodic | neel | 1e-06 | 3 | 0 | 3 | 0.000
xxx_1d_periodic | 1d | periodic | plus | 1e-04 | 3 | 2 | 1 | 0.667
xxx_1d_periodic | 1d | periodic | plus | 1e-06 | 3 | 2 | 1 | 0.667
xxx_1d_periodic | 1d | periodic | random_product | 1e-04 | 3 | 0 | 3 | 0.000
xxx_1d_periodic | 1d | periodic | random_product | 1e-06 | 3 | 0 | 3 | 0.000
ising_1d_periodic | 1d | periodic | all_zero | 1e-04 | 4 | 2 | 2 | 0.500
ising_1d_periodic | 1d | periodic | all_zero | 1e-06 | 4 | 1 | 3 | 0.250
ising_1d_periodic | 1d | periodic | neel | 1e-04 | 4 | 2 | 2 | 0.500
ising_1d_periodic | 1d | periodic | neel | 1e-06 | 4 | 1 | 3 | 0.250
ising_1d_periodic | 1d | periodic | plus | 1e-04 | 4 | 0 | 4 | 0.000
ising_1d_periodic | 1d | periodic | plus | 1e-06 | 4 | 0 | 4 | 0.000
ising_1d_periodic | 1d | periodic | random_product | 1e-04 | 4 | 0 | 4 | 0.000
ising_1d_periodic | 1d | periodic | random_product | 1e-06 | 4 | 0 | 4 | 0.000
xxx_1d_periodic | 1d | periodic | all_zero | 1e-04 | 12 | 8 | 4 | 0.667
xxx_1d_periodic | 1d | periodic | all_zero | 1e-06 | 12 | 8 | 4 | 0.667
xxx_1d_periodic | 1d | periodic | neel | 1e-04 | 12 | 0 | 12 | 0.000
xxx_1d_periodic | 1d | periodic | neel | 1e-06 | 12 | 0 | 12 | 0.000
xxx_1d_periodic | 1d | periodic | plus | 1e-04 | 12 | 8 | 4 | 0.667
xxx_1d_periodic | 1d | periodic | plus | 1e-06 | 12 | 8 | 4 | 0.667
xxx_1d_periodic | 1d | periodic | random_product | 1e-04 | 12 | 0 | 12 | 0.000
xxx_1d_periodic | 1d | periodic | random_product | 1e-06 | 12 | 0 | 12 | 0.000
ising_1d_periodic | 1d | periodic | all_zero | 1e-04 | 8 | 2 | 6 | 0.250
ising_1d_periodic | 1d | periodic | all_zero | 1e-06 | 8 | 1 | 7 | 0.125
ising_1d_periodic | 1d | periodic | neel | 1e-04 | 8 | 2 | 6 | 0.250
ising_1d_periodic | 1d | periodic | neel | 1e-06 | 8 | 1 | 7 | 0.125
ising_1d_periodic | 1d | periodic | plus | 1e-04 | 8 | 0 | 8 | 0.000
ising_1d_periodic | 1d | periodic | plus | 1e-06 | 8 | 0 | 8 | 0.000
ising_1d_periodic | 1d | periodic | random_product | 1e-04 | 8 | 0 | 8 | 0.000
ising_1d_periodic | 1d | periodic | random_product | 1e-06 | 8 | 0 | 8 | 0.000
xxx_1d_periodic | 1d | periodic | all_zero | 1e-04 | 24 | 16 | 8 | 0.667
xxx_1d_periodic | 1d | periodic | all_zero | 1e-06 | 24 | 16 | 8 | 0.667
xxx_1d_periodic | 1d | periodic | neel | 1e-04 | 24 | 0 | 24 | 0.000
xxx_1d_periodic | 1d | periodic | neel | 1e-06 | 24 | 0 | 24 | 0.000
xxx_1d_periodic | 1d | periodic | plus | 1e-04 | 24 | 16 | 8 | 0.667
xxx_1d_periodic | 1d | periodic | plus | 1e-06 | 24 | 16 | 8 | 0.667
xxx_1d_periodic | 1d | periodic | random_product | 1e-04 | 24 | 0 | 24 | 0.000
xxx_1d_periodic | 1d | periodic | random_product | 1e-06 | 24 | 0 | 24 | 0.000
ising_1d_periodic | 1d | periodic | all_zero | 1e-04 | 1 | 1 | 0 | 1.000
ising_1d_periodic | 1d | periodic | all_zero | 1e-06 | 1 | 1 | 0 | 1.000
ising_1d_periodic | 1d | periodic | neel | 1e-04 | 1 | 1 | 0 | 1.000
ising_1d_periodic | 1d | periodic | neel | 1e-06 | 1 | 1 | 0 | 1.000
ising_1d_periodic | 1d | periodic | plus | 1e-04 | 1 | 0 | 1 | 0.000
ising_1d_periodic | 1d | periodic | plus | 1e-06 | 1 | 0 | 1 | 0.000
ising_1d_periodic | 1d | periodic | random_product | 1e-04 | 1 | 0 | 1 | 0.000
ising_1d_periodic | 1d | periodic | random_product | 1e-06 | 1 | 0 | 1 | 0.000
xxx_1d_periodic | 1d | periodic | all_zero | 1e-04 | 3 | 2 | 1 | 0.667
xxx_1d_periodic | 1d | periodic | all_zero | 1e-06 | 3 | 2 | 1 | 0.667
xxx_1d_periodic | 1d | periodic | neel | 1e-04 | 3 | 0 | 3 | 0.000
xxx_1d_periodic | 1d | periodic | neel | 1e-06 | 3 | 0 | 3 | 0.000
xxx_1d_periodic | 1d | periodic | plus | 1e-04 | 3 | 2 | 1 | 0.667
xxx_1d_periodic | 1d | periodic | plus | 1e-06 | 3 | 2 | 1 | 0.667
xxx_1d_periodic | 1d | periodic | random_product | 1e-04 | 3 | 0 | 3 | 0.000
xxx_1d_periodic | 1d | periodic | random_product | 1e-06 | 3 | 0 | 3 | 0.000
ising_1d_periodic | 1d | periodic | all_zero | 1e-04 | 4 | 2 | 2 | 0.500
ising_1d_periodic | 1d | periodic | all_zero | 1e-06 | 4 | 1 | 3 | 0.250
ising_1d_periodic | 1d | periodic | neel | 1e-04 | 4 | 2 | 2 | 0.500
ising_1d_periodic | 1d | periodic | neel | 1e-06 | 4 | 1 | 3 | 0.250
ising_1d_periodic | 1d | periodic | plus | 1e-04 | 4 | 0 | 4 | 0.000
ising_1d_periodic | 1d | periodic | plus | 1e-06 | 4 | 0 | 4 | 0.000
ising_1d_periodic | 1d | periodic | random_product | 1e-04 | 4 | 0 | 4 | 0.000
ising_1d_periodic | 1d | periodic | random_product | 1e-06 | 4 | 0 | 4 | 0.000
xxx_1d_periodic | 1d | periodic | all_zero | 1e-04 | 12 | 8 | 4 | 0.667
xxx_1d_periodic | 1d | periodic | all_zero | 1e-06 | 12 | 8 | 4 | 0.667
xxx_1d_periodic | 1d | periodic | neel | 1e-04 | 12 | 0 | 12 | 0.000
xxx_1d_periodic | 1d | periodic | neel | 1e-06 | 12 | 0 | 12 | 0.000
xxx_1d_periodic | 1d | periodic | plus | 1e-04 | 12 | 8 | 4 | 0.667
xxx_1d_periodic | 1d | periodic | plus | 1e-06 | 12 | 8 | 4 | 0.667
xxx_1d_periodic | 1d | periodic | random_product | 1e-04 | 12 | 0 | 12 | 0.000
xxx_1d_periodic | 1d | periodic | random_product | 1e-06 | 12 | 0 | 12 | 0.000
ising_1d_periodic | 1d | periodic | all_zero | 1e-04 | 8 | 2 | 6 | 0.250
ising_1d_periodic | 1d | periodic | all_zero | 1e-06 | 8 | 1 | 7 | 0.125
ising_1d_periodic | 1d | periodic | neel | 1e-04 | 8 | 2 | 6 | 0.250
ising_1d_periodic | 1d | periodic | neel | 1e-06 | 8 | 1 | 7 | 0.125
ising_1d_periodic | 1d | periodic | plus | 1e-04 | 8 | 0 | 8 | 0.000
ising_1d_periodic | 1d | periodic | plus | 1e-06 | 8 | 0 | 8 | 0.000
ising_1d_periodic | 1d | periodic | random_product | 1e-04 | 8 | 0 | 8 | 0.000
ising_1d_periodic | 1d | periodic | random_product | 1e-06 | 8 | 0 | 8 | 0.000
xxx_1d_periodic | 1d | periodic | all_zero | 1e-04 | 24 | 16 | 8 | 0.667
xxx_1d_periodic | 1d | periodic | all_zero | 1e-06 | 24 | 16 | 8 | 0.667
xxx_1d_periodic | 1d | periodic | neel | 1e-04 | 24 | 0 | 24 | 0.000
xxx_1d_periodic | 1d | periodic | neel | 1e-06 | 24 | 0 | 24 | 0.000
xxx_1d_periodic | 1d | periodic | plus | 1e-04 | 24 | 16 | 8 | 0.667
xxx_1d_periodic | 1d | periodic | plus | 1e-06 | 24 | 16 | 8 | 0.667
xxx_1d_periodic | 1d | periodic | random_product | 1e-04 | 24 | 0 | 24 | 0.000
xxx_1d_periodic | 1d | periodic | random_product | 1e-06 | 24 | 0 | 24 | 0.000

## Practical accuracy failures (obs_max > 1e-3)

None.


## Strict accuracy failures (when reference available)

circuit | method | eps | fid_err | obs_max | worst_obs

---|---|---:|---:|---:|---

`ising_1d_periodic_n8_h1_dt0.1_L4_all_zero` | adaptive_hybrid_defect_1e-4 | 1e-04 | 3.479e-12 | 4.568e-08 | YY(7,0)
`ising_1d_periodic_n8_h1_dt0.1_L4_all_zero` | adaptive_hybrid_defect_1e-6 | 1e-06 | 8.389e-13 | 1.138e-08 | X(4)
`ising_1d_periodic_n8_h1_dt0.1_L4_neel` | adaptive_hybrid_defect_1e-4 | 1e-04 | 3.478e-12 | 4.568e-08 | YY(7,0)
`ising_1d_periodic_n8_h1_dt0.1_L4_neel` | adaptive_hybrid_defect_1e-6 | 1e-06 | 8.393e-13 | 1.138e-08 | X(4)
`ising_1d_periodic_n8_h1_dt0.1_L4_plus` | adaptive_hybrid_defect_1e-4 | 1e-04 | 2.816e-13 | 2.036e-10 | YY(7,0)
`ising_1d_periodic_n8_h1_dt0.1_L4_plus` | adaptive_hybrid_defect_1e-6 | 1e-06 | 2.816e-13 | 2.036e-10 | YY(7,0)
`ising_1d_periodic_n8_h1_dt0.1_L4_random_product` | adaptive_hybrid_defect_1e-4 | 1e-04 | 8.094e-13 | 5.575e-08 | ZZ(5,6)
`ising_1d_periodic_n8_h1_dt0.1_L4_random_product` | adaptive_hybrid_defect_1e-6 | 1e-06 | 8.094e-13 | 5.575e-08 | ZZ(5,6)
`ising_1d_periodic_n8_h1_dt0.1_L8_all_zero` | adaptive_hybrid_defect_1e-4 | 1e-04 | 6.474e-12 | 7.137e-07 | YY(7,0)
`ising_1d_periodic_n8_h1_dt0.1_L8_all_zero` | adaptive_hybrid_defect_1e-6 | 1e-06 | 3.380e-12 | 3.743e-07 | ZZ(5,6)
`ising_1d_periodic_n8_h1_dt0.1_L8_neel` | adaptive_hybrid_defect_1e-4 | 1e-04 | 6.475e-12 | 7.137e-07 | YY(7,0)
`ising_1d_periodic_n8_h1_dt0.1_L8_neel` | adaptive_hybrid_defect_1e-6 | 1e-06 | 3.380e-12 | 3.743e-07 | ZZ(5,6)
`ising_1d_periodic_n8_h1_dt0.1_L8_plus` | adaptive_hybrid_defect_1e-4 | 1e-04 | 2.751e-13 | 5.112e-09 | XX(3,4)
`ising_1d_periodic_n8_h1_dt0.1_L8_plus` | adaptive_hybrid_defect_1e-6 | 1e-06 | 2.751e-13 | 5.112e-09 | XX(3,4)
`ising_1d_periodic_n8_h1_dt0.1_L8_random_product` | adaptive_hybrid_defect_1e-4 | 1e-04 | 8.025e-13 | 6.505e-08 | YY(5,6)
`ising_1d_periodic_n8_h1_dt0.1_L8_random_product` | adaptive_hybrid_defect_1e-6 | 1e-06 | 8.025e-13 | 6.505e-08 | YY(5,6)
`ising_1d_periodic_n12_h1_dt0.1_L4_all_zero` | adaptive_hybrid_defect_1e-4 | 1e-04 | 6.394e-12 | 4.834e-08 | YY(11,0)
`ising_1d_periodic_n12_h1_dt0.1_L4_all_zero` | adaptive_hybrid_defect_1e-6 | 1e-06 | 4.694e-12 | 1.394e-08 | XX(8,9)
`ising_1d_periodic_n12_h1_dt0.1_L4_neel` | adaptive_hybrid_defect_1e-4 | 1e-04 | 6.395e-12 | 4.834e-08 | YY(11,0)
`ising_1d_periodic_n12_h1_dt0.1_L4_neel` | adaptive_hybrid_defect_1e-6 | 1e-06 | 4.694e-12 | 1.394e-08 | XX(8,9)
`ising_1d_periodic_n12_h1_dt0.1_L4_plus` | adaptive_hybrid_defect_1e-4 | 1e-04 | 2.141e-12 | 1.786e-07 | ZZ(6,7)
`ising_1d_periodic_n12_h1_dt0.1_L4_plus` | adaptive_hybrid_defect_1e-6 | 1e-06 | 2.141e-12 | 1.786e-07 | ZZ(6,7)
`ising_1d_periodic_n12_h1_dt0.1_L4_random_product` | adaptive_hybrid_defect_1e-4 | 1e-04 | 4.104e-12 | 5.268e-08 | YY(4,5)
`ising_1d_periodic_n12_h1_dt0.1_L4_random_product` | adaptive_hybrid_defect_1e-6 | 1e-06 | 4.104e-12 | 5.268e-08 | YY(4,5)
`xxx_1d_periodic_n12_dt0.1_L4_neel` | adaptive_hybrid_defect_1e-4 | 1e-04 | 3.745e-12 | 1.291e-07 | ZZ(11,0)
`xxx_1d_periodic_n12_dt0.1_L4_neel` | adaptive_hybrid_defect_1e-6 | 1e-06 | 3.745e-12 | 1.291e-07 | ZZ(11,0)
`xxx_1d_periodic_n12_dt0.1_L4_random_product` | adaptive_hybrid_defect_1e-4 | 1e-04 | 1.005e-11 | 7.857e-09 | ZZ(10,11)
`xxx_1d_periodic_n12_dt0.1_L4_random_product` | adaptive_hybrid_defect_1e-6 | 1e-06 | 1.005e-11 | 7.857e-09 | ZZ(10,11)
`ising_1d_periodic_n12_h1_dt0.1_L8_all_zero` | adaptive_hybrid_defect_1e-4 | 1e-04 | 2.101e-11 | 8.236e-07 | YY(11,0)
`ising_1d_periodic_n12_h1_dt0.1_L8_all_zero` | adaptive_hybrid_defect_1e-6 | 1e-06 | 1.852e-11 | 3.499e-07 | ZZ(5,6)
`ising_1d_periodic_n12_h1_dt0.1_L8_neel` | adaptive_hybrid_defect_1e-4 | 1e-04 | 2.101e-11 | 8.236e-07 | YY(11,0)
`ising_1d_periodic_n12_h1_dt0.1_L8_neel` | adaptive_hybrid_defect_1e-6 | 1e-06 | 1.852e-11 | 3.499e-07 | ZZ(5,6)
`ising_1d_periodic_n12_h1_dt0.1_L8_plus` | adaptive_hybrid_defect_1e-4 | 1e-04 | 1.098e-11 | 1.712e-07 | XX(5,6)
`ising_1d_periodic_n12_h1_dt0.1_L8_plus` | adaptive_hybrid_defect_1e-6 | 1e-06 | 1.098e-11 | 1.712e-07 | XX(5,6)
`ising_1d_periodic_n12_h1_dt0.1_L8_random_product` | adaptive_hybrid_defect_1e-4 | 1e-04 | 1.153e-11 | 1.405e-07 | XX(5,6)
`ising_1d_periodic_n12_h1_dt0.1_L8_random_product` | adaptive_hybrid_defect_1e-6 | 1e-06 | 1.153e-11 | 1.405e-07 | XX(5,6)
`xxx_1d_periodic_n12_dt0.1_L8_neel` | adaptive_hybrid_defect_1e-4 | 1e-04 | 3.730e-12 | 1.890e-07 | XX(11,0)
`xxx_1d_periodic_n12_dt0.1_L8_neel` | adaptive_hybrid_defect_1e-6 | 1e-06 | 3.730e-12 | 1.890e-07 | XX(11,0)
`xxx_1d_periodic_n12_dt0.1_L8_random_product` | adaptive_hybrid_defect_1e-4 | 1e-04 | 1.373e-11 | 5.492e-08 | ZZ(5,6)
`xxx_1d_periodic_n12_dt0.1_L8_random_product` | adaptive_hybrid_defect_1e-6 | 1e-06 | 1.373e-11 | 5.492e-08 | ZZ(5,6)

## Recommended setting

Compare eps=1e-4 vs eps=1e-6: eps=1e-4 should route more LR Pauli gates through TDVP, eps=1e-6 is more conservative. Use the TDVP fractions above plus practical accuracy failures to justify the default.

