# Enriched TDVP vs TEBD+SWAP benchmark

## Method summary

- **enriched_tdvp**: `gate_mode='hybrid'` with LR Pauli routing by projection defect \(d = 1 - \min(\mathrm{projected\_ratio}, 1)\): TDVP if \(d \le \epsilon\), else exact Pauli-product enrichment.\n
- **tebd_swaps**: `gate_mode='tebd'` and LR gates handled by SWAP networks.

Practical pass: pauli_obs_max_error <= 1e-3. Strict: fid<=1e-10 and obs_max<=1e-10 (when available).

## Fixed bond dimension scaling (summary)

family | n | layers | chi | method | eps | fid_err | obs_max | max_bond | swaps | wall_s

---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:

ising_2d | 9 | 2 | — | enriched_tdvp | 1e-04 | 5.073e-07 | 6.248e-04 | 5 | 0 | 0.13
ising_2d | 9 | 2 | — | enriched_tdvp | 1e-06 | 5.073e-07 | 6.248e-04 | 5 | 0 | 0.17
ising_2d | 9 | 2 | — | tebd_swaps | — | 0.000e+00 | 6.300e-04 | 8 | 60 | 0.05
ising_2d | 9 | 2 | 16 | enriched_tdvp | 1e-04 | 5.059e-07 | 6.249e-04 | 4 | 0 | 0.15
ising_2d | 9 | 2 | 16 | enriched_tdvp | 1e-06 | 5.059e-07 | 6.249e-04 | 4 | 0 | 0.15
ising_2d | 9 | 2 | 16 | tebd_swaps | — | 1.281e-09 | 6.300e-04 | 8 | 60 | 0.05
ising_2d | 9 | 2 | 32 | enriched_tdvp | 1e-04 | 5.059e-07 | 6.249e-04 | 4 | 0 | 0.15
ising_2d | 9 | 2 | 32 | enriched_tdvp | 1e-06 | 5.059e-07 | 6.249e-04 | 4 | 0 | 0.14
ising_2d | 9 | 2 | 32 | tebd_swaps | — | 1.281e-09 | 6.300e-04 | 8 | 60 | 0.05
ising_2d | 9 | 2 | 64 | enriched_tdvp | 1e-04 | 5.059e-07 | 6.249e-04 | 4 | 0 | 0.14
ising_2d | 9 | 2 | 64 | enriched_tdvp | 1e-06 | 5.059e-07 | 6.249e-04 | 4 | 0 | 0.15
ising_2d | 9 | 2 | 64 | tebd_swaps | — | 1.281e-09 | 6.300e-04 | 8 | 60 | 0.05
ising_2d | 9 | 2 | 128 | enriched_tdvp | 1e-04 | 5.059e-07 | 6.249e-04 | 4 | 0 | 0.16
ising_2d | 9 | 2 | 128 | enriched_tdvp | 1e-06 | 5.059e-07 | 6.249e-04 | 4 | 0 | 0.16
ising_2d | 9 | 2 | 128 | tebd_swaps | — | 1.281e-09 | 6.300e-04 | 8 | 60 | 0.05
ising_2d | 9 | 2 | 256 | enriched_tdvp | 1e-04 | 5.059e-07 | 6.249e-04 | 4 | 0 | 0.16
ising_2d | 9 | 2 | 256 | enriched_tdvp | 1e-06 | 5.059e-07 | 6.249e-04 | 4 | 0 | 0.15
ising_2d | 9 | 2 | 256 | tebd_swaps | — | 1.281e-09 | 6.300e-04 | 8 | 60 | 0.05
ising_2d | 9 | 2 | — | enriched_tdvp | 1e-04 | 5.752e-07 | 9.536e-02 | 16 | 0 | 0.19
ising_2d | 9 | 2 | — | enriched_tdvp | 1e-06 | 4.377e-07 | 9.537e-02 | 16 | 0 | 0.16
ising_2d | 9 | 2 | — | tebd_swaps | — | 0.000e+00 | 9.539e-02 | 16 | 60 | 0.05
ising_2d | 9 | 2 | 16 | enriched_tdvp | 1e-04 | 4.496e-07 | 9.535e-02 | 13 | 0 | 0.17
ising_2d | 9 | 2 | 16 | enriched_tdvp | 1e-06 | 3.834e-07 | 9.536e-02 | 11 | 0 | 0.15
ising_2d | 9 | 2 | 16 | tebd_swaps | — | 1.774e-09 | 9.539e-02 | 16 | 60 | 0.06
ising_2d | 9 | 2 | 32 | enriched_tdvp | 1e-04 | 4.496e-07 | 9.535e-02 | 13 | 0 | 0.17
ising_2d | 9 | 2 | 32 | enriched_tdvp | 1e-06 | 3.834e-07 | 9.536e-02 | 11 | 0 | 0.15
ising_2d | 9 | 2 | 32 | tebd_swaps | — | 1.774e-09 | 9.539e-02 | 16 | 60 | 0.05
ising_2d | 9 | 2 | 64 | enriched_tdvp | 1e-04 | 4.496e-07 | 9.535e-02 | 13 | 0 | 0.18
ising_2d | 9 | 2 | 64 | enriched_tdvp | 1e-06 | 3.834e-07 | 9.536e-02 | 11 | 0 | 0.15
ising_2d | 9 | 2 | 64 | tebd_swaps | — | 1.774e-09 | 9.539e-02 | 16 | 60 | 0.05
ising_2d | 9 | 2 | 128 | enriched_tdvp | 1e-04 | 4.496e-07 | 9.535e-02 | 13 | 0 | 0.17
ising_2d | 9 | 2 | 128 | enriched_tdvp | 1e-06 | 3.834e-07 | 9.536e-02 | 11 | 0 | 0.16
ising_2d | 9 | 2 | 128 | tebd_swaps | — | 1.774e-09 | 9.539e-02 | 16 | 60 | 0.05
ising_2d | 9 | 2 | 256 | enriched_tdvp | 1e-04 | 4.496e-07 | 9.535e-02 | 13 | 0 | 0.25
ising_2d | 9 | 2 | 256 | enriched_tdvp | 1e-06 | 3.834e-07 | 9.536e-02 | 11 | 0 | 0.18
ising_2d | 9 | 2 | 256 | tebd_swaps | — | 1.774e-09 | 9.539e-02 | 16 | 60 | 0.06
ising_2d | 9 | 2 | — | enriched_tdvp | 1e-04 | 5.197e-07 | 9.639e-01 | 5 | 0 | 0.16
ising_2d | 9 | 2 | — | enriched_tdvp | 1e-06 | 5.197e-07 | 9.639e-01 | 5 | 0 | 0.16
ising_2d | 9 | 2 | — | tebd_swaps | — | 2.697e-12 | 9.639e-01 | 8 | 60 | 0.05
ising_2d | 9 | 2 | 16 | enriched_tdvp | 1e-04 | 5.253e-07 | 9.639e-01 | 4 | 0 | 0.15
ising_2d | 9 | 2 | 16 | enriched_tdvp | 1e-06 | 5.253e-07 | 9.639e-01 | 4 | 0 | 0.16
ising_2d | 9 | 2 | 16 | tebd_swaps | — | 1.303e-09 | 9.639e-01 | 8 | 60 | 0.05
ising_2d | 9 | 2 | 32 | enriched_tdvp | 1e-04 | 5.253e-07 | 9.639e-01 | 4 | 0 | 0.14
ising_2d | 9 | 2 | 32 | enriched_tdvp | 1e-06 | 5.253e-07 | 9.639e-01 | 4 | 0 | 0.12
ising_2d | 9 | 2 | 32 | tebd_swaps | — | 1.303e-09 | 9.639e-01 | 8 | 60 | 0.05
ising_2d | 9 | 2 | 64 | enriched_tdvp | 1e-04 | 5.253e-07 | 9.639e-01 | 4 | 0 | 0.14
ising_2d | 9 | 2 | 64 | enriched_tdvp | 1e-06 | 5.253e-07 | 9.639e-01 | 4 | 0 | 0.14
ising_2d | 9 | 2 | 64 | tebd_swaps | — | 1.303e-09 | 9.639e-01 | 8 | 60 | 0.04
ising_2d | 9 | 2 | 128 | enriched_tdvp | 1e-04 | 5.253e-07 | 9.639e-01 | 4 | 0 | 0.13
ising_2d | 9 | 2 | 128 | enriched_tdvp | 1e-06 | 5.253e-07 | 9.639e-01 | 4 | 0 | 0.13
ising_2d | 9 | 2 | 128 | tebd_swaps | — | 1.303e-09 | 9.639e-01 | 8 | 60 | 0.05
ising_2d | 9 | 2 | 256 | enriched_tdvp | 1e-04 | 5.253e-07 | 9.639e-01 | 4 | 0 | 0.13
ising_2d | 9 | 2 | 256 | enriched_tdvp | 1e-06 | 5.253e-07 | 9.639e-01 | 4 | 0 | 0.13
ising_2d | 9 | 2 | 256 | tebd_swaps | — | 1.303e-09 | 9.639e-01 | 8 | 60 | 0.04
ising_2d | 9 | 2 | — | enriched_tdvp | 1e-04 | 8.224e-03 | 4.977e-01 | 15 | 0 | 0.15
ising_2d | 9 | 2 | — | enriched_tdvp | 1e-06 | 8.224e-03 | 4.977e-01 | 15 | 0 | 0.13
ising_2d | 9 | 2 | — | tebd_swaps | — | 0.000e+00 | 5.000e-01 | 16 | 60 | 0.05
ising_2d | 9 | 2 | 16 | enriched_tdvp | 1e-04 | 8.224e-03 | 4.977e-01 | 9 | 0 | 0.13
ising_2d | 9 | 2 | 16 | enriched_tdvp | 1e-06 | 8.224e-03 | 4.977e-01 | 9 | 0 | 0.13
ising_2d | 9 | 2 | 16 | tebd_swaps | — | 1.403e-09 | 5.000e-01 | 16 | 60 | 0.05
ising_2d | 9 | 2 | 32 | enriched_tdvp | 1e-04 | 8.224e-03 | 4.977e-01 | 9 | 0 | 0.13
ising_2d | 9 | 2 | 32 | enriched_tdvp | 1e-06 | 8.224e-03 | 4.977e-01 | 9 | 0 | 0.12
ising_2d | 9 | 2 | 32 | tebd_swaps | — | 1.403e-09 | 5.000e-01 | 16 | 60 | 0.04
ising_2d | 9 | 2 | 64 | enriched_tdvp | 1e-04 | 8.224e-03 | 4.977e-01 | 9 | 0 | 0.12
ising_2d | 9 | 2 | 64 | enriched_tdvp | 1e-06 | 8.224e-03 | 4.977e-01 | 9 | 0 | 0.14
ising_2d | 9 | 2 | 64 | tebd_swaps | — | 1.403e-09 | 5.000e-01 | 16 | 60 | 0.05
ising_2d | 9 | 2 | 128 | enriched_tdvp | 1e-04 | 8.224e-03 | 4.977e-01 | 9 | 0 | 0.14
ising_2d | 9 | 2 | 128 | enriched_tdvp | 1e-06 | 8.224e-03 | 4.977e-01 | 9 | 0 | 0.14
ising_2d | 9 | 2 | 128 | tebd_swaps | — | 1.403e-09 | 5.000e-01 | 16 | 60 | 0.04
ising_2d | 9 | 2 | 256 | enriched_tdvp | 1e-04 | 8.224e-03 | 4.977e-01 | 9 | 0 | 0.15
ising_2d | 9 | 2 | 256 | enriched_tdvp | 1e-06 | 8.224e-03 | 4.977e-01 | 9 | 0 | 0.15
ising_2d | 9 | 2 | 256 | tebd_swaps | — | 1.403e-09 | 5.000e-01 | 16 | 60 | 0.06
ising_2d | 9 | 4 | — | enriched_tdvp | 1e-04 | 7.171e-07 | 7.986e-03 | 11 | 0 | 0.36
ising_2d | 9 | 4 | — | enriched_tdvp | 1e-06 | 3.805e-05 | 7.945e-03 | 10 | 0 | 0.33
ising_2d | 9 | 4 | — | tebd_swaps | — | 1.245e-12 | 8.038e-03 | 16 | 120 | 0.09
ising_2d | 9 | 4 | 16 | enriched_tdvp | 1e-04 | 7.949e-07 | 7.980e-03 | 7 | 0 | 0.36
ising_2d | 9 | 4 | 16 | enriched_tdvp | 1e-06 | 2.536e-04 | 1.800e-02 | 7 | 0 | 0.37
ising_2d | 9 | 4 | 16 | tebd_swaps | — | 4.944e-09 | 8.038e-03 | 16 | 120 | 0.11
ising_2d | 9 | 4 | 32 | enriched_tdvp | 1e-04 | 7.949e-07 | 7.980e-03 | 7 | 0 | 0.39
ising_2d | 9 | 4 | 32 | enriched_tdvp | 1e-06 | 2.536e-04 | 1.800e-02 | 7 | 0 | 0.31
ising_2d | 9 | 4 | 32 | tebd_swaps | — | 4.944e-09 | 8.038e-03 | 16 | 120 | 0.08
ising_2d | 9 | 4 | 64 | enriched_tdvp | 1e-04 | 7.949e-07 | 7.980e-03 | 7 | 0 | 0.36
ising_2d | 9 | 4 | 64 | enriched_tdvp | 1e-06 | 2.536e-04 | 1.800e-02 | 7 | 0 | 0.36
ising_2d | 9 | 4 | 64 | tebd_swaps | — | 4.944e-09 | 8.038e-03 | 16 | 120 | 0.13
ising_2d | 9 | 4 | 128 | enriched_tdvp | 1e-04 | 7.949e-07 | 7.980e-03 | 7 | 0 | 0.43
ising_2d | 9 | 4 | 128 | enriched_tdvp | 1e-06 | 2.536e-04 | 1.800e-02 | 7 | 0 | 0.50
ising_2d | 9 | 4 | 128 | tebd_swaps | — | 4.944e-09 | 8.038e-03 | 16 | 120 | 0.14
ising_2d | 9 | 4 | 256 | enriched_tdvp | 1e-04 | 7.949e-07 | 7.980e-03 | 7 | 0 | 0.46
ising_2d | 9 | 4 | 256 | enriched_tdvp | 1e-06 | 2.536e-04 | 1.800e-02 | 7 | 0 | 0.33
ising_2d | 9 | 4 | 256 | tebd_swaps | — | 4.944e-09 | 8.038e-03 | 16 | 120 | 0.11
ising_2d | 9 | 4 | — | enriched_tdvp | 1e-04 | 1.795e-02 | 2.172e-01 | 16 | 0 | 0.43
ising_2d | 9 | 4 | — | enriched_tdvp | 1e-06 | 1.795e-02 | 2.172e-01 | 16 | 0 | 0.49
ising_2d | 9 | 4 | — | tebd_swaps | — | 0.000e+00 | 1.885e-01 | 16 | 120 | 0.11
ising_2d | 9 | 4 | 16 | enriched_tdvp | 1e-04 | 2.874e-02 | 2.099e-01 | 16 | 0 | 0.46
ising_2d | 9 | 4 | 16 | enriched_tdvp | 1e-06 | 2.874e-02 | 2.099e-01 | 16 | 0 | 0.39
ising_2d | 9 | 4 | 16 | tebd_swaps | — | 1.764e-09 | 1.885e-01 | 16 | 120 | 0.13
ising_2d | 9 | 4 | 32 | enriched_tdvp | 1e-04 | 2.874e-02 | 2.099e-01 | 16 | 0 | 0.47
ising_2d | 9 | 4 | 32 | enriched_tdvp | 1e-06 | 2.874e-02 | 2.099e-01 | 16 | 0 | 0.45
ising_2d | 9 | 4 | 32 | tebd_swaps | — | 1.764e-09 | 1.885e-01 | 16 | 120 | 0.13
ising_2d | 9 | 4 | 64 | enriched_tdvp | 1e-04 | 2.874e-02 | 2.099e-01 | 16 | 0 | 0.43
ising_2d | 9 | 4 | 64 | enriched_tdvp | 1e-06 | 2.874e-02 | 2.099e-01 | 16 | 0 | 0.41
ising_2d | 9 | 4 | 64 | tebd_swaps | — | 1.764e-09 | 1.885e-01 | 16 | 120 | 0.14
ising_2d | 9 | 4 | 128 | enriched_tdvp | 1e-04 | 2.874e-02 | 2.099e-01 | 16 | 0 | 0.47
ising_2d | 9 | 4 | 128 | enriched_tdvp | 1e-06 | 2.874e-02 | 2.099e-01 | 16 | 0 | 0.45
ising_2d | 9 | 4 | 128 | tebd_swaps | — | 1.764e-09 | 1.885e-01 | 16 | 120 | 0.13
ising_2d | 9 | 4 | 256 | enriched_tdvp | 1e-04 | 2.874e-02 | 2.099e-01 | 16 | 0 | 0.43
ising_2d | 9 | 4 | 256 | enriched_tdvp | 1e-06 | 2.874e-02 | 2.099e-01 | 16 | 0 | 0.37
ising_2d | 9 | 4 | 256 | tebd_swaps | — | 1.764e-09 | 1.885e-01 | 16 | 120 | 0.11
ising_2d | 9 | 4 | — | enriched_tdvp | 1e-04 | 5.575e-07 | 9.086e-01 | 10 | 0 | 0.41
ising_2d | 9 | 4 | — | enriched_tdvp | 1e-06 | 9.633e-06 | 9.074e-01 | 10 | 0 | 0.35
ising_2d | 9 | 4 | — | tebd_swaps | — | 4.796e-12 | 9.086e-01 | 16 | 120 | 0.10
ising_2d | 9 | 4 | 16 | enriched_tdvp | 1e-04 | 1.542e-06 | 9.085e-01 | 7 | 0 | 0.39
ising_2d | 9 | 4 | 16 | enriched_tdvp | 1e-06 | 1.422e-04 | 9.076e-01 | 7 | 0 | 0.32
ising_2d | 9 | 4 | 16 | tebd_swaps | — | 6.780e-09 | 9.086e-01 | 16 | 120 | 0.10
ising_2d | 9 | 4 | 32 | enriched_tdvp | 1e-04 | 1.542e-06 | 9.085e-01 | 7 | 0 | 0.39
ising_2d | 9 | 4 | 32 | enriched_tdvp | 1e-06 | 1.422e-04 | 9.076e-01 | 7 | 0 | 0.32
ising_2d | 9 | 4 | 32 | tebd_swaps | — | 6.780e-09 | 9.086e-01 | 16 | 120 | 0.10
ising_2d | 9 | 4 | 64 | enriched_tdvp | 1e-04 | 1.542e-06 | 9.085e-01 | 7 | 0 | 0.38
ising_2d | 9 | 4 | 64 | enriched_tdvp | 1e-06 | 1.422e-04 | 9.076e-01 | 7 | 0 | 0.32
ising_2d | 9 | 4 | 64 | tebd_swaps | — | 6.780e-09 | 9.086e-01 | 16 | 120 | 0.10
ising_2d | 9 | 4 | 128 | enriched_tdvp | 1e-04 | 1.542e-06 | 9.085e-01 | 7 | 0 | 0.43
ising_2d | 9 | 4 | 128 | enriched_tdvp | 1e-06 | 1.422e-04 | 9.076e-01 | 7 | 0 | 0.35
ising_2d | 9 | 4 | 128 | tebd_swaps | — | 6.780e-09 | 9.086e-01 | 16 | 120 | 0.11
ising_2d | 9 | 4 | 256 | enriched_tdvp | 1e-04 | 1.542e-06 | 9.085e-01 | 7 | 0 | 0.46
ising_2d | 9 | 4 | 256 | enriched_tdvp | 1e-06 | 1.422e-04 | 9.076e-01 | 7 | 0 | 0.33
ising_2d | 9 | 4 | 256 | tebd_swaps | — | 6.780e-09 | 9.086e-01 | 16 | 120 | 0.10
ising_2d | 9 | 4 | — | enriched_tdvp | 1e-04 | 4.223e-02 | 4.471e-01 | 16 | 0 | 0.39
ising_2d | 9 | 4 | — | enriched_tdvp | 1e-06 | 4.223e-02 | 4.471e-01 | 16 | 0 | 0.40
ising_2d | 9 | 4 | — | tebd_swaps | — | 0.000e+00 | 4.447e-01 | 16 | 120 | 0.11
ising_2d | 9 | 4 | 16 | enriched_tdvp | 1e-04 | 4.221e-02 | 4.470e-01 | 16 | 0 | 0.38
ising_2d | 9 | 4 | 16 | enriched_tdvp | 1e-06 | 4.221e-02 | 4.470e-01 | 16 | 0 | 0.38
ising_2d | 9 | 4 | 16 | tebd_swaps | — | 1.403e-09 | 4.447e-01 | 16 | 120 | 0.11
ising_2d | 9 | 4 | 32 | enriched_tdvp | 1e-04 | 4.221e-02 | 4.470e-01 | 16 | 0 | 0.37
ising_2d | 9 | 4 | 32 | enriched_tdvp | 1e-06 | 4.221e-02 | 4.470e-01 | 16 | 0 | 0.37
ising_2d | 9 | 4 | 32 | tebd_swaps | — | 1.403e-09 | 4.447e-01 | 16 | 120 | 0.11
ising_2d | 9 | 4 | 64 | enriched_tdvp | 1e-04 | 4.221e-02 | 4.470e-01 | 16 | 0 | 0.37
ising_2d | 9 | 4 | 64 | enriched_tdvp | 1e-06 | 4.221e-02 | 4.470e-01 | 16 | 0 | 0.37
ising_2d | 9 | 4 | 64 | tebd_swaps | — | 1.403e-09 | 4.447e-01 | 16 | 120 | 0.11
ising_2d | 9 | 4 | 128 | enriched_tdvp | 1e-04 | 4.221e-02 | 4.470e-01 | 16 | 0 | 0.37
ising_2d | 9 | 4 | 128 | enriched_tdvp | 1e-06 | 4.221e-02 | 4.470e-01 | 16 | 0 | 0.37
ising_2d | 9 | 4 | 128 | tebd_swaps | — | 1.403e-09 | 4.447e-01 | 16 | 120 | 0.11
ising_2d | 9 | 4 | 256 | enriched_tdvp | 1e-04 | 4.221e-02 | 4.470e-01 | 16 | 0 | 0.37
ising_2d | 9 | 4 | 256 | enriched_tdvp | 1e-06 | 4.221e-02 | 4.470e-01 | 16 | 0 | 0.37
ising_2d | 9 | 4 | 256 | tebd_swaps | — | 1.403e-09 | 4.447e-01 | 16 | 120 | 0.12
ising_2d | 9 | 2 | — | enriched_tdvp | 1e-04 | 8.295e-06 | 2.289e-03 | 7 | 0 | 0.17
ising_2d | 9 | 2 | — | enriched_tdvp | 1e-06 | 6.190e-07 | 2.369e-03 | 7 | 0 | 0.16
ising_2d | 9 | 2 | — | tebd_swaps | — | 0.000e+00 | 2.369e-03 | 8 | 60 | 0.05
ising_2d | 9 | 2 | 16 | enriched_tdvp | 1e-04 | 8.100e-07 | 2.363e-03 | 5 | 0 | 0.16
ising_2d | 9 | 2 | 16 | enriched_tdvp | 1e-06 | 6.206e-07 | 2.369e-03 | 5 | 0 | 0.16
ising_2d | 9 | 2 | 16 | tebd_swaps | — | 0.000e+00 | 2.369e-03 | 8 | 60 | 0.05
ising_2d | 9 | 2 | 32 | enriched_tdvp | 1e-04 | 8.100e-07 | 2.363e-03 | 5 | 0 | 0.16
ising_2d | 9 | 2 | 32 | enriched_tdvp | 1e-06 | 6.206e-07 | 2.369e-03 | 5 | 0 | 0.16
ising_2d | 9 | 2 | 32 | tebd_swaps | — | 0.000e+00 | 2.369e-03 | 8 | 60 | 0.05
ising_2d | 9 | 2 | 64 | enriched_tdvp | 1e-04 | 8.100e-07 | 2.363e-03 | 5 | 0 | 0.17
ising_2d | 9 | 2 | 64 | enriched_tdvp | 1e-06 | 6.206e-07 | 2.369e-03 | 5 | 0 | 0.15
ising_2d | 9 | 2 | 64 | tebd_swaps | — | 0.000e+00 | 2.369e-03 | 8 | 60 | 0.05
ising_2d | 9 | 2 | 128 | enriched_tdvp | 1e-04 | 8.100e-07 | 2.363e-03 | 5 | 0 | 0.18
ising_2d | 9 | 2 | 128 | enriched_tdvp | 1e-06 | 6.206e-07 | 2.369e-03 | 5 | 0 | 0.16
ising_2d | 9 | 2 | 128 | tebd_swaps | — | 0.000e+00 | 2.369e-03 | 8 | 60 | 0.05
ising_2d | 9 | 2 | 256 | enriched_tdvp | 1e-04 | 8.100e-07 | 2.363e-03 | 5 | 0 | 0.19
ising_2d | 9 | 2 | 256 | enriched_tdvp | 1e-06 | 6.206e-07 | 2.369e-03 | 5 | 0 | 0.17
ising_2d | 9 | 2 | 256 | tebd_swaps | — | 0.000e+00 | 2.369e-03 | 8 | 60 | 0.05
ising_2d | 9 | 2 | — | enriched_tdvp | 1e-04 | 9.649e-03 | 1.036e-01 | 16 | 0 | 0.19
ising_2d | 9 | 2 | — | enriched_tdvp | 1e-06 | 3.093e-06 | 9.295e-02 | 16 | 0 | 0.19
ising_2d | 9 | 2 | — | tebd_swaps | — | 0.000e+00 | 9.295e-02 | 16 | 60 | 0.06
ising_2d | 9 | 2 | 16 | enriched_tdvp | 1e-04 | 1.117e-06 | 9.289e-02 | 13 | 0 | 0.19
ising_2d | 9 | 2 | 16 | enriched_tdvp | 1e-06 | 1.117e-06 | 9.289e-02 | 13 | 0 | 0.16
ising_2d | 9 | 2 | 16 | tebd_swaps | — | 8.563e-10 | 9.295e-02 | 16 | 60 | 0.06
ising_2d | 9 | 2 | 32 | enriched_tdvp | 1e-04 | 1.117e-06 | 9.289e-02 | 13 | 0 | 0.16
ising_2d | 9 | 2 | 32 | enriched_tdvp | 1e-06 | 1.117e-06 | 9.289e-02 | 13 | 0 | 0.18
ising_2d | 9 | 2 | 32 | tebd_swaps | — | 8.563e-10 | 9.295e-02 | 16 | 60 | 0.06
ising_2d | 9 | 2 | 64 | enriched_tdvp | 1e-04 | 1.117e-06 | 9.289e-02 | 13 | 0 | 0.19
ising_2d | 9 | 2 | 64 | enriched_tdvp | 1e-06 | 1.117e-06 | 9.289e-02 | 13 | 0 | 0.19
ising_2d | 9 | 2 | 64 | tebd_swaps | — | 8.563e-10 | 9.295e-02 | 16 | 60 | 0.06
ising_2d | 9 | 2 | 128 | enriched_tdvp | 1e-04 | 1.117e-06 | 9.289e-02 | 13 | 0 | 0.19
ising_2d | 9 | 2 | 128 | enriched_tdvp | 1e-06 | 1.117e-06 | 9.289e-02 | 13 | 0 | 0.19
ising_2d | 9 | 2 | 128 | tebd_swaps | — | 8.563e-10 | 9.295e-02 | 16 | 60 | 0.06
ising_2d | 9 | 2 | 256 | enriched_tdvp | 1e-04 | 1.117e-06 | 9.289e-02 | 13 | 0 | 0.18
ising_2d | 9 | 2 | 256 | enriched_tdvp | 1e-06 | 1.117e-06 | 9.289e-02 | 13 | 0 | 0.19
ising_2d | 9 | 2 | 256 | tebd_swaps | — | 8.563e-10 | 9.295e-02 | 16 | 60 | 0.06
ising_2d | 9 | 2 | — | enriched_tdvp | 1e-04 | 5.016e-04 | 8.609e-01 | 7 | 0 | 0.19
ising_2d | 9 | 2 | — | enriched_tdvp | 1e-06 | 5.016e-04 | 8.609e-01 | 7 | 0 | 0.19
ising_2d | 9 | 2 | — | tebd_swaps | — | 1.720e-12 | 8.609e-01 | 8 | 60 | 0.06
ising_2d | 9 | 2 | 16 | enriched_tdvp | 1e-04 | 4.905e-04 | 8.609e-01 | 5 | 0 | 0.19
ising_2d | 9 | 2 | 16 | enriched_tdvp | 1e-06 | 4.909e-04 | 8.609e-01 | 5 | 0 | 0.18
ising_2d | 9 | 2 | 16 | tebd_swaps | — | 7.630e-11 | 8.609e-01 | 8 | 60 | 0.06
ising_2d | 9 | 2 | 32 | enriched_tdvp | 1e-04 | 4.905e-04 | 8.609e-01 | 5 | 0 | 0.20
ising_2d | 9 | 2 | 32 | enriched_tdvp | 1e-06 | 4.909e-04 | 8.609e-01 | 5 | 0 | 0.22
ising_2d | 9 | 2 | 32 | tebd_swaps | — | 7.630e-11 | 8.609e-01 | 8 | 60 | 0.06
ising_2d | 9 | 2 | 64 | enriched_tdvp | 1e-04 | 4.905e-04 | 8.609e-01 | 5 | 0 | 0.20
ising_2d | 9 | 2 | 64 | enriched_tdvp | 1e-06 | 4.909e-04 | 8.609e-01 | 5 | 0 | 0.20
ising_2d | 9 | 2 | 64 | tebd_swaps | — | 7.630e-11 | 8.609e-01 | 8 | 60 | 0.07
ising_2d | 9 | 2 | 128 | enriched_tdvp | 1e-04 | 4.905e-04 | 8.609e-01 | 5 | 0 | 0.20
ising_2d | 9 | 2 | 128 | enriched_tdvp | 1e-06 | 4.909e-04 | 8.609e-01 | 5 | 0 | 0.19
ising_2d | 9 | 2 | 128 | tebd_swaps | — | 7.630e-11 | 8.609e-01 | 8 | 60 | 0.06
ising_2d | 9 | 2 | 256 | enriched_tdvp | 1e-04 | 4.905e-04 | 8.609e-01 | 5 | 0 | 0.21
ising_2d | 9 | 2 | 256 | enriched_tdvp | 1e-06 | 4.909e-04 | 8.609e-01 | 5 | 0 | 0.19
ising_2d | 9 | 2 | 256 | tebd_swaps | — | 7.630e-11 | 8.609e-01 | 8 | 60 | 0.08
ising_2d | 9 | 2 | — | enriched_tdvp | 1e-04 | 8.393e-03 | 5.066e-01 | 16 | 0 | 0.30
ising_2d | 9 | 2 | — | enriched_tdvp | 1e-06 | 8.393e-03 | 5.066e-01 | 16 | 0 | 0.34
ising_2d | 9 | 2 | — | tebd_swaps | — | 0.000e+00 | 5.083e-01 | 16 | 60 | 0.07
ising_2d | 9 | 2 | 16 | enriched_tdvp | 1e-04 | 8.392e-03 | 5.066e-01 | 10 | 0 | 0.20
ising_2d | 9 | 2 | 16 | enriched_tdvp | 1e-06 | 8.392e-03 | 5.066e-01 | 10 | 0 | 0.19
ising_2d | 9 | 2 | 16 | tebd_swaps | — | 4.458e-10 | 5.083e-01 | 16 | 60 | 0.07
ising_2d | 9 | 2 | 32 | enriched_tdvp | 1e-04 | 8.392e-03 | 5.066e-01 | 10 | 0 | 0.19
ising_2d | 9 | 2 | 32 | enriched_tdvp | 1e-06 | 8.392e-03 | 5.066e-01 | 10 | 0 | 0.21
ising_2d | 9 | 2 | 32 | tebd_swaps | — | 4.458e-10 | 5.083e-01 | 16 | 60 | 0.07
ising_2d | 9 | 2 | 64 | enriched_tdvp | 1e-04 | 8.392e-03 | 5.066e-01 | 10 | 0 | 0.20
ising_2d | 9 | 2 | 64 | enriched_tdvp | 1e-06 | 8.392e-03 | 5.066e-01 | 10 | 0 | 0.20
ising_2d | 9 | 2 | 64 | tebd_swaps | — | 4.458e-10 | 5.083e-01 | 16 | 60 | 0.07
ising_2d | 9 | 2 | 128 | enriched_tdvp | 1e-04 | 8.392e-03 | 5.066e-01 | 10 | 0 | 0.21
ising_2d | 9 | 2 | 128 | enriched_tdvp | 1e-06 | 8.392e-03 | 5.066e-01 | 10 | 0 | 0.20
ising_2d | 9 | 2 | 128 | tebd_swaps | — | 4.458e-10 | 5.083e-01 | 16 | 60 | 0.07
ising_2d | 9 | 2 | 256 | enriched_tdvp | 1e-04 | 8.392e-03 | 5.066e-01 | 10 | 0 | 0.19
ising_2d | 9 | 2 | 256 | enriched_tdvp | 1e-06 | 8.392e-03 | 5.066e-01 | 10 | 0 | 0.20
ising_2d | 9 | 2 | 256 | tebd_swaps | — | 4.458e-10 | 5.083e-01 | 16 | 60 | 0.09
ising_2d | 9 | 4 | — | enriched_tdvp | 1e-04 | 5.164e-04 | 2.674e-02 | 15 | 0 | 0.46
ising_2d | 9 | 4 | — | enriched_tdvp | 1e-06 | 5.073e-04 | 2.753e-02 | 15 | 0 | 0.48
ising_2d | 9 | 4 | — | tebd_swaps | — | 4.781e-13 | 2.819e-02 | 16 | 120 | 0.13
ising_2d | 9 | 4 | 16 | enriched_tdvp | 1e-04 | 3.198e-04 | 2.783e-02 | 10 | 0 | 0.45
ising_2d | 9 | 4 | 16 | enriched_tdvp | 1e-06 | 2.013e-03 | 3.918e-02 | 9 | 0 | 0.50
ising_2d | 9 | 4 | 16 | tebd_swaps | — | 3.034e-09 | 2.819e-02 | 16 | 120 | 0.12
ising_2d | 9 | 4 | 32 | enriched_tdvp | 1e-04 | 3.198e-04 | 2.783e-02 | 10 | 0 | 0.44
ising_2d | 9 | 4 | 32 | enriched_tdvp | 1e-06 | 2.013e-03 | 3.918e-02 | 9 | 0 | 0.39
ising_2d | 9 | 4 | 32 | tebd_swaps | — | 3.034e-09 | 2.819e-02 | 16 | 120 | 0.12
ising_2d | 9 | 4 | 64 | enriched_tdvp | 1e-04 | 3.198e-04 | 2.783e-02 | 10 | 0 | 0.46
ising_2d | 9 | 4 | 64 | enriched_tdvp | 1e-06 | 2.013e-03 | 3.918e-02 | 9 | 0 | 0.41
ising_2d | 9 | 4 | 64 | tebd_swaps | — | 3.034e-09 | 2.819e-02 | 16 | 120 | 0.12
ising_2d | 9 | 4 | 128 | enriched_tdvp | 1e-04 | 3.198e-04 | 2.783e-02 | 10 | 0 | 0.44
ising_2d | 9 | 4 | 128 | enriched_tdvp | 1e-06 | 2.013e-03 | 3.918e-02 | 9 | 0 | 0.45
ising_2d | 9 | 4 | 128 | tebd_swaps | — | 3.034e-09 | 2.819e-02 | 16 | 120 | 0.28
ising_2d | 9 | 4 | 256 | enriched_tdvp | 1e-04 | 3.198e-04 | 2.783e-02 | 10 | 0 | 0.61
ising_2d | 9 | 4 | 256 | enriched_tdvp | 1e-06 | 2.013e-03 | 3.918e-02 | 9 | 0 | 0.46
ising_2d | 9 | 4 | 256 | tebd_swaps | — | 3.034e-09 | 2.819e-02 | 16 | 120 | 0.28
ising_2d | 9 | 4 | — | enriched_tdvp | 1e-04 | 4.489e-02 | 1.911e-01 | 16 | 0 | 1.00
ising_2d | 9 | 4 | — | enriched_tdvp | 1e-06 | 1.837e-02 | 1.904e-01 | 16 | 0 | 0.78
ising_2d | 9 | 4 | — | tebd_swaps | — | 0.000e+00 | 1.614e-01 | 16 | 120 | 0.28
ising_2d | 9 | 4 | 16 | enriched_tdvp | 1e-04 | 1.834e-02 | 1.902e-01 | 16 | 0 | 0.52
ising_2d | 9 | 4 | 16 | enriched_tdvp | 1e-06 | 1.834e-02 | 1.902e-01 | 16 | 0 | 0.64
ising_2d | 9 | 4 | 16 | tebd_swaps | — | 1.043e-09 | 1.614e-01 | 16 | 120 | 0.18
ising_2d | 9 | 4 | 32 | enriched_tdvp | 1e-04 | 1.834e-02 | 1.902e-01 | 16 | 0 | 0.52
ising_2d | 9 | 4 | 32 | enriched_tdvp | 1e-06 | 1.834e-02 | 1.902e-01 | 16 | 0 | 0.59
ising_2d | 9 | 4 | 32 | tebd_swaps | — | 1.043e-09 | 1.614e-01 | 16 | 120 | 0.16
ising_2d | 9 | 4 | 64 | enriched_tdvp | 1e-04 | 1.834e-02 | 1.902e-01 | 16 | 0 | 0.61
ising_2d | 9 | 4 | 64 | enriched_tdvp | 1e-06 | 1.834e-02 | 1.902e-01 | 16 | 0 | 0.55
ising_2d | 9 | 4 | 64 | tebd_swaps | — | 1.043e-09 | 1.614e-01 | 16 | 120 | 0.14
ising_2d | 9 | 4 | 128 | enriched_tdvp | 1e-04 | 1.834e-02 | 1.902e-01 | 16 | 0 | 0.53
ising_2d | 9 | 4 | 128 | enriched_tdvp | 1e-06 | 1.834e-02 | 1.902e-01 | 16 | 0 | 0.47
ising_2d | 9 | 4 | 128 | tebd_swaps | — | 1.043e-09 | 1.614e-01 | 16 | 120 | 0.14
ising_2d | 9 | 4 | 256 | enriched_tdvp | 1e-04 | 1.834e-02 | 1.902e-01 | 16 | 0 | 0.56
ising_2d | 9 | 4 | 256 | enriched_tdvp | 1e-06 | 1.834e-02 | 1.902e-01 | 16 | 0 | 0.51
ising_2d | 9 | 4 | 256 | tebd_swaps | — | 1.043e-09 | 1.614e-01 | 16 | 120 | 0.15
ising_2d | 9 | 4 | — | enriched_tdvp | 1e-04 | 4.869e-04 | 6.608e-01 | 15 | 0 | 0.52
ising_2d | 9 | 4 | — | enriched_tdvp | 1e-06 | 4.869e-04 | 6.608e-01 | 15 | 0 | 0.55
ising_2d | 9 | 4 | — | tebd_swaps | — | 1.949e-12 | 6.729e-01 | 16 | 120 | 0.13
ising_2d | 9 | 4 | 16 | enriched_tdvp | 1e-04 | 4.312e-04 | 6.627e-01 | 9 | 0 | 0.50
ising_2d | 9 | 4 | 16 | enriched_tdvp | 1e-06 | 1.227e-03 | 6.622e-01 | 9 | 0 | 0.42
ising_2d | 9 | 4 | 16 | tebd_swaps | — | 4.024e-09 | 6.729e-01 | 16 | 120 | 0.11
ising_2d | 9 | 4 | 32 | enriched_tdvp | 1e-04 | 4.312e-04 | 6.627e-01 | 9 | 0 | 0.72
ising_2d | 9 | 4 | 32 | enriched_tdvp | 1e-06 | 1.227e-03 | 6.622e-01 | 9 | 0 | 0.61
ising_2d | 9 | 4 | 32 | tebd_swaps | — | 4.024e-09 | 6.729e-01 | 16 | 120 | 0.14
ising_2d | 9 | 4 | 64 | enriched_tdvp | 1e-04 | 4.312e-04 | 6.627e-01 | 9 | 0 | 0.55
ising_2d | 9 | 4 | 64 | enriched_tdvp | 1e-06 | 1.227e-03 | 6.622e-01 | 9 | 0 | 0.56
ising_2d | 9 | 4 | 64 | tebd_swaps | — | 4.024e-09 | 6.729e-01 | 16 | 120 | 0.15
ising_2d | 9 | 4 | 128 | enriched_tdvp | 1e-04 | 4.312e-04 | 6.627e-01 | 9 | 0 | 0.47
ising_2d | 9 | 4 | 128 | enriched_tdvp | 1e-06 | 1.227e-03 | 6.622e-01 | 9 | 0 | 0.42
ising_2d | 9 | 4 | 128 | tebd_swaps | — | 4.024e-09 | 6.729e-01 | 16 | 120 | 0.14
ising_2d | 9 | 4 | 256 | enriched_tdvp | 1e-04 | 4.312e-04 | 6.627e-01 | 9 | 0 | 0.53
ising_2d | 9 | 4 | 256 | enriched_tdvp | 1e-06 | 1.227e-03 | 6.622e-01 | 9 | 0 | 0.43
ising_2d | 9 | 4 | 256 | tebd_swaps | — | 4.024e-09 | 6.729e-01 | 16 | 120 | 0.16
ising_2d | 9 | 4 | — | enriched_tdvp | 1e-04 | 4.104e-02 | 4.566e-01 | 16 | 0 | 0.63
ising_2d | 9 | 4 | — | enriched_tdvp | 1e-06 | 4.104e-02 | 4.566e-01 | 16 | 0 | 0.50
ising_2d | 9 | 4 | — | tebd_swaps | — | 0.000e+00 | 4.457e-01 | 16 | 120 | 0.17
ising_2d | 9 | 4 | 16 | enriched_tdvp | 1e-04 | 4.103e-02 | 4.566e-01 | 16 | 0 | 0.58
ising_2d | 9 | 4 | 16 | enriched_tdvp | 1e-06 | 4.103e-02 | 4.566e-01 | 16 | 0 | 0.59
ising_2d | 9 | 4 | 16 | tebd_swaps | — | 4.458e-10 | 4.457e-01 | 16 | 120 | 0.17
ising_2d | 9 | 4 | 32 | enriched_tdvp | 1e-04 | 4.103e-02 | 4.566e-01 | 16 | 0 | 0.56
ising_2d | 9 | 4 | 32 | enriched_tdvp | 1e-06 | 4.103e-02 | 4.566e-01 | 16 | 0 | 0.54
ising_2d | 9 | 4 | 32 | tebd_swaps | — | 4.458e-10 | 4.457e-01 | 16 | 120 | 0.15
ising_2d | 9 | 4 | 64 | enriched_tdvp | 1e-04 | 4.103e-02 | 4.566e-01 | 16 | 0 | 0.52
ising_2d | 9 | 4 | 64 | enriched_tdvp | 1e-06 | 4.103e-02 | 4.566e-01 | 16 | 0 | 0.51
ising_2d | 9 | 4 | 64 | tebd_swaps | — | 4.458e-10 | 4.457e-01 | 16 | 120 | 0.14
ising_2d | 9 | 4 | 128 | enriched_tdvp | 1e-04 | 4.103e-02 | 4.566e-01 | 16 | 0 | 0.63
ising_2d | 9 | 4 | 128 | enriched_tdvp | 1e-06 | 4.103e-02 | 4.566e-01 | 16 | 0 | 0.59
ising_2d | 9 | 4 | 128 | tebd_swaps | — | 4.458e-10 | 4.457e-01 | 16 | 120 | 0.18
ising_2d | 9 | 4 | 256 | enriched_tdvp | 1e-04 | 4.103e-02 | 4.566e-01 | 16 | 0 | 0.54
ising_2d | 9 | 4 | 256 | enriched_tdvp | 1e-06 | 4.103e-02 | 4.566e-01 | 16 | 0 | 0.55
ising_2d | 9 | 4 | 256 | tebd_swaps | — | 4.458e-10 | 4.457e-01 | 16 | 120 | 0.14
ising_2d | 9 | 2 | — | enriched_tdvp | 1e-04 | 3.643e-05 | 7.393e-03 | 8 | 0 | 0.39
ising_2d | 9 | 2 | — | enriched_tdvp | 1e-06 | 3.643e-05 | 7.393e-03 | 8 | 0 | 0.22
ising_2d | 9 | 2 | — | tebd_swaps | — | 0.000e+00 | 7.424e-03 | 8 | 60 | 0.07
ising_2d | 9 | 2 | 16 | enriched_tdvp | 1e-04 | 3.587e-05 | 7.393e-03 | 7 | 0 | 0.28
ising_2d | 9 | 2 | 16 | enriched_tdvp | 1e-06 | 3.587e-05 | 7.393e-03 | 7 | 0 | 0.21
ising_2d | 9 | 2 | 16 | tebd_swaps | — | 2.127e-09 | 7.424e-03 | 8 | 60 | 0.06
ising_2d | 9 | 2 | 32 | enriched_tdvp | 1e-04 | 3.587e-05 | 7.393e-03 | 7 | 0 | 0.22
ising_2d | 9 | 2 | 32 | enriched_tdvp | 1e-06 | 3.587e-05 | 7.393e-03 | 7 | 0 | 0.20
ising_2d | 9 | 2 | 32 | tebd_swaps | — | 2.127e-09 | 7.424e-03 | 8 | 60 | 0.06
ising_2d | 9 | 2 | 64 | enriched_tdvp | 1e-04 | 3.587e-05 | 7.393e-03 | 7 | 0 | 0.21
ising_2d | 9 | 2 | 64 | enriched_tdvp | 1e-06 | 3.587e-05 | 7.393e-03 | 7 | 0 | 0.24
ising_2d | 9 | 2 | 64 | tebd_swaps | — | 2.127e-09 | 7.424e-03 | 8 | 60 | 0.07
ising_2d | 9 | 2 | 128 | enriched_tdvp | 1e-04 | 3.587e-05 | 7.393e-03 | 7 | 0 | 0.22
ising_2d | 9 | 2 | 128 | enriched_tdvp | 1e-06 | 3.587e-05 | 7.393e-03 | 7 | 0 | 0.22
ising_2d | 9 | 2 | 128 | tebd_swaps | — | 2.127e-09 | 7.424e-03 | 8 | 60 | 0.07
ising_2d | 9 | 2 | 256 | enriched_tdvp | 1e-04 | 3.587e-05 | 7.393e-03 | 7 | 0 | 0.23
ising_2d | 9 | 2 | 256 | enriched_tdvp | 1e-06 | 3.587e-05 | 7.393e-03 | 7 | 0 | 0.34
ising_2d | 9 | 2 | 256 | tebd_swaps | — | 2.127e-09 | 7.424e-03 | 8 | 60 | 0.10
ising_2d | 9 | 2 | — | enriched_tdvp | 1e-04 | 1.279e-05 | 9.148e-02 | 16 | 0 | 0.30
ising_2d | 9 | 2 | — | enriched_tdvp | 1e-06 | 1.279e-05 | 9.148e-02 | 16 | 0 | 0.27
ising_2d | 9 | 2 | — | tebd_swaps | — | 0.000e+00 | 9.145e-02 | 16 | 60 | 0.09
ising_2d | 9 | 2 | 16 | enriched_tdvp | 1e-04 | 6.790e-06 | 9.145e-02 | 14 | 0 | 0.20
ising_2d | 9 | 2 | 16 | enriched_tdvp | 1e-06 | 6.790e-06 | 9.145e-02 | 14 | 0 | 0.20
ising_2d | 9 | 2 | 16 | tebd_swaps | — | 0.000e+00 | 9.145e-02 | 16 | 60 | 0.07
ising_2d | 9 | 2 | 32 | enriched_tdvp | 1e-04 | 6.790e-06 | 9.145e-02 | 14 | 0 | 0.23
ising_2d | 9 | 2 | 32 | enriched_tdvp | 1e-06 | 6.790e-06 | 9.145e-02 | 14 | 0 | 0.23
ising_2d | 9 | 2 | 32 | tebd_swaps | — | 0.000e+00 | 9.145e-02 | 16 | 60 | 0.10
ising_2d | 9 | 2 | 64 | enriched_tdvp | 1e-04 | 6.790e-06 | 9.145e-02 | 14 | 0 | 0.22
ising_2d | 9 | 2 | 64 | enriched_tdvp | 1e-06 | 6.790e-06 | 9.145e-02 | 14 | 0 | 0.24
ising_2d | 9 | 2 | 64 | tebd_swaps | — | 0.000e+00 | 9.145e-02 | 16 | 60 | 0.07
ising_2d | 9 | 2 | 128 | enriched_tdvp | 1e-04 | 6.790e-06 | 9.145e-02 | 14 | 0 | 0.27
ising_2d | 9 | 2 | 128 | enriched_tdvp | 1e-06 | 6.790e-06 | 9.145e-02 | 14 | 0 | 0.34
ising_2d | 9 | 2 | 128 | tebd_swaps | — | 0.000e+00 | 9.145e-02 | 16 | 60 | 0.09
ising_2d | 9 | 2 | 256 | enriched_tdvp | 1e-04 | 6.790e-06 | 9.145e-02 | 14 | 0 | 0.21
ising_2d | 9 | 2 | 256 | enriched_tdvp | 1e-06 | 6.790e-06 | 9.145e-02 | 14 | 0 | 0.28
ising_2d | 9 | 2 | 256 | tebd_swaps | — | 0.000e+00 | 9.145e-02 | 16 | 60 | 0.07
ising_2d | 9 | 2 | — | enriched_tdvp | 1e-04 | 0.000e+00 | 5.228e-01 | 8 | 0 | 0.23
ising_2d | 9 | 2 | — | enriched_tdvp | 1e-06 | 0.000e+00 | 5.228e-01 | 8 | 0 | 0.23
ising_2d | 9 | 2 | — | tebd_swaps | — | 0.000e+00 | 5.228e-01 | 8 | 60 | 0.08
ising_2d | 9 | 2 | 16 | enriched_tdvp | 1e-04 | 2.720e-07 | 5.228e-01 | 7 | 0 | 0.25
ising_2d | 9 | 2 | 16 | enriched_tdvp | 1e-06 | 2.720e-07 | 5.228e-01 | 7 | 0 | 0.19
ising_2d | 9 | 2 | 16 | tebd_swaps | — | 1.996e-10 | 5.228e-01 | 8 | 60 | 0.07
ising_2d | 9 | 2 | 32 | enriched_tdvp | 1e-04 | 2.720e-07 | 5.228e-01 | 7 | 0 | 0.26
ising_2d | 9 | 2 | 32 | enriched_tdvp | 1e-06 | 2.720e-07 | 5.228e-01 | 7 | 0 | 0.21
ising_2d | 9 | 2 | 32 | tebd_swaps | — | 1.996e-10 | 5.228e-01 | 8 | 60 | 0.09
ising_2d | 9 | 2 | 64 | enriched_tdvp | 1e-04 | 2.720e-07 | 5.228e-01 | 7 | 0 | 0.22
ising_2d | 9 | 2 | 64 | enriched_tdvp | 1e-06 | 2.720e-07 | 5.228e-01 | 7 | 0 | 0.24
ising_2d | 9 | 2 | 64 | tebd_swaps | — | 1.996e-10 | 5.228e-01 | 8 | 60 | 0.06
ising_2d | 9 | 2 | 128 | enriched_tdvp | 1e-04 | 2.720e-07 | 5.228e-01 | 7 | 0 | 0.26
ising_2d | 9 | 2 | 128 | enriched_tdvp | 1e-06 | 2.720e-07 | 5.228e-01 | 7 | 0 | 0.22
ising_2d | 9 | 2 | 128 | tebd_swaps | — | 1.996e-10 | 5.228e-01 | 8 | 60 | 0.07
ising_2d | 9 | 2 | 256 | enriched_tdvp | 1e-04 | 2.720e-07 | 5.228e-01 | 7 | 0 | 0.26
ising_2d | 9 | 2 | 256 | enriched_tdvp | 1e-06 | 2.720e-07 | 5.228e-01 | 7 | 0 | 0.25
ising_2d | 9 | 2 | 256 | tebd_swaps | — | 1.996e-10 | 5.228e-01 | 8 | 60 | 0.07
ising_2d | 9 | 2 | — | enriched_tdvp | 1e-04 | 8.497e-03 | 4.913e-01 | 16 | 0 | 0.29
ising_2d | 9 | 2 | — | enriched_tdvp | 1e-06 | 8.497e-03 | 4.913e-01 | 16 | 0 | 0.21
ising_2d | 9 | 2 | — | tebd_swaps | — | 0.000e+00 | 4.906e-01 | 16 | 60 | 0.07
ising_2d | 9 | 2 | 16 | enriched_tdvp | 1e-04 | 8.496e-03 | 4.914e-01 | 14 | 0 | 0.27
ising_2d | 9 | 2 | 16 | enriched_tdvp | 1e-06 | 8.496e-03 | 4.914e-01 | 14 | 0 | 0.21
ising_2d | 9 | 2 | 16 | tebd_swaps | — | 0.000e+00 | 4.906e-01 | 16 | 60 | 0.10
ising_2d | 9 | 2 | 32 | enriched_tdvp | 1e-04 | 8.496e-03 | 4.914e-01 | 14 | 0 | 0.24
ising_2d | 9 | 2 | 32 | enriched_tdvp | 1e-06 | 8.496e-03 | 4.914e-01 | 14 | 0 | 0.39
ising_2d | 9 | 2 | 32 | tebd_swaps | — | 0.000e+00 | 4.906e-01 | 16 | 60 | 0.07
ising_2d | 9 | 2 | 64 | enriched_tdvp | 1e-04 | 8.496e-03 | 4.914e-01 | 14 | 0 | 0.20
ising_2d | 9 | 2 | 64 | enriched_tdvp | 1e-06 | 8.496e-03 | 4.914e-01 | 14 | 0 | 0.19
ising_2d | 9 | 2 | 64 | tebd_swaps | — | 0.000e+00 | 4.906e-01 | 16 | 60 | 0.07
ising_2d | 9 | 2 | 128 | enriched_tdvp | 1e-04 | 8.496e-03 | 4.914e-01 | 14 | 0 | 0.18
ising_2d | 9 | 2 | 128 | enriched_tdvp | 1e-06 | 8.496e-03 | 4.914e-01 | 14 | 0 | 0.18
ising_2d | 9 | 2 | 128 | tebd_swaps | — | 0.000e+00 | 4.906e-01 | 16 | 60 | 0.06
ising_2d | 9 | 2 | 256 | enriched_tdvp | 1e-04 | 8.496e-03 | 4.914e-01 | 14 | 0 | 0.17
ising_2d | 9 | 2 | 256 | enriched_tdvp | 1e-06 | 8.496e-03 | 4.914e-01 | 14 | 0 | 0.18
ising_2d | 9 | 2 | 256 | tebd_swaps | — | 0.000e+00 | 4.906e-01 | 16 | 60 | 0.06
ising_2d | 9 | 4 | — | enriched_tdvp | 1e-04 | 1.965e-03 | 6.708e-02 | 16 | 0 | 0.41
ising_2d | 9 | 4 | — | enriched_tdvp | 1e-06 | 1.965e-03 | 6.708e-02 | 16 | 0 | 0.39
ising_2d | 9 | 4 | — | tebd_swaps | — | 0.000e+00 | 6.705e-02 | 16 | 120 | 0.11
ising_2d | 9 | 4 | 16 | enriched_tdvp | 1e-04 | 2.095e-03 | 6.748e-02 | 15 | 0 | 0.39
ising_2d | 9 | 4 | 16 | enriched_tdvp | 1e-06 | 2.095e-03 | 6.748e-02 | 15 | 0 | 0.39
ising_2d | 9 | 4 | 16 | tebd_swaps | — | 3.458e-09 | 6.705e-02 | 16 | 120 | 0.10
ising_2d | 9 | 4 | 32 | enriched_tdvp | 1e-04 | 2.095e-03 | 6.748e-02 | 15 | 0 | 0.41
ising_2d | 9 | 4 | 32 | enriched_tdvp | 1e-06 | 2.095e-03 | 6.748e-02 | 15 | 0 | 0.39
ising_2d | 9 | 4 | 32 | tebd_swaps | — | 3.458e-09 | 6.705e-02 | 16 | 120 | 0.11
ising_2d | 9 | 4 | 64 | enriched_tdvp | 1e-04 | 2.095e-03 | 6.748e-02 | 15 | 0 | 0.40
ising_2d | 9 | 4 | 64 | enriched_tdvp | 1e-06 | 2.095e-03 | 6.748e-02 | 15 | 0 | 0.42
ising_2d | 9 | 4 | 64 | tebd_swaps | — | 3.458e-09 | 6.705e-02 | 16 | 120 | 0.12
ising_2d | 9 | 4 | 128 | enriched_tdvp | 1e-04 | 2.095e-03 | 6.748e-02 | 15 | 0 | 0.42
ising_2d | 9 | 4 | 128 | enriched_tdvp | 1e-06 | 2.095e-03 | 6.748e-02 | 15 | 0 | 0.43
ising_2d | 9 | 4 | 128 | tebd_swaps | — | 3.458e-09 | 6.705e-02 | 16 | 120 | 0.26
ising_2d | 9 | 4 | 256 | enriched_tdvp | 1e-04 | 2.095e-03 | 6.748e-02 | 15 | 0 | 0.49
ising_2d | 9 | 4 | 256 | enriched_tdvp | 1e-06 | 2.095e-03 | 6.748e-02 | 15 | 0 | 0.50
ising_2d | 9 | 4 | 256 | tebd_swaps | — | 3.458e-09 | 6.705e-02 | 16 | 120 | 0.13
ising_2d | 9 | 4 | — | enriched_tdvp | 1e-04 | 1.715e-02 | 1.492e-01 | 16 | 0 | 0.49
ising_2d | 9 | 4 | — | enriched_tdvp | 1e-06 | 1.715e-02 | 1.492e-01 | 16 | 0 | 0.49
ising_2d | 9 | 4 | — | tebd_swaps | — | 0.000e+00 | 1.281e-01 | 16 | 120 | 0.29
ising_2d | 9 | 4 | 16 | enriched_tdvp | 1e-04 | 1.713e-02 | 1.490e-01 | 16 | 0 | 0.48
ising_2d | 9 | 4 | 16 | enriched_tdvp | 1e-06 | 1.713e-02 | 1.490e-01 | 16 | 0 | 0.71
ising_2d | 9 | 4 | 16 | tebd_swaps | — | 0.000e+00 | 1.281e-01 | 16 | 120 | 0.15
ising_2d | 9 | 4 | 32 | enriched_tdvp | 1e-04 | 1.713e-02 | 1.490e-01 | 16 | 0 | 0.57
ising_2d | 9 | 4 | 32 | enriched_tdvp | 1e-06 | 1.713e-02 | 1.490e-01 | 16 | 0 | 0.49
ising_2d | 9 | 4 | 32 | tebd_swaps | — | 0.000e+00 | 1.281e-01 | 16 | 120 | 0.14
ising_2d | 9 | 4 | 64 | enriched_tdvp | 1e-04 | 1.713e-02 | 1.490e-01 | 16 | 0 | 0.46
ising_2d | 9 | 4 | 64 | enriched_tdvp | 1e-06 | 1.713e-02 | 1.490e-01 | 16 | 0 | 0.45
ising_2d | 9 | 4 | 64 | tebd_swaps | — | 0.000e+00 | 1.281e-01 | 16 | 120 | 0.18
ising_2d | 9 | 4 | 128 | enriched_tdvp | 1e-04 | 1.713e-02 | 1.490e-01 | 16 | 0 | 0.47
ising_2d | 9 | 4 | 128 | enriched_tdvp | 1e-06 | 1.713e-02 | 1.490e-01 | 16 | 0 | 0.57
ising_2d | 9 | 4 | 128 | tebd_swaps | — | 0.000e+00 | 1.281e-01 | 16 | 120 | 0.14
ising_2d | 9 | 4 | 256 | enriched_tdvp | 1e-04 | 1.713e-02 | 1.490e-01 | 16 | 0 | 0.58
ising_2d | 9 | 4 | 256 | enriched_tdvp | 1e-06 | 1.713e-02 | 1.490e-01 | 16 | 0 | 0.48
ising_2d | 9 | 4 | 256 | tebd_swaps | — | 0.000e+00 | 1.281e-01 | 16 | 120 | 0.15
ising_2d | 9 | 4 | — | enriched_tdvp | 1e-04 | 2.229e-03 | 2.228e-01 | 16 | 0 | 0.48
ising_2d | 9 | 4 | — | enriched_tdvp | 1e-06 | 2.229e-03 | 2.228e-01 | 16 | 0 | 0.50
ising_2d | 9 | 4 | — | tebd_swaps | — | 0.000e+00 | 2.172e-01 | 16 | 120 | 0.14
ising_2d | 9 | 4 | 16 | enriched_tdvp | 1e-04 | 1.891e-03 | 2.191e-01 | 15 | 0 | 0.51
ising_2d | 9 | 4 | 16 | enriched_tdvp | 1e-06 | 1.891e-03 | 2.191e-01 | 15 | 0 | 0.50
ising_2d | 9 | 4 | 16 | tebd_swaps | — | 1.206e-09 | 2.172e-01 | 16 | 120 | 0.14
ising_2d | 9 | 4 | 32 | enriched_tdvp | 1e-04 | 1.891e-03 | 2.191e-01 | 15 | 0 | 0.49
ising_2d | 9 | 4 | 32 | enriched_tdvp | 1e-06 | 1.891e-03 | 2.191e-01 | 15 | 0 | 0.48
ising_2d | 9 | 4 | 32 | tebd_swaps | — | 1.206e-09 | 2.172e-01 | 16 | 120 | 0.14
ising_2d | 9 | 4 | 64 | enriched_tdvp | 1e-04 | 1.891e-03 | 2.191e-01 | 15 | 0 | 0.49
ising_2d | 9 | 4 | 64 | enriched_tdvp | 1e-06 | 1.891e-03 | 2.191e-01 | 15 | 0 | 0.47
ising_2d | 9 | 4 | 64 | tebd_swaps | — | 1.206e-09 | 2.172e-01 | 16 | 120 | 0.14
ising_2d | 9 | 4 | 128 | enriched_tdvp | 1e-04 | 1.891e-03 | 2.191e-01 | 15 | 0 | 0.46
ising_2d | 9 | 4 | 128 | enriched_tdvp | 1e-06 | 1.891e-03 | 2.191e-01 | 15 | 0 | 0.48
ising_2d | 9 | 4 | 128 | tebd_swaps | — | 1.206e-09 | 2.172e-01 | 16 | 120 | 0.13
ising_2d | 9 | 4 | 256 | enriched_tdvp | 1e-04 | 1.891e-03 | 2.191e-01 | 15 | 0 | 0.50
ising_2d | 9 | 4 | 256 | enriched_tdvp | 1e-06 | 1.891e-03 | 2.191e-01 | 15 | 0 | 0.67
ising_2d | 9 | 4 | 256 | tebd_swaps | — | 1.206e-09 | 2.172e-01 | 16 | 120 | 0.18
ising_2d | 9 | 4 | — | enriched_tdvp | 1e-04 | 3.366e-02 | 4.334e-01 | 16 | 0 | 0.60
ising_2d | 9 | 4 | — | enriched_tdvp | 1e-06 | 3.366e-02 | 4.334e-01 | 16 | 0 | 0.53
ising_2d | 9 | 4 | — | tebd_swaps | — | 0.000e+00 | 4.232e-01 | 16 | 120 | 0.15
ising_2d | 9 | 4 | 16 | enriched_tdvp | 1e-04 | 3.365e-02 | 4.333e-01 | 16 | 0 | 0.59
ising_2d | 9 | 4 | 16 | enriched_tdvp | 1e-06 | 3.365e-02 | 4.333e-01 | 16 | 0 | 0.57
ising_2d | 9 | 4 | 16 | tebd_swaps | — | 0.000e+00 | 4.232e-01 | 16 | 120 | 0.18
ising_2d | 9 | 4 | 32 | enriched_tdvp | 1e-04 | 3.365e-02 | 4.333e-01 | 16 | 0 | 0.56
ising_2d | 9 | 4 | 32 | enriched_tdvp | 1e-06 | 3.365e-02 | 4.333e-01 | 16 | 0 | 0.58
ising_2d | 9 | 4 | 32 | tebd_swaps | — | 0.000e+00 | 4.232e-01 | 16 | 120 | 0.17
ising_2d | 9 | 4 | 64 | enriched_tdvp | 1e-04 | 3.365e-02 | 4.333e-01 | 16 | 0 | 0.63
ising_2d | 9 | 4 | 64 | enriched_tdvp | 1e-06 | 3.365e-02 | 4.333e-01 | 16 | 0 | 0.82
ising_2d | 9 | 4 | 64 | tebd_swaps | — | 0.000e+00 | 4.232e-01 | 16 | 120 | 0.15
ising_2d | 9 | 4 | 128 | enriched_tdvp | 1e-04 | 3.365e-02 | 4.333e-01 | 16 | 0 | 0.83
ising_2d | 9 | 4 | 128 | enriched_tdvp | 1e-06 | 3.365e-02 | 4.333e-01 | 16 | 0 | 0.50
ising_2d | 9 | 4 | 128 | tebd_swaps | — | 0.000e+00 | 4.232e-01 | 16 | 120 | 0.14
ising_2d | 9 | 4 | 256 | enriched_tdvp | 1e-04 | 3.365e-02 | 4.333e-01 | 16 | 0 | 0.57
ising_2d | 9 | 4 | 256 | enriched_tdvp | 1e-06 | 3.365e-02 | 4.333e-01 | 16 | 0 | 0.52
ising_2d | 9 | 4 | 256 | tebd_swaps | — | 0.000e+00 | 4.232e-01 | 16 | 120 | 0.14

## Route statistics (enriched_tdvp only)

family | eps | lr_pauli | tdvp_frac | enriched_frac

---|---:|---:|---:|---:

ising_2d | 1e-04 | 18 | 0.889 | 0.111
ising_2d | 1e-06 | 18 | 0.889 | 0.111
ising_2d | 1e-04 | 18 | 0.889 | 0.111
ising_2d | 1e-06 | 18 | 0.889 | 0.111
ising_2d | 1e-04 | 18 | 0.889 | 0.111
ising_2d | 1e-06 | 18 | 0.889 | 0.111
ising_2d | 1e-04 | 18 | 0.889 | 0.111
ising_2d | 1e-06 | 18 | 0.889 | 0.111
ising_2d | 1e-04 | 18 | 0.889 | 0.111
ising_2d | 1e-06 | 18 | 0.889 | 0.111
ising_2d | 1e-04 | 18 | 0.889 | 0.111
ising_2d | 1e-06 | 18 | 0.889 | 0.111
ising_2d | 1e-04 | 18 | 0.444 | 0.556
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.444 | 0.556
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.444 | 0.556
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.444 | 0.556
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.444 | 0.556
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.444 | 0.556
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.889 | 0.111
ising_2d | 1e-06 | 18 | 0.889 | 0.111
ising_2d | 1e-04 | 18 | 0.889 | 0.111
ising_2d | 1e-06 | 18 | 0.889 | 0.111
ising_2d | 1e-04 | 18 | 0.889 | 0.111
ising_2d | 1e-06 | 18 | 0.889 | 0.111
ising_2d | 1e-04 | 18 | 0.889 | 0.111
ising_2d | 1e-06 | 18 | 0.889 | 0.111
ising_2d | 1e-04 | 18 | 0.889 | 0.111
ising_2d | 1e-06 | 18 | 0.889 | 0.111
ising_2d | 1e-04 | 18 | 0.889 | 0.111
ising_2d | 1e-06 | 18 | 0.889 | 0.111
ising_2d | 1e-04 | 18 | 0.389 | 0.611
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.222 | 0.778
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.222 | 0.778
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.222 | 0.778
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.222 | 0.778
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.222 | 0.778
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 36 | 0.889 | 0.111
ising_2d | 1e-06 | 36 | 0.722 | 0.278
ising_2d | 1e-04 | 36 | 0.889 | 0.111
ising_2d | 1e-06 | 36 | 0.556 | 0.444
ising_2d | 1e-04 | 36 | 0.889 | 0.111
ising_2d | 1e-06 | 36 | 0.556 | 0.444
ising_2d | 1e-04 | 36 | 0.889 | 0.111
ising_2d | 1e-06 | 36 | 0.556 | 0.444
ising_2d | 1e-04 | 36 | 0.889 | 0.111
ising_2d | 1e-06 | 36 | 0.556 | 0.444
ising_2d | 1e-04 | 36 | 0.889 | 0.111
ising_2d | 1e-06 | 36 | 0.556 | 0.444
ising_2d | 1e-04 | 36 | 0.583 | 0.417
ising_2d | 1e-06 | 36 | 0.472 | 0.528
ising_2d | 1e-04 | 36 | 0.556 | 0.444
ising_2d | 1e-06 | 36 | 0.361 | 0.639
ising_2d | 1e-04 | 36 | 0.556 | 0.444
ising_2d | 1e-06 | 36 | 0.361 | 0.639
ising_2d | 1e-04 | 36 | 0.556 | 0.444
ising_2d | 1e-06 | 36 | 0.361 | 0.639
ising_2d | 1e-04 | 36 | 0.556 | 0.444
ising_2d | 1e-06 | 36 | 0.361 | 0.639
ising_2d | 1e-04 | 36 | 0.556 | 0.444
ising_2d | 1e-06 | 36 | 0.361 | 0.639
ising_2d | 1e-04 | 36 | 0.861 | 0.139
ising_2d | 1e-06 | 36 | 0.722 | 0.278
ising_2d | 1e-04 | 36 | 0.889 | 0.111
ising_2d | 1e-06 | 36 | 0.556 | 0.444
ising_2d | 1e-04 | 36 | 0.889 | 0.111
ising_2d | 1e-06 | 36 | 0.556 | 0.444
ising_2d | 1e-04 | 36 | 0.889 | 0.111
ising_2d | 1e-06 | 36 | 0.556 | 0.444
ising_2d | 1e-04 | 36 | 0.889 | 0.111
ising_2d | 1e-06 | 36 | 0.556 | 0.444
ising_2d | 1e-04 | 36 | 0.889 | 0.111
ising_2d | 1e-06 | 36 | 0.556 | 0.444
ising_2d | 1e-04 | 36 | 0.556 | 0.444
ising_2d | 1e-06 | 36 | 0.472 | 0.528
ising_2d | 1e-04 | 36 | 0.389 | 0.611
ising_2d | 1e-06 | 36 | 0.389 | 0.611
ising_2d | 1e-04 | 36 | 0.389 | 0.611
ising_2d | 1e-06 | 36 | 0.389 | 0.611
ising_2d | 1e-04 | 36 | 0.389 | 0.611
ising_2d | 1e-06 | 36 | 0.389 | 0.611
ising_2d | 1e-04 | 36 | 0.389 | 0.611
ising_2d | 1e-06 | 36 | 0.389 | 0.611
ising_2d | 1e-04 | 36 | 0.389 | 0.611
ising_2d | 1e-06 | 36 | 0.389 | 0.611
ising_2d | 1e-04 | 18 | 0.889 | 0.111
ising_2d | 1e-06 | 18 | 0.778 | 0.222
ising_2d | 1e-04 | 18 | 0.889 | 0.111
ising_2d | 1e-06 | 18 | 0.778 | 0.222
ising_2d | 1e-04 | 18 | 0.889 | 0.111
ising_2d | 1e-06 | 18 | 0.778 | 0.222
ising_2d | 1e-04 | 18 | 0.889 | 0.111
ising_2d | 1e-06 | 18 | 0.778 | 0.222
ising_2d | 1e-04 | 18 | 0.889 | 0.111
ising_2d | 1e-06 | 18 | 0.778 | 0.222
ising_2d | 1e-04 | 18 | 0.889 | 0.111
ising_2d | 1e-06 | 18 | 0.778 | 0.222
ising_2d | 1e-04 | 18 | 0.389 | 0.611
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.222 | 0.778
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.222 | 0.778
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.222 | 0.778
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.222 | 0.778
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.222 | 0.778
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.833 | 0.167
ising_2d | 1e-06 | 18 | 0.833 | 0.167
ising_2d | 1e-04 | 18 | 0.833 | 0.167
ising_2d | 1e-06 | 18 | 0.667 | 0.333
ising_2d | 1e-04 | 18 | 0.833 | 0.167
ising_2d | 1e-06 | 18 | 0.667 | 0.333
ising_2d | 1e-04 | 18 | 0.833 | 0.167
ising_2d | 1e-06 | 18 | 0.667 | 0.333
ising_2d | 1e-04 | 18 | 0.833 | 0.167
ising_2d | 1e-06 | 18 | 0.667 | 0.333
ising_2d | 1e-04 | 18 | 0.833 | 0.167
ising_2d | 1e-06 | 18 | 0.667 | 0.333
ising_2d | 1e-04 | 18 | 0.389 | 0.611
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.222 | 0.778
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.222 | 0.778
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.222 | 0.778
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.222 | 0.778
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.222 | 0.778
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 36 | 0.722 | 0.278
ising_2d | 1e-06 | 36 | 0.667 | 0.333
ising_2d | 1e-04 | 36 | 0.778 | 0.222
ising_2d | 1e-06 | 36 | 0.500 | 0.500
ising_2d | 1e-04 | 36 | 0.778 | 0.222
ising_2d | 1e-06 | 36 | 0.500 | 0.500
ising_2d | 1e-04 | 36 | 0.778 | 0.222
ising_2d | 1e-06 | 36 | 0.500 | 0.500
ising_2d | 1e-04 | 36 | 0.778 | 0.222
ising_2d | 1e-06 | 36 | 0.500 | 0.500
ising_2d | 1e-04 | 36 | 0.778 | 0.222
ising_2d | 1e-06 | 36 | 0.500 | 0.500
ising_2d | 1e-04 | 36 | 0.556 | 0.444
ising_2d | 1e-06 | 36 | 0.472 | 0.528
ising_2d | 1e-04 | 36 | 0.472 | 0.528
ising_2d | 1e-06 | 36 | 0.389 | 0.611
ising_2d | 1e-04 | 36 | 0.472 | 0.528
ising_2d | 1e-06 | 36 | 0.389 | 0.611
ising_2d | 1e-04 | 36 | 0.472 | 0.528
ising_2d | 1e-06 | 36 | 0.389 | 0.611
ising_2d | 1e-04 | 36 | 0.472 | 0.528
ising_2d | 1e-06 | 36 | 0.389 | 0.611
ising_2d | 1e-04 | 36 | 0.472 | 0.528
ising_2d | 1e-06 | 36 | 0.389 | 0.611
ising_2d | 1e-04 | 36 | 0.694 | 0.306
ising_2d | 1e-06 | 36 | 0.694 | 0.306
ising_2d | 1e-04 | 36 | 0.750 | 0.250
ising_2d | 1e-06 | 36 | 0.444 | 0.556
ising_2d | 1e-04 | 36 | 0.750 | 0.250
ising_2d | 1e-06 | 36 | 0.444 | 0.556
ising_2d | 1e-04 | 36 | 0.750 | 0.250
ising_2d | 1e-06 | 36 | 0.444 | 0.556
ising_2d | 1e-04 | 36 | 0.750 | 0.250
ising_2d | 1e-06 | 36 | 0.444 | 0.556
ising_2d | 1e-04 | 36 | 0.750 | 0.250
ising_2d | 1e-06 | 36 | 0.444 | 0.556
ising_2d | 1e-04 | 36 | 0.556 | 0.444
ising_2d | 1e-06 | 36 | 0.472 | 0.528
ising_2d | 1e-04 | 36 | 0.472 | 0.528
ising_2d | 1e-06 | 36 | 0.472 | 0.528
ising_2d | 1e-04 | 36 | 0.472 | 0.528
ising_2d | 1e-06 | 36 | 0.472 | 0.528
ising_2d | 1e-04 | 36 | 0.472 | 0.528
ising_2d | 1e-06 | 36 | 0.472 | 0.528
ising_2d | 1e-04 | 36 | 0.472 | 0.528
ising_2d | 1e-06 | 36 | 0.472 | 0.528
ising_2d | 1e-04 | 36 | 0.472 | 0.528
ising_2d | 1e-06 | 36 | 0.472 | 0.528
ising_2d | 1e-04 | 18 | 0.778 | 0.222
ising_2d | 1e-06 | 18 | 0.778 | 0.222
ising_2d | 1e-04 | 18 | 0.778 | 0.222
ising_2d | 1e-06 | 18 | 0.778 | 0.222
ising_2d | 1e-04 | 18 | 0.778 | 0.222
ising_2d | 1e-06 | 18 | 0.778 | 0.222
ising_2d | 1e-04 | 18 | 0.778 | 0.222
ising_2d | 1e-06 | 18 | 0.778 | 0.222
ising_2d | 1e-04 | 18 | 0.778 | 0.222
ising_2d | 1e-06 | 18 | 0.778 | 0.222
ising_2d | 1e-04 | 18 | 0.778 | 0.222
ising_2d | 1e-06 | 18 | 0.778 | 0.222
ising_2d | 1e-04 | 18 | 0.389 | 0.611
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.222 | 0.778
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.222 | 0.778
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.222 | 0.778
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.222 | 0.778
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.222 | 0.778
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.778 | 0.222
ising_2d | 1e-06 | 18 | 0.778 | 0.222
ising_2d | 1e-04 | 18 | 0.778 | 0.222
ising_2d | 1e-06 | 18 | 0.778 | 0.222
ising_2d | 1e-04 | 18 | 0.778 | 0.222
ising_2d | 1e-06 | 18 | 0.778 | 0.222
ising_2d | 1e-04 | 18 | 0.778 | 0.222
ising_2d | 1e-06 | 18 | 0.778 | 0.222
ising_2d | 1e-04 | 18 | 0.778 | 0.222
ising_2d | 1e-06 | 18 | 0.778 | 0.222
ising_2d | 1e-04 | 18 | 0.778 | 0.222
ising_2d | 1e-06 | 18 | 0.778 | 0.222
ising_2d | 1e-04 | 18 | 0.389 | 0.611
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.222 | 0.778
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.222 | 0.778
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.222 | 0.778
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.222 | 0.778
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 18 | 0.222 | 0.778
ising_2d | 1e-06 | 18 | 0.222 | 0.778
ising_2d | 1e-04 | 36 | 0.667 | 0.333
ising_2d | 1e-06 | 36 | 0.667 | 0.333
ising_2d | 1e-04 | 36 | 0.667 | 0.333
ising_2d | 1e-06 | 36 | 0.667 | 0.333
ising_2d | 1e-04 | 36 | 0.667 | 0.333
ising_2d | 1e-06 | 36 | 0.667 | 0.333
ising_2d | 1e-04 | 36 | 0.667 | 0.333
ising_2d | 1e-06 | 36 | 0.667 | 0.333
ising_2d | 1e-04 | 36 | 0.667 | 0.333
ising_2d | 1e-06 | 36 | 0.667 | 0.333
ising_2d | 1e-04 | 36 | 0.667 | 0.333
ising_2d | 1e-06 | 36 | 0.667 | 0.333
ising_2d | 1e-04 | 36 | 0.556 | 0.444
ising_2d | 1e-06 | 36 | 0.472 | 0.528
ising_2d | 1e-04 | 36 | 0.389 | 0.611
ising_2d | 1e-06 | 36 | 0.389 | 0.611
ising_2d | 1e-04 | 36 | 0.389 | 0.611
ising_2d | 1e-06 | 36 | 0.389 | 0.611
ising_2d | 1e-04 | 36 | 0.389 | 0.611
ising_2d | 1e-06 | 36 | 0.389 | 0.611
ising_2d | 1e-04 | 36 | 0.389 | 0.611
ising_2d | 1e-06 | 36 | 0.389 | 0.611
ising_2d | 1e-04 | 36 | 0.389 | 0.611
ising_2d | 1e-06 | 36 | 0.389 | 0.611
ising_2d | 1e-04 | 36 | 0.667 | 0.333
ising_2d | 1e-06 | 36 | 0.667 | 0.333
ising_2d | 1e-04 | 36 | 0.667 | 0.333
ising_2d | 1e-06 | 36 | 0.667 | 0.333
ising_2d | 1e-04 | 36 | 0.667 | 0.333
ising_2d | 1e-06 | 36 | 0.667 | 0.333
ising_2d | 1e-04 | 36 | 0.667 | 0.333
ising_2d | 1e-06 | 36 | 0.667 | 0.333
ising_2d | 1e-04 | 36 | 0.667 | 0.333
ising_2d | 1e-06 | 36 | 0.667 | 0.333
ising_2d | 1e-04 | 36 | 0.667 | 0.333
ising_2d | 1e-06 | 36 | 0.667 | 0.333
ising_2d | 1e-04 | 36 | 0.556 | 0.444
ising_2d | 1e-06 | 36 | 0.472 | 0.528
ising_2d | 1e-04 | 36 | 0.472 | 0.528
ising_2d | 1e-06 | 36 | 0.472 | 0.528
ising_2d | 1e-04 | 36 | 0.472 | 0.528
ising_2d | 1e-06 | 36 | 0.472 | 0.528
ising_2d | 1e-04 | 36 | 0.472 | 0.528
ising_2d | 1e-06 | 36 | 0.472 | 0.528
ising_2d | 1e-04 | 36 | 0.472 | 0.528
ising_2d | 1e-06 | 36 | 0.472 | 0.528
ising_2d | 1e-04 | 36 | 0.472 | 0.528
ising_2d | 1e-06 | 36 | 0.472 | 0.528

## Practical accuracy failures (obs_max > 1e-3)

circuit | method | chi | eps | obs_max | fid_err | worst_obs | suspect

---|---|---:|---:|---:|---:|---|---

`ising2d_3x3_h0.5_dt0.1_L2_plus` | enriched_tdvp | — | 1e-04 | 9.536e-02 | 5.752e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L2_plus` | enriched_tdvp | — | 1e-06 | 9.537e-02 | 4.377e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L2_plus` | tebd_swaps | — | — | 9.539e-02 | 0.000e+00 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h0.5_dt0.1_L2_plus` | enriched_tdvp | 16 | 1e-04 | 9.535e-02 | 4.496e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L2_plus` | enriched_tdvp | 16 | 1e-06 | 9.536e-02 | 3.834e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L2_plus` | tebd_swaps | 16 | — | 9.539e-02 | 1.774e-09 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h0.5_dt0.1_L2_plus` | enriched_tdvp | 32 | 1e-04 | 9.535e-02 | 4.496e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L2_plus` | enriched_tdvp | 32 | 1e-06 | 9.536e-02 | 3.834e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L2_plus` | tebd_swaps | 32 | — | 9.539e-02 | 1.774e-09 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h0.5_dt0.1_L2_plus` | enriched_tdvp | 64 | 1e-04 | 9.535e-02 | 4.496e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L2_plus` | enriched_tdvp | 64 | 1e-06 | 9.536e-02 | 3.834e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L2_plus` | tebd_swaps | 64 | — | 9.539e-02 | 1.774e-09 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h0.5_dt0.1_L2_plus` | enriched_tdvp | 128 | 1e-04 | 9.535e-02 | 4.496e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L2_plus` | enriched_tdvp | 128 | 1e-06 | 9.536e-02 | 3.834e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L2_plus` | tebd_swaps | 128 | — | 9.539e-02 | 1.774e-09 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h0.5_dt0.1_L2_plus` | enriched_tdvp | 256 | 1e-04 | 9.535e-02 | 4.496e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L2_plus` | enriched_tdvp | 256 | 1e-06 | 9.536e-02 | 3.834e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L2_plus` | tebd_swaps | 256 | — | 9.539e-02 | 1.774e-09 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h0.5_dt0.1_L2_neel` | enriched_tdvp | — | 1e-04 | 9.639e-01 | 5.197e-07 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h0.5_dt0.1_L2_neel` | enriched_tdvp | — | 1e-06 | 9.639e-01 | 5.197e-07 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h0.5_dt0.1_L2_neel` | tebd_swaps | — | — | 9.639e-01 | 2.697e-12 | ZZ_vertical_lr(4,7) | True
`ising2d_3x3_h0.5_dt0.1_L2_neel` | enriched_tdvp | 16 | 1e-04 | 9.639e-01 | 5.253e-07 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h0.5_dt0.1_L2_neel` | enriched_tdvp | 16 | 1e-06 | 9.639e-01 | 5.253e-07 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h0.5_dt0.1_L2_neel` | tebd_swaps | 16 | — | 9.639e-01 | 1.303e-09 | ZZ_vertical_lr(4,7) | True
`ising2d_3x3_h0.5_dt0.1_L2_neel` | enriched_tdvp | 32 | 1e-04 | 9.639e-01 | 5.253e-07 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h0.5_dt0.1_L2_neel` | enriched_tdvp | 32 | 1e-06 | 9.639e-01 | 5.253e-07 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h0.5_dt0.1_L2_neel` | tebd_swaps | 32 | — | 9.639e-01 | 1.303e-09 | ZZ_vertical_lr(4,7) | True
`ising2d_3x3_h0.5_dt0.1_L2_neel` | enriched_tdvp | 64 | 1e-04 | 9.639e-01 | 5.253e-07 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h0.5_dt0.1_L2_neel` | enriched_tdvp | 64 | 1e-06 | 9.639e-01 | 5.253e-07 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h0.5_dt0.1_L2_neel` | tebd_swaps | 64 | — | 9.639e-01 | 1.303e-09 | ZZ_vertical_lr(4,7) | True
`ising2d_3x3_h0.5_dt0.1_L2_neel` | enriched_tdvp | 128 | 1e-04 | 9.639e-01 | 5.253e-07 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h0.5_dt0.1_L2_neel` | enriched_tdvp | 128 | 1e-06 | 9.639e-01 | 5.253e-07 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h0.5_dt0.1_L2_neel` | tebd_swaps | 128 | — | 9.639e-01 | 1.303e-09 | ZZ_vertical_lr(4,7) | True
`ising2d_3x3_h0.5_dt0.1_L2_neel` | enriched_tdvp | 256 | 1e-04 | 9.639e-01 | 5.253e-07 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h0.5_dt0.1_L2_neel` | enriched_tdvp | 256 | 1e-06 | 9.639e-01 | 5.253e-07 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h0.5_dt0.1_L2_neel` | tebd_swaps | 256 | — | 9.639e-01 | 1.303e-09 | ZZ_vertical_lr(4,7) | True
`ising2d_3x3_h0.5_dt0.1_L2_random_product` | enriched_tdvp | — | 1e-04 | 4.977e-01 | 8.224e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h0.5_dt0.1_L2_random_product` | enriched_tdvp | — | 1e-06 | 4.977e-01 | 8.224e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h0.5_dt0.1_L2_random_product` | tebd_swaps | — | — | 5.000e-01 | 0.000e+00 | ZZ_horizontal_periodic_lr(0,2) | True
`ising2d_3x3_h0.5_dt0.1_L2_random_product` | enriched_tdvp | 16 | 1e-04 | 4.977e-01 | 8.224e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h0.5_dt0.1_L2_random_product` | enriched_tdvp | 16 | 1e-06 | 4.977e-01 | 8.224e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h0.5_dt0.1_L2_random_product` | tebd_swaps | 16 | — | 5.000e-01 | 1.403e-09 | ZZ_horizontal_periodic_lr(0,2) | True
`ising2d_3x3_h0.5_dt0.1_L2_random_product` | enriched_tdvp | 32 | 1e-04 | 4.977e-01 | 8.224e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h0.5_dt0.1_L2_random_product` | enriched_tdvp | 32 | 1e-06 | 4.977e-01 | 8.224e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h0.5_dt0.1_L2_random_product` | tebd_swaps | 32 | — | 5.000e-01 | 1.403e-09 | ZZ_horizontal_periodic_lr(0,2) | True
`ising2d_3x3_h0.5_dt0.1_L2_random_product` | enriched_tdvp | 64 | 1e-04 | 4.977e-01 | 8.224e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h0.5_dt0.1_L2_random_product` | enriched_tdvp | 64 | 1e-06 | 4.977e-01 | 8.224e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h0.5_dt0.1_L2_random_product` | tebd_swaps | 64 | — | 5.000e-01 | 1.403e-09 | ZZ_horizontal_periodic_lr(0,2) | True
`ising2d_3x3_h0.5_dt0.1_L2_random_product` | enriched_tdvp | 128 | 1e-04 | 4.977e-01 | 8.224e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h0.5_dt0.1_L2_random_product` | enriched_tdvp | 128 | 1e-06 | 4.977e-01 | 8.224e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h0.5_dt0.1_L2_random_product` | tebd_swaps | 128 | — | 5.000e-01 | 1.403e-09 | ZZ_horizontal_periodic_lr(0,2) | True
`ising2d_3x3_h0.5_dt0.1_L2_random_product` | enriched_tdvp | 256 | 1e-04 | 4.977e-01 | 8.224e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h0.5_dt0.1_L2_random_product` | enriched_tdvp | 256 | 1e-06 | 4.977e-01 | 8.224e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h0.5_dt0.1_L2_random_product` | tebd_swaps | 256 | — | 5.000e-01 | 1.403e-09 | ZZ_horizontal_periodic_lr(0,2) | True
`ising2d_3x3_h0.5_dt0.1_L4_all_zero` | enriched_tdvp | — | 1e-04 | 7.986e-03 | 7.171e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L4_all_zero` | enriched_tdvp | — | 1e-06 | 7.945e-03 | 3.805e-05 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L4_all_zero` | tebd_swaps | — | — | 8.038e-03 | 1.245e-12 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L4_all_zero` | enriched_tdvp | 16 | 1e-04 | 7.980e-03 | 7.949e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L4_all_zero` | enriched_tdvp | 16 | 1e-06 | 1.800e-02 | 2.536e-04 | Y(6) | False
`ising2d_3x3_h0.5_dt0.1_L4_all_zero` | tebd_swaps | 16 | — | 8.038e-03 | 4.944e-09 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L4_all_zero` | enriched_tdvp | 32 | 1e-04 | 7.980e-03 | 7.949e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L4_all_zero` | enriched_tdvp | 32 | 1e-06 | 1.800e-02 | 2.536e-04 | Y(6) | False
`ising2d_3x3_h0.5_dt0.1_L4_all_zero` | tebd_swaps | 32 | — | 8.038e-03 | 4.944e-09 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L4_all_zero` | enriched_tdvp | 64 | 1e-04 | 7.980e-03 | 7.949e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L4_all_zero` | enriched_tdvp | 64 | 1e-06 | 1.800e-02 | 2.536e-04 | Y(6) | False
`ising2d_3x3_h0.5_dt0.1_L4_all_zero` | tebd_swaps | 64 | — | 8.038e-03 | 4.944e-09 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L4_all_zero` | enriched_tdvp | 128 | 1e-04 | 7.980e-03 | 7.949e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L4_all_zero` | enriched_tdvp | 128 | 1e-06 | 1.800e-02 | 2.536e-04 | Y(6) | False
`ising2d_3x3_h0.5_dt0.1_L4_all_zero` | tebd_swaps | 128 | — | 8.038e-03 | 4.944e-09 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L4_all_zero` | enriched_tdvp | 256 | 1e-04 | 7.980e-03 | 7.949e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L4_all_zero` | enriched_tdvp | 256 | 1e-06 | 1.800e-02 | 2.536e-04 | Y(6) | False
`ising2d_3x3_h0.5_dt0.1_L4_all_zero` | tebd_swaps | 256 | — | 8.038e-03 | 4.944e-09 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L4_plus` | enriched_tdvp | — | 1e-04 | 2.172e-01 | 1.795e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L4_plus` | enriched_tdvp | — | 1e-06 | 2.172e-01 | 1.795e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L4_plus` | tebd_swaps | — | — | 1.885e-01 | 0.000e+00 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h0.5_dt0.1_L4_plus` | enriched_tdvp | 16 | 1e-04 | 2.099e-01 | 2.874e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L4_plus` | enriched_tdvp | 16 | 1e-06 | 2.099e-01 | 2.874e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L4_plus` | tebd_swaps | 16 | — | 1.885e-01 | 1.764e-09 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h0.5_dt0.1_L4_plus` | enriched_tdvp | 32 | 1e-04 | 2.099e-01 | 2.874e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L4_plus` | enriched_tdvp | 32 | 1e-06 | 2.099e-01 | 2.874e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L4_plus` | tebd_swaps | 32 | — | 1.885e-01 | 1.764e-09 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h0.5_dt0.1_L4_plus` | enriched_tdvp | 64 | 1e-04 | 2.099e-01 | 2.874e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L4_plus` | enriched_tdvp | 64 | 1e-06 | 2.099e-01 | 2.874e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L4_plus` | tebd_swaps | 64 | — | 1.885e-01 | 1.764e-09 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h0.5_dt0.1_L4_plus` | enriched_tdvp | 128 | 1e-04 | 2.099e-01 | 2.874e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L4_plus` | enriched_tdvp | 128 | 1e-06 | 2.099e-01 | 2.874e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L4_plus` | tebd_swaps | 128 | — | 1.885e-01 | 1.764e-09 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h0.5_dt0.1_L4_plus` | enriched_tdvp | 256 | 1e-04 | 2.099e-01 | 2.874e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L4_plus` | enriched_tdvp | 256 | 1e-06 | 2.099e-01 | 2.874e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h0.5_dt0.1_L4_plus` | tebd_swaps | 256 | — | 1.885e-01 | 1.764e-09 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h0.5_dt0.1_L4_neel` | enriched_tdvp | — | 1e-04 | 9.086e-01 | 5.575e-07 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h0.5_dt0.1_L4_neel` | enriched_tdvp | — | 1e-06 | 9.074e-01 | 9.633e-06 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h0.5_dt0.1_L4_neel` | tebd_swaps | — | — | 9.086e-01 | 4.796e-12 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h0.5_dt0.1_L4_neel` | enriched_tdvp | 16 | 1e-04 | 9.085e-01 | 1.542e-06 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h0.5_dt0.1_L4_neel` | enriched_tdvp | 16 | 1e-06 | 9.076e-01 | 1.422e-04 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h0.5_dt0.1_L4_neel` | tebd_swaps | 16 | — | 9.086e-01 | 6.780e-09 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h0.5_dt0.1_L4_neel` | enriched_tdvp | 32 | 1e-04 | 9.085e-01 | 1.542e-06 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h0.5_dt0.1_L4_neel` | enriched_tdvp | 32 | 1e-06 | 9.076e-01 | 1.422e-04 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h0.5_dt0.1_L4_neel` | tebd_swaps | 32 | — | 9.086e-01 | 6.780e-09 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h0.5_dt0.1_L4_neel` | enriched_tdvp | 64 | 1e-04 | 9.085e-01 | 1.542e-06 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h0.5_dt0.1_L4_neel` | enriched_tdvp | 64 | 1e-06 | 9.076e-01 | 1.422e-04 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h0.5_dt0.1_L4_neel` | tebd_swaps | 64 | — | 9.086e-01 | 6.780e-09 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h0.5_dt0.1_L4_neel` | enriched_tdvp | 128 | 1e-04 | 9.085e-01 | 1.542e-06 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h0.5_dt0.1_L4_neel` | enriched_tdvp | 128 | 1e-06 | 9.076e-01 | 1.422e-04 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h0.5_dt0.1_L4_neel` | tebd_swaps | 128 | — | 9.086e-01 | 6.780e-09 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h0.5_dt0.1_L4_neel` | enriched_tdvp | 256 | 1e-04 | 9.085e-01 | 1.542e-06 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h0.5_dt0.1_L4_neel` | enriched_tdvp | 256 | 1e-06 | 9.076e-01 | 1.422e-04 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h0.5_dt0.1_L4_neel` | tebd_swaps | 256 | — | 9.086e-01 | 6.780e-09 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h0.5_dt0.1_L4_random_product` | enriched_tdvp | — | 1e-04 | 4.471e-01 | 4.223e-02 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h0.5_dt0.1_L4_random_product` | enriched_tdvp | — | 1e-06 | 4.471e-01 | 4.223e-02 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h0.5_dt0.1_L4_random_product` | tebd_swaps | — | — | 4.447e-01 | 0.000e+00 | ZZ_vertical_lr(4,7) | True
`ising2d_3x3_h0.5_dt0.1_L4_random_product` | enriched_tdvp | 16 | 1e-04 | 4.470e-01 | 4.221e-02 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h0.5_dt0.1_L4_random_product` | enriched_tdvp | 16 | 1e-06 | 4.470e-01 | 4.221e-02 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h0.5_dt0.1_L4_random_product` | tebd_swaps | 16 | — | 4.447e-01 | 1.403e-09 | ZZ_vertical_lr(4,7) | True
`ising2d_3x3_h0.5_dt0.1_L4_random_product` | enriched_tdvp | 32 | 1e-04 | 4.470e-01 | 4.221e-02 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h0.5_dt0.1_L4_random_product` | enriched_tdvp | 32 | 1e-06 | 4.470e-01 | 4.221e-02 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h0.5_dt0.1_L4_random_product` | tebd_swaps | 32 | — | 4.447e-01 | 1.403e-09 | ZZ_vertical_lr(4,7) | True
`ising2d_3x3_h0.5_dt0.1_L4_random_product` | enriched_tdvp | 64 | 1e-04 | 4.470e-01 | 4.221e-02 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h0.5_dt0.1_L4_random_product` | enriched_tdvp | 64 | 1e-06 | 4.470e-01 | 4.221e-02 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h0.5_dt0.1_L4_random_product` | tebd_swaps | 64 | — | 4.447e-01 | 1.403e-09 | ZZ_vertical_lr(4,7) | True
`ising2d_3x3_h0.5_dt0.1_L4_random_product` | enriched_tdvp | 128 | 1e-04 | 4.470e-01 | 4.221e-02 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h0.5_dt0.1_L4_random_product` | enriched_tdvp | 128 | 1e-06 | 4.470e-01 | 4.221e-02 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h0.5_dt0.1_L4_random_product` | tebd_swaps | 128 | — | 4.447e-01 | 1.403e-09 | ZZ_vertical_lr(4,7) | True
`ising2d_3x3_h0.5_dt0.1_L4_random_product` | enriched_tdvp | 256 | 1e-04 | 4.470e-01 | 4.221e-02 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h0.5_dt0.1_L4_random_product` | enriched_tdvp | 256 | 1e-06 | 4.470e-01 | 4.221e-02 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h0.5_dt0.1_L4_random_product` | tebd_swaps | 256 | — | 4.447e-01 | 1.403e-09 | ZZ_vertical_lr(4,7) | True
`ising2d_3x3_h1_dt0.1_L2_all_zero` | enriched_tdvp | — | 1e-04 | 2.289e-03 | 8.295e-06 | ZZ_vertical_lr(0,3) | False
`ising2d_3x3_h1_dt0.1_L2_all_zero` | enriched_tdvp | — | 1e-06 | 2.369e-03 | 6.190e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_all_zero` | tebd_swaps | — | — | 2.369e-03 | 0.000e+00 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_all_zero` | enriched_tdvp | 16 | 1e-04 | 2.363e-03 | 8.100e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_all_zero` | enriched_tdvp | 16 | 1e-06 | 2.369e-03 | 6.206e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_all_zero` | tebd_swaps | 16 | — | 2.369e-03 | 0.000e+00 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_all_zero` | enriched_tdvp | 32 | 1e-04 | 2.363e-03 | 8.100e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_all_zero` | enriched_tdvp | 32 | 1e-06 | 2.369e-03 | 6.206e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_all_zero` | tebd_swaps | 32 | — | 2.369e-03 | 0.000e+00 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_all_zero` | enriched_tdvp | 64 | 1e-04 | 2.363e-03 | 8.100e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_all_zero` | enriched_tdvp | 64 | 1e-06 | 2.369e-03 | 6.206e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_all_zero` | tebd_swaps | 64 | — | 2.369e-03 | 0.000e+00 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_all_zero` | enriched_tdvp | 128 | 1e-04 | 2.363e-03 | 8.100e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_all_zero` | enriched_tdvp | 128 | 1e-06 | 2.369e-03 | 6.206e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_all_zero` | tebd_swaps | 128 | — | 2.369e-03 | 0.000e+00 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_all_zero` | enriched_tdvp | 256 | 1e-04 | 2.363e-03 | 8.100e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_all_zero` | enriched_tdvp | 256 | 1e-06 | 2.369e-03 | 6.206e-07 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_all_zero` | tebd_swaps | 256 | — | 2.369e-03 | 0.000e+00 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_plus` | enriched_tdvp | — | 1e-04 | 1.036e-01 | 9.649e-03 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_plus` | enriched_tdvp | — | 1e-06 | 9.295e-02 | 3.093e-06 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_plus` | tebd_swaps | — | — | 9.295e-02 | 0.000e+00 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h1_dt0.1_L2_plus` | enriched_tdvp | 16 | 1e-04 | 9.289e-02 | 1.117e-06 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_plus` | enriched_tdvp | 16 | 1e-06 | 9.289e-02 | 1.117e-06 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_plus` | tebd_swaps | 16 | — | 9.295e-02 | 8.563e-10 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h1_dt0.1_L2_plus` | enriched_tdvp | 32 | 1e-04 | 9.289e-02 | 1.117e-06 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_plus` | enriched_tdvp | 32 | 1e-06 | 9.289e-02 | 1.117e-06 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_plus` | tebd_swaps | 32 | — | 9.295e-02 | 8.563e-10 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h1_dt0.1_L2_plus` | enriched_tdvp | 64 | 1e-04 | 9.289e-02 | 1.117e-06 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_plus` | enriched_tdvp | 64 | 1e-06 | 9.289e-02 | 1.117e-06 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_plus` | tebd_swaps | 64 | — | 9.295e-02 | 8.563e-10 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h1_dt0.1_L2_plus` | enriched_tdvp | 128 | 1e-04 | 9.289e-02 | 1.117e-06 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_plus` | enriched_tdvp | 128 | 1e-06 | 9.289e-02 | 1.117e-06 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_plus` | tebd_swaps | 128 | — | 9.295e-02 | 8.563e-10 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h1_dt0.1_L2_plus` | enriched_tdvp | 256 | 1e-04 | 9.289e-02 | 1.117e-06 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_plus` | enriched_tdvp | 256 | 1e-06 | 9.289e-02 | 1.117e-06 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L2_plus` | tebd_swaps | 256 | — | 9.295e-02 | 8.563e-10 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h1_dt0.1_L2_neel` | enriched_tdvp | — | 1e-04 | 8.609e-01 | 5.016e-04 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h1_dt0.1_L2_neel` | enriched_tdvp | — | 1e-06 | 8.609e-01 | 5.016e-04 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h1_dt0.1_L2_neel` | tebd_swaps | — | — | 8.609e-01 | 1.720e-12 | ZZ_vertical_lr(4,7) | True
`ising2d_3x3_h1_dt0.1_L2_neel` | enriched_tdvp | 16 | 1e-04 | 8.609e-01 | 4.905e-04 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h1_dt0.1_L2_neel` | enriched_tdvp | 16 | 1e-06 | 8.609e-01 | 4.909e-04 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h1_dt0.1_L2_neel` | tebd_swaps | 16 | — | 8.609e-01 | 7.630e-11 | ZZ_vertical_lr(4,7) | True
`ising2d_3x3_h1_dt0.1_L2_neel` | enriched_tdvp | 32 | 1e-04 | 8.609e-01 | 4.905e-04 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h1_dt0.1_L2_neel` | enriched_tdvp | 32 | 1e-06 | 8.609e-01 | 4.909e-04 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h1_dt0.1_L2_neel` | tebd_swaps | 32 | — | 8.609e-01 | 7.630e-11 | ZZ_vertical_lr(4,7) | True
`ising2d_3x3_h1_dt0.1_L2_neel` | enriched_tdvp | 64 | 1e-04 | 8.609e-01 | 4.905e-04 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h1_dt0.1_L2_neel` | enriched_tdvp | 64 | 1e-06 | 8.609e-01 | 4.909e-04 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h1_dt0.1_L2_neel` | tebd_swaps | 64 | — | 8.609e-01 | 7.630e-11 | ZZ_vertical_lr(4,7) | True
`ising2d_3x3_h1_dt0.1_L2_neel` | enriched_tdvp | 128 | 1e-04 | 8.609e-01 | 4.905e-04 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h1_dt0.1_L2_neel` | enriched_tdvp | 128 | 1e-06 | 8.609e-01 | 4.909e-04 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h1_dt0.1_L2_neel` | tebd_swaps | 128 | — | 8.609e-01 | 7.630e-11 | ZZ_vertical_lr(4,7) | True
`ising2d_3x3_h1_dt0.1_L2_neel` | enriched_tdvp | 256 | 1e-04 | 8.609e-01 | 4.905e-04 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h1_dt0.1_L2_neel` | enriched_tdvp | 256 | 1e-06 | 8.609e-01 | 4.909e-04 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h1_dt0.1_L2_neel` | tebd_swaps | 256 | — | 8.609e-01 | 7.630e-11 | ZZ_vertical_lr(4,7) | True
`ising2d_3x3_h1_dt0.1_L2_random_product` | enriched_tdvp | — | 1e-04 | 5.066e-01 | 8.393e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h1_dt0.1_L2_random_product` | enriched_tdvp | — | 1e-06 | 5.066e-01 | 8.393e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h1_dt0.1_L2_random_product` | tebd_swaps | — | — | 5.083e-01 | 0.000e+00 | ZZ_horizontal_periodic_lr(0,2) | True
`ising2d_3x3_h1_dt0.1_L2_random_product` | enriched_tdvp | 16 | 1e-04 | 5.066e-01 | 8.392e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h1_dt0.1_L2_random_product` | enriched_tdvp | 16 | 1e-06 | 5.066e-01 | 8.392e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h1_dt0.1_L2_random_product` | tebd_swaps | 16 | — | 5.083e-01 | 4.458e-10 | ZZ_horizontal_periodic_lr(0,2) | True
`ising2d_3x3_h1_dt0.1_L2_random_product` | enriched_tdvp | 32 | 1e-04 | 5.066e-01 | 8.392e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h1_dt0.1_L2_random_product` | enriched_tdvp | 32 | 1e-06 | 5.066e-01 | 8.392e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h1_dt0.1_L2_random_product` | tebd_swaps | 32 | — | 5.083e-01 | 4.458e-10 | ZZ_horizontal_periodic_lr(0,2) | True
`ising2d_3x3_h1_dt0.1_L2_random_product` | enriched_tdvp | 64 | 1e-04 | 5.066e-01 | 8.392e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h1_dt0.1_L2_random_product` | enriched_tdvp | 64 | 1e-06 | 5.066e-01 | 8.392e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h1_dt0.1_L2_random_product` | tebd_swaps | 64 | — | 5.083e-01 | 4.458e-10 | ZZ_horizontal_periodic_lr(0,2) | True
`ising2d_3x3_h1_dt0.1_L2_random_product` | enriched_tdvp | 128 | 1e-04 | 5.066e-01 | 8.392e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h1_dt0.1_L2_random_product` | enriched_tdvp | 128 | 1e-06 | 5.066e-01 | 8.392e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h1_dt0.1_L2_random_product` | tebd_swaps | 128 | — | 5.083e-01 | 4.458e-10 | ZZ_horizontal_periodic_lr(0,2) | True
`ising2d_3x3_h1_dt0.1_L2_random_product` | enriched_tdvp | 256 | 1e-04 | 5.066e-01 | 8.392e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h1_dt0.1_L2_random_product` | enriched_tdvp | 256 | 1e-06 | 5.066e-01 | 8.392e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h1_dt0.1_L2_random_product` | tebd_swaps | 256 | — | 5.083e-01 | 4.458e-10 | ZZ_horizontal_periodic_lr(0,2) | True
`ising2d_3x3_h1_dt0.1_L4_all_zero` | enriched_tdvp | — | 1e-04 | 2.674e-02 | 5.164e-04 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L4_all_zero` | enriched_tdvp | — | 1e-06 | 2.753e-02 | 5.073e-04 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L4_all_zero` | tebd_swaps | — | — | 2.819e-02 | 4.781e-13 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h1_dt0.1_L4_all_zero` | enriched_tdvp | 16 | 1e-04 | 2.783e-02 | 3.198e-04 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L4_all_zero` | enriched_tdvp | 16 | 1e-06 | 3.918e-02 | 2.013e-03 | Y(6) | False
`ising2d_3x3_h1_dt0.1_L4_all_zero` | tebd_swaps | 16 | — | 2.819e-02 | 3.034e-09 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h1_dt0.1_L4_all_zero` | enriched_tdvp | 32 | 1e-04 | 2.783e-02 | 3.198e-04 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L4_all_zero` | enriched_tdvp | 32 | 1e-06 | 3.918e-02 | 2.013e-03 | Y(6) | False
`ising2d_3x3_h1_dt0.1_L4_all_zero` | tebd_swaps | 32 | — | 2.819e-02 | 3.034e-09 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h1_dt0.1_L4_all_zero` | enriched_tdvp | 64 | 1e-04 | 2.783e-02 | 3.198e-04 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L4_all_zero` | enriched_tdvp | 64 | 1e-06 | 3.918e-02 | 2.013e-03 | Y(6) | False
`ising2d_3x3_h1_dt0.1_L4_all_zero` | tebd_swaps | 64 | — | 2.819e-02 | 3.034e-09 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h1_dt0.1_L4_all_zero` | enriched_tdvp | 128 | 1e-04 | 2.783e-02 | 3.198e-04 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L4_all_zero` | enriched_tdvp | 128 | 1e-06 | 3.918e-02 | 2.013e-03 | Y(6) | False
`ising2d_3x3_h1_dt0.1_L4_all_zero` | tebd_swaps | 128 | — | 2.819e-02 | 3.034e-09 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h1_dt0.1_L4_all_zero` | enriched_tdvp | 256 | 1e-04 | 2.783e-02 | 3.198e-04 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L4_all_zero` | enriched_tdvp | 256 | 1e-06 | 3.918e-02 | 2.013e-03 | Y(6) | False
`ising2d_3x3_h1_dt0.1_L4_all_zero` | tebd_swaps | 256 | — | 2.819e-02 | 3.034e-09 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h1_dt0.1_L4_plus` | enriched_tdvp | — | 1e-04 | 1.911e-01 | 4.489e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L4_plus` | enriched_tdvp | — | 1e-06 | 1.904e-01 | 1.837e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L4_plus` | tebd_swaps | — | — | 1.614e-01 | 0.000e+00 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h1_dt0.1_L4_plus` | enriched_tdvp | 16 | 1e-04 | 1.902e-01 | 1.834e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L4_plus` | enriched_tdvp | 16 | 1e-06 | 1.902e-01 | 1.834e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L4_plus` | tebd_swaps | 16 | — | 1.614e-01 | 1.043e-09 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h1_dt0.1_L4_plus` | enriched_tdvp | 32 | 1e-04 | 1.902e-01 | 1.834e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L4_plus` | enriched_tdvp | 32 | 1e-06 | 1.902e-01 | 1.834e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L4_plus` | tebd_swaps | 32 | — | 1.614e-01 | 1.043e-09 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h1_dt0.1_L4_plus` | enriched_tdvp | 64 | 1e-04 | 1.902e-01 | 1.834e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L4_plus` | enriched_tdvp | 64 | 1e-06 | 1.902e-01 | 1.834e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L4_plus` | tebd_swaps | 64 | — | 1.614e-01 | 1.043e-09 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h1_dt0.1_L4_plus` | enriched_tdvp | 128 | 1e-04 | 1.902e-01 | 1.834e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L4_plus` | enriched_tdvp | 128 | 1e-06 | 1.902e-01 | 1.834e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L4_plus` | tebd_swaps | 128 | — | 1.614e-01 | 1.043e-09 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h1_dt0.1_L4_plus` | enriched_tdvp | 256 | 1e-04 | 1.902e-01 | 1.834e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L4_plus` | enriched_tdvp | 256 | 1e-06 | 1.902e-01 | 1.834e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h1_dt0.1_L4_plus` | tebd_swaps | 256 | — | 1.614e-01 | 1.043e-09 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h1_dt0.1_L4_neel` | enriched_tdvp | — | 1e-04 | 6.608e-01 | 4.869e-04 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h1_dt0.1_L4_neel` | enriched_tdvp | — | 1e-06 | 6.608e-01 | 4.869e-04 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h1_dt0.1_L4_neel` | tebd_swaps | — | — | 6.729e-01 | 1.949e-12 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h1_dt0.1_L4_neel` | enriched_tdvp | 16 | 1e-04 | 6.627e-01 | 4.312e-04 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h1_dt0.1_L4_neel` | enriched_tdvp | 16 | 1e-06 | 6.622e-01 | 1.227e-03 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h1_dt0.1_L4_neel` | tebd_swaps | 16 | — | 6.729e-01 | 4.024e-09 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h1_dt0.1_L4_neel` | enriched_tdvp | 32 | 1e-04 | 6.627e-01 | 4.312e-04 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h1_dt0.1_L4_neel` | enriched_tdvp | 32 | 1e-06 | 6.622e-01 | 1.227e-03 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h1_dt0.1_L4_neel` | tebd_swaps | 32 | — | 6.729e-01 | 4.024e-09 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h1_dt0.1_L4_neel` | enriched_tdvp | 64 | 1e-04 | 6.627e-01 | 4.312e-04 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h1_dt0.1_L4_neel` | enriched_tdvp | 64 | 1e-06 | 6.622e-01 | 1.227e-03 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h1_dt0.1_L4_neel` | tebd_swaps | 64 | — | 6.729e-01 | 4.024e-09 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h1_dt0.1_L4_neel` | enriched_tdvp | 128 | 1e-04 | 6.627e-01 | 4.312e-04 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h1_dt0.1_L4_neel` | enriched_tdvp | 128 | 1e-06 | 6.622e-01 | 1.227e-03 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h1_dt0.1_L4_neel` | tebd_swaps | 128 | — | 6.729e-01 | 4.024e-09 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h1_dt0.1_L4_neel` | enriched_tdvp | 256 | 1e-04 | 6.627e-01 | 4.312e-04 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h1_dt0.1_L4_neel` | enriched_tdvp | 256 | 1e-06 | 6.622e-01 | 1.227e-03 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h1_dt0.1_L4_neel` | tebd_swaps | 256 | — | 6.729e-01 | 4.024e-09 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h1_dt0.1_L4_random_product` | enriched_tdvp | — | 1e-04 | 4.566e-01 | 4.104e-02 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h1_dt0.1_L4_random_product` | enriched_tdvp | — | 1e-06 | 4.566e-01 | 4.104e-02 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h1_dt0.1_L4_random_product` | tebd_swaps | — | — | 4.457e-01 | 0.000e+00 | ZZ_horizontal_periodic_lr(0,2) | True
`ising2d_3x3_h1_dt0.1_L4_random_product` | enriched_tdvp | 16 | 1e-04 | 4.566e-01 | 4.103e-02 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h1_dt0.1_L4_random_product` | enriched_tdvp | 16 | 1e-06 | 4.566e-01 | 4.103e-02 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h1_dt0.1_L4_random_product` | tebd_swaps | 16 | — | 4.457e-01 | 4.458e-10 | ZZ_horizontal_periodic_lr(0,2) | True
`ising2d_3x3_h1_dt0.1_L4_random_product` | enriched_tdvp | 32 | 1e-04 | 4.566e-01 | 4.103e-02 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h1_dt0.1_L4_random_product` | enriched_tdvp | 32 | 1e-06 | 4.566e-01 | 4.103e-02 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h1_dt0.1_L4_random_product` | tebd_swaps | 32 | — | 4.457e-01 | 4.458e-10 | ZZ_horizontal_periodic_lr(0,2) | True
`ising2d_3x3_h1_dt0.1_L4_random_product` | enriched_tdvp | 64 | 1e-04 | 4.566e-01 | 4.103e-02 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h1_dt0.1_L4_random_product` | enriched_tdvp | 64 | 1e-06 | 4.566e-01 | 4.103e-02 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h1_dt0.1_L4_random_product` | tebd_swaps | 64 | — | 4.457e-01 | 4.458e-10 | ZZ_horizontal_periodic_lr(0,2) | True
`ising2d_3x3_h1_dt0.1_L4_random_product` | enriched_tdvp | 128 | 1e-04 | 4.566e-01 | 4.103e-02 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h1_dt0.1_L4_random_product` | enriched_tdvp | 128 | 1e-06 | 4.566e-01 | 4.103e-02 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h1_dt0.1_L4_random_product` | tebd_swaps | 128 | — | 4.457e-01 | 4.458e-10 | ZZ_horizontal_periodic_lr(0,2) | True
`ising2d_3x3_h1_dt0.1_L4_random_product` | enriched_tdvp | 256 | 1e-04 | 4.566e-01 | 4.103e-02 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h1_dt0.1_L4_random_product` | enriched_tdvp | 256 | 1e-06 | 4.566e-01 | 4.103e-02 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h1_dt0.1_L4_random_product` | tebd_swaps | 256 | — | 4.457e-01 | 4.458e-10 | ZZ_horizontal_periodic_lr(0,2) | True
`ising2d_3x3_h2_dt0.1_L2_all_zero` | enriched_tdvp | — | 1e-04 | 7.393e-03 | 3.643e-05 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_all_zero` | enriched_tdvp | — | 1e-06 | 7.393e-03 | 3.643e-05 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_all_zero` | tebd_swaps | — | — | 7.424e-03 | 0.000e+00 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_all_zero` | enriched_tdvp | 16 | 1e-04 | 7.393e-03 | 3.587e-05 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_all_zero` | enriched_tdvp | 16 | 1e-06 | 7.393e-03 | 3.587e-05 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_all_zero` | tebd_swaps | 16 | — | 7.424e-03 | 2.127e-09 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_all_zero` | enriched_tdvp | 32 | 1e-04 | 7.393e-03 | 3.587e-05 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_all_zero` | enriched_tdvp | 32 | 1e-06 | 7.393e-03 | 3.587e-05 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_all_zero` | tebd_swaps | 32 | — | 7.424e-03 | 2.127e-09 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_all_zero` | enriched_tdvp | 64 | 1e-04 | 7.393e-03 | 3.587e-05 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_all_zero` | enriched_tdvp | 64 | 1e-06 | 7.393e-03 | 3.587e-05 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_all_zero` | tebd_swaps | 64 | — | 7.424e-03 | 2.127e-09 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_all_zero` | enriched_tdvp | 128 | 1e-04 | 7.393e-03 | 3.587e-05 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_all_zero` | enriched_tdvp | 128 | 1e-06 | 7.393e-03 | 3.587e-05 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_all_zero` | tebd_swaps | 128 | — | 7.424e-03 | 2.127e-09 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_all_zero` | enriched_tdvp | 256 | 1e-04 | 7.393e-03 | 3.587e-05 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_all_zero` | enriched_tdvp | 256 | 1e-06 | 7.393e-03 | 3.587e-05 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_all_zero` | tebd_swaps | 256 | — | 7.424e-03 | 2.127e-09 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_plus` | enriched_tdvp | — | 1e-04 | 9.148e-02 | 1.279e-05 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_plus` | enriched_tdvp | — | 1e-06 | 9.148e-02 | 1.279e-05 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_plus` | tebd_swaps | — | — | 9.145e-02 | 0.000e+00 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h2_dt0.1_L2_plus` | enriched_tdvp | 16 | 1e-04 | 9.145e-02 | 6.790e-06 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_plus` | enriched_tdvp | 16 | 1e-06 | 9.145e-02 | 6.790e-06 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_plus` | tebd_swaps | 16 | — | 9.145e-02 | 0.000e+00 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h2_dt0.1_L2_plus` | enriched_tdvp | 32 | 1e-04 | 9.145e-02 | 6.790e-06 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_plus` | enriched_tdvp | 32 | 1e-06 | 9.145e-02 | 6.790e-06 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_plus` | tebd_swaps | 32 | — | 9.145e-02 | 0.000e+00 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h2_dt0.1_L2_plus` | enriched_tdvp | 64 | 1e-04 | 9.145e-02 | 6.790e-06 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_plus` | enriched_tdvp | 64 | 1e-06 | 9.145e-02 | 6.790e-06 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_plus` | tebd_swaps | 64 | — | 9.145e-02 | 0.000e+00 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h2_dt0.1_L2_plus` | enriched_tdvp | 128 | 1e-04 | 9.145e-02 | 6.790e-06 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_plus` | enriched_tdvp | 128 | 1e-06 | 9.145e-02 | 6.790e-06 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_plus` | tebd_swaps | 128 | — | 9.145e-02 | 0.000e+00 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h2_dt0.1_L2_plus` | enriched_tdvp | 256 | 1e-04 | 9.145e-02 | 6.790e-06 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_plus` | enriched_tdvp | 256 | 1e-06 | 9.145e-02 | 6.790e-06 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L2_plus` | tebd_swaps | 256 | — | 9.145e-02 | 0.000e+00 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h2_dt0.1_L2_neel` | enriched_tdvp | — | 1e-04 | 5.228e-01 | 0.000e+00 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h2_dt0.1_L2_neel` | enriched_tdvp | — | 1e-06 | 5.228e-01 | 0.000e+00 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h2_dt0.1_L2_neel` | tebd_swaps | — | — | 5.228e-01 | 0.000e+00 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h2_dt0.1_L2_neel` | enriched_tdvp | 16 | 1e-04 | 5.228e-01 | 2.720e-07 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h2_dt0.1_L2_neel` | enriched_tdvp | 16 | 1e-06 | 5.228e-01 | 2.720e-07 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h2_dt0.1_L2_neel` | tebd_swaps | 16 | — | 5.228e-01 | 1.996e-10 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h2_dt0.1_L2_neel` | enriched_tdvp | 32 | 1e-04 | 5.228e-01 | 2.720e-07 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h2_dt0.1_L2_neel` | enriched_tdvp | 32 | 1e-06 | 5.228e-01 | 2.720e-07 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h2_dt0.1_L2_neel` | tebd_swaps | 32 | — | 5.228e-01 | 1.996e-10 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h2_dt0.1_L2_neel` | enriched_tdvp | 64 | 1e-04 | 5.228e-01 | 2.720e-07 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h2_dt0.1_L2_neel` | enriched_tdvp | 64 | 1e-06 | 5.228e-01 | 2.720e-07 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h2_dt0.1_L2_neel` | tebd_swaps | 64 | — | 5.228e-01 | 1.996e-10 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h2_dt0.1_L2_neel` | enriched_tdvp | 128 | 1e-04 | 5.228e-01 | 2.720e-07 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h2_dt0.1_L2_neel` | enriched_tdvp | 128 | 1e-06 | 5.228e-01 | 2.720e-07 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h2_dt0.1_L2_neel` | tebd_swaps | 128 | — | 5.228e-01 | 1.996e-10 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h2_dt0.1_L2_neel` | enriched_tdvp | 256 | 1e-04 | 5.228e-01 | 2.720e-07 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h2_dt0.1_L2_neel` | enriched_tdvp | 256 | 1e-06 | 5.228e-01 | 2.720e-07 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h2_dt0.1_L2_neel` | tebd_swaps | 256 | — | 5.228e-01 | 1.996e-10 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h2_dt0.1_L2_random_product` | enriched_tdvp | — | 1e-04 | 4.913e-01 | 8.497e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h2_dt0.1_L2_random_product` | enriched_tdvp | — | 1e-06 | 4.913e-01 | 8.497e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h2_dt0.1_L2_random_product` | tebd_swaps | — | — | 4.906e-01 | 0.000e+00 | ZZ_horizontal_periodic_lr(0,2) | True
`ising2d_3x3_h2_dt0.1_L2_random_product` | enriched_tdvp | 16 | 1e-04 | 4.914e-01 | 8.496e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h2_dt0.1_L2_random_product` | enriched_tdvp | 16 | 1e-06 | 4.914e-01 | 8.496e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h2_dt0.1_L2_random_product` | tebd_swaps | 16 | — | 4.906e-01 | 0.000e+00 | ZZ_horizontal_periodic_lr(0,2) | True
`ising2d_3x3_h2_dt0.1_L2_random_product` | enriched_tdvp | 32 | 1e-04 | 4.914e-01 | 8.496e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h2_dt0.1_L2_random_product` | enriched_tdvp | 32 | 1e-06 | 4.914e-01 | 8.496e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h2_dt0.1_L2_random_product` | tebd_swaps | 32 | — | 4.906e-01 | 0.000e+00 | ZZ_horizontal_periodic_lr(0,2) | True
`ising2d_3x3_h2_dt0.1_L2_random_product` | enriched_tdvp | 64 | 1e-04 | 4.914e-01 | 8.496e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h2_dt0.1_L2_random_product` | enriched_tdvp | 64 | 1e-06 | 4.914e-01 | 8.496e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h2_dt0.1_L2_random_product` | tebd_swaps | 64 | — | 4.906e-01 | 0.000e+00 | ZZ_horizontal_periodic_lr(0,2) | True
`ising2d_3x3_h2_dt0.1_L2_random_product` | enriched_tdvp | 128 | 1e-04 | 4.914e-01 | 8.496e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h2_dt0.1_L2_random_product` | enriched_tdvp | 128 | 1e-06 | 4.914e-01 | 8.496e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h2_dt0.1_L2_random_product` | tebd_swaps | 128 | — | 4.906e-01 | 0.000e+00 | ZZ_horizontal_periodic_lr(0,2) | True
`ising2d_3x3_h2_dt0.1_L2_random_product` | enriched_tdvp | 256 | 1e-04 | 4.914e-01 | 8.496e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h2_dt0.1_L2_random_product` | enriched_tdvp | 256 | 1e-06 | 4.914e-01 | 8.496e-03 | ZZ_horizontal_periodic_lr(0,2) | False
`ising2d_3x3_h2_dt0.1_L2_random_product` | tebd_swaps | 256 | — | 4.906e-01 | 0.000e+00 | ZZ_horizontal_periodic_lr(0,2) | True
`ising2d_3x3_h2_dt0.1_L4_all_zero` | enriched_tdvp | — | 1e-04 | 6.708e-02 | 1.965e-03 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L4_all_zero` | enriched_tdvp | — | 1e-06 | 6.708e-02 | 1.965e-03 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L4_all_zero` | tebd_swaps | — | — | 6.705e-02 | 0.000e+00 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h2_dt0.1_L4_all_zero` | enriched_tdvp | 16 | 1e-04 | 6.748e-02 | 2.095e-03 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L4_all_zero` | enriched_tdvp | 16 | 1e-06 | 6.748e-02 | 2.095e-03 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L4_all_zero` | tebd_swaps | 16 | — | 6.705e-02 | 3.458e-09 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h2_dt0.1_L4_all_zero` | enriched_tdvp | 32 | 1e-04 | 6.748e-02 | 2.095e-03 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L4_all_zero` | enriched_tdvp | 32 | 1e-06 | 6.748e-02 | 2.095e-03 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L4_all_zero` | tebd_swaps | 32 | — | 6.705e-02 | 3.458e-09 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h2_dt0.1_L4_all_zero` | enriched_tdvp | 64 | 1e-04 | 6.748e-02 | 2.095e-03 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L4_all_zero` | enriched_tdvp | 64 | 1e-06 | 6.748e-02 | 2.095e-03 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L4_all_zero` | tebd_swaps | 64 | — | 6.705e-02 | 3.458e-09 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h2_dt0.1_L4_all_zero` | enriched_tdvp | 128 | 1e-04 | 6.748e-02 | 2.095e-03 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L4_all_zero` | enriched_tdvp | 128 | 1e-06 | 6.748e-02 | 2.095e-03 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L4_all_zero` | tebd_swaps | 128 | — | 6.705e-02 | 3.458e-09 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h2_dt0.1_L4_all_zero` | enriched_tdvp | 256 | 1e-04 | 6.748e-02 | 2.095e-03 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L4_all_zero` | enriched_tdvp | 256 | 1e-06 | 6.748e-02 | 2.095e-03 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L4_all_zero` | tebd_swaps | 256 | — | 6.705e-02 | 3.458e-09 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h2_dt0.1_L4_plus` | enriched_tdvp | — | 1e-04 | 1.492e-01 | 1.715e-02 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h2_dt0.1_L4_plus` | enriched_tdvp | — | 1e-06 | 1.492e-01 | 1.715e-02 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h2_dt0.1_L4_plus` | tebd_swaps | — | — | 1.281e-01 | 0.000e+00 | ZZ_vertical_lr(0,3) | True
`ising2d_3x3_h2_dt0.1_L4_plus` | enriched_tdvp | 16 | 1e-04 | 1.490e-01 | 1.713e-02 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h2_dt0.1_L4_plus` | enriched_tdvp | 16 | 1e-06 | 1.490e-01 | 1.713e-02 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h2_dt0.1_L4_plus` | tebd_swaps | 16 | — | 1.281e-01 | 0.000e+00 | ZZ_vertical_lr(0,3) | True
`ising2d_3x3_h2_dt0.1_L4_plus` | enriched_tdvp | 32 | 1e-04 | 1.490e-01 | 1.713e-02 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h2_dt0.1_L4_plus` | enriched_tdvp | 32 | 1e-06 | 1.490e-01 | 1.713e-02 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h2_dt0.1_L4_plus` | tebd_swaps | 32 | — | 1.281e-01 | 0.000e+00 | ZZ_vertical_lr(0,3) | True
`ising2d_3x3_h2_dt0.1_L4_plus` | enriched_tdvp | 64 | 1e-04 | 1.490e-01 | 1.713e-02 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h2_dt0.1_L4_plus` | enriched_tdvp | 64 | 1e-06 | 1.490e-01 | 1.713e-02 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h2_dt0.1_L4_plus` | tebd_swaps | 64 | — | 1.281e-01 | 0.000e+00 | ZZ_vertical_lr(0,3) | True
`ising2d_3x3_h2_dt0.1_L4_plus` | enriched_tdvp | 128 | 1e-04 | 1.490e-01 | 1.713e-02 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h2_dt0.1_L4_plus` | enriched_tdvp | 128 | 1e-06 | 1.490e-01 | 1.713e-02 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h2_dt0.1_L4_plus` | tebd_swaps | 128 | — | 1.281e-01 | 0.000e+00 | ZZ_vertical_lr(0,3) | True
`ising2d_3x3_h2_dt0.1_L4_plus` | enriched_tdvp | 256 | 1e-04 | 1.490e-01 | 1.713e-02 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h2_dt0.1_L4_plus` | enriched_tdvp | 256 | 1e-06 | 1.490e-01 | 1.713e-02 | ZZ_vertical_lr(4,7) | False
`ising2d_3x3_h2_dt0.1_L4_plus` | tebd_swaps | 256 | — | 1.281e-01 | 0.000e+00 | ZZ_vertical_lr(0,3) | True
`ising2d_3x3_h2_dt0.1_L4_neel` | enriched_tdvp | — | 1e-04 | 2.228e-01 | 2.229e-03 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h2_dt0.1_L4_neel` | enriched_tdvp | — | 1e-06 | 2.228e-01 | 2.229e-03 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h2_dt0.1_L4_neel` | tebd_swaps | — | — | 2.172e-01 | 0.000e+00 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h2_dt0.1_L4_neel` | enriched_tdvp | 16 | 1e-04 | 2.191e-01 | 1.891e-03 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h2_dt0.1_L4_neel` | enriched_tdvp | 16 | 1e-06 | 2.191e-01 | 1.891e-03 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h2_dt0.1_L4_neel` | tebd_swaps | 16 | — | 2.172e-01 | 1.206e-09 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h2_dt0.1_L4_neel` | enriched_tdvp | 32 | 1e-04 | 2.191e-01 | 1.891e-03 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h2_dt0.1_L4_neel` | enriched_tdvp | 32 | 1e-06 | 2.191e-01 | 1.891e-03 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h2_dt0.1_L4_neel` | tebd_swaps | 32 | — | 2.172e-01 | 1.206e-09 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h2_dt0.1_L4_neel` | enriched_tdvp | 64 | 1e-04 | 2.191e-01 | 1.891e-03 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h2_dt0.1_L4_neel` | enriched_tdvp | 64 | 1e-06 | 2.191e-01 | 1.891e-03 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h2_dt0.1_L4_neel` | tebd_swaps | 64 | — | 2.172e-01 | 1.206e-09 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h2_dt0.1_L4_neel` | enriched_tdvp | 128 | 1e-04 | 2.191e-01 | 1.891e-03 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h2_dt0.1_L4_neel` | enriched_tdvp | 128 | 1e-06 | 2.191e-01 | 1.891e-03 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h2_dt0.1_L4_neel` | tebd_swaps | 128 | — | 2.172e-01 | 1.206e-09 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h2_dt0.1_L4_neel` | enriched_tdvp | 256 | 1e-04 | 2.191e-01 | 1.891e-03 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h2_dt0.1_L4_neel` | enriched_tdvp | 256 | 1e-06 | 2.191e-01 | 1.891e-03 | ZZ_vertical_lr(2,5) | False
`ising2d_3x3_h2_dt0.1_L4_neel` | tebd_swaps | 256 | — | 2.172e-01 | 1.206e-09 | ZZ_vertical_lr(2,5) | True
`ising2d_3x3_h2_dt0.1_L4_random_product` | enriched_tdvp | — | 1e-04 | 4.334e-01 | 3.366e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L4_random_product` | enriched_tdvp | — | 1e-06 | 4.334e-01 | 3.366e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L4_random_product` | tebd_swaps | — | — | 4.232e-01 | 0.000e+00 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h2_dt0.1_L4_random_product` | enriched_tdvp | 16 | 1e-04 | 4.333e-01 | 3.365e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L4_random_product` | enriched_tdvp | 16 | 1e-06 | 4.333e-01 | 3.365e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L4_random_product` | tebd_swaps | 16 | — | 4.232e-01 | 0.000e+00 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h2_dt0.1_L4_random_product` | enriched_tdvp | 32 | 1e-04 | 4.333e-01 | 3.365e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L4_random_product` | enriched_tdvp | 32 | 1e-06 | 4.333e-01 | 3.365e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L4_random_product` | tebd_swaps | 32 | — | 4.232e-01 | 0.000e+00 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h2_dt0.1_L4_random_product` | enriched_tdvp | 64 | 1e-04 | 4.333e-01 | 3.365e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L4_random_product` | enriched_tdvp | 64 | 1e-06 | 4.333e-01 | 3.365e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L4_random_product` | tebd_swaps | 64 | — | 4.232e-01 | 0.000e+00 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h2_dt0.1_L4_random_product` | enriched_tdvp | 128 | 1e-04 | 4.333e-01 | 3.365e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L4_random_product` | enriched_tdvp | 128 | 1e-06 | 4.333e-01 | 3.365e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L4_random_product` | tebd_swaps | 128 | — | 4.232e-01 | 0.000e+00 | ZZ_vertical_lr(3,6) | True
`ising2d_3x3_h2_dt0.1_L4_random_product` | enriched_tdvp | 256 | 1e-04 | 4.333e-01 | 3.365e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L4_random_product` | enriched_tdvp | 256 | 1e-06 | 4.333e-01 | 3.365e-02 | ZZ_vertical_lr(3,6) | False
`ising2d_3x3_h2_dt0.1_L4_random_product` | tebd_swaps | 256 | — | 4.232e-01 | 0.000e+00 | ZZ_vertical_lr(3,6) | True

## Recommended conclusion

Use the fixed-bond rows (same `chi`) to compare error-vs-budget and wall-time. Pay special attention to 2D XXX (`xxx_2d`) where vertical edges generate many long-range RXX/RYY/RZZ gates.

