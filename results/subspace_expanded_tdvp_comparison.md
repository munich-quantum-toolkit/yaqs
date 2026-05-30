# Subspace-expanded TDVP comparison

Strict: fidelity_error<=1e-10 and pauli_obs_max_error<=1e-10 (when fidelity available).

Practical: pauli_obs_max_error<=1e-3.

## Summary table (per circuit/method)

circuit | method | fid_err | obs_max | obs_mean | obs_l2 | worst_obs | bond_before | bond_after_expand | bond_final | wall_s

---|---|---:|---:|---:|---:|---|---:|---:|---:|---:

`tangent_blind_rxx_8q` | plain_tdvp | 1.554e-02 | 3.109e-02 | 1.884e-03 | 4.396e-02 | Z(1) | 2 | — | 2 | 0.00
`tangent_blind_rxx_8q` | current_projection_defect_router | 4.441e-16 | 2.220e-16 | 7.401e-17 | 7.364e-16 | Z(0) | 2 | — | 2 | 0.00
`tangent_blind_rxx_8q` | exact_pauli_enrichment | 4.441e-16 | 2.220e-16 | 7.401e-17 | 7.364e-16 | Z(0) | 2 | — | 2 | 0.00
`tangent_blind_rxx_8q` | subspace_expanded_tdvp | 4.441e-16 | 2.220e-16 | 7.401e-17 | 7.364e-16 | Z(0) | 2 | 2 | 2 | 0.01
`tangent_blind_ryy_8q` | plain_tdvp | 1.554e-02 | 3.109e-02 | 1.884e-03 | 4.396e-02 | Z(1) | 2 | — | 2 | 0.00
`tangent_blind_ryy_8q` | current_projection_defect_router | 4.441e-16 | 2.220e-16 | 7.401e-17 | 7.364e-16 | Z(0) | 2 | — | 2 | 0.00
`tangent_blind_ryy_8q` | exact_pauli_enrichment | 4.441e-16 | 2.220e-16 | 7.401e-17 | 7.364e-16 | Z(0) | 2 | — | 2 | 0.00
`tangent_blind_ryy_8q` | subspace_expanded_tdvp | 4.441e-16 | 2.220e-16 | 7.401e-17 | 7.364e-16 | Z(0) | 2 | 2 | 2 | 0.01
`sanity_rzz_8q` | plain_tdvp | 0.000e+00 | 3.331e-16 | 1.110e-16 | 1.105e-15 | Z(0) | 2 | — | 2 | 0.00
`sanity_rzz_8q` | current_projection_defect_router | 0.000e+00 | 3.331e-16 | 1.110e-16 | 1.105e-15 | Z(0) | 2 | — | 2 | 0.01
`sanity_rzz_8q` | exact_pauli_enrichment | 4.441e-16 | 2.220e-16 | 7.401e-17 | 7.364e-16 | Z(0) | 2 | — | 2 | 0.00
`sanity_rzz_8q` | subspace_expanded_tdvp | 2.220e-16 | 2.220e-16 | 7.401e-17 | 7.364e-16 | Z(0) | 2 | 2 | 2 | 0.01
`endpoint_prepared_ryy_8q` | plain_tdvp | 2.220e-15 | 2.331e-15 | 7.678e-16 | 7.421e-15 | Z(1) | 2 | — | 2 | 0.00
`endpoint_prepared_ryy_8q` | current_projection_defect_router | 2.220e-15 | 2.331e-15 | 7.678e-16 | 7.421e-15 | Z(1) | 2 | — | 2 | 0.00
`endpoint_prepared_ryy_8q` | exact_pauli_enrichment | 4.441e-16 | 3.331e-16 | 1.168e-16 | 1.117e-15 | Z(0) | 2 | — | 2 | 0.00
`endpoint_prepared_ryy_8q` | subspace_expanded_tdvp | 2.220e-16 | 8.228e-16 | 1.276e-16 | 1.342e-15 | Y(1) | 2 | 2 | 2 | 0.01
`endpoint_prepared_rxx_8q` | plain_tdvp | 2.442e-15 | 2.331e-15 | 7.678e-16 | 7.421e-15 | Z(1) | 2 | — | 2 | 0.00
`endpoint_prepared_rxx_8q` | current_projection_defect_router | 2.442e-15 | 2.331e-15 | 7.678e-16 | 7.421e-15 | Z(1) | 2 | — | 2 | 0.00
`endpoint_prepared_rxx_8q` | exact_pauli_enrichment | 4.441e-16 | 3.331e-16 | 1.168e-16 | 1.117e-15 | Z(0) | 2 | — | 2 | 0.00
`endpoint_prepared_rxx_8q` | subspace_expanded_tdvp | 2.220e-16 | 4.805e-16 | 1.105e-16 | 1.065e-15 | X(1) | 2 | 2 | 2 | 0.01
`stack_rxx_ryy_vacuum_10q` | plain_tdvp | 1.554e-02 | 3.109e-02 | 2.391e-03 | 5.385e-02 | Z(3) | 2 | — | 2 | 0.01
`stack_rxx_ryy_vacuum_10q` | current_projection_defect_router | 2.220e-16 | 2.220e-16 | 1.993e-17 | 3.682e-16 | Z(2) | 2 | — | 4 | 0.01
`stack_rxx_ryy_vacuum_10q` | exact_pauli_enrichment | 2.220e-16 | 2.220e-16 | 1.993e-17 | 3.682e-16 | Z(2) | 2 | — | 4 | 0.01
`stack_rxx_ryy_vacuum_10q` | subspace_expanded_tdvp | 2.220e-16 | 2.220e-16 | 3.110e-17 | 4.996e-16 | Z(2) | 2 | 2 | 4 | 0.02
`lr_stack_mixed_12q` | plain_tdvp | 2.636e-02 | 3.109e-02 | 2.358e-03 | 5.383e-02 | Z(3) | 2 | — | 2 | 0.02
`lr_stack_mixed_12q` | current_projection_defect_router | 2.220e-16 | 3.331e-16 | 7.467e-17 | 7.899e-16 | Z(1) | 2 | — | 4 | 0.03
`lr_stack_mixed_12q` | exact_pauli_enrichment | 4.441e-16 | 8.882e-16 | 1.378e-16 | 1.690e-15 | ZZ(6,7) | 2 | — | 4 | 0.02
`lr_stack_mixed_12q` | subspace_expanded_tdvp | 2.636e-02 | 3.109e-02 | 2.358e-03 | 5.383e-02 | Z(3) | 2 | 2 | 2 | 0.05
`rand_n8_d4_s0` | plain_tdvp | 8.891e-02 | 2.077e-01 | 2.795e-02 | 3.684e-01 | Z(3) | 2 | — | 4 | 0.02
`rand_n8_d4_s0` | current_projection_defect_router | 0.000e+00 | 1.277e-15 | 2.381e-16 | 2.132e-15 | Y(7) | 2 | — | 4 | 0.01
`rand_n8_d4_s0` | exact_pauli_enrichment | 0.000e+00 | 1.277e-15 | 2.381e-16 | 2.132e-15 | Y(7) | 2 | — | 4 | 0.01
`rand_n8_d4_s0` | subspace_expanded_tdvp | 0.000e+00 | 5.738e-14 | 5.903e-15 | 6.970e-14 | Y(3) | 2 | 2 | 4 | 0.03
`rand_n8_d4_s1` | plain_tdvp | 5.000e-02 | 7.572e-02 | 1.402e-02 | 1.469e-01 | Z(2) | 2 | — | 2 | 0.02
`rand_n8_d4_s1` | current_projection_defect_router | 4.441e-16 | 1.332e-15 | 4.653e-16 | 3.691e-15 | Z(6) | 2 | — | 4 | 0.02
`rand_n8_d4_s1` | exact_pauli_enrichment | 4.441e-16 | 1.332e-15 | 4.653e-16 | 3.691e-15 | Z(6) | 2 | — | 4 | 0.01
`rand_n8_d4_s1` | subspace_expanded_tdvp | 1.762e-03 | 1.587e-02 | 6.763e-04 | 1.641e-02 | X(1) | 2 | 2 | 4 | 0.03
`rand_n8_d4_s2` | plain_tdvp | 1.098e-01 | 2.074e-01 | 2.853e-02 | 3.527e-01 | Z(7) | 2 | — | 4 | 0.02
`rand_n8_d4_s2` | current_projection_defect_router | 0.000e+00 | 1.332e-15 | 3.679e-16 | 2.874e-15 | Z(1) | 2 | — | 4 | 0.01
`rand_n8_d4_s2` | exact_pauli_enrichment | 0.000e+00 | 1.332e-15 | 3.679e-16 | 2.874e-15 | Z(1) | 2 | — | 4 | 0.01
`rand_n8_d4_s2` | subspace_expanded_tdvp | 1.864e-04 | 4.036e-04 | 4.511e-05 | 6.905e-04 | Z(0) | 2 | 2 | 4 | 0.03
`rand_n8_d4_s3` | plain_tdvp | 1.198e-01 | 1.469e-01 | 2.866e-02 | 3.162e-01 | Z(0) | 2 | — | 3 | 0.02
`rand_n8_d4_s3` | current_projection_defect_router | 0.000e+00 | 1.110e-15 | 2.894e-16 | 2.479e-15 | Z(4) | 2 | — | 8 | 0.02
`rand_n8_d4_s3` | exact_pauli_enrichment | 0.000e+00 | 8.882e-16 | 2.406e-16 | 1.962e-15 | Z(4) | 2 | — | 8 | 0.01
`rand_n8_d4_s3` | subspace_expanded_tdvp | 7.280e-02 | 1.469e-01 | 1.634e-02 | 2.558e-01 | Z(0) | 2 | 2 | 4 | 0.03
`rand_n8_d4_s4` | plain_tdvp | 1.379e-01 | 2.649e-01 | 3.794e-02 | 4.833e-01 | Z(7) | 2 | — | 2 | 0.02
`rand_n8_d4_s4` | current_projection_defect_router | 0.000e+00 | 1.443e-15 | 4.284e-16 | 3.178e-15 | Z(0) | 2 | — | 4 | 0.02
`rand_n8_d4_s4` | exact_pauli_enrichment | 4.441e-16 | 1.110e-15 | 2.828e-16 | 2.266e-15 | Z(3) | 2 | — | 4 | 0.01
`rand_n8_d4_s4` | subspace_expanded_tdvp | 4.441e-16 | 4.441e-15 | 8.364e-16 | 8.141e-15 | X(1) | 2 | 2 | 4 | 0.03
`rand_n8_d8_s0` | plain_tdvp | 1.591e-01 | 2.592e-01 | 5.037e-02 | 4.867e-01 | Y(0) | 2 | — | 4 | 0.04
`rand_n8_d8_s0` | current_projection_defect_router | 4.206e-02 | 2.320e-01 | 1.047e-02 | 2.403e-01 | Y(0) | 2 | — | 4 | 0.04
`rand_n8_d8_s0` | exact_pauli_enrichment | 8.882e-16 | 1.665e-15 | 5.379e-16 | 4.089e-15 | Z(1) | 2 | — | 4 | 0.03
`rand_n8_d8_s0` | subspace_expanded_tdvp | 2.395e-02 | 6.948e-02 | 6.194e-03 | 9.448e-02 | X(0) | 2 | 2 | 4 | 0.08
`rand_n8_d8_s1` | plain_tdvp | 4.672e-02 | 1.030e-01 | 1.840e-02 | 1.709e-01 | XX(2,3) | 2 | — | 8 | 0.04
`rand_n8_d8_s1` | current_projection_defect_router | 0.000e+00 | 9.659e-15 | 1.812e-15 | 1.563e-14 | Y(6) | 2 | — | 8 | 0.03
`rand_n8_d8_s1` | exact_pauli_enrichment | 0.000e+00 | 2.331e-15 | 7.580e-16 | 5.849e-15 | Z(0) | 2 | — | 8 | 0.02
`rand_n8_d8_s1` | subspace_expanded_tdvp | 1.762e-03 | 1.522e-02 | 8.810e-04 | 1.647e-02 | X(1) | 2 | 2 | 8 | 0.06
`rand_n8_d8_s2` | plain_tdvp | 1.074e-01 | 1.875e-01 | 3.253e-02 | 3.185e-01 | X(7) | 2 | — | 8 | 0.05
`rand_n8_d8_s2` | current_projection_defect_router | 1.554e-14 | 2.360e-13 | 2.952e-14 | 3.029e-13 | X(0) | 2 | — | 8 | 0.04
`rand_n8_d8_s2` | exact_pauli_enrichment | 0.000e+00 | 3.553e-15 | 7.509e-16 | 6.226e-15 | Z(0) | 2 | — | 8 | 0.02
`rand_n8_d8_s2` | subspace_expanded_tdvp | 1.864e-04 | 3.505e-04 | 4.853e-05 | 6.148e-04 | Z(5) | 2 | 2 | 8 | 0.06
`rand_n8_d8_s3` | plain_tdvp | 1.616e-01 | 1.957e-01 | 5.215e-02 | 4.357e-01 | X(5) | 2 | — | 4 | 0.04
`rand_n8_d8_s3` | current_projection_defect_router | 2.220e-16 | 1.075e-13 | 3.075e-14 | 2.723e-13 | Z(5) | 2 | — | 8 | 0.04
`rand_n8_d8_s3` | exact_pauli_enrichment | 0.000e+00 | 4.906e-15 | 8.075e-16 | 7.521e-15 | X(5) | 2 | — | 8 | 0.02
`rand_n8_d8_s3` | subspace_expanded_tdvp | 1.188e-01 | 1.876e-01 | 3.867e-02 | 3.871e-01 | X(5) | 2 | 2 | 4 | 0.06
`rand_n8_d8_s4` | plain_tdvp | 1.402e-01 | 2.692e-01 | 3.547e-02 | 4.213e-01 | Z(7) | 2 | — | 2 | 0.03
`rand_n8_d8_s4` | current_projection_defect_router | 0.000e+00 | 8.615e-14 | 2.786e-14 | 2.201e-13 | Z(3) | 2 | — | 4 | 0.04
`rand_n8_d8_s4` | exact_pauli_enrichment | 4.441e-16 | 1.332e-15 | 3.718e-16 | 2.734e-15 | Y(1) | 2 | — | 4 | 0.02
`rand_n8_d8_s4` | subspace_expanded_tdvp | 8.882e-16 | 1.478e-13 | 2.455e-14 | 2.641e-13 | Z(7) | 2 | 2 | 4 | 0.07
`rand_n10_d4_s0` | plain_tdvp | 1.111e-01 | 1.339e-01 | 2.060e-02 | 2.949e-01 | Z(7) | 2 | — | 3 | 0.02
`rand_n10_d4_s0` | current_projection_defect_router | 0.000e+00 | 1.998e-15 | 6.248e-16 | 5.243e-15 | Y(1) | 2 | — | 8 | 0.02
`rand_n10_d4_s0` | exact_pauli_enrichment | 0.000e+00 | 1.998e-15 | 6.248e-16 | 5.243e-15 | Y(1) | 2 | — | 8 | 0.01
`rand_n10_d4_s0` | subspace_expanded_tdvp | 6.699e-02 | 1.339e-01 | 1.435e-02 | 2.549e-01 | Z(7) | 2 | 2 | 4 | 0.04
`rand_n10_d4_s1` | plain_tdvp | 1.151e-01 | 1.549e-01 | 2.044e-02 | 2.735e-01 | Z(7) | 2 | — | 4 | 0.02
`rand_n10_d4_s1` | current_projection_defect_router | 0.000e+00 | 6.439e-15 | 2.538e-15 | 2.051e-14 | Z(0) | 2 | — | 4 | 0.02
`rand_n10_d4_s1` | exact_pauli_enrichment | 8.882e-16 | 1.110e-15 | 2.738e-16 | 2.452e-15 | Z(7) | 2 | — | 4 | 0.02
`rand_n10_d4_s1` | subspace_expanded_tdvp | 7.341e-02 | 1.549e-01 | 1.277e-02 | 2.417e-01 | Z(7) | 2 | 2 | 4 | 0.04
`rand_n10_d4_s2` | plain_tdvp | 1.241e-02 | 1.307e-02 | 1.823e-03 | 2.372e-02 | Z(0) | 2 | — | 4 | 0.02
`rand_n10_d4_s2` | current_projection_defect_router | 0.000e+00 | 1.693e-14 | 1.212e-15 | 1.937e-14 | X(8) | 2 | — | 8 | 0.02
`rand_n10_d4_s2` | exact_pauli_enrichment | 0.000e+00 | 1.693e-14 | 1.212e-15 | 1.937e-14 | X(8) | 2 | — | 8 | 0.01
`rand_n10_d4_s2` | subspace_expanded_tdvp | 6.445e-03 | 1.307e-02 | 8.609e-04 | 1.835e-02 | Z(0) | 2 | 2 | 4 | 0.04
`rand_n10_d4_s3` | plain_tdvp | 1.521e-01 | 2.142e-01 | 3.143e-02 | 4.243e-01 | X(6) | 2 | — | 2 | 0.02
`rand_n10_d4_s3` | current_projection_defect_router | 0.000e+00 | 1.998e-15 | 5.708e-16 | 4.864e-15 | Z(3) | 2 | — | 4 | 0.02
`rand_n10_d4_s3` | exact_pauli_enrichment | 0.000e+00 | 1.998e-15 | 5.708e-16 | 4.864e-15 | Z(3) | 2 | — | 4 | 0.01
`rand_n10_d4_s3` | subspace_expanded_tdvp | 3.748e-02 | 1.054e-01 | 9.509e-03 | 1.483e-01 | X(2) | 2 | 2 | 4 | 0.03
`rand_n10_d4_s4` | plain_tdvp | 2.869e-02 | 5.119e-02 | 7.190e-03 | 9.247e-02 | Z(9) | 2 | — | 4 | 0.02
`rand_n10_d4_s4` | current_projection_defect_router | 0.000e+00 | 9.964e-15 | 1.523e-15 | 1.498e-14 | Y(9) | 2 | — | 4 | 0.02
`rand_n10_d4_s4` | exact_pauli_enrichment | 2.220e-15 | 2.442e-15 | 8.466e-16 | 7.264e-15 | X(4) | 2 | — | 4 | 0.01
`rand_n10_d4_s4` | subspace_expanded_tdvp | 8.882e-16 | 7.716e-15 | 9.634e-16 | 1.144e-14 | X(9) | 2 | 2 | 4 | 0.04
`rand_n10_d8_s0` | plain_tdvp | 1.155e-01 | 1.091e-01 | 2.756e-02 | 2.453e-01 | X(7) | 2 | — | 8 | 0.06
`rand_n10_d8_s0` | current_projection_defect_router | 1.332e-15 | 8.882e-15 | 1.901e-15 | 1.577e-14 | X(9) | 2 | — | 16 | 0.04
`rand_n10_d8_s0` | exact_pauli_enrichment | 0.000e+00 | 7.994e-15 | 1.256e-15 | 1.209e-14 | X(9) | 2 | — | 16 | 0.03
`rand_n10_d8_s0` | subspace_expanded_tdvp | 7.866e-02 | 1.501e-01 | 1.765e-02 | 2.348e-01 | X(7) | 2 | 2 | 16 | 0.08
`rand_n10_d8_s1` | plain_tdvp | 1.297e-01 | 1.403e-01 | 3.336e-02 | 3.223e-01 | Z(8) | 2 | — | 8 | 0.05
`rand_n10_d8_s1` | current_projection_defect_router | 3.344e-03 | 3.309e-02 | 2.097e-03 | 3.833e-02 | X(0) | 2 | — | 8 | 0.04
`rand_n10_d8_s1` | exact_pauli_enrichment | 0.000e+00 | 1.665e-15 | 4.817e-16 | 3.951e-15 | X(7) | 2 | — | 8 | 0.03
`rand_n10_d8_s1` | subspace_expanded_tdvp | 3.389e-02 | 1.392e-01 | 1.369e-02 | 2.128e-01 | Z(7) | 2 | 2 | 7 | 0.08
`rand_n10_d8_s2` | plain_tdvp | 1.593e-01 | 2.189e-01 | 3.442e-02 | 4.158e-01 | Y(5) | 2 | — | 4 | 0.04
`rand_n10_d8_s2` | current_projection_defect_router | 0.000e+00 | 1.490e-14 | 2.301e-15 | 2.318e-14 | X(8) | 2 | — | 8 | 0.04
`rand_n10_d8_s2` | exact_pauli_enrichment | 0.000e+00 | 1.490e-14 | 2.301e-15 | 2.318e-14 | X(8) | 2 | — | 8 | 0.03
`rand_n10_d8_s2` | subspace_expanded_tdvp | 6.445e-03 | 1.129e-02 | 1.110e-03 | 1.835e-02 | Z(0) | 2 | 2 | 5 | 0.08
`rand_n10_d8_s3` | plain_tdvp | 2.020e-01 | 2.055e-01 | 4.954e-02 | 5.155e-01 | Z(9) | 2 | — | 4 | 0.04
`rand_n10_d8_s3` | current_projection_defect_router | 0.000e+00 | 4.144e-13 | 6.395e-14 | 7.220e-13 | Y(6) | 2 | — | 16 | 0.05
`rand_n10_d8_s3` | exact_pauli_enrichment | 0.000e+00 | 7.216e-15 | 8.920e-16 | 9.778e-15 | X(9) | 2 | — | 16 | 0.03
`rand_n10_d8_s3` | subspace_expanded_tdvp | 3.750e-02 | 8.935e-02 | 1.139e-02 | 1.489e-01 | Z(2) | 2 | 2 | 8 | 0.09
`rand_n10_d8_s4` | plain_tdvp | 3.716e-02 | 3.953e-02 | 1.063e-02 | 1.029e-01 | Z(5) | 2 | — | 4 | 0.05
`rand_n10_d8_s4` | current_projection_defect_router | 1.110e-15 | 2.959e-14 | 8.363e-15 | 7.618e-14 | X(5) | 2 | — | 8 | 0.05
`rand_n10_d8_s4` | exact_pauli_enrichment | 0.000e+00 | 2.193e-14 | 2.329e-15 | 3.179e-14 | Y(8) | 2 | — | 8 | 0.03
`rand_n10_d8_s4` | subspace_expanded_tdvp | 0.000e+00 | 3.031e-14 | 5.728e-15 | 6.238e-14 | X(9) | 2 | 2 | 8 | 0.10
`rand_n12_d4_s0` | plain_tdvp | 1.438e-01 | 2.330e-01 | 1.695e-02 | 3.572e-01 | Z(0) | 2 | — | 3 | 0.02
`rand_n12_d4_s0` | current_projection_defect_router | 0.000e+00 | 1.887e-15 | 4.956e-16 | 4.940e-15 | Z(2) | 2 | — | 8 | 0.02
`rand_n12_d4_s0` | exact_pauli_enrichment | 0.000e+00 | 1.887e-15 | 4.956e-16 | 4.940e-15 | Z(2) | 2 | — | 8 | 0.02
`rand_n12_d4_s0` | subspace_expanded_tdvp | 1.423e-01 | 2.330e-01 | 1.654e-02 | 3.571e-01 | Z(0) | 2 | 2 | 4 | 0.04
`rand_n12_d4_s1` | plain_tdvp | 7.966e-02 | 1.018e-01 | 1.374e-02 | 2.076e-01 | Z(5) | 2 | — | 2 | 0.02
`rand_n12_d4_s1` | current_projection_defect_router | 4.441e-16 | 1.776e-15 | 4.920e-16 | 4.691e-15 | Z(1) | 2 | — | 8 | 0.02
`rand_n12_d4_s1` | exact_pauli_enrichment | 1.332e-15 | 3.220e-15 | 6.132e-16 | 6.752e-15 | Z(1) | 2 | — | 8 | 0.02
`rand_n12_d4_s1` | subspace_expanded_tdvp | 5.091e-02 | 1.010e-01 | 1.114e-02 | 1.937e-01 | Z(0) | 2 | 2 | 4 | 0.05
`rand_n12_d4_s2` | plain_tdvp | 3.912e-02 | 7.799e-02 | 6.094e-03 | 1.270e-01 | Z(4) | 2 | — | 4 | 0.02
`rand_n12_d4_s2` | current_projection_defect_router | 0.000e+00 | 2.998e-15 | 6.287e-16 | 6.628e-15 | Y(11) | 2 | — | 8 | 0.02
`rand_n12_d4_s2` | exact_pauli_enrichment | 0.000e+00 | 2.998e-15 | 6.287e-16 | 6.628e-15 | Y(11) | 2 | — | 8 | 0.02
`rand_n12_d4_s2` | subspace_expanded_tdvp | 3.908e-02 | 7.799e-02 | 6.085e-03 | 1.270e-01 | Z(4) | 2 | 2 | 4 | 0.04
`rand_n12_d4_s3` | plain_tdvp | 8.085e-02 | 1.298e-01 | 1.266e-02 | 2.029e-01 | Z(9) | 2 | — | 4 | 0.02
`rand_n12_d4_s3` | current_projection_defect_router | 0.000e+00 | 7.255e-15 | 1.767e-15 | 1.594e-14 | Y(10) | 2 | — | 8 | 0.03
`rand_n12_d4_s3` | exact_pauli_enrichment | 1.110e-15 | 4.045e-15 | 6.600e-16 | 6.539e-15 | Y(10) | 2 | — | 8 | 0.02
`rand_n12_d4_s3` | subspace_expanded_tdvp | 7.236e-02 | 1.298e-01 | 1.016e-02 | 1.979e-01 | Z(9) | 2 | 2 | 2 | 0.04
`rand_n12_d4_s4` | plain_tdvp | 2.923e-02 | 4.040e-02 | 4.851e-03 | 7.478e-02 | Z(2) | 2 | — | 2 | 0.02
`rand_n12_d4_s4` | current_projection_defect_router | 0.000e+00 | 7.855e-15 | 6.713e-16 | 9.369e-15 | X(7) | 2 | — | 8 | 0.02
`rand_n12_d4_s4` | exact_pauli_enrichment | 0.000e+00 | 7.855e-15 | 6.713e-16 | 9.369e-15 | X(7) | 2 | — | 8 | 0.02
`rand_n12_d4_s4` | subspace_expanded_tdvp | 2.049e-02 | 4.040e-02 | 3.191e-03 | 6.833e-02 | Z(2) | 2 | 2 | 4 | 0.08
`rand_n12_d8_s0` | plain_tdvp | 1.842e-01 | 2.332e-01 | 2.762e-02 | 3.987e-01 | Z(0) | 2 | — | 5 | 0.04
`rand_n12_d8_s0` | current_projection_defect_router | 0.000e+00 | 1.443e-14 | 2.404e-15 | 2.603e-14 | Z(9) | 2 | — | 8 | 0.05
`rand_n12_d8_s0` | exact_pauli_enrichment | 1.110e-15 | 1.688e-14 | 1.727e-15 | 2.281e-14 | X(9) | 2 | — | 8 | 0.03
`rand_n12_d8_s0` | subspace_expanded_tdvp | 1.718e-01 | 2.221e-01 | 2.575e-02 | 3.683e-01 | Z(0) | 2 | 2 | 8 | 0.09
`rand_n12_d8_s1` | plain_tdvp | 9.729e-02 | 1.034e-01 | 1.746e-02 | 2.134e-01 | Z(0) | 2 | — | 7 | 0.05
`rand_n12_d8_s1` | current_projection_defect_router | 8.882e-16 | 2.010e-14 | 2.773e-15 | 3.899e-14 | Z(5) | 2 | — | 16 | 0.05
`rand_n12_d8_s1` | exact_pauli_enrichment | 6.661e-16 | 2.065e-14 | 2.418e-15 | 3.708e-14 | Z(5) | 2 | — | 16 | 0.03
`rand_n12_d8_s1` | subspace_expanded_tdvp | 5.252e-02 | 1.035e-01 | 1.097e-02 | 1.845e-01 | Z(0) | 2 | 2 | 16 | 0.09
`rand_n12_d8_s2` | plain_tdvp | 6.530e-02 | 1.976e-01 | 1.500e-02 | 2.462e-01 | XX(4,5) | 2 | — | 6 | 0.05
`rand_n12_d8_s2` | current_projection_defect_router | 0.000e+00 | 1.549e-14 | 1.596e-15 | 2.117e-14 | Y(9) | 2 | — | 32 | 0.07
`rand_n12_d8_s2` | exact_pauli_enrichment | 0.000e+00 | 1.601e-14 | 1.734e-15 | 2.238e-14 | Y(9) | 2 | — | 32 | 0.04
`rand_n12_d8_s2` | subspace_expanded_tdvp | 1.163e-01 | 2.331e-01 | 2.681e-02 | 4.390e-01 | Z(9) | 2 | 2 | 8 | 0.09
`rand_n12_d8_s3` | plain_tdvp | 1.113e-01 | 1.207e-01 | 1.830e-02 | 2.201e-01 | Z(2) | 2 | — | 4 | 0.05
`rand_n12_d8_s3` | current_projection_defect_router | 0.000e+00 | 1.987e-14 | 7.685e-15 | 6.412e-14 | ZZ(6,7) | 2 | — | 16 | 0.06
`rand_n12_d8_s3` | exact_pauli_enrichment | 4.441e-16 | 3.569e-14 | 5.169e-15 | 6.259e-14 | Y(8) | 2 | — | 16 | 0.04
`rand_n12_d8_s3` | subspace_expanded_tdvp | 9.341e-02 | 1.207e-01 | 1.487e-02 | 2.142e-01 | Z(2) | 2 | 2 | 4 | 0.09
`rand_n12_d8_s4` | plain_tdvp | 6.564e-02 | 7.626e-02 | 1.273e-02 | 1.452e-01 | ZZ(6,7) | 2 | — | 7 | 0.05
`rand_n12_d8_s4` | current_projection_defect_router | 1.407e-02 | 3.767e-02 | 2.153e-03 | 5.272e-02 | Z(3) | 2 | — | 16 | 0.06
`rand_n12_d8_s4` | exact_pauli_enrichment | 0.000e+00 | 9.992e-15 | 1.378e-15 | 1.616e-14 | X(10) | 2 | — | 32 | 0.04
`rand_n12_d8_s4` | subspace_expanded_tdvp | 6.203e-02 | 8.804e-02 | 1.140e-02 | 1.530e-01 | ZZ(6,7) | 2 | 2 | 4 | 0.08

## Expansion diagnostics (subspace_expanded_tdvp only)

circuit | d_before | d_after | state_delta_from_expansion | bond_after_expand

---|---:|---:|---:|---:

`tangent_blind_rxx_8q` | 1.000e+00 | 0.000e+00 | 0.000e+00 | 2
`tangent_blind_ryy_8q` | 1.000e+00 | 0.000e+00 | 0.000e+00 | 2
`sanity_rzz_8q` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 2
`endpoint_prepared_ryy_8q` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 2
`endpoint_prepared_rxx_8q` | 0.000e+00 | 0.000e+00 | 0.000e+00 | 2
`stack_rxx_ryy_vacuum_10q` | 1.000e+00 | 0.000e+00 | 0.000e+00 | 2
`lr_stack_mixed_12q` | 3.331e-16 | 3.331e-16 | 0.000e+00 | 2
`rand_n8_d4_s0` | 5.785e-01 | 0.000e+00 | 0.000e+00 | 2
`rand_n8_d4_s1` | 5.663e-01 | 5.551e-16 | 1.332e-15 | 2
`rand_n8_d4_s2` | 6.572e-01 | 0.000e+00 | 0.000e+00 | 2
`rand_n8_d4_s3` | 6.018e-01 | 6.018e-01 | 0.000e+00 | 2
`rand_n8_d4_s4` | 8.768e-01 | 0.000e+00 | 0.000e+00 | 2
`rand_n8_d8_s0` | 5.785e-01 | 0.000e+00 | 0.000e+00 | 2
`rand_n8_d8_s1` | 5.663e-01 | 2.220e-16 | 8.882e-16 | 2
`rand_n8_d8_s2` | 6.572e-01 | 0.000e+00 | 0.000e+00 | 2
`rand_n8_d8_s3` | 6.018e-01 | 6.018e-01 | 0.000e+00 | 2
`rand_n8_d8_s4` | 8.768e-01 | 0.000e+00 | 0.000e+00 | 2
`rand_n10_d4_s0` | 8.867e-01 | 8.867e-01 | 0.000e+00 | 2
`rand_n10_d4_s1` | 1.340e-01 | 1.340e-01 | 6.661e-16 | 2
`rand_n10_d4_s2` | 6.664e-03 | 6.664e-03 | 6.661e-16 | 2
`rand_n10_d4_s3` | 9.833e-01 | 9.833e-01 | 4.441e-16 | 2
`rand_n10_d4_s4` | 5.244e-02 | 0.000e+00 | 8.882e-16 | 2
`rand_n10_d8_s0` | 8.003e-01 | 8.003e-01 | 0.000e+00 | 2
`rand_n10_d8_s1` | 1.340e-01 | 1.340e-01 | 0.000e+00 | 2
`rand_n10_d8_s2` | 6.664e-03 | 6.664e-03 | 1.998e-15 | 2
`rand_n10_d8_s3` | 9.833e-01 | 9.833e-01 | 4.441e-16 | 2
`rand_n10_d8_s4` | 6.269e-01 | 0.000e+00 | 2.220e-16 | 2
`rand_n12_d4_s0` | 1.000e+00 | 1.000e+00 | 0.000e+00 | 2
`rand_n12_d4_s1` | 6.675e-01 | 6.675e-01 | 0.000e+00 | 2
`rand_n12_d4_s2` | 7.371e-03 | 7.371e-03 | 0.000e+00 | 2
`rand_n12_d4_s3` | 2.357e-01 | 2.357e-01 | 0.000e+00 | 2
`rand_n12_d4_s4` | 1.000e+00 | 1.000e+00 | 1.110e-15 | 2
`rand_n12_d8_s0` | 1.000e+00 | 1.000e+00 | 0.000e+00 | 2
`rand_n12_d8_s1` | 6.675e-01 | 6.675e-01 | 2.220e-16 | 2
`rand_n12_d8_s2` | 7.371e-03 | 7.371e-03 | 4.441e-16 | 2
`rand_n12_d8_s3` | 2.357e-01 | 2.357e-01 | 2.220e-16 | 2
`rand_n12_d8_s4` | 5.759e-01 | 5.759e-01 | 1.332e-15 | 2
