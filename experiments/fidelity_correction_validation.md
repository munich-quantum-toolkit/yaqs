# Normalized fidelity correction — validation report

## Norm storage

`norm_before` / `norm_after` store the Euclidean L2 norm `‖ψ‖ = sqrt(⟨ψ|ψ⟩)`, not `⟨ψ|ψ⟩`.
Backfill used `F_norm = overlap_squared_raw / (norm_before² · norm_after²)` without rerunning simulations.

## Before/after infidelity at θ/(2π) ∈ {10⁻³, 10⁻², 10⁻¹}

| method | χ_max | x | I_raw | I_norm | Δ(I_raw−I_norm) | ‖approx‖ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hybrid_tdvp | 8 | 1e-03 | 6.265909e-06 | 6.251035e-06 | 1.487445e-08 | 1.000000 |
| hybrid_tdvp | 8 | 1e-02 | 6.231381e-04 | 6.231381e-04 | 1.332268e-15 | 1.000000 |
| hybrid_tdvp | 8 | 1e-01 | 3.280613e-02 | 3.280613e-02 | 0.000000e+00 | 1.000000 |
| hybrid_tdvp | 12 | 1e-03 | 1.624195e-07 | 1.614153e-07 | 1.004203e-09 | 1.000000 |
| hybrid_tdvp | 12 | 1e-02 | 1.619067e-05 | 1.609050e-05 | 1.001655e-07 | 1.000000 |
| hybrid_tdvp | 12 | 1e-01 | 1.394913e-03 | 1.394913e-03 | 1.332268e-15 | 1.000000 |
| hybrid_tdvp | 16 | 1e-03 | 2.229596e-09 | 2.212500e-09 | 1.709566e-11 | 1.000000 |
| hybrid_tdvp | 16 | 1e-02 | 2.532066e-10 | 2.514879e-10 | 1.718625e-12 | 1.000000 |
| hybrid_tdvp | 16 | 1e-01 | 8.909318e-12 | 8.963497e-12 | 5.417888e-14 | 1.000000 |
| tebd_swap | 8 | 1e-03 | 5.503614e-01 | 2.997828e-01 | 2.505786e-01 | 0.801337 |
| tebd_swap | 8 | 1e-02 | 5.510276e-01 | 3.002953e-01 | 2.507323e-01 | 0.801037 |
| tebd_swap | 8 | 1e-01 | 5.625603e-01 | 3.095156e-01 | 2.530446e-01 | 0.795943 |
| tebd_swap | 12 | 1e-03 | 4.203710e-01 | 2.440661e-01 | 1.763049e-01 | 0.875655 |
| tebd_swap | 12 | 1e-02 | 4.209809e-01 | 2.445488e-01 | 1.764320e-01 | 0.875474 |
| tebd_swap | 12 | 1e-01 | 4.312324e-01 | 2.521355e-01 | 1.790969e-01 | 0.872079 |
| tebd_swap | 16 | 1e-03 | -6.217249e-15 | 0.000000e+00 | 6.217249e-15 | 1.000000 |
| tebd_swap | 16 | 1e-02 | -2.220446e-15 | 0.000000e+00 | 2.220446e-15 | 1.000000 |
| tebd_swap | 16 | 1e-01 | 1.998401e-15 | 0.000000e+00 | 1.998401e-15 | 1.000000 |
| mpo_zipup | 8 | 1e-03 | 1.051320e-01 | 5.402539e-02 | 5.110665e-02 | 0.972612 |
| mpo_zipup | 8 | 1e-02 | 1.051320e-01 | 5.402539e-02 | 5.110665e-02 | 0.972612 |
| mpo_zipup | 8 | 1e-01 | 1.051320e-01 | 5.402539e-02 | 5.110665e-02 | 0.972612 |
| mpo_zipup | 12 | 1e-03 | 1.250900e-02 | 6.274185e-03 | 6.234820e-03 | 0.996858 |
| mpo_zipup | 12 | 1e-02 | 1.250900e-02 | 6.274185e-03 | 6.234820e-03 | 0.996858 |
| mpo_zipup | 12 | 1e-01 | 1.250900e-02 | 6.274185e-03 | 6.234820e-03 | 0.996858 |
| mpo_zipup | 16 | 1e-03 | -3.552714e-15 | 0.000000e+00 | 3.552714e-15 | 1.000000 |
| mpo_zipup | 16 | 1e-02 | -8.881784e-16 | 0.000000e+00 | 8.881784e-16 | 1.000000 |
| mpo_zipup | 16 | 1e-01 | 2.886580e-15 | 0.000000e+00 | 2.886580e-15 | 1.000000 |
| variational_mpo | 8 | 1e-03 | 1.051320e-01 | 5.402539e-02 | 5.110665e-02 | 0.972612 |
| variational_mpo | 8 | 1e-02 | 1.051320e-01 | 5.402539e-02 | 5.110665e-02 | 0.972612 |
| variational_mpo | 8 | 1e-01 | 1.051320e-01 | 5.402539e-02 | 5.110665e-02 | 0.972612 |
| variational_mpo | 12 | 1e-03 | 1.250900e-02 | 6.274185e-03 | 6.234820e-03 | 0.996858 |
| variational_mpo | 12 | 1e-02 | 1.250900e-02 | 6.274185e-03 | 6.234820e-03 | 0.996858 |
| variational_mpo | 12 | 1e-01 | 1.250900e-02 | 6.274185e-03 | 6.234820e-03 | 0.996858 |
| variational_mpo | 16 | 1e-03 | -3.552714e-15 | 0.000000e+00 | 3.552714e-15 | 1.000000 |
| variational_mpo | 16 | 1e-02 | -8.881784e-16 | 0.000000e+00 | 8.881784e-16 | 1.000000 |
| variational_mpo | 16 | 1e-01 | 2.886580e-15 | 0.000000e+00 | 2.886580e-15 | 1.000000 |

**Maximum |I_raw − I_norm| over all backfilled rows:** 2.634535e-01

## TDVP small-angle fitted exponents

Fit of `I ∝ x^α` on `x ∈ [0.0001, 0.01]`.

| χ_max | α_raw | α_norm |
| ---: | ---: | ---: |
| 8 | 1.9996 | 2.0000 |
| 12 | 1.8382 | 1.8382 |
| 16 | -0.6006 | -0.6009 |

## Method ordering at θ/(2π)=0.1

- χ=8 raw: hybrid_tdvp(3.281e-02) < mpo_zipup(1.051e-01) < variational_mpo(1.051e-01) < tebd_swap(5.626e-01)
- χ=8 norm: hybrid_tdvp(3.281e-02) < mpo_zipup(5.403e-02) < variational_mpo(5.403e-02) < tebd_swap(3.095e-01)
- χ=12 raw: hybrid_tdvp(1.395e-03) < mpo_zipup(1.251e-02) < variational_mpo(1.251e-02) < tebd_swap(4.312e-01)
- χ=12 norm: hybrid_tdvp(1.395e-03) < mpo_zipup(6.274e-03) < variational_mpo(6.274e-03) < tebd_swap(2.521e-01)
- χ=16 raw: tebd_swap(1.998e-15) < mpo_zipup(2.887e-15) < variational_mpo(2.887e-15) < hybrid_tdvp(8.909e-12)
- χ=16 norm: mpo_zipup(0.000e+00) < tebd_swap(0.000e+00) < variational_mpo(0.000e+00) < hybrid_tdvp(8.963e-12)

## χ_max selection

- Before: χ_full=16 (unchanged in backup meta)
- After: χ_full=16, χ_mid=12 (`relaxed_tdvp: smallest chi with TEBD/MPO/variational <= 1e-10 and TDVP <= 1e-8 (worst TDVP=2.213e-09)`)
- **No change** in selected χ.

## Substep interpretation (θ/(2π)=0.1)

TDVP substep series: compare I(n=1) vs I(n=64) and whether increasing n helps.

| χ | I_raw(n=1) | I_raw(n=64) | I_norm(n=1) | I_norm(n=64) |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 1.950984e-02 | 3.280613e-02 | 1.950984e-02 | 3.280613e-02 |
| 12 | 1.456000e-03 | 1.394913e-03 | 1.456000e-03 | 1.394913e-03 |
| 16 | -2.664535e-15 | 8.909318e-12 | 4.440892e-16 | 8.963497e-12 |

TDVP max |norm_loss| in angle sweep: 2.309e-07. Substep curves are essentially unchanged under normalization because TDVP preserves norm.

## Representative values (normalized)

| method | χ | x=1e-3 | x=1e-2 | x=1e-1 |
| --- | ---: | ---: | ---: | ---: |
| hybrid_tdvp | 8 | 6.251e-06 | 6.231e-04 | 3.281e-02 |
| hybrid_tdvp | 12 | 1.614e-07 | 1.609e-05 | 1.395e-03 |
| hybrid_tdvp | 16 | 2.213e-09 | 2.515e-10 | 8.963e-12 |
| tebd_swap | 8 | 2.998e-01 | 3.003e-01 | 3.095e-01 |
| tebd_swap | 12 | 2.441e-01 | 2.445e-01 | 2.521e-01 |
| tebd_swap | 16 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| mpo_zipup | 8 | 5.403e-02 | 5.403e-02 | 5.403e-02 |
| mpo_zipup | 12 | 6.274e-03 | 6.274e-03 | 6.274e-03 |
| mpo_zipup | 16 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| variational_mpo | 8 | 5.403e-02 | 5.403e-02 | 5.403e-02 |
| variational_mpo | 12 | 6.274e-03 | 6.274e-03 | 6.274e-03 |
| variational_mpo | 16 | 0.000e+00 | 0.000e+00 | 0.000e+00 |

## Scientific conclusions

- Selected χ_max unchanged (8 / 12 / 16).
- Non-TDVP methods at χ=16 remain at numerical precision.
- TEBD/MPO compressed-χ absolute infidelity drops (less pessimistic) when norm loss is factored out; method ordering at low χ can change quantitatively but the qualitative picture (TDVP best at low χ for small angles; TEBD routing floor; MPO exact at full χ) is unchanged.
- No automatic regeneration of fixed_resources / resource_frontier / convergence.
