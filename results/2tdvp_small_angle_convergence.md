# 2TDVP small-angle convergence diagnostic

## Summary

- **h1_L4_plus_rzz_4_7** / `current_local`: error 9.649e-03 → 3.679e-06, p≈1.6539359713053814, monotonic=True, class=finite-step integrator error
- **h1_L4_plus_rzz_4_7** / `local_full_sweep`: error 6.541e-03 → 3.888e-07, p≈2.004730705520417, monotonic=True, class=finite-step integrator error
- **h1_L4_plus_rzz_4_7** / `local_many_sweeps`: error 1.704e-04 → 8.686e-09, p≈2.032095498037779, monotonic=True, class=finite-step integrator error
- **h1_L4_plus_rzz_4_7** / `full_chain_many_sweeps`: error 1.704e-04 → 8.686e-09, p≈2.0320953793537146, monotonic=True, class=finite-step integrator error
- **random_rzz_4_7** / `current_local`: error 8.393e-03 → 2.560e-06, p≈1.7004266267696007, monotonic=True, class=finite-step integrator error
- **random_rzz_4_7** / `local_full_sweep`: error 6.532e-03 → 3.934e-07, p≈2.0024374921706474, monotonic=True, class=finite-step integrator error
- **random_rzz_4_7** / `local_many_sweeps`: error 2.144e-04 → 1.214e-08, p≈2.0132679049394624, monotonic=True, class=finite-step integrator error
- **random_rzz_4_7** / `full_chain_many_sweeps`: error 2.144e-04 → 1.214e-08, p≈2.0132681110235464, monotonic=True, class=finite-step integrator error
- **h2_L2_random_rzz_4_7** / `current_local`: error 8.497e-03 → 2.670e-06, p≈1.6958803619167004, monotonic=True, class=finite-step integrator error
- **h2_L2_random_rzz_4_7** / `local_full_sweep`: error 6.405e-03 → 3.857e-07, p≈2.0024147147826756, monotonic=True, class=finite-step integrator error
- **h2_L2_random_rzz_4_7** / `local_many_sweeps`: error 2.026e-04 → 1.144e-08, p≈2.013563287658694, monotonic=True, class=finite-step integrator error
- **h2_L2_random_rzz_4_7** / `full_chain_many_sweeps`: error 2.026e-04 → 1.144e-08, p≈2.013563421561998, monotonic=True, class=finite-step integrator error
- **control_rzz_2_5** / `current_local`: error 1.008e-08 → 6.021e-11, p≈1.0637972703915501, monotonic=True, class=finite-step integrator error
- **control_rzz_2_5** / `local_full_sweep`: error 1.603e-04 → 9.817e-09, p≈1.999498108789609, monotonic=True, class=finite-step integrator error
- **control_rzz_2_5** / `local_many_sweeps`: error 7.951e-06 → 5.466e-10, p≈1.9824615999900481, monotonic=True, class=finite-step integrator error
- **control_rzz_2_5** / `full_chain_many_sweeps`: error 7.951e-06 → 5.466e-10, p≈1.9815436382076927, monotonic=True, class=finite-step integrator error

## Angle scaling tables

| target | tdvp_config | error_original | error_smallest | estimated_order_p | monotonic |
|---|---|---:|---:|---:|---|
| h1_L4_plus_rzz_4_7 | current_local | 9.649e-03 | 3.679e-06 | 1.65 | True |
| h1_L4_plus_rzz_4_7 | local_full_sweep | 6.541e-03 | 3.888e-07 | 2.00 | True |
| h1_L4_plus_rzz_4_7 | local_many_sweeps | 1.704e-04 | 8.686e-09 | 2.03 | True |
| h1_L4_plus_rzz_4_7 | full_chain_many_sweeps | 1.704e-04 | 8.686e-09 | 2.03 | True |
| random_rzz_4_7 | current_local | 8.393e-03 | 2.560e-06 | 1.70 | True |
| random_rzz_4_7 | local_full_sweep | 6.532e-03 | 3.934e-07 | 2.00 | True |
| random_rzz_4_7 | local_many_sweeps | 2.144e-04 | 1.214e-08 | 2.01 | True |
| random_rzz_4_7 | full_chain_many_sweeps | 2.144e-04 | 1.214e-08 | 2.01 | True |
| h2_L2_random_rzz_4_7 | current_local | 8.497e-03 | 2.670e-06 | 1.70 | True |
| h2_L2_random_rzz_4_7 | local_full_sweep | 6.405e-03 | 3.857e-07 | 2.00 | True |
| h2_L2_random_rzz_4_7 | local_many_sweeps | 2.026e-04 | 1.144e-08 | 2.01 | True |
| h2_L2_random_rzz_4_7 | full_chain_many_sweeps | 2.026e-04 | 1.144e-08 | 2.01 | True |
| control_rzz_2_5 | current_local | 1.008e-08 | 6.021e-11 | 1.06 | True |
| control_rzz_2_5 | local_full_sweep | 1.603e-04 | 9.817e-09 | 2.00 | True |
| control_rzz_2_5 | local_many_sweeps | 7.951e-06 | 5.466e-10 | 1.98 | True |
| control_rzz_2_5 | full_chain_many_sweeps | 7.951e-06 | 5.466e-10 | 1.98 | True |

## Substep convergence

| target | m | error | wall_time | max_bond |
|---|---|---:|---:|---:|
| h1_L4_plus_rzz_4_7 | 1 | 1.704e-04 | 0.196 | 15 |
| h1_L4_plus_rzz_4_7 | 2 | 2.409e-04 | 0.542 | 15 |
| h1_L4_plus_rzz_4_7 | 4 | 3.077e-04 | 1.242 | 15 |
| h1_L4_plus_rzz_4_7 | 8 | 3.497e-04 | 2.057 | 15 |
| h1_L4_plus_rzz_4_7 | 16 | 3.730e-04 | 3.925 | 15 |
| h1_L4_plus_rzz_4_7 | 32 | 3.852e-04 | 7.610 | 15 |
| random_rzz_4_7 | 1 | 2.144e-04 | 0.218 | 15 |
| random_rzz_4_7 | 2 | 2.659e-04 | 0.413 | 15 |
| random_rzz_4_7 | 4 | 3.131e-04 | 0.946 | 15 |
| random_rzz_4_7 | 8 | 3.425e-04 | 2.193 | 15 |
| random_rzz_4_7 | 16 | 3.587e-04 | 3.803 | 15 |
| random_rzz_4_7 | 32 | 3.672e-04 | 8.101 | 15 |
| h2_L2_random_rzz_4_7 | 1 | 2.026e-04 | 0.299 | 15 |
| h2_L2_random_rzz_4_7 | 2 | 2.569e-04 | 0.713 | 15 |
| h2_L2_random_rzz_4_7 | 4 | 3.066e-04 | 1.316 | 15 |
| h2_L2_random_rzz_4_7 | 8 | 3.373e-04 | 2.369 | 15 |
| h2_L2_random_rzz_4_7 | 16 | 3.542e-04 | 4.556 | 15 |
| h2_L2_random_rzz_4_7 | 32 | 3.631e-04 | 8.259 | 15 |
| control_rzz_2_5 | 1 | 7.951e-06 | 0.136 | 5 |
| control_rzz_2_5 | 2 | 7.943e-06 | 0.483 | 4 |
| control_rzz_2_5 | 4 | 7.938e-06 | 1.203 | 4 |
| control_rzz_2_5 | 8 | 7.935e-06 | 1.960 | 4 |
| control_rzz_2_5 | 16 | 7.934e-06 | 3.609 | 4 |
| control_rzz_2_5 | 32 | 7.933e-06 | 8.187 | 4 |

## Reference sanity checks

| target | theta | tebd | enrichment | dense |
|---|---|---:|---:|---:|
| h1_L4_plus_rzz_4_7 | 0.2 | 9.159e-13 | 9.166e-13 | 9.168e-13 |
| random_rzz_4_7 | 0.2 | 9.148e-13 | 9.166e-13 | 9.164e-13 |
| h2_L2_random_rzz_4_7 | 0.2 | 1.783e-12 | 1.787e-12 | 1.788e-12 |
| control_rzz_2_5 | 0.2 | 0.000e+00 | 9.548e-13 | 0.000e+00 |

## Interpretation

- **h1_L4_plus_rzz_4_7**: finite-step integrator error
- **random_rzz_4_7**: finite-step integrator error
- **h2_L2_random_rzz_4_7**: finite-step integrator error
- **control_rzz_2_5**: finite-step integrator error

## Final answers

**h1_L4_plus_rzz_4_7**: θ smallest error=8.686e-09, θ original error=1.704e-04, p≈2.03.
**random_rzz_4_7**: θ smallest error=1.214e-08, θ original error=2.144e-04, p≈2.01.
**h2_L2_random_rzz_4_7**: θ smallest error=1.144e-08, θ original error=2.026e-04, p≈2.01.
**control_rzz_2_5**: θ smallest error=5.466e-10, θ original error=7.951e-06, p≈1.98.
1. **θ→0**: See per-target monotonic flag and error_at_smallest_theta in scaling table.
2. **Full-chain vs local**: Compare `full_chain_many_sweeps` vs `current_local` columns.
3. **Substepping**: See substep table vs enrichment at original θ.
4. **Order p**: From log-log fit on TDVP rows with 1e-14 < error < 1e-2.
5. **Parameter vs bug**: Classifications in Interpretation section.
6. **Production TDVP route**: Only viable if full-chain+substeps reach ≤1e-6 at circuit θ; else use enrichment.
