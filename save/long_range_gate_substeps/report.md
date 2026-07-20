# Long-range gate and TDVP substep benchmark report (seed-11 pilot)

- Command: `long_range_gate_study.py --seed 11 --output-dir /home/aaron/Github/yaqs/save/long_range_gate_substeps --resume`
- Total runtime: 411.8 s
- Output directory: `/home/aaron/Github/yaqs/save/long_range_gate_substeps`
- Authoritative store: `/home/aaron/Github/yaqs/save/long_range_gate_substeps/results.sqlite`

## Pilot note

This run uses seed 11 only. Summary values are direct seed-11 results (no seed aggregation).

## Quantitative answers

- RXX θ=0.1, χ=64, s=1: infidelity = 0.000e+00
- RXX θ=0.1, χ=64, s=2: infidelity = 4.699e-10
- RXX θ=0.1, χ=64, s=4: infidelity = 1.173e-10
- RXX θ=0.1, χ=64, s=8: infidelity = 2.930e-11

- RXX θ-scaling slope (s=1, χ=64): 2.001
- RYY θ-scaling slope (s=1, χ=64): nan
- RZZ θ-scaling slope (s=1, χ=64): nan

- TDVP cases with high infidelity and negligible discarded weight: 192

### Local vs full TDVP

- exact χ= s=: infidelity=1.332e-15
- full_tdvp χ=16 s=1: infidelity=0.000e+00
- full_tdvp χ=16 s=4: infidelity=1.173e-10
- full_tdvp χ=64 s=1: infidelity=0.000e+00
- full_tdvp χ=64 s=4: infidelity=1.173e-10
- hybrid_tdvp χ=16 s=1: infidelity=0.000e+00
- hybrid_tdvp χ=16 s=4: infidelity=1.173e-10
- hybrid_tdvp χ=64 s=1: infidelity=0.000e+00
- hybrid_tdvp χ=64 s=4: infidelity=1.173e-10
- local_vs_full_cross χ=16 s=1: infidelity=0.000e+00
- local_vs_full_cross χ=16 s=4: infidelity=0.000e+00
- local_vs_full_cross χ=64 s=1: infidelity=0.000e+00
- local_vs_full_cross χ=64 s=4: infidelity=0.000e+00

