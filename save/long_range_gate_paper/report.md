# Controlled accuracy of long-range gate application (v3 paper benchmark)

- Command: `completed benchmark`
- Total runtime: 6289.7 s
- Rows: 4833
- Sequence validation passed: True

## 1. TDVP angle-scaling exponents (χ=8)

- RXX: median slope=1.9868332219920803, range=[1.9705802336618785, 2.000394664612786] across seeds
- RYY: median slope=1.9956038436024153, range=[1.950033844642349, 1.9964541094297694] across seeds
- RZZ: median slope=1.9941037861764739, range=[1.9548821021660188, 1.9953883983848786] across seeds

## 2. Lowest final error by χ (gate 36)

- χ=8 commuting: best=hybrid_tdvp (1.513e-01)
- χ=8 mixed: best=hybrid_tdvp (1.660e-01)
- χ=16 commuting: best=hybrid_tdvp (5.916e-03)
- χ=16 mixed: best=hybrid_tdvp (1.785e-02)
- χ=32 commuting: best=mpo_zipup (0.000e+00)
- χ=32 mixed: best=hybrid_tdvp (4.928e-05)
- χ=64 commuting: best=tebd_swap (0.000e+00)
- χ=64 mixed: best=tebd_swap (0.000e+00)

## 3–4. Variational MPO vs zip-up / TDVP

- commuting χ=64 median infidelity: zip-up=0.000e+00, variational=0.000e+00, TDVP=9.558e-11
- mixed χ=64 median infidelity: zip-up=0.000e+00, variational=0.000e+00, TDVP=9.190e-12

## 5–6. Fixed-accuracy resources

- commuting hybrid_tdvp seed=11 infidelity_1e-2: χ=16
- commuting hybrid_tdvp seed=11 infidelity_1e-3: χ=32
- commuting hybrid_tdvp seed=11 max_obs_1e-2: χ=16
- commuting hybrid_tdvp seed=11 max_obs_1e-3: χ=32
- commuting hybrid_tdvp seed=22 infidelity_1e-2: χ=16
- commuting hybrid_tdvp seed=22 infidelity_1e-3: χ=32
- commuting hybrid_tdvp seed=22 max_obs_1e-2: χ=32
- commuting hybrid_tdvp seed=22 max_obs_1e-3: χ=32
- commuting hybrid_tdvp seed=33 infidelity_1e-2: χ=16
- commuting hybrid_tdvp seed=33 infidelity_1e-3: χ=32
- commuting hybrid_tdvp seed=33 max_obs_1e-2: χ=16
- commuting hybrid_tdvp seed=33 max_obs_1e-3: χ=32
- commuting tebd_swap seed=11 infidelity_1e-2: χ=64
- commuting tebd_swap seed=11 infidelity_1e-3: χ=64
- commuting tebd_swap seed=11 max_obs_1e-2: χ=64
- commuting tebd_swap seed=11 max_obs_1e-3: χ=64
- commuting tebd_swap seed=22 infidelity_1e-2: χ=32
- commuting tebd_swap seed=22 infidelity_1e-3: χ=64
- commuting tebd_swap seed=22 max_obs_1e-2: χ=64
- commuting tebd_swap seed=22 max_obs_1e-3: χ=64

## Pareto frontier (gate 36)

- hybrid_tdvp commuting χ=8: params=936, inf=5.696e-02
- hybrid_tdvp commuting χ=16: params=2728, inf=1.591e-03
- tebd_swap commuting χ=64: params=6824, inf=0.000e+00
- tebd_swap commuting χ=64: params=6824, inf=0.000e+00
- tebd_swap commuting χ=32: params=6824, inf=0.000e+00

## Supported claims vs overstatements

- Supported: method ordering and scaling trends when validation passes and seed spread is reported.
- Overstated without caution: single-gate runtime comparisons; discarded weight as error bound.
