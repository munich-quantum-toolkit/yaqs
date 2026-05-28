# Hybrid TDVP Benchmark Report

Generated: 2026-05-27T16:19:03.054279+00:00

## Settings

- **gate_mode_default**: `hybrid`
- **tdvp_circuit_full_sweep_default**: `False`
- **tdvp_sweeps_default**: `4`
- **svd_threshold**: `1e-14`
- **observable_panel**: `Pauli X, Y, Z on each qubit`
- **simulator_auto_pad**: `product states padded to chi >= 2 before gates`
- **notes**:
  - fidelity_error = 1 - |<psi_ref|psi_sim>|^2
  - obs_max_error = max |<O>_sim - <O>_ref| over X/Y/Z panel
  - padding study uses direct MPS path (no simulator auto-pad); pad=None is chi=1

## 1. Baseline: hybrid vs TEBD (4 sweeps, partial LR, simulator path)

| probe | mode | fidelity | 1-fidelity | obs_max | obs_l2 | worst_obs |
|---|---:|---:|---:|---:|---:|---|
| interior_lr_4q | hybrid | 1.000000 | 4.44e-16 | 0.0000 | 0.0000 | z@3 err=0.0000 |
| interior_lr_4q | tebd | 1.000000 | 4.44e-16 | 0.0000 | 0.0000 | z@3 err=0.0000 |
| interior_lr_8q | hybrid | 1.000000 | 4.44e-16 | 0.0000 | 0.0000 | z@7 err=0.0000 |
| interior_lr_8q | tebd | 1.000000 | -4.44e-16 | 0.0000 | 0.0000 | x@7 err=0.0000 |
| endpoint_lr_8q | hybrid | 1.000000 | -4.44e-16 | 0.0000 | 0.0000 | y@7 err=0.0000 |
| endpoint_lr_8q | tebd | 1.000000 | 0.00e+00 | 0.0000 | 0.0000 | z@7 err=0.0000 |
| mid_ctrl_8q | hybrid | 0.961940 | 3.81e-02 | 0.3827 | 0.5412 | y@4 err=0.3827 |
| mid_ctrl_8q | tebd | 1.000000 | 4.44e-16 | 0.0000 | 0.0000 | z@7 err=0.0000 |
| double_lr_6q | hybrid | 0.457867 | 5.42e-01 | 1.0000 | 1.4142 | y@5 err=1.0000 |
| double_lr_6q | tebd | 1.000000 | 4.44e-16 | 0.0000 | 0.0000 | z@3 err=0.0000 |
| mixed_nn_lr_4q | hybrid | 1.000000 | 2.22e-16 | 0.0000 | 0.0000 | y@3 err=0.0000 |
| mixed_nn_lr_4q | tebd | 1.000000 | 0.00e+00 | 0.0000 | 0.0000 | z@3 err=0.0000 |
| ghz_chain_6q | hybrid | 1.000000 | 4.44e-16 | 0.0000 | 0.0000 | z@5 err=0.0000 |
| ghz_chain_6q | tebd | 1.000000 | 4.44e-16 | 0.0000 | 0.0000 | z@5 err=0.0000 |

## 2. Effect of `tdvp_sweeps` (hybrid, partial LR, simulator)

| probe | sweeps | fidelity | 1-fidelity | obs_max | worst_obs |
|---|---:|---:|---:|---:|---|
| interior_lr_4q | 1 | 1.000000 | 0.00e+00 | 0.0000 | z@3 (0.0000) |
| interior_lr_4q | 2 | 1.000000 | 2.22e-16 | 0.0000 | z@3 (0.0000) |
| interior_lr_4q | 4 | 1.000000 | 4.44e-16 | 0.0000 | z@3 (0.0000) |
| interior_lr_4q | 8 | 1.000000 | 2.22e-16 | 0.0000 | z@3 (0.0000) |
| interior_lr_4q | 16 | 1.000000 | 2.22e-16 | 0.0000 | z@3 (0.0000) |
| interior_lr_4q | 32 | 1.000000 | 6.66e-16 | 0.0000 | z@3 (0.0000) |
| interior_lr_8q | 1 | 1.000000 | 4.44e-16 | 0.0000 | z@7 (0.0000) |
| interior_lr_8q | 2 | 1.000000 | 0.00e+00 | 0.0000 | z@7 (0.0000) |
| interior_lr_8q | 4 | 1.000000 | 4.44e-16 | 0.0000 | z@7 (0.0000) |
| interior_lr_8q | 8 | 1.000000 | -4.44e-16 | 0.0000 | z@6 (0.0000) |
| interior_lr_8q | 16 | 1.000000 | 8.88e-16 | 0.0000 | z@7 (0.0000) |
| interior_lr_8q | 32 | 1.000000 | 0.00e+00 | 0.0000 | z@7 (0.0000) |
| mid_ctrl_8q | 1 | 0.500000 | 5.00e-01 | 1.0000 | y@4 (1.0000) |
| mid_ctrl_8q | 2 | 0.853553 | 1.46e-01 | 0.7071 | y@4 (0.7071) |
| mid_ctrl_8q | 4 | 0.961940 | 3.81e-02 | 0.3827 | y@4 (0.3827) |
| mid_ctrl_8q | 8 | 0.990393 | 9.61e-03 | 0.1951 | y@0 (0.1951) |
| mid_ctrl_8q | 16 | 0.997592 | 2.41e-03 | 0.0980 | y@4 (0.0980) |
| mid_ctrl_8q | 32 | 0.999398 | 6.02e-04 | 0.0491 | y@4 (0.0491) |
| double_lr_6q | 1 | 0.500000 | 5.00e-01 | 1.0000 | y@5 (1.0000) |
| double_lr_6q | 2 | 0.480970 | 5.19e-01 | 1.0000 | y@5 (1.0000) |
| double_lr_6q | 4 | 0.457867 | 5.42e-01 | 1.0000 | y@5 (1.0000) |
| double_lr_6q | 8 | 0.443253 | 5.57e-01 | 1.0000 | y@5 (1.0000) |
| double_lr_6q | 16 | 0.435238 | 5.65e-01 | 1.0000 | y@5 (1.0000) |
| double_lr_6q | 32 | 0.431062 | 5.69e-01 | 1.0000 | y@5 (1.0000) |

## 3. Effect of initial bond padding (hybrid, direct MPS, 4 sweeps)

| probe | initial_pad | init_bond | fidelity | 1-fidelity | obs_max | worst_obs |
|---|---:|---:|---:|---:|---:|---|
| interior_lr_4q | None | 1 | 0.500000 | 5.00e-01 | 1.0000 | y@3 (1.0000) |
| interior_lr_4q | 2 | 2 | 1.000000 | 5.33e-15 | 0.0000 | z@2 (0.0000) |
| interior_lr_4q | 4 | 4 | 1.000000 | 5.33e-15 | 0.0000 | z@2 (0.0000) |
| interior_lr_4q | 8 | 4 | 1.000000 | 5.33e-15 | 0.0000 | z@2 (0.0000) |
| interior_lr_4q | 16 | 4 | 1.000000 | 5.33e-15 | 0.0000 | z@2 (0.0000) |
| interior_lr_8q | None | 1 | 0.500000 | 5.00e-01 | 1.0000 | y@7 (1.0000) |
| interior_lr_8q | 2 | 2 | 1.000000 | 1.11e-14 | 0.0000 | z@6 (0.0000) |
| interior_lr_8q | 4 | 4 | 1.000000 | 2.22e-16 | 0.0000 | z@7 (0.0000) |
| interior_lr_8q | 8 | 8 | 1.000000 | 4.22e-15 | 0.0000 | z@6 (0.0000) |
| interior_lr_8q | 16 | 16 | 1.000000 | 1.78e-15 | 0.0000 | z@6 (0.0000) |
| mid_ctrl_8q | None | 1 | 1.000000 | 7.55e-15 | 0.0000 | z@7 (0.0000) |
| mid_ctrl_8q | 2 | 2 | 0.961940 | 3.81e-02 | 0.3827 | y@4 (0.3827) |
| mid_ctrl_8q | 4 | 4 | 0.961940 | 3.81e-02 | 0.3827 | y@4 (0.3827) |
| mid_ctrl_8q | 8 | 8 | 0.961940 | 3.81e-02 | 0.3827 | y@4 (0.3827) |
| mid_ctrl_8q | 16 | 16 | 0.961940 | 3.81e-02 | 0.3827 | y@4 (0.3827) |
| double_lr_6q | None | 1 | 0.384639 | 6.15e-01 | 0.9989 | y@4 (0.9989) |
| double_lr_6q | 2 | 2 | 0.426777 | 5.73e-01 | 1.0000 | y@5 (1.0000) |
| double_lr_6q | 4 | 4 | 0.426777 | 5.73e-01 | 1.0000 | y@5 (1.0000) |
| double_lr_6q | 8 | 8 | 0.426777 | 5.73e-01 | 1.0000 | y@5 (1.0000) |
| double_lr_6q | 16 | 8 | 0.426777 | 5.73e-01 | 1.0000 | y@5 (1.0000) |

## 4. Partial LR vs symmetric LR+RL (`tdvp_circuit_full_sweep`)

| probe | full_sweep | fidelity | 1-fidelity | obs_max | worst_obs |
|---|---:|---:|---:|---:|---|
| interior_lr_4q | False | 1.000000 | 4.44e-16 | 0.0000 | z@3 (0.0000) |
| interior_lr_4q | True | 1.000000 | 0.00e+00 | 0.0000 | z@3 (0.0000) |
| interior_lr_8q | False | 1.000000 | 4.44e-16 | 0.0000 | z@7 (0.0000) |
| interior_lr_8q | True | 1.000000 | -4.44e-16 | 0.0000 | z@6 (0.0000) |
| mid_ctrl_8q | False | 0.961940 | 3.81e-02 | 0.3827 | y@4 (0.3827) |
| mid_ctrl_8q | True | 0.500000 | 5.00e-01 | 1.0000 | y@4 (1.0000) |
| double_lr_6q | False | 0.457867 | 5.42e-01 | 1.0000 | y@5 (1.0000) |
| double_lr_6q | True | 0.500000 | 5.00e-01 | 1.0000 | y@5 (1.0000) |

## Key takeaways for downstream analysis

- **Fidelity vs observables**: Z-only checks can miss errors; include Y (and X) in the panel.
- **Padding**: chi=1 (pad=None, direct path) fails interior LR probes; chi>=2 fixes single-LR cases.
- **Sweeps**: help mid-control partial LR converge in fidelity; do not fix double-LR or symmetric mode.
- **TEBD reference**: use `gate_mode=tebd` rows in baseline for exact comparison on hard probes.
