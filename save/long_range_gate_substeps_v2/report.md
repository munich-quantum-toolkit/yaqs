# Long-range gate and TDVP substep benchmark report (v2, seed-11 pilot)

- Command: `long_range_gate_study.py --seed 11 --output-dir ../../save/long_range_gate_substeps_v2 --validate-sequences`
- Total runtime: 514.1 s
- Output directory: `/home/aaron/Github/yaqs/save/long_range_gate_substeps_v2`
- Authoritative store: `/home/aaron/Github/yaqs/save/long_range_gate_substeps_v2/results.sqlite`
- Sequence validation passed: True

## Root cause (v1 sequence reference bug)

The v1 benchmark used `apply_two_qubit_dense` with a reshape `(before, 2, middle, 2, after)` that only matches the MPS/Qiskit statevector convention when `q0 + q1 = L - 1`. For general long-range pairs such as RZZ(2,8) and RZZ(3,9), the dense reference was wrong while TEBD/MPO/TDVP agreed with each other. This produced identical method infidelities against an incorrect exact trajectory (e.g. ~1.28e-3 at layer-internal gate 2 and ~0.313 final mixed-sequence error at χ=64).

## Quantitative answers

- Final commuting infidelity (tebd_swap, χ=64): 2.262e-09
- Final commuting infidelity (mpo_zipup, χ=64): 0.000e+00
- Final commuting infidelity (hybrid_tdvp, χ=64): 1.576e-11
- Final mixed infidelity (tebd_swap, χ=64): 0.000e+00
- Final mixed infidelity (mpo_zipup, χ=64): 0.000e+00
- Final mixed infidelity (hybrid_tdvp, χ=64): 4.832e-10
- Maximum TEBD infidelity at χ=64 (all gates): 8.128e-09
- Maximum MPO infidelity at χ=64 (all gates): 5.107e-15

### χ=8 angle-scaling slopes (s=1)

- RXX: 1.959e+00
- RYY: 1.927e+00
- RZZ: 1.934e+00

### Identity controls (A0, θ=0)

- RXX hybrid_tdvp χ=8: 1.776e-15
- RXX tebd_swap χ=8: 5.503e-01
- RXX mpo_zipup χ=8: 2.665e-15
- RYY hybrid_tdvp χ=8: 1.776e-15
- RYY tebd_swap χ=8: 5.503e-01
- RYY mpo_zipup χ=8: 2.665e-15
- RZZ hybrid_tdvp χ=8: 1.776e-15
- RZZ tebd_swap χ=8: 5.503e-01
- RZZ mpo_zipup χ=8: 2.665e-15
- RXX hybrid_tdvp χ=64: 1.776e-15
- RXX tebd_swap χ=64: 0.000e+00
- RXX mpo_zipup χ=64: 2.665e-15
- RYY hybrid_tdvp χ=64: 1.776e-15
- RYY tebd_swap χ=64: 0.000e+00
- RYY mpo_zipup χ=64: 2.665e-15
- RZZ hybrid_tdvp χ=64: 1.776e-15
- RZZ tebd_swap χ=64: 0.000e+00
- RZZ mpo_zipup χ=64: 2.665e-15

### Local vs full TDVP

| method | χmax | substeps | infidelity vs exact | infidelity local vs full | runtime | max bond |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_tdvp | 16 | 1 | 0.000e+00 | N/A | 0.090 | 16 |
| full_tdvp | 16 | 4 | 1.173e-10 | N/A | 0.289 | 16 |
| full_tdvp | 64 | 1 | 0.000e+00 | N/A | 0.043 | 16 |
| full_tdvp | 64 | 4 | 1.173e-10 | N/A | 0.069 | 16 |
| hybrid_tdvp | 16 | 1 | 0.000e+00 | N/A | 0.104 | 16 |
| hybrid_tdvp | 16 | 4 | 1.173e-10 | N/A | 0.117 | 16 |
| hybrid_tdvp | 64 | 1 | 0.000e+00 | N/A | 0.020 | 16 |
| hybrid_tdvp | 64 | 4 | 1.173e-10 | N/A | 0.193 | 16 |
- Local and full TDVP agree within numerical precision: True
- cross χ=16 s=1: local-vs-full infidelity=0.000e+00, runtime=0.043 s
- cross χ=16 s=4: local-vs-full infidelity=0.000e+00, runtime=0.193 s
- cross χ=64 s=1: local-vs-full infidelity=0.000e+00, runtime=0.074 s
- cross χ=64 s=4: local-vs-full infidelity=0.000e+00, runtime=0.199 s

### Discarded-weight diagnostic (TDVP only)

| experiment | TDVP rows | high infidelity | high inf + negligible discard | max inf (negligible discard) |
| --- | ---: | ---: | ---: | ---: |
| A | 144 | 24 | 0 | N/A |
| B | 36 | 8 | 0 | N/A |
| commuting | 432 | 160 | 0 | N/A |
| mixed | 432 | 207 | 0 | N/A |
