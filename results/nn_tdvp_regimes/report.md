# NN brickwork: TEBD vs all-TDVP

Stage **1**. Circuits: ``nn_brickwork`` only. ``tdvp_all`` applies TDVP to every two-qubit gate (including NN).

## Summary

- Comparisons: 162; TDVP wins: 2; TEBD wins: 115; similar: 45
- TDVP faster than TEBD: 0/162; TDVP more accurate (err ratio>1.5): 6/162
- TDVP lower peak χ than TEBD: 43/162

## Largest TDVP advantage (TEBD/TDVP error ratio)

- `nn_brickwork_n16_d8_s2_medium` χ=64: ratio=3.49, outcome=tdvp_wins
- `nn_brickwork_n16_d8_s0_medium` χ=32: ratio=2.80, outcome=tdvp_wins
- `nn_brickwork_n8_d4_s1_small` χ=16: ratio=1.63, outcome=similar
- `nn_brickwork_n8_d4_s1_small` χ=32: ratio=1.63, outcome=similar
- `nn_brickwork_n8_d4_s1_small` χ=64: ratio=1.63, outcome=similar
- `nn_brickwork_n12_d8_s2_medium` χ=32: ratio=1.61, outcome=similar

## Largest TEBD advantage (TDVP/TEBD error ratio)

- `nn_brickwork_n8_d12_s2_medium` χ=16: ratio=3103438286166.28
- `nn_brickwork_n8_d12_s2_medium` χ=32: ratio=3103438286166.28
- `nn_brickwork_n8_d12_s2_medium` χ=64: ratio=3103438286166.28
- `nn_brickwork_n8_d12_s1_medium` χ=16: ratio=3081024990233.07
- `nn_brickwork_n8_d12_s1_medium` χ=32: ratio=3081024990233.07
- `nn_brickwork_n8_d12_s1_medium` χ=64: ratio=3081024990233.07

## Sweep diagnostic

- No sweep pairs (default uses ``tdvp_sweeps=1`` only).

---

Total raw runs: 324.
