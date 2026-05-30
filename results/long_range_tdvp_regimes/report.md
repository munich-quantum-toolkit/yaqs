# Long-range TDVP regime map

Stage **1**. Primary metric: observable error vs reference.

## 1. Overall conclusion by family

| family | comparisons | TDVP win % | TEBD win % | mean TEBD/TDVP err | recommendation |
|---|---:|---:|---:|---:|---|
| dense_long_range | 108 | 1% | 89% | 9444481855329.91 | use TEBD+SWAP |
| flattened_2d_grid | 108 | 0% | 100% | 12807604139528.29 | use TEBD+SWAP |
| nn_brickwork | 108 | 0% | 0% | 0.98 | needs larger chi / reference unclear |
| periodic_1d | 108 | 0% | 100% | 7883738629742.69 | use TEBD+SWAP |
| random_all_to_all | 108 | 3% | 81% | 10560617393323.66 | use TEBD+SWAP |
| sparse_long_range | 108 | 33% | 55% | 2660389422249.09 | use TEBD+SWAP |

## 2. Best TDVP regimes

- `random_all_to_all_n8_d8_s0_medium` χ=16: TEBD/TDVP err ratio=79130981718975.6, outcome=tebd_wins
- `random_all_to_all_n8_d8_s0_medium` χ=32: TEBD/TDVP err ratio=79130981718975.6, outcome=tebd_wins
- `random_all_to_all_n8_d8_s0_medium` χ=64: TEBD/TDVP err ratio=79130981718975.6, outcome=tebd_wins
- `flattened_2d_grid_n8_d4_s0_medium` χ=16: TEBD/TDVP err ratio=75137626040403.0, outcome=tebd_wins
- `flattened_2d_grid_n8_d4_s0_medium` χ=32: TEBD/TDVP err ratio=75137626040403.0, outcome=tebd_wins
- `flattened_2d_grid_n8_d4_s0_medium` χ=64: TEBD/TDVP err ratio=75137626040403.0, outcome=tebd_wins
- `flattened_2d_grid_n16_d4_s1_medium` χ=16: TEBD/TDVP err ratio=68574140174185.2, outcome=tebd_wins
- `flattened_2d_grid_n16_d4_s1_medium` χ=32: TEBD/TDVP err ratio=68569393909556.7, outcome=tebd_wins

## 3. Worst TDVP regimes

- `sparse_long_range_n12_d4_s0_small` χ=32: TDVP/TEBD err ratio=263.9
- `sparse_long_range_n12_d4_s2_small` χ=16: TDVP/TEBD err ratio=237.0
- `sparse_long_range_n12_d4_s0_small` χ=16: TDVP/TEBD err ratio=143.3
- `sparse_long_range_n16_d4_s2_small` χ=16: TDVP/TEBD err ratio=98.8
- `sparse_long_range_n16_d4_s0_small` χ=16: TDVP/TEBD err ratio=55.0
- `sparse_long_range_n16_d8_s0_small` χ=32: TDVP/TEBD err ratio=40.1
- `sparse_long_range_n16_d8_s0_small` χ=16: TDVP/TEBD err ratio=38.2
- `sparse_long_range_n12_d4_s2_small` χ=32: TDVP/TEBD err ratio=30.4

## 4. Sweep diagnostic

- Default benchmark uses `tdvp_sweeps=1` only. Set `YAQS_LR_REGIME_INCLUDE_TDVP_SWEEP_SCAN=1` for sweep-grid diagnostics.

- No sweep pairs in this run.

## Standard TDVP padding diagnostic

**Stage 0 conclusions (reference):** unpadded standard TDVP underuses χ on periodic long-range gates; padding to bond dimension 4 fixes small periodic cases; padding beyond 4 does not help those cases; extra TDVP sweeps are not reliable refinement.

### Does padding fix periodic long-range gates?

- `periodic_1d_n12_d4_s0_small` χ=16: unpadded err=0.002687567981701709 fid=0.962819965149348; padded4 err=3.400e-12 fid=1.0000; TEBD err=1.541e-13
- `periodic_1d_n12_d4_s0_medium` χ=16: unpadded err=0.03358331970206226 fid=0.37333839481111475; padded4 err=9.085e-16 fid=1.0000; TEBD err=6.982e-16
- `periodic_1d_n12_d4_s1_small` χ=16: unpadded err=0.005544450430953804 fid=0.9217248969488389; padded4 err=1.283e-12 fid=1.0000; TEBD err=4.909e-14
- `periodic_1d_n12_d4_s1_medium` χ=16: unpadded err=0.05300759161668428 fid=0.12338850136563606; padded4 err=1.078e-14 fid=1.0000; TEBD err=1.366e-15
- `periodic_1d_n12_d4_s2_small` χ=16: unpadded err=0.0034282100300667647 fid=0.9555167525708822; padded4 err=7.645e-13 fid=1.0000; TEBD err=1.800e-13
- `periodic_1d_n12_d4_s2_medium` χ=16: unpadded err=0.01486551086915229 fid=0.31488995430852995; padded4 err=1.164e-14 fid=1.0000; TEBD err=7.123e-16
- `periodic_1d_n12_d4_s0_small` χ=32: unpadded err=0.002687567981701709 fid=0.962819965149348; padded4 err=3.400e-12 fid=1.0000; TEBD err=1.541e-13
- `periodic_1d_n12_d4_s0_medium` χ=32: unpadded err=0.03358331970206226 fid=0.37333839481111475; padded4 err=9.085e-16 fid=1.0000; TEBD err=6.982e-16

**periodic_1d:** unpadded TDVP often underuses χ and loses X-coherence; padded4 TDVP fixes the small periodic cases.

### Does padding preserve sparse-long-range advantages?

- 38/108 sparse cases favor padded4 or require padding; TEBD hits χ_max in 73/108 fixed-χ rows.
**sparse_long_range:** TDVP can be competitive when TEBD+SWAP hits χ_max; at larger χ, TEBD may become essentially exact.

### Is padding enough, or is TEBD still better?

- padded4_tdvp_wins: 89/648; tebd_wins: 397/648.

### Does padding merely match TEBD's bond dimension?

- padded4 peak χ < 85% of TEBD peak χ in 167/648 rows (TDVP can stay lower-χ).

### Should multi-sweep TDVP be used?

- Default benchmark uses one sweep only (`tdvp_sweeps=1`).
- **sweeps:** `tdvp_sweeps > 1` should remain diagnostic only.

Padding scan transitions: helpful=149/648 (see `tdvp_padding_scaling.csv`).

## 5. Bond-dimension diagnostic

- TDVP lower peak χ than TEBD in 274/648 matched pairs.

## 6. Accuracy vs runtime

- TDVP faster: 109/648; TDVP more accurate (err ratio>1.5): 467/648

## 7. Practical recommendation

- **dense_long_range** (medium, n=12, d=4, χ=16): TEBD preferable: TEBD is faster and equally/more accurate
- **dense_long_range** (medium, n=12, d=4, χ=32): TEBD preferable: TEBD is faster and equally/more accurate
- **dense_long_range** (medium, n=12, d=4, χ=64): TEBD preferable: TEBD is faster and equally/more accurate
- **dense_long_range** (medium, n=12, d=8, χ=16): TEBD preferable: TEBD is faster and equally/more accurate
- **dense_long_range** (medium, n=12, d=8, χ=32): TEBD preferable: TEBD is faster and equally/more accurate
- **dense_long_range** (medium, n=12, d=8, χ=64): TEBD preferable: TEBD is faster and equally/more accurate
- **dense_long_range** (medium, n=16, d=4, χ=16): TEBD preferable: TEBD is faster and equally/more accurate
- **dense_long_range** (medium, n=16, d=4, χ=32): TEBD preferable: TEBD is faster and equally/more accurate
- **dense_long_range** (medium, n=16, d=4, χ=64): TEBD preferable: TEBD is faster and equally/more accurate
- **dense_long_range** (medium, n=16, d=8, χ=16): Mixed / inconclusive: compare sweep_scaling and fixed_chi tables
- **dense_long_range** (medium, n=16, d=8, χ=32): TEBD preferable: TEBD is faster and equally/more accurate
- **dense_long_range** (medium, n=16, d=8, χ=64): TEBD preferable: TEBD is faster and equally/more accurate

---

Total raw runs: 1944; successful comparisons: 648.
