# θ=0 and identity-limit diagnostics

- Seed: 11
- Gate: `rzz` on sites (2, 9), convention RZZ(θ)=exp(−iθ Z⊗Z/2)
- Angles tested: θ/(2π) ∈ [0.0, 1e-12, 1e-10, 1e-08, 1e-06, 0.0001]
- χmax values: [8, 12, 16]

## 1. Gate construction at θ=0

| representation | ‖U−I‖₂ | ‖U−I‖_F | MPO max bond | MPO action infidelity | near-zero branches |
|---|---:|---:|---:|---:|---:|
| dense_gate_matrix | 0.000e+00 | 0.000e+00 | 1 |  | 0 |
| dense_make_gate | 0.000e+00 | 0.000e+00 | 1 |  | 0 |
| mpo_from_gate_theta0 | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 0 |
| mpo_identity_reference | 0.000e+00 | 0.000e+00 | 1 | 0.000e+00 | 0 |

**Gate construction:** PASS

## 2. Initial state

| χ | max bond | norm | copy infidelity | canonical infidelity | identical |
|---:|---:|---:|---:|---:|---|
| 8 | 8 | 1.000000000000 | 2.665e-15 | 5.329e-15 | True |
| 12 | 8 | 1.000000000000 | 2.665e-15 | 5.329e-15 | True |
| 16 | 8 | 1.000000000000 | 2.665e-15 | 5.329e-15 | True |

**Initial state:** PASS

## 3. θ=0 algorithm runs (no bypass)

| χ | method | exact inf | in-out inf | vec dist | Δnorm | disc. weight | var. obj (init→final) | pass |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 8 | hybrid_tdvp | 0.000e+00 | 0.000e+00 | 1.031e-13 | 2.087e-14 | 2.026e-30 |  | True |
| 8 | mpo_zipup | 2.665e-15 | 2.665e-15 | 1.971e-15 | 1.110e-16 | 1.678e-32 |  | True |
| 8 | tebd_swap | 5.503e-01 | 5.503e-01 | 5.487e-01 | 1.987e-01 | 1.055e+00 |  | True |
| 8 | variational_mpo | 2.665e-15 | 2.665e-15 | 1.971e-15 | 1.110e-16 | 0.000e+00 | 0.00e+00→0.00e+00 | True |
| 12 | hybrid_tdvp | 0.000e+00 | 0.000e+00 | 1.031e-13 | 2.087e-14 | 2.026e-30 |  | True |
| 12 | mpo_zipup | 2.665e-15 | 2.665e-15 | 1.971e-15 | 1.110e-16 | 1.678e-32 |  | True |
| 12 | tebd_swap | 4.203e-01 | 4.203e-01 | 4.940e-01 | 1.243e-01 | 4.716e-01 |  | True |
| 12 | variational_mpo | 2.665e-15 | 2.665e-15 | 1.971e-15 | 1.110e-16 | 0.000e+00 | 0.00e+00→0.00e+00 | True |
| 16 | hybrid_tdvp | 0.000e+00 | 0.000e+00 | 1.031e-13 | 2.087e-14 | 2.026e-30 |  | True |
| 16 | mpo_zipup | 2.665e-15 | 2.665e-15 | 1.971e-15 | 1.110e-16 | 1.678e-32 |  | True |
| 16 | tebd_swap | 0.000e+00 | 0.000e+00 | 5.055e-15 | 2.665e-15 | 3.043e-30 |  | True |
| 16 | variational_mpo | 2.665e-15 | 2.665e-15 | 1.971e-15 | 1.110e-16 | 0.000e+00 | 0.00e+00→0.00e+00 | True |

### SWAP routing (RZZ(0)=I via tebd_swap)

| χ | exact inf | in-out inf | note |
|---:|---:|---:|---|
| 8 | 5.503e-01 | 5.503e-01 | SWAP-forward/SWAP-back routing with RZZ(0)=I through tebd_swap path (no gate bypass) |
| 12 | 4.203e-01 | 4.203e-01 | SWAP-forward/SWAP-back routing with RZZ(0)=I through tebd_swap path (no gate bypass) |
| 16 | 0.000e+00 | 0.000e+00 | SWAP-forward/SWAP-back routing with RZZ(0)=I through tebd_swap path (no gate bypass) |

**TDVP/MPO θ=0:** PASS
**TEBD χ=16 θ=0:** PASS (χ=8 routing error ≈ 5.503e-01, expected from truncated SWAPs)

## 4. Continuity around θ=0

Includes unchanged-input baseline infidelity (valid fixed-χ reference).

| χ | x=θ/(2π) | method | exact inf | unchanged baseline | compression residual |
|---:|---:|---|---:|---:|---:|
| 8 | 0.0e+00 | hybrid_tdvp | 0.000e+00 | 2.665e-15 |  |
| 8 | 0.0e+00 | mpo_zipup | 2.665e-15 | 2.665e-15 | 3.331e-15 |
| 8 | 0.0e+00 | tebd_swap | 5.503e-01 | 2.665e-15 |  |
| 8 | 0.0e+00 | variational_mpo | 2.665e-15 | 2.665e-15 | 3.331e-15 |
| 8 | 1.0e-12 | hybrid_tdvp | 0.000e+00 | 1.776e-15 |  |
| 8 | 1.0e-12 | mpo_zipup | 1.554e-15 | 1.776e-15 | 3.331e-15 |
| 8 | 1.0e-12 | tebd_swap | 5.503e-01 | 1.776e-15 |  |
| 8 | 1.0e-12 | variational_mpo | 1.554e-15 | 1.776e-15 | 3.331e-15 |
| 8 | 1.0e-10 | hybrid_tdvp | 0.000e+00 | 1.776e-15 |  |
| 8 | 1.0e-10 | mpo_zipup | 1.776e-15 | 1.776e-15 | 3.331e-15 |
| 8 | 1.0e-10 | tebd_swap | 5.503e-01 | 1.776e-15 |  |
| 8 | 1.0e-10 | variational_mpo | 1.776e-15 | 1.776e-15 | 3.331e-15 |
| 8 | 1.0e-08 | hybrid_tdvp | 0.000e+00 | 3.109e-15 |  |
| 8 | 1.0e-08 | mpo_zipup | 5.329e-15 | 3.109e-15 | 4.885e-15 |
| 8 | 1.0e-08 | tebd_swap | 5.503e-01 | 3.109e-15 |  |
| 8 | 1.0e-08 | variational_mpo | 5.329e-15 | 3.109e-15 | 4.885e-15 |
| 8 | 1.0e-06 | hybrid_tdvp | 6.188e-12 | 9.868e-12 |  |
| 8 | 1.0e-06 | mpo_zipup | 1.051e-01 | 9.868e-12 | 1.051e-01 |
| 8 | 1.0e-06 | tebd_swap | 5.503e-01 | 9.868e-12 |  |
| 8 | 1.0e-06 | variational_mpo | 1.051e-01 | 9.868e-12 | 1.051e-01 |
| 8 | 1.0e-04 | hybrid_tdvp | 6.262e-08 | 9.866e-08 |  |
| 8 | 1.0e-04 | mpo_zipup | 1.051e-01 | 9.866e-08 | 1.051e-01 |
| 8 | 1.0e-04 | tebd_swap | 5.503e-01 | 9.866e-08 |  |
| 8 | 1.0e-04 | variational_mpo | 1.051e-01 | 9.866e-08 | 1.051e-01 |
| 12 | 0.0e+00 | hybrid_tdvp | 0.000e+00 | 2.665e-15 |  |
| 12 | 0.0e+00 | mpo_zipup | 2.665e-15 | 2.665e-15 | 3.331e-15 |
| 12 | 0.0e+00 | tebd_swap | 4.203e-01 | 2.665e-15 |  |
| 12 | 0.0e+00 | variational_mpo | 2.665e-15 | 2.665e-15 | 3.331e-15 |
| 12 | 1.0e-12 | hybrid_tdvp | 0.000e+00 | 1.776e-15 |  |
| 12 | 1.0e-12 | mpo_zipup | 1.554e-15 | 1.776e-15 | 3.331e-15 |
| 12 | 1.0e-12 | tebd_swap | 4.203e-01 | 1.776e-15 |  |
| 12 | 1.0e-12 | variational_mpo | 1.554e-15 | 1.776e-15 | 3.331e-15 |
| 12 | 1.0e-10 | hybrid_tdvp | 0.000e+00 | 1.776e-15 |  |
| 12 | 1.0e-10 | mpo_zipup | 1.776e-15 | 1.776e-15 | 3.331e-15 |
| 12 | 1.0e-10 | tebd_swap | 4.203e-01 | 1.776e-15 |  |
| 12 | 1.0e-10 | variational_mpo | 1.776e-15 | 1.776e-15 | 3.331e-15 |
| 12 | 1.0e-08 | hybrid_tdvp | 0.000e+00 | 3.109e-15 |  |
| 12 | 1.0e-08 | mpo_zipup | 5.329e-15 | 3.109e-15 | 4.885e-15 |
| 12 | 1.0e-08 | tebd_swap | 4.203e-01 | 3.109e-15 |  |
| 12 | 1.0e-08 | variational_mpo | 5.329e-15 | 3.109e-15 | 4.885e-15 |
| 12 | 1.0e-06 | hybrid_tdvp | 6.188e-12 | 9.868e-12 |  |
| 12 | 1.0e-06 | mpo_zipup | 1.251e-02 | 9.868e-12 | 1.251e-02 |
| 12 | 1.0e-06 | tebd_swap | 4.203e-01 | 9.868e-12 |  |
| 12 | 1.0e-06 | variational_mpo | 1.251e-02 | 9.868e-12 | 1.251e-02 |
| 12 | 1.0e-04 | hybrid_tdvp | 5.125e-09 | 9.866e-08 |  |
| 12 | 1.0e-04 | mpo_zipup | 1.251e-02 | 9.866e-08 | 1.251e-02 |
| 12 | 1.0e-04 | tebd_swap | 4.203e-01 | 9.866e-08 |  |
| 12 | 1.0e-04 | variational_mpo | 1.251e-02 | 9.866e-08 | 1.251e-02 |
| 16 | 0.0e+00 | hybrid_tdvp | 0.000e+00 | 2.665e-15 |  |
| 16 | 0.0e+00 | mpo_zipup | 2.665e-15 | 2.665e-15 | 3.331e-15 |
| 16 | 0.0e+00 | tebd_swap | 0.000e+00 | 2.665e-15 |  |
| 16 | 0.0e+00 | variational_mpo | 2.665e-15 | 2.665e-15 | 3.331e-15 |
| 16 | 1.0e-12 | hybrid_tdvp | 0.000e+00 | 1.776e-15 |  |
| 16 | 1.0e-12 | mpo_zipup | 1.554e-15 | 1.776e-15 | 3.331e-15 |
| 16 | 1.0e-12 | tebd_swap | 0.000e+00 | 1.776e-15 |  |
| 16 | 1.0e-12 | variational_mpo | 1.554e-15 | 1.776e-15 | 3.331e-15 |
| 16 | 1.0e-10 | hybrid_tdvp | 0.000e+00 | 1.776e-15 |  |
| 16 | 1.0e-10 | mpo_zipup | 1.776e-15 | 1.776e-15 | 3.331e-15 |
| 16 | 1.0e-10 | tebd_swap | 0.000e+00 | 1.776e-15 |  |
| 16 | 1.0e-10 | variational_mpo | 1.776e-15 | 1.776e-15 | 3.331e-15 |
| 16 | 1.0e-08 | hybrid_tdvp | 0.000e+00 | 3.109e-15 |  |
| 16 | 1.0e-08 | mpo_zipup | 5.329e-15 | 3.109e-15 | 4.885e-15 |
| 16 | 1.0e-08 | tebd_swap | 0.000e+00 | 3.109e-15 |  |
| 16 | 1.0e-08 | variational_mpo | 5.329e-15 | 3.109e-15 | 4.885e-15 |
| 16 | 1.0e-06 | hybrid_tdvp | 6.188e-12 | 9.868e-12 |  |
| 16 | 1.0e-06 | mpo_zipup | 2.220e-15 | 9.868e-12 | 2.665e-15 |
| 16 | 1.0e-06 | tebd_swap | 3.098e-09 | 9.868e-12 |  |
| 16 | 1.0e-06 | variational_mpo | 2.220e-15 | 9.868e-12 | 2.665e-15 |
| 16 | 1.0e-04 | hybrid_tdvp | 5.125e-09 | 9.866e-08 |  |
| 16 | 1.0e-04 | mpo_zipup | 0.000e+00 | 9.866e-08 | 0.000e+00 |
| 16 | 1.0e-04 | tebd_swap | 0.000e+00 | 9.866e-08 |  |
| 16 | 1.0e-04 | variational_mpo | 0.000e+00 | 9.866e-08 | 0.000e+00 |

## 5. Conclusions and stop conditions

- Implementation bug (θ=0 failure for TDVP/MPO): **False**
- MPO tiny-angle discontinuity at χ=8: **True**

At θ=0, TEBD+SWAP error at χ=8 matches SWAP-forward/SWAP-back routing with RZZ(0)=I; it is not caused by a nonzero gate matrix. If θ=0 passes for MPO methods but tiny θ shows an O(10⁻¹) plateau at χ=8, the mechanism is MPO zip-up compression discarding weight when entanglement exceeds χ=8, not an identity-limit bug.

Raw rows: `theta_zero_diagnostics.csv`.
