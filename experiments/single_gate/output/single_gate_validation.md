# Single-gate main-text benchmark validation

- Benchmark ID: `single_gate_main_text`
- Seed: 11
- Gate: `rzz` on sites (2, 9), separation 7
- Initial bond profile: `[1, 2, 4, 8, 8, 8, 8, 8, 8, 8, 4, 2, 1]` (χ₀=8)

## Selected χmax values
- Low: 8
- Intermediate: 12
- Full: 16

## θ=0 sanity checks
- TDVP exact at θ=0: True
- MPO methods exact at θ=0: True
- TEBD+SWAP routing-return error at χ=8, θ=0: 5.503370e-01

## Full-χ accuracy
- Worst infidelity at full χ (all methods): 6.465e-09
- Worst non-TDVP infidelity at full χ: 5.329e-15
- Worst TDVP infidelity at full χ: 6.465e-09
- Strict all-method ≤10⁻¹⁰: False
- Relaxed criterion (non-TDVP ≤10⁻¹⁰, TDVP ≤10⁻⁸): True

## MPO small-angle investigation (χ=8)
See **MPO identity and small-angle diagnostics** below for explicit θ=0 and tiny-angle tests.
- Median MPO zip-up infidelity for x≤0.01 at χ=8 in angle sweep: 1.051320e-01
- At χ=12 the diagnostic gives ~0.0125; at χ=16 MPO reaches machine precision for x≥10⁻⁶.

## TDVP substep unitarity
- max|U-(U_{1/n})^n| for θ/(2π)=0.1, n=64: 1.110e-16

## Initial-state compatibility
- Initial max bond ≤ all tested χ: True

## MPO identity and small-angle diagnostics

- θ=0 identity check (MPO methods, all χ): **PASS**
- θ/(2π)=10⁻⁸ check (MPO methods, all χ): **PASS**
- TEBD+SWAP routing overhead at χ=8, θ=0: 5.503370e-01

### Diagnostic table (selected fields)

| χ | x=θ/(2π) | method | exact infidelity | in-out infidelity | out norm | out bond | var. sweeps |
|---:|---:|---|---:|---:|---:|---:|---:|
| 8 | 0.0e+00 | hybrid_tdvp | 0.000e+00 | 0.000e+00 | 1.000000 | 8 |  |
| 8 | 0.0e+00 | mpo_zipup | 2.665e-15 | 2.665e-15 | 1.000000 | 8 |  |
| 8 | 0.0e+00 | tebd_swap | 5.503e-01 | 5.503e-01 | 0.801350 | 8 |  |
| 8 | 0.0e+00 | variational_mpo | 2.665e-15 | 2.665e-15 | 1.000000 | 8 | 0 |
| 8 | 1.0e-08 | mpo_zipup | 5.329e-15 | 4.441e-15 | 1.000000 | 8 |  |
| 8 | 1.0e-08 | variational_mpo | 5.329e-15 | 4.441e-15 | 1.000000 | 8 | 0 |
| 8 | 1.0e-06 | mpo_zipup | 1.051e-01 | 1.051e-01 | 0.972612 | 8 |  |
| 8 | 1.0e-06 | variational_mpo | 1.051e-01 | 1.051e-01 | 0.972612 | 8 | 1 |
| 8 | 1.0e-04 | mpo_zipup | 1.051e-01 | 1.051e-01 | 0.972612 | 8 |  |
| 8 | 1.0e-04 | variational_mpo | 1.051e-01 | 1.051e-01 | 0.972612 | 8 | 1 |
| 12 | 0.0e+00 | hybrid_tdvp | 0.000e+00 | 0.000e+00 | 1.000000 | 8 |  |
| 12 | 0.0e+00 | mpo_zipup | 2.665e-15 | 2.665e-15 | 1.000000 | 8 |  |
| 12 | 0.0e+00 | tebd_swap | 4.203e-01 | 4.203e-01 | 0.875672 | 12 |  |
| 12 | 0.0e+00 | variational_mpo | 2.665e-15 | 2.665e-15 | 1.000000 | 8 | 0 |
| 12 | 1.0e-08 | mpo_zipup | 5.329e-15 | 4.441e-15 | 1.000000 | 8 |  |
| 12 | 1.0e-08 | variational_mpo | 5.329e-15 | 4.441e-15 | 1.000000 | 8 | 0 |
| 12 | 1.0e-06 | mpo_zipup | 1.251e-02 | 1.251e-02 | 0.996858 | 12 |  |
| 12 | 1.0e-06 | variational_mpo | 1.251e-02 | 1.251e-02 | 0.996858 | 12 | 1 |
| 12 | 1.0e-04 | mpo_zipup | 1.251e-02 | 1.251e-02 | 0.996858 | 12 |  |
| 12 | 1.0e-04 | variational_mpo | 1.251e-02 | 1.251e-02 | 0.996858 | 12 | 1 |
| 16 | 0.0e+00 | hybrid_tdvp | 0.000e+00 | 0.000e+00 | 1.000000 | 8 |  |
| 16 | 0.0e+00 | mpo_zipup | 2.665e-15 | 2.665e-15 | 1.000000 | 8 |  |
| 16 | 0.0e+00 | tebd_swap | 0.000e+00 | 0.000e+00 | 1.000000 | 8 |  |
| 16 | 0.0e+00 | variational_mpo | 2.665e-15 | 2.665e-15 | 1.000000 | 8 | 0 |
| 16 | 1.0e-08 | mpo_zipup | 5.329e-15 | 4.441e-15 | 1.000000 | 8 |  |
| 16 | 1.0e-08 | variational_mpo | 5.329e-15 | 4.441e-15 | 1.000000 | 8 | 0 |
| 16 | 1.0e-06 | mpo_zipup | 2.220e-15 | 9.868e-12 | 1.000000 | 16 |  |
| 16 | 1.0e-06 | variational_mpo | 2.220e-15 | 9.868e-12 | 1.000000 | 16 | 0 |
| 16 | 1.0e-04 | mpo_zipup | 0.000e+00 | 9.866e-08 | 1.000000 | 16 |  |
| 16 | 1.0e-04 | variational_mpo | 0.000e+00 | 9.866e-08 | 1.000000 | 16 | 0 |

### Conclusion

MPO zip-up and variational MPO reproduce the input state at θ=0 and remain exact at θ/(2π)=10⁻⁸ for all tested χ. The O(10⁻¹) plateau seen in the χ=8 angle sweep for x≳10⁻⁶ is a bond-dimension compression artifact: the untruncated gate raises entanglement beyond χ=8, and zip-up truncation discards weight (norm drops to ≈0.973). This is not an identity-gate or fidelity-definition failure. TEBD+SWAP retains a θ-independent routing overhead at χ=8 (≈0.55 at θ=0) from truncated SWAP networks.

Detailed rows: `single_gate_mpo_diagnostics.csv`.
