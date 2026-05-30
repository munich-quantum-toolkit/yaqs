# Adaptive Generator-Enriched TDVP Benchmark

## Method summary

- single-qubit gates: direct tensor contraction
- nearest-neighbor two-qubit gates: TEBD/SVD
- long-range non-Pauli gates: local TDVP
- long-range Pauli rotations (rxx/ryy/rzz): adaptive routing
  - TDVP if projected-generator ratio >= tangent_blindness_tol
  - otherwise exact generator enrichment

## Correctness diagnostics

- **tangent_blind_ryy_8q** `adaptive_hybrid(max_bond=None,svd=1e-14)`: fidelity=4.441e-16, pauli_max=2.220e-16, max_bond=2, tdvp_lr_pauli=0, enriched_lr_pauli=1
- **tangent_blind_ryy_8q** `tebd_swaps(max_bond=None,svd=1e-14)`: fidelity=0.000e+00, pauli_max=3.331e-16, max_bond=2, tdvp_lr_pauli=—, enriched_lr_pauli=—
- **tangent_blind_ryy_8q** `tebd_swaps(max_bond=64,svd=1e-09)`: fidelity=0.000e+00, pauli_max=3.331e-16, max_bond=2, tdvp_lr_pauli=—, enriched_lr_pauli=—
- **tangent_blind_ryy_8q** `tebd_swaps(max_bond=128,svd=1e-09)`: fidelity=0.000e+00, pauli_max=3.331e-16, max_bond=2, tdvp_lr_pauli=—, enriched_lr_pauli=—
- **endpoint_prepared_ryy_8q** `adaptive_hybrid(max_bond=None,svd=1e-14)`: fidelity=6.661e-16, pauli_max=5.551e-16, max_bond=2, tdvp_lr_pauli=1, enriched_lr_pauli=0
- **endpoint_prepared_ryy_8q** `tebd_swaps(max_bond=None,svd=1e-14)`: fidelity=4.441e-16, pauli_max=8.327e-16, max_bond=2, tdvp_lr_pauli=—, enriched_lr_pauli=—
- **endpoint_prepared_ryy_8q** `tebd_swaps(max_bond=64,svd=1e-09)`: fidelity=4.441e-16, pauli_max=8.327e-16, max_bond=2, tdvp_lr_pauli=—, enriched_lr_pauli=—
- **endpoint_prepared_ryy_8q** `tebd_swaps(max_bond=128,svd=1e-09)`: fidelity=4.441e-16, pauli_max=8.327e-16, max_bond=2, tdvp_lr_pauli=—, enriched_lr_pauli=—
- **rzz_8q** `adaptive_hybrid(max_bond=None,svd=1e-14)`: fidelity=2.220e-16, pauli_max=2.220e-16, max_bond=2, tdvp_lr_pauli=1, enriched_lr_pauli=0
- **rzz_8q** `tebd_swaps(max_bond=None,svd=1e-14)`: fidelity=4.441e-16, pauli_max=2.220e-16, max_bond=2, tdvp_lr_pauli=—, enriched_lr_pauli=—
- **rzz_8q** `tebd_swaps(max_bond=64,svd=1e-09)`: fidelity=4.441e-16, pauli_max=2.220e-16, max_bond=2, tdvp_lr_pauli=—, enriched_lr_pauli=—
- **rzz_8q** `tebd_swaps(max_bond=128,svd=1e-09)`: fidelity=4.441e-16, pauli_max=2.220e-16, max_bond=2, tdvp_lr_pauli=—, enriched_lr_pauli=—
- **mixed_stack_10q** `adaptive_hybrid(max_bond=None,svd=1e-14)`: fidelity=2.220e-16, pauli_max=1.110e-16, max_bond=4, tdvp_lr_pauli=0, enriched_lr_pauli=2
- **mixed_stack_10q** `tebd_swaps(max_bond=None,svd=1e-14)`: fidelity=0.000e+00, pauli_max=3.331e-16, max_bond=4, tdvp_lr_pauli=—, enriched_lr_pauli=—
- **mixed_stack_10q** `tebd_swaps(max_bond=64,svd=1e-09)`: fidelity=0.000e+00, pauli_max=3.331e-16, max_bond=4, tdvp_lr_pauli=—, enriched_lr_pauli=—
- **mixed_stack_10q** `tebd_swaps(max_bond=128,svd=1e-09)`: fidelity=0.000e+00, pauli_max=3.331e-16, max_bond=4, tdvp_lr_pauli=—, enriched_lr_pauli=—
- **hard_mixed_stack_12q** `adaptive_hybrid(max_bond=None,svd=1e-14)`: fidelity=4.441e-16, pauli_max=9.992e-16, max_bond=4, tdvp_lr_pauli=2, enriched_lr_pauli=3
- **hard_mixed_stack_12q** `tebd_swaps(max_bond=None,svd=1e-14)`: fidelity=0.000e+00, pauli_max=2.878e-15, max_bond=4, tdvp_lr_pauli=—, enriched_lr_pauli=—
- **hard_mixed_stack_12q** `tebd_swaps(max_bond=64,svd=1e-09)`: fidelity=0.000e+00, pauli_max=2.878e-15, max_bond=4, tdvp_lr_pauli=—, enriched_lr_pauli=—
- **hard_mixed_stack_12q** `tebd_swaps(max_bond=128,svd=1e-09)`: fidelity=0.000e+00, pauli_max=2.878e-15, max_bond=4, tdvp_lr_pauli=—, enriched_lr_pauli=—

## Route statistics

- Total LR Pauli routed to TDVP: **676**
- Total LR Pauli routed to enrichment: **327**

## Position/distance sweep

See CSV for full per-pair detail (includes per-gate route JSON).

## Mixed-stack stability

The previously failing mixed stacks are included in correctness diagnostics and repeated-layer families.

## TEBD+SWAP comparison

Rows include both `adaptive_hybrid(...)` and `tebd_swaps(...)` for selected circuits.

## Paper-scale circuits

This script includes repeated-layer stress families up to n=20. Add manuscript circuits here as needed.

## Summary

This benchmark reports fidelity (when Qiskit statevector is feasible), Pauli observable errors, bond-dimension cost proxies, wall time, and per-long-range-Pauli route decisions.
