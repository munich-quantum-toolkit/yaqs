# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Stage 2a (smoke/validation): dense-reference and ordering validation (spec 3.1).

Checks, fully independently of the benchmark plumbing:
  * dense two-site gate application agrees with an independently built
    full 2^L x 2^L unitary (kron construction, site 0 = LSB), including
    nonadjacent pairs away from the q0+q1=L-1 special case;
  * sequential gate application composes correctly;
  * RXX, RYY, RZZ are all covered; theta is used directly (no wrapping);
  * theta=0 is the identity; exact gates are unitary;
  * normalized fidelity is invariant under global phase;
  * comparator exact-limit thresholds at a nonbinding bond dimension.

Writes logs/validation_dense.json. Exits nonzero on any failure.

Usage:
    uv run python paper_benchmarks/scripts/validate_dense.py
"""

from __future__ import annotations

import numpy as np
from pb_common import (
    LOGS_DIR,
    SG_L,
    add_experiment_path,
    limit_blas_threads,
    save_json,
)

limit_blas_threads()
add_experiment_path("single_gate")

from gate_runtime import (  # ruff: ignore[module-import-not-at-top-of-file]
    apply_gate_to_dense_state,
    apply_method,
    make_dag_node,
    make_gate,
    normalized_state_fidelity,
    prepare_initial_state,
)
from variational import apply_variational_mpo_gate  # ruff: ignore[module-import-not-at-top-of-file]

# Nonbinding cap for the exact-limit check: exact result of one long-range
# two-qubit gate on a chi0=8 state needs chi<=16.
NONBINDING_CHI = 64
TEBD_TOL = 1e-7
MPO_TOL = 1e-12
LOCAL_DIM = 2
PAULI = {
    "x": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def independent_gate_matrix(gate_type: str, theta: float) -> np.ndarray:
    """exp(-i theta P (x) P / 2) built by scipy-free eigendecomposition."""
    p = PAULI[gate_type[1]]
    pp = np.kron(p, p)
    evals, evecs = np.linalg.eigh(pp)
    return (evecs * np.exp(-1j * theta / 2.0 * evals)) @ evecs.conj().T


def independent_full_unitary(gate4: np.ndarray, q0: int, q1: int, length: int) -> np.ndarray:
    """Embed a 4x4 gate into the full Hilbert space with site 0 = LSB.

    Built via kron of single-site embeddings of the operator-Schmidt factors
    of the 4x4 gate; completely independent of the benchmark reshape code.
    """
    # Operator-Schmidt decompose gate4 = sum_k A_k (x) B_k acting on (q_left, q_right)
    left, right = min(q0, q1), max(q0, q1)
    g = gate4.reshape(2, 2, 2, 2)  # (out_l, out_r, in_l, in_r)
    m = np.transpose(g, (0, 2, 1, 3)).reshape(4, 4)  # (out_l,in_l) x (out_r,in_r)
    u, s, vh = np.linalg.svd(m)
    total = np.zeros((2**length, 2**length), dtype=np.complex128)
    eye = np.eye(2, dtype=np.complex128)
    for k in range(len(s)):
        if s[k] < 1e-15:
            continue
        a_k = (u[:, k] * s[k]).reshape(2, 2)
        b_k = vh[k, :].reshape(2, 2)
        term = np.array([[1.0]], dtype=np.complex128)
        # site 0 is the LSB: kron runs from the highest site down to site 0
        for site in reversed(range(length)):
            if site == left:
                op = a_k
            elif site == right:
                op = b_k
            else:
                op = eye
            term = np.kron(term, op)
        total += term
    return total


def main() -> int:
    rng = np.random.default_rng(7)
    checks: list[dict] = []
    failed = False

    def record(name: str, value: float, tol: float, **extra) -> None:
        nonlocal failed
        ok = bool(value < tol)
        failed = failed or not ok
        checks.append({"check": name, "value": float(value), "tol": tol, "pass": ok, **extra})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {value:.3e} (tol {tol:.0e})")

    length = 8  # independent-unitary checks need dense 2^L x 2^L matrices
    vec = rng.standard_normal(2**length) + 1j * rng.standard_normal(2**length)
    vec /= np.linalg.norm(vec)

    # Angles used directly, including values far outside any principal interval.
    thetas = [0.3, 2.0, 7.5, -1.2]
    pairs = [(2, 3), (1, 5), (0, 6), (2, 7), (3, 4)]  # includes q0+q1 != L-1 cases

    for gate_type in ("rxx", "ryy", "rzz"):
        worst = 0.0
        for theta in thetas:
            g4 = independent_gate_matrix(gate_type, theta)
            # YAQS gate matrix must agree with the independent construction
            yaqs4 = np.asarray(make_gate(gate_type, theta, 0, 1).matrix, dtype=np.complex128)
            record(
                f"gate_matrix_{gate_type}_theta{theta}", float(np.max(np.abs(yaqs4 - g4))), 1e-12,
            )
            for q0, q1 in pairs:
                full = independent_full_unitary(g4, q0, q1, length)
                ref = full @ vec
                got = apply_gate_to_dense_state(vec, g4, q0, q1, length)
                worst = max(worst, float(np.max(np.abs(ref - got))))
        record(f"dense_apply_vs_independent_{gate_type}", worst, 1e-12)

    seq = [("rxx", 0.7, 1, 4), ("rzz", 1.9, 0, 6), ("ryy", 2.4, 2, 5)]
    composed = np.eye(2**length, dtype=np.complex128)
    stepwise = vec.copy()
    for gate_type, theta, q0, q1 in seq:
        g4 = independent_gate_matrix(gate_type, theta)
        composed = independent_full_unitary(g4, q0, q1, length) @ composed
        stepwise = apply_gate_to_dense_state(stepwise, g4, q0, q1, length)
    record("sequence_vs_composed_unitary", float(np.max(np.abs(composed @ vec - stepwise))), 1e-12)

    for gate_type in ("rxx", "ryy", "rzz"):
        g0 = np.asarray(make_gate(gate_type, 0.0, 0, 1).matrix, dtype=np.complex128)
        record(f"theta0_identity_{gate_type}", float(np.max(np.abs(g0 - np.eye(4)))), 1e-14)
        g = np.asarray(make_gate(gate_type, 2.7, 0, 1).matrix, dtype=np.complex128)
        record(f"unitarity_{gate_type}", float(np.max(np.abs(g.conj().T @ g - np.eye(4)))), 1e-12)
        # theta and theta + 4*pi give the same unitary; theta + 2*pi flips sign.
        gp = np.asarray(make_gate(gate_type, 2.7 + 4 * np.pi, 0, 1).matrix, dtype=np.complex128)
        record(f"no_wrap_4pi_period_{gate_type}", float(np.max(np.abs(gp - g))), 1e-9)
        gm = np.asarray(make_gate(gate_type, 2.7 + 2 * np.pi, 0, 1).matrix, dtype=np.complex128)
        record(f"no_wrap_2pi_antiperiod_{gate_type}", float(np.max(np.abs(gm + g))), 1e-9)

    a = rng.standard_normal(64) + 1j * rng.standard_normal(64)
    b = a * 1.7 + 0.05 * (rng.standard_normal(64) + 1j * rng.standard_normal(64))
    f0 = normalized_state_fidelity(a, b)["fidelity_normalized"]
    f1 = normalized_state_fidelity(a, b * np.exp(1.234j))["fidelity_normalized"]
    f2 = normalized_state_fidelity(a * np.exp(-2.1j), b)["fidelity_normalized"]
    record("global_phase_invariance", max(abs(f1 - f0), abs(f2 - f0)), 1e-14)

    initial = prepare_initial_state(11)
    theta = 2.0 * np.pi * 0.1
    for gate_type in ("rxx", "ryy", "rzz"):
        gate = make_gate(gate_type, theta, 2, 9)
        g4 = np.asarray(gate.matrix, dtype=np.complex128)
        exact = apply_gate_to_dense_state(initial["vec"], g4, 2, 9, SG_L)
        node = make_dag_node(gate_type, theta, 2, 9, SG_L)
        for method, tol in (("tebd_swap", TEBD_TOL), ("mpo_zipup", MPO_TOL)):
            state, _, _ = apply_method(
                initial["mps"], node, method=method, chi=NONBINDING_CHI, substeps=1,
            )
            inf = normalized_state_fidelity(exact, state.to_vec())["infidelity_normalized"]
            record(f"exact_limit_{method}_{gate_type}", inf, tol)
        vres = apply_variational_mpo_gate(initial["mps"], node, chi=NONBINDING_CHI)
        inf = normalized_state_fidelity(exact, vres.state.to_vec())["infidelity_normalized"]
        record(f"exact_limit_variational_mpo_{gate_type}", inf, MPO_TOL)

    payload = {
        "description": "dense-reference and ordering validation (spec 3.1)",
        "length_independent_checks": length,
        "nonbinding_chi": NONBINDING_CHI,
        "tebd_tol": TEBD_TOL,
        "mpo_tol": MPO_TOL,
        "checks": checks,
        "all_pass": not failed,
    }
    save_json(LOGS_DIR / "validation_dense.json", payload)
    print(f"\n{'ALL PASS' if not failed else 'FAILURES PRESENT'} -> logs/validation_dense.json")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
