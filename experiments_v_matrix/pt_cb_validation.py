#!/usr/bin/env python3
"""Mandatory validation for causal-block operator entropy S_PT^cb.

Run: uv run --project /home/aaron/Github/yaqs python pt_cb_validation.py
"""

from __future__ import annotations

import math
import sys
from typing import cast

import numpy as np
from scipy.linalg import expm

from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer
from mqt.yaqs.characterization.memory.backends.tomography.process_tensors import (
    DenseProcessTensor,
    _unfuse_slot_index,
    causal_block_operator_entropy,
    causal_block_axis_indices,
    refold_unfused_to_upsilon,
    upsilon_to_unfused_operator,
)
from mqt.yaqs.characterization.memory.backends.tomography.process_tensors import convert_probe_callable
from mqt.yaqs.memory_characterizer import make_zero_psi

TOL = 1e-10


def _pt(length: int, k: int, j: float, *, g: float = 1.0) -> DenseProcessTensor:
    return cast(
        DenseProcessTensor,
        MemoryCharacterizer(parallel=False, show_progress=False).build_process_tensor(
            Hamiltonian.ising(length=length, J=j, g=g),
            AnalogSimParams(dt=0.1, max_bond_dim=64, order=1),
            timesteps=[0.1] * (k + 1),
            return_type="dense",
            method="exhaustive",
            compress_every=1,
        ),
    )


def _rank1_causal_upsilon(k: int, cut: int, *, seed: int = 0) -> np.ndarray:
    """Build an operator with operator-Schmidt rank 1 at causal cut ``cut``."""
    rng = np.random.default_rng(seed)
    op = upsilon_to_unfused_operator(np.eye(2 * 4**k, dtype=np.complex128), k)
    blocks = causal_block_axis_indices(k)
    left_axes = [i for b in blocks[:cut] for i in b]
    right_axes = [i for b in blocks[cut:] for i in b]
    perm = left_axes + right_axes
    inv = np.argsort(perm)
    t = np.ascontiguousarray(np.transpose(op, perm))
    dl = int(np.prod([t.shape[i] for i in range(len(left_axes))], dtype=np.int64))
    dr = int(np.prod([t.shape[i] for i in range(len(left_axes), len(left_axes) + len(right_axes))], dtype=np.int64))
    u = rng.standard_normal(dl) + 1j * rng.standard_normal(dl)
    v = rng.standard_normal(dr) + 1j * rng.standard_normal(dr)
    t2 = np.zeros_like(t)
    t2.reshape(dl, dr)[:, :] = np.outer(u, v.conj())
    op2 = np.transpose(t2, inv)
    return refold_unfused_to_upsilon(op2, k)


def _apply_local_unitary_block(
    op: np.ndarray, ket_axis: int, bra_axis: int, u: np.ndarray
) -> np.ndarray:
    """Apply the same local unitary on paired ket/bra axes."""
    out = np.tensordot(u, op, axes=([1], [ket_axis]))
    out = np.moveaxis(out, 0, ket_axis)
    out = np.tensordot(u.conj(), out, axes=([1], [bra_axis]))
    return np.moveaxis(out, 0, bra_axis)


def _causal_break_error(pt: DenseProcessTensor, k: int, cut: int, *, trials: int = 30) -> float:
    rng = np.random.default_rng(42)
    z = np.array([1.0, 0.0], dtype=np.complex128)
    x = np.array([0.0, 1.0], dtype=np.complex128)
    states = [z, x, (z + x) / np.sqrt(2)]
    idm = convert_probe_callable({"type": "unitary", "U": np.eye(2, dtype=np.complex128)})

    def mp(m, p):
        return convert_probe_callable((m, p))

    def rnd():
        return rng.choice([idm, convert_probe_callable((rng.choice(states), rng.choice(states)))])

    past_len, future_len = cut - 1, k - cut
    mx = 0.0
    for _ in range(trials):
        pa = [rnd() for _ in range(past_len)]
        pap = [rnd() for _ in range(past_len)]
        step = mp(rng.choice(states), rng.choice(states))
        fut = [rnd() for _ in range(future_len)]
        ra = pt.predict([*pa, step, *fut])
        rap = pt.predict([*pap, step, *fut])
        ta, tap = np.trace(ra), np.trace(rap)
        if abs(ta) > 1e-15 and abs(tap) > 1e-15:
            mx = max(mx, float(np.linalg.norm(ra / ta - rap / tap)))
    return mx


def main() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        status = "PASS" if cond else "FAIL"
        print(f"{status}: {name}" + (f" — {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    # Test A
    pt = _pt(1, 2, 0.0, g=0.0)
    for c in (1, 2):
        r = causal_block_operator_entropy(pt.to_matrix(), 2, c)
        check(f"A identity k=2 cut={c}", r["schmidt_rank"] == 1 and r["entropy"] < TOL, f"S={r['entropy']:.3e}")

    pt = _pt(1, 3, 0.0, g=0.0)
    for c in (1, 2, 3):
        r = causal_block_operator_entropy(pt.to_matrix(), 3, c)
        check(f"A identity k=3 cut={c}", r["schmidt_rank"] == 1 and r["entropy"] < TOL, f"S={r['entropy']:.3e}")

    # Test B: unitary channels (g=1, J=0 still product unitaries on system)
    pt = _pt(1, 3, 0.0, g=1.0)
    for c in (1, 2, 3):
        r = causal_block_operator_entropy(pt.to_matrix(), 3, c)
        check(f"B unitary k=3 cut={c}", r["schmidt_rank"] == 1 and r["entropy"] < TOL, f"S={r['entropy']:.3e}")

    # Test C: rank-1 at every causal cut (Markov-like factorization) and noisy Markov PT
    for k in (2, 3):
        for c in range(1, k + 1):
            ups = _rank1_causal_upsilon(k, c, seed=10 + k + c)
            r = causal_block_operator_entropy(ups, k, c)
            check(f"C rank-1 causal k={k} cut={c}", r["schmidt_rank"] == 1 and r["entropy"] < TOL)

    for k, g in ((2, 0.0), (3, 1.0)):
        pt = _pt(1, k, 0.0, g=g)
        for c in range(1, k + 1):
            r = causal_block_operator_entropy(pt.to_matrix(), k, c)
            check(
                f"C markov L=1 k={k} g={g:g} cut={c}",
                r["schmidt_rank"] == 1 and r["entropy"] < TOL,
                f"S={r['entropy']:.3e}",
            )

    # Test D: correlated (J=1, L=6, k=3, cut=2)
    pt = _pt(6, 3, 1.0)
    r = causal_block_operator_entropy(pt.to_matrix(), 3, 2)
    check("D correlated J=1", r["schmidt_rank"] > 1 and r["entropy"] > 0.0, f"rank={r['schmidt_rank']}, S={r['entropy']:.6f}")

    # Test E: production benchmark settings
    pt = _pt(6, 3, 0.0)
    r = causal_block_operator_entropy(pt.to_matrix(), 3, 2)
    cb = _causal_break_error(pt, 3, 2)
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    sv = mc.characterize(
        Hamiltonian.ising(6, J=0.0, g=1.0),
        AnalogSimParams(dt=0.1, max_bond_dim=64, order=1),
        num_interventions=3,
        cut=2,
        n_pasts=32,
        n_futures=32,
        initial_psi=make_zero_psi(6),
        rng=np.random.default_rng(0),
        intervention_style="haar",
    )
    check("E J=0 S_PT_cb", r["entropy"] < TOL and r["schmidt_rank"] == 1, f"S={r['entropy']:.3e}, rank={r['schmidt_rank']}")
    check("E causal-break", cb < TOL, f"err={cb:.3e}")
    check("E S_V", abs(sv.entropy(2)) < 0.05, f"S_V={sv.entropy(2):.6f}")
    if r["entropy"] >= TOL or r["schmidt_rank"] != 1:
        print("J=0 singular spectrum:", r["singular_values"][:8])
        print("blocks:", causal_block_axis_indices(3))
        print("left:", r["left_axes"], "right:", r["right_axes"])

    # Test F: invariance
    base = causal_block_operator_entropy(pt.to_matrix(), 3, 2)["entropy"]
    scaled = causal_block_operator_entropy(3.7 * pt.to_matrix(), 3, 2)["entropy"]
    check("F scale invariance", abs(base - scaled) < 1e-12, f"{base} vs {scaled}")

    blocks = causal_block_axis_indices(3)
    op = upsilon_to_unfused_operator(pt.to_matrix(), 3)
    # permute within block B_1
    b1 = blocks[1]
    perm = list(range(op.ndim))
    perm[b1[0]], perm[b1[1]] = b1[1], b1[0]
    op2 = np.transpose(op, perm)
    perm_inv = np.argsort(perm)
    op3 = np.transpose(op2, perm_inv)
    mat2 = refold_unfused_to_upsilon(op3, 3)
    perm_ent = causal_block_operator_entropy(mat2, 3, 2)["entropy"]
    check("F internal block permutation", abs(base - perm_ent) < 1e-12)

    u_rng = np.random.default_rng(123)
    a = u_rng.standard_normal((2, 2)) + 1j * u_rng.standard_normal((2, 2))
    q, _ = np.linalg.qr(a)
    op_u = _apply_local_unitary_block(op, 3, 5, q.astype(np.complex128))
    mat_u = refold_unfused_to_upsilon(op_u, 3)
    unitary_ent = causal_block_operator_entropy(mat_u, 3, 2)["entropy"]
    check("F local unitary within block", abs(base - unitary_ent) < 1e-12)

    print("\n" + ("ALL TESTS PASSED" if not failures else f"FAILED: {failures}"))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
