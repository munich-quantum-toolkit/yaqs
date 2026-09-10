#!/usr/bin/env python3
"""Unfuse Choi legs, search Markov factorization, operator-Schmidt audit.

Run: uv run --project /home/aaron/Github/yaqs python pt_choi_markov_audit.py
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.linalg import expm

from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer
from mqt.yaqs.characterization.memory.backends.tomography.basis import (
    assemble_fixed_basis,
    compute_dual_choi_basis,
)
from mqt.yaqs.characterization.memory.backends.tomography.data import assemble_upsilon
from mqt.yaqs.characterization.memory.backends.tomography.process_tensors import (
    DenseProcessTensor,
    _canonicalize_upsilon,
    encode_cptp_choi,
    trace_partial_dense,
)

# ---------------------------------------------------------------------------
# Axis utilities
# ---------------------------------------------------------------------------

# encode_cptp_choi: J += kron(emap(e_in), e_in)  =>  first kron factor = OUTPUT, second = INPUT
# Fused row/col index f = 2 * out + in
FUSED_OUT_FIRST = "out_first"  # f = 2*out + in
FUSED_IN_FIRST = "in_first"  # f = 2*in + out


def unfuse_4(fused: int, *, convention: str = FUSED_OUT_FIRST) -> tuple[int, int]:
    if convention == FUSED_OUT_FIRST:
        return fused // 2, fused % 2
    return fused % 2, fused // 2


def fuse_4(out: int, inp: int, *, convention: str = FUSED_OUT_FIRST) -> int:
    if convention == FUSED_OUT_FIRST:
        return 2 * out + inp
    return 2 * inp + out


def upsilon_to_unfused(
    upsilon: np.ndarray,
    k: int,
    *,
    fuse_convention: str = FUSED_OUT_FIRST,
) -> np.ndarray:
    """Reshape Upsilon (2·4^k, 2·4^k) to labelled unfused operator indices.

    Returns array with axes (in order):
      final_out_k, final_out_b,
      for t in 1..k: slot_t_out_k, slot_t_in_k, slot_t_out_b, slot_t_in_b
    """
    dims = [2] + [4] * k
    n = len(dims)
    # operator tensor: each subsystem has ket then bra
    shape: list[int] = []
    labels: list[str] = ["final_out_k", "final_out_b"]
    for t in range(1, k + 1):
        labels += [f"slot{t}_out_k", f"slot{t}_in_k", f"slot{t}_out_b", f"slot{t}_in_b"]
        shape += [2, 2, 2, 2]

    op = np.asarray(upsilon, dtype=np.complex128).reshape(*sum(([d, d] for d in dims), []))

    # Remap each fused 4-index slot into explicit out/in ket/bra
    out = op
    for t in range(k):
        slot = t + 1  # subsystem index in dims
        # current axes for this subsystem: 2*(slot) and 2*(slot)+1 in the interleaved list
        # build permutation to unfused order
        pass

    # Direct reshape via einsum-style index mapping
    unfused = np.zeros([2, 2] + [2, 2, 2, 2] * k, dtype=np.complex128)
    for indices in np.ndindex(*([2, 2] + [4] * k), *([2, 2] + [4] * k)):
        fk, fb = indices[0], indices[1]
        rest_k = indices[2 : 2 + k]
        rest_b = indices[2 + k + k : 2 + k + k + k]
        # bra side indices start at 2+k
        rest_k_b = indices[2 + k : 2 + k + k]
        # fix indexing
        idx_k = [fk, fb]
        idx_b = [indices[0 + n], indices[1 + n]] if False else []
    # clearer loop over matrix elements
    mat = np.asarray(upsilon, dtype=np.complex128).reshape(*dims, *dims)
    # mat[i0, i1, ..., i0', i1', ...]
    axes_k = list(range(n))
    axes_b = [n + i for i in range(n)]
    result = np.zeros([2, 2] + [2, 2, 2, 2] * k, dtype=np.complex128)
    for idx in np.ndindex(*dims, *dims):
        sub_k = idx[:n]
        sub_b = idx[n:]
        fo_k, fo_b = sub_k[0], sub_b[0]
        out_idx = [fo_k, fo_b]
        for t in range(k):
            ok, ik = unfuse_4(sub_k[t + 1], convention=fuse_convention)
            ob, ib = unfuse_4(sub_b[t + 1], convention=fuse_convention)
            out_idx.extend([ok, ik, ob, ib])
        result[tuple(out_idx)] = mat[idx]
    return result


def axis_labels(k: int) -> list[str]:
    labels = ["final_out_k", "final_out_b"]
    for t in range(1, k + 1):
        labels += [f"slot{t}_out_k", f"slot{t}_in_k", f"slot{t}_out_b", f"slot{t}_in_b"]
    return labels


def _subsystem_axes(sub: int, k: int) -> list[int]:
    """Flat unfused axis indices for subsystem sub (0=final, 1..k=slots)."""
    if sub == 0:
        return [0, 1]
    base = 2 + 4 * (sub - 1)
    return [base, base + 1, base + 2, base + 3]


def op_schmidt_spectrum_flat(
    op: np.ndarray,
    left_axes: list[int],
    right_axes: list[int],
    *,
    tol: float = 1e-12,
) -> tuple[np.ndarray, int, float]:
    """Operator Schmidt: matrix[(left axes), (right axes)]."""
    n = op.ndim
    used = set(left_axes + right_axes)
    if len(used) != len(left_axes) + len(right_axes):
        msg = "left and right axes must be disjoint"
        raise ValueError(msg)
    rest = [i for i in range(n) if i not in used]
    if rest:
        msg = f"axes {rest} not partitioned"
        raise ValueError(msg)
    perm = left_axes + right_axes
    t = np.transpose(op, perm)
    dl = int(np.prod([t.shape[i] for i in range(len(left_axes))]))
    dr = int(np.prod([t.shape[i] for i in range(len(left_axes), len(left_axes) + len(right_axes))]))
    mat = t.reshape(dl, dr)
    s = np.linalg.svd(mat, compute_uv=False)
    rank = int(np.sum(s > tol))
    w = s**2
    total = float(w.sum())
    ent = 0.0 if total < tol else float(-np.sum((w / total) * np.log(w / total + 1e-30)))
    return s, rank, ent


def trace_final_subsystem(upsilon: np.ndarray, k: int) -> np.ndarray:
    """Partial trace over subsystem 0 (final output) -> operator on slots 1..k only."""
    return trace_partial_dense(upsilon, [2] + [4] * k, keep=list(range(1, k + 1)))


def upsilon_slots_to_unfused(
    op_fused: np.ndarray,
    k: int,
    *,
    fuse_convention: str = FUSED_OUT_FIRST,
) -> np.ndarray:
    """Unfuse operator on k slot subsystems only (no final leg), each 4 -> 4 indices."""
    dims = [4] * k
    n = len(dims)
    mat = np.asarray(op_fused, dtype=np.complex128).reshape(*dims, *dims)
    result = np.zeros([2, 2, 2, 2] * k, dtype=np.complex128)
    for idx in np.ndindex(*dims, *dims):
        sub_k = idx[:n]
        sub_b = idx[n:]
        out_idx: list[int] = []
        for t in range(n):
            ok, ik = unfuse_4(sub_k[t], convention=fuse_convention)
            ob, ib = unfuse_4(sub_b[t], convention=fuse_convention)
            out_idx.extend([ok, ik, ob, ib])
        result[tuple(out_idx)] = mat[idx]
    return result


def op_schmidt_from_subsystems(
    upsilon: np.ndarray,
    k: int,
    left_subs: list[int],
    right_subs: list[int],
    *,
    fuse_convention: str = FUSED_OUT_FIRST,
) -> tuple[np.ndarray, int, float]:
    """Op-Schmidt on unfused operator; subsystem 0=final, 1..k = slots."""
    op = upsilon_to_unfused(upsilon, k, fuse_convention=fuse_convention)
    left_axes = [a for s in left_subs for a in _subsystem_axes(s, k)]
    right_axes = [a for s in right_subs for a in _subsystem_axes(s, k)]
    return op_schmidt_spectrum_flat(op, left_axes, right_axes)


def op_schmidt_channel_blocks(
    upsilon: np.ndarray,
    k: int,
    left_slots: list[int],
    right_slots: list[int],
    *,
    fuse_convention: str = FUSED_OUT_FIRST,
) -> tuple[np.ndarray, int, float]:
    """Op-Schmidt on slot subsystems after tracing final; slot indices 1..k."""
    reduced = trace_final_subsystem(upsilon, k)
    op = upsilon_slots_to_unfused(reduced, k, fuse_convention=fuse_convention)

    def slot_axes(slot: int) -> list[int]:
        base = 4 * (slot - 1)
        return [base, base + 1, base + 2, base + 3]

    left_axes = [a for s in left_slots for a in slot_axes(s)]
    right_axes = [a for s in right_slots for a in slot_axes(s)]
    return op_schmidt_spectrum_flat(op, left_axes, right_axes)


def von_neumann_entropy(rho: np.ndarray, *, tol: float = 1e-15) -> float:
    r = 0.5 * (rho + rho.conj().T)
    tr = np.trace(r)
    if abs(tr) < tol:
        return 0.0
    r = r / tr
    ev = np.clip(np.linalg.eigvalsh(r).real, 0.0, 1.0)
    nz = ev[ev > tol]
    return float(-np.sum(nz * np.log(nz))) if nz.size else 0.0


def partial_trace_subsystems(rho: np.ndarray, dims: list[int], keep: list[int]) -> np.ndarray:
    return trace_partial_dense(rho, dims, keep)


def mutual_info_subsystems(rho: np.ndarray, dims: list[int], left: list[int], right: list[int]) -> float:
    s_lr = von_neumann_entropy(rho)
    s_l = von_neumann_entropy(partial_trace_subsystems(rho, dims, left))
    s_r = von_neumann_entropy(partial_trace_subsystems(rho, dims, right))
    return s_l + s_r - s_lr


def build_production(*, length: int, k: int, j: float, g: float, dt: float) -> DenseProcessTensor:
    ham = Hamiltonian.ising(length=length, J=j, g=g)
    params = AnalogSimParams(dt=dt, max_bond_dim=64, order=1)
    return cast(
        DenseProcessTensor,
        MemoryCharacterizer(parallel=False, show_progress=False).build_process_tensor(
            ham,
            params,
            timesteps=[dt] * (k + 1),
            return_type="dense",
            method="exhaustive",
            compress_every=1,
        ),
    )


def markov_product_upsilon(
    rho0: np.ndarray,
    channel_chois: list[np.ndarray],
    *,
    fuse_convention: str = FUSED_OUT_FIRST,
) -> np.ndarray:
    """Standard product Υ = ρ0 ⊗ J1 ⊗ J2 ⊗ … in fused subsystem order [0,1,…,k]."""
    ops = [np.asarray(rho0, dtype=np.complex128).reshape(2, 2)] + list(channel_chois)
    return np.kron(ops[0], np.kron(*ops[1:])) if len(ops) > 1 else ops[0]


def apply_subsystem_permutation(upsilon: np.ndarray, k: int, perm: tuple[int, ...]) -> np.ndarray:
    """Permute subsystems {0,…,k} on both ket and bra sides."""
    dims = [2] + [4] * k
    n = len(dims)
    mat = upsilon.reshape(*dims, *dims)
    perm_k = list(perm) + [n + p for p in perm]
    return np.transpose(mat, perm_k).reshape(upsilon.shape)


def apply_fused_transpose(upsilon: np.ndarray, k: int, slot: int) -> np.ndarray:
    """Swap in/out within fused slot (partial transpose on input indices in out-first convention)."""
    dims = [2] + [4] * k
    mat = upsilon.reshape(*dims, *dims)
    # swap out/in within slot: f = 2*out+in -> swap means map (ok,oi) -> (oi,ok)
    # implement via unfused relabel
    op = upsilon_to_unfused(upsilon, k)
    labels = axis_labels(k)
    # swap out_k <-> in_k and out_b <-> in_b for slot
    sl = slot
    idx = {name: i for i, name in enumerate(labels)}
    perm = list(range(op.ndim))
    for suffix in ("_out_k", "_in_k", "_out_b", "_in_b"):
        a = idx[f"slot{sl}{suffix.replace('_k','_k').replace('out_k','out_k')}"]
    ok, ik, ob, ib = idx[f"slot{sl}_out_k"], idx[f"slot{sl}_in_k"], idx[f"slot{sl}_out_b"], idx[f"slot{sl}_in_b"]
    perm[ok], perm[ik] = ik, ok
    perm[ob], perm[ib] = ib, ob
    op2 = np.transpose(op, perm)
    # refold - use in_first convention after swap
    return refold_unfused(op2, k, fuse_convention=FUSED_IN_FIRST)


def refold_unfused(
    op: np.ndarray,
    k: int,
    *,
    fuse_convention: str = FUSED_OUT_FIRST,
) -> np.ndarray:
    dims = [2] + [4] * k
    n = len(dims)
    mat = np.zeros([*dims, *dims], dtype=np.complex128)
    for idx in np.ndindex(*dims, *dims):
        sub_k = list(idx[:n])
        sub_b = list(idx[n:])
        fo_k, fo_b = sub_k[0], sub_b[0]
        coords = [fo_k, fo_b]
        for t in range(1, n):
            ok = ik = ob = ib = 0
            ok, ik = unfuse_4(sub_k[t], convention=fuse_convention)
            ob, ib = unfuse_4(sub_b[t], convention=fuse_convention)
            coords.extend([ok, ik, ob, ib])
        mat[idx] = op[tuple(coords)]
    return mat.reshape(upsilon_shape(k))


def upsilon_shape(k: int) -> tuple[int, int]:
    d = 2 * (4**k)
    return d, d


def search_markov_equivalence(
    ups_prod: np.ndarray,
    rho0: np.ndarray,
    j_list: list[np.ndarray],
    k: int,
    *,
    atol: float = 1e-10,
) -> dict[str, object]:
    """Search subsystem permutations, global transpose, scale, slot in/out swap."""
    best = {"err": float("inf"), "transform": None}
    ups_p = _canonicalize_upsilon(ups_prod)
    norm_p = np.linalg.norm(ups_p)
    candidates: list[tuple[str, np.ndarray]] = []

    for perm in itertools.permutations(range(k + 1)):
        candidates.append((f"perm{perm}", apply_subsystem_permutation(
            markov_product_upsilon(rho0, j_list), k, perm
        )))

    j_t = [j.conj().T for j in j_list]
    candidates.append(("J.T product", markov_product_upsilon(rho0, j_t)))
    candidates.append(("rho0.T product", markov_product_upsilon(rho0.conj().T, j_list)))

    for slot in range(1, k + 1):
        base = markov_product_upsilon(rho0, j_list)
        candidates.append((f"slot{slot}_swap_inout", apply_fused_transpose(base, k, slot)))

    scales = [1.0, 1.0 / 4.0, 1.0 / 16.0, 4.0, 16.0, 1.0 / (4**k), float(4**k)]
    for name, mat in candidates:
        for sc in scales:
            ref = _canonicalize_upsilon(sc * mat)
            err = float(np.linalg.norm(ups_p - ref) / max(np.linalg.norm(ref), 1e-30))
            if err < best["err"]:
                best = {"err": err, "transform": f"{name}, scale={sc}", "ref_norm": float(np.linalg.norm(ref))}

    # Also test: dual-frame closed form vs product (different object)
    _, choi_basis, _, _ = assemble_fixed_basis(basis="tetrahedral")
    dual = compute_dual_choi_basis(choi_basis)
    frame = sum(dual[a].T for a in range(16))
    for perm in itertools.permutations(range(k + 1)):
        ref = apply_subsystem_permutation(
            markov_product_upsilon(rho0, [frame] * k), k, perm
        )
        ref = _canonicalize_upsilon(ref)
        err = float(np.linalg.norm(ups_p - ref) / max(np.linalg.norm(ref), 1e-30))
        if err < best["err"]:
            best = {"err": err, "transform": f"frame_sum perm{perm}", "ref_norm": float(np.linalg.norm(ref))}

    best["verified"] = bool(best["err"] < atol)
    return best


def identity_channel_audit() -> str:
    jid = encode_cptp_choi(lambda r: r)
    # as 4x4 operator on H_out ⊗ H_in (out-first reshape)
    j4 = jid.reshape(2, 2, 2, 2)  # [ok, ik, ob, ib] from kron rows/cols
    # normalized Choi state
    psi = jid / np.linalg.norm(jid)  # not trace norm
    rho = jid / np.trace(jid)
    rank = int(np.sum(np.linalg.eigvalsh(0.5 * (rho + rho.conj().T)).real > 1e-10))

    # state Schmidt in|out on ket indices only (Choi state vectorization)
    # Bell test: |Phi> = sum_i |ii>/sqrt(2)
    # Bell projector on out⊗in (4-dim)
    bell_ket = np.zeros(4, dtype=np.complex128)
    bell_ket[0] = 1 / np.sqrt(2)
    bell_ket[3] = 1 / np.sqrt(2)
    bell_proj = np.outer(bell_ket, bell_ket.conj())
    bell_fid = float(np.real(np.trace(bell_proj @ rho)))

    # op-Schmidt in|out within single channel (unfused pair 1 only)
    op = jid.reshape(1, 1, 2, 2, 2, 2)  # fake final dim 1
    op = np.asarray(jid, dtype=np.complex128).reshape(2, 2, 2, 2)  # ok,ik,ob,ib
    # treat as operator on (out_k,out_b) | (in_k,in_b) - wrong pairing
    # proper: left = output pair (axes 0,1), right = input pair (axes 2,3)
    mat = np.transpose(op, (0, 2, 1, 3)).reshape(4, 4)
    s_io = np.linalg.svd(mat, compute_uv=False)
    ent_io = von_neumann_entropy(np.diag(s_io**2 / (s_io**2).sum()))

    s_inout_state = mutual_info_subsystems(rho, [2, 2], [0], [1])

    lines = [
        f"encode_cptp_choi(identity) 4x4 matrix rank (as trace-norm state) = {rank}",
        f"Tr(J)={np.trace(jid):.4f}, normalized diag={np.diag(rho).real}",
        f"Fidelity to |Phi+> vector = {bell_fid:.6f}",
        f"State mutual info I(out:in) on vectorized J = {s_inout_state:.6f} (ln2={math.log(2):.6f})",
        f"Op-Schmidt SVs across output|input (ket legs): {s_io}",
        f"Op-Schmidt entropy (s^2 weights): {ent_io:.6f}",
    ]

    # two independent identity channels: product on slots 1 and 2
    ups_two = _canonicalize_upsilon(markov_product_upsilon(np.eye(2) / 2.0, [jid, jid]))
    s2, r2, e2 = op_schmidt_channel_blocks(ups_two, 2, left_slots=[1], right_slots=[2])
    lines.append(f"Two-slot product (subs {{1}}|{{2}}): op-Schmidt rank={r2}, SVs={s2[:4]}, ent={e2:.6f}")
    return "\n".join(lines)


def _embed_final(rho_or_j: np.ndarray, k: int) -> np.ndarray:
    """Embed with trivial 2x2 final factor for op-Schmidt tests on slots only."""
    if k == 1:
        return np.kron(np.eye(2, dtype=np.complex128) / 2.0, rho_or_j)
    return np.kron(np.eye(2, dtype=np.complex128) / 2.0, rho_or_j)


@dataclass
class AuditReport:
    sections: list[str]

    def add(self, title: str, body: str) -> None:
        self.sections.append(f"\n{'=' * 72}\n{title}\n{'=' * 72}\n{body}")

    def dump(self) -> str:
        return "\n".join(self.sections)


def main() -> None:
    dt = 0.1
    k = 2
    r = AuditReport([])

    # Production tensor: L=1, k=2, J=0, g=0 (identity dynamics)
    pt = build_production(length=1, k=k, j=0.0, g=0.0, dt=dt)
    ups = _canonicalize_upsilon(pt.to_matrix())
    rho0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    u_id = np.eye(2, dtype=np.complex128)
    jid = encode_cptp_choi(lambda r: u_id @ r @ u_id.conj().T)

    labels = axis_labels(k)
    r.add(
        "A. Unfused axis labels (from encode_cptp_choi: kron(output, input), f=2·out+in)",
        "Production reshape dims = [final(2), slot1(4), slot2(4), …]\n"
        f"Unfused labels: {', '.join(labels)}\n"
        f"predict contracts past as kron(encode_cptp_choi(E_t)).T on fused legs 1…k.\n"
        f"Unfused tensor shape: {upsilon_to_unfused(ups, k).shape}",
    )

    # B. Markov equivalence search
    search = search_markov_equivalence(ups, rho0, [jid, jid], k)
    r.add(
        "B. Markov product equivalence search (L=1, k=2, J=0, g=0)",
        f"Best relative error: {search['err']:.6e}\n"
        f"Best transform: {search['transform']}\n"
        f"Standard Markov product verified (err < 1e-10): {search['verified']}\n\n"
        + (
            "If not verified: equivalence to standard ρ₀⊗J⊗… Choi operator remains UNVERIFIED\n"
            "under tested permutations, in/out swaps, transposes, and scalar normalizations."
            if not search["verified"]
            else "Equivalence found under listed transform."
        ),
    )

    # C. Operator-Schmidt across complete channel blocks
    dims = [2, 4, 4]
    s_impl, rank_impl, ent_impl = op_schmidt_from_subsystems(ups, k, left_subs=[0, 1], right_subs=[2])
    s_ch, rank_ch, ent_ch = op_schmidt_channel_blocks(ups, k, left_slots=[1], right_slots=[2])
    ups_markov = _canonicalize_upsilon(markov_product_upsilon(rho0, [jid, jid]))
    s_m, rank_m, ent_m = op_schmidt_channel_blocks(ups_markov, k, left_slots=[1], right_slots=[2])

    r.add(
        "C. Genuine operator-Schmidt (unfused ket/bra matricization)",
        f"Production Υ (J=0, L=1, k=2):\n"
        f"  {{final,slot1}} | {{slot2}}: rank={rank_impl}, ent={ent_impl:.6f}, SVs={s_impl[:8]}\n"
        f"  {{slot1}} | {{slot2}} (channel blocks): rank={rank_ch}, ent={ent_ch:.6f}, SVs={s_ch[:8]}\n"
        f"Naive product ρ₀⊗J⊗J (same unfused convention):\n"
        f"  {{slot1}} | {{slot2}}: rank={rank_m}, ent={ent_m:.6f}, SVs={s_m[:8]}\n"
        f"Expected for exact Markov product across complete channels: rank=1, one SV.",
    )

    # D. Total correlations on positive Choi state (normalized Upsilon)
    ups_pos = ups if np.trace(ups).real > 0 else ups
    ups_norm = ups_pos / np.trace(ups_pos)
    mi01 = mutual_info_subsystems(ups_norm, dims, [0, 1], [2])
    mi12 = mutual_info_subsystems(ups_norm, dims, [1], [2])
    mi012 = mutual_info_subsystems(ups_norm, dims, [0, 1], [2])  # same as mi01 for this split
    r.add(
        "D. Total correlations I(L:R) on normalized Υ (subsystems 0=final, 1..k slots)",
        f"I({{0,1}}:{{2}}) = {mi01:.6f}\n"
        f"I({{1}}:{{2}}) = {mi12:.6f}\n"
        f"Product reference I({{1}}:{{2}}) = {mutual_info_subsystems(_canonicalize_upsilon(ups_markov), dims, [1], [2]):.6f}\n"
        f"For true product across {{1}}|{{2}}, I should be 0.",
    )

    r.add("E. Identity-channel sanity test", identity_channel_audit())

    # F. Four-way classification at L=2, k=2, J=0
    pt2 = build_production(length=2, k=k, j=0.0, g=1.0, dt=dt)
    ups2 = _canonicalize_upsilon(pt2.to_matrix())
    s_pt = float(
        __import__(
            "mqt.yaqs.characterization.memory.backends.tomography.process_tensors",
            fromlist=["cut_entanglement_entropy_from_upsilon"],
        ).cut_entanglement_entropy_from_upsilon(ups2, 2, k)
    )
    from mqt.yaqs.characterization.memory.backends.tomography.process_tensors import (
        cut_entanglement_entropy_from_upsilon,
    )
    from mqt.yaqs.characterization.memory.backends.tomography.constructor import build_process_tensor
    from mqt.yaqs.characterization.memory.backends.tomography.process_tensors import MPOProcessTensor

    s_pt = cut_entanglement_entropy_from_upsilon(ups2, 2, k)
    pt_mpo = cast(
        MPOProcessTensor,
        build_process_tensor(
            Hamiltonian.ising(2, J=0.0, g=1.0).mpo,
            AnalogSimParams(dt=dt, max_bond_dim=64, order=1),
            timesteps=[dt] * (k + 1),
            return_type="mpo",
            method="exhaustive",
            compress_every=1,
        ),
    )
    chi = pt_mpo.temporal_bond_dimension(2)
    s_ch2, rank_ch2, ent_ch2 = op_schmidt_channel_blocks(ups2, k, left_slots=[1], right_slots=[2])

    r.add(
        "F. Four-way classification (L=2, k=2, cut=2, J=0)",
        "1. Reduced Choi marginal entropy S_PT = cut_entanglement_entropy(c=2):\n"
        f"   {s_pt:.6f}  [NOT temporal memory entropy]\n"
        "2. MPO bond dimension chi (intervention-slot ordering {0,1}|{2}):\n"
        f"   {chi}  [NOT temporal memory entropy]\n"
        "3. Operator-Schmidt across complete channel blocks {1}|{2}:\n"
        f"   rank={rank_ch2}, entropy={ent_ch2:.6f}, SVs={s_ch2[:6]}\n"
        "4. Influence-functional / environment memory bond: N/A (Case-1 full Choi MPO)\n\n"
        "Standard Markov product equivalence to production: "
        + ("VERIFIED" if search["verified"] else "UNVERIFIED"),
    )

    text = r.dump()
    out = __import__("pathlib").Path("save/pt_cut_reference/pt_choi_markov_audit.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
