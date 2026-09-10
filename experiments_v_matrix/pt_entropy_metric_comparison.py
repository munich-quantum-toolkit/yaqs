#!/usr/bin/env python3
"""Compare S_V^full, I_PT, and legacy S_PT (MPO bond marginal) with entropy variants.

Explains why S_PT(c) can be O(1) at J=0 while S_V and I_PT vanish (Markov limit).
"""

from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path
from typing import Any, cast

import numpy as np

from common import (
    BETA,
    DT_DEFAULT,
    G_DEFAULT,
    characterizer,
    configure_matplotlib_prl,
    ising_chain,
    sim_params,
    write_csv,
)
from mqt.yaqs.characterization.memory.backends.tomography.process_tensors import (
    DenseProcessTensor,
    MPOProcessTensor,
    _canonicalize_upsilon,
    _temporal_bond_dimension_dense,
    causal_block_mutual_information,
    causal_block_operator_entropy,
    cut_entanglement_entropy_from_upsilon,
    trace_partial_dense,
    upsilon_to_unfused_operator,
)
from mqt.yaqs.characterization.memory.operational_memory.full_basis import (
    build_probe_set_from_catalog,
    enumerate_full_probe_catalog,
)
from mqt.yaqs.characterization.memory.operational_memory.response_matrix import (
    assemble_response_matrix,
    response_matrix_entropy,
)
from mqt.yaqs.characterization.memory.operational_memory.run import evaluate_probes_with_weights
from pt_cut_reference import _sv_metrics

LN2 = math.log(2.0)


def _von_neumann_probs(probs: np.ndarray, *, tol: float = 1e-15) -> float:
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > tol]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def _renyi2_probs(probs: np.ndarray, *, tol: float = 1e-15) -> float:
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > tol]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    purity = float(np.sum(p**2))
    if purity <= tol:
        return 0.0
    return float(-np.log(purity))


def _state_probs(rho: np.ndarray, *, tol: float = 1e-15) -> np.ndarray:
    herm = 0.5 * (rho + rho.conj().T)
    tr = float(np.trace(herm).real)
    if tr <= tol:
        return np.zeros(1, dtype=np.float64)
    evals = np.linalg.eigvalsh(herm).real
    evals = np.clip(evals, 0.0, None)
    evals = evals[evals > tol]
    return evals / evals.sum() if evals.size else np.zeros(1, dtype=np.float64)


def _mpo_bond_operator_schmidt(upsilon: np.ndarray, k: int, cut: int) -> dict[str, Any]:
    """Operator Schmidt across MPO bond: subsystems {0..cut-1} | {cut..k}."""
    rho = _canonicalize_upsilon(upsilon)
    dims = [2] + [4] * k
    dim_l = int(np.prod(dims[:cut], dtype=np.int64))
    dim_r = int(np.prod(dims[cut:], dtype=np.int64))
    rho4 = rho.reshape(dim_l, dim_r, dim_l, dim_r)
    mat = np.transpose(rho4, (0, 2, 1, 3)).reshape(dim_l * dim_l, dim_r * dim_r)
    s = np.linalg.svd(mat, compute_uv=False)
    weights = s**2
    total = float(weights.sum())
    p = weights / total if total > 1e-30 else np.zeros_like(weights)
    chi = 1
    try:
        chi = int(_temporal_bond_dimension_dense(rho, k, cut))
    except ValueError:
        chi = int(np.sum(s > 1e-12))
    return {
        "singular_values": s,
        "schmidt_weights_normalized": p,
        "entropy_vn": _von_neumann_probs(p),
        "entropy_renyi2": _renyi2_probs(p),
        "rank": int(np.sum(s > 1e-12)),
        "chi": chi,
    }


def _s_pt_spectrum(upsilon: np.ndarray, k: int, cut: int) -> dict[str, Any]:
    """Legacy S_PT: marginal on intervention legs {1,…,cut-1}."""
    rho = _canonicalize_upsilon(upsilon)
    dims = [2] + [4] * k
    if cut <= 1:
        return {"S_PT_vn": 0.0, "S_PT_renyi2": 0.0, "eigenvalues": np.array([1.0])}
    rho_past = trace_partial_dense(rho, dims, keep=list(range(1, cut)))
    probs = _state_probs(rho_past)
    return {
        "S_PT_vn": cut_entanglement_entropy_from_upsilon(rho, cut=cut, num_interventions=k),
        "S_PT_renyi2": _renyi2_probs(probs),
        "eigenvalues": probs,
    }


def _ipt_entropies(upsilon: np.ndarray, k: int, cut: int) -> dict[str, float]:
    """Von Neumann and Renyi-2 past-future mutual information at causal blocks."""
    from mqt.yaqs.characterization.memory.backends.tomography.process_tensors import (
        causal_block_axis_indices,
        _is_ket_unfused_axis,
    )

    mi = causal_block_mutual_information(upsilon, k, cut)
    op = upsilon_to_unfused_operator(upsilon, k)
    blocks = causal_block_axis_indices(k)
    p_axes = [i for b in blocks[:cut] for i in b]
    f_axes = [i for b in blocks[cut:] for i in b]
    p_ket = [a for a in p_axes if _is_ket_unfused_axis(a)]
    p_bra = [a for a in p_axes if not _is_ket_unfused_axis(a)]
    f_ket = [a for a in f_axes if _is_ket_unfused_axis(a)]
    f_bra = [a for a in f_axes if not _is_ket_unfused_axis(a)]
    perm = p_ket + f_ket + p_bra + f_bra
    tensor = np.transpose(op, perm)
    n_p, n_f = len(p_ket), len(f_ket)
    n_ket = n_p + n_f
    dim = 1 << n_ket
    dims = [2] * n_ket
    rho_pf = tensor.reshape(dim, dim).astype(np.complex128)
    rho_pf = 0.5 * (rho_pf + rho_pf.conj().T)
    tr = float(np.trace(rho_pf).real)
    rho_pf = rho_pf / tr if tr > 1e-15 else rho_pf
    rho_p = trace_partial_dense(rho_pf, dims, keep=list(range(n_p)))
    rho_f = trace_partial_dense(rho_pf, dims, keep=list(range(n_p, n_ket)))
    p_p = _state_probs(rho_p)
    p_f = _state_probs(rho_f)
    p_pf = _state_probs(rho_pf)
    s_p_r2, s_f_r2, s_pf_r2 = _renyi2_probs(p_p), _renyi2_probs(p_f), _renyi2_probs(p_pf)
    return {
        "I_PT_vn": float(mi["mutual_information"]),
        "I_PT_renyi2": float(s_p_r2 + s_f_r2 - s_pf_r2),
        "S_p_vn": float(mi["entropy_p"]),
        "S_f_vn": float(mi["entropy_f"]),
        "S_pf_vn": float(mi["entropy_pf"]),
    }


def _evaluate_j(
    *,
    length: int,
    k: int,
    cut: int,
    jv: float,
    mc: Any,
    params: Any,
    timesteps: list[float],
    probe_full: Any,
) -> dict[str, float | int]:
    ham = ising_chain(length=length, j=jv, g=G_DEFAULT)
    ham.ensure_encoded("mpo")
    pt_dense = cast(
        DenseProcessTensor,
        mc.build_process_tensor(
            ham,
            params,
            timesteps=timesteps,
            return_type="dense",
            method="exhaustive",
            compress_every=1,
        ),
    )
    pt_mpo = cast(
        MPOProcessTensor,
        mc.build_process_tensor(
            ham,
            params,
            timesteps=timesteps,
            return_type="mpo",
            method="exhaustive",
            compress_every=1,
        ),
    )
    ups = pt_dense.to_matrix()
    sv_full, _ = _sv_metrics(probe_full, pt_dense)
    spt = _s_pt_spectrum(ups, k, cut)
    mpo = _mpo_bond_operator_schmidt(ups, k, cut)
    ipt = _ipt_entropies(ups, k, cut)
    cb = causal_block_operator_entropy(ups, k, cut)
    cb_sv = np.asarray(cb["singular_values"], dtype=np.float64)
    cb_w = cb_sv**2 / max(float(np.sum(cb_sv**2)), 1e-30)
    return {
        "J": float(jv),
        "S_V_full": float(sv_full),
        "I_PT": float(ipt["I_PT_vn"]),
        "I_PT_renyi2": float(ipt["I_PT_renyi2"]),
        "S_PT": float(spt["S_PT_vn"]),
        "S_PT_renyi2": float(spt["S_PT_renyi2"]),
        "S_mpo_schmidt_vn": float(mpo["entropy_vn"]),
        "S_mpo_schmidt_renyi2": float(mpo["entropy_renyi2"]),
        "S_PT_cb": float(cb["entropy"]),
        "S_PT_cb_renyi2": _renyi2_probs(cb_w),
        "chi_mpo": int(mpo["chi"]),
        "rank_mpo_schmidt": int(mpo["rank"]),
        "S_PT_top_eval": float(spt["eigenvalues"][0]) if spt["eigenvalues"].size else 0.0,
    }


def _j0_report(rows: list[dict[str, float | int]], *, k: int, cut: int) -> str:
    j0 = next(r for r in rows if abs(float(r["J"])) < 1e-12)
    lines = [
        "J=0 diagnosis (Markov / uncoupled limit)",
        "--------------------------------------",
        f"  S_V^full     = {float(j0['S_V_full']):.3e}  (operational memory; expect ~0)",
        f"  I_PT (VN)    = {float(j0['I_PT']):.3e}  (causal-block MI; expect ~0)",
        f"  S_PT (VN)    = {float(j0['S_PT']):.3e}  (past-leg marginal; often ~ln 2 at c=2)",
        f"  S_mpo Schmidt= {float(j0['S_mpo_schmidt_vn']):.3e}  (operator Schmidt on bond {{0..{cut - 1}}}|{{{cut}..{k}}})",
        f"  S_PT^cb      = {float(j0['S_PT_cb']):.3e}  (operator Schmidt on causal blocks)",
        f"  chi_MPO      = {int(j0['chi_mpo'])}",
        "",
        "Why S_PT can be large at J=0:",
        "  S_PT traces out the final system leg and future intervention legs, then",
        "  computes S(rho_{legs 1..c-1}) in the *fused* Choi slot ordering.",
        "  At J=0 the dynamics are Markovian, but the dual-frame tomography ensemble",
        "  leaves the first intervention leg in a maximally mixed reduced state when",
        f"  cut>=2 and k>=2 — hence S_PT(c=2) ≈ ln 2 = {LN2:.6f} even when I_PT=0.",
        "",
        "  I_PT and S_V^full use the *causal* past|future split (complete channel blocks)",
        "  and measure correlations / operational response — both vanish when there is",
        "  no memory.",
        "",
        "Renyi-2 at J=0:",
        f"  S_PT_renyi2 = {float(j0['S_PT_renyi2']):.3f}",
        f"  I_PT_renyi2 = {float(j0['I_PT_renyi2']):.3e}",
        f"  S_mpo_schmidt_renyi2 = {float(j0['S_mpo_schmidt_renyi2']):.3f}",
    ]
    return "\n".join(lines)


def _control_benchmarks_text() -> str:
    return textwrap.dedent(
        """
        Control benchmarks (literature-relevant validation of S_V)
        ------------------------------------------------------------
        1. Markov limit (J=0):
           Expect S_V^full ≈ 0 and I_PT ≈ 0. S_PT may remain O(1) — not a failure of S_V.

        2. Causal-break / compositionality:
           Past-intervention swaps at the cut should not change future statistics
           (see pt_cut_reference._causal_break_error). Operational Markov test.

        3. Monotonicity with coupling:
           For fixed (c,k,L), S_V^full and I_PT should increase with |J| on this benchmark.

        4. Exhaustive vs finite budget:
           S_V^full from complete 64×64 ensemble; finite 8×8 subsets overestimate
           (see pt_response_comparison_prx subsampling bands).

        5. Cross-metric ordering:
           I_PT ≥ 0 always; S_V^full ≤ I_PT not required (different objects).
           Correlation across J is empirical, not universal (see multi-cut figure).

        6. Alternative PT entropies (this script):
           - S_mpo_schmidt: normalized |Schmidt|^2 across MPO vector bond
           - S_PT^cb: operator Schmidt on causal blocks (unnormalized operator)
           Compare which tracks S_V; I_PT is the causal-state analogue of S_V's split.

        7. Known literature context:
           - Process-tensor / process-matrix frameworks: temporal correlations in Choi space
           - Operational probing / intervention ensembles: memory from conditioned responses
           S_V is the latter; S_PT is a bond diagnostic that need not vanish Markovianly.
        """
    ).strip()


def _plot_comparison(rows: list[dict[str, float | int]], *, out_dir: Path, k: int, cut: int, length: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogFormatterMathtext, LogLocator

    configure_matplotlib_prl()
    j = np.asarray([float(r["J"]) for r in rows], dtype=np.float64)
    sv = np.asarray([float(r["S_V_full"]) for r in rows], dtype=np.float64)
    ipt_r2 = np.asarray([float(r["I_PT_renyi2"]) for r in rows], dtype=np.float64)
    spt_cb = np.asarray([float(r["S_PT_cb"]) for r in rows], dtype=np.float64)
    floor = 1e-12
    atol = 1e-15

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"wspace": 0.32})

    def _plot_metric(ax, y, *, label, color, marker, ls="-", log_floor: float | None = floor):
        y_arr = np.asarray(y, dtype=np.float64)
        if log_floor is not None:
            y_arr = np.maximum(y_arr, log_floor)
        ax.plot(j, y_arr, ls=ls, marker=marker, color=color, lw=1.7, ms=5, mew=0.75, label=label)

    _plot_metric(ax1, sv, label=r"$S_V^{\mathrm{full}}$", color="#0072B2", marker="o")
    _plot_metric(ax1, ipt_r2, label=r"$I_{\mathrm{PT}}$ (Renyi-2)", color="#D55E00", marker="s", ls="--")
    _plot_metric(ax1, spt_cb, label=r"$S_{\mathrm{PT}}^{\mathrm{cb}}$", color="#009E73", marker="^", ls=":")
    ax1.set_yscale("log")
    ax1.set_xlim(0.0, float(j.max()))
    ax1.set_xlabel(r"Coupling $J$")
    ax1.set_ylabel("Information measure (nats)")
    ax1.set_title(rf"$L={length}$, $k={k}$, $c={cut}$", fontsize=9)
    ax1.legend(frameon=False, fontsize=7.8, loc="lower right")
    ax1.yaxis.set_major_formatter(LogFormatterMathtext())
    ax1.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax1.tick_params(direction="in", top=True, right=True, which="both")
    ax1.text(0.03, 0.97, "(a)", transform=ax1.transAxes, fontsize=12, fontweight="bold", va="top")

    denom = np.maximum(np.abs(sv), atol)
    rel_ipt = np.abs(ipt_r2 - sv) / denom
    rel_spt = np.abs(spt_cb - sv) / denom
    both_small_ipt = (np.abs(sv) <= atol) & (np.abs(ipt_r2 - sv) <= atol)
    both_small_spt = (np.abs(sv) <= atol) & (np.abs(spt_cb - sv) <= atol)
    rel_ipt[both_small_ipt] = 0.0
    rel_spt[both_small_spt] = 0.0
    _plot_metric(
        ax2,
        rel_ipt,
        label=r"$|I_{\mathrm{PT}}^{(2)} - S_V^{\mathrm{full}}| / S_V^{\mathrm{full}}$",
        color="#D55E00",
        marker="s",
        ls="--",
    )
    _plot_metric(
        ax2,
        rel_spt,
        label=r"$|S_{\mathrm{PT}}^{\mathrm{cb}} - S_V^{\mathrm{full}}| / S_V^{\mathrm{full}}$",
        color="#009E73",
        marker="^",
        ls=":",
    )
    ax2.set_yscale("log")
    ax2.set_xlim(0.0, float(j.max()))
    ax2.set_xlabel(r"Coupling $J$")
    ax2.set_ylabel(r"Relative deviation from $S_V^{\mathrm{full}}$")
    ax2.legend(frameon=False, fontsize=7.0, loc="lower right")
    ax2.yaxis.set_major_formatter(LogFormatterMathtext())
    ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax2.tick_params(direction="in", top=True, right=True, which="both")
    ax2.text(0.03, 0.97, "(b)", transform=ax2.transAxes, fontsize=12, fontweight="bold", va="top")

    stem = out_dir / "fig_entropy_metric_comparison"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", dpi=600)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=600)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--length", type=int, default=2, help="Chain length (default 2 = fast).")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--cut", type=int, default=2)
    p.add_argument("--j-values", type=str, default="0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2")
    p.add_argument("--out-dir", type=Path, default=Path("save/pt_cut_reference"))
    p.add_argument("--paper-scale", action="store_true", help="Use L=6, k=3, c=2 (slow).")
    args = p.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    length = 6 if args.paper_scale else args.length
    k = 3 if args.paper_scale else args.k
    cut = 2 if args.paper_scale else args.cut
    j_values = [float(x.strip()) for x in args.j_values.split(",") if x.strip()]

    catalog = enumerate_full_probe_catalog(cut=cut, num_interventions=k)
    n_p, n_f = len(catalog.past_settings), len(catalog.future_settings)
    probe_full = build_probe_set_from_catalog(
        catalog,
        np.arange(n_p, dtype=np.int64),
        np.arange(n_f, dtype=np.int64),
    )

    mc = characterizer(parallel=False)
    params = sim_params(dt=DT_DEFAULT)
    timesteps = [DT_DEFAULT] * (k + 1)

    rows: list[dict[str, float | int]] = []
    for jv in j_values:
        print(f"J={jv:g}  L={length} k={k} c={cut}", flush=True)
        rows.append(
            _evaluate_j(
                length=length,
                k=k,
                cut=cut,
                jv=jv,
                mc=mc,
                params=params,
                timesteps=timesteps,
                probe_full=probe_full,
            )
        )

    write_csv(out_dir / "entropy_metric_comparison.csv", rows)
    _plot_comparison(rows, out_dir=out_dir, k=k, cut=cut, length=length)

    report = "\n\n".join(
        [
            f"Entropy metric comparison  L={length}, k={k}, c={cut}",
            f"Probe ensemble: |P|={n_p}, |F|={n_f}",
            _j0_report(rows, k=k, cut=cut),
            _control_benchmarks_text(),
        ]
    )
    (out_dir / "entropy_metric_comparison_report.txt").write_text(report + "\n", encoding="utf-8")
    print(report, flush=True)
    print(f"\nWrote {out_dir / 'fig_entropy_metric_comparison.png'}", flush=True)


if __name__ == "__main__":
    main()
