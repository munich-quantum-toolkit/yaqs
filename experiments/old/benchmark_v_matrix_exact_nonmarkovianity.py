#!/usr/bin/env python3
"""Exact V-matrix non-Markovianity benchmark (causal cut with split instrument).

At cut slot ``c`` (1-based), the instrument is **split**: the past probe supplies only the measurement
ket at ``c``, the future probe supplies only the preparation ket at ``c``, and the simulation applies
the corresponding project–normalize–reprepare step with ``φ`` from past ``i`` and ``ψ`` from future ``j``.

Choice of exact backend path:
- This script mirrors object-first diagnostics usage by wrapping each exact
  process in :class:`~mqt.yaqs.characterization.process_tensors.diagnostics.exact.ExactProbeProcess`
  and calling :func:`~mqt.yaqs.characterization.process_tensors.diagnostics.probe.probe_process`.
- The exact wrapper uses the existing rollout backend (`_simulate_sequences`)
  in final-state mode (`record_step_states=False`).
- Scheduling follows the process-tensor / **comb** convention: for ``k`` instrument slots there are
  ``k+1`` free-evolution segments ``U_1`` … ``U_{k+1}`` (evolve, then measure--reprepare, then evolve, …).
  ``timesteps`` passed to the backend has length ``k+1`` (default: ``dt`` repeated).
- That gives exact backend density-matrix outputs for sampled instrument sequences,
  without any Transformer model.

Initial states (``--n-seeds``):
- Site 0 is treated as the *system*; sites ``1..L-1`` are the *environment*, fixed to
  :math:`|0\\rangle` (tensor product :math:`|\\psi_{\\mathrm{sys}}\\rangle \\otimes |0\\rangle^{\\otimes (L-1)}`).
- ``n_seeds=1`` (default): full chain starts in :math:`|0\\ldots 0\\rangle` (previous behavior).
- ``n_seeds>1``: fix a list of :math:`n_{\\mathrm{seeds}}` such initial states **once per cut** (same list for
  every :math:`(J,g)` in that cut). For each initial state, build :math:`V` and compute metrics separately;
  **scalar** summary metrics are then averaged across initial states. Saved :math:`V` / spectra / row-distance
  arrays use **seed 0** for file compatibility with existing plots.

Diagnostic-only split-cut suite (weights, centering variants, shared probes): see
``experiments/benchmark_v_matrix_exact_diagnostics.py`` (does not change the main estimator).

Weight-aware candidate metrics (:math:`w_{ij}^{\\beta}`) and paired benchmark:
``experiments/benchmark_v_matrix_weighted_candidates.py`` (additive; default metric unchanged).

``--branch-weight-beta`` (default ``0``): set to ``0.5`` (sqrt) or ``1.0`` (linear) to build
:math:`V^{(\\beta)}_{i,(j,\\alpha)} = w_{ij}^{\\beta}[\\rho_{ij}]_\\alpha` with the same past-row centering;
uses the diagnostic rollout path for per-entry cumulative weights.

Optional ``--probe-convergence-study``:
- Varies probe budget :math:`m = N_p = N_f`, repeats over independent probe draws, compares to a large
 reference :math:`m_{\\mathrm{ref}}`. Uses :math:`n_{\\mathrm{seeds}}=1` (fixed :math:`|0\\ldots 0\\rangle`)
  and the four default physical cases only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from mqt.yaqs.characterization.process_tensors.core.encoding import pauli_xyz_to_rho, unpack_rho8
from mqt.yaqs.characterization.process_tensors.diagnostics.exact import (
    ExactProbeProcess,
    evaluate_exact_probe_set_with_diagnostics,
)
from mqt.yaqs.characterization.process_tensors.diagnostics.probe import (
    ProbeSet,
    analyze_v_matrix,
    probe_process,
    sample_split_cut_probes,
)
from mqt.yaqs.characterization.process_tensors.diagnostics.v_matrix_diag import (
    build_weighted_v_matrix,
    center_past_rows,
    prepare_branch_weights,
)
from mqt.yaqs.characterization.process_tensors.surrogates.utils import _random_pure_state
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument(
        "--cut",
        type=int,
        default=15,
        help="Causal cut slot c (1..k): instrument c is split (past: measurement only; future: preparation only).",
    )
    p.add_argument("--n-pasts", type=int, default=4)
    p.add_argument("--n-futures", type=int, default=4)
    p.add_argument("--g-fixed", type=float, default=1.0)
    p.add_argument("--j-sweep", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0")
    p.add_argument(
        "--cut-sweep",
        type=str,
        default=None,
        help='Optional cut sweep, e.g. "4,6,8,10,12,14,16". Defaults to single --cut behavior.',
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        dest="n_seeds",
        help=(
            "Initial states: env (sites 1..L-1) fixed to |0>, system (site 0) varies. "
            "n_seeds=1: single |0...0> (legacy). n_seeds>1: same fixed list per cut; "
            "one V per initial state; summary scalars are averaged across initial states."
        ),
    )
    p.add_argument("--out-dir", type=Path, default=Path("benchmark_v_matrix_exact_results"))
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate plots from existing summary.json/csv in --out-dir.",
    )
    p.add_argument("--parallel", action="store_true", default=True)
    p.add_argument("--no-parallel", dest="parallel", action="store_false")
    p.add_argument(
        "--probe-convergence-study",
        action="store_true",
        dest="probe_convergence_study",
        help=(
            "Run probe-budget convergence only: vary m=n_pasts=n_futures, repeat probe draws, "
            "compare to --convergence-m-ref. Implies n_seeds=1; uses four default cases; single --cut."
        ),
    )
    p.add_argument(
        "--convergence-m-list",
        type=str,
        default="4,8,16,32",
        help='Comma-separated probe budgets m (N_p=N_f=m), e.g. "4,8,16,32".',
    )
    p.add_argument(
        "--convergence-m-ref",
        type=int,
        default=64,
        help="Reference probe budget m_ref (large); errors vs mean metrics at m_ref over the same draws.",
    )
    p.add_argument(
        "--convergence-probe-draws",
        type=int,
        default=5,
        dest="convergence_probe_draws",
        help="Number of independent probe-set draws per m.",
    )
    p.add_argument(
        "--branch-weight-beta",
        type=float,
        default=1.0,
        dest="branch_weight_beta",
        help=(
            "Branch weight exponent β for V_{i,(j,α)} = w_ij^β [ρ_ij]_α after exact rollout (0 = unweighted, "
            "legacy behavior). Use 0.5 for sqrt weighting or 1.0 for linear; requires per-sequence cumulative "
            "weights (diagnostic rollout path)."
        ),
    )
    return p.parse_args()


def _parse_j_sweep(spec: str) -> list[float]:
    vals = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    if not vals:
        raise ValueError("j-sweep must contain at least one value.")
    return vals


def _parse_int_list(spec: str | None) -> list[int]:
    if spec is None:
        return []
    vals = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            vals.append(int(tok))
    return vals


def _list_initial_states_sys_env0(*, L: int, n_seeds: int, rng: np.random.Generator) -> list[np.ndarray]:
    """System = site 0; environment = sites 1..L-1 fixed to |0⟩."""
    if n_seeds < 1:
        msg = "n_seeds must be >= 1."
        raise ValueError(msg)
    if n_seeds == 1:
        psi = np.zeros(2**L, dtype=np.complex128)
        psi[0] = 1.0 + 0.0j
        return [psi]
    out: list[np.ndarray] = []
    for _ in range(n_seeds):
        psi_sys = _random_pure_state(rng)
        if L <= 1:
            out.append(psi_sys.astype(np.complex128))
            continue
        z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
        psi = psi_sys.astype(np.complex128)
        for _i in range(L - 1):
            psi = np.kron(psi, z)
        nrm = float(np.linalg.norm(psi))
        out.append(psi / max(nrm, 1e-15))
    return out


def _average_analysis_dicts(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    """Average scalar metrics; keep first seed's large arrays for saving (caller replaces)."""
    if not analyses:
        raise ValueError("empty analyses")
    keys_float = [
        "delta",
        "delta_norm",
        "entropy",
        "max_row_distance",
        "mean_row_distance",
        "median_row_distance",
    ]
    out: dict[str, Any] = {}
    for key in keys_float:
        out[key] = float(np.mean([float(a[key]) for a in analyses]))
    out["rank"] = int(round(float(np.mean([float(a["rank"]) for a in analyses]))))
    return out

def _probe_features_to_rho_ij(feat_ij: np.ndarray) -> np.ndarray:
    """Map per-entry probe features to :math:`2\\times 2` matrices (``rho8`` or ``pauli_xyz``)."""
    n_p, n_f, d = feat_ij.shape
    out = np.empty((n_p, n_f, 2, 2), dtype=np.complex128)
    if d == 8:
        for i in range(n_p):
            for j in range(n_f):
                out[i, j] = unpack_rho8(feat_ij[i, j])
    elif d == 3:
        for i in range(n_p):
            for j in range(n_f):
                out[i, j] = pauli_xyz_to_rho(feat_ij[i, j])
    else:
        msg = f"Expected last dim 8 (rho8) or 3 (pauli_xyz), got {d}."
        raise ValueError(msg)
    return out


def _save_point(
    *,
    out_point: Path,
    rhos: np.ndarray,
    v: np.ndarray,
    v_centered: np.ndarray,
    singular_values: np.ndarray,
    summary: dict[str, Any],
) -> None:
    out_point.mkdir(parents=True, exist_ok=True)
    np.save(out_point / "rhos.npy", rhos)
    np.save(out_point / "V.npy", v)
    np.save(out_point / "V_centered.npy", v_centered)
    np.save(out_point / "singular_values.npy", singular_values)
    (out_point / "summary.json").write_text(json.dumps(summary, indent=2))


def run_single_parameter_point(
    *,
    case_name: str,
    J: float,
    g: float,
    args: argparse.Namespace,
    probe_set: ProbeSet,
    shared_initial_states: list[np.ndarray],
    init_stack: np.ndarray,
    out_dir: Path,
    save_outputs: bool = True,
) -> dict[str, Any]:
    print(f"\ncase={case_name} J={J:.6f} g={g:.6f}", flush=True)
    op = MPO.ising(length=int(args.L), J=float(J), g=float(g))
    sim_params = AnalogSimParams(dt=float(args.dt), solver="MCWF", show_progress=False)

    n_seeds = int(args.n_seeds)
    if len(shared_initial_states) != n_seeds:
        msg = f"shared_initial_states length {len(shared_initial_states)} != n_seeds={n_seeds}."
        raise ValueError(msg)

    beta = float(getattr(args, "branch_weight_beta", 0.0))

    def _weighted_v_from_rollout(psi0: np.ndarray) -> dict[str, Any]:
        pauli_xyz_ij, weights_ij, _traces = evaluate_exact_probe_set_with_diagnostics(
            probe_set=probe_set,
            operator=op,
            sim_params=sim_params,
            initial_psi=psi0,
            parallel=bool(args.parallel),
        )
        w_clean, wmeta = prepare_branch_weights(weights_ij, log_warnings=True)
        v_w = build_weighted_v_matrix(pauli_xyz_ij, w_clean, beta)
        v_c = center_past_rows(v_w)
        ana_w = analyze_v_matrix(v_w, v_c)
        return {
            "pauli_xyz_ij": pauli_xyz_ij,
            "V": v_w,
            "V_centered": v_c,
            **ana_w,
            "weights_ij": weights_ij,
            "weight_preparation": wmeta,
        }

    weights_ij: np.ndarray | None = None
    weight_preparation: dict[str, Any] | None = None

    if n_seeds == 1:
        if abs(beta) <= 1e-15:
            exact_process = ExactProbeProcess(
                operator=op,
                sim_params=sim_params,
                initial_psi=shared_initial_states[0],
                parallel=bool(args.parallel),
            )
            out = probe_process(
                process=exact_process,
                cut=int(args.cut),
                k=int(args.k),
                probe_set=probe_set,
                return_v=True,
            )
            pauli_xyz_ij = out["pauli_xyz_ij"]
            v = out["V"]
            v_centered = out["V_centered"]
            ana = {k: out[k] for k in out if k not in ("probe_set", "pauli_xyz_ij")}
        else:
            out_w = _weighted_v_from_rollout(shared_initial_states[0])
            pauli_xyz_ij = out_w["pauli_xyz_ij"]
            v = out_w["V"]
            v_centered = out_w["V_centered"]
            weights_ij = out_w["weights_ij"]
            weight_preparation = out_w["weight_preparation"]
            ana = {
                k: out_w[k]
                for k in out_w
                if k not in ("pauli_xyz_ij", "V", "V_centered", "weights_ij", "weight_preparation")
            }
    else:
        ana_list: list[dict[str, Any]] = []
        pauli_xyz_first: np.ndarray | None = None
        for idx, psi0 in enumerate(shared_initial_states):
            if abs(beta) <= 1e-15:
                exact_process = ExactProbeProcess(
                    operator=op,
                    sim_params=sim_params,
                    initial_psi=psi0,
                    parallel=bool(args.parallel),
                )
                out_s = probe_process(
                    process=exact_process,
                    cut=int(args.cut),
                    k=int(args.k),
                    probe_set=probe_set,
                    return_v=True,
                )
            else:
                out_s = _weighted_v_from_rollout(psi0)
            pauli_xyz_ij_s = out_s["pauli_xyz_ij"]
            if idx == 0:
                pauli_xyz_first = pauli_xyz_ij_s
                if abs(beta) > 1e-15:
                    weights_ij = out_s["weights_ij"]
                    weight_preparation = out_s["weight_preparation"]
            ana_list.append(out_s)
        if pauli_xyz_first is None:
            raise RuntimeError("internal: no pauli_xyz_ij for n_seeds>1")
        pauli_xyz_ij = pauli_xyz_first
        ana0 = ana_list[0]
        v = ana0["V"]
        v_centered = ana0["V_centered"]
        ana_avg = _average_analysis_dicts(ana_list)
        ana = {
            **ana_avg,
            "singular_values": ana0["singular_values"],
            "row_distances": ana0["row_distances"],
        }

    point_summary: dict[str, Any] = {
        "case_name": case_name,
        "L": int(args.L),
        "k": int(args.k),
        "dt": float(args.dt),
        "cut": int(args.cut),
        "J": float(J),
        "g": float(g),
        "n_pasts": int(args.n_pasts),
        "n_futures": int(args.n_futures),
        "n_seeds": int(args.n_seeds),
        "branch_weight_beta": float(beta),
        "delta": float(ana["delta"]),
        "delta_norm": float(ana["delta_norm"]),
        "entropy": float(ana["entropy"]),
        "rank": int(ana["rank"]),
        "max_row_distance": float(ana["max_row_distance"]),
        "mean_row_distance": float(ana["mean_row_distance"]),
        "median_row_distance": float(ana["median_row_distance"]),
    }
    if n_seeds > 1:
        point_summary["metrics_averaged_over_initial_states"] = True
    if weight_preparation is not None:
        point_summary["weight_preparation"] = weight_preparation
    if save_outputs:
        rhos = _probe_features_to_rho_ij(pauli_xyz_ij)
        point_dir = out_dir / f"{case_name}_J{J:.3f}_g{g:.3f}"
        _save_point(
            out_point=point_dir,
            rhos=rhos,
            v=v,
            v_centered=v_centered,
            singular_values=ana["singular_values"],
            summary=point_summary,
        )
        np.save(point_dir / "past_features.npy", probe_set.past_features)
        np.save(point_dir / "future_features.npy", probe_set.future_features)
        np.save(point_dir / "initial_states.npy", init_stack)
        np.save(point_dir / "row_distances.npy", ana["row_distances"])
        if abs(beta) > 1e-15 and weights_ij is not None:
            np.save(point_dir / "weights_ij.npy", weights_ij)

    # Quick per-case readout.
    line = (
        f"delta={point_summary['delta']:.6e}, "
        f"delta_norm={point_summary['delta_norm']:.6e}, "
        f"entropy={point_summary['entropy']:.6e}, "
        f"rank={point_summary['rank']}, "
        f"mean_row_dist={point_summary['mean_row_distance']:.6e}"
    )
    if abs(beta) > 1e-15:
        line += f", branch_weight_beta={beta:g}"
    print(f"  {line}", flush=True)
    return point_summary


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def configure_matplotlib_for_paper() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "font.family": "sans-serif",
            "mathtext.default": "it",
        }
    )


def savefig_base(fig: Any, path_stem: Path) -> None:
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def _case_style(row: dict[str, Any]) -> dict[str, Any]:
    name = str(row.get("case_name", ""))
    jv = float(row.get("J", 0.0))
    gv = float(row.get("g", 0.0))
    if name == "trivial" or (abs(jv) <= 1e-15 and abs(gv) <= 1e-15):
        return {"color": "black", "marker": "x", "label": "trivial"}
    if name == "markov_ref" or (abs(jv) <= 1e-15 and abs(gv - 1.0) <= 1e-15):
        return {"color": "0.45", "marker": "s", "label": "markov_ref"}
    if jv < 0.8:
        return {"color": "#1f77b4", "marker": "o", "label": f"J={jv:g}"}
    return {"color": "#d62728", "marker": "^", "label": f"J={jv:g}"}


def _add_panel_label(ax: Any, label: str) -> None:
    ax.text(0.03, 0.95, label, transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")


def _safe_logscale_if_wide(ax: Any, ys: list[float], *, ratio_threshold: float = 100.0) -> None:
    ys_pos = [float(y) for y in ys if float(y) > 0.0]
    has_zero = any(float(y) <= 0.0 for y in ys)
    if len(ys_pos) < 2 or has_zero:
        return
    span = max(ys_pos) / max(min(ys_pos), 1e-300)
    if span >= ratio_threshold:
        ax.set_yscale("log")


def _coerce_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        for key in [
            "J",
            "g",
            "delta_norm",
            "entropy",
            "mean_row_distance",
            "cut",
        ]:
            if key in rr:
                try:
                    rr[key] = float(rr[key])
                except Exception:
                    pass
        out.append(rr)
    return out


def _find_representative_point_dirs(out_dir: Path) -> dict[str, Path]:
    candidates = {
        "trivial": ["trivial_J0.000_g0.000"],
        "markov_ref": ["markov_ref_J0.000_g1.000"],
        "j04": ["sweep_J0.400_g1.000", "nm_j0.4_J0.400_g1.000"],
        "j06": ["sweep_J0.600_g1.000", "nm_j0.6_J0.600_g1.000"],
        "j10": ["sweep_J1.000_g1.000", "nm_j1.0_J1.000_g1.000"],
        "j12": ["sweep_J1.200_g1.000", "nm_j1.2_J1.200_g1.000"],
        "j14": ["sweep_J1.400_g1.000", "nm_j1.4_J1.400_g1.000"],
        "j16": ["sweep_J1.600_g1.000", "nm_j1.6_J1.600_g1.000"],
        "j18": ["sweep_J1.800_g1.000", "nm_j1.8_J1.800_g1.000"],
        "j20": ["sweep_J2.000_g1.000", "nm_j2.0_J2.000_g1.000"],
    }
    found: dict[str, Path] = {}
    search_roots = [out_dir] + [p for p in out_dir.glob("cut_*") if p.is_dir()]
    for key, names in candidates.items():
        for root in search_roots:
            for name in names:
                p = root / name
                if p.exists() and p.is_dir():
                    found[key] = p
                    break
            if key in found:
                break
    return found


def plot_main_j_sweep(rows: list[dict[str, Any]], out_dir: Path, g_fixed: float) -> None:
    import matplotlib.pyplot as plt

    rows = _coerce_rows(rows)
    rows_g = sorted([r for r in rows if abs(float(r["g"]) - float(g_fixed)) <= 1e-12], key=lambda r: float(r["J"]))
    if not rows_g:
        return
    row_trivial = next((r for r in rows if str(r.get("case_name")) == "trivial"), None)
    row_markov = next((r for r in rows if str(r.get("case_name")) == "markov_ref"), None)
    sweep = [r for r in rows_g if float(r["J"]) > 1e-15]
    if not sweep:
        return

    fig, axs = plt.subplots(1, 2, figsize=(5.2, 2.45), constrained_layout=True)
    metrics = [("delta_norm", r"$\Delta_{\mathrm{norm}}$"), ("entropy", r"$S_V$")]
    panel_labels = ["(a)", "(b)"]
    j_vals = sorted({float(r["J"]) for r in rows_g})
    for ax, (metric, ylab), plab in zip(axs, metrics, panel_labels, strict=True):
        xs = [float(r["J"]) for r in sweep]
        ys = [float(r[metric]) for r in sweep]
        st = _case_style(sweep[-1])
        ax.plot(xs, ys, color=st["color"], marker=st["marker"], label="non-Markovian sweep")
        if row_trivial is not None:
            st_t = _case_style(row_trivial)
            ax.plot([0.0], [float(row_trivial[metric])], linestyle="None", marker=st_t["marker"], color=st_t["color"], label="trivial")
        if row_markov is not None:
            st_m = _case_style(row_markov)
            ax.plot([0.0], [float(row_markov[metric])], linestyle="None", marker=st_m["marker"], color=st_m["color"], label="markov_ref")
        ax.axhline(0.0, color="0.75", linewidth=0.8)
        if metric == "delta_norm":
            _safe_logscale_if_wide(ax, ys + ([float(row_trivial[metric])] if row_trivial else []) + ([float(row_markov[metric])] if row_markov else []))
        ax.set_xlabel(r"$J$")
        ax.set_ylabel(ylab)
        ax.set_xticks(j_vals)
        ax.grid(True, axis="y")
        _add_panel_label(ax, plab)
        if plab == "(a)":
            ax.legend(frameon=False, loc="best", handlelength=2.2)
    savefig_base(fig, out_dir / "fig_main_j_sweep")
    plt.close(fig)


def plot_singular_spectra(out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    reps = _find_representative_point_dirs(out_dir)
    need = ["trivial", "markov_ref", "j04", "j06", "j10", "j12", "j14", "j16", "j18", "j20"]
    if not all(k in reps for k in need):
        return

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.5), constrained_layout=True)
    mapping = [
        ("trivial", "trivial"),
        ("markov_ref", "markov_ref"),
        ("j04", "J=0.4"),
        ("j06", "J=0.6"),
        ("j10", "J=1.0"),
        ("j12", "J=1.2"),
        ("j14", "J=1.4"),
        ("j16", "J=1.6"),
        ("j18", "J=1.8"),
        ("j20", "J=2.0"),
    ]
    for key, label in mapping:
        sv_path = reps[key] / "singular_values.npy"
        if not sv_path.exists():
            continue
        s = np.load(sv_path).astype(np.float64)
        p = s * s
        denom = float(np.sum(p))
        if denom <= 0.0:
            x = np.array([0], dtype=np.int64)
            w = np.array([1e-30], dtype=np.float64)
        else:
            w = p / denom
            idx = np.arange(w.shape[0], dtype=np.int64)
            keep = w > 1e-16
            if np.any(keep):
                x = idx[keep]
                w = w[keep]
            else:
                x = np.array([0], dtype=np.int64)
                w = np.array([1e-30], dtype=np.float64)
        st = _case_style({"case_name": "sweep" if key.startswith("j") else key, "J": 0.6 if key == "j06" else (1.0 if key == "j10" else 0.0), "g": 0.0 if key == "trivial" else 1.0})
        ax.semilogy(x, w, marker=st["marker"], color=st["color"], label=label)
    ax.set_xlabel(r"singular-value index $n$")
    ax.set_ylabel(r"$p_n$")
    ax.grid(True, which="both", axis="y")
    ax.legend(frameon=False, loc="best")
    ax.set_ylim(1e-6, 1)
    savefig_base(fig, out_dir / "fig_singular_spectra")
    plt.close(fig)


def plot_cut_sweep(rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _coerce_rows(rows)
    if not rows or "cut" not in rows[0]:
        return

    def label(r: dict[str, Any]) -> str:
        name = str(r["case_name"])
        if name == "trivial":
            return "trivial"
        if name == "markov_ref":
            return "markov_ref"
        if abs(float(r["J"]) - 0.6) < 1e-12:
            return "J=0.6"
        if abs(float(r["J"]) - 1.0) < 1e-12:
            return "J=1.0"
        return ""

    keep = [r for r in rows if label(r)]
    if not keep:
        return

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.55), constrained_layout=True)
    metrics = [("entropy", r"$S_V$"), ("delta_norm", r"$\Delta_{\mathrm{norm}}$")]
    panel_labels = ["(a)", "(b)"]
    for ax, (metric, ylab), plab in zip(axs, metrics, panel_labels, strict=True):
        labels = ["trivial", "markov_ref", "J=0.6", "J=1.0"]
        all_ys: list[float] = []
        for lab in labels:
            sub = sorted([r for r in keep if label(r) == lab], key=lambda x: int(float(x["cut"])))
            if not sub:
                continue
            xs = [int(float(r["cut"])) for r in sub]
            ys = [float(r[metric]) for r in sub]
            all_ys.extend(ys)
            st = _case_style(sub[0])
            ax.plot(xs, ys, marker=st["marker"], color=st["color"], label=lab)
        ax.axhline(0.0, color="0.75", linewidth=0.8)
        if metric == "delta_norm":
            _safe_logscale_if_wide(ax, all_ys)
        ax.set_xlabel(r"cut $c$")
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y")
        _add_panel_label(ax, plab)
        if plab == "(a)":
            ax.legend(frameon=False, loc="best")
    savefig_base(fig, out_dir / "fig_cut_sweep")
    plt.close(fig)


def plot_row_distance_heatmaps(out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    reps = _find_representative_point_dirs(out_dir)
    keys = [("trivial", "trivial"), ("markov_ref", "markov_ref"), ("j10", "J=1.0")]
    if not all(k in reps for k, _ in keys):
        return
    mats = []
    for k, _lab in keys:
        p = reps[k] / "row_distances.npy"
        if not p.exists():
            return
        mats.append(np.load(p).astype(np.float64))
    vmax = max(float(np.max(m)) for m in mats)

    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35), constrained_layout=True)
    ims = []
    for ax, (k, lab), mat, plab in zip(axs, keys, mats, ["(a)", "(b)", "(c)"], strict=True):
        im = ax.imshow(mat, vmin=0.0, vmax=vmax, origin="lower", aspect="equal", interpolation="nearest")
        ims.append(im)
        ax.set_xlabel(r"past index $i$")
        ax.set_ylabel(r"past index $j$")
        ax.set_title("" if False else None)
        _add_panel_label(ax, plab)
        ax.text(0.52, 0.05, lab, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color="white")
    cbar = fig.colorbar(ims[-1], ax=axs, shrink=0.9)
    cbar.ax.set_ylabel("row distance", fontsize=9)
    savefig_base(fig, out_dir / "fig_row_distance_heatmaps")
    plt.close(fig)


def _make_plots(rows: list[dict[str, Any]], out_dir: Path, g_fixed: float) -> None:
    if not rows:
        return
    configure_matplotlib_for_paper()
    plot_main_j_sweep(rows, out_dir, g_fixed=g_fixed)
    plot_singular_spectra(out_dir)
    plot_row_distance_heatmaps(out_dir)


def _make_cut_sweep_plots(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    configure_matplotlib_for_paper()
    plot_cut_sweep(rows, out_dir)
    # Keep backward-compatible filenames requested earlier.
    import matplotlib.pyplot as plt

    rows_c = _coerce_rows(rows)
    def _label(r: dict[str, Any]) -> str:
        if str(r["case_name"]) == "trivial":
            return "trivial"
        if str(r["case_name"]) == "markov_ref":
            return "markov_ref"
        if abs(float(r["J"]) - 0.6) < 1e-12:
            return "J=0.6"
        if abs(float(r["J"]) - 1.0) < 1e-12:
            return "J=1.0"
        return ""
    data = [r for r in rows_c if _label(r)]
    if not data:
        return
    for metric, out_stem in [("entropy", "cut_sweep_entropy"), ("delta_norm", "cut_sweep_delta_norm")]:
        fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.5), constrained_layout=True)
        labels = ["trivial", "markov_ref", "J=0.6", "J=1.0"]
        ys_all: list[float] = []
        for lab in labels:
            sub = sorted([r for r in data if _label(r) == lab], key=lambda r: int(float(r["cut"])))
            if not sub:
                continue
            xs = [int(float(r["cut"])) for r in sub]
            ys = [float(r[metric]) for r in sub]
            ys_all.extend(ys)
            st = _case_style(sub[0])
            ax.plot(xs, ys, marker=st["marker"], color=st["color"], label=lab)
        if metric == "delta_norm":
            _safe_logscale_if_wide(ax, ys_all)
            ax.set_ylabel(r"$\Delta_{\mathrm{norm}}$")
        else:
            ax.set_ylabel(r"$S_V$")
        ax.set_xlabel(r"cut $c$")
        ax.grid(True, axis="y")
        ax.legend(frameon=False, loc="best")
        savefig_base(fig, out_dir / out_stem)
        plt.close(fig)


def _aggregate_cut_rows_from_subdirs(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_csv in sorted(out_dir.glob("cut_*/summary.csv")):
        with summary_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rr = dict(row)
                # Ensure cut column is present/consistent from folder name.
                try:
                    cut_dir = summary_csv.parent.name  # cut_XX
                    cut_val = int(cut_dir.split("_", 1)[1])
                    rr["cut"] = cut_val
                except Exception:
                    pass
                rows.append(rr)
    return rows


def run_benchmark_for_cut(
    *,
    cut: int,
    cases: list[tuple[str, float, float]],
    args: argparse.Namespace,
    out_dir: Path,
) -> list[dict[str, Any]]:
    cut_dir = out_dir / f"cut_{int(cut):02d}"
    cut_dir.mkdir(parents=True, exist_ok=True)

    # Per-cut shared probes with deterministic seed rule.
    seed_base = int(args.seed)
    shared_rng = np.random.default_rng(seed_base + 1000 * int(cut))
    shared_probe_set = sample_split_cut_probes(
        cut=int(cut),
        k=int(args.k),
        n_pasts=int(args.n_pasts),
        n_futures=int(args.n_futures),
        rng=shared_rng,
    )
    np.save(cut_dir / "shared_past_features.npy", shared_probe_set.past_features)
    np.save(cut_dir / "shared_future_features.npy", shared_probe_set.future_features)
    np.save(
        cut_dir / "shared_past_cut_measurement_kets.npy",
        np.stack([np.asarray(v, dtype=np.complex128) for v in shared_probe_set.past_cut_meas], axis=0),
    )
    np.save(
        cut_dir / "shared_future_cut_preparation_kets.npy",
        np.stack([np.asarray(v, dtype=np.complex128) for v in shared_probe_set.future_prep_cut], axis=0),
    )

    # Fixed initial-state ensemble for this cut only (reused for every J, g in the sweep).
    init_rng_states = np.random.default_rng(seed_base + 1000 * int(cut) + 50_000)
    shared_initial_list = _list_initial_states_sys_env0(
        L=int(args.L), n_seeds=int(args.n_seeds), rng=init_rng_states
    )
    init_stack = np.stack(shared_initial_list, axis=0)
    np.save(cut_dir / "shared_initial_states.npy", init_stack)

    # Reuse core per-point pipeline by cloning args with overridden cut.
    cut_args = argparse.Namespace(**vars(args))
    cut_args.cut = int(cut)

    rows: list[dict[str, Any]] = []
    for case_name, J, g in cases:
        rows.append(
            run_single_parameter_point(
                case_name=case_name,
                J=float(J),
                g=float(g),
                args=cut_args,
                probe_set=shared_probe_set,
                shared_initial_states=shared_initial_list,
                init_stack=init_stack,
                out_dir=cut_dir,
            )
        )

    (cut_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    _write_summary_csv(cut_dir / "summary.csv", rows)
    _make_plots(rows, cut_dir, g_fixed=float(args.g_fixed))
    return rows


def run_probe_convergence_study(*, args: argparse.Namespace, out_dir: Path) -> None:
    """Vary probe budget m = N_p = N_f; repeat independent probe draws; compare to m_ref."""
    seed_base = int(args.seed)
    cut = int(args.cut)
    m_list = _parse_int_list(str(args.convergence_m_list))
    m_ref = int(args.convergence_m_ref)
    n_draws = int(args.convergence_probe_draws)
    if n_draws < 1:
        msg = "convergence-probe-draws must be >= 1."
        raise ValueError(msg)
    if m_ref < 1:
        msg = "convergence-m-ref must be >= 1."
        raise ValueError(msg)
    if not m_list:
        msg = "convergence-m-list must contain at least one m."
        raise ValueError(msg)
    all_m = sorted({int(x) for x in m_list} | {m_ref})
    for m in all_m:
        if m < 1:
            msg = f"probe budget m must be >= 1, got {m}."
            raise ValueError(msg)

    study_dir = out_dir / "probe_convergence"
    study_dir.mkdir(parents=True, exist_ok=True)

    # Fixed single initial state |0...0> for the whole study (n_seeds=1).
    init_list = _list_initial_states_sys_env0(
        L=int(args.L), n_seeds=1, rng=np.random.default_rng(seed_base + 88_000 + cut)
    )
    init_stack = np.stack(init_list, axis=0)
    np.save(study_dir / "shared_initial_states.npy", init_stack)

    conv_cases = [
        ("trivial", 0.0, 0.0),
        ("markov_ref", 0.0, float(args.g_fixed)),
        ("nm_j0.6", 0.6, float(args.g_fixed)),
        ("nm_j1.0", 1.0, float(args.g_fixed)),
    ]

    metrics_to_track = [
        "delta",
        "delta_norm",
        "entropy",
        "max_row_distance",
        "mean_row_distance",
        "rank",
    ]

    detailed_rows: list[dict[str, Any]] = []

    for draw in range(n_draws):
        for m in all_m:
            probe_rng = np.random.default_rng(seed_base + 200_000 + int(m) * 1_000 + draw * 17_917 + cut * 3)
            probe_set = sample_split_cut_probes(
                cut=cut,
                k=int(args.k),
                n_pasts=int(m),
                n_futures=int(m),
                rng=probe_rng,
            )
            cut_args = argparse.Namespace(**vars(args))
            cut_args.cut = cut
            cut_args.n_pasts = int(m)
            cut_args.n_futures = int(m)
            cut_args.n_seeds = 1

            for case_name, J, g in conv_cases:
                summ = run_single_parameter_point(
                    case_name=case_name,
                    J=float(J),
                    g=float(g),
                    args=cut_args,
                    probe_set=probe_set,
                    shared_initial_states=init_list,
                    init_stack=init_stack,
                    out_dir=study_dir,
                    save_outputs=False,
                )
                row: dict[str, Any] = {
                    "probe_draw": int(draw),
                    "m": int(m),
                    "m_ref": int(m_ref),
                    "cut": int(cut),
                    **summ,
                }
                detailed_rows.append(row)

    summary_rows: list[dict[str, Any]] = []
    for case_name, J, g in conv_cases:
        ref_mean: dict[str, float] = {}
        for k in metrics_to_track:
            ref_vals = [
                float(r[k])
                for r in detailed_rows
                if str(r["case_name"]) == case_name and int(r["m"]) == m_ref and k in r
            ]
            ref_mean[k] = float(np.mean(ref_vals)) if ref_vals else float("nan")

        for m in all_m:
            srow: dict[str, Any] = {
                "case_name": case_name,
                "J": float(J),
                "g": float(g),
                "cut": int(cut),
                "L": int(args.L),
                "k": int(args.k),
                "m": int(m),
                "m_ref": int(m_ref),
                "n_probe_draws": int(n_draws),
            }
            for k in metrics_to_track:
                vals = [
                    float(r[k])
                    for r in detailed_rows
                    if str(r["case_name"]) == case_name and int(r["m"]) == m and k in r
                ]
                if not vals:
                    continue
                mu = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                srow[f"{k}_mean"] = mu
                srow[f"{k}_std"] = std
                rmu = ref_mean.get(k, float("nan"))
                if int(m) == int(m_ref):
                    srow[f"{k}_err_vs_ref"] = 0.0
                elif np.isfinite(rmu):
                    srow[f"{k}_err_vs_ref"] = float(abs(mu - rmu))
                else:
                    srow[f"{k}_err_vs_ref"] = float("nan")
            summary_rows.append(srow)

    _write_summary_csv(study_dir / "probe_convergence_detailed.csv", detailed_rows)
    _write_summary_csv(study_dir / "probe_convergence_summary.csv", summary_rows)
    (study_dir / "probe_convergence_detailed.json").write_text(json.dumps(detailed_rows, indent=2))
    (study_dir / "probe_convergence_summary.json").write_text(json.dumps(summary_rows, indent=2))

    print("\n=== Probe convergence study ===", flush=True)
    print(f"m values: {all_m}, m_ref={m_ref}, draws={n_draws}, cut={cut}", flush=True)
    print(f"Wrote {study_dir / 'probe_convergence_summary.csv'}", flush=True)


def main() -> None:
    args = _parse_args()
    if int(args.k) != 20:
        print(f"note: k={args.k} (requested default idea uses k=20)")
    if not (1 <= int(args.cut) <= int(args.k)):
        raise ValueError(f"cut must satisfy 1 <= cut <= k, got cut={args.cut}, k={args.k}")
    if int(args.n_seeds) < 1:
        raise ValueError("n-seeds must be >= 1.")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    j_vals = _parse_j_sweep(str(args.j_sweep))
    cut_sweep = _parse_int_list(args.cut_sweep)

    if bool(args.plot_only):
        summary_path = out_dir / "summary.json"
        if summary_path.exists():
            rows = json.loads(summary_path.read_text())
        else:
            csv_path = out_dir / "summary.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"No summary.json or summary.csv found under {out_dir}")
            rows = []
            with csv_path.open(newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    rows.append(dict(row))
        _make_plots(rows, out_dir, g_fixed=float(args.g_fixed))
        cut_sweep_csv = out_dir / "cut_sweep_summary.csv"
        if cut_sweep_csv.exists():
            rows_cs = []
            with cut_sweep_csv.open(newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    rows_cs.append(dict(row))
            _make_cut_sweep_plots(rows_cs, out_dir)
        else:
            # Fallback: build cut-sweep rows from existing cut_*/summary.csv files.
            rows_cs = _aggregate_cut_rows_from_subdirs(out_dir)
            if rows_cs:
                _write_summary_csv(out_dir / "cut_sweep_summary.csv", rows_cs)
                (out_dir / "cut_sweep_summary.json").write_text(json.dumps(rows_cs, indent=2))
                _make_cut_sweep_plots(rows_cs, out_dir)
        print(f"plot-only complete: wrote plots to {out_dir}", flush=True)
        return

    if bool(args.probe_convergence_study):
        if args.cut_sweep:
            print("note: --cut-sweep ignored when --probe-convergence-study is set.", flush=True)
        print("=== Exact V-Matrix: probe convergence study ===", flush=True)
        print(
            f"L={args.L}, k={args.k}, dt={args.dt}, cut={args.cut}, n_seeds=1 (forced)",
            flush=True,
        )
        run_probe_convergence_study(args=args, out_dir=out_dir)
        print(f"\nWrote probe convergence outputs under: {out_dir / 'probe_convergence'}", flush=True)
        return

    print("=== Exact V-Matrix Non-Markovianity Benchmark ===", flush=True)
    print(
        f"L={args.L}, k={args.k}, dt={args.dt}, cut={args.cut}, "
        f"N_p={args.n_pasts}, N_f={args.n_futures}, n_seeds={args.n_seeds}, "
        f"branch_weight_beta={float(args.branch_weight_beta):g}",
        flush=True,
    )

    all_rows: list[dict[str, Any]] = []

    # Cut selection:
    # - default: existing single-cut behavior
    # - optional: user-provided cut sweep with invalid cuts skipped
    if cut_sweep:
        valid_cuts = [c for c in cut_sweep if 1 <= int(c) <= int(args.k)]
        skipped = [c for c in cut_sweep if c not in valid_cuts]
        for c in skipped:
            print(f"note: skipping invalid cut={c} (must satisfy 1 <= cut <= k).", flush=True)
        if not valid_cuts:
            raise ValueError("No valid cuts in --cut-sweep.")
        cuts_to_run = sorted(set(valid_cuts))
        # Lighter per-cut suite requested.
        sweep_cases = [
            ("trivial", 0.0, 0.0),
            ("markov_ref", 0.0, float(args.g_fixed)),
            ("nm_j0.6", 0.6, float(args.g_fixed)),
            ("nm_j1.0", 1.0, float(args.g_fixed)),
        ]
    else:
        cuts_to_run = [int(args.cut)]
        # Preserve existing single-cut behavior.
        sweep_cases = [("trivial", 0.0, 0.0), ("markov_ref", 0.0, float(args.g_fixed))]
        for jv in j_vals:
            if abs(float(jv)) <= 1e-15:
                print("note: skipping duplicate sweep point J=0 (already included as markov_ref at g=g_fixed).", flush=True)
                continue
            sweep_cases.append(("sweep", float(jv), float(args.g_fixed)))

    for c in cuts_to_run:
        rows_cut = run_benchmark_for_cut(
            cut=int(c),
            cases=sweep_cases,
            args=args,
            out_dir=out_dir,
        )
        all_rows.extend(rows_cut)

    # Save global summaries.
    if len(cuts_to_run) == 1 and not cut_sweep:
        (out_dir / "summary.json").write_text(json.dumps(all_rows, indent=2))
        _write_summary_csv(out_dir / "summary.csv", all_rows)
        _make_plots(all_rows, out_dir, g_fixed=float(args.g_fixed))
    else:
        (out_dir / "cut_sweep_summary.json").write_text(json.dumps(all_rows, indent=2))
        _write_summary_csv(out_dir / "cut_sweep_summary.csv", all_rows)
        _make_cut_sweep_plots(all_rows, out_dir)

    # Final compact table.
    print("\n=== Summary ===", flush=True)
    header = [
        "case_name",
        "J",
        "g",
        "delta",
        "delta_norm",
        "entropy",
        "rank",
        "max_row_distance",
        "mean_row_distance",
    ]
    print(",".join(header), flush=True)
    for r in all_rows:
        row = [
            str(r["case_name"]),
            f"{float(r['J']):.6f}",
            f"{float(r['g']):.6f}",
            f"{float(r['delta']):.6e}",
            f"{float(r['delta_norm']):.6e}",
            f"{float(r['entropy']):.6e}",
            str(int(r["rank"])),
            f"{float(r['max_row_distance']):.6e}",
            f"{float(r['mean_row_distance']):.6e}",
        ]
        print(",".join(row), flush=True)

    if len(cuts_to_run) > 1:
        print("\n=== Cut-Grouped Summary ===", flush=True)
        cuts_unique = sorted({int(r["cut"]) for r in all_rows})
        for c in cuts_unique:
            print(f"cut={c}", flush=True)
            sub = [r for r in all_rows if int(r["cut"]) == int(c)]
            for r in sub:
                line = (
                    f"  case={r['case_name']}, J={float(r['J']):.3f}, g={float(r['g']):.3f}, "
                    f"delta_norm={float(r['delta_norm']):.6e}, entropy={float(r['entropy']):.6e}"
                )
                print(line, flush=True)

    print(f"\nWrote results to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
