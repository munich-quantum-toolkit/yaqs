#!/usr/bin/env python3
"""Compare operational :math:`S_V` with exhaustive :math:`S_V^{full}` and :math:`I_{PT}(c)`.

At fixed horizon ``k`` and causal cut ``c``, sweeps Ising coupling ``J`` using nested finite-basis
probe subsets and exact dense process tensors for :math:`I_{PT}`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import cast

import numpy as np

from common import (
    BETA,
    DT_DEFAULT,
    G_DEFAULT,
    L_DEFAULT,
    characterize,
    characterizer,
    configure_matplotlib_prl,
    ising_chain,
    load_csv,
    parse_float_list,
    save_figure,
    sim_params,
    write_csv,
)
from mqt.yaqs.characterization.memory.backends.exact import ExactBackend
from mqt.yaqs.characterization.memory.backends.tomography.process_tensors import (
    DenseProcessTensor,
    convert_probe_callable,
)
from mqt.yaqs.characterization.memory.operational_memory.branch_weights import compute_branch_weights
from mqt.yaqs.characterization.memory.operational_memory.full_basis import (
    build_probe_set_from_catalog,
    enumerate_full_probe_catalog,
    nested_probe_indices,
)
from mqt.yaqs.characterization.memory.operational_memory.response_matrix import (
    assemble_response_matrix,
    response_matrix_entropy,
)
from mqt.yaqs.characterization.memory.operational_memory.run import evaluate_probes_with_weights
from mqt.yaqs.characterization.memory.operational_memory.samples import ProbeSet
from mqt.yaqs.memory_characterizer import make_zero_psi

PROFILE_JS: dict[str, list[float]] = {
    "smoke": [0.0, 0.5, 1.0, 1.5, 2.0],
    "default": [0.2 * i for i in range(11)],
}
PROFILE_K: dict[str, int] = {
    "smoke": 3,
    "default": 3,
}


def center_cut(k: int) -> int:
    """Return the midpoint causal cut for ``k`` intervention legs."""
    if k < 1:
        msg = f"k must be positive, got {k}."
        raise ValueError(msg)
    return max(1, (k + 1) // 2)


def _resolve_j_values(args: argparse.Namespace) -> list[float]:
    if args.j_values:
        return parse_float_list(args.j_values)
    return list(PROFILE_JS[str(args.profile)])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=("smoke", "default"), default="default")
    p.add_argument("--k", type=int, default=None, help="Horizon (num_interventions).")
    p.add_argument("--cut", type=int, default=None, help="Causal cut (default k/2).")
    p.add_argument("--j-values", type=str, default=None, help="Comma-separated J sweep.")
    p.add_argument("--n-pasts", type=int, default=8)
    p.add_argument("--n-futures", type=int, default=8)
    p.add_argument("--n-pasts-fine", type=int, default=64, help="Probe grid for high-budget S_V.")
    p.add_argument("--n-futures-fine", type=int, default=64)
    p.add_argument("--pt-only", action="store_true", help="Compute S_PT_cb only (skip characterize).")
    p.add_argument("--sv-only", action="store_true", help="Compute S_V only (skip PT build).")
    p.add_argument("--refine-sv", action="store_true", help="Legacy: run high-sample S_V overlay only.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("save/pt_cut_reference"))
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--plot-ratio-only", action="store_true", help="Plot S_V/S_PT analysis from saved CSVs.")
    p.add_argument("--plot-prx-only", action="store_true", help="Plot PRX two-panel S vs J figure only.")
    p.add_argument("--summary-csv", type=Path, default=None)
    return p.parse_args()


def run_sv_sweep(
    args: argparse.Namespace,
    *,
    j_values: list[float],
    k: int,
    cut: int,
    n_pasts: int,
    n_futures: int,
) -> list[dict[str, float | int]]:
    """Sweep J and record operational S_V only."""
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mc = characterizer(parallel=False)
    params = sim_params(dt=DT_DEFAULT)
    psi0 = make_zero_psi(L_DEFAULT)

    rows: list[dict[str, float | int]] = []
    t0 = time.perf_counter()
    for jv in j_values:
        t_j = time.perf_counter()
        print(f"S_V J={jv:g}  (c={cut}, k={k}, m={n_pasts})", flush=True)
        ham = ising_chain(length=L_DEFAULT, j=float(jv), g=G_DEFAULT)
        sv_result = mc.characterize(
            ham,
            params,
            num_interventions=k,
            cut=cut,
            n_pasts=int(n_pasts),
            n_futures=int(n_futures),
            initial_psi=psi0,
            rng=np.random.default_rng(int(args.seed) + int(round(1000 * jv))),
            intervention_style="haar",
        )
        rows.append(
            {
                "J": float(jv),
                "cut": int(cut),
                "k": int(k),
                "L": int(L_DEFAULT),
                "entropy_SV": float(sv_result.entropy()),
                "n_pasts": int(n_pasts),
                "n_futures": int(n_futures),
            }
        )
        print(f"  done in {time.perf_counter() - t_j:.1f}s", flush=True)
    print(f"S_V sweep total: {time.perf_counter() - t0:.1f}s", flush=True)
    return rows


def _causal_break_error(pt: DenseProcessTensor, k: int, cut: int, *, trials: int = 30) -> float:
    """Max normalized trace distance under past-intervention swaps at the causal break."""
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


def _sv_metrics(probe_set: ProbeSet, backend: DenseProcessTensor | ExactBackend) -> tuple[float, float]:
    """Return ``(S_V, ||V||_F)`` for a probe set evaluated on ``backend``."""
    pauli, weights = evaluate_probes_with_weights(backend, probe_set)
    _raw, response = assemble_response_matrix(pauli, weights, beta=BETA, center=True)
    entropy = response_matrix_entropy(response)
    norm = float(np.linalg.norm(response, ord="fro"))
    return entropy, norm


def _save_j0_ipt_debug(
    pt: DenseProcessTensor,
    mi: dict[str, object],
    *,
    out_dir: Path,
) -> None:
    """Persist diagnostics when J=0 mutual-information sanity fails."""
    debug_path = out_dir / "j0_ipt_debug.npz"
    np.savez(
        debug_path,
        upsilon=pt.to_matrix(),
        blocks=np.array(mi["blocks"], dtype=object),
        block_labels=np.array(mi["block_labels"], dtype=object),
        p_axes=np.asarray(mi["p_axes"], dtype=np.int64),
        f_axes=np.asarray(mi["f_axes"], dtype=np.int64),
    )
    print(f"J=0 validation failed — wrote {debug_path}", flush=True)


def run_full_benchmark(args: argparse.Namespace) -> list[dict[str, float | int]]:
    """Sweep J recording ``S_V^full``, nested ``S_V``, and ``I_PT``."""
    k = int(args.k) if args.k is not None else PROFILE_K[str(args.profile)]
    cut = int(args.cut) if args.cut is not None else center_cut(k)
    if cut < 1 or cut > k:
        msg = f"cut must satisfy 1 <= cut <= k ({k}), got {cut}."
        raise ValueError(msg)

    j_values = _resolve_j_values(args)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = enumerate_full_probe_catalog(cut=cut, num_interventions=k)
    n_p_full = len(catalog.past_settings)
    n_f_full = len(catalog.future_settings)
    past_all = np.arange(n_p_full, dtype=np.int64)
    future_all = np.arange(n_f_full, dtype=np.int64)
    past_8, future_8 = nested_probe_indices(catalog, int(args.n_pasts), seed=int(args.seed))
    past_64, future_64 = nested_probe_indices(catalog, int(args.n_pasts_fine), seed=int(args.seed))
    probe_full = build_probe_set_from_catalog(catalog, past_all, future_all)
    probe_8 = build_probe_set_from_catalog(catalog, past_8, future_8)
    probe_64 = build_probe_set_from_catalog(catalog, past_64, future_64)

    catalog_meta = {
        "len_P_full": n_p_full,
        "len_F_full": n_f_full,
        "n_response_entries": n_p_full * n_f_full,
        "n_obs_channels": 3,
        "intervention_tomographically_complete": catalog.intervention_tomographically_complete,
        "output_informationally_complete": catalog.output_informationally_complete,
        "n_pasts_8": len(past_8),
        "n_futures_8": len(future_8),
        "n_pasts_64": len(past_64),
        "n_futures_64": len(future_64),
    }
    with (out_dir / "full_basis_catalog.json").open("w", encoding="utf-8") as f:
        json.dump(catalog_meta, f, indent=2)
    print(
        f"P_full={n_p_full}, F_full={n_f_full}, entries={n_p_full * n_f_full}, "
        f"nested budgets={len(past_8)}×{len(future_8)} and {len(past_64)}×{len(future_64)}",
        flush=True,
    )

    mc = characterizer(parallel=False)
    params = sim_params(dt=DT_DEFAULT)
    timesteps = [DT_DEFAULT] * (k + 1)
    psi0 = make_zero_psi(L_DEFAULT)

    rows: list[dict[str, float | int]] = []
    j0_failed = False
    cross_checks: list[dict[str, float]] = []
    t0 = time.perf_counter()
    for jv in j_values:
        t_j = time.perf_counter()
        print(f"J={jv:g}  (c={cut}, k={k})", flush=True)
        ham = ising_chain(length=L_DEFAULT, j=float(jv), g=G_DEFAULT)
        ham.ensure_encoded("mpo")
        row: dict[str, float | int] = {"J": float(jv), "cut": int(cut), "k": int(k), "L": int(L_DEFAULT)}

        t_pt = time.perf_counter()
        pt = cast(
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
        mi = pt.causal_block_mutual_information(cut)
        ups_trace = float(np.trace(pt.to_matrix()).real)
        row["I_PT"] = float(cast("float", mi["mutual_information"]))
        row["S_rho_P"] = float(cast("float", mi["entropy_p"]))
        row["S_rho_F"] = float(cast("float", mi["entropy_f"]))
        row["S_rho_PF"] = float(cast("float", mi["entropy_pf"]))
        row["process_tensor_trace"] = ups_trace
        row["process_tensor_min_eigenvalue"] = float(cast("float", mi["min_eigenvalue"]))
        row["causal_break_error"] = float(_causal_break_error(pt, k, cut))
        print(f"  PT {time.perf_counter() - t_pt:.1f}s  I_PT={row['I_PT']:.3e}", flush=True)

        t_sv = time.perf_counter()
        sv_full, norm_full = _sv_metrics(probe_full, pt)
        sv_8, norm_8 = _sv_metrics(probe_8, pt)
        sv_64, norm_64 = _sv_metrics(probe_64, pt)
        row["S_V_full"] = sv_full
        row["S_V_8"] = sv_8
        row["S_V_64"] = sv_64
        row["response_norm_full"] = norm_full
        row["response_norm_8"] = norm_8
        row["response_norm_64"] = norm_64
        print(
            f"  S_V {time.perf_counter() - t_sv:.1f}s  "
            f"full={sv_full:.4f} 8={sv_8:.4f} 64={sv_64:.4f}",
            flush=True,
        )

        if abs(jv) < 1e-12 or abs(jv - 1.0) < 1e-12:
            backend = ExactBackend(
                operator=ham.mpo,
                sim_params=params,
                initial_psi=psi0,
                parallel=False,
                show_progress=False,
            )
            pauli_pt, w_pt = evaluate_probes_with_weights(pt, probe_full)
            pauli_dir, w_dir = evaluate_probes_with_weights(backend, probe_full)
            w_analytic = compute_branch_weights(probe_full)
            pauli_rel = float(
                np.linalg.norm((pauli_pt - pauli_dir).ravel(), ord=2)
                / max(np.linalg.norm(pauli_dir.ravel(), ord=2), 1e-30)
            )
            _, V_pt = assemble_response_matrix(pauli_pt, w_analytic, beta=BETA, center=True)
            _, V_dir = assemble_response_matrix(pauli_dir, w_analytic, beta=BETA, center=True)
            v_rel = float(np.linalg.norm(V_pt - V_dir, ord="fro") / max(np.linalg.norm(V_dir, ord="fro"), 1e-30))
            cross_checks.append(
                {
                    "J": float(jv),
                    "pauli_rel_fro_error": pauli_rel,
                    "response_rel_fro_error_analytic_weights": v_rel,
                    "weight_max_abs_diff_sim_vs_analytic": float(np.max(np.abs(w_dir - w_analytic))),
                }
            )
            print(
                f"  cross-check pauli_rel={pauli_rel:.3e} V_rel(analytic w)={v_rel:.3e}",
                flush=True,
            )

        if abs(jv) < 1e-12 and (row["S_V_full"] >= 1e-10 or row["I_PT"] >= 1e-10):
            j0_failed = True
            _save_j0_ipt_debug(pt, mi, out_dir=out_dir)

        rows.append(row)
        print(f"  total {time.perf_counter() - t_j:.1f}s", flush=True)

    print(f"Benchmark total: {time.perf_counter() - t0:.1f}s", flush=True)
    if j0_failed:
        print("ABORT: J=0 S_V_full or I_PT sanity check failed.", flush=True)
    if cross_checks:
        with (out_dir / "response_cross_check.json").open("w", encoding="utf-8") as f:
            json.dump(cross_checks, f, indent=2)

    write_csv(out_dir / "summary_sv_full.csv", rows)
    np.savez(out_dir / "summary_sv_full.npz", **{key: np.asarray([r[key] for r in rows]) for key in rows[0]})
    with (out_dir / "summary_sv_full.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return rows


def run_benchmark(args: argparse.Namespace) -> list[dict[str, float | int]]:
    """Sweep J at the center cut and record S_V and S_PT."""
    k = int(args.k) if args.k is not None else PROFILE_K[str(args.profile)]
    cut = int(args.cut) if args.cut is not None else center_cut(k)
    if cut < 1 or cut > k:
        msg = f"cut must satisfy 1 <= cut <= k ({k}), got {cut}."
        raise ValueError(msg)

    if args.j_values:
        j_values = parse_float_list(args.j_values)
    elif args.profile == "default" and args.j_values is None:
        j_values = list(PROFILE_JS["default"])
    else:
        j_values = list(PROFILE_JS[str(args.profile)])

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mc = characterizer(parallel=False)
    params = sim_params(dt=DT_DEFAULT)
    timesteps = [DT_DEFAULT] * (k + 1)
    psi0 = make_zero_psi(L_DEFAULT)

    rows: list[dict[str, float | int]] = []
    t0 = time.perf_counter()
    for jv in j_values:
        t_j = time.perf_counter()
        print(f"J={jv:g}  (c={cut}, k={k})", flush=True)
        ham = ising_chain(length=L_DEFAULT, j=float(jv), g=G_DEFAULT)
        row: dict[str, float | int] = {
            "J": float(jv),
            "cut": int(cut),
            "k": int(k),
            "L": int(L_DEFAULT),
            "n_pasts": int(args.n_pasts),
            "n_futures": int(args.n_futures),
        }
        if not args.sv_only:
            t_pt = time.perf_counter()
            pt = mc.build_process_tensor(
                ham,
                params,
                timesteps=timesteps,
                return_type="mpo",
                method="direct",
                max_bond_dim=int(args.max_bond_dim),
                compress_every=16,
            )
            row["s_pt"] = float(pt.cut_entanglement_entropy(cut))
            row["chi"] = int(pt.temporal_bond_dimension(cut))
            print(f"  PT {time.perf_counter() - t_pt:.1f}s", flush=True)
        if not args.pt_only:
            t_sv = time.perf_counter()
            sv_result = mc.characterize(
                ham,
                params,
                num_interventions=k,
                cut=cut,
                n_pasts=int(args.n_pasts),
                n_futures=int(args.n_futures),
                initial_psi=psi0,
                rng=np.random.default_rng(int(args.seed) + int(round(1000 * jv))),
                intervention_style="haar",
            )
            row["entropy_SV"] = float(sv_result.entropy())
            print(f"  S_V {time.perf_counter() - t_sv:.1f}s", flush=True)
        rows.append(row)
        print(f"  total {time.perf_counter() - t_j:.1f}s", flush=True)

    print(f"Benchmark total: {time.perf_counter() - t0:.1f}s", flush=True)

    write_csv(out_dir / "summary.csv", rows)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return rows


def _positive_floor(values: np.ndarray, *, floor: float = 1e-6) -> np.ndarray:
    """Clip values for log-scale plotting."""
    return np.maximum(np.asarray(values, dtype=np.float64), floor)


def _dynamic_range_decades(values: np.ndarray, *, floor: float = 1e-6) -> float:
    """Return base-10 dynamic range of strictly positive entries."""
    pos = _positive_floor(values, floor=floor)
    pos = pos[pos > floor]
    if pos.size < 2:
        return 0.0
    return float(np.log10(pos.max()) - np.log10(pos.min()))


def _outlier_indices(values: np.ndarray, *, factor: float = 5.0) -> list[int]:
    """Flag points that exceed ``factor`` times adjacent neighbors (endpoint spike test)."""
    v = np.asarray(values, dtype=np.float64)
    n = v.size
    if n < 2:
        return []
    out: list[int] = []
    if v[0] > factor * max(v[1], 1e-30):
        out.append(0)
    for i in range(1, n - 1):
        local = max(v[i - 1], v[i + 1], 1e-30)
        if v[i] > factor * local:
            out.append(i)
    if v[-1] > factor * max(v[-2], 1e-30):
        out.append(n - 1)
    return out


def plot_results(
    rows: list[dict[str, float | int]],
    out_dir: Path,
    *,
    sv_fine_rows: list[dict[str, float | int]] | None = None,
) -> None:
    """Single-panel figure: S_PT and S_V vs J at the center cut."""
    import matplotlib.pyplot as plt

    if not rows:
        return

    configure_matplotlib_prl()
    rows = sorted(rows, key=lambda r: float(r["J"]))
    k = int(rows[0]["k"])
    cut = int(rows[0]["cut"])
    j_vals = np.asarray([float(r["J"]) for r in rows], dtype=np.float64)
    s_pt = np.asarray([float(r["s_pt"]) for r in rows], dtype=np.float64) if "s_pt" in rows[0] else None
    s_v = (
        np.asarray([max(float(r["entropy_SV"]), 0.0) for r in rows], dtype=np.float64)
        if "entropy_SV" in rows[0]
        else None
    )

    log_floor = 1e-6
    pt_outliers: list[int] = []
    if s_pt is not None:
        pt_outliers = _outlier_indices(s_pt)
        if pt_outliers:
            flagged = ", ".join(f"J={j_vals[i]:g} (S_PT={s_pt[i]:.3g})" for i in pt_outliers)
            print(f"S_PT outliers (>5× adjacent neighbors): {flagged}", flush=True)

    sv_check = s_v
    if sv_fine_rows:
        sv_check = np.asarray(
            [max(float(r["entropy_SV"]), 0.0) for r in sv_fine_rows],
            dtype=np.float64,
        )
    sv_outliers: list[int] = []
    if sv_check is not None:
        sv_outliers = _outlier_indices(sv_check)
        if sv_outliers and sv_fine_rows:
            j_f = np.asarray([float(r["J"]) for r in sv_fine_rows], dtype=np.float64)
            flagged = ", ".join(f"J={j_f[i]:g} (S_V={sv_check[i]:.3g})" for i in sv_outliers)
            print(f"S_V outliers (>5× adjacent neighbors): {flagged}", flush=True)
        elif sv_outliers and s_v is not None:
            flagged = ", ".join(f"J={j_vals[i]:g} (S_V={s_v[i]:.3g})" for i in sv_outliers)
            print(f"S_V outliers (>5× adjacent neighbors): {flagged}", flush=True)

    use_log_pt = s_pt is not None and _dynamic_range_decades(s_pt, floor=log_floor) > 1.0
    use_log_sv = sv_check is not None and _dynamic_range_decades(sv_check, floor=log_floor) > 1.0
    pt_range = _dynamic_range_decades(s_pt, floor=log_floor) if s_pt is not None else 0.0
    sv_range = _dynamic_range_decades(sv_check, floor=log_floor) if sv_check is not None else 0.0
    print(
        f"Scales: S_PT={'log' if use_log_pt else 'linear'} (range {pt_range:.1f} decades), "
        f"S_V={'log' if use_log_sv else 'linear' if sv_check is not None else 'n/a'} "
        f"(range {sv_range:.1f} decades)",
        flush=True,
    )

    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    color_pt = "#2166ac"
    color_sv = "#b2182b"
    if s_pt is not None:
        pt_plot = _positive_floor(s_pt, floor=log_floor) if use_log_pt else s_pt
        ax.plot(j_vals, pt_plot, "o-", color=color_pt, lw=1.8, ms=4.0, label=r"$S_{\mathrm{PT}}$")
        for i in pt_outliers:
            ax.plot(
                j_vals[i],
                pt_plot[i],
                "o",
                mfc="none",
                mec=color_pt,
                mew=1.4,
                ms=7.5,
                zorder=4,
                label="_nolegend_",
            )
        ax.set_ylabel(r"$S_{\mathrm{PT}}(c)$", color=color_pt)
        ax.tick_params(axis="y", labelcolor=color_pt)
        if use_log_pt:
            ax.set_yscale("log")

    ax.set_xlabel(r"Coupling $J$")
    ax.set_xlim(float(j_vals.min()), float(j_vals.max()))

    has_sv = sv_check is not None
    ax2 = ax.twinx() if has_sv else None
    if has_sv and s_v is not None:
        sv_plot = _positive_floor(s_v, floor=log_floor) if use_log_sv else s_v
        n_coarse = int(rows[0].get("n_pasts", 8))
        if sv_fine_rows:
            ax2.plot(
                j_vals,
                sv_plot,
                "s-",
                color=color_sv,
                lw=1.0,
                ms=2.5,
                alpha=0.35,
                label=rf"$S_V$ ({n_coarse}$\times${n_coarse})",
            )
            fine = sorted(sv_fine_rows, key=lambda r: float(r["J"]))
            j_fine = np.asarray([float(r["J"]) for r in fine], dtype=np.float64)
            s_v_fine = np.asarray([max(float(r["entropy_SV"]), 0.0) for r in fine], dtype=np.float64)
            sv_fine_plot = _positive_floor(s_v_fine, floor=log_floor) if use_log_sv else s_v_fine
            m_f = int(fine[0].get("n_pasts", 32))
            ax2.plot(
                j_fine,
                sv_fine_plot,
                "-",
                color=color_sv,
                lw=2.0,
                label=rf"$S_V$ ({m_f}$\times${m_f})",
            )
        else:
            ax2.plot(j_vals, sv_plot, "s-", color=color_sv, lw=1.8, ms=3.5, label=r"$S_V$")
            for i in sv_outliers:
                ax2.plot(
                    j_vals[i],
                    sv_plot[i],
                    "s",
                    mfc="none",
                    mec=color_sv,
                    mew=1.4,
                    ms=7.0,
                    zorder=4,
                    label="_nolegend_",
                )
        ax2.set_ylabel(rf"$S_V(c)$", color=color_sv)
        ax2.tick_params(axis="y", labelcolor=color_sv)
        if use_log_sv:
            ax2.set_yscale("log")

    legend_axes = ax.get_lines() + (ax2.get_lines() if ax2 is not None else [])
    lines = [line for line in legend_axes if not line.get_label().startswith("_")]
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, frameon=False, loc="upper left", fontsize=7.5)

    ax.text(
        0.98,
        0.04,
        rf"$c={cut}$, $k={k}$, $L={int(rows[0]['L'])}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.35",
    )

    for spine in ax.spines.values():
        spine.set_linewidth(0.85)
    ax.tick_params(direction="in", top=True, length=3.5, width=0.75)
    if ax2 is not None:
        ax2.tick_params(direction="in", right=True, length=3.5, width=0.75)

    save_figure(fig, out_dir / "fig_sv_spt_vs_j")


def _merge_sv_pt(
    rows: list[dict[str, float | int]],
    sv_fine_rows: list[dict[str, float | int]] | None,
) -> list[dict[str, float | int]]:
    """Merge S_PT from ``rows`` with S_V (prefer fine grid when available)."""
    if "s_pt" not in rows[0]:
        return []
    sv_by_j: dict[float, float] = {}
    if sv_fine_rows:
        for r in sv_fine_rows:
            sv_by_j[float(r["J"])] = max(float(r["entropy_SV"]), 0.0)
    elif "entropy_SV" in rows[0]:
        for r in rows:
            sv_by_j[float(r["J"])] = max(float(r["entropy_SV"]), 0.0)
    merged: list[dict[str, float | int]] = []
    for r in rows:
        jv = float(r["J"])
        if jv not in sv_by_j:
            continue
        s_pt = float(r["s_pt"])
        s_v = sv_by_j[jv]
        ratio = float(s_v / s_pt) if s_pt > 1e-15 else float("nan")
        merged.append(
            {
                "J": jv,
                "cut": int(r["cut"]),
                "k": int(r["k"]),
                "L": int(r["L"]),
                "entropy_SV": s_v,
                "s_pt": s_pt,
                "ratio_SV_over_SPT": ratio,
                "chi": int(r["chi"]) if "chi" in r else -1,
            }
        )
    return merged


def plot_ratio_analysis(
    rows: list[dict[str, float | int]],
    out_dir: Path,
    *,
    sv_fine_rows: list[dict[str, float | int]] | None = None,
) -> None:
    """Plot :math:`S_V/S_{\\mathrm{PT}}` vs ``J`` and ``S_V`` vs ``S_{\\mathrm{PT}}``."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    merged = _merge_sv_pt(rows, sv_fine_rows)
    if not merged:
        print("Skipping ratio plot: need both S_PT and S_V.", flush=True)
        return

    merged = sorted(merged, key=lambda r: float(r["J"]))
    write_csv(out_dir / "summary_ratio.csv", merged)

    configure_matplotlib_prl()
    j_all = np.asarray([float(r["J"]) for r in merged], dtype=np.float64)
    s_pt = np.asarray([float(r["s_pt"]) for r in merged], dtype=np.float64)
    s_v = np.asarray([float(r["entropy_SV"]) for r in merged], dtype=np.float64)
    ratio = np.asarray([float(r["ratio_SV_over_SPT"]) for r in merged], dtype=np.float64)

    # Exclude J=0: S_PT spike from Markovian Choi structure, not comparable to S_V scale.
    mask = j_all > 1e-12
    j_pos = j_all[mask]
    s_pt_pos = s_pt[mask]
    s_v_pos = s_v[mask]
    ratio_pos = ratio[mask]

    if j_pos.size >= 2:
        corr_j = float(np.corrcoef(j_pos, ratio_pos)[0, 1])
        slope, intercept = np.polyfit(j_pos, ratio_pos, 1)
        r2 = float(np.corrcoef(s_v_pos, s_pt_pos)[0, 1] ** 2)
        print(
            f"J>0: corr(J, S_V/S_PT)={corr_j:.3f}, "
            f"linear fit ratio ≈ {slope:.3f}·J + {intercept:.3f}, "
            f"R²(S_V vs S_PT)={r2:.3f}",
            flush=True,
        )

    k = int(merged[0]["k"])
    cut = int(merged[0]["cut"])
    fig, (ax_r, ax_sc) = plt.subplots(1, 2, figsize=(6.8, 2.6), constrained_layout=True)
    fig.patch.set_facecolor("white")

    # Panel (a): ratio vs J
    ax_r.set_facecolor("white")
    ax_r.plot(j_pos, ratio_pos, "o-", color="#542788", lw=1.6, ms=4.0)
    if j_pos.size >= 2:
        j_fit = np.linspace(float(j_pos.min()), float(j_pos.max()), 100)
        ax_r.plot(j_fit, slope * j_fit + intercept, "--", color="0.45", lw=1.2, label="linear fit")
    if np.any(j_all <= 1e-12):
        j0 = float(j_all[j_all <= 1e-12][0])
        r0 = float(ratio[j_all <= 1e-12][0])
        ax_r.plot(j0, max(r0, 1e-6), "o", mfc="none", mec="#542788", ms=7, mew=1.2)
        ax_r.annotate(
            r"$J{=}0$: excluded from fit",
            xy=(j0, max(r0, 1e-6)),
            xytext=(0.15, 0.15),
            textcoords="data",
            fontsize=7,
            color="0.35",
            arrowprops={"arrowstyle": "->", "lw": 0.6, "color": "0.45"},
        )
    ax_r.set_xlabel(r"Coupling $J$")
    ax_r.set_ylabel(r"$S_V / S_{\mathrm{PT}}$")
    ax_r.set_title(r"(a) Operational fraction", fontsize=9)
    if j_pos.size >= 2:
        ax_r.legend(frameon=False, fontsize=7, loc="upper left")

    # Panel (b): S_V vs S_PT, color = J
    ax_sc.set_facecolor("white")
    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=float(j_pos.min()) if j_pos.size else 0.0, vmax=float(j_pos.max()) if j_pos.size else 2.0)
    sc = ax_sc.scatter(s_pt_pos, s_v_pos, c=j_pos, cmap=cmap, norm=norm, s=28, edgecolors="0.2", linewidths=0.35)
    if s_pt_pos.size >= 2:
        pt_fit = np.linspace(float(s_pt_pos.min()), float(s_pt_pos.max()), 100)
        # Fit S_V = alpha * S_PT through origin-ish or affine
        alpha = float(np.dot(s_v_pos, s_pt_pos) / np.dot(s_pt_pos, s_pt_pos))
        ax_sc.plot(pt_fit, alpha * pt_fit, "--", color="0.45", lw=1.2, label=rf"$S_V \approx {alpha:.2f}\,S_{{\mathrm{{PT}}}}$")
    ax_sc.set_xlabel(r"$S_{\mathrm{PT}}(c)$")
    ax_sc.set_ylabel(r"$S_V(c)$")
    ax_sc.set_title(r"(b) Correlation", fontsize=9)
    cbar = fig.colorbar(sc, ax=ax_sc, shrink=0.92, pad=0.02)
    cbar.set_label(r"$J$", fontsize=8)
    if s_pt_pos.size >= 2:
        ax_sc.legend(frameon=False, fontsize=7, loc="upper left")
    if np.nanmax(s_v_pos) > 10 * np.nanmin(s_v_pos[s_v_pos > 0]) if np.any(s_v_pos > 0) else False:
        ax_sc.set_yscale("log")
        ax_sc.set_xscale("log")

    fig.suptitle(rf"$c={cut}$, $k={k}$, $L={int(merged[0]['L'])}$", fontsize=10, y=1.02)
    for ax in (ax_r, ax_sc):
        for spine in ax.spines.values():
            spine.set_linewidth(0.85)
        ax.tick_params(direction="in", top=True, right=True, length=3.5, width=0.75)

    save_figure(fig, out_dir / "fig_sv_spt_ratio")


def _parse_summary_row(row: dict[str, str]) -> dict[str, float | int]:
    """Parse a CSV row, skipping empty optional fields."""
    out: dict[str, float | int] = {}
    int_keys = {"cut", "k", "L", "chi", "n_pasts", "n_futures", "causal_block_rank"}
    for key, val in row.items():
        if not str(val).strip():
            continue
        out[key] = int(val) if key in int_keys else float(val)
    return out


def plot_figure_prx_sv_ipt(
    rows: list[dict[str, float | int]],
    out_dir: Path,
    *,
    allow_plot: bool = True,
    n_past_8: int = 8,
    n_past_64: int = 64,
) -> None:
    """Two-panel figure: entropies vs ``J`` and ``S_V^full`` vs ``I_PT``."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import LogFormatterMathtext
    from scipy import stats

    if not allow_plot:
        print("Skipping PRX figure: J=0 sanity check failed.", flush=True)
        return
    if not rows or "S_V_full" not in rows[0] or "I_PT" not in rows[0]:
        print("Skipping PRX figure: need S_V_full and I_PT.", flush=True)
        return

    rows = sorted(rows, key=lambda r: float(r["J"]))
    k = int(rows[0]["k"])
    cut = int(rows[0]["cut"])
    j_vals = np.asarray([float(r["J"]) for r in rows], dtype=np.float64)
    s_full = np.asarray([max(float(r["S_V_full"]), 0.0) for r in rows], dtype=np.float64)
    s_v8 = np.asarray([max(float(r["S_V_8"]), 0.0) for r in rows], dtype=np.float64)
    s_v64 = np.asarray([max(float(r["S_V_64"]), 0.0) for r in rows], dtype=np.float64)
    i_pt = np.asarray([max(float(r["I_PT"]), 0.0) for r in rows], dtype=np.float64)

    log_floor = 1e-12
    all_vals = np.concatenate([s_full, s_v8, s_v64, i_pt])
    pos = all_vals[all_vals > log_floor]
    use_log = pos.size >= 2 and float(np.log10(pos.max()) - np.log10(pos.min())) > 1.0

    configure_matplotlib_prl()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 2.75), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.30})
    fig.patch.set_facecolor("white")
    color_sv = "#8B0000"
    color_ipt = "#0072B2"

    def _floor(y: np.ndarray) -> np.ndarray:
        return np.maximum(y, log_floor) if use_log else y

    ax_a.set_facecolor("white")
    ax_a.plot(j_vals, _floor(s_full), "o-", color=color_sv, lw=2.0, ms=5.0, label=r"$S_V^{\mathrm{full}}$")
    ax_a.plot(
        j_vals,
        _floor(s_v64),
        "^-",
        color="#CC3311",
        lw=1.8,
        ms=4.2,
        label=rf"$S_V$ (${n_past_64}$\times${n_past_64}$)",
    )
    ax_a.plot(
        j_vals,
        _floor(s_v8),
        "s-",
        color="#CC3311",
        lw=1.2,
        ms=4.0,
        mfc="none",
        alpha=0.55,
        label=rf"$S_V$ (${n_past_8}$\times${n_past_8}$)",
    )
    ax_a.plot(j_vals, _floor(i_pt), "o-", color=color_ipt, lw=1.8, ms=4.5, alpha=0.9, label=r"$I_{\mathrm{PT}}$")
    ax_a.set_xlabel(r"Coupling $J$")
    ax_a.set_ylabel("Entropy")
    if use_log:
        ax_a.set_yscale("log")
        ax_a.yaxis.set_major_formatter(LogFormatterMathtext())
    ax_a.set_xlim(-0.05, float(j_vals.max()) + 0.05)
    ax_a.legend(frameon=False, loc="lower right", fontsize=7.5)
    ax_a.text(0.03, 0.97, "(a)", transform=ax_a.transAxes, fontsize=12, fontweight="bold", va="top")
    ax_a.text(
        0.97, 0.97, rf"$c={cut}$, $k={k}$, $L={int(rows[0]['L'])}$",
        transform=ax_a.transAxes, ha="right", va="top", fontsize=8, color="0.4",
    )

    ax_b.set_facecolor("white")
    fit_mask = (j_vals >= 0) & (i_pt > log_floor) & (s_full > log_floor)
    j_pos = j_vals[fit_mask]
    ipt_fit = i_pt[fit_mask]
    sv_fit = s_full[fit_mask]
    if j_pos.size:
        cmap = plt.get_cmap("plasma")
        cnorm = Normalize(vmin=float(j_pos.min()), vmax=float(j_pos.max()))
        if use_log:
            ax_b.set_xscale("log")
            ax_b.set_yscale("log")
        sc = ax_b.scatter(
            _floor(ipt_fit), _floor(sv_fit), c=j_pos, cmap=cmap, norm=cnorm, s=36,
            edgecolors="0.15", linewidths=0.4, zorder=3,
        )
        if ipt_fit.size >= 2:
            mask_pos = (ipt_fit > log_floor) & (sv_fit > log_floor)
            if np.count_nonzero(mask_pos) >= 2:
                log_i = np.log10(np.maximum(ipt_fit[mask_pos], log_floor))
                log_s = np.log10(np.maximum(sv_fit[mask_pos], log_floor))
                slope, intercept = np.polyfit(log_i, log_s, 1)
                x_line = np.geomspace(max(float(ipt_fit[mask_pos].min()), log_floor), float(ipt_fit[mask_pos].max()), 100)
                y_line = np.power(10.0, intercept + slope * np.log10(x_line))
                ax_b.plot(x_line, y_line, "--", color="0.35", lw=1.2, label="empirical guide to the eye", zorder=2)
                pearson = float(stats.pearsonr(log_i, log_s).statistic)
                spearman = float(stats.spearmanr(ipt_fit[mask_pos], sv_fit[mask_pos]).statistic)
                print(f"Panel (b): Pearson(log)={pearson:.4f}, Spearman={spearman:.4f}", flush=True)
                with (out_dir / "panel_b_ipt_fit.json").open("w", encoding="utf-8") as f:
                    json.dump({"pearson_log": pearson, "spearman": spearman}, f, indent=2)
            ax_b.legend(frameon=False, loc="lower right", fontsize=8.0)
        if use_log:
            ax_b.xaxis.set_major_formatter(LogFormatterMathtext())
            ax_b.yaxis.set_major_formatter(LogFormatterMathtext())
        cbar = fig.colorbar(sc, ax=ax_b, shrink=0.92, pad=0.02, aspect=18)
        cbar.ax.set_title(r"$J$", fontsize=9, pad=3)
    ax_b.set_xlabel(r"$I_{\mathrm{PT}}$")
    ax_b.set_ylabel(r"$S_V^{\mathrm{full}}$")
    ax_b.text(0.03, 0.97, "(b)", transform=ax_b.transAxes, fontsize=12, fontweight="bold", va="top")

    for ax in (ax_a, ax_b):
        for spine in ax.spines.values():
            spine.set_linewidth(0.85)
        ax.tick_params(direction="in", top=True, right=True, length=4.0, width=0.75)

    out_stem = out_dir / "fig_sv_spt_prx"
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)


def plot_convergence_figure(
    curves: list[dict[str, float | int | list[float] | list[int]]],
    out_dir: Path,
) -> None:
    """Supplementary figure: ``|S_V(m) - S_V^full|`` vs probe budget ``m``."""
    import matplotlib.pyplot as plt

    if not curves:
        return
    configure_matplotlib_prl()
    fig, ax = plt.subplots(figsize=(3.6, 2.8), constrained_layout=True)
    for entry in curves:
        m = np.asarray(entry["budgets"], dtype=np.int64)
        err = np.asarray(entry["abs_errors"], dtype=np.float64)
        ax.plot(m, np.maximum(err, 1e-16), "o-", ms=4, lw=1.4, label=f"J={float(entry['J']):g}")
    ax.set_xlabel(r"Probe budget $m$")
    ax.set_ylabel(r"$|S_V(m) - S_V^{\mathrm{full}}|$")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.legend(frameon=False, fontsize=8)
    out_stem = out_dir / "fig_sv_convergence"
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    fig.savefig(out_stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02, dpi=600)
    plt.close(fig)


def run_convergence_curves(
    args: argparse.Namespace,
    *,
    j_values: list[float],
    cut: int,
    k: int,
) -> list[dict[str, float | int | list[float] | list[int]]]:
    """Compute nested-budget convergence curves at selected ``J`` values."""
    catalog = enumerate_full_probe_catalog(cut=cut, num_interventions=k)
    past_all = np.arange(len(catalog.past_settings), dtype=np.int64)
    future_all = np.arange(len(catalog.future_settings), dtype=np.int64)
    probe_full = build_probe_set_from_catalog(catalog, past_all, future_all)
    max_budget = min(len(catalog.past_settings), len(catalog.future_settings))
    budgets = [b for b in (1, 2, 4, 8, 16, 32, 64) if b <= max_budget]
    past_perm, future_perm = nested_probe_indices(catalog, max_budget, seed=int(args.seed))
    past_perm = np.asarray(past_perm, dtype=np.int64)
    future_perm = np.asarray(future_perm, dtype=np.int64)

    mc = characterizer(parallel=False)
    params = sim_params(dt=DT_DEFAULT)
    timesteps = [DT_DEFAULT] * (k + 1)
    curves: list[dict[str, float | int | list[float] | list[int]]] = []
    for jv in j_values:
        ham = ising_chain(length=L_DEFAULT, j=float(jv), g=G_DEFAULT)
        pt = cast(
            DenseProcessTensor,
            mc.build_process_tensor(
                ham, params, timesteps=timesteps, return_type="dense", method="exhaustive", compress_every=1,
            ),
        )
        sv_full, _ = _sv_metrics(probe_full, pt)
        errs: list[float] = []
        for m in budgets:
            probe_m = build_probe_set_from_catalog(catalog, past_perm[:m], future_perm[:m])
            sv_m, _ = _sv_metrics(probe_m, pt)
            errs.append(abs(sv_m - sv_full))
        curves.append({"J": float(jv), "budgets": budgets, "abs_errors": errs, "S_V_full": sv_full})
    return curves


def _j0_sv_ipt_ok(rows: list[dict[str, float | int]]) -> bool:
    """Return False if J=0 fails Markov sanity for ``S_V^full`` or ``I_PT``."""
    for r in rows:
        if abs(float(r["J"])) < 1e-12:
            if float(r.get("S_V_full", 0.0)) >= 1e-10:
                return False
            if float(r.get("I_PT", 0.0)) >= 1e-10:
                return False
    return True


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    k = int(args.k) if args.k is not None else PROFILE_K[str(args.profile)]
    cut = int(args.cut) if args.cut is not None else center_cut(k)

    if args.plot_only or args.plot_prx_only:
        csv_path = args.summary_csv or (out_dir / "summary_sv_full.csv")
        rows = [_parse_summary_row(row) for row in load_csv(csv_path)]
    elif args.plot_ratio_only:
        csv_path = args.summary_csv or (out_dir / "summary.csv")
        rows = [_parse_summary_row(row) for row in load_csv(csv_path)]
        plot_results(rows, out_dir)
        return
    else:
        rows = run_full_benchmark(args)
        conv = run_convergence_curves(args, j_values=[0.0, 1.0, 2.0], cut=cut, k=k)
        with (out_dir / "convergence_curves.json").open("w", encoding="utf-8") as f:
            json.dump(conv, f, indent=2)
        plot_convergence_figure(conv, out_dir)

    allow_plot = _j0_sv_ipt_ok(rows)
    if not allow_plot:
        print("WARNING: J=0 S_V_full or I_PT is nonzero above tolerance.", flush=True)
    if allow_plot:
        import subprocess
        import sys

        script = Path(__file__).resolve().parent / "pt_response_comparison_prx.py"
        subprocess.run(
            [sys.executable, str(script), "--data-dir", str(out_dir)],
            check=True,
        )
        print(f"Wrote PRX figure: {out_dir / 'pt_response_comparison_prx.pdf'}", flush=True)
    print(f"Wrote results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
