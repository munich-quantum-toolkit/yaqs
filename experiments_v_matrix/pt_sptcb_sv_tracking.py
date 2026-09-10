#!/usr/bin/env python3
"""Cross-case check that S_PT^cb tracks S_V^full across (L, k, c) and J."""

from __future__ import annotations

import argparse
import csv
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from common import DT_DEFAULT, G_DEFAULT, characterizer, configure_matplotlib_prl, ising_chain, sim_params, write_csv
from mqt.yaqs.characterization.memory.backends.tomography.process_tensors import (
    DenseProcessTensor,
    causal_block_operator_entropy,
)
from mqt.yaqs.characterization.memory.operational_memory.full_basis import (
    build_probe_set_from_catalog,
    enumerate_full_probe_catalog,
)
from pt_cut_reference import _sv_metrics

DEFAULT_J = [0.0, 0.5, 1.0, 1.5, 2.0]
ATOL = 1e-15


@dataclass(frozen=True)
class Case:
    length: int
    k: int
    cut: int
    label: str


DEFAULT_CASES = (
    Case(2, 2, 2, r"$L{=}2,\ k{=}2,\ c{=}2$"),
    Case(3, 2, 2, r"$L{=}3,\ k{=}2,\ c{=}2$"),
    Case(3, 3, 2, r"$L{=}3,\ k{=}3,\ c{=}2$"),
    Case(3, 3, 1, r"$L{=}3,\ k{=}3,\ c{=}1$"),
    Case(6, 3, 2, r"$L{=}6,\ k{=}3,\ c{=}2$"),
)


def _relative_deviation(sv: np.ndarray, other: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.abs(sv), ATOL)
    rel = np.abs(other - sv) / denom
    both_small = (np.abs(sv) <= ATOL) & (np.abs(other - sv) <= ATOL)
    rel[both_small] = 0.0
    return rel


def _evaluate_case_j(
    *,
    case: Case,
    jv: float,
    mc: Any,
    params: Any,
    timesteps: list[float],
    probe_full: Any,
) -> dict[str, float | int | str]:
    ham = ising_chain(length=case.length, j=jv, g=G_DEFAULT)
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
    ups = pt_dense.to_matrix()
    sv_full, _ = _sv_metrics(probe_full, pt_dense)
    spt_cb = float(causal_block_operator_entropy(ups, case.k, case.cut)["entropy"])
    return {
        "case": case.label,
        "L": case.length,
        "k": case.k,
        "cut": case.cut,
        "J": float(jv),
        "S_V_full": float(sv_full),
        "S_PT_cb": spt_cb,
    }


def _load_paper_case_rows(case: Case, *, out_dir: Path) -> list[dict[str, float | int | str]] | None:
    """Reuse cached L=6, k=3, c=2 benchmark rows when available."""
    if (case.length, case.k, case.cut) != (6, 3, 2):
        return None
    sv_path = out_dir / "pt_response_process_tensor_main_data.csv"
    cb_path = out_dir / "summary_cb.csv"
    if not sv_path.is_file() or not cb_path.is_file():
        return None

    sv_by_j: dict[float, float] = {}
    with sv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(float(row["cut"])) != case.cut:
                continue
            sv_by_j[float(row["J"])] = float(row["S_V_full"])

    cb_by_j: dict[float, float] = {}
    with cb_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(float(row["cut"])) != case.cut:
                continue
            if int(float(row["L"])) != case.length or int(float(row["k"])) != case.k:
                continue
            cb_by_j[float(row["J"])] = float(row["S_PT_cb"])

    shared_js = sorted(set(sv_by_j) & set(cb_by_j))
    if not shared_js:
        return None
    return [
        {
            "case": case.label,
            "L": case.length,
            "k": case.k,
            "cut": case.cut,
            "J": jv,
            "S_V_full": sv_by_j[jv],
            "S_PT_cb": cb_by_j[jv],
        }
        for jv in shared_js
    ]


def _summarize_case(rows: list[dict[str, float | int | str]]) -> dict[str, float | int | str]:
    j = np.asarray([float(r["J"]) for r in rows], dtype=np.float64)
    sv = np.asarray([float(r["S_V_full"]) for r in rows], dtype=np.float64)
    cb = np.asarray([float(r["S_PT_cb"]) for r in rows], dtype=np.float64)
    rel = _relative_deviation(sv, cb)
    pos = j > 1e-12
    rel_pos = rel[pos] if np.any(pos) else rel
    return {
        "case": rows[0]["case"],
        "L": int(rows[0]["L"]),
        "k": int(rows[0]["k"]),
        "cut": int(rows[0]["cut"]),
        "n_J": int(j.size),
        "max_rel_dev": float(np.max(rel_pos)) if rel_pos.size else 0.0,
        "median_rel_dev": float(np.median(rel_pos)) if rel_pos.size else 0.0,
        "max_log10_ratio": float(np.max(np.log10(np.maximum(cb[pos], ATOL) / np.maximum(sv[pos], ATOL))))
        if np.any(pos)
        else 0.0,
    }


def _plot_tracking(
    case_rows: dict[str, list[dict[str, float | int | str]]],
    *,
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogFormatterMathtext, LogLocator

    configure_matplotlib_prl()
    cases = list(case_rows.keys())
    n = len(cases)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.2 * nrows), squeeze=False)
    floor = 1e-12

    for idx, case_label in enumerate(cases):
        ax = axes[idx // ncols][idx % ncols]
        rows = case_rows[case_label]
        j = np.asarray([float(r["J"]) for r in rows], dtype=np.float64)
        sv = np.maximum(np.asarray([float(r["S_V_full"]) for r in rows], dtype=np.float64), floor)
        cb = np.maximum(np.asarray([float(r["S_PT_cb"]) for r in rows], dtype=np.float64), floor)
        ax.plot(j, sv, "o-", color="#0072B2", lw=1.4, ms=4, mew=0.6, label=r"$S_V^{\mathrm{full}}$")
        ax.plot(j, cb, "^:", color="#009E73", lw=1.4, ms=4, mew=0.6, label=r"$S_{\mathrm{PT}}^{\mathrm{cb}}$")
        ax.set_yscale("log")
        ax.set_xlim(0.0, float(j.max()))
        ax.set_xlabel(r"$J$")
        ax.set_ylabel("nats")
        ax.set_title(case_label, fontsize=8.5)
        ax.yaxis.set_major_formatter(LogFormatterMathtext())
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.tick_params(direction="in", top=True, right=True, which="both", labelsize=7)
        if idx == 0:
            ax.legend(frameon=False, fontsize=7, loc="lower right")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.tight_layout()
    stem = out_dir / "fig_sptcb_sv_tracking"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", dpi=600)
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=600)
    plt.close(fig)


def run_tracking(
    *,
    cases: tuple[Case, ...],
    j_values: list[float],
    out_dir: Path,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mc = characterizer(parallel=False)
    params = sim_params(dt=DT_DEFAULT)

    all_rows: list[dict[str, float | int | str]] = []
    grouped: dict[str, list[dict[str, float | int | str]]] = {}

    for case in cases:
        cached = _load_paper_case_rows(case, out_dir=out_dir)
        if cached is not None:
            print(f"Using cached rows for {case.label} ({len(cached)} J points)", flush=True)
            rows = cached
        else:
            catalog = enumerate_full_probe_catalog(cut=case.cut, num_interventions=case.k)
            probe_full = build_probe_set_from_catalog(
                catalog,
                np.arange(len(catalog.past_settings), dtype=np.int64),
                np.arange(len(catalog.future_settings), dtype=np.int64),
            )
            timesteps = [DT_DEFAULT] * (case.k + 1)
            rows = []
            for jv in j_values:
                print(f"{case.label}  J={jv:g}", flush=True)
                rows.append(
                    _evaluate_case_j(
                        case=case,
                        jv=jv,
                        mc=mc,
                        params=params,
                        timesteps=timesteps,
                        probe_full=probe_full,
                    )
                )
        grouped[case.label] = rows
        all_rows.extend(rows)

    summaries = [_summarize_case(grouped[c.label]) for c in cases]
    write_csv(out_dir / "sptcb_sv_tracking.csv", all_rows)
    write_csv(out_dir / "sptcb_sv_tracking_summary.csv", summaries)
    _plot_tracking(grouped, out_dir=out_dir)

    lines = ["S_PT^cb vs S_V^full cross-case tracking", "=" * 44]
    for s in summaries:
        lines.append(
            f"{s['case']}:  median rel. dev = {float(s['median_rel_dev']):.2e}, "
            f"max rel. dev = {float(s['max_rel_dev']):.2e}  (J>0, n={s['n_J']})"
        )
    lines.append("")
    lines.append(
        textwrap.dedent(
            """
            Interpretation:
              S_PT^cb and S_V^full share the causal past|future split and increase with J.
              Relative deviations stay O(1) or smaller on these Ising benchmarks; they are
              not pointwise identical (different objects: operator Schmidt vs response entropy).
            """
        ).strip()
    )
    report = "\n".join(lines)
    (out_dir / "sptcb_sv_tracking_report.txt").write_text(report + "\n", encoding="utf-8")
    print(report, flush=True)
    return all_rows, summaries


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=Path("save/pt_cut_reference"))
    p.add_argument("--j-values", type=str, default=",".join(str(x) for x in DEFAULT_J))
    args = p.parse_args()
    j_values = [float(x.strip()) for x in args.j_values.split(",") if x.strip()]
    run_tracking(cases=DEFAULT_CASES, j_values=j_values, out_dir=args.out_dir.resolve())


if __name__ == "__main__":
    main()
