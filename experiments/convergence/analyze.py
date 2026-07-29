# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Analyze TDVP substep convergence and build summary tables."""

from __future__ import annotations

from typing import Any

import numpy as np

from config import (
    ABS_INF_TOL,
    CHI_VALUES,
    DT,
    OUTPUT_DIR,
    REL_INF_TOL,
    RELIABILITY_THRESHOLD,
    SUBSTEPS_EXTENDED,
    SUBSTEPS_N16,
    SUBSTEPS_N32,
    TARGET_STEPS,
    cache_key,
    config_hash,
)
from store import write_csv

# Prefer the finest converged n-vs-2n pair as the reference.
_CONVERGENCE_PAIRS: tuple[tuple[int, int], ...] = (
    (2, 4),
    (4, SUBSTEPS_EXTENDED),
    (SUBSTEPS_EXTENDED, SUBSTEPS_N16),
    (SUBSTEPS_N16, SUBSTEPS_N32),
)


def n_epsilon(rows: list[dict[str, Any]]) -> int:
    """Largest k such that 1-F_j < ε for every j≤k (j≥1)."""
    by_step = {int(float(r["trotter_step"])): r for r in rows}
    best = 0
    for k in range(1, TARGET_STEPS + 1):
        if k not in by_step:
            break
        if int(float(by_step[k].get("failed", 0) or 0)):
            break
        if float(by_step[k]["infidelity"]) >= RELIABILITY_THRESHOLD:
            break
        best = k
    return best


def _infidelity_agree(a: float, b: float) -> bool:
    diff = abs(a - b)
    if diff <= ABS_INF_TOL:
        return True
    scale = max(abs(a), abs(b), 1e-30)
    return (diff / scale) <= REL_INF_TOL


def trajectories_converged(
    rows_n: list[dict[str, Any]],
    rows_2n: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Compare n vs 2n: identical nε and infidelity agreement on reliable+first crossing."""
    ne = n_epsilon(rows_n)
    ne2 = n_epsilon(rows_2n)
    if ne != ne2:
        return False, f"nε mismatch: {ne} vs {ne2}"

    by_n = {int(float(r["trotter_step"])): r for r in rows_n}
    by_2 = {int(float(r["trotter_step"])): r for r in rows_2n}
    steps = list(range(1, ne + 1))
    if ne + 1 in by_n and ne + 1 in by_2:
        steps.append(ne + 1)
    for k in steps:
        if k not in by_n or k not in by_2:
            return False, f"missing step {k}"
        a = float(by_n[k]["infidelity"])
        b = float(by_2[k]["infidelity"])
        if not _infidelity_agree(a, b):
            return False, f"infidelity disagree at step {k}: {a:.6e} vs {b:.6e}"
    return True, f"converged (nε={ne})"


def pairwise_state_infidelity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    na = float(np.linalg.norm(vec_a))
    nb = float(np.linalg.norm(vec_b))
    if na <= 0 or nb <= 0:
        return 1.0
    a = vec_a / na
    b = vec_b / nb
    return float(1.0 - abs(np.vdot(a, b)) ** 2)


def load_statevector(chi: int, substeps: int, step: int) -> np.ndarray | None:
    key = cache_key(chi=chi, substeps=substeps)
    path = OUTPUT_DIR / "statevectors" / key / f"step_{step:03d}.npy"
    if not path.exists():
        return None
    return np.load(path)


def attach_pairwise(
    all_rows: list[dict[str, Any]],
    *,
    substeps_present: list[int],
) -> list[dict[str, Any]]:
    """Add pairwise_infidelity_vs_2n and D_n = |I_n - I_2n| where 2n exists."""
    by_key: dict[tuple[int, int, int], float] = {}
    for r in all_rows:
        key = (
            int(float(r["chi_max"])),
            int(float(r["tdvp_substeps"])),
            int(float(r["trotter_step"])),
        )
        by_key[key] = float(r["infidelity"])

    out: list[dict[str, Any]] = []
    for r in all_rows:
        row = dict(r)
        chi = int(float(r["chi_max"]))
        n = int(float(r["tdvp_substeps"]))
        step = int(float(r["trotter_step"]))
        n2 = 2 * n
        if n2 in substeps_present and step >= 0:
            va = load_statevector(chi, n, step)
            vb = load_statevector(chi, n2, step)
            if va is not None and vb is not None:
                row["pairwise_infidelity_vs_2n"] = pairwise_state_infidelity(va, vb)
            else:
                row["pairwise_infidelity_vs_2n"] = ""
            i_n = by_key.get((chi, n, step))
            i_2n = by_key.get((chi, n2, step))
            if i_n is not None and i_2n is not None:
                row["D_n"] = abs(i_n - i_2n)
            else:
                row["D_n"] = ""
        else:
            row["pairwise_infidelity_vs_2n"] = ""
            row["D_n"] = ""
        out.append(row)
    return out


def comparison_window_steps(rows_n: list[dict[str, Any]], rows_2n: list[dict[str, Any]]) -> list[int]:
    """Reliable interval plus first crossing (same window as trajectories_converged)."""
    ne = n_epsilon(rows_n)
    by_n = {int(float(r["trotter_step"])): r for r in rows_n}
    by_2 = {int(float(r["trotter_step"])): r for r in rows_2n}
    steps = list(range(1, ne + 1))
    if ne + 1 in by_n and ne + 1 in by_2:
        steps.append(ne + 1)
    return steps


def d_n_series(
    rows_n: list[dict[str, Any]],
    rows_2n: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Per-step D_n(t)=|I_n(t)-I_2n(t)| on the full shared step set."""
    by_n = {int(float(r["trotter_step"])): float(r["infidelity"]) for r in rows_n}
    by_2 = {int(float(r["trotter_step"])): float(r["infidelity"]) for r in rows_2n}
    steps = sorted(set(by_n) & set(by_2))
    return [
        {
            "trotter_step": k,
            "time": k * DT,
            "I_n": by_n[k],
            "I_2n": by_2[k],
            "D_n": abs(by_n[k] - by_2[k]),
        }
        for k in steps
        if k >= 0
    ]


def d_ratio_series(
    d_low: list[dict[str, Any]],
    d_high: list[dict[str, Any]],
    *,
    window_steps: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Per-step ratio D_high/D_low (e.g. D_8/D_4)."""
    by_low = {int(r["trotter_step"]): float(r["D_n"]) for r in d_low}
    by_high = {int(r["trotter_step"]): float(r["D_n"]) for r in d_high}
    steps = sorted(set(by_low) & set(by_high))
    if window_steps is not None:
        steps = [k for k in steps if k in set(window_steps)]
    out: list[dict[str, Any]] = []
    for k in steps:
        lo = by_low[k]
        hi = by_high[k]
        ratio = hi / lo if lo > 0 else (0.0 if hi == 0 else float("inf"))
        out.append(
            {
                "trotter_step": k,
                "time": k * DT,
                "D_low": lo,
                "D_high": hi,
                "D_ratio": ratio,
            }
        )
    return out


def summarize_d_ratios(ratios: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [float(r["D_ratio"]) for r in ratios if np.isfinite(float(r["D_ratio"]))]
    if not vals:
        return {"n_steps": 0, "median": None, "mean": None, "max": None, "decreasing": None}
    arr = np.asarray(vals, dtype=float)
    return {
        "n_steps": len(vals),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "fraction_lt_1": float(np.mean(arr < 1.0)),
        "decreasing": bool(float(np.median(arr)) < 1.0),
    }


def build_summary(
    by_chi_n: dict[tuple[int, int], list[dict[str, Any]]],
    *,
    substeps_used: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return summary rows and overall verdict dict."""
    summary: list[dict[str, Any]] = []
    converged_n_per_chi: dict[int, int | None] = {}
    n1_pass_per_chi: dict[int, bool] = {}

    for chi in CHI_VALUES:
        # Report all available n-vs-2n comparisons; select coarsest converged pair.
        ref_n: int | None = None
        conv_msg = "no pairs available"
        for n_low, n_high in _CONVERGENCE_PAIRS:
            if (chi, n_low) not in by_chi_n or (chi, n_high) not in by_chi_n:
                continue
            ok, msg = trajectories_converged(by_chi_n[(chi, n_low)], by_chi_n[(chi, n_high)])
            summary.append(
                {
                    "chi_max": chi,
                    "comparison": f"n{n_low}_vs_n{n_high}",
                    "n_eps_low": n_epsilon(by_chi_n[(chi, n_low)]),
                    "n_eps_high": n_epsilon(by_chi_n[(chi, n_high)]),
                    "converged": int(ok),
                    "detail": msg,
                }
            )
            conv_msg = msg
            if ok and ref_n is None:
                ref_n = n_low

        converged_n_per_chi[chi] = ref_n

        if ref_n is not None and (chi, 1) in by_chi_n and (chi, ref_n) in by_chi_n:
            rows1 = by_chi_n[(chi, 1)]
            rows_ref = by_chi_n[(chi, ref_n)]
            ne1 = n_epsilon(rows1)
            ne_ref = n_epsilon(rows_ref)
            ok, msg = trajectories_converged(rows1, rows_ref)
            pass_n1 = ok
            n1_pass_per_chi[chi] = pass_n1
            summary.append(
                {
                    "chi_max": chi,
                    "comparison": f"n1_vs_n{ref_n}",
                    "n_eps_low": ne1,
                    "n_eps_high": ne_ref,
                    "converged": int(pass_n1),
                    "detail": msg if pass_n1 else f"FAIL: {msg}",
                    "verdict": "PASS" if pass_n1 else "FAIL",
                    "converged_reference_n": ref_n,
                }
            )
        else:
            n1_pass_per_chi[chi] = False
            summary.append(
                {
                    "chi_max": chi,
                    "comparison": "n1_vs_reference",
                    "n_eps_low": n_epsilon(by_chi_n[(chi, 1)]) if (chi, 1) in by_chi_n else "",
                    "n_eps_high": "",
                    "converged": 0,
                    "detail": f"No converged reference ({conv_msg})",
                    "verdict": "FAIL",
                    "converged_reference_n": "",
                }
            )

        for n in substeps_used:
            if (chi, n) not in by_chi_n:
                continue
            rows = by_chi_n[(chi, n)]
            summary.append(
                {
                    "chi_max": chi,
                    "comparison": f"n_eps_n{n}",
                    "n_eps_low": n_epsilon(rows),
                    "n_eps_high": "",
                    "converged": "",
                    "detail": f"nε(χ={chi}, n={n})={n_epsilon(rows)}",
                    "tdvp_substeps": n,
                    "peak_param_count_final": int(float(rows[-1]["peak_param_count"])) if rows else "",
                    "failed_any": int(any(int(float(r.get("failed", 0) or 0)) for r in rows)),
                }
            )

    all_pass = all(n1_pass_per_chi.get(chi, False) for chi in CHI_VALUES)
    refs = [converged_n_per_chi[chi] for chi in CHI_VALUES]
    if all(r is not None for r in refs):
        smallest_converged = max(int(r) for r in refs if r is not None)
    else:
        smallest_converged = None

    production_n = None
    if all(r == SUBSTEPS_EXTENDED for r in refs):
        production_n = SUBSTEPS_EXTENDED

    verdict = {
        "n1_pass_all_chi": all_pass,
        "n1_pass_per_chi": n1_pass_per_chi,
        "converged_reference_n_per_chi": converged_n_per_chi,
        "smallest_converged_n_all_chi": smallest_converged,
        "production_substeps": production_n,
        "config_hash": config_hash(),
        "dt": DT,
    }
    return summary, verdict


def export_tables(
    all_rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    *,
    d_rows: list[dict[str, Any]] | None = None,
    d_ratio_rows: list[dict[str, Any]] | None = None,
) -> None:
    write_csv(OUTPUT_DIR / "tfim_tdvp_substeps.csv", all_rows)
    write_csv(OUTPUT_DIR / "tfim_tdvp_substeps_summary.csv", summary)
    if d_rows is not None:
        write_csv(OUTPUT_DIR / "tfim_tdvp_substeps_D.csv", d_rows)
    if d_ratio_rows is not None:
        write_csv(OUTPUT_DIR / "tfim_tdvp_substeps_D_ratios.csv", d_ratio_rows)


def build_d_tables(
    by_chi_n: dict[tuple[int, int], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build D_n and D_8/D_4 tables; return ratio diagnostics for the probe χ."""
    d_rows: list[dict[str, Any]] = []
    ratio_rows: list[dict[str, Any]] = []
    probe_diag: dict[str, Any] = {}

    for chi in CHI_VALUES:
        present = sorted({n for (c, n) in by_chi_n if c == chi})
        for n in present:
            n2 = 2 * n
            if (chi, n2) not in by_chi_n:
                continue
            series = d_n_series(by_chi_n[(chi, n)], by_chi_n[(chi, n2)])
            for row in series:
                d_rows.append(
                    {
                        "chi_max": chi,
                        "n": n,
                        "n2": n2,
                        **row,
                    }
                )

        if (chi, 4) in by_chi_n and (chi, 8) in by_chi_n and (chi, 16) in by_chi_n:
            d4 = d_n_series(by_chi_n[(chi, 4)], by_chi_n[(chi, 8)])
            d8 = d_n_series(by_chi_n[(chi, 8)], by_chi_n[(chi, 16)])
            window = comparison_window_steps(by_chi_n[(chi, 8)], by_chi_n[(chi, 16)])
            ratios = d_ratio_series(d4, d8, window_steps=window)
            for row in ratios:
                ratio_rows.append({"chi_max": chi, "ratio_label": "D8_over_D4", **row})
            # Also store full-timeline ratios for inspection.
            ratios_all = d_ratio_series(d4, d8, window_steps=None)
            for row in ratios_all:
                if int(row["trotter_step"]) not in set(window):
                    ratio_rows.append(
                        {
                            "chi_max": chi,
                            "ratio_label": "D8_over_D4_all_steps",
                            **row,
                        }
                    )
            stats = summarize_d_ratios(ratios)
            stats_all = summarize_d_ratios(ratios_all)
            if chi not in probe_diag:
                probe_diag[chi] = {
                    "window_steps": window,
                    "window": stats,
                    "all_steps": stats_all,
                }

    return d_rows, ratio_rows, probe_diag
