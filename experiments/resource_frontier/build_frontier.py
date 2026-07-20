# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Construct memory and runtime frontiers from measured TFIM runs."""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from typing import Any

import path_setup  # noqa: F401
from config import (
    BYTES_PER_COMPLEX128,
    DT,
    METHODS,
    MIB,
    OUTPUT_DIR,
    RELIABILITY_THRESHOLD,
    TARGET_STEPS,
)
from store import FrontierStore, write_csv


def reliable_through(rows: list[dict[str, Any]], n: int) -> bool:
    by_step = {int(float(r["trotter_step"])): r for r in rows}
    for k in range(1, n + 1):
        if k not in by_step:
            return False
        if int(float(by_step[k].get("failed", 0) or 0)):
            return False
        if float(by_step[k]["infidelity"]) >= RELIABILITY_THRESHOLD:
            return False
    return True


def group_by_config(rows: list[dict[str, Any]]) -> dict[tuple[str, int], list[dict[str, Any]]]:
    out: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[(str(r["method"]), int(float(r["chi_max"])))].append(r)
    return out


def preliminary_runtime_mins(
    raw_rows: list[dict[str, Any]],
) -> dict[int, dict[str, dict[str, Any] | None]]:
    """For each n and method, runtime-minimizing χ among reliable configs (exploratory)."""
    grouped = group_by_config(raw_rows)
    result: dict[int, dict[str, dict[str, Any] | None]] = {}
    for n in range(1, TARGET_STEPS + 1):
        result[n] = {}
        for method in METHODS:
            best: dict[str, Any] | None = None
            for (m, chi), rows in grouped.items():
                if m != method:
                    continue
                if not reliable_through(rows, n):
                    continue
                row_n = next(r for r in rows if int(float(r["trotter_step"])) == n)
                rt = float(row_n["cumulative_runtime_s"])
                peak_p = int(float(row_n["peak_param_count"]))
                cand = {"chi_max": chi, "runtime_s": rt, "peak_param_count": peak_p, "n": n}
                if best is None or rt < float(best["runtime_s"]):
                    best = cand
            result[n][method] = best
    return result


def build_memory_frontier(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = group_by_config(raw_rows)
    out: list[dict[str, Any]] = []
    terminated = {m: False for m in METHODS}
    prev_p = {m: 0 for m in METHODS}
    for n in range(1, TARGET_STEPS + 1):
        for method in METHODS:
            if terminated[method]:
                out.append(
                    {
                        "method": method,
                        "target_step": n,
                        "target_time": n * DT,
                        "P_star": "",
                        "M_star_bytes": "",
                        "M_star_MiB": "",
                        "chi_max": "",
                        "missing": 1,
                    }
                )
                continue
            best: dict[str, Any] | None = None
            for (m, chi), rows in grouped.items():
                if m != method:
                    continue
                if not reliable_through(rows, n):
                    continue
                row_n = next(r for r in rows if int(float(r["trotter_step"])) == n)
                peak_p = int(float(row_n["peak_param_count"]))
                cand = {
                    "method": method,
                    "target_step": n,
                    "target_time": n * DT,
                    "P_star": peak_p,
                    "M_star_bytes": peak_p * BYTES_PER_COMPLEX128,
                    "M_star_MiB": (peak_p * BYTES_PER_COMPLEX128) / MIB,
                    "chi_max": chi,
                    "missing": 0,
                }
                if best is None or peak_p < int(best["P_star"]):
                    best = cand
            if best is None:
                terminated[method] = True
                out.append(
                    {
                        "method": method,
                        "target_step": n,
                        "target_time": n * DT,
                        "P_star": "",
                        "M_star_bytes": "",
                        "M_star_MiB": "",
                        "chi_max": "",
                        "missing": 1,
                    }
                )
            else:
                if int(best["P_star"]) < prev_p[method] and prev_p[method] > 0:
                    best["nondecreasing_violation"] = 1
                else:
                    best["nondecreasing_violation"] = 0
                prev_p[method] = int(best["P_star"])
                out.append(best)
    return out


def _median_iqr(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    med = float(statistics.median(values))
    if len(values) >= 4:
        qs = statistics.quantiles(values, n=4, method="inclusive")
        return med, float(qs[0]), float(qs[2])
    return med, min(values), max(values)


def build_runtime_frontier(
    raw_rows: list[dict[str, Any]],
    timing_rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Prefer controlled timing medians when available; else exploratory main runs."""
    # Build timing lookup: (method, chi, n) -> list of cumulative runtimes
    timing: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    if timing_rows:
        by_rep: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
        for r in timing_rows:
            key = (str(r["method"]), int(float(r["chi_max"])), int(float(r["repeat"])))
            by_rep[key].append(r)
        for (method, chi, _rep), rows in by_rep.items():
            if not reliable_through(rows, max(int(float(r["trotter_step"])) for r in rows)):
                # Still use per-n if reliable through that n.
                pass
            for n in range(1, TARGET_STEPS + 1):
                if reliable_through(rows, n):
                    row_n = next(r for r in rows if int(float(r["trotter_step"])) == n)
                    timing[(method, chi, n)].append(float(row_n["cumulative_runtime_s"]))

    grouped = group_by_config(raw_rows)
    out: list[dict[str, Any]] = []
    terminated = {m: False for m in METHODS}
    prev_rt = {m: 0.0 for m in METHODS}
    for n in range(1, TARGET_STEPS + 1):
        for method in METHODS:
            if terminated[method]:
                out.append(
                    {
                        "method": method,
                        "target_step": n,
                        "target_time": n * DT,
                        "R_star_s": "",
                        "R_iqr_low_s": "",
                        "R_iqr_high_s": "",
                        "chi_max": "",
                        "missing": 1,
                        "timing_source": "",
                    }
                )
                continue
            best: dict[str, Any] | None = None
            chis = sorted({chi for (m, chi) in grouped if m == method})
            for chi in chis:
                rows = grouped[(method, chi)]
                if not reliable_through(rows, n):
                    continue
                if (method, chi, n) in timing and timing[(method, chi, n)]:
                    med, lo, hi = _median_iqr(timing[(method, chi, n)])
                    source = "timing_median"
                else:
                    row_n = next(r for r in rows if int(float(r["trotter_step"])) == n)
                    med = float(row_n["cumulative_runtime_s"])
                    lo = hi = med
                    source = "exploratory"
                cand = {
                    "method": method,
                    "target_step": n,
                    "target_time": n * DT,
                    "R_star_s": med,
                    "R_iqr_low_s": lo,
                    "R_iqr_high_s": hi,
                    "chi_max": chi,
                    "missing": 0,
                    "timing_source": source,
                }
                if best is None or med < float(best["R_star_s"]):
                    best = cand
            if best is None:
                terminated[method] = True
                out.append(
                    {
                        "method": method,
                        "target_step": n,
                        "target_time": n * DT,
                        "R_star_s": "",
                        "R_iqr_low_s": "",
                        "R_iqr_high_s": "",
                        "chi_max": "",
                        "missing": 1,
                        "timing_source": "",
                    }
                )
            else:
                if float(best["R_star_s"]) < prev_rt[method] - 1e-12 and prev_rt[method] > 0:
                    best["nondecreasing_violation"] = 1
                else:
                    best["nondecreasing_violation"] = 0
                prev_rt[method] = float(best["R_star_s"])
                out.append(best)
    return out


def _is_present(row: dict[str, Any]) -> bool:
    miss = row.get("missing", 1)
    if miss is None or miss == "":
        return False
    return int(float(miss)) == 0


def largest_common_reliable_step(mem_rows: list[dict[str, Any]]) -> int:
    best = 0
    for n in range(1, TARGET_STEPS + 1):
        ok = True
        for method in METHODS:
            row = next(
                (
                    r
                    for r in mem_rows
                    if r["method"] == method and int(float(r["target_step"])) == n and _is_present(r)
                ),
                None,
            )
            if row is None:
                ok = False
                break
        if ok:
            best = n
        else:
            break
    return best


def build_all() -> dict[str, Any]:
    store = FrontierStore(OUTPUT_DIR / "raw_runs.sqlite")
    raw = store.fetch_steps(tag="main")
    store.close()
    timing_path = OUTPUT_DIR / "timing_repeats.csv"
    timing_rows: list[dict[str, Any]] = []
    if timing_path.exists():
        import csv

        with timing_path.open(encoding="utf-8") as handle:
            timing_rows = list(csv.DictReader(handle))

    mem = build_memory_frontier(raw)
    rt = build_runtime_frontier(raw, timing_rows)
    write_csv(OUTPUT_DIR / "memory_frontier.csv", mem)
    write_csv(OUTPUT_DIR / "runtime_frontier.csv", rt)
    return {"memory": mem, "runtime": rt, "raw_n": len(raw), "timing_n": len(timing_rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build resource frontiers.")
    parser.parse_args(argv)
    summary = build_all()
    print(f"Wrote frontiers from {summary['raw_n']} raw rows, {summary['timing_n']} timing rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
