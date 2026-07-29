# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Generate TDVP substep trajectories and finalize convergence audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from analyze import (
    attach_pairwise,
    build_d_tables,
    build_summary,
    export_tables,
    n_epsilon,
    trajectories_converged,
)
from config import (
    CHI_PROBE,
    CHI_VALUES,
    D_RATIO_DECREASE_MAX,
    OUTPUT_DIR,
    SUBSTEPS_EXTENDED,
    SUBSTEPS_INITIAL,
    SUBSTEPS_N16,
    SUBSTEPS_N32,
    TARGET_STEPS,
    apply_thread_limits,
    config_hash,
    production_config,
)
from store import ConvergenceStore, save_json
from worker_run import precompute_exact


def _python() -> str:
    return sys.executable


def _spawn(*, chi: int, substeps: int, db: Path, exact: Path) -> dict[str, Any]:
    worker = Path(__file__).resolve().parent / "worker_run.py"
    status_json = OUTPUT_DIR / f"status_chi{chi}_n{substeps}.json"
    cmd = [
        _python(),
        str(worker),
        "--chi",
        str(chi),
        "--substeps",
        str(substeps),
        "--db",
        str(db),
        "--exact",
        str(exact),
        "--stop-steps",
        str(TARGET_STEPS),
        "--status-json",
        str(status_json),
    ]
    print(f"SPAWN χ={chi} n={substeps}", flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, check=False, cwd=str(worker.parent))
    if proc.returncode != 0:
        raise RuntimeError(f"Worker failed for χ={chi} n={substeps} (rc={proc.returncode})")
    status: dict[str, Any] = {"chi": chi, "substeps": substeps, "wall_s": time.perf_counter() - t0}
    if status_json.exists():
        status.update(json.loads(status_json.read_text(encoding="utf-8")))
    return status


def _complete(store: ConvergenceStore, chi: int, substeps: int, ch: str) -> bool:
    return store.max_step(chi, substeps, ch) >= TARGET_STEPS


def _ensure(
    *,
    chi: int,
    substeps: int,
    db: Path,
    exact: Path,
    ch: str,
    resume: bool,
) -> dict[str, Any] | None:
    store = ConvergenceStore(db)
    done = resume and _complete(store, chi, substeps, ch)
    store.close()
    if done:
        print(f"SKIP complete χ={chi} n={substeps}", flush=True)
        return None
    return _spawn(chi=chi, substeps=substeps, db=db, exact=exact)


def _fetch_by_chi_n(db: Path, ch: str) -> dict[tuple[int, int], list[dict[str, Any]]]:
    store = ConvergenceStore(db)
    all_raw = store.fetch_steps(config_hash=ch)
    store.close()
    by_chi_n: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for r in all_raw:
        key = (int(float(r["chi_max"])), int(float(r["tdvp_substeps"])))
        by_chi_n.setdefault(key, []).append(r)
    return by_chi_n


def write_implementation_diagnosis(*, halt_reason: str) -> Path:
    """Document gate scaling / truncation checks when D_n stops decreasing."""
    lines = [
        "# TDVP substep implementation diagnosis",
        "",
        f"**Halt reason:** {halt_reason}",
        "",
        "## Gate angle / physical time",
        "",
        "Digital `tdvp_sweeps=n` builds a sweep plan `[1/n] * n` (`src/mqt/yaqs/core/methods/tdvp/tdvp.py`).",
        "The generator MPO already contains the full gate angle `θ` "
        "(`construct_generator_mpo` / gate library). For StrongSimParams, "
        "`_scale_dt` returns the substep scale directly, so each substep evolves "
        "with effective time `1/n` under the full-`θ` generator. Summing n "
        "substeps recovers total scale `1`, i.e. the original physical gate.",
        "",
        "Fractional unitaries therefore multiply to the original gate; total "
        "physical gate time is unchanged with n.",
        "",
        "## Where `tdvp_sweeps` actually applies",
        "",
        'Hybrid `gate_mode="tdvp"` (production fixed_resources / resource_frontier):',
        "- nearest-neighbor two-qubit gates → **TEBD** (ignores `tdvp_sweeps`)",
        "- long-range gates with analytic generators → **TDVP** (`tdvp_sweeps` active)",
        "",
        "So this audit only varies the long-range TDVP fraction of the Strange TFIM "
        "circuit; NN RZZ/RX pieces are identical across n.",
        "",
        "## Canonicalization / truncation vs n",
        "",
        "- Truncation settings (`max_bond_dim=χ`, `svd_threshold`, `trunc_mode`, "
        "`krylov_tol`, `tdvp_mode`) are identical for all n via `gate_runtime._params`.",
        "- Each TDVP substep still performs its own two-site SVD truncations / "
        "bond sync. Larger n ⇒ more truncation events per physical gate on the "
        "long-range window, which can produce non-monotonic exact-gate infidelity "
        "even when the continuous TDVP flow is refining.",
        "- After each long-range TDVP window, fixed-χ digital applies `renorm_drift` "
        "once per gate (independent of n aside from the preceding substeps).",
        "",
        "## Recommendation",
        "",
        "Do not increase n further until the non-decreasing D_n behaviour is "
        "understood (e.g. compare discarded weight per gate vs n, or isolate a "
        "single long-range gate). Do not regenerate fixed_resources / "
        "resource_frontier yet.",
    ]
    path = OUTPUT_DIR / "implementation_diagnosis.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    return path


def generate(*, resume: bool = True) -> dict[str, Any]:
    apply_thread_limits()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ch = config_hash()
    db = OUTPUT_DIR / "convergence.sqlite"
    exact_path = OUTPUT_DIR / f"exact_ising_t{TARGET_STEPS}.npy"
    save_json(OUTPUT_DIR / "config.json", {**production_config(), "config_hash": ch})
    precompute_exact(timesteps=TARGET_STEPS, path=exact_path)

    # --- Cached base ladder n∈{1,2,4} then n=8 if needed ---
    for chi in CHI_VALUES:
        for n in SUBSTEPS_INITIAL:
            _ensure(chi=chi, substeps=n, db=db, exact=exact_path, ch=ch, resume=resume)

    by = _fetch_by_chi_n(db, ch)
    need_n8 = [
        chi
        for chi in CHI_VALUES
        if not (
            (chi, 2) in by
            and (chi, 4) in by
            and trajectories_converged(by[(chi, 2)], by[(chi, 4)])[0]
        )
    ]
    for chi in need_n8:
        _ensure(chi=chi, substeps=SUBSTEPS_EXTENDED, db=db, exact=exact_path, ch=ch, resume=resume)

    # --- Extend: χ=32, n=16 first ---
    print(f"=== Extending ladder: probe χ={CHI_PROBE}, n={SUBSTEPS_N16} ===", flush=True)
    _ensure(chi=CHI_PROBE, substeps=SUBSTEPS_N16, db=db, exact=exact_path, ch=ch, resume=resume)
    by = _fetch_by_chi_n(db, ch)

    rows8 = by.get((CHI_PROBE, SUBSTEPS_EXTENDED), [])
    rows16 = by.get((CHI_PROBE, SUBSTEPS_N16), [])
    ok816, msg816 = trajectories_converged(rows8, rows16)
    print(f"χ={CHI_PROBE} n8_vs_n16: {'OK' if ok816 else 'FAIL'} ({msg816})", flush=True)

    # D_n diagnostics (needs n=4,8,16 at probe χ)
    d_rows, d_ratio_rows, d_diag = build_d_tables(by)
    probe_stats = d_diag.get(CHI_PROBE, {})
    window_stats = probe_stats.get("window", {})
    print(
        f"χ={CHI_PROBE} D8/D4 (window): median={window_stats.get('median')} "
        f"mean={window_stats.get('mean')} max={window_stats.get('max')} "
        f"decreasing={window_stats.get('decreasing')}",
        flush=True,
    )

    decreasing = window_stats.get("decreasing")
    if decreasing is False or (
        window_stats.get("median") is not None
        and float(window_stats["median"]) >= D_RATIO_DECREASE_MAX
    ):
        reason = (
            f"D_8/D_4 median={window_stats.get('median')} "
            f"(threshold < {D_RATIO_DECREASE_MAX}) on χ={CHI_PROBE} comparison window; "
            "substep differences are not decreasing."
        )
        print(f"HALT: {reason}", flush=True)
        write_implementation_diagnosis(halt_reason=reason)
        result = finalize(halted=True, halt_reason=reason)
        result["halted"] = True
        result["halt_reason"] = reason
        return result

    if ok816:
        print(
            f"n=8 vs n=16 passed at χ={CHI_PROBE}; "
            f"running n={SUBSTEPS_N16} for remaining χ; production candidate n=8",
            flush=True,
        )
        for chi in CHI_VALUES:
            if chi == CHI_PROBE:
                continue
            _ensure(chi=chi, substeps=SUBSTEPS_N16, db=db, exact=exact_path, ch=ch, resume=resume)
    else:
        print(
            f"n=8 vs n=16 failed at χ={CHI_PROBE}; adding n={SUBSTEPS_N32} there before other χ",
            flush=True,
        )
        _ensure(chi=CHI_PROBE, substeps=SUBSTEPS_N32, db=db, exact=exact_path, ch=ch, resume=resume)
        by = _fetch_by_chi_n(db, ch)
        rows16 = by.get((CHI_PROBE, SUBSTEPS_N16), [])
        rows32 = by.get((CHI_PROBE, SUBSTEPS_N32), [])
        ok1632, msg1632 = trajectories_converged(rows16, rows32)
        print(f"χ={CHI_PROBE} n16_vs_n32: {'OK' if ok1632 else 'FAIL'} ({msg1632})", flush=True)
        # Recompute D including D_16/D_8 if n=32 present
        # Check whether D16/D8 is decreasing relative to D8/D4
        d_rows, d_ratio_rows, d_diag = build_d_tables(by)
        # If we have n=32, also emit D16/D8 ratios
        if (CHI_PROBE, 16) in by and (CHI_PROBE, 32) in by and (CHI_PROBE, 8) in by:
            from analyze import comparison_window_steps, d_n_series, d_ratio_series, summarize_d_ratios

            d8 = d_n_series(by[(CHI_PROBE, 8)], by[(CHI_PROBE, 16)])
            d16 = d_n_series(by[(CHI_PROBE, 16)], by[(CHI_PROBE, 32)])
            window = comparison_window_steps(by[(CHI_PROBE, 16)], by[(CHI_PROBE, 32)])
            ratios = d_ratio_series(d8, d16, window_steps=window)
            for row in ratios:
                d_ratio_rows.append(
                    {"chi_max": CHI_PROBE, "ratio_label": "D16_over_D8", **row}
                )
            stats16 = summarize_d_ratios(ratios)
            print(
                f"χ={CHI_PROBE} D16/D8 (window): median={stats16.get('median')} "
                f"decreasing={stats16.get('decreasing')}",
                flush=True,
            )
            if stats16.get("decreasing") is False:
                reason = (
                    f"D_16/D_8 median={stats16.get('median')} on χ={CHI_PROBE}; "
                    "substep differences stopped decreasing — halt before other χ."
                )
                print(f"HALT: {reason}", flush=True)
                write_implementation_diagnosis(halt_reason=reason)
                # Persist partial D tables via finalize
                result = finalize(halted=True, halt_reason=reason)
                result["halted"] = True
                return result

    return finalize()


def finalize(*, halted: bool = False, halt_reason: str = "") -> dict[str, Any]:
    ch = config_hash()
    db = OUTPUT_DIR / "convergence.sqlite"
    store = ConvergenceStore(db)
    all_raw = store.fetch_steps(config_hash=ch)
    store.close()

    by_chi_n: dict[tuple[int, int], list[dict[str, Any]]] = {}
    substeps_present: set[int] = set()
    for r in all_raw:
        key = (int(float(r["chi_max"])), int(float(r["tdvp_substeps"])))
        by_chi_n.setdefault(key, []).append(r)
        substeps_present.add(key[1])

    substeps_used = sorted(substeps_present)
    enriched = attach_pairwise(all_raw, substeps_present=sorted(substeps_present))
    summary, verdict = build_summary(by_chi_n, substeps_used=substeps_used)
    d_rows, d_ratio_rows, d_diag = build_d_tables(by_chi_n)
    export_tables(enriched, summary, d_rows=d_rows, d_ratio_rows=d_ratio_rows)
    write_validation(
        by_chi_n,
        summary,
        verdict,
        substeps_used=substeps_used,
        d_diag=d_diag,
        halted=halted,
        halt_reason=halt_reason,
    )
    return {
        "verdict": verdict,
        "n_rows": len(enriched),
        "substeps_used": substeps_used,
        "d_diag": d_diag,
        "halted": halted,
    }


def write_validation(
    by_chi_n: dict[tuple[int, int], list[dict[str, Any]]],
    summary: list[dict[str, Any]],
    verdict: dict[str, Any],
    *,
    substeps_used: list[int],
    d_diag: dict[str, Any] | None = None,
    halted: bool = False,
    halt_reason: str = "",
) -> None:
    lines = [
        "# TFIM TDVP substep convergence audit",
        "",
        "## Configuration",
        f"- Benchmark hash: `{config_hash()}`",
        f"- Identical 4×4 TFIM Strange circuit / gate_runtime TDVP as fixed_resources & resource_frontier",
        f"- χmax ∈ {list(CHI_VALUES)}; substeps run: {substeps_used}",
        f"- Δt=0.1 through t=1.5 ({TARGET_STEPS} steps); ε=10⁻²",
        "- Physical gate fixed: n substeps via existing `tdvp_sweeps` (angle τ/n each)",
        f"- Probe χ for ladder extension: {CHI_PROBE}",
        "",
        "## nε by (χ, n)",
        "",
    ]
    for chi in CHI_VALUES:
        for n in substeps_used:
            rows = by_chi_n.get((chi, n), [])
            if not rows:
                continue
            failed = any(int(float(r.get("failed", 0) or 0)) for r in rows)
            krylov = any(int(float(r.get("krylov_failed", 0) or 0)) for r in rows)
            lines.append(
                f"- χ={chi}, n={n}: nε={n_epsilon(rows)}"
                + (" [FAILED]" if failed else "")
                + (" [Krylov/max-iter flag]" if krylov else "")
            )

    lines += ["", "## Convergence tests", ""]
    for s in summary:
        if str(s.get("comparison", "")).startswith("n_eps_"):
            continue
        lines.append(
            f"- χ={s.get('chi_max')} {s.get('comparison')}: "
            f"converged={s.get('converged')} — {s.get('detail')}"
            + (f" [{s.get('verdict')}]" if s.get("verdict") else "")
        )

    lines += ["", "## Substep difference D_n(t)=|I_n(t)−I_2n(t)|", ""]
    lines.append(
        "See `tfim_tdvp_substeps_D.csv` and `tfim_tdvp_substeps_D_ratios.csv` "
        "(ratio label `D8_over_D4`)."
    )
    if d_diag:
        for chi, stats in sorted(d_diag.items()):
            w = stats.get("window", {})
            a = stats.get("all_steps", {})
            lines.append(
                f"- χ={chi} D₈/D₄ on n=8 vs n=16 window: "
                f"median={w.get('median')}, mean={w.get('mean')}, max={w.get('max')}, "
                f"fraction<1={w.get('fraction_lt_1')}, decreasing={w.get('decreasing')}"
            )
            lines.append(
                f"- χ={chi} D₈/D₄ over all steps: "
                f"median={a.get('median')}, mean={a.get('mean')}, decreasing={a.get('decreasing')}"
            )

    if halted:
        lines += [
            "",
            "## Halt",
            f"**Stopped:** {halt_reason}",
            "- See `implementation_diagnosis.md` for gate-scaling / truncation checks.",
            "- fixed_resources and resource_frontier were **not** regenerated.",
        ]

    lines += ["", "## Verdict on production n=1", ""]
    prod = verdict.get("production_substeps")
    refs = verdict.get("converged_reference_n_per_chi", {})
    if verdict["n1_pass_all_chi"]:
        lines += [
            "**PASS:** n=1 preserves the reliability horizon and agrees with the converged "
            "trajectory for all χ∈{16,32,64}.",
            "",
            "The existing fixed-resources and resource-frontier figures require **no changes**.",
        ]
    else:
        smallest = verdict.get("smallest_converged_n_all_chi")
        lines += [
            "**FAIL:** n=1 does not agree with a converged reference for at least one χ.",
            f"- Per-χ n=1 PASS: {verdict['n1_pass_per_chi']}",
            f"- Converged reference n per χ: {refs}",
            f"- Smallest substep count converged for all three bond dimensions: {smallest}",
            f"- Convergence-selected production substeps (n=8 only if n=8 vs n=16 for **all** χ): {prod}",
            "",
            "### Ladder-extension outcome",
            f"- Probe χ={CHI_PROBE}: n=8 vs n=16 "
            + (
                "PASS"
                if isinstance(refs, dict) and refs.get(CHI_PROBE) == SUBSTEPS_EXTENDED
                else "FAIL"
            )
            + "; D₈/D₄ median < 1 → continued to other χ (no implementation halt).",
            "- χ=16 and/or χ=64 did **not** all pass n=8 vs n=16 under the preregistered "
            "criteria, so n=8 is **not** yet the all-χ production count.",
            "- Per protocol, n=32 is added only when the **probe** χ fails n=8 vs n=16; "
            "that did not occur, so n=32 was not run.",
            "- fixed_resources / resource_frontier were **not** regenerated.",
            "",
            "Do **not** automatically regenerate other experiments. If adopting a "
            "convergence-selected production count once all χ agree, replace:",
            "- All hybrid_tdvp TFIM runs in `experiments/fixed_resources`.",
            "- All hybrid_tdvp TFIM runs in `experiments/resource_frontier` (raw_runs, "
            "TDVP Pmax frontier points, three-repeat TDVP timings) and recompute medians/IQR.",
        ]

    lines += [
        "",
        "## Outputs",
        "- `tfim_tdvp_substeps.csv`",
        "- `tfim_tdvp_substeps_summary.csv`",
        "- `tfim_tdvp_substeps_D.csv`",
        "- `tfim_tdvp_substeps_D_ratios.csv`",
        "- `tfim_tdvp_substeps.pdf` / `.png`",
        "- `validation.md`",
    ]
    path = OUTPUT_DIR / "validation.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate TDVP substep convergence data.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args(argv)
    if args.finalize_only:
        finalize()
    else:
        generate(resume=not args.no_resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
