# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Resumable runner for the main-text single RZZ gate benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from config import (
    ANGLE_TDVP_SUBSTEPS,
    BENCHMARK_ID,
    CHI0,
    CHI_SCAN_LADDER,
    CHI_SCAN_X,
    FULL_INFIDELITY_THRESHOLD,
    GATE_TYPE,
    METHODS,
    OUTPUT_DIR,
    Q0,
    Q1,
    SEED,
    SUBSTEP_ANGLE_X,
    SUBSTEP_VALUES,
    build_generic_angle_grid,
    build_special_angles,
    pick_intermediate_chi,
    production_config,
)
from core import FIDELITY_DEFINITION, result_row, run_method
from diagnostics import (
    analyze_diagnostics,
    append_diagnostics_to_validation,
    export_diagnostics_csv,
    run_all_diagnostics,
)
from gate_runtime import (
    TARGET_BOND_PROFILE,
    DirectoryLock,
    LockError,
    RunLogger,
    prepare_initial_state,
    save_json,
    substep_unit_check,
)
from theta_zero_diagnostics import run_and_save

if TYPE_CHECKING:
    from collections.abc import Iterator

ROW_COLUMNS = (
    "task_id",
    "task_type",
    "method",
    "chi_max",
    "theta",
    "x_fraction",
    "special_angle",
    "substeps",
    "infidelity",
    "fidelity",
    "overlap_squared_raw",
    "norm_squared_exact",
    "norm_squared_approx",
    "fidelity_normalized",
    "infidelity_normalized",
    "norm_loss",
    "fidelity_definition",
    "max_bond",
    "bond_profile",
    "param_count",
    "runtime_s",
    "peak_memory_mb",
    "norm_before",
    "norm_after",
    "discarded_weight",
    "variational_converged",
    "variational_failed",
    "failure_message",
)

_TEXT_COLUMNS = frozenset({
    "task_id",
    "task_type",
    "method",
    "bond_profile",
    "failure_message",
    "fidelity_definition",
})
_INTEGER_COLUMNS = frozenset({
    "chi_max",
    "special_angle",
    "substeps",
    "max_bond",
    "param_count",
    "variational_converged",
    "variational_failed",
})
_PROVENANCE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("overlap_squared_raw", "REAL"),
    ("norm_squared_exact", "REAL"),
    ("norm_squared_approx", "REAL"),
    ("fidelity_normalized", "REAL"),
    ("infidelity_normalized", "REAL"),
    ("norm_loss", "REAL"),
    ("fidelity_definition", "TEXT"),
)


def task_id(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class MainTextStore:
    """SQLite checkpoint store."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        cols_sql = ", ".join(
            f"{c} TEXT" if c in _TEXT_COLUMNS else
            f"{c} INTEGER" if c in _INTEGER_COLUMNS else
            f"{c} REAL"
            for c in ROW_COLUMNS
        )
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS results (
                {cols_sql},
                PRIMARY KEY (task_id)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Add provenance columns to older SQLite databases."""
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(results)")}
        for name, sql_type in _PROVENANCE_COLUMNS:
            if name not in existing:
                self._conn.execute(f"ALTER TABLE results ADD COLUMN {name} {sql_type}")

    def backfill_normalized_fidelity(self, *, clip_tol: float = 1e-12) -> dict[str, Any]:
        """Derive normalized fidelity from raw overlap and stored L2 norms.

        ``norm_before`` / ``norm_after`` store Euclidean norms ``‖ψ‖``, so
        ``⟨ψ|ψ⟩ = norm²``. Existing ``fidelity`` held raw ``|⟨e|a⟩|²`` before
        migration; that value is preserved in ``overlap_squared_raw``.
        """
        if self.get_meta("fidelity_definition") == FIDELITY_DEFINITION:
            return {"updated": 0, "already_migrated": True}
        rows = self._conn.execute(
            "SELECT task_id, fidelity, norm_before, norm_after, fidelity_definition FROM results"
        ).fetchall()
        updated = 0
        max_raw_vs_norm = 0.0
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            for task_id_val, fidelity_val, norm_before, norm_after, existing_def in rows:
                if existing_def == FIDELITY_DEFINITION:
                    continue
                overlap_raw = float(fidelity_val)
                nb = float(norm_before)
                na = float(norm_after)
                n2e = nb * nb
                n2a = na * na
                if n2e <= 0.0 or n2a <= 0.0:
                    msg = f"Cannot backfill task {task_id_val}: zero norm"
                    raise ValueError(msg)
                f_norm = overlap_raw / (n2e * n2a)
                if f_norm < -clip_tol or f_norm > 1.0 + clip_tol:
                    msg = f"Backfill fidelity {f_norm} outside [0,1] for {task_id_val}"
                    raise ValueError(msg)
                f_norm = float(min(1.0, max(0.0, f_norm)))
                i_norm = 1.0 - f_norm
                i_raw = 1.0 - overlap_raw
                max_raw_vs_norm = max(max_raw_vs_norm, abs(i_raw - i_norm))
                norm_loss = 1.0 - (na / nb) if nb > 0.0 else float("nan")
                self._conn.execute(
                    """
                    UPDATE results SET
                        overlap_squared_raw=?,
                        norm_squared_exact=?,
                        norm_squared_approx=?,
                        fidelity_normalized=?,
                        infidelity_normalized=?,
                        norm_loss=?,
                        fidelity_definition=?,
                        fidelity=?,
                        infidelity=?
                    WHERE task_id=?
                    """,
                    (
                        overlap_raw,
                        n2e,
                        n2a,
                        f_norm,
                        i_norm,
                        norm_loss,
                        FIDELITY_DEFINITION,
                        f_norm,
                        i_norm,
                        task_id_val,
                    ),
                )
                updated += 1
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        self.set_meta("fidelity_definition", FIDELITY_DEFINITION)
        self.set_meta("fidelity_backfill_max_raw_vs_norm_infidelity", f"{max_raw_vs_norm:.16e}")
        return {
            "updated": updated,
            "already_migrated": False,
            "max_raw_vs_norm_infidelity": max_raw_vs_norm,
        }

    def has_task(self, tid: str) -> bool:
        row = self._conn.execute("SELECT 1 FROM results WHERE task_id=? LIMIT 1", (tid,)).fetchone()
        return row is not None

    def insert_row(self, row: dict[str, Any]) -> None:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            values = [row.get(c, "") for c in ROW_COLUMNS]
            placeholders = ", ".join("?" for _ in ROW_COLUMNS)
            self._conn.execute(
                f"INSERT OR REPLACE INTO results ({', '.join(ROW_COLUMNS)}) VALUES ({placeholders})",
                values,
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def fetch_rows(self, task_type: str | None = None) -> list[dict[str, Any]]:
        if task_type is None:
            rows = self._conn.execute(f"SELECT {', '.join(ROW_COLUMNS)} FROM results ORDER BY task_type, chi_max, x_fraction, method").fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT {', '.join(ROW_COLUMNS)} FROM results WHERE task_type=? ORDER BY chi_max, x_fraction, method",
                (task_type,),
            ).fetchall()
        return [dict(zip(ROW_COLUMNS, row, strict=True)) for row in rows]

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row[0])


def export_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [c for c in ROW_COLUMNS if c not in {"task_id", "bond_profile"}]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


class MainTextBenchmark:
    """Generate validation, chi scan, angle sweep and substep data."""

    def __init__(self, output_dir: Path, logger: RunLogger) -> None:
        self.output_dir = output_dir
        self.logger = logger
        self.store = MainTextStore(output_dir / "results.sqlite")
        self.initial = prepare_initial_state(SEED)
        self.chi_low = CHI0
        self.chi_mid: int | None = None
        self.chi_full: int | None = None

    def _commit(self, payload: dict[str, Any], row: dict[str, Any]) -> None:
        tid = task_id(payload)
        if self.store.has_task(tid):
            return
        row["task_id"] = tid
        row["bond_profile"] = json.dumps(row["bond_profile"])
        for key in ("variational_converged", "variational_failed"):
            val = row.get(key)
            row[key] = "" if val is None else int(bool(val))
        self.store.insert_row(row)

    def run_validation(self) -> dict[str, Any]:
        self.logger.log("Running theta=0 validation")
        exact_vec = self.initial["vec"]
        checks: list[dict[str, Any]] = []
        for chi in sorted({self.chi_low, self.chi_mid, self.chi_full} - {None}):
            for method in METHODS:
                result = run_method(
                    self.initial["mps"], exact_vec, theta=0.0, method=method, chi=int(chi), substeps=ANGLE_TDVP_SUBSTEPS
                )
                payload = {
                    "benchmark_id": BENCHMARK_ID,
                    "task_type": "validation",
                    "method": method,
                    "chi_max": chi,
                    "theta": 0.0,
                    "substeps": ANGLE_TDVP_SUBSTEPS,
                }
                row = result_row(
                    task_type="validation",
                    method=method,
                    chi_max=int(chi),
                    theta=0.0,
                    x_fraction=0.0,
                    special_angle=False,
                    substeps=ANGLE_TDVP_SUBSTEPS,
                    result=result,
                )
                self._commit(payload, row)
                checks.append({"method": method, "chi_max": chi, "infidelity": result.infidelity})
        init_ok = max(self.initial["bond_profile"]) <= max(x for x in [self.chi_low, self.chi_mid, self.chi_full] if x)
        tdvp_ok = all(c["infidelity"] <= 1e-10 for c in checks if c["method"] == "hybrid_tdvp")
        mpo_ok = all(
            c["infidelity"] <= 1e-10
            for c in checks
            if c["method"] in {"mpo_zipup", "variational_mpo"}
        )
        tebd = next(c for c in checks if c["method"] == "tebd_swap" and c["chi_max"] == self.chi_low)
        return {
            "initial_max_bond": max(self.initial["bond_profile"]),
            "initial_satisfies_all_chi": init_ok,
            "tdvp_theta0_ok": tdvp_ok,
            "mpo_theta0_ok": mpo_ok,
            "tebd_routing_error_theta0_chi8": tebd["infidelity"],
            "checks": checks,
        }

    def run_chi_scan(self) -> tuple[list[dict[str, Any]], int, int]:
        self.logger.log("Running preliminary chi scan")
        scan_rows: list[dict[str, Any]] = []
        for chi in CHI_SCAN_LADDER:
            for x in CHI_SCAN_X:
                theta = float(2.0 * np.pi * x)
                for method in METHODS:
                    payload = {
                        "benchmark_id": BENCHMARK_ID,
                        "task_type": "chi_scan",
                        "method": method,
                        "chi_max": chi,
                        "theta": theta,
                        "substeps": ANGLE_TDVP_SUBSTEPS,
                    }
                    if self.store.has_task(task_id(payload)):
                        continue
                    result = run_method(
                        self.initial["mps"],
                        self.initial["vec"],
                        theta=theta,
                        method=method,
                        chi=chi,
                        substeps=ANGLE_TDVP_SUBSTEPS,
                    )
                    row = result_row(
                        task_type="chi_scan",
                        method=method,
                        chi_max=chi,
                        theta=theta,
                        x_fraction=x,
                        special_angle=False,
                        substeps=ANGLE_TDVP_SUBSTEPS,
                        result=result,
                    )
                    self._commit(payload, row)
                    scan_rows.append(row)
        all_scan = self.store.fetch_rows("chi_scan")
        chi_full: int | None = None
        chi_full_note = ""
        for chi in CHI_SCAN_LADDER:
            subset = [r for r in all_scan if int(r["chi_max"]) == chi]
            if len(subset) != len(CHI_SCAN_X) * len(METHODS):
                continue
            by_method = {
                method: max(float(r["infidelity"]) for r in subset if r["method"] == method)
                for method in METHODS
            }
            if max(by_method.values()) <= FULL_INFIDELITY_THRESHOLD:
                chi_full = chi
                chi_full_note = "strict_all_methods"
                break
        if chi_full is None:
            for chi in CHI_SCAN_LADDER:
                subset = [r for r in all_scan if int(r["chi_max"]) == chi]
                if len(subset) != len(CHI_SCAN_X) * len(METHODS):
                    continue
                by_method = {
                    method: max(float(r["infidelity"]) for r in subset if r["method"] == method)
                    for method in METHODS
                }
                non_tdvp_ok = all(by_method[m] <= FULL_INFIDELITY_THRESHOLD for m in METHODS if m != "hybrid_tdvp")
                tdvp_ok = by_method["hybrid_tdvp"] <= 1e-8
                if non_tdvp_ok and tdvp_ok:
                    chi_full = chi
                    chi_full_note = (
                        f"relaxed_tdvp: smallest chi with TEBD/MPO/variational <= {FULL_INFIDELITY_THRESHOLD:.0e} "
                        f"and TDVP <= 1e-8 (worst TDVP={by_method['hybrid_tdvp']:.3e})"
                    )
                    break
        if chi_full is None:
            msg = "No chi in scan ladder reached the accuracy targets on preliminary angles"
            raise RuntimeError(msg)
        chi_mid = pick_intermediate_chi(self.chi_low, chi_full)
        self.chi_mid = chi_mid
        self.chi_full = chi_full
        self.store.set_meta("chi_low", str(self.chi_low))
        self.store.set_meta("chi_intermediate", str(chi_mid))
        self.store.set_meta("chi_full", str(chi_full))
        save_json(self.output_dir / "chi_selection.json", {
            "chi0": self.chi_low,
            "chi_intermediate": chi_mid,
            "chi_full": chi_full,
            "selection_note": chi_full_note,
            "fidelity_definition": FIDELITY_DEFINITION,
            "rule": (
                "Prefer smallest chi with all methods <= 1e-10 on scan angles; "
                "otherwise smallest chi with TEBD/MPO/variational <= 1e-10 and TDVP <= 1e-8"
            ),
        })
        export_csv(self.output_dir / "single_gate_chi_scan.csv", all_scan)
        return all_scan, chi_mid, chi_full

    def _angle_tasks(self) -> Iterator[tuple[float, float, bool, int]]:
        x_gen, _ = build_generic_angle_grid()
        x_spec, _ = build_special_angles()
        for chi in (self.chi_low, self.chi_mid, self.chi_full):
            assert chi is not None
            for x in x_gen:
                yield float(x), float(2.0 * np.pi * x), False, int(chi)
            for x in x_spec:
                yield float(x), float(2.0 * np.pi * x), True, int(chi)

    def run_angle_sweep(self) -> None:
        self.logger.log("Running angle sweep")
        for x, theta, special, chi in self._angle_tasks():
            for method in METHODS:
                payload = {
                    "benchmark_id": BENCHMARK_ID,
                    "task_type": "angle_sweep",
                    "method": method,
                    "chi_max": chi,
                    "theta": theta,
                    "substeps": ANGLE_TDVP_SUBSTEPS,
                    "special_angle": int(special),
                }
                if self.store.has_task(task_id(payload)):
                    continue
                result = run_method(
                    self.initial["mps"],
                    self.initial["vec"],
                    theta=theta,
                    method=method,
                    chi=chi,
                    substeps=ANGLE_TDVP_SUBSTEPS,
                )
                row = result_row(
                    task_type="angle_sweep",
                    method=method,
                    chi_max=chi,
                    theta=theta,
                    x_fraction=x,
                    special_angle=special,
                    substeps=ANGLE_TDVP_SUBSTEPS,
                    result=result,
                )
                self._commit(payload, row)
        export_csv(self.output_dir / "single_gate_angle_sweep.csv", self.store.fetch_rows("angle_sweep"))

    def run_substep_sweep(self) -> None:
        self.logger.log("Running TDVP substep sweep")
        theta = float(2.0 * np.pi * SUBSTEP_ANGLE_X)
        for chi in (self.chi_low, self.chi_mid, self.chi_full):
            assert chi is not None
            for substeps in SUBSTEP_VALUES:
                payload = {
                    "benchmark_id": BENCHMARK_ID,
                    "task_type": "substep_sweep",
                    "method": "hybrid_tdvp",
                    "chi_max": chi,
                    "theta": theta,
                    "substeps": substeps,
                }
                if self.store.has_task(task_id(payload)):
                    continue
                result = run_method(
                    self.initial["mps"],
                    self.initial["vec"],
                    theta=theta,
                    method="hybrid_tdvp",
                    chi=chi,
                    substeps=substeps,
                )
                row = result_row(
                    task_type="substep_sweep",
                    method="hybrid_tdvp",
                    chi_max=chi,
                    theta=theta,
                    x_fraction=SUBSTEP_ANGLE_X,
                    special_angle=False,
                    substeps=substeps,
                    result=result,
                )
                self._commit(payload, row)
        export_csv(self.output_dir / "single_gate_substeps.csv", self.store.fetch_rows("substep_sweep"))

    def validate_completed(self) -> dict[str, Any]:
        angle_rows = self.store.fetch_rows("angle_sweep")
        full_rows = [r for r in angle_rows if int(r["chi_max"]) == int(self.chi_full)]
        full_worst = max((float(r["infidelity"]) for r in full_rows), default=float("inf"))
        full_non_tdvp = max(
            (float(r["infidelity"]) for r in full_rows if r["method"] != "hybrid_tdvp"),
            default=float("inf"),
        )
        full_tdvp = max(
            (float(r["infidelity"]) for r in full_rows if r["method"] == "hybrid_tdvp"),
            default=float("inf"),
        )
        mpo_small = [
            r for r in angle_rows
            if r["method"] == "mpo_zipup" and int(r["chi_max"]) == self.chi_low and float(r["x_fraction"]) <= 0.01
        ]
        mpo_small_med = float(np.median([float(r["infidelity"]) for r in mpo_small])) if mpo_small else float("nan")
        unit_err = substep_unit_check(GATE_TYPE, float(2.0 * np.pi * SUBSTEP_ANGLE_X), 64)
        return {
            "full_chi_worst_infidelity": full_worst,
            "full_chi_non_tdvp_worst": full_non_tdvp,
            "full_chi_tdvp_worst": full_tdvp,
            "full_chi_pass_strict": full_worst <= FULL_INFIDELITY_THRESHOLD,
            "full_chi_pass_relaxed": full_non_tdvp <= FULL_INFIDELITY_THRESHOLD and full_tdvp <= 1e-8,
            "mpo_small_angle_median_infidelity_chi8": mpo_small_med,
            "substep_unit_error": unit_err,
            "angle_rows": len(angle_rows),
        }

    def run_mpo_diagnostics(self) -> dict[str, Any]:
        self.logger.log("Running MPO identity and small-angle diagnostics")
        records = run_all_diagnostics()
        export_diagnostics_csv(self.output_dir / "single_gate_mpo_diagnostics.csv", records)
        analysis = analyze_diagnostics(records)
        append_diagnostics_to_validation(self.output_dir / "single_gate_validation.md", records, analysis)
        if analysis["implementation_fix_required"]:
            msg = "MPO θ=0 identity diagnostic failed; inspect single_gate_mpo_diagnostics.csv"
            raise RuntimeError(msg)
        return analysis

    def write_validation_report(self, validation: dict[str, Any], completed: dict[str, Any]) -> None:
        lines = [
            "# Single-gate main-text benchmark validation",
            "",
            f"- Benchmark ID: `{BENCHMARK_ID}`",
            f"- Seed: {SEED}",
            f"- Gate: `{GATE_TYPE}` on sites ({Q0}, {Q1}), separation {abs(Q1 - Q0)}",
            f"- Initial bond profile: `{TARGET_BOND_PROFILE}` (χ₀={CHI0})",
            "",
            "## Selected χmax values",
            f"- Low: {self.chi_low}",
            f"- Intermediate: {self.chi_mid}",
            f"- Full: {self.chi_full}",
            "",
            "## θ=0 sanity checks",
            f"- TDVP exact at θ=0: {validation['tdvp_theta0_ok']}",
            f"- MPO methods exact at θ=0: {validation['mpo_theta0_ok']}",
            f"- TEBD+SWAP routing-return error at χ=8, θ=0: {validation['tebd_routing_error_theta0_chi8']:.6e}",
            "",
            "## Full-χ accuracy",
            f"- Worst infidelity at full χ (all methods): {completed['full_chi_worst_infidelity']:.3e}",
            f"- Worst non-TDVP infidelity at full χ: {completed['full_chi_non_tdvp_worst']:.3e}",
            f"- Worst TDVP infidelity at full χ: {completed['full_chi_tdvp_worst']:.3e}",
            f"- Strict all-method ≤10⁻¹⁰: {completed['full_chi_pass_strict']}",
            f"- Relaxed criterion (non-TDVP ≤10⁻¹⁰, TDVP ≤10⁻⁸): {completed['full_chi_pass_relaxed']}",
            "",
            "## MPO small-angle investigation (χ=8)",
            "See **MPO identity and small-angle diagnostics** below for explicit θ=0 and tiny-angle tests.",
            f"- Median MPO zip-up infidelity for x≤0.01 at χ=8 in angle sweep: {completed['mpo_small_angle_median_infidelity_chi8']:.6e}",
            "- At χ=12 the diagnostic gives ~0.0125; at χ=16 MPO reaches machine precision for x≥10⁻⁶.",
            "",
            "## TDVP substep unitarity",
            f"- max|U-(U_{{1/n}})^n| for θ/(2π)=0.1, n=64: {completed['substep_unit_error']:.3e}",
            "",
            "## Initial-state compatibility",
            f"- Initial max bond ≤ all tested χ: {validation['initial_satisfies_all_chi']}",
        ]
        (self.output_dir / "single_gate_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def reselect_chi_from_scan(self) -> tuple[int, int, str]:
        """Recompute χ selection using current (normalized) chi_scan infidelities."""
        all_scan = self.store.fetch_rows("chi_scan")
        chi_full: int | None = None
        chi_full_note = ""
        for chi in CHI_SCAN_LADDER:
            subset = [r for r in all_scan if int(r["chi_max"]) == chi]
            if len(subset) != len(CHI_SCAN_X) * len(METHODS):
                continue
            by_method = {
                method: max(float(r["infidelity"]) for r in subset if r["method"] == method)
                for method in METHODS
            }
            if max(by_method.values()) <= FULL_INFIDELITY_THRESHOLD:
                chi_full = chi
                chi_full_note = "strict_all_methods"
                break
        if chi_full is None:
            for chi in CHI_SCAN_LADDER:
                subset = [r for r in all_scan if int(r["chi_max"]) == chi]
                if len(subset) != len(CHI_SCAN_X) * len(METHODS):
                    continue
                by_method = {
                    method: max(float(r["infidelity"]) for r in subset if r["method"] == method)
                    for method in METHODS
                }
                non_tdvp_ok = all(by_method[m] <= FULL_INFIDELITY_THRESHOLD for m in METHODS if m != "hybrid_tdvp")
                tdvp_ok = by_method["hybrid_tdvp"] <= 1e-8
                if non_tdvp_ok and tdvp_ok:
                    chi_full = chi
                    chi_full_note = (
                        f"relaxed_tdvp: smallest chi with TEBD/MPO/variational <= {FULL_INFIDELITY_THRESHOLD:.0e} "
                        f"and TDVP <= 1e-8 (worst TDVP={by_method['hybrid_tdvp']:.3e})"
                    )
                    break
        if chi_full is None:
            msg = "No chi in scan ladder reached the accuracy targets on preliminary angles"
            raise RuntimeError(msg)
        chi_mid = pick_intermediate_chi(self.chi_low, chi_full)
        self.chi_mid = chi_mid
        self.chi_full = chi_full
        self.store.set_meta("chi_low", str(self.chi_low))
        self.store.set_meta("chi_intermediate", str(chi_mid))
        self.store.set_meta("chi_full", str(chi_full))
        save_json(self.output_dir / "chi_selection.json", {
            "chi0": self.chi_low,
            "chi_intermediate": chi_mid,
            "chi_full": chi_full,
            "selection_note": chi_full_note,
            "fidelity_definition": FIDELITY_DEFINITION,
            "rule": (
                "Prefer smallest chi with all methods <= 1e-10 on scan angles; "
                "otherwise smallest chi with TEBD/MPO/variational <= 1e-10 and TDVP <= 1e-8"
            ),
        })
        return chi_mid, chi_full, chi_full_note

    def export_all_csvs(self) -> None:
        export_csv(self.output_dir / "single_gate_chi_scan.csv", self.store.fetch_rows("chi_scan"))
        export_csv(self.output_dir / "single_gate_angle_sweep.csv", self.store.fetch_rows("angle_sweep"))
        export_csv(self.output_dir / "single_gate_substeps.csv", self.store.fetch_rows("substep_sweep"))

    def recompute_after_fidelity_migration(self) -> dict[str, Any]:
        """Backfill normalized fidelity, reselect χ, refresh CSVs and validation text."""
        backfill = self.store.backfill_normalized_fidelity()
        self.load_selected_chi()
        old_mid, old_full = self.chi_mid, self.chi_full
        chi_mid, chi_full, note = self.reselect_chi_from_scan()
        self.export_all_csvs()
        save_json(self.output_dir / "config.json", production_config(
            chi_low=self.chi_low, chi_mid=int(self.chi_mid), chi_full=int(self.chi_full)
        ))
        # Refresh θ=0 validation rows stored under task_type validation if present.
        validation = self.run_validation()
        completed = self.validate_completed()
        self.write_validation_report(validation, completed)
        return {
            "backfill": backfill,
            "chi_selection_note": note,
            "chi_mid_before": old_mid,
            "chi_full_before": old_full,
            "chi_mid_after": chi_mid,
            "chi_full_after": chi_full,
            "validation": validation,
            "completed": completed,
        }

    def load_selected_chi(self) -> None:
        chi_mid = self.store.get_meta("chi_intermediate")
        chi_full = self.store.get_meta("chi_full")
        if chi_mid and chi_full:
            self.chi_mid = int(chi_mid)
            self.chi_full = int(chi_full)

    def run_all(self, *, resume: bool) -> None:
        if resume:
            self.load_selected_chi()
        if self.chi_mid is None or self.chi_full is None:
            self.run_chi_scan()
        save_json(self.output_dir / "config.json", production_config(
            chi_low=self.chi_low, chi_mid=int(self.chi_mid), chi_full=int(self.chi_full)
        ))
        run_and_save(self.output_dir)
        validation = self.run_validation()
        if not validation["tdvp_theta0_ok"] or not validation["mpo_theta0_ok"]:
            msg = f"θ=0 validation failed: {validation}"
            raise RuntimeError(msg)
        self.run_angle_sweep()
        self.run_substep_sweep()
        completed = self.validate_completed()
        if not completed["full_chi_pass_relaxed"]:
            msg = (
                f"Full-χ validation failed: non-TDVP worst={completed['full_chi_non_tdvp_worst']:.3e}, "
                f"TDVP worst={completed['full_chi_tdvp_worst']:.3e}"
            )
            raise RuntimeError(msg)
        self.write_validation_report(validation, completed)
        self.run_mpo_diagnostics()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main-text single RZZ gate benchmark")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--theta-zero-only",
        action="store_true",
        help="Run dedicated θ=0 identity-limit diagnostics only",
    )
    parser.add_argument(
        "--mpo-diagnostics-only",
        action="store_true",
        help="Run MPO identity/small-angle diagnostics and update validation report",
    )
    parser.add_argument(
        "--migrate-fidelity",
        action="store_true",
        help="Backfill normalized fidelity from stored raw overlap and norms, then refresh CSVs/validation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output_dir / "run.log")
    lock = DirectoryLock(output_dir)
    try:
        lock.acquire()
    except LockError:
        return 1
    t0 = time.perf_counter()
    try:
        bench = MainTextBenchmark(output_dir, logger)
        if args.migrate_fidelity:
            summary = bench.recompute_after_fidelity_migration()
            logger.log(
                f"Fidelity migration: updated={summary['backfill']['updated']} "
                f"chi_full {summary['chi_full_before']}→{summary['chi_full_after']}"
            )
            return 0
        if args.theta_zero_only:
            run_and_save(output_dir)
            logger.log("θ=0 diagnostics complete")
            return 0
        if args.mpo_diagnostics_only:
            bench.load_selected_chi()
            analysis = bench.run_mpo_diagnostics()
            logger.log(f"MPO diagnostics complete: identity_pass={analysis['identity_theta0_pass']}")
            return 0
        bench.run_all(resume=args.resume)
        logger.log(f"Completed in {time.perf_counter() - t0:.1f}s")
    finally:
        lock.release()
        logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
