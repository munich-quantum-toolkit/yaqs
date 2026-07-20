# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""SQLite/CSV storage for TDVP substep convergence audit."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

STEP_COLUMNS = (
    "chi_max",
    "tdvp_substeps",
    "config_hash",
    "cache_key",
    "trotter_step",
    "time",
    "infidelity",
    "state_norm",
    "peak_max_bond",
    "peak_param_count",
    "param_count",
    "cumulative_runtime_s",
    "step_runtime_s",
    "discarded_weight_step",
    "failed",
    "failure_message",
    "krylov_failed",
)


class ConvergenceStore:
    """Resumable per-step store keyed by (chi, substeps, config_hash, step)."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), timeout=60.0, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        text_cols = {"config_hash", "cache_key", "failure_message"}
        cols = ", ".join(f"{c} TEXT" if c in text_cols else f"{c} REAL" for c in STEP_COLUMNS)
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS steps (
                {cols},
                PRIMARY KEY (chi_max, tdvp_substeps, config_hash, trotter_step)
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

    def has_step(self, chi: int, substeps: int, config_hash: str, step: int) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM steps WHERE chi_max=? AND tdvp_substeps=? AND config_hash=? AND trotter_step=?",
            (chi, substeps, config_hash, step),
        ).fetchone()
        return row is not None

    def max_step(self, chi: int, substeps: int, config_hash: str) -> int:
        row = self._conn.execute(
            "SELECT MAX(trotter_step) FROM steps WHERE chi_max=? AND tdvp_substeps=? AND config_hash=?",
            (chi, substeps, config_hash),
        ).fetchone()
        return -1 if row is None or row[0] is None else int(float(row[0]))

    def insert_step(self, row: dict[str, Any]) -> None:
        payload = {c: row.get(c, "") for c in STEP_COLUMNS}
        placeholders = ", ".join("?" for _ in STEP_COLUMNS)
        self._conn.execute(
            f"INSERT OR REPLACE INTO steps ({', '.join(STEP_COLUMNS)}) VALUES ({placeholders})",
            [payload[c] for c in STEP_COLUMNS],
        )

    def fetch_steps(
        self,
        *,
        chi: int | None = None,
        substeps: int | None = None,
        config_hash: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        args: list[Any] = []
        if chi is not None:
            clauses.append("chi_max=?")
            args.append(chi)
        if substeps is not None:
            clauses.append("tdvp_substeps=?")
            args.append(substeps)
        if config_hash is not None:
            clauses.append("config_hash=?")
            args.append(config_hash)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(
            f"SELECT {', '.join(STEP_COLUMNS)} FROM steps{where} "
            "ORDER BY chi_max, tdvp_substeps, trotter_step",
            args,
        ).fetchall()
        return [dict(zip(STEP_COLUMNS, row, strict=True)) for row in rows]

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row[0])


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fields: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
    else:
        fields = fieldnames
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
