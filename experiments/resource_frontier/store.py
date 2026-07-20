# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""SQLite/CSV storage for resource-frontier TFIM runs."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any

RAW_COLUMNS = (
    "method",
    "chi_max",
    "trotter_step",
    "time",
    "infidelity",
    "state_norm",
    "cumulative_runtime_s",
    "step_runtime_s",
    "current_max_bond",
    "peak_max_bond",
    "param_count",
    "peak_param_count",
    "memory_bytes",
    "peak_memory_bytes",
    "discarded_weight_step",
    "largest_intermediate_elements",
    "failed",
    "failure_message",
    "converged",
    "source",
    "tag",
)


class FrontierStore:
    """Resumable per-step store keyed by (method, chi, step, tag)."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), timeout=60.0, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def close(self) -> None:
        self._conn.close()

    def _init_schema(self) -> None:
        text_cols = {"method", "failure_message", "source", "tag"}
        cols = ", ".join(f"{c} TEXT" if c in text_cols else f"{c} REAL" for c in RAW_COLUMNS)
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS raw_runs (
                {cols},
                PRIMARY KEY (method, chi_max, trotter_step, tag)
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

    def has_step(self, method: str, chi: int, step: int, *, tag: str = "main") -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM raw_runs WHERE method=? AND chi_max=? AND trotter_step=? AND tag=?",
            (method, chi, step, tag),
        ).fetchone()
        return row is not None

    def max_step(self, method: str, chi: int, *, tag: str = "main") -> int:
        row = self._conn.execute(
            "SELECT MAX(trotter_step) FROM raw_runs WHERE method=? AND chi_max=? AND tag=?",
            (method, chi, tag),
        ).fetchone()
        return -1 if row is None or row[0] is None else int(float(row[0]))

    def insert_step(self, row: dict[str, Any]) -> None:
        payload = {c: row.get(c, "") for c in RAW_COLUMNS}
        placeholders = ", ".join("?" for _ in RAW_COLUMNS)
        self._conn.execute(
            f"INSERT OR REPLACE INTO raw_runs ({', '.join(RAW_COLUMNS)}) VALUES ({placeholders})",
            [payload[c] for c in RAW_COLUMNS],
        )

    def fetch_steps(
        self,
        *,
        method: str | None = None,
        chi: int | None = None,
        tag: str = "main",
    ) -> list[dict[str, Any]]:
        clauses = ["tag=?"]
        args: list[Any] = [tag]
        if method is not None:
            clauses.append("method=?")
            args.append(method)
        if chi is not None:
            clauses.append("chi_max=?")
            args.append(chi)
        where = " AND ".join(clauses)
        rows = self._conn.execute(
            f"SELECT {', '.join(RAW_COLUMNS)} FROM raw_runs WHERE {where} "
            "ORDER BY method, chi_max, trotter_step",
            args,
        ).fetchall()
        return [dict(zip(RAW_COLUMNS, row, strict=True)) for row in rows]

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
    if not rows and fieldnames is None:
        path.write_text("", encoding="utf-8")
        return
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
