#!/usr/bin/env python3
"""Compare medium cut×J benchmark CSV against mc-process reference data."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAQS_REPO = Path("/home/aaron/Github/yaqs")
REF_BRANCH = "mc-process"
MEDIUM = ROOT / "save" / "medium" / "cut_vs_j" / "summary.csv"


def load_reference_csv(git_path: str) -> list[dict[str, str]]:
    """Load a CSV committed on mc-process via git show."""
    proc = subprocess.run(
        ["git", "-C", str(YAQS_REPO), "show", f"{REF_BRANCH}:{git_path}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return list(csv.DictReader(proc.stdout.splitlines()))


def load_local_csv(path: Path) -> list[dict[str, str]]:
    """Load a local benchmark summary CSV."""
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key_cut_j(row: dict[str, str]) -> tuple[int, float]:
    """Index rows by (cut, J)."""
    return int(float(row["cut"])), round(float(row["J"]), 10)


def compare_entropy_vs_j(*, local_csv: Path, ref_git_path: str) -> None:
    """Report max relative error between local and mc-process entropy tables."""
    ref = {key_cut_j(r): float(r["entropy"]) for r in load_reference_csv(ref_git_path)}
    loc = {key_cut_j(r): float(r["entropy"]) for r in load_local_csv(local_csv)}
    common = sorted(set(ref) & set(loc))
    if not common:
        print("No overlapping (cut, J) keys between reference and local results.")
        return

    print(f"Comparing {len(common)} (cut, J) points")
    print(f"  local:  {local_csv}")
    print(f"  ref:    {REF_BRANCH}:{ref_git_path}")
    print()

    rels: list[float] = []
    for k in common:
        s_ref, s_new = ref[k], loc[k]
        denom = max(abs(s_ref), 1e-12)
        if s_ref >= 1e-10:
            rels.append(abs(s_new - s_ref) / denom)

    if rels:
        rels.sort()
        print(f"Median rel. error (J>0, {len(rels)} points): {rels[len(rels) // 2]:.3e}")
        print(f"Max rel. error (J>0): {max(rels):.3e}")


def main() -> None:
    if not MEDIUM.is_file():
        print(f"Missing local results: {MEDIUM}", file=sys.stderr)
        print("Run: python run_medium.py", file=sys.stderr)
        sys.exit(1)
    if not (YAQS_REPO / ".git").is_dir():
        print(f"YAQS repo not found at {YAQS_REPO}", file=sys.stderr)
        sys.exit(1)

    compare_entropy_vs_j(
        local_csv=MEDIUM,
        ref_git_path="experiments/benchmark_entropy_vs_j_by_cut_results/summary.csv",
    )


if __name__ == "__main__":
    main()
