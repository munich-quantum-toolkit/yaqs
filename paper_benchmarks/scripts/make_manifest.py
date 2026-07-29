# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Stage 1 (audit): build the provenance manifest for every reused raw input.

Usage:
    uv run python paper_benchmarks/scripts/make_manifest.py
"""

from __future__ import annotations

from datetime import UTC, datetime

from pb_common import PB_DIR, RAW_DIR, RAW_NEW_DIR, git_commit, save_json, sha256sum

# Provenance notes for each reused input group.
#
# dense_reference_ordering_fix: whether the dataset was produced after the
#   general-pair dense-reference reshape fix (documented in
#   save/long_range_gate_substeps_v2/report.md). The corrected single-gate
#   pair (2, 9) on L=12 additionally satisfies q0+q1=L-1, for which even the
#   pre-fix reference was correct; validate_dense.py re-verifies the
#   convention independently either way.
# production_svd_semantics: trunc_mode="discarded_weight" with
#   svd_threshold=1e-13 and gate-library hard split cutoff 1e-14
#   (raw/svd_diagnostic/cutoff_semantics.json).
GROUPS = {
    "single_gate_corrected": {
        "source": "experiments/single_gate/output/",
        "generating_script": "experiments/single_gate/regenerate.py",
        "config": "experiments/single_gate/output/config.json",
        "dense_reference_ordering_fix": True,
        "production_svd_semantics": True,
        "notes": (
            "Corrected post-repair single-gate campaign "
            "(protocol compress_rightcanon_ltr+var_multistart+tdvp_n1_v1): "
            "RZZ, seed 11, sites (2,9), L=12, chi in {8,12,16}, "
            "TDVP angle-sweep substeps n=1."
        ),
    },
    "single_gate_validation": {
        "source": "experiments/single_gate/validation/",
        "generating_script": "experiments/single_gate/validation/audit.py",
        "config": "experiments/single_gate/validation/meta.json",
        "dense_reference_ordering_fix": True,
        "production_svd_semantics": True,
        "notes": (
            "Ten-seed robustness audit and variational root-cause diagnosis; "
            "diagnostic only, used for supplemental robustness statements."
        ),
    },
    "circuits_corrected": {
        "source": "experiments/fixed_resources/output_corrected/",
        "generating_script": "experiments/fixed_resources/generate_corrected.py",
        "config": "experiments/fixed_resources/output_corrected/config.json",
        "dense_reference_ordering_fix": True,
        "production_svd_semantics": True,
        "notes": (
            "Corrected 4x4 TFIM + Heisenberg fixed-chi circuit benchmark vs "
            "dense identical second-order Trotter reference (dt=0.1, 30 steps, "
            "snake ordering, open BC, TDVP n=2 on long-range gates only, "
            "chi_main=32); validated by chi=256 control and deterministic repeat."
        ),
    },
    "svd_diagnostic": {
        "source": "experiments/fixed_resources/output_svd_diagnostic/",
        "generating_script": "experiments/fixed_resources/run_svd_cutoff_diagnostic.py",
        "config": "experiments/fixed_resources/output_svd_diagnostic/cutoff_semantics.json",
        "dense_reference_ordering_fix": True,
        "production_svd_semantics": True,
        "notes": (
            "Internal SVD-cutoff audit only; documents discarded_weight "
            "semantics, production threshold 1e-13, gate-library hard split "
            "cutoff 1e-14. Not part of the plotting pipeline."
        ),
    },
}

EXCLUDED = [
    {
        "path": "experiments/single_gate/archive/pre_repair_20260723T135839Z/",
        "reason": "pre-repair single-gate data (broken MPS.compress / zip-up / variational; TDVP n=64)",
    },
    {
        "path": "experiments/fixed_resources/archive/pre_repair_20260723T145513Z/",
        "reason": "pre-repair circuit data (artefactually short zip-up horizons)",
    },
    {
        "path": "save/long_range_gate_paper/ (4833-row trials.csv)",
        "reason": (
            "generated 2026-07-23 14:38, before the compress/SVD/variational repairs "
            "landed; affected data per affected_benchmark_inventory.md"
        ),
    },
    {
        "path": "save/single_gate_angle_full/ (3-gate, 3-seed v4 angle sweep)",
        "reason": "same pre-repair vintage; superseded by corrected campaign + paper_benchmarks extension",
    },
    {
        "path": "experiments/resource_frontier/",
        "reason": "resource-frontier framing dropped from the manuscript; still needs regeneration per inventory",
    },
    {
        "path": "experiments/convergence/",
        "reason": "superseded: TDVP circuit subdivision re-validated inside fixed_resources (n=2)",
    },
]


def main() -> int:
    files = []
    for group, meta in GROUPS.items():
        group_dir = RAW_DIR / group
        for path in sorted(group_dir.rglob("*")):
            if not path.is_file():
                continue
            files.append({
                "group": group,
                "path": str(path.relative_to(PB_DIR)),
                "source_path": meta["source"] + path.name,
                "sha256": sha256sum(path),
                "bytes": path.stat().st_size,
                "generating_script": meta["generating_script"],
                "dense_reference_ordering_fix": meta["dense_reference_ordering_fix"],
                "production_svd_semantics": meta["production_svd_semantics"],
            })
    if RAW_NEW_DIR.exists():
        for path in sorted(RAW_NEW_DIR.rglob("*")):
            if not path.is_file() or path.suffix in {".lock", ".log"}:
                continue
            files.append({
                "group": "raw_new",
                "path": str(path.relative_to(PB_DIR)),
                "source_path": "generated by paper_benchmarks/scripts/",
                "sha256": sha256sum(path),
                "bytes": path.stat().st_size,
                "generating_script": "paper_benchmarks/scripts/generate_*.py",
                "dense_reference_ordering_fix": True,
                "production_svd_semantics": True,
            })

    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "git_dirty_note": (
            "working tree contains uncommitted experiment changes; raw copies are "
            "checksummed here and write-protected under paper_benchmarks/raw/"
        ),
        "groups": GROUPS,
        "excluded_datasets": EXCLUDED,
        "files": files,
        "validation": "see paper_benchmarks/validation_report.json",
    }
    save_json(PB_DIR / "data_manifest.json", manifest)
    print(f"data_manifest.json written: {len(files)} files, commit {manifest['git_commit'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
