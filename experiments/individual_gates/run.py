# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Individual-gates publication campaign runner.

Usage:
    uv run python experiments/individual_gates/run.py --stage validate
    uv run python experiments/individual_gates/run.py --stage campaign
    uv run python experiments/individual_gates/run.py --stage cnot_rank
    uv run python experiments/individual_gates/run.py --stage refinement
    uv run python experiments/individual_gates/run.py --stage all
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common import (  # noqa: E402
    DirectoryLock,
    DiscardedWeightTracker,
    apply_cx_dense_qiskit,
    apply_gate_dense_yaqs,
    apply_method,
    final_max_bond,
    final_param_count,
    git_revision,
    git_revision_for_hash,
    load_json,
    make_cx_dag_node,
    make_cx_gate,
    make_pauli_dag_node,
    make_pauli_gate,
    mps_bond_profile,
    normalized_state_fidelity,
    numerical_settings_dict,
    package_versions,
    prepare_initial_state,
    save_json,
    state_distance,
    task_id_from_payload,
    utc_now,
)
from config import (  # noqa: E402
    CAMPAIGN_ID,
    CHI_MAX_VALUES,
    CNOT_ORIENTATIONS,
    CNOT_RANK_CHI_VALUES,
    CNOT_RANK_CONTROL,
    CNOT_RANK_SVD_THRESHOLD,
    CNOT_RANK_TARGET,
    CNOT_RANK_TDVP_N_SUB,
    CX_GENERATOR_BRANCH,
    DIRECT_METHODS,
    EXPECTED_CAMPAIGN_ROWS,
    EXPECTED_CNOT_RANK_ROWS,
    METHODS,
    N_SUB_MAIN,
    OUTPUT_DIR,
    PAULI_GATES,
    Q0,
    Q1,
    REFINEMENT_CHI,
    REFINEMENT_CONTROL,
    REFINEMENT_CONTROL_SEED,
    REFINEMENT_FINE_N_SUB,
    REFINEMENT_KRYLOV_CONTROL_TOL,
    REFINEMENT_N_SUB,
    REFINEMENT_SEEDS,
    REFINEMENT_SVD_THRESHOLD,
    REFINEMENT_TARGET,
    SEEDS,
    X_VALUES,
    N,
    theta_from_x,
)
from validate import run_validation  # noqa: E402

TASKS_DIR = OUTPUT_DIR / "tasks"
CAMPAIGN_CSV = OUTPUT_DIR / "campaign_rows.csv"
REFINEMENT_CSV = OUTPUT_DIR / "refinement_rows.csv"
CNOT_RANK_CSV = OUTPUT_DIR / "cnot_rank_rows.csv"

ROW_FIELDS = [
    "task_id",
    "family",
    "gate",
    "control",
    "target",
    "theta",
    "x",
    "seed",
    "chi_max",
    "method",
    "n_sub",
    "svd_threshold",
    "infidelity_normalized",
    "fidelity_normalized",
    "norm_approx",
    "norm_exact",
    "norm_drift",
    "discarded_weight",
    "final_bond_profile",
    "final_max_bond",
    "final_param_count",
    "min_kept_singular",
    "cap_reached",
    "positive_weight_truncated",
    "cap_truncation_occurred",
    "git_commit",
    "git_dirty",
    "git_diff_hash",
]

# Legacy → current column names for in-place CSV migration (no recompute).
_LEGACY_RENAMES = {
    "hard_cap_binding": "cap_reached",
    "peak_bond": "final_max_bond",
    "peak_param_count": "final_param_count",
}


def _task_path(task_id: str) -> Path:
    return TASKS_DIR / f"{task_id}.json"


def _already_done(task_id: str) -> bool:
    return _task_path(task_id).is_file()


def _write_task(task_id: str, row: dict[str, Any]) -> None:
    save_json(_task_path(task_id), row)


def _base_payload(**kwargs: Any) -> dict[str, Any]:
    payload = {
        "campaign_id": CAMPAIGN_ID,
        "git": git_revision_for_hash(),
        "versions": package_versions(),
        "cx_generator_branch": CX_GENERATOR_BRANCH,
        "sites_paper_one_based": [Q0 + 1, Q1 + 1],
    }
    payload.update(kwargs)
    return payload


def _metric_row(
    *,
    task_id: str,
    family: str,
    gate: str,
    control: int | None,
    target: int | None,
    theta: float | None,
    x: float | None,
    seed: int,
    chi: int,
    method: str,
    n_sub: int,
    svd_threshold: float,
    exact_vec: np.ndarray,
    final_mps,
    discarded: float,
    tracker: DiscardedWeightTracker | None,
) -> dict[str, Any]:
    final_vec = final_mps.to_vec().astype(np.complex128)
    metrics = normalized_state_fidelity(exact_vec, final_vec)
    profile = mps_bond_profile(final_mps)
    min_sing = min(tracker.min_kept_singular) if tracker and tracker.min_kept_singular else float("nan")
    cap_reached = any(d == chi for d in profile[1:-1])
    git = git_revision()
    if tracker is not None:
        pos_trunc = tracker.positive_weight_truncated
        cap_trunc = tracker.cap_truncation_occurred
    else:
        pos_trunc = bool(discarded == discarded and discarded > 0.0)
        cap_trunc = ""
    return {
        "task_id": task_id,
        "family": family,
        "gate": gate,
        "control": "" if control is None else int(control),
        "target": "" if target is None else int(target),
        "theta": "" if theta is None else float(theta),
        "x": "" if x is None else float(x),
        "seed": int(seed),
        "chi_max": int(chi),
        "method": method,
        "n_sub": int(n_sub),
        "svd_threshold": float(svd_threshold),
        "infidelity_normalized": metrics["infidelity_normalized"],
        "fidelity_normalized": metrics["fidelity_normalized"],
        "norm_approx": metrics["norm_approx"],
        "norm_exact": metrics["norm_exact"],
        "norm_drift": metrics["norm_drift"],
        "discarded_weight": float(discarded) if discarded == discarded else 0.0,
        "final_bond_profile": json.dumps(profile),
        "final_max_bond": final_max_bond(profile),
        "final_param_count": final_param_count(profile),
        "min_kept_singular": float(min_sing),
        "cap_reached": bool(cap_reached),
        "positive_weight_truncated": bool(pos_trunc),
        "cap_truncation_occurred": cap_trunc if cap_trunc == "" else bool(cap_trunc),
        "git_commit": git["git_commit"],
        "git_dirty": git["git_dirty"],
        "git_diff_hash": git["git_diff_hash"],
    }


def migrate_row_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Rename legacy metric fields and fill truncation diagnostics when possible."""
    out = dict(row)
    for old, new in _LEGACY_RENAMES.items():
        if old in out and new not in out:
            out[new] = out.pop(old)
        elif old in out:
            out.pop(old)
    if "positive_weight_truncated" not in out:
        try:
            dw = float(out.get("discarded_weight", 0.0) or 0.0)
        except (TypeError, ValueError):
            dw = 0.0
        out["positive_weight_truncated"] = bool(dw > 0.0)
    if "cap_truncation_occurred" not in out:
        # Unknown for legacy rows: do not infer from cap_reached alone.
        out["cap_truncation_occurred"] = ""
    git = git_revision()
    out.setdefault("git_diff_hash", git.get("git_diff_hash", "unavailable"))
    return out


def migrate_campaign_csv() -> None:
    """Rewrite ``campaign_rows.csv`` with corrected field names (no recompute)."""
    if not CAMPAIGN_CSV.is_file():
        return
    with CAMPAIGN_CSV.open(encoding="utf-8", newline="") as fh:
        rows = [migrate_row_fields(r) for r in csv.DictReader(fh)]
    _write_csv(CAMPAIGN_CSV, rows, ROW_FIELDS)


def iter_campaign_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for gate in PAULI_GATES:
        for seed in SEEDS:
            for chi in CHI_MAX_VALUES:
                for x in X_VALUES:
                    theta = theta_from_x(x)
                    for method in METHODS:
                        specs.append(
                            {
                                "family": "pauli",
                                "gate": gate,
                                "control": None,
                                "target": None,
                                "q0": Q0,
                                "q1": Q1,
                                "theta": theta,
                                "x": float(x),
                                "seed": int(seed),
                                "chi": int(chi),
                                "method": method,
                                "n_sub": N_SUB_MAIN,
                                "svd_threshold": None,
                            }
                        )
    for control, target in CNOT_ORIENTATIONS:
        for seed in SEEDS:
            for chi in CHI_MAX_VALUES:
                for method in METHODS:
                    specs.append(
                        {
                            "family": "cnot",
                            "gate": "cx",
                            "control": int(control),
                            "target": int(target),
                            "q0": int(control),
                            "q1": int(target),
                            "theta": None,
                            "x": None,
                            "seed": int(seed),
                            "chi": int(chi),
                            "method": method,
                            "n_sub": N_SUB_MAIN,
                            "svd_threshold": None,
                        }
                    )
    return specs


def iter_cnot_rank_specs() -> list[dict[str, Any]]:
    """Fresh effective-zero CNOT-versus-χ_max specs (90 rows)."""
    specs: list[dict[str, Any]] = []
    control, target = CNOT_RANK_CONTROL, CNOT_RANK_TARGET
    for chi in CNOT_RANK_CHI_VALUES:
        for seed in SEEDS:
            for method in DIRECT_METHODS:
                specs.append(
                    {
                        "family": "cnot_rank",
                        "gate": "cx",
                        "control": control,
                        "target": target,
                        "seed": int(seed),
                        "chi": int(chi),
                        "method": method,
                        "n_sub": N_SUB_MAIN,
                        "svd_threshold": CNOT_RANK_SVD_THRESHOLD,
                        "resolution_label": "direct",
                    }
                )
            for n_sub in CNOT_RANK_TDVP_N_SUB:
                label = "fine_resolution" if n_sub == 256 else f"n_sub_{n_sub}"
                specs.append(
                    {
                        "family": "cnot_rank",
                        "gate": "cx",
                        "control": control,
                        "target": target,
                        "seed": int(seed),
                        "chi": int(chi),
                        "method": "gate_local_2tdvp",
                        "n_sub": int(n_sub),
                        "svd_threshold": CNOT_RANK_SVD_THRESHOLD,
                        "resolution_label": label,
                    }
                )
    return specs


def _exact_vec_for_spec(spec: dict[str, Any], init: dict[str, Any]) -> np.ndarray:
    if spec["family"] == "pauli":
        gate = make_pauli_gate(spec["gate"], float(spec["theta"]), Q0, Q1)
        return apply_gate_dense_yaqs(init["vec"], N, Q0, Q1, gate)
    gate = make_cx_gate(int(spec["control"]), int(spec["target"]))
    return apply_gate_dense_yaqs(init["vec"], N, int(spec["control"]), int(spec["target"]), gate)


def _node_for_spec(spec: dict[str, Any]):
    if spec["family"] == "pauli":
        return make_pauli_dag_node(spec["gate"], float(spec["theta"]), Q0, Q1)
    return make_cx_dag_node(int(spec["control"]), int(spec["target"]))


def run_campaign(*, resume: bool = True) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    specs = iter_campaign_specs()
    if len(specs) != EXPECTED_CAMPAIGN_ROWS:
        msg = f"Spec count {len(specs)} != expected {EXPECTED_CAMPAIGN_ROWS}"
        raise RuntimeError(msg)

    # Prefer migrating the existing authoritative CSV over recomputing Pauli rows.
    if resume and CAMPAIGN_CSV.is_file():
        with CAMPAIGN_CSV.open(encoding="utf-8", newline="") as fh:
            existing = list(csv.DictReader(fh))
        if len(existing) == EXPECTED_CAMPAIGN_ROWS:
            rows = [migrate_row_fields(r) for r in existing]
            _write_csv(CAMPAIGN_CSV, rows, ROW_FIELDS)
            summary = {
                "stage": "campaign",
                "n_rows": len(rows),
                "n_new": 0,
                "n_resumed": len(rows),
                "csv": str(CAMPAIGN_CSV),
                "migrated_fields_only": True,
                "git": git_revision(),
                "note": (
                    "Preserved existing campaign CSV (no Pauli recomputation). "
                    "Rows retain their recorded Git provenance."
                ),
            }
            save_json(OUTPUT_DIR / "campaign_summary.json", summary)
            return summary

    initials = {seed: prepare_initial_state(seed) for seed in SEEDS}
    rows: list[dict[str, Any]] = []
    n_new = 0
    n_skip = 0

    for spec in specs:
        settings = numerical_settings_dict(
            chi=spec["chi"],
            method=spec["method"],
            n_sub=spec["n_sub"],
            svd_threshold=spec["svd_threshold"],
        )
        payload = _base_payload(
            stage="campaign",
            family=spec["family"],
            gate=spec["gate"],
            control=spec["control"],
            target=spec["target"],
            theta=spec["theta"],
            x=spec["x"],
            seed=spec["seed"],
            settings=settings,
        )
        tid = task_id_from_payload(payload)
        if resume and _already_done(tid):
            stored = load_json(_task_path(tid))
            rows.append(migrate_row_fields(stored["row"]))
            n_skip += 1
            continue

        init = initials[spec["seed"]]
        exact = _exact_vec_for_spec(spec, init)
        node = _node_for_spec(spec)
        tracker = DiscardedWeightTracker()
        final_mps, discarded = apply_method(
            init["mps"],
            node,
            method=spec["method"],
            chi=spec["chi"],
            n_sub=spec["n_sub"],
            svd_threshold=spec["svd_threshold"],
            tracker=tracker,
        )
        thr = settings["svd_threshold"]
        row = _metric_row(
            task_id=tid,
            family=spec["family"],
            gate=spec["gate"],
            control=spec["control"],
            target=spec["target"],
            theta=spec["theta"],
            x=spec["x"],
            seed=spec["seed"],
            chi=spec["chi"],
            method=spec["method"],
            n_sub=spec["n_sub"],
            svd_threshold=thr,
            exact_vec=exact,
            final_mps=final_mps,
            discarded=discarded,
            tracker=tracker,
        )
        _write_task(tid, {"payload": payload, "row": row, "completed_utc": utc_now()})
        rows.append(row)
        n_new += 1

    _write_csv(CAMPAIGN_CSV, rows, ROW_FIELDS)
    ids = [r["task_id"] for r in rows]
    if len(ids) != len(set(ids)):
        msg = "Duplicate task_ids in campaign rows"
        raise RuntimeError(msg)
    if len(rows) != EXPECTED_CAMPAIGN_ROWS:
        msg = f"Campaign rows {len(rows)} != {EXPECTED_CAMPAIGN_ROWS}"
        raise RuntimeError(msg)

    summary = {
        "stage": "campaign",
        "n_rows": len(rows),
        "n_new": n_new,
        "n_resumed": n_skip,
        "csv": str(CAMPAIGN_CSV),
        "git": git_revision(),
        "note": "Rows retain their recorded Git provenance.",
    }
    save_json(OUTPUT_DIR / "campaign_summary.json", summary)
    return summary


def run_cnot_rank(*, resume: bool = True) -> dict[str, Any]:
    """Fresh effective-zero CNOT-versus-χ_max dataset (90 rows)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    specs = iter_cnot_rank_specs()
    if len(specs) != EXPECTED_CNOT_RANK_ROWS:
        msg = f"CNOT-rank spec count {len(specs)} != {EXPECTED_CNOT_RANK_ROWS}"
        raise RuntimeError(msg)

    initials = {seed: prepare_initial_state(seed) for seed in SEEDS}
    exacts = {
        seed: apply_gate_dense_yaqs(
            initials[seed]["vec"],
            N,
            CNOT_RANK_CONTROL,
            CNOT_RANK_TARGET,
            make_cx_gate(CNOT_RANK_CONTROL, CNOT_RANK_TARGET),
        )
        for seed in SEEDS
    }
    node = make_cx_dag_node(CNOT_RANK_CONTROL, CNOT_RANK_TARGET)

    rows: list[dict[str, Any]] = []
    state_cache: dict[tuple[int, int, int], np.ndarray] = {}
    n_new = 0
    n_skip = 0

    for spec in specs:
        settings = numerical_settings_dict(
            chi=spec["chi"],
            method=spec["method"],
            n_sub=spec["n_sub"],
            svd_threshold=spec["svd_threshold"],
        )
        payload = _base_payload(
            stage="cnot_rank",
            family=spec["family"],
            gate=spec["gate"],
            control=spec["control"],
            target=spec["target"],
            seed=spec["seed"],
            settings=settings,
            resolution_label=spec["resolution_label"],
        )
        tid = task_id_from_payload(payload)
        if resume and _already_done(tid):
            stored = load_json(_task_path(tid))
            row = migrate_row_fields(stored["row"])
            rows.append(row)
            npy = OUTPUT_DIR / "cnot_rank_states" / f"{tid}.npy"
            if npy.is_file() and spec["method"] == "gate_local_2tdvp":
                state_cache[(spec["seed"], spec["chi"], spec["n_sub"])] = np.load(npy)
            n_skip += 1
            continue

        init = initials[spec["seed"]]
        tracker = DiscardedWeightTracker()
        final_mps, discarded = apply_method(
            init["mps"],
            node,
            method=spec["method"],
            chi=spec["chi"],
            n_sub=spec["n_sub"],
            svd_threshold=spec["svd_threshold"],
            tracker=tracker,
        )
        row = _metric_row(
            task_id=tid,
            family="cnot_rank",
            gate="cx",
            control=spec["control"],
            target=spec["target"],
            theta=None,
            x=None,
            seed=spec["seed"],
            chi=spec["chi"],
            method=spec["method"],
            n_sub=spec["n_sub"],
            svd_threshold=float(spec["svd_threshold"]),
            exact_vec=exacts[spec["seed"]],
            final_mps=final_mps,
            discarded=discarded,
            tracker=tracker,
        )
        row["resolution_label"] = spec["resolution_label"]
        _write_task(tid, {"payload": payload, "row": row, "completed_utc": utc_now()})
        if spec["method"] == "gate_local_2tdvp":
            npy = OUTPUT_DIR / "cnot_rank_states" / f"{tid}.npy"
            npy.parent.mkdir(parents=True, exist_ok=True)
            vec = final_mps.to_vec().astype(np.complex128)
            np.save(npy, vec)
            state_cache[(spec["seed"], spec["chi"], spec["n_sub"])] = vec
        rows.append(row)
        n_new += 1

    # Phase-aligned distances between n=128 and fine-resolution n=256.
    distances_128_256: list[dict[str, Any]] = []
    for chi in CNOT_RANK_CHI_VALUES:
        for seed in SEEDS:
            v128 = state_cache.get((seed, chi, 128))
            v256 = state_cache.get((seed, chi, 256))
            if v128 is None or v256 is None:
                # Attempt reload from disk via task scan.
                for row in rows:
                    if row["method"] == "gate_local_2tdvp" and int(row["seed"]) == seed and int(row["chi_max"]) == chi:
                        npy = OUTPUT_DIR / "cnot_rank_states" / f"{row['task_id']}.npy"
                        if npy.is_file():
                            state_cache[(seed, chi, int(row["n_sub"]))] = np.load(npy)
                v128 = state_cache.get((seed, chi, 128))
                v256 = state_cache.get((seed, chi, 256))
            dist = float("nan") if v128 is None or v256 is None else state_distance(v128, v256)
            distances_128_256.append(
                {
                    "seed": seed,
                    "chi_max": chi,
                    "phase_aligned_distance_n128_n256": dist,
                    "note": "Fine-resolution control only; small distance does not imply convergence.",
                }
            )

    cnot_fields = [*ROW_FIELDS, "resolution_label"]
    _write_csv(CNOT_RANK_CSV, rows, cnot_fields)
    if len(rows) != EXPECTED_CNOT_RANK_ROWS:
        msg = f"CNOT-rank rows {len(rows)} != {EXPECTED_CNOT_RANK_ROWS}"
        raise RuntimeError(msg)

    summary = {
        "stage": "cnot_rank",
        "n_rows": len(rows),
        "n_new": n_new,
        "n_resumed": n_skip,
        "csv": str(CNOT_RANK_CSV),
        "control": CNOT_RANK_CONTROL,
        "target": CNOT_RANK_TARGET,
        "sites_paper_one_based": [CNOT_RANK_CONTROL + 1, CNOT_RANK_TARGET + 1],
        "chi_max_values": list(CNOT_RANK_CHI_VALUES),
        "tdvp_n_sub": list(CNOT_RANK_TDVP_N_SUB),
        "svd_threshold": CNOT_RANK_SVD_THRESHOLD,
        "distances_n128_n256": distances_128_256,
        "git": git_revision(),
        "note": "Separate from production-threshold campaign CNOT rows; do not merge.",
    }
    save_json(OUTPUT_DIR / "cnot_rank_summary.json", summary)
    return summary


def run_refinement(*, resume: bool = True) -> dict[str, Any]:
    """CNOT refinement diagnostic over all campaign seeds (forward orientation, χ=8)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    gate = make_cx_gate(REFINEMENT_CONTROL, REFINEMENT_TARGET)
    node = make_cx_dag_node(REFINEMENT_CONTROL, REFINEMENT_TARGET)
    n_values = list(REFINEMENT_N_SUB) + [REFINEMENT_FINE_N_SUB]
    rows: list[dict[str, Any]] = []
    seed_summaries: dict[str, dict[str, float]] = {}

    for seed in REFINEMENT_SEEDS:
        init = prepare_initial_state(seed)
        exact = apply_gate_dense_yaqs(init["vec"], N, REFINEMENT_CONTROL, REFINEMENT_TARGET, gate)
        qiskit = apply_cx_dense_qiskit(init["vec"], REFINEMENT_CONTROL, REFINEMENT_TARGET)
        if float(np.linalg.norm(exact - qiskit)) > 1e-12:
            msg = f"Refinement exact endpoint disagrees with Qiskit for seed {seed}"
            raise RuntimeError(msg)

        state_dir = OUTPUT_DIR / "refinement_states" / f"seed_{seed}"
        state_dir.mkdir(parents=True, exist_ok=True)
        results: dict[int, dict[str, Any]] = {}

        for n_sub in n_values:
            settings = numerical_settings_dict(
                chi=REFINEMENT_CHI,
                method="gate_local_2tdvp",
                n_sub=n_sub,
                svd_threshold=REFINEMENT_SVD_THRESHOLD,
            )
            payload = _base_payload(
                stage="refinement",
                family="cnot_refinement",
                gate="cx",
                control=REFINEMENT_CONTROL,
                target=REFINEMENT_TARGET,
                seed=seed,
                settings=settings,
            )
            tid = task_id_from_payload(payload)
            npy_path = state_dir / f"n{n_sub}.npy"
            if resume and _already_done(tid) and npy_path.is_file():
                results[n_sub] = load_json(_task_path(tid))
                continue

            tracker = DiscardedWeightTracker()
            final_mps, discarded = apply_method(
                init["mps"],
                node,
                method="gate_local_2tdvp",
                chi=REFINEMENT_CHI,
                n_sub=n_sub,
                svd_threshold=REFINEMENT_SVD_THRESHOLD,
                tracker=tracker,
            )
            vec = final_mps.to_vec().astype(np.complex128)
            metrics = normalized_state_fidelity(exact, vec)
            profile = mps_bond_profile(final_mps)
            record = {
                "payload": payload,
                "task_id": tid,
                "n_sub": n_sub,
                "row": {
                    "task_id": tid,
                    "seed": seed,
                    "n_sub": n_sub,
                    "infidelity_vs_exact": metrics["infidelity_normalized"],
                    "norm_drift": metrics["norm_drift"],
                    "discarded_weight": float(discarded) if discarded == discarded else 0.0,
                    "final_bond_profile": json.dumps(profile),
                    "final_max_bond": final_max_bond(profile),
                    "min_kept_singular": min(tracker.min_kept_singular) if tracker.min_kept_singular else float("nan"),
                    "cap_reached": any(d == REFINEMENT_CHI for d in profile[1:-1]),
                    "positive_weight_truncated": tracker.positive_weight_truncated,
                    "cap_truncation_occurred": tracker.cap_truncation_occurred,
                    "distance_to_finest": None,
                    "adjacent_refinement_distance": None,
                },
                "completed_utc": utc_now(),
            }
            _write_task(
                tid,
                record
                | {
                    "vec_norm": float(np.linalg.norm(vec)),
                    "vec_hash": task_id_from_payload({"re": vec.real.tolist(), "im": vec.imag.tolist()}),
                },
            )
            np.save(npy_path, vec)
            results[n_sub] = record

        fine = np.load(state_dir / f"n{REFINEMENT_FINE_N_SUB}.npy")
        for n_sub in REFINEMENT_N_SUB:
            rec = results[n_sub]
            vec = np.load(state_dir / f"n{n_sub}.npy")
            row = migrate_row_fields(dict(rec["row"]))
            row["distance_to_finest"] = state_distance(fine, vec)
            if 2 * n_sub in set(n_values):
                vec2 = np.load(state_dir / f"n{2 * n_sub}.npy")
                row["adjacent_refinement_distance"] = state_distance(vec, vec2)
            else:
                row["adjacent_refinement_distance"] = ""
            rows.append(row)

        fine_rec = results[REFINEMENT_FINE_N_SUB]
        fine_row = migrate_row_fields(dict(fine_rec["row"]))
        fine_row["distance_to_finest"] = 0.0
        # No 2n partner for n=1024; successive distance lives on the n=512 row.
        fine_row["adjacent_refinement_distance"] = ""
        n512 = np.load(state_dir / "n512.npy")
        adjacent_512_1024 = state_distance(n512, fine)
        rows.append(fine_row)
        seed_summaries[str(seed)] = {"adjacent_distance_n512_n1024": adjacent_512_1024}

    fields = [
        "task_id",
        "seed",
        "n_sub",
        "infidelity_vs_exact",
        "norm_drift",
        "discarded_weight",
        "final_bond_profile",
        "final_max_bond",
        "min_kept_singular",
        "cap_reached",
        "positive_weight_truncated",
        "cap_truncation_occurred",
        "distance_to_finest",
        "adjacent_refinement_distance",
    ]
    _write_csv(REFINEMENT_CSV, rows, fields)

    # Tight-Krylov control at n=1024, tol=1e-14.
    control_init = prepare_initial_state(REFINEMENT_CONTROL_SEED)
    control_exact = apply_gate_dense_yaqs(
        control_init["vec"], N, REFINEMENT_CONTROL, REFINEMENT_TARGET, gate
    )
    control_state_dir = OUTPUT_DIR / "refinement_states" / f"seed_{REFINEMENT_CONTROL_SEED}"
    krylov_control = _run_refinement_krylov_control(
        control_init,
        node,
        control_exact,
        seed=REFINEMENT_CONTROL_SEED,
        state_dir=control_state_dir,
        resume=resume,
    )

    summary = {
        "stage": "refinement",
        "seeds": list(REFINEMENT_SEEDS),
        "chi_max": REFINEMENT_CHI,
        "control": REFINEMENT_CONTROL,
        "target": REFINEMENT_TARGET,
        "sites_paper_one_based": [REFINEMENT_CONTROL + 1, REFINEMENT_TARGET + 1],
        "svd_threshold": REFINEMENT_SVD_THRESHOLD,
        "n_sub_values": list(n_values),
        "csv": str(REFINEMENT_CSV),
        "seed_summaries": seed_summaries,
        "note": (
            "n=1024 is a fine-resolution endpoint only; all displayed "
            "statistics aggregate the three campaign seeds."
        ),
        "krylov_control": krylov_control,
        "git": git_revision(),
    }
    save_json(OUTPUT_DIR / "refinement_summary.json", summary)
    save_json(OUTPUT_DIR / "refinement_controls.json", krylov_control)
    return summary


def _run_refinement_krylov_control(
    init: dict[str, Any],
    node,
    exact: np.ndarray,
    *,
    seed: int,
    state_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    """Archive n=1024 TDVP with krylov_tol=1e-14 vs the 1e-12 result."""
    settings = numerical_settings_dict(
        chi=REFINEMENT_CHI,
        method="gate_local_2tdvp",
        n_sub=REFINEMENT_FINE_N_SUB,
        svd_threshold=REFINEMENT_SVD_THRESHOLD,
        krylov_tol=REFINEMENT_KRYLOV_CONTROL_TOL,
    )
    payload = _base_payload(
        stage="refinement_krylov_control",
        family="cnot_refinement",
        gate="cx",
        control=REFINEMENT_CONTROL,
        target=REFINEMENT_TARGET,
        seed=seed,
        settings=settings,
    )
    tid = task_id_from_payload(payload)
    npy_path = state_dir / f"n{REFINEMENT_FINE_N_SUB}_krylov1e-14.npy"
    if resume and _already_done(tid) and npy_path.is_file():
        vec = np.load(npy_path)
        stored = load_json(_task_path(tid))
    else:
        tracker = DiscardedWeightTracker()
        final_mps, discarded = apply_method(
            init["mps"],
            node,
            method="gate_local_2tdvp",
            chi=REFINEMENT_CHI,
            n_sub=REFINEMENT_FINE_N_SUB,
            svd_threshold=REFINEMENT_SVD_THRESHOLD,
            krylov_tol=REFINEMENT_KRYLOV_CONTROL_TOL,
            tracker=tracker,
        )
        vec = final_mps.to_vec().astype(np.complex128)
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, vec)
        metrics = normalized_state_fidelity(exact, vec)
        stored = {
            "payload": payload,
            "task_id": tid,
            "infidelity_vs_exact": metrics["infidelity_normalized"],
            "discarded_weight": float(discarded) if discarded == discarded else 0.0,
            "completed_utc": utc_now(),
        }
        _write_task(tid, stored)

    baseline = np.load(state_dir / f"n{REFINEMENT_FINE_N_SUB}.npy")
    return {
        "n_sub": REFINEMENT_FINE_N_SUB,
        "chi_max": REFINEMENT_CHI,
        "seed": seed,
        "svd_threshold": REFINEMENT_SVD_THRESHOLD,
        "krylov_tol_control": REFINEMENT_KRYLOV_CONTROL_TOL,
        "krylov_tol_baseline": 1e-12,
        "phase_aligned_distance_to_baseline_1e-12": state_distance(baseline, vec),
        "infidelity_vs_exact": stored.get("infidelity_vs_exact"),
        "task_id": tid,
        "git": git_revision(),
    }


def write_manifest() -> dict[str, Any]:
    git = git_revision()
    manifest = {
        "campaign_id": CAMPAIGN_ID,
        "completed_utc": utc_now(),
        "git": git,
        "git_diff_hash": git["git_diff_hash"],
        "versions": package_versions(),
        "cx_generator_branch": CX_GENERATOR_BRANCH,
        "artifacts": {
            "campaign_rows": str(CAMPAIGN_CSV) if CAMPAIGN_CSV.is_file() else None,
            "cnot_rank_rows": str(CNOT_RANK_CSV) if CNOT_RANK_CSV.is_file() else None,
            "refinement_rows": str(REFINEMENT_CSV) if REFINEMENT_CSV.is_file() else None,
            "refinement_controls": str(OUTPUT_DIR / "refinement_controls.json"),
            "validation_report": str(OUTPUT_DIR / "validation_report.json"),
        },
    }
    save_json(OUTPUT_DIR / "manifest.json", manifest)
    save_json(OUTPUT_DIR / "meta.json", manifest)
    return manifest


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("validate", "campaign", "cnot_rank", "refinement", "migrate", "all"),
        required=True,
    )
    parser.add_argument("--no-resume", action="store_true", help="Recompute tasks even if present")
    args = parser.parse_args(argv)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lock = DirectoryLock(OUTPUT_DIR)
    lock.acquire()
    try:
        resume = not args.no_resume
        if args.stage == "migrate":
            migrate_campaign_csv()
            print(f"Migrated field names in {CAMPAIGN_CSV}")
            write_manifest()
            return 0
        if args.stage in {"validate", "all"}:
            print("Stage: validate")
            report = run_validation()
            save_json(OUTPUT_DIR / "validation_report.json", report)
            print("Validation PASS")
        if args.stage in {"campaign", "all"}:
            print("Stage: campaign (resume; will not recompute existing Pauli/CNOT tasks)")
            summary = run_campaign(resume=resume)
            print(f"Campaign done: {summary['n_rows']} rows ({summary['n_new']} new, {summary['n_resumed']} resumed)")
        if args.stage in {"cnot_rank", "all"}:
            print("Stage: cnot_rank")
            summary = run_cnot_rank(resume=resume)
            print(f"CNOT-rank done: {summary['n_rows']} rows ({summary['n_new']} new, {summary['n_resumed']} resumed)")
        if args.stage in {"refinement", "all"}:
            print("Stage: refinement")
            summary = run_refinement(resume=resume)
            print(f"Refinement done: wrote {summary['csv']}")
        if args.stage == "all":
            print("Stage: validate (post-data)")
            report = run_validation()
            save_json(OUTPUT_DIR / "validation_report.json", report)
            print("Post-data validation PASS")
        write_manifest()
    finally:
        lock.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
