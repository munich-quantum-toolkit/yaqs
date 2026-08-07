# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Matched variational-MPO accuracy control for the displayed gate cases.

This campaign writes separate source data and never enters the plotting method
list.  It compares variational endpoint compression with the ordinary MPO
contract-and-truncate initializer on exactly the cells represented in Fig. 1.

Run from the repository root with::

    uv run python -m experiments.individual_gates.variational_control
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
for _path in (_REPO_ROOT, _HERE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common import (  # noqa: E402
    apply_gate_dense_yaqs,
    apply_method,
    digital_params,
    git_revision,
    make_cx_dag_node,
    make_cx_gate,
    make_pauli_dag_node,
    make_pauli_gate,
    mps_bond_profile,
    normalized_state_fidelity,
    prepare_initial_state,
    state_distance,
    task_id_from_payload,
)
from config import (  # noqa: E402
    CNOT_RANK_CHI_VALUES,
    CNOT_RANK_CONTROL,
    CNOT_RANK_SVD_THRESHOLD,
    CNOT_RANK_TARGET,
    PAULI_GATES,
    Q0,
    Q1,
    SEEDS,
    SVD_THRESHOLD,
    TRUNC_MODE,
    X_VALUES,
    N,
    theta_from_x,
)

from experiments.variational_mpo import apply_variational_mpo_node  # noqa: E402

OUTPUT_DIR = _HERE / "output" / "variational_mpo_control"
TASKS_DIR = OUTPUT_DIR / "single_gate_tasks"
ROWS_PATH = OUTPUT_DIR / "single_gate_rows.csv"
SUMMARY_PATH = OUTPUT_DIR / "single_gate_summary.json"
SUMMARY_MD_PATH = OUTPUT_DIR / "single_gate_summary.md"
CAMPAIGN_ID = "variational_mpo_single_gate_control_v1"
MAX_SWEEPS = 32
EQUALITY_TOLERANCE = 1e-12
MONOTONICITY_TOLERANCE = 2e-12

FIELDS = (
    "task_id",
    "family",
    "gate",
    "control",
    "target",
    "theta",
    "x",
    "seed",
    "chi_max",
    "svd_threshold",
    "mpo_infidelity",
    "variational_infidelity",
    "variational_minus_mpo",
    "state_distance_to_mpo",
    "variational_norm",
    "mpo_norm",
    "variational_bond_profile",
    "mpo_bond_profile",
    "objective_initial",
    "objective_final",
    "mpo_initializer_objective",
    "input_initializer_objective",
    "initializer_runtimes_s",
    "best_initializer",
    "sweeps",
    "converged",
    "rejected_nonimproving_updates",
    "objective_trace",
    "update_trace",
    "target_max_bond",
    "target_parameter_count",
    "variational_runtime_s",
    "fidelity_to_target",
    "core_source_sha256",
    "control_source_sha256",
    "git_commit",
    "git_dirty",
    "git_diff_hash",
)


def _core_source_hash() -> str:
    from experiments import variational_mpo

    return hashlib.sha256(Path(variational_mpo.__file__).read_bytes()).hexdigest()


def _control_source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _normalized_state_distance(first: np.ndarray, second: np.ndarray) -> float:
    first_vector = np.asarray(first, dtype=np.complex128).reshape(-1)
    second_vector = np.asarray(second, dtype=np.complex128).reshape(-1)
    first_vector = first_vector / np.linalg.norm(first_vector)
    second_vector = second_vector / np.linalg.norm(second_vector)
    return state_distance(first_vector, second_vector)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for gate in PAULI_GATES:
        for seed in SEEDS:
            for x_value in X_VALUES:
                specs.append(
                    {
                        "family": "pauli",
                        "gate": gate,
                        "control": "",
                        "target": "",
                        "q0": Q0,
                        "q1": Q1,
                        "theta": theta_from_x(x_value),
                        "x": float(x_value),
                        "seed": int(seed),
                        "chi_max": 8,
                        "svd_threshold": SVD_THRESHOLD,
                    }
                )
    for seed in SEEDS:
        for chi in CNOT_RANK_CHI_VALUES:
            specs.append(
                {
                    "family": "cnot_rank",
                    "gate": "cx",
                    "control": CNOT_RANK_CONTROL,
                    "target": CNOT_RANK_TARGET,
                    "q0": CNOT_RANK_CONTROL,
                    "q1": CNOT_RANK_TARGET,
                    "theta": "",
                    "x": "",
                    "seed": int(seed),
                    "chi_max": int(chi),
                    "svd_threshold": CNOT_RANK_SVD_THRESHOLD,
                }
            )
    if len(specs) != 87:
        msg = f"Expected 87 displayed control cells, got {len(specs)}."
        raise RuntimeError(msg)
    return specs


def _gate_and_node(spec: dict[str, Any]):
    if spec["family"] == "pauli":
        gate = make_pauli_gate(spec["gate"], float(spec["theta"]), spec["q0"], spec["q1"])
        node = make_pauli_dag_node(spec["gate"], float(spec["theta"]), spec["q0"], spec["q1"], N)
        return gate, node
    gate = make_cx_gate(spec["control"], spec["target"])
    node = make_cx_dag_node(spec["control"], spec["target"], N)
    return gate, node


def _payload(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "campaign_id": CAMPAIGN_ID,
        "spec": spec,
        "max_sweeps": MAX_SWEEPS,
        "trunc_mode": TRUNC_MODE,
        "core_source_sha256": _core_source_hash(),
        "control_source_sha256": _control_source_hash(),
        "git": git_revision(),
    }


def _run_one(spec: dict[str, Any], initial: dict[str, Any], task_id: str) -> dict[str, Any]:
    gate, node = _gate_and_node(spec)
    exact = apply_gate_dense_yaqs(initial["vec"], N, spec["q0"], spec["q1"], gate)
    chi = int(spec["chi_max"])
    threshold = float(spec["svd_threshold"])

    mpo_state, _ = apply_method(
        initial["mps"],
        node,
        method="mpo_zipup",
        chi=chi,
        n_sub=1,
        svd_threshold=threshold,
    )
    compression_params = digital_params(
        chi,
        method="mpo_zipup",
        n_sub=1,
        svd_threshold=threshold,
    )
    result = apply_variational_mpo_node(
        initial["mps"],
        node,
        compression_params=compression_params,
        max_sweeps=MAX_SWEEPS,
    )
    if not result.converged:
        msg = f"Variational fit did not converge for task {task_id}."
        raise RuntimeError(msg)

    mpo_metrics = normalized_state_fidelity(exact, mpo_state.to_vec())
    variational_metrics = normalized_state_fidelity(exact, result.state.to_vec())
    delta = variational_metrics["infidelity_normalized"] - mpo_metrics["infidelity_normalized"]
    if delta > EQUALITY_TOLERANCE:
        msg = f"Variational infidelity exceeds its MPO initializer by {delta:.3e} for task {task_id}."
        raise RuntimeError(msg)
    if any(np.diff(result.objective_trace) > MONOTONICITY_TOLERANCE) or any(
        np.diff(result.update_trace) > MONOTONICITY_TOLERANCE
    ):
        msg = f"Nonmonotone accepted objective trace for task {task_id}."
        raise RuntimeError(msg)

    git = git_revision()
    return {
        "task_id": task_id,
        "family": spec["family"],
        "gate": spec["gate"],
        "control": spec["control"],
        "target": spec["target"],
        "theta": spec["theta"],
        "x": spec["x"],
        "seed": spec["seed"],
        "chi_max": chi,
        "svd_threshold": threshold,
        "mpo_infidelity": mpo_metrics["infidelity_normalized"],
        "variational_infidelity": variational_metrics["infidelity_normalized"],
        "variational_minus_mpo": delta,
        "state_distance_to_mpo": _normalized_state_distance(mpo_state.to_vec(), result.state.to_vec()),
        "variational_norm": variational_metrics["norm_approx"],
        "mpo_norm": mpo_metrics["norm_approx"],
        "variational_bond_profile": json.dumps(mps_bond_profile(result.state)),
        "mpo_bond_profile": json.dumps(mps_bond_profile(mpo_state)),
        "objective_initial": result.objective_initial,
        "objective_final": result.objective_final,
        "mpo_initializer_objective": result.initializer_objectives["mpo_contract_compress"],
        "input_initializer_objective": result.initializer_objectives["input"],
        "initializer_runtimes_s": json.dumps(result.initializer_runtimes_s, sort_keys=True),
        "best_initializer": result.best_initializer,
        "sweeps": result.sweeps,
        "converged": result.converged,
        "rejected_nonimproving_updates": result.rejected_nonimproving_updates,
        "objective_trace": json.dumps(result.objective_trace),
        "update_trace": json.dumps(result.update_trace),
        "target_max_bond": result.target_max_bond,
        "target_parameter_count": result.target_parameter_count,
        "variational_runtime_s": result.runtime_s,
        "fidelity_to_target": result.fidelity_to_target,
        "core_source_sha256": _core_source_hash(),
        "control_source_sha256": _control_source_hash(),
        "git_commit": git["git_commit"],
        "git_dirty": git["git_dirty"],
        "git_diff_hash": git["git_diff_hash"],
    }


def run(*, resume: bool = True) -> list[dict[str, Any]]:
    """Run or resume all 87 displayed endpoint controls."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    initials = {seed: prepare_initial_state(seed) for seed in SEEDS}
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(_specs(), start=1):
        payload = _payload(spec)
        task_id = task_id_from_payload(payload)
        task_path = TASKS_DIR / f"{task_id}.json"
        if resume and task_path.is_file():
            stored = json.loads(task_path.read_text(encoding="utf-8"))
            row = stored["row"]
        else:
            row = _run_one(spec, initials[int(spec["seed"])], task_id)
            _write_json(task_path, {"payload": payload, "row": row})
        rows.append(row)
        _write_csv(ROWS_PATH, rows)
        print(f"[{index:02d}/87] {spec['gate']} seed={spec['seed']} chi={spec['chi_max']}", flush=True)
    return rows


def _median_case(rows: list[dict[str, Any]], predicate) -> dict[str, float]:
    selected = [row for row in rows if predicate(row)]
    return {
        "n": len(selected),
        "mpo_infidelity": statistics.median(float(row["mpo_infidelity"]) for row in selected),
        "variational_infidelity": statistics.median(float(row["variational_infidelity"]) for row in selected),
    }


def summarize(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Create machine-readable and compact Markdown summaries."""
    if rows is None:
        with ROWS_PATH.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    if len(rows) != 87:
        msg = f"Expected 87 completed rows, got {len(rows)}."
        raise RuntimeError(msg)

    deltas = [float(row["variational_minus_mpo"]) for row in rows]
    summary = {
        "campaign_id": CAMPAIGN_ID,
        "n_rows": len(rows),
        "all_converged": all(str(row["converged"]).lower() == "true" for row in rows),
        "equal_within_1e-12": sum(abs(delta) <= EQUALITY_TOLERANCE for delta in deltas),
        "improved_beyond_1e-12": sum(delta < -EQUALITY_TOLERANCE for delta in deltas),
        "worse_beyond_1e-12": sum(delta > EQUALITY_TOLERANCE for delta in deltas),
        "minimum_variational_minus_mpo": min(deltas),
        "maximum_variational_minus_mpo": max(deltas),
        "median_absolute_difference": statistics.median(abs(delta) for delta in deltas),
        "maximum_state_distance_to_mpo": max(float(row["state_distance_to_mpo"]) for row in rows),
        "maximum_sweeps": max(int(row["sweeps"]) for row in rows),
        "maximum_chi16_cnot_infidelity": max(
            float(row["variational_infidelity"])
            for row in rows
            if row["family"] == "cnot_rank" and int(row["chi_max"]) == 16
        ),
        "selected_medians": {
            "pauli_x_1e-2": _median_case(
                rows, lambda row: row["family"] == "pauli" and np.isclose(float(row["x"]), 1e-2)
            ),
            "pauli_x_0.25": _median_case(
                rows, lambda row: row["family"] == "pauli" and np.isclose(float(row["x"]), 0.25)
            ),
            "cnot_chi8": _median_case(rows, lambda row: row["family"] == "cnot_rank" and int(row["chi_max"]) == 8),
        },
        "cnot_medians_by_cap": {
            str(chi): _median_case(
                rows,
                lambda row, selected_chi=chi: row["family"] == "cnot_rank" and int(row["chi_max"]) == selected_chi,
            )
            for chi in CNOT_RANK_CHI_VALUES
        },
        "source_data": str(ROWS_PATH),
        "core_source_sha256": _core_source_hash(),
        "control_source_sha256": _control_source_hash(),
        "git": git_revision(),
    }
    _write_json(SUMMARY_PATH, summary)

    lines = [
        "# Variational MPO single-gate control",
        "",
        f"- Completed cells: {summary['n_rows']}",
        f"- Equal to MPO within 1e-12: {summary['equal_within_1e-12']}",
        f"- Improved beyond 1e-12: {summary['improved_beyond_1e-12']}",
        f"- Worse beyond 1e-12: {summary['worse_beyond_1e-12']}",
        f"- Maximum sweeps: {summary['maximum_sweeps']}",
        "",
        "| Case | MPO median infidelity | Variational median infidelity |",
        "| --- | ---: | ---: |",
    ]
    for label, values in summary["selected_medians"].items():
        lines.append(f"| {label} | {values['mpo_infidelity']:.8e} | {values['variational_infidelity']:.8e} |")
    SUMMARY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    rows = None if args.summarize_only else run(resume=not args.no_resume)
    summary = summarize(rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
