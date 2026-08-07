# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Mandatory validation stage for the individual-gates campaign."""

from __future__ import annotations

import copy
import csv
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
from common import (
    I2,
    PAULI,
    DiscardedWeightTracker,
    apply_cx_dense_qiskit,
    apply_gate_dense_yaqs,
    apply_method,
    cx_matrix,
    dense_h_cx,
    digital_params,
    generator_mpo_dense,
    git_revision,
    independent_r_pp_matrix,
    make_cx_dag_node,
    make_cx_gate,
    make_pauli_dag_node,
    make_pauli_gate,
    normalized_state_fidelity,
    package_versions,
    prepare_initial_state,
    track_truncate,
    two_site_h_cx,
)
from config import (
    CHI_MAX_VALUES,
    CNOT_RANK_CHI_VALUES,
    CNOT_RANK_CONTROL,
    CNOT_RANK_SVD_THRESHOLD,
    CNOT_RANK_TARGET,
    EFFECTIVE_ZERO_SVD_THRESHOLD,
    EXPECTED_CAMPAIGN_ROWS,
    EXPECTED_CNOT_RANK_ROWS,
    EXPECTED_CNOT_ROWS,
    EXPECTED_PAULI_ROWS,
    MIN_KEEP,
    OUTPUT_DIR,
    Q0,
    Q1,
    SEEDS,
    SVD_THRESHOLD,
    VALIDATION_MATRIX_TOL,
    N,
    theta_from_x,
)
from scipy.linalg import expm

from mqt.yaqs.digital import digital_tjm
from mqt.yaqs.digital.digital_tjm import apply_two_qubit_gate, construct_generator_mpo

if TYPE_CHECKING:
    from collections.abc import Iterator


def _fail(failures: list[str], msg: str) -> None:
    failures.append(msg)


def check_pauli_convention(failures: list[str], residuals: dict[str, Any]) -> None:
    pauli_res: dict[str, float] = {}
    for gate_type in ("rxx", "ryy", "rzz"):
        for x in (0.0, 1e-3, 0.25):
            theta = theta_from_x(x)
            indep = independent_r_pp_matrix(gate_type, theta)
            yaqs = np.asarray(make_pauli_gate(gate_type, theta, 0, 1).matrix, dtype=np.complex128)
            err = float(np.max(np.abs(yaqs - indep)))
            pauli_res[f"{gate_type}_x{x}_vs_formula"] = err
            if err > VALIDATION_MATRIX_TOL:
                _fail(failures, f"{gate_type} x={x}: YAQS vs cos/sin formula err={err:.3e}")
            p = PAULI[gate_type[-1]]
            expm_u = expm(-0.5j * theta * np.kron(p, p))
            err2 = float(np.max(np.abs(yaqs - expm_u)))
            pauli_res[f"{gate_type}_x{x}_vs_expm"] = err2
            if err2 > VALIDATION_MATRIX_TOL:
                _fail(failures, f"{gate_type} x={x}: YAQS vs expm err={err2:.3e}")
    residuals["pauli_matrix"] = pauli_res


def check_cx_generator_algebra(failures: list[str], residuals: dict[str, Any]) -> None:
    h = two_site_h_cx()
    u = expm(-1j * h)
    cx = cx_matrix()
    err = float(np.max(np.abs(u - cx)))
    residuals["expm_minus_i_H_CX_vs_CX"] = err
    if err > VALIDATION_MATRIX_TOL:
        _fail(failures, f"||expm(-i H_CX)-CX||={err:.3e} > tol")

    path_res: dict[str, float] = {}
    for n in (1, 2, 3, 4, 8, 16, 32):
        u_n = np.linalg.matrix_power(expm(-1j * h / n), n)
        err_n = float(np.max(np.abs(u_n - cx)))
        path_res[str(n)] = err_n
        if err_n > VALIDATION_MATRIX_TOL:
            _fail(failures, f"||(expm(-i H_CX/{n}))^{n}-CX||={err_n:.3e}")
    residuals["cx_path_power_residuals"] = path_res


def _local_mpo_generator_lr(gate) -> np.ndarray:
    """2-site generator in left-right site order (left = min site = LSB)."""
    from common import _mpo_site_matrix

    mpo, first, last = construct_generator_mpo(gate, N)
    left = _mpo_site_matrix(mpo.tensors[first])
    right = _mpo_site_matrix(mpo.tensors[last])
    return np.kron(right, left)


def check_cx_generator_mpo(failures: list[str], residuals: dict[str, Any]) -> None:
    mpo_res: dict[str, float] = {}
    for control, target in ((Q0, Q1), (Q1, Q0)):
        gate = make_cx_gate(control, target)
        local = _local_mpo_generator_lr(gate)
        left_site, right_site = min(control, target), max(control, target)
        if control == left_site and target == right_site:
            indep = (np.pi / 4.0) * np.kron(I2 - PAULI["x"], I2 - PAULI["z"])
        elif control == right_site and target == left_site:
            indep = (np.pi / 4.0) * np.kron(I2 - PAULI["z"], I2 - PAULI["x"])
        else:
            _fail(failures, f"Unexpected orientation control={control} target={target}")
            continue
        err = float(np.max(np.abs(local - indep)))
        mpo_res[f"local_CX({control},{target})"] = err
        if err > VALIDATION_MATRIX_TOL:
            _fail(
                failures,
                f"Generator MPO vs dense H_CX for CX({control},{target}) err={err:.3e}",
            )
        dense = dense_h_cx(control, target, N)
        mpo_dense = generator_mpo_dense(gate, N)
        err_full = float(np.max(np.abs(mpo_dense - dense)))
        mpo_res[f"full_CX({control},{target})"] = err_full
        if err_full > VALIDATION_MATRIX_TOL:
            _fail(failures, f"Full MPO densification vs H_CX CX({control},{target}) err={err_full:.3e}")
    residuals["generator_mpo"] = mpo_res


def check_cx_qiskit_endpoint(failures: list[str], residuals: dict[str, Any]) -> None:
    init = prepare_initial_state(11)
    qiskit_res: dict[str, float] = {}
    for control, target in ((Q0, Q1), (Q1, Q0)):
        gate = make_cx_gate(control, target)
        yaqs = apply_gate_dense_yaqs(init["vec"], N, control, target, gate)
        qiskit = apply_cx_dense_qiskit(init["vec"], control, target, N)
        err = float(np.linalg.norm(yaqs - qiskit))
        qiskit_res[f"CX({control},{target})"] = err
        if err > 1e-12:
            _fail(failures, f"YAQS dense CX({control},{target}) vs Qiskit err={err:.3e}")
    residuals["qiskit_endpoint"] = qiskit_res


@contextmanager
def _trace_dispatch() -> Iterator[dict[str, int]]:
    counts = {"tdvp": 0, "mpo": 0, "tebd": 0}
    orig_tdvp = digital_tjm.apply_two_qubit_gate_tdvp
    orig_mpo = digital_tjm.apply_long_range_gate_mpo
    orig_tebd = digital_tjm.apply_two_qubit_gate_tebd

    def wrap_tdvp(*args, **kwargs):
        counts["tdvp"] += 1
        return orig_tdvp(*args, **kwargs)

    def wrap_mpo(*args, **kwargs):
        counts["mpo"] += 1
        return orig_mpo(*args, **kwargs)

    def wrap_tebd(*args, **kwargs):
        counts["tebd"] += 1
        return orig_tebd(*args, **kwargs)

    digital_tjm.apply_two_qubit_gate_tdvp = wrap_tdvp  # type: ignore[assignment]
    digital_tjm.apply_long_range_gate_mpo = wrap_mpo  # type: ignore[assignment]
    digital_tjm.apply_two_qubit_gate_tebd = wrap_tebd  # type: ignore[assignment]
    try:
        yield counts
    finally:
        digital_tjm.apply_two_qubit_gate_tdvp = orig_tdvp
        digital_tjm.apply_long_range_gate_mpo = orig_mpo
        digital_tjm.apply_two_qubit_gate_tebd = orig_tebd


def check_cx_tdvp_dispatch(failures: list[str], residuals: dict[str, Any]) -> None:
    init = prepare_initial_state(11)
    dispatch: dict[str, dict[str, int]] = {}
    for control, target in ((Q0, Q1), (Q1, Q0)):
        node = make_cx_dag_node(control, target)
        params = digital_params(8, method="gate_local_2tdvp", n_sub=1)
        with _trace_dispatch() as counts:
            apply_two_qubit_gate(copy.deepcopy(init["mps"]), node, params)
        dispatch[f"CX({control},{target})"] = dict(counts)
        if counts["tdvp"] != 1 or counts["mpo"] != 0:
            _fail(failures, f"Separated CNOT CX({control},{target}) under full-tdvp dispatch counts={counts}")
    residuals["tdvp_dispatch_counts"] = dispatch


def check_min_keep_no_padding(failures: list[str]) -> None:
    init = prepare_initial_state(11)
    node = make_pauli_dag_node("rzz", theta_from_x(0.1), Q0, Q1)
    tracker = DiscardedWeightTracker()
    with track_truncate(tracker):
        apply_method(init["mps"], node, method="gate_local_2tdvp", chi=8, n_sub=1, tracker=tracker)
    if not tracker.min_keep_args:
        _fail(failures, "No truncate calls observed during TDVP RZZ")
        return
    if any(mk != MIN_KEEP for mk in tracker.min_keep_args):
        _fail(failures, f"min_keep args {tracker.min_keep_args} != {MIN_KEEP}")
    for s_list, keep, mk in zip(tracker.singular_lists, tracker.keep_counts, tracker.min_keep_args, strict=True):
        if mk != 1:
            _fail(failures, f"min_keep={mk} observed")
        if len(s_list) >= 2 and keep >= 2:
            s0 = s_list[0]
            s1 = s_list[1]
            if s0 > 0 and s1 <= 1e-30 * s0:
                _fail(failures, f"Exact-zero singular retained: s={s_list[:4]} keep={keep}")


def check_swap_route_and_ordering(failures: list[str]) -> None:
    init = prepare_initial_state(11)
    node = make_pauli_dag_node("rxx", theta_from_x(0.1), Q0, Q1)
    with _trace_dispatch() as counts:
        apply_method(init["mps"], node, method="tebd_swap", chi=8, n_sub=1)
    if counts["tebd"] < 1:
        _fail(failures, f"tebd_swap did not enter TEBD path; counts={counts}")

    g_fwd = make_cx_gate(Q0, Q1)
    g_rev = make_cx_gate(Q1, Q0)
    if g_fwd.sites != [Q0, Q1]:
        _fail(failures, f"Forward CX sites={g_fwd.sites}")
    if g_rev.sites != [Q1, Q0]:
        _fail(failures, f"Reverse CX sites={g_rev.sites}")


def check_row_inventory(failures: list[str], residuals: dict[str, Any]) -> None:
    inventory = {
        "expected_pauli_rows": EXPECTED_PAULI_ROWS,
        "expected_cnot_rows": EXPECTED_CNOT_ROWS,
        "expected_campaign_rows": EXPECTED_CAMPAIGN_ROWS,
        "expected_cnot_rank_rows": EXPECTED_CNOT_RANK_ROWS,
        "seeds": list(SEEDS),
        "chi_max_campaign": list(CHI_MAX_VALUES),
        "chi_max_cnot_rank": list(CNOT_RANK_CHI_VALUES),
    }
    residuals["row_inventory"] = inventory
    if EXPECTED_PAULI_ROWS != 432:
        _fail(failures, f"EXPECTED_PAULI_ROWS={EXPECTED_PAULI_ROWS} != 432")
    if EXPECTED_CNOT_ROWS != 36:
        _fail(failures, f"EXPECTED_CNOT_ROWS={EXPECTED_CNOT_ROWS} != 36")
    if EXPECTED_CAMPAIGN_ROWS != 468:
        _fail(failures, f"EXPECTED_CAMPAIGN_ROWS={EXPECTED_CAMPAIGN_ROWS} != 468")
    if EXPECTED_CNOT_RANK_ROWS != 90:
        _fail(failures, f"EXPECTED_CNOT_RANK_ROWS={EXPECTED_CNOT_RANK_ROWS} != 90")
    if len(SEEDS) != 3 or len(CHI_MAX_VALUES) != 2:
        _fail(failures, "Unexpected seeds or chi grid")

    campaign_csv = OUTPUT_DIR / "campaign_rows.csv"
    if campaign_csv.is_file():
        with campaign_csv.open(encoding="utf-8", newline="") as fh:
            n = sum(1 for _ in csv.DictReader(fh))
        inventory["actual_campaign_rows"] = n
        if n != EXPECTED_CAMPAIGN_ROWS:
            _fail(failures, f"campaign_rows.csv has {n} rows, expected {EXPECTED_CAMPAIGN_ROWS}")
    else:
        inventory["actual_campaign_rows"] = None

    cnot_csv = OUTPUT_DIR / "cnot_rank_rows.csv"
    if cnot_csv.is_file():
        with cnot_csv.open(encoding="utf-8", newline="") as fh:
            n = sum(1 for _ in csv.DictReader(fh))
        inventory["actual_cnot_rank_rows"] = n
        if n != EXPECTED_CNOT_RANK_ROWS:
            _fail(failures, f"cnot_rank_rows.csv has {n} rows, expected {EXPECTED_CNOT_RANK_ROWS}")
    else:
        inventory["actual_cnot_rank_rows"] = None


def check_deterministic_replay(failures: list[str]) -> None:
    init = prepare_initial_state(11)
    node = make_pauli_dag_node("ryy", theta_from_x(1e-3), Q0, Q1)
    a, _ = apply_method(init["mps"], node, method="gate_local_2tdvp", chi=8, n_sub=1)
    b, _ = apply_method(init["mps"], node, method="gate_local_2tdvp", chi=8, n_sub=1)
    err = float(np.linalg.norm(a.to_vec() - b.to_vec()))
    if err > 1e-14:
        _fail(failures, f"Deterministic replay mismatch err={err:.3e}")
    metrics = normalized_state_fidelity(init["vec"], a.to_vec())
    if not (0.0 <= metrics["fidelity_normalized"] <= 1.0):
        _fail(failures, f"Fidelity out of range: {metrics}")


def check_n128_n256_distances(failures: list[str], residuals: dict[str, Any]) -> None:
    """Report phase-aligned ||Ψ_128 − Ψ_256|| for every state and cap when available."""
    summary_path = OUTPUT_DIR / "cnot_rank_summary.json"
    if not summary_path.is_file():
        residuals["n128_to_n256_distances"] = {"status": "cnot_rank_not_yet_run"}
        return
    import json

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    dists = summary.get("distances_n128_n256", [])
    residuals["n128_to_n256_distances"] = dists
    if len(dists) != len(CNOT_RANK_CHI_VALUES) * len(SEEDS):
        _fail(
            failures,
            f"Expected {len(CNOT_RANK_CHI_VALUES) * len(SEEDS)} n128–n256 distances, got {len(dists)}",
        )
    for entry in dists:
        d = entry.get("phase_aligned_distance_n128_n256")
        if d is None or (isinstance(d, float) and np.isnan(d)):
            _fail(
                failures,
                f"Missing n128–n256 distance for seed={entry.get('seed')} chi={entry.get('chi_max')}",
            )


def check_threshold_controls(failures: list[str], residuals: dict[str, Any]) -> None:
    """Production-threshold vs effective-zero n=1 controls at χ=8 and 16."""
    campaign_csv = OUTPUT_DIR / "campaign_rows.csv"
    cnot_csv = OUTPUT_DIR / "cnot_rank_rows.csv"
    if not campaign_csv.is_file() or not cnot_csv.is_file():
        residuals["production_vs_effective_zero_n1"] = {"status": "datasets_incomplete"}
        return

    with campaign_csv.open(encoding="utf-8", newline="") as fh:
        camp = list(csv.DictReader(fh))
    with cnot_csv.open(encoding="utf-8", newline="") as fh:
        rank = list(csv.DictReader(fh))

    controls: list[dict[str, Any]] = []
    for chi in (8, 16):
        for seed in SEEDS:
            for method in ("gate_local_2tdvp", "mpo_zipup", "tebd_swap"):
                c_rows = [
                    r
                    for r in camp
                    if r["family"] == "cnot"
                    and r["gate"] == "cx"
                    and int(r["control"]) == CNOT_RANK_CONTROL
                    and int(r["target"]) == CNOT_RANK_TARGET
                    and int(r["seed"]) == seed
                    and int(r["chi_max"]) == chi
                    and r["method"] == method
                    and int(float(r["n_sub"])) == 1
                ]
                r_rows = [
                    r
                    for r in rank
                    if int(r["seed"]) == seed
                    and int(r["chi_max"]) == chi
                    and r["method"] == method
                    and int(float(r["n_sub"])) == 1
                ]
                if not c_rows or not r_rows:
                    continue
                c_inf = float(c_rows[0]["infidelity_normalized"])
                r_inf = float(r_rows[0]["infidelity_normalized"])
                controls.append(
                    {
                        "seed": seed,
                        "chi_max": chi,
                        "method": method,
                        "production_svd_threshold": SVD_THRESHOLD,
                        "effective_zero_svd_threshold": CNOT_RANK_SVD_THRESHOLD,
                        "infidelity_production": c_inf,
                        "infidelity_effective_zero": r_inf,
                        "abs_diff": abs(c_inf - r_inf),
                    }
                )
    residuals["production_vs_effective_zero_n1"] = controls
    if len(controls) != 18:  # 2 chi × 3 seeds × 3 methods
        _fail(failures, f"Expected 18 production-vs-zero controls, got {len(controls)}")


def check_state_cache_distances(residuals: dict[str, Any]) -> None:
    """Optional direct reload of n=128/256 states for residual reporting."""
    states_dir = OUTPUT_DIR / "cnot_rank_states"
    if not states_dir.is_dir():
        return
    # Distances already recorded in cnot_rank_summary; keep a file-count note.
    residuals["cnot_rank_state_files"] = len(list(states_dir.glob("*.npy")))


def run_validation() -> dict[str, Any]:
    """Run all mandatory checks. Returns a report dict; raises on failure."""
    failures: list[str] = []
    residuals: dict[str, Any] = {}
    check_row_inventory(failures, residuals)
    check_pauli_convention(failures, residuals)
    check_cx_generator_algebra(failures, residuals)
    check_cx_generator_mpo(failures, residuals)
    check_cx_qiskit_endpoint(failures, residuals)
    check_cx_tdvp_dispatch(failures, residuals)
    check_min_keep_no_padding(failures)
    check_swap_route_and_ordering(failures)
    check_deterministic_replay(failures)
    check_n128_n256_distances(failures, residuals)
    check_threshold_controls(failures, residuals)
    check_state_cache_distances(residuals)

    git = git_revision()
    report = {
        "stage": "validate",
        "pass": len(failures) == 0,
        "n_failures": len(failures),
        "failures": failures,
        "residuals": residuals,
        "effective_zero_svd_threshold": EFFECTIVE_ZERO_SVD_THRESHOLD,
        "production_svd_threshold": SVD_THRESHOLD,
        "git": git,
        "git_diff_hash": git["git_diff_hash"],
        "versions": package_versions(),
        "sites_paper_one_based": [Q0 + 1, Q1 + 1],
        "cnot_rank_orientation_code": [CNOT_RANK_CONTROL, CNOT_RANK_TARGET],
    }
    if failures:
        msg = "Validation failed:\n" + "\n".join(f"  - {f}" for f in failures)
        raise RuntimeError(msg)
    return report
