# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Structural projector validation suite.

Locality identities, exterior cancellation, nearest-neighbor exactness, and the
minimal-support long-range obstruction (analytical projectors plus production
product-state stall under unmodified full_tdvp).

Usage:
    uv run python -m experiments.structural_checks.run
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    CHI,
    DTYPE,
    EXPERIMENT_DIR,
    EXTERIOR_SEED,
    EXTERIOR_SITES,
    EXTERIOR_TERM_FLOOR,
    FIXED_RANK_GEOMETRIES,
    GENERATOR_SEED,
    MPS_SEEDS,
    NN_GEOMETRIES,
    OBSTRUCTION_N,
    OBSTRUCTION_P2_TOL,
    OBSTRUCTION_SITES,
    OBSTRUCTION_THETA,
    OUTPUT_DIR,
    PRODUCT_SWEEP_CONFIGS,
    PRODUCTION_STALL_DIST_TOL,
    PRODUCTION_STALL_INFIDELITY_TOL,
    REL_TOL,
    TWO_SITE_GEOMETRIES,
    D,
    N,
    bond_profile,
)
from .dense_projectors import (
    FixtureError,
    apply_b_contract,
    apply_k_contract,
    apply_s_contract,
    apply_two_site_op,
    apply_xx,
    assert_nonvacuous_action,
    build_p1_full,
    build_p2_full,
    compute_schmidt,
    fixed_rank_window,
    infidelity,
    localized_p1_action,
    localized_p2_action,
    make_generic_generator,
    projector_diagnostics,
    random_exact_rank_state,
    relative_residual,
    two_site_window,
)
from .rank_tracing import trace_split_ranks


def sci_tex(value: float) -> str:
    """Ordinary LaTeX scientific notation without siunitx."""
    x = float(value)
    if x == 0.0:
        return "0"
    mant, exp_s = f"{x:.2e}".split("e")
    return f"{mant}\\times10^{{{int(exp_s)}}}"


def _package_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "yaqs": "unknown",
    }
    try:
        from mqt import yaqs

        versions["yaqs"] = getattr(yaqs, "__version__", "unknown")
    except Exception:
        pass
    return versions


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _conditioning_fields(sch) -> dict[str, float]:
    return {
        "sigma_min_retained": sch.sigma_min_retained,
        "sigma_max_discarded": sch.sigma_max_discarded,
    }


def check_projector_algebra(sch, seed: int) -> list[dict[str, Any]]:
    """Hermitian / idempotent diagnostics for full P1 and P2."""
    rows = []
    for name, builder in (("P1", build_p1_full), ("P2", build_p2_full)):
        p = builder(sch)
        diag = projector_diagnostics(p)
        ok = diag["hermitian_rel"] < REL_TOL and diag["idempotent_rel"] < REL_TOL
        rows.append(
            {
                "check_family": "projector_algebra",
                "geometry": name,
                "seed": seed,
                "N": sch.n,
                "chi": CHI,
                "site_a_zero_based": "",
                "site_b_zero_based": "",
                "window_start_zero_based": "",
                "window_end_zero_based": "",
                "generator_norm": "",
                "full_action_norm": "",
                "absolute_residual": max(diag["hermitian_rel"], diag["idempotent_rel"]),
                "relative_residual": max(diag["hermitian_rel"], diag["idempotent_rel"]),
                "hermitian_rel": diag["hermitian_rel"],
                "idempotent_rel": diag["idempotent_rel"],
                **_conditioning_fields(sch),
                "pass": ok,
            }
        )
    return rows


def check_fixed_rank(
    sch,
    seed: int,
    generator: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    p1 = build_p1_full(sch)
    for geometry, (q0, q1) in FIXED_RANK_GEOMETRIES.items():
        x = apply_two_site_op(sch.psi, generator, q0, q1, n=sch.n)
        full = p1 @ x
        assert_nonvacuous_action(full, x)
        windowed = localized_p1_action(x, q0, q1, sch)
        abs_res, rel, full_norm = relative_residual(full, windowed)
        w0, w1 = fixed_rank_window(q0, q1)
        ok = rel < REL_TOL
        rows.append(
            {
                "check_family": "fixed_rank_locality",
                "geometry": geometry,
                "seed": seed,
                "N": sch.n,
                "chi": CHI,
                "site_a_zero_based": q0,
                "site_b_zero_based": q1,
                "window_start_zero_based": w0,
                "window_end_zero_based": w1,
                "generator_norm": float(np.linalg.norm(x)),
                "full_action_norm": full_norm,
                "absolute_residual": abs_res,
                "relative_residual": rel,
                **_conditioning_fields(sch),
                "pass": ok,
            }
        )
    return rows


def check_two_site(
    sch,
    seed: int,
    generator: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    p2 = build_p2_full(sch)
    for geometry, (q0, q1) in TWO_SITE_GEOMETRIES.items():
        x = apply_two_site_op(sch.psi, generator, q0, q1, n=sch.n)
        full = p2 @ x
        assert_nonvacuous_action(full, x)
        windowed = localized_p2_action(x, q0, q1, sch)
        abs_res, rel, full_norm = relative_residual(full, windowed)
        w0, w1 = two_site_window(q0, q1)
        ok = rel < REL_TOL
        rows.append(
            {
                "check_family": "two_site_locality",
                "geometry": geometry,
                "seed": seed,
                "N": sch.n,
                "chi": CHI,
                "site_a_zero_based": q0,
                "site_b_zero_based": q1,
                "window_start_zero_based": w0,
                "window_end_zero_based": w1,
                "generator_norm": float(np.linalg.norm(x)),
                "full_action_norm": full_norm,
                "absolute_residual": abs_res,
                "relative_residual": rel,
                **_conditioning_fields(sch),
                "pass": ok,
            }
        )
    return rows


def check_nn_exactness(
    sch,
    seed: int,
    generator: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    p2 = build_p2_full(sch)
    for geometry, (q0, q1) in NN_GEOMETRIES.items():
        x = apply_two_site_op(sch.psi, generator, q0, q1, n=sch.n)
        proj = p2 @ x
        abs_res = float(np.linalg.norm(x - proj))
        x_norm = float(np.linalg.norm(x))
        rel = abs_res / x_norm if x_norm > 0 else np.inf
        ok = rel < REL_TOL
        rows.append(
            {
                "check_family": "nearest_neighbor_exactness",
                "geometry": geometry,
                "seed": seed,
                "N": sch.n,
                "chi": CHI,
                "site_a_zero_based": q0,
                "site_b_zero_based": q1,
                "window_start_zero_based": q0,
                "window_end_zero_based": q1,
                "generator_norm": x_norm,
                "full_action_norm": float(np.linalg.norm(proj)),
                "absolute_residual": abs_res,
                "relative_residual": rel,
                **_conditioning_fields(sch),
                "pass": ok,
            }
        )
    return rows


def check_exterior_cancellation(sch, generator: np.ndarray) -> list[dict[str, Any]]:
    """Explicit nonzero-term exterior cancellation for seed 101, sites (2,5)."""
    q0, q1 = EXTERIOR_SITES
    x = apply_two_site_op(sch.psi, generator, q0, q1, n=sch.n)
    x_norm = float(np.linalg.norm(x))
    floor = EXTERIOR_TERM_FLOOR * x_norm
    rows: list[dict[str, Any]] = []

    # Fixed-rank exterior pairs
    p1_exterior = np.zeros_like(x)
    for k in range(q0):
        s_term = apply_s_contract(x, k, sch)
        b_term = apply_b_contract(x, k + 1, sch)
        diff = s_term - b_term
        p1_exterior += diff
        s_n = float(np.linalg.norm(s_term))
        b_n = float(np.linalg.norm(b_term))
        abs_res = float(np.linalg.norm(diff))
        rel = abs_res / x_norm if x_norm > 0 else np.inf
        ok = s_n > floor and b_n > floor and rel < REL_TOL
        rows.append(
            {
                "check_family": "exterior_cancellation_P1",
                "side": "left",
                "pair_index": k,
                "term_a": "S",
                "term_a_index": k,
                "term_b": "B",
                "term_b_index": k + 1,
                "term_a_norm": s_n,
                "term_b_norm": b_n,
                "term_a_rel": s_n / x_norm if x_norm > 0 else np.inf,
                "term_b_rel": b_n / x_norm if x_norm > 0 else np.inf,
                "absolute_residual": abs_res,
                "relative_residual": rel,
                "x_norm": x_norm,
                "term_floor": floor,
                "pass": ok,
            }
        )
    for k in range(q1 + 1, sch.n):
        s_term = apply_s_contract(x, k, sch)
        b_term = apply_b_contract(x, k, sch)
        diff = s_term - b_term
        p1_exterior += diff
        s_n = float(np.linalg.norm(s_term))
        b_n = float(np.linalg.norm(b_term))
        abs_res = float(np.linalg.norm(diff))
        rel = abs_res / x_norm if x_norm > 0 else np.inf
        ok = s_n > floor and b_n > floor and rel < REL_TOL
        rows.append(
            {
                "check_family": "exterior_cancellation_P1",
                "side": "right",
                "pair_index": k,
                "term_a": "S",
                "term_a_index": k,
                "term_b": "B",
                "term_b_index": k,
                "term_a_norm": s_n,
                "term_b_norm": b_n,
                "term_a_rel": s_n / x_norm if x_norm > 0 else np.inf,
                "term_b_rel": b_n / x_norm if x_norm > 0 else np.inf,
                "absolute_residual": abs_res,
                "relative_residual": rel,
                "x_norm": x_norm,
                "term_floor": floor,
                "pass": ok,
            }
        )
    total_p1 = float(np.linalg.norm(p1_exterior)) / x_norm if x_norm > 0 else np.inf
    rows.append(
        {
            "check_family": "exterior_cancellation_P1",
            "side": "total",
            "pair_index": -1,
            "term_a": "sum",
            "term_a_index": -1,
            "term_b": "sum",
            "term_b_index": -1,
            "term_a_norm": float(np.linalg.norm(p1_exterior)),
            "term_b_norm": 0.0,
            "term_a_rel": total_p1,
            "term_b_rel": 0.0,
            "absolute_residual": float(np.linalg.norm(p1_exterior)),
            "relative_residual": total_p1,
            "x_norm": x_norm,
            "term_floor": floor,
            "pass": total_p1 < REL_TOL,
        }
    )

    # Two-site exterior pairs
    p2_exterior = np.zeros_like(x)
    for k in range(q0):
        k_term = apply_k_contract(x, k, sch)
        s_term = apply_s_contract(x, k + 1, sch)
        diff = k_term - s_term
        p2_exterior += diff
        k_n = float(np.linalg.norm(k_term))
        s_n = float(np.linalg.norm(s_term))
        abs_res = float(np.linalg.norm(diff))
        rel = abs_res / x_norm if x_norm > 0 else np.inf
        ok = k_n > floor and s_n > floor and rel < REL_TOL
        rows.append(
            {
                "check_family": "exterior_cancellation_P2",
                "side": "left",
                "pair_index": k,
                "term_a": "K",
                "term_a_index": k,
                "term_b": "S",
                "term_b_index": k + 1,
                "term_a_norm": k_n,
                "term_b_norm": s_n,
                "term_a_rel": k_n / x_norm if x_norm > 0 else np.inf,
                "term_b_rel": s_n / x_norm if x_norm > 0 else np.inf,
                "absolute_residual": abs_res,
                "relative_residual": rel,
                "x_norm": x_norm,
                "term_floor": floor,
                "pass": ok,
            }
        )
    for k in range(q1, sch.n - 1):
        k_term = apply_k_contract(x, k, sch)
        s_term = apply_s_contract(x, k, sch)
        diff = k_term - s_term
        p2_exterior += diff
        k_n = float(np.linalg.norm(k_term))
        s_n = float(np.linalg.norm(s_term))
        abs_res = float(np.linalg.norm(diff))
        rel = abs_res / x_norm if x_norm > 0 else np.inf
        ok = k_n > floor and s_n > floor and rel < REL_TOL
        rows.append(
            {
                "check_family": "exterior_cancellation_P2",
                "side": "right",
                "pair_index": k,
                "term_a": "K",
                "term_a_index": k,
                "term_b": "S",
                "term_b_index": k,
                "term_a_norm": k_n,
                "term_b_norm": s_n,
                "term_a_rel": k_n / x_norm if x_norm > 0 else np.inf,
                "term_b_rel": s_n / x_norm if x_norm > 0 else np.inf,
                "absolute_residual": abs_res,
                "relative_residual": rel,
                "x_norm": x_norm,
                "term_floor": floor,
                "pass": ok,
            }
        )
    total_p2 = float(np.linalg.norm(p2_exterior)) / x_norm if x_norm > 0 else np.inf
    rows.append(
        {
            "check_family": "exterior_cancellation_P2",
            "side": "total",
            "pair_index": -1,
            "term_a": "sum",
            "term_a_index": -1,
            "term_b": "sum",
            "term_b_index": -1,
            "term_a_norm": float(np.linalg.norm(p2_exterior)),
            "term_b_norm": 0.0,
            "term_a_rel": total_p2,
            "term_b_rel": 0.0,
            "absolute_residual": float(np.linalg.norm(p2_exterior)),
            "relative_residual": total_p2,
            "x_norm": x_norm,
            "term_floor": floor,
            "pass": total_p2 < REL_TOL,
        }
    )
    return rows


def check_analytical_obstruction() -> list[dict[str, Any]]:
    """Dense analytical obstruction on |0000⟩ with X_0 X_3 (paper X_1 X_4)."""
    n = OBSTRUCTION_N
    psi = np.zeros(2**n, dtype=np.complex128)
    psi[0] = 1.0
    # Product state: all Schmidt ranks are 1; build projectors with chi=1.
    from .dense_projectors import SchmidtData

    lefts: dict[int, np.ndarray] = {}
    rights: dict[int, np.ndarray] = {}
    spectra: dict[int, np.ndarray] = {}
    p_left = {0: np.ones((1, 1), dtype=np.complex128)}
    p_right = {n: np.ones((1, 1), dtype=np.complex128)}
    ranks_ok = True
    for cut in range(1, n):
        matrix = psi.reshape(2**cut, 2 ** (n - cut))
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        rank = 1
        numerical_rank = int(np.sum(s > 1e-12 * max(s[0], 1e-30)))
        ranks_ok = ranks_ok and numerical_rank == rank
        lefts[cut] = u[:, :rank]
        rights[cut] = vh[:rank, :].conj().T
        spectra[cut] = s
        p_left[cut] = lefts[cut] @ lefts[cut].conj().T
        p_right[cut] = rights[cut] @ rights[cut].conj().T
    sch = SchmidtData(
        psi=psi,
        n=n,
        profile=[1] * (n + 1),
        lefts=lefts,
        rights=rights,
        spectra=spectra,
        p_left=p_left,
        p_right=p_right,
        sigma_min_retained=1.0,
        sigma_max_discarded=0.0,
    )
    q0, q1 = OBSTRUCTION_SITES
    x = apply_xx(psi, q0, q1, n=n)
    p1x = build_p1_full(sch) @ x
    p2x = build_p2_full(sch) @ x
    x_norm = float(np.linalg.norm(x))
    p1_norm = float(np.linalg.norm(p1x))
    p2_norm = float(np.linalg.norm(p2x))
    theta = OBSTRUCTION_THETA
    psi_exact = np.cos(theta / 2.0) * psi - 1j * np.sin(theta / 2.0) * x
    fid_inf = infidelity(psi, psi_exact)

    term_rows = []
    all_terms_vanish = True
    for k in range(n - 1):
        kn = float(np.linalg.norm(apply_k_contract(x, k, sch)))
        term_rows.append({"term": "K", "index": k, "norm": kn})
        all_terms_vanish = all_terms_vanish and kn < OBSTRUCTION_P2_TOL
    for k in range(n):
        sn = float(np.linalg.norm(apply_s_contract(x, k, sch)))
        term_rows.append({"term": "S", "index": k, "norm": sn})
        all_terms_vanish = all_terms_vanish and sn < OBSTRUCTION_P2_TOL

    p1_rel = p1_norm / x_norm if x_norm > 0 else np.inf
    p2_rel = p2_norm / x_norm if x_norm > 0 else np.inf
    ok = (
        abs(x_norm - 1.0) < REL_TOL
        and p1_norm < OBSTRUCTION_P2_TOL
        and p2_norm < OBSTRUCTION_P2_TOL
        and abs(fid_inf - 0.5) < REL_TOL
        and ranks_ok
        and all_terms_vanish
    )
    return [
        {
            "check_family": "analytical_obstruction",
            "geometry": "product_xx",
            "N": n,
            "site_a_zero_based": q0,
            "site_b_zero_based": q1,
            "site_a_one_based": q0 + 1,
            "site_b_one_based": q1 + 1,
            "theta": theta,
            "x_norm": x_norm,
            "p1_action_norm": p1_norm,
            "p1_action_rel": p1_rel,
            "p2_action_norm": p2_norm,
            "p2_action_rel": p2_rel,
            "infidelity_vs_unchanged": fid_inf,
            "all_schmidt_ranks_one": ranks_ok,
            "all_projector_terms_vanish": all_terms_vanish,
            "term_norms_json": json.dumps(term_rows),
            "absolute_residual": p2_norm,
            "relative_residual": p2_rel,
            "pass": ok,
        }
    ]


def _product_state_refs():
    """Shared exact RXX endpoint and DAG node for product-state sweeps."""
    from mqt.yaqs.core.data_structures.mps import MPS

    from .production_helpers import apply_gate_dense_yaqs, make_dag_node, make_gate

    q0, q1 = OBSTRUCTION_SITES
    theta = OBSTRUCTION_THETA
    n = OBSTRUCTION_N
    node = make_dag_node("rxx", theta, q0, q1, n)
    psi0 = MPS(n, state="zeros").to_vec().astype(np.complex128)
    gate = make_gate("rxx", theta, q0, q1)
    psi_exact = apply_gate_dense_yaqs(psi0, n, q0, q1, gate)
    return n, q0, q1, node, psi0, psi_exact


def check_production_product_sweep() -> list[dict[str, Any]]:
    """Unmodified production full_tdvp on RXX|0000⟩; expect stall (peak χ=1)."""
    from mqt.yaqs.core.data_structures.mps import MPS

    from .production_helpers import (
        apply_full_tdvp,
        mps_bond_profile,
        normalized_infidelity,
        phase_align,
    )

    n, q0, q1, node, psi0, psi_exact = _product_state_refs()
    rows = []
    for cfg in PRODUCT_SWEEP_CONFIGS:
        chi = int(cfg["chi_max"])
        n_sub = int(cfg["n_sub"])
        initial = MPS(n, state="zeros")
        with trace_split_ranks() as rank_trace:
            final_mps, _discarded = apply_full_tdvp(
                initial,
                node,
                chi=chi,
                substeps=n_sub,
            )
        final_vec = final_mps.to_vec().astype(np.complex128)
        aligned = phase_align(psi0, final_vec)
        dist = float(np.linalg.norm(aligned - psi0))
        inf = normalized_infidelity(psi_exact, final_vec)
        profile = mps_bond_profile(final_mps)
        peak_rank = int(rank_trace.peak_retained_rank)
        bonds_all_one = profile == [1] * (n + 1)
        ok = (
            dist < PRODUCTION_STALL_DIST_TOL
            and abs(inf - 0.5) < PRODUCTION_STALL_INFIDELITY_TOL
            and peak_rank == 1
            and bonds_all_one
        )
        rows.append(
            {
                "check_family": "production_product_sweep",
                "geometry": "rxx_product",
                "N": n,
                "site_a_zero_based": q0,
                "site_b_zero_based": q1,
                "site_a_one_based": q0 + 1,
                "site_b_one_based": q1 + 1,
                "chi_max": chi,
                "n_sub": n_sub,
                "phase_aligned_distance": dist,
                "infidelity_vs_exact": inf,
                "final_bond_profile": json.dumps(profile),
                "peak_retained_rank_during_split": peak_rank,
                "pass": ok,
            }
        )
    return rows


def _max_rel(rows: list[dict[str, Any]], family: str | None = None) -> float:
    vals = [
        float(r["relative_residual"])
        for r in rows
        if "relative_residual" in r
        and r["relative_residual"] != ""
        and (family is None or r.get("check_family") == family)
    ]
    return float(max(vals)) if vals else float("nan")


def write_table(
    projector_rows: list[dict[str, Any]],
    exterior_rows: list[dict[str, Any]],
    obstruction_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    """Theorem-facing REVTeX ``table*``; maxima over seeds; one-based sites."""

    def max_rel(family: str, geometries: set[str] | None = None) -> float:
        vals = [
            float(r["relative_residual"])
            for r in projector_rows
            if r["check_family"] == family and (geometries is None or r["geometry"] in geometries)
        ]
        return max(vals) if vals else float("nan")

    def exterior_summary(family: str) -> tuple[float, float]:
        pair_rows = [r for r in exterior_rows if r["check_family"] == family and r["side"] != "total"]
        min_term_rel = min(min(float(r["term_a_rel"]), float(r["term_b_rel"])) for r in pair_rows)
        max_pair = max(float(r["relative_residual"]) for r in pair_rows)
        return min_term_rel, max_pair

    r1 = max_rel(
        "fixed_rank_locality",
        {"interior", "left_boundary", "right_boundary"},
    )
    r2 = max_rel(
        "two_site_locality",
        {"adjacent_interior", "separated_interior", "left_boundary", "right_boundary"},
    )
    nn = max_rel("nearest_neighbor_exactness")
    p1_min_term, p1_ext = exterior_summary("exterior_cancellation_P1")
    p2_min_term, p2_ext = exterior_summary("exterior_cancellation_P2")
    anal = obstruction_rows[0]
    p1_rel = float(anal["p1_action_rel"])
    p2_rel = float(anal["p2_action_rel"])

    caption = (
        r"Structural validation of gate-local TDVP projectors. "
        r"Unless noted, residuals are maxima over three deterministic exact-rank "
        r"complex MPS states ($N=8$, maximum bond dimension $\chi=4$, seeds "
        r"$101$--$103$) acted on by one fixed normalized generic Hermitian "
        r"two-site generator. "
        r"The minimal-support obstruction uses $|\psi\rangle=|0000\rangle$ with "
        r"$H_g=X_1X_4$ (paper sites)."
    )

    lines = [
        r"% Auto-generated by experiments/structural_checks/run.py",
        r"% Sites are one-based (paper convention).",
        r"\begin{table*}",
        r"\caption{" + caption + "}",
        r"\label{tab:structural}",
        r"\begin{tabular}{llc}",
        r"\hline\hline",
        r"Mathematical statement & Diagnostic & Result \\",
        r"\hline",
        (
            r"Gate-window locality & "
            r"$\max r_1$ (interior and boundary) & "
            f"${sci_tex(r1)}$ \\\\"
        ),
        (
            r"Two-site gate-window locality & "
            r"$\max r_2$ (adjacent, separated, boundary) & "
            f"${sci_tex(r2)}$ \\\\"
        ),
        (
            r"Exterior cancellation for $P^{[1]}$ & "
            r"$\min\|{\rm term}\|/\|H_g\psi\|$, $\max r_{\rm pair}$ & "
            f"${sci_tex(p1_min_term)}$, ${sci_tex(p1_ext)}$ \\\\"
        ),
        (
            r"Exterior cancellation for $P^{[2]}$ & "
            r"$\min\|{\rm term}\|/\|H_g\psi\|$, $\max r_{\rm pair}$ & "
            f"${sci_tex(p2_min_term)}$, ${sci_tex(p2_ext)}$ \\\\"
        ),
        (
            r"Nearest-neighbor instantaneous exactness & "
            rf"$\max r_{{\mathrm{{NN}}}}$ & "
            f"${sci_tex(nn)}$ \\\\"
        ),
        (
            r"Minimal-support long-range obstruction & "
            r"$\|P^{[1]}X_1X_4\psi\|/\|X_1X_4\psi\|$, "
            r"$\|P^{[2]}X_1X_4\psi\|/\|X_1X_4\psi\|$ & "
            f"${sci_tex(p1_rel)}$, ${sci_tex(p2_rel)}$ \\\\"
        ),
        r"\hline\hline",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_production_stall_table(stall_rows: list[dict[str, Any]], path: Path) -> None:
    """Compact production product-state stall values for manuscript prose."""
    lines = [
        r"% Auto-generated by experiments/structural_checks/run.py",
        r"% Production full_tdvp on |0000> with R_XX(pi/2) on paper sites (1,4).",
        r"\begin{table}",
        (
            r"\caption{Production product-state stall under unmodified "
            r"``full\_tdvp'': $|\psi\rangle=|0000\rangle$, "
            r"$R_{XX}(\pi/2)$ on paper sites $(1,4)$. "
            r"The state is unchanged, the infidelity relative to the exact "
            r"endpoint is $1/2$, and the peak bond dimension remains one.}"
        ),
        r"\label{tab:production-stall}",
        r"\begin{tabular}{ccccc}",
        r"\hline\hline",
        r"$\chi_{\max}$ & $n_{\mathrm{sub}}$ & $\|\psi'-\psi\|$ & $1-F$ & peak $\chi$ \\",
        r"\hline",
    ]
    for r in stall_rows:
        lines.append(
            f"{r['chi_max']} & {r['n_sub']} & "
            f"${sci_tex(float(r['phase_aligned_distance']))}$ & "
            f"${sci_tex(float(r['infidelity_vs_exact']))}$ & "
            f"{r['peak_retained_rank_during_split']} \\\\"
        )
    lines.extend(
        [
            r"\hline\hline",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Structural checks: locality identities and production product-state stall")

    generator = make_generic_generator(GENERATOR_SEED)
    projector_rows: list[dict[str, Any]] = []
    exterior_rows: list[dict[str, Any]] = []
    fixture_error: str | None = None

    try:
        for seed in MPS_SEEDS:
            psi = random_exact_rank_state(seed, n=N, chi=CHI, d=D)
            sch = compute_schmidt(psi, n=N, chi=CHI)
            projector_rows.extend(check_projector_algebra(sch, seed))
            projector_rows.extend(check_fixed_rank(sch, seed, generator))
            projector_rows.extend(check_two_site(sch, seed, generator))
            projector_rows.extend(check_nn_exactness(sch, seed, generator))
            if seed == EXTERIOR_SEED:
                exterior_rows = check_exterior_cancellation(sch, generator)
    except FixtureError as exc:
        fixture_error = str(exc)
        print(f"FIXTURE FAILURE: {exc}")

    obstruction_rows = check_analytical_obstruction()
    production_rows = check_production_product_sweep()

    projector_fields = [
        "check_family",
        "geometry",
        "seed",
        "N",
        "chi",
        "site_a_zero_based",
        "site_b_zero_based",
        "window_start_zero_based",
        "window_end_zero_based",
        "generator_norm",
        "full_action_norm",
        "absolute_residual",
        "relative_residual",
        "sigma_min_retained",
        "sigma_max_discarded",
        "hermitian_rel",
        "idempotent_rel",
        "pass",
    ]
    exterior_fields = [
        "check_family",
        "side",
        "pair_index",
        "term_a",
        "term_a_index",
        "term_b",
        "term_b_index",
        "term_a_norm",
        "term_b_norm",
        "term_a_rel",
        "term_b_rel",
        "absolute_residual",
        "relative_residual",
        "x_norm",
        "term_floor",
        "pass",
    ]
    obstruction_fields = [
        "check_family",
        "geometry",
        "N",
        "site_a_zero_based",
        "site_b_zero_based",
        "site_a_one_based",
        "site_b_one_based",
        "theta",
        "x_norm",
        "p1_action_norm",
        "p1_action_rel",
        "p2_action_norm",
        "p2_action_rel",
        "infidelity_vs_unchanged",
        "all_schmidt_ranks_one",
        "all_projector_terms_vanish",
        "term_norms_json",
        "chi_max",
        "n_sub",
        "phase_aligned_distance",
        "infidelity_vs_exact",
        "final_bond_profile",
        "peak_retained_rank_during_split",
        "absolute_residual",
        "relative_residual",
        "pass",
    ]
    obstruction_combined = [{**dict.fromkeys(obstruction_fields, ""), **r} for r in obstruction_rows]
    obstruction_combined.extend({**dict.fromkeys(obstruction_fields, ""), **r} for r in production_rows)

    paths = {
        "projector_checks": OUTPUT_DIR / "projector_checks.csv",
        "exterior_cancellation": OUTPUT_DIR / "exterior_cancellation.csv",
        "obstruction_checks": OUTPUT_DIR / "obstruction_checks.csv",
        "summary": OUTPUT_DIR / "summary.json",
        "table": OUTPUT_DIR / "table_structural.tex",
        "table_production_stall": OUTPUT_DIR / "table_production_stall.tex",
    }
    _write_csv(paths["projector_checks"], projector_rows, projector_fields)
    _write_csv(paths["exterior_cancellation"], exterior_rows, exterior_fields)
    _write_csv(paths["obstruction_checks"], obstruction_combined, obstruction_fields)
    write_table(projector_rows, exterior_rows, obstruction_rows, paths["table"])
    write_production_stall_table(production_rows, paths["table_production_stall"])

    all_pass_flags = (
        [bool(r["pass"]) for r in projector_rows]
        + [bool(r["pass"]) for r in exterior_rows]
        + [bool(r["pass"]) for r in obstruction_rows]
        + [bool(r["pass"]) for r in production_rows]
    )
    if fixture_error is not None:
        all_pass_flags.append(False)
    all_pass = all(all_pass_flags) if all_pass_flags else False

    max_by_family: dict[str, float] = {}
    for family in (
        "projector_algebra",
        "fixed_rank_locality",
        "two_site_locality",
        "nearest_neighbor_exactness",
        "exterior_cancellation_P1",
        "exterior_cancellation_P2",
        "analytical_obstruction",
    ):
        source = projector_rows + exterior_rows + obstruction_rows
        max_by_family[family] = _max_rel(source, family)

    summary = {
        "description": (
            "Structural projector validation: locality identities plus the "
            "minimal-support obstruction "
            "(analytical_obstruction / production_product_sweep)"
        ),
        "generated_utc": datetime.now(UTC).isoformat(),
        "experiment_dir": "experiments/structural_checks",
        "dtype": DTYPE,
        "configuration": {
            "N": N,
            "d": D,
            "chi": CHI,
            "mps_seeds": list(MPS_SEEDS),
            "generator_seed": GENERATOR_SEED,
            "bond_profile": bond_profile(),
            "fixed_rank_geometries": {k: list(v) for k, v in FIXED_RANK_GEOMETRIES.items()},
            "two_site_geometries": {k: list(v) for k, v in TWO_SITE_GEOMETRIES.items()},
            "nn_geometries": {k: list(v) for k, v in NN_GEOMETRIES.items()},
            "exterior_seed": EXTERIOR_SEED,
            "exterior_sites": list(EXTERIOR_SITES),
            "product_sweep_configs": list(PRODUCT_SWEEP_CONFIGS),
        },
        "tolerances": {
            "relative_residual": REL_TOL,
            "obstruction_p2": OBSTRUCTION_P2_TOL,
            "production_stall_distance": PRODUCTION_STALL_DIST_TOL,
            "production_stall_infidelity": PRODUCTION_STALL_INFIDELITY_TOL,
            "exterior_term_floor_factor": EXTERIOR_TERM_FLOOR,
        },
        **_package_versions(),
        "fixture_error": fixture_error,
        "maximum_residual_by_family": max_by_family,
        "n_cases": len(all_pass_flags),
        "n_passed": sum(all_pass_flags),
        "n_failed": len(all_pass_flags) - sum(all_pass_flags),
        "all_pass": all_pass,
        "artifacts": {k: str(v.relative_to(EXPERIMENT_DIR)) for k, v in paths.items()},
        "analytical_obstruction": obstruction_rows,
        "production_product_sweep": production_rows,
    }
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print()
    print("=== Structural checks summary ===")
    for family, value in max_by_family.items():
        print(f"  max residual [{family}]: {value:.3e}")
    print("  analytical obstruction:")
    for r in obstruction_rows:
        status = "PASS" if r["pass"] else "FAIL"
        print(
            f"    ||P1 X||/||X||={float(r['p1_action_rel']):.3e} "
            f"||P2 X||/||X||={float(r['p2_action_rel']):.3e} "
            f"infidelity_unchanged={float(r['infidelity_vs_unchanged']):.6f} [{status}]"
        )
    print("  production_product_sweep (unmodified full_tdvp):")
    for r in production_rows:
        status = "PASS" if r["pass"] else "FAIL"
        print(
            f"    chi={r['chi_max']}, n_sub={r['n_sub']}: "
            f"dist={r['phase_aligned_distance']:.3e}, "
            f"infidelity={r['infidelity_vs_exact']:.6f}, "
            f"peak_rank={r['peak_retained_rank_during_split']} [{status}]"
        )
    print(f"\n{summary['n_passed']}/{summary['n_cases']} cases passed; n_failed={summary['n_failed']}")
    print("ALL PASS" if all_pass else "FAILURES PRESENT")
    print(f"Artifacts written under {OUTPUT_DIR}")
    print(f"Manuscript table: {paths['table']}")
    print(f"Production stall table: {paths['table_production_stall']}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
