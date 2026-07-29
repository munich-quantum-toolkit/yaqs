# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Bounded pilot: is complete-generator TDVP advantageous when D_H << D_U?

Candidates:
  1. 2D QAOA/Ising cost layer  U_C(gamma) = exp(-i gamma sum_<ij> Z_i Z_j) on |+>^N
  2. Collective one-axis twisting  U_OAT(kappa) = exp(-i kappa/(N-1) sum_{i<j} X_i X_j) on |0>^N

Usage:
    uv run python experiments/generator_rank_pilot/run_pilot.py [--validate-only]
        [--candidate {qaoa,oat,all}]

Raw CSV: experiments/generator_rank_pilot/output/pilot_results.csv (new directory,
existing benchmark data untouched).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import path_setup  # noqa: F401
import pilot_lib as pl
from gate_runtime import normalized_state_fidelity
from variational import tt_svd_from_vec

from mqt.yaqs.core import linalg as yaqs_linalg
from mqt.yaqs.core.data_structures.mps import MPS

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
CSV_PATH = OUTPUT_DIR / "pilot_results.csv"
META_PATH = OUTPUT_DIR / "meta.json"

CSV_FIELDS = [
    "candidate", "size_label", "length", "angle", "chi", "method", "ordering", "substeps",
    "infidelity", "fidelity", "state_norm", "final_max_bond", "peak_max_bond",
    "final_param_count", "peak_param_count", "est_transient_elements",
    "total_discarded_weight", "max_step_discarded_weight", "max_event_discarded_weight",
    "n_svd", "n_evolver_calls", "runtime_s", "d_h", "d_u", "exact_max_bond", "notes",
]


def kron_chain(ops: dict[int, np.ndarray], length: int) -> np.ndarray:
    """Dense operator with site i on bit i (little-endian, matching MPS.to_vec)."""
    eye = np.eye(2, dtype=np.complex128)
    out = np.array([[1.0]], dtype=np.complex128)
    for site in reversed(range(length)):
        out = np.kron(out, ops.get(site, eye))
    return out


def dense_generator(terms: list[tuple[float, list[tuple[int, str]]]], length: int) -> np.ndarray:
    paulis = {
        "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    }
    h = np.zeros((2**length, 2**length), dtype=np.complex128)
    for coeff, ops in terms:
        h += coeff * kron_chain({site: paulis[p] for site, p in ops}, length)
    return h


def validate() -> dict[str, Any]:
    """Tiny-instance validations (2x2 QAOA, N=4 OAT). Raises AssertionError on failure."""
    report: dict[str, Any] = {}

    # Bit-order convention of MPS.to_vec.
    basis = MPS(4, state="basis", basis_string="1000")
    idx = int(np.argmax(np.abs(basis.to_vec())))
    assert idx in {1, 8}, f"unexpected basis index {idx}"
    little_endian = idx == 1
    report["to_vec_little_endian"] = little_endian
    assert little_endian, "kron_chain assumes little-endian to_vec; adjust if this fails"

    for cand, width_or_n in (("qaoa", 2), ("oat", 4)):
        length = width_or_n**2 if cand == "qaoa" else width_or_n
        angle = 0.7 if cand == "qaoa" else 0.5
        if cand == "qaoa":
            gamma = angle / 2.0
            gates_a = pl.qaoa_gate_list(width_or_n, angle, "horiz_first")
            gates_b = pl.qaoa_gate_list(width_or_n, angle, "vert_first")
            h_mpo = pl.qaoa_generator_mpo(width_or_n, gamma)
            terms = [(gamma, [(a, "Z"), (b, "Z")]) for a, b in pl.grid_edges(width_or_n)]
            init_kind = "x+"
        else:
            gates_a = pl.oat_gate_list(width_or_n, angle, "lexicographic")
            gates_b = pl.oat_gate_list(width_or_n, angle, "by_distance")
            h_mpo = pl.oat_generator_mpo(width_or_n, angle)
            coeff = angle / (width_or_n - 1)
            terms = [
                (coeff, [(i, "X"), (j, "X")])
                for i in range(width_or_n)
                for j in range(i + 1, width_or_n)
            ]
            init_kind = "zeros"

        init_vec = pl.dense_initial(length, init_kind)
        ref_a = pl.dense_reference(length, init_kind, gates_a)
        ref_b = pl.dense_reference(length, init_kind, gates_b)
        assert np.max(np.abs(ref_a - ref_b)) < 1e-12, f"{cand}: gate orderings disagree"

        # Generator exponential agrees with the product of constituent gates.
        h_dense = dense_generator(terms, length)
        ref_gen = yaqs_linalg.expm(-1j * h_dense) @ init_vec
        err = float(np.max(np.abs(ref_gen - ref_a)))
        report[f"{cand}_expm_vs_gates_maxabs"] = err
        assert err < 1e-10, f"{cand}: expm(-iH) disagrees with gate product ({err})"

        # Generator MPO applied to a random state agrees with dense H (validates MPO + bit order).
        rng = np.random.default_rng(7)
        rvec = rng.normal(size=2**length) + 1j * rng.normal(size=2**length)
        rvec /= np.linalg.norm(rvec)
        rmps = tt_svd_from_vec(rvec, length, chi_max=64)
        happlied = copy.deepcopy(rmps)
        h_mpo.multiply(happlied, sim_params=None, compress=False)
        err = float(np.max(np.abs(happlied.to_vec() - h_dense @ rmps.to_vec())))
        report[f"{cand}_hmpo_vs_dense_maxabs"] = err
        assert err < 1e-10, f"{cand}: generator MPO disagrees with dense H ({err})"

        # Untruncated production routes agree with the exact reference.
        init_mps = MPS(length, state=init_kind)
        for label, gates in (("gatewise_a", gates_a), ("gatewise_b", gates_b)):
            rec = pl.run_mpo_gatewise(
                init_mps, gates, chi=64, ordering=label, exact_vec=ref_a, exact_mps=None
            )
            report[f"{cand}_{label}_untrunc_infidelity"] = rec["infidelity"]
            assert rec["infidelity"] < 1e-10, f"{cand} {label}: untruncated route inexact"
            assert abs(rec["state_norm"] - 1.0) < 1e-8, f"{cand} {label}: norm drift"
        layer_mpo = pl.build_layer_unitary_mpo(gates_a, length)
        rec = pl.run_mpo_layer(init_mps, layer_mpo, chi=64, exact_vec=ref_a, exact_mps=None)
        report[f"{cand}_layer_untrunc_infidelity"] = rec["infidelity"]
        assert rec["infidelity"] < 1e-10, f"{cand}: untruncated layer-MPO route inexact"

        # TDVP genuinely evolves under the complete generator.
        rec = pl.run_tdvp_layer(init_mps, h_mpo, chi=16, substeps=8, exact_vec=ref_a, exact_mps=None)
        report[f"{cand}_tdvp_n8_chi16_infidelity"] = rec["infidelity"]
        assert rec["infidelity"] < 1e-3, f"{cand}: TDVP n=8 far from exact ({rec['infidelity']})"
        assert abs(rec["state_norm"] - 1.0) < 1e-6, f"{cand}: TDVP norm drift"
        moved = normalized_state_fidelity(init_vec, ref_a)["fidelity_normalized"]
        report[f"{cand}_initial_vs_exact_fidelity"] = moved
        assert moved < 0.999, f"{cand}: layer barely moves the state; test not meaningful"

        # Zero-angle identity: gate routes and TDVP (H=0 handled by Lanczos breakdown or skip).
        zero_gates = [(name, 0.0, a, b) for name, _, a, b in gates_a]
        rec = pl.run_mpo_gatewise(
            init_mps, zero_gates, chi=64, ordering="zero", exact_vec=init_vec, exact_mps=None
        )
        assert rec["infidelity"] < 1e-12, f"{cand}: zero-angle gatewise fails"
        h0 = pl.qaoa_generator_mpo(width_or_n, 0.0) if cand == "qaoa" else pl.oat_generator_mpo(width_or_n, 0.0)
        try:
            rec = pl.run_tdvp_layer(init_mps, h0, chi=16, substeps=1, exact_vec=init_vec, exact_mps=None)
            report[f"{cand}_tdvp_zero_infidelity"] = rec["infidelity"]
            assert rec["infidelity"] < 1e-10, f"{cand}: zero-angle TDVP fails"
        except (ValueError, ZeroDivisionError, FloatingPointError) as exc:  # Lanczos on H=0
            report[f"{cand}_tdvp_zero_infidelity"] = f"skipped ({exc})"

        # Exact-MPS construction (tt_svd) reproduces the dense reference.
        exact_mps = tt_svd_from_vec(ref_a, length, chi_max=4096)
        fid = normalized_state_fidelity(ref_a, exact_mps.to_vec())["fidelity_normalized"]
        assert 1.0 - fid < 1e-12, f"{cand}: tt_svd exact MPS mismatch"

        report[f"{cand}_tiny_d_h"] = pl.mpo_max_bond(h_mpo)
        report[f"{cand}_tiny_d_u"] = pl.mpo_max_bond(layer_mpo)

    return report


def append_rows(rows: list[dict[str, Any]]) -> None:
    exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def run_config(
    *,
    candidate: str,
    size_label: str,
    length: int,
    angle: float,
    chis: list[int],
    init_kind: str,
    gates_by_ordering: dict[str, list[tuple[str, float, int, int]]],
    h_mpo,
    exact_vec: np.ndarray | None,
    exact_mps: MPS,
    d_h: int,
    d_u: int,
    substep_grid: tuple[int, ...] = (1, 2, 4, 8),
    representative_n16: bool = False,
) -> None:
    init_mps = MPS(length, state=init_kind)
    exact_max_bond = max(pl.bond_profile(exact_mps))
    layer_mpo = pl.build_layer_unitary_mpo(next(iter(gates_by_ordering.values())), length)

    base = {
        "candidate": candidate, "size_label": size_label, "length": length,
        "angle": angle, "d_h": d_h, "d_u": d_u, "exact_max_bond": exact_max_bond,
    }

    for chi in chis:
        rows: list[dict[str, Any]] = []
        tdvp_recs: dict[int, dict[str, Any]] = {}
        substeps = list(substep_grid)
        for n in substeps:
            rec = pl.run_tdvp_layer(
                init_mps, h_mpo, chi=chi, substeps=n, exact_vec=exact_vec, exact_mps=exact_mps
            )
            tdvp_recs[n] = rec
            rows.append({**base, "chi": chi, **rec})
        if representative_n16 and chi == max(chis):
            inf8, inf4 = tdvp_recs[8]["infidelity"], tdvp_recs[4]["infidelity"]
            if abs(inf8 - inf4) > 0.1 * max(abs(inf8), 1e-16):
                rec = pl.run_tdvp_layer(
                    init_mps, h_mpo, chi=chi, substeps=16, exact_vec=exact_vec, exact_mps=exact_mps
                )
                rec["notes"] = "n16_topup"
                rows.append({**base, "chi": chi, **rec})

        for ordering, gates in gates_by_ordering.items():
            rec = pl.run_mpo_gatewise(
                init_mps, gates, chi=chi, ordering=ordering, exact_vec=exact_vec, exact_mps=exact_mps
            )
            rows.append({**base, "chi": chi, **rec})

        rec = pl.run_mpo_layer(init_mps, layer_mpo, chi=chi, exact_vec=exact_vec, exact_mps=exact_mps)
        rows.append({**base, "chi": chi, **rec})

        # Variational layer fit: init from the first gatewise ordering's result.
        _first_ordering, first_gates = next(iter(gates_by_ordering.items()))
        var_init = MPS(length, state=init_kind)
        params = pl._params(chi, gate_mode="mpo", tdvp_sweeps=1)  # noqa: SLF001
        from gate_runtime import make_dag_node

        from mqt.yaqs.digital.digital_tjm import apply_two_qubit_gate
        for name, theta, q0, q1 in first_gates:
            apply_two_qubit_gate(var_init, make_dag_node(name, theta, q0, q1, length), params)
        try:
            rec = pl.run_variational_layer(
                exact_mps, var_init, chi=chi, exact_vec=exact_vec, exact_mps=exact_mps
            )
        except RuntimeError as exc:
            rec = {"method": "variational_layer", "ordering": "", "substeps": 0,
                   "notes": f"FAILED: {exc}", "infidelity": "", "fidelity": ""}
        rows.append({**base, "chi": chi, **rec})

        rec = pl.run_oracle(exact_mps, chi=chi, exact_vec=exact_vec, exact_mps=exact_mps)
        rows.append({**base, "chi": chi, **rec})

        append_rows(rows)
        best_inf = min((r["infidelity"] for r in rows if r.get("infidelity") != ""), default=float("nan"))
        print(  # noqa: T201
            f"[{candidate} {size_label} angle={angle} chi={chi}] {len(rows)} runs, "
            f"best infidelity {best_inf:.3e}",
            flush=True,
        )


def run_qaoa() -> None:
    for width, chis in ((4, [2, 4, 8, 16, 32]), (5, [4, 8, 16, 32, 64])):
        length = width * width
        for theta in (0.3, 0.7):
            gamma = theta / 2.0
            gates = {
                "horiz_first": pl.qaoa_gate_list(width, theta, "horiz_first"),
                "vert_first": pl.qaoa_gate_list(width, theta, "vert_first"),
            }
            h_mpo = pl.qaoa_generator_mpo(width, gamma)
            d_h = pl.mpo_max_bond(h_mpo)
            layer_for_du = pl.build_layer_unitary_mpo(gates["horiz_first"], length)
            d_u = pl.mpo_max_bond(layer_for_du)
            t0 = time.perf_counter()
            if length <= 20:
                exact_vec = pl.dense_reference(length, "x+", gates["horiz_first"])
                exact_mps = tt_svd_from_vec(exact_vec, length, chi_max=4096)
            else:
                exact_vec = None
                exact_mps = pl.exact_mps_reference(length, "x+", gates["horiz_first"])
                # Cross-check against the independent uncapped layer-MPO route.
                chk = MPS(length, state="x+")
                layer_for_du.multiply(chk, sim_params=None, compress=False)
                chk.compress(1e-15, max_bond_dim=None, trunc_mode="discarded_weight")
                fid = pl.mps_fidelity(exact_mps, chk)
                assert 1.0 - fid < 1e-9, f"5x5 exact-reference cross-check failed ({1.0 - fid})"
            print(  # noqa: T201
                f"[qaoa {width}x{width} theta={theta}] reference built in "
                f"{time.perf_counter() - t0:.1f}s, D_H={d_h}, D_U={d_u}, "
                f"exact max bond={max(pl.bond_profile(exact_mps))}",
                flush=True,
            )
            run_config(
                candidate="qaoa", size_label=f"{width}x{width}", length=length, angle=theta,
                chis=chis, init_kind="x+", gates_by_ordering=gates, h_mpo=h_mpo,
                exact_vec=exact_vec, exact_mps=exact_mps, d_h=d_h, d_u=d_u,
                representative_n16=(width == 5 and theta == 0.3),
            )


def run_oat() -> None:
    for n_qubits, chis in ((16, [2, 4, 8, 16]), (20, [2, 4, 8, 16, 32])):
        for kappa in (0.5, 1.5):
            gates = {
                "lexicographic": pl.oat_gate_list(n_qubits, kappa, "lexicographic"),
                "by_distance": pl.oat_gate_list(n_qubits, kappa, "by_distance"),
            }
            h_mpo = pl.oat_generator_mpo(n_qubits, kappa)
            d_h = pl.mpo_max_bond(h_mpo)
            layer_for_du = pl.build_layer_unitary_mpo(gates["lexicographic"], n_qubits)
            d_u = pl.mpo_max_bond(layer_for_du)
            t0 = time.perf_counter()
            exact_vec = pl.dense_reference(n_qubits, "zeros", gates["lexicographic"])
            exact_mps = tt_svd_from_vec(exact_vec, n_qubits, chi_max=4096)
            print(  # noqa: T201
                f"[oat N={n_qubits} kappa={kappa}] reference built in "
                f"{time.perf_counter() - t0:.1f}s, D_H={d_h}, D_U={d_u}, "
                f"exact max bond={max(pl.bond_profile(exact_mps))}",
                flush=True,
            )
            run_config(
                candidate="oat", size_label=f"N{n_qubits}", length=n_qubits, angle=kappa,
                chis=chis, init_kind="zeros", gates_by_ordering=gates, h_mpo=h_mpo,
                exact_vec=exact_vec, exact_mps=exact_mps, d_h=d_h, d_u=d_u,
                representative_n16=(n_qubits == 20 and kappa == 0.5),
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--candidate", choices=["qaoa", "oat", "all"], default="all")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    print("Running tiny-instance validations...", flush=True)  # noqa: T201
    t0 = time.perf_counter()
    validation = validate()
    print(f"Validations passed in {time.perf_counter() - t0:.1f}s", flush=True)  # noqa: T201
    for key, val in validation.items():
        print(f"  {key}: {val}", flush=True)  # noqa: T201

    META_PATH.write_text(json.dumps({
        "settings": pl.settings_note(),
        "angle_convention": {
            "qaoa": "edge Rzz(theta)=exp(-i theta/2 ZZ); layer U_C=exp(-i gamma H_C) with gamma=theta/2; "
                    "angle column = theta (0.3, 0.7) => gamma in {0.15, 0.35}",
            "oat": "pair Rxx(theta)=exp(-i theta/2 XX) with theta=2*kappa/(N-1); "
                   "angle column = kappa",
        },
        "validation": {k: (v if isinstance(v, (bool, str)) else float(v)) for k, v in validation.items()},
    }, indent=2))

    if args.validate_only:
        return

    if CSV_PATH.exists():
        CSV_PATH.unlink()

    if args.candidate in {"qaoa", "all"}:
        run_qaoa()
    if args.candidate in {"oat", "all"}:
        run_oat()
    print(f"Done. Results: {CSV_PATH}", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()
