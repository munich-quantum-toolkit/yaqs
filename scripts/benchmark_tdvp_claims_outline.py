#!/usr/bin/env python
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Lightweight benchmark for Circuit TDVP paper claims (laptop-feasible).

Run::

    uv run python -m scripts.benchmark_tdvp_claims_outline

Environment::

    YAQS_TDVP_CLAIMS_STAGE=0|1|2
    YAQS_TDVP_CLAIMS_OVERWRITE=0|1
    YAQS_TDVP_CLAIMS_MAX_CASES=0
    YAQS_TDVP_CLAIMS_INCLUDE_FLOQUET=0|1
    YAQS_TDVP_CLAIMS_INCLUDE_DENSE_FAILURE=0|1
    YAQS_TDVP_CLAIMS_INCLUDE_SWEEP_SCAN=0|1
    YAQS_TDVP_CLAIMS_OUTPUT_DIR=results/tdvp_claims
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from mqt.yaqs.core.data_structures.mps import MPS

from scripts.benchmark_long_range_tdvp_regimes import (
    CircuitSpec,
    RawRun,
    _assign_dispatch_fields,
    _assign_padding_fields,
    _run_method,
    compute_reference,
    make_circuit_family,
    run_method,
    validate_reference_convention,
    validate_tdvp_padding,
)
from scripts.benchmark_nn_tdvp_regimes import run_nn_method
from scripts.benchmark_utils import _mean_bond_dim, _prep_initial_state
from scripts.yaqs_reference_utils import (
    ReferenceConvention,
    check_high_fidelity_observable_consistency,
    compare_statevectors,
    qiskit_reference_vec,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STAGE = int(os.environ.get("YAQS_TDVP_CLAIMS_STAGE", "0"))
OVERWRITE = os.environ.get("YAQS_TDVP_CLAIMS_OVERWRITE", "0").strip() not in {"", "0", "false", "False"}
MAX_CASES = int(os.environ.get("YAQS_TDVP_CLAIMS_MAX_CASES", "0"))
INCLUDE_FLOQUET = os.environ.get(
    "YAQS_TDVP_CLAIMS_INCLUDE_FLOQUET",
    "1" if STAGE >= 1 else "0",
).strip() not in {"", "0", "false", "False"}
INCLUDE_DENSE_FAILURE = os.environ.get("YAQS_TDVP_CLAIMS_INCLUDE_DENSE_FAILURE", "0").strip() not in {
    "",
    "0",
    "false",
    "False",
}
INCLUDE_SWEEP_SCAN = os.environ.get("YAQS_TDVP_CLAIMS_INCLUDE_SWEEP_SCAN", "0").strip() not in {
    "",
    "0",
    "false",
    "False",
}
OUTDIR = Path(os.environ.get("YAQS_TDVP_CLAIMS_OUTPUT_DIR", "results/tdvp_claims"))

BASE_SEED = 4242
TDVP_SWEEPS_DEFAULT = 1
EXACT_N_MAX = 16
HIGH_BUDGET_CHI = 256

LrMethod = Literal["tebd_swap", "hybrid_tdvp_unpadded", "hybrid_tdvp_padded4"]
NnMethod = Literal["tebd_nn", "tdvp_all"]
MethodName = LrMethod | NnMethod

METHODS_BY_FAMILY: dict[str, list[str]] = {
    "nn_brickwork": ["tebd_nn", "tdvp_all"],
    "periodic_1d": ["tebd_swap", "hybrid_tdvp_unpadded", "hybrid_tdvp_padded4"],
    "sparse_long_range": ["tebd_swap", "hybrid_tdvp_unpadded", "hybrid_tdvp_padded4"],
    "dense_long_range_failure": ["tebd_swap", "hybrid_tdvp_unpadded", "hybrid_tdvp_padded4"],
    "heisenberg_floquet_2d": ["tebd_swap", "hybrid_tdvp_unpadded", "hybrid_tdvp_padded4"],
}

STAGE_CONFIGS: dict[int, dict[str, Any]] = {
    0: {
        "n_values": [8, 10],
        "depth_values": [2, 4],
        "chi_values": [8, 16, 32],
        "seeds": [0, 1],
        "angle_regimes": ["small"],
        "families": ["nn_brickwork", "periodic_1d", "sparse_long_range"],
        "exact_n_max": EXACT_N_MAX,
    },
    1: {
        "n_values": [8, 12, 16],
        "depth_values": [4, 8],
        "chi_values": [8, 16, 24, 32, 48],
        "seeds": [0, 1, 2],
        "angle_regimes": ["small", "medium"],
        "families": ["nn_brickwork", "periodic_1d", "sparse_long_range"],
        "exact_n_max": EXACT_N_MAX,
        "floquet": {
            "lattices": [(3, 4), (4, 4)],
            "cycles": [2],
            "J_over_pi_values": [0.005, 0.01, 0.02, 0.04, 0.06, 0.08],
            "chi_values": [16, 32, 64],
            "seeds": [0, 1],
        },
    },
    2: {
        "n_values": [12, 16, 24],
        "depth_values": [8, 12, 16],
        "chi_values": [8, 12, 16, 24, 32, 48, 64],
        "seeds": [0, 1, 2, 3],
        "angle_regimes": ["small", "medium"],
        "families": ["sparse_long_range"],
        "optional_families": ["periodic_1d"],
        "exact_n_max": EXACT_N_MAX,
        "floquet": {
            "lattices": [(4, 4), (5, 4), (5, 5)],
            "cycles": [2, 3],
            "J_over_pi_values": [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10],
            "chi_values": [16, 32, 64, 128],
            "seeds": [0, 1, 2],
        },
    },
}


@dataclass
class ClaimsSpec:
    """One benchmark case (Pauli circuits or Floquet application)."""

    case_id: str
    family: str
    n_qubits: int
    depth: int
    seed: int
    angle_regime: str
    qc: QuantumCircuit
    lr_pairs: tuple[tuple[int, int], ...]
    nn_pairs: tuple[tuple[int, int], ...]
    topology: str = "claims"
    lattice_shape: tuple[int, int] | None = None
    J_over_pi: float | None = None
    initial_state: str = "plus"
    floquet_patches: tuple[tuple[str, tuple[int, ...]], ...] = field(default_factory=tuple)

    def to_circuit_spec(self) -> CircuitSpec:
        return CircuitSpec(
            family=self.family,
            n_qubits=self.n_qubits,
            depth=self.depth,
            seed=self.seed,
            angle_regime=self.angle_regime,
            case_id=self.case_id,
            qc=self.qc,
            lr_pairs=self.lr_pairs,
            nn_pairs=self.nn_pairs,
            topology=self.topology,
            lx=self.lattice_shape[0] if self.lattice_shape else None,
            ly=self.lattice_shape[1] if self.lattice_shape else None,
        )


# ---------------------------------------------------------------------------
# Circuit builders
# ---------------------------------------------------------------------------


def _rng_for(*parts: Any) -> np.random.Generator:
    h = hash(parts) % 10_000_007
    return np.random.default_rng(BASE_SEED + h)


def _sample_angle(regime: str, rng: np.random.Generator) -> float:
    if regime == "small":
        return float(rng.uniform(0.02, 0.08))
    if regime == "medium":
        return float(rng.uniform(0.08, 0.20))
    return float(rng.uniform(0.02, 0.08))


def _add_pauli(qc: QuantumCircuit, name: str, theta: float, i: int, j: int) -> None:
    getattr(qc, name)(theta, i, j)


def _sample_distinct_pair(n: int, rng: np.random.Generator) -> tuple[int, int]:
    a, b = (int(x) for x in rng.choice(n, size=2, replace=False))
    return (a, b) if a < b else (b, a)


def make_sparse_long_range_claims(
    *,
    n: int,
    depth: int,
    seed: int,
    angle_regime: str,
) -> ClaimsSpec:
    rng = _rng_for("sparse", n, depth, seed, angle_regime)
    qc = QuantumCircuit(n)
    lr_pairs: set[tuple[int, int]] = set()
    nn_pairs: set[tuple[int, int]] = set()
    min_range = max(2, n // 3)
    num_lr = max(1, n // 4)
    for layer in range(depth):
        for i in range(n - 1):
            _add_pauli(qc, "rzz", _sample_angle(angle_regime, rng), i, i + 1)
            nn_pairs.add((i, i + 1))
        candidates = [(i, j) for i in range(n) for j in range(i + min_range, n)]
        rng.shuffle(candidates)
        gnames = ("rxx", "ryy", "rzz")
        for pair in candidates[: min(num_lr, len(candidates))]:
            g = gnames[layer % 3]
            _add_pauli(qc, g, _sample_angle(angle_regime, rng), pair[0], pair[1])
            lr_pairs.add(pair)
        qc.barrier()
    cid = f"sparse_long_range_n{n}_d{depth}_s{seed}_{angle_regime}"
    return ClaimsSpec(
        case_id=cid,
        family="sparse_long_range",
        n_qubits=n,
        depth=depth,
        seed=seed,
        angle_regime=angle_regime,
        qc=qc,
        lr_pairs=tuple(sorted(lr_pairs)),
        nn_pairs=tuple(sorted(nn_pairs)),
        topology="sparse_lr_claims",
    )


def make_claims_spec(family: str, *, n: int, depth: int, seed: int, angle_regime: str) -> ClaimsSpec:
    if family == "sparse_long_range":
        return make_sparse_long_range_claims(n=n, depth=depth, seed=seed, angle_regime=angle_regime)
    base = make_circuit_family(family, n=n, depth=depth, seed=seed, angle_regime=angle_regime)
    return ClaimsSpec(
        case_id=base.case_id,
        family=base.family,
        n_qubits=base.n_qubits,
        depth=base.depth,
        seed=base.seed,
        angle_regime=base.angle_regime,
        qc=base.qc,
        lr_pairs=base.lr_pairs,
        nn_pairs=base.nn_pairs,
        topology=base.topology,
    )


def _snake_index(row: int, col: int, num_rows: int, num_cols: int) -> int:
    if row % 2 == 0:
        return row * num_cols + col
    return row * num_cols + (num_cols - 1 - col)


def _heisenberg_bond_gate(
    qc: QuantumCircuit,
    i: int,
    j: int,
    *,
    J: float,
    h_i: float,
    h_j: float,
) -> None:
    qc.rz(-2.0 * h_i, i)
    qc.rz(-2.0 * h_j, j)
    theta = -2.0 * J
    qc.rxx(theta, i, j)
    qc.ryy(theta, i, j)
    qc.rzz(theta, i, j)


def make_heisenberg_floquet_spec(
    *,
    lx: int,
    ly: int,
    cycles: int,
    seed: int,
    J_over_pi: float,
) -> ClaimsSpec:
    n = lx * ly
    rng = _rng_for("floquet", lx, ly, cycles, seed, J_over_pi)
    J = float(J_over_pi * np.pi)
    h_fields = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    qc = QuantumCircuit(n)
    lr_pairs: set[tuple[int, int]] = set()
    nn_pairs: set[tuple[int, int]] = set()

    def add_bond(r1: int, c1: int, r2: int, c2: int) -> None:
        i = _snake_index(r1, c1, lx, ly)
        j = _snake_index(r2, c2, lx, ly)
        if i == j:
            return
        a, b = (i, j) if i < j else (j, i)
        _heisenberg_bond_gate(qc, a, b, J=J, h_i=float(h_fields[a]), h_j=float(h_fields[b]))
        if abs(a - b) == 1:
            nn_pairs.add((a, b))
        else:
            lr_pairs.add((a, b))

    for _ in range(cycles):
        for row in range(lx):
            for col in range(0, ly - 1, 2):
                add_bond(row, col, row, col + 1)
        for row in range(lx):
            for col in range(1, ly - 1, 2):
                add_bond(row, col, row, col + 1)
        for col in range(ly):
            for row in range(0, lx - 1, 2):
                add_bond(row, col, row + 1, col)
        for col in range(ly):
            for row in range(1, lx - 1, 2):
                add_bond(row, col, row + 1, col)
        qc.barrier()

    cr, cc = lx // 2, ly // 2
    patches: list[tuple[str, tuple[int, ...]]] = [
        ("1x1", (_snake_index(cr, cc, lx, ly),)),
        ("2x2", tuple(
            sorted({
                _snake_index(cr + dr, cc + dc, lx, ly)
                for dr in (0, 1)
                for dc in (0, 1)
                if cr + dr < lx and cc + dc < ly
            })
        )),
    ]
    if ly >= 2:
        patches.append(("1x2", (_snake_index(cr, cc, lx, ly), _snake_index(cr, min(cc + 1, ly - 1), lx, ly))))
    cid = f"heisenberg_floquet_2d_{lx}x{ly}_c{cycles}_J{J_over_pi:.4g}_s{seed}"
    return ClaimsSpec(
        case_id=cid,
        family="heisenberg_floquet_2d",
        n_qubits=n,
        depth=cycles,
        seed=seed,
        angle_regime="floquet",
        qc=qc,
        lr_pairs=tuple(sorted(lr_pairs)),
        nn_pairs=tuple(sorted(nn_pairs)),
        topology=f"2d_snake_{lx}x{ly}",
        lattice_shape=(lx, ly),
        J_over_pi=J_over_pi,
        initial_state="neel",
        floquet_patches=tuple(patches),
    )


def build_claims_specs(cfg: dict[str, Any]) -> list[ClaimsSpec]:
    specs: list[ClaimsSpec] = []
    families = list(cfg["families"])
    if INCLUDE_DENSE_FAILURE and "dense_long_range_failure" not in families:
        families.append("dense_long_range_failure")
    if STAGE == 2 and cfg.get("optional_families"):
        for fam in cfg["optional_families"]:
            if fam not in families:
                families.append(fam)
    for family in families:
        fam = "dense_long_range" if family == "dense_long_range_failure" else family
        for n in cfg["n_values"]:
            for depth in cfg["depth_values"]:
                for seed in cfg["seeds"]:
                    for ang in cfg["angle_regimes"]:
                        if family == "sparse_long_range":
                            specs.append(
                                make_sparse_long_range_claims(
                                    n=n, depth=depth, seed=seed, angle_regime=ang
                                )
                            )
                            continue
                        base = make_circuit_family(
                            fam, n=n, depth=depth, seed=seed, angle_regime=ang
                        )
                        cid = base.case_id
                        if family == "dense_long_range_failure":
                            cid = cid.replace("dense_long_range", "dense_long_range_failure")
                        specs.append(
                            ClaimsSpec(
                                case_id=cid,
                                family=family,
                                n_qubits=base.n_qubits,
                                depth=base.depth,
                                seed=base.seed,
                                angle_regime=base.angle_regime,
                                qc=base.qc,
                                lr_pairs=base.lr_pairs,
                                nn_pairs=base.nn_pairs,
                                topology=base.topology,
                            )
                        )
    if INCLUDE_FLOQUET and "floquet" in cfg:
        fc = cfg["floquet"]
        for lx, ly in fc["lattices"]:
            for cycles in fc["cycles"]:
                for jop in fc["J_over_pi_values"]:
                    for seed in fc["seeds"]:
                        specs.append(
                            make_heisenberg_floquet_spec(
                                lx=lx, ly=ly, cycles=cycles, seed=seed, J_over_pi=float(jop)
                            )
                        )
    if MAX_CASES > 0:
        return specs[:MAX_CASES]
    return specs


# ---------------------------------------------------------------------------
# Collision entropy / Floquet observables
# ---------------------------------------------------------------------------


def _z_string_expectation(vec: np.ndarray, n: int, sites: list[int], pattern: int) -> float:
    dim = 1 << n
    vec = np.asarray(vec, dtype=np.complex128).ravel()
    m = len(sites)
    out = 0.0
    for idx in range(dim):
        amp2 = float(abs(vec[idx]) ** 2)
        if amp2 == 0.0:
            continue
        sign = 1
        for k in range(m):
            if (pattern >> k) & 1:
                if (idx >> sites[k]) & 1:
                    sign *= -1
        out += sign * amp2
    return out


def ipr2_z_from_statevector(vec: np.ndarray, n: int, sites: tuple[int, ...]) -> float:
    m = len(sites)
    if m == 0:
        return 1.0
    total = 0.0
    n_pat = 1 << m
    site_list = list(sites)
    for pattern in range(n_pat):
        z_exp = _z_string_expectation(vec, n, site_list, pattern)
        total += z_exp * z_exp
    return total / n_pat


def s2_z_from_ipr2(ipr2: float) -> float:
    if ipr2 <= 0.0:
        return float("nan")
    return float(-math.log(ipr2))


def _floquet_reference_vec(spec: ClaimsSpec) -> tuple[np.ndarray | None, str, int | None]:
    prep = QuantumCircuit(spec.n_qubits)
    _prep_initial_state(prep, spec.initial_state, seed=spec.seed)  # type: ignore[arg-type]
    full = prep.compose(spec.qc)
    if spec.n_qubits <= EXACT_N_MAX:
        return qiskit_reference_vec(full), "exact_statevector", None
    try:
        cs = spec.to_circuit_spec()
        ref_mps, rtype, rchi, _ = compute_reference(cs, exact_n_max=0)
        if isinstance(ref_mps, MPS):
            return np.asarray(ref_mps.to_vec(), dtype=np.complex128), rtype, rchi
    except Exception:
        pass
    return None, "none", None


def compute_floquet_patch_rows(
    spec: ClaimsSpec,
    mps: MPS,
    *,
    method: str,
    chi: int,
    ref_vec: np.ndarray | None,
    ref_type: str,
) -> list[dict[str, Any]]:
    vec = np.asarray(mps.to_vec(), dtype=np.complex128)
    rows: list[dict[str, Any]] = []
    parity_errs: list[float] = []
    for patch_shape, sites in spec.floquet_patches:
        ipr2 = ipr2_z_from_statevector(vec, spec.n_qubits, sites)
        s2 = s2_z_from_ipr2(ipr2)
        ipr2_err = s2_err = None
        if ref_vec is not None:
            ref_ipr2 = ipr2_z_from_statevector(ref_vec, spec.n_qubits, sites)
            ref_s2 = s2_z_from_ipr2(ref_ipr2)
            ipr2_err = abs(ipr2 - ref_ipr2)
            s2_err = abs(s2 - ref_s2) if not math.isnan(s2) and not math.isnan(ref_s2) else None
            m = len(sites)
            for pattern in range(1 << m):
                z_got = _z_string_expectation(vec, spec.n_qubits, list(sites), pattern)
                z_ref = _z_string_expectation(ref_vec, spec.n_qubits, list(sites), pattern)
                parity_errs.append(abs(z_got - z_ref))
        rows.append({
            "case_id": spec.case_id,
            "family": spec.family,
            "method": method,
            "chi_max": chi,
            "J_over_pi": spec.J_over_pi,
            "seed": spec.seed,
            "lattice_shape": f"{spec.lattice_shape[0]}x{spec.lattice_shape[1]}" if spec.lattice_shape else "",
            "depth_or_cycles": spec.depth,
            "patch_shape": patch_shape,
            "patch_sites": json.dumps(list(sites)),
            "ipr2_z": ipr2,
            "s2_z": s2,
            "ipr2_z_error": ipr2_err,
            "s2_z_error": s2_err,
            "reference_type": ref_type,
        })
    if rows and parity_errs:
        mean_pe = float(np.mean(parity_errs))
        max_pe = float(np.max(parity_errs))
        for row in rows:
            row["mean_abs_z_parity_error"] = mean_pe
            row["max_abs_z_parity_error"] = max_pe
            row["num_z_parity_terms"] = len(parity_errs)
    return rows


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_claims_benchmark(*, chi: int = 32) -> None:
    """Quick pre-flight checks before the claims benchmark."""
    print("=== Claims benchmark validation ===")
    n = 4
    prep = QuantumCircuit(n)
    _prep_initial_state(prep, "plus", seed=0)  # type: ignore[arg-type]
    ref_id = qiskit_reference_vec(prep)
    from mqt.yaqs.core.data_structures.state import State
    from mqt.yaqs.digital.digital_tjm import apply_single_qubit_gate

    st = State(n, initial="zeros", representation="mps")
    mps = st.mps
    for node in circuit_to_dag(prep).topological_op_nodes():
        apply_single_qubit_gate(mps, node)
    f_id, _, _ = compare_statevectors(np.asarray(mps.to_vec()), ref_id, n=n)
    if f_id < 1.0 - 1e-10:
        raise SystemExit(f"Identity/prep validation failed: f={f_id:.3e}")

    nn = make_claims_spec("nn_brickwork", n=8, depth=2, seed=0, angle_regime="small")
    ref, _, _, _ = compute_reference(nn.to_circuit_spec(), exact_n_max=16)
    tebd = run_nn_method(
        nn.to_circuit_spec(),
        method="tebd_nn",
        chi=chi,
        tdvp_sweeps=1,
        sweep_scan_enabled=False,
        ref=ref,
        ref_type="exact_statevector",
        ref_chi=None,
        ref_sweeps=None,
    )
    hybrid = run_method(
        nn.to_circuit_spec(),
        method="hybrid_tdvp_unpadded",
        chi=chi,
        tdvp_sweeps=1,
        ref=ref,
        ref_type="exact_statevector",
        ref_chi=None,
        ref_sweeps=None,
        bond_history=[],
    )
    if tebd.status != "ok" or hybrid.status != "ok":
        raise SystemExit("NN-only TEBD/hybrid validation failed")
    if hybrid.lr_gate_count != 0:
        raise SystemExit("NN circuit should have lr_gate_count=0")

    qc = QuantumCircuit(8)
    qc.rzz(0.1, 0, 7)
    per = CircuitSpec(
        family="periodic_1d",
        n_qubits=8,
        depth=1,
        seed=0,
        angle_regime="small",
        case_id="validation_periodic",
        qc=qc,
        lr_pairs=((0, 7),),
        nn_pairs=(),
        topology="validation",
    )
    _, _, d_h, _ = _run_method(qc, per, method="hybrid_tdvp_unpadded", chi=chi, tdvp_sweeps=1)
    _, _, d_t, _ = _run_method(qc, per, method="tebd_swap", chi=chi, tdvp_sweeps=1)
    if d_h.tdvp_lr_gate_count != 1:
        raise SystemExit(f"Expected tdvp_lr=1, got {d_h.tdvp_lr_gate_count}")
    if d_t.tebd_swap_gate_count != 1:
        raise SystemExit(f"Expected tebd_swap=1, got {d_t.tebd_swap_gate_count}")

    validate_tdvp_padding(chi=chi)
    print("=== Claims validation PASSED ===\n")


# ---------------------------------------------------------------------------
# Run / IO
# ---------------------------------------------------------------------------


def _resolve_sweeps() -> list[int]:
    if INCLUDE_SWEEP_SCAN:
        return [1, 2, 4]
    return [TDVP_SWEEPS_DEFAULT]


def _run_key(spec: ClaimsSpec, method: str, chi: int, sweeps: int) -> str:
    j = "" if spec.J_over_pi is None else f"|J{spec.J_over_pi:.6g}"
    return f"{spec.case_id}|{method}|chi{chi}|sweeps{sweeps}{j}"


def _run_to_row(spec: ClaimsSpec, run: Any, *, chi: int) -> dict[str, Any]:
    d = asdict(run)
    d["lattice_shape"] = (
        f"{spec.lattice_shape[0]}x{spec.lattice_shape[1]}" if spec.lattice_shape else ""
    )
    d["depth_or_cycles"] = spec.depth
    d["J_over_pi"] = spec.J_over_pi if spec.J_over_pi is not None else ""
    d["chi_max"] = chi
    return d


def _run_floquet_lr_method(
    spec: ClaimsSpec,
    *,
    method: LrMethod,
    chi: int,
    tdvp_sweeps: int,
    ref: Any,
    ref_type: str,
    ref_chi: int | None,
) -> tuple[RawRun, MPS]:
    """Simulate Floquet circuits with Néel initial state."""
    from scripts.benchmark_long_range_tdvp_regimes import (
        REFERENCE_CONVENTION,
        _apply_grouped_observable_errors,
    )

    cs = spec.to_circuit_spec()
    mps, wall, dispatch, padding = _run_method(
        spec.qc,
        cs,
        method=method,
        chi=chi,
        tdvp_sweeps=tdvp_sweeps,
        initial_state=spec.initial_state,
    )
    run = RawRun(
        family=spec.family,
        n_qubits=spec.n_qubits,
        depth=spec.depth,
        seed=spec.seed,
        angle_regime=spec.angle_regime,
        method=method,
        chi_max=chi,
        tdvp_sweeps=tdvp_sweeps,
        svd_threshold=1e-10,
        lanczos_tol=1e-8,
        reference_type=ref_type,
        reference_available=ref is not None,
        reference_chi=ref_chi,
        case_id=spec.case_id,
        sweep_scan_enabled=INCLUDE_SWEEP_SCAN,
    )
    run.wall_time_s = wall
    run.peak_bond_dim = int(getattr(mps, "_bench_peak_bond", mps.get_max_bond()))  # type: ignore[attr-defined]
    run.final_max_bond_dim = int(mps.get_max_bond())
    run.mean_bond_dim = _mean_bond_dim(mps)
    run.hit_chi_max = run.final_max_bond_dim >= chi
    _assign_dispatch_fields(run, dispatch)
    _assign_padding_fields(run, padding)
    if ref is not None and isinstance(ref, np.ndarray):
        convention: ReferenceConvention = REFERENCE_CONVENTION
        vec = np.asarray(mps.to_vec(), dtype=np.complex128)
        f_dir, f_rev, conv_sv = compare_statevectors(vec, ref, n=spec.n_qubits)
        run.fidelity_direct = f_dir
        run.fidelity_bit_reversed = f_rev
        if conv_sv == "bit_reversed" and f_rev > f_dir + 1e-10:
            convention = "bit_reversed"
            run.infidelity_to_reference = 1.0 - f_rev
        else:
            convention = "direct"
            run.infidelity_to_reference = 1.0 - f_dir
        run.fidelity_to_reference = 1.0 - run.infidelity_to_reference
        run.reference_convention_used = convention
        _apply_grouped_observable_errors(run, cs, mps, ref, convention=convention)
        warning = check_high_fidelity_observable_consistency(
            run.fidelity_to_reference, run.max_abs_observable_error,
        )
        run.fidelity_observable_consistency_warning = warning
    run.status = "ok"
    return run, mps


def execute_claims_run(
    spec: ClaimsSpec,
    method: str,
    chi: int,
    tdvp_sweeps: int,
    ref: Any,
    ref_type: str,
    ref_chi: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cs = spec.to_circuit_spec()
    floquet_rows: list[dict[str, Any]] = []
    mps: MPS | None = None

    if spec.family == "heisenberg_floquet_2d" and method in METHODS_BY_FAMILY["heisenberg_floquet_2d"]:
        run, mps = _run_floquet_lr_method(
            spec,
            method=method,  # type: ignore[arg-type]
            chi=chi,
            tdvp_sweeps=tdvp_sweeps,
            ref=ref,
            ref_type=ref_type,
            ref_chi=ref_chi,
        )
    elif method in ("tebd_nn", "tdvp_all"):
        run = run_nn_method(
            cs,
            method=method,  # type: ignore[arg-type]
            chi=chi,
            tdvp_sweeps=tdvp_sweeps,
            sweep_scan_enabled=INCLUDE_SWEEP_SCAN,
            ref=ref,
            ref_type=ref_type,
            ref_chi=ref_chi,
            ref_sweeps=None,
        )
    else:
        run = run_method(
            cs,
            method=method,  # type: ignore[arg-type]
            chi=chi,
            tdvp_sweeps=tdvp_sweeps,
            ref=ref,
            ref_type=ref_type,
            ref_chi=ref_chi,
            ref_sweeps=None,
            bond_history=[],
        )

    row = _run_to_row(spec, run, chi=chi)
    if spec.family == "heisenberg_floquet_2d" and run.status == "ok" and mps is not None:
        ref_vec = ref if isinstance(ref, np.ndarray) else None
        floquet_rows = compute_floquet_patch_rows(
            spec, mps, method=method, chi=chi, ref_vec=ref_vec, ref_type=ref_type,
        )
    return row, floquet_rows


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    v = row.get(key)
    if v in (None, ""):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _hit_chi(row: dict[str, Any]) -> bool:
    return row.get("hit_chi_max") in ("True", True, "true", 1)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _classify_lr_outcome(tdvp: dict[str, Any], tebd: dict[str, Any]) -> str:
    tol = 1e-5
    t_err = _f(tdvp, "mean_abs_observable_error", float("inf"))
    e_err = _f(tebd, "mean_abs_observable_error", float("inf"))
    if t_err < tol and e_err < tol:
        return "both_good"
    if t_err > 0.1 and e_err > 0.1:
        return "both_bad"
    if t_err <= e_err / 2:
        return "tdvp_win"
    if e_err <= t_err / 2:
        return "tebd_win"
    return "similar"


def summarize_claims(raw: list[dict[str, Any]], floquet: list[dict[str, Any]], outdir: Path) -> None:
    ok = [r for r in raw if r.get("status") == "ok"]
    lr_methods = {"tebd_swap", "hybrid_tdvp_unpadded", "hybrid_tdvp_padded4"}

    main_rows: list[dict[str, Any]] = []
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ok:
        if r.get("method") not in lr_methods:
            continue
        key = (
            f"{r['case_id']}|chi{r['chi_max']}|sweeps{r.get('tdvp_sweeps', 1)}"
        )
        by_case[key].append(r)
    for _k, rows in by_case.items():
        tebd = next((r for r in rows if r["method"] == "tebd_swap"), None)
        unp = next((r for r in rows if r["method"] == "hybrid_tdvp_unpadded"), None)
        pad = next((r for r in rows if r["method"] == "hybrid_tdvp_padded4"), None)
        if tebd is None:
            continue
        for label, tdvp in (("unpadded", unp), ("padded4", pad)):
            if tdvp is None:
                continue
            te = _f(tebd, "mean_abs_observable_error", 1e-15)
            t_err = _f(tdvp, "mean_abs_observable_error", 1e-15)
            ti = _f(tebd, "infidelity_to_reference", 1e-15)
            tdi = _f(tdvp, "infidelity_to_reference", 1e-15)
            main_rows.append({
                "case_id": tebd["case_id"],
                "family": tebd["family"],
                "n_qubits": tebd["n_qubits"],
                "depth_or_cycles": tebd.get("depth_or_cycles", tebd.get("depth")),
                "seed": tebd["seed"],
                "angle_regime": tebd["angle_regime"],
                "chi_max": tebd["chi_max"],
                "tdvp_variant": label,
                "tebd_error": te,
                "tdvp_error": t_err,
                "tdvp_error_over_tebd_error": t_err / max(te, 1e-15),
                "tdvp_infidelity_over_tebd_infidelity": tdi / max(ti, 1e-15),
                "tdvp_time_over_tebd_time": _f(tdvp, "wall_time_s") / max(_f(tebd, "wall_time_s"), 1e-9),
                "tdvp_peak_chi_over_tebd_peak_chi": _f(tdvp, "peak_bond_dim") / max(_f(tebd, "peak_bond_dim"), 1),
                "outcome": _classify_lr_outcome(tdvp, tebd),
            })
    _write_csv(outdir / "main_method_comparison.csv", main_rows)

    fixed: list[dict[str, Any]] = []
    for row in main_rows:
        tebd_key = f"{row['case_id']}|chi{row['chi_max']}"
        tebd_row = next(
            (r for r in ok if r["method"] == "tebd_swap" and f"{r['case_id']}|chi{r['chi_max']}" == tebd_key),
            None,
        )
        if tebd_row is None:
            continue
        te = _f(tebd_row, "mean_abs_observable_error")
        ti = _f(tebd_row, "infidelity_to_reference")
        limited = te > 1e-5 or ti > 1e-6
        bond_lim = _hit_chi(tebd_row) or _f(tebd_row, "peak_bond_dim") >= 0.9 * _f(tebd_row, "chi_max")
        if not limited:
            continue
        fixed.append({
            **row,
            "tebd_accuracy_limited": limited,
            "tebd_bond_limited": bond_lim,
            "tebd_hit_chi_max": tebd_row.get("hit_chi_max"),
            "tdvp_hit_chi_max": next(
                (
                    r.get("hit_chi_max")
                    for r in ok
                    if r["case_id"] == row["case_id"]
                    and r["chi_max"] == row["chi_max"]
                    and r["method"] == ("hybrid_tdvp_padded4" if row["tdvp_variant"] == "padded4" else "hybrid_tdvp_unpadded")
                ),
                "",
            ),
        })
    _write_csv(outdir / "fixed_chi_advantage.csv", fixed)

    # recovery chi
    recovery: list[dict[str, Any]] = []
    tdvp_targets = [r for r in ok if r["method"] in ("hybrid_tdvp_unpadded", "hybrid_tdvp_padded4")]
    chi_list = sorted({int(float(r["chi_max"])) for r in ok})
    for tdvp in tdvp_targets:
        chi_tdvp = int(float(tdvp["chi_max"]))
        if chi_tdvp not in (16, 24, 32):
            continue
        t_err = _f(tdvp, "mean_abs_observable_error", float("inf"))
        match_key = (
            tdvp["case_id"], tdvp.get("angle_regime"), tdvp.get("seed"),
            tdvp.get("depth_or_cycles", tdvp.get("depth")),
        )
        rec_chi = float("nan")
        rec_time = float("nan")
        status = "not_recovered"
        for chi_t in chi_list:
            tebd = next(
                (
                    r for r in ok
                    if r["method"] == "tebd_swap"
                    and r["case_id"] == tdvp["case_id"]
                    and int(float(r["chi_max"])) == chi_t
                    and r.get("seed") == tdvp.get("seed")
                    and r.get("angle_regime") == tdvp.get("angle_regime")
                ),
                None,
            )
            if tebd is None:
                continue
            if _f(tebd, "mean_abs_observable_error", float("inf")) <= t_err:
                rec_chi = float(chi_t)
                rec_time = _f(tebd, "wall_time_s")
                status = "recovered"
                break
        recovery.append({
            "case_id": tdvp["case_id"],
            "family": tdvp["family"],
            "method": tdvp["method"],
            "tdvp_chi": chi_tdvp,
            "tdvp_error": t_err,
            "tebd_recovery_chi": rec_chi,
            "recovery_chi_ratio": rec_chi / chi_tdvp if not math.isnan(rec_chi) else "",
            "recovery_status": status,
            "tdvp_time": _f(tdvp, "wall_time_s"),
            "tebd_recovery_time": rec_time,
            "time_ratio": rec_time / max(_f(tdvp, "wall_time_s"), 1e-9) if not math.isnan(rec_time) else "",
        })
    _write_csv(outdir / "recovery_chi.csv", recovery)

    # padding diagnostic
    pad_rows: list[dict[str, Any]] = []
    for fam in ("periodic_1d", "sparse_long_range"):
        groups: dict[str, tuple[dict[str, Any] | None, dict[str, Any] | None]] = {}
        for r in ok:
            if r.get("family") != fam:
                continue
            gk = f"{r['case_id']}|chi{r['chi_max']}|sweeps{r.get('tdvp_sweeps', 1)}"
            unp, pad = groups.get(gk, (None, None))
            if r["method"] == "hybrid_tdvp_unpadded":
                unp = r
            elif r["method"] == "hybrid_tdvp_padded4":
                pad = r
            groups[gk] = (unp, pad)
        for gk, (unp, pad) in groups.items():
            if unp is None or pad is None:
                continue
            pe = _f(unp, "mean_abs_observable_error", 1e-15)
            ce = _f(pad, "mean_abs_observable_error", 1e-15)
            ratio = ce / pe
            if ratio <= 0.1:
                cls = "padding_required"
            elif ratio <= 0.5:
                cls = "padding_helpful"
            elif ratio >= 1.3:
                cls = "padding_harmful"
            elif abs(ratio - 1.0) < 0.3:
                cls = "padding_neutral"
            else:
                cls = "padding_neutral"
            pad_rows.append({
                "case_id": unp["case_id"],
                "family": fam,
                "chi_max": unp["chi_max"],
                "unpadded_error": pe,
                "padded4_error": ce,
                "error_ratio": ratio,
                "classification": cls,
            })
    _write_csv(outdir / "padding_diagnostic.csv", pad_rows)

    # NN control
    nn_rows: list[dict[str, Any]] = []
    nn_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ok:
        if r.get("family") != "nn_brickwork":
            continue
        nn_groups[f"{r['case_id']}|chi{r['chi_max']}"].append(r)
    for _k, rows in nn_groups.items():
        tebd = next((r for r in rows if r["method"] == "tebd_nn"), None)
        tdvp = next((r for r in rows if r["method"] == "tdvp_all"), None)
        if tebd is None or tdvp is None:
            continue
        te = _f(tebd, "mean_abs_observable_error", 1e-15)
        t_err = _f(tdvp, "mean_abs_observable_error", 1e-15)
        nn_rows.append({
            "case_id": tebd["case_id"],
            "chi_max": tebd["chi_max"],
            "tebd_error": te,
            "tdvp_error": t_err,
            "tdvp_error_over_tebd_error": t_err / max(te, 1e-15),
            "tdvp_time_over_tebd_time": _f(tdvp, "wall_time_s") / max(_f(tebd, "wall_time_s"), 1e-9),
            "outcome": _classify_lr_outcome(tdvp, tebd),
        })
    _write_csv(outdir / "nn_control.csv", nn_rows)
    _write_csv(outdir / "floquet_collision_entropy.csv", floquet)

    _make_plots(raw, pad_rows, recovery, nn_rows, floquet, main_rows, outdir / "plots")
    _write_report(raw, main_rows, fixed, pad_rows, nn_rows, recovery, floquet, outdir / "report.md")


def _make_plots(
    raw: list[dict[str, Any]],
    pad_rows: list[dict[str, Any]],
    recovery: list[dict[str, Any]],
    nn_rows: list[dict[str, Any]],
    floquet: list[dict[str, Any]],
    main_rows: list[dict[str, Any]],
    plotdir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-untyped]
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return
    plotdir.mkdir(parents=True, exist_ok=True)
    ok = [r for r in raw if r.get("status") == "ok"]

    # Plot 1: periodic padding
    per = [r for r in ok if r.get("family") == "periodic_1d"]
    if per:
        fig, ax = plt.subplots(figsize=(7, 4))
        for method, label in (
            ("tebd_swap", "TEBD+SWAP"),
            ("hybrid_tdvp_unpadded", "TDVP unpadded"),
            ("hybrid_tdvp_padded4", "TDVP padded4"),
        ):
            pts = sorted(
                [(int(float(r["chi_max"])), _f(r, "mean_abs_observable_error")) for r in per if r["method"] == method],
                key=lambda x: x[0],
            )
            if pts:
                xs, ys = zip(*pts, strict=True)
                ax.plot(xs, ys, marker="o", label=label)
        ax.set_xlabel("chi_max")
        ax.set_ylabel("mean observable error")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title("periodic_1d padding diagnostic")
        fig.tight_layout()
        fig.savefig(plotdir / "padding_periodic_diagnostic.png", dpi=120)
        plt.close(fig)

    # Plot 2: sparse fixed chi
    sparse = [r for r in ok if r.get("family") == "sparse_long_range"]
    if sparse:
        fig, ax = plt.subplots(figsize=(7, 4))
        for method, label in (
            ("tebd_swap", "TEBD+SWAP"),
            ("hybrid_tdvp_unpadded", "TDVP unpadded"),
            ("hybrid_tdvp_padded4", "TDVP padded4"),
        ):
            pts = sorted(
                [(int(float(r["chi_max"])), _f(r, "mean_abs_observable_error")) for r in sparse if r["method"] == method],
                key=lambda x: x[0],
            )
            if pts:
                xs, ys = zip(*pts, strict=True)
                ax.plot(xs, ys, marker="o", label=label)
        ax.set_xlabel("chi_max")
        ax.set_ylabel("mean observable error")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title("sparse_long_range fixed-chi")
        fig.tight_layout()
        fig.savefig(plotdir / "sparse_fixed_chi_error_vs_chi.png", dpi=120)
        plt.close(fig)

    if recovery:
        fig, ax = plt.subplots(figsize=(6, 4))
        xs = [_f(r, "tdvp_chi") for r in recovery if r.get("recovery_status") == "recovered"]
        ys = [_f(r, "recovery_chi_ratio") for r in recovery if r.get("recovery_status") == "recovered"]
        if xs:
            ax.scatter(xs, ys, alpha=0.7)
            ax.set_xlabel("TDVP chi")
            ax.set_ylabel("TEBD recovery chi / TDVP chi")
            ax.set_title("Recovery chi")
            fig.tight_layout()
            fig.savefig(plotdir / "recovery_chi.png", dpi=120)
            plt.close(fig)

    if nn_rows:
        fig, ax = plt.subplots(figsize=(6, 4))
        ratios = [_f(r, "tdvp_error_over_tebd_error") for r in nn_rows]
        ax.hist(ratios, bins=min(20, max(5, len(ratios))))
        ax.axvline(1.0, color="k", ls="--")
        ax.set_xlabel("TDVP error / TEBD error")
        ax.set_title("NN control")
        fig.tight_layout()
        fig.savefig(plotdir / "nn_control_tebd_vs_tdvp.png", dpi=120)
        plt.close(fig)

    if floquet:
        fig, ax = plt.subplots(figsize=(7, 4))
        for method in ("tebd_swap", "hybrid_tdvp_padded4"):
            pts = sorted(
                [
                    (float(r["J_over_pi"]), _f(r, "s2_z_error") if r.get("s2_z_error") not in (None, "") else _f(r, "s2_z"))
                    for r in floquet
                    if r["method"] == method and r.get("patch_shape") == "1x1"
                ],
                key=lambda x: x[0],
            )
            if pts:
                xs, ys = zip(*pts, strict=True)
                ax.plot(xs, ys, marker="o", label=method)
        ax.set_xlabel("J/pi")
        ax.set_ylabel("S2_Z error (1x1 patch)")
        ax.legend()
        ax.set_title("Floquet collision entropy")
        fig.tight_layout()
        fig.savefig(plotdir / "floquet_collision_entropy_vs_J.png", dpi=120)
        plt.close(fig)

    if main_rows:
        outcomes = {"tdvp_win": 0, "tebd_win": 0, "similar": 0, "both_good": 0, "both_bad": 0}
        for row in main_rows:
            outcome = str(row.get("outcome", "similar"))
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        fig, ax = plt.subplots(figsize=(6, 3))
        labels = list(outcomes.keys())
        counts = [outcomes[k] for k in labels]
        ax.barh(labels, counts)
        ax.set_xlabel("count")
        ax.set_title("LR method outcome regime map")
        fig.tight_layout()
        fig.savefig(plotdir / "regime_map.png", dpi=120)
        plt.close(fig)


def _write_report(
    raw: list[dict[str, Any]],
    main_rows: list[dict[str, Any]],
    fixed: list[dict[str, Any]],
    pad_rows: list[dict[str, Any]],
    nn_rows: list[dict[str, Any]],
    recovery: list[dict[str, Any]],
    floquet: list[dict[str, Any]],
    path: Path,
) -> None:
    lines = [
        "# TDVP claims benchmark\n\n",
        "## Executive summary\n\n",
        "The main supported claim is fixed-χ observable accuracy improvement in "
        "sparse long-range/routing-limited regimes.\n\n",
        "The benchmark does not attempt to show universal TDVP superiority.\n\n",
        "TEBD remains the preferred nearest-neighbor update.\n\n",
        "Runtime advantage is not assumed; wall-clock is reported honestly.\n\n",
        "The Floquet application is exploratory unless it clearly shows improved "
        "fixed-χ collision-entropy accuracy.\n\n",
    ]
    if fixed:
        wins = sum(1 for r in fixed if _f(r, "tdvp_error_over_tebd_error", 1) < 0.5)
        lines.append(
            f"- Fixed-χ advantage subset: {len(fixed)} cases; TDVP wins (2× error) in {wins}.\n"
        )
    if pad_rows:
        req = sum(1 for r in pad_rows if r["classification"] == "padding_required")
        help_ = sum(1 for r in pad_rows if r["classification"] == "padding_helpful")
        lines.append(f"- Padding: required={req}, helpful={help_} over {len(pad_rows)} pairs.\n")
    if nn_rows:
        med = float(np.median([_f(r, "tdvp_error_over_tebd_error") for r in nn_rows]))
        lines.append(f"- NN control median TDVP/TEBD error ratio: {med:.2f}\n")

    lines.append("\n## Validation and dispatch checks\n\n")
    lines.append("- Pre-flight validation runs before the benchmark loop.\n")

    lines.append("\n## Padding diagnostic\n\n")
    per_pad = [r for r in pad_rows if "periodic" in str(r.get("case_id", ""))]
    if per_pad:
        lines.append(f"- periodic_1d padding pairs: {len(per_pad)}\n")
    lines.append("- Does padding fix periodic gates? See `padding_diagnostic.csv`.\n")
    lines.append("- Is padded4 always better? Expected: no.\n")

    lines.append("\n## Sparse long-range fixed-χ advantage\n\n")
    sparse_main = [r for r in main_rows if r.get("family") == "sparse_long_range"]
    lines.append(f"- sparse comparisons: {len(sparse_main)}\n")

    lines.append("\n## Recovery χ\n\n")
    rec_ok = [r for r in recovery if r.get("recovery_status") == "recovered"]
    lines.append(f"- Recovered within tested χ: {len(rec_ok)}/{len(recovery)}\n")

    lines.append("\n## Nearest-neighbor control\n\n")
    if nn_rows:
        tebd_better = sum(1 for r in nn_rows if _f(r, "tdvp_error_over_tebd_error") >= 1.0)
        lines.append(f"- Cases with TDVP ≥ TEBD error: {tebd_better}/{len(nn_rows)}\n")

    lines.append("\n## Heisenberg Floquet mini-application\n\n")
    lines.append(f"- Floquet patch rows: {len(floquet)} (exploratory).\n")

    lines.append("\n## Failure/scope cases\n\n")
    dense = [r for r in raw if r.get("family") == "dense_long_range_failure"]
    lines.append(f"- dense_long_range_failure rows: {len(dense)}\n")

    lines.append("\n## Recommended paper figures\n\n")
    lines.append("- `plots/sparse_fixed_chi_error_vs_chi.png`\n")
    lines.append("- `plots/recovery_chi.png`\n")
    lines.append("- `plots/padding_periodic_diagnostic.png`\n")
    lines.append("- `plots/nn_control_tebd_vs_tdvp.png`\n")
    lines.append(f"\n---\n\nTotal runs: {len(raw)}.\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    if STAGE not in STAGE_CONFIGS:
        raise SystemExit(f"Unknown YAQS_TDVP_CLAIMS_STAGE={STAGE}")
    cfg = STAGE_CONFIGS[STAGE]
    outdir = OUTDIR if OUTDIR.is_absolute() else Path(__file__).resolve().parents[1] / OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    validate_reference_convention(chi=max(cfg["chi_values"]))
    validate_claims_benchmark(chi=max(cfg["chi_values"]))

    specs = build_claims_specs(cfg)
    raw_path = outdir / "raw.csv"
    if OVERWRITE and raw_path.exists():
        raw_path.unlink()
    existing: set[str] = set()
    if raw_path.exists() and not OVERWRITE:
        with raw_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing.add(
                    f"{row.get('case_id')}|{row.get('method')}|chi{row.get('chi_max')}"
                    f"|sweeps{row.get('tdvp_sweeps', '1')}"
                    f"|J{row.get('J_over_pi', '')}"
                )

    sweeps_list = _resolve_sweeps()
    chi_values = list(cfg["chi_values"])
    if INCLUDE_FLOQUET and "floquet" in cfg:
        chi_values = sorted(set(chi_values) | set(cfg["floquet"]["chi_values"]))

    floquet_all: list[dict[str, Any]] = []
    floquet_path = outdir / "floquet_collision_entropy.csv"
    if OVERWRITE and floquet_path.exists():
        floquet_path.unlink()

    ref_cache: dict[str, tuple[Any, str, int | None, int | None]] = {}
    total = sum(
        len(METHODS_BY_FAMILY.get(s.family, []))
        * (
            len(cfg["floquet"]["chi_values"])
            if s.family == "heisenberg_floquet_2d" and "floquet" in cfg
            else len(chi_values)
        )
        * (
            1
            if s.family in ("nn_brickwork", "heisenberg_floquet_2d") or not INCLUDE_SWEEP_SCAN
            else len(sweeps_list)
        )
        for s in specs
    )
    print(f"Planned runs (approx): {total}")

    case_num = 0
    print(f"Claims stage {STAGE}: {len(specs)} specs, out={outdir}")

    for spec in specs:
        methods = METHODS_BY_FAMILY.get(spec.family, [])
        chis = chi_values
        if spec.family == "heisenberg_floquet_2d" and "floquet" in cfg:
            chis = list(cfg["floquet"]["chi_values"])
        if spec.case_id not in ref_cache:
            if spec.family == "heisenberg_floquet_2d":
                rv, rt, rc = _floquet_reference_vec(spec)
                ref_cache[spec.case_id] = (rv, rt, rc, None)
            else:
                ref_cache[spec.case_id] = compute_reference(
                    spec.to_circuit_spec(), exact_n_max=int(cfg.get("exact_n_max", EXACT_N_MAX))
                )
        ref, ref_type, ref_chi, ref_sw = ref_cache[spec.case_id]

        for chi in chis:
            sw_list = [1] if spec.family in ("nn_brickwork", "heisenberg_floquet_2d") else sweeps_list
            if spec.family not in ("nn_brickwork",) and not INCLUDE_SWEEP_SCAN:
                sw_list = [TDVP_SWEEPS_DEFAULT]
            for method in methods:
                for sweeps in sw_list if method != "tebd_nn" else [1]:
                    case_num += 1
                    key = _run_key(spec, method, chi, sweeps)
                    if key in existing:
                        print(f"[{case_num}] SKIP {key}")
                        continue
                    try:
                        row, frows = execute_claims_run(
                            spec, method, chi, sweeps, ref, ref_type, ref_chi,
                        )
                    except Exception as exc:
                        row = {
                            "case_id": spec.case_id,
                            "family": spec.family,
                            "method": method,
                            "chi_max": chi,
                            "status": "failed",
                            "error_message": f"{type(exc).__name__}: {exc}",
                        }
                        frows = []
                    write_header = not raw_path.exists() or raw_path.stat().st_size == 0
                    with raw_path.open("a", newline="", encoding="utf-8") as f:
                        if row:
                            keys = list(row.keys())
                            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                            if write_header:
                                w.writeheader()
                            w.writerow(row)
                    existing.add(key)
                    if frows:
                        floquet_all.extend(frows)
                        _write_csv(floquet_path, floquet_all)
                    print(
                        f"[{case_num}] {spec.case_id} method={method} chi={chi} "
                        f"sweeps={sweeps} status={row.get('status', '?')}"
                    )

    raw: list[dict[str, Any]] = []
    if raw_path.exists():
        with raw_path.open(newline="", encoding="utf-8") as f:
            raw = list(csv.DictReader(f))
    if floquet_path.exists():
        with floquet_path.open(newline="", encoding="utf-8") as f:
            floquet_all = list(csv.DictReader(f))
    summarize_claims(raw, floquet_all, outdir)
    print(f"\nDone. {len(raw)} rows in {raw_path}")
    print(f"Report: {outdir / 'report.md'}")


if __name__ == "__main__":
    main()
