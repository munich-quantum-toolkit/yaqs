#!/usr/bin/env python3
"""Matched L=16 BUG/2TDVP benchmarks for the manuscript parameter grid.

The timed region contains only repeated calls to the evolution kernel.  State
construction, deterministic noisy padding, MPO construction, exact reference
evolution, warm-up, vector conversion, work counters, and rank diagnostics are
all outside the timed region.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.core.methods import matrix_exponential
from mqt.yaqs.core.methods.bug import bug as bug_evolve
from mqt.yaqs.core.methods.tdvp import primitives
from mqt.yaqs.core.methods.tdvp import tdvp as tdvp_evolve


LENGTH = 16
COUPLING = 1.0
FIELD = 1.05
TOTAL_TIME = 1.0
DT_GRID = (0.01, 0.005, 0.0025, 0.00125)
THRESHOLD = 1e-12
MAX_BOND = 512
KRYLOV_TOL = 1e-12
INITIAL_CHI = 4
NOISE_SCALE = 1e-10
SEED = 20260812


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="tfim,hs", help="Comma-separated subset of tfim,hs")
    parser.add_argument("--dts", default=",".join(map(str, DT_GRID)), help="Comma-separated time steps")
    parser.add_argument("--total-time", type=float, default=TOTAL_TIME)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("raw_results.json"))
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--skip-diagnostics", action="store_true")
    return parser.parse_args()


def direct_ising_mpo() -> MPO:
    """Construct the same uncompressed bond-3 TFIM MPO as YAQS-Julia."""
    identity = np.eye(2, dtype=np.complex128)
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    left = np.array([identity, -COUPLING * z, -FIELD * x])[None, :]
    bulk = np.zeros((3, 3, 2, 2), dtype=np.complex128)
    bulk[0, 0], bulk[0, 1], bulk[0, 2] = identity, -COUPLING * z, -FIELD * x
    bulk[1, 2], bulk[2, 2] = z, identity
    right = np.array([[-FIELD * x], [z], [identity]])
    tensors = [left, *([bulk.copy() for _ in range(LENGTH - 2)]), right]
    mpo = MPO()
    mpo.custom([np.transpose(tensor, (2, 3, 0, 1)).copy() for tensor in tensors], transpose=False)
    return mpo


def direct_haldane_shastry_mpo() -> MPO:
    """Construct Julia's exact uncompressed finite-state-machine HS MPO."""
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128) / 2
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128) / 2
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128) / 2
    identity = np.eye(2, dtype=np.complex128)
    spin_ops = (sx, sy, sz)
    tensors: list[np.ndarray] = []
    for site in range(1, LENGTH + 1):
        left_dim = 1 if site == 1 else 3 * (site - 1) + 2
        right_dim = 1 if site == LENGTH else 3 * site + 2
        tensor = np.zeros((left_dim, 2, 2, right_dim), dtype=np.complex128)
        if site == 1:
            tensor[0, :, :, 0] = identity
            for axis, operator in enumerate(spin_ops, start=1):
                tensor[0, :, :, axis] = operator
        elif site == LENGTH:
            for start in range(1, LENGTH):
                coupling = COUPLING * (np.pi / LENGTH) ** 2 / np.sin(np.pi * (site - start) / LENGTH) ** 2
                for axis, operator in enumerate(spin_ops, start=1):
                    left = (start - 1) * 3 + axis
                    tensor[left, :, :, 0] += coupling * operator
            tensor[left_dim - 1, :, :, 0] = identity
        else:
            tensor[0, :, :, 0] = identity
            for axis, operator in enumerate(spin_ops, start=1):
                right = (site - 1) * 3 + axis
                tensor[0, :, :, right] = operator
            for start in range(1, site):
                coupling = COUPLING * (np.pi / LENGTH) ** 2 / np.sin(np.pi * (site - start) / LENGTH) ** 2
                for axis, operator in enumerate(spin_ops, start=1):
                    virtual = (start - 1) * 3 + axis
                    tensor[virtual, :, :, virtual] = identity
                    tensor[virtual, :, :, right_dim - 1] += coupling * operator
            tensor[left_dim - 1, :, :, right_dim - 1] = identity
        tensors.append(np.transpose(tensor, (1, 2, 0, 3)).copy())
    mpo = MPO()
    mpo.custom(tensors, transpose=False)
    return mpo


def padded_initial_state(model: str) -> MPS:
    """Create, perturb once, and right-canonicalize one shared chi=4 MPS."""
    rng = np.random.default_rng(SEED + (0 if model == "tfim" else 1))
    if model == "tfim":
        local_vectors = [np.array([1, 1], dtype=np.complex128) / np.sqrt(2) for _ in range(LENGTH)]
    else:
        local_vectors = [
            np.array([1, 0], dtype=np.complex128) if site % 2 == 0 else np.array([0, 1], dtype=np.complex128)
            for site in range(LENGTH)
        ]
    # Work in Julia's (left, physical, right) order while reproducing the
    # existing shared-fixture padding protocol.
    tensors = [vector.reshape(1, 2, 1).copy() for vector in local_vectors]
    for bond in range(LENGTH - 1):
        left = tensors[bond]
        right = tensors[bond + 1]
        old = left.shape[2]
        padded_left = np.zeros((left.shape[0], left.shape[1], INITIAL_CHI), dtype=np.complex128)
        padded_left[:, :, :old] = left
        shape = padded_left[:, :, old:].shape
        padded_left[:, :, old:] = NOISE_SCALE * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
        padded_right = np.zeros((INITIAL_CHI, right.shape[1], right.shape[2]), dtype=np.complex128)
        padded_right[:old, :, :] = right
        shape = padded_right[old:, :, :].shape
        padded_right[old:, :, :] = NOISE_SCALE * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
        tensors[bond] = padded_left
        tensors[bond + 1] = padded_right
    for site in range(LENGTH - 1, 0, -1):
        tensor = tensors[site]
        left, physical, right = tensor.shape
        qh, rh = np.linalg.qr(tensor.reshape(left, physical * right).conj().T, mode="reduced")
        transfer = rh.conj().T
        tensors[site] = qh.conj().T.reshape(qh.shape[1], physical, right)
        previous = tensors[site - 1]
        tensors[site - 1] = (previous.reshape(-1, left) @ transfer).reshape(
            previous.shape[0], previous.shape[1], transfer.shape[1]
        )
    tensors[0] /= np.linalg.norm(tensors[0])
    state = MPS(LENGTH, tensors=[np.transpose(tensor, (1, 0, 2)).copy() for tensor in tensors])
    state.set_center(0)
    return state


def bond_profile(state: MPS) -> list[int]:
    return [int(tensor.shape[2]) for tensor in state.tensors[:-1]]


def exact_sparse_hamiltonian(model: str) -> sparse.csr_matrix:
    """Build the exact Hamiltonian in YAQS's site-0-LSB vector convention."""
    dim = 1 << LENGTH
    basis = np.arange(dim, dtype=np.int64)
    if model == "tfim":
        diagonal = np.zeros(dim, dtype=np.float64)
        for site in range(LENGTH - 1):
            zi = 1 - 2 * ((basis >> site) & 1)
            zj = 1 - 2 * ((basis >> (site + 1)) & 1)
            diagonal -= COUPLING * zi * zj
        rows = [basis]
        cols = [basis]
        data = [diagonal]
        for site in range(LENGTH):
            rows.append(basis ^ (1 << site))
            cols.append(basis)
            data.append(np.full(dim, -FIELD, dtype=np.float64))
    else:
        diagonal = np.zeros(dim, dtype=np.float64)
        rows = [basis]
        cols = [basis]
        data = [diagonal]
        for left in range(LENGTH):
            for right in range(left + 1, LENGTH):
                coupling = COUPLING * (np.pi / LENGTH) ** 2 / np.sin(np.pi * (right - left) / LENGTH) ** 2
                left_bits = (basis >> left) & 1
                right_bits = (basis >> right) & 1
                diagonal += coupling * (1 - 2 * left_bits) * (1 - 2 * right_bits) / 4
                opposite = left_bits != right_bits
                source = basis[opposite]
                rows.append(source ^ (1 << left) ^ (1 << right))
                cols.append(source)
                data.append(np.full(source.size, coupling / 2, dtype=np.float64))
    row = np.concatenate(rows)
    col = np.concatenate(cols)
    values = np.concatenate(data)
    return sparse.coo_matrix((values, (row, col)), shape=(dim, dim)).tocsr()


def phase_error(reference: np.ndarray, approximate: np.ndarray) -> float:
    overlap = np.vdot(reference, approximate)
    phase = 1.0 + 0.0j if overlap == 0 else overlap / abs(overlap)
    return float(np.linalg.norm(approximate - phase * reference))


def infidelity(reference: np.ndarray, approximate: np.ndarray) -> float:
    denominator = float(np.vdot(reference, reference).real * np.vdot(approximate, approximate).real)
    fidelity = float(abs(np.vdot(reference, approximate)) ** 2 / denominator)
    return max(0.0, 1.0 - fidelity)


def z_expectations(vector: np.ndarray) -> np.ndarray:
    probability = np.abs(vector) ** 2
    probability /= probability.sum()
    basis = np.arange(vector.size, dtype=np.int64)
    return np.asarray([np.sum(probability * (1 - 2 * ((basis >> site) & 1))) for site in range(LENGTH)])


def parameters(dt: float) -> AnalogSimParams:
    return AnalogSimParams(
        elapsed_time=dt,
        dt=dt,
        max_bond_dim=MAX_BOND,
        trunc_mode="relative_discarded_weight",
        svd_threshold=THRESHOLD,
        krylov_tol=KRYLOV_TOL,
        tdvp_mode="2site",
        get_state=True,
    )


def run_to_final(
    method: str,
    initial: MPS,
    mpo: MPO,
    params: AnalogSimParams,
    steps: int,
    *,
    diagnostics: bool,
) -> tuple[MPS, dict[str, int], int, dict[str, list[int]]]:
    evolve = bug_evolve if method == "bug" else tdvp_evolve
    state = deepcopy(initial)
    local_mpo = deepcopy(mpo)
    checkpoints: dict[str, list[int]] = {}

    def checkpoint(stage: str, checkpoint_state: MPS, *, reflected: bool) -> None:
        snapshot = deepcopy(checkpoint_state)
        if reflected:
            snapshot.flip_network()
        checkpoints[stage] = bond_profile(snapshot)

    matrix_exponential.reset_krylov_stats()
    matrix_exponential.enable_krylov_stats(enabled=diagnostics)
    max_chi = max(bond_profile(state))
    try:
        for step in range(steps):
            if method == "bug" and step == 0:
                evolve(state, local_mpo, params, checkpoint=checkpoint)
            else:
                evolve(state, local_mpo, params)
            max_chi = max(max_chi, max(bond_profile(state)))
    finally:
        matrix_exponential.enable_krylov_stats(enabled=False)
    return state, matrix_exponential.get_krylov_stats(), max_chi, checkpoints


def time_method(method: str, initial: MPS, mpo: MPO, params: AnalogSimParams, steps: int) -> float:
    evolve = bug_evolve if method == "bug" else tdvp_evolve
    state = deepcopy(initial)
    local_mpo = deepcopy(mpo)
    gc.collect()
    start = time.perf_counter()
    for _ in range(steps):
        evolve(state, local_mpo, params)
    return time.perf_counter() - start


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    dts = [float(item) for item in args.dts.split(",") if item.strip()]
    if not set(models) <= {"tfim", "hs"}:
        raise ValueError(f"Unsupported models: {models}")
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")

    # One shared pure-NumPy, matrix-free adaptive Lanczos implementation for
    # every BUG and 2TDVP local exponential.
    primitives.DENSE_THRESHOLD = -1
    matrix_exponential.NUMBA_THRESHOLD = sys.maxsize

    payload: dict[str, Any] = {
        "protocol": {
            "length": LENGTH,
            "total_time": args.total_time,
            "dt_grid": dts,
            "threshold": THRESHOLD,
            "trunc_mode": "relative_discarded_weight",
            "min_keep": 2,
            "max_bond_dim": MAX_BOND,
            "initial_chi": INITIAL_CHI,
            "noise_scale": NOISE_SCALE,
            "seed": SEED,
            "krylov_max_dim": 25,
            "krylov_tol": KRYLOV_TOL,
            "krylov_backend": "shared pure-NumPy matrix-free adaptive Lanczos",
            "timing_repetitions": args.repetitions,
            "timing_excludes": [
                "state and MPO construction",
                "initial padding",
                "warm-up",
                "garbage collection",
                "exact reference",
                "rank diagnostics",
                "Krylov work counters",
                "state-vector conversion",
            ],
        },
        "models": {},
    }
    write_json(args.output, payload)

    for model in models:
        print(f"[{model}] constructing shared state, MPO, and reference", flush=True)
        initial = padded_initial_state(model)
        mpo = direct_ising_mpo() if model == "tfim" else direct_haldane_shastry_mpo()
        initial_vector = initial.to_vec().copy()
        hamiltonian = None if args.skip_reference else exact_sparse_hamiltonian(model)
        reference = None if hamiltonian is None else expm_multiply((-1j * args.total_time) * hamiltonian, initial_vector)
        reference_z = None if reference is None else z_expectations(reference)
        reference_energy = (
            None
            if reference is None or hamiltonian is None
            else float(np.vdot(reference, hamiltonian @ reference).real / np.vdot(reference, reference).real)
        )
        model_result: dict[str, Any] = {
            "initial_bond_profile": bond_profile(initial),
            "initial_norm": float(np.vdot(initial_vector, initial_vector).real),
            "mpo_bond_profile": [int(tensor.shape[3]) for tensor in mpo.tensors[:-1]],
            "reference_norm": None if reference is None else float(np.vdot(reference, reference).real),
            "runs": {},
        }
        payload["models"][model] = model_result
        write_json(args.output, payload)

        for dt in dts:
            steps = round(args.total_time / dt)
            if not math.isclose(steps * dt, args.total_time, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"total time {args.total_time} is not divisible by dt={dt}")
            params = parameters(dt)
            print(f"[{model} dt={dt}] warming both methods", flush=True)
            # A one-step warm-up is enough because the selected backend has no
            # JIT compilation; it also primes imports and contraction caches.
            bug_evolve(deepcopy(initial), deepcopy(mpo), params)
            tdvp_evolve(deepcopy(initial), deepcopy(mpo), params)

            run_result: dict[str, Any] = {"steps": steps, "methods": {}}
            diagnostic_states: dict[str, MPS] = {}
            if not args.skip_diagnostics:
                for method in ("bug", "2tdvp"):
                    print(f"[{model} dt={dt}] diagnostic replay: {method}", flush=True)
                    state, stats, max_chi, checkpoints = run_to_final(
                        method, initial, mpo, params, steps, diagnostics=True
                    )
                    diagnostic_states[method] = state
                    vector = state.to_vec().copy()
                    z_values = z_expectations(vector)
                    energy = (
                        None
                        if hamiltonian is None
                        else float(np.vdot(vector, hamiltonian @ vector).real / np.vdot(vector, vector).real)
                    )
                    run_result["methods"][method] = {
                        "krylov_calls": stats["calls"],
                        "krylov_operator_applications": stats["operator_applications"],
                        "max_chi": max_chi,
                        "final_bond_profile": bond_profile(state),
                        "first_step_bug_checkpoints": checkpoints,
                        "norm": float(np.vdot(vector, vector).real),
                        "phase_aligned_state_error": None if reference is None else phase_error(reference, vector),
                        "infidelity": None if reference is None else infidelity(reference, vector),
                        "max_abs_z_error": None if reference_z is None else float(np.max(np.abs(z_values - reference_z))),
                        "rms_z_error": None if reference_z is None else float(np.sqrt(np.mean((z_values - reference_z) ** 2))),
                        "energy_abs_error": None
                        if energy is None or reference_energy is None
                        else abs(energy - reference_energy),
                    }

            timings: dict[str, list[float]] = {"bug": [], "2tdvp": []}
            for repetition in range(args.repetitions):
                # Alternate order to reduce systematic thermal/order bias.
                order = ("bug", "2tdvp") if repetition % 2 == 0 else ("2tdvp", "bug")
                for method in order:
                    print(
                        f"[{model} dt={dt}] timing {repetition + 1}/{args.repetitions}: {method}",
                        flush=True,
                    )
                    timings[method].append(time_method(method, initial, mpo, params, steps))
            for method in ("bug", "2tdvp"):
                method_result = run_result["methods"].setdefault(method, {})
                method_result["runtime_samples_seconds"] = timings[method]
                method_result["runtime_median_seconds"] = statistics.median(timings[method])
                method_result["runtime_min_seconds"] = min(timings[method])
                method_result["runtime_max_seconds"] = max(timings[method])
            bug_median = run_result["methods"]["bug"]["runtime_median_seconds"]
            tdvp_median = run_result["methods"]["2tdvp"]["runtime_median_seconds"]
            run_result["tdvp_over_bug_runtime"] = tdvp_median / bug_median
            run_result["bug_speedup_percent"] = 100 * (tdvp_median - bug_median) / tdvp_median
            model_result["runs"][format(dt, ".8g")] = run_result
            write_json(args.output, payload)
            print(
                f"[{model} dt={dt}] median BUG={bug_median:.3f}s, "
                f"2TDVP={tdvp_median:.3f}s, ratio={tdvp_median / bug_median:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
