#!/usr/bin/env python3
"""Exact V-matrix non-Markovianity benchmark (causal cut with split instrument).

At cut slot ``c`` (1-based), the instrument is **split**: the past probe supplies only the measurement
ket at ``c``, the future probe supplies only the preparation ket at ``c``, and the simulation applies
the corresponding project–normalize–reprepare step with ``φ`` from past ``i`` and ``ψ`` from future ``j``.

Choice of exact backend path:
- This script uses the existing exact surrogate workflow rollout backend
  (`_simulate_sequences`) in final-state mode (`record_step_states=False`).
- Scheduling follows the process-tensor / **comb** convention: for ``k`` instrument slots there are
  ``k+1`` free-evolution segments ``U_1`` … ``U_{k+1}`` (evolve, then measure--reprepare, then evolve, …).
  ``timesteps`` passed to the backend has length ``k+1`` (default: ``dt`` repeated).
- That gives exact backend density-matrix outputs for sampled instrument sequences,
  without any Transformer model.

Initial states (``--n-seeds``):
- Site 0 is treated as the *system*; sites ``1..L-1`` are the *environment*, fixed to
  :math:`|0\\rangle` (tensor product :math:`|\\psi_{\\mathrm{sys}}\\rangle \\otimes |0\\rangle^{\\otimes (L-1)}`).
- ``n_seeds=1`` (default): full chain starts in :math:`|0\\ldots 0\\rangle` (previous behavior).
- ``n_seeds>1``: fix a list of :math:`n_{\\mathrm{seeds}}` such initial states **once per cut** (same list for
  every :math:`(J,g)` in that cut). For each initial state, build :math:`V` and compute metrics separately;
  **scalar** summary metrics are then averaged across initial states. Saved :math:`V` / spectra / row-distance
  arrays use **seed 0** for file compatibility with existing plots.

Optional ``--probe-convergence-study``:
- Varies probe budget :math:`m = N_p = N_f`, repeats over independent probe draws, compares to a large
 reference :math:`m_{\\mathrm{ref}}`. Uses :math:`n_{\\mathrm{seeds}}=1` (fixed :math:`|0\\ldots 0\\rangle`)
  and the four default physical cases only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from mqt.yaqs.characterization.process_tensors.core.encoding import unpack_rho8
from mqt.yaqs.characterization.process_tensors.surrogates.utils import _random_pure_state, _sample_random_intervention_parts
from mqt.yaqs.characterization.process_tensors.surrogates.workflow import _psi_from_rank1_projector, _simulate_sequences
from mqt.yaqs.characterization.process_tensors.core.utils import make_mcwf_static_context
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument(
        "--cut",
        type=int,
        default=10,
        help="Causal cut slot c (1..k): instrument c is split (past: measurement only; future: preparation only).",
    )
    p.add_argument("--n-pasts", type=int, default=32)
    p.add_argument("--n-futures", type=int, default=32)
    p.add_argument("--g-fixed", type=float, default=1.0)
    p.add_argument("--j-sweep", type=str, default="0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0")
    p.add_argument(
        "--cut-sweep",
        type=str,
        default=None,
        help='Optional cut sweep, e.g. "4,6,8,10,12,14,16". Defaults to single --cut behavior.',
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        dest="n_seeds",
        help=(
            "Initial states: env (sites 1..L-1) fixed to |0>, system (site 0) varies. "
            "n_seeds=1: single |0...0> (legacy). n_seeds>1: same fixed list per cut; "
            "one V per initial state; summary scalars are averaged across initial states."
        ),
    )
    p.add_argument("--out-dir", type=Path, default=Path("benchmark_v_matrix_exact_results"))
    p.add_argument("--disable-z-only", action="store_true")
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Only generate plots from existing summary.json/csv in --out-dir.",
    )
    p.add_argument("--parallel", action="store_true", default=True)
    p.add_argument("--no-parallel", dest="parallel", action="store_false")
    p.add_argument(
        "--probe-convergence-study",
        action="store_true",
        dest="probe_convergence_study",
        help=(
            "Run probe-budget convergence only: vary m=n_pasts=n_futures, repeat probe draws, "
            "compare to --convergence-m-ref. Implies n_seeds=1; uses four default cases; single --cut."
        ),
    )
    p.add_argument(
        "--convergence-m-list",
        type=str,
        default="4,8,16,32",
        help='Comma-separated probe budgets m (N_p=N_f=m), e.g. "4,8,16,32".',
    )
    p.add_argument(
        "--convergence-m-ref",
        type=int,
        default=64,
        help="Reference probe budget m_ref (large); errors vs mean metrics at m_ref over the same draws.",
    )
    p.add_argument(
        "--convergence-probe-draws",
        type=int,
        default=5,
        dest="convergence_probe_draws",
        help="Number of independent probe-set draws per m.",
    )
    return p.parse_args()


def _parse_j_sweep(spec: str) -> list[float]:
    vals = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    if not vals:
        raise ValueError("j-sweep must contain at least one value.")
    return vals


def _parse_int_list(spec: str | None) -> list[int]:
    if spec is None:
        return []
    vals = []
    for tok in spec.split(","):
        tok = tok.strip()
        if tok:
            vals.append(int(tok))
    return vals


def _sample_step(rng: np.random.Generator) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    rho_prep, effect, feat = _sample_random_intervention_parts(rng)
    psi_meas = _psi_from_rank1_projector(effect)
    psi_prep = _psi_from_rank1_projector(rho_prep)
    return feat.astype(np.float32), (psi_meas, psi_prep)


def _sample_cut_measurement_only(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Past side at cut: rank-1 effect only; returns (choi_feat32, psi_meas)."""
    _rho_prep, effect, feat = _sample_random_intervention_parts(rng)
    psi_meas = _psi_from_rank1_projector(effect)
    return feat.astype(np.float32), psi_meas


def _sample_cut_preparation_only(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Future side at cut: rank-1 preparation only; returns (choi_feat32, psi_prep)."""
    rho_prep, _effect, feat = _sample_random_intervention_parts(rng)
    psi_prep = _psi_from_rank1_projector(rho_prep)
    return feat.astype(np.float32), psi_prep


def _build_past_and_future_sets(
    *,
    cut: int,
    k: int,
    n_pasts: int,
    n_futures: int,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[list[tuple[np.ndarray, np.ndarray]]],
    list[np.ndarray],
    list[np.ndarray],
    list[list[tuple[np.ndarray, np.ndarray]]],
]:
    """Sample past/future probes with split instrument at slot ``cut`` (1-based).

    Past: full instruments at slots ``1..cut-1``, then measurement-only at slot ``cut``.
    Future: preparation-only at slot ``cut``, then full instruments at ``cut+1..k``.
    """
    c = int(cut)
    kk = int(k)
    if not (1 <= c <= kk):
        msg = f"cut must satisfy 1 <= cut <= k, got cut={cut}, k={k}"
        raise ValueError(msg)
    past_full = c - 1
    future_full = kk - c

    # Past: (c-1) full slots + 1 feature row for measurement-only at cut.
    past_nfeat = past_full + 1
    past_features = np.empty((n_pasts, past_nfeat, 32), dtype=np.float32)
    past_pairs: list[list[tuple[np.ndarray, np.ndarray]]] = []
    past_cut_meas: list[np.ndarray] = []

    for i in range(n_pasts):
        pairs_i: list[tuple[np.ndarray, np.ndarray]] = []
        for t in range(past_full):
            feat, pair = _sample_step(rng)
            past_features[i, t] = feat
            pairs_i.append(pair)
        feat_m, psi_m = _sample_cut_measurement_only(rng)
        past_features[i, past_full] = feat_m
        past_cut_meas.append(psi_m)
        past_pairs.append(pairs_i)

    # Future: 1 row for prep-only at cut + (k-c) full instruments.
    fut_nfeat = 1 + future_full
    future_features = np.empty((n_futures, fut_nfeat, 32), dtype=np.float32)
    future_prep_cut: list[np.ndarray] = []
    future_pairs: list[list[tuple[np.ndarray, np.ndarray]]] = []

    for j in range(n_futures):
        feat_p, psi_p = _sample_cut_preparation_only(rng)
        future_features[j, 0] = feat_p
        future_prep_cut.append(psi_p)
        pairs_j: list[tuple[np.ndarray, np.ndarray]] = []
        for t in range(future_full):
            feat, pair = _sample_step(rng)
            future_features[j, 1 + t] = feat
            pairs_j.append(pair)
        future_pairs.append(pairs_j)

    return past_features, future_features, past_pairs, past_cut_meas, future_prep_cut, future_pairs


def _list_initial_states_sys_env0(*, L: int, n_seeds: int, rng: np.random.Generator) -> list[np.ndarray]:
    """System = site 0; environment = sites 1..L-1 fixed to |0⟩."""
    if n_seeds < 1:
        msg = "n_seeds must be >= 1."
        raise ValueError(msg)
    if n_seeds == 1:
        psi = np.zeros(2**L, dtype=np.complex128)
        psi[0] = 1.0 + 0.0j
        return [psi]
    out: list[np.ndarray] = []
    for _ in range(n_seeds):
        psi_sys = _random_pure_state(rng)
        if L <= 1:
            out.append(psi_sys.astype(np.complex128))
            continue
        z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
        psi = psi_sys.astype(np.complex128)
        for _i in range(L - 1):
            psi = np.kron(psi, z)
        nrm = float(np.linalg.norm(psi))
        out.append(psi / max(nrm, 1e-15))
    return out


def _build_all_pairs_grid(
    *,
    k: int,
    cut: int,
    past_pairs: list[list[tuple[np.ndarray, np.ndarray]]],
    past_cut_meas: list[np.ndarray],
    future_prep_cut: list[np.ndarray],
    future_pairs: list[list[tuple[np.ndarray, np.ndarray]]],
) -> tuple[list[list[tuple[np.ndarray, np.ndarray]]], int, int]:
    """Concatenate past probes, split pair at slot ``cut``, then future probes."""
    c = int(cut)
    kk = int(k)
    n_p = len(past_pairs)
    n_f = len(future_pairs)
    if len(past_cut_meas) != n_p:
        msg = f"past_cut_meas length {len(past_cut_meas)} != n_pasts={n_p}."
        raise ValueError(msg)
    if len(future_prep_cut) != n_f:
        msg = f"future_prep_cut length {len(future_prep_cut)} != n_futures={n_f}."
        raise ValueError(msg)
    all_pairs: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for i in range(n_p):
        for j in range(n_f):
            full: list[tuple[np.ndarray, np.ndarray]] = []
            for t in range(c - 1):
                full.append(past_pairs[i][t])
            full.append((past_cut_meas[i], future_prep_cut[j]))
            for t in range(kk - c):
                full.append(future_pairs[j][t])
            if len(full) != kk:
                raise RuntimeError("Internal error: full sequence length mismatch.")
            all_pairs.append(full)
    return all_pairs, n_p, n_f


def _simulate_output_rhos_for_initial_state(
    *,
    operator: MPO,
    sim_params: AnalogSimParams,
    static_ctx: Any,
    k: int,
    all_pairs: list[list[tuple[np.ndarray, np.ndarray]]],
    n_p: int,
    n_f: int,
    initial_psi: np.ndarray,
    parallel: bool,
) -> np.ndarray:
    n_tot = n_p * n_f
    if len(all_pairs) != n_tot:
        msg = f"all_pairs length {len(all_pairs)} != n_p*n_f={n_tot}."
        raise ValueError(msg)
    initial_psis = [np.asarray(initial_psi, dtype=np.complex128).copy() for _ in range(n_tot)]
    final_packed = _simulate_sequences(
        operator=operator,
        sim_params=sim_params,
        timesteps=[float(sim_params.dt)] * (int(k) + 1),
        psi_pairs_list=all_pairs,
        initial_psis=initial_psis,
        static_ctx=static_ctx,
        parallel=bool(parallel),
        show_progress=True,
        record_step_states=False,
    )
    if not isinstance(final_packed, np.ndarray):
        raise RuntimeError("Expected ndarray output from exact simulation.")
    if final_packed.shape[0] != n_tot:
        raise RuntimeError(
            f"Expected {n_tot} final states from exact simulation, got {final_packed.shape[0]}."
        )
    # Assumption: `_simulate_sequences(..., record_step_states=False)` preserves input sequence order.
    # We construct `all_pairs` in nested (past_i, future_j) order, so reshape(n_p, n_f, 8) matches that grid.
    return final_packed.reshape(n_p, n_f, 8).astype(np.float32)


def _average_analysis_dicts(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    """Average scalar metrics; keep first seed's large arrays for saving (caller replaces)."""
    if not analyses:
        raise ValueError("empty analyses")
    keys_float = [
        "delta",
        "delta_norm",
        "entropy",
        "participation_ratio",
        "max_row_distance",
        "mean_row_distance",
        "median_row_distance",
    ]
    out: dict[str, Any] = {}
    for key in keys_float:
        out[key] = float(np.mean([float(a[key]) for a in analyses]))
    out["rank"] = int(round(float(np.mean([float(a["rank"]) for a in analyses]))))
    return out


def _average_z_packs(z_list: list[dict[str, Any]]) -> dict[str, Any]:
    z0 = z_list[0]
    return {
        "vz": z0["vz"],
        "vz_centered": z0["vz_centered"],
        "singular_values_z": z0["singular_values_z"],
        "delta_z": float(np.mean([float(z["delta_z"]) for z in z_list])),
        "delta_z_norm": float(np.mean([float(z["delta_z_norm"]) for z in z_list])),
        "entropy_z": float(np.mean([float(z["entropy_z"]) for z in z_list])),
        "rank_z": int(round(float(np.mean([float(z["rank_z"]) for z in z_list])))),
        "participation_ratio_z": float(np.mean([float(z["participation_ratio_z"]) for z in z_list])),
    }


def _build_v_matrix(rho8_ij: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_p, n_f, d_out = rho8_ij.shape
    # V is built from rho8 (fixed real vectorization of output density matrices).
    # Since rho8 is a fixed linear encoding, row-variation/SVD diagnostics are representation-consistent.
    v = rho8_ij.reshape(n_p, n_f * d_out).astype(np.float64)
    v_centered = v - v.mean(axis=0, keepdims=True)
    return v, v_centered


def _pairwise_row_distances(v: np.ndarray) -> np.ndarray:
    n = int(v.shape[0])
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            d[i, j] = float(np.linalg.norm(v[i] - v[j]))
    return d


def _analyze_v_matrix(v: np.ndarray, v_centered: np.ndarray) -> dict[str, Any]:
    fro_v_sq = float(np.linalg.norm(v, ord="fro") ** 2)
    fro_c_sq = float(np.linalg.norm(v_centered, ord="fro") ** 2)
    delta_norm = float(fro_c_sq / fro_v_sq) if fro_v_sq > 0.0 else 0.0

    s = np.linalg.svd(v_centered, compute_uv=False).astype(np.float64)
    p = s * s
    p_sum = float(np.sum(p))
    if p_sum <= 0.0:
        entropy = 0.0
        pr = 0.0
    else:
        q = p / p_sum
        q = np.clip(q, 1e-30, 1.0)
        entropy = float(-np.sum(q * np.log(q)))
        p2 = float(np.sum(p * p))
        pr = float((p_sum * p_sum) / p2) if p2 > 0.0 else 0.0

    tol = 1e-10 * max(1.0, float(s[0]) if s.size > 0 else 1.0)
    rank = int(np.sum(s > tol))

    dmat = _pairwise_row_distances(v)
    tri = dmat[np.triu_indices(dmat.shape[0], k=1)]
    max_d = float(np.max(tri)) if tri.size else 0.0
    mean_d = float(np.mean(tri)) if tri.size else 0.0
    med_d = float(np.median(tri)) if tri.size else 0.0

    return {
        "delta": fro_c_sq,
        "delta_norm": delta_norm,
        "singular_values": s,
        "entropy": entropy,
        "rank": rank,
        "participation_ratio": pr,
        "row_distances": dmat,
        "max_row_distance": max_d,
        "mean_row_distance": mean_d,
        "median_row_distance": med_d,
    }


def _rho8_to_rho(rho8_ij: np.ndarray) -> np.ndarray:
    n_p, n_f, _ = rho8_ij.shape
    out = np.empty((n_p, n_f, 2, 2), dtype=np.complex128)
    for i in range(n_p):
        for j in range(n_f):
            out[i, j] = unpack_rho8(rho8_ij[i, j])
    return out


def _z_from_rhos(rhos_ij: np.ndarray) -> np.ndarray:
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    n_p, n_f, _, _ = rhos_ij.shape
    vz = np.empty((n_p, n_f), dtype=np.float64)
    for i in range(n_p):
        for j in range(n_f):
            vz[i, j] = float(np.trace(rhos_ij[i, j] @ z).real)
    return vz


def _analyze_z_matrix(vz: np.ndarray) -> dict[str, Any]:
    v = vz.astype(np.float64)
    v_c = v - v.mean(axis=0, keepdims=True)
    res = _analyze_v_matrix(v, v_c)
    return {
        "vz": v,
        "vz_centered": v_c,
        "delta_z": float(res["delta"]),
        "delta_z_norm": float(res["delta_norm"]),
        "singular_values_z": res["singular_values"],
        "entropy_z": float(res["entropy"]),
        "rank_z": int(res["rank"]),
        "participation_ratio_z": float(res["participation_ratio"]),
    }


def _save_point(
    *,
    out_point: Path,
    rhos: np.ndarray,
    v: np.ndarray,
    v_centered: np.ndarray,
    singular_values: np.ndarray,
    summary: dict[str, Any],
    z_pack: dict[str, Any] | None,
) -> None:
    out_point.mkdir(parents=True, exist_ok=True)
    np.save(out_point / "rhos.npy", rhos)
    np.save(out_point / "V.npy", v)
    np.save(out_point / "V_centered.npy", v_centered)
    np.save(out_point / "singular_values.npy", singular_values)
    if z_pack is not None:
        np.save(out_point / "vz.npy", z_pack["vz"])
        np.save(out_point / "vz_centered.npy", z_pack["vz_centered"])
        np.save(out_point / "singular_values_z.npy", z_pack["singular_values_z"])
    (out_point / "summary.json").write_text(json.dumps(summary, indent=2))


def run_single_parameter_point(
    *,
    case_name: str,
    J: float,
    g: float,
    args: argparse.Namespace,
    shared_past_features: np.ndarray,
    shared_future_features: np.ndarray,
    shared_past_pairs: list[list[tuple[np.ndarray, np.ndarray]]],
    shared_past_cut_meas: list[np.ndarray],
    shared_future_prep_cut: list[np.ndarray],
    shared_future_pairs: list[list[tuple[np.ndarray, np.ndarray]]],
    shared_initial_states: list[np.ndarray],
    init_stack: np.ndarray,
    out_dir: Path,
    include_z: bool,
    save_outputs: bool = True,
) -> dict[str, Any]:
    print(f"\ncase={case_name} J={J:.6f} g={g:.6f}", flush=True)
    op = MPO.ising(length=int(args.L), J=float(J), g=float(g))
    sim_params = AnalogSimParams(dt=float(args.dt), solver="MCWF", show_progress=False)
    static_ctx = make_mcwf_static_context(op, sim_params, noise_model=None)

    n_seeds = int(args.n_seeds)
    if len(shared_initial_states) != n_seeds:
        msg = f"shared_initial_states length {len(shared_initial_states)} != n_seeds={n_seeds}."
        raise ValueError(msg)

    all_pairs, n_p, n_f = _build_all_pairs_grid(
        k=int(args.k),
        cut=int(args.cut),
        past_pairs=shared_past_pairs,
        past_cut_meas=shared_past_cut_meas,
        future_prep_cut=shared_future_prep_cut,
        future_pairs=shared_future_pairs,
    )

    if n_seeds == 1:
        rho8_ij = _simulate_output_rhos_for_initial_state(
            operator=op,
            sim_params=sim_params,
            static_ctx=static_ctx,
            k=int(args.k),
            all_pairs=all_pairs,
            n_p=n_p,
            n_f=n_f,
            initial_psi=shared_initial_states[0],
            parallel=bool(args.parallel),
        )
        rhos = _rho8_to_rho(rho8_ij)
        v, v_centered = _build_v_matrix(rho8_ij)
        ana = _analyze_v_matrix(v, v_centered)
        z_pack: dict[str, Any] | None = None
        if include_z:
            vz = _z_from_rhos(rhos)
            z_pack = _analyze_z_matrix(vz)
    else:
        ana_list: list[dict[str, Any]] = []
        z_list: list[dict[str, Any]] = []
        rho8_first: np.ndarray | None = None
        for idx, psi0 in enumerate(shared_initial_states):
            rho8_ij_s = _simulate_output_rhos_for_initial_state(
                operator=op,
                sim_params=sim_params,
                static_ctx=static_ctx,
                k=int(args.k),
                all_pairs=all_pairs,
                n_p=n_p,
                n_f=n_f,
                initial_psi=psi0,
                parallel=bool(args.parallel),
            )
            if idx == 0:
                rho8_first = rho8_ij_s
            v_s, v_c_s = _build_v_matrix(rho8_ij_s)
            ana_list.append(_analyze_v_matrix(v_s, v_c_s))
            if include_z:
                rhos_s = _rho8_to_rho(rho8_ij_s)
                vz_s = _z_from_rhos(rhos_s)
                z_list.append(_analyze_z_matrix(vz_s))
        if rho8_first is None:
            raise RuntimeError("internal: no rho8 for n_seeds>1")
        rho8_ij = rho8_first
        rhos = _rho8_to_rho(rho8_ij)
        v, v_centered = _build_v_matrix(rho8_ij)
        ana_avg = _average_analysis_dicts(ana_list)
        ana0 = ana_list[0]
        ana = {
            **ana_avg,
            "singular_values": ana0["singular_values"],
            "row_distances": ana0["row_distances"],
        }
        if include_z:
            z_pack = _average_z_packs(z_list)
        else:
            z_pack = None

    point_summary: dict[str, Any] = {
        "case_name": case_name,
        "L": int(args.L),
        "k": int(args.k),
        "dt": float(args.dt),
        "cut": int(args.cut),
        "J": float(J),
        "g": float(g),
        "n_pasts": int(args.n_pasts),
        "n_futures": int(args.n_futures),
        "n_seeds": int(args.n_seeds),
        "delta": float(ana["delta"]),
        "delta_norm": float(ana["delta_norm"]),
        "entropy": float(ana["entropy"]),
        "rank": int(ana["rank"]),
        "participation_ratio": float(ana["participation_ratio"]),
        "max_row_distance": float(ana["max_row_distance"]),
        "mean_row_distance": float(ana["mean_row_distance"]),
        "median_row_distance": float(ana["median_row_distance"]),
    }
    if n_seeds > 1:
        point_summary["metrics_averaged_over_initial_states"] = True
    if z_pack is not None:
        point_summary.update(
            {
                "delta_z": float(z_pack["delta_z"]),
                "delta_z_norm": float(z_pack["delta_z_norm"]),
                "entropy_z": float(z_pack["entropy_z"]),
                "rank_z": int(z_pack["rank_z"]),
                "participation_ratio_z": float(z_pack["participation_ratio_z"]),
            }
        )

    if save_outputs:
        point_dir = out_dir / f"{case_name}_J{J:.3f}_g{g:.3f}"
        _save_point(
            out_point=point_dir,
            rhos=rhos,
            v=v,
            v_centered=v_centered,
            singular_values=ana["singular_values"],
            summary=point_summary,
            z_pack=z_pack,
        )
        np.save(point_dir / "past_features.npy", shared_past_features)
        np.save(point_dir / "future_features.npy", shared_future_features)
        np.save(point_dir / "initial_states.npy", init_stack)
        np.save(point_dir / "row_distances.npy", ana["row_distances"])

    # Quick per-case readout.
    print(
        "  "
        + ", ".join(
            [
                f"delta={point_summary['delta']:.6e}",
                f"delta_norm={point_summary['delta_norm']:.6e}",
                f"entropy={point_summary['entropy']:.6e}",
                f"rank={point_summary['rank']}",
                f"PR={point_summary['participation_ratio']:.6e}",
                f"mean_row_dist={point_summary['mean_row_distance']:.6e}",
            ]
        ),
        flush=True,
    )
    return point_summary


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def configure_matplotlib_for_paper() -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "font.family": "sans-serif",
            "mathtext.default": "it",
        }
    )


def savefig_base(fig: Any, path_stem: Path) -> None:
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def _case_style(row: dict[str, Any]) -> dict[str, Any]:
    name = str(row.get("case_name", ""))
    jv = float(row.get("J", 0.0))
    gv = float(row.get("g", 0.0))
    if name == "trivial" or (abs(jv) <= 1e-15 and abs(gv) <= 1e-15):
        return {"color": "black", "marker": "x", "label": "trivial"}
    if name == "markov_ref" or (abs(jv) <= 1e-15 and abs(gv - 1.0) <= 1e-15):
        return {"color": "0.45", "marker": "s", "label": "markov_ref"}
    if jv < 0.8:
        return {"color": "#1f77b4", "marker": "o", "label": f"J={jv:g}"}
    return {"color": "#d62728", "marker": "^", "label": f"J={jv:g}"}


def _add_panel_label(ax: Any, label: str) -> None:
    ax.text(0.03, 0.95, label, transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")


def _safe_logscale_if_wide(ax: Any, ys: list[float], *, ratio_threshold: float = 100.0) -> None:
    ys_pos = [float(y) for y in ys if float(y) > 0.0]
    has_zero = any(float(y) <= 0.0 for y in ys)
    if len(ys_pos) < 2 or has_zero:
        return
    span = max(ys_pos) / max(min(ys_pos), 1e-300)
    if span >= ratio_threshold:
        ax.set_yscale("log")


def _coerce_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        for key in [
            "J",
            "g",
            "delta_norm",
            "entropy",
            "participation_ratio",
            "mean_row_distance",
            "delta_z_norm",
            "entropy_z",
            "cut",
        ]:
            if key in rr:
                try:
                    rr[key] = float(rr[key])
                except Exception:
                    pass
        out.append(rr)
    return out


def _find_representative_point_dirs(out_dir: Path) -> dict[str, Path]:
    candidates = {
        "trivial": ["trivial_J0.000_g0.000"],
        "markov_ref": ["markov_ref_J0.000_g1.000"],
        "j04": ["sweep_J0.400_g1.000", "nm_j0.4_J0.400_g1.000"],
        "j06": ["sweep_J0.600_g1.000", "nm_j0.6_J0.600_g1.000"],
        "j10": ["sweep_J1.000_g1.000", "nm_j1.0_J1.000_g1.000"],
        "j12": ["sweep_J1.200_g1.000", "nm_j1.2_J1.200_g1.000"],
        "j14": ["sweep_J1.400_g1.000", "nm_j1.4_J1.400_g1.000"],
        "j16": ["sweep_J1.600_g1.000", "nm_j1.6_J1.600_g1.000"],
        "j18": ["sweep_J1.800_g1.000", "nm_j1.8_J1.800_g1.000"],
        "j20": ["sweep_J2.000_g1.000", "nm_j2.0_J2.000_g1.000"],
    }
    found: dict[str, Path] = {}
    search_roots = [out_dir] + [p for p in out_dir.glob("cut_*") if p.is_dir()]
    for key, names in candidates.items():
        for root in search_roots:
            for name in names:
                p = root / name
                if p.exists() and p.is_dir():
                    found[key] = p
                    break
            if key in found:
                break
    return found


def plot_main_j_sweep(rows: list[dict[str, Any]], out_dir: Path, g_fixed: float, include_z: bool) -> None:
    _ = include_z
    import matplotlib.pyplot as plt

    rows = _coerce_rows(rows)
    rows_g = sorted([r for r in rows if abs(float(r["g"]) - float(g_fixed)) <= 1e-12], key=lambda r: float(r["J"]))
    if not rows_g:
        return
    row_trivial = next((r for r in rows if str(r.get("case_name")) == "trivial"), None)
    row_markov = next((r for r in rows if str(r.get("case_name")) == "markov_ref"), None)
    sweep = [r for r in rows_g if float(r["J"]) > 1e-15]
    if not sweep:
        return

    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.45), constrained_layout=True)
    metrics = [("delta_norm", r"$\Delta_{\mathrm{norm}}$"), ("entropy", r"$S_V$"), ("participation_ratio", r"$\mathrm{PR}$")]
    panel_labels = ["(a)", "(b)", "(c)"]
    j_vals = sorted({float(r["J"]) for r in rows_g})
    for ax, (metric, ylab), plab in zip(axs, metrics, panel_labels, strict=True):
        xs = [float(r["J"]) for r in sweep]
        ys = [float(r[metric]) for r in sweep]
        st = _case_style(sweep[-1])
        ax.plot(xs, ys, color=st["color"], marker=st["marker"], label="non-Markovian sweep")
        if row_trivial is not None:
            st_t = _case_style(row_trivial)
            ax.plot([0.0], [float(row_trivial[metric])], linestyle="None", marker=st_t["marker"], color=st_t["color"], label="trivial")
        if row_markov is not None:
            st_m = _case_style(row_markov)
            ax.plot([0.0], [float(row_markov[metric])], linestyle="None", marker=st_m["marker"], color=st_m["color"], label="markov_ref")
        ax.axhline(0.0, color="0.75", linewidth=0.8)
        if metric == "delta_norm":
            _safe_logscale_if_wide(ax, ys + ([float(row_trivial[metric])] if row_trivial else []) + ([float(row_markov[metric])] if row_markov else []))
        ax.set_xlabel(r"$J$")
        ax.set_ylabel(ylab)
        ax.set_xticks(j_vals)
        ax.grid(True, axis="y")
        _add_panel_label(ax, plab)
        if plab == "(a)":
            ax.legend(frameon=False, loc="best", handlelength=2.2)
    savefig_base(fig, out_dir / "fig_main_j_sweep")
    plt.close(fig)


def plot_singular_spectra(out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    reps = _find_representative_point_dirs(out_dir)
    need = ["trivial", "markov_ref", "j04", "j06", "j10", "j12", "j14", "j16", "j18", "j20"]
    if not all(k in reps for k in need):
        return

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.5), constrained_layout=True)
    mapping = [
        ("trivial", "trivial"),
        ("markov_ref", "markov_ref"),
        ("j04", "J=0.4"),
        ("j06", "J=0.6"),
        ("j10", "J=1.0"),
        ("j12", "J=1.2"),
        ("j14", "J=1.4"),
        ("j16", "J=1.6"),
        ("j18", "J=1.8"),
        ("j20", "J=2.0"),
    ]
    for key, label in mapping:
        sv_path = reps[key] / "singular_values.npy"
        if not sv_path.exists():
            continue
        s = np.load(sv_path).astype(np.float64)
        p = s * s
        denom = float(np.sum(p))
        if denom <= 0.0:
            x = np.array([0], dtype=np.int64)
            w = np.array([1e-30], dtype=np.float64)
        else:
            w = p / denom
            idx = np.arange(w.shape[0], dtype=np.int64)
            keep = w > 1e-16
            if np.any(keep):
                x = idx[keep]
                w = w[keep]
            else:
                x = np.array([0], dtype=np.int64)
                w = np.array([1e-30], dtype=np.float64)
        st = _case_style({"case_name": "sweep" if key.startswith("j") else key, "J": 0.6 if key == "j06" else (1.0 if key == "j10" else 0.0), "g": 0.0 if key == "trivial" else 1.0})
        ax.semilogy(x, w, marker=st["marker"], color=st["color"], label=label)
    ax.set_xlabel(r"singular-value index $n$")
    ax.set_ylabel(r"$p_n$")
    ax.grid(True, which="both", axis="y")
    ax.legend(frameon=False, loc="best")
    ax.set_ylim(1e-6, 1)
    savefig_base(fig, out_dir / "fig_singular_spectra")
    plt.close(fig)


def plot_cut_sweep(rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    rows = _coerce_rows(rows)
    if not rows or "cut" not in rows[0]:
        return

    def label(r: dict[str, Any]) -> str:
        name = str(r["case_name"])
        if name == "trivial":
            return "trivial"
        if name == "markov_ref":
            return "markov_ref"
        if abs(float(r["J"]) - 0.6) < 1e-12:
            return "J=0.6"
        if abs(float(r["J"]) - 1.0) < 1e-12:
            return "J=1.0"
        return ""

    keep = [r for r in rows if label(r)]
    if not keep:
        return

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.55), constrained_layout=True)
    metrics = [("entropy", r"$S_V$"), ("delta_norm", r"$\Delta_{\mathrm{norm}}$")]
    panel_labels = ["(a)", "(b)"]
    for ax, (metric, ylab), plab in zip(axs, metrics, panel_labels, strict=True):
        labels = ["trivial", "markov_ref", "J=0.6", "J=1.0"]
        all_ys: list[float] = []
        for lab in labels:
            sub = sorted([r for r in keep if label(r) == lab], key=lambda x: int(float(x["cut"])))
            if not sub:
                continue
            xs = [int(float(r["cut"])) for r in sub]
            ys = [float(r[metric]) for r in sub]
            all_ys.extend(ys)
            st = _case_style(sub[0])
            ax.plot(xs, ys, marker=st["marker"], color=st["color"], label=lab)
        ax.axhline(0.0, color="0.75", linewidth=0.8)
        if metric == "delta_norm":
            _safe_logscale_if_wide(ax, all_ys)
        ax.set_xlabel(r"cut $c$")
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y")
        _add_panel_label(ax, plab)
        if plab == "(a)":
            ax.legend(frameon=False, loc="best")
    savefig_base(fig, out_dir / "fig_cut_sweep")
    plt.close(fig)


def plot_full_vs_z(rows: list[dict[str, Any]], out_dir: Path, g_fixed: float) -> None:
    import matplotlib.pyplot as plt

    rows = _coerce_rows(rows)
    rows_g = sorted([r for r in rows if abs(float(r["g"]) - float(g_fixed)) <= 1e-12], key=lambda r: float(r["J"]))
    if not rows_g or not all(("entropy_z" in r and "delta_z_norm" in r) for r in rows_g):
        return

    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.55), constrained_layout=True)
    metrics = [("entropy", "entropy_z", r"$S_V$"), ("delta_norm", "delta_z_norm", r"$\Delta_{\mathrm{norm}}$")]
    panel_labels = ["(a)", "(b)"]
    for ax, (m_full, m_z, ylab), plab in zip(axs, metrics, panel_labels, strict=True):
        labels = ["trivial", "markov_ref", "J=0.6", "J=1.0"]
        for lab in labels:
            if lab == "trivial":
                sub = [r for r in rows if str(r.get("case_name")) == "trivial"]
            elif lab == "markov_ref":
                sub = [r for r in rows if str(r.get("case_name")) == "markov_ref" and abs(float(r["g"]) - float(g_fixed)) <= 1e-12]
            elif lab == "J=0.6":
                sub = [r for r in rows_g if abs(float(r["J"]) - 0.6) < 1e-12]
            else:
                sub = [r for r in rows_g if abs(float(r["J"]) - 1.0) < 1e-12]
            if not sub:
                continue
            rr = sub[0]
            st = _case_style(rr)
            x = float(rr["J"])
            ax.plot([x], [float(rr[m_full])], linestyle="-", marker=st["marker"], color=st["color"])
            ax.plot([x], [float(rr[m_z])], linestyle="--", marker=st["marker"], color=st["color"])
        ax.set_xlabel(r"$J$")
        ax.set_ylabel(ylab)
        ax.grid(True, axis="y")
        _add_panel_label(ax, plab)
    axs[0].plot([], [], "-", color="0.2", label="full-state")
    axs[0].plot([], [], "--", color="0.2", label="Z-only")
    axs[0].legend(frameon=False, loc="best")
    savefig_base(fig, out_dir / "fig_full_vs_z")
    plt.close(fig)


def plot_row_distance_heatmaps(out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    reps = _find_representative_point_dirs(out_dir)
    keys = [("trivial", "trivial"), ("markov_ref", "markov_ref"), ("j10", "J=1.0")]
    if not all(k in reps for k, _ in keys):
        return
    mats = []
    for k, _lab in keys:
        p = reps[k] / "row_distances.npy"
        if not p.exists():
            return
        mats.append(np.load(p).astype(np.float64))
    vmax = max(float(np.max(m)) for m in mats)

    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35), constrained_layout=True)
    ims = []
    for ax, (k, lab), mat, plab in zip(axs, keys, mats, ["(a)", "(b)", "(c)"], strict=True):
        im = ax.imshow(mat, vmin=0.0, vmax=vmax, origin="lower", aspect="equal", interpolation="nearest")
        ims.append(im)
        ax.set_xlabel(r"past index $i$")
        ax.set_ylabel(r"past index $j$")
        ax.set_title("" if False else None)
        _add_panel_label(ax, plab)
        ax.text(0.52, 0.05, lab, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color="white")
    cbar = fig.colorbar(ims[-1], ax=axs, shrink=0.9)
    cbar.ax.set_ylabel("row distance", fontsize=9)
    savefig_base(fig, out_dir / "fig_row_distance_heatmaps")
    plt.close(fig)


def _make_plots(rows: list[dict[str, Any]], out_dir: Path, include_z: bool, g_fixed: float) -> None:
    if not rows:
        return
    configure_matplotlib_for_paper()
    plot_main_j_sweep(rows, out_dir, g_fixed=g_fixed, include_z=include_z)
    plot_singular_spectra(out_dir)
    plot_full_vs_z(rows, out_dir, g_fixed=g_fixed)
    plot_row_distance_heatmaps(out_dir)


def _make_cut_sweep_plots(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    configure_matplotlib_for_paper()
    plot_cut_sweep(rows, out_dir)
    # Keep backward-compatible filenames requested earlier.
    import matplotlib.pyplot as plt

    rows_c = _coerce_rows(rows)
    def _label(r: dict[str, Any]) -> str:
        if str(r["case_name"]) == "trivial":
            return "trivial"
        if str(r["case_name"]) == "markov_ref":
            return "markov_ref"
        if abs(float(r["J"]) - 0.6) < 1e-12:
            return "J=0.6"
        if abs(float(r["J"]) - 1.0) < 1e-12:
            return "J=1.0"
        return ""
    data = [r for r in rows_c if _label(r)]
    if not data:
        return
    for metric, out_stem in [("entropy", "cut_sweep_entropy"), ("delta_norm", "cut_sweep_delta_norm")]:
        fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.5), constrained_layout=True)
        labels = ["trivial", "markov_ref", "J=0.6", "J=1.0"]
        ys_all: list[float] = []
        for lab in labels:
            sub = sorted([r for r in data if _label(r) == lab], key=lambda r: int(float(r["cut"])))
            if not sub:
                continue
            xs = [int(float(r["cut"])) for r in sub]
            ys = [float(r[metric]) for r in sub]
            ys_all.extend(ys)
            st = _case_style(sub[0])
            ax.plot(xs, ys, marker=st["marker"], color=st["color"], label=lab)
        if metric == "delta_norm":
            _safe_logscale_if_wide(ax, ys_all)
            ax.set_ylabel(r"$\Delta_{\mathrm{norm}}$")
        else:
            ax.set_ylabel(r"$S_V$")
        ax.set_xlabel(r"cut $c$")
        ax.grid(True, axis="y")
        ax.legend(frameon=False, loc="best")
        savefig_base(fig, out_dir / out_stem)
        plt.close(fig)


def _aggregate_cut_rows_from_subdirs(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_csv in sorted(out_dir.glob("cut_*/summary.csv")):
        with summary_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rr = dict(row)
                # Ensure cut column is present/consistent from folder name.
                try:
                    cut_dir = summary_csv.parent.name  # cut_XX
                    cut_val = int(cut_dir.split("_", 1)[1])
                    rr["cut"] = cut_val
                except Exception:
                    pass
                rows.append(rr)
    return rows


def run_benchmark_for_cut(
    *,
    cut: int,
    cases: list[tuple[str, float, float]],
    args: argparse.Namespace,
    out_dir: Path,
    include_z: bool,
) -> list[dict[str, Any]]:
    cut_dir = out_dir / f"cut_{int(cut):02d}"
    cut_dir.mkdir(parents=True, exist_ok=True)

    # Per-cut shared probes with deterministic seed rule.
    seed_base = int(args.seed)
    shared_rng = np.random.default_rng(seed_base + 1000 * int(cut))
    (
        shared_past_features,
        shared_future_features,
        shared_past_pairs,
        shared_past_cut_meas,
        shared_future_prep_cut,
        shared_future_pairs,
    ) = _build_past_and_future_sets(
        cut=int(cut),
        k=int(args.k),
        n_pasts=int(args.n_pasts),
        n_futures=int(args.n_futures),
        rng=shared_rng,
    )
    np.save(cut_dir / "shared_past_features.npy", shared_past_features)
    np.save(cut_dir / "shared_future_features.npy", shared_future_features)
    np.save(
        cut_dir / "shared_past_cut_measurement_kets.npy",
        np.stack([np.asarray(v, dtype=np.complex128) for v in shared_past_cut_meas], axis=0),
    )
    np.save(
        cut_dir / "shared_future_cut_preparation_kets.npy",
        np.stack([np.asarray(v, dtype=np.complex128) for v in shared_future_prep_cut], axis=0),
    )

    # Fixed initial-state ensemble for this cut only (reused for every J, g in the sweep).
    init_rng_states = np.random.default_rng(seed_base + 1000 * int(cut) + 50_000)
    shared_initial_list = _list_initial_states_sys_env0(
        L=int(args.L), n_seeds=int(args.n_seeds), rng=init_rng_states
    )
    init_stack = np.stack(shared_initial_list, axis=0)
    np.save(cut_dir / "shared_initial_states.npy", init_stack)

    # Reuse core per-point pipeline by cloning args with overridden cut.
    cut_args = argparse.Namespace(**vars(args))
    cut_args.cut = int(cut)

    rows: list[dict[str, Any]] = []
    for case_name, J, g in cases:
        rows.append(
            run_single_parameter_point(
                case_name=case_name,
                J=float(J),
                g=float(g),
                args=cut_args,
                shared_past_features=shared_past_features,
                shared_future_features=shared_future_features,
                shared_past_pairs=shared_past_pairs,
                shared_past_cut_meas=shared_past_cut_meas,
                shared_future_prep_cut=shared_future_prep_cut,
                shared_future_pairs=shared_future_pairs,
                shared_initial_states=shared_initial_list,
                init_stack=init_stack,
                out_dir=cut_dir,
                include_z=include_z,
            )
        )

    (cut_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    _write_summary_csv(cut_dir / "summary.csv", rows)
    _make_plots(rows, cut_dir, include_z=include_z, g_fixed=float(args.g_fixed))
    return rows


def run_probe_convergence_study(*, args: argparse.Namespace, out_dir: Path, include_z: bool) -> None:
    """Vary probe budget m = N_p = N_f; repeat independent probe draws; compare to m_ref."""
    seed_base = int(args.seed)
    cut = int(args.cut)
    m_list = _parse_int_list(str(args.convergence_m_list))
    m_ref = int(args.convergence_m_ref)
    n_draws = int(args.convergence_probe_draws)
    if n_draws < 1:
        msg = "convergence-probe-draws must be >= 1."
        raise ValueError(msg)
    if m_ref < 1:
        msg = "convergence-m-ref must be >= 1."
        raise ValueError(msg)
    if not m_list:
        msg = "convergence-m-list must contain at least one m."
        raise ValueError(msg)
    all_m = sorted({int(x) for x in m_list} | {m_ref})
    for m in all_m:
        if m < 1:
            msg = f"probe budget m must be >= 1, got {m}."
            raise ValueError(msg)

    study_dir = out_dir / "probe_convergence"
    study_dir.mkdir(parents=True, exist_ok=True)

    # Fixed single initial state |0...0> for the whole study (n_seeds=1).
    init_list = _list_initial_states_sys_env0(
        L=int(args.L), n_seeds=1, rng=np.random.default_rng(seed_base + 88_000 + cut)
    )
    init_stack = np.stack(init_list, axis=0)
    np.save(study_dir / "shared_initial_states.npy", init_stack)

    conv_cases = [
        ("trivial", 0.0, 0.0),
        ("markov_ref", 0.0, float(args.g_fixed)),
        ("nm_j0.6", 0.6, float(args.g_fixed)),
        ("nm_j1.0", 1.0, float(args.g_fixed)),
    ]

    metrics_to_track = [
        "delta",
        "delta_norm",
        "entropy",
        "participation_ratio",
        "max_row_distance",
        "mean_row_distance",
        "rank",
    ]
    if include_z:
        metrics_to_track.extend(["delta_z", "delta_z_norm", "entropy_z", "rank_z", "participation_ratio_z"])

    detailed_rows: list[dict[str, Any]] = []

    for draw in range(n_draws):
        for m in all_m:
            probe_rng = np.random.default_rng(seed_base + 200_000 + int(m) * 1_000 + draw * 17_917 + cut * 3)
            p_feat, f_feat, p_pairs, p_cut_m, f_prep_c, f_pairs = _build_past_and_future_sets(
                cut=cut,
                k=int(args.k),
                n_pasts=int(m),
                n_futures=int(m),
                rng=probe_rng,
            )
            cut_args = argparse.Namespace(**vars(args))
            cut_args.cut = cut
            cut_args.n_pasts = int(m)
            cut_args.n_futures = int(m)
            cut_args.n_seeds = 1

            for case_name, J, g in conv_cases:
                summ = run_single_parameter_point(
                    case_name=case_name,
                    J=float(J),
                    g=float(g),
                    args=cut_args,
                    shared_past_features=p_feat,
                    shared_future_features=f_feat,
                    shared_past_pairs=p_pairs,
                    shared_past_cut_meas=p_cut_m,
                    shared_future_prep_cut=f_prep_c,
                    shared_future_pairs=f_pairs,
                    shared_initial_states=init_list,
                    init_stack=init_stack,
                    out_dir=study_dir,
                    include_z=include_z,
                    save_outputs=False,
                )
                row: dict[str, Any] = {
                    "probe_draw": int(draw),
                    "m": int(m),
                    "m_ref": int(m_ref),
                    "cut": int(cut),
                    **summ,
                }
                detailed_rows.append(row)

    summary_rows: list[dict[str, Any]] = []
    for case_name, J, g in conv_cases:
        ref_mean: dict[str, float] = {}
        for k in metrics_to_track:
            ref_vals = [
                float(r[k])
                for r in detailed_rows
                if str(r["case_name"]) == case_name and int(r["m"]) == m_ref and k in r
            ]
            ref_mean[k] = float(np.mean(ref_vals)) if ref_vals else float("nan")

        for m in all_m:
            srow: dict[str, Any] = {
                "case_name": case_name,
                "J": float(J),
                "g": float(g),
                "cut": int(cut),
                "L": int(args.L),
                "k": int(args.k),
                "m": int(m),
                "m_ref": int(m_ref),
                "n_probe_draws": int(n_draws),
            }
            for k in metrics_to_track:
                vals = [
                    float(r[k])
                    for r in detailed_rows
                    if str(r["case_name"]) == case_name and int(r["m"]) == m and k in r
                ]
                if not vals:
                    continue
                mu = float(np.mean(vals))
                std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                srow[f"{k}_mean"] = mu
                srow[f"{k}_std"] = std
                rmu = ref_mean.get(k, float("nan"))
                if int(m) == int(m_ref):
                    srow[f"{k}_err_vs_ref"] = 0.0
                elif np.isfinite(rmu):
                    srow[f"{k}_err_vs_ref"] = float(abs(mu - rmu))
                else:
                    srow[f"{k}_err_vs_ref"] = float("nan")
            summary_rows.append(srow)

    _write_summary_csv(study_dir / "probe_convergence_detailed.csv", detailed_rows)
    _write_summary_csv(study_dir / "probe_convergence_summary.csv", summary_rows)
    (study_dir / "probe_convergence_detailed.json").write_text(json.dumps(detailed_rows, indent=2))
    (study_dir / "probe_convergence_summary.json").write_text(json.dumps(summary_rows, indent=2))

    print("\n=== Probe convergence study ===", flush=True)
    print(f"m values: {all_m}, m_ref={m_ref}, draws={n_draws}, cut={cut}", flush=True)
    print(f"Wrote {study_dir / 'probe_convergence_summary.csv'}", flush=True)


def main() -> None:
    args = _parse_args()
    if int(args.k) != 20:
        print(f"note: k={args.k} (requested default idea uses k=20)")
    if not (1 <= int(args.cut) <= int(args.k)):
        raise ValueError(f"cut must satisfy 1 <= cut <= k, got cut={args.cut}, k={args.k}")
    if int(args.n_seeds) < 1:
        raise ValueError("n-seeds must be >= 1.")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    include_z = not bool(args.disable_z_only)
    j_vals = _parse_j_sweep(str(args.j_sweep))
    cut_sweep = _parse_int_list(args.cut_sweep)

    if bool(args.plot_only):
        summary_path = out_dir / "summary.json"
        if summary_path.exists():
            rows = json.loads(summary_path.read_text())
        else:
            csv_path = out_dir / "summary.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"No summary.json or summary.csv found under {out_dir}")
            rows = []
            with csv_path.open(newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    rows.append(dict(row))
        _make_plots(rows, out_dir, include_z=include_z, g_fixed=float(args.g_fixed))
        cut_sweep_csv = out_dir / "cut_sweep_summary.csv"
        if cut_sweep_csv.exists():
            rows_cs = []
            with cut_sweep_csv.open(newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    rows_cs.append(dict(row))
            _make_cut_sweep_plots(rows_cs, out_dir)
        else:
            # Fallback: build cut-sweep rows from existing cut_*/summary.csv files.
            rows_cs = _aggregate_cut_rows_from_subdirs(out_dir)
            if rows_cs:
                _write_summary_csv(out_dir / "cut_sweep_summary.csv", rows_cs)
                (out_dir / "cut_sweep_summary.json").write_text(json.dumps(rows_cs, indent=2))
                _make_cut_sweep_plots(rows_cs, out_dir)
        print(f"plot-only complete: wrote plots to {out_dir}", flush=True)
        return

    if bool(args.probe_convergence_study):
        if args.cut_sweep:
            print("note: --cut-sweep ignored when --probe-convergence-study is set.", flush=True)
        print("=== Exact V-Matrix: probe convergence study ===", flush=True)
        print(
            f"L={args.L}, k={args.k}, dt={args.dt}, cut={args.cut}, n_seeds=1 (forced), "
            f"include_z={include_z}",
            flush=True,
        )
        run_probe_convergence_study(args=args, out_dir=out_dir, include_z=include_z)
        print(f"\nWrote probe convergence outputs under: {out_dir / 'probe_convergence'}", flush=True)
        return

    print("=== Exact V-Matrix Non-Markovianity Benchmark ===", flush=True)
    print(
        f"L={args.L}, k={args.k}, dt={args.dt}, cut={args.cut}, "
        f"N_p={args.n_pasts}, N_f={args.n_futures}, n_seeds={args.n_seeds}, include_z={include_z}",
        flush=True,
    )

    all_rows: list[dict[str, Any]] = []

    # Cut selection:
    # - default: existing single-cut behavior
    # - optional: user-provided cut sweep with invalid cuts skipped
    if cut_sweep:
        valid_cuts = [c for c in cut_sweep if 1 <= int(c) <= int(args.k)]
        skipped = [c for c in cut_sweep if c not in valid_cuts]
        for c in skipped:
            print(f"note: skipping invalid cut={c} (must satisfy 1 <= cut <= k).", flush=True)
        if not valid_cuts:
            raise ValueError("No valid cuts in --cut-sweep.")
        cuts_to_run = sorted(set(valid_cuts))
        # Lighter per-cut suite requested.
        sweep_cases = [
            ("trivial", 0.0, 0.0),
            ("markov_ref", 0.0, float(args.g_fixed)),
            ("nm_j0.6", 0.6, float(args.g_fixed)),
            ("nm_j1.0", 1.0, float(args.g_fixed)),
        ]
    else:
        cuts_to_run = [int(args.cut)]
        # Preserve existing single-cut behavior.
        sweep_cases = [("trivial", 0.0, 0.0), ("markov_ref", 0.0, float(args.g_fixed))]
        for jv in j_vals:
            if abs(float(jv)) <= 1e-15:
                print("note: skipping duplicate sweep point J=0 (already included as markov_ref at g=g_fixed).", flush=True)
                continue
            sweep_cases.append(("sweep", float(jv), float(args.g_fixed)))

    for c in cuts_to_run:
        rows_cut = run_benchmark_for_cut(
            cut=int(c),
            cases=sweep_cases,
            args=args,
            out_dir=out_dir,
            include_z=include_z,
        )
        all_rows.extend(rows_cut)

    # Save global summaries.
    if len(cuts_to_run) == 1 and not cut_sweep:
        (out_dir / "summary.json").write_text(json.dumps(all_rows, indent=2))
        _write_summary_csv(out_dir / "summary.csv", all_rows)
        _make_plots(all_rows, out_dir, include_z=include_z, g_fixed=float(args.g_fixed))
    else:
        (out_dir / "cut_sweep_summary.json").write_text(json.dumps(all_rows, indent=2))
        _write_summary_csv(out_dir / "cut_sweep_summary.csv", all_rows)
        _make_cut_sweep_plots(all_rows, out_dir)

    # Final compact table.
    print("\n=== Summary ===", flush=True)
    header = [
        "case_name",
        "J",
        "g",
        "delta",
        "delta_norm",
        "entropy",
        "rank",
        "participation_ratio",
        "max_row_distance",
        "mean_row_distance",
    ]
    if include_z:
        header.extend(["delta_z", "entropy_z"])
    print(",".join(header), flush=True)
    for r in all_rows:
        row = [
            str(r["case_name"]),
            f"{float(r['J']):.6f}",
            f"{float(r['g']):.6f}",
            f"{float(r['delta']):.6e}",
            f"{float(r['delta_norm']):.6e}",
            f"{float(r['entropy']):.6e}",
            str(int(r["rank"])),
            f"{float(r['participation_ratio']):.6e}",
            f"{float(r['max_row_distance']):.6e}",
            f"{float(r['mean_row_distance']):.6e}",
        ]
        if include_z:
            row.extend([f"{float(r['delta_z']):.6e}", f"{float(r['entropy_z']):.6e}"])
        print(",".join(row), flush=True)

    if len(cuts_to_run) > 1:
        print("\n=== Cut-Grouped Summary ===", flush=True)
        cuts_unique = sorted({int(r["cut"]) for r in all_rows})
        for c in cuts_unique:
            print(f"cut={c}", flush=True)
            sub = [r for r in all_rows if int(r["cut"]) == int(c)]
            for r in sub:
                line = (
                    f"  case={r['case_name']}, J={float(r['J']):.3f}, g={float(r['g']):.3f}, "
                    f"delta_norm={float(r['delta_norm']):.6e}, entropy={float(r['entropy']):.6e}"
                )
                if include_z and "entropy_z" in r:
                    line += f", delta_z={float(r['delta_z']):.6e}, entropy_z={float(r['entropy_z']):.6e}"
                print(line, flush=True)

    print(f"\nWrote results to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
