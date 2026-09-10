#!/usr/bin/env python3
"""Spectral diagnostics explaining weak- vs strong-coupling agreement of S_V and S_temporal."""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from common import BETA, DT_DEFAULT, G_DEFAULT, L_DEFAULT, characterizer, configure_matplotlib_prl, ising_chain, sim_params, write_csv
from mqt.yaqs.characterization.memory.backends.tomography.process_tensors import (
    DenseProcessTensor,
    MPOProcessTensor,
    causal_block_operator_entropy,
)
from mqt.yaqs.characterization.memory.operational_memory.full_basis import (
    build_probe_set_from_catalog,
    enumerate_full_probe_catalog,
)
from mqt.yaqs.characterization.memory.operational_memory.response_matrix import assemble_response_matrix
from mqt.yaqs.characterization.memory.operational_memory.run import evaluate_probes_with_weights
from pt_cut_reference import _causal_break_error
from pt_response_process_tensor_main import CUTS, K, _clip_entropy

# ---------------------------------------------------------------------------
# Configuration (edit here for robustness grids)
# ---------------------------------------------------------------------------

OUT_DIR_DEFAULT = Path("save/diagnostics_temporal_entropy")

L_REF = L_DEFAULT
K_REF = K
G_REF = G_DEFAULT
DT_REF = DT_DEFAULT
CUTS_REF = CUTS

J_SPECTRA = (1.0, 3.0, 6.0)
J_STEP = 0.2
J_MAX = 6.0

RANK_RTOL = 1e-12
WEIGHT_TOL = 1e-30
CHI_EPSILONS = (1e-2, 1e-3, 1e-4)
CUMULATIVE_THRESHOLDS = (0.90, 0.99, 0.999)

MPO_TOL = 1e-10
MPO_COMPRESS_EVERY = 16

VALID_MIN_EIGENVALUE = -1e-10
VALID_CAUSAL_BREAK = 0.05
VALID_HERMITICITY = 1e-10

TIMESTEP_DT_VALUES = (0.1, 0.05, 0.025)
TIMESTEP_JS = (3.0, 6.0)
TIMESTEP_CUTS = CUTS_REF
# Fixed-k timestep sweep: holding total time fixed would require k=7,15 at smaller dt,
# which is infeasible for exact PT-MPO / dense builds. We keep k=k_ref and vary dt only.

FINITE_L_VALUES = (6, 8)
FINITE_K_VALUES = (3, 4)  # k=4 finite-window check only (direct MPO)
FINITE_JS = (3.0, 6.0)
FINITE_CUTS = CUTS_REF

K_EXHAUSTIVE_MAX = K_REF  # exhaustive dense PT is only feasible at the reference window
K_HARD_MAX = K_REF  # refuse to run above this k

OBJECT_TYPES = (
    "process_tensor",
    "response_centered",
    "response_uncentered",
)


@dataclass
class DiagnosticConfig:
    """Runtime configuration written to ``config.json``."""

    out_dir: str
    L_ref: int = L_REF
    k_ref: int = K_REF
    g: float = G_REF
    dt_ref: float = DT_REF
    cuts: tuple[int, ...] = CUTS_REF
    j_spectra: tuple[float, ...] = J_SPECTRA
    j_step: float = J_STEP
    j_max: float = J_MAX
    rank_rtol: float = RANK_RTOL
    validation: dict[str, float] = field(
        default_factory=lambda: {
            "min_eigenvalue": VALID_MIN_EIGENVALUE,
            "causal_break_error": VALID_CAUSAL_BREAK,
            "hermiticity_error": VALID_HERMITICITY,
        }
    )
    timestep_dt_values: tuple[float, ...] = TIMESTEP_DT_VALUES
    timestep_js: tuple[float, ...] = TIMESTEP_JS
    timestep_cuts: tuple[int, ...] = TIMESTEP_CUTS
    finite_L_values: tuple[int, ...] = FINITE_L_VALUES
    finite_k_values: tuple[int, ...] = FINITE_K_VALUES
    finite_js: tuple[float, ...] = FINITE_JS
    finite_cuts: tuple[int, ...] = FINITE_CUTS
    timestep_mode: str = (
        "fixed_k_varying_T: timestep robustness holds k=k_ref and varies dt; "
        "total evolution time T=(k+1)*dt therefore changes. "
        "A fixed-T refinement grid (k=7,15 at smaller dt) is infeasible."
    )
    mpo_cross_check_note: str = (
        "Dense causal-block SVD is the exact operator-Schmidt spectrum for the "
        "causal channel-block grouping. MPOProcessTensor.compute_schmidt_spectrum(cut) "
        "uses the canonical MPO tensor bond between sites cut-1 and cut; these blockings "
        "coincide only when the MPO site cut matches the causal-block partition."
    )


def _j_grid(*, j_max: float, j_step: float) -> list[float]:
    n = int(round(j_max / j_step))
    return [round(i * j_step, 12) for i in range(n + 1)]


def _normalize_j(jv: float, *, j_step: float = J_STEP) -> float:
    return round(round(float(jv) / j_step) * j_step, 12)


def _entropy_probs(probs: np.ndarray) -> float:
    p = probs[probs > WEIGHT_TOL]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def _effective_rank(probs: np.ndarray) -> float:
    s = _entropy_probs(probs)
    return float(np.exp(s)) if s > 0.0 else 1.0


def _ipr(probs: np.ndarray) -> float:
    p = probs[probs > WEIGHT_TOL]
    if p.size == 0:
        return 1.0
    return float(1.0 / np.sum(p**2))


def _numerical_rank(singular_values: np.ndarray, *, rtol: float) -> int:
    s = np.asarray(singular_values, dtype=np.float64)
    if s.size == 0 or s[0] <= 0.0:
        return 0
    return int(np.sum(s > rtol * s[0]))


def _weights_from_sv(singular_values: np.ndarray) -> np.ndarray:
    s = np.asarray(singular_values, dtype=np.float64)
    total = float(np.sum(s**2))
    if total <= WEIGHT_TOL:
        return np.zeros_like(s)
    return (s**2) / total


def _cumulative_rank(probs: np.ndarray, threshold: float) -> int:
    p = np.asarray(probs, dtype=np.float64)
    if p.size == 0:
        return 0
    csum = np.cumsum(p)
    idx = np.searchsorted(csum, threshold, side="left")
    return int(min(idx + 1, p.size))


def _chi_epsilon(probs: np.ndarray, epsilon: float) -> int:
    """Minimum rank keeping tail squared-Frobenius mass below ``epsilon**2``."""
    p = np.asarray(probs, dtype=np.float64)
    if p.size == 0:
        return 0
    tail = np.cumsum(p[::-1])[::-1]
    target = epsilon**2
    ok = tail <= target + 1e-15
    if not np.any(ok):
        return int(p.size)
    return int(p.size - np.argmax(ok))


def summarize_spectrum(
    singular_values: np.ndarray,
    *,
    rtol: float = RANK_RTOL,
) -> dict[str, float | int]:
    """Return scalar spectral diagnostics from descending singular values."""
    s = np.sort(np.asarray(singular_values, dtype=np.float64))[::-1]
    p = _weights_from_sv(s)
    out: dict[str, float | int] = {
        "entropy": _clip_entropy(_entropy_probs(p)),
        "effective_rank": _effective_rank(p),
        "p1": float(p[0]) if p.size else 0.0,
        "numerical_rank": _numerical_rank(s, rtol=rtol),
        "ipr": _ipr(p),
    }
    rank_keys = {0.90: "r_90", 0.99: "r_99", 0.999: "r_999"}
    for thr, key in rank_keys.items():
        out[key] = _cumulative_rank(p, thr)
    for eps in CHI_EPSILONS:
        key = f"chi_{eps:.0e}".replace("e-0", "e-")
        out[key] = _chi_epsilon(p, eps)
    return out


def _hermiticity_error_upsilon(upsilon: np.ndarray) -> float:
    u = np.asarray(upsilon, dtype=np.complex128)
    num = float(np.linalg.norm(u - u.conj().T, ord="fro"))
    den = max(float(np.linalg.norm(u, ord="fro")), 1e-30)
    return num / den


def _min_eigenvalue_upsilon(upsilon: np.ndarray) -> float:
    u = 0.5 * (np.asarray(upsilon, dtype=np.complex128) + np.asarray(upsilon, dtype=np.complex128).conj().T)
    return float(np.min(np.linalg.eigvalsh(u).real))


def _pt_singular_values(upsilon: np.ndarray, k: int, cut: int) -> np.ndarray:
    cb = causal_block_operator_entropy(upsilon, k, cut, rtol=RANK_RTOL, weight_tol=WEIGHT_TOL)
    s = np.asarray(cb["singular_values"], dtype=np.float64)
    return np.sort(s)[::-1]


def _response_singular_values(
    backend: DenseProcessTensor | MPOProcessTensor,
    probe_full: Any,
    *,
    center: bool,
) -> np.ndarray:
    pauli, weights = evaluate_probes_with_weights(backend, probe_full)
    _raw, response = assemble_response_matrix(pauli, weights, beta=BETA, center=center)
    s = np.linalg.svd(response, compute_uv=False).astype(np.float64)
    return np.sort(s)[::-1]


def _physical_time(k: int, dt: float) -> float:
    return float((k + 1) * dt)


def _assert_feasible(*, k: int, length: int, run_kind: str) -> None:
    k_max = 4 if run_kind == "finite_env" else K_HARD_MAX
    if k > k_max:
        msg = (
            f"Refusing k={k} ({run_kind}): exact PT at k>{k_max} is infeasible. "
            "Edit limits at the top of pt_temporal_entropy_diagnostics.py if verified."
        )
        raise ValueError(msg)
    if length > 10:
        msg = f"Refusing L={length} ({run_kind}): large environments are not configured for this script."
        raise ValueError(msg)


def _build_timesteps(k: int, dt: float) -> list[float]:
    return [float(dt)] * (k + 1)


def _probe_set(cut: int, k: int) -> Any:
    catalog = enumerate_full_probe_catalog(cut=cut, num_interventions=k)
    return build_probe_set_from_catalog(
        catalog,
        np.arange(len(catalog.past_settings), dtype=np.int64),
        np.arange(len(catalog.future_settings), dtype=np.int64),
    )


def _resolve_build_method(*, k: int, run_kind: str, smoke: bool = False) -> str:
    """Pick PT builder.

    * Smoke runs: always exhaustive at ``k=k_ref`` (seconds per point).
    * Production ``main``: exhaustive dense at ``k<=K_EXHAUSTIVE_MAX``.
    * Production ``spectra`` / robustness: exact direct MPO (matches S_temporal).
    """
    if smoke:
        return "exhaustive"
    if run_kind == "main" and k <= K_EXHAUSTIVE_MAX:
        return "exhaustive"
    return "direct_mpo"


def _build_process_tensor(
    *,
    jv: float,
    mc: Any,
    params: Any,
    timesteps: list[float],
    length: int,
    method: str,
) -> tuple[DenseProcessTensor, MPOProcessTensor | None, str]:
    """Return dense PT, optional MPO, and the method actually used."""
    if method == "exhaustive":
        pt = _build_dense_pt(jv=jv, mc=mc, params=params, timesteps=timesteps, length=length)
        return pt, None, "exhaustive"
    pt_mpo = _build_mpo_pt(jv=jv, mc=mc, params=params, timesteps=timesteps, length=length)
    return pt_mpo.to_dense(), pt_mpo, "direct_mpo"


def _point_key(
    *,
    run_kind: str,
    jv: float,
    k: int,
    length: int,
    dt: float,
) -> str:
    return f"{run_kind}|J={jv:g}|k={k}|L={length}|dt={dt:g}"


def _load_progress(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data.get("completed", []))


def _save_progress(path: Path, completed: set[str]) -> None:
    path.write_text(json.dumps({"completed": sorted(completed)}, indent=2), encoding="utf-8")


def _load_scalar_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_validation_rows(path: Path) -> list[dict[str, Any]]:
    return _load_scalar_rows(path)


def _build_dense_pt(
    *,
    jv: float,
    mc: Any,
    params: Any,
    timesteps: list[float],
    length: int,
) -> DenseProcessTensor:
    ham = ising_chain(length=length, j=float(jv), g=G_REF)
    ham.ensure_encoded("mpo")
    return cast(
        DenseProcessTensor,
        mc.build_process_tensor(
            ham,
            params,
            timesteps=timesteps,
            return_type="dense",
            method="exhaustive",
            compress_every=1,
        ),
    )


def _build_mpo_pt(
    *,
    jv: float,
    mc: Any,
    params: Any,
    timesteps: list[float],
    length: int,
) -> MPOProcessTensor:
    ham = ising_chain(length=length, j=float(jv), g=G_REF)
    ham.ensure_encoded("mpo")
    return cast(
        MPOProcessTensor,
        mc.build_process_tensor(
            ham,
            params,
            timesteps=timesteps,
            return_type="mpo",
            method="direct",
            max_bond_dim=None,
            tol=MPO_TOL,
            compress_every=MPO_COMPRESS_EVERY,
        ),
    )


def _validation_row(
    *,
    jv: float,
    cut: int,
    k: int,
    length: int,
    dt: float,
    pt: DenseProcessTensor,
    upsilon: np.ndarray,
    cfg: DiagnosticConfig,
) -> dict[str, float | int | str | bool]:
    trace = float(np.trace(upsilon).real)
    herm = _hermiticity_error_upsilon(upsilon)
    min_eval = _min_eigenvalue_upsilon(upsilon)
    validation_note = ""
    try:
        mi = pt.causal_block_mutual_information(cut, psd_tol=1e-9)
        min_eval = float(mi["min_eigenvalue"])
    except ValueError as exc:
        validation_note = str(exc)
    try:
        cb_err = float(_causal_break_error(pt, k, cut))
    except Exception as exc:  # noqa: BLE001
        cb_err = float("nan")
        validation_note = validation_note or str(exc)
    valid = (
        min_eval >= cfg.validation["min_eigenvalue"]
        and cb_err <= cfg.validation["causal_break_error"]
        and herm <= cfg.validation["hermiticity_error"]
        and not validation_note
    )
    return {
        "J": float(jv),
        "cut": int(cut),
        "k": int(k),
        "L": int(length),
        "dt": float(dt),
        "process_tensor_trace": trace,
        "minimum_eigenvalue": min_eval,
        "causal_break_error": cb_err,
        "hermiticity_error": herm,
        "valid": bool(valid),
        "validation_note": validation_note,
    }


def _scalar_row(
    *,
    jv: float,
    cut: int,
    k: int,
    length: int,
    dt: float,
    object_type: str,
    singular_values: np.ndarray,
    valid: bool,
    run_kind: str = "main",
    build_method: str = "",
    total_time: float = 0.0,
) -> dict[str, float | int | str | bool]:
    stats = summarize_spectrum(singular_values, rtol=RANK_RTOL)
    row: dict[str, float | int | str | bool] = {
        "run_kind": run_kind,
        "build_method": build_method,
        "total_time": float(total_time),
        "J": float(jv),
        "cut": int(cut),
        "k": int(k),
        "L": int(length),
        "dt": float(dt),
        "object_type": object_type,
        "valid": bool(valid),
        "entropy": float(stats["entropy"]),
        "effective_rank": float(stats["effective_rank"]),
        "p1": float(stats["p1"]),
        "numerical_rank": int(stats["numerical_rank"]),
        "ipr": float(stats["ipr"]),
        "r_90": int(stats["r_90"]),
        "r_99": int(stats["r_99"]),
        "r_999": int(stats["r_999"]),
        "chi_1e-2": int(stats["chi_1e-2"]),
        "chi_1e-3": int(stats["chi_1e-3"]),
        "chi_1e-4": int(stats["chi_1e-4"]),
        "n_singular_values": int(singular_values.size),
    }
    return row


def _mpo_cross_check(
    pt_mpo: MPOProcessTensor,
    upsilon: np.ndarray,
    *,
    k: int,
    cut: int,
) -> dict[str, float | int | str]:
    dense_s = _pt_singular_values(upsilon, k, cut)
    dense_p = _weights_from_sv(dense_s)
    try:
        mpo_s = np.sort(np.asarray(pt_mpo.compute_schmidt_spectrum(cut), dtype=np.float64))[::-1]
    except Exception as exc:  # noqa: BLE001
        return {
            "cut": int(cut),
            "k": int(k),
            "mpo_available": False,
            "error": str(exc),
            "max_abs_weight_error": float("nan"),
            "max_rel_singular_value_error": float("nan"),
            "entropy_difference": float("nan"),
            "dense_rank": int(dense_s.size),
            "mpo_rank": 0,
        }
    mpo_p = _weights_from_sv(mpo_s)
    n = min(dense_p.size, mpo_p.size)
    if n == 0:
        return {
            "cut": int(cut),
            "k": int(k),
            "mpo_available": True,
            "max_abs_weight_error": 0.0,
            "max_rel_singular_value_error": 0.0,
            "entropy_difference": 0.0,
            "dense_rank": int(dense_s.size),
            "mpo_rank": int(mpo_s.size),
        }
    max_w = float(np.max(np.abs(dense_p[:n] - mpo_p[:n])))
    rel = np.abs(dense_s[:n] - mpo_s[:n]) / np.maximum(dense_s[:n], 1e-30)
    max_rel = float(np.max(rel)) if rel.size else 0.0
    ent_dense = _entropy_probs(dense_p)
    ent_mpo = _entropy_probs(mpo_p)
    return {
        "cut": int(cut),
        "k": int(k),
        "mpo_available": True,
        "max_abs_weight_error": max_w,
        "max_rel_singular_value_error": max_rel,
        "entropy_difference": float(abs(ent_dense - ent_mpo)),
        "dense_rank": int(dense_s.size),
        "mpo_rank": int(mpo_s.size),
    }


def evaluate_point(
    *,
    jv: float,
    k: int,
    length: int,
    dt: float,
    cuts: tuple[int, ...],
    mc: Any,
    cfg: DiagnosticConfig,
    store_spectra: bool,
    run_kind: str,
    check_mpo: bool = False,
    build_method: str | None = None,
    smoke: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray], list[dict[str, Any]]]:
    """Build one PT and evaluate all cuts."""
    _assert_feasible(k=k, length=length, run_kind=run_kind)
    params = sim_params(dt=dt)
    timesteps = _build_timesteps(k, dt)
    method = build_method or _resolve_build_method(k=k, run_kind=run_kind, smoke=smoke)
    t0 = time.perf_counter()
    print(f"  build J={jv:g} L={length} k={k} dt={dt} T={_physical_time(k, dt):g} ({run_kind}, {method})", flush=True)
    pt, pt_mpo, method_used = _build_process_tensor(
        jv=jv, mc=mc, params=params, timesteps=timesteps, length=length, method=method
    )
    # Never build a second MPO when cross-checking: reuse the direct-MPO object.
    if check_mpo and pt_mpo is None and method_used == "exhaustive":
        print("    note: MPO cross-check skipped (exhaustive build has no MPO object)", flush=True)
    upsilon = pt.to_matrix()
    print(f"    done in {time.perf_counter() - t0:.1f}s ({method_used})", flush=True)
    total_t = _physical_time(k, dt)

    scalar_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    mpo_checks: list[dict[str, Any]] = []
    spectra_arrays: dict[str, np.ndarray] = {}

    for cut in cuts:
        if cut < 1 or cut > k:
            continue
        probe_full = _probe_set(cut, k)
        val = _validation_row(
            jv=jv, cut=cut, k=k, length=length, dt=dt, pt=pt, upsilon=upsilon, cfg=cfg
        )
        validation_rows.append(val)
        valid = bool(val["valid"])

        s_pt = _pt_singular_values(upsilon, k, cut)
        s_cent = _response_singular_values(pt, probe_full, center=True)
        s_unc = _response_singular_values(pt, probe_full, center=False)

        for obj_type, s in (
            ("process_tensor", s_pt),
            ("response_centered", s_cent),
            ("response_uncentered", s_unc),
        ):
            scalar_rows.append(
                _scalar_row(
                    jv=jv,
                    cut=cut,
                    k=k,
                    length=length,
                    dt=dt,
                    object_type=obj_type,
                    singular_values=s,
                    valid=valid,
                    run_kind=run_kind,
                    build_method=method_used,
                    total_time=total_t,
                )
            )

        if store_spectra:
            tag = f"J{jv:g}_cut{cut}_k{k}_L{length}_dt{dt}"
            spectra_arrays[f"{tag}__process_tensor__sv"] = s_pt
            spectra_arrays[f"{tag}__process_tensor__p"] = _weights_from_sv(s_pt)
            spectra_arrays[f"{tag}__response_centered__sv"] = s_cent
            spectra_arrays[f"{tag}__response_centered__p"] = _weights_from_sv(s_cent)
            spectra_arrays[f"{tag}__response_uncentered__sv"] = s_unc
            spectra_arrays[f"{tag}__response_uncentered__p"] = _weights_from_sv(s_unc)

        if check_mpo and pt_mpo is not None:
            mpo_checks.append(_mpo_cross_check(pt_mpo, upsilon, k=k, cut=cut))

    return scalar_rows, validation_rows, spectra_arrays, mpo_checks


def _parse_spectra_key(key: str) -> dict[str, Any]:
    """Parse ``J3_cut2_k3_L6_dt0.1__process_tensor__sv`` metadata."""
    head, object_type, _field = key.split("__")
    chunks = head.split("_")
    return {
        "key": key,
        "J": float(chunks[0][1:]),
        "cut": int(chunks[1][3:]),
        "k": int(chunks[2][1:]),
        "L": int(chunks[3][1:]),
        "dt": float(chunks[4][2:]),
        "object_type": object_type,
    }


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    exists = path.is_file()
    keys = list(rows[0].keys())
    with path.open("a" if exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def run_main_diagnostics(
    cfg: DiagnosticConfig,
    *,
    smoke: bool,
    resume: bool,
    plot_only: bool,
    phase: str = "all",
) -> None:
    out = Path(cfg.out_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    scalar_path = out / "scalar_diagnostics.csv"
    validation_path = out / "validation.csv"
    progress_path = out / "progress.json"
    spectra_npz = out / "spectra_selected.npz"

    all_scalar = _load_scalar_rows(scalar_path) if resume else []
    all_validation = _load_validation_rows(validation_path) if resume else []
    completed = _load_progress(progress_path) if resume else set()
    spectra_arrays: dict[str, np.ndarray] = {}
    if resume and spectra_npz.is_file():
        loaded = np.load(spectra_npz)
        spectra_arrays = {key: loaded[key] for key in loaded.files}
    spectra_index: list[dict[str, Any]] = []
    if resume and (out / "spectra_index.json").is_file():
        spectra_index = json.loads((out / "spectra_index.json").read_text(encoding="utf-8"))
    mpo_checks: list[dict[str, Any]] = []
    if resume and (out / "mpo_cross_check.json").is_file():
        mpo_checks = json.loads((out / "mpo_cross_check.json").read_text(encoding="utf-8"))

    if plot_only:
        if not all_scalar:
            raise SystemExit(f"No scalar rows in {scalar_path}; run diagnostics first.")
        _make_figures(all_scalar, spectra_arrays, cfg, fig_dir=fig_dir)
        summary = _interpretation_text(all_scalar, cfg)
        print(summary, flush=True)
        (out / "interpretation_summary.txt").write_text(summary + "\n", encoding="utf-8")
        return

    if not resume:
        if scalar_path.is_file():
            scalar_path.unlink()
        if validation_path.is_file():
            validation_path.unlink()
        if progress_path.is_file():
            progress_path.unlink()
        all_scalar = []
        all_validation = []
        completed = set()

    mc = characterizer(parallel=False)
    j_sweep = [0.0, 1.0, 2.0] if smoke else _j_grid(j_max=cfg.j_max, j_step=cfg.j_step)
    j_spectra = (1.0,) if smoke else cfg.j_spectra

    total_time = _physical_time(cfg.k_ref, cfg.dt_ref)

    def _record_point(
        *,
        jv: float,
        k: int,
        length: int,
        dt: float,
        cuts: tuple[int, ...],
        store_spectra: bool,
        run_kind: str,
        write_scalar: bool,
        check_mpo: bool = False,
        build_method: str | None = None,
    ) -> None:
        key = _point_key(run_kind=run_kind, jv=jv, k=k, length=length, dt=dt)
        if key in completed:
            print(f"  skip {key} (already done)", flush=True)
            return
        rows, vals, arrays, mpo = evaluate_point(
            jv=jv,
            k=k,
            length=length,
            dt=dt,
            cuts=cuts,
            mc=mc,
            cfg=cfg,
            store_spectra=store_spectra,
            run_kind=run_kind,
            check_mpo=check_mpo,
            build_method=build_method,
            smoke=smoke,
        )
        if write_scalar:
            all_scalar.extend(rows)
            all_validation.extend(vals)
            _append_csv(scalar_path, rows)
            _append_csv(validation_path, vals)
        mpo_checks.extend(mpo)
        for arr_key, arr in arrays.items():
            spectra_arrays[arr_key] = arr
            meta = _parse_spectra_key(arr_key)
            meta["field"] = arr_key.rsplit("__", 1)[-1]
            if meta not in spectra_index:
                spectra_index.append(meta)
        completed.add(key)
        _save_progress(progress_path, completed)

    # Full J sweep (scalar summaries only)
    if phase in {"all", "main"}:
        print("=== Main J sweep (scalar summaries, exhaustive at k=k_ref) ===", flush=True)
        for jv in j_sweep:
            _record_point(
                jv=jv,
                k=cfg.k_ref,
                length=cfg.L_ref,
                dt=cfg.dt_ref,
                cuts=cfg.cuts,
                store_spectra=False,
                run_kind="main",
                write_scalar=True,
            )

    # Selected J spectra: single direct-MPO build (matches S_temporal path)
    if phase in {"all", "spectra"}:
        print("=== Selected-J spectra (direct MPO, one build per J) ===", flush=True)
        for jv in j_spectra:
            _record_point(
                jv=jv,
                k=cfg.k_ref,
                length=cfg.L_ref,
                dt=cfg.dt_ref,
                cuts=cfg.cuts,
                store_spectra=True,
                run_kind="spectra",
                write_scalar=smoke,
                check_mpo=not smoke,
                build_method=None if smoke else "direct_mpo",
            )

    # Timestep / finite robustness (skipped in smoke — production uses direct MPO at k=k_ref)
    if phase in {"all", "robustness"} and not smoke:
        print("=== Timestep robustness (fixed k, varying dt and T) ===", flush=True)
        dt_vals = (0.1,) if smoke else cfg.timestep_dt_values
        ts_js = (3.0,) if smoke else cfg.timestep_js
        for jv in ts_js:
            for dt in dt_vals:
                _record_point(
                    jv=jv,
                    k=cfg.k_ref,
                    length=cfg.L_ref,
                    dt=dt,
                    cuts=cfg.timestep_cuts,
                    store_spectra=False,
                    run_kind="timestep",
                    write_scalar=True,
                    build_method=None if smoke else "direct_mpo",
                )

        print("=== Finite environment (L and k grid at k<=4) ===", flush=True)
        l_vals = (6,) if smoke else cfg.finite_L_values
        k_vals = (3,) if smoke else cfg.finite_k_values
        fe_js = (3.0,) if smoke else cfg.finite_js
        for jv in fe_js:
            for length in l_vals:
                for k in k_vals:
                    _record_point(
                        jv=jv,
                        k=k,
                        length=length,
                        dt=cfg.dt_ref,
                        cuts=tuple(c for c in cfg.finite_cuts if c <= k),
                        store_spectra=False,
                        run_kind="finite_env",
                        write_scalar=True,
                        build_method="direct_mpo",
                    )

    if phase in {"all", "main", "spectra", "robustness"}:
        np.savez(out / "spectra_selected.npz", **spectra_arrays)
        (out / "spectra_index.json").write_text(json.dumps(spectra_index, indent=2), encoding="utf-8")
        (out / "mpo_cross_check.json").write_text(json.dumps(mpo_checks, indent=2), encoding="utf-8")

    if phase in {"all", "plot"}:
        _make_figures(all_scalar, spectra_arrays, cfg, fig_dir=fig_dir)
        summary = _interpretation_text(all_scalar, cfg)
        print(summary, flush=True)
        (out / "interpretation_summary.txt").write_text(summary + "\n", encoding="utf-8")


def _valid_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if bool(r.get("valid", True)) and r.get("run_kind") == "main"]


def _make_figures(
    scalar_rows: list[dict[str, Any]],
    spectra_arrays: dict[str, np.ndarray],
    cfg: DiagnosticConfig,
    *,
    fig_dir: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    configure_matplotlib_prl()
    plt.rcParams.update({"font.size": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42})

    colors = {
        "process_tensor": "#0072B2",
        "response_centered": "#D55E00",
        "response_uncentered": "#009E73",
    }
    markers = {"process_tensor": "o", "response_centered": "s", "response_uncentered": "^"}

    for cut in cfg.cuts:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for jv in cfg.j_spectra:
            for obj in OBJECT_TYPES:
                key = f"J{jv:g}_cut{cut}_k{cfg.k_ref}_L{cfg.L_ref}_dt{cfg.dt_ref}__{obj}__p"
                if key not in spectra_arrays:
                    continue
                p = spectra_arrays[key]
                idx = np.arange(1, p.size + 1)
                ax.plot(idx, np.maximum(p, 1e-16), marker=markers[obj], ms=3, lw=0.8, label=f"J={jv:g} {obj}")
        ax.set_yscale("log")
        ax.set_xlabel("Mode index $i$")
        ax.set_ylabel(r"Normalized weight $p_i$")
        ax.set_title(rf"Cut $c={cut}$")
        ax.legend(fontsize=6.5, ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / f"spectra_weights_cut_{cut}.pdf")
        fig.savefig(fig_dir / f"spectra_weights_cut_{cut}.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for jv in cfg.j_spectra:
            for obj in OBJECT_TYPES:
                key = f"J{jv:g}_cut{cut}_k{cfg.k_ref}_L{cfg.L_ref}_dt{cfg.dt_ref}__{obj}__p"
                if key not in spectra_arrays:
                    continue
                p = spectra_arrays[key]
                csum = np.cumsum(p)
                ax.plot(np.arange(1, p.size + 1), csum, marker=markers[obj], ms=3, lw=0.8, label=f"J={jv:g} {obj}")
        ax.set_xlabel("Rank $r$")
        ax.set_ylabel(r"Cumulative weight $C(r)$")
        ax.set_title(rf"Cut $c={cut}$")
        ax.legend(fontsize=6.5, ncol=2, frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / f"cumulative_weights_cut_{cut}.pdf")
        fig.savefig(fig_dir / f"cumulative_weights_cut_{cut}.png")
        plt.close(fig)

    main_rows = _valid_rows(scalar_rows)

    def _plot_vs_j(metric: str, fname: str, ylabel: str, logy: bool = False) -> None:
        fig, axes = plt.subplots(1, len(cfg.cuts), figsize=(3.2 * len(cfg.cuts), 2.8), sharey=False)
        if len(cfg.cuts) == 1:
            axes = [axes]
        for ax, cut in zip(axes, cfg.cuts, strict=True):
            for obj in OBJECT_TYPES:
                sub = sorted(
                    [r for r in main_rows if int(r["cut"]) == cut and r["object_type"] == obj],
                    key=lambda r: float(r["J"]),
                )
                if not sub:
                    continue
                j = np.array([float(r["J"]) for r in sub if float(r["J"]) > 0])
                y = np.array([float(r[metric]) for r in sub if float(r["J"]) > 0])
                ax.plot(j, y, marker=markers[obj], color=colors[obj], lw=1.2, ms=3.5, label=obj)
            ax.set_xlabel(r"Coupling $J$")
            ax.set_ylabel(ylabel)
            ax.set_title(rf"$c={cut}$")
            if logy:
                ax.set_yscale("log")
        handles = [
            Line2D([0], [0], color=colors[o], marker=markers[o], lw=1.2, ms=4, label=o) for o in OBJECT_TYPES
        ]
        fig.legend(handles=handles, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))
        fig.tight_layout()
        fig.savefig(fig_dir / fname)
        plt.close(fig)

    _plot_vs_j("entropy", "entropy_vs_J.pdf", r"Entropy $S$", logy=True)
    _plot_vs_j("p1", "dominant_weight_vs_J.pdf", r"Dominant weight $p_1$", logy=True)
    _plot_vs_j("effective_rank", "effective_rank_vs_J.pdf", r"Effective rank $\exp(S)$", logy=True)
    _plot_vs_j("ipr", "ipr_vs_J.pdf", r"Inverse participation ratio $R_2$", logy=True)
    _plot_vs_j("r_99", "r99_vs_J.pdf", r"Rank $r_{99}$")
    _plot_vs_j("chi_1e-3", "chi_1e-3_vs_J.pdf", r"Truncation rank $\chi_{10^{-3}}$")

    fig, axes = plt.subplots(1, len(cfg.cuts), figsize=(3.2 * len(cfg.cuts), 2.8))
    if len(cfg.cuts) == 1:
        axes = [axes]
    for ax, cut in zip(axes, cfg.cuts, strict=True):
        for obj, ls in (("response_centered", "-"), ("response_uncentered", "--")):
            sub = sorted(
                [r for r in main_rows if int(r["cut"]) == cut and r["object_type"] == obj],
                key=lambda r: float(r["J"]),
            )
            j = np.array([float(r["J"]) for r in sub if float(r["J"]) > 0])
            y = np.array([float(r["entropy"]) for r in sub if float(r["J"]) > 0])
            ax.plot(j, y, ls=ls, color=colors[obj], marker=markers[obj], lw=1.2, ms=3.5, label=obj)
        ax.set_xlabel(r"Coupling $J$")
        ax.set_ylabel(r"Entropy $S$")
        ax.set_title(rf"$c={cut}$")
        ax.set_yscale("log")
    fig.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    fig.savefig(fig_dir / "centered_vs_uncentered.pdf")
    plt.close(fig)

    ts_rows = [r for r in scalar_rows if r.get("run_kind") == "timestep" and bool(r.get("valid", True))]
    if ts_rows:
        for jv in cfg.timestep_js:
            fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
            for ax, metric, ylab in ((axes[0], "entropy", r"$S$"), (axes[1], "p1", r"$p_1$")):
                for obj in OBJECT_TYPES:
                    sub = sorted(
                        [
                            r
                            for r in ts_rows
                            if r["object_type"] == obj and abs(float(r["J"]) - jv) < 1e-9
                        ],
                        key=lambda r: float(r["dt"]),
                    )
                    if not sub:
                        continue
                    x = [float(r["dt"]) for r in sub]
                    y = [float(r[metric]) for r in sub]
                    ax.plot(x, y, marker=markers[obj], color=colors[obj], lw=1.2, ms=4, label=obj)
                ax.set_xlabel(r"Timestep $\Delta t$")
                ax.set_ylabel(ylab)
                ax.invert_xaxis()
            fig.suptitle(rf"Timestep robustness ($J={jv:g}$, fixed $k={cfg.k_ref}$, $T=(k+1)\Delta t$)", y=1.02)
            fig.legend(frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
            fig.tight_layout()
            fig.savefig(fig_dir / f"timestep_robustness_J{jv:g}.pdf")
            plt.close(fig)
        # Combined alias for spec output name
        import shutil

        j_first = cfg.timestep_js[0]
        src = fig_dir / f"timestep_robustness_J{j_first:g}.pdf"
        dst = fig_dir / "timestep_robustness.pdf"
        if src.is_file():
            shutil.copy(src, dst)

    fe_rows = [r for r in scalar_rows if r.get("run_kind") == "finite_env" and bool(r.get("valid", True))]
    if fe_rows:
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        labels: list[str] = []
        for obj in OBJECT_TYPES:
            sub = sorted(
                [
                    r
                    for r in fe_rows
                    if r["object_type"] == obj and abs(float(r["J"]) - 3.0) < 1e-9 and int(r["cut"]) == 2
                ],
                key=lambda r: (int(r["L"]), int(r["k"])),
            )
            if not sub:
                continue
            labels = [f"L={int(r['L'])},k={int(r['k'])}" for r in sub]
            y = [float(r["entropy"]) for r in sub]
            ax.plot(range(len(labels)), y, marker=markers[obj], color=colors[obj], lw=1.2, ms=4, label=obj)
        if labels:
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=20, ha="right")
            ax.set_ylabel(r"Entropy $S$")
            ax.set_title(r"Finite environment ($J=3$, $c=2$)")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(fig_dir / "finite_environment_robustness.pdf")
        plt.close(fig)


def _interpretation_text(scalar_rows: list[dict[str, Any]], cfg: DiagnosticConfig) -> str:
    lines: list[str] = []
    main_valid = _valid_rows(scalar_rows)
    lines.append("=== Interpretation table (valid main-run points) ===")
    header = f"{'c':>2} {'J':>4} {'object':<20} {'S':>10} {'exp(S)':>10} {'p1':>10} {'r99':>6}"
    lines.append(header)
    lines.append("-" * len(header))
    for cut in cfg.cuts:
        for jv in cfg.j_spectra:
            for obj in OBJECT_TYPES:
                hit = next(
                    (
                        r
                        for r in main_valid
                        if int(r["cut"]) == cut
                        and abs(float(r["J"]) - jv) < 1e-9
                        and r["object_type"] == obj
                    ),
                    None,
                )
                if hit is None:
                    continue
                lines.append(
                    f"{cut:2d} {jv:4.1f} {obj:<20} "
                    f"{float(hit['entropy']):10.4e} {float(hit['effective_rank']):10.4e} "
                    f"{float(hit['p1']):10.4e} {int(hit['r_99']):6d}"
                )

    def _delta(metric: str, cut: int, j_weak: float = 1.0, j_strong: float = 6.0, obj: str = "process_tensor"):
        def _get(jv: float) -> float:
            r = next(
                (
                    x
                    for x in main_valid
                    if int(x["cut"]) == cut and abs(float(x["J"]) - jv) < 1e-9 and x["object_type"] == obj
                ),
                None,
            )
            return float(r[metric]) if r else float("nan")

        return _get(j_strong) / max(_get(j_weak), 1e-30)

    lines.append("\n=== Strong-coupling divergence mechanisms (J=1 vs J=6, c=2) ===")
    p1_pt_ratio = _delta("p1", 2, obj="process_tensor")
    rank_cent_ratio = _delta("effective_rank", 2, obj="response_centered")
    s_cent_j6 = next(
        (
            float(r["entropy"])
            for r in main_valid
            if int(r["cut"]) == 2 and abs(float(r["J"]) - 6.0) < 1e-9 and r["object_type"] == "response_centered"
        ),
        float("nan"),
    )
    s_unc_j6 = next(
        (
            float(r["entropy"])
            for r in main_valid
            if int(r["cut"]) == 2 and abs(float(r["J"]) - 6.0) < 1e-9 and r["object_type"] == "response_uncentered"
        ),
        float("nan"),
    )
    centering_effect = s_cent_j6 / max(s_unc_j6, 1e-30)

    flags: list[str] = []
    if p1_pt_ratio > 1.5:
        flags.append("A: PT spectrum becomes more dominant-mode weighted at strong coupling")
    if rank_cent_ratio > 1.5:
        flags.append("B: Centered response spectrum broadens (exp(S) increases)")
    elif rank_cent_ratio < 0.9:
        flags.append("B: Centered response effective rank decreases at strong coupling")
    if abs(centering_effect - 1.0) > 0.15:
        flags.append(
            f"C: Centering materially changes response entropy (S_cent/S_unc={centering_effect:.2f} at J=6)"
        )
    ts_rows = [r for r in scalar_rows if r.get("run_kind") == "timestep" and int(r["cut"]) == 2]
    if ts_rows:
        by_dt: dict[float, float] = {}
        for dt in sorted({float(r["dt"]) for r in ts_rows}):
            r = next(
                (
                    x
                    for x in ts_rows
                    if abs(float(x["dt"]) - dt) < 1e-12 and x["object_type"] == "response_centered"
                ),
                None,
            )
            if r:
                by_dt[dt] = float(r["p1"])
        if len(by_dt) >= 2 and (max(by_dt.values()) - min(by_dt.values())) / max(min(by_dt.values()), 1e-30) > 0.1:
            flags.append("D: Timestep refinement shifts centered-response p1 by >10%")
    fe_rows = [r for r in scalar_rows if r.get("run_kind") == "finite_env" and int(r["cut"]) == 2]
    if fe_rows:
        ent = [
            float(r["entropy"])
            for r in fe_rows
            if r["object_type"] == "response_centered" and abs(float(r["J"]) - 3.0) < 1e-9
        ]
        if ent and (max(ent) - min(ent)) / max(min(ent), 1e-30) > 0.1:
            flags.append("E: Finite L/k grid shifts centered-response entropy by >10% at J=3")
    if not flags:
        flags.append("No single mechanism dominates; inspect spectra plots and CSV rows.")
    for line in flags:
        lines.append(f"  - {line}")

    lines.append(
        textwrap.dedent(
            f"""
            Catalog ceiling: log(4096) = {np.log(4096):.4f} nats for S_V^full at k={cfg.k_ref}.
            {cfg.mpo_cross_check_note}
            """
        ).strip()
    )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--smoke", action="store_true", help="Reduced grids for quick testing.")
    p.add_argument("--resume", action="store_true", help="Skip completed points recorded in progress.json.")
    p.add_argument("--plot-only", action="store_true", help="Regenerate figures/summary from saved CSV/NPZ.")
    p.add_argument(
        "--phase",
        choices=("all", "main", "spectra", "robustness", "plot"),
        default="all",
        help="Run a subset: main J sweep, selected spectra, robustness grids, or plot-only.",
    )
    p.add_argument("--j-max", type=float, default=J_MAX)
    p.add_argument("--j-step", type=float, default=J_STEP)
    args = p.parse_args()

    cfg = DiagnosticConfig(
        out_dir=str(args.out_dir.resolve()),
        j_max=float(args.j_max),
        j_step=float(args.j_step),
    )
    phase = "plot" if args.plot_only else str(args.phase)
    run_main_diagnostics(
        cfg,
        smoke=bool(args.smoke),
        resume=bool(args.resume),
        plot_only=bool(args.plot_only),
        phase=phase,
    )


if __name__ == "__main__":
    main()
