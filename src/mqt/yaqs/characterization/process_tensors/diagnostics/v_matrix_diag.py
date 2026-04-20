"""Diagnostic-only helpers for V-matrix variants, spectra, and weighted masks (not the main estimator)."""

from __future__ import annotations

from typing import Any

import numpy as np


def center_past_rows(v: np.ndarray) -> np.ndarray:
    """Subtract the mean over past rows (axis 0), matching the current V-matrix centering."""
    vv = np.asarray(v, dtype=np.float64)
    return vv - vv.mean(axis=0, keepdims=True)


def prepare_branch_weights(
    weights_ij: np.ndarray,
    *,
    log_warnings: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate physical branch weights; clamp negatives to 0; record NaN/inf as diagnostic failure.

    Does **not** renormalize weights across entries. Returns cleaned weights safe for ``w**beta``.
    """
    w = np.asarray(weights_ij, dtype=np.float64)
    meta: dict[str, Any] = {
        "weight_data_invalid": False,
        "nan_count": int(np.isnan(w).sum()),
        "posinf_count": int(np.isposinf(w).sum()),
        "neginf_count": int(np.isneginf(w).sum()),
        "negative_count": int((w < 0).sum()),
        "warnings": [],
    }
    if meta["nan_count"] or meta["posinf_count"] or meta["neginf_count"]:
        meta["weight_data_invalid"] = True
        meta["warnings"].append("Non-finite weights detected; replaced with 0 for V construction.")
    if meta["negative_count"]:
        meta["warnings"].append("Negative weights clamped to 0.")
        if log_warnings:
            import warnings

            warnings.warn(
                "prepare_branch_weights: clamped negative cumulative weights to 0.",
                stacklevel=2,
            )
    w_clean = w.copy()
    w_clean[w_clean < 0] = 0.0
    w_clean = np.nan_to_num(w_clean, nan=0.0, posinf=0.0, neginf=0.0)
    return w_clean, meta


def build_weighted_v_matrix(
    rho8_ij: np.ndarray,
    weights_ij: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Construct :math:`V^{(\\beta)}_{i,(j,\\alpha)} = w_{ij}^{\\beta} [\\rho_{ij}]_\\alpha` (flattened).

    ``rho8_ij`` is ``(n_pasts, n_futures, 8)``; ``weights_ij`` is ``(n_pasts, n_futures)``.
    """
    n_p, n_f, d_out = rho8_ij.shape
    w = np.asarray(weights_ij, dtype=np.float64).reshape(n_p, n_f)
    rho = np.asarray(rho8_ij, dtype=np.float64).reshape(n_p, n_f, d_out)
    scale = np.power(w, float(beta))
    scale_exp = np.repeat(scale[:, :, np.newaxis], d_out, axis=2)
    return (rho * scale_exp).reshape(n_p, n_f * d_out)


def build_weighted_v_candidate_triple(
    rho8_ij: np.ndarray,
    weights_ij: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Build unweighted (β=0), ``weighted_sqrt`` (β=0.5), ``weighted_linear`` (β=1) raw and past-centered matrices.

    Returns:
        ``(raw_by_name, centered_by_name, weight_meta)`` where names are
        ``unweighted``, ``weighted_sqrt``, ``weighted_linear``.
    """
    w_clean, wmeta = prepare_branch_weights(weights_ij)
    raw: dict[str, np.ndarray] = {
        "unweighted": build_weighted_v_matrix(rho8_ij, w_clean, 0.0),
        "weighted_sqrt": build_weighted_v_matrix(rho8_ij, w_clean, 0.5),
        "weighted_linear": build_weighted_v_matrix(rho8_ij, w_clean, 1.0),
    }
    centered: dict[str, np.ndarray] = {k: center_past_rows(v) for k, v in raw.items()}
    return raw, centered, wmeta


def build_v_variants(rho8_ij: np.ndarray) -> dict[str, np.ndarray]:
    """Build raw and several centered variants of V from ``rho8_ij`` (n_p, n_f, 8)."""
    n_p, n_f, d_out = rho8_ij.shape
    v_raw = rho8_ij.reshape(n_p, n_f * d_out).astype(np.float64)
    col_mean = v_raw.mean(axis=0, keepdims=True)
    v_centered_past = v_raw - col_mean
    global_mean = float(np.mean(v_raw))
    v_centered_global = v_raw - global_mean
    row_mean = v_raw.mean(axis=1, keepdims=True)
    col_mean2 = v_raw.mean(axis=0, keepdims=True)
    v_centered_rowcol = v_raw - row_mean - col_mean2 + global_mean
    return {
        "V_raw": v_raw,
        "V_centered_past": v_centered_past,
        "V_centered_global": v_centered_global,
        "V_centered_rowcol": v_centered_rowcol,
    }


def _svd_spectrum_entropy_rank(
    mat: np.ndarray,
    *,
    tol_ratio: float = 1e-10,
    discarded_weight_threshold: float | None = 1e-3,
    min_keep: int = 1,
) -> tuple[np.ndarray, float, int, float, float]:
    """SVD diagnostics with optional TDVP-style discarded-weight truncation.

    If ``discarded_weight_threshold`` is not ``None``, the returned singular spectrum is
    truncated by accumulating the squared singular values from smallest to largest and
    stopping before the cumulative discarded weight exceeds the threshold.
    """
    s_full = np.linalg.svd(mat, compute_uv=False).astype(np.float64)
    s = s_full
    discarded_weight = 0.0
    if s.size:
        thr = max(float(discarded_weight_threshold), 0.0)
        keep = int(s.size)
        min_keep_eff = max(1, min(int(min_keep), int(s.size)))
        discard = 0.0
        for idx, sval in enumerate(reversed(s)):
            next_discard = discard + float(sval * sval)
            if next_discard > thr:
                keep = max(int(s.size) - idx, min_keep_eff)
                break
            discard = next_discard
        else:
            # Threshold allows discarding everything; keep a tiny but non-zero core.
            keep = min_keep_eff
        discarded_weight = float(np.sum(np.square(s[keep:])))
        s = s[:keep]

    p = s * s
    p_sum = float(np.sum(p))
    if p_sum <= 0.0:
        return s, 0.0, 0, 0.0, 0.0
    q = np.clip(p / p_sum, 1e-30, 1.0)
    entropy = float(-np.sum(q * np.log(q)))
    pr = float(np.sum(q * q))
    tol = tol_ratio * max(1.0, float(s[0]) if s.size > 0 else 1.0)
    rank = int(np.sum(s > tol))
    if discarded_weight_threshold is None:
        discarded_weight = 0.0
    return s, entropy, rank, pr, discarded_weight


def matrix_diagnostic_metrics(
    mat: np.ndarray,
    name: str = "V",
    *,
    discarded_weight_threshold: float | None = 1e-12,
    min_keep: int = 1,
) -> dict[str, Any]:
    """Frobenius norms, SVD spectrum stats, row/column norm statistics."""
    fro = float(np.linalg.norm(mat, ord="fro"))
    fro_sq = float(fro**2)
    discarded_weight_threshold = 1e-3
    s, ent, rank, pr, discarded_weight = _svd_spectrum_entropy_rank(
        mat,
        discarded_weight_threshold=discarded_weight_threshold,
        min_keep=min_keep,
    )
    row_norms = np.linalg.norm(mat, axis=1)
    col_norms = np.linalg.norm(mat, axis=0)
    return {
        "name": name,
        "frobenius": fro,
        "frobenius_sq": fro_sq,
        "singular_values": s,
        "entropy_sv": ent,
        "rank_tol": rank,
        "participation_ratio": pr,
        "sv_truncated_count": int(s.size),
        "sv_discarded_weight": float(discarded_weight),
        "sv_discarded_weight_threshold": None if discarded_weight_threshold is None else float(discarded_weight_threshold),
        "mean_row_norm": float(np.mean(row_norms)) if row_norms.size else 0.0,
        "std_row_norm": float(np.std(row_norms, ddof=1)) if row_norms.size > 1 else 0.0,
        "mean_col_norm": float(np.mean(col_norms)) if col_norms.size else 0.0,
        "std_col_norm": float(np.std(col_norms, ddof=1)) if col_norms.size > 1 else 0.0,
    }


def pairwise_row_distances(v: np.ndarray) -> np.ndarray:
    n = int(v.shape[0])
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            d[i, j] = float(np.linalg.norm(v[i] - v[j]))
    return d


def row_distance_summary(v: np.ndarray) -> dict[str, float]:
    dmat = pairwise_row_distances(v)
    tri = dmat[np.triu_indices(dmat.shape[0], k=1)]
    return {
        "max_row_distance": float(np.max(tri)) if tri.size else 0.0,
        "mean_row_distance": float(np.mean(tri)) if tri.size else 0.0,
        "median_row_distance": float(np.median(tri)) if tri.size else 0.0,
    }


def apply_weight_mask_and_scales(
    v_raw: np.ndarray,
    weights_ij: np.ndarray,
    *,
    mask_threshold: float,
) -> dict[str, np.ndarray]:
    """Diagnostic-only: masked and weighted variants of the same V layout (weights shape `(n_p, n_f)`)."""
    n_p, n_f = weights_ij.shape
    d = v_raw.shape[1] // n_f
    w_exp = np.repeat(np.asarray(weights_ij, dtype=np.float64), d, axis=1)
    sqrt_w = np.sqrt(np.clip(w_exp, 0.0, None))
    v_w_sqrt = v_raw * sqrt_w
    v_w_lin = v_raw * w_exp
    v_masked_2d = v_raw.copy()
    for i in range(n_p):
        for j in range(n_f):
            if weights_ij[i, j] < mask_threshold:
                v_masked_2d[i, j * d : (j + 1) * d] = 0.0
    return {
        f"masked_lt_{mask_threshold:g}": v_masked_2d,
        "weighted_sqrt": v_w_sqrt,
        "weighted_linear": v_w_lin,
    }


def delta_norm_of_centered(v_raw: np.ndarray, v_c: np.ndarray) -> float:
    fro_v_sq = float(np.linalg.norm(v_raw, ord="fro") ** 2)
    fro_c_sq = float(np.linalg.norm(v_c, ord="fro") ** 2)
    return float(fro_c_sq / fro_v_sq) if fro_v_sq > 0.0 else 0.0


def entry_centered_block_norms(v_centered_past: np.ndarray, n_f: int, d: int = 8) -> np.ndarray:
    """Per-(i,j) Frobenius norm of the centered block (length d) for each future column."""
    n_p = v_centered_past.shape[0]
    out = np.zeros((n_p, n_f), dtype=np.float64)
    for i in range(n_p):
        for j in range(n_f):
            blk = v_centered_past[i, j * d : (j + 1) * d]
            out[i, j] = float(np.linalg.norm(blk))
    return out


def correlation_weight_vs_entry_norm(
    weights: np.ndarray,
    entry_norms: np.ndarray,
) -> float:
    w = weights.reshape(-1)
    e = entry_norms.reshape(-1)
    if w.size < 2 or np.std(w) < 1e-30 or np.std(e) < 1e-30:
        return float("nan")
    return float(np.corrcoef(w, e)[0, 1])


def correlation_terminated_vs_entry_norm(
    terminated: np.ndarray,
    entry_norms: np.ndarray,
) -> float:
    t = terminated.astype(np.float64).reshape(-1)
    e = entry_norms.reshape(-1)
    if t.size < 2 or np.std(t) < 1e-30 or np.std(e) < 1e-30:
        return float("nan")
    return float(np.corrcoef(t, e)[0, 1])


def traces_flat_to_ij_arrays(
    traces: list[dict[str, Any]],
    *,
    n_p: int,
    n_f: int,
) -> dict[str, np.ndarray]:
    """Map rollout traces (same order as :func:`build_all_pairs_grid`) to per-entry fields shaped ``(n_p, n_f)``."""
    n_tot = n_p * n_f
    if len(traces) != n_tot:
        msg = f"Expected {n_tot} traces, got {len(traces)}."
        raise ValueError(msg)
    term = np.zeros((n_p, n_f), dtype=bool)
    brk = np.full((n_p, n_f), np.nan, dtype=np.float64)
    wfin = np.zeros((n_p, n_f), dtype=np.float64)
    nsteps = np.zeros((n_p, n_f), dtype=np.int32)
    min_sp = np.zeros((n_p, n_f), dtype=np.float64)
    max_sp = np.zeros((n_p, n_f), dtype=np.float64)
    mean_sp = np.zeros((n_p, n_f), dtype=np.float64)
    any_skip = np.zeros((n_p, n_f), dtype=bool)
    for ii in range(n_p):
        for jj in range(n_f):
            idx = ii * n_f + jj
            tr = traces[idx]
            term[ii, jj] = bool(tr["terminated_early"])
            bs = tr.get("break_step")
            brk[ii, jj] = float(bs) if bs is not None else np.nan
            wfin[ii, jj] = float(tr["cumulative_weight_final"])
            nsteps[ii, jj] = int(tr["num_steps_completed"])
            min_sp[ii, jj] = float(tr["min_step_prob"])
            max_sp[ii, jj] = float(tr["max_step_prob"])
            mean_sp[ii, jj] = float(tr["mean_step_prob"])
            any_skip[ii, jj] = bool(tr.get("any_prob_skipped_renormalize", False))
    return {
        "terminated_early": term,
        "break_step": brk,
        "cumulative_weight_final": wfin,
        "num_steps_completed": nsteps,
        "min_step_prob": min_sp,
        "max_step_prob": max_sp,
        "mean_step_prob": mean_sp,
        "any_prob_skipped_renormalize": any_skip,
    }


def weight_threshold_fractions(weights: np.ndarray, thresholds: list[float]) -> dict[str, float]:
    """Fraction of entries with cumulative weight below each threshold."""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    out: dict[str, float] = {}
    for thr in thresholds:
        key = f"fraction_weight_lt_{thr:g}"
        out[key] = float(np.mean(w < thr)) if w.size else 0.0
    return out


def summarize_weight_by_index(weights: np.ndarray) -> dict[str, Any]:
    """Mean weight along past rows and future columns."""
    w = np.asarray(weights, dtype=np.float64)
    return {
        "mean_weight_by_past": [float(x) for x in np.mean(w, axis=1)],
        "mean_weight_by_future": [float(x) for x in np.mean(w, axis=0)],
    }


def analyze_weight_scheme_pair(
    v_raw: np.ndarray,
    v_centered: np.ndarray,
    *,
    scheme_name: str,
) -> dict[str, Any]:
    """Full metrics for one weighting scheme: raw + past-centered, row distances, ``delta_norm``."""
    mr = matrix_diagnostic_metrics(v_raw, f"{scheme_name}_raw")
    mc = matrix_diagnostic_metrics(v_centered, f"{scheme_name}_centered_past")
    dn = delta_norm_of_centered(v_raw, v_centered)
    _drop = {"singular_values", "name"}
    return {
        "scheme": scheme_name,
        "metrics_raw": {k: mr[k] for k in mr if k not in _drop},
        "metrics_centered_past": {k: mc[k] for k in mc if k not in _drop},
        "singular_values_raw": mr["singular_values"],
        "singular_values_centered_past": mc["singular_values"],
        "delta_norm": float(dn),
        "row_distance_raw": row_distance_summary(v_raw),
        "row_distance_centered_past": row_distance_summary(v_centered),
        "pairwise_row_distances_centered_past": pairwise_row_distances(v_centered),
    }
