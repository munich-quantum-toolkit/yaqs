"""Data loading and singular-spectrum normalization for the comparison figure."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer
from mqt.yaqs.characterization.memory.backends.tomography.process_tensors import (
    DenseProcessTensor,
    causal_block_operator_entropy,
)
from mqt.yaqs.characterization.memory.operational_memory.full_basis import (
    build_probe_set_from_catalog,
    enumerate_full_probe_catalog,
)
from mqt.yaqs.characterization.memory.operational_memory.response_matrix import (
    assemble_response_matrix,
)
from mqt.yaqs.characterization.memory.operational_memory.run import evaluate_probes_with_weights

from .constants import (
    BETA,
    CUMULATIVE_CAP,
    DT_DEFAULT,
    G_DEFAULT,
    K_DEFAULT,
    L_DEFAULT,
    RANK_RTOL,
    SPECTRUM_CUT,
    SPECTRUM_JS,
    SPECTRUM_MAX_MODES,
    SPECTRUM_MIN_MODES,
    SPECTRUM_SUM_TOL,
    WEIGHT_TOL,
)

QuantityKind = Literal["S_V", "S_MPO"]


@dataclass(frozen=True)
class EntropyPoint:
    """One entropy measurement at a fixed cut and coupling."""

    cut: int
    j: float
    s_v: float
    s_mpo: float


@dataclass(frozen=True)
class SpectrumCurve:
    """Normalized singular-weight spectrum for one quantity at one coupling."""

    quantity: QuantityKind
    cut: int
    j: float
    weights: np.ndarray
    singular_values: np.ndarray
    regime: str


def load_entropy_table(path: Path) -> list[EntropyPoint]:
    """Load panel-(a) entropy rows from a bundled CSV."""
    points: list[EntropyPoint] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            points.append(
                EntropyPoint(
                    cut=int(float(row["cut"])),
                    j=float(row["J"]),
                    s_v=float(row["S_V"]),
                    s_mpo=float(row["S_MPO"]),
                )
            )
    return points


def entropy_series(
    points: list[EntropyPoint],
    *,
    cut: int,
    quantity: QuantityKind,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted ``(J, entropy)`` arrays for one cut and quantity."""
    subset = sorted((p for p in points if p.cut == cut), key=lambda p: p.j)
    j = np.asarray([p.j for p in subset], dtype=np.float64)
    if quantity == "S_V":
        y = np.asarray([p.s_v for p in subset], dtype=np.float64)
    else:
        y = np.asarray([p.s_mpo for p in subset], dtype=np.float64)
    y = np.where(y > 0.0, y, np.nan)
    return j, y


def normalize_singular_weights(singular_values: np.ndarray) -> np.ndarray:
    """Return descending normalized weights ``p_i = s_i^2 / sum_j s_j^2``."""
    s = np.sort(np.asarray(singular_values, dtype=np.float64))[::-1]
    total = float(np.sum(s**2))
    if total <= WEIGHT_TOL:
        return np.zeros_like(s)
    weights = (s**2) / total
    assert abs(float(np.sum(weights)) - 1.0) < SPECTRUM_SUM_TOL or weights.size == 0
    return weights


def resolved_mask(singular_values: np.ndarray, *, rtol: float = RANK_RTOL) -> np.ndarray:
    """Boolean mask for numerically resolved singular values."""
    s = np.asarray(singular_values, dtype=np.float64)
    if s.size == 0 or s[0] <= 0.0:
        return np.zeros_like(s, dtype=bool)
    return s > rtol * s[0]


def meaningful_weights(
    singular_values: np.ndarray,
    weights: np.ndarray,
    *,
    rtol: float = RANK_RTOL,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter to non-positive or numerically unreliable weights; no padding."""
    s = np.asarray(singular_values, dtype=np.float64)
    p = np.asarray(weights, dtype=np.float64)
    keep = resolved_mask(s, rtol=rtol) & (p > WEIGHT_TOL)
    return s[keep], p[keep]


def modes_to_display(weights: np.ndarray) -> int:
    """Modes to span for fair comparison: cumulative cap, min 6, max 8."""
    p = np.asarray(weights, dtype=np.float64)
    if p.size == 0:
        return 0
    cumulative = np.cumsum(p)
    n_cum = int(np.searchsorted(cumulative, CUMULATIVE_CAP) + 1)
    return int(min(p.size, max(min(n_cum, SPECTRUM_MAX_MODES), min(SPECTRUM_MIN_MODES, p.size))))


def _ising_chain(*, length: int, j: float) -> Hamiltonian:
    ham = Hamiltonian.ising(length=length, J=float(j), g=G_DEFAULT)
    ham.ensure_encoded("mpo")
    return ham


def _build_process_tensor(j: float) -> DenseProcessTensor:
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    params = AnalogSimParams(dt=DT_DEFAULT)
    timesteps = [DT_DEFAULT] * (K_DEFAULT + 1)
    ham = _ising_chain(length=L_DEFAULT, j=j)
    pt = mc.build_process_tensor(
        ham,
        params,
        timesteps=timesteps,
        return_type="dense",
        method="exhaustive",
        compress_every=1,
    )
    if not isinstance(pt, DenseProcessTensor):
        msg = f"Expected DenseProcessTensor, got {type(pt).__name__}."
        raise TypeError(msg)
    return pt


def _probe_set(cut: int):
    catalog = enumerate_full_probe_catalog(cut=cut, num_interventions=K_DEFAULT)
    n_p = len(catalog.past_settings)
    n_f = len(catalog.future_settings)
    return build_probe_set_from_catalog(
        catalog,
        np.arange(n_p, dtype=np.int64),
        np.arange(n_f, dtype=np.int64),
    )


def singular_values_response(pt: DenseProcessTensor, *, cut: int) -> np.ndarray:
    """Singular values of the centered response matrix at ``cut``."""
    probe = _probe_set(cut)
    pauli, weights = evaluate_probes_with_weights(pt, probe)
    _raw, response = assemble_response_matrix(pauli, weights, beta=BETA, center=True)
    s = np.linalg.svd(response, compute_uv=False).astype(np.float64)
    return np.sort(s)[::-1]


def singular_values_process_tensor(pt: DenseProcessTensor, *, cut: int) -> np.ndarray:
    """Operator-Schmidt singular values of the causal-block process tensor at ``cut``."""
    cb = causal_block_operator_entropy(
        pt.to_matrix(),
        K_DEFAULT,
        cut,
        rtol=RANK_RTOL,
        weight_tol=WEIGHT_TOL,
    )
    s = np.asarray(cb["singular_values"], dtype=np.float64)
    return np.sort(s)[::-1]


def spectrum_at(
    *,
    quantity: QuantityKind,
    cut: int,
    j: float,
    regime: str,
    pt: DenseProcessTensor | None = None,
) -> SpectrumCurve:
    """Compute one normalized spectrum for panel (b)."""
    tensor = pt if pt is not None else _build_process_tensor(j)
    if quantity == "S_V":
        sv = singular_values_response(tensor, cut=cut)
    else:
        sv = singular_values_process_tensor(tensor, cut=cut)
    weights = normalize_singular_weights(sv)
    return SpectrumCurve(
        quantity=quantity,
        cut=cut,
        j=float(j),
        weights=weights,
        singular_values=sv,
        regime=regime,
    )


def build_panel_b_curves(*, cut: int = SPECTRUM_CUT) -> list[SpectrumCurve]:
    """Assemble panel-(b) spectra: both quantities at each coupling in ``SPECTRUM_JS``."""
    curves: list[SpectrumCurve] = []
    pt_cache: dict[float, DenseProcessTensor] = {}
    for jv in SPECTRUM_JS:
        pt_cache[jv] = _build_process_tensor(jv)
        for quantity in ("S_V", "S_MPO"):
            curves.append(
                spectrum_at(
                    quantity=quantity,
                    cut=cut,
                    j=jv,
                    regime=f"J={jv:g}",
                    pt=pt_cache[jv],
                )
            )
    print(f"Panel (b) spectra at c={cut} for J={', '.join(f'{j:g}' for j in SPECTRUM_JS)}")
    for curve in curves:
        p1 = float(curve.weights[0]) if curve.weights.size else float("nan")
        tail = 1.0 - p1 if curve.weights.size else float("nan")
        print(
            f"  {curve.quantity} J={curve.j:g}, c={cut}: p1={p1:.5f}, 1-p1={tail:.3e}, "
            f"resolved_modes={int(resolved_mask(curve.singular_values).sum())}"
        )
    return curves


def panel_b_mode_span(curves: list[SpectrumCurve]) -> int:
    """Maximum mode index to display fairly across panel-(b) curves."""
    if not curves:
        return 0
    return max(modes_to_display(c.weights) for c in curves)
