# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Data generation and loading for the updated Figure 4 comparison."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer
from mqt.yaqs.characterization.memory.backends.tomography.process_tensors import DenseProcessTensor
from mqt.yaqs.characterization.memory.operational_memory.response_matrix import (
    assemble_response_matrix,
    compute_spectrum,
)
from mqt.yaqs.characterization.memory.operational_memory.run import evaluate_probes_with_weights

from .constants import (
    CUMULATIVE_CAP,
    DT_DEFAULT,
    G_DEFAULT,
    K_DEFAULT,
    L_DEFAULT,
    RANK_RTOL,
    SPECTRUM_CUT,
    SPECTRUM_DISCARDED_WEIGHT_THRESHOLD,
    SPECTRUM_JS,
    SPECTRUM_MAX_MODES,
    SPECTRUM_MIN_MODES,
    SPECTRUM_SUM_TOL,
    WEIGHT_TOL,
)
from .full_basis import build_probe_set_from_catalog, enumerate_full_probe_catalog

if TYPE_CHECKING:
    from pathlib import Path

    from mqt.yaqs.characterization.memory.operational_memory.samples import ProbeSet

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
    """Load panel-(a) entropy rows from CSV."""
    points: list[EntropyPoint] = []
    with path.open(newline="", encoding="utf-8") as handle:
        points.extend(
            EntropyPoint(
                cut=int(float(row["cut"])),
                j=float(row["J"]),
                s_v=float(row["S_V"]),
                s_mpo=float(row["S_MPO"]),
            )
            for row in csv.DictReader(handle)
        )
    return points


def write_entropy_table(path: Path, points: list[EntropyPoint]) -> None:
    """Write panel-(a) entropy rows using deterministic LF line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("cut", "J", "S_V", "S_MPO"), lineterminator="\n")
        writer.writeheader()
        for point in sorted(points, key=lambda item: (item.cut, item.j)):
            writer.writerow({
                "cut": point.cut,
                "J": f"{point.j:.12g}",
                "S_V": f"{point.s_v:.17g}",
                "S_MPO": f"{point.s_mpo:.17g}",
            })


def entropy_series(
    points: list[EntropyPoint],
    *,
    cut: int,
    quantity: QuantityKind,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted ``(J, entropy)`` arrays for one cut and quantity."""
    subset = sorted((point for point in points if point.cut == cut), key=lambda point: point.j)
    j_values = np.asarray([point.j for point in subset], dtype=np.float64)
    if quantity == "S_V":
        entropy = np.asarray([point.s_v for point in subset], dtype=np.float64)
    else:
        entropy = np.asarray([point.s_mpo for point in subset], dtype=np.float64)
    return j_values, np.where(entropy > 0.0, entropy, np.nan)


def normalize_singular_weights(singular_values: np.ndarray) -> np.ndarray:
    """Return descending normalized weights ``p_i = s_i^2 / sum_j s_j^2``."""
    singular = np.sort(np.asarray(singular_values, dtype=np.float64))[::-1]
    total = float(np.sum(singular**2))
    if total <= WEIGHT_TOL:
        return np.zeros_like(singular)
    weights = (singular**2) / total
    assert abs(float(np.sum(weights)) - 1.0) < SPECTRUM_SUM_TOL or weights.size == 0
    return weights


def resolved_mask(singular_values: np.ndarray, *, rtol: float = RANK_RTOL) -> np.ndarray:
    """Return a mask for numerically resolved singular values."""
    singular = np.asarray(singular_values, dtype=np.float64)
    if singular.size == 0 or singular[0] <= 0.0:
        return np.zeros_like(singular, dtype=bool)
    return singular > rtol * singular[0]


def modes_to_display(weights: np.ndarray) -> int:
    """Choose a common displayed mode span using the original Figure 4 rule."""
    probabilities = np.asarray(weights, dtype=np.float64)
    if probabilities.size == 0:
        return 0
    cumulative = np.cumsum(probabilities)
    n_cumulative = int(np.searchsorted(cumulative, CUMULATIVE_CAP) + 1)
    return int(
        min(
            probabilities.size,
            max(
                min(n_cumulative, SPECTRUM_MAX_MODES),
                min(SPECTRUM_MIN_MODES, probabilities.size),
            ),
        )
    )


def build_process_tensor(j_value: float) -> DenseProcessTensor:
    """Reconstruct the exact dense ``k=3`` process tensor at one coupling."""
    characterizer = MemoryCharacterizer(parallel=False, show_progress=False)
    process = characterizer.build_process_tensor(
        Hamiltonian.ising(length=L_DEFAULT, J=float(j_value), g=G_DEFAULT),
        AnalogSimParams(dt=DT_DEFAULT),
        timesteps=[DT_DEFAULT] * (K_DEFAULT + 1),
        return_type="dense",
        check=True,
        parallel=False,
    )
    if not isinstance(process, DenseProcessTensor):
        message = f"Expected DenseProcessTensor, got {type(process).__name__}."
        raise TypeError(message)
    return process


def build_full_probe_set(cut: int) -> ProbeSet:
    """Build the original exhaustive tetrahedral probe grid for one cut."""
    catalog = enumerate_full_probe_catalog(cut=cut, num_interventions=K_DEFAULT)
    return build_probe_set_from_catalog(
        catalog,
        np.arange(len(catalog.past_settings), dtype=np.int64),
        np.arange(len(catalog.future_settings), dtype=np.int64),
    )


def response_spectrum(
    process: DenseProcessTensor,
    probe_set: ProbeSet,
) -> tuple[np.ndarray, float]:
    """Return the updated raw-IXYZ response spectrum and entropy."""
    pauli_ixyz, complete_weights = evaluate_probes_with_weights(process, probe_set)
    response = assemble_response_matrix(pauli_ixyz, complete_weights)
    analysis = compute_spectrum(
        response,
        discarded_weight_threshold=SPECTRUM_DISCARDED_WEIGHT_THRESHOLD,
    )
    singular_values = np.asarray(analysis["singular_values_full"], dtype=np.float64)
    return np.sort(singular_values)[::-1], float(analysis["entropy"])


def process_tensor_spectrum(
    process: DenseProcessTensor,
    *,
    cut: int,
) -> tuple[np.ndarray, float]:
    """Return the causal-block operator-Schmidt spectrum and its entropy."""
    analysis = process.compute_temporal_entropy(cut, rtol=RANK_RTOL, weight_tol=WEIGHT_TOL)
    singular_values = np.asarray(analysis["singular_values"], dtype=np.float64)
    return np.sort(singular_values)[::-1], float(analysis["entropy"])


def spectrum_curve(
    *,
    quantity: QuantityKind,
    cut: int,
    j_value: float,
    singular_values: np.ndarray,
) -> SpectrumCurve:
    """Build a plot-ready spectrum record."""
    return SpectrumCurve(
        quantity=quantity,
        cut=cut,
        j=float(j_value),
        weights=normalize_singular_weights(singular_values),
        singular_values=np.asarray(singular_values, dtype=np.float64),
        regime=f"J={j_value:g}",
    )


def panel_b_mode_span(curves: list[SpectrumCurve]) -> int:
    """Return the common number of displayed modes for panels (b) and (c)."""
    if not curves:
        return 0
    return max(modes_to_display(curve.weights) for curve in curves)


def _j_tag(j_value: float) -> str:
    return f"{j_value:g}".replace("-", "m").replace(".", "p")


def save_spectrum_curves(path: Path, curves: list[SpectrumCurve]) -> None:
    """Store full selected spectra in a compact NPZ archive."""
    payload: dict[str, np.ndarray] = {
        "j_values": np.asarray(SPECTRUM_JS, dtype=np.float64),
        "cut": np.asarray([SPECTRUM_CUT], dtype=np.int64),
    }
    for curve in curves:
        quantity_tag = "response" if curve.quantity == "S_V" else "mpo"
        payload[f"{quantity_tag}_J_{_j_tag(curve.j)}"] = curve.singular_values
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_spectrum_curves(path: Path) -> list[SpectrumCurve]:
    """Load selected response and PT-MPO spectra from NPZ."""
    curves: list[SpectrumCurve] = []
    with np.load(path) as archive:
        cut = int(np.asarray(archive["cut"]).reshape(-1)[0])
        j_values = np.asarray(archive["j_values"], dtype=np.float64)
        for j_value in j_values:
            for quantity, quantity_tag in (("S_V", "response"), ("S_MPO", "mpo")):
                singular_values = np.asarray(
                    archive[f"{quantity_tag}_J_{_j_tag(float(j_value))}"],
                    dtype=np.float64,
                )
                curves.append(
                    spectrum_curve(
                        quantity=quantity,
                        cut=cut,
                        j_value=float(j_value),
                        singular_values=singular_values,
                    )
                )
    return curves
