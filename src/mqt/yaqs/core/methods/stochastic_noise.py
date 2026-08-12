# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Application and sampling of stochastic gate-local digital noise."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe

from ..data_structures.stochastic_noise_model import XBasisDissipativeNoiseModel, XYZPauliNoiseModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from ..data_structures.mps import MPS
    from ..data_structures.stochastic_noise_model import StochasticNoiseModel

_PAULI_MATRICES: dict[str, NDArray[np.complex128]] = {
    "X": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "Z": np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}


def _validate_gate_sites(sites: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(site) for site in sites)
    if len(normalized) not in {1, 2}:
        msg = f"Stochastic noise models support one- and two-qubit gates, got sites {list(normalized)}."
        raise ValueError(msg)
    if len(set(normalized)) != len(normalized):
        msg = f"Gate sites must be distinct, got {list(normalized)}."
        raise ValueError(msg)
    return normalized


def sample_xyz_pauli_event(
    model: XYZPauliNoiseModel,
    rng: np.random.Generator,
) -> str | None:
    """Sample one mutually exclusive I/X/Y/Z event for a touched qubit.

    The first draw decides between identity and an error. If an error occurs,
    a second draw selects uniformly from X, Y, and Z.

    Args:
        model: XYZ Pauli model.
        rng: Trajectory-local random generator.

    Returns:
        The selected Pauli label, or ``None`` for identity.
    """
    if model.is_noiseless or rng.random() >= model.p:
        return None
    axes = ("X", "Y", "Z")
    return axes[int(rng.integers(len(axes)))]


def _apply_pauli_event(state: MPS, axis: str, site: int) -> None:
    state.tensors[site] = oe.contract("ab, bcd->acd", _PAULI_MATRICES[axis], state.tensors[site])


def _apply_x_basis_dissipative_channel(
    state: MPS,
    model: XBasisDissipativeNoiseModel,
    site: int,
    rng: np.random.Generator,
) -> None:
    """Sample and apply one state-dependent Kraus branch at ``site``.

    Args:
        state: MPS trajectory, updated in place.
        model: X-basis dissipative model.
        site: Target site.
        rng: Trajectory-local random generator.

    Raises:
        ValueError: If the Kraus weights are invalid or cannot be normalized.
    """
    if state.orthogonality_center is None:
        state.set_canonical_form(site, decomposition="QR")
    elif state.orthogonality_center != site:
        state.shift_center_to(site, decomposition="QR")

    tensor = state.tensors[site]
    candidates = [oe.contract("ab, bcd->acd", kraus, tensor) for kraus in model.kraus_operators()]
    weights = np.asarray([float(np.vdot(candidate, candidate).real) for candidate in candidates], dtype=np.float64)
    incoming_norm = float(np.vdot(tensor, tensor).real)
    total = float(math.fsum(weights))
    if not math.isfinite(total) or total <= np.finfo(np.float64).tiny:
        msg = "Both dissipative Kraus branches have zero or non-finite total weight."
        raise ValueError(msg)
    if not math.isclose(total, incoming_norm, rel_tol=1e-10, abs_tol=1e-10):
        msg = f"Dissipative Kraus weights {total} do not reproduce the incoming squared norm {incoming_norm}."
        raise ValueError(msg)

    probabilities = weights / total
    uniform = float(rng.random())
    if not probabilities[0]:
        branch = 1
    elif not probabilities[1]:
        branch = 0
    else:
        branch = 0 if uniform < probabilities[0] else 1
    selected_weight = float(weights[branch])
    if selected_weight <= np.finfo(np.float64).tiny:
        msg = "Selected dissipative Kraus branch is not numerically normalizable."
        raise ValueError(msg)
    state.tensors[site] = candidates[branch] / math.sqrt(selected_weight)


def apply_stochastic_noise(
    state: MPS,
    model: StochasticNoiseModel,
    sites: Sequence[int],
    rng: np.random.Generator,
) -> None:
    """Apply a stochastic gate-local noise model after one ideal gate.

    Args:
        state: MPS trajectory, updated in place.
        model: Supported stochastic gate-local noise model.
        sites: Touched qubits in gate order.
        rng: Trajectory-local random generator.
    """
    gate_sites = _validate_gate_sites(sites)
    if model.is_noiseless:
        return
    if isinstance(model, XYZPauliNoiseModel):
        for site in gate_sites:
            event = sample_xyz_pauli_event(model, rng)
            if event is not None:
                _apply_pauli_event(state, event, site)
        return
    for site in gate_sites:
        _apply_x_basis_dissipative_channel(state, model, site, rng)
