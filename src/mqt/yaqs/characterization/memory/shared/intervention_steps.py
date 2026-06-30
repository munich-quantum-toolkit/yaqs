# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared parsing and application of intervention probe steps."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from mqt.yaqs.characterization.memory.backends.surrogates.utils import InterventionMap
from mqt.yaqs.characterization.memory.shared.utils import (
    _apply_backend_unitary_site_zero,
    _apply_cut_preparation_step,
    _reprepare_backend_state_forced,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.core.data_structures.mps import MPS

AnyInterventionStep = dict[str, Any] | tuple[Any, Any]


class _ParsedStep(NamedTuple):
    """Parsed fields for one intervention probe step."""

    kind: str
    unitary: NDArray[np.complex128] | None = None
    psi_meas: NDArray[np.complex128] | None = None
    psi_reset: NDArray[np.complex128] | None = None
    psi_prep: NDArray[np.complex128] | None = None


def _reshape_ket(psi: Any) -> NDArray[np.complex128]:
    """Reshape an input ket to length 2.

    Args:
        psi: Ket-like array.

    Returns:
        Complex ket of shape ``(2,)``.
    """
    return np.asarray(psi, dtype=np.complex128).reshape(2)


def _parse_intervention_step(step: AnyInterventionStep) -> _ParsedStep:
    """Parse a probe-grid step dict or measure/prepare pair.

    Args:
        step: Structured dict step or ``(psi_meas, psi_prep)`` pair.

    Returns:
        Parsed step fields.

    Raises:
        ValueError: If the step type is unsupported.
    """
    if isinstance(step, dict):
        step_type = str(step.get("type", "")).lower()
        if step_type == "unitary":
            return _ParsedStep(
                "unitary",
                unitary=np.asarray(step["U"], dtype=np.complex128).reshape(2, 2),
            )
        if step_type == "cut_measurement":
            psi_meas = _reshape_ket(step["psi_meas"])
            return _ParsedStep(
                "cut_measurement",
                psi_meas=psi_meas,
                psi_reset=_reshape_ket(step.get("psi_reset", psi_meas)),
            )
        if step_type == "cut_preparation":
            return _ParsedStep("cut_preparation", psi_prep=_reshape_ket(step["psi_prep"]))
        msg = f"Unsupported probe step type: {step_type!r}"
        raise ValueError(msg)
    psi_meas, psi_prep = step
    return _ParsedStep(
        "measure_prepare",
        psi_meas=_reshape_ket(psi_meas),
        psi_prep=_reshape_ket(psi_prep),
    )


def compute_born_probability(rho: NDArray[np.complex128], psi: NDArray[np.complex128]) -> float:
    """Compute Born probability ``<psi|rho|psi>`` for a rank-one effect.

    Args:
        rho: ``2 x 2`` density matrix.
        psi: Length-2 ket.

    Returns:
        Real probability in ``[0, 1]``.
    """
    r = np.asarray(rho, dtype=np.complex128).reshape(2, 2)
    ket = np.asarray(psi, dtype=np.complex128).reshape(2)
    return float(np.real(np.vdot(ket, r @ ket)))


def build_intervention_operator(step: AnyInterventionStep) -> InterventionMap | NDArray[np.complex128]:
    """Build an intervention map or unitary matrix for a probe step.

    Args:
        step: Structured dict step or measure/prepare ket pair.

    Returns:
        :class:`InterventionMap` or ``2 x 2`` unitary for process-tensor contraction.
    """
    parsed = _parse_intervention_step(step)
    if parsed.kind == "unitary":
        assert parsed.unitary is not None
        return parsed.unitary
    if parsed.kind == "cut_measurement":
        assert parsed.psi_meas is not None
        assert parsed.psi_reset is not None
        return InterventionMap(
            rho_prep=np.outer(parsed.psi_reset, parsed.psi_reset.conj()),
            effect=np.outer(parsed.psi_meas, parsed.psi_meas.conj()),
        )
    if parsed.kind == "cut_preparation":
        assert parsed.psi_prep is not None
        return InterventionMap(
            rho_prep=np.outer(parsed.psi_prep, parsed.psi_prep.conj()),
            effect=np.eye(2, dtype=np.complex128),
        )
    assert parsed.psi_meas is not None
    assert parsed.psi_prep is not None
    return InterventionMap(
        rho_prep=np.outer(parsed.psi_prep, parsed.psi_prep.conj()),
        effect=np.outer(parsed.psi_meas, parsed.psi_meas.conj()),
    )


def apply_intervention_to_rho(rho: NDArray[np.complex128], step: AnyInterventionStep) -> NDArray[np.complex128]:
    """Apply one intervention step to a single-qubit density matrix.

    Args:
        rho: ``2 x 2`` density matrix before the step.
        step: MP pair, unitary dict, or structured probe step.

    Returns:
        Trace-normalized ``2 x 2`` state after the step.
    """
    op = build_intervention_operator(step)
    r = np.asarray(rho, dtype=np.complex128).reshape(2, 2)
    out = op @ r @ op.conj().T if isinstance(op, np.ndarray) else op(r)
    tr = np.trace(out)
    if abs(tr) > 1e-15:
        out /= tr
    return out


def compute_intervention_probability(rho: NDArray[np.complex128], step: AnyInterventionStep) -> float:
    """Compute the branch-weight probability for one intervention step.

    Args:
        rho: ``2 x 2`` state before the step.
        step: MP pair, unitary dict, or structured probe step.

    Returns:
        Step probability used in branch-weight accumulation.
    """
    parsed = _parse_intervention_step(step)
    if parsed.kind in {"unitary", "cut_preparation"}:
        return 1.0
    assert parsed.psi_meas is not None
    return compute_born_probability(rho, parsed.psi_meas)


def apply_intervention_to_backend(
    state: MPS | NDArray[np.complex128],
    step: AnyInterventionStep,
    *,
    solver: str,
    chain_length: int,
) -> tuple[MPS | NDArray[np.complex128], float]:
    """Apply one intervention step to a backend state on site 0.

    Args:
        state: Current backend state (dense vector for MCWF, MPS for TJM).
        step: Structured dict step or measure/prepare ket pair.
        solver: Backend solver name.
        chain_length: Number of qubits in the chain.

    Returns:
        Tuple ``(state_out, step_prob)`` after the intervention.
    """
    parsed = _parse_intervention_step(step)
    if parsed.kind == "unitary":
        assert parsed.unitary is not None
        return _apply_backend_unitary_site_zero(state, parsed.unitary, solver), 1.0
    if parsed.kind == "cut_measurement":
        assert parsed.psi_meas is not None
        assert parsed.psi_reset is not None
        state_out, prob = _reprepare_backend_state_forced(state, parsed.psi_meas, parsed.psi_reset, solver)
        return state_out, float(prob)
    if parsed.kind == "cut_preparation":
        assert parsed.psi_prep is not None
        state_out, prob = _apply_cut_preparation_step(
            state,
            parsed.psi_prep,
            solver,
            chain_length=chain_length,
        )
        return state_out, float(prob)
    assert parsed.psi_meas is not None
    assert parsed.psi_prep is not None
    state_out, prob = _reprepare_backend_state_forced(state, parsed.psi_meas, parsed.psi_prep, solver)
    return state_out, float(prob)
