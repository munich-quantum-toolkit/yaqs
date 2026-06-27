# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Analytic and trace-based branch weights for operational memory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .grid import assemble_probe_sequence

if TYPE_CHECKING:
    from .samples import ProbeSet

ProbeStep = dict[str, Any] | tuple[np.ndarray, np.ndarray]

_RHO0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)


def compute_born_prob(rho: np.ndarray, psi: np.ndarray) -> float:
    """Return Born probability ``<psi|rho|psi>`` for a rank-one effect.

    Args:
        rho: ``2 x 2`` density matrix.
        psi: Length-2 ket.

    Returns:
        Real probability in ``[0, 1]``.
    """
    r = np.asarray(rho, dtype=np.complex128).reshape(2, 2)
    ket = np.asarray(psi, dtype=np.complex128).reshape(2)
    return float(np.real(np.vdot(ket, r @ ket)))


def _step_probability(rho: np.ndarray, step: ProbeStep) -> float:
    """Compute the measurement probability for one intervention step.

    Args:
        rho: ``2 x 2`` state before the step.
        step: MP pair, unitary dict, or structured probe step.

    Returns:
        Step probability used in branch-weight accumulation.

    Raises:
        ValueError: If the step type is unsupported.
    """
    if isinstance(step, dict):
        step_type = str(step.get("type", "")).lower()
        if step_type in {"unitary", "prepare_only"}:
            return 1.0
        if step_type == "measure_only":
            psi_meas = np.asarray(step["psi_meas"], dtype=np.complex128).reshape(2)
            return compute_born_prob(rho, psi_meas)
        msg = f"Unsupported probe step type: {step_type!r}"
        raise ValueError(msg)
    psi_meas, _ = step
    return compute_born_prob(rho, psi_meas)


def _apply_step(rho: np.ndarray, step: ProbeStep) -> np.ndarray:
    """Apply one intervention step to a single-qubit state.

    Args:
        rho: ``2 x 2`` density matrix before the step.
        step: MP pair, unitary dict, or structured probe step.

    Returns:
        Normalized ``2 x 2`` state after the step.

    Raises:
        ValueError: If the step type is unsupported.
    """
    r = np.asarray(rho, dtype=np.complex128).reshape(2, 2)
    if isinstance(step, dict):
        step_type = str(step.get("type", "")).lower()
        if step_type == "unitary":
            u = np.asarray(step["U"], dtype=np.complex128).reshape(2, 2)
            out = u @ r @ u.conj().T
        elif step_type == "measure_only":
            psi_meas = np.asarray(step["psi_meas"], dtype=np.complex128).reshape(2)
            prob = compute_born_prob(r, psi_meas)
            if prob > 1e-15:
                if "psi_reset" in step:
                    psi_reset = np.asarray(step["psi_reset"], dtype=np.complex128).reshape(2)
                    ket = psi_reset / max(float(np.linalg.norm(psi_reset)), 1e-15)
                else:
                    ket = psi_meas / max(float(np.linalg.norm(psi_meas)), 1e-15)
                out = np.outer(ket, ket.conj())
            else:
                out = np.zeros((2, 2), dtype=np.complex128)
        elif step_type == "prepare_only":
            psi_prep = np.asarray(step["psi_prep"], dtype=np.complex128).reshape(2)
            ket = psi_prep / max(float(np.linalg.norm(psi_prep)), 1e-15)
            out = np.outer(ket, ket.conj())
        else:
            msg = f"Unsupported probe step type: {step_type!r}"
            raise ValueError(msg)
    else:
        psi_meas, psi_prep = step
        prob = compute_born_prob(r, psi_meas)
        ket = np.asarray(psi_prep, dtype=np.complex128).reshape(2)
        ket /= max(float(np.linalg.norm(ket)), 1e-15)
        out = np.outer(ket, ket.conj()) if prob > 1e-15 else np.zeros((2, 2), dtype=np.complex128)
    tr = np.trace(out)
    if abs(tr) > 1e-15:
        out /= tr
    return out


def compute_branch_weight(steps: list[Any], *, cut: int) -> float:
    """Compute analytic branch weight from step probabilities up to ``cut``.

    Args:
        steps: Full intervention sequence.
        cut: Causal cut index.

    Returns:
        Cumulative branch weight ``prod_t p_t`` for ``t < cut``.
    """
    rho = _RHO0.copy()
    weight = 1.0
    for t in range(min(int(cut), len(steps))):
        sp = _step_probability(rho, steps[t])
        weight *= sp
        if weight < 1e-15:
            return float(weight)
        rho = _apply_step(rho, steps[t])
    return float(weight)


def compute_branch_weights(probe_set: ProbeSet) -> np.ndarray:
    r"""Compute analytic branch weights :math:`w_{\alpha,m}` at the causal cut.

    Args:
        probe_set: Sampled split-cut probes.

    Returns:
        Array of shape ``(n_pasts, n_futures)`` constant across future columns per past.
    """
    n_p = len(probe_set.past_pairs)
    n_f = len(probe_set.future_pairs)
    c = int(probe_set.cut)
    w = np.empty((n_p, n_f), dtype=np.float64)
    for i in range(n_p):
        w_i = compute_branch_weight(assemble_probe_sequence(probe_set, i, 0), cut=c)
        w[i, :] = w_i
    return w


def compute_analytic_weights(probe_set: ProbeSet) -> np.ndarray:
    """Compute analytic branch weights for comb/surrogate backends.

    Args:
        probe_set: Sampled split-cut probes.

    Returns:
        Branch-weight array of shape ``(n_pasts, n_futures)``.
    """
    return compute_branch_weights(probe_set)


def compute_trace_weights(
    traces: list[dict[str, Any]],
    *,
    n_pasts: int,
    n_futures: int,
    cut: int,
) -> np.ndarray:
    """Compute branch weights from simulated step probabilities through ``cut``.

    Args:
        traces: Per-sequence diagnostic dicts with ``step_probs`` (flat grid order).
        n_pasts: Number of past probe branches.
        n_futures: Number of future probe branches.
        cut: Causal cut index.

    Returns:
        Branch-weight array of shape ``(n_pasts, n_futures)``.
    """
    n_p, n_f = int(n_pasts), int(n_futures)
    w = np.zeros((n_p, n_f), dtype=np.float64)
    c = int(cut)
    for ii in range(n_p):
        for jj in range(n_f):
            probs = traces[ii * n_f + jj]["step_probs"]
            n = min(c, len(probs))
            w[ii, jj] = float(np.prod(probs[:n])) if n else 1.0
    return w
