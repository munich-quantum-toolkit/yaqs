# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Operational-memory backend protocol and orchestration."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

import numpy as np

from ..shared.interventions import DEFAULT_INTERVENTION_STYLE
from .grid import assemble_probe_grid, compute_delayed_length
from .response_matrix import assemble_response_matrix, compute_spectrum
from .samples import ProbeSet, sample_probes

if TYPE_CHECKING:
    from mqt.yaqs.core.parallel_utils import ExecutionConfig


class SupportsEvaluateProbesWeighted(Protocol):
    """Protocol for backends that implement :meth:`evaluate_probes_weighted`."""

    def evaluate_probes_weighted(self, probe_set: ProbeSet) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate probe responses with complete retained-record probabilities.

        Args:
            probe_set: Sampled split-cut probes.

        Returns:
            Tuple ``(pauli_ixyz_ij, weights_ij)`` with shapes ``(n_pasts, n_futures, 4)`` and
            ``(n_pasts, n_futures)``. Pauli channels are ordered ``(I, X, Y, Z)``.
        """


OperationalMemoryBackend: TypeAlias = SupportsEvaluateProbesWeighted
"""Split-cut backend returning normalized responses and complete record probabilities.

Backends must implement :meth:`evaluate_probes_weighted`. Normalized Pauli responses alone are
insufficient because retained-outcome probabilities depend on the process dynamics.
"""


def evaluate_probes_with_weights(
    process: OperationalMemoryBackend,
    probe_set: ProbeSet,
    *,
    initial_rho: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate responses and complete process-aware retained-record probabilities.

    Args:
        process: Backend implementing :meth:`evaluate_probes_weighted`.
        probe_set: Sampled split-cut probes.
        initial_rho: Optional site-0 state at the process boundary. Surrogate backends require
            this state because it conditions their predictions and first outcome probability.

    Returns:
        Tuple ``(pauli_ixyz_ij, weights_ij)`` supplied by the backend.

    Raises:
        TypeError: If ``process`` does not implement weighted probing.
    """
    weighted_fn = getattr(process, "evaluate_probes_weighted", None)
    if callable(weighted_fn):
        if initial_rho is None:
            pauli_ixyz_ij, weights_ij = weighted_fn(probe_set)
        else:
            pauli_ixyz_ij, weights_ij = weighted_fn(probe_set, initial_rho=initial_rho)
        return np.asarray(pauli_ixyz_ij, dtype=np.float64), np.asarray(weights_ij, dtype=np.float64)
    msg = (
        f"{type(process).__name__} must implement evaluate_probes_weighted; "
        "normalized probe responses do not determine retained-outcome probabilities"
    )
    raise TypeError(msg)


def _exact_backend_cls_if_needed(*, delay: int, parallel: bool | None) -> type | None:
    """Return :class:`~mqt.yaqs.characterization.memory.backends.exact.ExactBackend` when needed.

    Args:
        delay: Number of soft-reset slots at the causal cut.
        parallel: Optional parallelism override for the exact backend.

    Returns:
        The exact backend class, or ``None`` when delay and parallel are both inactive.
    """
    if delay > 0 or parallel is not None:
        from ..backends.exact import ExactBackend  # ruff:ignore[import-outside-top-level]

        return ExactBackend
    return None


def _validate_probe_set_geometry(
    probe_set: ProbeSet,
    *,
    cut: int,
    num_interventions: int,
) -> None:
    """Ensure a pre-sampled probe grid matches the requested split-cut geometry.

    Args:
        probe_set: Pre-sampled probes.
        cut: Requested causal cut index.
        num_interventions: Requested base sequence length.

    Raises:
        ValueError: If ``probe_set`` was built for different ``cut`` or ``num_interventions``.
    """
    if int(probe_set.cut) != int(cut) or int(probe_set.num_interventions) != int(num_interventions):
        msg = (
            f"probe_set was built for cut={probe_set.cut}, "
            f"num_interventions={probe_set.num_interventions}, but cut={cut}, "
            f"num_interventions={num_interventions} were requested."
        )
        raise ValueError(msg)


def _resolve_probe_set(
    probe_set: ProbeSet | None,
    *,
    cut: int,
    num_interventions: int,
    n_pasts: int,
    n_futures: int,
    rng: np.random.Generator | None,
    intervention_style: str,
) -> ProbeSet:
    """Return a probe set, sampling internally when ``probe_set`` is omitted.

    Args:
        probe_set: Optional pre-sampled probes.
        cut: Causal cut index.
        num_interventions: Base sequence length.
        n_pasts: Past probe count for internal sampling.
        n_futures: Future probe count for internal sampling.
        rng: RNG for internal sampling.
        intervention_style: Intervention style for internal sampling.

    Returns:
        Probe grid for split-cut characterization.
    """
    if probe_set is not None:
        return probe_set
    sample_rng = np.random.default_rng() if rng is None else rng
    return sample_probes(
        cut=cut,
        num_interventions=num_interventions,
        n_pasts=n_pasts,
        n_futures=n_futures,
        rng=sample_rng,
        intervention_style=intervention_style,
    )


def _setup_delayed_probing(
    probe_set: ProbeSet,
    *,
    delay: int,
    num_interventions: int,
    process: OperationalMemoryBackend,
    exact_backend_cls: type | None,
) -> tuple[ProbeSet, list[Any] | None]:
    """Expand probe geometry when soft-reset delay slots are requested.

    Args:
        probe_set: Base probe grid without delay slots.
        delay: Number of ``(|0>, |0>)`` slots at the causal cut.
        num_interventions: Base sequence length (excluding delay).
        process: Operational-memory backend under test.
        exact_backend_cls: Exact backend class when delay or parallel overrides apply.

    Returns:
        Tuple ``(sim_probe_set, intervention_steps_list)`` where ``intervention_steps_list`` is
        ``None`` when ``delay == 0``.

    Raises:
        ValueError: If ``delay > 0`` with a backend that cannot simulate custom sequences.
    """
    if delay <= 0:
        return probe_set, None
    if exact_backend_cls is None or not isinstance(process, exact_backend_cls):
        msg = "delay > 0 requires an exact Hamiltonian characterize backend."
        raise ValueError(msg)
    intervention_steps_list, _, _ = assemble_probe_grid(probe_set, delay=delay)
    sim_probe_set = replace(
        probe_set, num_interventions=compute_delayed_length(num_interventions=num_interventions, delay=delay)
    )
    return sim_probe_set, intervention_steps_list


def _evaluate_operational_memory_probes(
    process: OperationalMemoryBackend,
    sim_probe_set: ProbeSet,
    *,
    exact_backend_cls: type | None,
    execution_override: ExecutionConfig | None,
    intervention_steps_list: list[Any] | None,
    initial_rho: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate split-cut probe responses on the selected backend.

    Args:
        process: Operational-memory backend.
        sim_probe_set: Probe grid passed to the backend (may include delay expansion).
        exact_backend_cls: Exact backend class when delay or parallel overrides apply.
        execution_override: Optional execution configuration for the exact backend.
        intervention_steps_list: Optional per-probe custom intervention sequences.
        initial_rho: Optional site-0 state passed to a surrogate backend.

    Returns:
        Tuple ``(pauli_ixyz_ij, weights_ij)``.
    """
    use_exact_weighted = (
        exact_backend_cls is not None
        and isinstance(process, exact_backend_cls)
        and (intervention_steps_list is not None or execution_override is not None)
    )
    if not use_exact_weighted:
        return evaluate_probes_with_weights(process, sim_probe_set, initial_rho=initial_rho)
    eval_kwargs: dict[str, Any] = {}
    if intervention_steps_list is not None:
        eval_kwargs["intervention_steps_list"] = intervention_steps_list
    if execution_override is not None:
        eval_kwargs["_execution"] = execution_override
    return process.evaluate_probes_weighted(sim_probe_set, **eval_kwargs)


def run_memory_characterization(
    *,
    process: OperationalMemoryBackend,
    cut: int,
    num_interventions: int,
    n_pasts: int = 32,
    n_futures: int = 32,
    rng: np.random.Generator | None = None,
    probe_set: ProbeSet | None = None,
    intervention_style: str = DEFAULT_INTERVENTION_STYLE,
    parallel: bool | None = None,
    delay: int = 0,
    initial_rho: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run split-cut probing and assemble response-matrix diagnostics.

    Args:
        process: Operational-memory backend (exact, process tensor, or surrogate).
        cut: Causal cut index.
        num_interventions: Base sequence length (past + cut + future legs; excludes ``delay`` slots).
        n_pasts: Past probe count when sampling internally.
        n_futures: Future probe count when sampling internally.
        rng: RNG for internal probe sampling.
        probe_set: Pre-sampled probes (optional).
        intervention_style: ``"haar"``, ``"clifford"``, or ``"measure_prepare"`` for internal sampling.
        parallel: Override parallelism for :class:`~mqt.yaqs.characterization.memory.backends.exact.ExactBackend`.
        delay: Number of ``(|0>, |0>)`` soft-reset slots to insert at the causal break.
        initial_rho: Optional site-0 state after the initial evolution segment and before the
            first intervention. Surrogate backends require this state.

    Returns:
        Dict with ``entropy``, ``modes``, ``singular_values``, ``response_matrix``,
        ``pauli_ixyz_ij``, ``probe_set``, and ``weights_ij``. The response matrix has shape
        ``(4 * n_futures, n_pasts)`` with future-probe ``(I, X, Y, Z)`` rows and history
        columns.

    Raises:
        ValueError: If ``delay`` is negative, a supplied ``probe_set`` was built for a
            different ``cut`` or ``num_interventions``, or ``delay > 0`` with a backend that does not
            support custom sequences, or a backend returns invalid weighted responses.
    """
    if delay < 0:
        msg = f"delay must be >= 0, got {delay}"
        raise ValueError(msg)

    exact_backend_cls = _exact_backend_cls_if_needed(delay=delay, parallel=parallel)
    execution_override: ExecutionConfig | None = None
    if parallel is not None and exact_backend_cls is not None and isinstance(process, exact_backend_cls):
        from ..backends.exact import ExactBackend  # ruff:ignore[import-outside-top-level]

        assert isinstance(process, ExactBackend)
        execution_override = process.execution_config(parallel=parallel)
    if probe_set is not None:
        _validate_probe_set_geometry(probe_set, cut=cut, num_interventions=num_interventions)
    probe_set = _resolve_probe_set(
        probe_set,
        cut=cut,
        num_interventions=num_interventions,
        n_pasts=n_pasts,
        n_futures=n_futures,
        rng=rng,
        intervention_style=intervention_style,
    )
    sim_probe_set, intervention_steps_list = _setup_delayed_probing(
        probe_set,
        delay=delay,
        num_interventions=num_interventions,
        process=process,
        exact_backend_cls=exact_backend_cls,
    )
    pauli_ixyz_ij, weights_ij = _evaluate_operational_memory_probes(
        process,
        sim_probe_set,
        exact_backend_cls=exact_backend_cls,
        execution_override=execution_override,
        intervention_steps_list=intervention_steps_list,
        initial_rho=initial_rho,
    )
    response_matrix = assemble_response_matrix(pauli_ixyz_ij, weights_ij)
    ana = compute_spectrum(response_matrix)
    out: dict[str, Any] = {
        "pauli_ixyz_ij": pauli_ixyz_ij,
        **ana,
        "probe_set": probe_set,
        "response_matrix": response_matrix,
        "weights_ij": weights_ij,
    }
    return out
