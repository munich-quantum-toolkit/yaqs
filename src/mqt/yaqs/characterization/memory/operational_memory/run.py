# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Operational-memory backend protocol and orchestration."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from .branch_weights import compute_branch_weights
from .grid import assemble_delayed_probe_grid, compute_delayed_length
from .memory_matrix import assemble_memory_matrix, compute_spectrum
from .samples import ProbeSet, sample_probes

if TYPE_CHECKING:
    from mqt.yaqs.core.parallel_utils import ExecutionConfig


class OperationalMemoryBackend(Protocol):
    """Protocol for split-cut probing backends used in operational memory.

    Implement **either** :meth:`evaluate_probes_weighted` (simulation trace weights, e.g.
    :class:`~mqt.yaqs.characterization.memory.backends.exact.ExactBackend`) **or**
    :meth:`evaluate_probes` (black-box Pauli responses for process tensors and surrogates).
    :func:`evaluate_probes_weighted_for` dispatches to the implemented method.
    """

    def evaluate_probes_weighted(self, probe_set: ProbeSet) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate weighted probe responses.

        Args:
            probe_set: Sampled split-cut probes.

        Returns:
            Tuple ``(pauli_xyz_ij, weights_ij)`` with shapes ``(n_pasts, n_futures, 4)`` and
            ``(n_pasts, n_futures)``.
        """

    def evaluate_probes(self, probe_set: ProbeSet) -> np.ndarray:
        """Evaluate unweighted probe responses.

        Args:
            probe_set: Sampled split-cut probes.

        Returns:
            Pauli tomography array of shape ``(n_pasts, n_futures, 4)``.
        """


def evaluate_probes_weighted_for(
    process: OperationalMemoryBackend,
    probe_set: ProbeSet,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate weighted probe responses; analytic weights unless the class overrides.

    Args:
        process: Backend implementing :meth:`evaluate_probes_weighted` or :meth:`evaluate_probes`.
        probe_set: Sampled split-cut probes.

    Returns:
        Tuple ``(pauli_xyz_ij, weights_ij)``.

    Raises:
        TypeError: If ``process`` implements neither weighted nor unweighted probing.
    """
    weighted_fn = getattr(process, "evaluate_probes_weighted", None)
    if callable(weighted_fn):
        pauli_xyz_ij, weights_ij = weighted_fn(probe_set)
        return np.asarray(pauli_xyz_ij, dtype=np.float64), np.asarray(weights_ij, dtype=np.float64)
    evaluate_fn = getattr(process, "evaluate_probes", None)
    if callable(evaluate_fn):
        pauli_xyz_ij = np.asarray(evaluate_fn(probe_set), dtype=np.float64)
        return pauli_xyz_ij, compute_branch_weights(probe_set)
    msg = f"{type(process).__name__} must implement evaluate_probes_weighted or evaluate_probes"
    raise TypeError(msg)


def run_operational_memory(
    *,
    process: OperationalMemoryBackend,
    cut: int,
    num_interventions: int,
    n_pasts: int = 32,
    n_futures: int = 32,
    rng: np.random.Generator | None = None,
    probe_set: ProbeSet | None = None,
    return_raw: bool = False,
    intervention_mode: str = "split_cut_unitary",
    unitary_ensemble: str = "haar",
    parallel: bool | None = None,
    delay: int = 0,
) -> dict[str, Any]:
    """Run split-cut probing and assemble memory-matrix diagnostics.

    Args:
        process: Operational-memory backend (exact, process tensor, or surrogate).
        cut: Causal cut index.
        num_interventions: Base sequence length (past + cut + future legs; excludes ``delay`` slots).
        n_pasts: Past probe count when sampling internally.
        n_futures: Future probe count when sampling internally.
        rng: RNG for internal probe sampling.
        probe_set: Pre-sampled probes (optional).
        return_raw: If True, include uncentered ``memory_matrix_raw``.
        intervention_mode: Passed to internal
            :func:`~mqt.yaqs.characterization.memory.operational_memory.samples.sample_probes`.
        unitary_ensemble: Passed to internal sampling.
        parallel: Override parallelism for :class:`~mqt.yaqs.characterization.memory.backends.exact.ExactBackend`.
        delay: Number of ``(|0>, |0>)`` soft-reset slots to insert at the causal break.

    Returns:
        Dict with ``entropy``, ``modes``, ``singular_values``, ``memory_matrix``,
        ``probe_set``, and optional ``weights_ij``.

    Raises:
        ValueError: If ``delay`` is negative, a supplied ``probe_set`` was built for a
            different ``cut`` or ``num_interventions``, or ``delay > 0`` with a backend that does not
            support custom sequences.
    """
    if delay < 0:
        msg = f"delay must be >= 0, got {delay}"
        raise ValueError(msg)

    exact_backend_cls: type | None = None
    if delay > 0 or parallel is not None:
        from ..backends.exact import ExactBackend  # noqa: PLC0415

        exact_backend_cls = ExactBackend

    execution_override: ExecutionConfig | None = None
    if parallel is not None and exact_backend_cls is not None and isinstance(process, exact_backend_cls):
        execution_override = process.execution_config(parallel=parallel)
    if probe_set is not None and (
        int(probe_set.cut) != int(cut) or int(probe_set.num_interventions) != int(num_interventions)
    ):
        msg = (
            f"probe_set was built for cut={probe_set.cut}, "
            f"num_interventions={probe_set.num_interventions}, but cut={cut}, "
            f"num_interventions={num_interventions} were requested."
        )
        raise ValueError(msg)
    if probe_set is None:
        if rng is None:
            rng = np.random.default_rng()
        probe_set = sample_probes(
            cut=cut,
            num_interventions=num_interventions,
            n_pasts=n_pasts,
            n_futures=n_futures,
            rng=rng,
            intervention_mode=intervention_mode,
            unitary_ensemble=unitary_ensemble,
        )
    intervention_steps_list = None
    sim_probe_set = probe_set
    if delay > 0:
        if exact_backend_cls is None or not isinstance(process, exact_backend_cls):
            msg = "delay > 0 requires an exact Hamiltonian characterize backend."
            raise ValueError(msg)
        intervention_steps_list, _, _ = assemble_delayed_probe_grid(probe_set, delay=delay)
        sim_probe_set = replace(probe_set, num_interventions=compute_delayed_length(num_interventions=num_interventions, delay=delay))

    if (
        exact_backend_cls is not None
        and isinstance(process, exact_backend_cls)
        and (delay > 0 or execution_override is not None)
    ):
        eval_kwargs: dict[str, Any] = {}
        if intervention_steps_list is not None:
            eval_kwargs["intervention_steps_list"] = intervention_steps_list
        if execution_override is not None:
            eval_kwargs["_execution"] = execution_override
        pauli_xyz_ij, weights_ij = process.evaluate_probes_weighted(sim_probe_set, **eval_kwargs)
    else:
        pauli_xyz_ij, weights_ij = evaluate_probes_weighted_for(process, sim_probe_set)
    m_raw, memory_matrix = assemble_memory_matrix(pauli_xyz_ij, weights_ij)
    ana = compute_spectrum(memory_matrix)
    out: dict[str, Any] = {
        "pauli_xyz_ij": pauli_xyz_ij,
        **ana,
        "probe_set": probe_set,
        "memory_matrix": memory_matrix,
        "weights_ij": weights_ij,
    }
    if return_raw:
        out["memory_matrix_raw"] = m_raw
    return out
