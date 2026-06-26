# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Operational-memory backend protocol and orchestration."""

from __future__ import annotations

from typing import Any, Protocol, cast

import numpy as np

from mqt.yaqs.core.parallel_utils import merge_execution_config

from .branch_weights import compute_analytic_weights
from .memory_matrix import assemble_memory_matrix, compute_spectrum
from .samples import ProbeSet, sample_probes


class OperationalMemoryBackend(Protocol):
    """Protocol for backends that evaluate weighted split-cut probes.

    Implement :meth:`evaluate_probes_weighted` (for example :class:`~mqt.yaqs.characterization.memory.backends.exact.ExactBackend`).
    """

    def evaluate_probes_weighted(self, probe_set: ProbeSet) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate weighted probe responses.

        Args:
            probe_set: Sampled split-cut probes.

        Returns:
            Tuple ``(pauli_xyz_ij, weights_ij)`` with shapes ``(n_p, n_f, 4)`` and ``(n_p, n_f)``.
        """


class CombProbeBackend(Protocol):
    """Protocol for comb/surrogate backends exposing :meth:`evaluate_probes`.

    Weighted evaluation is supplied by :func:`evaluate_probes_weighted_for`.
    """

    def evaluate_probes(self, probe_set: ProbeSet) -> np.ndarray:
        """Evaluate unweighted probe responses.

        Args:
            probe_set: Sampled split-cut probes.

        Returns:
            Pauli tomography array of shape ``(n_pasts, n_futures, 4)``.
        """


MemoryProcessBackend = OperationalMemoryBackend | CombProbeBackend


# Back-compat alias until call sites migrate.
_ProbeProcess = MemoryProcessBackend


def evaluate_probes_weighted_for(
    process: MemoryProcessBackend,
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
    if "evaluate_probes_weighted" in process.__class__.__dict__:
        weighted = cast("OperationalMemoryBackend", process)
        pauli_xyz_ij, weights_ij = weighted.evaluate_probes_weighted(probe_set)
        return np.asarray(pauli_xyz_ij, dtype=np.float32), np.asarray(weights_ij, dtype=np.float64)
    if "evaluate_probes" in process.__class__.__dict__:
        comb = cast("CombProbeBackend", process)
        pauli_xyz_ij = np.asarray(comb.evaluate_probes(probe_set), dtype=np.float32)
        return pauli_xyz_ij, compute_analytic_weights(probe_set)
    msg = f"{type(process).__name__} must implement evaluate_probes_weighted or evaluate_probes"
    raise TypeError(msg)


def run_operational_memory(
    *,
    process: MemoryProcessBackend,
    cut: int,
    k: int,
    n_pasts: int = 32,
    n_futures: int = 32,
    rng: np.random.Generator | None = None,
    probe_set: ProbeSet | None = None,
    return_raw: bool = False,
    intervention_mode: str = "unitary_break_mp",
    unitary_ensemble: str = "haar",
    parallel: bool | None = None,
) -> dict[str, Any]:
    """Run split-cut probing and assemble memory-matrix diagnostics.

    Args:
        process: Operational-memory backend (exact, comb, or surrogate).
        cut: Causal cut index.
        k: Total sequence length.
        n_pasts: Past probe count when sampling internally.
        n_futures: Future probe count when sampling internally.
        rng: RNG for internal probe sampling.
        probe_set: Pre-sampled probes (optional).
        return_raw: If True, include uncentered ``memory_matrix_raw``.
        intervention_mode: Passed to internal :func:`~mqt.yaqs.characterization.memory.operational_memory.samples.sample_probes`.
        unitary_ensemble: Passed to internal sampling.
        parallel: Override parallelism for :class:`~mqt.yaqs.characterization.memory.backends.exact.ExactBackend`.

    Returns:
        Dict with ``entropy``, ``rank``, ``singular_values``, ``memory_matrix``,
        ``probe_set``, and optional ``weights_ij``.
    """
    if parallel is not None:
        from ..backends.exact import ExactBackend  # noqa: PLC0415

        if isinstance(process, ExactBackend):
            process._execution = merge_execution_config(  # noqa: SLF001
                process._execution,  # noqa: SLF001
                parallel=parallel,
            )
    if probe_set is None:
        if rng is None:
            rng = np.random.default_rng()
        probe_set = sample_probes(
            cut=cut,
            k=k,
            n_pasts=n_pasts,
            n_futures=n_futures,
            rng=rng,
            intervention_mode=intervention_mode,
            unitary_ensemble=unitary_ensemble,
        )
    pauli_xyz_ij, weights_ij = evaluate_probes_weighted_for(process, probe_set)
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
