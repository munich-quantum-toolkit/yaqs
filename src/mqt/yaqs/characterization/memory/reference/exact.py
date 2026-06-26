# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact process probing helpers built on the rollout backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mqt.yaqs.core.parallel_utils import ExecutionConfig, merge_execution_config

from ..combs.core.encoding import packed_rho8_to_pauli_batch
from ..combs.core.utils import StochasticSolver, make_mcwf_static_context
from ..combs.surrogates.workflow import _simulate_sequences, simulate_final_states_with_diagnostics
from ..diagnostics.probe import ProbeSet, _build_all_pairs_grid

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.mpo import MPO
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


class ExactProbeProcess:
    """Exact rollout-backed probe process with internal static context."""

    def __init__(
        self,
        *,
        operator: MPO,
        sim_params: AnalogSimParams,
        initial_psi: np.ndarray,
        parallel: bool = True,
        show_progress: bool = False,
        _execution: ExecutionConfig | None = None,
    ) -> None:
        """Initialize exact probe backend with reusable MCWF static context.

        Args:
            operator: Hamiltonian MPO.
            sim_params: Analog simulation parameters.
            initial_psi: Initial state vector for rollouts.
            parallel: Whether to parallelize sequence simulation.
            show_progress: Whether to show a progress bar during simulation.
        """
        self.operator = operator
        self.sim_params = sim_params
        self.initial_psi = np.asarray(initial_psi, dtype=np.complex128).copy()
        self._execution = merge_execution_config(_execution, parallel=parallel, show_progress=show_progress)
        self._static_ctx = make_mcwf_static_context(operator, sim_params, noise_model=None)

    @property
    def parallel(self) -> bool:
        """Whether parallel rollout execution is enabled."""
        return self._execution.parallel

    def evaluate_probe_set(self, probe_set: ProbeSet) -> np.ndarray:
        """Run exact backend for all (past, future) probe combinations.

        Args:
            probe_set: Sampled split-cut probes.

        Returns:
            Array of shape ``(n_pasts, n_futures, 4)`` with Pauli tomography ``(I, X, Y, Z)``
            from the final single-qubit reduced state.

        Raises:
            RuntimeError: If the backend returns an unexpected result count.
            TypeError: If the backend output is not an ndarray.
        """
        all_pairs, n_p, n_f = _build_all_pairs_grid(probe_set)
        n_tot = n_p * n_f
        initial_psis = [self.initial_psi.copy() for _ in range(n_tot)]
        final_packed = _simulate_sequences(
            operator=self.operator,
            sim_params=self.sim_params,
            timesteps=[float(self.sim_params.dt)] * (int(probe_set.k) + 1),
            psi_pairs_list=all_pairs,
            initial_psis=initial_psis,
            static_ctx=self._static_ctx,
            parallel=self._execution.parallel,
            show_progress=self._execution.show_progress,
            _execution=self._execution,
            record_step_states=False,
        )
        if not isinstance(final_packed, np.ndarray):
            msg = "Expected ndarray output from exact simulation."
            raise TypeError(msg)
        if final_packed.shape[0] != n_tot:
            msg = f"Expected {n_tot} final states from exact simulation, got {final_packed.shape[0]}."
            raise RuntimeError(msg)
        packed_flat = np.asarray(final_packed, dtype=np.float32).reshape(n_p * n_f, 8)
        return packed_rho8_to_pauli_batch(packed_flat).reshape(n_p, n_f, 4).astype(np.float32)


def evaluate_exact_probe_set_with_diagnostics(
    *,
    probe_set: ProbeSet,
    operator: MPO,
    sim_params: AnalogSimParams,
    initial_psi: np.ndarray,
    parallel: bool = True,
    show_progress: bool = False,
    solver: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    r"""Exact rollout with per-sequence diagnostics (branch weights, early termination).

    Args:
        probe_set: Sampled split-cut probes.
        operator: Hamiltonian MPO.
        sim_params: Analog simulation parameters.
        initial_psi: Initial state vector for rollouts.
        parallel: Whether to parallelize sequence simulation.
        show_progress: Whether to show a progress bar.
        solver: Stochastic solver (``"MCWF"`` or ``"TJM"``).

    Returns:
        ``(pauli_ij, weights_ij, traces_flat)`` where ``pauli_ij`` has shape
        ``(n_pasts, n_futures, 4)`` (Pauli tomography from the final reduced state),
        ``weights_ij`` holds break weights :math:`w_{\alpha,m}` from simulated step
        probabilities through cut ``c``, and
        ``traces_flat[i * n_f + j]`` matches the sequence order of :func:`_build_all_pairs_grid`.

    Raises:
        TypeError: If the backend output is not an ndarray.
    """
    all_pairs, n_p, n_f = _build_all_pairs_grid(probe_set)
    n_tot = n_p * n_f
    initial_psis = [np.asarray(initial_psi, dtype=np.complex128).copy() for _ in range(n_tot)]
    exec_cfg = merge_execution_config(None, parallel=parallel, show_progress=show_progress)
    resolved_solver: StochasticSolver = "MCWF" if solver is None else solver  # type: ignore[assignment]
    static_ctx = make_mcwf_static_context(operator, sim_params, noise_model=None) if resolved_solver == "MCWF" else None
    final_packed, traces = simulate_final_states_with_diagnostics(
        operator=operator,
        sim_params=sim_params,
        timesteps=[float(sim_params.dt)] * (int(probe_set.k) + 1),
        psi_pairs_list=all_pairs,
        initial_psis=initial_psis,
        static_ctx=static_ctx,
        parallel=exec_cfg.parallel,
        show_progress=exec_cfg.show_progress,
        solver=resolved_solver,
        _execution=exec_cfg,
    )
    if not isinstance(final_packed, np.ndarray):
        msg = "Expected ndarray output from exact simulation."
        raise TypeError(msg)
    pauli_xyz = packed_rho8_to_pauli_batch(final_packed.reshape(n_p * n_f, 8)).reshape(n_p, n_f, 4).astype(np.float32)
    w = np.zeros((n_p, n_f), dtype=np.float64)
    cut = int(probe_set.cut)
    for ii in range(n_p):
        for jj in range(n_f):
            probs = traces[ii * n_f + jj]["step_probs"]
            n = min(cut, len(probs))
            w[ii, jj] = float(np.prod(probs[:n])) if n else 1.0
    return pauli_xyz, w, traces
