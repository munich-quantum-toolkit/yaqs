# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact Hamiltonian probing via traced sequence simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mqt.yaqs.core.parallel_utils import ExecutionConfig, merge_execution_config

from ..operational_memory.grid import assemble_probe_grid
from ..shared.encoding import decode_packed_pauli_batch
from ..shared.utils import StochasticSolver, make_mcwf_static_context, validate_stochastic_solver
from .sequences.workflow import simulate_sequences

if TYPE_CHECKING:
    from mqt.yaqs.analog.mcwf import MCWFContext
    from mqt.yaqs.core.data_structures.mpo import MPO
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

    from ..operational_memory.samples import ProbeSet


def _resolve_sequence_grid(
    probe_set: ProbeSet,
    intervention_steps_list: list[list[Any]] | None,
) -> tuple[list[list[Any]], int, int]:
    """Resolve the flat intervention-sequence grid for simulation.

    Args:
        probe_set: Sampled split-cut probes.
        intervention_steps_list: Optional pre-built sequence list (experiment geometries).

    Returns:
        Tuple ``(all_pairs, n_pasts, n_futures)``.

    Raises:
        ValueError: If ``intervention_steps_list`` length does not match the probe grid.
    """
    if intervention_steps_list is None:
        return assemble_probe_grid(probe_set)
    n_p = len(probe_set.past_pairs)
    n_f = len(probe_set.future_pairs)
    if len(intervention_steps_list) != n_p * n_f:
        msg = f"intervention_steps_list length {len(intervention_steps_list)} != n_pasts * n_futures ({n_p * n_f})"
        raise ValueError(msg)
    return intervention_steps_list, n_p, n_f


def _branch_weights_from_simulation(
    simulation_diagnostics: list[dict[str, Any]],
    *,
    n_pasts: int,
    n_futures: int,
    cut: int,
) -> np.ndarray:
    """Compute branch weights from simulated step probabilities through ``cut``.

    Args:
        simulation_diagnostics: Per-sequence diagnostic dicts with ``step_probs`` (flat grid order).
        n_pasts: Number of past probe branches.
        n_futures: Number of future probe branches.
        cut: Causal cut index.

    Returns:
        Branch-weight array of shape ``(n_pasts, n_futures)``.
    """
    w = np.zeros((n_pasts, n_futures), dtype=np.float64)
    for past_idx in range(n_pasts):
        for future_idx in range(n_futures):
            probs = simulation_diagnostics[past_idx * n_futures + future_idx]["step_probs"]
            n = min(cut, len(probs))
            w[past_idx, future_idx] = float(np.prod(probs[:n])) if n else 1.0
    return w


class ExactBackend:
    """Exact MCWF/TJM backend for weighted split-cut probe evaluation.

    Builds a reusable static MCWF context internally and dispatches sequence
    simulation via :func:`~mqt.yaqs.characterization.memory.backends.sequences.workflow.simulate_sequences`
    with ``traced=True``.
    """

    def __init__(
        self,
        *,
        operator: MPO,
        sim_params: AnalogSimParams,
        initial_psi: np.ndarray,
        parallel: bool = True,
        show_progress: bool = False,
        solver: StochasticSolver | None = None,
        _execution: ExecutionConfig | None = None,
    ) -> None:
        """Initialize the exact probe backend.

        Args:
            operator: Hamiltonian MPO.
            sim_params: Analog simulation parameters.
            initial_psi: Initial state vector for sequences.
            parallel: Whether to parallelize sequence simulation.
            show_progress: Whether to show a progress bar during simulation.
            solver: Stochastic solver (``"MCWF"`` or ``"TJM"``); defaults to ``"MCWF"``.
        """
        self.operator = operator
        self.sim_params = sim_params
        self.initial_psi = np.asarray(initial_psi, dtype=np.complex128).copy()
        self._solver = validate_stochastic_solver(solver)
        self._execution = merge_execution_config(_execution, parallel=parallel, show_progress=show_progress)
        self._static_ctx = (
            make_mcwf_static_context(operator, sim_params, noise_model=None) if self._solver == "MCWF" else None
        )

    @property
    def parallel(self) -> bool:
        """Whether parallel sequence execution is enabled."""
        return self._execution.parallel

    def execution_config(self, *, parallel: bool | None = None) -> ExecutionConfig:
        """Return execution settings, optionally overriding parallelism for one call.

        Args:
            parallel: When set, merge a one-shot ``parallel`` override into the backend config.

        Returns:
            Effective :class:`~mqt.yaqs.core.parallel_utils.ExecutionConfig`.
        """
        if parallel is None:
            return self._execution
        return merge_execution_config(self._execution, parallel=parallel)

    def evaluate_probes_weighted(
        self,
        probe_set: ProbeSet,
        *,
        intervention_steps_list: list[list[Any]] | None = None,
        _execution: ExecutionConfig | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate weighted probe responses via traced simulation.

        Args:
            probe_set: Sampled split-cut probes.
            intervention_steps_list: Optional pre-built sequence grid (experiment geometries).
            _execution: Optional one-shot execution override for this evaluation.

        Returns:
            Tuple ``(pauli_xyz_ij, weights_ij)``.
        """
        pauli_xyz, weights_ij, _traces = simulate_exact(
            probe_set=probe_set,
            operator=self.operator,
            sim_params=self.sim_params,
            initial_psi=self.initial_psi,
            parallel=(exec_cfg := _execution or self._execution).parallel,
            show_progress=exec_cfg.show_progress,
            solver=self._solver,
            _execution=exec_cfg,
            intervention_steps_list=intervention_steps_list,
            static_ctx=self._static_ctx,
        )
        return pauli_xyz, weights_ij

    def evaluate_probes(self, probe_set: ProbeSet) -> np.ndarray:
        """Evaluate unweighted Pauli probe responses.

        Args:
            probe_set: Sampled split-cut probes.

        Returns:
            Array of shape ``(n_pasts, n_futures, 4)``.
        """
        pauli_xyz_ij, _weights_ij = self.evaluate_probes_weighted(probe_set)
        return pauli_xyz_ij


def simulate_exact(
    *,
    probe_set: ProbeSet,
    operator: MPO,
    sim_params: AnalogSimParams,
    initial_psi: np.ndarray,
    parallel: bool = True,
    show_progress: bool = False,
    solver: StochasticSolver | None = None,
    _execution: ExecutionConfig | None = None,
    intervention_steps_list: list[list[Any]] | None = None,
    static_ctx: MCWFContext | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    r"""Exact simulation with per-sequence diagnostics (branch weights, early termination).

    Args:
        probe_set: Sampled split-cut probes.
        operator: Hamiltonian MPO.
        sim_params: Analog simulation parameters.
        initial_psi: Initial state vector for sequences.
        parallel: Whether to parallelize sequence simulation.
        show_progress: Whether to show a progress bar.
        solver: Stochastic solver (``"MCWF"`` or ``"TJM"``).
        _execution: Optional internal execution configuration.
        intervention_steps_list: Optional pre-built sequence grid (experiment geometries).
        static_ctx: Optional reusable MCWF static context (built when omitted for MCWF).

    Returns:
        ``(pauli_ij, weights_ij, traces_flat)`` where ``pauli_ij`` has shape
        ``(n_pasts, n_futures, 4)``, ``weights_ij`` holds break weights through cut ``c``,
        and ``traces_flat[i * n_f + j]`` matches the sequence order of the grid.

    Raises:
        TypeError: If the backend output is not an ndarray.
    """
    all_pairs, n_p, n_f = _resolve_sequence_grid(probe_set, intervention_steps_list)
    n_tot = n_p * n_f
    initial_psis = [np.asarray(initial_psi, dtype=np.complex128).copy() for _ in range(n_tot)]
    exec_cfg = merge_execution_config(_execution, parallel=parallel, show_progress=show_progress)
    resolved_solver = validate_stochastic_solver(solver)
    if static_ctx is None and resolved_solver == "MCWF":
        static_ctx = make_mcwf_static_context(operator, sim_params, noise_model=None)
    result = simulate_sequences(
        operator=operator,
        sim_params=sim_params,
        timesteps=[float(sim_params.dt)] * (int(probe_set.num_interventions) + 1),
        intervention_steps_list=all_pairs,
        initial_psis=initial_psis,
        static_ctx=static_ctx,
        parallel=exec_cfg.parallel,
        show_progress=exec_cfg.show_progress,
        record_step_states=False,
        traced=True,
        solver=resolved_solver,
        _execution=exec_cfg,
    )
    if not isinstance(result, tuple):
        msg = "Expected traced simulation output."
        raise TypeError(msg)
    final_packed, traces = result
    if not isinstance(final_packed, np.ndarray):
        msg = "Expected ndarray output from exact simulation."
        raise TypeError(msg)
    pauli_xyz = decode_packed_pauli_batch(final_packed.reshape(n_p * n_f, 8)).reshape(n_p, n_f, 4)
    w = _branch_weights_from_simulation(traces, n_pasts=n_p, n_futures=n_f, cut=int(probe_set.cut))
    return pauli_xyz, w, traces
