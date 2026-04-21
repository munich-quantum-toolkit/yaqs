"""Exact process probing helpers built on the rollout backend."""

from __future__ import annotations

from typing import Any

import numpy as np

from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from ..core.encoding import packed_rho8_to_pauli_xyz_batch
from ..core.utils import make_mcwf_static_context
from ..surrogates.workflow import _simulate_sequences, simulate_final_states_with_diagnostics
from .probe import ProbeSet, build_all_pairs_grid


class ExactProbeProcess:
    """Exact rollout-backed probe process with internal static context."""

    def __init__(
        self,
        *,
        operator: MPO,
        sim_params: AnalogSimParams,
        initial_psi: np.ndarray,
        parallel: bool = True,
    ) -> None:
        self.operator = operator
        self.sim_params = sim_params
        self.initial_psi = np.asarray(initial_psi, dtype=np.complex128).copy()
        self.parallel = bool(parallel)
        self._static_ctx = make_mcwf_static_context(operator, sim_params, noise_model=None)

    def evaluate_probe_set(self, probe_set: ProbeSet) -> np.ndarray:
        """Run exact backend for all (past, future) probe combinations.

        Returns:
            Array of shape ``(n_pasts, n_futures, 3)`` — Pauli :math:`(x,y,z)` expectations
            from the final single-qubit reduced state (see :mod:`~mqt.yaqs.characterization.process_tensors.core.encoding`).
        """
        all_pairs, n_p, n_f = build_all_pairs_grid(probe_set)
        n_tot = n_p * n_f
        initial_psis = [self.initial_psi.copy() for _ in range(n_tot)]
        final_packed = _simulate_sequences(
            operator=self.operator,
            sim_params=self.sim_params,
            timesteps=[float(self.sim_params.dt)] * (int(probe_set.k) + 1),
            psi_pairs_list=all_pairs,
            initial_psis=initial_psis,
            static_ctx=self._static_ctx,
            parallel=self.parallel,
            show_progress=True,
            record_step_states=False,
        )
        if not isinstance(final_packed, np.ndarray):
            raise RuntimeError("Expected ndarray output from exact simulation.")
        if final_packed.shape[0] != n_tot:
            msg = f"Expected {n_tot} final states from exact simulation, got {final_packed.shape[0]}."
            raise RuntimeError(msg)
        xyz = packed_rho8_to_pauli_xyz_batch(final_packed.reshape(n_p * n_f, 8)).reshape(n_p, n_f, 3)
        return xyz.astype(np.float32)


def evaluate_exact_probe_set_with_diagnostics(
    *,
    probe_set: ProbeSet,
    operator: MPO,
    sim_params: AnalogSimParams,
    initial_psi: np.ndarray,
    parallel: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Exact rollout with per-sequence diagnostics (branch weights, early termination).

    Returns:
        ``(pauli_xyz_ij, weights_ij, traces_flat)`` where ``pauli_xyz_ij`` has shape
        ``(n_pasts, n_futures, 3)`` (Pauli expectations :math:`x,y,z` from the final reduced state),
        ``weights_ij[i,j] = cumulative_weight_final``, and ``traces_flat[i * n_f + j]`` matches the
        sequence order of :func:`build_all_pairs_grid`.
    """
    all_pairs, n_p, n_f = build_all_pairs_grid(probe_set)
    n_tot = n_p * n_f
    initial_psis = [np.asarray(initial_psi, dtype=np.complex128).copy() for _ in range(n_tot)]
    final_packed, traces = simulate_final_states_with_diagnostics(
        operator=operator,
        sim_params=sim_params,
        timesteps=[float(sim_params.dt)] * (int(probe_set.k) + 1),
        psi_pairs_list=all_pairs,
        initial_psis=initial_psis,
        static_ctx=make_mcwf_static_context(operator, sim_params, noise_model=None),
        parallel=bool(parallel),
        show_progress=True,
    )
    if not isinstance(final_packed, np.ndarray):
        raise RuntimeError("Expected ndarray output from exact simulation.")
    pauli_xyz = packed_rho8_to_pauli_xyz_batch(final_packed.reshape(n_p * n_f, 8)).reshape(n_p, n_f, 3).astype(np.float32)
    w = np.zeros((n_p, n_f), dtype=np.float64)
    for ii in range(n_p):
        for jj in range(n_f):
            idx = ii * n_f + jj
            w[ii, jj] = float(traces[idx]["cumulative_weight_final"])
    return pauli_xyz, w, traces

