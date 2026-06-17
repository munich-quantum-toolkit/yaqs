# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Ensemble evolution helpers for analog simulations."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from ..core.data_structures.simulation_parameters import EvolutionMode
from ..core.methods.bug import bug
from ..core.methods.tdvp import tdvp

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.data_structures.mpo import MPO
    from ..core.data_structures.mps import MPS
    from ..core.data_structures.simulation_parameters import AnalogSimParams


def _unitary_step(state: MPS, hamiltonian: MPO, sim_params: AnalogSimParams) -> None:
    """Advance one unitary time step according to ``sim_params.evolution_mode`` (TDVP or BUG).

    Args:
        state (MPS): MPS to evolve in-place.
        hamiltonian (MPO): Hamiltonian as an MPO.
        sim_params (AnalogSimParams): Analog simulation parameters (time step, bond limits, etc.).
    """
    if sim_params.evolution_mode == EvolutionMode.TDVP:
        tdvp(state, hamiltonian, sim_params)
    elif sim_params.evolution_mode == EvolutionMode.BUG:
        bug(state, hamiltonian, sim_params)


def _step_correlator_phis(
    sim_params: AnalogSimParams,
    hamiltonian: MPO,
    phis: list[MPS],
) -> None:
    """Advance all ``multi_time_observables`` auxiliary MPS states one unitary step.

    Args:
        sim_params: Analog simulation parameters.
        hamiltonian: Hamiltonian MPO shared with the primary state.
        phis: One auxiliary state per ``(A, B)`` pair (each ``B|psi⟩``).
    """
    for phi in phis:
        _unitary_step(phi, hamiltonian, sim_params)


def ensemble_member_worker(
    args: tuple[int, MPS, AnalogSimParams, MPO],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128] | None]:
    r"""Run one deterministic unitary ensemble member (no stochastic noise).

    For each pair ``(A, B)`` in ``sim_params.multi_time_observables`` the worker builds
    :math:`|\phi_B\rangle = B|\psi(0)\rangle`, evolves :math:`|\psi\rangle` and each
    :math:`|\phi_B\rangle` with the same unitary propagator, and records
    :math:`\langle\psi(t)|A|\phi_B(t)\rangle` at sampled times.

    The autocorrelation :math:`C_{OO}(t)` is obtained by passing ``(O, O)`` as a pair.

    Args:
        args:
            ``(member_index, initial_state, sim_params, hamiltonian)``. The index is ignored by
            the evolution but is kept for TJM-style parallel worker signatures.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128] | None]:
            ``(observable_results, diagnostics, multi_time_results)``. ``multi_time_results`` is ``None``
            when ``sim_params.multi_time_observables`` is empty; otherwise it has shape
            ``(n_pairs, n_times)`` if ``sample_timesteps`` else ``(n_pairs, 1)``.
    """
    _idx, initial_state, sim_params, hamiltonian = args
    state = copy.deepcopy(initial_state)
    last_index = len(sim_params.times) - 1
    pairs = sim_params.multi_time_observables
    n_pairs = len(pairs)

    num_cols = len(sim_params.times) if sim_params.sample_timesteps else 1
    diagnostics = np.zeros((3, num_cols), dtype=np.float64)
    if sim_params.sample_timesteps:
        observable_results = np.zeros((len(sim_params.sorted_observables), len(sim_params.times)), dtype=np.float64)
    else:
        observable_results = np.zeros((len(sim_params.sorted_observables), 1), dtype=np.float64)

    phis: list[MPS] = []
    multi_time_results: NDArray[np.complex128] | None = None
    if n_pairs > 0:
        if sim_params.sample_timesteps:
            multi_time_results = np.zeros((n_pairs, len(sim_params.times)), dtype=np.complex128)
        else:
            multi_time_results = np.zeros((n_pairs, 1), dtype=np.complex128)
        for _probe_a, b_op in pairs:
            phi_b = copy.deepcopy(state)
            phi_b.apply_local(b_op)
            phis.append(phi_b)

    if sim_params.sample_timesteps:
        state.record_diagnostics(diagnostics, 0)
        state.evaluate_observables(sim_params, observable_results, 0)
        if multi_time_results is not None:
            for p, (probe_a, _b_op) in enumerate(pairs):
                multi_time_results[p, 0] = phis[p].mixed_expectation(state, probe_a)
    elif last_index == 0:
        state.record_diagnostics(diagnostics, 0)
        state.evaluate_observables(sim_params, observable_results)
        if multi_time_results is not None:
            for p, (probe_a, _b_op) in enumerate(pairs):
                multi_time_results[p, 0] = phis[p].mixed_expectation(state, probe_a)

    for j, _ in enumerate(sim_params.times[1:], start=1):
        _unitary_step(state, hamiltonian, sim_params)
        _step_correlator_phis(sim_params, hamiltonian, phis)

        if sim_params.sample_timesteps:
            state.record_diagnostics(diagnostics, j)
            state.evaluate_observables(sim_params, observable_results, j)
            if multi_time_results is not None:
                for p, (probe_a, _b_op) in enumerate(pairs):
                    multi_time_results[p, j] = phis[p].mixed_expectation(state, probe_a)
        elif j == last_index:
            state.record_diagnostics(diagnostics, 0)
            state.evaluate_observables(sim_params, observable_results)
            if multi_time_results is not None:
                for p, (probe_a, _b_op) in enumerate(pairs):
                    multi_time_results[p, 0] = phis[p].mixed_expectation(state, probe_a)

    return observable_results, diagnostics, multi_time_results
