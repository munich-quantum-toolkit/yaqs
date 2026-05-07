# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unitary ensemble evolution helpers for analog simulations."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from ..analog.autocorrelator import apply_observable_inplace, mixed_expectation
from ..core.data_structures.simulation_parameters import EvolutionMode
from ..core.methods.bug import bug
from ..core.methods.tdvp import local_dynamic_tdvp

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.data_structures.networks import MPO, MPS
    from ..core.data_structures.simulation_parameters import AnalogSimParams, Observable


def _unitary_step(state: MPS, hamiltonian: MPO, sim_params: AnalogSimParams) -> None:
    """Advance one unitary time step according to ``sim_params.evolution_mode`` (TDVP or BUG).

    Args:
        state (MPS): MPS to evolve in-place.
        hamiltonian (MPO): Hamiltonian as an MPO.
        sim_params (AnalogSimParams): Analog simulation parameters (time step, bond limits, etc.).
    """
    if sim_params.evolution_mode == EvolutionMode.TDVP:
        local_dynamic_tdvp(state, hamiltonian, sim_params)
    elif sim_params.evolution_mode == EvolutionMode.BUG:
        bug(state, hamiltonian, sim_params)


def unitary_ensemble_member_worker(
    args: tuple[int, MPS, AnalogSimParams, MPO],
) -> tuple[NDArray[np.float64], NDArray[np.complex128] | None]:
    """Run one deterministic unitary ensemble member (no stochastic noise).

    When ``sim_params.compute_autocorrelator`` is True, maintains an auxiliary MPS
    :math:`|\phi\rangle = O|\psi\rangle` with the same unitary evolution as :math:`|\psi\rangle` and
    records :math:`\langle\psi(t)|O|\phi(t)\rangle` at sampled times (ensemble drivers average over
    members). This matches the dense convention
    :math:`\langle\psi|U^\dagger O U O|\psi\rangle` when ``O`` is Hermitian.

    Args:
        args:
            ``(member_index, initial_state, sim_params, hamiltonian)``. The index is ignored by the
            evolution but is kept for TJM-style parallel worker signatures.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.complex128] | None]:
            Pair ``(observable_results, autocorr_results)`` where ``observable_results`` has shape
            ``(n_obs, n_times)`` if ``sample_timesteps`` else ``(n_obs, 1)``, and
            ``autocorr_results`` is ``None`` or a 1D array of length ``n_times`` or ``1`` matching
            the sampling mode.

    Raises:
        ValueError: If ``compute_autocorrelator`` is True but ``autocorrelator_observable`` is not set.
    """
    _idx, initial_state, sim_params, hamiltonian = args
    state = copy.deepcopy(initial_state)
    last_index = len(sim_params.times) - 1

    if sim_params.sample_timesteps:
        observable_results = np.zeros((len(sim_params.sorted_observables), len(sim_params.times)), dtype=np.float64)
    else:
        observable_results = np.zeros((len(sim_params.sorted_observables), 1), dtype=np.float64)

    autocorr_results: NDArray[np.complex128] | None = None
    phi: MPS | None = None
    if sim_params.compute_autocorrelator:
        if sim_params.autocorrelator_observable is None:
            msg = "compute_autocorrelator=True requires autocorrelator_observable to be set."
            raise ValueError(msg)

        phi = copy.deepcopy(state)
        apply_observable_inplace(phi, sim_params.autocorrelator_observable)
        if sim_params.sample_timesteps:
            autocorr_results = np.zeros(len(sim_params.times), dtype=np.complex128)
        else:
            autocorr_results = np.zeros(1, dtype=np.complex128)

    if sim_params.sample_timesteps:
        state.evaluate_observables(sim_params, observable_results, 0)
        if autocorr_results is not None:
            assert phi is not None
            assert sim_params.autocorrelator_observable is not None
            autocorr_results[0] = mixed_expectation(state, phi, sim_params.autocorrelator_observable)

    for j, _ in enumerate(sim_params.times[1:], start=1):
        _unitary_step(state, hamiltonian, sim_params)
        if phi is not None:
            _unitary_step(phi, hamiltonian, sim_params)

        if sim_params.sample_timesteps:
            state.evaluate_observables(sim_params, observable_results, j)
            if autocorr_results is not None:
                assert phi is not None
                assert sim_params.autocorrelator_observable is not None
                autocorr_results[j] = mixed_expectation(state, phi, sim_params.autocorrelator_observable)
        elif j == last_index:
            state.evaluate_observables(sim_params, observable_results)
            if autocorr_results is not None:
                assert phi is not None
                assert sim_params.autocorrelator_observable is not None
                autocorr_results[0] = mixed_expectation(state, phi, sim_params.autocorrelator_observable)

    return observable_results, autocorr_results

