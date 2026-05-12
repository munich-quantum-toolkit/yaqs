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
from ..core.methods.tdvp import local_dynamic_tdvp

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.data_structures.networks import MPO, MPS
    from ..core.data_structures.simulation_parameters import AnalogSimParams


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


def _step_auxiliary_states(
    sim_params: AnalogSimParams,
    hamiltonian: MPO,
    autocorr_phi: MPS | None,
    two_time_phis: list[MPS],
) -> None:
    if autocorr_phi is not None:
        _unitary_step(autocorr_phi, hamiltonian, sim_params)
    for phi_p in two_time_phis:
        _unitary_step(phi_p, hamiltonian, sim_params)


def ensemble_member_worker(
    args: tuple[int, MPS, AnalogSimParams, MPO],
) -> tuple[NDArray[np.float64], NDArray[np.complex128] | None, NDArray[np.complex128] | None]:
    r"""Run one deterministic unitary ensemble member (no stochastic noise).

    When ``sim_params.compute_autocorrelator`` is True, maintains an auxiliary MPS
    :math:`|\phi\rangle = O|\psi\rangle` with the same unitary evolution as :math:`|\psi\rangle` and
    records :math:`\langle\psi(t)|O|\phi(t)\rangle` at sampled times (ensemble drivers average over
    members). This matches the dense convention
    :math:`\langle\psi|U^\dagger O U O|\psi\rangle` when ``O`` is Hermitian.

    When ``sim_params.two_time_correlators`` is non-empty, for each pair :math:`(A, B)` the worker
    builds :math:`|\phi_B\rangle = B|\psi\rangle`, evolves :math:`|\psi\rangle` and each
    :math:`|\phi_B\rangle` with the same unitary propagator, and records
    :math:`\langle\psi(t)|A|\phi_B(t)\rangle` at sampled times.

    Args:
        args:
            ``(member_index, initial_state, sim_params, hamiltonian)``. The index is ignored by the
            evolution but is kept for TJM-style parallel worker signatures.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.complex128] | None, NDArray[np.complex128] | None]:
            ``(observable_results, autocorr_results, two_time_results)``. The latter two entries
            are ``None`` when the corresponding feature is off. ``two_time_results`` has shape
            ``(n_pairs, n_times)`` if ``sample_timesteps`` else ``(n_pairs, 1)``.

    Raises:
        ValueError: If ``compute_autocorrelator`` is True but ``autocorrelator_observable`` is not set.
    """
    _idx, initial_state, sim_params, hamiltonian = args
    state = copy.deepcopy(initial_state)
    last_index = len(sim_params.times) - 1
    pairs = sim_params.two_time_correlators
    n_pairs = len(pairs)

    if sim_params.sample_timesteps:
        observable_results = np.zeros((len(sim_params.sorted_observables), len(sim_params.times)), dtype=np.float64)
    else:
        observable_results = np.zeros((len(sim_params.sorted_observables), 1), dtype=np.float64)

    autocorr_results: NDArray[np.complex128] | None = None
    autocorr_phi: MPS | None = None
    if sim_params.compute_autocorrelator:
        if sim_params.autocorrelator_observable is None:
            msg = "compute_autocorrelator=True requires autocorrelator_observable to be set."
            raise ValueError(msg)

        autocorr_phi = copy.deepcopy(state)
        autocorr_phi.apply_local(sim_params.autocorrelator_observable)
        if sim_params.sample_timesteps:
            autocorr_results = np.zeros(len(sim_params.times), dtype=np.complex128)
        else:
            autocorr_results = np.zeros(1, dtype=np.complex128)

    two_time_phis: list[MPS] = []
    two_time_results: NDArray[np.complex128] | None = None
    if n_pairs > 0:
        if sim_params.sample_timesteps:
            two_time_results = np.zeros((n_pairs, len(sim_params.times)), dtype=np.complex128)
        else:
            two_time_results = np.zeros((n_pairs, 1), dtype=np.complex128)
        for _probe_a, b_op in pairs:
            phi_b = copy.deepcopy(state)
            phi_b.apply_local(b_op)
            two_time_phis.append(phi_b)

    if sim_params.sample_timesteps:
        state.evaluate_observables(sim_params, observable_results, 0)
        if autocorr_results is not None:
            assert autocorr_phi is not None
            assert sim_params.autocorrelator_observable is not None
            autocorr_results[0] = autocorr_phi.mixed_expectation(state, sim_params.autocorrelator_observable)
        if two_time_results is not None:
            for p, (probe_a, _b_op) in enumerate(pairs):
                two_time_results[p, 0] = two_time_phis[p].mixed_expectation(state, probe_a)
    elif last_index == 0:
        state.evaluate_observables(sim_params, observable_results)
        if autocorr_results is not None:
            assert autocorr_phi is not None
            assert sim_params.autocorrelator_observable is not None
            autocorr_results[0] = autocorr_phi.mixed_expectation(state, sim_params.autocorrelator_observable)
        if two_time_results is not None:
            for p, (probe_a, _b_op) in enumerate(pairs):
                two_time_results[p, 0] = two_time_phis[p].mixed_expectation(state, probe_a)

    for j, _ in enumerate(sim_params.times[1:], start=1):
        _unitary_step(state, hamiltonian, sim_params)
        _step_auxiliary_states(sim_params, hamiltonian, autocorr_phi, two_time_phis)

        if sim_params.sample_timesteps:
            state.evaluate_observables(sim_params, observable_results, j)
            if autocorr_results is not None:
                assert autocorr_phi is not None
                assert sim_params.autocorrelator_observable is not None
                autocorr_results[j] = autocorr_phi.mixed_expectation(state, sim_params.autocorrelator_observable)
            if two_time_results is not None:
                for p, (probe_a, _b_op) in enumerate(pairs):
                    two_time_results[p, j] = two_time_phis[p].mixed_expectation(state, probe_a)
        elif j == last_index:
            state.evaluate_observables(sim_params, observable_results)
            if autocorr_results is not None:
                assert autocorr_phi is not None
                assert sim_params.autocorrelator_observable is not None
                autocorr_results[0] = autocorr_phi.mixed_expectation(state, sim_params.autocorrelator_observable)
            if two_time_results is not None:
                for p, (probe_a, _b_op) in enumerate(pairs):
                    two_time_results[p, 0] = two_time_phis[p].mixed_expectation(state, probe_a)

    return observable_results, autocorr_results, two_time_results
