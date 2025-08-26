# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for analog simulation with the Tensor Jump Method.

This module provides unit tests for the analog simulation functions implemented in the
AnalogTJM module. It verifies that the initialization and time evolution routines for
the Tensor Jump Method (TJM) work as expected in various configurations, including both
first and second order evolution schemes, with and without timestep sampling.

The tests cover:
  - Initialization: Ensuring that a half time step of dissipation followed by a stochastic process
    is correctly applied to the initial state.
  - Step-through evolution: Verifying that dynamic_tdvp, apply_dissipation, and stochastic_process
    are called with the proper arguments during a single time step.
  - Analog simulation (order=2): Checking the shape of the results when running a second order evolution,
    with and without sampling timesteps.
  - Analog simulation (order=1): Checking the shape of the results when running a first order evolution,
    with and without sampling timesteps.

These tests ensure that the evolution functions correctly integrate the MPS state under the
specified Hamiltonian and noise model, and that observable measurements are properly aggregated.
"""

# ignore non-lowercase variable names for physics notation
# ruff: noqa: N806

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np

from mqt.yaqs import simulator
from mqt.yaqs.analog.analog_tjm import analog_tjm_1, analog_tjm_2, initialize, step_through
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import X, Z


def test_initialize() -> None:
    """Test that initialize applies a half-time dissipation and then a stochastic process to the MPS.

    This test creates an Ising MPO and an MPS of length 5, along with a minimal NoiseModel and AnalogSimParams.
    It patches the functions apply_dissipation and stochastic_process to ensure that initialize calls them with the
    correct arguments: apply_dissipation should be called with dt/2, and stochastic_process with dt.
    """
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_ising(L, J, g)
    state = MPS(L)
    noise_model = NoiseModel([{"name": "lowering", "sites": [i], "strength": 0.1} for i in range(L)])
    measurements = [Observable(X(), site) for site in range(L)]
    sim_params = AnalogSimParams(
        measurements,
        elapsed_time=0.2,
        dt=0.2,
        sample_timesteps=False,
        num_traj=1,
        max_bond_dim=2,
        threshold=1e-6,
        order=1,
    )
    with (
        patch("mqt.yaqs.analog.analog_tjm.apply_dissipation") as mock_dissipation,
        patch("mqt.yaqs.analog.analog_tjm.stochastic_process") as mock_stochastic_process,
    ):
        initialize(state, noise_model, sim_params)
        mock_dissipation.assert_called_once_with(state, noise_model, sim_params.dt / 2, sim_params)
        mock_stochastic_process.assert_called_once_with(state, noise_model, sim_params.dt, sim_params)


def test_step_through() -> None:
    """Test that step_through calls dynamic_tdvp, apply_dissipation, and stochastic_process with correct arguments.

    This test creates an Ising MPO and an MPS of length 5, along with a minimal NoiseModel and AnalogSimParams.
    It patches dynamic_tdvp, apply_dissipation, and stochastic_process to ensure that step_through calls each of them
    correctly: dynamic_tdvp should be called with the state, H, and sim_params, and both apply_dissipation and
    stochastic_process should be called with dt.
    """
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_ising(L, J, g)
    state = MPS(L)
    noise_model = NoiseModel([{"name": "lowering", "sites": [i], "strength": 0.1} for i in range(L)])
    measurements = [Observable(X(), site) for site in range(L)]
    sim_params = AnalogSimParams(
        measurements,
        elapsed_time=0.2,
        dt=0.2,
        sample_timesteps=False,
        num_traj=1,
        max_bond_dim=2,
        threshold=1e-6,
        order=1,
    )
    with (
        patch("mqt.yaqs.analog.analog_tjm.local_dynamic_tdvp") as mock_dynamic_tdvp,
        patch("mqt.yaqs.analog.analog_tjm.apply_dissipation") as mock_dissipation,
        patch("mqt.yaqs.analog.analog_tjm.stochastic_process") as mock_stochastic_process,
    ):
        step_through(state, H, noise_model, sim_params)
        mock_dynamic_tdvp(state, H, sim_params)
        mock_dissipation.assert_called_once_with(state, noise_model, sim_params.dt, sim_params)
        mock_stochastic_process.assert_called_once_with(state, noise_model, sim_params.dt, sim_params)


def test_analog_tjm_2() -> None:
    """Test the analog_tjm_2 function for a two-site evolution (order=2) without sampling timesteps.

    This test creates an Ising MPO and an MPS of length 5, with no noise model.
    It calls analog_tjm_2 with sim_params configured for order 2 and sample_timesteps False.
    The returned results array should have shape (num_observables, 1).
    """
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_ising(L, J, g)
    state = MPS(L)
    noise_model = None
    measurements = [Observable(Z(), site) for site in range(L)]
    sim_params = AnalogSimParams(
        measurements,
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=False,
        num_traj=1,
        max_bond_dim=4,
        threshold=1e-6,
        order=2,
    )
    args = (0, state, noise_model, sim_params, H)
    results = analog_tjm_2(args)
    assert results.shape == (len(measurements), 1), "Results incorrect shape"


def test_analog_tjm_2_sample_timesteps() -> None:
    """Test the analog_tjm_2 function for a two-site evolution (order=2) with sampling timesteps.

    This test creates an Ising MPO and an MPS of length 5, with no noise model,
    and sim_params with sample_timesteps True. The resulting results array should have shape
    (num_observables, len(sim_params.times)).
    """
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_ising(L, J, g)
    state = MPS(L)
    noise_model = None
    measurements = [Observable(Z(), site) for site in range(L)]
    sim_params = AnalogSimParams(
        measurements,
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        num_traj=1,
        max_bond_dim=4,
        threshold=1e-6,
        order=2,
    )
    args = (0, state, noise_model, sim_params, H)
    results = analog_tjm_2(args)
    assert results.shape == (len(measurements), len(sim_params.times)), "Results incorrect shape"


def test_analog_tjm_1() -> None:
    """Test the analog_tjm_1 function for a one-site evolution (order=1) without sampling timesteps.

    This test creates an Ising MPO and an MPS of length 5, with no noise model,
    and sim_params with order 1 and sample_timesteps False.
    The resulting results array should have shape (num_observables, 1).
    """
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_ising(L, J, g)
    state = MPS(L)
    noise_model = None
    measurements = [Observable(Z(), site) for site in range(L)]
    sim_params = AnalogSimParams(
        measurements,
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=False,
        num_traj=1,
        max_bond_dim=4,
        threshold=1e-6,
        order=1,
    )
    args = (0, state, noise_model, sim_params, H)
    results = analog_tjm_1(args)
    assert results.shape == (len(measurements), 1), "Results incorrect shape"


def test_analog_tjm_1_sample_timesteps() -> None:
    """Test the analog_tjm_1 function for a one-site evolution (order=1) with sampling timesteps.

    This test creates an Ising MPO and an MPS of length 5, with no noise model,
    and sim_params with sample_timesteps True.
    The results array should have shape (num_observables, len(sim_params.times)).
    """
    L = 5
    J = 1
    g = 0.5
    H = MPO()
    H.init_ising(L, J, g)
    state = MPS(L)
    noise_model = None
    measurements = [Observable(Z(), site) for site in range(L)]
    sim_params = AnalogSimParams(
        measurements,
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        num_traj=1,
        max_bond_dim=4,
        threshold=1e-6,
        order=1,
    )
    args = (0, state, noise_model, sim_params, H)
    results = analog_tjm_1(args)
    assert results.shape == (len(measurements), len(sim_params.times)), "Results incorrect shape"


def test_analog_simulation_twositeprocesses() -> None:
    """Test analog simulation with two-site crosstalk processes against QuTiP reference.

    This test simulates a 3-qubit Ising chain with both single-site Pauli-X noise and
    neighboring two-site crosstalk X⊗X noise. It compares the YAQS analog simulation
    results against hardcoded QuTiP master equation results with a tolerance of 0.1
    for all Z observables at all time points.

    The test uses the same parameters as analogTWOSITEcheck.py:
    - L=3, J=1.0, g=0.5
    - gamma_single=0.02, gamma_pair=0.01
    - T=1.0, dt=0.05, num_traj=200
    """
    # Parameters matching analogTWOSITEcheck.py
    L = 3
    J = 1.0
    g = 0.5
    gamma_single = 0.02
    gamma_pair = 0.01
    T = 1.0
    dt = 0.05
    num_traj = 400

    # Setup YAQS simulation
    H = MPO()
    H.init_ising(L, J, g)
    state = MPS(L, state="zeros")
    observables = [Observable(Z(), i) for i in range(L)]
    sim_params = AnalogSimParams(
        observables=observables,
        elapsed_time=T,
        dt=dt,
        num_traj=num_traj,
        max_bond_dim=64,
        min_bond_dim=2,
        threshold=1e-10,
        order=2,
        sample_timesteps=True,
        get_state=False,
    )

    # Setup noise model with single-site and two-site crosstalk processes
    processes = [{"name": "pauli_x", "sites": [i], "strength": gamma_single} for i in range(L)]
    processes += [{"name": "crosstalk_xx", "sites": [i, i + 1], "strength": gamma_pair} for i in range(L - 1)]
    noise = NoiseModel(processes)

    # Run simulation
    simulator.run(state, H, sim_params, noise, parallel=True)

    # Hardcoded QuTiP reference solution
    expected_results = [
        [
            1.0,
            0.9957595387085492,
            0.989068537861746,
            0.9799951962331045,
            0.9686368930494625,
            0.9551186308467075,
            0.9395910913283494,
            0.9222283389300958,
            0.9032252097089084,
            0.8827944277523293,
            0.8611634997325803,
            0.8385714280891042,
            0.8152653063954242,
            0.7914968423015236,
            0.7675188651840964,
            0.7435818712832375,
            0.7199306579260094,
            0.6968010984985046,
            0.6744171032041366,
            0.6529878097650909,
            0.6327050454858952,
        ],
        [
            1.0,
            0.9947673850218341,
            0.9871414981439525,
            0.9773043133235998,
            0.9655275369620003,
            0.9521575556050013,
            0.9375965200751353,
            0.9222806564841823,
            0.9066570069612527,
            0.8911598843450562,
            0.8761883965926414,
            0.8620860664049642,
            0.8491239779302621,
            0.8374879983781462,
            0.8272709723219848,
            0.8184701569889284,
            0.8109899797799391,
            0.8046500024603839,
            0.7991974939190101,
            0.7943239338217639,
            0.7896845077283686,
        ],
        [
            1.0,
            0.9957595387085492,
            0.989068537861746,
            0.9799951962331045,
            0.9686368930494625,
            0.9551186308467075,
            0.9395910913283494,
            0.9222283389300958,
            0.9032252097089086,
            0.8827944277523293,
            0.8611634997325803,
            0.838571428089104,
            0.8152653063954242,
            0.7914968423015236,
            0.7675188651840964,
            0.7435818712832375,
            0.7199306579260094,
            0.6968010984985046,
            0.6744171032041368,
            0.6529878097650911,
            0.6327050454858952,
        ],
    ]

    # Collect YAQS results
    yaqs_results = np.zeros((L, len(sim_params.times)))
    for _idx, obs in enumerate(sim_params.sorted_observables):
        site = obs.sites if isinstance(obs.sites, int) else obs.sites[0]
        yaqs_results[site, :] = obs.results

    # Compare with tolerance of 0.1
    tolerance = 0.1
    for site in range(L):
        for time_idx in range(len(sim_params.times)):
            expected = expected_results[site][time_idx]
            actual = yaqs_results[site, time_idx]
            assert abs(expected - actual) < tolerance, (
                f"Site {site}, time {sim_params.times[time_idx]:.2f}: "
                f"expected {expected:.6f}, got {actual:.6f}, diff {abs(expected - actual):.6f}"
            )


def test_analog_simulation_two_site_lowering_against_qutip() -> None:
    """Analog simulation with single-site and two-site lowering against QuTiP.

    This test simulates a 3-qubit Ising chain with both single-site lowering (sigma-)
    and adjacent two-site lowering (sigma- x sigma-) noise processes. It compares YAQS
    analog simulation results to a hardcoded QuTiP master-equation reference
    with a tolerance of 0.1 across all Z observables and time points. The setup
    matches the reference parameters: L=3, J=1.0, g=0.5, gamma_single=0.02,
    gamma_pair=0.01, T=1.0, dt=0.05, and num_traj=200.
    """
    # Setup YAQS simulation (same parameters as reference)
    L = 3
    H = MPO()
    H.init_ising(L, 1.0, 0.5)
    state = MPS(L, state="zeros")
    observables = [Observable(Z(), i) for i in range(L)]
    sim_params = AnalogSimParams(
        observables=observables,
        elapsed_time=1.0,
        dt=0.05,
        num_traj=200,
        max_bond_dim=64,
        min_bond_dim=2,
        threshold=1e-10,
        order=2,
        sample_timesteps=True,
        get_state=False,
    )

    gamma_single = 0.02
    gamma_pair = 0.01
    # Noise model with single-site lowering and adjacent two-site lowering
    processes: list[dict[str, Any]] = [{"name": "lowering", "sites": [i], "strength": gamma_single} for i in range(L)]
    processes += [{"name": "lowering_two", "sites": [i, i + 1], "strength": gamma_pair} for i in range(L - 1)]
    noise = NoiseModel(processes)

    # Run simulation
    simulator.run(state, H, sim_params, noise, parallel=True)

    # Collect YAQS results per site (site-major layout)
    yaqs_results = np.zeros((L, len(sim_params.times)))
    for _idx, obs in enumerate(sim_params.sorted_observables):
        site = obs.sites if isinstance(obs.sites, int) else obs.sites[0]
        yaqs_results[site, :] = obs.results

    expected = np.asarray([
        [
            1.0000000000000000,
            0.9987519272263703,
            0.9950257733838441,
            0.9888717325009401,
            0.9803701315747982,
            0.9696302545838585,
            0.9567887490974905,
            0.9420076418328850,
            0.9254719943623211,
            0.9073872362346228,
            0.8879762177723114,
            0.8674760301377559,
            0.8461346407370148,
            0.8242074001985631,
            0.8019534724965074,
            0.7796322463493451,
            0.7574997830044093,
            0.7358053541584250,
            0.7147881235851911,
            0.6946740230770052,
            0.6756728658483249,
        ],
        [
            1.0000000000000000,
            0.9987550471745019,
            0.9950752625974710,
            0.9891191847065749,
            0.9811389736390038,
            0.9714667942643668,
            0.9604969258940412,
            0.9486646038289549,
            0.9364227534930116,
            0.9242179032576668,
            0.9124666118045819,
            0.9015336863444420,
            0.8917134095304454,
            0.8832147918831563,
            0.8761516472885322,
            0.8705380650825371,
            0.8662894560296098,
            0.8632291147594634,
            0.8610999682102124,
            0.8595808205461875,
            0.8583061459455201,
        ],
        [
            1.0000000000000000,
            0.9987519272263703,
            0.9950257733838441,
            0.9888717325009401,
            0.9803701315747982,
            0.9696302545838585,
            0.9567887490974905,
            0.9420076418328850,
            0.9254719943623211,
            0.9073872362346228,
            0.8879762177723114,
            0.8674760301377559,
            0.8461346407370148,
            0.8242074001985631,
            0.8019534724965072,
            0.7796322463493451,
            0.7574997830044091,
            0.7358053541584247,
            0.7147881235851911,
            0.6946740230770052,
            0.6756728658483246,
        ],
    ])
    assert expected.shape == yaqs_results.shape

    # Use same tolerance as existing two-site crosstalk test for trajectory agreement
    tolerance = 0.1
    assert np.allclose(yaqs_results, expected, atol=tolerance), (
        f"Max abs diff {np.max(np.abs(yaqs_results - expected))} exceeds {tolerance}"
    )
