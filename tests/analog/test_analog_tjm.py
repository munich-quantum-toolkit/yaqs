# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
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
  - Lowering noise: TJM jump probabilities and ensemble observables must agree with MCWF.

These tests ensure that the evolution functions correctly integrate the MPS state under the
specified Hamiltonian and noise model, and that observable measurements are properly aggregated.
"""

# ignore non-lowercase variable names for physics notation
# ruff:file-ignore[non-lowercase-variable-in-function]

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np
import pytest

from mqt.yaqs import (
    AnalogSimParams,
    Hamiltonian,
    NoiseModel,
    Observable,
    Simulator,
    State,
)
from mqt.yaqs.analog import analog_tjm as analog_module
from mqt.yaqs.analog.analog_tjm import analog_tjm_1, analog_tjm_2, initialize, step_through
from mqt.yaqs.analog.mcwf import MCWFContext, preprocess_mcwf
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import EvolutionMode
from mqt.yaqs.core.libraries.gate_library import X, Z
from mqt.yaqs.core.methods.dissipation import apply_dissipation
from mqt.yaqs.core.methods.stochastic_process import calculate_stochastic_factor, stochastic_process
from mqt.yaqs.core.methods.tdvp.tdvp import tdvp
from tests.conftest import YAQS_TEST_SEED

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


def test_initialize() -> None:
    """Test that initialize applies a half-time dissipation and then a stochastic process to the MPS.

    This test creates an Ising MPO and an MPS of length 5, along with a minimal NoiseModel and AnalogSimParams.
    It patches the functions apply_dissipation and stochastic_process to ensure that initialize calls them with the
    correct arguments: apply_dissipation should be called with dt/2, and stochastic_process with dt.
    """
    L = 5
    J = 1
    g = 0.5
    MPO.ising(L, J, g)

    state = MPS(L)
    noise_model = NoiseModel([{"name": "lowering", "sites": [i], "strength": 0.1} for i in range(L)])
    sim_params = AnalogSimParams(
        observables=[Observable(X(), site) for site in range(L)],
        elapsed_time=0.2,
        dt=0.2,
        num_traj=1,
        max_bond_dim=2,
        sample_timesteps=False,
    )
    with (
        patch("mqt.yaqs.analog.analog_tjm.apply_dissipation") as mock_dissipation,
        patch("mqt.yaqs.analog.analog_tjm.stochastic_process") as mock_stochastic_process,
    ):
        initialize(state, noise_model, sim_params)
        mock_dissipation.assert_called_once_with(state, noise_model, sim_params.dt / 2, sim_params)
        mock_stochastic_process.assert_called_once_with(state, noise_model, sim_params.dt, sim_params, rng=None)


def test_step_through() -> None:
    """Test that step_through calls unitary evolution, dissipation, and stochastic_process.

    This test creates an Ising MPO and an MPS of length 5, along with a minimal NoiseModel and AnalogSimParams.
    It patches apply_unitary_evolution, apply_dissipation, and stochastic_process to ensure that step_through
    calls each of them correctly.
    """
    L = 5
    J = 1
    g = 0.5
    H = MPO.ising(L, J, g)

    state = MPS(L)
    noise_model = NoiseModel([{"name": "lowering", "sites": [i], "strength": 0.1} for i in range(L)])
    sim_params = AnalogSimParams(
        observables=[Observable(X(), site) for site in range(L)],
        elapsed_time=0.2,
        dt=0.2,
        num_traj=1,
        max_bond_dim=2,
        sample_timesteps=False,
    )
    with (
        patch("mqt.yaqs.analog.analog_tjm.apply_unitary_evolution") as mock_unitary,
        patch("mqt.yaqs.analog.analog_tjm.apply_dissipation") as mock_dissipation,
        patch("mqt.yaqs.analog.analog_tjm.stochastic_process") as mock_stochastic_process,
    ):
        step_through(state, H, noise_model, sim_params, current_time=0.2)
        mock_unitary.assert_called_once_with(state, H, sim_params)
        mock_dissipation.assert_called_once_with(state, noise_model, sim_params.dt, sim_params)
        mock_stochastic_process.assert_called_once_with(state, noise_model, sim_params.dt, sim_params, rng=None)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("sample_timesteps", [False, True])
def test_analog_tjm_shape_via_simulator(order: int, *, sample_timesteps: bool) -> None:
    """Simulator-driven analog TJM produces per-observable trajectories of the expected shape.

    Covers both one-site (order=1) and two-site (order=2) evolution, with and without
    intermediate time sampling. Per-trajectory rows on ``result.trajectories[i]`` must
    have one column when ``sample_timesteps=False`` and ``len(sim_params.times)``
    columns otherwise.
    """
    length = 5
    state = State(length, initial="zeros")
    hamiltonian = Hamiltonian.ising(length, J=1.0, g=0.5)
    observables = [Observable(Z(), site) for site in range(length)]
    sim_params = AnalogSimParams(
        observables=observables,
        elapsed_time=0.2,
        dt=0.2,
        num_traj=1,
        max_bond_dim=2,
        order=order,
        sample_timesteps=sample_timesteps,
    )

    result = Simulator(parallel=False, show_progress=False).run(state, hamiltonian, sim_params)

    expected_cols = len(sim_params.times) if sample_timesteps else 1
    assert result.expectation_values is not None
    assert result.trajectories is not None
    for traj in result.trajectories:
        assert traj.shape == (sim_params.num_traj, expected_cols)


@pytest.mark.parametrize("two_site_process", ["crosstalk_xx", "lowering_two"])
def test_analog_two_site_jump_operators_smoke(two_site_process: str) -> None:
    """Smoke test: analog TJM runs with single-site plus one adjacent two-site jump process.

    Replaces former QuTiP golden-trajectory integration tests; keeps both crosstalk and
    lowering_two library names exercised at minimal cost.
    """
    length = 2
    hamiltonian = Hamiltonian.ising(length, 1.0, 0.5)
    state = State(length, initial="zeros")
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
        num_traj=20,
        max_bond_dim=8,
        order=2,
        sample_timesteps=False,
        random_seed=YAQS_TEST_SEED,
    )
    noise = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 0.02},
        {"name": two_site_process, "sites": [0, 1], "strength": 0.01},
    ])
    result = Simulator(parallel=False, show_progress=False).run(state, hamiltonian, sim_params, noise)

    results = result.expectation_values[0]
    assert results is not None
    z_mean = np.real(results)
    assert np.isfinite(z_mean).all()
    assert np.all(np.abs(z_mean) <= 1.0 + 1e-6)


class _NeverJumpRng:
    """RNG stub that always takes the no-jump branch in ``stochastic_process``."""

    @staticmethod
    def random() -> float:
        return 1.0


def _lowering_noise_model(length: int, *, gamma: float = 1.0) -> NoiseModel:
    return NoiseModel([{"name": "lowering", "sites": [i], "strength": gamma} for i in range(length)])


def _zero_hamiltonian_mpo(length: int) -> MPO:
    mpo = MPO.identity(length)
    for i in range(len(mpo.tensors)):
        mpo.tensors[i] *= 0.0
    return mpo


def _mcwf_jump_probability(state: MPS, ctx: MCWFContext) -> float:
    """Reference jump probability for one MCWF step from the current MPS.

    Returns:
        Jump probability ``1 - ||exp(-i H_eff dt) |psi>||^2`` for the given state.
    """
    psi: NDArray[np.complex128] = state.to_vec()
    psi /= np.linalg.norm(psi)
    assert ctx.step_propagator is not None, "Test systems must fit the dense MCWF propagator."
    psi_next = ctx.step_propagator @ psi
    return float(1.0 - np.vdot(psi_next, psi_next).real)


def _tjm_jump_probability_after_dissipation(
    state: MPS,
    noise_model: NoiseModel,
    dt: float,
    sim_params: AnalogSimParams,
) -> float:
    dissipated = copy.deepcopy(state)
    apply_dissipation(dissipated, noise_model, dt, sim_params)
    assert dissipated.orthogonality_center == 0
    return float(calculate_stochastic_factor(dissipated))


def _entangled_after_tdvp(*, length: int, dt: float) -> MPS:
    state = MPS(length, state="ones")
    hamiltonian = MPO.ising(length, J=1.0, g=0.5)
    sim_params = AnalogSimParams(elapsed_time=0.0, dt=dt, max_bond_dim=64, svd_threshold=1e-10)
    tdvp(state, hamiltonian, sim_params)
    return state


@pytest.mark.parametrize(
    "prepare_state",
    [
        pytest.param(lambda: MPS(5, state="ones"), id="product_all_excited"),
        pytest.param(lambda: _entangled_after_tdvp(length=5, dt=0.05), id="entangled_ising_tdvp"),
    ],
)
def test_lowering_jump_probability_matches_mcwf_after_dissipation(
    prepare_state: Callable[[], MPS],
) -> None:
    """TJM dissipative norm loss must match the MCWF jump probability for lowering noise."""
    length = 5
    dt = 0.05
    state = prepare_state()
    hamiltonian = _zero_hamiltonian_mpo(length)
    noise_model = _lowering_noise_model(length)
    sim_params = AnalogSimParams(elapsed_time=0.0, dt=dt, max_bond_dim=64, svd_threshold=1e-10)
    ctx = preprocess_mcwf(
        psi_initial=state.to_vec(),
        h_sparse=hamiltonian.to_sparse_matrix(),
        noise_model=noise_model,
        sim_params=sim_params,
        num_sites=state.length,
        physical_dimensions=state.physical_dimensions,
    )

    p_mcwf = _mcwf_jump_probability(state, ctx)
    p_tjm = _tjm_jump_probability_after_dissipation(state, noise_model, dt, sim_params)

    np.testing.assert_allclose(p_tjm, p_mcwf, rtol=0.0, atol=5e-4)


@pytest.mark.parametrize(
    "hamiltonian_factory",
    [
        pytest.param(_zero_hamiltonian_mpo, id="h_zero"),
    ],
)
def test_lowering_jump_probability_stable_over_repeated_no_jump_steps(
    hamiltonian_factory: Callable[[int], MPO],
) -> None:
    """Jump probabilities must stay aligned with MCWF across many dissipative substeps."""
    length = 5
    dt = 0.05
    n_steps = 10

    hamiltonian = hamiltonian_factory(length)
    noise_model = _lowering_noise_model(length)
    sim_params = AnalogSimParams(elapsed_time=0.0, dt=dt, max_bond_dim=64, svd_threshold=1e-10)
    initial = MPS(length, state="ones")
    ctx = preprocess_mcwf(
        psi_initial=initial.to_vec(),
        h_sparse=hamiltonian.to_sparse_matrix(),
        noise_model=noise_model,
        sim_params=sim_params,
        num_sites=initial.length,
        physical_dimensions=initial.physical_dimensions,
    )

    state = MPS(length, state="ones")
    state.normalize("B")
    never_jump = _NeverJumpRng()

    for _ in range(n_steps):
        p_mcwf = _mcwf_jump_probability(state, ctx)
        p_tjm = _tjm_jump_probability_after_dissipation(state, noise_model, dt, sim_params)
        np.testing.assert_allclose(p_tjm, p_mcwf, rtol=0.0, atol=1e-3)

        apply_dissipation(state, noise_model, dt, sim_params)
        stochastic_process(state, noise_model, dt, sim_params, rng=cast("np.random.Generator", never_jump))
        state.normalize("B")


def test_tjm_and_mcwf_lowering_mean_excitation_agreement() -> None:
    """End-to-end smoke: ensemble-averaged excitation density from TJM and MCWF should agree.

    The per-step jump-probability checks above are the primary regression guard; this
    single short trajectory ensemble only verifies that the full solvers stay consistent.
    """
    length = 5
    gamma = 1.0
    t_max = 1.5
    dt = 0.1
    num_traj = 20

    hamiltonian = Hamiltonian.ising(length, J=0.0, g=0.0)
    noise_model = _lowering_noise_model(length, gamma=gamma)
    observables = [Observable("z", sites=i) for i in range(length)]

    sim = Simulator(parallel=True, show_progress=False)

    tjm_params = AnalogSimParams(
        observables=observables,
        elapsed_time=t_max,
        dt=dt,
        num_traj=num_traj,
        random_seed=YAQS_TEST_SEED,
        max_bond_dim=64,
        svd_threshold=1e-10,
        order=1,
    )
    mcwf_params = AnalogSimParams(
        observables=observables,
        elapsed_time=t_max,
        dt=dt,
        num_traj=num_traj,
        random_seed=YAQS_TEST_SEED,
        max_bond_dim=64,
        svd_threshold=1e-10,
    )

    tjm_result = sim.run(
        State(length, initial="ones", representation="mps"),
        hamiltonian,
        tjm_params,
        noise_model,
    )
    mcwf_result = sim.run(
        State(length, initial="ones", representation="vector"),
        hamiltonian,
        mcwf_params,
        noise_model,
    )

    n_tjm = np.mean([(1.0 - z) / 2.0 for z in tjm_result.expectation_values], axis=0)
    n_mcwf = np.mean([(1.0 - z) / 2.0 for z in mcwf_result.expectation_values], axis=0)

    # v0.5.0 showed ~0.03 excess excitation in TJM at t=2; keep tolerance below that.
    mean_abs_diff = float(np.mean(np.abs(n_tjm - n_mcwf)))
    assert mean_abs_diff < 0.03, f"mean |n_TJM - n_MCWF| = {mean_abs_diff:.4f}"


def test_analog_tjm_1_dispatches_bug(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default order-1 TJM honors EvolutionMode.BUG for each unitary interval."""
    calls = {"bug": 0, "tdvp": 0}

    def fake_bug(state: MPS, _hamiltonian: MPO, _sim_params: AnalogSimParams, *, normalize: bool = True) -> None:
        del normalize
        calls["bug"] += 1
        state.set_center(0)

    def fake_tdvp(_state: MPS, _hamiltonian: MPO, _sim_params: AnalogSimParams) -> None:
        calls["tdvp"] += 1

    monkeypatch.setattr("mqt.yaqs.analog.evolution.bug", fake_bug)
    monkeypatch.setattr("mqt.yaqs.analog.evolution.tdvp", fake_tdvp)

    length = 3
    state = MPS(length, state="zeros")
    state.set_canonical_form(0)
    hamiltonian = MPO.ising(length, 1.0, 0.5)
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), site) for site in range(length)],
        elapsed_time=0.2,
        dt=0.1,
        num_traj=1,
        order=1,
        evolution_mode=EvolutionMode.BUG,
        sample_timesteps=True,
        max_bond_dim=4,
    )
    analog_tjm_1((0, state, None, sim_params, hamiltonian))
    assert calls["bug"] == 2
    assert calls["tdvp"] == 0


def test_simulator_order1_honors_bug_evolution_mode() -> None:
    """Ordinary single-State default-order Simulator runs invoke BUG when requested."""
    length = 3
    state = State(length, initial="zeros")
    hamiltonian = Hamiltonian.ising(length, J=1.0, g=0.5)
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), site) for site in range(length)],
        elapsed_time=0.1,
        dt=0.1,
        num_traj=1,
        order=1,
        evolution_mode=EvolutionMode.BUG,
        max_bond_dim=8,
        sample_timesteps=True,
    )
    result = Simulator(parallel=False, show_progress=False).run(state, hamiltonian, sim_params)
    assert result.expectation_values is not None
    assert result.expectation_values[0].shape == (len(sim_params.times),)


def test_analog_tjm_1_uses_operator_for_each_interval() -> None:
    """Order-1 TJM applies the matching MPO on each analog interval."""
    first = MPO.ising(2, 1.0, 0.5)
    second = MPO.ising(2, 1.0, 2.0)
    state = MPS(2)
    sim_params = AnalogSimParams(
        observables=[Observable("z", 0)],
        elapsed_time=0.2,
        dt=0.1,
        order=1,
        sample_timesteps=False,
        num_traj=1,
    )
    seen: list[MPO] = []

    def capture(_state: MPS, hamiltonian: MPO, _sim_params: AnalogSimParams) -> None:
        seen.append(hamiltonian)

    with patch("mqt.yaqs.analog.analog_tjm.apply_unitary_evolution", side_effect=capture):
        analog_tjm_1((0, state, None, sim_params, (first, second)))
    assert seen == [first, second]


def test_analog_tjm_2_uses_operator_for_each_interval() -> None:
    """Order-2 TJM uses each interval MPO for evolution and measurement."""
    first = MPO.ising(2, 1.0, 0.5)
    second = MPO.ising(2, 1.0, 1.0)
    third = MPO.ising(2, 1.0, 2.0)
    state = MPS(2)
    sim_params = AnalogSimParams(
        observables=[Observable("z", 0)],
        elapsed_time=0.3,
        dt=0.1,
        order=2,
        sample_timesteps=True,
        num_traj=1,
    )
    with (
        patch.object(
            analog_module,
            "step_through",
            side_effect=lambda current_state, *_args, **_kwargs: current_state,
        ) as mock_step,
        patch.object(analog_module, "sample", return_value=None) as mock_sample,
    ):
        analog_tjm_2((0, state, None, sim_params, (first, second, third)))

    assert [call.args[1] for call in mock_step.call_args_list] == [first, second]
    assert [call.args[1] for call in mock_sample.call_args_list] == [first, second, third]


def test_analog_tjm_2_sample_at_measures_only_requested_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    """sample_at records order-2 measurement copies only at the selected times."""
    sampled: list[int] = []
    original = analog_module.sample

    def record_sample(
        phi: MPS,
        hamiltonian: MPO,
        noise_model: NoiseModel | None,
        sim_params: AnalogSimParams,
        results: NDArray[np.float64],
        j: int,
        rng: np.random.Generator | None = None,
        diagnostics: NDArray[np.float64] | None = None,
    ) -> MPS | None:
        sampled.append(j)
        return original(phi, hamiltonian, noise_model, sim_params, results, j, rng=rng, diagnostics=diagnostics)

    monkeypatch.setattr(analog_module, "sample", record_sample)
    sim_params = AnalogSimParams(
        observables=[Observable("z", 0)],
        elapsed_time=0.4,
        dt=0.1,
        order=2,
        sample_timesteps=True,
        get_state=True,
        num_traj=1,
    )
    analog_tjm_2((0, MPS(1), None, sim_params, MPO.ising(1, 0.0, 0.0)), sample_at=(2, 4))
    assert sampled == [2, 4]


def test_analog_tjm_1_sample_at_writes_only_selected_columns() -> None:
    """Order-1 sample_at fills selected columns and leaves the others untouched."""
    hamiltonian = MPO.ising(1, 0.0, 1.0)
    sim_params = AnalogSimParams(
        observables=[Observable("z", 0)],
        elapsed_time=0.4,
        dt=0.1,
        order=1,
        sample_timesteps=True,
        num_traj=1,
    )
    selected, _, _ = analog_tjm_1((0, MPS(1), None, sim_params, hamiltonian), sample_at=(1, 3))
    full, _, _ = analog_tjm_1((0, MPS(1), None, sim_params, hamiltonian))
    selected_z = np.asarray(selected[0], dtype=float)
    full_z = np.asarray(full[0], dtype=float)
    np.testing.assert_allclose(selected_z[1], full_z[1])
    np.testing.assert_allclose(selected_z[3], full_z[3])
    np.testing.assert_allclose(selected_z[[0, 2, 4]], 0.0)


def test_analog_tjm_2_zero_duration_empty_sample_at_writes_nothing() -> None:
    """A single-time order-2 run with sample_at=() records neither diagnostics nor observables."""
    sim_params = AnalogSimParams(
        observables=[Observable("z", 0)],
        elapsed_time=0.0,
        dt=0.1,
        order=2,
        sample_timesteps=True,
        get_state=True,
        num_traj=1,
    )
    results, diagnostics, state = analog_tjm_2(
        (0, MPS(1), None, sim_params, MPO.ising(1, 0.0, 0.0)),
        sample_at=(),
    )
    assert state is not None
    np.testing.assert_allclose(np.asarray(results, dtype=float), 0.0)
    np.testing.assert_allclose(diagnostics, 0.0)


def test_sample_at_multiple_indices_require_sample_timesteps() -> None:
    """Multiple sample_at indices need a full time-grid result buffer."""
    sim_params = AnalogSimParams(
        observables=[Observable("z", 0)],
        elapsed_time=0.2,
        dt=0.1,
        order=1,
        sample_timesteps=False,
    )
    with pytest.raises(ValueError, match="sample_timesteps=True"):
        analog_tjm_1((0, MPS(1), None, sim_params, MPO.ising(1, 0.0, 0.0)), sample_at=(1, 2))


@pytest.mark.parametrize(("backend", "order"), [(analog_tjm_1, 1), (analog_tjm_2, 2)])
@pytest.mark.parametrize("bad_index", [-1, 5])
def test_sample_at_rejects_out_of_range_indices(
    backend: Callable[..., tuple[np.ndarray, np.ndarray, MPS | None]],
    order: int,
    bad_index: int,
) -> None:
    """sample_at indices must lie on the analog time grid."""
    sim_params = AnalogSimParams(
        observables=[Observable("z", 0)],
        elapsed_time=0.2,
        dt=0.1,
        order=order,
        sample_timesteps=True,
    )
    with pytest.raises(ValueError, match="outside the time grid"):
        backend((0, MPS(1), None, sim_params, MPO.ising(1, 0.0, 0.0)), sample_at=(bad_index,))


def test_analog_tjm_2_get_state_when_sample_at_omits_final_time() -> None:
    """Order-2 get_state still returns the physical sample if sample_at skips the last time."""
    hamiltonian = MPO.ising(1, 0.0, 1.0)
    sim_params = AnalogSimParams(
        observables=[Observable("z", 0)],
        elapsed_time=0.3,
        dt=0.1,
        order=2,
        sample_timesteps=True,
        get_state=True,
        num_traj=1,
    )
    args = (0, MPS(1), None, sim_params, hamiltonian)
    omitted_results, _, omitted_state = analog_tjm_2(args, sample_at=(1,))
    full_results, _, full_state = analog_tjm_2(
        (0, MPS(1), None, sim_params, hamiltonian),
    )
    assert omitted_state is not None
    assert full_state is not None
    np.testing.assert_allclose(omitted_state.to_vec(), full_state.to_vec())
    np.testing.assert_allclose(omitted_results[0, 1], full_results[0, 1])
    np.testing.assert_allclose(omitted_results[0, -1], 0.0)
