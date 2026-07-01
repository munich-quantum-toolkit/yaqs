# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for deterministic unitary ensemble evolution in analog simulations.

Includes spin-chain two-time correlator checks against dense ED (see ``mqt.yaqs.analog.ensemble``).
"""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs import AnalogSimParams, Hamiltonian, NoiseModel, Observable, Simulator, State
from mqt.yaqs.analog.ensemble import ensemble_member_worker
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import EvolutionMode
from mqt.yaqs.core.libraries.gate_library import BaseGate, X, Y, Z


def test_unitary_ensemble_observable_average() -> None:
    """Aggregate observables over ensemble members in list-of-state analog runs."""
    length = 2
    hamiltonian = Hamiltonian.ising(length, J=0.6, g=0.2)
    initial_states = [State(length, initial="zeros"), State(length, initial="ones")]

    observable = Observable(Z(), 0)
    sim_params = AnalogSimParams(
        observables=[observable],
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
    )

    result = Simulator(parallel=False, show_progress=False).run(
        initial_states, hamiltonian, sim_params, noise_model=None
    )

    assert result.trajectories[0] is not None
    assert result.expectation_values[0] is not None
    assert result.trajectories[0].shape == (len(initial_states), len(sim_params.times))
    np.testing.assert_allclose(result.expectation_values[0], np.mean(result.trajectories[0], axis=0))


def test_unitary_ensemble_autocorrelator_outputs_mean_matrix_row() -> None:
    """Autocorrelation (O,O) pair yields a ``(1, n_times)`` ensemble-mean result."""
    length = 2
    hamiltonian = Hamiltonian.ising(length, J=0.5, g=0.1)
    initial_states = [State(length, initial="zeros"), State(length, initial="ones")]
    correlator_op = Observable(Z(), 0)

    sim_params = AnalogSimParams(
        observables=[],
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        multi_time_observables=[(correlator_op, correlator_op)],
    )

    result = Simulator(parallel=False, show_progress=False).run(
        initial_states, hamiltonian, sim_params, noise_model=None
    )

    assert result.multi_time_times is not None
    assert result.multi_time_results is not None
    assert result.multi_time_results.shape == (1, len(sim_params.times))
    assert np.iscomplexobj(result.multi_time_results)
    np.testing.assert_allclose(result.multi_time_results[0, 0], 1.0 + 0.0j, atol=1e-10)


def test_unitary_ensemble_multi_time_observables_mean_matrix() -> None:
    """Aggregate multi_time_observables pairs to an ensemble-mean matrix."""
    length = 2
    hamiltonian = Hamiltonian.ising(length, J=0.2, g=0.1)
    initial_states = [State(length, initial="zeros"), State(length, initial="ones")]
    z0 = Observable(Z(), 0)
    z1 = Observable(Z(), 1)
    pairs: list[tuple[Observable, Observable]] = [(z0, z1), (z1, z0)]

    sim_params = AnalogSimParams(
        observables=[],
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        multi_time_observables=pairs,
    )

    result = Simulator(parallel=False, show_progress=False).run(
        initial_states, hamiltonian, sim_params, noise_model=None
    )

    assert result.multi_time_times is not None
    assert result.multi_time_results is not None
    assert result.multi_time_results.shape == (len(pairs), len(sim_params.times))
    assert np.iscomplexobj(result.multi_time_results)


def test_unitary_ensemble_t0_only_records_when_not_sampling_timesteps() -> None:
    """When only ``t=0`` exists and sampling is off, observable/correlators are still recorded."""
    length = 2
    hamiltonian = Hamiltonian.ising(length, J=0.2, g=0.1)
    initial_states = [State(length, initial="zeros")]
    z0 = Observable(Z(), 0)
    z1 = Observable(Z(), 1)

    sim_params = AnalogSimParams(
        observables=[z0],
        elapsed_time=0.0,
        dt=0.1,
        sample_timesteps=False,
        multi_time_observables=[(z0, z0), (z0, z1)],
    )

    result = Simulator(parallel=False, show_progress=False).run(
        initial_states, hamiltonian, sim_params, noise_model=None
    )

    assert result.expectation_values[0] is not None
    assert result.expectation_values[0].shape == (1,)
    np.testing.assert_allclose(result.expectation_values[0][0], 1.0, atol=1e-10)

    assert result.multi_time_results is not None
    assert result.multi_time_results.shape == (2, 1)
    # (Z0, Z0) autocorrelator at t=0: <0|Z0^2|0> = 1
    np.testing.assert_allclose(result.multi_time_results[0, 0], 1.0 + 0.0j, atol=1e-10)


def test_unitary_ensemble_clears_multi_time_outputs_when_feature_disabled() -> None:
    """Reusing ``sim_params`` should clear prior multi_time_observables outputs when feature is off."""
    length = 2
    hamiltonian = Hamiltonian.ising(length, J=0.2, g=0.1)
    initial_states = [State(length, initial="zeros"), State(length, initial="ones")]
    z0 = Observable(Z(), 0)
    z1 = Observable(Z(), 1)

    sim_params = AnalogSimParams(
        observables=[],
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        multi_time_observables=[(z0, z0), (z0, z1)],
    )

    result = Simulator(parallel=False, show_progress=False).run(
        initial_states, hamiltonian, sim_params, noise_model=None
    )
    assert result.multi_time_results is not None
    assert result.multi_time_times is not None

    sim_params_off = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
    )

    result_off = Simulator(parallel=False, show_progress=False).run(
        initial_states, hamiltonian, sim_params_off, noise_model=None
    )
    assert result_off.multi_time_results is None
    assert result_off.multi_time_times is None


def test_list_mps_analog_ensemble_rejects_non_mps_representation() -> None:
    """List-of-MPS analog ensemble only supports the mps representation path."""
    length = 2
    hamiltonian = Hamiltonian.ising(length, J=0.2, g=0.1)
    states = [
        State(length, initial="zeros", representation="density_matrix"),
        State(length, initial="ones", representation="density_matrix"),
    ]
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
    )
    with pytest.raises(
        ValueError, match=r"list\[State\] analog ensemble currently supports only State\.representation='mps'\."
    ):
        Simulator(parallel=False, show_progress=False).run(states, hamiltonian, sim_params, noise_model=None)


def test_list_mps_analog_ensemble_rejects_empty_state_list() -> None:
    """Empty list[MPS] must fail before evolution."""
    length = 2
    hamiltonian = Hamiltonian.ising(length, J=0.2, g=0.1)
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
    )
    with pytest.raises(ValueError, match="initial_state list must not be empty"):
        Simulator(parallel=False, show_progress=False).run([], hamiltonian, sim_params, noise_model=None)


def test_list_mps_analog_ensemble_rejects_state_length_mismatch() -> None:
    """All ensemble MPS chain lengths must match the Hamiltonian MPO length."""
    hamiltonian = Hamiltonian.ising(2, J=0.2, g=0.1)
    states = [State(3, initial="zeros"), State(3, initial="ones")]
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
    )
    with pytest.raises(ValueError, match=r"State\.length=3 does not match Hamiltonian\.length=2"):
        Simulator(parallel=False, show_progress=False).run(states, hamiltonian, sim_params, noise_model=None)


def test_list_mps_analog_ensemble_rejects_get_state() -> None:
    """get_state is not supported together with list[MPS] analog ensemble mode."""
    length = 2
    hamiltonian = Hamiltonian.ising(length, J=0.2, g=0.1)
    states = [State(length, initial="zeros"), State(length, initial="ones")]
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
        get_state=True,
    )
    with pytest.raises(ValueError, match="get_state=True is not supported for list\\[State\\] analog ensemble mode"):
        Simulator(parallel=False, show_progress=False).run(states, hamiltonian, sim_params, noise_model=None)


def test_list_mps_unitary_ensemble_parallel_worker_path() -> None:
    """parallel=True with multiple members exercises the process-pool ensemble worker."""
    length = 2
    hamiltonian = Hamiltonian.ising(length, J=0.2, g=0.1)
    states = [State(length, initial="zeros"), State(length, initial="ones")]
    z0 = Observable(Z(), 0)
    z1 = Observable(Z(), 1)
    sim_params = AnalogSimParams(
        observables=[z0],
        elapsed_time=0.15,
        dt=0.05,
        multi_time_observables=[(z0, z0), (z0, z1)],
    )
    result = Simulator(parallel=True, show_progress=False).run(states, hamiltonian, sim_params, noise_model=None)
    assert result.expectation_values[0] is not None
    assert result.multi_time_results is not None


def test_unitary_ensemble_uses_bug_evolution_mode_via_simulator() -> None:
    """BUG tensor evolution should be exercised by the high-level Simulator path."""
    length = 2
    hamiltonian = Hamiltonian.ising(length, J=0.2, g=0.1)
    states = [State(length, initial="zeros"), State(length, initial="ones")]
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.05,
        dt=0.05,
        evolution_mode=EvolutionMode.BUG,
        max_bond_dim=64,
        svd_threshold=1e-10,
    )
    result = Simulator(parallel=False, show_progress=False).run(states, hamiltonian, sim_params, noise_model=None)
    assert result.expectation_values[0] is not None
    assert result.expectation_values[0].shape == (len(sim_params.times),)
    assert result.multi_time_results is None


def test_unitary_ensemble_final_timestep_when_not_sampling_via_simulator() -> None:
    """``sample_timesteps=False`` with multi-time pairs records correlators on the last step."""
    length = 2
    hamiltonian = Hamiltonian.ising(length, J=0.2, g=0.1)
    states = [State(length, initial="zeros"), State(length, initial="ones")]
    z0 = Observable(Z(), 0)
    z1 = Observable(Z(), 1)
    sim_params = AnalogSimParams(
        observables=[z0],
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=False,
        multi_time_observables=[(z0, z0), (z0, z1)],
        max_bond_dim=64,
        svd_threshold=1e-10,
    )
    assert len(sim_params.times) >= 3
    result = Simulator(parallel=False, show_progress=False).run(states, hamiltonian, sim_params, noise_model=None)
    assert result.expectation_values[0] is not None
    assert result.expectation_values[0].shape == (1,)
    assert result.multi_time_results is not None
    assert result.multi_time_results.shape == (2, 1)


def test_list_initial_states_with_noise_raises() -> None:
    """Raise an explicit error for noisy analog runs with list[MPS]."""
    length = 2
    hamiltonian = Hamiltonian.ising(length, J=1.0, g=0.2)
    initial_states = [State(length, initial="zeros"), State(length, initial="ones")]
    noise_model = NoiseModel([{"name": "lowering", "sites": [0], "strength": 0.1}])
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
    )

    with pytest.raises(
        ValueError,
        match=(
            r"(?s)list\[State\] with noisy analog simulation is not supported yet\."
            r".*list\[State\] with no noise.*single State for noisy simulation"
        ),
    ):
        Simulator(parallel=False, show_progress=False).run(
            initial_states, hamiltonian, sim_params, noise_model=noise_model
        )


# --- Spin-chain ED helpers and integration tests ---


def _embed_one_site_operator(length: int, site: int, op2: np.ndarray) -> np.ndarray:
    left_dim = 2**site
    right_dim = 2 ** (length - site - 1)
    return np.kron(np.kron(np.eye(left_dim, dtype=np.complex128), op2), np.eye(right_dim, dtype=np.complex128))


def _embed_two_site_operator(length: int, site_left: int, op4: np.ndarray) -> np.ndarray:
    left_dim = 2**site_left
    right_dim = 2 ** (length - site_left - 2)
    return np.kron(np.kron(np.eye(left_dim, dtype=np.complex128), op4), np.eye(right_dim, dtype=np.complex128))


def _spin_current_bond_matrix(j_coupling: float) -> np.ndarray:
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    return 0.25 * j_coupling * (np.kron(x, y) - np.kron(y, x))


def _periodic_bond_endpoints(length: int) -> list[tuple[int, int]]:
    return [(i, (i + 1) % length) for i in range(length)]


def _spin_current_observable_for_periodic_bond(site_a: int, site_b: int, j_xy: float) -> Observable:
    return Observable(BaseGate(_spin_current_bond_matrix(j_xy)), sites=[site_a, site_b])


def _build_xxz_tf_open_chain_dense(length: int, j_xy: float, delta: float, h_x: float) -> np.ndarray:
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    xx = np.kron(x, x)
    yy = np.kron(y, y)
    zz = np.kron(z, z)
    dim = 2**length
    h_dense = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(length - 1):
        h_dense += 0.25 * j_xy * _embed_two_site_operator(length, i, xx)
        h_dense += 0.25 * j_xy * _embed_two_site_operator(length, i, yy)
        h_dense += 0.25 * delta * _embed_two_site_operator(length, i, zz)
    for r in range(length):
        h_dense += 0.5 * h_x * _embed_one_site_operator(length, r, x)
    return h_dense


def _ed_first_k_basis_two_time_means(
    *,
    length: int,
    j_xy: float,
    delta: float,
    h_x: float,
    times: np.ndarray,
    k: int,
    site: int,
    probes: list[tuple[np.ndarray, np.ndarray]],
) -> list[np.ndarray]:
    dim = 2**length
    k_eff = min(k, dim)
    h_dense = _build_xxz_tf_open_chain_dense(length, j_xy, delta, h_x)
    evals, evecs = np.linalg.eigh(h_dense)
    evecs_h = evecs.conj().T

    embedded_b = [_embed_one_site_operator(length, site, b2) for _a2, b2 in probes]
    embedded_a = [_embed_one_site_operator(length, site, a2) for a2, _b2 in probes]

    out = np.zeros((len(probes), len(times)), dtype=np.complex128)
    for t_idx, t in enumerate(times):
        phases = np.exp(-1j * evals * t)
        u_t = (evecs * phases[np.newaxis, :]) @ evecs_h
        for p_idx, (a_op, b_op) in enumerate(zip(embedded_a, embedded_b, strict=True)):
            m = u_t.conj().T @ a_op @ u_t @ b_op
            out[p_idx, t_idx] = np.sum(np.diag(m)[:k_eff]) / float(k_eff)

    return [out[p_idx] for p_idx in range(len(probes))]


def test_xxz_transverse_unitary_ensemble_pauli_and_two_time_vs_ed() -> None:
    """Match YAQS two-time correlators against dense ED for a transverse-field XXZ setup.

    Parameters are tuned for CI runtime while keeping max error below ``1e-5`` against ED:
    ``k=16`` ensemble members (was 64) and ``parallel=True`` (unitary, deterministic).
    """
    length = 6
    j_xy = 1.0
    delta = 0.7
    h_x = 0.5
    t_final = 2.0
    dt = 0.05
    k = 16
    mid = length // 2

    h_mpo = Hamiltonian.pauli(
        length=length,
        two_body=[(0.25 * j_xy, "X", "X"), (0.25 * j_xy, "Y", "Y"), (0.25 * delta, "Z", "Z")],
        one_body=[(0.5 * h_x, "X")],
        bc="open",
    )
    states = [State(length, initial="basis", basis_string=format(i, f"0{length}b")) for i in range(k)]

    ox = Observable(X(), mid)
    oy = Observable(Y(), mid)
    oz = Observable(Z(), mid)
    pairs = [(ox, ox), (oy, oy), (oz, oz), (oz, ox)]

    sim_params = AnalogSimParams(
        observables=[],
        elapsed_time=t_final,
        dt=dt,
        max_bond_dim=256,
        svd_threshold=1e-12,
        krylov_tol=1e-12,
        sample_timesteps=True,
        multi_time_observables=pairs,
    )
    result = Simulator(parallel=True, show_progress=False).run(states, h_mpo, sim_params, noise_model=None)
    assert result.multi_time_results is not None
    assert result.multi_time_times is not None
    yaqs = np.asarray(result.multi_time_results, dtype=np.complex128)
    times = np.asarray(result.multi_time_times, dtype=np.float64)

    x2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    y2 = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    z2 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    ed_series = _ed_first_k_basis_two_time_means(
        length=length,
        j_xy=j_xy,
        delta=delta,
        h_x=h_x,
        times=times,
        k=k,
        site=mid,
        probes=[(x2, x2), (y2, y2), (z2, z2), (z2, x2)],
    )
    ed = np.vstack(ed_series)

    max_abs_per_pair = np.max(np.abs(yaqs - ed), axis=1)
    assert np.all(max_abs_per_pair < 1e-5), f"pairwise max errors were {max_abs_per_pair!r}"


def test_two_time_correlator_probe_row_diagonal_matches_expectation_at_t0() -> None:
    """At ``t=0``, two-time block diagonal entry ``(r,r)`` must match dense ``<psi|j_r j_r|psi>``."""
    length = 4
    j_xy = 1.0
    delta = 0.8
    mps = MPS(length, state="random", pad=8)
    mps.normalize("B")

    row = tuple(_spin_current_observable_for_periodic_bond(a, b, j_xy) for a, b in _periodic_bond_endpoints(length))
    s_index = 1
    obs_s = row[s_index]
    pairs = [(obs_r, obs_s) for obs_r in row]

    h = MPO.pauli(
        length=length,
        two_body=[(0.25 * j_xy, "X", "X"), (0.25 * j_xy, "Y", "Y"), (0.25 * delta, "Z", "Z")],
        bc="periodic",
    )

    sim_params = AnalogSimParams(
        observables=[],
        elapsed_time=0.0,
        dt=0.5,
        max_bond_dim=32,
        svd_threshold=1e-10,
        order=1,
        sample_timesteps=True,
        multi_time_observables=pairs,
    )

    _, _, mat = ensemble_member_worker((0, mps, sim_params, h))
    assert mat is not None
    val_worker = float(np.real(mat[s_index, 0]))

    psi = np.asarray(mps.to_vec(), dtype=np.complex128)
    j_bond = _spin_current_bond_matrix(j_xy)
    jr = _embed_two_site_operator(length, 1, j_bond)
    expected = float(np.real(np.vdot(psi, jr @ jr @ psi)))
    assert val_worker == pytest.approx(expected, rel=0, abs=1e-6)
