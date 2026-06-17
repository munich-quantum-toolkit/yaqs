# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for simulation parameters classes.

This module contains unit tests for the Observable and AnalogSimParams classes used in
quantum simulation. It verifies that:
  - An Observable is correctly initialized with valid parameters and that invalid parameters
    raise an appropriate error.
  - AnalogSimParams instances are created with the correct attributes (such as elapsed_time, dt, times,
    sample_timesteps, and num_traj) both with explicit and default values.
  - allocate_observable_buffers properly sets up expectation_values and trajectories arrays
    depending on whether sample_timesteps is True or False.
"""

# ignore non-lowercase variable names for physics notation
# ruff: noqa: PLC2701, SLF001 -- white-box tests of parameter validation and TDVP internals

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.result import Result, aggregate_trajectories, allocate_observable_buffers
from mqt.yaqs.core.data_structures.simulation_parameters import (
    SIMULATION_PRESETS,
    AnalogSimParams,
    Observable,
    StrongSimParams,
    WeakSimParams,
    _validate_tdvp_sweeps,
)
from mqt.yaqs.core.libraries.gate_library import GateLibrary, X
from mqt.yaqs.core.methods.tdvp import primitives as tdvp_primitives

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.core.data_structures.simulation_parameters import (
        GateMode,
        SimulationPreset,
        TDVPMode,
    )


def test_observable_creation_valid() -> None:
    """Test that an Observable is created correctly with valid parameters.

    This test constructs an Observable with the name "x" on site 0 and verifies gate and site.
    """
    gate = X()
    site = 0
    obs = Observable(gate, site)

    assert np.array_equal(obs.gate.matrix, np.array([[0, 1], [1, 0]]))
    assert obs.sites == site


def test_analog_simparams_basic() -> None:
    """Test that AnalogSimParams is initialized with correct parameters.

    This test creates a AnalogSimParams instance with a single observable, total time elapsed_time, time step dt,
    sample_timesteps flag set to True, and a specified number of trajectories num_traj. It then verifies that the
    observables, elapsed_time, dt, times array, sample_timesteps flag, and num_traj are set correctly.
    """
    obs_list = [Observable(X(), 0)]
    elapsed_time = 1.0
    dt = 0.2
    params = AnalogSimParams(observables=obs_list, elapsed_time=elapsed_time, dt=dt, num_traj=50)

    assert params.observables == obs_list
    assert params.elapsed_time == elapsed_time
    assert params.dt == dt
    expected_times = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert np.allclose(params.times, expected_times), "Times array should match numpy.arange(0, elapsed_time+dt, dt)."
    assert params.sample_timesteps is True
    assert params.num_traj == 50


def test_analog_simparams_defaults() -> None:
    """Test the default parameters for AnalogSimParams.

    This test constructs a AnalogSimParams instance with an empty observable list and total time elapsed_time,
    and verifies that default values for dt, sample_timesteps, number of trajectories (num_traj), max_bond_dim,
    svd_threshold, and order are correctly assigned.
    """
    obs_list = [Observable(X(), 0)]
    params = AnalogSimParams(observables=obs_list)

    assert params.elapsed_time == pytest.approx(0.1)
    assert params.dt == pytest.approx(0.1)
    assert params.sample_timesteps is True
    # times should be np.arange(0, elapsed_time+dt, dt)
    assert np.isclose(params.times[-1], 0.1)
    balanced = SIMULATION_PRESETS["balanced"]
    assert params.preset == "balanced"
    assert params.num_traj == balanced["num_traj"]
    assert params.max_bond_dim == balanced["max_bond_dim"]
    assert params.svd_threshold == pytest.approx(balanced["svd_threshold"])
    assert params.krylov_tol == pytest.approx(balanced["krylov_tol"])
    assert params.order == 1


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ("fast", SIMULATION_PRESETS["fast"]),
        ("balanced", SIMULATION_PRESETS["balanced"]),
        ("accurate", SIMULATION_PRESETS["accurate"]),
        ("exact", SIMULATION_PRESETS["exact"]),
    ],
)
def test_analog_simparams_presets(preset: SimulationPreset, expected: dict[str, float | int | None]) -> None:
    """AnalogSimParams resolves svd_threshold, max_bond_dim, num_traj, and krylov_tol from presets."""
    params = AnalogSimParams(preset=preset)
    assert params.preset == preset
    assert params.svd_threshold == pytest.approx(expected["svd_threshold"])
    assert params.max_bond_dim == expected["max_bond_dim"]
    assert params.num_traj == expected["num_traj"]
    assert params.krylov_tol == pytest.approx(expected["krylov_tol"])


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ("fast", SIMULATION_PRESETS["fast"]),
        ("balanced", SIMULATION_PRESETS["balanced"]),
        ("accurate", SIMULATION_PRESETS["accurate"]),
        ("exact", SIMULATION_PRESETS["exact"]),
    ],
)
def test_strong_simparams_presets(preset: SimulationPreset, expected: dict[str, float | int | None]) -> None:
    """StrongSimParams resolves svd_threshold, max_bond_dim, num_traj, and krylov_tol from presets."""
    params = StrongSimParams(preset=preset)
    assert params.preset == preset
    assert params.svd_threshold == pytest.approx(expected["svd_threshold"])
    assert params.max_bond_dim == expected["max_bond_dim"]
    assert params.num_traj == expected["num_traj"]
    assert params.krylov_tol == pytest.approx(expected["krylov_tol"])


def test_analog_simparams_default_constructor_uses_balanced() -> None:
    """AnalogSimParams() uses the balanced preset by default."""
    params = AnalogSimParams()
    balanced = SIMULATION_PRESETS["balanced"]
    assert params.preset == "balanced"
    assert params.svd_threshold == pytest.approx(balanced["svd_threshold"])
    assert params.max_bond_dim == balanced["max_bond_dim"]
    assert params.num_traj == balanced["num_traj"]
    assert params.krylov_tol == pytest.approx(balanced["krylov_tol"])


def test_weak_simparams_default_constructor_uses_balanced() -> None:
    """WeakSimParams(shots=...) uses the balanced preset by default."""
    params = WeakSimParams(shots=100)
    balanced = SIMULATION_PRESETS["balanced"]
    assert params.preset == "balanced"
    assert params.svd_threshold == pytest.approx(balanced["svd_threshold"])
    assert params.max_bond_dim == balanced["max_bond_dim"]
    assert params.krylov_tol == pytest.approx(balanced["krylov_tol"])
    assert params.gate_mode == "mpo"


def test_gate_mode_defaults_and_validation() -> None:
    """Strong and weak digital params default to mpo and validate gate_mode names."""
    assert StrongSimParams().gate_mode == "mpo"
    assert WeakSimParams(shots=1).gate_mode == "mpo"
    assert StrongSimParams(gate_mode="full-tdvp").gate_mode == "full-tdvp"
    assert StrongSimParams(gate_mode="mpo").gate_mode == "mpo"
    with pytest.raises(ValueError, match="gate_mode"):
        StrongSimParams(gate_mode=cast("GateMode", "invalid"))


def test_tdvp_mode_defaults_and_validation() -> None:
    """All simulation params default to 2site TDVP."""
    assert AnalogSimParams().tdvp_mode == "2site"
    assert StrongSimParams().tdvp_mode == "2site"
    assert WeakSimParams(shots=1).tdvp_mode == "2site"
    assert StrongSimParams(tdvp_mode="1site").tdvp_mode == "1site"
    assert StrongSimParams(tdvp_mode="2site").tdvp_mode == "2site"
    with pytest.raises(ValueError, match="tdvp_mode"):
        StrongSimParams(tdvp_mode=cast("TDVPMode", "invalid"))


def test_tdvp_sweeps_defaults_and_validation() -> None:
    """Analog, strong, and weak params default tdvp_sweeps to 1 and validate inputs."""
    assert AnalogSimParams().tdvp_sweeps == 1
    assert StrongSimParams().tdvp_sweeps == 1
    assert WeakSimParams(shots=1).tdvp_sweeps == 1
    assert StrongSimParams(tdvp_sweeps=3).tdvp_sweeps == 3
    with pytest.raises(ValueError, match="tdvp_sweeps"):
        StrongSimParams(tdvp_sweeps=0)
    with pytest.raises(ValueError, match="tdvp_sweeps"):
        StrongSimParams(tdvp_sweeps=-1)


@pytest.mark.parametrize("invalid", [1.5, True])
def test_tdvp_sweeps_rejects_non_int(invalid: object) -> None:
    """tdvp_sweeps must be a true int, not bool or float."""
    with pytest.raises(TypeError, match="tdvp_sweeps"):
        _validate_tdvp_sweeps(cast("Any", invalid))


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ("fast", SIMULATION_PRESETS["fast"]),
        ("balanced", SIMULATION_PRESETS["balanced"]),
        ("accurate", SIMULATION_PRESETS["accurate"]),
        ("exact", SIMULATION_PRESETS["exact"]),
    ],
)
def test_weak_simparams_presets(preset: SimulationPreset, expected: dict[str, float | int | None]) -> None:
    """WeakSimParams resolves svd_threshold, max_bond_dim, and krylov_tol from the shared presets."""
    params = WeakSimParams(shots=100, preset=preset)
    assert params.preset == preset
    assert params.shots == 100
    assert params.svd_threshold == pytest.approx(expected["svd_threshold"])
    assert params.max_bond_dim == expected["max_bond_dim"]
    assert params.krylov_tol == pytest.approx(expected["krylov_tol"])


def test_analog_simparams_preset_explicit_overrides() -> None:
    """Explicit numerical arguments override presets."""
    params = AnalogSimParams(preset="fast", svd_threshold=1e-8, max_bond_dim=512, num_traj=10, krylov_tol=1e-12)
    assert params.svd_threshold == pytest.approx(1e-8)
    assert params.max_bond_dim == 512
    assert params.num_traj == 10
    assert params.krylov_tol == pytest.approx(1e-12)


def test_analog_simparams_krylov_tol_overrides_preset_only() -> None:
    """Explicit ``krylov_tol`` overrides the preset without affecting other preset fields."""
    params = AnalogSimParams(preset="balanced", krylov_tol=1e-8)
    balanced = SIMULATION_PRESETS["balanced"]
    assert params.preset == "balanced"
    assert params.svd_threshold == pytest.approx(balanced["svd_threshold"])
    assert params.max_bond_dim == balanced["max_bond_dim"]
    assert params.num_traj == balanced["num_traj"]
    assert params.krylov_tol == pytest.approx(1e-8)


def test_analog_simparams_max_bond_dim_none_overrides_preset() -> None:
    """Explicit ``max_bond_dim=None`` removes the bond cap without changing other preset fields."""
    params = AnalogSimParams(preset="balanced", max_bond_dim=None)
    balanced = SIMULATION_PRESETS["balanced"]
    assert params.preset == "balanced"
    assert params.max_bond_dim is None
    assert params.svd_threshold == pytest.approx(balanced["svd_threshold"])
    assert params.num_traj == balanced["num_traj"]
    assert params.krylov_tol == pytest.approx(balanced["krylov_tol"])


def test_strong_simparams_preset_explicit_overrides() -> None:
    """Explicit numerical arguments override presets."""
    params = StrongSimParams(preset="fast", svd_threshold=1e-8, max_bond_dim=512, num_traj=10, krylov_tol=1e-12)
    assert params.svd_threshold == pytest.approx(1e-8)
    assert params.max_bond_dim == 512
    assert params.num_traj == 10
    assert params.krylov_tol == pytest.approx(1e-12)


def test_weak_simparams_preset_explicit_overrides() -> None:
    """Explicit numerical arguments override presets."""
    params = WeakSimParams(shots=100, preset="fast", svd_threshold=1e-8, max_bond_dim=512, krylov_tol=1e-12)
    assert params.shots == 100
    assert params.svd_threshold == pytest.approx(1e-8)
    assert params.max_bond_dim == 512
    assert params.krylov_tol == pytest.approx(1e-12)


def test_strong_simparams_rejects_invalid_preset() -> None:
    """Invalid preset names raise ValueError."""
    with pytest.raises(ValueError, match="preset must be one of"):
        StrongSimParams(preset="invalid")  # ty: ignore[invalid-argument-type]


def test_strong_simparams_rejects_none_preset() -> None:
    """preset=None is not supported."""
    with pytest.raises(ValueError, match="preset must be one of"):
        StrongSimParams(preset=None)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("bad_tol", [0.0, -1.0, float("inf"), float("nan")])
def test_simparams_rejects_invalid_krylov_tol(bad_tol: float) -> None:
    """krylov_tol must be finite and strictly positive."""
    with pytest.raises(ValueError, match="krylov_tol must be a finite positive float"):
        _ = AnalogSimParams(krylov_tol=bad_tol)


@pytest.mark.parametrize("bad_threshold", [0.0, -1.0, float("inf"), float("nan")])
def test_simparams_rejects_invalid_svd_threshold(bad_threshold: float) -> None:
    """svd_threshold must be finite and strictly positive."""
    with pytest.raises(ValueError, match="svd_threshold must be a finite positive float"):
        _ = AnalogSimParams(svd_threshold=bad_threshold)


def test_krylov_tol_propagates_to_expm_krylov(monkeypatch: pytest.MonkeyPatch) -> None:
    """TDVP Krylov helper must pass krylov_tol down to expm_krylov(tol=...)."""
    seen: dict[str, float] = {}

    def fake_expm_krylov(
        _matrix_free_operator: object,
        vec: np.ndarray,
        _dt: float,
        max_lanczos_iterations: int = 25,
        tol: float = 1e-12,
    ) -> np.ndarray:
        _ = max_lanczos_iterations
        seen["tol"] = float(tol)
        return vec

    monkeypatch.setattr(tdvp_primitives, "expm_krylov", fake_expm_krylov)

    tensor = np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)

    _ = tdvp_primitives._evolve_local_tensor_krylov(
        projector=lambda x: x,
        tensor=tensor,
        dt=0.1,
        proj_args=(),
        krylov_tol=1e-7,
    )

    assert seen["tol"] == pytest.approx(1e-7)


def test_allocate_observable_buffers_with_sample_timesteps() -> None:
    """allocate_observable_buffers shapes buffers when sample_timesteps is True."""
    sim_params = AnalogSimParams(
        observables=[Observable(X(), 1)],
        elapsed_time=1.0,
        dt=0.5,
        num_traj=10,
        sample_timesteps=True,
    )
    trajectories, expectation_values, times = allocate_observable_buffers(sim_params, 1, num_traj=10)

    assert times is sim_params.times
    assert expectation_values[0].shape == (3,)
    assert trajectories[0].shape == (10, 3)


def test_allocate_observable_buffers_without_sample_timesteps() -> None:
    """allocate_observable_buffers uses a single time column when sample_timesteps is False."""
    sim_params = AnalogSimParams(
        observables=[Observable(X(), 0)],
        elapsed_time=1.0,
        dt=0.25,
        num_traj=5,
        sample_timesteps=False,
    )
    trajectories, expectation_values, times = allocate_observable_buffers(sim_params, 1, num_traj=5)

    assert times is not None
    assert times.shape == (1,)
    assert expectation_values[0].shape == (1,)
    assert trajectories[0].shape == (5, 1)


def test_observable_from_string_entropy_and_spectrum_with_list_sites() -> None:
    """Constructor maps 'entropy' and 'schmidt_spectrum' and accepts list[int] sites."""
    cut = [3, 4]
    obs_ent = Observable("entropy", sites=cut)
    obs_ssp = Observable("schmidt_spectrum", sites=cut)

    assert obs_ent.gate.name == "entropy"
    assert obs_ssp.gate.name == "schmidt_spectrum"
    # meta-observables use identity placeholders for BaseGate compatibility
    assert np.allclose(obs_ent.gate.matrix, np.eye(2))
    assert np.allclose(obs_ssp.gate.matrix, np.eye(2))
    assert obs_ent.sites == cut
    assert obs_ssp.sites == cut


def test_observable_from_string_falls_back_to_pvm() -> None:
    """Any other string is interpreted as a PVM bitstring; gate must store that bitstring."""
    bitstring = "10101"
    obs = Observable(bitstring, sites=None)
    assert obs.gate.name == "pvm"
    # gate must expose the queried bitstring
    assert hasattr(obs.gate, "bitstring")
    assert obs.gate.bitstring == bitstring
    # PVM uses identity placeholder matrix for compatibility in your implementation
    assert np.allclose(obs.gate.matrix, np.eye(2))


def test_observable_from_gate_instance_keeps_gate_and_sites_int() -> None:
    """Passing a concrete BaseGate instance should be preserved and sites can be an int."""
    x_gate = GateLibrary.x()
    obs = Observable(x_gate, sites=5)
    # same object semantics not required; equality via matrix is sufficient
    assert obs.gate.name == "x"
    assert np.allclose(obs.gate.matrix, x_gate.matrix)
    assert obs.sites == 5


def test_observable_from_gate_instance_with_list_sites() -> None:
    """Gate instance + list[int] sites should preserve the list (for two-site ops)."""
    cz_gate = GateLibrary.cz()
    obs = Observable(cz_gate, sites=[1, 3])
    assert obs.gate.name == "cz"
    assert obs.sites == [1, 3]


def test_aggregate_trajectories_regular_observable_mean() -> None:
    """Regular observables: results = mean(trajectories, axis=0).

    We create a single-site Z observable with a (num_traj x T) trajectory array and
    verify that `results` equals the columnwise mean.
    """
    # Observable to aggregate
    z_obs = Observable(GateLibrary.z(), sites=0)

    # Two trajectories across 3 time steps → mean is easy to verify
    traj = np.array(
        [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
        dtype=np.float64,
    )
    sim = AnalogSimParams(observables=[z_obs], elapsed_time=0.2, dt=0.1, num_traj=2)
    run_result = Result(sim_params=sim, observables=[z_obs], trajectories=[traj], expectation_values=[np.empty(1)])

    aggregate_trajectories(run_result)

    expected = traj.mean(axis=0)
    np.testing.assert_allclose(run_result.expectation_values[0], expected)


def test_aggregate_trajectories_schmidt_concatenation() -> None:
    """Schmidt spectrum: results = concatenation of raveled arrays from list entries.

    Provide a list of arrays with different shapes (1D/2D) to confirm `.ravel()` and
    `np.concatenate` behavior.
    """
    ss_obs = Observable(GateLibrary.schmidt_spectrum(), sites=[1, 2])

    # List of arrays (the method requires a list, not a single ndarray)
    a = np.array([0.8, 0.6], dtype=np.float64)
    b = np.array([0.4, 0.3], dtype=np.float64)  # will ravel to [0.4, 0.3]
    c = np.array([0.2, 0.1], dtype=np.float64)  # will ravel to [0.2, 0.1]
    traj_arr = np.array([a, b, c])

    sim = AnalogSimParams(observables=[ss_obs], elapsed_time=0.1, dt=0.1, num_traj=3)
    run_result = Result(sim_params=sim, observables=[ss_obs], trajectories=[traj_arr], expectation_values=[np.empty(1)])

    aggregate_trajectories(run_result)

    np.testing.assert_allclose(
        run_result.expectation_values[0], np.array([0.8, 0.6, 0.4, 0.3, 0.2, 0.1], dtype=np.float64)
    )


def test_aggregate_trajectories_mixed_regular_and_schmidt() -> None:
    """Combination: both regular and Schmidt observables are updated correctly."""
    # Regular observable with 3 trajectories x 2 time steps
    x_obs = Observable(GateLibrary.x(), sites=2)
    x_traj = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], dtype=np.float64)

    ss_obs = Observable(GateLibrary.schmidt_spectrum(), sites=[0, 1])
    ss_traj = np.array([np.array([1.0, 0.5], dtype=np.float64), np.array([0.5, 0.25], dtype=np.float64)])

    sim = AnalogSimParams(observables=[x_obs, ss_obs], elapsed_time=0.2, dt=0.1, num_traj=3)
    run_result = Result(
        sim_params=sim,
        observables=[x_obs, ss_obs],
        trajectories=[x_traj, ss_traj],
        expectation_values=[np.empty(2), np.empty(4)],
    )

    aggregate_trajectories(run_result)

    np.testing.assert_allclose(run_result.expectation_values[0], np.array([1.0, 1.0], dtype=np.float64))
    np.testing.assert_allclose(run_result.expectation_values[1], np.array([1.0, 0.5, 0.5, 0.25], dtype=np.float64))


def test_aggregate_trajectories_schmidt_requires_array() -> None:
    """For Schmidt spectrum, trajectories must be a *array*; list should raise AssertionError."""
    ss_obs = Observable(GateLibrary.schmidt_spectrum(), sites=[2, 3])
    bad_traj = [0.9, 0.1]

    sim = AnalogSimParams(observables=[ss_obs], elapsed_time=0.1, dt=0.1)

    run_result = Result(
        sim_params=sim,
        observables=[ss_obs],
        trajectories=cast("list[NDArray]", [bad_traj]),
        expectation_values=[np.empty(1)],
    )

    with pytest.raises(AssertionError):
        aggregate_trajectories(run_result)


def test_strong_params_sorting_and_fields() -> None:
    """Constructor sorts non-PVM observables by site; PVM observables are appended."""
    obs_z3 = Observable(GateLibrary.z(), sites=3)
    obs_x2 = Observable(GateLibrary.x(), sites=2)
    obs_y1 = Observable(GateLibrary.y(), sites=1)
    obs_ssp = Observable(GateLibrary.schmidt_spectrum(), sites=[1, 2])

    params = StrongSimParams(
        observables=[obs_z3, obs_x2, obs_y1, obs_ssp],
        num_traj=7,
        max_bond_dim=128,
        get_state=True,
        sample_layers=True,
        num_mid_measurements=2,
    )

    assert params.sorted_observables[0] is obs_y1
    assert params.sorted_observables[1] is obs_ssp
    assert params.sorted_observables[2] is obs_x2
    assert params.sorted_observables[3] is obs_z3

    # Mapping from user order -> sorted worker row indices
    assert params.observable_sorted_indices == (3, 2, 0, 1)

    # Ordering is derived from the current observables list, not cached at construction.
    params.observables.append(Observable(GateLibrary.z(), sites=0))
    assert len(params.sorted_observables) == 5
    assert params.observable_sorted_indices == (4, 3, 1, 2, 0)

    # Parameter fields are retained
    assert params.num_traj == 7
    assert params.max_bond_dim == 128
    assert params.svd_threshold == pytest.approx(SIMULATION_PRESETS["balanced"]["svd_threshold"])
    assert params.get_state is True
    assert params.sample_layers is True
    assert params.num_mid_measurements == 2


def test_strong_params_rejects_mixed_pvm_with_non_pvm() -> None:
    """Constructor must assert when mixing PVM with non-PVM observables."""
    pvm = Observable(GateLibrary.pvm("101"), sites=None)
    z0 = Observable(GateLibrary.z(), sites=0)
    with pytest.raises(AssertionError):
        _ = StrongSimParams(observables=[pvm, z0])


def test_strong_params_accepts_all_pvm_or_all_non_pvm() -> None:
    """Constructor allows all-PVM and all-non-PVM sets."""
    # All PVM
    p1 = Observable(GateLibrary.pvm("0"), sites=None)
    p2 = Observable(GateLibrary.pvm("1"), sites=None)
    _ = StrongSimParams(observables=[p1, p2])  # should not raise

    # All non-PVM
    z0 = Observable(GateLibrary.z(), sites=0)
    x1 = Observable(GateLibrary.x(), sites=1)
    _ = StrongSimParams(observables=[z0, x1])  # should not raise


def test_strong_aggregate_regular_mean() -> None:
    """Regular observables: results = mean(trajectories, axis=0)."""
    x = Observable(GateLibrary.x(), sites=2)
    traj = np.array(
        [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    params = StrongSimParams(observables=[x], num_traj=3)
    run_result = Result(sim_params=params, observables=[x], trajectories=[traj], expectation_values=[np.empty(3)])
    aggregate_trajectories(run_result)

    np.testing.assert_allclose(run_result.expectation_values[0], traj.mean(axis=0))


def test_strong_aggregate_schmidt_concat() -> None:
    """Schmidt spectrum: concatenation of raveled list entries."""
    ssp = Observable(GateLibrary.schmidt_spectrum(), sites=[0, 1])
    ssp_traj = np.array([
        np.array([0.9, 0.8], dtype=np.float64),
        np.array([0.6, 0.4], dtype=np.float64),
        np.array([0.2, 0.1], dtype=np.float64),
    ])

    params = StrongSimParams(observables=[ssp], num_traj=3)
    run_result = Result(sim_params=params, observables=[ssp], trajectories=[ssp_traj], expectation_values=[np.empty(6)])
    aggregate_trajectories(run_result)

    np.testing.assert_allclose(
        run_result.expectation_values[0], np.array([0.9, 0.8, 0.6, 0.4, 0.2, 0.1], dtype=np.float64)
    )


def test_strong_aggregate_mixed_regular_and_schmidt() -> None:
    """Combination case: regular and Schmidt updated correctly in one call."""
    z = Observable(GateLibrary.z(), sites=0)
    z_traj = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    ssp = Observable(GateLibrary.schmidt_spectrum(), sites=[1, 2])
    ssp_traj = np.array([np.array([1.0, 0.5], dtype=np.float64), np.array([0.5, 0.25], dtype=np.float64)])

    params = StrongSimParams(observables=[z, ssp], num_traj=2)
    run_result = Result(
        sim_params=params,
        observables=[z, ssp],
        trajectories=[z_traj, ssp_traj],
        expectation_values=[np.empty(2), np.empty(4)],
    )
    aggregate_trajectories(run_result)

    np.testing.assert_allclose(run_result.expectation_values[0], np.array([2.0, 3.0], dtype=np.float64))
    np.testing.assert_allclose(run_result.expectation_values[1], np.array([1.0, 0.5, 0.5, 0.25], dtype=np.float64))


def test_strong_aggregate_schmidt_requires_array() -> None:
    """Schmidt branch must assert if trajectories is not an array."""
    ssp = Observable(GateLibrary.schmidt_spectrum(), sites=[0, 1])
    bad_traj = [0.9, 0.1]

    params = StrongSimParams(observables=[ssp], num_traj=1)

    run_result = Result(
        sim_params=params,
        observables=[ssp],
        trajectories=cast("list[NDArray]", [bad_traj]),
        expectation_values=[np.empty(1)],
    )

    with pytest.raises(AssertionError):
        aggregate_trajectories(run_result)


@pytest.mark.parametrize(
    ("param_cls", "kwargs"),
    [
        (AnalogSimParams, {}),
        (WeakSimParams, {"shots": 1}),
        (StrongSimParams, {}),
    ],
)
def test_random_seed_rejects_invalid_type(
    param_cls: type[AnalogSimParams | WeakSimParams | StrongSimParams],
    kwargs: dict[str, object],
) -> None:
    """random_seed must be None or int."""
    with pytest.raises(TypeError, match="random_seed must be int or None"):
        param_cls(random_seed="not-a-seed", **kwargs)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("param_cls", "kwargs"),
    [
        (AnalogSimParams, {}),
        (WeakSimParams, {"shots": 1}),
        (StrongSimParams, {}),
    ],
)
def test_random_seed_rejects_negative(
    param_cls: type[AnalogSimParams | WeakSimParams | StrongSimParams],
    kwargs: dict[str, object],
) -> None:
    """random_seed must be non-negative when set."""
    with pytest.raises(ValueError, match="random_seed must be non-negative"):
        param_cls(random_seed=-1, **kwargs)  # ty: ignore[invalid-argument-type]
