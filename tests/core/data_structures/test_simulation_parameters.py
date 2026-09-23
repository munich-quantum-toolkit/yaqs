# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for simulation parameters classes.

This module contains unit tests for simulation parameter classes. It verifies that:
  - AnalogSimParams instances are created with the correct attributes (such as elapsed_time, dt, times,
    sample_timesteps, and num_traj) both with explicit and default values.
  - allocate_observable_buffers properly sets up expectation_values and trajectories arrays
    depending on whether sample_timesteps is True or False.
"""

# ignore non-lowercase variable names for physics notation
# ruff:file-ignore[import-private-name, private-member-access] -- white-box tests of parameter validation and TDVP internals

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.observable import Observable
from mqt.yaqs.core.data_structures.result import Result, aggregate_trajectories, allocate_observable_buffers
from mqt.yaqs.core.data_structures.simulation_parameters import (
    SIMULATION_PRESETS,
    AnalogSimParams,
    DigitalSimParams,
    EvolutionMode,
    _validate_analog_time_grid,
    _validate_tdvp_sweeps,
)
from mqt.yaqs.core.methods.tdvp import primitives as tdvp_primitives

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from mqt.yaqs.core.data_structures.simulation_parameters import (
        GateMode,
        SimulationPreset,
        TDVPMode,
    )


def test_analog_simparams_basic() -> None:
    """Test that AnalogSimParams is initialized with correct parameters.

    This test creates a AnalogSimParams instance with a single observable, total time elapsed_time, time step dt,
    sample_timesteps flag set to True, and a specified number of trajectories num_traj. It then verifies that the
    observables, elapsed_time, dt, times array, sample_timesteps flag, and num_traj are set correctly.
    """
    obs_list = [Observable("x", 0)]
    elapsed_time = 1.0
    dt = 0.2
    params = AnalogSimParams(observables=obs_list, elapsed_time=elapsed_time, dt=dt, num_traj=50)

    assert params.observables == obs_list
    assert params.elapsed_time == elapsed_time
    assert params.dt == dt
    expected_times = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert np.allclose(params.times, expected_times), "Times array should sample 0..elapsed_time in steps of dt."
    assert params.sample_timesteps is True
    assert params.num_traj == 50


def test_analog_simparams_times_no_float_overshoot() -> None:
    """``elapsed_time=0.2, dt=0.1`` must yield ``[0, 0.1, 0.2]``, not an extra 0.3."""
    params = AnalogSimParams(observables=[Observable("x", 0)], elapsed_time=0.2, dt=0.1)
    np.testing.assert_allclose(params.times, [0.0, 0.1, 0.2])
    assert params.times[-1] == pytest.approx(params.elapsed_time)


def test_analog_simparams_zero_elapsed_time() -> None:
    """``elapsed_time=0`` with a valid ``dt`` yields a single-point grid at ``t=0``."""
    params = AnalogSimParams(observables=[Observable("x", 0)], elapsed_time=0.0, dt=0.1)
    assert params.elapsed_time == pytest.approx(0.0)
    assert params.dt == pytest.approx(0.1)
    np.testing.assert_allclose(params.times, [0.0])


@pytest.mark.parametrize(
    ("elapsed_time", "dt"),
    [
        (100.1, 0.1),
        (1.0, 1.0 / 9015),
    ],
)
def test_analog_simparams_accepts_float64_rounding_dust(elapsed_time: float, dt: float) -> None:
    """Ordinary float rounding remains valid on longer and fine-dt grids."""
    params = AnalogSimParams(observables=[Observable("x", 0)], elapsed_time=elapsed_time, dt=dt)

    assert params.times[-1] == pytest.approx(elapsed_time, rel=0.0, abs=0.0)


def test_analog_time_grid_validator_accepts_fine_float64_ratio_without_allocating_grid() -> None:
    """Fine ``dt`` rounding is valid without constructing its 123-million-point grid."""
    assert _validate_analog_time_grid(1.23456789, 1e-8) == 123_456_789


@pytest.mark.parametrize(
    ("elapsed_time", "dt"),
    [
        (0.15, 0.1),
        (0.25, 0.1),
        (5e-13, 1e-12),
        (1.5e-12, 1e-12),
        (1.0, 1e9),
    ],
)
def test_analog_simparams_rejects_nonintegral_duration(elapsed_time: float, dt: float) -> None:
    """Non-integral ``elapsed_time/dt`` must raise rather than mislabel the final time."""
    with pytest.raises(ValueError, match="integer multiple"):
        AnalogSimParams(observables=[Observable("x", 0)], elapsed_time=elapsed_time, dt=dt)


@pytest.mark.parametrize(
    ("elapsed_time", "dt", "match"),
    [
        (-0.1, 0.1, "non-negative"),
        (0.1, 0.0, "positive"),
        (0.1, -0.1, "positive"),
        (float("nan"), 0.1, "finite"),
        (float("inf"), 0.1, "finite"),
        (0.1, float("nan"), "finite"),
        (0.1, float("inf"), "finite"),
        (1e308, 1e-308, "elapsed_time / dt must be finite"),
    ],
)
def test_analog_simparams_rejects_invalid_time_parameters(elapsed_time: float, dt: float, match: str) -> None:
    """Non-finite, zero, or negative time parameters raise clear ValueErrors."""
    with pytest.raises(ValueError, match=match):
        AnalogSimParams(observables=[Observable("x", 0)], elapsed_time=elapsed_time, dt=dt)


@pytest.mark.parametrize(("elapsed_time", "dt"), [(True, 0.1), (0.1, False), ("0.1", 0.1), (0.1, None)])
def test_analog_simparams_rejects_non_numeric_time_parameters(elapsed_time: object, dt: object) -> None:
    """Booleans and non-numeric time parameters raise clear TypeErrors."""
    with pytest.raises(TypeError, match="real number"):
        AnalogSimParams(
            observables=[Observable("x", 0)],
            elapsed_time=cast("Any", elapsed_time),
            dt=cast("Any", dt),
        )


def test_analog_simparams_defaults() -> None:
    """Test the default parameters for AnalogSimParams.

    This test constructs a AnalogSimParams instance with an empty observable list and total time elapsed_time,
    and verifies that default values for dt, sample_timesteps, number of trajectories (num_traj), max_bond_dim,
    svd_threshold, and order are correctly assigned.
    """
    obs_list = [Observable("x", 0)]
    params = AnalogSimParams(observables=obs_list)

    assert params.elapsed_time == pytest.approx(0.1)
    assert params.dt == pytest.approx(0.1)
    assert params.sample_timesteps is True
    # times should be 0..elapsed_time inclusive with spacing dt
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
def test_digital_simparams_presets(preset: SimulationPreset, expected: dict[str, float | int | None]) -> None:
    """DigitalSimParams resolves svd_threshold, max_bond_dim, num_traj, and krylov_tol from presets."""
    params = DigitalSimParams(preset=preset, get_state=True)
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


def test_digital_simparams_shots_default_uses_balanced() -> None:
    """DigitalSimParams(shots=...) uses the balanced preset by default."""
    params = DigitalSimParams(shots=100)
    balanced = SIMULATION_PRESETS["balanced"]
    assert params.preset == "balanced"
    assert params.svd_threshold == pytest.approx(balanced["svd_threshold"])
    assert params.max_bond_dim == balanced["max_bond_dim"]
    assert params.krylov_tol == pytest.approx(balanced["krylov_tol"])
    assert params.gate_mode == "mpo"


def test_gate_mode_defaults_and_validation() -> None:
    """Digital params default to mpo and validate gate_mode names."""
    assert DigitalSimParams(get_state=True).gate_mode == "mpo"
    assert DigitalSimParams(shots=1).gate_mode == "mpo"
    assert DigitalSimParams(get_state=True, gate_mode="full-tdvp").gate_mode == "full-tdvp"
    assert DigitalSimParams(get_state=True, gate_mode="mpo").gate_mode == "mpo"
    with pytest.raises(ValueError, match="gate_mode"):
        DigitalSimParams(get_state=True, gate_mode=cast("GateMode", "invalid"))


def test_tdvp_mode_defaults_and_validation() -> None:
    """All simulation params default to 2site TDVP."""
    assert AnalogSimParams().tdvp_mode == "2site"
    assert DigitalSimParams(get_state=True).tdvp_mode == "2site"
    assert DigitalSimParams(shots=1).tdvp_mode == "2site"
    assert DigitalSimParams(get_state=True, tdvp_mode="1site").tdvp_mode == "1site"
    assert DigitalSimParams(get_state=True, tdvp_mode="2site").tdvp_mode == "2site"
    with pytest.raises(ValueError, match="tdvp_mode"):
        DigitalSimParams(get_state=True, tdvp_mode=cast("TDVPMode", "invalid"))


def test_tdvp_sweeps_defaults_and_validation() -> None:
    """Analog and digital params default tdvp_sweeps to 1 and validate inputs."""
    assert AnalogSimParams().tdvp_sweeps == 1
    assert DigitalSimParams(get_state=True).tdvp_sweeps == 1
    assert DigitalSimParams(shots=1).tdvp_sweeps == 1
    assert DigitalSimParams(get_state=True, tdvp_sweeps=3).tdvp_sweeps == 3
    with pytest.raises(ValueError, match="tdvp_sweeps"):
        DigitalSimParams(get_state=True, tdvp_sweeps=0)
    with pytest.raises(ValueError, match="tdvp_sweeps"):
        DigitalSimParams(get_state=True, tdvp_sweeps=-1)


@pytest.mark.parametrize("invalid", [1.5, True])
def test_tdvp_sweeps_rejects_non_int(invalid: object) -> None:
    """tdvp_sweeps must be a true int, not bool or float."""
    with pytest.raises(TypeError, match="tdvp_sweeps"):
        _validate_tdvp_sweeps(cast("Any", invalid))


_INTEGER_CONTROL_CASES = [
    pytest.param(
        lambda value: AnalogSimParams(num_traj=cast("Any", value)),
        "num_traj",
        1,
        id="analog-num-traj",
    ),
    pytest.param(
        lambda value: DigitalSimParams(num_traj=cast("Any", value)),
        "num_traj",
        1,
        id="digital-num-traj",
    ),
    pytest.param(
        lambda value: AnalogSimParams(max_bond_dim=cast("Any", value)),
        "max_bond_dim",
        1,
        id="analog-max-bond-dim",
    ),
    pytest.param(
        lambda value: DigitalSimParams(max_bond_dim=cast("Any", value)),
        "max_bond_dim",
        1,
        id="digital-max-bond-dim",
    ),
    pytest.param(
        lambda value: DigitalSimParams(num_mid_measurements=cast("Any", value)),
        "num_mid_measurements",
        0,
        id="digital-num-mid-measurements",
    ),
    pytest.param(
        lambda value: DigitalSimParams(shots=cast("Any", value)),
        "shots",
        1,
        id="digital-shots",
    ),
    pytest.param(
        lambda value: AnalogSimParams(tdvp_sweeps=cast("Any", value)),
        "tdvp_sweeps",
        1,
        id="analog-tdvp-sweeps",
    ),
    pytest.param(
        lambda value: DigitalSimParams(tdvp_sweeps=cast("Any", value)),
        "tdvp_sweeps",
        1,
        id="digital-tdvp-sweeps",
    ),
]


@pytest.mark.parametrize(("factory", "field", "minimum"), _INTEGER_CONTROL_CASES)
@pytest.mark.parametrize("invalid", [True, 1.0, 1.5, "1", np.nan, np.inf])
def test_simulation_integer_controls_reject_wrong_types(
    factory: Callable[[object], AnalogSimParams | DigitalSimParams],
    field: str,
    minimum: int,
    invalid: object,
) -> None:
    """Allocation and execution controls require true integer values."""
    del minimum
    with pytest.raises(TypeError, match=field):
        factory(invalid)


@pytest.mark.parametrize(("factory", "field", "minimum"), _INTEGER_CONTROL_CASES)
@pytest.mark.parametrize("distance_below_minimum", [1, 2])
def test_simulation_integer_controls_reject_values_below_minimum(
    factory: Callable[[object], AnalogSimParams | DigitalSimParams],
    field: str,
    minimum: int,
    distance_below_minimum: int,
) -> None:
    """Allocation and execution controls enforce their documented lower bounds."""
    with pytest.raises(ValueError, match=field):
        factory(minimum - distance_below_minimum)


def test_simulation_integer_controls_accept_and_normalize_numpy_integers() -> None:
    """Python and NumPy integers share one stored control representation."""
    analog = AnalogSimParams(
        num_traj=cast("Any", np.int64(2)),
        max_bond_dim=np.int64(3),
        tdvp_sweeps=cast("Any", np.int64(2)),
        order=cast("Any", np.int64(2)),
    )
    digital = DigitalSimParams(
        shots=cast("Any", np.int64(4)),
        num_traj=cast("Any", np.int64(2)),
        max_bond_dim=np.int64(3),
        num_mid_measurements=cast("Any", np.int64(0)),
        tdvp_sweeps=cast("Any", np.int64(2)),
    )

    assert (analog.num_traj, analog.max_bond_dim, analog.tdvp_sweeps, analog.order) == (2, 3, 2, 2)
    assert (digital.shots, digital.num_traj, digital.max_bond_dim, digital.num_mid_measurements) == (4, 2, 3, 0)
    assert digital.tdvp_sweeps == 2
    assert all(
        type(value) is int
        for value in (
            analog.num_traj,
            analog.max_bond_dim,
            analog.tdvp_sweeps,
            analog.order,
            digital.shots,
            digital.num_traj,
            digital.max_bond_dim,
            digital.num_mid_measurements,
            digital.tdvp_sweeps,
        )
    )


@pytest.mark.parametrize("invalid", [True, 1.0, 1.5, "1", np.nan, np.inf])
def test_analog_order_rejects_wrong_types(invalid: object) -> None:
    """Analog order accepts only true integers."""
    with pytest.raises(TypeError, match="order"):
        AnalogSimParams(order=cast("Any", invalid))


@pytest.mark.parametrize("invalid", [-1, 0, 3])
def test_analog_order_rejects_unsupported_values(invalid: int) -> None:
    """Analog order accepts only the two implemented algorithms."""
    with pytest.raises(ValueError, match="order must be 1 or 2"):
        AnalogSimParams(order=invalid)


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ("fast", SIMULATION_PRESETS["fast"]),
        ("balanced", SIMULATION_PRESETS["balanced"]),
        ("accurate", SIMULATION_PRESETS["accurate"]),
        ("exact", SIMULATION_PRESETS["exact"]),
    ],
)
def test_digital_simparams_shots_presets(preset: SimulationPreset, expected: dict[str, float | int | None]) -> None:
    """DigitalSimParams resolves svd_threshold, max_bond_dim, and krylov_tol from the shared presets."""
    params = DigitalSimParams(shots=100, preset=preset)
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


def test_digital_simparams_preset_explicit_overrides() -> None:
    """Explicit numerical arguments override presets."""
    params = DigitalSimParams(
        preset="fast", svd_threshold=1e-8, max_bond_dim=512, num_traj=10, krylov_tol=1e-12, get_state=True
    )
    assert params.svd_threshold == pytest.approx(1e-8)
    assert params.max_bond_dim == 512
    assert params.num_traj == 10
    assert params.krylov_tol == pytest.approx(1e-12)


def test_digital_simparams_shots_preset_explicit_overrides() -> None:
    """Explicit numerical arguments override presets."""
    params = DigitalSimParams(shots=100, preset="fast", svd_threshold=1e-8, max_bond_dim=512, krylov_tol=1e-12)
    assert params.shots == 100
    assert params.svd_threshold == pytest.approx(1e-8)
    assert params.max_bond_dim == 512
    assert params.krylov_tol == pytest.approx(1e-12)


def test_digital_simparams_rejects_invalid_preset() -> None:
    """Invalid preset names raise ValueError."""
    with pytest.raises(ValueError, match="preset must be one of"):
        DigitalSimParams(preset="invalid", get_state=True)  # ty: ignore[invalid-argument-type]


def test_digital_simparams_rejects_none_preset() -> None:
    """preset=None is not supported."""
    with pytest.raises(ValueError, match="preset must be one of"):
        DigitalSimParams(preset=None, get_state=True)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("bad_tol", [0.0, -1.0, float("inf"), float("nan")])
def test_simparams_rejects_invalid_krylov_tol(bad_tol: float) -> None:
    """krylov_tol must be finite and strictly positive."""
    with pytest.raises(ValueError, match="krylov_tol"):
        _ = AnalogSimParams(krylov_tol=bad_tol)


@pytest.mark.parametrize("bad_threshold", [-1.0, float("inf"), float("nan")])
def test_simparams_rejects_invalid_svd_threshold(bad_threshold: float) -> None:
    """svd_threshold must be finite and non-negative."""
    with pytest.raises(ValueError, match="svd_threshold"):
        _ = AnalogSimParams(svd_threshold=bad_threshold)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [("krylov_tol", True), ("svd_threshold", "1e-6")],
)
def test_simparams_rejects_coerced_real_controls(field: str, invalid: object) -> None:
    """Numerical tolerances reject Boolean and string coercion."""
    kwargs: dict[str, Any] = {field: invalid}
    with pytest.raises(TypeError, match=field):
        AnalogSimParams(**kwargs)


def test_simparams_allows_zero_svd_threshold() -> None:
    """Zero SVD threshold is allowed for exact-zero / hard-cap truncation protocols."""
    params = AnalogSimParams(svd_threshold=0.0)
    assert params.svd_threshold == pytest.approx(0.0)


def test_evolution_mode_validation() -> None:
    """AnalogSimParams accepts BUG and rejects unknown evolution modes."""
    params = AnalogSimParams(evolution_mode=EvolutionMode.BUG)
    assert params.evolution_mode is EvolutionMode.BUG
    assert AnalogSimParams(evolution_mode="bug").evolution_mode is EvolutionMode.BUG

    with pytest.raises(ValueError, match="evolution_mode"):
        _ = AnalogSimParams(evolution_mode="not-a-mode")


@pytest.mark.parametrize(
    "trunc_mode",
    ["discarded_weight", "relative", "hard_cutoff", "relative_discarded_weight"],
)
def test_trunc_mode_accepted_by_analog_and_digital(trunc_mode: str) -> None:
    """Analog and digital params accept every mode validated by _validate_trunc_mode."""
    assert AnalogSimParams(trunc_mode=trunc_mode).trunc_mode == trunc_mode
    assert DigitalSimParams(get_state=True, trunc_mode=trunc_mode).trunc_mode == trunc_mode


@pytest.mark.parametrize("bad", ["nope", ["discarded_weight"], 1, None])
def test_trunc_mode_rejects_unsupported_values(bad: object) -> None:
    """Non-supported trunc_mode values raise ValueError for analog and digital params."""
    with pytest.raises(ValueError, match="trunc_mode"):
        _ = AnalogSimParams(trunc_mode=cast("Any", bad))
    with pytest.raises(ValueError, match="trunc_mode"):
        _ = DigitalSimParams(get_state=True, trunc_mode=cast("Any", bad))


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
        observables=[Observable("x", 1)],
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
        observables=[Observable("x", 0)],
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


def test_aggregate_trajectories_regular_observable_mean() -> None:
    """Regular observables: results = mean(trajectories, axis=0).

    We create a single-site Z observable with a (num_traj x T) trajectory array and
    verify that `results` equals the columnwise mean.
    """
    # Observable to aggregate
    z_obs = Observable("z", sites=0)

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
    ss_obs = Observable("schmidt_spectrum", sites=[1, 2])

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
    x_obs = Observable("x", sites=2)
    x_traj = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], dtype=np.float64)

    ss_obs = Observable("schmidt_spectrum", sites=[0, 1])
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
    ss_obs = Observable("schmidt_spectrum", sites=[2, 3])
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


def test_digital_params_sorting_and_fields() -> None:
    """Constructor sorts non-PVM observables by site; PVM observables are appended."""
    obs_z3 = Observable("z", sites=3)
    obs_x2 = Observable("x", sites=2)
    obs_y1 = Observable("y", sites=1)
    obs_ssp = Observable("schmidt_spectrum", sites=[1, 2])

    params = DigitalSimParams(
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
    params.observables.append(Observable("z", sites=0))
    assert len(params.sorted_observables) == 5
    assert params.observable_sorted_indices == (4, 3, 1, 2, 0)

    # Parameter fields are retained
    assert params.num_traj == 7
    assert params.max_bond_dim == 128
    assert params.svd_threshold == pytest.approx(SIMULATION_PRESETS["balanced"]["svd_threshold"])
    assert params.get_state is True
    assert params.sample_layers is True
    assert params.num_mid_measurements == 2


def test_analog_params_rejects_mixed_pvm_with_non_pvm() -> None:
    """The analog constructor rejects mixed PVM and non-PVM observables."""
    pvm = Observable("101")
    z0 = Observable("z", sites=0)
    with pytest.raises(ValueError, match="Mixed observable and projective-measurement simulation is not supported"):
        _ = AnalogSimParams(observables=[pvm, z0])


@pytest.mark.parametrize(
    "multi_time_observables",
    [
        [(Observable("z", 0),)],
        [(Observable("z", 0), object())],
    ],
)
def test_analog_params_rejects_invalid_multi_time_observable_pairs(multi_time_observables: object) -> None:
    """Each two-time entry must contain exactly two Observable objects."""
    with pytest.raises(TypeError, match=r"multi_time_observables\[0\].*pair of Observable"):
        AnalogSimParams(multi_time_observables=cast("Any", multi_time_observables))


@pytest.mark.parametrize(
    "unsupported",
    [Observable("entropy", [0, 1]), Observable("00")],
)
def test_analog_params_rejects_unsupported_multi_time_observables(unsupported: Observable) -> None:
    """Two-time pairs accept only local operators supported by ensemble evolution."""
    with pytest.raises(ValueError, match="one- or two-site operator observables"):
        AnalogSimParams(multi_time_observables=[(unsupported, Observable("z", 0))])


def test_digital_params_rejects_mixed_pvm_with_non_pvm() -> None:
    """The digital constructor rejects mixed PVM and non-PVM observables."""
    pvm = Observable("101")
    z0 = Observable("z", sites=0)
    with pytest.raises(ValueError, match="Mixed observable and projective-measurement simulation is not supported"):
        _ = DigitalSimParams(observables=[pvm, z0])


def test_digital_params_accepts_all_pvm_or_all_non_pvm() -> None:
    """Constructor allows all-PVM and all-non-PVM sets."""
    # All PVM
    p1 = Observable("0")
    p2 = Observable("1")
    _ = DigitalSimParams(observables=[p1, p2])  # should not raise

    # All non-PVM
    z0 = Observable("z", sites=0)
    x1 = Observable("x", sites=1)
    _ = DigitalSimParams(observables=[z0, x1])  # should not raise


def test_digital_aggregate_regular_mean() -> None:
    """Regular observables: results = mean(trajectories, axis=0)."""
    x = Observable("x", sites=2)
    traj = np.array(
        [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    params = DigitalSimParams(observables=[x], num_traj=3)
    run_result = Result(sim_params=params, observables=[x], trajectories=[traj], expectation_values=[np.empty(3)])
    aggregate_trajectories(run_result)

    np.testing.assert_allclose(run_result.expectation_values[0], traj.mean(axis=0))


def test_digital_aggregate_schmidt_concat() -> None:
    """Schmidt spectrum: concatenation of raveled list entries."""
    ssp = Observable("schmidt_spectrum", sites=[0, 1])
    ssp_traj = np.array([
        np.array([0.9, 0.8], dtype=np.float64),
        np.array([0.6, 0.4], dtype=np.float64),
        np.array([0.2, 0.1], dtype=np.float64),
    ])

    params = DigitalSimParams(observables=[ssp], num_traj=3)
    run_result = Result(sim_params=params, observables=[ssp], trajectories=[ssp_traj], expectation_values=[np.empty(6)])
    aggregate_trajectories(run_result)

    np.testing.assert_allclose(
        run_result.expectation_values[0], np.array([0.9, 0.8, 0.6, 0.4, 0.2, 0.1], dtype=np.float64)
    )


def test_digital_aggregate_mixed_regular_and_schmidt() -> None:
    """Combination case: regular and Schmidt updated correctly in one call."""
    z = Observable("z", sites=0)
    z_traj = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    ssp = Observable("schmidt_spectrum", sites=[1, 2])
    ssp_traj = np.array([np.array([1.0, 0.5], dtype=np.float64), np.array([0.5, 0.25], dtype=np.float64)])

    params = DigitalSimParams(observables=[z, ssp], num_traj=2)
    run_result = Result(
        sim_params=params,
        observables=[z, ssp],
        trajectories=[z_traj, ssp_traj],
        expectation_values=[np.empty(2), np.empty(4)],
    )
    aggregate_trajectories(run_result)

    np.testing.assert_allclose(run_result.expectation_values[0], np.array([2.0, 3.0], dtype=np.float64))
    np.testing.assert_allclose(run_result.expectation_values[1], np.array([1.0, 0.5, 0.5, 0.25], dtype=np.float64))


def test_digital_aggregate_schmidt_requires_array() -> None:
    """Schmidt branch must assert if trajectories is not an array."""
    ssp = Observable("schmidt_spectrum", sites=[0, 1])
    bad_traj = [0.9, 0.1]

    params = DigitalSimParams(observables=[ssp], num_traj=1)

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
        (DigitalSimParams, {"shots": 1}),
        (DigitalSimParams, {"get_state": True}),
    ],
)
def test_random_seed_rejects_invalid_type(
    param_cls: type[AnalogSimParams | DigitalSimParams],
    kwargs: dict[str, object],
) -> None:
    """random_seed must be None or int."""
    with pytest.raises(TypeError, match="random_seed must be an integer"):
        param_cls(random_seed="not-a-seed", **kwargs)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("param_cls", "kwargs"),
    [
        (AnalogSimParams, {}),
        (DigitalSimParams, {"shots": 1}),
        (DigitalSimParams, {"get_state": True}),
    ],
)
def test_random_seed_rejects_negative(
    param_cls: type[AnalogSimParams | DigitalSimParams],
    kwargs: dict[str, object],
) -> None:
    """random_seed must be non-negative when set."""
    with pytest.raises(ValueError, match="random_seed must be >= 0"):
        param_cls(random_seed=-1, **kwargs)  # ty: ignore[invalid-argument-type]


def test_simparams_accept_numpy_scalar_flags_and_seed() -> None:
    """Equivalent NumPy Boolean and integer controls normalize to Python scalars."""
    analog = AnalogSimParams(
        sample_timesteps=cast("Any", np.zeros((), dtype=np.bool_)[()]),
        get_state=cast("Any", np.ones((), dtype=np.bool_)[()]),
        random_seed=cast("Any", np.int64(4)),
    )
    digital = DigitalSimParams(
        get_state=cast("Any", np.ones((), dtype=np.bool_)[()]),
        sample_layers=cast("Any", np.zeros((), dtype=np.bool_)[()]),
        random_seed=cast("Any", np.int64(5)),
    )

    assert analog.sample_timesteps is False
    assert analog.get_state is True
    assert analog.random_seed == 4
    assert digital.get_state is True
    assert digital.sample_layers is False
    assert digital.random_seed == 5


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda: AnalogSimParams(sample_timesteps=cast("Any", "false")), id="analog-sampling"),
        pytest.param(lambda: AnalogSimParams(get_state=cast("Any", 1)), id="analog-state"),
        pytest.param(lambda: DigitalSimParams(get_state=cast("Any", "true")), id="digital-state"),
        pytest.param(lambda: DigitalSimParams(sample_layers=cast("Any", 0)), id="digital-sampling"),
    ],
)
def test_simparams_reject_non_boolean_flags(factory: Callable[[], object]) -> None:
    """Simulation output flags require Boolean values rather than truthiness."""
    with pytest.raises(TypeError, match="must be a boolean"):
        factory()


def test_digital_simparams_allows_outputless_program_configuration() -> None:
    """An output-less digital configuration can propagate program state."""
    params = DigitalSimParams()

    assert params.observables == []
    assert params.shots is None
    assert not params.get_state


def test_digital_simparams_allows_sample_layers_without_observables() -> None:
    """sample_layers may be set alone so a SimulationProgram can inject observables later."""
    params = DigitalSimParams(sample_layers=True)

    assert params.sample_layers
    assert params.observables == []


def test_digital_simparams_rejects_positional_arguments() -> None:
    """All DigitalSimParams constructor arguments are keyword-only."""
    obs = [Observable("z", 0)]
    with pytest.raises(TypeError, match=r"keyword-only|takes 1 positional"):
        DigitalSimParams(obs)  # ty: ignore[too-many-positional-arguments]
    with pytest.raises(TypeError, match=r"keyword-only|takes 1 positional"):
        DigitalSimParams(obs, 7, 64)  # ty: ignore[too-many-positional-arguments]
    with pytest.raises(TypeError, match=r"keyword-only|takes 1 positional"):
        DigitalSimParams(1024, 4)  # ty: ignore[too-many-positional-arguments]
