# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the public TDVP entry point and sweep orchestration."""

# ignore non-lowercase variable names for physics notation
# ruff: noqa: N806, PLC2701

from __future__ import annotations

from typing import Any, cast
from unittest.mock import patch

import pytest

from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable, StrongSimParams
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs.core.methods.tdvp import tdvp
from mqt.yaqs.core.methods.tdvp.tdvp import _run_sweeps, evolve_window


def test_run_sweeps_invokes_substeps() -> None:
    """_run_sweeps batches symmetric substeps for analog and digital."""
    L = 3
    H = MPO.ising(L, 1.0, 0.5)
    state = MPS(L, state="zeros")

    captured_analog: list[float] = []

    def _capture_analog(*_args: object, sweep_plan: list[float] | None = None, **_kwargs: object) -> None:
        """Record analog substep scales passed via ``sweep_plan``."""
        if sweep_plan is not None:
            captured_analog.extend(sweep_plan)

    analog_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        tdvp_sweeps=3,
        sample_timesteps=True,
    )
    _run_sweeps(_capture_analog, state, H, analog_params)
    assert len(captured_analog) == 3
    for scale in captured_analog:
        assert scale == pytest.approx(1 / 3)

    captured_plan: list[float] = []

    def _capture_plan(*_args: object, sweep_plan: list[float] | None = None, **_kwargs: object) -> None:
        """Record digital substep scales passed via ``sweep_plan``."""
        if sweep_plan is not None:
            captured_plan.extend(sweep_plan)

    digital_params = StrongSimParams(observables=[Observable(Z(), 0)], tdvp_sweeps=3, preset="exact")
    _run_sweeps(_capture_plan, state, H, digital_params)
    assert len(captured_plan) == 3
    for scale in captured_plan:
        assert scale == pytest.approx(1 / 3)

    captured_plan.clear()
    digital_one = StrongSimParams(observables=[Observable(Z(), 0)], tdvp_sweeps=1, preset="exact")
    _run_sweeps(_capture_plan, state, H, digital_one)
    assert captured_plan == [1.0]

    analog_one = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
        tdvp_sweeps=1,
        sample_timesteps=True,
    )
    captured_analog.clear()
    _run_sweeps(_capture_analog, state, H, analog_one)
    assert captured_analog == captured_plan


def test_dynamic_sweep_scaling() -> None:
    """Explicit dynamic mode honors tdvp_sweeps via a sweep plan."""
    L = 4
    H = MPO.ising(L, 1.0, 0.5)
    state = MPS(L, state="zeros")
    sim_params = StrongSimParams(
        observables=[Observable(Z(), 0)],
        tdvp_sweeps=2,
        preset="exact",
        tdvp_mode="dynamic",
    )

    with patch("mqt.yaqs.core.methods.tdvp.integrators.sweep_dynamic") as mock_sweep:
        tdvp(state, H, sim_params)
        assert mock_sweep.call_count == 1
        assert len(mock_sweep.call_args.kwargs["sweep_plan"]) == 2


def test_tdvp_mode_dispatch() -> None:
    """Tdvp routes each tdvp_mode to the matching private sweep kernel."""
    L = 4
    H = MPO.ising(L, 1.0, 0.5)
    state = MPS(L, state="zeros")

    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], preset="exact", tdvp_mode="1site")
    with patch("mqt.yaqs.core.methods.tdvp.integrators.sweep_1site") as mock_one:
        tdvp(state, H, sim_params)
        mock_one.assert_called_once()

    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], preset="exact", tdvp_mode="2site")
    with patch("mqt.yaqs.core.methods.tdvp.integrators.sweep_2site") as mock_two:
        tdvp(state, H, sim_params)
        mock_two.assert_called_once()

    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], preset="exact", tdvp_mode="dynamic")
    with patch("mqt.yaqs.core.methods.tdvp.integrators.sweep_dynamic") as mock_dyn:
        tdvp(state, H, sim_params)
        mock_dyn.assert_called_once()


def test_tdvp_default_mode_is_2site() -> None:
    """AnalogSimParams default tdvp_mode uses two-site TDVP."""
    L = 4
    H = MPO.ising(L, 1.0, 0.5)
    state = MPS(L, state="zeros")
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=False,
    )

    with patch("mqt.yaqs.core.methods.tdvp.integrators.sweep_2site") as mock_two:
        tdvp(state, H, sim_params)
        mock_two.assert_called_once()


def test_strong_default_mode_is_2site() -> None:
    """StrongSimParams default tdvp_mode uses two-site TDVP."""
    L = 4
    H = MPO.ising(L, 1.0, 0.5)
    state = MPS(L, state="zeros")
    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], preset="exact")

    with patch("mqt.yaqs.core.methods.tdvp.integrators.sweep_2site") as mock_two:
        tdvp(state, H, sim_params)
        mock_two.assert_called_once()


def test_evolve_window_rejects_single_site_window() -> None:
    """Window-local TDVP requires at least two sites."""
    state = MPS(1, state="zeros")
    hamiltonian = MPO.ising(1, 1.0, 0.5)
    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], preset="exact")
    with pytest.raises(ValueError, match="at least two sites"):
        evolve_window(state, hamiltonian, sim_params)


def test_evolve_window_no_drift_renorm() -> None:
    """Window-local TDVP disables per-sweep drift renorm before grafting."""
    L = 4
    H = MPO.ising(L, 1.0, 0.5)
    state = MPS(L, state="zeros")
    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], preset="exact")

    with patch("mqt.yaqs.core.methods.tdvp.integrators.sweep_2site") as mock_two:
        evolve_window(state, H, sim_params)
        mock_two.assert_called_once()
        assert mock_two.call_args.kwargs.get("drift_renorm") is False


def test_tdvp_rejects_invalid_tdvp_sweeps_at_runtime() -> None:
    """Mutated tdvp_sweeps below one fails fast in the sweep runner."""
    state = MPS(4, state="zeros")
    hamiltonian = MPO.ising(4, 1.0, 0.5)
    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], preset="exact")
    sim_params.tdvp_sweeps = 0
    with pytest.raises(ValueError, match="tdvp_sweeps"):
        tdvp(state, hamiltonian, sim_params)


def test_tdvp_rejects_unknown_tdvp_mode_at_runtime() -> None:
    """Mutated tdvp_mode outside the supported set raises instead of falling through."""
    state = MPS(4, state="zeros")
    hamiltonian = MPO.ising(4, 1.0, 0.5)
    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], preset="exact")
    sim_params.tdvp_mode = cast("Any", "invalid")
    with pytest.raises(ValueError, match="tdvp_mode"):
        tdvp(state, hamiltonian, sim_params)


def test_tdvp_rejects_operator_length_mismatch() -> None:
    """MPS and MPO length mismatch is rejected before sweep dispatch."""
    state = MPS(3, state="zeros")
    hamiltonian = MPO.ising(4, 1.0, 0.5)
    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], preset="exact")
    with pytest.raises(ValueError, match="same number of sites"):
        tdvp(state, hamiltonian, sim_params)


def test_tdvp_2site_falls_back_on_single_site() -> None:
    """Two-site TDVP on a one-site chain falls back to 1-site TDVP."""
    state = MPS(1, state="zeros")
    hamiltonian = MPO.ising(1, 1.0, 0.5)
    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], preset="exact", tdvp_mode="2site")
    with patch("mqt.yaqs.core.methods.tdvp.integrators.sweep_1site") as mock_one:
        tdvp(state, hamiltonian, sim_params)
        mock_one.assert_called_once()


def test_run_sweeps_rejects_invalid_tdvp_sweeps() -> None:
    """_run_sweeps validates tdvp_sweeps before invoking the integrator."""
    state = MPS(3, state="zeros")
    hamiltonian = MPO.ising(3, 1.0, 0.5)
    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], preset="exact")
    sim_params.tdvp_sweeps = 0
    with pytest.raises(ValueError, match="tdvp_sweeps"):
        _run_sweeps(lambda *_a, **_k: None, state, hamiltonian, sim_params)


def test_single_site_fallback_uses_1site() -> None:
    """Two-site and dynamic modes on a one-site chain fall back to 1-site TDVP."""
    H = MPO.ising(1, 1.0, 0.5)
    state = MPS(1, state="zeros")
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
        sample_timesteps=False,
    )

    with patch("mqt.yaqs.core.methods.tdvp.integrators.sweep_1site") as mock_one:
        tdvp(state, H, sim_params)
        mock_one.assert_called_once()
