# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import X, Z, XX
from mqt.yaqs.core.methods.scheduled_jumps import apply_scheduled_jumps, has_scheduled_jump


def test_has_scheduled_jump() -> None:
    """Tests the has_scheduled_jump helper function."""
    # Case 1: noise_model is None
    assert not has_scheduled_jump(None, 1.0, 0.1)

    # Case 2: scheduled_jumps list is empty
    noise_model = NoiseModel()
    assert not has_scheduled_jump(noise_model, 1.0, 0.1)

    # Case 3: Exact match
    scheduled_jumps = [{"time": 1.0, "sites": [0], "name": "x"}]
    noise_model = NoiseModel(scheduled_jumps=scheduled_jumps)
    assert has_scheduled_jump(noise_model, 1.0, 0.1)

    # Case 4: Match within tolerance
    assert has_scheduled_jump(noise_model, 1.0001, 0.1)
    
    # Case 5: No match
    assert not has_scheduled_jump(noise_model, 2.0, 0.1)


def test_apply_scheduled_jumps_single_site() -> None:
    """Tests applying a single-site scheduled jump."""
    L = 2
    state = MPS(L, state="zeros") # |00>
    state.normalize("B")

    scheduled_jumps = [{"time": 1.0, "sites": [0], "name": "x"}]
    noise_model = NoiseModel(scheduled_jumps=scheduled_jumps)
    
    sim_params = AnalogSimParams(dt=0.1, get_state=True)
    
    # Apply jump at t=1.0
    new_state = apply_scheduled_jumps(state, noise_model, 1.0, sim_params)
    
    # Expect |10>
    # Measure Z on site 0 and 1
    z_obs0 = Observable(Z(), sites=0)
    z_obs1 = Observable(Z(), sites=1)
    
    assert np.isclose(new_state.expect(z_obs0), -1.0)
    assert np.isclose(new_state.expect(z_obs1), 1.0)


def test_apply_scheduled_jumps_two_site() -> None:
    """Tests applying a two-site scheduled jump."""
    L = 2
    state = MPS(L, state="zeros") # |00>
    state.normalize("B")

    # ZZ jump should do nothing to |00> in terms of expectation?
    # No, let's use XX jump. |00> -> |11>
    scheduled_jumps = [{"time": 1.0, "sites": [0, 1], "name": "crosstalk_xx"}]
    noise_model = NoiseModel(scheduled_jumps=scheduled_jumps)
    
    sim_params = AnalogSimParams(dt=0.1, get_state=True)
    
    # Apply jump at t=1.0
    new_state = apply_scheduled_jumps(state, noise_model, 1.0, sim_params)
    
    # Expect |11>
    z_obs0 = Observable(Z(), sites=0)
    z_obs1 = Observable(Z(), sites=1)
    
    new_state.set_canonical_form(0)
    exp0 = new_state.expect(z_obs0)
    new_state.set_canonical_form(1)
    exp1 = new_state.expect(z_obs1)
    
    assert np.isclose(exp0, -1.0)
    assert np.isclose(exp1, -1.0)


def test_apply_scheduled_jumps_multiple() -> None:
    """Tests applying multiple scheduled jumps at the same time."""
    L = 3
    state = MPS(L, state="zeros") # |000>
    state.normalize("B")

    # Jumps on site 0 and site 2 at t=1.0
    scheduled_jumps = [
        {"time": 1.0, "sites": [0], "name": "x"},
        {"time": 1.0, "sites": [2], "name": "x"},
    ]
    noise_model = NoiseModel(scheduled_jumps=scheduled_jumps)
    
    sim_params = AnalogSimParams(dt=0.1, get_state=True)
    
    # Apply jumps at t=1.0
    new_state = apply_scheduled_jumps(state, noise_model, 1.0, sim_params)
    
    # Expect |101>
    z_obs0 = Observable(Z(), sites=0)
    z_obs1 = Observable(Z(), sites=1)
    z_obs2 = Observable(Z(), sites=2)
    
    new_state.set_canonical_form(0)
    assert np.isclose(new_state.expect(z_obs0), -1.0)
    new_state.set_canonical_form(1)
    assert np.isclose(new_state.expect(z_obs1), 1.0)
    new_state.set_canonical_form(2)
    assert np.isclose(new_state.expect(z_obs2), -1.0)
