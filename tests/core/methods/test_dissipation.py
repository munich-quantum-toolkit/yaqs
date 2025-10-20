# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the Dissipation module."""

from __future__ import annotations

import numpy as np

from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.core.methods.dissipation import apply_dissipation, is_adjacent, is_longrange

rng = np.random.default_rng()


def test_apply_dissipation_one_site_canonical_0() -> None:
    """Test that apply_dissipation correctly shifts the MPS to be site-canonical at site 0.

    This test constructs a simple product-state MPS of length 3, where each tensor is of shape (pdim, 1, 1),
    representing an unentangled state. A minimal NoiseModel with one jump operator is created with a small strength,
    and apply_dissipation is applied with a small time step dt. Finally, the test checks that the orthogonality
    center of the MPS is shifted to site 0, as expected.
    """
    # 1) Create a simple product-state MPS of length 3.
    length = 3
    pdim = 2
    tensors = []
    for _ in range(length):
        # Create a random 2-element vector, normalize it, and reshape to (pdim, 1, 1)
        vec = rng.random(pdim).astype(complex)
        vec /= np.linalg.norm(vec)
        tensor = vec.reshape(pdim, 1, 1)
        tensors.append(tensor)

    state = MPS(length=length, tensors=tensors, physical_dimensions=[pdim] * length)

    noise_model = NoiseModel([
        {"name": name, "sites": [i], "strength": 0.1} for i in range(length) for name in ["lowering", "pauli_z"]
    ])
    dt = 0.1
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)

    apply_dissipation(state, noise_model, dt, sim_params)

    canonical_site = state.check_canonical_form()
    assert canonical_site[0] == 0, (
        f"MPS should be site-canonical at site 0 after apply_dissipation, but got canonical site: {canonical_site}"
    )


def test_apply_dissipation_two_site_canonical_0() -> None:
    """Test that apply_dissipation correctly shifts the MPS to be site-canonical at site 0.

    This test constructs a simple product-state MPS of length 3, where each tensor is of shape (pdim, 1, 1),
    representing an unentangled state. A minimal NoiseModel with two 2-site jump operators is created
    with a small strength, and apply_dissipation is applied with a small time step dt.
    Finally, the test checks that the orthogonality center of the MPS is shifted to site 0, as expected.
    """
    # 1) Create a simple product-state MPS of length 3.
    length = 3
    pdim = 2
    tensors = []
    for _ in range(length):
        # Create a random 2-element vector, normalize it, and reshape to (pdim, 1, 1)
        vec = rng.random(pdim).astype(complex)
        vec /= np.linalg.norm(vec)
        tensor = vec.reshape(pdim, 1, 1)
        tensors.append(tensor)

    state = MPS(length=length, tensors=tensors, physical_dimensions=[pdim] * length)

    noise_model = NoiseModel([
        {"name": name, "sites": [i, i + 1], "strength": 0.1}
        for i in range(length - 1)
        for name in ["crosstalk_xx", "crosstalk_yy"]
    ])
    dt = 0.1
    sim_params = AnalogSimParams(get_state=True, elapsed_time=0.0)

    apply_dissipation(state, noise_model, dt, sim_params)

    canonical_site = state.check_canonical_form()
    assert canonical_site[0] == 0, (
        f"MPS should be site-canonical at site 0 after apply_dissipation, but got canonical site: {canonical_site}"
    )


def test_is_adjacent_and_is_longrange() -> None:
    """Test adjacency helpers for two-site processes.

    Verifies that `is_adjacent` returns True for nearest neighbors and False otherwise,
    and that `is_longrange` returns True only for non-neighbor pairs.
    """
    proc_adj = {"sites": [0, 1]}
    proc_adj_unsorted = {"sites": [2, 1]}
    proc_long = {"sites": [0, 2]}
    proc_far = {"sites": [1, 3]}

    assert is_adjacent(proc_adj) is True
    assert is_adjacent(proc_adj_unsorted) is True
    assert is_adjacent(proc_long) is False
    assert is_adjacent(proc_far) is False

    assert is_longrange(proc_adj) is False
    assert is_longrange(proc_adj_unsorted) is False
    assert is_longrange(proc_long) is True
    assert is_longrange(proc_far) is True
