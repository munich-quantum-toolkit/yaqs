# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"Tests for the Dissipation module."

from __future__ import annotations

import numpy as np

from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.methods.dissipation import apply_dissipation


def test_apply_dissipation_site_canonical_0() -> None:
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
        vec = np.random.rand(pdim).astype(complex)
        vec /= np.linalg.norm(vec)
        tensor = vec.reshape(pdim, 1, 1)
        tensors.append(tensor)

    state = MPS(length=length, tensors=tensors, physical_dimensions=[pdim] * length)

    # 2) Create a minimal NoiseModel with a single process and a small strength.
    noise_model = NoiseModel(processes=["relaxation"], strengths=[0.1])
    dt = 0.1

    # 3) Apply dissipation to the MPS.
    apply_dissipation(state, noise_model, dt)

    # 4) Verify that the MPS is site-canonical at site 0.
    canonical_site = state.check_canonical_form()
    assert canonical_site[0] == 0, (
        f"MPS should be site-canonical at site 0 after apply_dissipation, but got canonical site: {canonical_site}"
    )
