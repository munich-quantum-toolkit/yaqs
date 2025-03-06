# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import numpy as np

from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.methods.dissipation import apply_dissipation


def test_apply_dissipation_site_canonical_0() -> None:
    """Check that after calling apply_dissipation, the MPS is site-canonical at site 0.
    This relies on the code's logic to shift orthogonality left after each site.
    """
    # 1) Create a small MPS of length 3 (for example),
    #    with random or simple product tensors.
    length = 3
    pdim = 2
    # We'll build trivial rank-3 tensors: shape => (pdim, 1, 1).
    # So it's effectively a product state with no entanglement.
    tensors = []
    for _ in range(length):
        # random 2-element vector => shape (2,1,1)
        vec = np.random.rand(pdim).astype(complex)
        vec /= np.linalg.norm(vec)  # normalize local
        tensor = vec.reshape(pdim, 1, 1)
        tensors.append(tensor)

    state = MPS(length=length, tensors=tensors, physical_dimensions=[pdim] * length)

    # 2) Create a minimal NoiseModel. Suppose we have 1 jump operator (like sigma_z or random).
    #    We set a small strength for a small dt test.
    noise_model = NoiseModel(processes=["relaxation"], strengths=[0.1])

    dt = 0.1

    # 3) Apply dissipation
    apply_dissipation(state, noise_model, dt)

    # 4) Now check that MPS is site-canonical at site 0.
    canonical_site = state.check_canonical_form()
    assert canonical_site[0] == 0, (
        f"MPS should be site-canonical at site 0 after apply_dissipation, but got canonical site: {canonical_site}"
    )
