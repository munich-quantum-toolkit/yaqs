# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for core MPO tensor utility helpers."""

from __future__ import annotations

import numpy as np

from mqt.yaqs.core.data_structures.mpo_utils import gate_tensor_lr_order, identity_mpo_site
from mqt.yaqs.core.libraries.gate_library import GateLibrary


def test_gate_tensor_lr_order_swaps_sites() -> None:
    """Gate tensor axes follow ascending MPS site order."""
    gate = GateLibrary.cx()
    gate.set_sites(2, 0)
    ordered = gate_tensor_lr_order(gate)
    gate.set_sites(0, 2)
    direct = gate_tensor_lr_order(gate)
    np.testing.assert_allclose(ordered, direct)


def test_identity_mpo_site_shape() -> None:
    """Identity MPO site has bond dimension one."""
    site = identity_mpo_site(2)
    assert site.shape == (2, 2, 1, 1)
    np.testing.assert_allclose(site[:, :, 0, 0], np.eye(2))
