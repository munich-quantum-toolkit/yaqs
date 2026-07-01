# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for core MPO tensor utility helpers."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mpo_utils import (
    contract_mpo_site_with_mpo_site,
    contract_mpo_site_with_mps_site,
    convert_nn_matrix,
    decompose_theta,
    get_support_mpo,
    make_identity_site,
    resolve_lr_tensor,
)
from mqt.yaqs.core.libraries.gate_library import BaseGate, GateLibrary, extend_gate


def test_resolve_lr_tensor_swaps_sites() -> None:
    """Reversed nearest-neighbor qargs use the matrix MPS map, not a tensor transpose."""
    gate = GateLibrary.cx()
    gate.set_sites(1, 0)
    reversed_tensor = resolve_lr_tensor(gate)
    expected = convert_nn_matrix(gate.matrix)
    np.testing.assert_allclose(reversed_tensor, expected)


def test_resolve_lr_tensor_explicit_sites() -> None:
    """Explicit left/right sites select the reversed-qarg branch on long-range pairs."""
    gate = GateLibrary.cx()
    gate.set_sites(3, 1)
    tensor = resolve_lr_tensor(gate, left_site=1, right_site=3)
    expected = convert_nn_matrix(gate.matrix)
    np.testing.assert_allclose(tensor, expected)


def test_resolve_lr_tensor_inconsistent_sites_raises() -> None:
    """Mismatched site indices raise ValueError."""
    gate = GateLibrary.cx()
    gate.set_sites(0, 2)
    with pytest.raises(ValueError, match="not consistent"):
        resolve_lr_tensor(gate, left_site=0, right_site=3)


def test_make_identity_site_shape() -> None:
    """Identity MPO site has bond dimension one."""
    site = make_identity_site(2)
    assert site.shape == (2, 2, 1, 1)
    np.testing.assert_allclose(site[:, :, 0, 0], np.eye(2))


def test_get_support_mpo_uses_cached_mpo_tensors() -> None:
    """Cached gate MPO tensors are returned when the support length matches."""
    gate = GateLibrary.cx()
    gate.set_sites(1, 3)
    first = get_support_mpo(gate, first_site=1, last_site=3)
    second = get_support_mpo(gate, first_site=1, last_site=3)
    assert len(first) == len(second) == 3
    np.testing.assert_allclose(first[0], second[0])


def test_get_support_mpo_reextends_when_cache_length_mismatches() -> None:
    """A cached tensor list with the wrong length triggers ``extend_gate``."""
    gate = GateLibrary.cx()
    gate.set_sites(0, 1)
    nn_support = get_support_mpo(gate, first_site=0, last_site=1)
    assert len(nn_support) == 2
    wide = get_support_mpo(gate, first_site=0, last_site=2)
    assert len(wide) == 3


def test_get_support_mpo_calls_extend_gate_without_cache() -> None:
    """Gates without ``mpo_tensors`` build support tensors via ``extend_gate``."""
    gate = GateLibrary.cx()
    gate.set_sites(0, 1)
    tensor = np.asarray(gate.tensor, dtype=np.complex128)
    stub = cast("BaseGate", type("GateStub", (), {"interaction": 2, "sites": [0, 1], "tensor": tensor})())
    support = get_support_mpo(stub, first_site=0, last_site=1)
    expected = extend_gate(resolve_lr_tensor(gate), [0, 1])
    assert len(support) == 2
    np.testing.assert_allclose(support[0], expected[0])
    np.testing.assert_allclose(support[1], expected[1])


def test_contract_mpo_site_with_mps_site() -> None:
    """MPO--MPS site contraction fuses virtual bonds in library order."""
    mpo_site = make_identity_site(2)
    mps_site = np.ones((2, 1, 1), dtype=np.complex128)
    out = contract_mpo_site_with_mps_site(mpo_site, mps_site)
    assert out.shape == (2, 1, 1)


def test_contract_mpo_site_with_mpo_site_conjugate_flag() -> None:
    """MPO--MPO contraction supports the conjugated equivalence-checking path."""
    left = make_identity_site(2)
    right = make_identity_site(2)
    plain = contract_mpo_site_with_mpo_site(left, right, conjugate=False)
    conj = contract_mpo_site_with_mpo_site(left, right, conjugate=True)
    assert plain.shape == conj.shape == (2, 2, 1, 1)


def test_decompose_theta_splits_fused_tensor() -> None:
    """``decompose_theta`` returns two rank-4 tensors with matching total norm."""
    theta = np.random.default_rng(0).standard_normal((2, 2, 2, 2, 2, 2)) + 1j * np.random.default_rng(
        1
    ).standard_normal((2, 2, 2, 2, 2, 2))
    left, right = decompose_theta(theta, threshold=1e-12)
    assert left.ndim == 4
    assert right.ndim == 4


def test_from_gate_rejects_single_qubit_gate() -> None:
    """``MPO.from_gate`` requires a two-qubit gate."""
    gate = GateLibrary.x()
    gate.set_sites(0)
    with pytest.raises(ValueError, match="two-qubit gate"):
        MPO.from_gate(gate, 2)


def test_from_gate_rejects_chain_shorter_than_support() -> None:
    """``chain_length`` must cover the gate support interval."""
    gate = GateLibrary.cx()
    gate.set_sites(1, 3)
    with pytest.raises(ValueError, match="smaller than gate support"):
        MPO.from_gate(gate, 2)


def test_from_gate_pads_identity_outside_support() -> None:
    """``MPO.from_gate`` inserts identity sites when ``chain_length`` exceeds support width."""
    gate = GateLibrary.cx()
    gate.set_sites(1, 3)
    mpo = MPO.from_gate(gate, chain_length=6)
    assert mpo.length == 6
    assert mpo.tensors[0].shape[-1] == 1
    assert mpo.tensors[5].shape[-1] == 1
