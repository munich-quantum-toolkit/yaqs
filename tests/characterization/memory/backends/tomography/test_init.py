# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Smoke tests for tomography package re-exports."""

from __future__ import annotations

from mqt.yaqs.characterization.memory.backends import tomography
from mqt.yaqs.characterization.memory.backends.tomography.basis import TomographyBasis
from mqt.yaqs.characterization.memory.backends.tomography.process_tensors import DenseProcessTensor, MPOProcessTensor


def test_tomography_package_reexports() -> None:
    """Public symbols resolve through the package import path."""
    assert set(tomography.__all__) == {"DenseProcessTensor", "MPOProcessTensor", "TomographyBasis"}
    assert tomography.DenseProcessTensor is DenseProcessTensor
    assert tomography.MPOProcessTensor is MPOProcessTensor
    assert tomography.TomographyBasis is TomographyBasis
    for name in (
        "assemble_fixed_basis",
        "get_basis_states",
        "get_choi_basis",
        "SequenceData",
        "build_process_tensor",
        "evaluate_probes",
        "compute_temporal_entropy",
    ):
        assert not hasattr(tomography, name), name
