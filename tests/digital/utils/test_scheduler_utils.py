# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for equivalence scheduling helpers."""

from __future__ import annotations

from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.digital.utils.scheduler_utils import gates_have_disjoint_sites, partition_disjoint_gate_batches


def test_gates_have_disjoint_sites() -> None:
    """Disjoint-site predicate matches partition batching expectations."""
    x0 = GateLibrary.x()
    x0.set_sites(0)
    x1 = GateLibrary.x()
    x1.set_sites(1)
    cx = GateLibrary.cx()
    cx.set_sites(0, 1)

    assert gates_have_disjoint_sites(x0, x1) is True
    assert gates_have_disjoint_sites(x0, cx) is False


def test_partition_disjoint_gate_batches() -> None:
    """Disjoint single-qubit gates belong to one batch; overlapping gates do not."""
    x0 = GateLibrary.x()
    x0.set_sites(0)
    x1 = GateLibrary.x()
    x1.set_sites(1)
    cx = GateLibrary.cx()
    cx.set_sites(0, 1)

    batches = partition_disjoint_gate_batches([x0, x1])
    assert len(batches) == 1
    assert len(batches[0]) == 2

    batches_overlap = partition_disjoint_gate_batches([x0, cx])
    assert len(batches_overlap) == 2
