# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Scheduling helpers for parallel equivalence checking."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...core.libraries.gate_library import BaseGate


def gates_have_disjoint_sites(left: BaseGate, right: BaseGate) -> bool:
    """Return whether two gates act on disjoint qubit sets."""
    return not (set(left.sites) & set(right.sites))


def partition_disjoint_gate_batches(gates: list[BaseGate]) -> list[list[BaseGate]]:
    """Partition gates into batches of pairwise disjoint site sets.

    Gates within a batch can be applied in parallel; batches are applied sequentially.

    Returns:
        Disjoint gate batches in input order (within each batch).
    """
    batches: list[list[BaseGate]] = []
    for gate in gates:
        for batch in batches:
            if all(gates_have_disjoint_sites(gate, other) for other in batch):
                batch.append(gate)
                break
        else:
            batches.append([gate])
    return batches
