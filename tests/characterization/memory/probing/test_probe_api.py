# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff: noqa: PLR6301 -- DummyProcess mimics a protocol-style backend object

"""Tests for split-cut probe_process framework."""

from __future__ import annotations

import numpy as np

from mqt.yaqs.characterization.memory.probing.probe import ProbeSet, probe_process


def test_probe_process_uses_object_backend() -> None:
    """probe_process delegates evaluation to a user-supplied process object."""

    class DummyProcess:
        def evaluate_probe_set(self, probe_set: ProbeSet) -> np.ndarray:
            n_p = len(probe_set.past_pairs)
            n_f = len(probe_set.future_pairs)
            return np.zeros((n_p, n_f, 3), dtype=np.float32)

    out = probe_process(process=DummyProcess(), cut=1, k=1, n_pasts=2, n_futures=3, rng=np.random.default_rng(7))
    assert out["pauli_xyz_ij"].shape == (2, 3, 3)
    assert "entropy" in out
