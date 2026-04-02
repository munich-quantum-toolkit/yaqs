# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Encoding utilities shared by process-tensor and surrogate workflows.

Core modules must be used by both stacks; this file therefore contains only the fixed-basis Choi
feature table encoding, which is used by the process-tensor basis code and by surrogate utilities.
"""

from __future__ import annotations

import numpy as np

def _flatten_choi4_to_real32(j: np.ndarray) -> np.ndarray:
    """Vectorize 4x4 complex Choi matrix to 32 floats (Re/Im, row-major)."""
    m = np.asarray(j, dtype=np.complex128).reshape(4, 4)
    flat = m.reshape(-1)
    interleaved = np.stack([flat.real, flat.imag], axis=-1).astype(np.float32)
    return interleaved.reshape(-1)


def build_choi_feature_table(choi_matrices: list[np.ndarray]) -> np.ndarray:
    """Return array of shape (16, 32): one feature row per fixed-basis index."""
    rows = [_flatten_choi4_to_real32(c) for c in choi_matrices]
    return np.stack(rows, axis=0)

