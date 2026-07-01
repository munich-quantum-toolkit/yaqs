# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Time-dependent variational principle (TDVP) methods.

Public entry point: :func:`mqt.yaqs.core.methods.tdvp.tdvp.tdvp`.
Low-level kernels live in :mod:`mqt.yaqs.core.methods.tdvp.primitives`.
Sweep integrators live in :mod:`mqt.yaqs.core.methods.tdvp.integrators`.
Sweep helpers live in :mod:`mqt.yaqs.core.methods.tdvp.sweep_utils`.
"""

from __future__ import annotations

from .tdvp import tdvp

__all__ = ["tdvp"]
