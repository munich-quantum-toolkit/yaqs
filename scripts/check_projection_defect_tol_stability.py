#!/usr/bin/env python
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Alias for the projection defect tolerance stability sweep.

Run:

    uv run python -m scripts.check_projection_defect_tol_stability

This forwards to :mod:`scripts.check_projection_accept_ratio_stability`, which
implements the projection-defect tolerance sweep and writes:

- ``results/projection_defect_tol_stability.csv``
- ``results/projection_defect_tol_stability.md``
"""

from scripts.check_projection_accept_ratio_stability import main as _main


if __name__ == "__main__":
    _main()

