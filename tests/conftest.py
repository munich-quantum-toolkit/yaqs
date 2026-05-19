# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared pytest configuration and constants for YAQS tests."""

from __future__ import annotations

import os

# Cap BLAS/OpenMP threads in pytest-xdist workers before numerical libraries spin
# up pools (reduces intermittent OpenBLAS segfaults on some Linux aarch64 setups).
for _name, _val in (
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OMP_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
):
    os.environ.setdefault(_name, _val)

# Default seed for stochastic integration tests (TJM, noisy simulator.run, etc.).
YAQS_TEST_SEED = 42
