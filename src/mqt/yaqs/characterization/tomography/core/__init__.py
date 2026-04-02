# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Shared tomography infrastructure.

Exact comb reconstruction lives in :mod:`mqt.yaqs.characterization.tomography.exact`.
Neural surrogates live in :mod:`mqt.yaqs.characterization.tomography.surrogate`.

Rule: anything in ``core/`` must be used by at least two of ``exact/``, ``estimate/``, and
``surrogate/``.
"""
