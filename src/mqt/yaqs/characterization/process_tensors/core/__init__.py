# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Shared internal helpers for process-tensor tomography and surrogates.

This subpackage is internal; call sites should import from
:mod:`mqt.yaqs.characterization.process_tensors.tomography` or
:mod:`mqt.yaqs.characterization.process_tensors.surrogates` instead of relying on these helpers.
"""

from .encoding import build_choi_feature_table
from .metrics import _rel_fro_error, _trace_distance
from .utils import make_mcwf_static_context

__all__ = [
    "build_choi_feature_table",
    "make_mcwf_static_context",
    "_rel_fro_error",
    "_trace_distance",
]
