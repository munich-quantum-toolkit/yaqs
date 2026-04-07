# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Shared internal helpers for process-tensor tomography and surrogates.

This subpackage is internal; call sites should import from
:mod:`mqt.yaqs.characterization.process_tensors.tomography` or
:mod:`mqt.yaqs.characterization.process_tensors.surrogates` instead of relying on these helpers.
"""

from .encoding import build_choi_feature_table, normalize_rho_from_backend_output, pack_rho8, unpack_rho8
from .metrics import (
    _mean_frobenius_mse_rho8,
    _mean_trace_distance_rho8,
    _rel_fro_error,
    _trace_distance,
)
from .utils import make_mcwf_static_context

__all__ = [
    "build_choi_feature_table",
    "make_mcwf_static_context",
    "_rel_fro_error",
    "_trace_distance",
    "normalize_rho_from_backend_output",
    "pack_rho8",
    "unpack_rho8",
    "_mean_frobenius_mse_rho8",
    "_mean_trace_distance_rho8",
]
