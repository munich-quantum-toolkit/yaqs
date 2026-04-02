# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Process-tensor tomography (exact/exhaustive).

This subpackage constructs a process tensor from exhaustive discrete intervention sequences (size
``16^k`` for ``k`` steps), optionally under MCWF/TJM noise, and returns a
:class:`~mqt.yaqs.characterization.process_tensors.tomography.combs.DenseComb` or
:class:`~mqt.yaqs.characterization.process_tensors.tomography.combs.MPOComb`.
"""

from .basis import TomographyBasis, build_basis_for_fixed_alphabet, get_basis_states, get_choi_basis
from .combs import DenseComb, MPOComb
from .constructor import construct_process_tensor, run_all_sequences
from .data import SequenceData

__all__ = [
    "DenseComb",
    "MPOComb",
    "SequenceData",
    "TomographyBasis",
    "build_basis_for_fixed_alphabet",
    "construct_process_tensor",
    "get_basis_states",
    "get_choi_basis",
    "run_all_sequences",
]
