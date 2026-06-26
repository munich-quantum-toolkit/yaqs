# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Process-tensor tomography (exact/exhaustive).

This subpackage constructs a process tensor from exhaustive discrete intervention sequences (size
``16^k`` for ``k`` steps), optionally under MCWF/TJM noise, and returns a
:class:`~mqt.yaqs.characterization.memory.backends.tomography.combs.DenseComb` or
:class:`~mqt.yaqs.characterization.memory.backends.tomography.combs.MPOComb`.
"""

from .basis import TomographyBasis as TomographyBasis
from .basis import assemble_fixed_basis as assemble_fixed_basis
from .basis import get_basis_states as get_basis_states
from .basis import get_choi_basis as get_choi_basis
from .combs import DenseComb as DenseComb
from .combs import MPOComb as MPOComb
from .constructor import build_process_tensor as build_process_tensor
from .constructor import run_all_sequences as run_all_sequences
from .data import SequenceData as SequenceData
