# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Process-tensors characterization.

This package contains:

- :mod:`mqt.yaqs.characterization.process_tensors.tomography`: exhaustive/discrete process-tensor reconstruction
  returning :class:`~mqt.yaqs.characterization.process_tensors.tomography.data.SequenceData`.
- :mod:`mqt.yaqs.characterization.process_tensors.surrogates`: neural surrogate utilities and workflows.

The submodule :mod:`mqt.yaqs.characterization.process_tensors.core` contains shared internal helpers.
"""

# Public entry points only.
from .surrogates.model import TransformerComb
from .surrogates.workflow import create_surrogate, generate_data
from .tomography.constructor import construct_process_tensor

__all__ = [
    # Exact PT reconstruction (returns DenseComb or MPOComb)
    "construct_process_tensor",
    # Surrogate PT learning
    "TransformerComb",
    "create_surrogate",
    "generate_data",
]
