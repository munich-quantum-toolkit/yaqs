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
from .diagnostics.probe import probe_process

__all__ = [
    # Surrogate PT learning
    "TransformerComb",
    # Exact PT reconstruction (returns DenseComb or MPOComb)
    "construct_process_tensor",
    "create_surrogate",
    "generate_data",
    "probe_process",
]
