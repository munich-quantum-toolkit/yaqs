# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Characterization entry point for YAQS.

**Public API** (``__all__``):

- :func:`generate_data` — simulate rollouts; returns a :class:`~torch.utils.data.TensorDataset`.
- :func:`create_surrogate` — end-to-end training from sampled data.
- :class:`TransformerComb` — neural module; :meth:`~TransformerComb.fit` takes training (and optional validation) :class:`~torch.utils.data.TensorDataset` instances.

Process-tensor tomography (:func:`construct`, :class:`SequenceData`, comb types, etc.) and surrogate
training tools are available directly from this module.
"""

from __future__ import annotations

from mqt.yaqs.characterization.process_tensors.tomography import (
    DenseComb,
    MPOComb,
    SequenceData,
    construct,
)
from mqt.yaqs.characterization.process_tensors.surrogates.model import TransformerComb
from mqt.yaqs.characterization.process_tensors.surrogates.workflow import create_surrogate, generate_data

__all__ = [
    # Exact process-tensor tomography
    "construct",
    "SequenceData",
    "DenseComb",
    "MPOComb",
    # Surrogate tooling
    "TransformerComb",
    "generate_data",
    "create_surrogate",
]
