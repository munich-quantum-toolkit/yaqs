# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Tomography entry point for YAQS.

**Public API** (``__all__``):

- :func:`generate_data` — simulate rollouts; returns a :class:`~torch.utils.data.TensorDataset`.
- :func:`create_surrogate` — end-to-end training from sampled data.
- :class:`TransformerComb` — neural module; :meth:`~TransformerComb.fit` takes training (and optional validation) :class:`~torch.utils.data.TensorDataset` instances.

Process tomography (:func:`construct`, :class:`SequenceData`, comb types, etc.) is importable from this module but is not part of ``__all__``; prefer :mod:`mqt.yaqs.characterization.tomography.process_tensor` for explicit imports.
"""

from __future__ import annotations

from mqt.yaqs.characterization.tomography.process_tensor import (
    DenseComb,
    MPOComb,
    SequenceData,
    construct,
)
from mqt.yaqs.characterization.tomography.surrogate.model import TransformerComb
from mqt.yaqs.characterization.tomography.surrogate.workflow import create_surrogate, generate_data

__all__ = ["create_surrogate", "generate_data", "TransformerComb"]
