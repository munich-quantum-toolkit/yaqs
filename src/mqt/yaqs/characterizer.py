# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Characterization entry point for YAQS.

**Public API** (``__all__``):

- :func:`generate_data` — simulate rollouts; returns a :class:`~torch.utils.data.TensorDataset`.
- :func:`create_surrogate` — end-to-end training from sampled data.
- :class:`TransformerComb` — neural module; :meth:`~TransformerComb.fit` takes training (and optional validation) :class:`~torch.utils.data.TensorDataset` instances.
- :func:`construct_process_tensor` — exhaustive process-tensor tomography returning a comb.

Surrogate and exact tomography entry points are available directly from this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mqt.yaqs.characterization.process_tensors.surrogates.model import TransformerComb
from mqt.yaqs.characterization.process_tensors.surrogates.workflow import create_surrogate, generate_data
from mqt.yaqs.characterization.process_tensors.tomography import DenseComb, MPOComb, construct_process_tensor

if TYPE_CHECKING:
    from typing import TypeAlias


Comb: TypeAlias = DenseComb | MPOComb | TransformerComb


__all__ = [
    # Surrogate tooling
    "TransformerComb",
    # Exact process-tensor tomography
    "construct_process_tensor",
    "create_surrogate",
    "generate_data",
]
