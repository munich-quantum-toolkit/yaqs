# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Tomography entry point for YAQS.

**Public API** (``__all__``):

- :func:`generate_data` — simulate rollouts; returns a :class:`~torch.utils.data.TensorDataset`.
- :func:`create_surrogate` — end-to-end training from sampled data.
- :class:`TransformerComb` — neural module; :meth:`~TransformerComb.fit` takes training (and optional validation) :class:`~torch.utils.data.TensorDataset` instances.

Process tomography (``run_exhaustive``, ``run_estimate``, comb types, etc.) remains importable from
this module for backward compatibility but is not part of ``__all__``; prefer explicit imports from
:mod:`mqt.yaqs.characterization.tomography` for those entry points in new code.
"""

from __future__ import annotations

from mqt.yaqs.characterization.tomography.estimate.estimator import TomographyEstimate
from mqt.yaqs.characterization.tomography.estimate.estimate import run_estimate
from mqt.yaqs.characterization.tomography.exact.combs import DenseComb, MPOComb
from mqt.yaqs.characterization.tomography.exact.exhaustive import run_exhaustive
from mqt.yaqs.characterization.tomography.surrogate.model import TransformerComb
from mqt.yaqs.characterization.tomography.surrogate.workflow import create_surrogate, generate_data

__all__ = ["create_surrogate", "generate_data", "TransformerComb"]
