# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Neural surrogates and comb-sequence simulation helpers.

Public workflow API: :func:`~mqt.yaqs.characterization.memory.backends.surrogates.workflow.sample_train_dataset`
(returns :class:`~torch.utils.data.TensorDataset`) and
:func:`~mqt.yaqs.characterization.memory.backends.surrogates.workflow.train_surrogate_model`.
:class:`~mqt.yaqs.characterization.memory.backends.surrogates.model.TransformerComb` holds the network and
:meth:`~mqt.yaqs.characterization.memory.backends.surrogates.model.TransformerComb.fit` training loop.
Sequence traces: :mod:`mqt.yaqs.characterization.memory.backends.surrogates.data`.

**Terminology** — See :mod:`mqt.yaqs.characterization.memory.backends.tomography.data` (**sequence** vs
stochastic **trajectory** under a noise model).
"""
