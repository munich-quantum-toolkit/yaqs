# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Neural surrogates and sequence-rollout helpers.

Public workflow API: :func:`~mqt.yaqs.characterization.process_tensors.surrogates.workflow.generate_data`
(returns :class:`~torch.utils.data.TensorDataset`) and
:func:`~mqt.yaqs.characterization.process_tensors.surrogates.workflow.create_surrogate`.
:class:`~mqt.yaqs.characterization.process_tensors.surrogates.model.TransformerComb` holds the network and
:meth:`~mqt.yaqs.characterization.process_tensors.surrogates.model.TransformerComb.fit` training loop.
Rollout samples: :mod:`mqt.yaqs.characterization.process_tensors.surrogates.data`.

**Terminology** — See :mod:`mqt.yaqs.characterization.process_tensors.tomography.data` (**sequence** vs
stochastic **trajectory** under a noise model).
"""
