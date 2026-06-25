# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Neural surrogates and sequence-rollout helpers.

Public workflow API: :func:`~mqt.yaqs.characterization.memory.combs.surrogates.workflow.generate_data`
(returns :class:`~torch.utils.data.TensorDataset`) and
:func:`~mqt.yaqs.characterization.memory.combs.surrogates.workflow.create_surrogate`.
:class:`~mqt.yaqs.characterization.memory.combs.surrogates.model.TransformerComb` holds the network and
:meth:`~mqt.yaqs.characterization.memory.combs.surrogates.model.TransformerComb.fit` training loop.
Rollout samples: :mod:`mqt.yaqs.characterization.memory.combs.surrogates.data`.

**Terminology** — See :mod:`mqt.yaqs.characterization.memory.combs.tomography.data` (**sequence** vs
stochastic **trajectory** under a noise model).
"""
