# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Neural surrogates for operational-memory dynamics.

Public workflow API: :func:`~mqt.yaqs.characterization.memory.backends.surrogates.workflow.build_training_dataset`
(returns :class:`~torch.utils.data.TensorDataset`) and
:func:`~mqt.yaqs.characterization.memory.backends.surrogates.workflow.train_surrogate_model`.
:class:`~mqt.yaqs.characterization.memory.backends.surrogates.model.ProcessTensorSurrogate` holds the network and
:meth:`~mqt.yaqs.characterization.memory.backends.surrogates.model.ProcessTensorSurrogate.fit` training loop.
Sequence traces: :mod:`mqt.yaqs.characterization.memory.backends.surrogates.data`.
Comb-schedule simulation: :mod:`mqt.yaqs.characterization.memory.backends.sequences`.

**Terminology** — See :mod:`mqt.yaqs.characterization.memory.backends.tomography.data` (**sequence** vs
stochastic **trajectory** under a noise model).
"""
