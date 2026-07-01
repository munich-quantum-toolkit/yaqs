# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Process-tensor schedule sequence simulation (parallel engine for exact and surrogate paths).

:func:`~mqt.yaqs.characterization.memory.backends.sequences.workflow.simulate_sequences`
dispatches intervention sequences via :func:`~mqt.yaqs.core.parallel_utils.run_indexed_jobs`.
Pool workers live in :mod:`.workers`.
"""

from .workflow import simulate_sequences

__all__ = ["simulate_sequences"]
