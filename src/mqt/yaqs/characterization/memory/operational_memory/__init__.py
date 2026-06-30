# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Split-cut operational memory protocol and helpers.

Submodules:

- :mod:`.samples` — :class:`ProbeSet`, :func:`sample_probes`, intervention step sampling
- :mod:`.grid` — :func:`assemble_probe_sequence`, :func:`assemble_probe_grid`
- :mod:`.branch_weights` — :func:`compute_analytic_weights`, :func:`compute_trace_weights`
- :mod:`.memory_matrix` — weighted assembly and :func:`compute_spectrum`
- :mod:`.run` — :func:`run_operational_memory`, backend protocols
- :mod:`.results` — :class:`CharacterizationResult`
- :mod:`.interventions` — user-facing intervention encoding for predict/train paths
"""

from .branch_weights import (
    compute_analytic_weights,
    compute_born_prob,
    compute_branch_weight,
    compute_branch_weights,
    compute_trace_weights,
)
from .grid import assemble_probe_grid, assemble_probe_sequence
from .results import CharacterizationResult, merge_cut_results, pack_result
from .run import (
    ProcessTensorProbeBackend,
    MemoryProcessBackend,
    OperationalMemoryBackend,
    evaluate_probes_weighted_for,
    run_operational_memory,
)
from .samples import ProbeSet, extract_ket, resolve_unitary_sampler, sample_probes

__all__ = [
    "CharacterizationResult",
    "ProcessTensorProbeBackend",
    "MemoryProcessBackend",
    "OperationalMemoryBackend",
    "ProbeSet",
    "assemble_probe_grid",
    "assemble_probe_sequence",
    "compute_analytic_weights",
    "compute_born_prob",
    "compute_branch_weight",
    "compute_branch_weights",
    "compute_trace_weights",
    "evaluate_probes_weighted_for",
    "extract_ket",
    "merge_cut_results",
    "pack_result",
    "resolve_unitary_sampler",
    "run_operational_memory",
    "sample_probes",
]
