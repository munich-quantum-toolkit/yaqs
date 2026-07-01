# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared helpers for process-tensor tomography and surrogate backends.

Submodules (import explicitly; symbols are not re-exported here):

- :mod:`.encoding` — ``encode_rho_pauli``, ``pack_rho8``, ``extract_ket``, …
- :mod:`.interventions` — ``InterventionMap``, ``encode_interventions``, sampling helpers
- :mod:`.intervention_steps` — probe-step parsing and backend application
- :mod:`.metrics` — ``compute_trace_distance``, ``mean_frobenius_mse_rho8``, …
- :mod:`.utils` — site-0 MCWF/TJM evolution, ``make_mcwf_static_context``, …

Named ``shared`` (not ``core``) to avoid clashing with :mod:`mqt.yaqs.core`.
"""
