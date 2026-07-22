# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Process-tensor construction (dense tomography and direct MPO).

``return_type="dense"`` runs exhaustive discrete-basis tomography
(``16**num_interventions`` sequences) and returns
:class:`~mqt.yaqs.characterization.memory.backends.tomography.process_tensors.DenseProcessTensor`.

``return_type="mpo"`` builds an MPO by direct construction (noiseless) and returns
:class:`~mqt.yaqs.characterization.memory.backends.tomography.process_tensors.MPOProcessTensor`.
"""

from .basis import TomographyBasis as TomographyBasis
from .basis import assemble_fixed_basis as assemble_fixed_basis
from .basis import get_basis_states as get_basis_states
from .basis import get_choi_basis as get_choi_basis
from .constructor import build_process_tensor as build_process_tensor
from .data import SequenceData as SequenceData
from .process_tensors import DenseProcessTensor as DenseProcessTensor
from .process_tensors import MPOProcessTensor as MPOProcessTensor
from .process_tensors import compute_temporal_entropy as compute_temporal_entropy
from .process_tensors import evaluate_probes as evaluate_probes
