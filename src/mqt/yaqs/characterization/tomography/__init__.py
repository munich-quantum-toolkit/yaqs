# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Internal implementation package for YAQS tomography.

Users should prefer the top-level public façade :mod:`mqt.yaqs.tomography`.
"""

from .process_tensor import (
    DenseComb,
    MPOComb,
    SequenceData,
    construct,
)
from .surrogate.model import TransformerComb
from .surrogate.workflow import create_surrogate, generate_data

from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def run(
    *,
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float],
    num_trajectories: int = 100,
    noise_model: NoiseModel | None = None,
    parallel: bool = False,
    basis: "TomographyBasis" = "tetrahedral",
    basis_seed: int | None = None,
) -> MPOComb | DenseComb:
    """Legacy façade returning a comb directly.

    Returns an :class:`~mqt.yaqs.characterization.tomography.process_tensor.combs.MPOComb`
    from exhaustive discrete process tomography.
    """
    # Import here to keep this module's import graph simple.
    from .process_tensor import TomographyBasis

    if basis not in ("standard", "tetrahedral", "random"):
        raise ValueError(f"Unknown tomography basis {basis!r}.")

    data = construct(
        operator,
        sim_params,
        timesteps=timesteps,
        parallel=parallel,
        num_trajectories=num_trajectories,
        noise_model=noise_model,
        basis=basis,
        basis_seed=basis_seed,
    )
    # Dense reconstruction keeps numerical structure (needed for PSD/CI tests,
    # especially for edge cases like `timesteps=[0.0, 0.0]`).
    return data.to_dense_comb()

__all__ = [
    "DenseComb",
    "MPOComb",
    "SequenceData",
    "TransformerComb",
    "construct",
    "run",
    "create_surrogate",
    "generate_data",
]
