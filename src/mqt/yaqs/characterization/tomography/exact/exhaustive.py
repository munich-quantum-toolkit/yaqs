# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Exact (exhaustive) process tomography: evaluate all ``16^k`` discrete basis sequences."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from mqt.yaqs.core.data_structures.networks import MPO

from ..estimate.estimate import run_estimate
from ..estimate.basis import TomographyBasis

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
    from .combs import DenseComb, MPOComb


def run_exhaustive(
    operator: MPO,
    sim_params: "AnalogSimParams",
    timesteps: list[float] | None = None,
    *,
    output: Literal["dense", "mpo"] = "dense",
    noise_model: "NoiseModel | None" = None,
    parallel: bool = True,
    num_trajectories: int = 100,
    compress_every: int = 100,
    tol: float = 1e-12,
    max_bond_dim: int | None = None,
    n_sweeps: int = 2,
    basis: TomographyBasis = "tetrahedral",
    basis_seed: int | None = None,
) -> "DenseComb | MPOComb":
    """Exhaustive tomography comb estimation (deterministic over all basis sequences)."""
    return run_estimate(
        operator,
        sim_params,
        timesteps=timesteps,
        mode="exhaustive",
        output=output,
        noise_model=noise_model,
        parallel=parallel,
        num_samples=None,
        seed=0,
        num_trajectories=num_trajectories,
        compress_every=compress_every,
        tol=tol,
        max_bond_dim=max_bond_dim,
        n_sweeps=n_sweeps,
        basis=basis,
        basis_seed=basis_seed,
    )

