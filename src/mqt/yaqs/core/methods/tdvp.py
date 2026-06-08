# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""TDVP public entry point and sweep batching.

Low-level kernels live in :mod:`mqt.yaqs.core.methods.tdvp_primitives`.
Sweep integrators live in :mod:`mqt.yaqs.core.methods.tdvp_integrators`.
Sweep helpers live in :mod:`mqt.yaqs.core.methods.tdvp_sweep_utils`.
Long-range digital gate bond support lives in
:mod:`mqt.yaqs.core.methods.tdvp_bond_support`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ..data_structures.simulation_parameters import AnalogSimParams
from . import tdvp_integrators

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..data_structures.mpo import MPO
    from ..data_structures.mps import MPS
    from ..data_structures.simulation_parameters import StrongSimParams, WeakSimParams


Mode = Literal["1site", "2site", "dynamic"]


def _build_tdvp_sweep_plan(
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
) -> list[float]:
    """Build the sweep plan for one evolution step (gate or analog ``dt``).

    Each substep is symmetric (LTR then RTL) at ``step_time / tdvp_sweeps``.

    Returns:
        Ordered step-scale factors for one batched kernel call.
    """
    step_scale = 1.0 / sim_params.tdvp_sweeps
    return [step_scale for _ in range(sim_params.tdvp_sweeps)]


def _run_sweeps(
    evolve_once: Callable[..., None],
    state: MPS,
    operator: MPO,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    /,
    *args: object,
    **kwargs: object,
) -> None:
    """Run ``sim_params.tdvp_sweeps`` TDVP substeps per evolution step.

    Public TDVP entry points validate inputs and delegate here. Private ``_*_sweep``
    kernels perform one symmetric substep and accept ``step_scale``; they must not
    re-enter the public wrappers.

    Substep geometry (``_build_tdvp_sweep_plan``): ``tdvp_sweeps`` symmetric substeps
    (LTR then RTL each) at ``step_time / tdvp_sweeps`` for analog ``dt`` and digital gates.
    """
    evolve_once(
        state,
        operator,
        sim_params,
        *args,
        sweep_plan=_build_tdvp_sweep_plan(sim_params),
        **kwargs,
    )


def tdvp(
    state: MPS,
    operator: MPO,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    mode: Mode = "dynamic",
    *,
    support_bonds: frozenset[int] | None = None,
) -> None:
    """Evolve an MPS under an MPO operator via TDVP.

    The operator may represent a Hamiltonian (analog simulation) or a gate
    generator MPO (digital simulation).

    Args:
        state: MPS state updated in place.
        operator: MPO defining the local generator at each site.
        sim_params: Simulation parameters including ``dt`` or gate time,
            ``tdvp_sweeps``, truncation, and Krylov settings.
        mode: TDVP integrator variant. ``"dynamic"`` (default) adaptively
            chooses single- or two-site updates per bond; ``"1site"`` and
            ``"2site"`` force 1TDVP or 2TDVP respectively.
        support_bonds: Optional window-local bond indices for long-range digital
            gate support during dynamic TDVP. Requires ``StrongSimParams`` or
            ``WeakSimParams``; analog callers should omit this.

    Raises:
        ValueError: If ``state`` and ``operator`` lengths mismatch, if
            ``mode="2site"`` with fewer than two sites, if ``support_bonds``
            is set with a non-dynamic mode, or if ``support_bonds`` is set with
            ``AnalogSimParams``.
    """
    if operator.length != state.length:
        msg = "MPS and operator must have the same number of sites."
        raise ValueError(msg)
    if mode == "2site" and operator.length < 2:
        msg = "Operator is too short for a two-site update (2TDVP)."
        raise ValueError(msg)
    if support_bonds is not None and mode != "dynamic":
        msg = "support_bonds is only supported with mode='dynamic'."
        raise ValueError(msg)
    if support_bonds is not None and isinstance(sim_params, AnalogSimParams):
        msg = "support_bonds is not supported with AnalogSimParams."
        raise ValueError(msg)

    if mode == "dynamic" and operator.length == 1:
        mode = "1site"

    if mode == "1site":
        _run_sweeps(tdvp_integrators._single_site_tdvp_sweep, state, operator, sim_params)
    elif mode == "2site":
        _run_sweeps(tdvp_integrators._two_site_tdvp_sweep, state, operator, sim_params)
    elif support_bonds:
        _run_sweeps(tdvp_integrators._dynamic_tdvp_sweep, state, operator, sim_params, support_bonds)
    else:
        _run_sweeps(tdvp_integrators._dynamic_tdvp_sweep, state, operator, sim_params)
