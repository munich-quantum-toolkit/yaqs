# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""TDVP public entry point and sweep batching."""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import integrators

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...data_structures.mpo import MPO
    from ...data_structures.mps import MPS
    from ...data_structures.simulation_parameters import AnalogSimParams, StrongSimParams, WeakSimParams


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

    Public TDVP entry points validate inputs and delegate here. Sweep kernels
    perform one symmetric substep and accept ``step_scale``; they must not
    re-enter the public wrappers.

    Each substep is symmetric (LTR then RTL) at ``step_time / tdvp_sweeps`` for
    analog ``dt`` and digital gates.

    Args:
        evolve_once: One-sweep integrator, e.g. :func:`~integrators.sweep_2site`.
        state: MPS updated in place.
        operator: Generator MPO.
        sim_params: Supplies ``tdvp_sweeps`` and evolution settings.
        *args: Extra positional arguments forwarded to ``evolve_once``.
        **kwargs: Extra keyword arguments forwarded to ``evolve_once``.

    Raises:
        ValueError: If ``sim_params.tdvp_sweeps`` is less than 1.

    """
    if sim_params.tdvp_sweeps < 1:
        msg = f"tdvp_sweeps must be >= 1, got {sim_params.tdvp_sweeps}."
        raise ValueError(msg)
    step_scale = 1.0 / sim_params.tdvp_sweeps
    sweep_plan = [step_scale] * sim_params.tdvp_sweeps
    evolve_once(
        state,
        operator,
        sim_params,
        *args,
        sweep_plan=sweep_plan,
        **kwargs,
    )


def tdvp(
    state: MPS,
    operator: MPO,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
) -> None:
    """Evolve an MPS under an MPO operator via TDVP.

    The operator may represent a Hamiltonian (analog simulation) or a gate
    generator MPO (digital simulation).

    Args:
        state: MPS state updated in place.
        operator: MPO defining the local generator at each site.
        sim_params: Simulation parameters including ``dt`` or gate time,
            ``tdvp_sweeps``, ``tdvp_mode``, truncation, and Krylov settings.

    Raises:
        ValueError: If ``state`` and ``operator`` lengths mismatch or if
            ``tdvp_mode="2site"`` with fewer than two sites (except a one-site
            chain, which falls back to 1TDVP).

    """
    if operator.length != state.length:
        msg = "MPS and operator must have the same number of sites."
        raise ValueError(msg)
    if state.orthogonality_center is not None:
        state.require_orthogonality_center(0, context="tdvp")
    tdvp_mode = sim_params.tdvp_mode
    if tdvp_mode in {"2site", "dynamic"} and operator.length == 1:
        tdvp_mode = "1site"
    elif tdvp_mode == "2site" and operator.length < 2:
        msg = "Operator is too short for a two-site update (2TDVP)."
        raise ValueError(msg)

    if tdvp_mode == "1site":
        _run_sweeps(integrators.sweep_1site, state, operator, sim_params)
    elif tdvp_mode == "2site":
        _run_sweeps(integrators.sweep_2site, state, operator, sim_params)
    elif tdvp_mode == "dynamic":
        _run_sweeps(integrators.sweep_dynamic, state, operator, sim_params)
    else:
        msg = f'tdvp_mode must be one of ("1site", "2site", "dynamic"), got {tdvp_mode!r}.'
        raise ValueError(msg)


def evolve_window(
    state: MPS,
    operator: MPO,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
) -> None:
    """Evolve a window-local MPS without post-sweep renormalization before grafting.

    Used by :func:`mqt.yaqs.digital.digital_tjm.apply_two_qubit_gate_tdvp` for
    long-range gates whose tensors are copied back into a larger chain.

    Args:
        state: Window-local MPS updated in place.
        operator: Window-local generator MPO.
        sim_params: Truncation and Krylov settings for TDVP.

    Raises:
        ValueError: If the window-local MPS has fewer than two sites.

    """
    if state.length < 2:
        msg = "evolve_window requires an MPS window with at least two sites."
        raise ValueError(msg)
    _run_sweeps(
        integrators.sweep_2site,
        state,
        operator,
        sim_params,
        drift_renorm=False,
    )
