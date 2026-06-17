# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Simulation Parameters for each type of simulation allowed in YAQS.

This module provides classes for representing observables and simulation parameters
for quantum simulations. It defines the Observable class for measurement, as well as
the PhysicsSimParams, WeakSimParams, and StrongSimParams classes for configuring simulation
runs. These classes encapsulate settings such as simulation time, time steps, bond dimension limits,
thresholds, and window sizes. Simulation outputs are stored on
:class:`~mqt.yaqs.core.data_structures.result.Result`, not on these parameter objects.
"""

from __future__ import annotations

import copy
from enum import Enum
from typing import TYPE_CHECKING, Literal, TypedDict

import numpy as np

from mqt.yaqs.core.libraries.gate_library import GateLibrary

if TYPE_CHECKING:
    from mqt.yaqs.core.libraries.gate_library import BaseGate

SimulationPreset = Literal["fast", "balanced", "accurate", "exact"]
GateMode = Literal["tdvp", "full-tdvp", "swaps", "mpo"]
TDVPMode = Literal["1site", "2site", "dynamic"]


class PresetTypes(TypedDict):
    """Built-in numerical settings for a simulation preset."""

    svd_threshold: float
    max_bond_dim: int | None
    num_traj: int
    krylov_tol: float


SIMULATION_PRESETS: dict[SimulationPreset, PresetTypes] = {
    "fast": {"svd_threshold": 1e-3, "max_bond_dim": 16, "num_traj": 128, "krylov_tol": 1e-3},
    "balanced": {"svd_threshold": 1e-6, "max_bond_dim": 128, "num_traj": 256, "krylov_tol": 1e-4},
    "accurate": {"svd_threshold": 1e-9, "max_bond_dim": 4096, "num_traj": 1024, "krylov_tol": 1e-6},
    "exact": {"svd_threshold": 1e-13, "max_bond_dim": None, "num_traj": 1024, "krylov_tol": 1e-12},
}


_USE_PRESET = object()


def _validate_preset(preset: SimulationPreset) -> SimulationPreset:
    """Validate ``preset`` names at runtime.

    Args:
        preset: Built-in simulation preset name. Must be one of ``"fast"``, ``"balanced"``,
            ``"accurate"``, or ``"exact"`` (see ``SIMULATION_PRESETS``).

    Returns:
        The validated preset name.

    Raises:
        ValueError: If ``preset`` is not a supported preset name.
    """
    if preset not in SIMULATION_PRESETS:
        msg = f"preset must be one of {sorted(SIMULATION_PRESETS)!r}, got {preset!r}."
        raise ValueError(msg)
    return preset


def _validate_random_seed(random_seed: int | None) -> None:
    """Validate ``random_seed`` before storing it on simulation parameter objects.

    Args:
        random_seed: Base seed for reproducible stochastic runs, or ``None`` for unseeded RNG.

    Raises:
        TypeError: If ``random_seed`` is not ``None`` or an ``int``.
        ValueError: If ``random_seed`` is negative.
    """
    if random_seed is None:
        return
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        msg = f"random_seed must be int or None, got {type(random_seed).__name__}."
        raise TypeError(msg)
    if random_seed < 0:
        msg = f"random_seed must be non-negative, got {random_seed}."
        raise ValueError(msg)


def _validate_gate_mode(mode: GateMode) -> GateMode:
    """Validate ``gate_mode`` for digital MPS circuit simulation.

    Args:
        mode: Two-qubit gate update mode for the MPS digital backend.

    Returns:
        The validated mode name.

    Raises:
        ValueError: If ``mode`` is not a supported value.
    """
    allowed = ("tdvp", "full-tdvp", "swaps", "mpo")
    if mode not in allowed:
        msg = f"gate_mode must be one of {allowed!r}, got {mode!r}."
        raise ValueError(msg)
    return mode


def _validate_tdvp_sweeps(tdvp_sweeps: int) -> int:
    """Validate ``tdvp_sweeps`` for TDVP evolution substeps.

    Args:
        tdvp_sweeps: Number of TDVP substeps per evolution step (analog ``dt`` or circuit gate).

    Returns:
        The validated sweep count.

    Raises:
        TypeError: If ``tdvp_sweeps`` is not an ``int``.
        ValueError: If ``tdvp_sweeps`` is less than 1.
    """
    if isinstance(tdvp_sweeps, bool) or not isinstance(tdvp_sweeps, int):
        msg = f"tdvp_sweeps must be int, got {type(tdvp_sweeps).__name__}."
        raise TypeError(msg)
    if tdvp_sweeps < 1:
        msg = f"tdvp_sweeps must be >= 1, got {tdvp_sweeps}."
        raise ValueError(msg)
    return tdvp_sweeps


def _validate_tdvp_mode(tdvp_mode: TDVPMode) -> TDVPMode:
    """Validate ``tdvp_mode`` for TDVP integrator geometry.

    All simulation parameter classes default to ``"2site"``.

    Args:
        tdvp_mode: Integrator variant (``"1site"``, ``"2site"``, or ``"dynamic"``).

    Returns:
        The validated mode name.

    Raises:
        ValueError: If ``tdvp_mode`` is not a supported value.
    """
    allowed = ("1site", "2site", "dynamic")
    if tdvp_mode not in allowed:
        msg = f"tdvp_mode must be one of {allowed!r}, got {tdvp_mode!r}."
        raise ValueError(msg)
    return tdvp_mode


def _validate_krylov_tol(krylov_tol: float) -> float:
    """Validate the Krylov/Lanczos matrix exponential tolerance.

    Args:
        krylov_tol: Tolerance for adaptive Krylov/Lanczos matrix exponentials.

    Returns:
        The validated tolerance as a float.

    Raises:
        ValueError: If ``krylov_tol`` is non-finite or not strictly positive.
    """
    krylov_tol = float(krylov_tol)
    if not np.isfinite(krylov_tol) or krylov_tol <= 0.0:
        msg = f"krylov_tol must be a finite positive float, got {krylov_tol!r}."
        raise ValueError(msg)
    return krylov_tol


def _validate_svd_threshold(svd_threshold: float) -> float:
    """Validate the SVD truncation threshold.

    Args:
        svd_threshold: Tolerance for SVD-based bond truncation during simulation.

    Returns:
        The validated threshold as a float.

    Raises:
        ValueError: If ``svd_threshold`` is non-finite or not strictly positive.
    """
    svd_threshold = float(svd_threshold)
    if not np.isfinite(svd_threshold) or svd_threshold <= 0.0:
        msg = f"svd_threshold must be a finite positive float, got {svd_threshold!r}."
        raise ValueError(msg)
    return svd_threshold


def _resolve_max_bond_dim(max_bond_dim: int | object | None, preset_value: int | None) -> int | None:
    """Resolve ``max_bond_dim`` from an explicit value or the preset default.

    Args:
        max_bond_dim: Explicit cap, ``None`` for no cap, or ``_USE_PRESET`` to keep the preset value.
        preset_value: ``max_bond_dim`` from the selected preset.

    Returns:
        The resolved maximum bond dimension.

    Raises:
        TypeError: If ``max_bond_dim`` is not an ``int``, ``None``, or ``_USE_PRESET``.
    """
    if max_bond_dim is _USE_PRESET:
        return preset_value
    if isinstance(max_bond_dim, int):
        return max_bond_dim
    if max_bond_dim is None:
        return None
    msg = f"max_bond_dim must be int, None, or omitted, got {type(max_bond_dim).__name__}."
    raise TypeError(msg)


class EvolutionMode(Enum):
    """Enumerates the different modes of tensor evolution in the simulation."""

    TDVP = "tdvp"
    BUG = "bug"


class Observable:
    """Measurement metadata for a quantum simulation.

    Describes *what* to measure (gate and sites). Per-run expectation values and
    trajectories are stored on :class:`~mqt.yaqs.core.data_structures.result.Result`.

    Attributes:
        gate: The gate that acts as the observable.
        sites: The site or site indices on which this observable is measured.
    """

    def __init__(self, gate: BaseGate | str, sites: int | list[int] | None = None) -> None:
        """Initializes an Observable instance.

        Args:
            gate: The gate that will act as the observable.
            sites: The qubit or site indices on which this observable is measured.
        """
        if isinstance(gate, str):
            if gate == "entropy":
                gate = GateLibrary.entropy()
            elif gate == "schmidt_spectrum":
                gate = GateLibrary.schmidt_spectrum()
            elif gate == "pvm":
                gate = GateLibrary.pvm(gate)
            elif hasattr(GateLibrary, gate):
                attr = getattr(GateLibrary, gate)
                try:
                    gate = attr()
                except TypeError:
                    gate = GateLibrary.pvm(gate)
            else:
                gate = GateLibrary.pvm(gate)
        assert hasattr(GateLibrary, gate.name), f"Observable {gate.name} not found in GateLibrary."
        self.gate = copy.deepcopy(gate)
        if gate.name != "pvm":
            assert sites is not None
            self.sites = sites
            self.gate.set_sites(self.sites)


def _prepare_observable_ordering(observables: list[Observable]) -> tuple[list[Observable], tuple[int, ...]]:
    """Prepare a sorted evaluation order and a user-index to sorted-row mapping.

    Non-PVM observables are evaluated in ascending site order to reduce the number of
    orthogonality-center shifts in MPS-based backends. The sorting is *stable* for
    ties (observables on the same site keep their original list order). PVM
    observables are appended in their original relative order.

    Args:
        observables: Observables in the order supplied by the user.

    Returns:
        ``(sorted_observables, observable_sorted_indices)`` where
        ``observable_sorted_indices[user_i]`` is the corresponding row index in the
        sorted worker buffers for ``observables[user_i]``.
    """
    if not observables:
        return [], ()

    indexed = list(enumerate(observables))
    sortable = [(i, obs) for i, obs in indexed if obs.gate.name != "pvm"]
    pvm_pairs = [(i, obs) for i, obs in indexed if obs.gate.name == "pvm"]

    def _site_sort_key(pair: tuple[int, Observable]) -> tuple[int, int]:
        user_i, obs = pair
        sites = obs.sites
        site = sites[0] if isinstance(sites, list) else sites
        assert isinstance(site, int)
        return (site, user_i)

    sorted_pairs = sorted(sortable, key=_site_sort_key) + pvm_pairs
    sorted_observables = [obs for _user_i, obs in sorted_pairs]

    user_to_sorted: list[int] = [0] * len(observables)
    for sorted_i, (user_i, _obs) in enumerate(sorted_pairs):
        user_to_sorted[user_i] = sorted_i

    return sorted_observables, tuple(user_to_sorted)


class _ObservableOrderingMixin:
    """Computed observable ordering from the current ``observables`` list."""

    observables: list[Observable]

    @property
    def sorted_observables(self) -> list[Observable]:
        """Observables sorted by site for efficient MPS evaluation."""
        sorted_obs, _indices = _prepare_observable_ordering(self.observables)
        return sorted_obs

    @property
    def observable_sorted_indices(self) -> tuple[int, ...]:
        """Maps each user-list index to the corresponding sorted worker-buffer row."""
        _sorted_obs, indices = _prepare_observable_ordering(self.observables)
        return indices


class AnalogSimParams(_ObservableOrderingMixin):
    """Analog Simulation Parameters.

    A class to represent the parameters for an analog simulation.

    Attributes:
        observables: List of observables tracked during the simulation.
        sorted_observables: Observables sorted by site for efficient MPS evaluation (computed
            from the current :attr:`observables` list).
        observable_sorted_indices: Maps each user-list index to the corresponding row in
            sorted worker buffers (computed from the current :attr:`observables` list).
        elapsed_time: Total simulation time.
        dt: Simulation time step.
        times: Array of sampled times from ``0`` to ``elapsed_time`` with spacing ``dt``.
        sample_timesteps: If ``True``, record values at all sampled timesteps.
        num_traj: Number of trajectories (for stochastic open-system evolution).
        random_seed: If set, seeds per-trajectory jump RNG and static noise sampling for reproducible runs.
        max_bond_dim: Maximum allowed bond dimension, or ``None`` for no cap. Omit the
            constructor argument to use the preset value; pass ``None`` explicitly for no cap.
        preset: Preset controlling ``svd_threshold``, ``max_bond_dim``, ``num_traj``, and ``krylov_tol``.
            Default is ``"balanced"``. ``"fast"`` is intended for quick tests and
            examples, ``"accurate"`` for high-quality production runs, and ``"exact"`` for
            strict reference/debug settings (still subject to timestep and sampling error).
            Explicit ``svd_threshold``, ``max_bond_dim``, ``num_traj``, and ``krylov_tol`` override the preset.
        krylov_tol: Tolerance for the adaptive Krylov/Lanczos matrix exponential used in TDVP updates.
            Smaller values are more accurate but may require more Krylov vectors. Explicit values
            override the preset.
        trunc_mode: Truncation mode used in TDVP (``"discarded_weight"`` or ``"relative"``).
        svd_threshold: SVD truncation threshold for bond dimension control.
        order: Integration order.
        get_state: If ``True``, request the final state on the returned :class:`~mqt.yaqs.Result`.
        multi_time_observables: Optional list of ``(A, B)`` observable pairs for unitary-ensemble
            two-time correlators. Each entry computes ``<psi(t)|A U(t) B|psi(0)>``.
            Autocorrelation is the special case ``(O, O)``. Results are indexed by pair position.
        tdvp_sweeps: Number of TDVP substeps per time step ``dt``. Each substep is a
            symmetric integrator step (LTR then RTL) at evolution time ``dt / tdvp_sweeps``.
            Default is ``1``.
        tdvp_mode: TDVP integrator geometry (``"1site"``, ``"2site"``, or ``"dynamic"``).
            Default is ``"2site"``.
    """

    def __init__(
        self,
        observables: list[Observable] | None = None,
        elapsed_time: float = 0.1,
        dt: float = 0.1,
        num_traj: int | None = None,
        max_bond_dim: int | object | None = _USE_PRESET,
        trunc_mode: str = "discarded_weight",
        svd_threshold: float | None = None,
        krylov_tol: float | None = None,
        order: int = 1,
        *,
        preset: SimulationPreset = "balanced",
        sample_timesteps: bool = True,
        evolution_mode: EvolutionMode = EvolutionMode.TDVP,
        get_state: bool = False,
        random_seed: int | None = None,
        multi_time_observables: list[tuple[Observable, Observable]] | None = None,
        tdvp_sweeps: int = 1,
        tdvp_mode: TDVPMode = "2site",
    ) -> None:
        """Physics simulation parameters initialization.

        Initializes parameters for a physics-based quantum simulation.

        Args:
            observables: List of observables to measure during the simulation.
            elapsed_time: Total simulation time.
            dt: Time step interval.
            num_traj: Number of simulation samples.
            random_seed: If set, makes stochastic trajectories and noise-model sampling reproducible.
            max_bond_dim: Maximum bond dimension allowed, or ``None`` for no cap. Omit to use
                the preset value; pass ``None`` explicitly for no cap.
            preset: Preset controlling ``svd_threshold``, ``max_bond_dim``, ``num_traj``, and ``krylov_tol``.
                Default is ``"balanced"``. ``"fast"`` is intended for quick tests and
                examples, ``"accurate"`` for high-quality production runs, and ``"exact"`` for
                strict reference/debug settings (still subject to timestep and sampling error).
                Explicit ``svd_threshold``, ``max_bond_dim``, ``num_traj``, and ``krylov_tol`` override the preset.
            krylov_tol: Tolerance for the adaptive Krylov/Lanczos matrix exponential used in TDVP updates.
                Smaller values are more accurate but may require more Krylov vectors. Explicit values
                override the preset.
            trunc_mode: TDVP truncation mode (``"discarded_weight"`` or ``"relative"``).
            svd_threshold: SVD truncation threshold for bond dimension control.
            order: Order of approximation or numerical scheme.
            sample_timesteps: Whether to sample at intermediate time steps.
            evolution_mode: Tensor evolution mode (default ``EvolutionMode.TDVP``).
            get_state: If ``True``, request the final state on the returned :class:`~mqt.yaqs.Result`.
            multi_time_observables: For ``list[State]`` unitary ensemble runs only, list of ``(A, B)``
                pairs evaluated as ``<psi(t)|A U(t) B|psi(0)>``. Autocorrelation is the special
                case ``(O, O)``.
            tdvp_sweeps: Number of TDVP substeps per time step ``dt``. Each substep is a
                symmetric integrator step at ``dt / tdvp_sweeps`` (default ``1``).
            tdvp_mode: TDVP integrator geometry (``"1site"``, ``"2site"``, or ``"dynamic"``).
                Default is ``"2site"``.
        """
        _validate_random_seed(random_seed)
        preset_values = SIMULATION_PRESETS[_validate_preset(preset)]
        self.preset = preset
        obs_list: list[Observable] = [] if observables is None else list(observables)
        assert all(n.gate.name == "pvm" for n in obs_list) or all(n.gate.name != "pvm" for n in obs_list), (
            "We currently have not implemented mixed observable and projective-measurement simulation."
        )
        self.observables = obs_list

        self.elapsed_time = elapsed_time
        self.dt = dt
        self.times = np.arange(0, elapsed_time + dt, dt)
        self.sample_timesteps = sample_timesteps
        self.num_traj = num_traj if num_traj is not None else preset_values["num_traj"]
        self.max_bond_dim = _resolve_max_bond_dim(max_bond_dim, preset_values["max_bond_dim"])
        self.trunc_mode = trunc_mode
        self.svd_threshold = _validate_svd_threshold(
            svd_threshold if svd_threshold is not None else preset_values["svd_threshold"]
        )
        self.krylov_tol = _validate_krylov_tol(krylov_tol if krylov_tol is not None else preset_values["krylov_tol"])
        self.order = order
        self.evolution_mode = evolution_mode
        self.get_state = get_state
        self.random_seed = random_seed
        self.multi_time_observables: list[tuple[Observable, Observable]] = (
            [] if multi_time_observables is None else list(multi_time_observables)
        )
        self.tdvp_sweeps = _validate_tdvp_sweeps(tdvp_sweeps)
        self.tdvp_mode = _validate_tdvp_mode(tdvp_mode)


class StrongSimParams(_ObservableOrderingMixin):
    """Strong Circuit Simulation Parameters.

    A class to represent the parameters for a strong simulation.

    Attributes:
        dt: A placeholder property for code compatibility.
        observables: A list of observables to be tracked during the simulation.
        sorted_observables: Observables sorted by site for efficient MPS evaluation (computed
            from the current :attr:`observables` list).
        observable_sorted_indices: Maps each user-list index to the corresponding row in
            sorted worker buffers (computed from the current :attr:`observables` list).
        num_traj: The number of trajectories to simulate.
        random_seed: If set, seeds per-trajectory jump RNG and static noise
            sampling for reproducible runs.
        max_bond_dim: The maximum bond dimension for the simulation, or ``None`` for no cap. Omit
            the constructor argument to use the preset value; pass ``None`` explicitly for no cap.
        preset: Preset controlling ``svd_threshold``, ``max_bond_dim``, ``num_traj``, and ``krylov_tol``.
            Default is ``"balanced"``. ``"fast"`` is intended for quick tests and
            examples, ``"accurate"`` for high-quality production runs, and ``"exact"`` for
            strict reference/debug settings (still subject to timestep and sampling error).
            Explicit ``svd_threshold``, ``max_bond_dim``, ``num_traj``, and ``krylov_tol`` override the preset.
        krylov_tol: Tolerance for the adaptive Krylov/Lanczos matrix exponential used in TDVP updates.
            Smaller values are more accurate but may require more Krylov vectors. Explicit values
            override the preset.
        trunc_mode: The type of truncation performed in TDVP. Options are
            ``"discarded_weight"`` and ``"relative"``.
        svd_threshold: SVD truncation threshold for bond dimension control.
        window_size: The size of the window for the simulation. Default is ``None``.
        get_state: If ``True``, request the final state on the returned
            :class:`~mqt.yaqs.Result`.
        gate_mode: Two-qubit gate update mode on the MPS digital backend
            (``"swaps"``, ``"tdvp"``, ``"full-tdvp"``, or ``"mpo"``). Default is
            ``"mpo"``. Hybrid ``"tdvp"`` applies TEBD to nearest-neighbor gates and
            two-site TDVP (2TDVP) on a local window for long-range gates.
        tdvp_sweeps: Number of symmetric TDVP substeps per gate. Each substep integrates
            evolution time ``1 / tdvp_sweeps`` of the unit gate. Default is ``1``.
        tdvp_mode: TDVP integrator geometry (``"1site"``, ``"2site"``, or
            ``"dynamic"``). Default is ``"2site"`` for circuit simulation.
    """

    # Properties set as placeholders for code compatibility
    dt = 1

    def __init__(
        self,
        observables: list[Observable] | None = None,
        num_traj: int | None = None,
        max_bond_dim: int | object | None = _USE_PRESET,
        trunc_mode: str = "discarded_weight",
        svd_threshold: float | None = None,
        krylov_tol: float | None = None,
        *,
        preset: SimulationPreset = "balanced",
        get_state: bool = False,
        sample_layers: bool = False,
        num_mid_measurements: int = 0,
        random_seed: int | None = None,
        gate_mode: GateMode = "mpo",
        tdvp_sweeps: int = 1,
        tdvp_mode: TDVPMode = "2site",
    ) -> None:
        """Strong circuit simulation parameters initialization.

        Initializes parameters for a strong quantum circuit simulation.

        Args:
            observables: List of observables to measure during simulation.
            num_traj: Number of trajectories to simulate.
            max_bond_dim: Maximum bond dimension allowed in simulation, or ``None`` for no cap. Omit
                to use the preset value; pass ``None`` explicitly for no cap.
            preset: Preset controlling ``svd_threshold``, ``max_bond_dim``, ``num_traj``, and ``krylov_tol``.
                Default is ``"balanced"``. ``"fast"`` is intended for quick tests and
                examples, ``"accurate"`` for high-quality production runs, and ``"exact"`` for
                strict reference/debug settings (still subject to timestep and sampling error).
                Explicit ``svd_threshold``, ``max_bond_dim``, ``num_traj``, and ``krylov_tol`` override the preset.
            krylov_tol: Tolerance for the adaptive Krylov/Lanczos matrix exponential used in TDVP updates.
                Smaller values are more accurate but may require more Krylov vectors. Explicit values
                override the preset.
            trunc_mode: TDVP truncation mode (``"discarded_weight"`` or ``"relative"``).
            svd_threshold: SVD truncation threshold for bond dimension control.
            get_state: If ``True``, request the final state on the returned :class:`~mqt.yaqs.Result`.
            sample_layers: If ``True``, record observables at sampled circuit layers.
            num_mid_measurements: Number of mid-circuit measurement barriers when sampling layers.
            random_seed: If set, makes stochastic trajectories and noise-model sampling reproducible.
            gate_mode: Two-qubit gate update mode (default ``"mpo"``). Hybrid ``"tdvp"`` uses
                TEBD for nearest-neighbor gates and two-site TDVP (2TDVP) on a local window
                for long-range gates.
            tdvp_sweeps: Number of symmetric TDVP substeps per gate at evolution time
                ``1 / tdvp_sweeps`` (default ``1``).
            tdvp_mode: TDVP integrator geometry (``"1site"``, ``"2site"``, or ``"dynamic"``).
                Default is ``"2site"``. Selects the integrator for :func:`~mqt.yaqs.core.methods.tdvp.tdvp.tdvp`
                (for example ``gate_mode="full-tdvp"``); hybrid long-range gates always use
                2TDVP via :func:`~mqt.yaqs.core.methods.tdvp.tdvp.evolve_window`.
        """
        _validate_random_seed(random_seed)
        preset_values = SIMULATION_PRESETS[_validate_preset(preset)]
        self.preset = preset
        obs_list: list[Observable] = [] if observables is None else list(observables)
        assert all(n.gate.name == "pvm" for n in obs_list) or all(n.gate.name != "pvm" for n in obs_list), (
            "We currently have not implemented mixed observable and projective-measurement simulation."
        )
        self.observables = obs_list

        self.num_traj = num_traj if num_traj is not None else preset_values["num_traj"]
        self.max_bond_dim = _resolve_max_bond_dim(max_bond_dim, preset_values["max_bond_dim"])
        self.trunc_mode = trunc_mode
        self.svd_threshold = _validate_svd_threshold(
            svd_threshold if svd_threshold is not None else preset_values["svd_threshold"]
        )
        self.krylov_tol = _validate_krylov_tol(krylov_tol if krylov_tol is not None else preset_values["krylov_tol"])
        self.get_state = get_state
        self.sample_layers = sample_layers
        self.num_mid_measurements = num_mid_measurements
        self.random_seed = random_seed
        self.gate_mode = _validate_gate_mode(gate_mode)
        self.tdvp_sweeps = _validate_tdvp_sweeps(tdvp_sweeps)
        self.tdvp_mode = _validate_tdvp_mode(tdvp_mode)


class WeakSimParams:
    """A class to represent the parameters for a weak simulation.

    Attributes:
        dt: A placeholder property for code compatibility.
        num_traj: A placeholder property for code compatibility.
        shots: The number of shots for the simulation.
        max_bond_dim: The maximum bond dimension for the simulation, or ``None`` for no cap. Omit
            the constructor argument to use the preset value; pass ``None`` explicitly for no cap.
        preset: Preset controlling ``svd_threshold``, ``max_bond_dim``, and ``krylov_tol``.
            Default is ``"balanced"``. ``"fast"`` is intended for quick tests and
            examples, ``"accurate"`` for high-quality production runs, and ``"exact"`` for
            strict reference/debug settings.
            Explicit ``svd_threshold``, ``max_bond_dim``, and ``krylov_tol`` override the preset.
        krylov_tol: Tolerance for the adaptive Krylov/Lanczos matrix exponential used in TDVP updates.
            Smaller values are more accurate but may require more Krylov vectors. Explicit values
            override the preset.
        trunc_mode: The type of truncation performed in TDVP. Options are
            ``"discarded_weight"`` and ``"relative"``.
        svd_threshold: SVD truncation threshold for bond dimension control.
        window_size: The window size for the simulation.
        get_state: If ``True``, request the final state on the returned
            :class:`~mqt.yaqs.Result`.
        gate_mode: Two-qubit gate update mode on the MPS digital backend
            (``"swaps"``, ``"tdvp"``, ``"full-tdvp"``, or ``"mpo"``).
        tdvp_sweeps: Number of TDVP substeps per evolution step. Each substep is a
            symmetric integrator step (LTR then RTL) at evolution time ``1 / tdvp_sweeps``
            of the unit gate time. Default is ``1``.
        tdvp_mode: TDVP integrator geometry (``"1site"``, ``"2site"``, or ``"dynamic"``).
            Default is ``"2site"`` for circuit simulation.
    """

    # Properties set as placeholders for code compatibility
    dt = 1
    num_traj = 0

    def __init__(
        self,
        shots: int,
        max_bond_dim: int | object | None = _USE_PRESET,
        trunc_mode: str = "discarded_weight",
        svd_threshold: float | None = None,
        krylov_tol: float | None = None,
        *,
        preset: SimulationPreset = "balanced",
        get_state: bool = False,
        random_seed: int | None = None,
        gate_mode: GateMode = "mpo",
        tdvp_sweeps: int = 1,
        tdvp_mode: TDVPMode = "2site",
    ) -> None:
        """Weak circuit simulation initialization.

        Initializes parameters for a weak circuit simulation.

        Args:
            shots: Number of measurement shots to simulate.
            max_bond_dim: Maximum bond dimension for simulation, or ``None`` for no cap. Omit to
                use the preset value; pass ``None`` explicitly for no cap.
            preset: Preset controlling ``svd_threshold``, ``max_bond_dim``, and ``krylov_tol``.
                Default is ``"balanced"``. ``"fast"`` is intended for quick tests and
                examples, ``"accurate"`` for high-quality production runs, and ``"exact"`` for
                strict reference/debug settings.
                Explicit ``svd_threshold``, ``max_bond_dim``, and ``krylov_tol`` override the preset.
            krylov_tol: Tolerance for the adaptive Krylov/Lanczos matrix exponential used in TDVP updates.
                Smaller values are more accurate but may require more Krylov vectors. Explicit values
                override the preset.
            trunc_mode: TDVP truncation mode (``"discarded_weight"`` or ``"relative"``).
            svd_threshold: SVD truncation threshold for bond dimension control.
            get_state: If ``True``, request the final state on the returned :class:`~mqt.yaqs.Result`.
            random_seed: If set, makes per-shot jump RNG reproducible.
            gate_mode: Two-qubit gate update mode (default ``"mpo"``).
            tdvp_sweeps: Number of TDVP substeps per evolution step. Each substep is a
                symmetric integrator step at ``1 / tdvp_sweeps`` of the unit gate time
                (default ``1``).
            tdvp_mode: TDVP integrator geometry (``"1site"``, ``"2site"``, or ``"dynamic"``).
                Default is ``"2site"``.
        """
        _validate_random_seed(random_seed)
        preset_values = SIMULATION_PRESETS[_validate_preset(preset)]
        self.preset = preset
        self.shots = shots
        self.max_bond_dim = _resolve_max_bond_dim(max_bond_dim, preset_values["max_bond_dim"])
        self.trunc_mode = trunc_mode
        self.svd_threshold = _validate_svd_threshold(
            svd_threshold if svd_threshold is not None else preset_values["svd_threshold"]
        )
        self.krylov_tol = _validate_krylov_tol(krylov_tol if krylov_tol is not None else preset_values["krylov_tol"])
        self.get_state = get_state
        self.random_seed = random_seed
        self.gate_mode = _validate_gate_mode(gate_mode)
        self.tdvp_sweeps = _validate_tdvp_sweeps(tdvp_sweeps)
        self.tdvp_mode = _validate_tdvp_mode(tdvp_mode)
