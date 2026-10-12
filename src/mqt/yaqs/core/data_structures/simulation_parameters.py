# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Simulation Parameters for each type of simulation allowed in YAQS.

This module provides :class:`AnalogSimParams` and :class:`DigitalSimParams` for
configuring simulation runs. These classes encapsulate settings such as simulation
time, time steps, bond dimension limits, and thresholds. Simulation outputs are stored on
:class:`~mqt.yaqs.core.data_structures.result.Result`, not on these parameter objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypedDict, cast

import numpy as np

from mqt.yaqs.core.linalg.svd_utils import TruncMode  # ruff: ignore[typing-only-first-party-import]

from .._validation import validate_bool, validate_finite_real, validate_integer
from .observable import Observable

SimulationPreset = Literal["fast", "balanced", "accurate", "exact"]
GateMode = Literal["tdvp", "full-tdvp", "swaps", "mpo"]
TDVPMode = Literal["1site", "2site", "dynamic"]


class _PresetTypes(TypedDict):
    """Built-in numerical settings for a simulation preset."""

    svd_threshold: float
    max_bond_dim: int | None
    num_traj: int
    krylov_tol: float


_SIMULATION_PRESETS: dict[SimulationPreset, _PresetTypes] = {
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
            ``"accurate"``, or ``"exact"``.

    Returns:
        The validated preset name.

    Raises:
        ValueError: If ``preset`` is not a supported preset name.
    """
    if preset not in _SIMULATION_PRESETS:
        msg = f"preset must be one of {sorted(_SIMULATION_PRESETS)!r}, got {preset!r}."
        raise ValueError(msg)
    return preset


def _validate_random_seed(random_seed: int | None) -> int | None:
    """Validate and normalize a simulation random seed.

    Args:
        random_seed: Base seed for reproducible stochastic runs, or ``None`` for unseeded RNG.

    Returns:
        The seed normalized to a Python integer, or ``None``.
    """
    if random_seed is None:
        return None
    return validate_integer(random_seed, name="random_seed", minimum=0)


def _validate_analog_time_grid(elapsed_time: float, dt: float) -> int:
    """Validate analog time parameters and return the integer number of steps.

    Backends evolve every interval with the full ``dt``, so ``elapsed_time`` must be an
    integer multiple of ``dt`` within a scale-aware tolerance. Relabeling the final
    timestamp without a fractional step would otherwise disagree with the evolution.

    Args:
        elapsed_time: Total simulation time (must be finite and ``>= 0``).
        dt: Fixed time step (must be finite and ``> 0``).

    Returns:
        Non-negative integer step count ``n`` such that ``times = dt * arange(n + 1)``.

    Raises:
        TypeError: If ``elapsed_time`` or ``dt`` is a bool or non-numeric.
        ValueError: If values are non-finite, ``dt <= 0``, ``elapsed_time < 0``, or
            ``elapsed_time`` is not an integer multiple of ``dt``.
    """
    if isinstance(elapsed_time, bool) or not isinstance(elapsed_time, (int, float, np.floating, np.integer)):
        msg = f"elapsed_time must be a real number, got {type(elapsed_time).__name__}."
        raise TypeError(msg)
    if isinstance(dt, bool) or not isinstance(dt, (int, float, np.floating, np.integer)):
        msg = f"dt must be a real number, got {type(dt).__name__}."
        raise TypeError(msg)

    try:
        elapsed_f = float(elapsed_time)
        dt_f = float(dt)
    except OverflowError as exc:
        msg = "elapsed_time and dt must be representable as finite floats."
        raise ValueError(msg) from exc

    if not np.isfinite(elapsed_f):
        msg = f"elapsed_time must be finite, got {elapsed_time!r}."
        raise ValueError(msg)
    if not np.isfinite(dt_f):
        msg = f"dt must be finite, got {dt!r}."
        raise ValueError(msg)
    if dt_f <= 0.0:
        msg = f"dt must be positive, got {dt_f}."
        raise ValueError(msg)
    if elapsed_f < 0.0:
        msg = f"elapsed_time must be non-negative, got {elapsed_f}."
        raise ValueError(msg)
    if not elapsed_f > 0.0:
        return 0

    n_float = elapsed_f / dt_f
    if not np.isfinite(n_float):
        msg = f"elapsed_time / dt must be finite, got {n_float}."
        raise ValueError(msg)

    n_steps = round(n_float)
    # Bound by a float64 times-grid allocation: nbytes must fit in a platform size index.
    max_steps = np.iinfo(np.intp).max // np.dtype(np.float64).itemsize - 1
    if n_steps > max_steps:
        msg = f"elapsed_time / dt yields too many time steps ({n_steps})."
        raise ValueError(msg)

    evolved_time = n_steps * dt_f
    residual = abs(elapsed_f - evolved_time)
    # Account only for accumulated float64 roundoff and keep the tolerance well
    # below half a step so a genuinely fractional duration cannot be relabeled.
    roundoff_tol = max(
        np.spacing(elapsed_f),
        abs(n_steps) * np.spacing(dt_f),
        8 * np.finfo(np.float64).eps * max(elapsed_f, evolved_time, dt_f),
    )
    tol = min(roundoff_tol, 0.25 * dt_f)
    if n_steps <= 0 or residual > tol:
        msg = (
            f"elapsed_time ({elapsed_f}) must be an integer multiple of dt ({dt_f}); "
            f"got elapsed_time/dt = {n_float} (nearest integer {n_steps}, time residual {residual})."
        )
        raise ValueError(msg)
    return n_steps


def _build_analog_time_grid(elapsed_time: float, dt: float) -> tuple[float, float, np.ndarray]:
    """Validate analog time controls and construct their complete fixed-step grid.

    Args:
        elapsed_time: Total evolution time.
        dt: Fixed evolution step.

    Returns:
        Normalized elapsed time, normalized step size, and the corresponding time grid.
    """
    n_steps = _validate_analog_time_grid(elapsed_time, dt)
    elapsed = float(elapsed_time)
    step = float(dt)
    times = step * np.arange(n_steps + 1, dtype=np.float64)
    if n_steps > 0:
        times[-1] = elapsed
    return elapsed, step, times


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

    """
    return validate_integer(tdvp_sweeps, name="tdvp_sweeps", minimum=1)


def _validate_num_traj(num_traj: int) -> int:
    """Validate and normalize a trajectory count.

    Returns:
        Positive Python integer trajectory count.
    """
    return validate_integer(num_traj, name="num_traj", minimum=1)


def _validate_num_mid_measurements(num_mid_measurements: int) -> int:
    """Validate and normalize a mid-circuit measurement count.

    Returns:
        Non-negative Python integer measurement count.
    """
    return validate_integer(num_mid_measurements, name="num_mid_measurements", minimum=0)


def _validate_shots(shots: int | None) -> int | None:
    """Validate and normalize an optional measurement-shot count.

    Returns:
        Positive Python integer shot count, or ``None``.
    """
    if shots is None:
        return None
    return validate_integer(shots, name="shots", minimum=1)


def _validate_order(order: int) -> int:
    """Validate and normalize the analog integration order.

    Returns:
        Implemented integration order, one or two.

    Raises:
        ValueError: If the integer is not one or two.
    """
    normalized = validate_integer(order, name="order")
    if normalized not in {1, 2}:
        msg = f"order must be 1 or 2, got {normalized}."
        raise ValueError(msg)
    return normalized


def _validate_max_bond_dim(max_bond_dim: int | None) -> int | None:
    """Validate and normalize an optional maximum bond dimension.

    Returns:
        Positive Python integer bond cap, or ``None``.
    """
    if max_bond_dim is None:
        return None
    return validate_integer(max_bond_dim, name="max_bond_dim", minimum=1)


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
    normalized = validate_finite_real(krylov_tol, name="krylov_tol")
    if normalized <= 0.0:
        msg = f"krylov_tol must be positive, got {normalized!r}."
        raise ValueError(msg)
    return normalized


def _validate_svd_threshold(svd_threshold: float) -> float:
    """Validate the SVD truncation threshold.

    Args:
        svd_threshold: Tolerance for SVD-based bond truncation during simulation.
            Zero is allowed: it disables tolerance-based truncation for discarded-
            weight modes and removes only exact zeros for ``hard_cutoff``, while a
            hard bond cap may still reduce rank.

    Returns:
        The validated threshold as a float.

    Raises:
        ValueError: If ``svd_threshold`` is non-finite or negative.
    """
    normalized = validate_finite_real(svd_threshold, name="svd_threshold")
    if normalized < 0.0:
        msg = f"svd_threshold must be non-negative, got {normalized!r}."
        raise ValueError(msg)
    return normalized


def _resolve_max_bond_dim(max_bond_dim: int | object | None, preset_value: int | None) -> int | None:
    """Resolve ``max_bond_dim`` from an explicit value or the preset default.

    Args:
        max_bond_dim: Explicit cap, ``None`` for no cap, or ``_USE_PRESET`` to keep the preset value.
        preset_value: ``max_bond_dim`` from the selected preset.

    Returns:
        The resolved maximum bond dimension.

    """
    if max_bond_dim is _USE_PRESET:
        return _validate_max_bond_dim(preset_value)
    if max_bond_dim is None:
        return None
    return validate_integer(max_bond_dim, name="max_bond_dim", minimum=1)


_ALLOWED_TRUNC_MODES = frozenset({
    "discarded_weight",
    "relative",
    "hard_cutoff",
    "relative_discarded_weight",
})


def _validate_trunc_mode(trunc_mode: str) -> TruncMode:
    """Validate the SVD truncation mode name.

    Args:
        trunc_mode: Truncation mode string.

    Returns:
        The validated truncation mode.

    Raises:
        ValueError: If ``trunc_mode`` is not a string or not a supported value.
    """
    if not isinstance(trunc_mode, str):
        # Public contract documents ValueError for any unsupported trunc_mode value.
        msg = f"trunc_mode must be one of {sorted(_ALLOWED_TRUNC_MODES)!r}, got {trunc_mode!r}."
        raise ValueError(msg)  # ruff: ignore[type-check-without-type-error]
    if trunc_mode not in _ALLOWED_TRUNC_MODES:
        msg = f"trunc_mode must be one of {sorted(_ALLOWED_TRUNC_MODES)!r}, got {trunc_mode!r}."
        raise ValueError(msg)
    return cast("TruncMode", trunc_mode)


def _validate_evolution_mode(evolution_mode: str) -> Literal["tdvp", "bug"]:
    """Validate the analog evolution mode.

    Args:
        evolution_mode: ``"tdvp"`` or ``"bug"``.

    Returns:
        The validated mode string.

    Raises:
        ValueError: If ``evolution_mode`` is not a supported value.
    """
    allowed = ("tdvp", "bug")
    if isinstance(evolution_mode, str) and evolution_mode in allowed:
        return evolution_mode
    msg = f"evolution_mode must be one of {allowed!r}, got {evolution_mode!r}."
    raise ValueError(msg)


def _validate_observable_mix(observables: list[Observable]) -> None:
    """Reject mixed projective-measurement and ordinary observables.

    Args:
        observables: Observables supplied for one simulation.

    Raises:
        TypeError: If an entry is not an :class:`Observable`.
        ValueError: If the list contains both projective-measurement and ordinary observables.
    """
    for index, observable in enumerate(observables):
        if not isinstance(observable, Observable):
            msg = f"observables[{index}] must be an Observable, got {type(observable).__name__}."
            raise TypeError(msg)
    has_pvm = any(observable.name == "pvm" for observable in observables)
    has_ordinary = any(observable.name != "pvm" for observable in observables)
    if has_pvm and has_ordinary:
        msg = "Mixed observable and projective-measurement simulation is not supported."
        raise ValueError(msg)


def _validate_multi_time_observables(value: object) -> list[tuple[Observable, Observable]]:
    """Validate and normalize two-time observable pairs.

    Args:
        value: ``None`` or a sequence of two-element observable sequences.

    Returns:
        Observable pairs normalized to a list of tuples.

    Raises:
        TypeError: If the outer value is not a sequence, an entry is not a pair,
            or either pair element is not an :class:`Observable`.
        ValueError: If a pair contains an unsupported observable type.
    """
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = "multi_time_observables must be a sequence of Observable pairs."
        raise TypeError(msg)

    pairs: list[tuple[Observable, Observable]] = []
    for index, pair in enumerate(value):
        if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
            msg = f"multi_time_observables[{index}] must be a pair of Observable objects."
            raise TypeError(msg)
        first, second = pair
        if not isinstance(first, Observable) or not isinstance(second, Observable):
            msg = f"multi_time_observables[{index}] must be a pair of Observable objects."
            raise TypeError(msg)
        if any(observable.type != "operator" or observable.interaction not in {1, 2} for observable in pair):
            msg = f"multi_time_observables[{index}] must contain one- or two-site operator observables."
            raise ValueError(msg)
        pairs.append((first, second))
    return pairs


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
    sortable = [(i, obs) for i, obs in indexed if obs.name != "pvm"]
    pvm_pairs = [(i, obs) for i, obs in indexed if obs.name == "pvm"]

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
    def _sorted_observables(self) -> list[Observable]:
        """Observables sorted by site for efficient MPS evaluation."""
        sorted_obs, _indices = _prepare_observable_ordering(self.observables)
        return sorted_obs

    @property
    def _observable_sorted_indices(self) -> tuple[int, ...]:
        """Maps each user-list index to the corresponding sorted worker-buffer row."""
        _sorted_obs, indices = _prepare_observable_ordering(self.observables)
        return indices


class AnalogSimParams(_ObservableOrderingMixin):
    """Settings for analog Hamiltonian evolution.

    Constructor settings remain mutable. :meth:`~mqt.yaqs.Simulator.run`
    validates them and rebuilds the fixed time grid before each run.

    Attributes:
        times: Sample times from zero through ``elapsed_time``, spaced by ``dt``.
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
        evolution_mode: Literal["tdvp", "bug"] = "tdvp",
        get_state: bool = False,
        random_seed: int | None = None,
        multi_time_observables: list[tuple[Observable, Observable]] | None = None,
        tdvp_sweeps: int = 1,
        tdvp_mode: TDVPMode = "2site",
    ) -> None:
        """Configure analog evolution and its outputs.

        Args:
            observables: Observables to measure, in result order.
            elapsed_time: Finite, non-negative duration; must be an integer multiple of ``dt``.
            dt: Finite, positive fixed time step.
            num_traj: Positive stochastic trajectory count.
            max_bond_dim: Positive bond cap. Omit to use the preset; explicit ``None`` removes the cap.
            trunc_mode: SVD rule: ``"discarded_weight"``, ``"relative"``,
                ``"hard_cutoff"``, or ``"relative_discarded_weight"``.
            svd_threshold: SVD threshold. Zero disables tolerance-based truncation
                in discarded-weight modes.
            krylov_tol: Tolerance for the local Krylov/Lanczos matrix exponential.
            order: TJM integration order, one or two.
            preset: ``"fast"``, ``"balanced"``, ``"accurate"``, or ``"exact"``.
                Sets ``num_traj``, ``max_bond_dim``, ``svd_threshold``, and ``krylov_tol``;
                explicit values override it. ``"exact"`` still has timestep and sampling error.
            sample_timesteps: Record every time step rather than only the final values.
            evolution_mode: ``"tdvp"`` (default) or ``"bug"`` for MPS evolution.
            get_state: Retain the final state in :class:`~mqt.yaqs.Result`.
            random_seed: Seed for stochastic jumps and static noise sampling.
            multi_time_observables: One- or two-site pairs ``(A, B)`` for noiseless
                ``list[State]`` ensembles, computing ``<psi(t)|A U(t) B|psi(0)>``.
            tdvp_sweeps: Positive number of symmetric TDVP substeps, each of duration
                ``dt / tdvp_sweeps``.
            tdvp_mode: TDVP geometry: ``"1site"``, ``"2site"``, or ``"dynamic"``.
        """
        normalized_seed = _validate_random_seed(random_seed)
        preset_values = _SIMULATION_PRESETS[_validate_preset(preset)]
        self.preset = preset
        obs_list: list[Observable] = [] if observables is None else list(observables)
        _validate_observable_mix(obs_list)
        self.observables = obs_list

        self.elapsed_time, self.dt, self.times = _build_analog_time_grid(elapsed_time, dt)
        self.sample_timesteps = validate_bool(sample_timesteps, name="sample_timesteps")
        self.num_traj = _validate_num_traj(num_traj if num_traj is not None else preset_values["num_traj"])
        self.max_bond_dim = _resolve_max_bond_dim(max_bond_dim, preset_values["max_bond_dim"])
        self.trunc_mode = _validate_trunc_mode(trunc_mode)
        self.svd_threshold = _validate_svd_threshold(
            svd_threshold if svd_threshold is not None else preset_values["svd_threshold"]
        )
        self.krylov_tol = _validate_krylov_tol(krylov_tol if krylov_tol is not None else preset_values["krylov_tol"])
        self.order = _validate_order(order)
        self.evolution_mode = _validate_evolution_mode(evolution_mode)
        self.get_state = validate_bool(get_state, name="get_state")
        self.random_seed = normalized_seed
        self.multi_time_observables = _validate_multi_time_observables(multi_time_observables)
        self.tdvp_sweeps = _validate_tdvp_sweeps(tdvp_sweeps)
        self.tdvp_mode = _validate_tdvp_mode(tdvp_mode)


class DigitalSimParams(_ObservableOrderingMixin):
    """Settings for MPS circuit simulation.

    Select outputs with ``observables``, ``shots``, and ``get_state``.
    A standalone run needs at least one output; a program segment can propagate
    the state without recording outputs. Shots sample the final computational
    basis, independently of the configured observables.

    Attributes:
        dt: Compatibility time step for analog evolution helpers.
    """

    dt = 1

    def __init__(
        self,
        *,
        observables: list[Observable] | None = None,
        shots: int | None = None,
        num_traj: int | None = None,
        max_bond_dim: int | object | None = _USE_PRESET,
        trunc_mode: str = "discarded_weight",
        svd_threshold: float | None = None,
        krylov_tol: float | None = None,
        preset: SimulationPreset = "balanced",
        get_state: bool = False,
        sample_layers: bool = False,
        num_mid_measurements: int = 0,
        random_seed: int | None = None,
        gate_mode: GateMode = "mpo",
        tdvp_sweeps: int = 1,
        tdvp_mode: TDVPMode = "2site",
    ) -> None:
        """Configure circuit evolution and its outputs.

        Args:
            observables: Observables to measure, in result order.
            shots: Positive total bitstring-sample budget, or ``None`` to skip readout.
                Requires an all-qubit state. With noise, samples are distributed
                across trajectories; noiseless readout uses one final state.
            num_traj: Positive noisy trajectory count for observables and diagnostics.
                Independent of ``shots`` and ignored without noise. In noisy runs
                with both outputs, ``shots < num_traj`` gives some trajectories zero shots.
            max_bond_dim: Positive bond cap. Omit to use the preset; explicit ``None`` removes the cap.
            trunc_mode: SVD rule: ``"discarded_weight"``, ``"relative"``,
                ``"hard_cutoff"``, or ``"relative_discarded_weight"``.
            svd_threshold: SVD truncation threshold.
            krylov_tol: Tolerance for the local Krylov/Lanczos matrix exponential.
            preset: ``"fast"``, ``"balanced"``, ``"accurate"``, or ``"exact"``.
                Sets ``num_traj``, ``max_bond_dim``, ``svd_threshold``, and ``krylov_tol``;
                explicit values override it. Sampling and numerical errors can remain.
            get_state: Retain the final state in :class:`~mqt.yaqs.Result`.
            sample_layers: Record observables at ``SAMPLE_OBSERVABLES`` barriers.
            num_mid_measurements: Non-negative number of sampling barriers when sampling layers.
            random_seed: Seed for stochastic jumps and static noise sampling;
                does not seed final shot readout.
            gate_mode: Gate update mode: ``"swaps"``, ``"tdvp"``, ``"full-tdvp"``, or ``"mpo"``.
            tdvp_sweeps: Positive number of symmetric TDVP substeps per gate.
            tdvp_mode: TDVP geometry: ``"1site"``, ``"2site"``, or ``"dynamic"``.
        """
        normalized_seed = _validate_random_seed(random_seed)
        preset_values = _SIMULATION_PRESETS[_validate_preset(preset)]
        self.preset = preset
        obs_list: list[Observable] = [] if observables is None else list(observables)
        _validate_observable_mix(obs_list)
        self.observables = obs_list

        self.shots = _validate_shots(shots)

        # ``sample_layers`` may be set without observables here so a
        # :class:`~mqt.yaqs.SimulationProgram` can inject program-wide observables later.
        self.num_traj = _validate_num_traj(num_traj if num_traj is not None else preset_values["num_traj"])
        self.max_bond_dim = _resolve_max_bond_dim(max_bond_dim, preset_values["max_bond_dim"])
        self.trunc_mode = _validate_trunc_mode(trunc_mode)
        self.svd_threshold = _validate_svd_threshold(
            svd_threshold if svd_threshold is not None else preset_values["svd_threshold"]
        )
        self.krylov_tol = _validate_krylov_tol(krylov_tol if krylov_tol is not None else preset_values["krylov_tol"])
        self.get_state = validate_bool(get_state, name="get_state")
        self.sample_layers = validate_bool(sample_layers, name="sample_layers")
        self.num_mid_measurements = _validate_num_mid_measurements(num_mid_measurements)
        self.random_seed = normalized_seed
        self.gate_mode = _validate_gate_mode(gate_mode)
        self.tdvp_sweeps = _validate_tdvp_sweeps(tdvp_sweeps)
        self.tdvp_mode = _validate_tdvp_mode(tdvp_mode)


def _validate_simulation_controls(sim_params: AnalogSimParams | DigitalSimParams) -> None:
    """Validate and normalize mutable controls before a simulation run.

    Analog validation also rebuilds :attr:`AnalogSimParams.times` from the
    current ``elapsed_time`` and ``dt`` values.

    Args:
        sim_params: Analog or digital parameters supplied to an execution boundary.

    """
    sim_params.observables = list(sim_params.observables)
    _validate_observable_mix(sim_params.observables)
    sim_params.num_traj = _validate_num_traj(sim_params.num_traj)
    sim_params.max_bond_dim = _validate_max_bond_dim(sim_params.max_bond_dim)
    sim_params.trunc_mode = _validate_trunc_mode(sim_params.trunc_mode)
    sim_params.svd_threshold = _validate_svd_threshold(sim_params.svd_threshold)
    sim_params.krylov_tol = _validate_krylov_tol(sim_params.krylov_tol)
    sim_params.get_state = validate_bool(sim_params.get_state, name="get_state")
    sim_params.random_seed = _validate_random_seed(sim_params.random_seed)
    sim_params.tdvp_sweeps = _validate_tdvp_sweeps(sim_params.tdvp_sweeps)
    sim_params.tdvp_mode = _validate_tdvp_mode(sim_params.tdvp_mode)
    if isinstance(sim_params, AnalogSimParams):
        sim_params.elapsed_time, sim_params.dt, sim_params.times = _build_analog_time_grid(
            sim_params.elapsed_time,
            sim_params.dt,
        )
        sim_params.sample_timesteps = validate_bool(sim_params.sample_timesteps, name="sample_timesteps")
        sim_params.order = _validate_order(sim_params.order)
        sim_params.evolution_mode = _validate_evolution_mode(sim_params.evolution_mode)
        sim_params.multi_time_observables = _validate_multi_time_observables(sim_params.multi_time_observables)
        return
    sim_params.shots = _validate_shots(sim_params.shots)
    sim_params.sample_layers = validate_bool(sim_params.sample_layers, name="sample_layers")
    sim_params.num_mid_measurements = _validate_num_mid_measurements(sim_params.num_mid_measurements)
    sim_params.gate_mode = _validate_gate_mode(sim_params.gate_mode)
