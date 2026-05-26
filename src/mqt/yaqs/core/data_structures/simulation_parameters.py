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
from typing import TYPE_CHECKING, Literal

import numpy as np

from mqt.yaqs.core.libraries.gate_library import GateLibrary

if TYPE_CHECKING:
    from mqt.yaqs.core.libraries.gate_library import BaseGate

AccuracyPreset = Literal["fast", "balanced", "accurate"]

ACCURACY_PRESETS: dict[AccuracyPreset, dict[str, float | int]] = {
    "fast": {"threshold": 1e-3, "max_bond_dim": 16, "num_traj": 64},
    "balanced": {"threshold": 1e-6, "max_bond_dim": 128, "num_traj": 256},
    "accurate": {"threshold": 1e-9, "max_bond_dim": 4096, "num_traj": 1000},
}

_EXPERT_DEFAULT_THRESHOLD = 1e-9
_EXPERT_DEFAULT_MAX_BOND_DIM = 4096
_EXPERT_DEFAULT_NUM_TRAJ = 1000


def _validate_accuracy(accuracy: AccuracyPreset | None) -> AccuracyPreset | None:
    """Validate ``accuracy`` preset names at runtime.

    Args:
        accuracy: Preset name or ``None`` for expert defaults.

    Returns:
        The validated preset name, or ``None``.

    Raises:
        ValueError: If ``accuracy`` is not a supported preset name.
    """
    if accuracy is None:
        return None
    if accuracy not in ACCURACY_PRESETS:
        msg = f"accuracy must be one of {sorted(ACCURACY_PRESETS)!r}, got {accuracy!r}."
        raise ValueError(msg)
    return accuracy


def _get_accuracy_preset(accuracy: AccuracyPreset | None) -> dict[str, float | int]:
    """Return the preset values for ``accuracy`` or expert defaults for ``None``.

    Returns:
        Mapping with ``threshold``, ``max_bond_dim``, and ``num_traj`` keys.
    """
    accuracy = _validate_accuracy(accuracy)
    if accuracy is None:
        return {
            "threshold": _EXPERT_DEFAULT_THRESHOLD,
            "max_bond_dim": _EXPERT_DEFAULT_MAX_BOND_DIM,
            "num_traj": _EXPERT_DEFAULT_NUM_TRAJ,
        }
    return ACCURACY_PRESETS[accuracy]


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


class AnalogSimParams:
    """Analog Simulation Parameters.

    A class to represent the parameters for an analog simulation.

    Attributes:
        observables: List of observables tracked during the simulation.
        sorted_observables: Observables sorted by site and name.
        elapsed_time: Total simulation time.
        dt: Simulation time step.
        times: Array of sampled times from ``0`` to ``elapsed_time`` with spacing ``dt``.
        sample_timesteps: If ``True``, record values at all sampled timesteps.
        num_traj: Number of trajectories (for stochastic open-system evolution).
        random_seed: If set, seeds per-trajectory jump RNG and static noise sampling for reproducible runs.
        max_bond_dim: Maximum allowed bond dimension.
        accuracy: Preset controlling ``threshold``, ``max_bond_dim``, and ``num_traj``.
            Default is ``"balanced"``. ``"accurate"`` matches the previous high-accuracy
            defaults. ``accuracy=None`` selects the same expert defaults as ``"accurate"``.
            Explicit ``threshold``, ``max_bond_dim``, and ``num_traj`` override the preset.
        trunc_mode: Truncation mode used in TDVP (``"discarded_weight"`` or ``"relative"``).
        threshold: Truncation threshold.
        order: Integration order.
        get_state: If ``True``, request the final state on the returned :class:`~mqt.yaqs.Result`.
        multi_time_observables: Optional list of ``(A, B)`` observable pairs for unitary-ensemble
            two-time correlators. Each entry computes ``<psi(t)|A U(t) B|psi(0)>``.
            Autocorrelation is the special case ``(O, O)``. Results are indexed by pair position.
    """

    def __init__(
        self,
        observables: list[Observable] | None = None,
        elapsed_time: float = 0.1,
        dt: float = 0.1,
        num_traj: int | None = None,
        max_bond_dim: int | None = None,
        min_bond_dim: int = 2,
        trunc_mode: str = "discarded_weight",
        threshold: float | None = None,
        order: int = 1,
        *,
        accuracy: AccuracyPreset | None = "balanced",
        sample_timesteps: bool = True,
        evolution_mode: EvolutionMode = EvolutionMode.TDVP,
        get_state: bool = False,
        random_seed: int | None = None,
        multi_time_observables: list[tuple[Observable, Observable]] | None = None,
    ) -> None:
        """Physics simulation parameters initialization.

        Initializes parameters for a physics-based quantum simulation.

        Args:
            observables: List of observables to measure during the simulation.
            elapsed_time: Total simulation time.
            dt: Time step interval.
            num_traj: Number of simulation samples.
            random_seed: If set, makes stochastic trajectories and noise-model sampling reproducible.
            max_bond_dim: Maximum bond dimension allowed.
            min_bond_dim: Minimum bond dimension used to improve TDVP accuracy when possible.
            accuracy: Preset controlling ``threshold``, ``max_bond_dim``, and ``num_traj``.
                Default is ``"balanced"``. ``"accurate"`` matches the previous high-accuracy
                defaults. ``accuracy=None`` selects the same expert defaults as ``"accurate"``.
                Explicit ``threshold``, ``max_bond_dim``, and ``num_traj`` override the preset.
            trunc_mode: TDVP truncation mode (``"discarded_weight"`` or ``"relative"``).
            threshold: Threshold for simulation accuracy.
            order: Order of approximation or numerical scheme.
            sample_timesteps: Whether to sample at intermediate time steps.
            evolution_mode: Tensor evolution mode (default ``EvolutionMode.TDVP``).
            get_state: If ``True``, request the final state on the returned :class:`~mqt.yaqs.Result`.
            multi_time_observables: For ``list[State]`` unitary ensemble runs only, list of ``(A, B)``
                pairs evaluated as ``<psi(t)|A U(t) B|psi(0)>``. Autocorrelation is the special
                case ``(O, O)``.

        """
        _validate_random_seed(random_seed)
        preset = _get_accuracy_preset(accuracy)
        obs_list: list[Observable] = [] if observables is None else list(observables)
        assert all(n.gate.name == "pvm" for n in obs_list) or all(n.gate.name != "pvm" for n in obs_list), (
            "We currently have not implemented mixed observable and projective-measurement simulation."
        )
        self.observables = obs_list

        if self.observables:
            sortable = [obs for obs in self.observables if obs.gate.name != "pvm"]
            unsorted = [obs for obs in self.observables if obs.gate.name == "pvm"]
            sorted_obs = sorted(
                sortable,
                key=lambda obs: obs.sites[0] if isinstance(obs.sites, list) else obs.sites,
            )
            self.sorted_observables = sorted_obs + unsorted
        else:
            self.sorted_observables = []

        self.elapsed_time = elapsed_time
        self.dt = dt
        self.times = np.arange(0, elapsed_time + dt, dt)
        self.sample_timesteps = sample_timesteps
        self.num_traj = num_traj if num_traj is not None else int(preset["num_traj"])
        self.max_bond_dim = max_bond_dim if max_bond_dim is not None else int(preset["max_bond_dim"])
        self.min_bond_dim = min_bond_dim
        self.trunc_mode = trunc_mode
        self.threshold = threshold if threshold is not None else float(preset["threshold"])
        self.order = order
        self.evolution_mode = evolution_mode
        self.get_state = get_state
        self.random_seed = random_seed
        self.multi_time_observables: list[tuple[Observable, Observable]] = (
            [] if multi_time_observables is None else list(multi_time_observables)
        )


class StrongSimParams:
    """Strong Circuit Simulation Parameters.

    A class to represent the parameters for a strong simulation.

    Attributes:
        dt: A placeholder property for code compatibility.
        observables: A list of observables to be tracked during the simulation.
        sorted_observables: A list of observables sorted by site and name.
        num_traj: The number of trajectories to simulate.
        random_seed: If set, seeds per-trajectory jump RNG and static noise
            sampling for reproducible runs.
        max_bond_dim: The maximum bond dimension for the simulation.
        accuracy: Preset controlling ``threshold``, ``max_bond_dim``, and ``num_traj``.
            Default is ``"balanced"``. ``"accurate"`` matches the previous high-accuracy
            defaults. ``accuracy=None`` selects the same expert defaults as ``"accurate"``.
            Explicit ``threshold``, ``max_bond_dim``, and ``num_traj`` override the preset.
        min_bond_dim: The minimum bond dimension if possible which gives TDVP
            better accuracy. Default is 2.
        trunc_mode: The type of truncation performed in TDVP. Options are
            ``"discarded_weight"`` and ``"relative"``.
        threshold: The threshold value for the simulation.
        window_size: The size of the window for the simulation. Default is ``None``.
        get_state: If ``True``, request the final state on the returned
            :class:`~mqt.yaqs.Result`.
    """

    # Properties set as placeholders for code compatibility
    dt = 1

    def __init__(
        self,
        observables: list[Observable] | None = None,
        num_traj: int | None = None,
        max_bond_dim: int | None = None,
        min_bond_dim: int = 2,
        trunc_mode: str = "discarded_weight",
        threshold: float | None = None,
        *,
        accuracy: AccuracyPreset | None = "balanced",
        get_state: bool = False,
        sample_layers: bool = False,
        num_mid_measurements: int = 0,
        random_seed: int | None = None,
    ) -> None:
        """Strong circuit simulation parameters initialization.

        Initializes parameters for a strong quantum circuit simulation.

        Args:
            observables: List of observables to measure during simulation.
            num_traj: Number of trajectories to simulate.
            max_bond_dim: Maximum bond dimension allowed in simulation.
            min_bond_dim: Minimum bond dimension when TDVP can use it for better accuracy.
            accuracy: Preset controlling ``threshold``, ``max_bond_dim``, and ``num_traj``.
                Default is ``"balanced"``. ``"accurate"`` matches the previous high-accuracy
                defaults. ``accuracy=None`` selects the same expert defaults as ``"accurate"``.
                Explicit ``threshold``, ``max_bond_dim``, and ``num_traj`` override the preset.
            trunc_mode: TDVP truncation mode (``"discarded_weight"`` or ``"relative"``).
            threshold: Threshold for simulation accuracy.
            get_state: If ``True``, request the final state on the returned :class:`~mqt.yaqs.Result`.
            sample_layers: If ``True``, record observables at sampled circuit layers.
            num_mid_measurements: Number of mid-circuit measurement barriers when sampling layers.
            random_seed: If set, makes stochastic trajectories and noise-model sampling reproducible.
        """
        _validate_random_seed(random_seed)
        preset = _get_accuracy_preset(accuracy)
        obs_list: list[Observable] = [] if observables is None else list(observables)
        assert all(n.gate.name == "pvm" for n in obs_list) or all(n.gate.name != "pvm" for n in obs_list), (
            "We currently have not implemented mixed observable and projective-measurement simulation."
        )
        self.observables = obs_list

        if self.observables:
            sortable = [obs for obs in self.observables if obs.gate.name != "pvm"]
            unsorted = [obs for obs in self.observables if obs.gate.name == "pvm"]
            sorted_obs = sorted(
                sortable,
                key=lambda obs: obs.sites[0] if isinstance(obs.sites, list) else obs.sites,
            )
            self.sorted_observables = sorted_obs + unsorted
        else:
            self.sorted_observables = []

        self.num_traj = num_traj if num_traj is not None else int(preset["num_traj"])
        self.max_bond_dim = max_bond_dim if max_bond_dim is not None else int(preset["max_bond_dim"])
        self.min_bond_dim = min_bond_dim
        self.trunc_mode = trunc_mode
        self.threshold = threshold if threshold is not None else float(preset["threshold"])
        self.get_state = get_state
        self.sample_layers = sample_layers
        self.num_mid_measurements = num_mid_measurements
        self.random_seed = random_seed


class WeakSimParams:
    """A class to represent the parameters for a weak simulation.

    Attributes:
        dt: A placeholder property for code compatibility.
        num_traj: A placeholder property for code compatibility.
        shots: The number of shots for the simulation.
        max_bond_dim: The maximum bond dimension for the simulation.
        accuracy: Preset controlling ``threshold`` and ``max_bond_dim``.
            Default is ``"balanced"``. ``"accurate"`` matches the previous high-accuracy
            defaults. ``accuracy=None`` selects the same expert defaults as ``"accurate"``.
            Explicit ``threshold`` and ``max_bond_dim`` override the preset.
        min_bond_dim: The minimum bond dimension if possible which gives TDVP
            better accuracy. Default is 2.
        trunc_mode: The type of truncation performed in TDVP. Options are
            ``"discarded_weight"`` and ``"relative"``.
        threshold: The threshold value for the simulation.
        window_size: The window size for the simulation.
        get_state: If ``True``, request the final state on the returned
            :class:`~mqt.yaqs.Result`.
    """

    # Properties set as placeholders for code compatibility
    dt = 1
    num_traj = 0

    def __init__(
        self,
        shots: int,
        max_bond_dim: int | None = None,
        min_bond_dim: int = 2,
        trunc_mode: str = "discarded_weight",
        threshold: float | None = None,
        *,
        accuracy: AccuracyPreset | None = "balanced",
        get_state: bool = False,
        random_seed: int | None = None,
    ) -> None:
        """Weak circuit simulation initialization.

        Initializes parameters for a weak circuit simulation.

        Args:
            shots: Number of measurement shots to simulate.
            max_bond_dim: Maximum bond dimension for simulation.
            min_bond_dim: Minimum bond dimension when TDVP can use it for better accuracy.
            accuracy: Preset controlling ``threshold`` and ``max_bond_dim``.
                Default is ``"balanced"``. ``"accurate"`` matches the previous high-accuracy
                defaults. ``accuracy=None`` selects the same expert defaults as ``"accurate"``.
                Explicit ``threshold`` and ``max_bond_dim`` override the preset.
            trunc_mode: TDVP truncation mode (``"discarded_weight"`` or ``"relative"``).
            threshold: Accuracy threshold for truncating tensors.
            get_state: If ``True``, request the final state on the returned :class:`~mqt.yaqs.Result`.
            random_seed: If set, makes per-shot jump RNG reproducible.
        """
        _validate_random_seed(random_seed)
        preset = _get_accuracy_preset(accuracy)
        self.shots = shots
        self.max_bond_dim = max_bond_dim if max_bond_dim is not None else int(preset["max_bond_dim"])
        self.min_bond_dim = min_bond_dim
        self.trunc_mode = trunc_mode
        self.threshold = threshold if threshold is not None else float(preset["threshold"])
        self.get_state = get_state
        self.random_seed = random_seed
