# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Result wrapper returned by :meth:`~mqt.yaqs.Simulator.run`.

This module defines :class:`Result`, a thin wrapper around the populated simulation
parameters that were used for a run. It exposes the parts of those parameters that
are produced by the simulator -- observables, the sampled noise model, the final
output state, analog multi-time correlator results, and (for weak digital simulations)
measurement counts -- through properties so user code does not need to reach into
``sim_params`` directly.

Properties are ordered by primary simulation mode (analog first, digital weak last)
to match the rest of the :class:`~mqt.yaqs.Simulator` surface.

Future refactors will move these outputs off of the simulation parameter classes and
onto :class:`Result` as first-class fields. The proxy properties here document that
intent and keep the call site stable in the meantime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .simulation_parameters import AnalogSimParams, WeakSimParams

if TYPE_CHECKING:
    from numpy import complex128, float64
    from numpy.typing import NDArray

    from .noise_model import NoiseModel
    from .simulation_parameters import Observable, StrongSimParams
    from .state import State


@dataclass
class Result:
    """Result of a :meth:`~mqt.yaqs.Simulator.run` call.

    This is currently a thin wrapper around the simulation-parameter object that was
    populated during the run; observable data still lives on the individual
    :class:`~mqt.yaqs.core.data_structures.simulation_parameters.Observable` instances
    referenced by :attr:`sim_params`. The convenience properties below provide a
    stable accessor surface so call sites do not have to introspect the parameter type.
    """

    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams

    @property
    def observables(self) -> list[Observable]:
        """Return the observables that were measured during the run.

        ``WeakSimParams`` does not track observables; this property returns an empty list
        for weak simulations.
        """
        if isinstance(self.sim_params, WeakSimParams):
            return []
        return self.sim_params.observables

    @property
    def noise_model(self) -> NoiseModel | None:
        """Return the sampled noise model used for the run, if any."""
        return self.sim_params.noise_model

    @property
    def output_state(self) -> State | None:
        """Return the final state when ``get_state=True``, else ``None``."""
        return self.sim_params.output_state

    @property
    def multi_time_times(self) -> NDArray[float64] | None:
        """Return the time grid for two-time correlators (analog ensembles only)."""
        if isinstance(self.sim_params, AnalogSimParams):
            return self.sim_params.multi_time_observables_times
        return None

    @property
    def multi_time_results(self) -> NDArray[complex128] | None:
        """Return the two-time correlator results (analog ensembles only)."""
        if isinstance(self.sim_params, AnalogSimParams):
            return self.sim_params.multi_time_observables_results
        return None

    @property
    def counts(self) -> dict[int, int] | None:
        """Return the aggregated measurement counts for weak simulations, else ``None``."""
        if isinstance(self.sim_params, WeakSimParams):
            return getattr(self.sim_params, "results", None)
        return None
