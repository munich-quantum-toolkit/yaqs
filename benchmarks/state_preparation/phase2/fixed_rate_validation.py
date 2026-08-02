# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Independent exhaustive oracle for fixed-rate Pauli circuit-TJM branches.

This module deliberately does not call the production trajectory sampler.  It
constructs the exact branch law implied by one gate-local fixed-rate Pauli
instruction: scalar no-jump drift followed by normalization gives an identity
branch, while at most one Pauli jump is selected at each noisy gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, cast

import numpy as np

from mqt.yaqs.optimization import GateNoiseContext, KrotovNoiseMap, TJMNoiseInstruction

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import GateNoiseProvider, KrotovTJMOptions, ParameterizedCircuit

_PAULI_PROCESS_PREFIXES = ("pauli_", "crosstalk_", "longrange_crosstalk_")
_UNITARY_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class FixedRateGateBranch:
    """One exact outcome of the fixed-rate Pauli invocation after a gate."""

    gate_index: int
    outcome_label: str
    process_index: int | None
    probability: float
    noise_map: KrotovNoiseMap


@dataclass(frozen=True, slots=True)
class FixedRateCircuitBranch:
    """One complete ordered circuit-TJM branch and its exact probability."""

    gate_branches: tuple[FixedRateGateBranch, ...]
    probability: float

    @property
    def noise_maps(self) -> tuple[KrotovNoiseMap, ...]:
        """Physical replay maps in circuit-gate order."""
        return tuple(branch.noise_map for branch in self.gate_branches)

    @property
    def outcome_labels(self) -> tuple[str, ...]:
        """Diagnostic branch labels in circuit-gate order."""
        return tuple(branch.outcome_label for branch in self.gate_branches)


@dataclass(frozen=True, slots=True)
class FixedRateBranchTree:
    """Exact per-gate and full-circuit fixed-rate Pauli branch tree."""

    gate_branches: tuple[tuple[FixedRateGateBranch, ...], ...]
    circuit_branches: tuple[FixedRateCircuitBranch, ...]


def _identity_gate_branch(gate_index: int, resolved_angle: float | None) -> FixedRateGateBranch:
    """Return the sole outcome for a gate excluded from the noise profile."""
    return FixedRateGateBranch(
        gate_index=gate_index,
        outcome_label="not_applied",
        process_index=None,
        probability=1.0,
        noise_map=KrotovNoiseMap(
            source_gate_index=gate_index,
            resolved_native_angle=resolved_angle,
            is_identity=True,
        ),
    )


def _process_operator(
    process: Mapping[str, object],
    process_index: int,
) -> tuple[NDArray[np.complex128], tuple[int, ...]]:
    """Return and validate one unitary Pauli jump operator.

    Args:
        process: Fixed-rate process record from a gate-local noise model.
        process_index: Position of the record in that noise model.

    Returns:
        An immutable dense operator and its physical sites.

    Raises:
        ValueError: If the process is not a finite unitary Pauli process with
            consistent support.
    """
    name = process.get("name")
    if not isinstance(name, str) or not name.startswith(_PAULI_PROCESS_PREFIXES):
        msg = f"Process {process_index} is not a recognized fixed-rate Pauli process: {name!r}."
        raise ValueError(msg)

    raw_sites = process.get("sites")
    if not isinstance(raw_sites, list) or not raw_sites or any(type(site) is not int for site in raw_sites):
        msg = f"Process {process_index} has invalid sites {raw_sites!r}."
        raise ValueError(msg)
    sites = tuple(cast("list[int]", raw_sites))

    if "matrix" in process:
        operator = np.asarray(process["matrix"], dtype=np.complex128)
    else:
        factors = process.get("factors")
        if not isinstance(factors, tuple) or len(factors) != 2:
            msg = f"Process {process_index} has no dense matrix or two-factor Pauli representation."
            raise ValueError(msg)
        operator = np.kron(
            np.asarray(factors[0], dtype=np.complex128),
            np.asarray(factors[1], dtype=np.complex128),
        )

    dimension = 2 ** len(sites)
    if operator.shape != (dimension, dimension) or not np.all(np.isfinite(operator)):
        msg = f"Process {process_index} has an invalid operator shape or non-finite entries."
        raise ValueError(msg)
    identity = np.eye(dimension, dtype=np.complex128)
    if not np.allclose(operator.conj().T @ operator, identity, atol=_UNITARY_TOLERANCE, rtol=_UNITARY_TOLERANCE):
        msg = f"Process {process_index} is not unitary, so the fixed-rate Pauli branch oracle does not apply."
        raise ValueError(msg)

    immutable = np.array(operator, dtype=np.complex128, copy=True, order="C")
    immutable.flags.writeable = False
    return immutable, sites


def _fixed_rate_gate_branches(
    instruction: TJMNoiseInstruction,
    context: GateNoiseContext,
    dt: float,
) -> tuple[FixedRateGateBranch, ...]:
    """Enumerate no-jump and at-most-one-jump outcomes for one noisy gate.

    Args:
        instruction: Gate-local fixed-rate TJM instruction.
        context: Resolved physical placement for the gate.
        dt: Positive circuit-TJM time step.

    Returns:
        The complete normalized outcome law for this gate invocation.

    Raises:
        ValueError: If any process is not a valid fixed-rate unitary Pauli process.
        RuntimeError: If the independently constructed probabilities fail to
            sum to one within numerical tolerance.
    """
    processes = cast("list[Mapping[str, object]]", instruction.noise_model.processes)
    rates: list[float] = []
    operators: list[tuple[NDArray[np.complex128], tuple[int, ...]]] = []
    for process_index, process in enumerate(processes):
        strength = process.get("strength")
        if type(strength) is not float or not math.isfinite(strength) or strength < 0.0:
            msg = f"Process {process_index} has invalid fixed rate {strength!r}."
            raise ValueError(msg)
        rates.append(strength)
        operators.append(_process_operator(process, process_index))

    total_rate = math.fsum(rates)
    no_jump_probability = math.exp(-dt * total_rate)
    branches = [
        FixedRateGateBranch(
            gate_index=context.gate_index,
            outcome_label="no_jump",
            process_index=None,
            probability=no_jump_probability,
            noise_map=KrotovNoiseMap(
                normalized=total_rate > 0.0,
                channel_id=instruction.channel_id,
                outcome_labels=("no_jump",),
                source_gate_index=context.gate_index,
                resolved_native_angle=context.resolved_angle,
                is_identity=True,
            ),
        )
    ]
    if total_rate > 0.0:
        jump_probability = -math.expm1(-dt * total_rate)
        for process_index, (process, rate, operator) in enumerate(zip(processes, rates, operators, strict=True)):
            if rate <= 0.0:
                continue
            process_name = cast("str", process["name"])
            branches.append(
                FixedRateGateBranch(
                    gate_index=context.gate_index,
                    outcome_label=process_name,
                    process_index=process_index,
                    probability=jump_probability * rate / total_rate,
                    noise_map=KrotovNoiseMap(
                        operators=(operator,),
                        normalized=True,
                        jump_process_index=process_index,
                        channel_id=instruction.channel_id,
                        outcome_labels=(process_name,),
                        source_gate_index=context.gate_index,
                        resolved_native_angle=context.resolved_angle,
                        is_identity=False,
                    ),
                )
            )

    if not math.isclose(math.fsum(branch.probability for branch in branches), 1.0, rel_tol=0.0, abs_tol=1e-14):
        msg = f"Gate {context.gate_index} branch probabilities do not sum to one."
        raise RuntimeError(msg)
    return tuple(branches)


def enumerate_fixed_rate_pauli_branches(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    provider: GateNoiseProvider,
    tjm_options: KrotovTJMOptions,
) -> FixedRateBranchTree:
    """Enumerate the exact branch tree for a fixed-rate Pauli provider.

    Args:
        circuit: Parameterized circuit whose post-gate placement is enumerated.
        theta: Resolved circuit parameters.
        provider: Gate-local provider returning fixed-rate Pauli TJM models.
        tjm_options: Placement and time-step settings.

    Returns:
        Exact per-gate outcomes and their full Cartesian-product branch tree.

    Raises:
        TypeError: If the provider returns a non-TJM instruction.
        RuntimeError: If a gate or full-circuit branch law fails normalization.
    """
    parameters = np.asarray(theta, dtype=np.float64)
    x_value = np.array([], dtype=np.float64)
    per_gate: list[tuple[FixedRateGateBranch, ...]] = []
    provider_rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(0)))

    for gate_index, gate in enumerate(circuit.gates):
        _matrix, sites, resolved_angle = circuit.gate_matrix_and_angle(gate, parameters, x_value)
        context = GateNoiseContext(
            gate_index=gate_index,
            gate_name=gate.name,
            sites=sites,
            arity=len(sites),
            resolved_angle=resolved_angle,
            logical_gate_id=gate.logical_gate_id if gate.logical_gate_id is not None else gate_index,
            native_gate_id=gate.native_gate_id if gate.native_gate_id is not None else gate_index,
            parameter_index=gate.param_index,
        )
        selected = (
            gate.noise_enabled
            and (tjm_options.noisy_gate_indices is None or gate_index in tjm_options.noisy_gate_indices)
            and (tjm_options.apply_noise_to == "all" or len(sites) == 2)
        )
        if not selected:
            per_gate.append((_identity_gate_branch(gate_index, resolved_angle),))
            continue
        instruction = provider(context, provider_rng)
        if instruction is None:
            per_gate.append((_identity_gate_branch(gate_index, resolved_angle),))
            continue
        if not isinstance(instruction, TJMNoiseInstruction):
            msg = "The fixed-rate Pauli branch oracle requires a TJMNoiseInstruction provider."
            raise TypeError(msg)
        per_gate.append(_fixed_rate_gate_branches(instruction, context, tjm_options.dt))

    circuit_branches = tuple(
        FixedRateCircuitBranch(
            gate_branches=tuple(gate_outcomes),
            probability=math.prod(outcome.probability for outcome in gate_outcomes),
        )
        for gate_outcomes in product(*per_gate)
    )
    if not math.isclose(
        math.fsum(branch.probability for branch in circuit_branches),
        1.0,
        rel_tol=0.0,
        abs_tol=2e-14,
    ):
        msg = "Full circuit branch probabilities do not sum to one."
        raise RuntimeError(msg)
    return FixedRateBranchTree(tuple(per_gate), circuit_branches)


__all__ = [
    "FixedRateBranchTree",
    "FixedRateCircuitBranch",
    "FixedRateGateBranch",
    "enumerate_fixed_rate_pauli_branches",
]
