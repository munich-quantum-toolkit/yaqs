# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Krotov-inspired discrete adjoint optimization for parameterized circuits on MPS.

This module transfers the gate-local Krotov optimization scheme from dense
state-vector simulation to the MPS/MPO backend of YAQS. The continuous-time
forward-state/costate structure of Krotov's method is replaced by gate-index
locality: a forward sweep produces the intermediate MPS states
``|psi_k> = U_k ... U_1 |psi_0>``, a backward sweep propagates the terminal
costate ``|chi_M>`` (the loss derivative with respect to the final state) through
the adjoint gates, and the gate-local contribution

``Delta_k = (d alpha_k / d theta_k) * 2 Re <chi_k | (-i/2 G_k) | psi_k>``

is evaluated as an MPS overlap. Three training variants are provided:

- :func:`train_krotov_online`: sample-wise sequential stale-adjoint sweeps,
- :func:`train_krotov_batch`: full-batch updates equal to the empirical-loss gradient,
- :func:`train_krotov_hybrid`: online phase followed by batch refinement.

Gate applications are exact by default (no truncation), so the batch direction
coincides with the exact gradient of the empirical loss for noise-free circuits.
A truncation configuration can be supplied to trade accuracy for bond-dimension
control on larger systems.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import opt_einsum as oe

from ..core.data_structures.mps import MPS
from ..core.methods.decompositions import merge_two_site, split_two_site

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ..core.data_structures.simulation_parameters import Observable
    from ..core.methods.decompositions import TruncMode
    from .parameterized_circuit import ParameterizedCircuit, ParameterizedGate

    StatePrep = Callable[[NDArray[np.float64]], MPS]

_BCE_EPS = 1e-7

_SWAP_MATRIX = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    dtype=np.complex128,
)

LossKind = Literal["mse", "bce"]
ScheduleKind = Literal["constant", "inverse", "exp"]


@dataclass
class KrotovTruncation:
    """Truncation settings for gate applications during Krotov sweeps.

    The defaults keep all singular values, so forward states, costates, and
    gate-local contributions are exact for noise-free circuits.

    Attributes:
        max_bond_dim: Hard cap on the bond dimension, or ``None`` for no cap.
        svd_threshold: Truncation threshold passed to the SVD split.
        trunc_mode: Truncation mode (``"discarded_weight"`` or ``"relative"``).
        min_bond_dim: Minimum number of singular values to keep.
    """

    max_bond_dim: int | None = None
    svd_threshold: float = 0.0
    trunc_mode: str = "discarded_weight"
    min_bond_dim: int = 1


@dataclass
class KrotovReadout:
    """Terminal readout and loss specification.

    The scalar readout is ``z = <psi_M | O | psi_M>`` for a local one- or two-site
    observable ``O``. Two loss models are supported:

    - ``"mse"``: prediction ``z + bias`` with labels in ``{-1, +1}`` and squared
      error loss. The scalar bias is optional and trained with its exact gradient.
    - ``"bce"``: probability ``p = (z + 1) / 2`` with labels in ``{0, 1}`` and
      binary cross-entropy loss.

    Attributes:
        observable: Local observable defining the readout.
        loss: Loss model, ``"mse"`` or ``"bce"``.
        use_bias: Whether a trainable scalar bias is added to the prediction
            (``"mse"`` only).
    """

    observable: Observable
    loss: LossKind = "mse"
    use_bias: bool = False

    def __post_init__(self) -> None:
        """Validates the readout configuration.

        Raises:
            ValueError: If the loss kind is unknown or a bias is requested for BCE.
        """
        if self.loss not in {"mse", "bce"}:
            msg = f"Unknown loss kind: {self.loss!r}. Supported: 'mse', 'bce'."
            raise ValueError(msg)
        if self.use_bias and self.loss != "mse":
            msg = "A trainable bias is only supported for the 'mse' loss."
            raise ValueError(msg)


@dataclass
class KrotovOptions:
    """Hyperparameters for Krotov training.

    Attributes:
        max_iterations: Number of outer iterations (epochs).
        switch_iteration: Hybrid only: last iteration of the online phase.
        online_step_size: Step size of the online (stale-adjoint) phase.
        batch_step_size: Step size of the batch (exact-gradient) phase.
        online_schedule: Learning-rate schedule of the online phase.
        batch_schedule: Learning-rate schedule of the batch phase.
        online_decay: Decay rate of the online schedule.
        batch_decay: Decay rate of the batch schedule.
        seed: Seed for the per-epoch sample permutation of the online phase.
        truncation: Truncation settings for all gate applications.
    """

    max_iterations: int = 100
    switch_iteration: int = 20
    online_step_size: float = 0.1
    batch_step_size: float = 0.1
    online_schedule: ScheduleKind = "constant"
    batch_schedule: ScheduleKind = "constant"
    online_decay: float = 0.0
    batch_decay: float = 0.0
    seed: int = 0
    truncation: KrotovTruncation = field(default_factory=KrotovTruncation)


@dataclass
class KrotovResult:
    """Result of a Krotov training run.

    Attributes:
        theta: Final trainable gate parameters.
        bias: Final scalar bias (``0.0`` if the readout has no bias).
        trace: Per-iteration record with keys ``"step"``, ``"phase"``, ``"loss"``,
            ``"step_size"``, ``"gradient_norm"``, and ``"update_norm"``.
    """

    theta: NDArray[np.float64]
    bias: float
    trace: dict[str, list[float | int | str]]


def _step_size(base: float, iteration: int, schedule: ScheduleKind, decay: float) -> float:
    """Evaluate the scheduled step size for one iteration.

    Args:
        base: Base step size.
        iteration: One-based iteration counter within the phase.
        schedule: Schedule kind (``"constant"``, ``"inverse"``, or ``"exp"``).
        decay: Decay rate of the schedule.

    Returns:
        The step size for this iteration.

    Raises:
        ValueError: If the schedule kind is unknown.
    """
    exponent = max(iteration - 1, 0)
    if schedule == "constant":
        return base
    if schedule == "inverse":
        return base / (1.0 + decay * exponent)
    if schedule == "exp":
        return base * float(np.exp(-decay * exponent))
    msg = f"Unknown learning-rate schedule: {schedule!r}"
    raise ValueError(msg)


def _apply_one_site(state: MPS, matrix: NDArray[np.complex128], site: int) -> None:
    """Apply a single-qubit operator to an MPS in place.

    Args:
        state: The MPS to update.
        matrix: ``2 x 2`` operator.
        site: Site the operator acts on.
    """
    state.tensors[site] = np.asarray(
        oe.contract("ab,bcd->acd", matrix, state.tensors[site]), dtype=np.complex128
    )


def _apply_two_site_adjacent(
    state: MPS, matrix: NDArray[np.complex128], left_site: int, truncation: KrotovTruncation
) -> None:
    """Apply a two-qubit operator to adjacent sites of an MPS in place.

    Args:
        state: The MPS to update.
        matrix: ``4 x 4`` operator on the merged index ``|q_left, q_right>``.
        left_site: The lower of the two adjacent sites.
        truncation: Truncation settings for the SVD split.
    """
    right_site = left_site + 1
    left_tensor = state.tensors[left_site]
    right_tensor = state.tensors[right_site]
    d_left, d_right = left_tensor.shape[0], right_tensor.shape[0]

    merged = merge_two_site(left_tensor, right_tensor)
    updated = np.asarray(oe.contract("ab,bcd->acd", matrix, merged), dtype=np.complex128)
    new_left, new_right = split_two_site(
        updated,
        [d_left, d_right],
        svd_distribution="right",
        trunc_mode=cast("TruncMode", truncation.trunc_mode),
        threshold=truncation.svd_threshold,
        max_bond_dim=truncation.max_bond_dim,
        min_bond_dim=truncation.min_bond_dim,
    )
    state.tensors[left_site] = new_left
    state.tensors[right_site] = new_right


def _apply_operator(
    state: MPS,
    matrix: NDArray[np.complex128],
    sites: tuple[int, ...],
    truncation: KrotovTruncation,
) -> None:
    """Apply a one- or two-site operator (ascending-site convention) to an MPS in place.

    Long-range two-qubit operators are handled by SWAP chains that bring the upper
    site adjacent to the lower one and back.

    Args:
        state: The MPS to update.
        matrix: ``2 x 2`` or ``4 x 4`` operator on the merged index ``|q_min, q_max>``.
        sites: Sites the operator acts on, sorted ascending.
        truncation: Truncation settings for the SVD splits.
    """
    if len(sites) == 1:
        _apply_one_site(state, matrix, sites[0])
        return

    left_site, right_site = sites
    if right_site == left_site + 1:
        _apply_two_site_adjacent(state, matrix, left_site, truncation)
        return

    for i in range(right_site - 1, left_site, -1):
        _apply_two_site_adjacent(state, _SWAP_MATRIX, i, truncation)
    _apply_two_site_adjacent(state, matrix, left_site, truncation)
    for i in range(left_site + 1, right_site):
        _apply_two_site_adjacent(state, _SWAP_MATRIX, i, truncation)


def _expectation(state: MPS, observable: Observable) -> float:
    """Compute ``<state | O | state>`` by full contraction, without canonical-form assumptions.

    Args:
        state: The (normalized) MPS.
        observable: Local one- or two-site observable.

    Returns:
        The real expectation value.
    """
    ket = copy.deepcopy(state)
    ket.apply_local(observable)
    value = state.scalar_product(ket)
    return float(np.real(value))


def _resolve_initial_state(
    initial_state: MPS | StatePrep | None,
    x: NDArray[np.float64],
    num_qubits: int,
) -> MPS:
    """Build the circuit input state for one sample.

    Args:
        initial_state: Fixed MPS, sample-dependent state-preparation callable, or
            ``None`` for the all-zeros state.
        x: Input sample.
        num_qubits: Number of qubits.

    Returns:
        A fresh MPS instance for this sample.
    """
    if initial_state is None:
        return MPS(num_qubits)
    if isinstance(initial_state, MPS):
        return copy.deepcopy(initial_state)
    return initial_state(x)


def forward_states(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    x: NDArray[np.float64],
    initial_state: MPS,
    truncation: KrotovTruncation,
) -> list[MPS]:
    """Propagate the initial state through the circuit, recording every intermediate MPS.

    Args:
        circuit: The parameterized circuit.
        theta: Trainable parameter vector.
        x: Input sample.
        initial_state: Circuit input state (consumed; pass a fresh copy).
        truncation: Truncation settings for gate applications.

    Returns:
        States ``[|psi_0>, |psi_1>, ..., |psi_M>]`` with ``|psi_k>`` the state after
        the first ``k`` gates.
    """
    states = [initial_state]
    for gate in circuit.gates:
        state = copy.deepcopy(states[-1])
        matrix, sites = circuit.gate_matrix(gate, theta, x)
        _apply_operator(state, matrix, sites, truncation)
        states.append(state)
    return states


def _loss_and_costate_factor(z: float, bias: float, y: float, loss: LossKind) -> tuple[float, float]:
    """Evaluate the sample loss and the scalar prefactor of the terminal costate.

    The terminal costate is ``|chi_M> = factor * O |psi_M>`` with the factor defined
    by the first variation of the sample loss with respect to the final state.

    Args:
        z: Readout expectation value.
        bias: Scalar bias (``"mse"`` only).
        y: Sample label.
        loss: Loss kind.

    Returns:
        A tuple ``(loss_value, costate_factor)``.
    """
    if loss == "mse":
        residual = z + bias - y
        return residual**2, 2.0 * residual
    probability = float(np.clip((z + 1.0) / 2.0, _BCE_EPS, 1.0 - _BCE_EPS))
    loss_value = -(y * np.log(probability) + (1.0 - y) * np.log(1.0 - probability))
    dloss_dp = -y / probability + (1.0 - y) / (1.0 - probability)
    return float(loss_value), 0.5 * float(dloss_dp)


def terminal_costate(final_state: MPS, readout: KrotovReadout, y: float, bias: float) -> tuple[MPS, float, float]:
    """Construct the terminal costate ``|chi_M>`` for one sample.

    Args:
        final_state: Final circuit state ``|psi_M>``.
        readout: Readout and loss specification.
        y: Sample label.
        bias: Scalar bias of the prediction.

    Returns:
        A tuple ``(chi, loss_value, z)`` with the (unnormalized) costate MPS, the
        sample loss, and the readout expectation value.
    """
    z = _expectation(final_state, readout.observable)
    loss_value, factor = _loss_and_costate_factor(z, bias, y, readout.loss)
    chi = copy.deepcopy(final_state)
    chi.apply_local(readout.observable)
    chi.tensors[0] = chi.tensors[0] * factor
    return chi, loss_value, z


def backward_costates(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    x: NDArray[np.float64],
    chi_terminal: MPS,
    truncation: KrotovTruncation,
) -> list[MPS]:
    """Propagate the terminal costate backward through the adjoint circuit.

    Args:
        circuit: The parameterized circuit.
        theta: Trainable parameter vector used for the adjoint gates.
        x: Input sample.
        chi_terminal: Terminal costate ``|chi_M>``.
        truncation: Truncation settings for gate applications.

    Returns:
        Costates ``[|chi_0>, ..., |chi_M>]`` with ``|chi_k> = U_{k+1}^† ... U_M^† |chi_M>``.
    """
    num_gates = len(circuit.gates)
    costates: list[MPS] = [chi_terminal] * (num_gates + 1)
    costates[num_gates] = chi_terminal
    for k in range(num_gates - 1, -1, -1):
        chi = copy.deepcopy(costates[k + 1])
        matrix, sites = circuit.gate_matrix(circuit.gates[k], theta, x)
        _apply_operator(chi, matrix.conj().T, sites, truncation)
        costates[k] = chi
    return costates


def _gate_contribution(
    circuit: ParameterizedCircuit,
    gate: ParameterizedGate,
    chi_after: MPS,
    psi_after: MPS,
    truncation: KrotovTruncation,
) -> float:
    """Evaluate the gate-local Krotov contribution ``2 Re <chi | D | psi>``.

    Args:
        circuit: The parameterized circuit.
        gate: The trainable circuit factor.
        chi_after: Costate after gate ``k``.
        psi_after: Forward state after gate ``k``.
        truncation: Truncation settings for applying the derivative operator.

    Returns:
        The contribution ``(d alpha / d theta) * 2 Re <chi | dU/d(alpha) U^{-1} | psi>``.
    """
    operator, sites = circuit.derivative_operator(gate)
    perturbed = copy.deepcopy(psi_after)
    _apply_operator(perturbed, operator, sites, truncation)
    overlap = chi_after.scalar_product(perturbed)
    return gate.angle_scale * 2.0 * float(np.real(overlap))


def sample_contribution(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    x: NDArray[np.float64],
    y: float,
    readout: KrotovReadout,
    bias: float,
    initial_state: MPS,
    truncation: KrotovTruncation,
) -> tuple[NDArray[np.float64], float, float]:
    """Compute the gate-wise Krotov contribution of one sample.

    For fixed parameters this equals the exact gradient of the sample loss with
    respect to the gate-supported parameters. Only one costate is kept in memory;
    it is propagated backward while contributions are collected.

    Args:
        circuit: The parameterized circuit.
        theta: Trainable parameter vector.
        x: Input sample.
        y: Sample label.
        readout: Readout and loss specification.
        bias: Scalar bias of the prediction.
        initial_state: Circuit input state for this sample (consumed).
        truncation: Truncation settings for gate applications.

    Returns:
        A tuple ``(contribution, loss_value, z)``.
    """
    states = forward_states(circuit, theta, x, initial_state, truncation)
    chi, loss_value, z = terminal_costate(states[-1], readout, y, bias)

    contribution = np.zeros(circuit.num_params, dtype=np.float64)
    for k in range(len(circuit.gates) - 1, -1, -1):
        gate = circuit.gates[k]
        if gate.param_index is not None:
            contribution[gate.param_index] += _gate_contribution(circuit, gate, chi, states[k + 1], truncation)
        if k > 0:
            matrix, sites = circuit.gate_matrix(gate, theta, x)
            _apply_operator(chi, matrix.conj().T, sites, truncation)
    return contribution, loss_value, z


def _batch_quantities(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    inputs: NDArray[np.float64],
    labels: NDArray[np.float64],
    readout: KrotovReadout,
    bias: float,
    initial_state: MPS | StatePrep | None,
    truncation: KrotovTruncation,
) -> tuple[NDArray[np.float64], float, float]:
    """Average sample contributions, the bias gradient, and the loss over a batch.

    Args:
        circuit: The parameterized circuit.
        theta: Trainable parameter vector.
        inputs: Batch inputs of shape ``(N, ...)``.
        labels: Batch labels of shape ``(N,)``.
        readout: Readout and loss specification.
        bias: Scalar bias of the prediction.
        initial_state: Circuit input state specification.
        truncation: Truncation settings for gate applications.

    Returns:
        A tuple ``(mean_contribution, bias_gradient, mean_loss)``.
    """
    mean_contribution = np.zeros(circuit.num_params, dtype=np.float64)
    bias_gradient = 0.0
    mean_loss = 0.0
    for x, y in zip(inputs, labels):
        state = _resolve_initial_state(initial_state, x, circuit.num_qubits)
        contribution, loss_value, z = sample_contribution(
            circuit, theta, x, float(y), readout, bias, state, truncation
        )
        mean_contribution += contribution
        mean_loss += loss_value
        if readout.use_bias:
            bias_gradient += 2.0 * (z + bias - float(y))
    num_samples = len(inputs)
    return mean_contribution / num_samples, bias_gradient / num_samples, mean_loss / num_samples


def empirical_loss(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    inputs: NDArray[np.float64],
    labels: NDArray[np.float64],
    readout: KrotovReadout,
    bias: float = 0.0,
    initial_state: MPS | StatePrep | None = None,
    truncation: KrotovTruncation | None = None,
) -> float:
    """Evaluate the empirical loss of the circuit on a dataset.

    Args:
        circuit: The parameterized circuit.
        theta: Trainable parameter vector.
        inputs: Batch inputs of shape ``(N, ...)``.
        labels: Batch labels of shape ``(N,)``.
        readout: Readout and loss specification.
        bias: Scalar bias of the prediction.
        initial_state: Circuit input state specification.
        truncation: Truncation settings; defaults to exact application.

    Returns:
        The mean sample loss.
    """
    truncation = truncation if truncation is not None else KrotovTruncation()
    total = 0.0
    for x, y in zip(inputs, labels):
        state = _resolve_initial_state(initial_state, x, circuit.num_qubits)
        for gate in circuit.gates:
            matrix, sites = circuit.gate_matrix(gate, theta, x)
            _apply_operator(state, matrix, sites, truncation)
        z = _expectation(state, readout.observable)
        loss_value, _ = _loss_and_costate_factor(z, bias, float(y), readout.loss)
        total += loss_value
    return total / len(inputs)


def _online_sample_update(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    x: NDArray[np.float64],
    y: float,
    readout: KrotovReadout,
    bias: float,
    step_size: float,
    initial_state: MPS,
    truncation: KrotovTruncation,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply one stale-adjoint online Krotov sweep for a single sample.

    The backward costates are computed once at the pre-sweep parameters and kept
    fixed, while the forward state is refreshed gate by gate with the already
    updated parameters.

    Args:
        circuit: The parameterized circuit.
        theta: Pre-sweep trainable parameter vector (not modified).
        x: Input sample.
        y: Sample label.
        readout: Readout and loss specification.
        bias: Scalar bias of the prediction.
        step_size: Online step size.
        initial_state: Circuit input state for this sample (consumed).
        truncation: Truncation settings for gate applications.

    Returns:
        A tuple ``(new_theta, contribution)`` with the post-sweep parameters and the
        gate-local signals observed during the sweep.
    """
    states = forward_states(circuit, theta, x, initial_state, truncation)
    chi_terminal_state, _, _ = terminal_costate(states[-1], readout, y, bias)
    costates = backward_costates(circuit, theta, x, chi_terminal_state, truncation)

    new_theta = theta.copy()
    current = states[0]
    contribution = np.zeros(circuit.num_params, dtype=np.float64)

    for k, gate in enumerate(circuit.gates):
        if gate.param_index is not None:
            gate_output = copy.deepcopy(current)
            matrix, sites = circuit.gate_matrix(gate, new_theta, x)
            _apply_operator(gate_output, matrix, sites, truncation)
            signal = _gate_contribution(circuit, gate, costates[k + 1], gate_output, truncation)
            contribution[gate.param_index] += signal
            new_theta[gate.param_index] -= step_size * signal
        matrix, sites = circuit.gate_matrix(gate, new_theta, x)
        _apply_operator(current, matrix, sites, truncation)

    return new_theta, contribution


def _init_trace() -> dict[str, list[float | int | str]]:
    """Create an empty training trace.

    Returns:
        Trace dictionary with empty per-iteration records.
    """
    return {"step": [], "phase": [], "loss": [], "step_size": [], "gradient_norm": [], "update_norm": []}


def _record(
    trace: dict[str, list[float | int | str]],
    step: int,
    phase: str,
    loss: float,
    step_size: float,
    gradient_norm: float,
    update_norm: float,
) -> None:
    """Append one iteration record to the training trace.

    Args:
        trace: Training trace.
        step: Iteration counter.
        phase: Phase label (``"init"``, ``"online"``, or ``"batch"``).
        loss: Empirical loss after the iteration.
        step_size: Step size used in the iteration.
        gradient_norm: Norm of the (mean) update direction.
        update_norm: Norm of the parameter change.
    """
    trace["step"].append(step)
    trace["phase"].append(phase)
    trace["loss"].append(loss)
    trace["step_size"].append(step_size)
    trace["gradient_norm"].append(gradient_norm)
    trace["update_norm"].append(update_norm)


def _online_epoch(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    bias: float,
    inputs: NDArray[np.float64],
    labels: NDArray[np.float64],
    readout: KrotovReadout,
    step_size: float,
    epoch_seed: int,
    initial_state: MPS | StatePrep | None,
    truncation: KrotovTruncation,
) -> tuple[NDArray[np.float64], float, float]:
    """Run one online epoch: a stale-adjoint sweep per sample in random order.

    Args:
        circuit: The parameterized circuit.
        theta: Trainable parameter vector.
        bias: Scalar bias of the prediction.
        inputs: Training inputs.
        labels: Training labels.
        readout: Readout and loss specification.
        step_size: Online step size.
        epoch_seed: Seed for the sample permutation.
        initial_state: Circuit input state specification.
        truncation: Truncation settings for gate applications.

    Returns:
        A tuple ``(theta, bias, gradient_norm)`` after the epoch.
    """
    rng = np.random.default_rng(epoch_seed)
    permutation = rng.permutation(len(inputs))
    contributions = []
    for idx in permutation:
        state = _resolve_initial_state(initial_state, inputs[idx], circuit.num_qubits)
        theta, contribution = _online_sample_update(
            circuit, theta, inputs[idx], float(labels[idx]), readout, bias, step_size, state, truncation
        )
        contributions.append(contribution)

    bias_gradient = 0.0
    if readout.use_bias:
        for x, y in zip(inputs, labels):
            state = _resolve_initial_state(initial_state, x, circuit.num_qubits)
            for gate in circuit.gates:
                matrix, sites = circuit.gate_matrix(gate, theta, x)
                _apply_operator(state, matrix, sites, truncation)
            z = _expectation(state, readout.observable)
            bias_gradient += 2.0 * (z + bias - float(y))
        bias_gradient /= len(inputs)
        bias -= step_size * bias_gradient

    mean_contribution = np.mean(np.asarray(contributions), axis=0)
    gradient_norm = float(np.sqrt(np.linalg.norm(mean_contribution) ** 2 + bias_gradient**2))
    return theta, bias, gradient_norm


def _batch_epoch(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    bias: float,
    inputs: NDArray[np.float64],
    labels: NDArray[np.float64],
    readout: KrotovReadout,
    step_size: float,
    initial_state: MPS | StatePrep | None,
    truncation: KrotovTruncation,
) -> tuple[NDArray[np.float64], float, float, float]:
    """Run one batch epoch: a single full-batch (exact-gradient) update.

    Args:
        circuit: The parameterized circuit.
        theta: Trainable parameter vector.
        bias: Scalar bias of the prediction.
        inputs: Training inputs.
        labels: Training labels.
        readout: Readout and loss specification.
        step_size: Batch step size.
        initial_state: Circuit input state specification.
        truncation: Truncation settings for gate applications.

    Returns:
        A tuple ``(theta, bias, gradient_norm, pre_update_loss)`` after the update.
    """
    mean_contribution, bias_gradient, mean_loss = _batch_quantities(
        circuit, theta, inputs, labels, readout, bias, initial_state, truncation
    )
    theta = theta - step_size * mean_contribution
    if readout.use_bias:
        bias -= step_size * bias_gradient
    gradient_norm = float(np.sqrt(np.linalg.norm(mean_contribution) ** 2 + bias_gradient**2))
    return theta, bias, gradient_norm, mean_loss


def _train(
    circuit: ParameterizedCircuit,
    readout: KrotovReadout,
    inputs: NDArray[np.float64],
    labels: NDArray[np.float64],
    initial_theta: NDArray[np.float64],
    options: KrotovOptions,
    initial_state: MPS | StatePrep | None,
    initial_bias: float,
    switch_iteration: int,
) -> KrotovResult:
    """Shared training loop for the online, batch, and hybrid variants.

    Args:
        circuit: The parameterized circuit.
        readout: Readout and loss specification.
        inputs: Training inputs of shape ``(N, ...)``.
        labels: Training labels of shape ``(N,)``.
        initial_theta: Initial trainable parameter vector.
        options: Training hyperparameters.
        initial_state: Circuit input state specification.
        initial_bias: Initial scalar bias.
        switch_iteration: Last online iteration (``0`` for pure batch, ``>= max_iterations``
            for pure online).

    Returns:
        The training result with final parameters and trace.
    """
    inputs = np.asarray(inputs)
    labels = np.asarray(labels, dtype=np.float64)
    theta = np.asarray(initial_theta, dtype=np.float64).copy()
    bias = float(initial_bias)
    truncation = options.truncation

    trace = _init_trace()
    loss = empirical_loss(circuit, theta, inputs, labels, readout, bias, initial_state, truncation)
    _record(trace, 0, "init", loss, 0.0, 0.0, 0.0)

    for iteration in range(1, options.max_iterations + 1):
        theta_before = theta.copy()
        bias_before = bias
        if iteration <= switch_iteration:
            step = _step_size(options.online_step_size, iteration, options.online_schedule, options.online_decay)
            theta, bias, gradient_norm = _online_epoch(
                circuit,
                theta,
                bias,
                inputs,
                labels,
                readout,
                step,
                options.seed + iteration,
                initial_state,
                truncation,
            )
            phase = "online"
        else:
            phase_iteration = iteration - switch_iteration
            step = _step_size(options.batch_step_size, phase_iteration, options.batch_schedule, options.batch_decay)
            theta, bias, gradient_norm, _ = _batch_epoch(
                circuit, theta, bias, inputs, labels, readout, step, initial_state, truncation
            )
            phase = "batch"

        loss = empirical_loss(circuit, theta, inputs, labels, readout, bias, initial_state, truncation)
        update_norm = float(np.sqrt(np.linalg.norm(theta - theta_before) ** 2 + (bias - bias_before) ** 2))
        _record(trace, iteration, phase, loss, step, gradient_norm, update_norm)

    return KrotovResult(theta=theta, bias=bias, trace=trace)


def train_krotov_online(
    circuit: ParameterizedCircuit,
    readout: KrotovReadout,
    inputs: NDArray[np.float64],
    labels: NDArray[np.float64],
    *,
    initial_theta: NDArray[np.float64],
    options: KrotovOptions | None = None,
    initial_state: MPS | StatePrep | None = None,
    initial_bias: float = 0.0,
) -> KrotovResult:
    """Train with sample-wise sequential stale-adjoint Krotov sweeps.

    Args:
        circuit: The parameterized circuit.
        readout: Readout and loss specification.
        inputs: Training inputs of shape ``(N, ...)``.
        labels: Training labels of shape ``(N,)``.
        initial_theta: Initial trainable parameter vector.
        options: Training hyperparameters; defaults to :class:`KrotovOptions`.
        initial_state: Fixed input MPS, per-sample state-preparation callable, or
            ``None`` for the all-zeros state.
        initial_bias: Initial scalar bias (``"mse"`` with ``use_bias`` only).

    Returns:
        The training result with final parameters and trace.
    """
    options = options if options is not None else KrotovOptions()
    return _train(
        circuit,
        readout,
        inputs,
        labels,
        initial_theta,
        options,
        initial_state,
        initial_bias,
        switch_iteration=options.max_iterations,
    )


def train_krotov_batch(
    circuit: ParameterizedCircuit,
    readout: KrotovReadout,
    inputs: NDArray[np.float64],
    labels: NDArray[np.float64],
    *,
    initial_theta: NDArray[np.float64],
    options: KrotovOptions | None = None,
    initial_state: MPS | StatePrep | None = None,
    initial_bias: float = 0.0,
) -> KrotovResult:
    """Train with full-batch Krotov updates (exact empirical-loss gradient descent).

    Args:
        circuit: The parameterized circuit.
        readout: Readout and loss specification.
        inputs: Training inputs of shape ``(N, ...)``.
        labels: Training labels of shape ``(N,)``.
        initial_theta: Initial trainable parameter vector.
        options: Training hyperparameters; defaults to :class:`KrotovOptions`.
        initial_state: Fixed input MPS, per-sample state-preparation callable, or
            ``None`` for the all-zeros state.
        initial_bias: Initial scalar bias (``"mse"`` with ``use_bias`` only).

    Returns:
        The training result with final parameters and trace.
    """
    options = options if options is not None else KrotovOptions()
    return _train(
        circuit,
        readout,
        inputs,
        labels,
        initial_theta,
        options,
        initial_state,
        initial_bias,
        switch_iteration=0,
    )


def train_krotov_hybrid(
    circuit: ParameterizedCircuit,
    readout: KrotovReadout,
    inputs: NDArray[np.float64],
    labels: NDArray[np.float64],
    *,
    initial_theta: NDArray[np.float64],
    options: KrotovOptions | None = None,
    initial_state: MPS | StatePrep | None = None,
    initial_bias: float = 0.0,
) -> KrotovResult:
    """Train with an online-to-batch hybrid Krotov schedule.

    Online stale-adjoint sweeps are used for the first ``options.switch_iteration``
    iterations, after which the method switches to full-batch updates.

    Args:
        circuit: The parameterized circuit.
        readout: Readout and loss specification.
        inputs: Training inputs of shape ``(N, ...)``.
        labels: Training labels of shape ``(N,)``.
        initial_theta: Initial trainable parameter vector.
        options: Training hyperparameters; defaults to :class:`KrotovOptions`.
        initial_state: Fixed input MPS, per-sample state-preparation callable, or
            ``None`` for the all-zeros state.
        initial_bias: Initial scalar bias (``"mse"`` with ``use_bias`` only).

    Returns:
        The training result with final parameters and trace.
    """
    options = options if options is not None else KrotovOptions()
    return _train(
        circuit,
        readout,
        inputs,
        labels,
        initial_theta,
        options,
        initial_state,
        initial_bias,
        switch_iteration=options.switch_iteration,
    )


__all__ = [
    "KrotovOptions",
    "KrotovReadout",
    "KrotovResult",
    "KrotovTruncation",
    "backward_costates",
    "empirical_loss",
    "forward_states",
    "sample_contribution",
    "terminal_costate",
    "train_krotov_batch",
    "train_krotov_hybrid",
    "train_krotov_online",
]
