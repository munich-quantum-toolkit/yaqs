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
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import opt_einsum as oe

from ..core import linalg
from ..core.data_structures.mps import MPS
from ..core.data_structures.noise_model import NoiseModel
from ..core.methods.decompositions import merge_two_site, split_two_site

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ..core.data_structures.simulation_parameters import Observable
    from ..core.methods.decompositions import TruncMode
    from .parameterized_circuit import ParameterizedCircuit, ParameterizedGate

    StatePrep = Callable[[NDArray[np.float64]], MPS]
    TargetState = MPS | NDArray[np.complex128]

_BCE_EPS = 1e-7

_SWAP_MATRIX = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    dtype=np.complex128,
)

LossKind = Literal["mse", "bce"]
ScheduleKind = Literal["constant", "inverse", "exp"]
NoiseApplicationMode = Literal["all", "two-qubit"]
TrajectoryUpdateMode = Literal["independent", "cross"]


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


@dataclass(frozen=True)
class KrotovTJMOptions:
    """Trajectory settings for noisy circuit-TJM Krotov optimization.

    The noisy optimizer treats YAQS circuit TJM as an ensemble of pure MPS
    trajectories. During one gradient evaluation the realized trajectory maps are
    fixed and replayed in the backward sweep. This is a pathwise-gradient
    approximation: jump probabilities, MPS canonicalization, and truncation
    choices are not differentiated.

    Attributes:
        num_trajectories: Number of pure MPS trajectories.
        random_seed: Base seed for reproducible trajectory realizations.
        dt: Effective circuit-TJM time step after each noisy gate.
        apply_noise_to: Whether to apply local noise after every gate or only
            after two-qubit gates.
        noisy_gate_indices: Optional explicit gate indices where noise may be
            applied. This is useful when one logical noisy gate is decomposed
            into several optimizer primitives and noise should be sampled only
            once for the logical gate.
        trajectory_update: Whether state-preparation updates use independent
            trajectory pairings or the cross-trajectory density-matrix expansion
            from Goerz and Jacobs, Quantum Sci. Technol. 3 045005 (2018).
        differentiate_jump_normalization: Reserved for a future exact normalized
            jump pullback. The current implementation must keep this ``False``.
    """

    num_trajectories: int = 1
    random_seed: int = 0
    dt: float = 1.0
    apply_noise_to: NoiseApplicationMode = "all"
    noisy_gate_indices: tuple[int, ...] | None = None
    trajectory_update: TrajectoryUpdateMode = "independent"
    differentiate_jump_normalization: bool = False
    use_crn: bool = False

    def __post_init__(self) -> None:
        """Validate trajectory options.

        Raises:
            ValueError: If an option is outside the supported first-implementation range.
        """
        if self.num_trajectories < 1:
            msg = "num_trajectories must be at least 1."
            raise ValueError(msg)
        if self.dt <= 0.0 or not np.isfinite(self.dt):
            msg = f"dt must be a positive finite float, got {self.dt!r}."
            raise ValueError(msg)
        if self.apply_noise_to not in {"all", "two-qubit"}:
            msg = f"Unknown noise application mode: {self.apply_noise_to!r}."
            raise ValueError(msg)
        if self.noisy_gate_indices is not None and any(index < 0 for index in self.noisy_gate_indices):
            msg = f"noisy_gate_indices must be nonnegative, got {self.noisy_gate_indices!r}."
            raise ValueError(msg)
        if self.trajectory_update not in {"independent", "cross"}:
            msg = f"Unknown trajectory update mode: {self.trajectory_update!r}."
            raise ValueError(msg)
        if self.differentiate_jump_normalization:
            msg = "Exact normalized-jump pullbacks are not implemented; keep differentiate_jump_normalization=False."
            raise ValueError(msg)


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


@dataclass(frozen=True)
class KrotovNoiseMap:
    """Realized fixed circuit-TJM map after one circuit gate.

    The map is represented as a sequence of local linear operators applied to an
    MPS in order. If ``normalized`` is true, the forward trajectory normalized the
    state after the realized noise map. The first implementation uses the
    documented pathwise physical-operator approximation in the backward pass: it
    replays the adjoint local operators and ignores the derivative of this
    normalization.

    Attributes:
        operators: Local operators in forward application order.
        normalized: Whether the forward state was normalized after the operators.
        jump_process_index: Index of the sampled jump process in the local noise
            model, or ``None`` for a no-jump realization.
    """

    operators: tuple[tuple[NDArray[np.complex128], tuple[int, ...]], ...] = ()
    normalized: bool = False
    jump_process_index: int | None = None


@dataclass
class KrotovTrajectory:
    """Forward storage for one fixed circuit-TJM trajectory.

    Attributes:
        states: States after each gate and its realized noise map, including the
            initial state at index 0.
        gate_outputs: States immediately after each unitary circuit gate and
            before the realized noise map.
        noise_maps: Realized maps after each circuit gate.
    """

    states: list[MPS]
    gate_outputs: list[MPS]
    noise_maps: list[KrotovNoiseMap]


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
    state.tensors[site] = np.asarray(oe.contract("ab,bcd->acd", matrix, state.tensors[site]), dtype=np.complex128)


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


def _noise_model_is_trivial(noise_model: NoiseModel | None) -> bool:
    """Return whether a noise model has no positive-strength processes."""
    return noise_model is None or all(np.isclose(float(proc["strength"]), 0.0) for proc in noise_model.processes)


def _process_is_pauli(process: dict[str, Any]) -> bool:
    """Return whether a process has Pauli jump operators with ``J^dag J = I``."""
    return str(process["name"]) in {
        "pauli_x",
        "pauli_y",
        "pauli_z",
        "crosstalk_xx",
        "crosstalk_yy",
        "crosstalk_zz",
        "crosstalk_xy",
        "crosstalk_yx",
        "crosstalk_zy",
        "crosstalk_zx",
        "crosstalk_yz",
        "crosstalk_xz",
        "longrange_crosstalk_xx",
        "longrange_crosstalk_yy",
        "longrange_crosstalk_zz",
        "longrange_crosstalk_xy",
        "longrange_crosstalk_yx",
        "longrange_crosstalk_zy",
        "longrange_crosstalk_zx",
        "longrange_crosstalk_yz",
        "longrange_crosstalk_xz",
    }


def _process_sites(process: dict[str, Any]) -> tuple[int, ...]:
    """Return sorted process sites."""
    return tuple(int(site) for site in process["sites"])


def _local_noise_model(noise_model: NoiseModel | None, gate_sites: tuple[int, ...]) -> NoiseModel | None:
    """Restrict a global noise model to processes local to a circuit gate.

    Returns:
        Local noise model containing only processes supported on the gate sites,
        or ``None`` if no global noise model was supplied.
    """
    if noise_model is None:
        return None
    gate_site_set = set(gate_sites)
    local_processes = [
        copy.deepcopy(process)
        for process in noise_model.processes
        if set(_process_sites(process)).issubset(gate_site_set)
    ]
    return NoiseModel(local_processes)


def _jump_operator(process: dict[str, Any]) -> tuple[NDArray[np.complex128], tuple[int, ...]]:
    """Return the jump operator matrix and sites for one noise process.

    Raises:
        NotImplementedError: If a long-range non-Pauli process has no dense matrix.
    """
    sites = _process_sites(process)
    if len(sites) == 1 or "matrix" in process:
        return np.asarray(process["matrix"], dtype=np.complex128), sites
    if "factors" in process:
        factors = process["factors"]
        return np.asarray(np.kron(factors[0], factors[1]), dtype=np.complex128), sites
    msg = f"Noise process {process['name']!r} has no usable local matrix or factors."
    raise NotImplementedError(msg)


def _dissipation_operator(process: dict[str, Any], dt: float) -> tuple[NDArray[np.complex128], tuple[int, ...]]:
    """Return the linear no-jump drift operator for one noise process."""
    sites = _process_sites(process)
    gamma = float(process["strength"])
    dim = 2 ** len(sites)
    if _process_is_pauli(process):
        return (np.exp(-0.5 * dt * gamma) * np.eye(dim, dtype=np.complex128)).astype(np.complex128), sites

    jump_op, _ = _jump_operator(process)
    mat = jump_op.conj().T @ jump_op
    return np.asarray(linalg.expm(-0.5 * dt * gamma * mat), dtype=np.complex128), sites


def _apply_noise_map(state: MPS, noise_map: KrotovNoiseMap, truncation: KrotovTruncation) -> None:
    """Apply a realized circuit-TJM map to an MPS in place."""
    for matrix, sites in noise_map.operators:
        _apply_operator(state, matrix, sites, truncation)
    if noise_map.normalized:
        _normalize_preserving_phase(state)


def _pullback_noise_map(costate: MPS, noise_map: KrotovNoiseMap, truncation: KrotovTruncation) -> None:
    """Apply the adjoint of a realized circuit-TJM map to a costate in place.

    This is the fixed-trajectory physical-operator approximation: if the forward
    trajectory normalized after the realized noise map, the derivative of that
    normalization is ignored here. MPS canonicalization and truncation are also
    treated as representation choices rather than differentiable maps.
    """
    for matrix, sites in reversed(noise_map.operators):
        _apply_operator(costate, matrix.conj().T, sites, truncation)


def _state_norm(state: MPS) -> float:
    """Return the real full-network norm of an MPS."""
    return max(0.0, float(np.real(state.scalar_product(state))))


def _normalize_preserving_phase(state: MPS) -> None:
    """Normalize an MPS without changing its represented global phase.

    MPS SVD canonicalization may choose singular-vector signs/gauges
    arbitrarily. For Krotov adjoints, that representation phase must stay
    consistent with the pre-normalized trajectory state; otherwise terminal
    costates and stored forward gate outputs can acquire incompatible phases.
    """
    reference = copy.deepcopy(state)
    state.normalize("B", decomposition="SVD")
    overlap = reference.scalar_product(state)
    if abs(overlap) > 0.0:
        state.tensors[0] *= np.conj(overlap) / abs(overlap)


def _jump_probability_weights(
    state: MPS,
    local_noise_model: NoiseModel,
    dt: float,
    truncation: KrotovTruncation,
) -> list[float]:
    """Compute unnormalized jump probabilities for a post-drift MPS.

    Returns:
        Unnormalized jump probabilities in ``local_noise_model.processes`` order.
    """
    weights = []
    for process in local_noise_model.processes:
        gamma = float(process["strength"])
        if gamma <= 0.0:
            weights.append(0.0)
            continue
        jump_op, sites = _jump_operator(process)
        jumped = copy.deepcopy(state)
        _apply_operator(jumped, jump_op, sites, truncation)
        weights.append(dt * gamma * _state_norm(jumped))
    return weights


def _sample_noise_map_and_apply(
    state: MPS,
    local_noise_model: NoiseModel | None,
    truncation: KrotovTruncation,
    tjm_options: KrotovTJMOptions,
    rng: np.random.Generator,
) -> KrotovNoiseMap:
    """Sample a fixed circuit-TJM map and apply it to ``state`` in place.

    Returns:
        Realized noise map applied to ``state``.
    """
    if _noise_model_is_trivial(local_noise_model):
        return KrotovNoiseMap()

    assert local_noise_model is not None
    operators: list[tuple[NDArray[np.complex128], tuple[int, ...]]] = []
    positive_processes = [process for process in local_noise_model.processes if float(process["strength"]) > 0.0]
    pauli_only = all(_process_is_pauli(process) for process in positive_processes)
    for process in local_noise_model.processes:
        if float(process["strength"]) > 0.0:
            matrix, sites = _dissipation_operator(process, tjm_options.dt)
            operators.append((matrix, sites))
            _apply_operator(state, matrix, sites, truncation)

    jump_probability = float(np.clip(1.0 - _state_norm(state), 0.0, 1.0))
    if rng.random() >= jump_probability:
        _normalize_preserving_phase(state)
        if pauli_only:
            return KrotovNoiseMap()
        return KrotovNoiseMap(operators=tuple(operators), normalized=True)

    weights = _jump_probability_weights(state, local_noise_model, tjm_options.dt, truncation)
    total = float(np.sum(weights))
    if total <= 0.0:
        _normalize_preserving_phase(state)
        if pauli_only:
            return KrotovNoiseMap()
        return KrotovNoiseMap(operators=tuple(operators), normalized=True)

    probabilities = np.asarray(weights, dtype=np.float64) / total
    jump_index = int(rng.choice(len(local_noise_model.processes), p=probabilities))
    jump_op, sites = _jump_operator(local_noise_model.processes[jump_index])
    operators.append((jump_op, sites))
    _apply_operator(state, jump_op, sites, truncation)
    _normalize_preserving_phase(state)
    if pauli_only:
        return KrotovNoiseMap(operators=((jump_op, sites),), jump_process_index=jump_index)
    return KrotovNoiseMap(operators=tuple(operators), normalized=True, jump_process_index=jump_index)


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


def _mps_from_statevector(vector: NDArray[np.complex128]) -> MPS:
    """Convert a dense qubit statevector into an exact normalized MPS.

    Args:
        vector: One-dimensional dense statevector in YAQS/Qiskit little-endian basis order.

    Returns:
        An exact MPS representation of the normalized statevector.

    Raises:
        ValueError: If the vector is not one-dimensional, has non-qubit dimension, or is zero.
    """
    dense = np.asarray(vector, dtype=np.complex128)
    if dense.ndim != 1:
        msg = f"Target statevector must be one-dimensional, got shape {dense.shape}."
        raise ValueError(msg)
    if dense.size < 2:
        msg = "Target statevector must contain at least one qubit."
        raise ValueError(msg)

    num_qubits = int(np.log2(dense.size))
    if 2**num_qubits != dense.size:
        msg = f"Target statevector length must be a power of two, got {dense.size}."
        raise ValueError(msg)

    norm = float(np.linalg.norm(dense))
    if norm <= 0.0 or not np.isfinite(norm):
        msg = "Target statevector must have nonzero finite norm."
        raise ValueError(msg)

    current = (dense / norm).reshape([2] * num_qubits)
    left_bond = 1
    left_to_right_tensors: list[NDArray[np.complex128]] = []
    for _site in range(num_qubits - 1):
        matrix = current.reshape(left_bond * 2, -1)
        u_matrix, singular_values, vh_matrix = np.linalg.svd(matrix, full_matrices=False)
        right_bond = len(singular_values)
        left_to_right_tensors.append(
            np.asarray(u_matrix.reshape(left_bond, 2, right_bond).transpose(1, 0, 2), dtype=np.complex128)
        )
        current = np.diag(singular_values) @ vh_matrix
        left_bond = right_bond

    left_to_right_tensors.append(np.asarray(current.reshape(left_bond, 2, 1).transpose(1, 0, 2), dtype=np.complex128))

    # YAQS stores site 0 as the least significant qubit, while the left-to-right
    # SVD above factors the dense tensor from the most significant axis.
    tensors = [tensor.transpose(0, 2, 1) for tensor in reversed(left_to_right_tensors)]
    return MPS(num_qubits, tensors=tensors)


def _normalized_mps_copy(state: MPS) -> MPS:
    """Return a normalized deep copy of an MPS.

    Args:
        state: Input MPS.

    Returns:
        A fresh normalized MPS.

    Raises:
        ValueError: If the MPS norm is zero or not finite.
    """
    norm_sq = state.scalar_product(state)
    if abs(float(np.imag(norm_sq))) > 1e-10 * max(1.0, abs(float(np.real(norm_sq)))):
        msg = f"Target MPS norm has a non-negligible imaginary part: {norm_sq}."
        raise ValueError(msg)
    norm = float(np.sqrt(float(np.real(norm_sq))))
    if norm <= 0.0 or not np.isfinite(norm):
        msg = "Target MPS must have nonzero finite norm."
        raise ValueError(msg)

    normalized = copy.deepcopy(state)
    normalized.tensors[0] /= norm
    return normalized


def _resolve_target_state(target_state: TargetState, num_qubits: int) -> MPS:
    """Normalize and validate a state-preparation target.

    Args:
        target_state: Target MPS or dense statevector.
        num_qubits: Required number of qubits.

    Returns:
        A normalized target MPS.

    Raises:
        ValueError: If the target length is incompatible with the circuit.
    """
    target = (
        _normalized_mps_copy(target_state) if isinstance(target_state, MPS) else _mps_from_statevector(target_state)
    )
    if target.length != num_qubits:
        msg = f"Target state has {target.length} qubits, but the circuit has {num_qubits}."
        raise ValueError(msg)
    return target


def _resolve_fixed_initial_state(initial_state: MPS | None, num_qubits: int) -> MPS:
    """Build a fresh fixed input state for state preparation.

    Args:
        initial_state: Fixed input MPS, or ``None`` for the all-zeros state.
        num_qubits: Number of qubits.

    Returns:
        A fresh MPS instance.

    Raises:
        ValueError: If the fixed state length is incompatible with the circuit.
    """
    if initial_state is not None and initial_state.length != num_qubits:
        msg = f"Initial state has {initial_state.length} qubits, but the circuit has {num_qubits}."
        raise ValueError(msg)
    return copy.deepcopy(initial_state) if initial_state is not None else MPS(num_qubits)


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


def _should_apply_noise(gate_index: int, gate_sites: tuple[int, ...], tjm_options: KrotovTJMOptions) -> bool:
    """Return whether local TJM noise should be applied after a gate."""
    if tjm_options.noisy_gate_indices is not None and gate_index not in tjm_options.noisy_gate_indices:
        return False
    return tjm_options.apply_noise_to == "all" or len(gate_sites) == 2


def forward_tjm_trajectory(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    x: NDArray[np.float64],
    initial_state: MPS,
    truncation: KrotovTruncation,
    noise_model: NoiseModel | None,
    tjm_options: KrotovTJMOptions,
    rng: np.random.Generator,
    noise_maps: list[KrotovNoiseMap] | None = None,
) -> KrotovTrajectory:
    """Propagate one fixed circuit-TJM trajectory through a parameterized circuit.

    If ``noise_maps`` are supplied, they are replayed exactly. Otherwise new
    realized maps are sampled with ``rng``. The returned trajectory stores both
    post-gate/pre-noise states and post-noise states for the Krotov backward pass.

    Args:
        circuit: The parameterized circuit.
        theta: Trainable parameter vector.
        x: Input sample.
        initial_state: Circuit input state (consumed).
        truncation: Truncation settings for gate and noise applications.
        noise_model: Fixed YAQS noise model, or ``None`` for noiseless evolution.
        tjm_options: Circuit-TJM trajectory settings.
        rng: Random number generator used when sampling maps.
        noise_maps: Optional fixed realized maps to replay.

    Returns:
        Forward trajectory storage for this realization.

    Raises:
        ValueError: If a fixed map list does not match the circuit length.
    """
    if noise_maps is not None and len(noise_maps) != len(circuit.gates):
        msg = f"Expected {len(circuit.gates)} fixed noise maps, got {len(noise_maps)}."
        raise ValueError(msg)

    states = [initial_state]
    gate_outputs: list[MPS] = []
    realized_maps: list[KrotovNoiseMap] = []

    for gate_index, gate in enumerate(circuit.gates):
        phi = copy.deepcopy(states[-1])
        matrix, sites = circuit.gate_matrix(gate, theta, x)
        _apply_operator(phi, matrix, sites, truncation)
        gate_outputs.append(copy.deepcopy(phi))

        state_after_noise = copy.deepcopy(phi)
        if noise_maps is not None:
            noise_map = noise_maps[gate_index]
            _apply_noise_map(state_after_noise, noise_map, truncation)
        elif _should_apply_noise(gate_index, sites, tjm_options) and not _noise_model_is_trivial(noise_model):
            local_model = _local_noise_model(noise_model, sites)
            noise_map = _sample_noise_map_and_apply(state_after_noise, local_model, truncation, tjm_options, rng)
        else:
            noise_map = KrotovNoiseMap()
        realized_maps.append(noise_map)
        states.append(state_after_noise)

    return KrotovTrajectory(states=states, gate_outputs=gate_outputs, noise_maps=realized_maps)


def _trajectory_seed(base_seed: int, iteration: int, trajectory_index: int) -> int:
    """Build a reproducible seed for one trajectory and optimizer iteration.

    Returns:
        Integer seed.
    """
    return int(base_seed + 1_000_003 * iteration + trajectory_index)


def _forward_tjm_trajectories(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    x: NDArray[np.float64],
    initial_state: MPS,
    truncation: KrotovTruncation,
    noise_model: NoiseModel | None,
    tjm_options: KrotovTJMOptions,
    iteration: int = 0,
    fixed_noise_maps: list[list[KrotovNoiseMap]] | None = None,
) -> list[KrotovTrajectory]:
    """Generate or replay an ensemble of circuit-TJM trajectories.

    Returns:
        List of trajectory forward-storage objects.

    Raises:
        ValueError: If the fixed map list count does not match the trajectory count.
    """
    if fixed_noise_maps is not None and len(fixed_noise_maps) != tjm_options.num_trajectories:
        msg = f"Expected {tjm_options.num_trajectories} trajectory map lists, got {len(fixed_noise_maps)}."
        raise ValueError(msg)

    trajectories = []
    for traj_idx in range(tjm_options.num_trajectories):
        rng = np.random.default_rng(_trajectory_seed(tjm_options.random_seed, iteration, traj_idx))
        maps = None if fixed_noise_maps is None else fixed_noise_maps[traj_idx]
        trajectories.append(
            forward_tjm_trajectory(
                circuit,
                theta,
                x,
                copy.deepcopy(initial_state),
                truncation,
                noise_model,
                tjm_options,
                rng,
                maps,
            )
        )
    return trajectories


def _trajectory_contribution(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    x: NDArray[np.float64],
    trajectory: KrotovTrajectory,
    chi_terminal: MPS,
    truncation: KrotovTruncation,
) -> NDArray[np.float64]:
    """Backpropagate one trajectory costate and collect gate-local contributions.

    Returns:
        Gate-parameter contribution vector for this trajectory.
    """
    contribution = np.zeros(circuit.num_params, dtype=np.float64)
    lambda_after_noise = chi_terminal

    for k in range(len(circuit.gates) - 1, -1, -1):
        gate = circuit.gates[k]
        chi_tilde = copy.deepcopy(lambda_after_noise)
        _pullback_noise_map(chi_tilde, trajectory.noise_maps[k], truncation)
        if gate.param_index is not None:
            contribution[gate.param_index] += _gate_contribution(
                circuit,
                gate,
                chi_tilde,
                trajectory.gate_outputs[k],
                truncation,
            )
        matrix, sites = circuit.gate_matrix(gate, theta, x)
        _apply_operator(chi_tilde, matrix.conj().T, sites, truncation)
        lambda_after_noise = chi_tilde

    return contribution


def _cross_gate_contribution(
    circuit: ParameterizedCircuit,
    gate: ParameterizedGate,
    costates_after_noise: list[MPS],
    gate_outputs: list[MPS],
    truncation: KrotovTruncation,
) -> float:
    r"""Evaluate the gate-local cross-trajectory Krotov signal.

    For a trainable gate derivative ``D_k = alpha'_k (-i G_k / 2)`` this computes

    ``-(1/R^2) sum_rs 2 Re <xi_r|D_k|psi_s><psi_s|xi_r>``.

    The negative sign makes the returned signal a loss gradient contribution,
    matching :func:`_gate_contribution` for the independent-trajectory objective.

    Returns:
        Gate-local contribution for the trainable parameter attached to ``gate``.
    """
    if not costates_after_noise or not gate_outputs:
        return 0.0

    operator, sites = circuit.derivative_operator(gate)
    scale = 1.0 / (len(costates_after_noise) * len(gate_outputs))
    signal = 0.0
    for gate_output in gate_outputs:
        perturbed = copy.deepcopy(gate_output)
        _apply_operator(perturbed, operator, sites, truncation)
        for costate in costates_after_noise:
            derivative_overlap = costate.scalar_product(perturbed)
            density_overlap = gate_output.scalar_product(costate)
            signal -= gate.angle_scale * 2.0 * float(np.real(derivative_overlap * density_overlap)) * scale
    return signal


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
    chi.tensors[0] *= factor
    return chi, loss_value, z


def state_preparation_terminal_costate(final_state: MPS, target_state: TargetState) -> tuple[MPS, float, float]:
    """Construct the terminal costate for pure target-state preparation.

    The objective is the infidelity ``L = 1 - |<psi_target | psi_M>|^2``. Its
    terminal costate is ``|chi_M> = -<psi_target | psi_M> |psi_target>`` so that
    the existing gate-local contribution is the exact gradient of the infidelity.

    Args:
        final_state: Final circuit state ``|psi_M>``.
        target_state: Normalized or unnormalized target MPS/statevector.

    Returns:
        A tuple ``(chi, loss_value, fidelity)`` with the terminal costate MPS,
        the infidelity, and the fidelity.
    """
    target = _resolve_target_state(target_state, final_state.length)
    return _state_preparation_terminal_costate_from_mps(final_state, target)


def _state_preparation_terminal_costate_from_mps(final_state: MPS, target: MPS) -> tuple[MPS, float, float]:
    """Construct the state-preparation terminal costate for a resolved target MPS.

    Args:
        final_state: Final circuit state.
        target: Normalized target MPS with matching length.

    Returns:
        A tuple ``(chi, loss_value, fidelity)``.
    """
    overlap = target.scalar_product(final_state)
    fidelity = float(abs(overlap) ** 2)
    chi = copy.deepcopy(target)
    chi.tensors[0] *= -overlap
    return chi, 1.0 - fidelity, fidelity


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


def state_preparation_contribution(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target_state: TargetState,
    initial_state: MPS,
    truncation: KrotovTruncation,
) -> tuple[NDArray[np.float64], float, float]:
    """Compute the Krotov contribution for a pure state-preparation objective.

    For fixed parameters this equals the exact gradient of the target-state
    infidelity ``1 - |<psi_target | psi_M>|^2`` with respect to all gate-supported
    parameters.

    Args:
        circuit: The parameterized circuit.
        theta: Trainable parameter vector.
        target_state: Target MPS or dense statevector.
        initial_state: Circuit input state (consumed).
        truncation: Truncation settings for gate applications.

    Returns:
        A tuple ``(contribution, loss_value, fidelity)``.
    """
    x = np.array([], dtype=np.float64)
    target = _resolve_target_state(target_state, circuit.num_qubits)
    return _state_preparation_contribution_resolved(circuit, theta, x, target, initial_state, truncation)


def _state_preparation_contribution_resolved(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    x: NDArray[np.float64],
    target: MPS,
    initial_state: MPS,
    truncation: KrotovTruncation,
) -> tuple[NDArray[np.float64], float, float]:
    """Compute state-preparation contribution for an already normalized target.

    Args:
        circuit: The parameterized circuit.
        theta: Trainable parameter vector.
        x: Empty input sample used for data-free state preparation.
        target: Normalized target MPS.
        initial_state: Circuit input state (consumed).
        truncation: Truncation settings for gate applications.

    Returns:
        A tuple ``(contribution, loss_value, fidelity)``.
    """
    states = forward_states(circuit, theta, x, initial_state, truncation)
    chi, loss_value, fidelity = _state_preparation_terminal_costate_from_mps(states[-1], target)

    contribution = np.zeros(circuit.num_params, dtype=np.float64)
    for k in range(len(circuit.gates) - 1, -1, -1):
        gate = circuit.gates[k]
        if gate.param_index is not None:
            contribution[gate.param_index] += _gate_contribution(circuit, gate, chi, states[k + 1], truncation)
        if k > 0:
            matrix, sites = circuit.gate_matrix(gate, theta, x)
            _apply_operator(chi, matrix.conj().T, sites, truncation)
    return contribution, loss_value, fidelity


def state_preparation_metrics(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target_state: TargetState,
    *,
    initial_state: MPS | None = None,
    truncation: KrotovTruncation | None = None,
) -> tuple[float, float]:
    """Evaluate target-state infidelity and fidelity.

    Args:
        circuit: The parameterized circuit.
        theta: Trainable parameter vector.
        target_state: Target MPS or dense statevector.
        initial_state: Fixed circuit input MPS, or ``None`` for the all-zeros state.
        truncation: Truncation settings; defaults to exact application.

    Returns:
        A tuple ``(loss_value, fidelity)``.
    """
    x = np.array([], dtype=np.float64)
    truncation = truncation if truncation is not None else KrotovTruncation()
    target = _resolve_target_state(target_state, circuit.num_qubits)
    state = _resolve_fixed_initial_state(initial_state, circuit.num_qubits)
    return _state_preparation_metrics_resolved(circuit, theta, x, target, state, truncation)


def _state_preparation_metrics_resolved(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    x: NDArray[np.float64],
    target: MPS,
    initial_state: MPS,
    truncation: KrotovTruncation,
) -> tuple[float, float]:
    """Evaluate state-preparation metrics for a resolved target MPS.

    Args:
        circuit: The parameterized circuit.
        theta: Trainable parameter vector.
        x: Empty input sample used for data-free state preparation.
        target: Normalized target MPS.
        initial_state: Circuit input state (consumed).
        truncation: Truncation settings for gate applications.

    Returns:
        A tuple ``(loss_value, fidelity)``.
    """
    state = initial_state
    for gate in circuit.gates:
        matrix, sites = circuit.gate_matrix(gate, theta, x)
        _apply_operator(state, matrix, sites, truncation)
    overlap = target.scalar_product(state)
    fidelity = float(abs(overlap) ** 2)
    return 1.0 - fidelity, fidelity


def state_preparation_loss(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target_state: TargetState,
    *,
    initial_state: MPS | None = None,
    truncation: KrotovTruncation | None = None,
) -> float:
    """Evaluate the target-state infidelity.

    Args:
        circuit: The parameterized circuit.
        theta: Trainable parameter vector.
        target_state: Target MPS or dense statevector.
        initial_state: Fixed circuit input MPS, or ``None`` for the all-zeros state.
        truncation: Truncation settings; defaults to exact application.

    Returns:
        Infidelity ``1 - |<psi_target | psi_M>|^2``.
    """
    loss_value, _ = state_preparation_metrics(
        circuit, theta, target_state, initial_state=initial_state, truncation=truncation
    )
    return loss_value


def noisy_state_preparation_contribution(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target_state: TargetState,
    noise_model: NoiseModel | None,
    tjm_options: KrotovTJMOptions,
    initial_state: MPS,
    truncation: KrotovTruncation,
    *,
    iteration: int = 0,
    fixed_noise_maps: list[list[KrotovNoiseMap]] | None = None,
) -> tuple[NDArray[np.float64], float, float, list[KrotovTrajectory]]:
    """Compute the noisy circuit-TJM state-preparation Krotov contribution.

    The noisy loss is ``1 - mean_r |<target|psi_r>|^2`` over pure MPS
    trajectories. With ``trajectory_update="independent"``, terminal costates
    include the ``1/R`` factor, so trajectory contributions are summed directly.
    With ``trajectory_update="cross"``, the contribution is the density-matrix
    trajectory expansion ``-(1/R^2) sum_rs 2 Re <xi_r|D|psi_s><psi_s|xi_r>``.

    Args:
        circuit: The parameterized circuit.
        theta: Trainable parameter vector.
        target_state: Target MPS or dense statevector.
        noise_model: Fixed YAQS noise model. It is never modified or optimized.
        tjm_options: Trajectory settings.
        initial_state: Circuit input state (consumed).
        truncation: Truncation settings for gate/noise applications.
        iteration: Outer optimizer iteration used to refresh trajectory seeds.
        fixed_noise_maps: Optional fixed realizations for common-random-number
            finite differences.

    Returns:
        Tuple ``(contribution, loss, mean_fidelity, trajectories)``.
    """
    if tjm_options.trajectory_update == "cross":
        return noisy_state_preparation_cross_contribution(
            circuit,
            theta,
            target_state,
            noise_model,
            tjm_options,
            initial_state,
            truncation,
            iteration=iteration,
            fixed_noise_maps=fixed_noise_maps,
        )

    x = np.array([], dtype=np.float64)
    target = _resolve_target_state(target_state, circuit.num_qubits)
    trajectories = _forward_tjm_trajectories(
        circuit,
        theta,
        x,
        initial_state,
        truncation,
        noise_model,
        tjm_options,
        iteration,
        fixed_noise_maps,
    )

    contribution = np.zeros(circuit.num_params, dtype=np.float64)
    fidelity_sum = 0.0
    scale = 1.0 / tjm_options.num_trajectories
    for trajectory in trajectories:
        chi, _loss_value, fidelity = _state_preparation_terminal_costate_from_mps(trajectory.states[-1], target)
        chi.tensors[0] *= scale
        fidelity_sum += fidelity
        contribution += _trajectory_contribution(circuit, theta, x, trajectory, chi, truncation)

    mean_fidelity = fidelity_sum * scale
    return contribution, 1.0 - mean_fidelity, mean_fidelity, trajectories


def noisy_state_preparation_cross_contribution(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target_state: TargetState,
    noise_model: NoiseModel | None,
    tjm_options: KrotovTJMOptions,
    initial_state: MPS,
    truncation: KrotovTruncation,
    *,
    iteration: int = 0,
    fixed_noise_maps: list[list[KrotovNoiseMap]] | None = None,
) -> tuple[NDArray[np.float64], float, float, list[KrotovTrajectory]]:
    """Compute the cross-trajectory circuit-TJM state-preparation contribution.

    This implements the finite-trajectory density-matrix expansion of the Krotov
    update from Goerz and Jacobs, Quantum Sci. Technol. 3 045005 (2018). Forward
    states and backward target states are still represented as pure MPS
    trajectories, but every backward trajectory is paired with every forward
    trajectory in each gate-local update.

    Returns:
        Tuple ``(contribution, loss, mean_fidelity, trajectories)``.
    """
    x = np.array([], dtype=np.float64)
    target = _resolve_target_state(target_state, circuit.num_qubits)
    trajectories = _forward_tjm_trajectories(
        circuit,
        theta,
        x,
        initial_state,
        truncation,
        noise_model,
        tjm_options,
        iteration,
        fixed_noise_maps,
    )

    stale_costates = [
        _trajectory_chi_tildes(circuit, theta, x, trajectory, copy.deepcopy(target), truncation)
        for trajectory in trajectories
    ]

    contribution = np.zeros(circuit.num_params, dtype=np.float64)
    for k, gate in enumerate(circuit.gates):
        if gate.param_index is None:
            continue
        costates_after_noise = [costates[k] for costates in stale_costates]
        gate_outputs = [trajectory.gate_outputs[k] for trajectory in trajectories]
        contribution[gate.param_index] += _cross_gate_contribution(
            circuit,
            gate,
            costates_after_noise,
            gate_outputs,
            truncation,
        )

    fidelities = [float(abs(target.scalar_product(trajectory.states[-1])) ** 2) for trajectory in trajectories]
    mean_fidelity = float(np.mean(fidelities))
    return contribution, 1.0 - mean_fidelity, mean_fidelity, trajectories


def noisy_state_preparation_metrics(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target_state: TargetState,
    noise_model: NoiseModel | None,
    tjm_options: KrotovTJMOptions,
    *,
    initial_state: MPS | None = None,
    truncation: KrotovTruncation | None = None,
    iteration: int = 0,
    fixed_noise_maps: list[list[KrotovNoiseMap]] | None = None,
) -> tuple[float, float, list[float]]:
    """Evaluate noisy state-preparation infidelity and trajectory fidelities.

    Returns:
        Tuple ``(loss, mean_fidelity, trajectory_fidelities)``.
    """
    truncation = truncation if truncation is not None else KrotovTruncation()
    target = _resolve_target_state(target_state, circuit.num_qubits)
    state = _resolve_fixed_initial_state(initial_state, circuit.num_qubits)
    trajectories = _forward_tjm_trajectories(
        circuit,
        theta,
        np.array([], dtype=np.float64),
        state,
        truncation,
        noise_model,
        tjm_options,
        iteration,
        fixed_noise_maps,
    )
    fidelities = [float(abs(target.scalar_product(trajectory.states[-1])) ** 2) for trajectory in trajectories]
    mean_fidelity = float(np.mean(fidelities))
    return 1.0 - mean_fidelity, mean_fidelity, fidelities


def noisy_state_preparation_loss(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target_state: TargetState,
    noise_model: NoiseModel | None,
    tjm_options: KrotovTJMOptions,
    *,
    initial_state: MPS | None = None,
    truncation: KrotovTruncation | None = None,
    iteration: int = 0,
    fixed_noise_maps: list[list[KrotovNoiseMap]] | None = None,
) -> float:
    """Evaluate noisy circuit-TJM state-preparation infidelity.

    Returns:
        Noisy trajectory-averaged infidelity.
    """
    loss_value, _mean_fidelity, _fidelities = noisy_state_preparation_metrics(
        circuit,
        theta,
        target_state,
        noise_model,
        tjm_options,
        initial_state=initial_state,
        truncation=truncation,
        iteration=iteration,
        fixed_noise_maps=fixed_noise_maps,
    )
    return loss_value


def noisy_sample_contribution(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    x: NDArray[np.float64],
    y: float,
    readout: KrotovReadout,
    bias: float,
    noise_model: NoiseModel | None,
    tjm_options: KrotovTJMOptions,
    initial_state: MPS,
    truncation: KrotovTruncation,
    *,
    iteration: int = 0,
    fixed_noise_maps: list[list[KrotovNoiseMap]] | None = None,
) -> tuple[NDArray[np.float64], float, float, list[float], list[KrotovTrajectory]]:
    """Compute one noisy supervised sample contribution with YAQS trajectory averaging.

    Trajectory expectation values are averaged first, and the scalar loss is then
    applied to the averaged readout. This intentionally differs from averaging
    per-trajectory losses.

    Args:
        circuit: The parameterized circuit.
        theta: Trainable parameter vector.
        x: Input sample.
        y: Sample label.
        readout: Readout and loss specification.
        bias: Scalar bias of the prediction.
        noise_model: Fixed YAQS noise model.
        tjm_options: Trajectory settings.
        initial_state: Circuit input state (consumed).
        truncation: Truncation settings for gate/noise applications.
        iteration: Outer optimizer iteration used to refresh trajectory seeds.
        fixed_noise_maps: Optional fixed realizations for common-random-number
            finite differences.

    Returns:
        Tuple ``(contribution, loss, averaged_readout, trajectory_readouts, trajectories)``.
    """
    trajectories = _forward_tjm_trajectories(
        circuit,
        theta,
        x,
        initial_state,
        truncation,
        noise_model,
        tjm_options,
        iteration,
        fixed_noise_maps,
    )
    trajectory_readouts = [_expectation(trajectory.states[-1], readout.observable) for trajectory in trajectories]
    zbar = float(np.mean(trajectory_readouts))
    loss_value, factor = _loss_and_costate_factor(zbar, bias, y, readout.loss)

    contribution = np.zeros(circuit.num_params, dtype=np.float64)
    scale = 1.0 / tjm_options.num_trajectories
    for trajectory in trajectories:
        chi = copy.deepcopy(trajectory.states[-1])
        chi.apply_local(readout.observable)
        chi.tensors[0] *= factor * scale
        contribution += _trajectory_contribution(circuit, theta, x, trajectory, chi, truncation)

    return contribution, loss_value, zbar, trajectory_readouts, trajectories


def noisy_sample_loss(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    x: NDArray[np.float64],
    y: float,
    readout: KrotovReadout,
    bias: float,
    noise_model: NoiseModel | None,
    tjm_options: KrotovTJMOptions,
    *,
    initial_state: MPS | None = None,
    truncation: KrotovTruncation | None = None,
    iteration: int = 0,
    fixed_noise_maps: list[list[KrotovNoiseMap]] | None = None,
) -> tuple[float, float, list[float]]:
    """Evaluate a noisy supervised sample loss with trajectory-averaged readout.

    Returns:
        Tuple ``(loss, averaged_readout, trajectory_readouts)``.
    """
    truncation = truncation if truncation is not None else KrotovTruncation()
    trajectories = _forward_tjm_trajectories(
        circuit,
        theta,
        x,
        _resolve_fixed_initial_state(initial_state, circuit.num_qubits),
        truncation,
        noise_model,
        tjm_options,
        iteration,
        fixed_noise_maps,
    )
    trajectory_readouts = [_expectation(trajectory.states[-1], readout.observable) for trajectory in trajectories]
    zbar = float(np.mean(trajectory_readouts))
    loss_value, _factor = _loss_and_costate_factor(zbar, bias, y, readout.loss)
    return loss_value, zbar, trajectory_readouts


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
    for x, y in zip(inputs, labels, strict=False):
        state = _resolve_initial_state(initial_state, x, circuit.num_qubits)
        contribution, loss_value, z = sample_contribution(circuit, theta, x, float(y), readout, bias, state, truncation)
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
    for x, y in zip(inputs, labels, strict=False):
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


def _init_state_preparation_trace() -> dict[str, list[float | int | str]]:
    """Create an empty state-preparation training trace.

    Returns:
        Trace dictionary with an additional ``"fidelity"`` record.
    """
    trace = _init_trace()
    trace["fidelity"] = []
    return trace


def _record_state_preparation(
    trace: dict[str, list[float | int | str]],
    step: int,
    phase: str,
    loss: float,
    fidelity: float,
    step_size: float,
    gradient_norm: float,
    update_norm: float,
) -> None:
    """Append one state-preparation iteration record to the trace.

    Args:
        trace: Training trace.
        step: Iteration counter.
        phase: Phase label (``"init"``, ``"online"``, or ``"batch"``).
        loss: Infidelity after the iteration.
        fidelity: Fidelity after the iteration.
        step_size: Step size used in the iteration.
        gradient_norm: Norm of the update direction.
        update_norm: Norm of the parameter change.
    """
    _record(trace, step, phase, loss, step_size, gradient_norm, update_norm)
    trace["fidelity"].append(fidelity)


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
        for x, y in zip(inputs, labels, strict=False):
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
    theta -= step_size * mean_contribution
    if readout.use_bias:
        bias -= step_size * bias_gradient
    gradient_norm = float(np.sqrt(np.linalg.norm(mean_contribution) ** 2 + bias_gradient**2))
    return theta, bias, gradient_norm, mean_loss


def _state_preparation_online_update(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: MPS,
    step_size: float,
    initial_state: MPS,
    truncation: KrotovTruncation,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply one stale-adjoint online sweep for state preparation.

    Args:
        circuit: The parameterized circuit.
        theta: Pre-sweep trainable parameter vector.
        target: Normalized target MPS.
        step_size: Online step size.
        initial_state: Circuit input state (consumed).
        truncation: Truncation settings for gate applications.

    Returns:
        A tuple ``(new_theta, contribution)`` with the post-sweep parameters and
        observed gate-local signals.
    """
    x = np.array([], dtype=np.float64)
    states = forward_states(circuit, theta, x, initial_state, truncation)
    chi_terminal_state, _, _ = _state_preparation_terminal_costate_from_mps(states[-1], target)
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


def _state_preparation_batch_epoch(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: MPS,
    step_size: float,
    initial_state: MPS | None,
    truncation: KrotovTruncation,
) -> tuple[NDArray[np.float64], float, float, float]:
    """Run one full-gradient state-preparation update.

    Args:
        circuit: The parameterized circuit.
        theta: Trainable parameter vector.
        target: Normalized target MPS.
        step_size: Batch step size.
        initial_state: Fixed input MPS, or ``None`` for the all-zeros state.
        truncation: Truncation settings for gate applications.

    Returns:
        A tuple ``(theta, gradient_norm, pre_update_loss, pre_update_fidelity)``.
    """
    contribution, loss_value, fidelity = _state_preparation_contribution_resolved(
        circuit,
        theta,
        np.array([], dtype=np.float64),
        target,
        _resolve_fixed_initial_state(initial_state, circuit.num_qubits),
        truncation,
    )
    theta -= step_size * contribution
    gradient_norm = float(np.linalg.norm(contribution))
    return theta, gradient_norm, loss_value, fidelity


def _train_state_preparation(
    circuit: ParameterizedCircuit,
    target_state: TargetState,
    initial_theta: NDArray[np.float64],
    options: KrotovOptions,
    initial_state: MPS | None,
    switch_iteration: int,
) -> KrotovResult:
    """Shared training loop for state-preparation online, batch, and hybrid variants.

    Args:
        circuit: The parameterized circuit.
        target_state: Target MPS or dense statevector.
        initial_theta: Initial trainable parameter vector.
        options: Training hyperparameters.
        initial_state: Fixed input MPS, or ``None`` for the all-zeros state.
        switch_iteration: Last online iteration (``0`` for pure batch).

    Returns:
        The training result with final parameters and trace.
    """
    theta = np.asarray(initial_theta, dtype=np.float64).copy()
    truncation = options.truncation
    target = _resolve_target_state(target_state, circuit.num_qubits)
    _resolve_fixed_initial_state(initial_state, circuit.num_qubits)

    trace = _init_state_preparation_trace()
    loss, fidelity = _state_preparation_metrics_resolved(
        circuit,
        theta,
        np.array([], dtype=np.float64),
        target,
        _resolve_fixed_initial_state(initial_state, circuit.num_qubits),
        truncation,
    )
    _record_state_preparation(trace, 0, "init", loss, fidelity, 0.0, 0.0, 0.0)

    for iteration in range(1, options.max_iterations + 1):
        theta_before = theta.copy()
        if iteration <= switch_iteration:
            step = _step_size(options.online_step_size, iteration, options.online_schedule, options.online_decay)
            theta, contribution = _state_preparation_online_update(
                circuit,
                theta,
                target,
                step,
                _resolve_fixed_initial_state(initial_state, circuit.num_qubits),
                truncation,
            )
            gradient_norm = float(np.linalg.norm(contribution))
            phase = "online"
        else:
            phase_iteration = iteration - switch_iteration
            step = _step_size(options.batch_step_size, phase_iteration, options.batch_schedule, options.batch_decay)
            theta, gradient_norm, _, _ = _state_preparation_batch_epoch(
                circuit,
                theta,
                target,
                step,
                initial_state,
                truncation,
            )
            phase = "batch"

        loss, fidelity = _state_preparation_metrics_resolved(
            circuit,
            theta,
            np.array([], dtype=np.float64),
            target,
            _resolve_fixed_initial_state(initial_state, circuit.num_qubits),
            truncation,
        )
        update_norm = float(np.linalg.norm(theta - theta_before))
        _record_state_preparation(trace, iteration, phase, loss, fidelity, step, gradient_norm, update_norm)

    return KrotovResult(theta=theta, bias=0.0, trace=trace)


def _trajectory_chi_tildes(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    x: NDArray[np.float64],
    trajectory: KrotovTrajectory,
    chi_terminal: MPS,
    truncation: KrotovTruncation,
) -> list[MPS]:
    """Return stale costates after pulling back through each realized noise map."""
    chi_tildes: list[MPS] = [chi_terminal] * len(circuit.gates)
    lambda_after_noise = chi_terminal
    for k in range(len(circuit.gates) - 1, -1, -1):
        gate = circuit.gates[k]
        chi_tilde = copy.deepcopy(lambda_after_noise)
        _pullback_noise_map(chi_tilde, trajectory.noise_maps[k], truncation)
        chi_tildes[k] = copy.deepcopy(chi_tilde)
        matrix, sites = circuit.gate_matrix(gate, theta, x)
        _apply_operator(chi_tilde, matrix.conj().T, sites, truncation)
        lambda_after_noise = chi_tilde
    return chi_tildes


def _noisy_state_preparation_batch_epoch(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: MPS,
    noise_model: NoiseModel | None,
    tjm_options: KrotovTJMOptions,
    step_size: float,
    initial_state: MPS | None,
    truncation: KrotovTruncation,
    iteration: int,
    fixed_noise_maps: list[list['KrotovNoiseMap']] | None = None,
) -> tuple[NDArray[np.float64], float, float, float]:
    """Run one noisy full-trajectory-gradient state-preparation update.

    Returns:
        Tuple ``(theta, gradient_norm, pre_update_loss, pre_update_fidelity)``.
    """
    contribution, loss_value, fidelity, _trajectories = noisy_state_preparation_contribution(
        circuit,
        theta,
        target,
        noise_model,
        tjm_options,
        _resolve_fixed_initial_state(initial_state, circuit.num_qubits),
        truncation,
        iteration=iteration,
        fixed_noise_maps=fixed_noise_maps,
    )
    theta -= step_size * contribution
    return theta, float(np.linalg.norm(contribution)), loss_value, fidelity


def _noisy_state_preparation_online_update(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: MPS,
    noise_model: NoiseModel | None,
    tjm_options: KrotovTJMOptions,
    step_size: float,
    initial_state: MPS | None,
    truncation: KrotovTruncation,
    iteration: int,
    fixed_noise_maps: list[list['KrotovNoiseMap']] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run one stale-adjoint noisy online sweep for state preparation.

    Returns:
        Tuple ``(new_theta, contribution)`` after the sweep.
    """
    if tjm_options.trajectory_update == "cross":
        return _noisy_state_preparation_online_cross_update(
            circuit,
            theta,
            target,
            noise_model,
            tjm_options,
            step_size,
            initial_state,
            truncation,
            iteration,
            fixed_noise_maps,
        )

    x = np.array([], dtype=np.float64)
    initial = _resolve_fixed_initial_state(initial_state, circuit.num_qubits)
    trajectories = _forward_tjm_trajectories(
        circuit,
        theta,
        x,
        initial,
        truncation,
        noise_model,
        tjm_options,
        iteration,
        fixed_noise_maps,
    )

    scale = 1.0 / tjm_options.num_trajectories
    stale_costates = []
    for trajectory in trajectories:
        chi, _loss_value, _fidelity = _state_preparation_terminal_costate_from_mps(trajectory.states[-1], target)
        chi.tensors[0] *= scale
        stale_costates.append(_trajectory_chi_tildes(circuit, theta, x, trajectory, chi, truncation))

    new_theta = theta.copy()
    current_states = [_resolve_fixed_initial_state(initial_state, circuit.num_qubits) for _ in trajectories]
    contribution = np.zeros(circuit.num_params, dtype=np.float64)

    for k, gate in enumerate(circuit.gates):
        if gate.param_index is not None:
            signal = 0.0
            for traj_idx, current in enumerate(current_states):
                gate_output = copy.deepcopy(current)
                matrix, sites = circuit.gate_matrix(gate, new_theta, x)
                _apply_operator(gate_output, matrix, sites, truncation)
                signal += _gate_contribution(circuit, gate, stale_costates[traj_idx][k], gate_output, truncation)
            contribution[gate.param_index] += signal
            new_theta[gate.param_index] -= step_size * signal

        for traj_idx, current in enumerate(current_states):
            matrix, sites = circuit.gate_matrix(gate, new_theta, x)
            _apply_operator(current, matrix, sites, truncation)
            _apply_noise_map(current, trajectories[traj_idx].noise_maps[k], truncation)

    return new_theta, contribution


def _noisy_state_preparation_online_cross_update(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: MPS,
    noise_model: NoiseModel | None,
    tjm_options: KrotovTJMOptions,
    step_size: float,
    initial_state: MPS | None,
    truncation: KrotovTruncation,
    iteration: int,
    fixed_noise_maps: list[list['KrotovNoiseMap']] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Run one stale-adjoint online sweep with cross-trajectory pairings.

    The realized TJM maps are fixed from the guess parameters. Forward states are
    refreshed with partially updated parameters, while the target costates remain
    stale, matching the existing online Krotov structure.

    Returns:
        Tuple ``(new_theta, contribution)`` after the sweep.
    """
    x = np.array([], dtype=np.float64)
    initial = _resolve_fixed_initial_state(initial_state, circuit.num_qubits)
    trajectories = _forward_tjm_trajectories(
        circuit,
        theta,
        x,
        initial,
        truncation,
        noise_model,
        tjm_options,
        iteration,
        fixed_noise_maps,
    )
    stale_costates = [
        _trajectory_chi_tildes(circuit, theta, x, trajectory, copy.deepcopy(target), truncation)
        for trajectory in trajectories
    ]

    new_theta = theta.copy()
    current_states = [_resolve_fixed_initial_state(initial_state, circuit.num_qubits) for _ in trajectories]
    contribution = np.zeros(circuit.num_params, dtype=np.float64)

    for k, gate in enumerate(circuit.gates):
        gate_outputs: list[MPS] = []
        matrix, sites = circuit.gate_matrix(gate, new_theta, x)
        for current in current_states:
            gate_output = copy.deepcopy(current)
            _apply_operator(gate_output, matrix, sites, truncation)
            gate_outputs.append(gate_output)

        if gate.param_index is not None:
            costates_after_noise = [costates[k] for costates in stale_costates]
            signal = _cross_gate_contribution(circuit, gate, costates_after_noise, gate_outputs, truncation)
            contribution[gate.param_index] += signal
            new_theta[gate.param_index] -= step_size * signal
            matrix, sites = circuit.gate_matrix(gate, new_theta, x)
            gate_outputs = []
            for current in current_states:
                gate_output = copy.deepcopy(current)
                _apply_operator(gate_output, matrix, sites, truncation)
                gate_outputs.append(gate_output)

        for traj_idx, gate_output in enumerate(gate_outputs):
            current = copy.deepcopy(gate_output)
            _apply_noise_map(current, trajectories[traj_idx].noise_maps[k], truncation)
            current_states[traj_idx] = current

    return new_theta, contribution


def _train_noisy_state_preparation(
    circuit: ParameterizedCircuit,
    target_state: TargetState,
    noise_model: NoiseModel | None,
    tjm_options: KrotovTJMOptions,
    initial_theta: NDArray[np.float64],
    options: KrotovOptions,
    initial_state: MPS | None,
    switch_iteration: int,
) -> KrotovResult:
    """Shared noisy state-preparation loop for online, batch, and hybrid modes.

    Returns:
        Training result with final parameters and noisy state-preparation trace.
    """
    theta = np.asarray(initial_theta, dtype=np.float64).copy()
    truncation = options.truncation
    target = _resolve_target_state(target_state, circuit.num_qubits)
    _resolve_fixed_initial_state(initial_state, circuit.num_qubits)

    fixed_maps = None
    if tjm_options.use_crn:
        trajs = _forward_tjm_trajectories(
            circuit,
            theta,
            np.array([], dtype=np.float64),
            _resolve_fixed_initial_state(initial_state, circuit.num_qubits),
            truncation,
            noise_model,
            tjm_options,
            options.seed,
        )
        fixed_maps = [traj.noise_maps for traj in trajs]

    trace = _init_state_preparation_trace()
    loss, fidelity, _fidelities = noisy_state_preparation_metrics(
        circuit,
        theta,
        target,
        noise_model,
        tjm_options,
        initial_state=initial_state,
        truncation=truncation,
        iteration=0,
        fixed_noise_maps=fixed_maps,
    )
    _record_state_preparation(trace, 0, "init", loss, fidelity, 0.0, 0.0, 0.0)

    for iteration in range(1, options.max_iterations + 1):
        theta_before = theta.copy()
        if iteration <= switch_iteration:
            step = _step_size(options.online_step_size, iteration, options.online_schedule, options.online_decay)
            theta, contribution = _noisy_state_preparation_online_update(
                circuit,
                theta,
                target,
                noise_model,
                tjm_options,
                step,
                initial_state,
                truncation,
                options.seed + iteration,
                fixed_noise_maps=fixed_maps,
            )
            gradient_norm = float(np.linalg.norm(contribution))
            phase = "online"
        else:
            phase_iteration = iteration - switch_iteration
            step = _step_size(options.batch_step_size, phase_iteration, options.batch_schedule, options.batch_decay)
            theta, gradient_norm, _loss_before, _fidelity_before = _noisy_state_preparation_batch_epoch(
                circuit,
                theta,
                target,
                noise_model,
                tjm_options,
                step,
                initial_state,
                truncation,
                options.seed + iteration,
                fixed_noise_maps=fixed_maps,
            )
            phase = "batch"

        loss, fidelity, _fidelities = noisy_state_preparation_metrics(
            circuit,
            theta,
            target,
            noise_model,
            tjm_options,
            initial_state=initial_state,
            truncation=truncation,
            iteration=options.seed + iteration,
            fixed_noise_maps=fixed_maps,
        )
        update_norm = float(np.linalg.norm(theta - theta_before))
        _record_state_preparation(trace, iteration, phase, loss, fidelity, step, gradient_norm, update_norm)

    return KrotovResult(theta=theta, bias=0.0, trace=trace)


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


def train_krotov_state_preparation_online(
    circuit: ParameterizedCircuit,
    target_state: TargetState,
    *,
    initial_theta: NDArray[np.float64],
    options: KrotovOptions | None = None,
    initial_state: MPS | None = None,
) -> KrotovResult:
    """Train a circuit to prepare a target state with online Krotov sweeps.

    Args:
        circuit: The parameterized circuit.
        target_state: Target MPS or dense statevector.
        initial_theta: Initial trainable parameter vector.
        options: Training hyperparameters; defaults to :class:`KrotovOptions`.
        initial_state: Fixed input MPS, or ``None`` for the all-zeros state.

    Returns:
        The training result with final parameters and a trace containing
        ``"loss"`` and ``"fidelity"``.
    """
    options = options if options is not None else KrotovOptions()
    return _train_state_preparation(
        circuit,
        target_state,
        initial_theta,
        options,
        initial_state,
        switch_iteration=options.max_iterations,
    )


def train_krotov_state_preparation_batch(
    circuit: ParameterizedCircuit,
    target_state: TargetState,
    *,
    initial_theta: NDArray[np.float64],
    options: KrotovOptions | None = None,
    initial_state: MPS | None = None,
) -> KrotovResult:
    """Train a circuit to prepare a target state with full-gradient updates.

    Args:
        circuit: The parameterized circuit.
        target_state: Target MPS or dense statevector.
        initial_theta: Initial trainable parameter vector.
        options: Training hyperparameters; defaults to :class:`KrotovOptions`.
        initial_state: Fixed input MPS, or ``None`` for the all-zeros state.

    Returns:
        The training result with final parameters and a trace containing
        ``"loss"`` and ``"fidelity"``.
    """
    options = options if options is not None else KrotovOptions()
    return _train_state_preparation(
        circuit,
        target_state,
        initial_theta,
        options,
        initial_state,
        switch_iteration=0,
    )


def train_krotov_state_preparation_hybrid(
    circuit: ParameterizedCircuit,
    target_state: TargetState,
    *,
    initial_theta: NDArray[np.float64],
    options: KrotovOptions | None = None,
    initial_state: MPS | None = None,
) -> KrotovResult:
    """Train a circuit to prepare a target state with online-to-batch Krotov updates.

    Args:
        circuit: The parameterized circuit.
        target_state: Target MPS or dense statevector.
        initial_theta: Initial trainable parameter vector.
        options: Training hyperparameters; defaults to :class:`KrotovOptions`.
        initial_state: Fixed input MPS, or ``None`` for the all-zeros state.

    Returns:
        The training result with final parameters and a trace containing
        ``"loss"`` and ``"fidelity"``.
    """
    options = options if options is not None else KrotovOptions()
    return _train_state_preparation(
        circuit,
        target_state,
        initial_theta,
        options,
        initial_state,
        switch_iteration=options.switch_iteration,
    )


def train_krotov_noisy_state_preparation_online(
    circuit: ParameterizedCircuit,
    target_state: TargetState,
    noise_model: NoiseModel | None,
    tjm_options: KrotovTJMOptions | None = None,
    *,
    initial_theta: NDArray[np.float64],
    options: KrotovOptions | None = None,
    initial_state: MPS | None = None,
) -> KrotovResult:
    """Train a noisy circuit-TJM state-preparation objective with online sweeps.

    Returns:
        Training result with final parameters and trace.
    """
    options = options if options is not None else KrotovOptions()
    tjm_options = tjm_options if tjm_options is not None else KrotovTJMOptions()
    return _train_noisy_state_preparation(
        circuit,
        target_state,
        noise_model,
        tjm_options,
        initial_theta,
        options,
        initial_state,
        switch_iteration=options.max_iterations,
    )


def train_krotov_noisy_state_preparation_batch(
    circuit: ParameterizedCircuit,
    target_state: TargetState,
    noise_model: NoiseModel | None,
    tjm_options: KrotovTJMOptions | None = None,
    *,
    initial_theta: NDArray[np.float64],
    options: KrotovOptions | None = None,
    initial_state: MPS | None = None,
) -> KrotovResult:
    """Train a noisy circuit-TJM state-preparation objective with batch updates.

    Returns:
        Training result with final parameters and trace.
    """
    options = options if options is not None else KrotovOptions()
    tjm_options = tjm_options if tjm_options is not None else KrotovTJMOptions()
    return _train_noisy_state_preparation(
        circuit,
        target_state,
        noise_model,
        tjm_options,
        initial_theta,
        options,
        initial_state,
        switch_iteration=0,
    )


def train_krotov_noisy_state_preparation_hybrid(
    circuit: ParameterizedCircuit,
    target_state: TargetState,
    noise_model: NoiseModel | None,
    tjm_options: KrotovTJMOptions | None = None,
    *,
    initial_theta: NDArray[np.float64],
    options: KrotovOptions | None = None,
    initial_state: MPS | None = None,
) -> KrotovResult:
    """Train a noisy circuit-TJM state-preparation objective with a hybrid schedule.

    Returns:
        Training result with final parameters and trace.
    """
    options = options if options is not None else KrotovOptions()
    tjm_options = tjm_options if tjm_options is not None else KrotovTJMOptions()
    return _train_noisy_state_preparation(
        circuit,
        target_state,
        noise_model,
        tjm_options,
        initial_theta,
        options,
        initial_state,
        switch_iteration=options.switch_iteration,
    )


__all__ = [
    "KrotovNoiseMap",
    "KrotovOptions",
    "KrotovReadout",
    "KrotovResult",
    "KrotovTJMOptions",
    "KrotovTrajectory",
    "KrotovTruncation",
    "backward_costates",
    "empirical_loss",
    "forward_states",
    "forward_tjm_trajectory",
    "noisy_sample_contribution",
    "noisy_sample_loss",
    "noisy_state_preparation_contribution",
    "noisy_state_preparation_cross_contribution",
    "noisy_state_preparation_loss",
    "noisy_state_preparation_metrics",
    "sample_contribution",
    "state_preparation_contribution",
    "state_preparation_loss",
    "state_preparation_metrics",
    "state_preparation_terminal_costate",
    "terminal_costate",
    "train_krotov_batch",
    "train_krotov_hybrid",
    "train_krotov_noisy_state_preparation_batch",
    "train_krotov_noisy_state_preparation_hybrid",
    "train_krotov_noisy_state_preparation_online",
    "train_krotov_online",
    "train_krotov_state_preparation_batch",
    "train_krotov_state_preparation_hybrid",
    "train_krotov_state_preparation_online",
]
