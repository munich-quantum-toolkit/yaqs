# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Surrogate workflow: build training data and train models.

**Public API** (see ``__all__``): :func:`build_training_dataset`, :func:`train_surrogate_model`.

:func:`build_training_dataset` returns a :class:`~torch.utils.data.TensorDataset` for
:meth:`~mqt.yaqs.characterization.memory.backends.surrogates.model.ProcessTensorSurrogate.fit`.

Sequence simulation is delegated to
:mod:`~mqt.yaqs.characterization.memory.backends.sequences`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from mqt.yaqs.core.data_structures.mps import MPS

from .....core._validation import validate_integer

if TYPE_CHECKING:
    import types

    from torch.utils.data import TensorDataset

    from mqt.yaqs.analog.mcwf import MCWFContext
    from mqt.yaqs.characterization.memory.backends.surrogates.model import ProcessTensorSurrogate
    from mqt.yaqs.core.data_structures.mpo import MPO
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
    from mqt.yaqs.core.parallel_utils import ExecutionConfig

from ...shared.interventions import DEFAULT_INTERVENTION_STYLE, normalize_style, sample_train_interventions
from ...shared.utils import (
    StochasticSolver,
    _dense_state_to_mps,
    make_mcwf_static_context,
    resolve_stochastic_solver,
    validate_qubit_memory_operator,
)
from ..sequences.workflow import simulate_sequences
from .data import SequenceRecord, stack_sequence_records
from .utils import sample_density_matrix, sample_initial_psi


def _require_torch() -> types.ModuleType:
    """Import PyTorch or raise with an installation hint.

    Returns:
        The ``torch`` module.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    try:
        import torch  # ruff:ignore[import-outside-top-level]
    except ImportError as exc:
        msg = "PyTorch is required for surrogate training; install with `uv sync --extra torch`."
        raise ImportError(msg) from exc
    return torch


def pack_dataset(
    rho0: np.ndarray,
    e_features: np.ndarray,
    rho_seq: np.ndarray,
) -> TensorDataset:
    """Pack sequence records into a PyTorch :class:`~torch.utils.data.TensorDataset`.

    Args:
        rho0: Array of shape ``(N, 8)``.
        e_features: Array of shape ``(N, K, d_e)``.
        rho_seq: Array of shape ``(N, K, 8)``.

    Returns:
        TensorDataset with tensors ``(e_features, rho0, rho_seq)`` in that order.
    """
    import torch  # ruff:ignore[import-outside-top-level]
    from torch.utils.data import TensorDataset  # ruff:ignore[import-outside-top-level]

    return TensorDataset(
        torch.as_tensor(e_features, dtype=torch.float32),
        torch.as_tensor(rho0, dtype=torch.float32),
        torch.as_tensor(rho_seq, dtype=torch.float32),
    )


def _sample_initial_tjm_state(
    rho_in: np.ndarray,
    *,
    length: int,
    rng: np.random.Generator,
    init_mode: str,
) -> MPS:
    """Build a TJM initial state without a full dense chain vector.

    Eigenstate initialization needs only site 0. Purified initialization needs
    site 0 and one auxiliary qubit. Remaining environment sites are appended as
    product ``|0>`` tensors.

    Args:
        rho_in: Reduced density matrix on site 0.
        length: Total number of qubits in the chain.
        rng: Random number generator used for eigenstate sampling.
        init_mode: ``"eigenstate"`` or ``"purified"``.

    Returns:
        MPS with the sampled state on site 0 and no full-chain dense allocation.
    """
    compact_length = min(length, 2)
    compact_vector = cast(
        "np.ndarray",
        sample_initial_psi(
            rho_in,
            length=compact_length,
            rng=rng,
            init_mode=init_mode,
        ),
    )
    compact_state = _dense_state_to_mps(compact_vector, length=compact_length)
    if compact_length == length:
        return compact_state

    zero_tensor = np.array([1.0, 0.0], dtype=np.complex128).reshape(2, 1, 1)
    tensors = [tensor.copy() for tensor in compact_state.tensors]
    tensors.extend(zero_tensor.copy() for _ in range(length - compact_length))
    state = MPS(length, tensors=tensors)
    state.set_center(compact_length - 1)
    return state


def build_training_dataset(
    operator: MPO,
    sim_params: AnalogSimParams,
    *,
    num_interventions: int,
    n: int,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    parallel: bool = True,
    show_progress: bool = True,
    timesteps: list[float] | None = None,
    init_mode: str = "eigenstate",
    solver: StochasticSolver | None = None,
    intervention_style: str = DEFAULT_INTERVENTION_STYLE,
    _execution: ExecutionConfig | None = None,
) -> TensorDataset:
    """Simulate intervention sequences and pack a surrogate training dataset.

    Args:
        operator: Hamiltonian MPO. The chain length is inferred from ``operator.length``.
        sim_params: Analog simulation parameters.
        num_interventions: Positive integer number of intervention steps.
        n: Positive integer number of sequences to simulate.
        rng: Optional RNG (overrides ``seed`` if provided).
        seed: Optional seed used to create a default RNG.
        parallel: Whether to parallelize over sequences.
        show_progress: Whether to show progress bars.
        timesteps: Optional process-tensor schedule evolution durations (defaults to
            ``[sim_params.dt] * (num_interventions + 1)``).
        init_mode: Initial-state sampling mode (see :func:`sample_initial_psi`).
        solver: Stochastic solver (``"MCWF"`` or ``"TJM"``); defaults to ``"MCWF"``.
        intervention_style: Training intervention style (``"haar"``, ``"clifford"``, or
            ``"measure_prepare"``).

    Returns:
        A :class:`~torch.utils.data.TensorDataset` with tensors ``(E_features, rho0, rho_seq)``.

    Raises:
        ValueError: If either count is not positive, ``timesteps`` has the wrong length, or
            ``operator`` is not a qubit Hamiltonian.
    """
    resolved_num_interventions = validate_integer(num_interventions, name="num_interventions", minimum=1)
    n_sequences = validate_integer(n, name="n", minimum=1)

    validate_qubit_memory_operator(operator)
    chain_length = int(operator.length)
    if timesteps is None:
        timesteps = [float(sim_params.dt)] * (resolved_num_interventions + 1)
    if len(timesteps) != resolved_num_interventions + 1:
        msg = (
            f"Process-tensor schedule: timesteps length must be num_interventions+1="
            f"{resolved_num_interventions + 1}, got {len(timesteps)}."
        )
        raise ValueError(msg)

    _require_torch()
    stochastic_solver = resolve_stochastic_solver(solver=solver)

    static_ctx: MCWFContext | None = None
    if stochastic_solver == "MCWF":
        static_ctx = make_mcwf_static_context(operator, sim_params, noise_model=None)

    if rng is None:
        rng = np.random.default_rng(0 if seed is None else int(seed))

    intervention_steps_list: list[list[Any]] = []
    initial_psis: list[np.ndarray | MPS] = []
    choi_feature_rows_per_sequence: list[np.ndarray] = []

    for _ in range(n_sequences):
        rho_in = sample_density_matrix(rng)
        step_pairs, choi_rows = sample_train_interventions(
            resolved_num_interventions,
            normalize_style(str(intervention_style)),
            rng,
        )
        intervention_steps_list.append(step_pairs)
        choi_feature_rows_per_sequence.append(choi_rows.astype(np.float32))
        if stochastic_solver == "TJM":
            initial_psis.append(
                _sample_initial_tjm_state(
                    rho_in,
                    length=chain_length,
                    rng=rng,
                    init_mode=init_mode,
                )
            )
        else:
            initial_psi = cast(
                "np.ndarray",
                sample_initial_psi(rho_in, length=chain_length, rng=rng, init_mode=init_mode),
            )
            initial_psis.append(initial_psi)

    samples = cast(
        "list[SequenceRecord]",
        simulate_sequences(
            operator=operator,
            sim_params=sim_params,
            timesteps=timesteps,
            intervention_steps_list=intervention_steps_list,
            initial_psis=initial_psis,
            e_features_rows=choi_feature_rows_per_sequence,
            parallel=bool(parallel),
            show_progress=bool(show_progress),
            record_step_states=True,
            static_ctx=static_ctx,
            context_vec=None,
            solver=stochastic_solver,
            _execution=_execution,
        ),
    )
    rho0_batch, features_batch, rho_seq_batch, _ctx = stack_sequence_records(samples)
    return pack_dataset(rho0_batch, features_batch, rho_seq_batch)


def train_surrogate_model(
    operator: MPO,
    sim_params: AnalogSimParams,
    *,
    num_interventions: int,
    n: int,
    seed: int | None = None,
    parallel: bool = True,
    show_progress: bool = True,
    timesteps: list[float] | None = None,
    init_mode: str = "eigenstate",
    model_kwargs: dict[str, Any] | None = None,
    train_kwargs: dict[str, Any] | None = None,
    solver: StochasticSolver | None = None,
    intervention_style: str = DEFAULT_INTERVENTION_STYLE,
    _execution: ExecutionConfig | None = None,
) -> ProcessTensorSurrogate:
    """Train a surrogate model end-to-end on simulated sequence records.

    Args:
        operator: Hamiltonian MPO.
        sim_params: Analog simulation parameters.
        num_interventions: Positive integer number of intervention steps.
        n: Positive integer number of sequences to simulate for training.
        seed: Seed used for data generation RNG.
        parallel: Whether to parallelize data generation.
        show_progress: Whether to show progress bars.
        timesteps: Optional per-step durations passed to :func:`build_training_dataset`.
        init_mode: Initial-state sampling mode passed to :func:`build_training_dataset`.
        solver: Stochastic solver (``"MCWF"`` or ``"TJM"``) passed to
            :func:`build_training_dataset`; defaults to ``"MCWF"``.
        intervention_style: Training intervention style passed to :func:`build_training_dataset`.
        model_kwargs: Optional keyword arguments forwarded to :class:`ProcessTensorSurrogate`.
        train_kwargs: Optional keyword arguments forwarded to :meth:`ProcessTensorSurrogate.fit`.

    Returns:
        Trained :class:`ProcessTensorSurrogate`.
    """
    resolved_num_interventions = validate_integer(num_interventions, name="num_interventions", minimum=1)
    n_sequences = validate_integer(n, name="n", minimum=1)

    import torch  # ruff:ignore[import-outside-top-level]

    from .model import ProcessTensorSurrogate  # ruff:ignore[import-outside-top-level]

    rng = np.random.default_rng(0 if seed is None else int(seed))
    train_data = build_training_dataset(
        operator,
        sim_params,
        num_interventions=resolved_num_interventions,
        n=n_sequences,
        rng=rng,
        parallel=bool(parallel),
        show_progress=bool(show_progress),
        timesteps=timesteps,
        init_mode=init_mode,
        solver=solver,
        intervention_style=intervention_style,
        _execution=_execution,
    )

    resolved_model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    resolved_train_kwargs = {} if train_kwargs is None else dict(train_kwargs)
    device_arg = resolved_train_kwargs.pop("device", None)
    if device_arg is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg) if isinstance(device_arg, str) else device_arg
    d_e = int(train_data.tensors[0].shape[-1])
    model = ProcessTensorSurrogate(d_e=d_e, d_rho=8, **resolved_model_kwargs).to(device)

    model.fit(train_data, device=device, **resolved_train_kwargs)
    return model


__all__ = ["build_training_dataset", "train_surrogate_model"]
