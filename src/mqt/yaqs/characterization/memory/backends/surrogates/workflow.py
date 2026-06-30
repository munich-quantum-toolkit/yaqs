# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Surrogate workflow: sample training data and train models.

**Public API** (see ``__all__``): :func:`sample_train_dataset`, :func:`train_surrogate_model`.

:func:`sample_train_dataset` returns a :class:`~torch.utils.data.TensorDataset` for
:meth:`~mqt.yaqs.characterization.memory.backends.surrogates.model.ProcessTensorSurrogate.fit`.

Sequence simulation is delegated to
:mod:`~mqt.yaqs.characterization.memory.backends.sequences`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    import types

    from torch.utils.data import TensorDataset

    from mqt.yaqs.analog.mcwf import MCWFContext
    from mqt.yaqs.characterization.memory.backends.surrogates.model import ProcessTensorSurrogate
    from mqt.yaqs.core.data_structures.mpo import MPO
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
    from mqt.yaqs.core.parallel_utils import ExecutionConfig

from ...operational_memory.interventions import DEFAULT_INTERVENTION_STYLE, normalize_style, sample_train_interventions
from ...shared.utils import StochasticSolver, make_mcwf_static_context, resolve_stochastic_solver
from ..sequences.workflow import simulate_sequences
from .data import SeqTrace, stack_traces
from .utils import sample_density_matrix, sample_initial_psi


def _require_torch() -> types.ModuleType:
    """Import PyTorch or raise with an installation hint.

    Returns:
        The ``torch`` module.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        msg = "PyTorch is required for surrogate training; install with `uv sync --extra torch`."
        raise ImportError(msg) from exc
    return torch


def pack_dataset(
    rho0: np.ndarray,
    e_features: np.ndarray,
    rho_seq: np.ndarray,
) -> TensorDataset:
    """Pack sequence trace arrays into a PyTorch :class:`~torch.utils.data.TensorDataset`.

    Args:
        rho0: Array of shape ``(N, 8)``.
        e_features: Array of shape ``(N, K, d_e)``.
        rho_seq: Array of shape ``(N, K, 8)``.

    Returns:
        TensorDataset with tensors ``(e_features, rho0, rho_seq)`` in that order.
    """
    import torch  # noqa: PLC0415
    from torch.utils.data import TensorDataset  # noqa: PLC0415

    return TensorDataset(
        torch.as_tensor(e_features, dtype=torch.float32),
        torch.as_tensor(rho0, dtype=torch.float32),
        torch.as_tensor(rho_seq, dtype=torch.float32),
    )


def sample_train_dataset(
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
    """Sample intervention sequences and pack a surrogate training dataset.

    Args:
        operator: Hamiltonian MPO. The chain length is inferred from ``operator.length``.
        sim_params: Analog simulation parameters.
        k: Number of intervention steps.
        n: Number of sampled sequences.
        rng: Optional RNG (overrides ``seed`` if provided).
        seed: Optional seed used to create a default RNG.
        parallel: Whether to parallelize over sequences.
        show_progress: Whether to show progress bars.
        timesteps: Optional comb evolution durations (defaults to ``[sim_params.dt] * (k+1)``).
        init_mode: Initial-state sampling mode (see :func:`sample_initial_psi`).
        solver: Optional stochastic solver override (``"MCWF"`` or ``"TJM"``).
        style: Training intervention style (``"haar"``, ``"clifford"``, or ``"measure_prepare"``).

    Returns:
        A :class:`~torch.utils.data.TensorDataset` with tensors ``(E_features, rho0, rho_seq)``.

    Raises:
        ValueError: If ``timesteps`` has the wrong length (must be ``k+1``).
    """
    chain_length = int(operator.length)
    if timesteps is None:
        timesteps = [float(sim_params.dt)] * (int(num_interventions) + 1)
    if len(timesteps) != int(num_interventions) + 1:
        msg = (
            f"Comb schedule: timesteps length must be num_interventions+1="
            f"{int(num_interventions) + 1}, got {len(timesteps)}."
        )
        raise ValueError(msg)

    _require_torch()
    stochastic_solver = resolve_stochastic_solver(sim_params, solver=solver)

    static_ctx: MCWFContext | None = None
    if stochastic_solver == "MCWF":
        static_ctx = make_mcwf_static_context(operator, sim_params, noise_model=None)

    if rng is None:
        rng = np.random.default_rng(0 if seed is None else int(seed))

    intervention_steps_list: list[list[Any]] = []
    initial_psis: list[np.ndarray] = []
    choi_feature_rows_per_sequence: list[np.ndarray] = []

    for _ in range(int(n)):
        rho_in = sample_density_matrix(rng)
        step_pairs, choi_rows = sample_train_interventions(
            int(num_interventions),
            normalize_style(str(intervention_style)),
            rng,
        )
        intervention_steps_list.append(step_pairs)
        choi_feature_rows_per_sequence.append(choi_rows.astype(np.float32))
        initial_psi = sample_initial_psi(rho_in, length=int(chain_length), rng=rng, init_mode=init_mode)
        if isinstance(initial_psi, tuple):
            initial_psi = initial_psi[0]
        initial_psis.append(initial_psi)

    samples = cast(
        "list[SeqTrace]",
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
    rho0_batch, features_batch, rho_seq_batch, _ctx = stack_traces(samples)
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
    """Train a surrogate model end-to-end on simulated sequence traces.

    Args:
        operator: Hamiltonian MPO.
        sim_params: Analog simulation parameters.
        k: Number of intervention steps.
        n: Number of sampled sequences.
        seed: Seed used for data generation RNG.
        parallel: Whether to parallelize data generation.
        show_progress: Whether to show progress bars.
        timesteps: Optional per-step durations passed to :func:`sample_train_dataset`.
        init_mode: Initial-state sampling mode passed to :func:`sample_train_dataset`.
        solver: Optional stochastic solver override passed to :func:`sample_train_dataset`.
        style: Training intervention style passed to :func:`sample_train_dataset`.
        model_kwargs: Optional keyword arguments forwarded to :class:`ProcessTensorSurrogate`.
        train_kwargs: Optional keyword arguments forwarded to :meth:`ProcessTensorSurrogate.fit`.

    Returns:
        Trained :class:`ProcessTensorSurrogate`.
    """
    import torch  # noqa: PLC0415

    from .model import ProcessTensorSurrogate  # noqa: PLC0415

    rng = np.random.default_rng(0 if seed is None else int(seed))
    train_data = sample_train_dataset(
        operator,
        sim_params,
        num_interventions=int(num_interventions),
        n=int(n),
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


__all__ = ["sample_train_dataset", "train_surrogate_model"]
