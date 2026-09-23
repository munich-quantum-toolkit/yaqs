# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Operational memory characterization entry point for YAQS."""

# ruff:file-ignore[any-type, import-outside-top-level] -- lazy torch imports, unified dispatch targets

from __future__ import annotations

from concurrent.futures import CancelledError
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

from mqt.yaqs.characterization.memory.backends.tomography import DenseProcessTensor, MPOProcessTensor
from mqt.yaqs.characterization.memory.backends.tomography.constructor import (
    build_process_tensor as _build_process_tensor,
)
from mqt.yaqs.characterization.memory.backends.tomography.process_tensors import convert_probe_callable
from mqt.yaqs.characterization.memory.operational_memory.results import (
    CharacterizationResult,
    merge_cut_results,
    pack_result,
)
from mqt.yaqs.characterization.memory.operational_memory.run import run_memory_characterization
from mqt.yaqs.characterization.memory.operational_memory.samples import ProbeSet, sample_probes
from mqt.yaqs.characterization.memory.shared.encoding import (
    coerce_rho_matrix,
    normalize_backend_rho,
    pack_rho8,
    unpack_rho8,
)
from mqt.yaqs.characterization.memory.shared.interventions import (
    DEFAULT_INTERVENTION_STYLE,
    InterventionSequence,
    encode_interventions,
    expand_interventions,
    normalize_style,
)
from mqt.yaqs.characterization.memory.shared.utils import (
    DEFAULT_VECTOR_MAX_QUBITS,
    CharacterizerRepresentation,
    representation_to_solver,
    resolve_characterizer_representation,
    validate_qubit_memory_operator,
)
from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
from mqt.yaqs.core.data_structures.simulation_parameters import (
    _validate_simulation_controls,  # ruff: ignore[import-private-name] -- facade validates mutable controls before memory backends read them
)
from mqt.yaqs.core.parallel_utils import ExecutionConfig, MPContext, merge_execution_config

from .core._validation import validate_choice, validate_integer

if TYPE_CHECKING:
    from numpy.random import Generator
    from torch.utils.data import TensorDataset

    from mqt.yaqs.characterization.memory.backends.surrogates.model import ProcessTensorSurrogate
    from mqt.yaqs.characterization.memory.backends.tomography.basis import TomographyBasis
    from mqt.yaqs.core.data_structures.mpo import MPO
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


_DEFAULT_CHARACTERIZATION_PRESET = "balanced"
_CHARACTERIZER_REPRESENTATIONS: tuple[CharacterizerRepresentation, ...] = ("vector", "mps", "auto")
_CHARACTERIZATION_PRESETS: dict[str, tuple[int, int]] = {
    "quick": (8, 8),
    "balanced": (32, 32),
    "accurate": (128, 128),
}


def _resolve_probe_grid(
    preset: str,
    n_pasts: int | None,
    n_futures: int | None,
) -> tuple[int, int]:
    """Resolve past/future probe grid sizes from preset or overrides.

    Args:
        preset: ``"quick"``, ``"balanced"``, or ``"accurate"``.
        n_pasts: Optional positive integer override for the number of past probes.
        n_futures: Optional positive integer override for the number of future probes.

    Returns:
        Tuple ``(n_pasts, n_futures)``.

    Raises:
        ValueError: If ``preset`` is unknown or an explicit probe count is not positive.
    """
    if preset not in _CHARACTERIZATION_PRESETS:
        msg = f"preset must be one of {sorted(_CHARACTERIZATION_PRESETS)!r}, got {preset!r}."
        raise ValueError(msg)
    defaults = _CHARACTERIZATION_PRESETS[preset]
    return (
        validate_integer(defaults[0] if n_pasts is None else n_pasts, name="n_pasts", minimum=1),
        validate_integer(defaults[1] if n_futures is None else n_futures, name="n_futures", minimum=1),
    )


def _coerce_probe_set(probe_set: Any) -> ProbeSet | None:
    """Normalize ``probe_set=`` input for :meth:`MemoryCharacterizer.characterize`.

    Args:
        probe_set: ``None``, a :class:`CharacterizationResult`, or a :class:`ProbeSet`.

    Returns:
        :class:`ProbeSet` to reuse, or ``None`` to sample fresh probes.

    Raises:
        ValueError: If a prior result has no reusable probes or multiple cuts.
        TypeError: If ``probe_set`` is not ``None``, :class:`CharacterizationResult`, or :class:`ProbeSet`.
    """
    if probe_set is None:
        return None
    if isinstance(probe_set, CharacterizationResult):
        if len(probe_set.by_cut) != 1:
            msg = "probe_set from a prior characterize() result requires exactly one cut."
            raise ValueError(msg)
        entry = next(iter(probe_set.by_cut.values()))
        if entry.probe_set is None:
            msg = "Prior characterize() result has no stored probes to reuse."
            raise ValueError(msg)
        return entry.probe_set
    if isinstance(probe_set, ProbeSet):
        return probe_set
    msg = f"probe_set must be None, CharacterizationResult, or ProbeSet, got {type(probe_set).__name__}."
    raise TypeError(msg)


def _require_hamiltonian(hamiltonian: Hamiltonian) -> MPO:
    """Encode a :class:`Hamiltonian` as MPO or raise.

    Args:
        hamiltonian: User-facing Hamiltonian object.

    Returns:
        Encoded MPO operator.

    Raises:
        TypeError: If ``hamiltonian`` is not a :class:`Hamiltonian`.
    """
    if not isinstance(hamiltonian, Hamiltonian):
        msg = "Pass a Hamiltonian; use Hamiltonian.ising(...) or Hamiltonian(...)."
        raise TypeError(msg)
    hamiltonian.ensure_mpo()
    operator = hamiltonian.mpo
    validate_qubit_memory_operator(operator)
    return operator


def _resolve_num_interventions(target: Any, num_interventions: int | None) -> int:
    """Infer ``num_interventions`` from an explicit value or process-tensor/surrogate target.

    Args:
        target: Process tensor, surrogate, or other characterized object.
        num_interventions: Optional explicit positive integer intervention count.

    Returns:
        Resolved ``num_interventions``.

    Raises:
        ValueError: If the value cannot be inferred or is not positive.
    """
    if num_interventions is None:
        k_attr = getattr(target, "_num_interventions_for_probe", None)
        if not callable(k_attr):
            msg = "num_interventions must be provided when the target does not define _num_interventions_for_probe()."
            raise ValueError(msg)
        num_interventions = k_attr()
    return validate_integer(num_interventions, name="num_interventions", minimum=1)


def _default_cut(num_interventions: int, cut: int | None) -> int:
    """Resolve causal cut, defaulting to the interior cut ``(num_interventions + 1) // 2``.

    Args:
        num_interventions: Positive integer intervention sequence length.
        cut: Optional integer cut in ``[1, num_interventions]``.

    Returns:
        Valid cut in ``[1, num_interventions]``.

    Raises:
        ValueError: If ``num_interventions`` is not positive or the resolved cut is out of range.
    """
    resolved_num_interventions = validate_integer(num_interventions, name="num_interventions", minimum=1)
    c = (resolved_num_interventions + 1) // 2 if cut is None else validate_integer(cut, name="cut", minimum=1)
    if not (1 <= c <= resolved_num_interventions):
        msg = f"cut must satisfy 1 <= cut <= num_interventions ({resolved_num_interventions}), got {c}."
        raise ValueError(msg)
    return c


def _matches_hamiltonian(target: Any) -> bool:
    """Return whether ``target`` is a Hamiltonian characterize/predict target.

    Args:
        target: Object passed to :meth:`MemoryCharacterizer.characterize` or :meth:`MemoryCharacterizer.predict`.

    Returns:
        ``True`` if ``target`` is a :class:`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian`.
    """
    return isinstance(target, Hamiltonian)


def _matches_process_tensor(target: Any) -> bool:
    """Return whether ``target`` is a reference process tensor predict target.

    Args:
        target: Object passed to :meth:`MemoryCharacterizer.predict` or info-theory helpers.

    Returns:
        ``True`` if ``target`` is a :class:`DenseProcessTensor` or :class:`MPOProcessTensor`.
    """
    return isinstance(target, (DenseProcessTensor, MPOProcessTensor))


class MemoryCharacterizer:
    """Entry point for operational memory workflows.

    **Build:** :meth:`train`, :meth:`sample` (advanced), :meth:`build_process_tensor`

    **Use:** :meth:`predict` (surrogate or reference process-tensor dynamics), :meth:`characterize` (memory metrics),
    :meth:`compute_qmi`, :meth:`compute_cmi` (reference process-tensor information metrics)

    Hamiltonian-based workflows currently support qubit Hamiltonians only.

    Attributes:
        parallel: Whether sequence simulations run in parallel via a process pool.
        max_workers: Maximum worker processes when ``parallel=True``.
        show_progress: Whether to display a tqdm progress bar.
        representation: ``"vector"`` (MCWF), ``"mps"`` (TJM), or ``"auto"``.
        vector_max_qubits: Auto cutover: vector up to this many qubits, then mps.
        mp_context: Multiprocessing context.
        max_retries: Maximum retry attempts for transient worker errors.
        retry_exceptions: Exception types that trigger a retry.
    """

    def __init__(
        self,
        *,
        parallel: bool = True,
        max_workers: int | None = None,
        show_progress: bool = True,
        representation: CharacterizerRepresentation = "auto",
        vector_max_qubits: int = DEFAULT_VECTOR_MAX_QUBITS,
        mp_context: MPContext = "auto",
        max_retries: int = 10,
        retry_exceptions: tuple[type[BaseException], ...] = (CancelledError, TimeoutError, OSError),
    ) -> None:
        """Configure execution and representation defaults for characterization workflows.

        Args:
            parallel: Boolean that enables parallel sequence simulation.
            max_workers: Positive worker-process cap, or ``None`` for the default.
            show_progress: Boolean that controls tqdm progress bars.
            representation: ``"vector"``, ``"mps"``, or ``"auto"`` stochastic solver choice.
            vector_max_qubits: Non-negative auto cutover from vector to MPS simulation.
            mp_context: ``"auto"``, ``"fork"``, or ``"spawn"`` multiprocessing start method.
            max_retries: Non-negative retry count for transient worker failures.
            retry_exceptions: Exception types that trigger a worker retry.
        """
        self._execution = ExecutionConfig(
            parallel=parallel,
            max_workers=max_workers,
            show_progress=show_progress,
            mp_context=mp_context,
            max_retries=max_retries,
            retry_exceptions=retry_exceptions,
        )
        self.representation = validate_choice(
            representation,
            name="representation",
            allowed=_CHARACTERIZER_REPRESENTATIONS,
        )
        self.vector_max_qubits = validate_integer(vector_max_qubits, name="vector_max_qubits", minimum=0)

    @property
    def parallel(self) -> bool:
        """Whether parallel sequence simulation is enabled."""
        return self._execution.parallel

    @property
    def max_workers(self) -> int:
        """Resolved worker-process cap for parallel sequence jobs."""
        return self._execution.resolved_max_workers()

    @property
    def show_progress(self) -> bool:
        """Whether progress bars are shown during sequence simulation."""
        return self._execution.show_progress

    @property
    def mp_context(self) -> MPContext:
        """Multiprocessing context used for worker pools."""
        return self._execution.mp_context

    @property
    def max_retries(self) -> int:
        """Maximum retry attempts for transient worker failures."""
        return self._execution.max_retries

    @property
    def retry_exceptions(self) -> tuple[type[BaseException], ...]:
        """Exception types that trigger a worker retry."""
        return self._execution.retry_exceptions

    def _solver_for(self, hamiltonian: Hamiltonian) -> Literal["MCWF", "TJM"]:
        """Resolve stochastic solver for a Hamiltonian under this characterizer's representation.

        Returns:
            ``"MCWF"`` or ``"TJM"`` from the resolved characterizer representation.
        """
        rep = resolve_characterizer_representation(
            hamiltonian.length,
            self.representation,
            vector_max_qubits=self.vector_max_qubits,
        )
        return representation_to_solver(rep)

    def build_process_tensor(
        self,
        hamiltonian: Hamiltonian,
        sim_params: AnalogSimParams,
        timesteps: list[float] | None = None,
        *,
        noise_model: NoiseModel | None = None,
        num_trajectories: int = 100,
        basis: TomographyBasis = "tetrahedral",
        basis_seed: int | None = None,
        return_type: Literal["dense", "mpo"] = "mpo",
        check: bool = True,
        atol: float = 1e-8,
        compress_every: int = 16,
        tol: float = 1e-12,
        max_bond_dim: int | None = None,
        n_sweeps: int = 2,
        parallel: bool | None = None,
        initial_rho: np.ndarray | None = None,
        initial_rho_atol: float = 1e-8,
    ) -> DenseProcessTensor | MPOProcessTensor:
        """Build a process tensor via dense tomography or direct MPO construction.

        - ``return_type="mpo"`` (default): direct MPO construction (noiseless only; the uncapped
          path grows as ``16**num_interventions``).
        - ``return_type="dense"``: exhaustive tomography (scales as ``16**num_interventions``;
          supports ``noise_model``).

        Args:
            hamiltonian: System Hamiltonian.
            sim_params: Analog simulation parameters.
            timesteps: Optional process-tensor schedule evolution durations (length
                ``num_interventions + 1``; defaults to ``[dt, dt]`` for one intervention leg).
            noise_model: Optional noise model (dense tomography only).
            num_trajectories: Positive integer Monte Carlo trajectories per tomography sample
                (dense only).
            basis: Intervention / Choi basis name.
            basis_seed: Optional RNG seed for basis construction.
            return_type: ``"mpo"`` (direct construction, default) or ``"dense"`` (tomography).
            check: Whether to validate CPTP properties during dense construction.
            atol: CPTP check tolerance.
            compress_every: Positive integer interval for compressing accumulated direct-MPO terms.
            tol: MPO compression tolerance.
            max_bond_dim: Optional positive integer cap on the branch ensemble and MPO bond dimension
                for direct construction. The supported default, ``None``, retains all branches. A
                finite cap can violate process-tensor semantics, positivity, and causal normalization.
            n_sweeps: Non-negative integer number of MPO compression sweeps.
            parallel: Override instance parallel setting for dense tomography or MPO construction.
            initial_rho: Optional expected site-0 reference after ``U_0``; validated when provided.
            initial_rho_atol: Tolerance for optional ``initial_rho`` validation.

        Returns:
            Dense or MPO process tensor depending on ``return_type``.
        """
        _validate_simulation_controls(sim_params)
        resolved_num_trajectories = num_trajectories
        resolved_max_bond_dim = max_bond_dim
        resolved_compress_every = compress_every
        resolved_n_sweeps = n_sweeps
        if return_type == "dense":
            resolved_num_trajectories = validate_integer(
                num_trajectories,
                name="num_trajectories",
                minimum=1,
            )
        elif return_type == "mpo":
            resolved_max_bond_dim = (
                None if max_bond_dim is None else validate_integer(max_bond_dim, name="max_bond_dim", minimum=1)
            )
            resolved_compress_every = validate_integer(compress_every, name="compress_every", minimum=1)
            resolved_n_sweeps = validate_integer(n_sweeps, name="n_sweeps", minimum=0)

        operator = _require_hamiltonian(hamiltonian)
        execution = self._execution if parallel is None else merge_execution_config(self._execution, parallel=parallel)
        return _build_process_tensor(
            operator,
            sim_params,
            timesteps,
            noise_model=noise_model,
            num_trajectories=resolved_num_trajectories,
            basis=basis,
            basis_seed=basis_seed,
            return_type=return_type,
            check=check,
            atol=atol,
            compress_every=resolved_compress_every,
            tol=tol,
            max_bond_dim=resolved_max_bond_dim,
            n_sweeps=resolved_n_sweeps,
            solver=self._solver_for(hamiltonian),
            parallel=execution.parallel,
            initial_rho=initial_rho,
            initial_rho_atol=initial_rho_atol,
            _execution=execution,
        )

    def sample(
        self,
        hamiltonian: Hamiltonian,
        sim_params: AnalogSimParams,
        *,
        num_interventions: int,
        n: int,
        rng: Generator | None = None,
        seed: int | None = None,
        timesteps: list[float] | None = None,
        init_mode: str = "eigenstate",
        intervention_style: str = DEFAULT_INTERVENTION_STYLE,
        parallel: bool | None = None,
        show_progress: bool | None = None,
    ) -> TensorDataset:
        """Sample intervention sequences for surrogate training (advanced).

        Args:
            hamiltonian: System Hamiltonian.
            sim_params: Analog simulation parameters.
            num_interventions: Positive integer number of intervention steps per sequence.
            n: Positive integer number of training sequences.
            rng: Optional RNG (overrides ``seed``).
            seed: Optional seed when ``rng`` is omitted.
            timesteps: Optional process-tensor schedule of length ``num_interventions + 1``.
            init_mode: Initial-state sampling mode for training sequences.
            intervention_style: ``"haar"``, ``"clifford"``, or ``"measure_prepare"``.
            parallel: Override instance parallel setting.
            show_progress: Override instance progress-bar setting.

        Returns:
            PyTorch ``TensorDataset`` with ``(E_features, rho0, rho_seq)`` tensors.

        """
        resolved_num_interventions = validate_integer(num_interventions, name="num_interventions", minimum=1)
        resolved_n = validate_integer(n, name="n", minimum=1)
        operator = _require_hamiltonian(hamiltonian)
        from mqt.yaqs.characterization.memory.backends.surrogates.workflow import (
            build_training_dataset as _build_training_dataset,
        )

        return _build_training_dataset(
            operator,
            sim_params,
            num_interventions=resolved_num_interventions,
            n=resolved_n,
            rng=rng,
            seed=seed,
            timesteps=timesteps,
            init_mode=init_mode,
            solver=self._solver_for(hamiltonian),
            intervention_style=intervention_style,
            parallel=self._execution.parallel if parallel is None else parallel,
            show_progress=self._execution.show_progress if show_progress is None else show_progress,
            _execution=self._execution,
        )

    def train(
        self,
        hamiltonian: Hamiltonian,
        sim_params: AnalogSimParams,
        *,
        num_interventions: int,
        n: int,
        seed: int | None = None,
        timesteps: list[float] | None = None,
        init_mode: str = "eigenstate",
        intervention_style: str = DEFAULT_INTERVENTION_STYLE,
        model_kwargs: dict | None = None,
        train_kwargs: dict | None = None,
        parallel: bool | None = None,
        show_progress: bool | None = None,
    ) -> ProcessTensorSurrogate:
        """Train a Transformer surrogate on simulated intervention sequences.

        Args:
            hamiltonian: System Hamiltonian.
            sim_params: Analog simulation parameters.
            num_interventions: Positive integer training sequence length (stored on the model).
            n: Positive integer number of training sequences.
            seed: Optional RNG seed for data sampling and weight init.
            timesteps: Optional process-tensor schedule of length ``num_interventions + 1``.
            init_mode: Initial-state sampling mode for training sequences.
            intervention_style: Training intervention style.
            model_kwargs: Optional overrides for :class:`ProcessTensorSurrogate` construction.
            train_kwargs: Optional overrides for the training loop.
            parallel: Override instance parallel setting.
            show_progress: Override instance progress-bar setting.

        Returns:
            Trained :class:`~mqt.yaqs.characterization.memory.backends.surrogates.model.ProcessTensorSurrogate`.

        """
        resolved_num_interventions = validate_integer(num_interventions, name="num_interventions", minimum=1)
        resolved_n = validate_integer(n, name="n", minimum=1)
        operator = _require_hamiltonian(hamiltonian)
        from mqt.yaqs.characterization.memory.backends.surrogates.workflow import (
            train_surrogate_model as _train_surrogate_model,
        )

        return _train_surrogate_model(
            operator,
            sim_params,
            num_interventions=resolved_num_interventions,
            n=resolved_n,
            seed=seed,
            timesteps=timesteps,
            init_mode=init_mode,
            intervention_style=intervention_style,
            solver=self._solver_for(hamiltonian),
            model_kwargs=model_kwargs,
            train_kwargs=train_kwargs,
            parallel=self._execution.parallel if parallel is None else parallel,
            show_progress=self._execution.show_progress if show_progress is None else show_progress,
            _execution=self._execution,
        )

    @overload
    def characterize(
        self,
        hamiltonian: Hamiltonian,
        sim_params: AnalogSimParams,
        /,
        *,
        num_interventions: int,
        cut: int | None = None,
        cuts: Literal["all"] | list[int] | None = None,
        preset: str = _DEFAULT_CHARACTERIZATION_PRESET,
        n_pasts: int | None = None,
        n_futures: int | None = None,
        intervention_style: str = DEFAULT_INTERVENTION_STYLE,
        rng: Generator | None = None,
        probe_set: Any | None = None,
        initial_psi: np.ndarray | None = None,
        delay: int | None = None,
    ) -> CharacterizationResult: ...

    @overload
    def characterize(
        self,
        target: Any,
        /,
        *,
        cut: int | None = None,
        cuts: Literal["all"] | list[int] | None = None,
        num_interventions: int | None = None,
        preset: str = _DEFAULT_CHARACTERIZATION_PRESET,
        n_pasts: int | None = None,
        n_futures: int | None = None,
        intervention_style: str = DEFAULT_INTERVENTION_STYLE,
        rng: Generator | None = None,
        probe_set: Any | None = None,
        initial_rho: np.ndarray | None = None,
        parallel: bool | None = None,
        delay: int | None = None,
    ) -> CharacterizationResult: ...

    def characterize(
        self,
        target: Any,
        sim_params: AnalogSimParams | None = None,
        /,
        *,
        num_interventions: int | None = None,
        cut: int | None = None,
        cuts: Literal["all"] | list[int] | None = None,
        preset: str = _DEFAULT_CHARACTERIZATION_PRESET,
        n_pasts: int | None = None,
        n_futures: int | None = None,
        intervention_style: str = DEFAULT_INTERVENTION_STYLE,
        rng: Generator | None = None,
        probe_set: Any | None = None,
        initial_psi: np.ndarray | None = None,
        initial_rho: np.ndarray | None = None,
        parallel: bool | None = None,
        delay: int | None = None,
        **probe_kwargs: Any,
    ) -> CharacterizationResult:
        """Return operational memory diagnostics for a Hamiltonian, surrogate, or process tensor.

        For a Hamiltonian, pass ``sim_params`` and ``num_interventions``. For process
        tensors and surrogates, ``num_interventions`` is inferred from the target when
        omitted. Default interior cut is ``(num_interventions + 1) // 2``.

        Args:
            target: Hamiltonian, trained surrogate, or reference process tensor.
            sim_params: Required for Hamiltonian targets only.
            num_interventions: Positive integer base split-cut sequence length (required for
                Hamiltonian targets). An explicit ``delay`` adds ``delay + 1`` physical interventions.
            cut: Single integer causal cut in ``[1, num_interventions]``; mutually exclusive with
                ``cuts``.
            cuts: ``"all"`` or a list of integer cuts in ``[1, num_interventions]`` for multi-cut
                Hamiltonian sweeps.
            preset: Probe-grid preset (``"quick"``, ``"balanced"``, ``"accurate"``).
            n_pasts: Optional positive integer number of past probes.
            n_futures: Optional positive integer number of future probes.
            intervention_style: ``"haar"``, ``"clifford"``, or ``"measure_prepare"``.
            rng: RNG for probe sampling.
            probe_set: Prior :class:`CharacterizationResult` or :class:`ProbeSet` to reuse.
            initial_psi: Optional initial state for Hamiltonian exact simulation.
            initial_rho: Site-0 state after the initial evolution segment and before the first
                intervention. Required for surrogate targets and unsupported for Hamiltonian or
                process-tensor targets.
            parallel: Override parallelism for process-tensor/surrogate probing.
            delay: Optional non-negative integer conditioned-reset bridge length (Hamiltonian only).
                ``None`` uses the standard causal break. Every integer value from zero uses the
                paper's separate boundary interventions.
            **probe_kwargs: Unsupported; pass explicit keyword arguments instead.

        Returns:
            Diagnostics with per-cut entropy, modes, spectrum, and stored probes.

        Raises:
            TypeError: If a Hamiltonian is given without ``sim_params``, or an explicit or
                inferred size is not an integer.
            ValueError: If ``num_interventions`` is missing or not positive, a probe count is
                not positive, a cut is out of range, both ``cut`` and ``cuts`` are given,
                ``cuts`` is an empty list, ``probe_set`` is reused across multiple cuts, or a
                conditioned-reset delay is invalid or unsupported by the target.
        """
        n_p, n_f = _resolve_probe_grid(preset, n_pasts, n_futures)
        resolved_delay = None if delay is None else validate_integer(delay, name="delay", minimum=0)
        if "intervention_mode" in probe_kwargs or "unitary_ensemble" in probe_kwargs:
            msg = "Use intervention_style= instead of intervention_mode= / unitary_ensemble=."
            raise ValueError(msg)
        if probe_kwargs:
            unknown = ", ".join(sorted(probe_kwargs))
            msg = f"Unsupported probe_kwargs: {unknown}."
            raise ValueError(msg)
        resolved_style = normalize_style(intervention_style)
        resolved_probe_set = _coerce_probe_set(probe_set)

        if initial_rho is not None and (_matches_hamiltonian(target) or _matches_process_tensor(target)):
            msg = "initial_rho is supported only for surrogate characterization."
            raise ValueError(msg)

        if resolved_delay is not None and not _matches_hamiltonian(target):
            msg = "delay is supported for Hamiltonian characterize() only."
            raise ValueError(msg)

        if _matches_hamiltonian(target):
            if sim_params is None:
                msg = "characterize(hamiltonian, sim_params, num_interventions=...) requires AnalogSimParams."
                raise TypeError(msg)
            _validate_simulation_controls(sim_params)
            if num_interventions is None:
                msg = "characterize(hamiltonian, sim_params, ...) requires num_interventions=."
                raise ValueError(msg)
            resolved_num_interventions = validate_integer(num_interventions, name="num_interventions", minimum=1)
            return self._characterize_hamiltonian(
                target,
                sim_params,
                num_interventions=resolved_num_interventions,
                cut=cut,
                cuts=cuts,
                n_pasts=n_p,
                n_futures=n_f,
                rng=rng,
                probe_set=resolved_probe_set,
                initial_psi=initial_psi,
                intervention_style=resolved_style,
                delay=resolved_delay,
            )

        resolved_num_interventions = _resolve_num_interventions(target, num_interventions)
        cut_list = self._resolve_cut_list(resolved_num_interventions, cut=cut, cuts=cuts)
        if resolved_probe_set is not None and len(cut_list) > 1:
            msg = "probe_set cannot be reused across multiple cuts; omit probe_set for multi-cut characterize()."
            raise ValueError(msg)
        if len(cut_list) == 1:
            return self._characterize_target(
                target,
                cut=cut_list[0],
                num_interventions=resolved_num_interventions,
                n_pasts=n_p,
                n_futures=n_f,
                rng=rng,
                probe_set=resolved_probe_set,
                initial_rho=initial_rho,
                parallel=parallel,
                intervention_style=resolved_style,
                delay=resolved_delay,
            )
        parts: dict[int, CharacterizationResult] = {}
        for c in cut_list:
            parts[c] = self._characterize_target(
                target,
                cut=c,
                num_interventions=resolved_num_interventions,
                n_pasts=n_p,
                n_futures=n_f,
                rng=rng,
                probe_set=None,
                initial_rho=initial_rho,
                parallel=parallel,
                intervention_style=resolved_style,
                delay=resolved_delay,
            )
        return merge_cut_results(parts)

    @staticmethod
    def compute_qmi(
        process_tensor: DenseProcessTensor | MPOProcessTensor,
        /,
        *,
        past: str = "all",
        base: int = 2,
    ) -> float:
        """Compute quantum mutual information from a reference process tensor.

        For an MPO input, this method densifies the complete process tensor. For
        ``k`` intervention legs, the dense complex matrix uses ``64 * 16**k``
        bytes before analysis workspace.

        Args:
            process_tensor: Dense or MPO reference process tensor.
            past: Past legs to include: ``"all"``, ``"first"``, or ``"last"``.
            base: Log base for entropy.

        Returns:
            Quantum mutual information between the final site and the selected past legs.

        Raises:
            TypeError: If ``process_tensor`` is not a reference process tensor.
        """
        if not _matches_process_tensor(process_tensor):
            msg = f"compute_qmi requires a reference process tensor, got {type(process_tensor).__name__}."
            raise TypeError(msg)
        return process_tensor.qmi(base=base, past=past)

    @staticmethod
    def compute_cmi(
        process_tensor: DenseProcessTensor | MPOProcessTensor,
        /,
        *,
        base: int = 2,
    ) -> float:
        r"""Compute conditional mutual information from a reference process tensor.

        For an MPO input, this method densifies the complete process tensor. For
        ``k`` intervention legs, the dense complex matrix uses ``64 * 16**k``
        bytes before analysis workspace.

        Args:
            process_tensor: Dense or MPO reference process tensor.
            base: Log base for entropy.

        Returns:
            Conditional mutual information :math:`I(F : P_{<k} \\mid P_k)`.

        Raises:
            TypeError: If ``process_tensor`` is not a reference process tensor.
        """
        if not _matches_process_tensor(process_tensor):
            msg = f"compute_cmi requires a reference process tensor, got {type(process_tensor).__name__}."
            raise TypeError(msg)
        return process_tensor.cmi(base=base)

    @staticmethod
    def _resolve_cut_list(
        num_interventions: int,
        *,
        cut: int | None,
        cuts: Literal["all"] | list[int] | None,
    ) -> list[int]:
        """Resolve the list of cuts to characterize.

        Args:
            num_interventions: Positive integer intervention sequence length.
            cut: Optional integer cut in ``[1, num_interventions]``.
            cuts: ``"all"`` or an explicit list of integer cuts in ``[1, num_interventions]``.

        Returns:
            List of cut indices to evaluate.

        Raises:
            ValueError: If both ``cut`` and ``cuts`` are provided, ``cuts`` is an
                empty list, ``num_interventions`` is not positive, or a cut is out of range.
        """
        resolved_num_interventions = validate_integer(num_interventions, name="num_interventions", minimum=1)
        if cuts is not None and cut is not None:
            msg = "Specify only one of cut=... or cuts=..., not both."
            raise ValueError(msg)
        if cuts is not None:
            if cuts != "all" and len(cuts) == 0:
                msg = "cuts must be 'all' or a non-empty list of cut indices."
                raise ValueError(msg)
            if cuts == "all":
                return list(range(1, resolved_num_interventions + 1))
            return [_default_cut(resolved_num_interventions, c) for c in cuts]
        if cut is not None:
            return [_default_cut(resolved_num_interventions, cut)]
        return [_default_cut(resolved_num_interventions, None)]

    def _characterize_target(
        self,
        target: Any,
        *,
        cut: int,
        num_interventions: int,
        n_pasts: int,
        n_futures: int,
        rng: Generator | None,
        probe_set: ProbeSet | None,
        initial_rho: np.ndarray | None,
        parallel: bool | None,
        intervention_style: str,
        delay: int | None = None,
    ) -> CharacterizationResult:
        """Characterize a process tensor or surrogate via internal split-cut probing.

        A conditioned-reset ``delay`` is rejected in :meth:`characterize` before this path is reached.

        Returns:
            Single-cut :class:`~mqt.yaqs.characterization.memory.operational_memory.results.CharacterizationResult`.
        """
        resolved_cut = _default_cut(num_interventions, cut)
        out = run_memory_characterization(
            process=target,
            cut=resolved_cut,
            num_interventions=num_interventions,
            n_pasts=n_pasts,
            n_futures=n_futures,
            rng=rng,
            probe_set=probe_set,
            initial_rho=initial_rho,
            parallel=parallel if parallel is not None else self._execution.parallel,
            delay=delay,
            intervention_style=intervention_style,
        )
        return pack_result(out, cut=resolved_cut)

    def _characterize_hamiltonian(
        self,
        hamiltonian: Hamiltonian,
        sim_params: AnalogSimParams,
        *,
        num_interventions: int,
        cut: int | None,
        cuts: Literal["all"] | list[int] | None,
        n_pasts: int,
        n_futures: int,
        rng: Generator | None,
        probe_set: ProbeSet | None,
        initial_psi: np.ndarray | None,
        intervention_style: str,
        delay: int | None = None,
    ) -> CharacterizationResult:
        """Characterize a Hamiltonian via exact stochastic sequences and branch weights.

        Returns:
            Single- or multi-cut
            :class:`~mqt.yaqs.characterization.memory.operational_memory.results.CharacterizationResult`.

        Raises:
            ValueError: If ``probe_set`` is given for a multi-cut request.
        """
        cut_list = MemoryCharacterizer._resolve_cut_list(num_interventions, cut=cut, cuts=cuts)
        if probe_set is not None and len(cut_list) > 1:
            msg = "probe_set cannot be reused across multiple cuts; omit probe_set for multi-cut characterize()."
            raise ValueError(msg)
        from mqt.yaqs.characterization.memory.backends.exact import ExactBackend

        operator = _require_hamiltonian(hamiltonian)
        solver = self._solver_for(hamiltonian)
        backend = ExactBackend(
            operator=operator,
            sim_params=sim_params,
            initial_psi=initial_psi,
            parallel=self._execution.parallel,
            show_progress=self._execution.show_progress,
            solver=solver,
            _execution=self._execution,
        )
        parts: dict[int, CharacterizationResult] = {}
        for c in cut_list:
            resolved_cut = _default_cut(num_interventions, c)
            local_probe_set = probe_set
            if local_probe_set is None:
                local_rng = rng if rng is not None else np.random.default_rng()
                local_probe_set = sample_probes(
                    cut=resolved_cut,
                    num_interventions=num_interventions,
                    n_pasts=n_pasts,
                    n_futures=n_futures,
                    rng=local_rng,
                    intervention_style=intervention_style,
                )
            out = run_memory_characterization(
                process=backend,
                cut=resolved_cut,
                num_interventions=num_interventions,
                probe_set=local_probe_set,
                delay=delay,
            )
            parts[resolved_cut] = pack_result(out, cut=resolved_cut)
        return merge_cut_results(parts) if len(parts) > 1 else parts[cut_list[0]]

    def predict(  # ruff:ignore[no-self-use] -- public instance API
        self,
        target: Any,
        rho0: np.ndarray,
        sequence: InterventionSequence,
        /,
        *,
        num_interventions: int | None = None,
        return_sequence: bool = False,
        rng: Generator | None = None,
    ) -> np.ndarray:
        r"""Predict site-0 reduced-state dynamics under an intervention sequence.

        Supports trained surrogates and reference process tensors. For process tensors,
        ``rho0`` must match the stored reference initial state (site-0 density matrix after
        ``U_0`` from ``|0\\rangle^{\\otimes L}``).

        Args:
            target: Trained surrogate or reference process tensor.
            rho0: Initial ``2 x 2`` density matrix or packed length-8 vector.
            sequence: Intervention kind string, per-slot list, or expanded sequence.
            num_interventions: Positive integer sequence length; inferred from ``target`` when omitted.
            return_sequence: If True, return the full ``num_interventions``-step trajectory
                instead of the final state only.
            rng: RNG for stochastic intervention sampling.

        Returns:
            Final (or full) site-0 reduced density matrix.

        Raises:
            TypeError: If the explicit or inferred ``num_interventions`` is not an integer,
                or ``target`` does not support surrogate-style prediction.
            ValueError: If ``num_interventions`` is not positive or
                ``return_sequence=True`` for a process-tensor target.
        """
        resolved_num_interventions = _resolve_num_interventions(target, num_interventions)
        if _matches_process_tensor(target) and return_sequence:
            msg = "return_sequence=True is not supported for process tensor targets."
            raise ValueError(msg)
        local_rng = rng if rng is not None else np.random.default_rng()
        seq = sequence

        if _matches_process_tensor(target):
            rho_mat = coerce_rho_matrix(rho0)
            target.check_initial_rho(rho_mat)
            if isinstance(seq, str):
                slots = expand_interventions(seq, num_interventions=resolved_num_interventions, _rng=local_rng)
            else:
                slots = list(seq)
            steps, _ = encode_interventions(slots, num_interventions=resolved_num_interventions, rng=local_rng)
            callables = [convert_probe_callable(s) for s in steps]
            rho_out = target.predict(callables)
            return np.asarray(rho_out, dtype=np.complex128)

        rho_mat = coerce_rho_matrix(rho0)
        predict_fn = getattr(target, "predict", None)
        if not callable(predict_fn):
            msg = f"Unsupported predict target type: {type(target).__name__}"
            raise TypeError(msg)
        _steps, e_features = encode_interventions(seq, num_interventions=resolved_num_interventions, rng=local_rng)
        packed0 = pack_rho8(normalize_backend_rho(rho_mat)).astype(np.float32)
        pred = predict_fn(
            e_features[np.newaxis, ...],
            packed0[np.newaxis, ...],
            return_numpy=True,
        )
        if return_sequence:
            return np.stack([unpack_rho8(row) for row in pred[0]], axis=0).astype(np.complex128)
        return unpack_rho8(pred[0, -1, :])


__all__ = ["MemoryCharacterizer"]
