# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Operational memory characterization entry point for YAQS."""

# ruff: noqa: ANN401, PLC0415 -- lazy torch imports, unified dispatch targets

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
from mqt.yaqs.characterization.memory.operational_memory.run import run_operational_memory
from mqt.yaqs.characterization.memory.operational_memory.samples import sample_probes
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
    map_probe_kwargs,
)
from mqt.yaqs.characterization.memory.shared.utils import (
    DEFAULT_VECTOR_MAX_QUBITS,
    CharacterizerRepresentation,
    make_zero_psi,
    representation_to_solver,
    resolve_characterizer_representation,
)
from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
from mqt.yaqs.core.parallel_utils import ExecutionConfig, MPContext, merge_execution_config

if TYPE_CHECKING:
    from numpy.random import Generator
    from torch.utils.data import TensorDataset

    from mqt.yaqs.characterization.memory.backends.surrogates.model import ProcessTensorSurrogate
    from mqt.yaqs.characterization.memory.backends.tomography.basis import TomographyBasis
    from mqt.yaqs.characterization.memory.operational_memory.samples import ProbeSet
    from mqt.yaqs.core.data_structures.mpo import MPO
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


_DEFAULT_CHARACTERIZATION_PRESET = "balanced"
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
        n_pasts: Optional override for the number of past probes.
        n_futures: Optional override for the number of future probes.

    Returns:
        Tuple ``(n_pasts, n_futures)``.

    Raises:
        ValueError: If ``preset`` is unknown.
    """
    if preset not in _CHARACTERIZATION_PRESETS:
        msg = f"preset must be one of {sorted(_CHARACTERIZATION_PRESETS)!r}, got {preset!r}."
        raise ValueError(msg)
    defaults = _CHARACTERIZATION_PRESETS[preset]
    return (
        int(defaults[0] if n_pasts is None else n_pasts),
        int(defaults[1] if n_futures is None else n_futures),
    )


def _resolve_probe_bundle(probe_set: Any) -> ProbeSet | None:
    """Accept a prior :class:`CharacterizationResult` or internal probe bundle.

    Args:
        probe_set: ``None``, a :class:`CharacterizationResult`, or an internal probe set.

    Returns:
        Internal probe bundle, or ``None``.

    Raises:
        ValueError: If a prior result has no reusable probes or multiple cuts.
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
    return probe_set


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
    hamiltonian.ensure_encoded("mpo")
    return hamiltonian.mpo


def _resolve_num_interventions(target: Any, num_interventions: int | None) -> int:
    """Infer ``num_interventions`` from an explicit value or process-tensor/surrogate target.

    Args:
        target: Process tensor, surrogate, or other characterized object.
        num_interventions: Optional explicit intervention count.

    Returns:
        Resolved ``num_interventions``.

    Raises:
        ValueError: If ``num_interventions`` cannot be inferred from ``target``.
    """
    if num_interventions is not None:
        return int(num_interventions)
    k_attr = getattr(target, "_num_interventions_for_probe", None)
    if callable(k_attr):
        return int(k_attr())
    msg = "num_interventions must be provided when the target does not define _num_interventions_for_probe()."
    raise ValueError(msg)


def _default_cut(num_interventions: int, cut: int | None) -> int:
    """Resolve causal cut, defaulting to the interior cut ``(num_interventions + 1) // 2``.

    Args:
        num_interventions: Intervention sequence length.
        cut: Optional explicit cut.

    Returns:
        Valid cut in ``[1, num_interventions]``.

    Raises:
        ValueError: If the resolved cut is out of range.
    """
    resolved_num_interventions = int(num_interventions)
    c = (resolved_num_interventions + 1) // 2 if cut is None else int(cut)
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
            parallel: Whether to parallelize sequence simulation.
            max_workers: Cap on worker processes when ``parallel=True``.
            show_progress: Whether to show tqdm progress bars.
            representation: ``"vector"``, ``"mps"``, or ``"auto"`` stochastic solver choice.
            vector_max_qubits: Auto cutover threshold from vector to MPS simulation.
            mp_context: Multiprocessing start method.
            max_retries: Retries for transient worker failures.
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
        self.representation = representation
        self.vector_max_qubits = int(vector_max_qubits)

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
        return_type: Literal["dense", "mpo"] = "dense",
        check: bool = True,
        atol: float = 1e-8,
        compress_every: int = 100,
        tol: float = 1e-12,
        max_bond_dim: int | None = None,
        n_sweeps: int = 2,
        parallel: bool | None = None,
        initial_rho: np.ndarray | None = None,
        initial_rho_atol: float = 1e-8,
    ) -> DenseProcessTensor | MPOProcessTensor:
        """Build an exhaustive reference process tensor (validation only; scales as ``16**num_interventions``).

        Args:
            hamiltonian: System Hamiltonian.
            sim_params: Analog simulation parameters.
            timesteps: Optional process-tensor schedule evolution durations (length
                ``num_interventions + 1``; defaults to ``[dt, dt]`` for one intervention leg).
            noise_model: Optional noise model during tomography sequences.
            num_trajectories: Monte Carlo trajectories per tomography sample.
            basis: Intervention basis for process-tensor tomography.
            basis_seed: Optional RNG seed for basis construction.
            return_type: ``"dense"`` or ``"mpo"`` process-tensor storage.
            check: Whether to validate CPTP properties during construction.
            atol: CPTP check tolerance.
            compress_every: MPO compression cadence during construction.
            tol: MPO compression tolerance.
            max_bond_dim: Optional MPO bond-dimension cap.
            n_sweeps: MPO variational refinement sweeps.
            parallel: Override instance parallel setting.
            initial_rho: Optional expected site-0 reference after ``U_0``; validated when provided.
            initial_rho_atol: Tolerance for optional ``initial_rho`` validation.

        Returns:
            Dense or MPO reference process tensor for small-horizon validation.
        """
        operator = _require_hamiltonian(hamiltonian)
        execution = self._execution if parallel is None else merge_execution_config(self._execution, parallel=parallel)
        return _build_process_tensor(
            operator,
            sim_params,
            timesteps,
            noise_model=noise_model,
            num_trajectories=num_trajectories,
            basis=basis,
            basis_seed=basis_seed,
            return_type=return_type,
            check=check,
            atol=atol,
            compress_every=compress_every,
            tol=tol,
            max_bond_dim=max_bond_dim,
            n_sweeps=n_sweeps,
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
            num_interventions: Number of intervention steps per sequence.
            n: Number of training sequences.
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
        operator = _require_hamiltonian(hamiltonian)
        from mqt.yaqs.characterization.memory.backends.surrogates.workflow import (
            sample_train_dataset as _sample_train_dataset,
        )

        return _sample_train_dataset(
            operator,
            sim_params,
            num_interventions=num_interventions,
            n=n,
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
            num_interventions: Training sequence length (stored on the model).
            n: Number of training sequences.
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
        operator = _require_hamiltonian(hamiltonian)
        from mqt.yaqs.characterization.memory.backends.surrogates.workflow import (
            train_surrogate_model as _train_surrogate_model,
        )

        return _train_surrogate_model(
            operator,
            sim_params,
            num_interventions=num_interventions,
            n=n,
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
        delay: int = 0,
        **probe_kwargs: Any,
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
        parallel: bool | None = None,
        delay: int = 0,
        **probe_kwargs: Any,
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
        parallel: bool | None = None,
        delay: int = 0,
        **probe_kwargs: Any,
    ) -> CharacterizationResult:
        """Return operational memory diagnostics for a Hamiltonian, surrogate, or process tensor.

        For a Hamiltonian, pass ``sim_params`` and ``num_interventions``. For process
        tensors and surrogates, ``num_interventions`` is inferred from the target when
        omitted. Default interior cut is ``(num_interventions + 1) // 2``.

        Args:
            target: Hamiltonian, trained surrogate, or reference process tensor.
            sim_params: Required for Hamiltonian targets only.
            num_interventions: Intervention sequence length (required for Hamiltonian targets).
            cut: Single causal cut; mutually exclusive with ``cuts``.
            cuts: ``"all"`` or explicit list for multi-cut Hamiltonian sweeps.
            preset: Probe-grid preset (``"quick"``, ``"balanced"``, ``"accurate"``).
            n_pasts: Override number of past probes.
            n_futures: Override number of future probes.
            intervention_style: ``"haar"``, ``"clifford"``, or ``"measure_prepare"``.
            rng: RNG for probe sampling.
            probe_set: Prior :class:`CharacterizationResult` or internal probe bundle to reuse.
            initial_psi: Optional initial state for Hamiltonian exact simulation.
            parallel: Override parallelism for process-tensor/surrogate probing.
            delay: Soft-reset slots ``(|0>, |0>)`` inserted at the causal break (Hamiltonian only).
            **probe_kwargs: Advanced overrides forwarded to internal probe sampling.

        Returns:
            Diagnostics with per-cut entropy, modes, spectrum, and stored probes.

        Raises:
            TypeError: If a Hamiltonian is given without ``sim_params``.
            ValueError: If ``num_interventions`` is missing for a Hamiltonian target, both
                ``cut`` and ``cuts`` are given, ``cuts`` is an empty list, ``probe_set`` is
                reused across multiple cuts, ``delay > 0`` on a process-tensor/surrogate
                target, or ``delay > 0`` on a non-exact backend.
        """
        n_p, n_f = _resolve_probe_grid(preset, n_pasts, n_futures)
        probe_kw = {**map_probe_kwargs(intervention_style), **probe_kwargs}
        resolved_probe_set = _resolve_probe_bundle(probe_set)

        if delay > 0 and not _matches_hamiltonian(target):
            msg = "delay > 0 is supported for Hamiltonian characterize() only."
            raise ValueError(msg)

        if _matches_hamiltonian(target):
            if sim_params is None:
                msg = "characterize(hamiltonian, sim_params, num_interventions=...) requires AnalogSimParams."
                raise TypeError(msg)
            if num_interventions is None:
                msg = "characterize(hamiltonian, sim_params, ...) requires num_interventions=."
                raise ValueError(msg)
            return self._characterize_hamiltonian(
                target,
                sim_params,
                num_interventions=int(num_interventions),
                cut=cut,
                cuts=cuts,
                n_pasts=n_p,
                n_futures=n_f,
                rng=rng,
                probe_set=resolved_probe_set,
                initial_psi=initial_psi,
                probe_kw=probe_kw,
                delay=delay,
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
                parallel=parallel,
                probe_kw=probe_kw,
                delay=delay,
            )
        parts: dict[int, CharacterizationResult] = {}
        for c in cut_list:
            parts[int(c)] = self._characterize_target(
                target,
                cut=int(c),
                num_interventions=resolved_num_interventions,
                n_pasts=n_p,
                n_futures=n_f,
                rng=rng,
                probe_set=None,
                parallel=parallel,
                probe_kw=probe_kw,
                delay=delay,
            )
        return merge_cut_results(parts)

    @staticmethod
    def compute_qmi(
        process_tensor: DenseProcessTensor | MPOProcessTensor,
        /,
        *,
        past: str = "all",
        base: int = 2,
        check_psd: bool = False,
        assume_canonical: bool = False,
    ) -> float:
        """Compute quantum mutual information from a reference process tensor.

        Args:
            process_tensor: Dense or MPO reference process tensor.
            past: Past legs to include: ``"all"``, ``"first"``, or ``"last"``.
            base: Log base for entropy.
            check_psd: If ``True``, validate PSD before normalizing.
            assume_canonical: If ``True``, treat the stored matrix as already canonicalized.

        Returns:
            Quantum mutual information between the final site and the selected past legs.

        Raises:
            TypeError: If ``process_tensor`` is not a reference process tensor.
        """
        if not _matches_process_tensor(process_tensor):
            msg = f"compute_qmi requires a reference process tensor, got {type(process_tensor).__name__}."
            raise TypeError(msg)
        return process_tensor.qmi(
            base=base,
            past=past,
            check_psd=check_psd,
            assume_canonical=assume_canonical,
        )

    @staticmethod
    def compute_cmi(
        process_tensor: DenseProcessTensor | MPOProcessTensor,
        /,
        *,
        base: int = 2,
        check_psd: bool = False,
        assume_canonical: bool = False,
    ) -> float:
        r"""Compute conditional mutual information from a reference process tensor.

        Args:
            process_tensor: Dense or MPO reference process tensor.
            base: Log base for entropy.
            check_psd: Passed through to the process-tensor implementation.
            assume_canonical: If ``True``, treat the stored matrix as already canonicalized.

        Returns:
            Conditional mutual information :math:`I(F : P_{<k} \\mid P_k)`.

        Raises:
            TypeError: If ``process_tensor`` is not a reference process tensor.
        """
        if not _matches_process_tensor(process_tensor):
            msg = f"compute_cmi requires a reference process tensor, got {type(process_tensor).__name__}."
            raise TypeError(msg)
        return process_tensor.cmi(
            base=base,
            check_psd=check_psd,
            assume_canonical=assume_canonical,
        )

    @staticmethod
    def _resolve_cut_list(
        num_interventions: int,
        *,
        cut: int | None,
        cuts: Literal["all"] | list[int] | None,
    ) -> list[int]:
        """Resolve the list of cuts to characterize.

        Args:
            num_interventions: Intervention sequence length.
            cut: Optional single cut.
            cuts: ``"all"`` or explicit cut list.

        Returns:
            Sorted list of cut indices to evaluate.

        Raises:
            ValueError: If both ``cut`` and ``cuts`` are provided, or ``cuts`` is an
                empty list.
        """
        if cuts is not None and cut is not None:
            msg = "Specify only one of cut=... or cuts=..., not both."
            raise ValueError(msg)
        if cuts is not None:
            if cuts != "all" and len(cuts) == 0:
                msg = "cuts must be 'all' or a non-empty list of cut indices."
                raise ValueError(msg)
            return list(range(1, int(num_interventions) + 1)) if cuts == "all" else [int(c) for c in cuts]
        if cut is not None:
            return [int(cut)]
        return [_default_cut(int(num_interventions), None)]

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
        parallel: bool | None,
        probe_kw: dict[str, Any],
        delay: int = 0,
    ) -> CharacterizationResult:
        """Characterize a process tensor or surrogate via internal split-cut probing.

        ``delay > 0`` is rejected in :meth:`characterize` before this path is reached.

        Returns:
            Single-cut :class:`~mqt.yaqs.characterization.memory.operational_memory.results.CharacterizationResult`.
        """
        resolved_cut = _default_cut(int(num_interventions), cut)
        out = run_operational_memory(
            process=target,
            cut=resolved_cut,
            num_interventions=int(num_interventions),
            n_pasts=n_pasts,
            n_futures=n_futures,
            rng=rng,
            probe_set=probe_set,
            return_raw=True,
            parallel=parallel if parallel is not None else self._execution.parallel,
            delay=delay,
            **probe_kw,
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
        probe_kw: dict[str, Any],
        delay: int = 0,
    ) -> CharacterizationResult:
        """Characterize a Hamiltonian via exact stochastic sequences and branch weights.

        Returns:
            Single- or multi-cut
            :class:`~mqt.yaqs.characterization.memory.operational_memory.results.CharacterizationResult`.

        Raises:
            ValueError: If ``probe_set`` is given for a multi-cut request.
        """
        from mqt.yaqs.characterization.memory.backends.exact import ExactBackend

        operator = _require_hamiltonian(hamiltonian)
        cut_list = MemoryCharacterizer._resolve_cut_list(int(num_interventions), cut=cut, cuts=cuts)
        if probe_set is not None and len(cut_list) > 1:
            msg = "probe_set cannot be reused across multiple cuts; omit probe_set for multi-cut characterize()."
            raise ValueError(msg)
        psi0 = (
            np.asarray(initial_psi, dtype=np.complex128)
            if initial_psi is not None
            else make_zero_psi(hamiltonian.length)
        )
        backend = ExactBackend(
            operator=operator,
            sim_params=sim_params,
            initial_psi=psi0,
            parallel=self._execution.parallel,
            show_progress=self._execution.show_progress,
            solver=self._solver_for(hamiltonian),
            _execution=self._execution,
        )
        parts: dict[int, CharacterizationResult] = {}
        for c in cut_list:
            resolved_cut = _default_cut(int(num_interventions), int(c))
            local_probe_set = probe_set
            if local_probe_set is None:
                local_rng = rng if rng is not None else np.random.default_rng()
                local_probe_set = sample_probes(
                    cut=resolved_cut,
                    num_interventions=int(num_interventions),
                    n_pasts=n_pasts,
                    n_futures=n_futures,
                    rng=local_rng,
                    **probe_kw,
                )
            out = run_operational_memory(
                process=backend,
                cut=resolved_cut,
                num_interventions=int(num_interventions),
                probe_set=local_probe_set,
                return_raw=True,
                delay=delay,
            )
            parts[int(resolved_cut)] = pack_result(out, cut=resolved_cut)
        return merge_cut_results(parts) if len(parts) > 1 else parts[cut_list[0]]

    def predict(  # noqa: PLR6301 -- public instance API
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
            num_interventions: Sequence length; inferred from ``target`` when omitted.
            return_sequence: If True, return the full ``num_interventions``-step trajectory
                instead of the final state only.
            rng: RNG for stochastic intervention sampling.

        Returns:
            Final (or full) site-0 reduced density matrix.

        Raises:
            ValueError: If ``return_sequence=True`` for a process-tensor target.
            TypeError: If ``target`` does not support surrogate-style prediction.
        """
        local_rng = rng if rng is not None else np.random.default_rng()
        seq = sequence

        if _matches_process_tensor(target):
            if return_sequence:
                msg = "return_sequence=True is not supported for process tensor targets."
                raise ValueError(msg)
            rho_mat = coerce_rho_matrix(rho0)
            target.check_initial_rho(rho_mat)
            resolved_num_interventions = _resolve_num_interventions(target, num_interventions)
            if isinstance(seq, str):
                slots = expand_interventions(seq, num_interventions=resolved_num_interventions, _rng=local_rng)
            else:
                slots = list(seq)
            steps, _ = encode_interventions(slots, num_interventions=resolved_num_interventions, rng=local_rng)
            callables = [convert_probe_callable(s) for s in steps]
            rho_out = target.predict(callables)
            return np.asarray(rho_out, dtype=np.complex128)

        rho_mat = coerce_rho_matrix(rho0)
        resolved_num_interventions = _resolve_num_interventions(target, num_interventions)
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
