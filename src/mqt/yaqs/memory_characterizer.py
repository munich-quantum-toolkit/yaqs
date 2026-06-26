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

from mqt.yaqs.characterization.memory.backends.tomography import DenseComb, MPOComb, build_process_tensor
from mqt.yaqs.characterization.memory.backends.tomography.combs import convert_probe_callable
from mqt.yaqs.characterization.memory.operational_memory.interventions import (
    InterventionSequence,
    encode_sequence,
    map_probe_kwargs,
)
from mqt.yaqs.characterization.memory.operational_memory.results import (
    CharacterizationResult,
    merge_cut_results,
    pack_result,
)
from mqt.yaqs.characterization.memory.operational_memory.run import run_operational_memory
from mqt.yaqs.characterization.memory.operational_memory.samples import sample_probes
from mqt.yaqs.characterization.memory.shared.encoding import (
    normalize_backend_rho,
    pack_rho8,
    unpack_rho8,
)
from mqt.yaqs.characterization.memory.shared.utils import (
    DEFAULT_VECTOR_MAX_QUBITS,
    CharacterizerRepresentation,
    representation_to_solver,
    resolve_characterizer_representation,
)
from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
from mqt.yaqs.core.parallel_utils import ExecutionConfig, MPContext, merge_execution_config

if TYPE_CHECKING:
    from numpy.random import Generator
    from torch.utils.data import TensorDataset

    from mqt.yaqs.characterization.memory.backends.surrogates.model import TransformerComb
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


def resolve_probe_bundle(probe_set: Any) -> ProbeSet | None:
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


def make_zero_psi(length: int) -> np.ndarray:
    """Return ``|0...0>`` on ``length`` qubits as a state vector.

    Args:
        length: Chain length.

    Returns:
        Product computational-zero state vector.
    """
    dim = 2 ** int(length)
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    return psi


def _resolve_k(target: Any, k: int | None) -> int:
    """Infer sequence length ``k`` from an explicit value or comb/surrogate target.

    Args:
        target: Comb, surrogate, or other characterized object.
        k: Optional explicit sequence length.

    Returns:
        Resolved ``k``.

    Raises:
        ValueError: If ``k`` cannot be inferred from ``target``.
    """
    if k is not None:
        return int(k)
    k_attr = getattr(target, "_k_for_probe", None)
    if callable(k_attr):
        return int(k_attr())
    msg = "k must be provided when the target does not define _k_for_probe()."
    raise ValueError(msg)


def _default_cut(k: int, cut: int | None) -> int:
    """Resolve causal cut, defaulting to the interior cut ``(k + 1) // 2``.

    Args:
        k: Sequence length.
        cut: Optional explicit cut.

    Returns:
        Valid cut in ``[1, k]``.

    Raises:
        ValueError: If the resolved cut is out of range.
    """
    resolved_k = int(k)
    c = (resolved_k + 1) // 2 if cut is None else int(cut)
    if not (1 <= c <= resolved_k):
        msg = f"cut must satisfy 1 <= cut <= k ({resolved_k}), got {c}."
        raise ValueError(msg)
    return c


def coerce_rho_matrix(rho0: np.ndarray) -> np.ndarray:
    """Normalize an initial state to a ``2 x 2`` density matrix.

    Args:
        rho0: Packed length-8 vector or ``2 x 2`` matrix.

    Returns:
        Complex density matrix.

    Raises:
        ValueError: If ``rho0`` has an unsupported shape.
    """
    arr = np.asarray(rho0, dtype=np.complex128)
    if arr.shape == (8,):
        return unpack_rho8(arr.astype(np.float64))
    if arr.shape == (2, 2):
        return arr
    msg = f"rho0 must be shape (2, 2) or packed length-8, got {arr.shape}."
    raise ValueError(msg)


def matches_hamiltonian(target: Any) -> bool:
    """Return whether ``target`` is a Hamiltonian characterize/predict target."""
    return isinstance(target, Hamiltonian)


def matches_comb(target: Any) -> bool:
    """Return whether ``target`` is a reference comb predict target."""
    return isinstance(target, (DenseComb, MPOComb))


class MemoryCharacterizer:
    """Entry point for operational memory workflows.

    **Build:** :meth:`train`, :meth:`sample` (advanced), :meth:`build_comb`

    **Use:** :meth:`predict` (surrogate or reference-comb dynamics), :meth:`characterize` (memory metrics)

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
        """Resolve stochastic solver for a Hamiltonian under this characterizer's representation."""
        rep = resolve_characterizer_representation(
            hamiltonian.length,
            self.representation,
            vector_max_qubits=self.vector_max_qubits,
        )
        return representation_to_solver(rep)

    def build_comb(
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
    ) -> DenseComb | MPOComb:
        """Build an exhaustive reference comb (validation only; scales as ``16^k``).

        Args:
            hamiltonian: System Hamiltonian.
            sim_params: Analog simulation parameters.
            timesteps: Optional per-step durations; defaults from ``sim_params.dt``.
            noise_model: Optional noise model during tomography sequences.
            num_trajectories: Monte Carlo trajectories per tomography sample.
            basis: Intervention basis for process-tensor tomography.
            basis_seed: Optional RNG seed for basis construction.
            return_type: ``"dense"`` or ``"mpo"`` comb storage.
            check: Whether to validate CPTP properties during construction.
            atol: CPTP check tolerance.
            compress_every: MPO compression cadence during construction.
            tol: MPO compression tolerance.
            max_bond_dim: Optional MPO bond-dimension cap.
            n_sweeps: MPO variational refinement sweeps.
            parallel: Override instance parallel setting.

        Returns:
            Dense or MPO reference comb for small ``k`` validation.
        """
        operator = _require_hamiltonian(hamiltonian)
        execution = self._execution if parallel is None else merge_execution_config(self._execution, parallel=parallel)
        return build_process_tensor(
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
            _execution=execution,
        )

    def sample(
        self,
        hamiltonian: Hamiltonian,
        sim_params: AnalogSimParams,
        *,
        k: int,
        n: int,
        rng: Generator | None = None,
        seed: int | None = None,
        timesteps: list[float] | None = None,
        init_mode: str = "eigenstate",
        interventions: str = "measure_prepare",
        parallel: bool | None = None,
        show_progress: bool | None = None,
    ) -> TensorDataset:
        """Sample intervention sequences for surrogate training (advanced).

        Args:
            hamiltonian: System Hamiltonian.
            sim_params: Analog simulation parameters.
            k: Number of intervention steps per sequence.
            n: Number of training sequences.
            rng: Optional RNG (overrides ``seed``).
            seed: Optional seed when ``rng`` is omitted.
            timesteps: Optional comb schedule of length ``k + 1``.
            init_mode: Initial-state sampling mode for training sequences.
            interventions: ``"haar"``, ``"clifford"``, or ``"measure_prepare"``.
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
            k=k,
            n=n,
            rng=rng,
            seed=seed,
            timesteps=timesteps,
            init_mode=init_mode,
            solver=self._solver_for(hamiltonian),
            interventions=interventions,
            parallel=self._execution.parallel if parallel is None else parallel,
            show_progress=self._execution.show_progress if show_progress is None else show_progress,
            _execution=self._execution,
        )

    def train(
        self,
        hamiltonian: Hamiltonian,
        sim_params: AnalogSimParams,
        *,
        k: int,
        n: int,
        seed: int | None = None,
        timesteps: list[float] | None = None,
        init_mode: str = "eigenstate",
        interventions: str = "measure_prepare",
        model_kwargs: dict | None = None,
        train_kwargs: dict | None = None,
        parallel: bool | None = None,
        show_progress: bool | None = None,
    ) -> TransformerComb:
        """Train a Transformer surrogate on simulated intervention sequences.

        Args:
            hamiltonian: System Hamiltonian.
            sim_params: Analog simulation parameters.
            k: Training sequence length (stored on the model).
            n: Number of training sequences.
            seed: Optional RNG seed for data sampling and weight init.
            timesteps: Optional comb schedule of length ``k + 1``.
            init_mode: Initial-state sampling mode for training sequences.
            interventions: Training intervention kind.
            model_kwargs: Optional overrides for :class:`TransformerComb` construction.
            train_kwargs: Optional overrides for the training loop.
            parallel: Override instance parallel setting.
            show_progress: Override instance progress-bar setting.

        Returns:
            Trained :class:`~mqt.yaqs.characterization.memory.backends.surrogates.model.TransformerComb`.
        """
        operator = _require_hamiltonian(hamiltonian)
        from mqt.yaqs.characterization.memory.backends.surrogates.workflow import (
            train_surrogate_model as _train_surrogate_model,
        )

        return _train_surrogate_model(
            operator,
            sim_params,
            k=k,
            n=n,
            seed=seed,
            timesteps=timesteps,
            init_mode=init_mode,
            interventions=interventions,
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
        k: int,
        cut: int | None = None,
        cuts: Literal["all"] | list[int] | None = None,
        preset: str = _DEFAULT_CHARACTERIZATION_PRESET,
        n_pasts: int | None = None,
        n_futures: int | None = None,
        interventions: str = "haar",
        rng: Generator | None = None,
        probe_set: Any | None = None,
        initial_psi: np.ndarray | None = None,
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
        k: int | None = None,
        preset: str = _DEFAULT_CHARACTERIZATION_PRESET,
        n_pasts: int | None = None,
        n_futures: int | None = None,
        interventions: str = "haar",
        rng: Generator | None = None,
        probe_set: Any | None = None,
        parallel: bool | None = None,
        **probe_kwargs: Any,
    ) -> CharacterizationResult: ...

    def characterize(
        self,
        target: Any,
        sim_params: AnalogSimParams | None = None,
        /,
        *,
        k: int | None = None,
        cut: int | None = None,
        cuts: Literal["all"] | list[int] | None = None,
        preset: str = _DEFAULT_CHARACTERIZATION_PRESET,
        n_pasts: int | None = None,
        n_futures: int | None = None,
        interventions: str = "haar",
        rng: Generator | None = None,
        probe_set: Any | None = None,
        initial_psi: np.ndarray | None = None,
        parallel: bool | None = None,
        **probe_kwargs: Any,
    ) -> CharacterizationResult:
        """Return operational memory diagnostics for a Hamiltonian, surrogate, or comb.

        For a Hamiltonian, pass ``sim_params`` and ``k``. For combs/surrogates, ``k`` is
        inferred from the target when omitted. Default interior cut is ``(k + 1) // 2``.

        Args:
            target: Hamiltonian, trained surrogate, or reference comb.
            sim_params: Required for Hamiltonian targets only.
            k: Intervention sequence length (required for Hamiltonian targets).
            cut: Single causal cut; mutually exclusive with ``cuts``.
            cuts: ``"all"`` or explicit list for multi-cut Hamiltonian sweeps.
            preset: Probe-grid preset (``"quick"``, ``"balanced"``, ``"accurate"``).
            n_pasts: Override number of past probes.
            n_futures: Override number of future probes.
            interventions: ``"haar"``, ``"clifford"``, or ``"measure_prepare"``.
            rng: RNG for probe sampling.
            probe_set: Prior :class:`CharacterizationResult` or internal probe bundle to reuse.
            initial_psi: Optional initial state for Hamiltonian exact simulation.
            parallel: Override parallelism for comb/surrogate probing.
            **probe_kwargs: Advanced overrides forwarded to internal probe sampling.

        Returns:
            Diagnostics with per-cut entropy, rank, spectrum, and stored probes.

        Raises:
            TypeError: If a Hamiltonian is given without ``sim_params``.
            ValueError: If ``k`` is missing for a Hamiltonian target.
        """
        n_p, n_f = _resolve_probe_grid(preset, n_pasts, n_futures)
        probe_kw = {**map_probe_kwargs(interventions), **probe_kwargs}
        resolved_probe_set = resolve_probe_bundle(probe_set)

        if matches_hamiltonian(target):
            if sim_params is None:
                msg = "characterize(hamiltonian, sim_params, k=...) requires AnalogSimParams."
                raise TypeError(msg)
            if k is None:
                msg = "characterize(hamiltonian, sim_params, ...) requires k=."
                raise ValueError(msg)
            return self._characterize_hamiltonian(
                target,
                sim_params,
                k=int(k),
                cut=cut,
                cuts=cuts,
                n_pasts=n_p,
                n_futures=n_f,
                rng=rng,
                probe_set=resolved_probe_set,
                initial_psi=initial_psi,
                probe_kw=probe_kw,
            )

        resolved_k = _resolve_k(target, k)
        cut_list = self._resolve_cut_list(resolved_k, cut=cut, cuts=cuts)
        if len(cut_list) == 1:
            return self._characterize_target(
                target,
                cut=cut_list[0],
                k=resolved_k,
                n_pasts=n_p,
                n_futures=n_f,
                rng=rng,
                probe_set=resolved_probe_set,
                parallel=parallel,
                probe_kw=probe_kw,
            )
        parts: dict[int, CharacterizationResult] = {}
        for c in cut_list:
            parts[int(c)] = self._characterize_target(
                target,
                cut=int(c),
                k=resolved_k,
                n_pasts=n_p,
                n_futures=n_f,
                rng=rng,
                probe_set=None,
                parallel=parallel,
                probe_kw=probe_kw,
            )
        return merge_cut_results(parts)

    def _resolve_cut_list(
        self,
        k: int,
        *,
        cut: int | None,
        cuts: Literal["all"] | list[int] | None,
    ) -> list[int]:
        """Resolve the list of cuts to characterize.

        Args:
            k: Sequence length.
            cut: Optional single cut.
            cuts: ``"all"`` or explicit cut list.

        Returns:
            Sorted list of cut indices to evaluate.
        """
        if cuts is not None:
            return list(range(1, int(k) + 1)) if cuts == "all" else [int(c) for c in cuts]
        if cut is not None:
            return [int(cut)]
        return [_default_cut(int(k), None)]

    def _characterize_target(
        self,
        target: Any,
        *,
        cut: int,
        k: int,
        n_pasts: int,
        n_futures: int,
        rng: Generator | None,
        probe_set: ProbeSet | None,
        parallel: bool | None,
        probe_kw: dict[str, Any],
    ) -> CharacterizationResult:
        """Characterize a comb or surrogate via internal split-cut probing."""
        resolved_cut = _default_cut(int(k), cut)
        out = run_operational_memory(
            process=target,
            cut=resolved_cut,
            k=int(k),
            n_pasts=n_pasts,
            n_futures=n_futures,
            rng=rng,
            probe_set=probe_set,
            return_raw=True,
            parallel=parallel if parallel is not None else self._execution.parallel,
            **probe_kw,
        )
        return pack_result(out, cut=resolved_cut)

    def _characterize_hamiltonian(
        self,
        hamiltonian: Hamiltonian,
        sim_params: AnalogSimParams,
        *,
        k: int,
        cut: int | None,
        cuts: Literal["all"] | list[int] | None,
        n_pasts: int,
        n_futures: int,
        rng: Generator | None,
        probe_set: ProbeSet | None,
        initial_psi: np.ndarray | None,
        probe_kw: dict[str, Any],
    ) -> CharacterizationResult:
        """Characterize a Hamiltonian via exact stochastic sequences and branch weights."""
        from mqt.yaqs.characterization.memory.backends.exact import ExactBackend

        operator = _require_hamiltonian(hamiltonian)
        cut_list = self._resolve_cut_list(int(k), cut=cut, cuts=cuts)
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
            resolved_cut = _default_cut(int(k), int(c))
            local_probe_set = probe_set
            if local_probe_set is None:
                local_rng = rng if rng is not None else np.random.default_rng()
                local_probe_set = sample_probes(
                    cut=resolved_cut,
                    k=int(k),
                    n_pasts=n_pasts,
                    n_futures=n_futures,
                    rng=local_rng,
                    **probe_kw,
                )
            out = run_operational_memory(
                process=backend,
                cut=resolved_cut,
                k=int(k),
                probe_set=local_probe_set,
                return_raw=True,
            )
            parts[int(resolved_cut)] = pack_result(out, cut=resolved_cut)
        return merge_cut_results(parts) if len(parts) > 1 else parts[cut_list[0]]

    def predict(
        self,
        target: Any,
        rho0: np.ndarray,
        sequence: InterventionSequence,
        /,
        *,
        k: int | None = None,
        return_sequence: bool = False,
        rng: Generator | None = None,
    ) -> np.ndarray:
        """Predict site-0 reduced-state dynamics under an intervention sequence.

        Supports trained surrogates and reference combs. For combs, ``rho0`` is accepted
        for API symmetry but not used (the comb contracts from the tomographic reference state).

        Args:
            target: Trained surrogate or reference comb.
            rho0: Initial ``2 x 2`` density matrix or packed length-8 vector.
            sequence: Intervention kind string, per-slot list, or expanded sequence.
            k: Sequence length; inferred from ``target`` when omitted.
            return_sequence: If True, return the full ``k``-step trajectory instead of the
                final state only.
            rng: RNG for stochastic intervention sampling.

        Returns:
            Final (or full) site-0 reduced density matrix.
        """
        local_rng = rng if rng is not None else np.random.default_rng()
        rho_mat = coerce_rho_matrix(rho0)
        seq = sequence

        if matches_comb(target):
            resolved_k = _resolve_k(target, k)
            if isinstance(seq, str):
                from mqt.yaqs.characterization.memory.operational_memory.interventions import expand_sequence

                slots = expand_sequence(seq, k=resolved_k, rng=local_rng)
            else:
                slots = list(seq)
            steps, _ = encode_sequence(slots, k=resolved_k, rng=local_rng)
            callables = [convert_probe_callable(s) for s in steps]
            rho_out = target.predict(callables)
            return np.asarray(rho_out, dtype=np.complex128)

        resolved_k = _resolve_k(target, k)
        _steps, e_features = encode_sequence(seq, k=resolved_k, rng=local_rng)
        packed0 = pack_rho8(normalize_backend_rho(rho_mat)).astype(np.float32)
        pred = target.predict(
            e_features[np.newaxis, ...],
            packed0[np.newaxis, ...],
            return_numpy=True,
        )
        if return_sequence:
            return np.stack([unpack_rho8(row) for row in pred[0]], axis=0).astype(np.complex128)
        return unpack_rho8(pred[0, -1, :])


__all__ = ["MemoryCharacterizer"]
