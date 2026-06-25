# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Operational memory characterization entry point for YAQS."""

# ruff: noqa: ANN401, D102, PLC0415 -- lazy torch imports, unified dispatch targets

from __future__ import annotations

from concurrent.futures import CancelledError
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

from mqt.yaqs.characterization.memory.combs.core.encoding import (
    normalize_rho_from_backend_output,
    pack_rho8,
    unpack_rho8,
)
from mqt.yaqs.characterization.memory.combs.core.utils import (
    DEFAULT_VECTOR_MAX_QUBITS,
    CharacterizerRepresentation,
    representation_to_solver,
    resolve_characterizer_representation,
)
from mqt.yaqs.characterization.memory.combs.tomography import DenseComb, MPOComb, construct_process_tensor
from mqt.yaqs.characterization.memory.combs.tomography.combs import _probe_step_to_callable
from mqt.yaqs.characterization.memory.diagnostics.probe import (
    analyze_v_matrix,
    build_weighted_v_from_probe,
    probe_process,
    sample_split_cut_probes,
)
from mqt.yaqs.characterization.memory.diagnostics.results import (
    CharacterizationResult,
    _merge_results,
    _result_from_probe_dict,
)
from mqt.yaqs.characterization.memory.interventions import (
    InterventionSequence,
    encode_sequence,
    probe_kwargs_from_interventions,
)
from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
from mqt.yaqs.core.parallel_utils import ExecutionConfig, MPContext, merge_execution_config

if TYPE_CHECKING:
    from numpy.random import Generator
    from torch.utils.data import TensorDataset

    from mqt.yaqs.characterization.memory.combs.surrogates.model import TransformerComb
    from mqt.yaqs.characterization.memory.combs.tomography.basis import TomographyBasis
    from mqt.yaqs.characterization.memory.diagnostics.probe import ProbeSet
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
    if preset not in _CHARACTERIZATION_PRESETS:
        msg = f"preset must be one of {sorted(_CHARACTERIZATION_PRESETS)!r}, got {preset!r}."
        raise ValueError(msg)
    defaults = _CHARACTERIZATION_PRESETS[preset]
    return (
        int(defaults[0] if n_pasts is None else n_pasts),
        int(defaults[1] if n_futures is None else n_futures),
    )


def _require_hamiltonian(hamiltonian: Hamiltonian) -> MPO:
    if not isinstance(hamiltonian, Hamiltonian):
        msg = "Pass a Hamiltonian; use Hamiltonian.ising(...) or Hamiltonian(...)."
        raise TypeError(msg)
    hamiltonian.ensure_encoded("mpo")
    return hamiltonian.mpo


def _default_product_zero_psi(length: int) -> np.ndarray:
    dim = 2 ** int(length)
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    return psi


def _resolve_k(target: Any, k: int | None) -> int:
    if k is not None:
        return int(k)
    k_attr = getattr(target, "_k_for_probe", None)
    if callable(k_attr):
        return int(k_attr())
    msg = "k must be provided when the target does not define _k_for_probe()."
    raise ValueError(msg)


def _default_cut(k: int, cut: int | None) -> int:
    resolved_k = int(k)
    c = (resolved_k + 1) // 2 if cut is None else int(cut)
    if not (1 <= c <= resolved_k):
        msg = f"cut must satisfy 1 <= cut <= k ({resolved_k}), got {c}."
        raise ValueError(msg)
    return c


def _as_rho_matrix(rho0: np.ndarray) -> np.ndarray:
    arr = np.asarray(rho0, dtype=np.complex128)
    if arr.shape == (8,):
        return unpack_rho8(arr.astype(np.float64))
    if arr.shape == (2, 2):
        return arr
    msg = f"rho0 must be shape (2, 2) or packed length-8, got {arr.shape}."
    raise ValueError(msg)


def _is_hamiltonian_target(target: Any) -> bool:
    return isinstance(target, Hamiltonian)


def _is_comb_target(target: Any) -> bool:
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
        return self._execution.parallel

    @property
    def max_workers(self) -> int:
        return self._execution.resolved_max_workers()

    @property
    def show_progress(self) -> bool:
        return self._execution.show_progress

    @property
    def mp_context(self) -> MPContext:
        return self._execution.mp_context

    @property
    def max_retries(self) -> int:
        return self._execution.max_retries

    @property
    def retry_exceptions(self) -> tuple[type[BaseException], ...]:
        return self._execution.retry_exceptions

    def _solver_for(self, hamiltonian: Hamiltonian) -> Literal["MCWF", "TJM"]:
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
        """Build an exhaustive reference comb (validation only; scales as ``16^k``)."""
        operator = _require_hamiltonian(hamiltonian)
        execution = self._execution if parallel is None else merge_execution_config(self._execution, parallel=parallel)
        return construct_process_tensor(
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
        """Sample intervention sequences for surrogate training (advanced)."""
        operator = _require_hamiltonian(hamiltonian)
        from mqt.yaqs.characterization.memory.combs.surrogates.workflow import generate_data as _generate_data

        return _generate_data(
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
        """Train a surrogate on simulated intervention sequences."""
        operator = _require_hamiltonian(hamiltonian)
        from mqt.yaqs.characterization.memory.combs.surrogates.workflow import create_surrogate as _create_surrogate

        return _create_surrogate(
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
        probe_set: ProbeSet | None = None,
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
        probe_set: ProbeSet | None = None,
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
        probe_set: ProbeSet | None = None,
        initial_psi: np.ndarray | None = None,
        parallel: bool | None = None,
        **probe_kwargs: Any,
    ) -> CharacterizationResult:
        """Return operational memory diagnostics for a Hamiltonian, surrogate, or comb."""
        n_p, n_f = _resolve_probe_grid(preset, n_pasts, n_futures)
        probe_kw = {**probe_kwargs_from_interventions(interventions), **probe_kwargs}

        if _is_hamiltonian_target(target):
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
                probe_set=probe_set,
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
                probe_set=probe_set,
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
        return _merge_results(parts)

    def _resolve_cut_list(
        self,
        k: int,
        *,
        cut: int | None,
        cuts: Literal["all"] | list[int] | None,
    ) -> list[int]:
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
        resolved_cut = _default_cut(int(k), cut)
        out = probe_process(
            process=target,
            cut=resolved_cut,
            k=int(k),
            n_pasts=n_pasts,
            n_futures=n_futures,
            rng=rng,
            probe_set=probe_set,
            return_v=True,
            parallel=parallel if parallel is not None else self._execution.parallel,
            **probe_kw,
        )
        return _result_from_probe_dict(out, cut=resolved_cut)

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
        from mqt.yaqs.characterization.memory.reference.exact import (
            evaluate_exact_probe_set_with_diagnostics,
        )

        operator = _require_hamiltonian(hamiltonian)
        cut_list = self._resolve_cut_list(int(k), cut=cut, cuts=cuts)
        parts: dict[int, CharacterizationResult] = {}
        for c in cut_list:
            resolved_cut = _default_cut(int(k), int(c))
            local_probe_set = probe_set
            if local_probe_set is None:
                local_rng = rng if rng is not None else np.random.default_rng()
                local_probe_set = sample_split_cut_probes(
                    cut=resolved_cut,
                    k=int(k),
                    n_pasts=n_pasts,
                    n_futures=n_futures,
                    rng=local_rng,
                    **probe_kw,
                )
            psi0 = (
                np.asarray(initial_psi, dtype=np.complex128)
                if initial_psi is not None
                else _default_product_zero_psi(hamiltonian.length)
            )
            pauli_xyz, weights_ij, _traces = evaluate_exact_probe_set_with_diagnostics(
                probe_set=local_probe_set,
                operator=operator,
                sim_params=sim_params,
                initial_psi=psi0,
                parallel=self._execution.parallel,
                show_progress=self._execution.show_progress,
                solver=self._solver_for(hamiltonian),
            )
            v, v_centered = build_weighted_v_from_probe(pauli_xyz, weights_ij)
            ana = analyze_v_matrix(v, v_centered)
            out: dict[str, Any] = {**ana, "V_centered": v_centered}
            parts[int(resolved_cut)] = _result_from_probe_dict(out, cut=resolved_cut)
        return _merge_results(parts) if len(parts) > 1 else parts[cut_list[0]]

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
        """
        local_rng = rng if rng is not None else np.random.default_rng()
        rho_mat = _as_rho_matrix(rho0)
        seq = sequence

        if _is_comb_target(target):
            resolved_k = _resolve_k(target, k)
            if isinstance(seq, str):
                from mqt.yaqs.characterization.memory.interventions import expand_intervention_sequence

                slots = expand_intervention_sequence(seq, k=resolved_k, rng=local_rng)
            else:
                slots = list(seq)
            steps, _ = encode_sequence(slots, k=resolved_k, rng=local_rng)
            callables = [_probe_step_to_callable(s) for s in steps]
            rho_out = target.predict(callables)
            return np.asarray(rho_out, dtype=np.complex128)

        resolved_k = _resolve_k(target, k)
        _steps, e_features = encode_sequence(seq, k=resolved_k, rng=local_rng)
        packed0 = pack_rho8(normalize_rho_from_backend_output(rho_mat)).astype(np.float32)
        pred = target.predict(
            e_features[np.newaxis, ...],
            packed0[np.newaxis, ...],
            return_numpy=True,
        )
        if return_sequence:
            return np.stack([unpack_rho8(row) for row in pred[0]], axis=0).astype(np.complex128)
        return unpack_rho8(pred[0, -1, :])


__all__ = ["MemoryCharacterizer"]
