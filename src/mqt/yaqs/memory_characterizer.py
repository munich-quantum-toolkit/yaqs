# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Operational memory characterization entry point for YAQS."""

# ruff: noqa: ANN401, D102, PLC0415 -- module-level shortcuts, Attributes in class docstring, lazy torch imports

from __future__ import annotations

import importlib
from concurrent.futures import CancelledError
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from mqt.yaqs.characterization.memory.combs.tomography import DenseComb, MPOComb, construct_process_tensor
from mqt.yaqs.characterization.memory.diagnostics.probe import (
    ProbeSet,
    analyze_v_matrix,
    build_weighted_v_from_probe,
    probe_process,
    sample_split_cut_probes,
)
from mqt.yaqs.characterization.memory.diagnostics.results import CutDiagnostics, ProbeResult
from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
from mqt.yaqs.core.parallel_utils import ExecutionConfig, MPContext, merge_execution_config

if TYPE_CHECKING:
    from numpy.random import Generator
    from torch.utils.data import TensorDataset

    from mqt.yaqs.characterization.memory.combs.surrogates.model import TransformerComb
    from mqt.yaqs.characterization.memory.combs.tomography.basis import TomographyBasis
    from mqt.yaqs.core.data_structures.mpo import MPO
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

_LAZY_EXPORTS = {
    "TransformerComb": ("mqt.yaqs.characterization.memory.combs.surrogates.model", "TransformerComb"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_EXPORTS:
        module_path, attr = _LAZY_EXPORTS[name]
        return getattr(importlib.import_module(module_path), attr)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def _require_hamiltonian(hamiltonian: Hamiltonian) -> MPO:
    """Resolve a user :class:`Hamiltonian` to its MPO backend representation.

    Returns:
        MPO operator backing ``hamiltonian``.

    Raises:
        TypeError: If ``hamiltonian`` is not a :class:`Hamiltonian`.
    """
    if not isinstance(hamiltonian, Hamiltonian):
        msg = "Pass a Hamiltonian; use Hamiltonian.ising(...) or Hamiltonian(...)."
        raise TypeError(msg)
    hamiltonian.ensure_encoded("mpo")
    return hamiltonian.mpo


def _default_product_zero_psi(length: int) -> np.ndarray:
    """Return |0...0> on ``length`` qubits as a state vector."""
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
    msg = "k must be provided when the probe target does not define _k_for_probe()."
    raise ValueError(msg)


def _default_cut(k: int, cut: int | None) -> int:
    resolved_k = int(k)
    c = (resolved_k + 1) // 2 if cut is None else int(cut)
    if not (1 <= c <= resolved_k):
        msg = f"cut must satisfy 1 <= cut <= k ({resolved_k}), got {c}."
        raise ValueError(msg)
    return c


def _surrogate_model_if_any(target: Any) -> Any:
    """Return ``target`` when it is a trained surrogate without importing torch."""
    target_type = type(target)
    if target_type.__name__ == "TransformerComb" and target_type.__module__.endswith(".surrogates.model"):
        return target
    return None


class MemoryCharacterizer:
    """Entry point for operational memory characterization workflows.

  **Build artifacts** (optional intermediate steps — not diagnostics themselves):

    - :meth:`train` / :meth:`sample` — surrogate model and training data
    - :meth:`build_comb` — small-``k`` reference comb for validation

  **V-matrix diagnostics** (entropy, rank, singular spectrum) — always via ``probe*``:

    +-----------------------------+------------------------------------------+
    | Method                      | Where probe responses come from          |
    +=============================+==========================================+
    | :meth:`probe_exact`         | Full :class:`~mqt.yaqs.Simulator`       |
    |                             | rollouts (ground truth)                  |
    +-----------------------------+------------------------------------------+
    | :meth:`probe`               | A trained surrogate or reference comb    |
    |                             | you built with :meth:`train` or          |
    |                             | :meth:`build_comb`                       |
    +-----------------------------+------------------------------------------+
    | :meth:`probe_from_responses`| Pre-computed response grids              |
    +-----------------------------+------------------------------------------+

    Each ``probe*`` method returns a :class:`~mqt.yaqs.characterization.memory.diagnostics.results.ProbeResult`
    with :meth:`~mqt.yaqs.characterization.memory.diagnostics.results.ProbeResult.entropy`,
    :meth:`~mqt.yaqs.characterization.memory.diagnostics.results.ProbeResult.rank`, and
    :meth:`~mqt.yaqs.characterization.memory.diagnostics.results.ProbeResult.singular_values`.

    :meth:`characterize` is a shortcut that calls :meth:`train` then :meth:`probe` at
    multiple cuts.

    Attributes:
        parallel: Whether probe rollouts run in parallel via a process pool.
        max_workers: Maximum worker processes when ``parallel=True``.
        show_progress: Whether to display a tqdm progress bar.
        mp_context: Multiprocessing context (``"auto"``, ``"fork"``, or ``"spawn"``).
        max_retries: Maximum retry attempts for transient worker errors.
        retry_exceptions: Exception types that trigger a retry.
    """

    def __init__(
        self,
        *,
        parallel: bool = True,
        max_workers: int | None = None,
        show_progress: bool = True,
        mp_context: MPContext = "auto",
        max_retries: int = 10,
        retry_exceptions: tuple[type[BaseException], ...] = (CancelledError, TimeoutError, OSError),
    ) -> None:
        """Initialize with execution-side configuration.

        Args:
            parallel: If ``True`` (default), use a process pool for rollout-heavy paths.
            max_workers: Maximum worker processes when running in parallel.
            show_progress: Show a tqdm progress bar during rollouts.
            mp_context: Multiprocessing start method.
            max_retries: Maximum retries for transient worker errors.
            retry_exceptions: Exception types that trigger a retry.
        """
        self._execution = ExecutionConfig(
            parallel=parallel,
            max_workers=max_workers,
            show_progress=show_progress,
            mp_context=mp_context,
            max_retries=max_retries,
            retry_exceptions=retry_exceptions,
        )

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

        Returns:
            Dense or MPO process tensor comb.
        """
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
        solver: str | None = None,
        parallel: bool | None = None,
        show_progress: bool | None = None,
    ) -> TensorDataset:
        """Sample intervention rollouts for surrogate training.

        Returns:
            PyTorch dataset of past/future intervention rollouts.
        """
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
            solver=solver,
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
        model_kwargs: dict | None = None,
        train_kwargs: dict | None = None,
        parallel: bool | None = None,
        show_progress: bool | None = None,
    ) -> TransformerComb:
        """Train a :class:`TransformerComb` surrogate on sampled rollouts.

        Returns:
            Trained surrogate comb model.
        """
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
            model_kwargs=model_kwargs,
            train_kwargs=train_kwargs,
            parallel=self._execution.parallel if parallel is None else parallel,
            show_progress=self._execution.show_progress if show_progress is None else show_progress,
            _execution=self._execution,
        )

    def probe(
        self,
        target: Any,
        *,
        cut: int | None = None,
        k: int | None = None,
        n_pasts: int = 32,
        n_futures: int = 32,
        rng: Generator | None = None,
        probe_set: ProbeSet | None = None,
        return_v: bool = True,
        parallel: bool | None = None,
        **probe_kwargs: Any,
    ) -> ProbeResult:
        """Probe a process model and return V-matrix diagnostics.

        Call after :meth:`train` or :meth:`build_comb` with the returned surrogate or
        comb. Samples split-cut interventions, queries ``target`` for each response
        pair, builds the weighted V matrix, and returns entropy, rank, and singular
        values in a :class:`~mqt.yaqs.characterization.memory.diagnostics.results.ProbeResult`.

        Args:
            target: ``TransformerComb``, ``DenseComb``, or ``MPOComb``.

        Returns:
            :class:`~mqt.yaqs.characterization.memory.diagnostics.results.ProbeResult`
            at the resolved cut.
        """
        resolved_k = _resolve_k(target, k)
        resolved_cut = _default_cut(resolved_k, cut)
        out = probe_process(
            process=target,
            cut=resolved_cut,
            k=resolved_k,
            n_pasts=n_pasts,
            n_futures=n_futures,
            rng=rng,
            probe_set=probe_set,
            return_v=return_v,
            parallel=parallel if parallel is not None else self._execution.parallel,
            **probe_kwargs,
        )
        model = _surrogate_model_if_any(target)
        return ProbeResult.from_probe_process_dict(out, cut=resolved_cut, model=model)

    def probe_exact(
        self,
        hamiltonian: Hamiltonian,
        sim_params: AnalogSimParams,
        *,
        cut: int | None = None,
        k: int,
        n_pasts: int = 32,
        n_futures: int = 32,
        rng: Generator | None = None,
        probe_set: ProbeSet | None = None,
        initial_psi: np.ndarray | None = None,
        return_v: bool = True,
        **probe_kwargs: Any,
    ) -> ProbeResult:
        """Probe via exact simulator rollouts and return V-matrix diagnostics.

        Ground-truth path: runs the full analog simulator for each split-cut
        intervention, assembles the weighted V matrix, and returns a
        :class:`~mqt.yaqs.characterization.memory.diagnostics.results.ProbeResult`.

        Returns:
            :class:`~mqt.yaqs.characterization.memory.diagnostics.results.ProbeResult`
            at the resolved cut.
        """
        from mqt.yaqs.characterization.memory.reference.exact import (
            evaluate_exact_probe_set_with_diagnostics,
        )

        operator = _require_hamiltonian(hamiltonian)
        resolved_cut = _default_cut(int(k), cut)
        if probe_set is None:
            if rng is None:
                rng = np.random.default_rng()
            sample_kw = {
                key: probe_kwargs[key] for key in ("intervention_mode", "unitary_ensemble") if key in probe_kwargs
            }
            probe_set = sample_split_cut_probes(
                cut=resolved_cut,
                k=int(k),
                n_pasts=n_pasts,
                n_futures=n_futures,
                rng=rng,
                **sample_kw,
            )
        psi0 = (
            np.asarray(initial_psi, dtype=np.complex128)
            if initial_psi is not None
            else _default_product_zero_psi(hamiltonian.length)
        )
        pauli_xyz, weights_ij, _traces = evaluate_exact_probe_set_with_diagnostics(
            probe_set=probe_set,
            operator=operator,
            sim_params=sim_params,
            initial_psi=psi0,
            parallel=self._execution.parallel,
            show_progress=self._execution.show_progress,
        )
        v, v_centered = build_weighted_v_from_probe(pauli_xyz, weights_ij)
        ana = analyze_v_matrix(v, v_centered)
        out: dict[str, Any] = {
            "pauli_xyz_ij": pauli_xyz,
            "weights_ij": weights_ij,
            "probe_set": probe_set,
            **ana,
        }
        if return_v:
            out["V"] = v
            out["V_centered"] = v_centered
        return ProbeResult.from_probe_process_dict(out, cut=resolved_cut)

    @staticmethod
    def probe_from_responses(
        pauli_xyz_ij: np.ndarray,
        weights_ij: np.ndarray,
        probe_set: ProbeSet,
        *,
        cut: int | None = None,
        return_v: bool = True,
    ) -> ProbeResult:
        """Assemble V-matrix diagnostics from pre-computed probe response grids.

        No simulation is run. Use when responses were produced elsewhere or saved from
        a prior :meth:`probe_exact` call.

        Returns:
            :class:`~mqt.yaqs.characterization.memory.diagnostics.results.ProbeResult`
            at the resolved cut.
        """
        resolved_cut = _default_cut(int(probe_set.k), cut if cut is not None else int(probe_set.cut))
        v, v_centered = build_weighted_v_from_probe(
            np.asarray(pauli_xyz_ij, dtype=np.float32),
            np.asarray(weights_ij, dtype=np.float64),
        )
        ana = analyze_v_matrix(v, v_centered)
        out: dict[str, Any] = {
            "pauli_xyz_ij": np.asarray(pauli_xyz_ij, dtype=np.float32),
            "weights_ij": np.asarray(weights_ij, dtype=np.float64),
            "probe_set": probe_set,
            **ana,
        }
        if return_v:
            out["V"] = v
            out["V_centered"] = v_centered
        return ProbeResult.from_probe_process_dict(out, cut=resolved_cut)

    def characterize(
        self,
        hamiltonian: Hamiltonian,
        sim_params: AnalogSimParams,
        *,
        k: int,
        n: int,
        cuts: Literal["all"] | list[int] = "all",
        n_pasts: int = 32,
        n_futures: int = 32,
        seed: int | None = None,
        timesteps: list[float] | None = None,
        init_mode: str = "eigenstate",
        model_kwargs: dict | None = None,
        train_kwargs: dict | None = None,
        rng: Generator | None = None,
        **probe_kwargs: Any,
    ) -> ProbeResult:
        """Train a surrogate and return V-matrix diagnostics at one or more cuts.

        Shortcut for :meth:`train` followed by :meth:`probe`. To train once and probe
        many times, call those methods separately.

        Returns:
            Multi-cut :class:`~mqt.yaqs.characterization.memory.diagnostics.results.ProbeResult`;
            :attr:`~mqt.yaqs.characterization.memory.diagnostics.results.ProbeResult.model`
            holds the trained surrogate.
        """
        model = self.train(
            hamiltonian,
            sim_params,
            k=k,
            n=n,
            seed=seed,
            timesteps=timesteps,
            init_mode=init_mode,
            model_kwargs=model_kwargs,
            train_kwargs=train_kwargs,
        )
        cut_list = list(range(1, int(k) + 1)) if cuts == "all" else [int(c) for c in cuts]
        by_cut: dict[int, CutDiagnostics] = {}
        for c in cut_list:
            part = self.probe(
                model,
                cut=c,
                k=k,
                n_pasts=n_pasts,
                n_futures=n_futures,
                rng=rng,
                **probe_kwargs,
            )
            by_cut[int(c)] = part.by_cut[int(c)]
        return ProbeResult(by_cut=by_cut, model=model)

    def characterize_comb(
        self,
        hamiltonian: Hamiltonian,
        sim_params: AnalogSimParams,
        *,
        cut: int | None = None,
        k: int,
        timesteps: list[float] | None = None,
        n_pasts: int = 32,
        n_futures: int = 32,
        rng: Generator | None = None,
        probe_set: ProbeSet | None = None,
        return_v: bool = True,
        parallel: bool | None = None,
        **comb_kwargs: Any,
    ) -> ProbeResult:
        """Build a reference comb and return V-matrix diagnostics (small ``k`` only).

        Shortcut for :meth:`build_comb` followed by :meth:`probe`. Prefer calling
        those methods separately when reusing the comb at multiple cuts.

        Returns:
            :class:`~mqt.yaqs.characterization.memory.diagnostics.results.ProbeResult`
            at the resolved cut.
        """
        comb = self.build_comb(
            hamiltonian,
            sim_params,
            timesteps,
            parallel=parallel,
            **comb_kwargs,
        )
        return self.probe(
            comb,
            cut=cut,
            k=k,
            n_pasts=n_pasts,
            n_futures=n_futures,
            rng=rng,
            probe_set=probe_set,
            return_v=return_v,
        )


def train_surrogate(hamiltonian: Hamiltonian, sim_params: AnalogSimParams, /, **kwargs: Any) -> TransformerComb:
    """Train a surrogate via a default :class:`MemoryCharacterizer`.

    Returns:
        Trained :class:`TransformerComb`.
    """
    return MemoryCharacterizer().train(hamiltonian, sim_params, **kwargs)


def sample_rollouts(hamiltonian: Hamiltonian, sim_params: AnalogSimParams, /, **kwargs: Any) -> TensorDataset:
    """Sample training rollouts via a default :class:`MemoryCharacterizer`.

    Returns:
        PyTorch rollout dataset.
    """
    return MemoryCharacterizer().sample(hamiltonian, sim_params, **kwargs)


def characterize_memory(hamiltonian: Hamiltonian, sim_params: AnalogSimParams, /, **kwargs: Any) -> ProbeResult:
    """Train and probe memory via a default :class:`MemoryCharacterizer`.

    Returns:
        Multi-cut :class:`~mqt.yaqs.characterization.memory.diagnostics.results.ProbeResult`.
    """
    return MemoryCharacterizer().characterize(hamiltonian, sim_params, **kwargs)


__all__ = [
    "MemoryCharacterizer",
    "characterize_memory",
    "sample_rollouts",
    "train_surrogate",
]
