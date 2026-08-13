# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""User-facing Hamiltonian specification for YAQS analog simulations."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse

from .hamiltonian_utils import (
    attach_mpo,
    sparse_to_csr,
)
from .mpo import MPO
from .state_utils import infer_chain_length

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["Hamiltonian"]

# Match preprocess_mcwf: warn when full Hilbert-space matrices become expensive.
_LARGE_HILBERT_DIM = 2**14
_PARAMETERIZED_HERMITICITY_MAX_DIM = 4096

_TermFactory = Callable[[object], "Hamiltonian | MPO"]
_ParameterSchedule = Callable[[float], object]
_ParameterizedTerm = tuple[_TermFactory, _ParameterSchedule]


class Hamiltonian:
    """Hamiltonian for :meth:`~mqt.yaqs.Simulator.run` (analog evolution).

    Build via classmethods (``ising``, ``pauli``, …), pass ``tensors`` / ``matrix`` /
    ``sparse_matrix``, or pair parameterized factories with schedules. These
    choices describe **source data**, not the simulation backend.

    Pair with :class:`~mqt.yaqs.core.data_structures.state.State`: the state's
    ``representation`` alone selects the backend (``"mps"`` → TJM / MPO, ``"vector"`` →
    MCWF / sparse, ``"density_matrix"`` → Lindblad / sparse).
    :meth:`~mqt.yaqs.Simulator.run` converts and caches the required MPO or sparse form.
    """

    def __init__(
        self,
        length: int | None = None,
        *,
        tensors: list[NDArray[np.complex128]] | None = None,
        matrix: NDArray[np.complex128] | None = None,
        sparse_matrix: scipy.sparse.spmatrix | None = None,
        parameterized_terms: Sequence[_ParameterizedTerm] | None = None,
        physical_dimension: int = 2,
    ) -> None:
        """Build a Hamiltonian from manual tensor or matrix data.

        For preset Hamiltonians use :meth:`ising`, :meth:`heisenberg`, etc.

        Args:
            length: Number of sites. Inferred from ``len(tensors)`` or matrix dimension when omitted.
            tensors: MPO tensor cores.
            matrix: Dense operator matrix.
            sparse_matrix: Sparse operator.
            parameterized_terms: Non-empty sequence of ``(factory, schedule)``
                pairs. At time ``t``, YAQS calls ``factory(schedule(t))``. Each
                factory must return a static :class:`Hamiltonian` or :class:`MPO`.
            physical_dimension: Local Hilbert-space dimension (uniform sites).

        Raises:
            ValueError: If no source is given, sources are mutually exclusive,
                shapes are invalid, or parameterized construction has no valid
                explicit ``length``.
        """
        if physical_dimension <= 0:
            msg = "physical_dimension must be a positive integer."
            raise ValueError(msg)

        sources = [tensors is not None, matrix is not None, sparse_matrix is not None, parameterized_terms is not None]
        if sum(sources) != 1:
            msg = (
                "Pass exactly one of tensors, matrix, sparse_matrix, or parameterized_terms, "
                "or use a classmethod preset."
            )
            raise ValueError(msg)

        self.physical_dimension = physical_dimension
        self._tensors: list[NDArray[np.complex128]] | None = None
        self._matrix: NDArray[np.complex128] | None = None
        self._sparse_matrix: scipy.sparse.csr_matrix | None = None
        self._mpo: MPO | None = None
        self._parameterized_terms: tuple[_ParameterizedTerm, ...] | None = None

        if parameterized_terms is not None:
            self._init_from_parameterized_terms(parameterized_terms, length)
        elif tensors is not None:
            self._init_from_tensors(tensors, length)
        elif matrix is not None:
            self._init_from_matrix(matrix, length)
        else:
            assert sparse_matrix is not None
            self._init_from_sparse_matrix(sparse_matrix, length)

    def _init_from_parameterized_terms(
        self,
        parameterized_terms: Sequence[_ParameterizedTerm],
        length: int | None,
    ) -> None:
        """Validate and store paired parameterized factories and schedules.

        Raises:
            TypeError: If ``length`` or a pair has the wrong type.
            ValueError: If ``length`` is not positive or the sequence is empty.
        """
        if isinstance(length, bool) or not isinstance(length, int):
            msg = "length must be a positive integer for parameterized_terms."
            raise TypeError(msg)
        if length <= 0:
            msg = "length must be a positive integer for parameterized_terms."
            raise ValueError(msg)
        if isinstance(parameterized_terms, (str, bytes)) or not isinstance(parameterized_terms, Sequence):
            msg = "parameterized_terms must be a non-empty sequence of (factory, schedule) pairs."
            raise TypeError(msg)
        raw_terms = tuple(parameterized_terms)
        if not raw_terms:
            msg = "parameterized_terms must be a non-empty sequence of (factory, schedule) pairs."
            raise ValueError(msg)

        normalized: list[_ParameterizedTerm] = []
        for index, item in enumerate(raw_terms):
            if not isinstance(item, tuple) or len(item) != 2:
                msg = f"parameterized_terms[{index}] must be a (factory, schedule) tuple."
                raise TypeError(msg)
            factory, schedule = item
            if not callable(factory):
                msg = f"parameterized_terms[{index}] factory must be callable."
                raise TypeError(msg)
            if not callable(schedule):
                msg = f"parameterized_terms[{index}] schedule must be callable."
                raise TypeError(msg)
            normalized.append((factory, schedule))

        self.length = length
        self._parameterized_terms = tuple(normalized)

    @property
    def is_parameterized(self) -> bool:
        """Whether this Hamiltonian is defined by paired factories and schedules."""
        return getattr(self, "_parameterized_terms", None) is not None

    @staticmethod
    def _validate_parameter_value(value: object, *, term_index: int, time: float) -> object:
        """Validate one schedule output and return it unchanged.

        Returns:
            The original validated value.

        Raises:
            ValueError: If the value is empty, non-numeric, or non-finite.
        """
        try:
            array = np.asarray(value)
            finite = array.size > 0 and np.issubdtype(array.dtype, np.number) and bool(np.all(np.isfinite(array)))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            msg = f"parameterized_terms[{term_index}] schedule returned a non-finite numeric value at time {time}."
            raise ValueError(msg)
        return value

    def _parameters_at(self, time: float) -> tuple[object, ...]:
        """Evaluate every paired schedule at ``time``.

        Returns:
            Validated parameter values in term order.

        Raises:
            ValueError: If ``time`` or a schedule output is non-finite.
        """
        resolved_time = float(time)
        if not np.isfinite(resolved_time):
            msg = f"time must be finite, got {resolved_time!r}."
            raise ValueError(msg)
        if self._parameterized_terms is None:
            msg = "Static Hamiltonians do not have parameter schedules."
            raise ValueError(msg)
        return tuple(
            self._validate_parameter_value(schedule(resolved_time), term_index=index, time=resolved_time)
            for index, (_factory, schedule) in enumerate(self._parameterized_terms)
        )

    @staticmethod
    def _validate_resolved_mpo(
        mpo: MPO,
        *,
        expected_length: int,
        term_index: int,
        check_hermiticity: bool = True,
    ) -> None:
        """Validate one factory's resolved MPO structure and optional Hermiticity.

        Raises:
            ValueError: If length, tensor structure, boundaries, or Hermiticity is invalid.
        """
        if mpo.length != expected_length or len(mpo.tensors) != expected_length:
            msg = f"parameterized_terms[{term_index}] factory returned length {mpo.length}; expected {expected_length}."
            raise ValueError(msg)
        for site, tensor in enumerate(mpo.tensors):
            if tensor.ndim != 4 or tensor.shape[0] != tensor.shape[1] or not np.all(np.isfinite(tensor)):
                msg = f"parameterized_terms[{term_index}] factory returned an invalid MPO tensor at site {site}."
                raise ValueError(msg)
        if mpo.tensors[0].shape[2] != 1 or mpo.tensors[-1].shape[3] != 1 or not mpo.check_if_valid_mpo():
            msg = f"parameterized_terms[{term_index}] factory returned an MPO with invalid virtual bonds."
            raise ValueError(msg)

        total_dimension = int(np.prod([tensor.shape[0] for tensor in mpo.tensors]))
        # Dense reconstruction scales exponentially; above this cutoff the factory
        # remains responsible for satisfying the Hermiticity contract.
        if check_hermiticity and total_dimension <= _PARAMETERIZED_HERMITICITY_MAX_DIM:
            matrix = mpo.to_matrix()
            if not np.allclose(matrix, matrix.conj().T, rtol=1e-10, atol=1e-12):
                msg = f"parameterized_terms[{term_index}] factory returned a non-Hermitian operator."
                raise ValueError(msg)

    def _resolve_parameters(self, parameters: Sequence[object]) -> MPO:
        """Resolve validated parameter values into one static MPO.

        Returns:
            Sum of the static MPO terms returned by the paired factories.

        Raises:
            ValueError: If values, lengths, dimensions, or operators are invalid.
        """
        if self._parameterized_terms is None:
            msg = "Static Hamiltonians cannot resolve parameter values."
            raise ValueError(msg)
        return self._resolve_factories(
            tuple(factory for factory, _schedule in self._parameterized_terms),
            parameters,
            length=self.length,
        )

    @classmethod
    def _resolve_factories(
        cls,
        factories: Sequence[_TermFactory],
        parameters: Sequence[object],
        *,
        length: int,
        check_hermiticity: bool = True,
    ) -> MPO:
        """Resolve factory callables and parameters into one validated MPO.

        Returns:
            The validated sum of all resolved terms.

        Raises:
            TypeError: If a factory returns an unsupported type.
            ValueError: If values, lengths, dimensions, or operators are invalid.
        """
        if len(parameters) != len(factories):
            msg = f"Expected {len(factories)} parameter values, got {len(parameters)}."
            raise ValueError(msg)

        mpos: list[MPO] = []
        physical_legs: tuple[tuple[int, int], ...] | None = None
        for index, (factory, parameter) in enumerate(zip(factories, parameters, strict=True)):
            resolved = factory(parameter)
            if isinstance(resolved, Hamiltonian):
                if resolved.is_parameterized:
                    msg = (
                        f"parameterized_terms[{index}] factory must return a static Hamiltonian, "
                        "not a parameterized one."
                    )
                    raise ValueError(msg)
                resolved.ensure_mpo()
                mpo = resolved.mpo
            elif isinstance(resolved, MPO):
                mpo = resolved
            else:
                msg = (
                    f"parameterized_terms[{index}] factory must return Hamiltonian or MPO, "
                    f"got {type(resolved).__name__}."
                )
                raise TypeError(msg)
            cls._validate_resolved_mpo(
                mpo,
                expected_length=length,
                term_index=index,
                check_hermiticity=check_hermiticity,
            )
            current_legs = tuple((tensor.shape[0], tensor.shape[1]) for tensor in mpo.tensors)
            if physical_legs is not None and current_legs != physical_legs:
                msg = f"parameterized_terms[{index}] factory returned physical dimensions incompatible with term 0."
                raise ValueError(msg)
            physical_legs = current_legs
            mpos.append(mpo)
        return MPO.mpo_sum(mpos)

    def _resolve_at(self, time: float) -> MPO:
        """Resolve this parameterized Hamiltonian at one time.

        Returns:
            The static MPO at ``time``.
        """
        return self._resolve_parameters(self._parameters_at(time))

    def _init_from_tensors(
        self,
        tensors: list[NDArray[np.complex128]],
        length: int | None,
    ) -> None:
        """Validate and store MPO tensor cores, then materialize the MPO.

        Raises:
            ValueError: If ``tensors`` is empty or ``length`` disagrees with ``len(tensors)``.
        """
        if len(tensors) == 0:
            msg = "tensors must be a non-empty list of MPO cores."
            raise ValueError(msg)
        n_sites = len(tensors)
        if length is not None and length != n_sites:
            msg = f"length={length} does not match len(tensors)={n_sites}."
            raise ValueError(msg)
        self.length = n_sites if length is None else length
        self._tensors = [np.asarray(t, dtype=np.complex128) for t in tensors]
        self.ensure_mpo()

    def _init_from_matrix(
        self,
        matrix: NDArray[np.complex128],
        length: int | None,
    ) -> None:
        """Validate and store a dense Hamiltonian matrix.

        Raises:
            ValueError: If ``matrix`` is not square or disagrees with ``length``.
        """
        mat = np.asarray(matrix, dtype=np.complex128)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            msg = "matrix must be a square 2-D array."
            raise ValueError(msg)
        hilbert_dim = mat.shape[0]
        if length is None:
            self.length = infer_chain_length(hilbert_dim, physical_dimension=self.physical_dimension)
        else:
            expected = self.physical_dimension**length
            if hilbert_dim != expected:
                msg = f"matrix dimension {hilbert_dim} does not match physical_dimension**length={expected}."
                raise ValueError(msg)
            self.length = length
        self._matrix = mat

    def _init_from_sparse_matrix(
        self,
        sparse_matrix: scipy.sparse.spmatrix,
        length: int | None,
    ) -> None:
        """Validate and store a sparse Hamiltonian matrix.

        Raises:
            ValueError: If ``sparse_matrix`` is not square or disagrees with ``length``.
        """
        sparse = sparse_to_csr(sparse_matrix)
        hilbert_dim = sparse.shape[0]
        if sparse.shape[0] != sparse.shape[1]:
            msg = "sparse_matrix must be square."
            raise ValueError(msg)
        if length is None:
            self.length = infer_chain_length(hilbert_dim, physical_dimension=self.physical_dimension)
        else:
            expected = self.physical_dimension**length
            if hilbert_dim != expected:
                msg = f"sparse_matrix dimension {hilbert_dim} does not match physical_dimension**length={expected}."
                raise ValueError(msg)
            self.length = length
        self._sparse_matrix = sparse

    @classmethod
    def from_mpo(cls, mpo: MPO) -> Hamiltonian:
        """Wrap an existing :class:`MPO`.

        Returns:
            A :class:`Hamiltonian` referencing ``mpo``.
        """
        wrapped = cls.__new__(cls)
        attach_mpo(wrapped, mpo)
        return wrapped

    @classmethod
    def ising(
        cls,
        length: int,
        J: float,  # ruff:ignore[invalid-argument-name]
        g: float,
        *,
        bc: str = "open",
        physical_dimension: int = 2,
        tol: float = 1e-12,
        max_bond_dim: int | None = None,
        n_sweeps: int = 2,
    ) -> Hamiltonian:
        """Transverse-field Ising Hamiltonian (delegates to :meth:`MPO.ising`).

        Returns:
            A :class:`Hamiltonian` wrapping the constructed MPO.
        """
        return cls.from_mpo(
            MPO.ising(
                length,
                J,
                g,
                bc=bc,
                physical_dimension=physical_dimension,
                tol=tol,
                max_bond_dim=max_bond_dim,
                n_sweeps=n_sweeps,
            ),
        )

    @classmethod
    def heisenberg(
        cls,
        length: int,
        Jx: float,  # ruff:ignore[invalid-argument-name]
        Jy: float,  # ruff:ignore[invalid-argument-name]
        Jz: float,  # ruff:ignore[invalid-argument-name]
        h: float = 0.0,
        *,
        bc: str = "open",
        physical_dimension: int = 2,
        tol: float = 1e-12,
        max_bond_dim: int | None = None,
        n_sweeps: int = 2,
    ) -> Hamiltonian:
        """Heisenberg Hamiltonian (delegates to :meth:`MPO.heisenberg`).

        Returns:
            A :class:`Hamiltonian` wrapping the constructed MPO.
        """
        return cls.from_mpo(
            MPO.heisenberg(
                length,
                Jx,
                Jy,
                Jz,
                h,
                bc=bc,
                physical_dimension=physical_dimension,
                tol=tol,
                max_bond_dim=max_bond_dim,
                n_sweeps=n_sweeps,
            ),
        )

    @classmethod
    def pauli(
        cls,
        *,
        length: int,
        two_body: list[tuple[complex | float, str, str]] | None = None,
        one_body: list[tuple[complex | float, str]] | None = None,
        bc: str = "open",
        physical_dimension: int = 2,
        tol: float = 1e-12,
        max_bond_dim: int | None = None,
        n_sweeps: int = 2,
    ) -> Hamiltonian:
        """Pauli-string Hamiltonian from one- and two-body terms (delegates to :meth:`MPO.pauli`).

        Returns:
            A :class:`Hamiltonian` wrapping the constructed MPO.
        """
        return cls.from_mpo(
            MPO.pauli(
                length=length,
                two_body=two_body,
                one_body=one_body,
                bc=bc,
                physical_dimension=physical_dimension,
                tol=tol,
                max_bond_dim=max_bond_dim,
                n_sweeps=n_sweeps,
            ),
        )

    @classmethod
    def fermi_hubbard_1d(
        cls,
        length: int,
        t: float,
        u: float,
        *,
        jordan_wigner: bool = False,
    ) -> Hamiltonian:
        """1D Fermi-Hubbard Hamiltonian (delegates to :meth:`MPO.fermi_hubbard_1d`).

        Returns:
            A :class:`Hamiltonian` wrapping the constructed MPO.
        """
        return cls.from_mpo(MPO.fermi_hubbard_1d(length, t=t, u=u, jordan_wigner=jordan_wigner))

    @classmethod
    def coupled_transmon(
        cls,
        length: int,
        qubit_dim: int,
        resonator_dim: int,
        qubit_freq: float,
        resonator_freq: float,
        anharmonicity: float,
        coupling: float,
    ) -> Hamiltonian:
        """Coupled transmon-resonator chain (delegates to :meth:`MPO.coupled_transmon`).

        Returns:
            A :class:`Hamiltonian` wrapping the constructed MPO.
        """
        return cls.from_mpo(
            MPO.coupled_transmon(
                length,
                qubit_dim,
                resonator_dim,
                qubit_freq,
                resonator_freq,
                anharmonicity,
                coupling,
            ),
        )

    @staticmethod
    def _warn_large_hilbert_dim(dim: int, *, action: str) -> None:
        """Emit the same large-system RuntimeWarning used by ``preprocess_mcwf``."""
        if dim <= _LARGE_HILBERT_DIM:
            return
        msg = (
            f"Hilbert-space dimension {dim} is large when {action}. "
            "This may be very slow or run out of memory. "
            "Prefer an MPO preset, Hamiltonian.from_mpo(...), or tensors= for large TJM runs."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=3)

    def ensure_mpo(self) -> Hamiltonian:
        """Materialize and cache an MPO form (used by TJM / ``State.representation='mps'``).

        Dense and sparse sources are converted via :meth:`MPO.from_matrix` (sparse is densified
        only when this path is requested). Large Hilbert-space conversions emit a
        ``RuntimeWarning`` matching the ``preprocess_mcwf`` threshold.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If no data is available to build an MPO.
        """
        if self.is_parameterized:
            msg = "A parameterized Hamiltonian must be resolved at a concrete time before MPO materialization."
            raise ValueError(msg)
        if self._mpo is not None:
            return self
        if self._tensors is not None:
            mpo = MPO()
            mpo.custom([np.asarray(t, dtype=np.complex128) for t in self._tensors])
            self._mpo = mpo
            return self
        if self._matrix is not None:
            self._warn_large_hilbert_dim(self._matrix.shape[0], action="factorizing a dense matrix into an MPO")
            self._mpo = MPO.from_matrix(self._matrix, self.physical_dimension)
            return self
        if self._sparse_matrix is not None:
            dim = self._sparse_matrix.shape[0]
            if self._matrix is None:
                self._warn_large_hilbert_dim(dim, action="densifying a sparse matrix to build an MPO")
                self._matrix = self._sparse_matrix.toarray()
            else:
                self._warn_large_hilbert_dim(dim, action="factorizing a dense matrix into an MPO")
            self._mpo = MPO.from_matrix(self._matrix, self.physical_dimension)
            return self
        msg = "No Hamiltonian data available to build an MPO."
        raise ValueError(msg)

    def ensure_sparse(self) -> Hamiltonian:
        """Materialize and cache a sparse matrix (used by MCWF / Lindblad).

        Prefers the authoritative dense matrix when present so a later sparse
        conversion after ``ensure_mpo()`` does not rebuild from a possibly
        truncated MPO.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If no data is available to build a sparse matrix.
        """
        if self.is_parameterized:
            msg = "A parameterized Hamiltonian cannot be materialized as one static sparse matrix."
            raise ValueError(msg)
        if self._sparse_matrix is not None:
            return self
        if self._matrix is not None:
            self._sparse_matrix = scipy.sparse.csr_matrix(self._matrix)
            return self
        if self._mpo is not None:
            self._sparse_matrix = sparse_to_csr(self._mpo.to_sparse_matrix())
            return self
        if self._tensors is not None:
            self.ensure_mpo()
            assert self._mpo is not None
            self._sparse_matrix = sparse_to_csr(self._mpo.to_sparse_matrix())
            return self
        msg = "Cannot build sparse matrix from Hamiltonian specification."
        raise ValueError(msg)

    @property
    def mpo(self) -> MPO:
        """Cached MPO, if one has been materialized.

        Raises:
            RuntimeError: If no MPO has been materialized yet; call :meth:`ensure_mpo`.
        """
        if self._mpo is None:
            msg = "MPO is not available; call ensure_mpo() first."
            raise RuntimeError(msg)
        return self._mpo

    @property
    def sparse_matrix(self) -> scipy.sparse.csr_matrix:
        """Cached sparse matrix, if one has been materialized.

        Raises:
            RuntimeError: If no sparse matrix has been materialized yet; call :meth:`ensure_sparse`.
        """
        if self._sparse_matrix is None:
            msg = "Sparse matrix is not available; call ensure_sparse() first."
            raise RuntimeError(msg)
        return self._sparse_matrix

    @property
    def matrix(self) -> NDArray[np.complex128]:
        """Cached dense matrix, if one has been materialized.

        Raises:
            RuntimeError: If no dense matrix has been materialized yet.
        """
        if self._matrix is None:
            msg = "Dense matrix is not available."
            raise RuntimeError(msg)
        return self._matrix

    def to_matrix(self) -> NDArray[np.complex128]:
        """Dense matrix (converts from cached MPO/sparse without requiring prior encode).

        Returns:
            Dense Hamiltonian matrix on the full Hilbert space.

        Raises:
            RuntimeError: If no materialized data is available to convert.
        """
        if self._matrix is not None:
            return np.asarray(self._matrix, dtype=np.complex128)
        if self._mpo is not None:
            return self._mpo.to_matrix()
        if self._sparse_matrix is not None:
            return self._sparse_matrix.toarray()
        msg = "Hamiltonian has no materialized data to convert to a dense matrix."
        raise RuntimeError(msg)

    def to_sparse_matrix(self) -> scipy.sparse.csr_matrix:
        """Sparse matrix (converts from cached forms; does not mutate caches).

        Prefer :meth:`ensure_sparse` when the sparse form should be cached for reuse.

        Returns:
            Sparse Hamiltonian matrix on the full Hilbert space.

        Raises:
            RuntimeError: If no materialized data is available to convert.
        """
        if self._sparse_matrix is not None:
            return self._sparse_matrix
        if self._mpo is not None:
            return self._mpo.to_sparse_matrix()
        if self._matrix is not None:
            return scipy.sparse.csr_matrix(self._matrix)
        msg = "Hamiltonian has no materialized data to convert to sparse form."
        raise RuntimeError(msg)
