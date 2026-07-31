# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""User-facing Hamiltonian specification for YAQS analog simulations."""

from __future__ import annotations

import warnings
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


class Hamiltonian:
    """Hamiltonian for :meth:`~mqt.yaqs.Simulator.run` (analog evolution).

    Build via classmethods (``ising``, ``pauli``, …) or pass ``tensors`` / ``matrix`` /
    ``sparse_matrix``. These choices describe **source data**, not the simulation backend.

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
        physical_dimension: int = 2,
    ) -> None:
        """Build a Hamiltonian from manual tensor or matrix data.

        For preset Hamiltonians use :meth:`ising`, :meth:`heisenberg`, etc.

        Args:
            length: Number of sites. Inferred from ``len(tensors)`` or matrix dimension when omitted.
            tensors: MPO tensor cores.
            matrix: Dense operator matrix.
            sparse_matrix: Sparse operator.
            physical_dimension: Local Hilbert-space dimension (uniform sites).

        Raises:
            ValueError: If no manual data is given, data are mutually exclusive, shapes are invalid,
                or ``physical_dimension`` is not positive.
        """
        if physical_dimension <= 0:
            msg = "physical_dimension must be a positive integer."
            raise ValueError(msg)

        manual = [tensors is not None, matrix is not None, sparse_matrix is not None]
        if sum(manual) != 1:
            msg = "Pass exactly one of tensors, matrix, or sparse_matrix, or use a classmethod preset."
            raise ValueError(msg)

        self.physical_dimension = physical_dimension
        self._tensors: list[NDArray[np.complex128]] | None = None
        self._matrix: NDArray[np.complex128] | None = None
        self._sparse_matrix: scipy.sparse.csr_matrix | None = None
        self._mpo: MPO | None = None

        if tensors is not None:
            self._init_from_tensors(tensors, length)
        elif matrix is not None:
            self._init_from_matrix(matrix, length)
        else:
            assert sparse_matrix is not None
            self._init_from_sparse_matrix(sparse_matrix, length)

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
