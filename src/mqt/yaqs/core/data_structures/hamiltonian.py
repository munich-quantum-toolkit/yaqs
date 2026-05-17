# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""User-facing Hamiltonian specification for YAQS analog simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import scipy.sparse

from .networks import MPO
from .state import _infer_qubit_length

if TYPE_CHECKING:
    from numpy.typing import NDArray

Representation = Literal["mpo", "sparse", "dense"]

_ALLOWED_REPRESENTATIONS = frozenset({"mpo", "sparse", "dense"})


def _sparse_to_csr(matrix: scipy.sparse.spmatrix) -> scipy.sparse.csr_matrix:
    """Return ``matrix`` as CSR (copies only when needed)."""
    if isinstance(matrix, scipy.sparse.csr_matrix):
        return matrix
    return scipy.sparse.csr_matrix(matrix)


def _validate_representation(value: str) -> Representation:
    """Validate and return a Hamiltonian representation label.

    Raises:
        ValueError: If ``value`` is not ``"mpo"``, ``"sparse"``, or ``"dense"``.
    """
    if value not in _ALLOWED_REPRESENTATIONS:
        msg = f"Invalid representation {value!r}. Allowed values are 'mpo', 'sparse', or 'dense'."
        raise ValueError(msg)
    return cast("Representation", value)


class Hamiltonian:
    """Hamiltonian for :func:`~mqt.yaqs.simulator.run` (analog evolution).

    Build via classmethods (``ising``, ``hamiltonian``, …) or pass ``tensors`` / ``matrix`` /
    ``sparse_matrix``. Materialization happens at construction; reuse the same instance across
    ``run`` loops.

    Pair with :class:`~mqt.yaqs.core.data_structures.state.State`: TJM and ensembles need
    ``representation="mpo"``; MCWF and Lindblad use a sparse matrix derived from the Hamiltonian
    at ``run`` time when the state is dense.
    """

    def __init__(
        self,
        length: int | None = None,
        *,
        representation: Representation | None = None,
        tensors: list[NDArray[np.complex128]] | None = None,
        matrix: NDArray[np.complex128] | None = None,
        sparse_matrix: scipy.sparse.spmatrix | None = None,
        physical_dimension: int = 2,
    ) -> None:
        """Build a Hamiltonian from manual tensor or matrix data.

        For preset Hamiltonians use :meth:`ising`, :meth:`hamiltonian`, etc.

        Args:
            length: Number of sites. Inferred from ``len(tensors)`` or matrix dimension when omitted.
            representation: Only for ambiguous cases; usually inferred from manual data.
            tensors: MPO tensor cores; infers ``representation="mpo"``.
            matrix: Dense operator matrix; infers ``representation="dense"``.
            sparse_matrix: Sparse operator; infers ``representation="sparse"``.
            physical_dimension: Local Hilbert-space dimension for MPO construction from ``tensors``.

        Raises:
            ValueError: If no manual data is given, data are mutually exclusive, or shapes are invalid.
        """
        manual = [tensors is not None, matrix is not None, sparse_matrix is not None]
        if sum(manual) != 1:
            msg = "Pass exactly one of tensors, matrix, or sparse_matrix, or use a classmethod preset."
            raise ValueError(msg)

        self.physical_dimension = physical_dimension
        self._tensors: list[NDArray[np.complex128]] | None = None
        self._matrix: NDArray[np.complex128] | None = None
        self._sparse_matrix: scipy.sparse.csr_matrix | None = None
        self.representation: Representation
        self._mpo: MPO | None = None
        self._encoded_as: Representation | None = None

        if tensors is not None:
            if len(tensors) == 0:
                msg = "tensors must be a non-empty list of MPO cores."
                raise ValueError(msg)
            n_sites = len(tensors)
            if length is not None and length != n_sites:
                msg = f"length={length} does not match len(tensors)={n_sites}."
                raise ValueError(msg)
            self.length = n_sites if length is None else length
            self._tensors = [np.asarray(t, dtype=np.complex128) for t in tensors]
            if representation is not None and representation != "mpo":
                msg = "representation is inferred as 'mpo' from tensors=; omit representation=."
                raise ValueError(msg)
            self.representation = "mpo"
        elif matrix is not None:
            mat = np.asarray(matrix, dtype=np.complex128)
            if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                msg = "matrix must be a square 2-D array."
                raise ValueError(msg)
            hilbert_dim = mat.shape[0]
            if length is None:
                self.length = _infer_qubit_length(hilbert_dim)
            else:
                self.length = length
            self._matrix = mat
            if representation is not None and representation != "dense":
                msg = "representation is inferred as 'dense' from matrix=; omit representation=."
                raise ValueError(msg)
            self.representation = "dense"
        else:
            assert sparse_matrix is not None
            sparse = _sparse_to_csr(sparse_matrix)
            hilbert_dim = sparse.shape[0]
            if sparse.shape[0] != sparse.shape[1]:
                msg = "sparse_matrix must be square."
                raise ValueError(msg)
            if length is None:
                self.length = _infer_qubit_length(hilbert_dim)
            else:
                self.length = length
            self._sparse_matrix = sparse
            if representation is not None and representation != "sparse":
                msg = "representation is inferred as 'sparse' from sparse_matrix=; omit representation=."
                raise ValueError(msg)
            self.representation = "sparse"

        self._encode(self.representation)

    @classmethod
    def from_mpo(cls, mpo: MPO) -> Hamiltonian:
        """Wrap an existing :class:`MPO` (already encoded as ``"mpo"``).

        Returns:
            A :class:`Hamiltonian` referencing ``mpo``.
        """
        wrapped = cls.__new__(cls)
        wrapped.length = mpo.length
        wrapped.physical_dimension = mpo.physical_dimension
        wrapped.representation = "mpo"
        wrapped._tensors = None
        wrapped._matrix = None
        wrapped._sparse_matrix = None
        wrapped._mpo = mpo
        wrapped._encoded_as = "mpo"
        return wrapped

    @classmethod
    def ising(
        cls,
        length: int,
        J: float,  # noqa: N803
        g: float,
        **kwargs: Any,
    ) -> Hamiltonian:
        """Transverse-field Ising Hamiltonian (delegates to :meth:`MPO.ising`)."""
        return cls.from_mpo(MPO.ising(length, J, g, **kwargs))

    @classmethod
    def heisenberg(
        cls,
        length: int,
        Jx: float,  # noqa: N803
        Jy: float,  # noqa: N803
        Jz: float,  # noqa: N803
        h: float = 0.0,
        **kwargs: Any,
    ) -> Hamiltonian:
        """Heisenberg Hamiltonian (delegates to :meth:`MPO.heisenberg`)."""
        return cls.from_mpo(MPO.heisenberg(length, Jx, Jy, Jz, h, **kwargs))

    @classmethod
    def hamiltonian(cls, **kwargs: Any) -> Hamiltonian:
        """Generic Pauli Hamiltonian (delegates to :meth:`MPO.hamiltonian`)."""
        return cls.from_mpo(MPO.hamiltonian(**kwargs))

    @classmethod
    def fermi_hubbard_1d(cls, **kwargs: Any) -> Hamiltonian:
        """1D Fermi-Hubbard Hamiltonian (delegates to :meth:`MPO.fermi_hubbard_1d`)."""
        return cls.from_mpo(MPO.fermi_hubbard_1d(**kwargs))

    @classmethod
    def coupled_transmon(cls, **kwargs: Any) -> Hamiltonian:
        """Coupled transmon-resonator chain (delegates to :meth:`MPO.coupled_transmon`)."""
        return cls.from_mpo(MPO.coupled_transmon(**kwargs))

    def _build_mpo(self) -> MPO:
        """Build or return the MPO representation.

        Raises:
            ValueError: If this Hamiltonian was created from matrix data only.
        """
        if self._mpo is None:
            if self._matrix is not None or self._sparse_matrix is not None:
                msg = "Cannot build an MPO from matrix or sparse_matrix data; use tensors= or a preset classmethod."
                raise ValueError(msg)
            if self._tensors is None:
                msg = "No MPO specification available."
                raise ValueError(msg)
            mpo = MPO()
            mpo.custom([np.asarray(t, dtype=np.complex128) for t in self._tensors])
            self._mpo = mpo
        return self._mpo

    @property
    def mpo(self) -> MPO:
        """MPO when encoded as ``"mpo"``.

        Raises:
            RuntimeError: If not encoded as ``"mpo"``.
        """
        if self._encoded_as != "mpo" or self._mpo is None:
            msg = f"MPO is not available for representation={self.representation!r}."
            raise RuntimeError(msg)
        return self._mpo

    @property
    def sparse_matrix(self) -> scipy.sparse.csr_matrix:
        """Sparse matrix when encoded as ``"sparse"``.

        Raises:
            RuntimeError: If not encoded as ``"sparse"``.
        """
        if self._encoded_as != "sparse" or self._sparse_matrix is None:
            msg = f"Sparse matrix is not available for representation={self.representation!r}."
            raise RuntimeError(msg)
        return self._sparse_matrix

    @property
    def matrix(self) -> NDArray[np.complex128]:
        """Dense matrix when encoded as ``"dense"``.

        Raises:
            RuntimeError: If not encoded as ``"dense"``.
        """
        if self._encoded_as != "dense" or self._matrix is None:
            msg = f"Dense matrix is not available for representation={self.representation!r}."
            raise RuntimeError(msg)
        return self._matrix

    def _encode(self, representation: Representation | None = None) -> Hamiltonian:
        """Materialize internal storage for the requested representation.

        Returns:
            ``self`` for chaining.
        """
        rep: Representation = (
            self.representation if representation is None else _validate_representation(representation)
        )
        if self._encoded_as == rep:
            if rep == "mpo" and self._mpo is not None:
                return self
            if rep == "sparse" and self._sparse_matrix is not None:
                return self
            if rep == "dense" and self._matrix is not None:
                return self

        if rep == "mpo":
            self._build_mpo()
        elif rep == "sparse":
            if self._sparse_matrix is None:
                if self._mpo is not None:
                    self._sparse_matrix = _sparse_to_csr(self._mpo.to_sparse_matrix())
                elif self._matrix is not None:
                    self._sparse_matrix = scipy.sparse.csr_matrix(self._matrix)
                else:
                    msg = "Cannot build sparse matrix from Hamiltonian specification."
                    raise ValueError(msg)
        elif rep == "dense":
            if self._matrix is None:
                if self._sparse_matrix is not None:
                    self._matrix = self._sparse_matrix.toarray()
                elif self._mpo is not None:
                    self._matrix = self._mpo.to_matrix()
                else:
                    msg = "Cannot build dense matrix from Hamiltonian specification."
                    raise ValueError(msg)
        else:
            msg = f"Unknown representation: {rep!r}"
            raise ValueError(msg)
        self._encoded_as = rep
        return self

    def _ensure_encoded(self, representation: Representation | None = None) -> Hamiltonian:
        """Materialize ``representation`` if needed (used by :func:`~mqt.yaqs.simulator.run`).

        Returns:
            ``self`` for chaining.
        """
        return self._encode(representation)

    def to_matrix(self) -> NDArray[np.complex128]:
        """Dense matrix (converts from cached MPO/sparse without changing :attr:`representation`)."""
        if self._matrix is not None:
            return np.asarray(self._matrix, dtype=np.complex128)
        if self._mpo is not None:
            return self._mpo.to_matrix()
        if self._sparse_matrix is not None:
            return self._sparse_matrix.toarray()
        msg = "Hamiltonian has no materialized data to convert to a dense matrix."
        raise RuntimeError(msg)

    def to_sparse_matrix(self) -> scipy.sparse.csr_matrix:
        """Sparse matrix (converts from cached forms without changing :attr:`representation`)."""
        if self._sparse_matrix is not None:
            return self._sparse_matrix
        if self._mpo is not None:
            return self._mpo.to_sparse_matrix()
        if self._matrix is not None:
            return scipy.sparse.csr_matrix(self._matrix)
        msg = "Hamiltonian has no materialized data to convert to sparse form."
        raise RuntimeError(msg)
