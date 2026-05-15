# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""User-facing initial-state specification for YAQS simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from .networks import MPS

if TYPE_CHECKING:
    from numpy.typing import NDArray

Representation = Literal["mps", "vector", "density_matrix"]


def _infer_qubit_length(hilbert_dim: int) -> int:
    """Infer the number of qubits from a Hilbert-space dimension ``2**n``.

    Args:
        hilbert_dim: Dimension of the Hilbert space.

    Returns:
        Number of qubits ``n``.

    Raises:
        ValueError: If ``hilbert_dim`` is not a positive power of two.
    """
    if hilbert_dim < 1 or (hilbert_dim & (hilbert_dim - 1)) != 0:
        msg = (
            f"Hilbert-space dimension {hilbert_dim} is not a power of two; "
            "pass ``length`` explicitly for non-uniform physical dimensions."
        )
        raise ValueError(msg)
    return int(hilbert_dim.bit_length() - 1)


def _normalize_vector(vec: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Return a unit-norm copy of a state vector.

    Raises:
        ValueError: If the vector has zero norm.
    """
    vec = np.asarray(vec, dtype=np.complex128).reshape(-1)
    norm = np.linalg.norm(vec)
    if norm == 0:
        msg = "State vector must be non-zero."
        raise ValueError(msg)
    return vec / norm


def _normalize_density_matrix(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Return a trace-one copy of a density matrix.

    Raises:
        ValueError: If ``rho`` is not square or has zero trace.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        msg = "density_matrix must be a square 2-D array."
        raise ValueError(msg)
    trace = np.trace(rho)
    if np.isclose(trace, 0.0):
        msg = "density_matrix must have non-zero trace."
        raise ValueError(msg)
    if not np.isclose(trace, 1.0):
        rho /= trace
    return rho


class State:
    """Initial quantum state for :func:`~mqt.yaqs.simulator.run`.

    Holds problem-side configuration (length, preset, physical dimensions). Call
    :meth:`encode` (or let ``run`` do so) to materialize an :class:`~mqt.yaqs.core.data_structures.networks.MPS`,
    dense vector, or density matrix for the chosen ``AnalogSimParams.representation``.

    Manual data (mutually exclusive):

    - ``tensors``: list of MPS cores → MPS representation.
    - ``vector``: 1-D array → dense state vector.
    - ``density_matrix``: 2-D array → density matrix.

    If none of these are given, an MPS is built from ``initial`` (and related kwargs).
    """

    def __init__(
        self,
        length: int | None = None,
        *,
        initial: str = "zeros",
        physical_dimensions: list[int] | int | None = None,
        tensors: list[NDArray[np.complex128]] | None = None,
        vector: NDArray[np.complex128] | None = None,
        density_matrix: NDArray[np.complex128] | None = None,
        pad: int | None = None,
        basis_string: str | None = None,
    ) -> None:
        """Build a state specification.

        Args:
            length: Number of lattice sites. Inferred from ``tensors`` (list length),
                or from the Hilbert-space dimension of ``vector`` / ``density_matrix``
                when that dimension is a power of two. Required for preset-only construction.
            initial: Product-state preset passed to :class:`MPS` when no manual data is given.
                Same options as ``MPS(..., state=...)`` (``"zeros"``, ``"ones"``, ``"x+"``, …).
            physical_dimensions: Per-site physical dimension(s); default is qubits (2).
            tensors: MPS tensor cores (rank-3 arrays per site) → MPS representation.
            vector: 1-D state vector → dense ``representation='vector'`` data.
            density_matrix: 2-D square array → ``representation='density_matrix'`` data.
            pad: Bond-dimension padding passed through to :class:`MPS` (preset/tensor paths only).
            basis_string: Computational-basis string when ``initial="basis"``.

        Raises:
            ValueError: If more than one of ``tensors``, ``vector``, and ``density_matrix`` is set,
                if ``length`` cannot be inferred, or if array shapes are invalid, or if a vector or
                density matrix has zero norm or trace.
        """
        manual = [tensors is not None, vector is not None, density_matrix is not None]
        if sum(manual) > 1:
            msg = "Specify at most one of tensors, vector, and density_matrix."
            raise ValueError(msg)

        self.initial = initial
        self.physical_dimensions = physical_dimensions
        self._tensors: list[NDArray[np.complex128]] | None = None
        self.pad = pad
        self.basis_string = basis_string
        self._encoded_as: Representation | None = None
        self._mps: MPS | None = None
        self._vector: NDArray[np.complex128] | None = None
        self._density_matrix: NDArray[np.complex128] | None = None

        if tensors is not None:
            if len(tensors) == 0:
                msg = "tensors must be a non-empty list of MPS cores."
                raise ValueError(msg)
            n_sites = len(tensors)
            if length is not None and length != n_sites:
                msg = f"length={length} does not match len(tensors)={n_sites}."
                raise ValueError(msg)
            self.length = n_sites if length is None else length
            self._tensors = [np.asarray(t, dtype=np.complex128) for t in tensors]
        elif vector is not None:
            self._vector = _normalize_vector(vector)
            hilbert_dim = self._vector.size
            if length is None:
                self.length = _infer_qubit_length(hilbert_dim)
            else:
                self.length = length
        elif density_matrix is not None:
            self._density_matrix = _normalize_density_matrix(density_matrix)
            hilbert_dim = self._density_matrix.shape[0]
            if length is None:
                self.length = _infer_qubit_length(hilbert_dim)
            else:
                self.length = length
        else:
            if length is None:
                msg = "length is required when not passing tensors, vector, or density_matrix."
                raise ValueError(msg)
            self.length = length

    @classmethod
    def from_mps(cls, mps: MPS) -> State:
        """Wrap an existing :class:`MPS` as a :class:`State` (already in MPS form).

        Returns:
            A :class:`State` that references ``mps`` and is already encoded as ``"mps"``.
        """
        wrapped = cls(
            mps.length,
            tensors=list(mps.tensors),
            physical_dimensions=list(mps.physical_dimensions),
        )
        wrapped._mps = mps
        wrapped._encoded_as = "mps"
        return wrapped

    def _build_mps(self) -> MPS:
        """Return the MPS representation, building it from tensors or presets if needed.

        Returns:
            The materialized :class:`MPS`.

        Raises:
            ValueError: If this :class:`State` was created from ``vector`` or ``density_matrix``.
        """
        if self._mps is None:
            if self._vector is not None or self._density_matrix is not None:
                msg = (
                    "Cannot build an MPS from a State initialized with vector or "
                    "density_matrix; use tensors= or a preset initial=."
                )
                raise ValueError(msg)
            self._mps = MPS(
                self.length,
                tensors=self._tensors,
                physical_dimensions=self.physical_dimensions,
                state=self.initial,
                pad=self.pad,
                basis_string=self.basis_string,
            )
        return self._mps

    @property
    def mps(self) -> MPS:
        """Tensor-network representation after :meth:`encode` with ``"mps"`` (or a compatible encoding).

        Raises:
            RuntimeError: If :meth:`encode` with ``"mps"`` has not been called.
        """
        if self._mps is None:
            msg = "MPS not materialized; call encode('mps') first."
            raise RuntimeError(msg)
        return self._mps

    @property
    def vector(self) -> NDArray[np.complex128]:
        """Normalized dense state vector after :meth:`encode` with ``"vector"``.

        Raises:
            RuntimeError: If :meth:`encode` with ``"vector"`` has not been called.
        """
        if self._vector is None:
            msg = "Vector not materialized; call encode('vector') first."
            raise RuntimeError(msg)
        return self._vector

    @property
    def density_matrix(self) -> NDArray[np.complex128]:
        """Density matrix after :meth:`encode` with ``"density_matrix"``.

        Raises:
            RuntimeError: If :meth:`encode` with ``"density_matrix"`` has not been called.
        """
        if self._density_matrix is None:
            msg = "Density matrix not materialized; call encode('density_matrix') first."
            raise RuntimeError(msg)
        return self._density_matrix

    def encode(self, representation: Representation) -> State:
        r"""Materialize the state for the requested simulation representation.

        Args:
            representation: ``"mps"`` (B-normalized MPS), ``"vector"`` (unit-norm dense
                :math:`|\\psi\\rangle`), or ``"density_matrix"`` (density operator).

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If ``representation`` is not recognized or cannot be produced from
                the data supplied at construction.
        """
        if self._encoded_as == representation:
            return self
        if representation == "mps":
            mps = self._build_mps()
            mps.normalize("B")
        elif representation == "vector":
            if self._vector is not None:
                self._vector = _normalize_vector(self._vector)
            else:
                mps = self._build_mps()
                self._vector = _normalize_vector(mps.to_vec())
        elif representation == "density_matrix":
            if self._density_matrix is not None:
                self._density_matrix = _normalize_density_matrix(self._density_matrix)
            elif self._vector is not None:
                vec = _normalize_vector(self._vector)
                dim = vec.size
                self._density_matrix = np.outer(vec, vec.conj()).reshape(dim, dim)
            else:
                mps = self._build_mps()
                vec = _normalize_vector(mps.to_vec())
                dim = vec.size
                self._density_matrix = np.outer(vec, vec.conj()).reshape(dim, dim)
        else:
            msg = f"Unknown representation: {representation!r}"
            raise ValueError(msg)
        self._encoded_as = representation
        return self
