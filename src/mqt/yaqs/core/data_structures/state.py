# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""User-facing initial-state specification for YAQS simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .mps import MPS
from .state_utils import (
    Representation,
    infer_qubit_length,
    normalize_density_matrix,
    normalize_vector,
    preset_is_product_state,
    product_state_vector,
    reject_preset_only_kwargs,
    resolve_physical_dimensions,
    validate_representation,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["Representation", "State"]


class State:
    """Initial quantum state for :meth:`~mqt.yaqs.Simulator.run`.

    Specify *what* to simulate (length, preset, optional raw data) and *how* to represent it
    during evolution (:attr:`representation`). Materialization happens at construction;
    pass the ``State`` to :meth:`~mqt.yaqs.Simulator.run` (including in parameter loops).

    - **Presets** — ``State(L, initial="zeros")``; default ``representation="mps"`` (TJM).
      For MCWF or Lindblad, set ``representation="vector"`` or ``"density_matrix"``.
    - **Manual data** — exactly one of ``tensors``, ``vector``, or ``density_matrix``; the
      representation is inferred (do not pass ``representation=``).

    Circuit simulation requires ``representation="mps"`` (checked in ``run``).
    """

    def __init__(
        self,
        length: int | None = None,
        *,
        initial: str = "zeros",
        representation: Representation | None = None,
        physical_dimensions: list[int] | int | None = None,
        tensors: list[NDArray[np.complex128]] | None = None,
        vector: NDArray[np.complex128] | None = None,
        density_matrix: NDArray[np.complex128] | None = None,
        pad: int | None = None,
        basis_string: str | None = None,
        seed: int | None = None,
    ) -> None:
        """Build a state specification.

        Args:
            length: Number of lattice sites. Inferred from ``tensors`` (list length),
                or from the Hilbert-space dimension of ``vector`` / ``density_matrix``
                when that dimension is a power of two. Required for preset-only construction.
            initial: Product-state preset passed to :class:`MPS` when no manual data is given.
                Same options as ``MPS(..., state=...)`` (``"zeros"``, ``"ones"``, ``"x+"``, …).
            representation: For preset-only states: ``"mps"`` (default, TJM), ``"vector"`` (MCWF),
                or ``"density_matrix"`` (Lindblad). Ignored when ``tensors``, ``vector``, or
                ``density_matrix`` is passed (inferred automatically).
            physical_dimensions: Per-site physical dimension(s); default is qubits (2).
            tensors: MPS tensor cores (rank-3 arrays per site); ``representation`` inferred as ``"mps"``.
            vector: 1-D state vector; ``representation`` inferred as ``"vector"``.
            density_matrix: 2-D square array; ``representation`` inferred as ``"density_matrix"``.
            pad: Bond-dimension padding passed through to :class:`MPS` (preset/tensor paths only).
            basis_string: Computational-basis string when ``initial="basis"``.
            seed: RNG seed for ``initial="random"`` when building dense product vectors.

        Raises:
            ValueError: If more than one of ``tensors``, ``vector``, and ``density_matrix`` is set,
                if ``length`` cannot be inferred, if ``length`` is not positive, or if array shapes
                are invalid, or if a vector or density matrix has zero norm or trace.
        """
        if length is not None and length <= 0:
            msg = "length must be a positive integer."
            raise ValueError(msg)

        manual = [tensors is not None, vector is not None, density_matrix is not None]
        if sum(manual) > 1:
            msg = "Specify at most one of tensors, vector, and density_matrix."
            raise ValueError(msg)

        self.initial = initial
        self.physical_dimensions = physical_dimensions
        self._tensors: list[NDArray[np.complex128]] | None = None
        self.pad = pad
        self.basis_string = basis_string
        self.seed = seed
        self._encoded_as: Representation | None = None
        self._mps: MPS | None = None
        self._vector: NDArray[np.complex128] | None = None
        self._density_matrix: NDArray[np.complex128] | None = None

        if tensors is not None:
            reject_preset_only_kwargs(initial=initial, pad=pad, basis_string=basis_string, seed=seed)
            if len(tensors) == 0:
                msg = "tensors must be a non-empty list of MPS cores."
                raise ValueError(msg)
            n_sites = len(tensors)
            if length is not None and length != n_sites:
                msg = f"length={length} does not match len(tensors)={n_sites}."
                raise ValueError(msg)
            self.length = n_sites if length is None else length
            self._tensors = [np.asarray(t, dtype=np.complex128) for t in tensors]
            if representation is not None and representation != "mps":
                msg = "representation is inferred as 'mps' from tensors=; omit representation=."
                raise ValueError(msg)
            self.representation = "mps"
        elif vector is not None:
            reject_preset_only_kwargs(initial=initial, pad=pad, basis_string=basis_string, seed=seed)
            self._vector = normalize_vector(vector)
            hilbert_dim = self._vector.size
            if length is None:
                self.length = infer_qubit_length(hilbert_dim)
            else:
                expected = int(np.prod(resolve_physical_dimensions(length, physical_dimensions)))
                if hilbert_dim != expected:
                    msg = f"vector size {hilbert_dim} does not match Hilbert dimension {expected} for length={length}."
                    raise ValueError(msg)
                self.length = length
            if representation is not None and representation != "vector":
                msg = "representation is inferred as 'vector' from vector=; omit representation=."
                raise ValueError(msg)
            self.representation = "vector"
        elif density_matrix is not None:
            reject_preset_only_kwargs(initial=initial, pad=pad, basis_string=basis_string, seed=seed)
            self._density_matrix = normalize_density_matrix(density_matrix)
            hilbert_dim = self._density_matrix.shape[0]
            if length is None:
                self.length = infer_qubit_length(hilbert_dim)
            else:
                expected = int(np.prod(resolve_physical_dimensions(length, physical_dimensions)))
                if hilbert_dim != expected:
                    msg = (
                        f"density_matrix dimension {hilbert_dim} does not match Hilbert dimension "
                        f"{expected} for length={length}."
                    )
                    raise ValueError(msg)
                self.length = length
            if representation is not None and representation != "density_matrix":
                msg = "representation is inferred as 'density_matrix' from density_matrix=; omit representation=."
                raise ValueError(msg)
            self.representation = "density_matrix"
        else:
            if length is None:
                msg = "length is required when not passing tensors, vector, or density_matrix."
                raise ValueError(msg)
            self.length = length
            self.representation = "mps" if representation is None else validate_representation(representation)

        self._encode(self.representation)

    def ensure_encoded(self, representation: Representation | None = None) -> State:
        """Materialize ``representation`` if needed (used by :meth:`~mqt.yaqs.Simulator.run`).

        Returns:
            ``self`` for chaining.
        """
        return self._encode(representation)

    @classmethod
    def from_mps(cls, mps: MPS) -> State:
        """Wrap an existing :class:`MPS` as a :class:`State` (already in MPS form).

        Returns:
            A :class:`State` that references ``mps`` and is already encoded as ``"mps"``.
        """
        wrapped = cls(mps.length, physical_dimensions=list(mps.physical_dimensions))
        wrapped._tensors = [np.asarray(t, dtype=np.complex128) for t in mps.tensors]
        wrapped._mps = mps
        wrapped._encoded_as = "mps"
        wrapped.representation = "mps"
        wrapped._encode("mps")
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

    def _can_build_dense_from_preset(self) -> bool:
        """Whether :meth:`_encode` can materialize a dense vector without an MPS.

        Returns:
            ``True`` if the state was specified by a product preset (not tensors / haar-random).
        """
        if self._tensors is not None:
            return False
        if not preset_is_product_state(self.initial):
            return False
        return not (self.initial == "basis" and self.basis_string is None)

    def _dense_vector_from_preset(self) -> NDArray[np.complex128]:
        """Normalized product-state vector for the stored preset.

        Returns:
            Dense vector for ``self.initial`` without building an :class:`MPS`.
        """
        return product_state_vector(
            self.length,
            self.initial,
            self.physical_dimensions,
            basis_string=self.basis_string,
            seed=self.seed,
        )

    @property
    def mps(self) -> MPS:
        """MPS when :attr:`representation` is ``"mps"``.

        Raises:
            RuntimeError: If the state is not encoded as ``"mps"``.
        """
        if self._encoded_as != "mps" or self._mps is None:
            msg = f"MPS is not available for representation={self.representation!r}."
            raise RuntimeError(msg)
        return self._mps

    @property
    def vector(self) -> NDArray[np.complex128]:
        """Dense state vector when :attr:`representation` is ``"vector"``.

        Raises:
            RuntimeError: If the state is not encoded as ``"vector"``.
        """
        if self._encoded_as != "vector" or self._vector is None:
            msg = f"State vector is not available for representation={self.representation!r}."
            raise RuntimeError(msg)
        return self._vector

    @property
    def density_matrix(self) -> NDArray[np.complex128]:
        """Density matrix when :attr:`representation` is ``"density_matrix"``.

        Raises:
            RuntimeError: If the state is not encoded as ``"density_matrix"``.
        """
        if self._encoded_as != "density_matrix" or self._density_matrix is None:
            msg = f"Density matrix is not available for representation={self.representation!r}."
            raise RuntimeError(msg)
        return self._density_matrix

    def _encode(self, representation: Representation | None = None) -> State:
        r"""Materialize internal storage for the requested representation (used by ``run``).

        Args:
            representation: Defaults to :attr:`representation`.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If ``representation`` is not recognized or cannot be produced from
                the data supplied at construction.
        """
        rep = self.representation if representation is None else validate_representation(representation)
        if self._encoded_as == rep:
            return self
        if rep == "mps":
            mps = self._build_mps()
            mps.normalize("B")
            self._mps = mps
        elif rep == "vector":
            if self._vector is not None:
                self._vector = normalize_vector(self._vector)
            elif self._can_build_dense_from_preset():
                self._vector = self._dense_vector_from_preset()
            else:
                mps = self._build_mps()
                self._vector = normalize_vector(mps.to_vec())
        elif rep == "density_matrix":
            if self._density_matrix is not None:
                self._density_matrix = normalize_density_matrix(self._density_matrix)
            elif self._vector is not None:
                vec = normalize_vector(self._vector)
                dim = vec.size
                self._density_matrix = np.asarray(np.outer(vec, vec.conj()).reshape(dim, dim), dtype=np.complex128)
            elif self._can_build_dense_from_preset():
                vec = self._dense_vector_from_preset()
                self._vector = vec
                dim = vec.size
                self._density_matrix = np.asarray(np.outer(vec, vec.conj()).reshape(dim, dim), dtype=np.complex128)
            else:
                mps = self._build_mps()
                vec = normalize_vector(mps.to_vec())
                dim = vec.size
                self._density_matrix = np.asarray(np.outer(vec, vec.conj()).reshape(dim, dim), dtype=np.complex128)
        else:
            msg = f"Unknown representation: {rep!r}"
            raise ValueError(msg)
        self._encoded_as = rep
        return self
