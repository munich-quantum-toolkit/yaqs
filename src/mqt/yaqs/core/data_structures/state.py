# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""User-facing initial-state specification for YAQS simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from .networks import MPS

if TYPE_CHECKING:
    from numpy.typing import NDArray

Representation = Literal["mps", "vector", "density_matrix"]

_ALLOWED_REPRESENTATIONS = frozenset({"mps", "vector", "density_matrix"})


def _validate_representation(value: str) -> Representation:
    """Validate and return a simulation representation label.

    Raises:
        ValueError: If ``value`` is not ``"mps"``, ``"vector"``, or ``"density_matrix"``.
    """
    if value not in _ALLOWED_REPRESENTATIONS:
        msg = f"Invalid representation {value!r}. Allowed values are 'mps', 'vector', or 'density_matrix'."
        raise ValueError(msg)
    return cast("Representation", value)


def _reject_preset_only_kwargs(
    *,
    initial: str,
    pad: int | None,
    basis_string: str | None,
    seed: int | None,
) -> None:
    """Raise if preset-only kwargs are passed together with manual state data."""
    if initial != "zeros":
        msg = "initial= and other preset options apply only to preset State construction."
        raise ValueError(msg)
    if pad is not None:
        msg = "pad applies only to preset State construction."
        raise ValueError(msg)
    if basis_string is not None:
        msg = "basis_string applies only to preset State construction."
        raise ValueError(msg)
    if seed is not None:
        msg = "seed applies only to preset State construction."
        raise ValueError(msg)


_PRODUCT_STATE_PRESETS = frozenset({
    "zeros",
    "ones",
    "x+",
    "x-",
    "y+",
    "y-",
    "Neel",
    "wall",
    "random",
    "basis",
})


def _preset_is_product_state(initial: str) -> bool:
    """Return whether ``initial`` names a rank-1 product preset (not entangled).

    Returns:
        ``True`` for product presets such as ``"zeros"`` or ``"x+"``; ``False`` for ``"haar-random"``.
    """
    return initial in _PRODUCT_STATE_PRESETS


def _resolve_physical_dimensions(
    length: int,
    physical_dimensions: list[int] | int | None,
) -> list[int]:
    """Resolve per-site physical dimensions (default: qubits).

    Returns:
        A list of length ``length`` with each site's physical dimension.

    Raises:
        ValueError: If a list is passed with the wrong length.
    """
    if physical_dimensions is None:
        return [2] * length
    if isinstance(physical_dimensions, int):
        return [physical_dimensions] * length
    if len(physical_dimensions) != length:
        msg = f"physical_dimensions length {len(physical_dimensions)} != {length}."
        raise ValueError(msg)
    return list(physical_dimensions)


def _local_vector_for_preset(
    site: int,
    initial: str,
    local_dim: int,
    *,
    length: int,
    basis_string: str | None,
    rng: np.random.Generator,
) -> NDArray[np.complex128]:
    """Local state vector for one site (matches :class:`MPS` preset rules).

    Returns:
        A length-``local_dim`` complex vector for the site.

    Raises:
        ValueError: If ``initial`` is unknown, ``basis`` is missing ``basis_string``, or local
            dimension is too small for the preset.
    """
    vector = np.zeros(local_dim, dtype=np.complex128)
    if initial == "zeros":
        vector[0] = 1.0
    elif initial == "ones":
        if local_dim < 2:
            msg = "ones preset requires local dimension at least 2."
            raise ValueError(msg)
        vector[1] = 1.0
    elif initial == "x+":
        vector[0] = 1 / np.sqrt(2)
        vector[1] = 1 / np.sqrt(2)
    elif initial == "x-":
        vector[0] = 1 / np.sqrt(2)
        vector[1] = -1 / np.sqrt(2)
    elif initial == "y+":
        vector[0] = 1 / np.sqrt(2)
        vector[1] = 1j / np.sqrt(2)
    elif initial == "y-":
        vector[0] = 1 / np.sqrt(2)
        vector[1] = -1j / np.sqrt(2)
    elif initial == "Neel":
        if site % 2:
            vector[0] = 1.0
        elif local_dim > 1:
            vector[1] = 1.0
        else:
            vector[0] = 1.0
    elif initial == "wall":
        if site < length // 2:
            vector[0] = 1.0
        elif local_dim > 1:
            vector[1] = 1.0
        else:
            vector[0] = 1.0
    elif initial == "random":
        if local_dim < 2:
            msg = "random preset requires local dimension at least 2."
            raise ValueError(msg)
        vector[0] = rng.random()
        vector[1] = 1.0 - vector[0]
    elif initial == "basis":
        if basis_string is None:
            msg = "basis_string must be provided for initial='basis'."
            raise ValueError(msg)
        idx = int(basis_string[site])
        if idx >= local_dim:
            msg = f"basis index {idx} out of range for local dimension {local_dim}."
            raise ValueError(msg)
        vector[idx] = 1.0
    else:
        msg = f"Unknown product-state preset: {initial!r}"
        raise ValueError(msg)
    return vector


def _product_state_vector(
    length: int,
    initial: str,
    physical_dimensions: list[int] | int | None,
    *,
    basis_string: str | None = None,
    seed: int | None = None,
) -> NDArray[np.complex128]:
    """Build a normalized product-state vector without constructing an MPS.

    Returns:
        The normalized dense state vector of length ``prod(physical_dimensions)``.
    """
    dims = _resolve_physical_dimensions(length, physical_dimensions)
    rng = np.random.default_rng(seed)
    psi = _local_vector_for_preset(0, initial, dims[0], length=length, basis_string=basis_string, rng=rng)
    # kron(v_site, psi) so site 0 remains the least significant index (MPS to_vec order).
    for site in range(1, length):
        local = _local_vector_for_preset(site, initial, dims[site], length=length, basis_string=basis_string, rng=rng)
        psi = np.kron(local, psi)
    vec = np.asarray(psi, dtype=np.complex128).reshape(-1)
    return _normalize_vector(vec)


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

    Returns:
        The normalized vector.

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

    Specify *what* to simulate (length, preset, optional raw data) and *how* to represent it
    during evolution (:attr:`representation`). Pass the ``State`` to ``run``; do not call
    :meth:`_encode` or access :attr:`mps`, :attr:`vector`, or :attr:`density_matrix` beforehand.

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
        self.seed = seed
        self._encoded_as: Representation | None = None
        self._mps: MPS | None = None
        self._vector: NDArray[np.complex128] | None = None
        self._density_matrix: NDArray[np.complex128] | None = None

        if tensors is not None:
            _reject_preset_only_kwargs(initial=initial, pad=pad, basis_string=basis_string, seed=seed)
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
            _reject_preset_only_kwargs(initial=initial, pad=pad, basis_string=basis_string, seed=seed)
            self._vector = _normalize_vector(vector)
            hilbert_dim = self._vector.size
            if length is None:
                self.length = _infer_qubit_length(hilbert_dim)
            else:
                self.length = length
            if representation is not None and representation != "vector":
                msg = "representation is inferred as 'vector' from vector=; omit representation=."
                raise ValueError(msg)
            self.representation = "vector"
        elif density_matrix is not None:
            _reject_preset_only_kwargs(initial=initial, pad=pad, basis_string=basis_string, seed=seed)
            self._density_matrix = _normalize_density_matrix(density_matrix)
            hilbert_dim = self._density_matrix.shape[0]
            if length is None:
                self.length = _infer_qubit_length(hilbert_dim)
            else:
                self.length = length
            if representation is not None and representation != "density_matrix":
                msg = (
                    "representation is inferred as 'density_matrix' from density_matrix=; "
                    "omit representation=."
                )
                raise ValueError(msg)
            self.representation = "density_matrix"
        else:
            if length is None:
                msg = "length is required when not passing tensors, vector, or density_matrix."
                raise ValueError(msg)
            self.length = length
            self.representation = "mps" if representation is None else _validate_representation(representation)

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
        if not _preset_is_product_state(self.initial):
            return False
        return not (self.initial == "basis" and self.basis_string is None)

    def _dense_vector_from_preset(self) -> NDArray[np.complex128]:
        """Normalized product-state vector for the stored preset.

        Returns:
            Dense vector for ``self.initial`` without building an :class:`MPS`.
        """
        return _product_state_vector(
            self.length,
            self.initial,
            self.physical_dimensions,
            basis_string=self.basis_string,
            seed=self.seed,
        )

    @property
    def mps(self) -> MPS:
        """Internal MPS after materialization (not for use before :func:`~mqt.yaqs.simulator.run`).

        Raises:
            RuntimeError: If the state has not been materialized as ``"mps"``.
        """
        if self._encoded_as != "mps" or self._mps is None:
            msg = "MPS is not available; pass this State to simulator.run instead."
            raise RuntimeError(msg)
        return self._mps

    @property
    def vector(self) -> NDArray[np.complex128]:
        """Internal dense vector after materialization (not for use before ``run``).

        Raises:
            RuntimeError: If the state has not been materialized as ``"vector"``.
        """
        if self._encoded_as != "vector" or self._vector is None:
            msg = "State vector is not available; pass this State to simulator.run instead."
            raise RuntimeError(msg)
        return self._vector

    @property
    def density_matrix(self) -> NDArray[np.complex128]:
        """Internal density matrix after materialization (not for use before ``run``).

        Raises:
            RuntimeError: If the state has not been materialized as ``"density_matrix"``.
        """
        if self._encoded_as != "density_matrix" or self._density_matrix is None:
            msg = "Density matrix is not available; pass this State to simulator.run instead."
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
        rep = self.representation if representation is None else _validate_representation(representation)
        if self._encoded_as == rep:
            return self
        if rep == "mps":
            mps = self._build_mps()
            mps.normalize("B")
        elif rep == "vector":
            if self._vector is not None:
                self._vector = _normalize_vector(self._vector)
            elif self._can_build_dense_from_preset():
                self._vector = self._dense_vector_from_preset()
            else:
                mps = self._build_mps()
                self._vector = _normalize_vector(mps.to_vec())
        elif rep == "density_matrix":
            if self._density_matrix is not None:
                self._density_matrix = _normalize_density_matrix(self._density_matrix)
            elif self._vector is not None:
                vec = _normalize_vector(self._vector)
                dim = vec.size
                self._density_matrix = np.outer(vec, vec.conj()).reshape(dim, dim)
            elif self._can_build_dense_from_preset():
                vec = self._dense_vector_from_preset()
                self._vector = vec
                dim = vec.size
                self._density_matrix = np.outer(vec, vec.conj()).reshape(dim, dim)
            else:
                mps = self._build_mps()
                vec = _normalize_vector(mps.to_vec())
                dim = vec.size
                self._density_matrix = np.outer(vec, vec.conj()).reshape(dim, dim)
        else:
            msg = f"Unknown representation: {rep!r}"
            raise ValueError(msg)
        self._encoded_as = rep
        return self
