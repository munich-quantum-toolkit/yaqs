# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Internal helpers for :class:`~mqt.yaqs.core.data_structures.state.State`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

Representation = Literal["mps", "vector", "density_matrix"]

_ALLOWED_REPRESENTATIONS = frozenset({"mps", "vector", "density_matrix"})


def validate_representation(value: str) -> Representation:
    """Validate and return a simulation representation label.

    Returns:
        A valid ``"mps"``, ``"vector"``, or ``"density_matrix"`` label.

    Raises:
        ValueError: If ``value`` is not ``"mps"``, ``"vector"``, or ``"density_matrix"``.
    """
    if value not in _ALLOWED_REPRESENTATIONS:
        msg = f"Invalid representation {value!r}. Allowed values are 'mps', 'vector', or 'density_matrix'."
        raise ValueError(msg)
    return cast("Representation", value)


def reject_preset_only_kwargs(
    *,
    initial: str,
    pad: int | None,
    basis_string: str | None,
    seed: int | None,
) -> None:
    """Raise if preset-only kwargs are passed together with manual state data.

    Raises:
        ValueError: If any preset-only keyword is set together with manual state data.
    """
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


def preset_is_product_state(initial: str) -> bool:
    """Return whether ``initial`` names a rank-1 product preset (not entangled).

    Returns:
        ``True`` for product presets such as ``"zeros"`` or ``"x+"``; ``False`` for ``"haar-random"``.
    """
    return initial in _PRODUCT_STATE_PRESETS


def resolve_physical_dimensions(
    length: int,
    physical_dimensions: list[int] | int | None,
) -> list[int]:
    """Resolve per-site physical dimensions (default: qubits).

    Returns:
        A list of length ``length`` with each site's physical dimension.

    Raises:
        ValueError: If a list is passed with the wrong length, or any dimension is not a
            positive integer.
    """
    if physical_dimensions is None:
        return [2] * length
    if isinstance(physical_dimensions, int):
        if physical_dimensions <= 0:
            msg = (
                f"resolve_physical_dimensions: physical_dimensions must be a positive integer, "
                f"got {physical_dimensions}."
            )
            raise ValueError(msg)
        return [physical_dimensions] * length
    if len(physical_dimensions) != length:
        msg = f"physical_dimensions length {len(physical_dimensions)} != {length}."
        raise ValueError(msg)
    resolved: list[int] = []
    for i, dim in enumerate(physical_dimensions):
        if not isinstance(dim, int) or dim <= 0:
            msg = f"resolve_physical_dimensions: physical_dimensions[{i}] must be a positive integer, got {dim!r}."
            raise ValueError(msg)
        resolved.append(dim)
    return resolved


def local_vector_for_preset(
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
    elif initial in {"x+", "x-", "y+", "y-"}:
        if local_dim < 2:
            msg = f"{initial} preset requires local dimension at least 2."
            raise ValueError(msg)
        if initial == "x+":
            vector[0] = 1 / np.sqrt(2)
            vector[1] = 1 / np.sqrt(2)
        elif initial == "x-":
            vector[0] = 1 / np.sqrt(2)
            vector[1] = -1 / np.sqrt(2)
        elif initial == "y+":
            vector[0] = 1 / np.sqrt(2)
            vector[1] = 1j / np.sqrt(2)
        else:
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
        if site >= len(basis_string):
            msg = f"basis_string length {len(basis_string)} is too short for site {site} (chain length {length})."
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


def product_state_vector(
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
    dims = resolve_physical_dimensions(length, physical_dimensions)
    rng = np.random.default_rng(seed)
    psi = local_vector_for_preset(0, initial, dims[0], length=length, basis_string=basis_string, rng=rng)
    # kron(v_site, psi) so site 0 remains the least significant index (MPS to_vec order).
    for site in range(1, length):
        local = local_vector_for_preset(site, initial, dims[site], length=length, basis_string=basis_string, rng=rng)
        psi = np.kron(local, psi)
    vec = np.asarray(psi, dtype=np.complex128).reshape(-1)
    return normalize_vector(vec)


def infer_qubit_length(hilbert_dim: int) -> int:
    """Infer the number of qubits from a Hilbert-space dimension ``2**n``.

    Args:
        hilbert_dim: Dimension of the Hilbert space.

    Returns:
        Number of qubits ``n``.
    """
    return infer_chain_length(hilbert_dim, physical_dimension=2)


def infer_chain_length(hilbert_dim: int, *, physical_dimension: int) -> int:
    """Infer chain length from ``hilbert_dim == physical_dimension**length``.

    Args:
        hilbert_dim: Dimension of the Hilbert space.
        physical_dimension: Local Hilbert-space dimension at each site.

    Returns:
        Number of sites ``length``.

    Raises:
        ValueError: If ``physical_dimension`` is not positive or ``hilbert_dim`` is not an
            exact power of ``physical_dimension``.
    """
    if physical_dimension <= 0:
        msg = "physical_dimension must be a positive integer."
        raise ValueError(msg)
    if hilbert_dim < 1:
        msg = f"Hilbert-space dimension {hilbert_dim} must be positive."
        raise ValueError(msg)
    length = round(np.log(hilbert_dim) / np.log(physical_dimension))
    if physical_dimension**length != hilbert_dim:
        msg = (
            f"Hilbert-space dimension {hilbert_dim} is not physical_dimension**length "
            f"for physical_dimension={physical_dimension}; pass ``length`` explicitly."
        )
        raise ValueError(msg)
    return int(length)


def normalize_vector(vec: NDArray[np.complex128]) -> NDArray[np.complex128]:
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


def _resolve_site_dims(
    length: int,
    physical_dimensions: list[int] | int | None,
    *,
    local_dim: int = 2,
) -> list[int]:
    """Return per-site Hilbert-space dimensions for embedding helpers."""
    if physical_dimensions is not None:
        return resolve_physical_dimensions(length, physical_dimensions)
    if not isinstance(local_dim, int) or local_dim <= 0:
        msg = f"_resolve_site_dims: local_dim must be a positive integer, got {local_dim!r}."
        raise ValueError(msg)
    return [local_dim] * length


def embed_one_site_operator(
    op: NDArray[np.complex128],
    length: int,
    site: int,
    *,
    local_dim: int = 2,
    physical_dimensions: list[int] | int | None = None,
) -> NDArray[np.complex128]:
    """Embed a one-site operator into the full Hilbert space.

    Uses the same qubit indexing as :meth:`~mqt.yaqs.core.data_structures.mps.MPS.to_vec`,
    Qiskit little-endian circuits, and weak-simulation bitstring keys: site ``0`` is the
    least significant bit in the flat state index.

    Args:
        op: Local ``(local_dim, local_dim)`` operator matrix.
        length: Number of sites in the chain.
        site: Site index on which ``op`` acts.
        local_dim: Uniform local dimension when ``physical_dimensions`` is not provided.
        physical_dimensions: Optional per-site dimensions (broadcast int or length-``length``
            list). When set, overrides ``local_dim``.

    Returns:
        Dense ``(prod(local_dim**length), prod(local_dim**length))`` embedded operator.

    Raises:
        ValueError: If ``site`` is out of range or ``op`` has the wrong shape.
    """
    if site < 0 or site >= length:
        msg = f"site {site} out of range for length {length}."
        raise ValueError(msg)
    dims = _resolve_site_dims(length, physical_dimensions, local_dim=local_dim)
    site_dim = dims[site]
    op_arr = np.asarray(op, dtype=np.complex128)
    if op_arr.shape != (site_dim, site_dim):
        msg = f"op must have shape ({site_dim}, {site_dim}), got {op_arr.shape}."
        raise ValueError(msg)
    res = np.eye(1, dtype=np.complex128)
    for k in range(length):
        eye_k = np.eye(dims[k], dtype=np.complex128)
        local = op_arr if k == site else eye_k
        res = np.kron(local, res)
    return np.asarray(res, dtype=np.complex128)


def embed_adjacent_two_site_operator(
    op4: NDArray[np.complex128],
    length: int,
    site_left: int,
    *,
    local_dim: int = 2,
    physical_dimensions: list[int] | int | None = None,
) -> NDArray[np.complex128]:
    """Embed a two-site operator on neighboring sites ``(site_left, site_left + 1)``.

    Indexing matches :func:`embed_one_site_operator` (site ``0`` = LSB).

    Args:
        op4: Local ``(local_dim**2, local_dim**2)`` operator on the adjacent pair.
        length: Number of sites in the chain.
        site_left: Left site index of the pair.
        local_dim: Uniform local dimension when ``physical_dimensions`` is not provided.
        physical_dimensions: Optional per-site dimensions (broadcast int or length-``length``
            list). When set, overrides ``local_dim``.

    Returns:
        Dense embedded operator on the full Hilbert space.

    Raises:
        ValueError: If the pair is not inside the chain or ``op4`` has the wrong shape.
    """
    site_right = site_left + 1
    if site_left < 0 or site_right >= length:
        msg = f"adjacent pair ({site_left}, {site_right}) invalid for length {length}."
        raise ValueError(msg)
    dims = _resolve_site_dims(length, physical_dimensions, local_dim=local_dim)
    pair_dim = dims[site_left] * dims[site_right]
    op_arr = np.asarray(op4, dtype=np.complex128)
    if op_arr.shape != (pair_dim, pair_dim):
        msg = f"op4 must have shape ({pair_dim}, {pair_dim}), got {op_arr.shape}."
        raise ValueError(msg)
    res = np.eye(1, dtype=np.complex128)
    site = 0
    while site < length:
        if site == site_left:
            res = np.kron(op_arr, res)
            site += 2
        else:
            res = np.kron(np.eye(dims[site], dtype=np.complex128), res)
            site += 1
    return np.asarray(res, dtype=np.complex128)


def embed_two_site_factors(
    op1: NDArray[np.complex128],
    op2: NDArray[np.complex128],
    length: int,
    site1: int,
    site2: int,
    *,
    local_dim: int = 2,
    physical_dimensions: list[int] | int | None = None,
) -> NDArray[np.complex128]:
    """Embed a product of local operators on two (possibly non-adjacent) sites.

    Args:
        op1: Local operator on ``site1``.
        op2: Local operator on ``site2``.
        length: Number of sites in the chain.
        site1: First site index.
        site2: Second site index.
        local_dim: Uniform local dimension when ``physical_dimensions`` is not provided.
        physical_dimensions: Optional per-site dimensions (broadcast int or length-``length``
            list). When set, overrides ``local_dim``.

    Returns:
        Dense embedded operator on the full Hilbert space.

    Raises:
        ValueError: If site indices are invalid or operator shapes mismatch.
    """
    if site1 == site2:
        msg = "site1 and site2 must differ."
        raise ValueError(msg)
    for site in (site1, site2):
        if site < 0 or site >= length:
            msg = f"site {site} out of range for length {length}."
            raise ValueError(msg)
    op1_arr = np.asarray(op1, dtype=np.complex128)
    op2_arr = np.asarray(op2, dtype=np.complex128)
    dims = _resolve_site_dims(length, physical_dimensions, local_dim=local_dim)
    if op1_arr.shape != (dims[site1], dims[site1]) or op2_arr.shape != (dims[site2], dims[site2]):
        msg = (
            f"local operators must match site dimensions "
            f"({dims[site1]}, {dims[site1]}) and ({dims[site2]}, {dims[site2]})."
        )
        raise ValueError(msg)
    res = np.eye(1, dtype=np.complex128)
    for k in range(length):
        if k == site1:
            local = op1_arr
        elif k == site2:
            local = op2_arr
        else:
            local = np.eye(dims[k], dtype=np.complex128)
        res = np.kron(local, res)
    return np.asarray(res, dtype=np.complex128)


def normalize_density_matrix(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
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
