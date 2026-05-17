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
