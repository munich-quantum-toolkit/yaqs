# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Dense clean-circuit out-of-time-order correlator helpers.

The functions in this module evaluate Pauli OTOCs from a supplied dense clean
unitary. They do not construct noisy final MPOs. Clean-circuit OTOC measures
cross-cut perturbation spreading capacity, while final-MPO entropy measures
accumulated operator-Schmidt complexity of the noisy verification MPO.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

Normalization = Literal["raw", "bounded"]

_PAULI_MATRICES: dict[str, NDArray[np.complex128]] = {
    "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def pauli_matrix(label: str) -> NDArray[np.complex128]:
    """Return a single-qubit Pauli matrix.

    Args:
        label: Pauli label. Supported values are ``"I"``, ``"X"``, ``"Y"``, and ``"Z"``.

    Returns:
        The requested Pauli matrix as a complex NumPy array.

    Raises:
        ValueError: If ``label`` is not supported.
    """
    try:
        return _PAULI_MATRICES[label].copy()
    except KeyError as exc:
        msg = f"Unsupported Pauli label {label!r}. Expected one of {tuple(_PAULI_MATRICES)}."
        raise ValueError(msg) from exc


def embed_single_site_operator(
    local_op: np.ndarray,
    site: int,
    num_qubits: int,
) -> NDArray[np.complex128]:
    """Embed a single-site operator into a dense spin-chain operator.

    Site index 0 is the leftmost qubit in the Kronecker product. For two qubits,
    ``site=0`` means ``local_op`` tensor ``I`` and ``site=1`` means ``I`` tensor
    ``local_op``.

    Args:
        local_op: Single-qubit operator with shape ``(2, 2)``.
        site: Site where ``local_op`` is embedded.
        num_qubits: Number of qubits in the spin chain.

    Returns:
        Dense embedded operator with shape ``(2**num_qubits, 2**num_qubits)``.

    Raises:
        ValueError: If the operator shape, site, or number of qubits is invalid.
    """
    _validate_num_qubits(num_qubits)
    _validate_site(site, num_qubits)

    local_array = np.asarray(local_op, dtype=np.complex128)
    if local_array.shape != (2, 2):
        msg = f"local_op must have shape (2, 2), but has shape {local_array.shape}."
        raise ValueError(msg)

    factors = [pauli_matrix("I") for _ in range(num_qubits)]
    factors[site] = local_array

    embedded = factors[0]
    for factor in factors[1:]:
        embedded = np.kron(embedded, factor)
    return np.asarray(embedded, dtype=np.complex128)


def pair_pauli_otoc(
    unitary: np.ndarray,
    source_site: int,
    target_site: int,
    source_pauli: str,
    target_pauli: str,
    num_qubits: int,
    normalization: Normalization = "bounded",
) -> float:
    """Compute a dense clean-circuit Pauli OTOC for one source-target pair.

    The source Pauli is evolved as ``W_i(D) = U(D).conj().T @ W_i @ U(D)`` and
    evaluated against the target Pauli using the normalized commutator norm.

    Args:
        unitary: Dense clean unitary matrix ``U(D)``.
        source_site: Site of the initially local source Pauli.
        target_site: Site of the target Pauli.
        source_pauli: Source Pauli label.
        target_pauli: Target Pauli label.
        num_qubits: Number of qubits in the spin chain.
        normalization: ``"raw"`` returns ``Tr(C.conj().T @ C).real / (2*d)``;
            ``"bounded"`` returns half of the raw value.

    Returns:
        The requested OTOC value as a Python float.
    """
    unitary_array = _validate_unitary_shape(unitary, num_qubits)
    _validate_site(source_site, num_qubits)
    _validate_site(target_site, num_qubits)
    _validate_normalization(normalization)

    source = embed_single_site_operator(pauli_matrix(source_pauli), source_site, num_qubits)
    target = embed_single_site_operator(pauli_matrix(target_pauli), target_site, num_qubits)

    evolved_source = unitary_array.conj().T @ source @ unitary_array
    commutator = evolved_source @ target - target @ evolved_source
    dimension = 2**num_qubits
    raw = _real_float(np.trace(commutator.conj().T @ commutator) / (2 * dimension))

    if normalization == "raw":
        return raw
    return raw / 2


def averaged_pair_otoc(
    unitary: np.ndarray,
    source_site: int,
    target_site: int,
    num_qubits: int,
    paulis: Sequence[str] = ("X", "Y", "Z"),
    normalization: Normalization = "bounded",
) -> float:
    """Average the pair OTOC over all supplied source and target Pauli labels.

    Args:
        unitary: Dense clean unitary matrix ``U(D)``.
        source_site: Site of the initially local source Pauli.
        target_site: Site of the target Pauli.
        num_qubits: Number of qubits in the spin chain.
        paulis: Pauli labels to use for both the source and target averages.
        normalization: OTOC normalization convention.

    Returns:
        Pauli-averaged OTOC value as a Python float.
    """
    pauli_labels = _validate_paulis(paulis)
    values = [
        pair_pauli_otoc(
            unitary,
            source_site,
            target_site,
            source_pauli,
            target_pauli,
            num_qubits,
            normalization,
        )
        for source_pauli in pauli_labels
        for target_pauli in pauli_labels
    ]
    return float(np.mean(values))


def cross_cut_otoc(
    unitary: np.ndarray,
    left_sites: Sequence[int],
    right_sites: Sequence[int],
    num_qubits: int,
    paulis: Sequence[str] = ("X", "Y", "Z"),
    normalization: Normalization = "bounded",
) -> dict[str, float]:
    """Compute directed and symmetric clean-circuit OTOC averages across a cut.

    Example:
        ```python
        from mqt.yaqs.core.methods.otoc import cross_cut_otoc

        result = cross_cut_otoc(
            unitary=U,
            left_sites=range(3),
            right_sites=range(3, 6),
            num_qubits=6,
            normalization="bounded",
        )

        symmetric_otoc = result["symmetric"]
        ```

    Args:
        unitary: Dense clean unitary matrix ``U(D)``.
        left_sites: Sites on the left side of the spatial cut.
        right_sites: Sites on the right side of the spatial cut.
        num_qubits: Number of qubits in the spin chain.
        paulis: Pauli labels to average over.
        normalization: OTOC normalization convention.

    Returns:
        Dictionary containing ``"left_to_right"``, ``"right_to_left"``, and
        ``"symmetric"`` OTOC values.
    """
    left = _validate_sites(left_sites, num_qubits, "left_sites")
    right = _validate_sites(right_sites, num_qubits, "right_sites")
    pauli_labels = _validate_paulis(paulis)
    _validate_unitary_shape(unitary, num_qubits)
    _validate_normalization(normalization)

    left_to_right = _average_directed_cross_cut(unitary, left, right, num_qubits, pauli_labels, normalization)
    right_to_left = _average_directed_cross_cut(unitary, right, left, num_qubits, pauli_labels, normalization)
    symmetric = 0.5 * (left_to_right + right_to_left)

    return {
        "left_to_right": left_to_right,
        "right_to_left": right_to_left,
        "symmetric": symmetric,
    }


def cross_cut_otoc_by_depth(
    unitary_by_depth: Mapping[int, np.ndarray] | Sequence[tuple[int, np.ndarray]],
    left_sites: Sequence[int],
    right_sites: Sequence[int],
    num_qubits: int,
    paulis: Sequence[str] = ("X", "Y", "Z"),
    normalization: Normalization = "bounded",
) -> list[dict[str, float | int]]:
    """Compute cross-cut clean-circuit OTOC rows for a depth series.

    If ``unitary_by_depth`` is a mapping, depths are processed in sorted key order.
    If it is a sequence of ``(depth, unitary)`` pairs, the given order is preserved.

    Example:
        ```python
        rows = cross_cut_otoc_by_depth(
            unitary_by_depth={D: U_D for D, U_D in unitary_by_depth.items()},
            left_sites=range(3),
            right_sites=range(3, 6),
            num_qubits=6,
        )
        ```

    Args:
        unitary_by_depth: Mapping or ordered sequence of depths to dense clean unitaries.
        left_sites: Sites on the left side of the spatial cut.
        right_sites: Sites on the right side of the spatial cut.
        num_qubits: Number of qubits in the spin chain.
        paulis: Pauli labels to average over.
        normalization: OTOC normalization convention.

    Returns:
        A list of dictionaries with ``"depth"``, ``"left_to_right"``,
        ``"right_to_left"``, and ``"symmetric"`` entries.
    """
    if isinstance(unitary_by_depth, Mapping):
        depth_unitary_pairs = [(depth, unitary_by_depth[depth]) for depth in sorted(unitary_by_depth)]
    else:
        depth_unitary_pairs = list(unitary_by_depth)

    rows: list[dict[str, float | int]] = []
    for depth, unitary in depth_unitary_pairs:
        values = cross_cut_otoc(unitary, left_sites, right_sites, num_qubits, paulis, normalization)
        rows.append({"depth": depth, **values})
    return rows


def _average_directed_cross_cut(
    unitary: np.ndarray,
    source_sites: Sequence[int],
    target_sites: Sequence[int],
    num_qubits: int,
    paulis: Sequence[str],
    normalization: Normalization,
) -> float:
    values = [
        averaged_pair_otoc(unitary, source_site, target_site, num_qubits, paulis, normalization)
        for source_site in source_sites
        for target_site in target_sites
    ]
    return float(np.mean(values))


def _validate_num_qubits(num_qubits: int) -> None:
    if num_qubits <= 0:
        msg = f"num_qubits must be positive, but is {num_qubits}."
        raise ValueError(msg)


def _validate_site(site: int, num_qubits: int) -> None:
    if site < 0 or site >= num_qubits:
        msg = f"site must satisfy 0 <= site < num_qubits, but got site={site} and num_qubits={num_qubits}."
        raise ValueError(msg)


def _validate_sites(sites: Sequence[int], num_qubits: int, name: str) -> list[int]:
    site_list = list(sites)
    if not site_list:
        msg = f"{name} must be non-empty."
        raise ValueError(msg)
    for site in site_list:
        _validate_site(site, num_qubits)
    return site_list


def _validate_paulis(paulis: Sequence[str]) -> tuple[str, ...]:
    pauli_labels = tuple(paulis)
    if not pauli_labels:
        msg = "paulis must be non-empty."
        raise ValueError(msg)
    for label in pauli_labels:
        pauli_matrix(label)
    return pauli_labels


def _validate_normalization(normalization: str) -> None:
    if normalization not in {"raw", "bounded"}:
        msg = f"normalization must be 'raw' or 'bounded', but is {normalization!r}."
        raise ValueError(msg)


def _validate_unitary_shape(unitary: np.ndarray, num_qubits: int) -> NDArray[np.complex128]:
    _validate_num_qubits(num_qubits)
    unitary_array = np.asarray(unitary, dtype=np.complex128)
    dimension = 2**num_qubits
    expected_shape = (dimension, dimension)
    if unitary_array.shape != expected_shape:
        msg = (
            f"unitary must have shape {expected_shape} for num_qubits={num_qubits}, "
            f"but has shape {unitary_array.shape}."
        )
        raise ValueError(msg)
    return unitary_array


def _real_float(value: complex) -> float:
    real_value = float(np.real(value))
    imaginary_value = float(np.imag(value))
    tolerance = 1e-10 * max(1.0, abs(real_value))
    if abs(imaginary_value) > tolerance:
        msg = f"Expected a real OTOC value, but the imaginary residual is {imaginary_value}."
        raise ValueError(msg)
    return real_value
