# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Named observable definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np

from .operator_matrices import (
    CONTROLLED_X,
    CONTROLLED_Z,
    HADAMARD,
    IDENTITY,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    PROJECTOR_ONE,
    PROJECTOR_ZERO,
    SWAP_MATRIX,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray

ObservableKind = Literal["operator", "diagnostic"]


@dataclass(frozen=True)
class ObservableDefinition:
    """Definition returned by a named observable factory.

    Attributes:
        name: Canonical observable name.
        matrix: Local operator matrix, or ``None`` for a state diagnostic.
        interaction: Number of sites used by a local operator.
        kind: Whether the definition is an operator or a state diagnostic.
    """

    name: str
    matrix: NDArray[np.complex128] | None
    interaction: int
    kind: ObservableKind = "operator"


def _operator(name: str, matrix: ArrayLike, interaction: int) -> ObservableDefinition:
    """Return an independent operator definition.

    Args:
        name: Canonical observable name.
        matrix: Operator matrix.
        interaction: Number of sites on which the matrix acts.

    Returns:
        A named operator definition.
    """
    return ObservableDefinition(name, np.array(matrix, dtype=np.complex128, copy=True), interaction)


class ObservableLibrary:
    """Factories for the built-in Hermitian observables."""

    _ALIASES: ClassVar[dict[str, str]] = {"i": "id", "iden": "id"}
    _NON_HERMITIAN_GATE_NAMES: ClassVar[frozenset[str]] = frozenset({
        "cp",
        "create",
        "destroy",
        "p",
        "rx",
        "rxx",
        "ry",
        "ryy",
        "rz",
        "rzz",
        "s",
        "sdg",
        "sx",
        "sxdg",
        "t",
        "tdg",
        "u",
        "u1",
        "u2",
        "u3",
    })

    @staticmethod
    def x() -> ObservableDefinition:
        """Return the Pauli-X observable."""
        return _operator("x", PAULI_X, 1)

    @staticmethod
    def y() -> ObservableDefinition:
        """Return the Pauli-Y observable."""
        return _operator("y", PAULI_Y, 1)

    @staticmethod
    def z() -> ObservableDefinition:
        """Return the Pauli-Z observable."""
        return _operator("z", PAULI_Z, 1)

    @staticmethod
    def h() -> ObservableDefinition:
        """Return the Hadamard observable."""
        return _operator("h", HADAMARD, 1)

    @staticmethod
    def id() -> ObservableDefinition:
        """Return the one-site identity observable."""
        return _operator("id", IDENTITY, 1)

    @staticmethod
    def xx() -> ObservableDefinition:
        """Return the two-site Pauli-XX observable."""
        return _operator("xx", np.kron(PAULI_X, PAULI_X), 2)

    @staticmethod
    def yy() -> ObservableDefinition:
        """Return the two-site Pauli-YY observable."""
        return _operator("yy", np.kron(PAULI_Y, PAULI_Y), 2)

    @staticmethod
    def zz() -> ObservableDefinition:
        """Return the two-site Pauli-ZZ observable."""
        return _operator("zz", np.kron(PAULI_Z, PAULI_Z), 2)

    @staticmethod
    def cx() -> ObservableDefinition:
        """Return the two-site controlled-X observable."""
        return _operator("cx", CONTROLLED_X, 2)

    @staticmethod
    def cz() -> ObservableDefinition:
        """Return the two-site controlled-Z observable."""
        return _operator("cz", CONTROLLED_Z, 2)

    @staticmethod
    def swap() -> ObservableDefinition:
        """Return the two-site SWAP observable."""
        return _operator("swap", SWAP_MATRIX, 2)

    @staticmethod
    def p0() -> ObservableDefinition:
        """Return the one-site projector onto zero."""
        return _operator("p0", PROJECTOR_ZERO, 1)

    @staticmethod
    def p1() -> ObservableDefinition:
        """Return the one-site projector onto one."""
        return _operator("p1", PROJECTOR_ONE, 1)

    @staticmethod
    def position(*, positions: ArrayLike) -> ObservableDefinition:
        """Return a position operator diagonal in a supplied local basis.

        Args:
            positions: Real position values for the local basis.

        Returns:
            The position observable definition.

        Raises:
            ValueError: If ``positions`` is complex, empty, non-finite, or not one-dimensional.
        """
        values = np.asarray(positions)
        if np.iscomplexobj(values):
            msg = "positions must contain only real values."
            raise ValueError(msg)
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            msg = "positions must be a non-empty one-dimensional array."
            raise ValueError(msg)
        if not np.all(np.isfinite(values)):
            msg = "positions must contain only finite values."
            raise ValueError(msg)
        return _operator("position", np.diag(values), 1)

    @staticmethod
    def entropy() -> ObservableDefinition:
        """Return an entanglement-entropy diagnostic definition."""
        return ObservableDefinition("entropy", None, 0, "diagnostic")

    @staticmethod
    def schmidt_spectrum() -> ObservableDefinition:
        """Return a Schmidt-spectrum diagnostic definition."""
        return ObservableDefinition("schmidt_spectrum", None, 0, "diagnostic")

    @classmethod
    def resolve(cls, name: str, **kwargs: object) -> ObservableDefinition:
        """Resolve a public name through an explicit observable factory.

        Args:
            name: Public observable name.
            **kwargs: Arguments for a configurable factory.

        Returns:
            The named observable definition.

        Raises:
            ValueError: If the name belongs to a non-Hermitian gate or is unknown.
        """
        canonical_name = cls._ALIASES.get(name, name)
        factories: dict[str, Callable[..., ObservableDefinition]] = {
            "cx": cls.cx,
            "cz": cls.cz,
            "entropy": cls.entropy,
            "h": cls.h,
            "id": cls.id,
            "p0": cls.p0,
            "p1": cls.p1,
            "position": cls.position,
            "schmidt_spectrum": cls.schmidt_spectrum,
            "swap": cls.swap,
            "x": cls.x,
            "xx": cls.xx,
            "y": cls.y,
            "yy": cls.yy,
            "z": cls.z,
            "zz": cls.zz,
        }
        if canonical_name in factories:
            return factories[canonical_name](**kwargs)
        if canonical_name in cls._NON_HERMITIAN_GATE_NAMES:
            msg = f"{name!r} is a gate name, not a Hermitian observable."
            raise ValueError(msg)
        msg = f"Unknown observable {name!r}."
        raise ValueError(msg)
