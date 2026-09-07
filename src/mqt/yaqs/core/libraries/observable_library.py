# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Named observable definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from .operator_matrices import (
    IDENTITY,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    PROJECTOR_ONE,
    PROJECTOR_ZERO,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray

ObservableType = Literal["operator", "diagnostic"]


@dataclass(frozen=True)
class ObservableDefinition:
    """Definition returned by a named observable factory.

    Attributes:
        name: Canonical observable name.
        matrix: Local operator matrix, or ``None`` for a state diagnostic.
        interaction: Number of sites used by a local operator.
        type: Whether the definition is an operator or a state diagnostic.
    """

    name: str
    matrix: NDArray[np.complex128] | None
    interaction: int
    type: ObservableType = "operator"


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

    @staticmethod
    def x() -> ObservableDefinition:
        """Return the Pauli-X observable definition.

        Returns:
            The Pauli-X observable definition.
        """
        return _operator("x", PAULI_X, 1)

    @staticmethod
    def y() -> ObservableDefinition:
        """Return the Pauli-Y observable definition.

        Returns:
            The Pauli-Y observable definition.
        """
        return _operator("y", PAULI_Y, 1)

    @staticmethod
    def z() -> ObservableDefinition:
        """Return the Pauli-Z observable definition.

        Returns:
            The Pauli-Z observable definition.
        """
        return _operator("z", PAULI_Z, 1)

    @staticmethod
    def id() -> ObservableDefinition:
        """Return the one-site identity observable definition.

        Returns:
            The one-site identity observable definition.
        """
        return _operator("id", IDENTITY, 1)

    @staticmethod
    def xx() -> ObservableDefinition:
        """Return the two-site Pauli-XX observable definition.

        Returns:
            The two-site Pauli-XX observable definition.
        """
        return _operator("xx", np.kron(PAULI_X, PAULI_X), 2)

    @staticmethod
    def yy() -> ObservableDefinition:
        """Return the two-site Pauli-YY observable definition.

        Returns:
            The two-site Pauli-YY observable definition.
        """
        return _operator("yy", np.kron(PAULI_Y, PAULI_Y), 2)

    @staticmethod
    def zz() -> ObservableDefinition:
        """Return the two-site Pauli-ZZ observable definition.

        Returns:
            The two-site Pauli-ZZ observable definition.
        """
        return _operator("zz", np.kron(PAULI_Z, PAULI_Z), 2)

    @staticmethod
    def p0() -> ObservableDefinition:
        """Return the one-site projector onto zero.

        Returns:
            The projector onto the zero state.
        """
        return _operator("p0", PROJECTOR_ZERO, 1)

    @staticmethod
    def p1() -> ObservableDefinition:
        """Return the one-site projector onto one.

        Returns:
            The projector onto the one state.
        """
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
        """Return an entanglement-entropy diagnostic definition.

        Returns:
            The entanglement-entropy diagnostic definition.
        """
        return ObservableDefinition("entropy", None, 0, "diagnostic")

    @staticmethod
    def schmidt_spectrum() -> ObservableDefinition:
        """Return a Schmidt-spectrum diagnostic definition.

        Returns:
            The Schmidt-spectrum diagnostic definition.
        """
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
            ValueError: If the observable name is unknown.
        """
        factories: dict[str, Callable[..., ObservableDefinition]] = {
            "entropy": cls.entropy,
            "id": cls.id,
            "p0": cls.p0,
            "p1": cls.p1,
            "position": cls.position,
            "schmidt_spectrum": cls.schmidt_spectrum,
            "x": cls.x,
            "xx": cls.xx,
            "y": cls.y,
            "yy": cls.yy,
            "z": cls.z,
            "zz": cls.zz,
        }
        if name in factories:
            return factories[name](**kwargs)
        msg = f"Unknown observable {name!r}."
        raise ValueError(msg)
