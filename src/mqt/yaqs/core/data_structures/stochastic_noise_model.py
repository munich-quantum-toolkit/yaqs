# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Stochastic noise models for digital circuit simulation.

Unlike :class:`~mqt.yaqs.NoiseModel`, the models in this module do not
describe continuous-time Lindblad processes. Their parameter ``p`` controls a
gate-local stochastic channel and is never interpreted as a Lindblad rate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeAlias, TypeGuard

import numpy as np


def _validate_probability(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        msg = "p must be a real number (booleans are not allowed)."
        raise TypeError(msg)
    parameter = float(value)
    if not math.isfinite(parameter):
        msg = f"p must be finite, got {parameter}."
        raise ValueError(msg)
    if not 0.0 <= parameter <= 1.0:
        msg = f"p must lie in [0, 1], got {parameter}."
        raise ValueError(msg)
    return parameter


@dataclass(frozen=True, slots=True)
class XYZPauliNoiseModel:
    r"""Gate-local XYZ Pauli channel for digital circuit simulation.

    ``p`` is the total local error probability. Each application samples one
    mutually exclusive member of the I, X, Y, and Z channel:

    .. math::

       P(I)=1-p,\qquad P(X)=P(Y)=P(Z)=\frac{p}{3}.

    The channel is applied independently to every qubit touched by an ideal
    one- or two-qubit gate.

    Args:
        p: Total local Pauli-error probability in the closed interval ``[0, 1]``.
    """

    p: float

    def __post_init__(self) -> None:
        """Validate and normalize the channel parameter."""
        object.__setattr__(self, "p", _validate_probability(self.p))

    @property
    def probabilities(self) -> dict[str, float]:
        """The one-qubit I/X/Y/Z probabilities."""
        pauli_probability = self.p / 3.0
        return {
            "I": 1.0 - self.p,
            "X": pauli_probability,
            "Y": pauli_probability,
            "Z": pauli_probability,
        }

    @property
    def is_noiseless(self) -> bool:
        """Whether this model is exactly the ideal channel."""
        return not self.p


@dataclass(frozen=True, slots=True)
class XBasisDissipativeNoiseModel:
    r"""Gate-local X-basis dissipative channel for digital circuit simulation.

    The channel irreversibly maps :math:`|-⟩` to :math:`|+⟩` through

    .. math::

       K_0=|+⟩\langle+|+\sqrt{1-p}|-⟩\langle-|,\qquad
       K_1=\sqrt{p}|+⟩\langle-|.

    Args:
        p: X-basis damping parameter in the closed interval ``[0, 1]``. It
            parametrizes the Kraus operators applied after each supported gate
            to every touched qubit; it is not an outer Bernoulli probability.
    """

    p: float

    def __post_init__(self) -> None:
        """Validate and normalize the channel parameter."""
        object.__setattr__(self, "p", _validate_probability(self.p))

    def kraus_operators(self) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of the complete Kraus pair ``(K0, K1)``."""
        plus = np.asarray([1.0, 1.0], dtype=np.complex128) / math.sqrt(2.0)
        minus = np.asarray([1.0, -1.0], dtype=np.complex128) / math.sqrt(2.0)
        plus_projector = np.outer(plus, plus.conj())
        minus_projector = np.outer(minus, minus.conj())
        k0 = plus_projector + math.sqrt(1.0 - self.p) * minus_projector
        k1 = math.sqrt(self.p) * np.outer(plus, minus.conj())
        return k0, k1

    @property
    def is_noiseless(self) -> bool:
        """Whether this model is exactly the ideal channel."""
        return not self.p


StochasticNoiseModel: TypeAlias = XYZPauliNoiseModel | XBasisDissipativeNoiseModel


def _is_stochastic_noise_model(value: object) -> TypeGuard[StochasticNoiseModel]:
    """Return whether ``value`` is a supported stochastic gate-local noise model."""
    return isinstance(value, (XYZPauliNoiseModel, XBasisDissipativeNoiseModel))
