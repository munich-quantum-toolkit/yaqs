# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the state-preparation benchmark's product-Pauli sampler."""

from __future__ import annotations

import copy
import math
from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

import benchmarks.state_preparation.noise as noise_module
import mqt.yaqs.optimization.krotov as krotov_module
from benchmarks.state_preparation import (
    PauliDistribution,
    sample_local_pauli,
    sample_product_pauli_channel,
)
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.optimization import (
    GateNoiseContext,
    KrotovTJMOptions,
    KrotovTruncation,
    ParameterizedCircuit,
    ParameterizedGate,
    RandomUnitaryInstruction,
    forward_tjm_trajectory,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpy.typing import NDArray

    from mqt.yaqs.optimization import GateNoiseProvider, KrotovNoiseMap, LocalOperator


_LABELS = ("I", "X", "Y", "Z")
_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
_PAULIS: Mapping[str, NDArray[np.complex128]] = {
    "I": _I,
    "X": _X,
    "Y": _Y,
    "Z": _Z,
}
_TJM_BOUNDARY_MARGIN = 1e-12


@dataclass
class _SequenceRNG:
    """Minimal deterministic stand-in for ``Generator.random``."""

    values: tuple[float, ...]
    calls: int = 0

    def random(self) -> float:
        """Return the next configured scalar draw.

        Raises:
            AssertionError: If production consumes more draws than configured.
        """
        if self.calls >= len(self.values):
            msg = "The sampler consumed more random draws than expected."
            raise AssertionError(msg)
        value = self.values[self.calls]
        self.calls += 1
        return value


@dataclass
class _ScalarRNG:
    """RNG test double that returns one arbitrary scalar-like value."""

    value: object
    calls: int = 0

    def random(self) -> object:
        """Return the configured value and record the draw."""
        self.calls += 1
        return self.value


@dataclass
class _TJMOutcomeRNG:
    """Controlled RNG that selects one actual circuit-TJM branch."""

    jump_draw: float
    choice_index: int | None
    random_calls: int = 0
    choice_calls: int = 0
    choice_probabilities: NDArray[np.float64] | None = None

    def random(self) -> float:
        """Return the configured jump/no-jump decision scalar."""
        self.random_calls += 1
        return self.jump_draw

    def choice(
        self,
        size: int,
        *,
        p: NDArray[np.float64],
    ) -> int:
        """Return the configured jump-process index after recording weights.

        Raises:
            AssertionError: If a no-jump oracle unexpectedly requests a choice
                or the requested index is outside the supplied process range.
        """
        if self.choice_index is None:
            msg = "A no-jump TJM outcome unexpectedly requested a process choice."
            raise AssertionError(msg)
        if not 0 <= self.choice_index < size:
            msg = f"Configured TJM choice {self.choice_index} lies outside range(0, {size})."
            raise AssertionError(msg)
        self.choice_calls += 1
        self.choice_probabilities = np.asarray(p, dtype=np.float64).copy()
        return self.choice_index


def _as_generator(rng: _SequenceRNG) -> np.random.Generator:
    """Return a deterministic test double cast to the production RNG protocol."""
    return cast("np.random.Generator", rng)


def _as_tjm_generator(rng: _TJMOutcomeRNG) -> np.random.Generator:
    """Return a controlled TJM test double cast to the production RNG protocol."""
    return cast("np.random.Generator", rng)


def _sample_local(
    distribution: object,
    site: object,
    rng: np.random.Generator,
) -> LocalOperator | None:
    """Return one sample while keeping deliberately invalid test inputs typed."""
    return sample_local_pauli(
        cast("PauliDistribution", distribution),
        cast("int", site),
        rng,
    )


def _sample_product(
    first_site: object,
    second_site: object,
    first_distribution: object,
    second_distribution: object,
    rng: np.random.Generator,
) -> tuple[LocalOperator, ...]:
    """Return one product sample while keeping invalid test inputs typed."""
    return sample_product_pauli_channel(
        cast("int", first_site),
        cast("int", second_site),
        cast("PauliDistribution", first_distribution),
        cast("PauliDistribution", second_distribution),
        rng,
    )


def _probabilities(distribution: Mapping[str, Any]) -> dict[str, float]:
    """Return a complete canonical probability dictionary."""
    return {label: float(distribution.get(label, 0.0)) for label in _LABELS}


def _joint_probabilities(
    first_distribution: Mapping[str, Any],
    second_distribution: Mapping[str, Any],
) -> dict[tuple[str, str], float]:
    """Return the exact product probability of every two-site branch."""
    first = _probabilities(first_distribution)
    second = _probabilities(second_distribution)
    return {
        (first_label, second_label): first[first_label] * second[second_label]
        for first_label, second_label in product(_LABELS, repeat=2)
    }


def _branch_draw(distribution: Mapping[str, Any], label: str) -> float:
    """Return an interior scalar draw for one positive-probability branch.

    Raises:
        ValueError: If the requested branch has zero probability.
    """
    probabilities = _probabilities(distribution)
    lower = sum(probabilities[item] for item in _LABELS[: _LABELS.index(label)])
    probability = probabilities[label]
    if probability <= 0.0:
        msg = f"Cannot choose an interior draw for zero-probability branch {label}."
        raise ValueError(msg)
    return lower + probability / 2.0


def _outcome_labels(
    operators: tuple[LocalOperator, ...],
    first_site: int,
    second_site: int,
) -> tuple[str, str]:
    """Return both local outcomes reconstructed from identity-elided operators."""
    by_site = {operator.sites[0]: operator.label for operator in operators}
    return cast("tuple[str, str]", (by_site.get(first_site, "I"), by_site.get(second_site, "I")))


def _density(vector: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Return the pure-state density matrix of a state vector."""
    return np.outer(vector, vector.conj())


def _apply_sampled_operators(
    vector: NDArray[np.complex128],
    operators: tuple[LocalOperator, ...],
) -> NDArray[np.complex128]:
    """Return the state after applying a realized branch through the MPS one-site path."""
    state = MPS.from_statevector(vector)
    for operator in operators:
        state.apply_local_operator(operator.matrix, operator.sites)
    return np.asarray(state.to_vec(), dtype=np.complex128)


def _combined_tjm_rates(r: float, dt: float) -> dict[tuple[str, str], float]:
    """Construct the test-only combined 15-branch TJM calibration.

    Returns:
        A rate for every non-identity product branch.

    Raises:
        ValueError: If the no-jump branch has zero probability.
    """
    identity_probability = (1.0 - 3.0 * r) ** 2
    if math.isclose(identity_probability, 1.0, rel_tol=0.0, abs_tol=1e-15):
        return {}
    if math.isclose(identity_probability, 0.0, rel_tol=0.0, abs_tol=1e-15):
        msg = "The logarithmic TJM oracle is undefined when P_II is zero."
        raise ValueError(msg)
    total_rate = -math.log(identity_probability) / dt
    branch_probabilities = _joint_probabilities(
        {"I": 1.0 - 3.0 * r, "X": r, "Y": r, "Z": r},
        {"I": 1.0 - 3.0 * r, "X": r, "Y": r, "Z": r},
    )
    return {
        branch: total_rate * probability / (1.0 - identity_probability)
        for branch, probability in branch_probabilities.items()
        if branch != ("I", "I")
    }


def _local_tjm_probabilities(r: float, dt: float) -> dict[str, float]:
    """Construct the test-only calibrated one-step local TJM distribution.

    Returns:
        The exact one-step no-jump and Pauli-jump probabilities.

    Raises:
        ValueError: If the local no-jump branch has zero probability.
    """
    identity_probability = 1.0 - 3.0 * r
    if math.isclose(identity_probability, 1.0, rel_tol=0.0, abs_tol=1e-15):
        return {"I": 1.0, "X": 0.0, "Y": 0.0, "Z": 0.0}
    if math.isclose(identity_probability, 0.0, rel_tol=0.0, abs_tol=1e-15):
        msg = "The logarithmic TJM oracle is undefined when the local no-jump probability is zero."
        raise ValueError(msg)
    total_rate = -math.log(identity_probability) / dt
    rate = total_rate / 3.0
    no_jump = math.exp(-total_rate * dt)
    jump_probability = (1.0 - no_jump) * rate / total_rate
    return {"I": no_jump, "X": jump_probability, "Y": jump_probability, "Z": jump_probability}


def _combined_tjm_model(
    r: float,
    dt: float,
) -> tuple[NoiseModel, tuple[tuple[str, str], ...]]:
    """Return the calibrated combined model and its process branch order."""
    rates = _combined_tjm_rates(r, dt)
    branches = tuple(rates)
    processes: list[dict[str, Any]] = []
    for first_label, second_label in branches:
        if first_label == "I":
            name = f"pauli_{second_label.lower()}"
            sites = [1]
        elif second_label == "I":
            name = f"pauli_{first_label.lower()}"
            sites = [0]
        else:
            name = f"crosstalk_{first_label.lower()}{second_label.lower()}"
            sites = [0, 1]
        processes.append({
            "name": name,
            "sites": sites,
            "strength": rates[first_label, second_label],
        })
    return NoiseModel(processes), branches


def _local_tjm_model(site: int, r: float, dt: float) -> NoiseModel:
    """Return a calibrated three-Pauli TJM model for one site.

    Raises:
        ValueError: If the local no-jump probability is zero or one.
    """
    identity_probability = 1.0 - 3.0 * r
    if identity_probability <= 0.0 or identity_probability >= 1.0:
        msg = "The local TJM model helper requires 0 < r < 1/3."
        raise ValueError(msg)
    rate = -math.log(identity_probability) / (3.0 * dt)
    return NoiseModel([{"name": f"pauli_{label.lower()}", "sites": [site], "strength": rate} for label in _LABELS[1:]])


def _sample_actual_tjm(
    state: MPS,
    noise_model: NoiseModel,
    dt: float,
    rng: _TJMOutcomeRNG,
) -> KrotovNoiseMap:
    """Sample and apply one real YAQS circuit-TJM invocation.

    Returns:
        The realized replay map from the existing optimizer machinery.
    """
    sampler = cast(
        "Callable[[MPS, NoiseModel | None, KrotovTruncation, KrotovTJMOptions, np.random.Generator], KrotovNoiseMap]",
        vars(krotov_module)["_sample_noise_map_and_apply"],
    )
    return sampler(
        state,
        noise_model,
        KrotovTruncation(),
        KrotovTJMOptions(dt=dt),
        _as_tjm_generator(rng),
    )


def _actual_combined_tjm_density(
    vector: NDArray[np.complex128],
    r: float,
    dt: float,
) -> NDArray[np.complex128]:
    """Return density evolution by enumerating actual combined TJM outcomes."""
    model, branches = _combined_tjm_model(r, dt)
    distribution = {"I": 1.0 - 3.0 * r, "X": r, "Y": r, "Z": r}
    branch_probabilities = _joint_probabilities(distribution, distribution)
    rates = _combined_tjm_rates(r, dt)
    expected_choice_probabilities = np.asarray(
        [rates[branch] for branch in branches],
        dtype=np.float64,
    )
    expected_choice_probabilities /= np.sum(expected_choice_probabilities)
    jump_probability = 1.0 - branch_probabilities["I", "I"]
    result = np.zeros((4, 4), dtype=np.complex128)

    no_jump_state = MPS.from_statevector(vector)
    no_jump_rng = _TJMOutcomeRNG(jump_probability + _TJM_BOUNDARY_MARGIN, None)
    no_jump_map = _sample_actual_tjm(no_jump_state, model, dt, no_jump_rng)
    assert no_jump_rng.random_calls == 1
    assert no_jump_rng.choice_calls == 0
    assert no_jump_map.operators == ()
    result += branch_probabilities["I", "I"] * _density(np.asarray(no_jump_state.to_vec(), dtype=np.complex128))

    for choice_index, branch in enumerate(branches):
        jumped_state = MPS.from_statevector(vector)
        jump_rng = _TJMOutcomeRNG(jump_probability - _TJM_BOUNDARY_MARGIN, choice_index)
        jump_map = _sample_actual_tjm(jumped_state, model, dt, jump_rng)
        assert jump_rng.random_calls == 1
        assert jump_rng.choice_calls == 1
        assert jump_rng.choice_probabilities is not None
        np.testing.assert_allclose(
            jump_rng.choice_probabilities,
            expected_choice_probabilities,
            atol=1e-12,
        )
        assert len(jump_map.operators) == 1
        result += branch_probabilities[branch] * _density(np.asarray(jumped_state.to_vec(), dtype=np.complex128))
    return result


def _actual_sequential_local_tjm_density(
    vector: NDArray[np.complex128],
    r: float,
    dt: float,
) -> NDArray[np.complex128]:
    """Return density evolution by enumerating two actual local TJM calls."""
    models = (_local_tjm_model(0, r, dt), _local_tjm_model(1, r, dt))
    distribution = _local_tjm_probabilities(r, dt)
    branch_probabilities = _joint_probabilities(distribution, distribution)
    jump_probability = 1.0 - distribution["I"]
    result = np.zeros((4, 4), dtype=np.complex128)

    for branch, probability in branch_probabilities.items():
        state = MPS.from_statevector(vector)
        for model, label in zip(models, branch, strict=True):
            if label == "I":
                rng = _TJMOutcomeRNG(jump_probability + _TJM_BOUNDARY_MARGIN, None)
            else:
                rng = _TJMOutcomeRNG(
                    jump_probability - _TJM_BOUNDARY_MARGIN,
                    _LABELS[1:].index(label),
                )
            noise_map = _sample_actual_tjm(state, model, dt, rng)
            assert rng.random_calls == 1
            if label == "I":
                assert rng.choice_calls == 0
                assert noise_map.operators == ()
            else:
                assert rng.choice_calls == 1
                assert rng.choice_probabilities is not None
                np.testing.assert_allclose(
                    rng.choice_probabilities,
                    np.full(3, 1.0 / 3.0),
                    atol=1e-12,
                )
                assert len(noise_map.operators) == 1
        result += probability * _density(np.asarray(state.to_vec(), dtype=np.complex128))
    return result


def _density_from_branch_probabilities(
    vector: NDArray[np.complex128],
    probabilities: Mapping[tuple[str, str], float],
) -> NDArray[np.complex128]:
    """Return density evolution under explicitly enumerated Pauli branches."""
    initial_density = _density(vector)
    result = np.zeros((4, 4), dtype=np.complex128)
    for (first_label, second_label), probability in probabilities.items():
        matrix = np.kron(_PAULIS[second_label], _PAULIS[first_label])
        result += probability * matrix @ initial_density @ matrix.conj().T
    return result


def _noise_map_signature(noise_map: KrotovNoiseMap) -> tuple[tuple[str | None, tuple[int, ...]], ...]:
    """Return the state-independent portion of one realized provider map."""
    return tuple(
        (
            next(
                (
                    label
                    for label, matrix in _PAULIS.items()
                    if label != "I" and np.array_equal(realized_matrix, matrix)
                ),
                None,
            ),
            sites,
        )
        for realized_matrix, sites in noise_map.operators
    )


def test_public_product_pauli_api_is_exported() -> None:
    """The benchmark package should expose the sampler's public API."""
    distribution = cast("PauliDistribution", {"I": 1.0})

    assert distribution["I"] == pytest.approx(1.0)
    assert noise_module.sample_local_pauli is sample_local_pauli
    assert noise_module.sample_product_pauli_channel is sample_product_pauli_channel


@pytest.mark.parametrize(
    "distribution",
    [
        {"I": 1.0},
        {"I": np.float64(1.0)},
        {"X": 1.0},
        {"I": 0.1, "X": 0.2, "Y": 0.3, "Z": 0.4},
        {"Z": 0.4, "Y": 0.3, "X": 0.2, "I": 0.1},
        {"I": 0.5, "X": 0.5 + 5e-13},
    ],
    ids=("identity", "numpy-real", "missing-zero-keys", "canonical", "reordered", "sum-within-tolerance"),
)
def test_local_sampler_accepts_valid_mappings(distribution: Mapping[str, Any]) -> None:
    """Valid mappings may omit zero-valued keys and ignore insertion order."""
    rng = _SequenceRNG((0.0,))

    _sample_local(distribution, np.int64(2), _as_generator(rng))

    assert rng.calls == 1


@pytest.mark.parametrize(
    ("distribution", "error", "match"),
    [
        (cast("Any", [1.0, 0.0, 0.0, 0.0]), TypeError, "mapping"),
        ({}, ValueError, "sum"),
        ({cast("Any", 1): 1.0}, TypeError, "labels"),
        ({"i": 1.0}, ValueError, "unknown"),
        ({"A": 1.0}, ValueError, "unknown"),
        ({"I": -0.1, "X": 1.1}, ValueError, "lie"),
        ({"I": 1.1}, ValueError, "lie"),
        ({"I": 10**10000}, ValueError, "lie"),
        ({"I": 0.4, "X": 0.5}, ValueError, "sum"),
        ({"I": 0.5, "X": 0.5 + 2e-12}, ValueError, "sum"),
        ({"I": True}, TypeError, "real"),
        ({"I": np.bool_(1)}, TypeError, "real"),
        ({"I": 1.0 + 0.0j}, TypeError, "real"),
        ({"I": cast("Any", "1.0")}, TypeError, "real"),
        ({"I": np.nan}, ValueError, "finite"),
        ({"I": np.inf}, ValueError, "finite"),
        ({"I": -np.inf}, ValueError, "finite"),
    ],
    ids=(
        "not-mapping",
        "empty",
        "non-string-key",
        "lowercase-key",
        "unknown-key",
        "negative",
        "above-one",
        "overflowing-real",
        "not-normalized",
        "outside-tolerance",
        "boolean",
        "numpy-boolean",
        "complex",
        "string",
        "nan",
        "positive-infinity",
        "negative-infinity",
    ),
)
def test_local_sampler_rejects_invalid_distributions_without_drawing(
    distribution: object,
    error: type[Exception],
    match: str,
) -> None:
    """Malformed distributions fail before consuming the random stream."""
    rng = _SequenceRNG((0.25,))

    with pytest.raises(error, match=match):
        _sample_local(distribution, 0, _as_generator(rng))

    assert rng.calls == 0


@pytest.mark.parametrize(
    ("site", "error", "match"),
    [
        (True, TypeError, "bool"),
        (-1, ValueError, "nonnegative"),
        (1.5, TypeError, "integer"),
        ("1", TypeError, "integer"),
    ],
)
def test_local_sampler_rejects_invalid_sites_before_drawing(
    site: object,
    error: type[Exception],
    match: str,
) -> None:
    """Invalid support cannot advance a trajectory's RNG."""
    rng = _SequenceRNG((0.25,))

    with pytest.raises(error, match=match):
        _sample_local({"I": 1.0}, site, _as_generator(rng))

    assert rng.calls == 0


@pytest.mark.parametrize(
    ("draw", "error"),
    [
        (True, TypeError),
        (np.bool_(0), TypeError),
        ("0.5", TypeError),
        (np.array([0.5]), TypeError),
        (np.nan, ValueError),
        (np.inf, ValueError),
        (-0.1, ValueError),
        (1.0, ValueError),
        (1.1, ValueError),
        (10**10000, ValueError),
    ],
    ids=(
        "boolean",
        "numpy-boolean",
        "string",
        "array",
        "nan",
        "infinity",
        "negative",
        "one",
        "above-one",
        "overflowing-real",
    ),
)
def test_local_sampler_rejects_invalid_rng_draws(draw: object, error: type[Exception]) -> None:
    """Malformed generator outputs fail after exactly one attempted draw."""
    rng = _ScalarRNG(draw)

    with pytest.raises(error, match=r"rng\.random"):
        _sample_local({"I": 1.0}, 0, cast("np.random.Generator", rng))

    assert rng.calls == 1


@pytest.mark.parametrize(
    ("draw", "expected_label"),
    [
        (0.0, None),
        (np.nextafter(0.125, 0.0), None),
        (0.125, "X"),
        (np.nextafter(0.375, 0.0), "X"),
        (0.375, "Y"),
        (np.nextafter(0.625, 0.0), "Y"),
        (0.625, "Z"),
        (np.nextafter(1.0, 0.0), "Z"),
    ],
)
def test_local_sampler_uses_canonical_half_open_boundaries(
    draw: float,
    expected_label: str | None,
) -> None:
    """An exact cumulative boundary advances to the following Pauli."""
    distribution = {"Z": 0.375, "Y": 0.25, "X": 0.25, "I": 0.125}
    rng = _SequenceRNG((draw,))

    operator = _sample_local(distribution, 3, _as_generator(rng))

    assert rng.calls == 1
    if expected_label is None:
        assert operator is None
    else:
        assert operator is not None
        assert operator.label == expected_label
        assert operator.sites == (3,)
        np.testing.assert_array_equal(operator.matrix, _PAULIS[expected_label])


@pytest.mark.parametrize(
    ("draw", "expected_label"),
    [
        (0.0, "X"),
        (np.nextafter(0.5, 0.0), "X"),
        (0.5, "Z"),
        (np.nextafter(1.0, 0.0), "Z"),
    ],
)
def test_local_sampler_skips_zero_width_branches(draw: float, expected_label: str) -> None:
    """Zero-probability identities and Paulis can never be selected."""
    rng = _SequenceRNG((draw,))

    operator = _sample_local({"X": 0.5, "Z": 0.5}, 0, _as_generator(rng))

    assert operator is not None
    assert operator.label == expected_label
    assert rng.calls == 1


def test_local_sampler_assigns_roundoff_gap_to_last_positive_branch() -> None:
    """A tolerated sub-unit sum cannot leave the upper random interval unassigned."""
    rng = _SequenceRNG((np.nextafter(1.0, 0.0),))

    operator = _sample_local({"I": 0.5, "Z": 0.5 - 5e-13}, 0, _as_generator(rng))

    assert operator is not None
    assert operator.label == "Z"
    assert rng.calls == 1


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ({"I": 1.0}, {"I": 1.0}, ()),
        ({"X": 1.0}, {"I": 1.0}, (("X", 4),)),
        ({"I": 1.0}, {"Z": 1.0}, (("Z", 1),)),
        ({"X": 1.0}, {"Y": 1.0}, (("X", 4), ("Y", 1))),
    ],
    ids=("zero-errors", "first-only", "second-only", "two-errors"),
)
def test_product_sampler_elides_identity_and_preserves_call_order(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    expected: tuple[tuple[str, int], ...],
) -> None:
    """Product outcomes contain zero, one, or two bare one-site operators."""
    rng = _SequenceRNG((0.0, 0.0))

    operators = _sample_product(4, 1, first, second, _as_generator(rng))

    assert rng.calls == 2
    assert tuple((operator.label, operator.sites[0]) for operator in operators) == expected
    for operator in operators:
        assert operator.label is not None
        assert operator.sites in {(4,), (1,)}
        assert operator.matrix.shape == (2, 2)
        np.testing.assert_array_equal(operator.matrix, _PAULIS[operator.label])


@pytest.mark.parametrize(("first_label", "second_label"), product(_LABELS, repeat=2))
def test_product_sampler_realizes_all_sixteen_branches_as_single_site_operators(
    first_label: str,
    second_label: str,
) -> None:
    """Every joint branch is two independent identity-elided local outcomes."""
    distribution = {"I": 0.1, "X": 0.2, "Y": 0.3, "Z": 0.4}
    rng = _SequenceRNG((
        _branch_draw(distribution, first_label),
        _branch_draw(distribution, second_label),
    ))

    operators = _sample_product(0, 2, distribution, distribution, _as_generator(rng))

    assert rng.calls == 2
    assert _outcome_labels(operators, 0, 2) == (first_label, second_label)
    assert len(operators) == int(first_label != "I") + int(second_label != "I")
    assert all(operator.matrix.shape == (2, 2) and len(operator.sites) == 1 for operator in operators)


def test_product_sampler_calls_public_local_sampler_twice_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    """The implementation composes two public local calls with one shared RNG."""
    first_distribution = {"I": 1.0}
    second_distribution = {"Z": 1.0}
    rng = _SequenceRNG((0.2, 0.8))
    calls: list[tuple[PauliDistribution, int, np.random.Generator]] = []

    def fake_local_sampler(
        distribution: PauliDistribution,
        site: int,
        generator: np.random.Generator,
    ) -> LocalOperator | None:
        calls.append((distribution, site, generator))
        return None

    monkeypatch.setattr(noise_module, "sample_local_pauli", fake_local_sampler)

    assert (
        noise_module.sample_product_pauli_channel(
            7,
            2,
            cast("PauliDistribution", first_distribution),
            cast("PauliDistribution", second_distribution),
            _as_generator(rng),
        )
        == ()
    )
    assert calls == [
        (first_distribution, 7, _as_generator(rng)),
        (second_distribution, 2, _as_generator(rng)),
    ]


@pytest.mark.parametrize(
    ("first_site", "second_site", "first_distribution", "second_distribution", "error"),
    [
        (1, 1, {"I": 1.0}, {"I": 1.0}, ValueError),
        (-1, 2, {"I": 1.0}, {"I": 1.0}, ValueError),
        (1, True, {"I": 1.0}, {"I": 1.0}, TypeError),
        (1, 2, {"I": 1.0}, {"I": 0.5}, ValueError),
        (1, 2, {"I": 1.0}, {"A": 1.0}, ValueError),
    ],
)
def test_product_sampler_prevalidates_both_calls(
    monkeypatch: pytest.MonkeyPatch,
    first_site: object,
    second_site: object,
    first_distribution: Mapping[str, Any],
    second_distribution: Mapping[str, Any],
    error: type[Exception],
) -> None:
    """Invalid second-call input cannot partially consume or invoke the sampler."""
    calls = 0

    def fake_local_sampler(
        distribution: PauliDistribution,
        site: int,
        generator: np.random.Generator,
    ) -> LocalOperator | None:
        del distribution, site, generator
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(noise_module, "sample_local_pauli", fake_local_sampler)
    rng = _SequenceRNG((0.1, 0.2))

    with pytest.raises(error):
        noise_module.sample_product_pauli_channel(
            cast("Any", first_site),
            cast("Any", second_site),
            cast("PauliDistribution", first_distribution),
            cast("PauliDistribution", second_distribution),
            _as_generator(rng),
        )

    assert calls == 0
    assert rng.calls == 0


def test_bit_flip_product_probability_algebra() -> None:
    """Two independent bit flips produce the expected four nonzero branches."""
    probabilities = _joint_probabilities({"I": 0.8, "X": 0.2}, {"I": 0.8, "X": 0.2})

    assert probabilities["I", "I"] == pytest.approx(0.64)
    assert probabilities["I", "X"] == pytest.approx(0.16)
    assert probabilities["X", "I"] == pytest.approx(0.16)
    assert probabilities["X", "X"] == pytest.approx(0.04)
    assert all(
        probability == pytest.approx(0.0)
        for branch, probability in probabilities.items()
        if "Y" in branch or "Z" in branch
    )
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_ballarin_all_sixteen_probabilities_and_marginals() -> None:
    """Ballarin probabilities factorize into correct joint classes and marginals."""
    r = 0.07
    local = {"I": 1.0 - 3.0 * r, "X": r, "Y": r, "Z": r}
    probabilities = _joint_probabilities(local, local)

    assert len(probabilities) == 16
    assert probabilities["I", "I"] == pytest.approx((1.0 - 3.0 * r) ** 2)
    one_sided = [
        probability for (first, second), probability in probabilities.items() if (first == "I") != (second == "I")
    ]
    two_sided = [
        probability for (first, second), probability in probabilities.items() if first != "I" and second != "I"
    ]
    assert len(one_sided) == 6
    assert one_sided == pytest.approx([r * (1.0 - 3.0 * r)] * 6)
    assert len(two_sided) == 9
    assert two_sided == pytest.approx([r**2] * 9)
    assert sum(probabilities.values()) == pytest.approx(1.0)
    for label in _LABELS:
        assert sum(probability for (first, _), probability in probabilities.items() if first == label) == (
            pytest.approx(local[label])
        )
        assert sum(probability for (_, second), probability in probabilities.items() if second == label) == (
            pytest.approx(local[label])
        )


def test_asymmetric_product_probabilities_factorize() -> None:
    """Different local channels retain both marginals and exact factorization."""
    first = {"I": 0.4, "X": 0.1, "Y": 0.2, "Z": 0.3}
    second = {"I": 0.1, "X": 0.2, "Y": 0.3, "Z": 0.4}
    probabilities = _joint_probabilities(first, second)

    for first_label, second_label in product(_LABELS, repeat=2):
        assert probabilities[first_label, second_label] == pytest.approx(first[first_label] * second[second_label])
    for label in _LABELS:
        assert sum(probability for (first_label, _), probability in probabilities.items() if first_label == label) == (
            pytest.approx(first[label])
        )
        assert sum(
            probability for (_, second_label), probability in probabilities.items() if second_label == label
        ) == pytest.approx(second[label])


@pytest.mark.parametrize(
    ("r", "distribution"),
    [
        (0.0, {"I": 1.0, "X": 0.0, "Y": 0.0, "Z": 0.0}),
        (1.0 / 3.0, {"I": 0.0, "X": 1.0 / 3.0, "Y": 1.0 / 3.0, "Z": 1.0 / 3.0}),
    ],
)
def test_ballarin_endpoint_distributions_are_valid(r: float, distribution: Mapping[str, Any]) -> None:
    """Both the zero-rate and maximum valid local Ballarin channels sample."""
    rng = _SequenceRNG((0.0, 0.0))

    operators = _sample_product(0, 1, distribution, distribution, _as_generator(rng))

    assert rng.calls == 2
    expected_operator_count = 0 if math.isclose(r, 0.0, abs_tol=0.0) else 2
    assert len(operators) == expected_operator_count
    if r > 0.0:
        assert _outcome_labels(operators, 0, 1) == ("X", "X")


def test_fixed_seed_reproducibility_and_mapping_order_independence() -> None:
    """Equal seeds reproduce outcomes regardless of mapping insertion order."""
    canonical = {"I": 0.25, "X": 0.25, "Y": 0.25, "Z": 0.25}
    reordered = {"Z": 0.25, "Y": 0.25, "X": 0.25, "I": 0.25}

    def samples(seed: int, distribution: Mapping[str, Any]) -> list[tuple[str, str]]:
        rng = np.random.default_rng(seed)
        return [_outcome_labels(_sample_product(0, 1, distribution, distribution, rng), 0, 1) for _ in range(64)]

    first = samples(2026, canonical)
    second = samples(2026, reordered)
    different = samples(2027, canonical)

    assert first == second
    assert first != different


def test_seeded_joint_frequencies_match_the_product_law() -> None:
    """All branch frequencies remain inside a conservative seven-sigma band."""
    sample_count = 20_000
    r = 0.1
    distribution = {"I": 1.0 - 3.0 * r, "X": r, "Y": r, "Z": r}
    expected = _joint_probabilities(distribution, distribution)
    rng = np.random.default_rng(314159)
    counts: Counter[tuple[str, str]] = Counter()

    for _ in range(sample_count):
        operators = _sample_product(0, 1, distribution, distribution, rng)
        counts[_outcome_labels(operators, 0, 1)] += 1

    for branch, probability in expected.items():
        observed = counts[branch] / sample_count
        standard_error = math.sqrt(probability * (1.0 - probability) / sample_count)
        tolerance = 7.0 * standard_error + 1.0 / sample_count
        assert observed == pytest.approx(probability, abs=tolerance)


def test_two_local_x_operators_match_dense_x_tensor_x_without_svd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A product branch uses only one-site updates and preserves MPS bonds."""
    vector = np.array(
        [1.0 + 0.2j, -0.3 + 0.4j, 0.8 - 0.5j, -0.7 - 0.1j],
        dtype=np.complex128,
    )
    vector /= np.linalg.norm(vector)
    state = MPS.from_statevector(vector)
    shapes_before = tuple(tensor.shape for tensor in state.tensors)
    rng = _SequenceRNG((0.0, 0.0))
    operators = _sample_product(
        0,
        1,
        {"X": 1.0},
        {"X": 1.0},
        _as_generator(rng),
    )

    def reject_two_site_update(*args: object, **kwargs: object) -> None:
        del args, kwargs
        msg = "A product-Pauli branch must not invoke a two-site SVD."
        raise AssertionError(msg)

    monkeypatch.setattr(MPS, "_apply_adjacent_two_site_operator", reject_two_site_update)
    for operator in operators:
        state.apply_local_operator(operator.matrix, operator.sites)

    np.testing.assert_allclose(state.to_vec(), np.kron(_X, _X) @ vector, atol=1e-12)
    assert tuple(tensor.shape for tensor in state.tensors) == shapes_before


@pytest.mark.parametrize(
    ("first_distribution", "second_distribution", "expected"),
    [
        ({"X": 1.0}, {"I": 1.0}, np.array([0.0, 1.0, 0.0, 0.0])),
        ({"I": 1.0}, {"X": 1.0}, np.array([0.0, 0.0, 1.0, 0.0])),
    ],
    ids=("first-site-is-low-bit", "second-site-is-high-bit"),
)
def test_one_sided_product_outcomes_follow_dense_little_endian_order(
    first_distribution: Mapping[str, Any],
    second_distribution: Mapping[str, Any],
    expected: NDArray[np.float64],
) -> None:
    """Site zero acts on the right Kronecker factor of a dense state vector."""
    rng = _SequenceRNG((0.0, 0.0))
    operators = _sample_product(
        0,
        1,
        first_distribution,
        second_distribution,
        _as_generator(rng),
    )

    evolved = _apply_sampled_operators(
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
        operators,
    )

    np.testing.assert_allclose(evolved, expected, atol=1e-12)


def test_provider_sampling_is_independent_of_mps_norm_and_truncation() -> None:
    """WP3 maps depend on only context and RNG, not the evolving MPS representation."""
    distribution = {"I": 0.25, "X": 0.25, "Y": 0.25, "Z": 0.25}

    def provider(
        context: GateNoiseContext,
        rng: np.random.Generator,
    ) -> RandomUnitaryInstruction:
        operators = _sample_product(
            context.sites[0],
            context.sites[1],
            distribution,
            distribution,
            rng,
        )
        labels = _outcome_labels(operators, context.sites[0], context.sites[1])
        return RandomUnitaryInstruction(operators, "product-pauli", labels)

    typed_provider = cast("GateNoiseProvider", provider)
    circuit = ParameterizedCircuit(2, [ParameterizedGate("rzz", (0, 1), angle_offset=0.61)])
    vector = np.array([0.4, 0.3j, -0.2 + 0.1j, 0.7], dtype=np.complex128)
    normalized = MPS.from_statevector(vector)
    rescaled = copy.deepcopy(normalized)
    rescaled.tensors[0] *= 3.7

    exact = forward_tjm_trajectory(
        circuit,
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        normalized,
        KrotovTruncation(),
        None,
        KrotovTJMOptions(),
        np.random.default_rng(41),
        noise_provider=typed_provider,
    )
    truncated = forward_tjm_trajectory(
        circuit,
        np.array([], dtype=np.float64),
        np.array([], dtype=np.float64),
        rescaled,
        KrotovTruncation(max_bond_dim=1, svd_threshold=0.2),
        None,
        KrotovTJMOptions(),
        np.random.default_rng(41),
        noise_provider=typed_provider,
    )

    assert len(exact.noise_maps) == len(truncated.noise_maps) == 1
    assert _noise_map_signature(exact.noise_maps[0]) == _noise_map_signature(truncated.noise_maps[0])
    assert exact.noise_maps[0].outcome_labels == truncated.noise_maps[0].outcome_labels
    assert exact.noise_maps[0].channel_id == truncated.noise_maps[0].channel_id == "product-pauli"


def test_calibrated_tjm_oracle_rates_reproduce_branch_probabilities() -> None:
    """Combined and local one-step calibrations recover the exact product law."""
    r = 0.07
    dt = 0.8
    local = {"I": 1.0 - 3.0 * r, "X": r, "Y": r, "Z": r}
    expected = _joint_probabilities(local, local)
    rates = _combined_tjm_rates(r, dt)
    total_rate = sum(rates.values())
    no_jump = math.exp(-total_rate * dt)

    assert total_rate == pytest.approx(-math.log(expected["I", "I"]) / dt)
    assert no_jump == pytest.approx(expected["I", "I"])
    for branch, rate in rates.items():
        probability = (1.0 - no_jump) * rate / total_rate
        assert probability == pytest.approx(expected[branch])

    calibrated_local = _local_tjm_probabilities(r, dt)
    for label in _LABELS:
        assert calibrated_local[label] == pytest.approx(local[label])
    assert _joint_probabilities(calibrated_local, calibrated_local) == pytest.approx(expected)


def test_calibrated_tjm_oracle_guards_logarithmic_endpoints() -> None:
    """The reference is trivial at r=0 and refuses the singular r=1/3 case."""
    assert _combined_tjm_rates(0.0, 1.0) == {}
    assert _local_tjm_probabilities(0.0, 1.0) == {"I": 1.0, "X": 0.0, "Y": 0.0, "Z": 0.0}
    with pytest.raises(ValueError, match="P_II is zero"):
        _combined_tjm_rates(1.0 / 3.0, 1.0)
    with pytest.raises(ValueError, match="no-jump probability is zero"):
        _local_tjm_probabilities(1.0 / 3.0, 1.0)


def test_product_sampler_matches_exhaustive_kraus_and_tjm_references() -> None:
    """Four independent constructions yield the same truncation-free density map."""
    r = 0.07
    dt = 0.8
    distribution = {"I": 1.0 - 3.0 * r, "X": r, "Y": r, "Z": r}
    expected_probabilities = _joint_probabilities(distribution, distribution)
    vector = np.array(
        [1.0 + 0.2j, -0.4 + 0.7j, 0.3 - 0.8j, -0.2 - 0.1j],
        dtype=np.complex128,
    )
    vector /= np.linalg.norm(vector)

    production_density = np.zeros((4, 4), dtype=np.complex128)
    for branch, probability in expected_probabilities.items():
        rng = _SequenceRNG((
            _branch_draw(distribution, branch[0]),
            _branch_draw(distribution, branch[1]),
        ))
        operators = _sample_product(0, 1, distribution, distribution, _as_generator(rng))
        evolved = _apply_sampled_operators(vector, operators)
        production_density += probability * _density(evolved)

    exhaustive_density = _density_from_branch_probabilities(vector, expected_probabilities)

    initial_density = _density(vector)
    kraus_density = np.zeros((4, 4), dtype=np.complex128)
    for branch, probability in expected_probabilities.items():
        kraus = math.sqrt(probability) * np.kron(_PAULIS[branch[1]], _PAULIS[branch[0]])
        kraus_density += kraus @ initial_density @ kraus.conj().T

    combined_tjm_density = _actual_combined_tjm_density(vector, r, dt)
    local_tjm_density = _actual_sequential_local_tjm_density(vector, r, dt)

    np.testing.assert_allclose(production_density, exhaustive_density, atol=1e-12)
    np.testing.assert_allclose(kraus_density, exhaustive_density, atol=1e-12)
    np.testing.assert_allclose(combined_tjm_density, exhaustive_density, atol=1e-12)
    np.testing.assert_allclose(local_tjm_density, exhaustive_density, atol=1e-12)
