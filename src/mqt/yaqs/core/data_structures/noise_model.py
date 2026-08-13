# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Noise Models.

This module defines the NoiseModel class, which represents a noise model in a quantum system.
It stores a list of noise processes and their corresponding strengths, and automatically retrieves
the associated jump operator matrices from the NoiseLibrary. These jump operators are used to simulate
the effects of noise in quantum simulations.
"""

from __future__ import annotations

import copy
import logging
import math
import re
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.stats import truncnorm

from ..libraries.noise_library import NoiseLibrary
from .state_utils import resolve_physical_dimensions

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
    from mqt.yaqs.core.libraries.gate_library import BaseGate


PAULI_MAP = {
    "x": NoiseLibrary.pauli_x().matrix,
    "y": NoiseLibrary.pauli_y().matrix,
    "z": NoiseLibrary.pauli_z().matrix,
}

# Literal fixed library names (no runtime reflection over NoiseLibrary attributes).
_FIXED_OPERATOR_NAMES: frozenset[str] = frozenset({
    "raising",
    "lowering",
    "pauli_x",
    "pauli_y",
    "pauli_z",
    "raising_two",
    "lowering_two",
    "crosstalk_xx",
    "crosstalk_yy",
    "crosstalk_zz",
    "crosstalk_xy",
    "crosstalk_yx",
    "crosstalk_zy",
    "crosstalk_zx",
    "crosstalk_yz",
    "crosstalk_xz",
    "x",
    "y",
    "z",
})

_LIBRARY_CLASSES: dict[str, type] = {
    "raising": NoiseLibrary.raising,
    "lowering": NoiseLibrary.lowering,
    "pauli_x": NoiseLibrary.pauli_x,
    "pauli_y": NoiseLibrary.pauli_y,
    "pauli_z": NoiseLibrary.pauli_z,
    "raising_two": NoiseLibrary.raising_two,
    "lowering_two": NoiseLibrary.lowering_two,
    "crosstalk_xx": NoiseLibrary.crosstalk_xx,
    "crosstalk_yy": NoiseLibrary.crosstalk_yy,
    "crosstalk_zz": NoiseLibrary.crosstalk_zz,
    "crosstalk_xy": NoiseLibrary.crosstalk_xy,
    "crosstalk_yx": NoiseLibrary.crosstalk_yx,
    "crosstalk_zy": NoiseLibrary.crosstalk_zy,
    "crosstalk_zx": NoiseLibrary.crosstalk_zx,
    "crosstalk_yz": NoiseLibrary.crosstalk_yz,
    "crosstalk_xz": NoiseLibrary.crosstalk_xz,
}

_CROSSTALK_RE = re.compile(r"^crosstalk_[xyz]{2}$")
_LONGRANGE_CROSSTALK_RE = re.compile(r"^longrange_crosstalk_[xyz]{2}$")
_SUPPORTED_DISTRIBUTIONS = frozenset({"normal", "lognormal", "truncated_normal"})
_DISTRIBUTION_KEYS = frozenset({"distribution", "mean", "std"})

logger = logging.getLogger(__name__)


def _is_bool(value: object) -> bool:
    return isinstance(value, bool)


def _require_mapping(entry: object, kind: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        msg = f"Each {kind} must be a dictionary."
        raise TypeError(msg)
    return cast("dict[str, Any]", entry)


def _validate_name(name: object, kind: str) -> str:
    if not isinstance(name, str):
        msg = f"{kind} 'name' must be a string."
        raise TypeError(msg)
    if not name:
        msg = f"{kind} 'name' must be a nonempty string."
        raise ValueError(msg)
    return name


def _normalize_sites(sites: object, kind: str) -> list[int]:
    if not isinstance(sites, (list, tuple)):
        msg = f"{kind} 'sites' must be a list or tuple of integers."
        raise TypeError(msg)
    if len(sites) not in {1, 2}:
        msg = f"{kind} must have exactly 1 or 2 sites, got {len(sites)}."
        raise ValueError(msg)
    normalized: list[int] = []
    for site in sites:
        if _is_bool(site) or not isinstance(site, (int, np.integer)):
            msg = f"{kind} site indices must be integers (booleans are not allowed)."
            raise TypeError(msg)
        site_int = int(site)
        if site_int < 0:
            msg = f"{kind} site indices must be nonnegative, got {site_int}."
            raise ValueError(msg)
        normalized.append(site_int)
    if len(normalized) == 2 and normalized[0] == normalized[1]:
        msg = f"{kind} two-site indices must be distinct, got {normalized}."
        raise ValueError(msg)
    return normalized


def _validate_finite_nonnegative_real(value: object, label: str) -> float:
    if _is_bool(value) or not isinstance(value, (int, float, np.floating, np.integer)):
        msg = f"{label} must be a real number (booleans are not allowed)."
        raise TypeError(msg)
    number = float(value)
    if not math.isfinite(number):
        msg = f"{label} must be finite, got {number}."
        raise ValueError(msg)
    if number < 0:
        msg = (
            f"{label} must be nonnegative (got {number}). "
            "Standard TJM/MCWF jump probabilities require nonnegative rates; "
            "YAQS does not currently implement signed density-matrix generators "
            "or non-Markovian reverse-jump unravelings."
        )
        raise ValueError(msg)
    return number


def _validate_finite_real(value: object, label: str) -> float:
    if _is_bool(value) or not isinstance(value, (int, float, np.floating, np.integer)):
        msg = f"{label} must be a real number (booleans are not allowed)."
        raise TypeError(msg)
    number = float(value)
    if not math.isfinite(number):
        msg = f"{label} must be finite, got {number}."
        raise ValueError(msg)
    return number


def _validate_strength(strength: object) -> float | dict[str, Any]:
    if isinstance(strength, dict):
        strength_dict = cast("dict[str, Any]", strength)
        unknown = set(strength_dict) - _DISTRIBUTION_KEYS
        if unknown:
            msg = f"Unknown distribution keys: {sorted(unknown)}. Supported keys: {sorted(_DISTRIBUTION_KEYS)}."
            raise ValueError(msg)
        if "distribution" not in strength_dict:
            msg = "Noise strength dict must contain 'distribution' key."
            raise ValueError(msg)
        dist_type = strength_dict["distribution"]
        if dist_type not in _SUPPORTED_DISTRIBUTIONS:
            msg = f"Unsupported distribution type: {dist_type}. Supported: {sorted(_SUPPORTED_DISTRIBUTIONS)}."
            raise ValueError(msg)
        mean = _validate_finite_real(strength_dict.get("mean", 0.0), "distribution mean")
        std = _validate_finite_real(strength_dict.get("std", 0.0), "distribution std")
        if std < 0:
            msg = f"distribution std must be nonnegative, got {std}."
            raise ValueError(msg)
        return {"distribution": dist_type, "mean": mean, "std": std}
    return _validate_finite_nonnegative_real(strength, "process strength")


def _as_square_matrix(value: object, label: str) -> NDArray[np.complex128]:
    try:
        array = np.array(value, dtype=np.complex128, copy=True)
    except (TypeError, ValueError) as exc:
        msg = f"{label} must be a numeric array."
        raise TypeError(msg) from exc
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        msg = f"{label} must be a square 2-D array, got shape {array.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{label} entries must be finite."
        raise ValueError(msg)
    return array


def _crosstalk_suffix(name: str) -> str | None:
    if _CROSSTALK_RE.fullmatch(name):
        return name.rsplit("_", 1)[-1]
    if _LONGRANGE_CROSSTALK_RE.fullmatch(name):
        return name.rsplit("_", 1)[-1]
    return None


def _supported_operator_message() -> str:
    fixed = ", ".join(sorted(_FIXED_OPERATOR_NAMES))
    return (
        f"Supported fixed names: {fixed}. "
        "Also accepted: names matching crosstalk_[xyz]{2} or longrange_crosstalk_[xyz]{2}."
    )


def _crosstalk_pauli_letters(suffix: str, *, swapped: bool) -> tuple[str, str]:
    """Return Pauli letters for ascending sites, honoring caller site order when swapped."""
    a, b = suffix[0], suffix[1]
    return (b, a) if swapped else (a, b)


class NoiseModel:
    """A class to represent a noise model with arbitrary-site jump operators.

    Attributes:
        processes: A list of noise processes affecting the system. Each process is
            a ``dict`` with:

            - ``name``: process name or identifier.
            - ``sites``: indices of sites this process acts on.
            - ``strength``: noise strength (nonnegative Lindblad rate ``gamma``).
            - ``matrix``: matrix representing the operator on those sites (for
              1-site and adjacent 2-site processes).
            - ``factors``: tuple of two 1-site operator matrices (for long-range
              2-site processes).
        scheduled_jumps: A list of scheduled jump dictionaries applied at specific
            times (supported only for single-State analog MPS TJM).
    """

    def __init__(
        self, processes: list[dict[str, Any]] | None = None, scheduled_jumps: list[dict[str, Any]] | None = None
    ) -> None:
        """Initialize the NoiseModel.

        Args:
            processes: A list of noise process dictionaries affecting the quantum
                system. Each dict must contain ``name``, ``sites``, and
                ``strength``. Optional keys:

                - ``matrix``: local jump operator L as a ``d x d`` array for
                  1-site or adjacent 2-site processes. When provided, YAQS uses
                  this matrix and does not look up ``name`` in the library.
                  Custom full two-site matrices must use ascending site order.
                - ``factors``: pair of 1-site matrices for non-adjacent 2-site
                  processes (required unless ``name`` matches
                  ``crosstalk_[xyz]{2}`` or ``longrange_crosstalk_[xyz]{2}``).

                ``strength`` is the Lindblad rate ``gamma``; jump operators are
                formed as ``sqrt(gamma) * L`` during simulation. Fixed strengths
                must be finite and nonnegative.
            scheduled_jumps: A list of scheduled jumps to apply at specific times.
                Each dict must contain ``time``, ``sites``, and ``name``. An
                optional ``matrix`` overrides the library lookup. Two-site
                scheduled jumps must be nearest-neighbor.

        Raises:
            TypeError: If inputs have incorrect types (including booleans used
                as sites, strengths, or times).

        Note:
            Invalid values (for example negative strengths, bad site lists, or
            unknown operator names) raise ``ValueError``.
        """
        self.processes: list[dict[str, Any]] = []
        self.scheduled_jumps: list[dict[str, Any]] = []

        if scheduled_jumps is not None:
            if not isinstance(scheduled_jumps, (list, tuple)):
                msg = "scheduled_jumps must be a list or tuple of dictionaries."
                raise TypeError(msg)
            for jump in scheduled_jumps:
                self.scheduled_jumps.append(self._normalize_scheduled_jump(jump))

        if processes is None:
            return
        if not isinstance(processes, (list, tuple)):
            msg = "processes must be a list or tuple of dictionaries."
            raise TypeError(msg)

        self.processes = [self._normalize_process(original) for original in processes]

    @staticmethod
    def _normalize_scheduled_jump(jump: object) -> dict[str, Any]:
        original = _require_mapping(jump, "scheduled jump")
        for key in ("time", "sites", "name"):
            if key not in original:
                msg = f"Each scheduled jump must have a '{key}' key."
                raise ValueError(msg)

        jump_dict = dict(original)
        if "factors" in jump_dict:
            msg = "Scheduled jumps do not accept 'factors'; use 'matrix' for custom operators."
            raise ValueError(msg)
        jump_dict["name"] = _validate_name(jump_dict["name"], "Scheduled jump")
        jump_dict["time"] = _validate_finite_real(jump_dict["time"], "Scheduled jump time")
        sites = _normalize_sites(jump_dict["sites"], "Scheduled jump")
        user_matrix = "matrix" in jump_dict
        if len(sites) == 2:
            sorted_sites = sorted(sites)
            swapped = sorted_sites != list(sites)
            if abs(sorted_sites[1] - sorted_sites[0]) != 1:
                msg = (
                    f"Scheduled jump acts on non-adjacent sites {sites}. "
                    "Only nearest-neighbor scheduled jumps are supported."
                )
                raise ValueError(msg)
            if swapped and user_matrix:
                msg = f"Custom full scheduled-jump matrices require ascending site order; got sites {sites}."
                raise ValueError(msg)
            jump_dict["sites"] = sorted_sites
        else:
            swapped = False
            jump_dict["sites"] = sites

        if user_matrix:
            jump_dict["matrix"] = _as_square_matrix(jump_dict["matrix"], "Scheduled jump matrix")
        else:
            suffix = _crosstalk_suffix(jump_dict["name"])
            if suffix is not None:
                a, b = _crosstalk_pauli_letters(suffix, swapped=swapped)
                jump_dict["matrix"] = np.array(np.kron(PAULI_MAP[a], PAULI_MAP[b]), copy=True)
            else:
                jump_dict["matrix"] = NoiseModel.get_operator(jump_dict["name"])
        return jump_dict

    @staticmethod
    def _normalize_one_site_process(
        proc: dict[str, Any],
        sites: list[int],
        name: str,
        *,
        user_matrix: bool,
        factors_provided: bool,
    ) -> dict[str, Any]:
        proc["sites"] = sites
        if factors_provided:
            msg = "One-site processes do not accept 'factors'."
            raise ValueError(msg)
        if user_matrix:
            proc["matrix"] = _as_square_matrix(proc["matrix"], "Process matrix")
        else:
            proc["matrix"] = NoiseModel.get_operator(name)
        return proc

    @staticmethod
    def _normalize_non_adjacent_two_site_process(
        proc: dict[str, Any],
        name: str,
        *,
        user_matrix: bool,
        user_factors: object,
        swapped: bool,
    ) -> dict[str, Any]:
        """Normalize a long-range two-site process onto per-site factors.

        Args:
            proc: Process dict being normalized (``sites`` already sorted).
            name: Validated process name.
            user_matrix: Whether the caller supplied a full ``matrix``.
            user_factors: Caller-supplied ``factors``, or ``None`` if omitted.
            swapped: Whether the original site order was descending.

        Returns:
            The updated process dict with ``factors`` set and ``matrix`` removed.

        Raises:
            ValueError: If a full matrix is supplied, factors are missing for a
                non-crosstalk name, or supplied factors fail validation.
        """
        if user_matrix:
            msg = "Non-adjacent two-site processes require 'factors' (a full 'matrix' embedding is not accepted here)."
            raise ValueError(msg)

        suffix = _crosstalk_suffix(name)
        if user_factors is None:
            if suffix is None:
                msg = (
                    "Non-adjacent 2-site processes must specify 'factors' unless named "
                    "crosstalk_[xyz]{2} or longrange_crosstalk_[xyz]{2}."
                )
                raise ValueError(msg)
            a, b = _crosstalk_pauli_letters(suffix, swapped=swapped)
            proc["factors"] = (
                np.array(PAULI_MAP[a], copy=True),
                np.array(PAULI_MAP[b], copy=True),
            )
        else:
            proc["factors"] = _factors_for_sorted_sites(user_factors, swapped=swapped)

        proc.pop("matrix", None)
        return proc

    @staticmethod
    def _normalize_two_site_process(
        proc: dict[str, Any],
        sites: list[int],
        name: str,
        *,
        user_matrix: bool,
        factors_provided: bool,
        user_factors: object,
    ) -> dict[str, Any]:
        sorted_sites = sorted(sites)
        swapped = sorted_sites != list(sites)
        if swapped and user_matrix:
            msg = (
                "Custom full two-site matrices require ascending site order; "
                f"got sites {list(sites)}. Use ascending sites or supply 'factors'."
            )
            raise ValueError(msg)
        proc["sites"] = sorted_sites
        i, j = sorted_sites
        if abs(j - i) != 1:
            return NoiseModel._normalize_non_adjacent_two_site_process(
                proc,
                name,
                user_matrix=user_matrix,
                user_factors=user_factors,
                swapped=swapped,
            )

        if factors_provided:
            msg = "Adjacent two-site processes use 'matrix', not 'factors'."
            raise ValueError(msg)
        suffix = _crosstalk_suffix(name)
        if user_matrix:
            proc["matrix"] = _as_square_matrix(proc["matrix"], "Process matrix")
        elif suffix is not None:
            a, b = _crosstalk_pauli_letters(suffix, swapped=swapped)
            proc["matrix"] = np.array(np.kron(PAULI_MAP[a], PAULI_MAP[b]), copy=True)
        else:
            proc["matrix"] = NoiseModel.get_operator(name)
        proc.pop("factors", None)
        return proc

    @staticmethod
    def _normalize_process(original: object) -> dict[str, Any]:
        source = _require_mapping(original, "noise process")
        for key in ("name", "sites", "strength"):
            if key not in source:
                msg = f"Each process must have a '{key}' key."
                raise ValueError(msg)

        proc = dict(source)
        name = _validate_name(proc["name"], "Process")
        proc["name"] = name
        proc["strength"] = _validate_strength(proc["strength"])

        sites = _normalize_sites(proc["sites"], "Process")
        user_matrix = "matrix" in source
        factors_provided = "factors" in source
        user_factors = source.get("factors")
        if factors_provided and user_factors is None:
            msg = "Process 'factors' must be a sequence of exactly two square matrices, not None."
            raise ValueError(msg)
        if user_matrix and factors_provided:
            msg = "Process cannot specify both 'matrix' and 'factors'."
            raise ValueError(msg)

        if len(sites) == 2:
            return NoiseModel._normalize_two_site_process(
                proc,
                sites,
                name,
                user_matrix=user_matrix,
                factors_provided=factors_provided,
                user_factors=user_factors,
            )
        return NoiseModel._normalize_one_site_process(
            proc,
            sites,
            name,
            user_matrix=user_matrix,
            factors_provided=factors_provided,
        )

    def sample(self, rng: np.random.Generator | int | None = None) -> NoiseModel:
        """Sample a concrete NoiseModel from any distribution-based strengths.

        For each process:

        - If ``strength`` is a float, it is kept as is.
        - If ``strength`` is a dict describing a distribution, a value is sampled.

        Negative draws from ``normal`` are clamped to ``0.0`` with a warning.
        Sampled strengths are re-validated before the resolved model is returned.

        Args:
            rng: The random number generator or seed to use for sampling.
                If None, a new generator is created.

        Returns:
            A new NoiseModel instance where all process strengths are concrete
            floats. This sampled model represents the specific realization of
            static disorder used for a simulation run.

        Raises:
            ValueError: If an unsupported distribution type is provided or a
                sampled strength is invalid.
        """
        generator = np.random.default_rng(rng)
        new_processes: list[dict[str, Any]] = []
        for proc in self.processes:
            new_proc = copy.deepcopy(proc)
            strength_val = proc["strength"]

            if isinstance(strength_val, dict):
                dist_type = strength_val["distribution"]
                mean = strength_val["mean"]
                std = strength_val["std"]

                if dist_type == "normal":
                    sampled_val = float(generator.normal(loc=mean, scale=std))
                    if sampled_val < 0:
                        logger.warning(
                            "Sampled noise strength %f using 'normal' distribution (mean=%f, std=%f) "
                            "was negative and clamped to 0.0.",
                            sampled_val,
                            mean,
                            std,
                        )
                    sampled_val = float(max(0.0, sampled_val))
                elif dist_type == "lognormal":
                    sampled_val = float(generator.lognormal(mean=mean, sigma=std))
                elif dist_type == "truncated_normal":
                    if math.isclose(std, 0.0, abs_tol=1e-8):
                        sampled_val = float(max(0.0, mean))
                    else:
                        a_norm = (0.0 - mean) / std
                        b_norm = (np.inf - mean) / std
                        sampled_val = float(truncnorm.rvs(a_norm, b_norm, loc=mean, scale=std, random_state=generator))
                else:
                    msg = f"Unsupported distribution type: {dist_type}"
                    raise ValueError(msg)
                new_proc["strength"] = _validate_finite_nonnegative_real(sampled_val, "sampled process strength")
            else:
                new_proc["strength"] = _validate_finite_nonnegative_real(strength_val, "process strength")

            new_processes.append(new_proc)

        new_model = object.__new__(NoiseModel)
        new_model.processes = new_processes
        new_model.scheduled_jumps = copy.deepcopy(self.scheduled_jumps)
        return new_model

    @staticmethod
    def get_operator(name: str) -> NDArray[np.complex128]:
        """Retrieve the operator matrix for a supported library name.

        Args:
            name: Name of the noise process (fixed library name, short Pauli
                ``x``/``y``/``z``, or an exact ``crosstalk_[xyz]{2}`` label).

        Returns:
            The matrix representation of the operator.

        Raises:
            ValueError: If ``name`` is not a supported operator name.
        """
        if name in PAULI_MAP:
            return np.array(PAULI_MAP[name], copy=True)
        suffix = _crosstalk_suffix(name)
        if suffix is not None:
            return np.array(np.kron(PAULI_MAP[suffix[0]], PAULI_MAP[suffix[1]]), copy=True)
        operator_class = _LIBRARY_CLASSES.get(name)
        if operator_class is None:
            msg = f"Unknown noise operator '{name}'. {_supported_operator_message()}"
            raise ValueError(msg)
        operator: BaseGate = operator_class()
        return np.array(operator.matrix, dtype=np.complex128, copy=True)


def _validate_factors(factors: object) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    if not isinstance(factors, (list, tuple)) or len(factors) != 2:
        msg = "Process 'factors' must be a sequence of exactly two square matrices."
        raise ValueError(msg)
    left = _as_square_matrix(factors[0], "Process factor[0]")
    right = _as_square_matrix(factors[1], "Process factor[1]")
    return left, right


def _factors_for_sorted_sites(
    factors: object,
    *,
    swapped: bool,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Validate factors and reorder them to match ascending site indices.

    Args:
        factors: Caller-supplied pair of square factor matrices.
        swapped: Whether the original site order was descending.

    Returns:
        Factor matrices ordered for the sorted ``sites`` list.
    """
    left, right = _validate_factors(factors)
    if swapped:
        return right, left
    return left, right


_PAULI_1 = (PAULI_MAP["x"], PAULI_MAP["y"], PAULI_MAP["z"])
_PAULI_2 = tuple(np.kron(a, b) for a in _PAULI_1 for b in _PAULI_1)


def _matches_up_to_unit_phase(mat: NDArray[np.complex128], reference: NDArray[np.complex128]) -> bool:
    """Return True if ``mat`` equals ``exp(i φ) * reference`` for real φ (not a scale)."""
    if mat.shape != reference.shape:
        return False
    idx = np.unravel_index(int(np.argmax(np.abs(reference))), reference.shape)
    ref_val = reference[idx]
    mat_val = mat[idx]
    if abs(ref_val) < 1e-14 or abs(mat_val) < 1e-14:
        return bool(np.allclose(mat, reference, atol=1e-10, rtol=0.0))
    phase = mat_val / ref_val
    if not np.isclose(abs(phase), 1.0, atol=1e-10, rtol=0.0):
        return False
    return bool(np.allclose(mat, phase * reference, atol=1e-10, rtol=0.0))


def _is_unit_phase_pauli(mat: NDArray[np.complex128]) -> bool:
    return any(_matches_up_to_unit_phase(mat, p) for p in _PAULI_1)


def _is_unit_phase_pauli_kron(mat: NDArray[np.complex128]) -> bool:
    return any(_matches_up_to_unit_phase(mat, p) for p in _PAULI_2)


def is_pauli(proc: dict[str, Any]) -> bool:
    """Return True if the process operators match Pauli structure (unit-modulus phase only).

    Recognizes one-site X/Y/Z, adjacent Kronecker products of Paulis, and long-range
    factor pairs that are each Pauli. Scaled operators such as ``2 * X`` are not Pauli
    for TJM's scalar dissipator shortcut, which assumes ``L^† L = I``.
    """
    sites = proc["sites"]
    if len(sites) == 1:
        if "matrix" not in proc:
            return False
        return _is_unit_phase_pauli(np.asarray(proc["matrix"], dtype=np.complex128))
    if len(sites) != 2:
        return False
    if abs(sites[1] - sites[0]) == 1 and "matrix" in proc:
        return _is_unit_phase_pauli_kron(np.asarray(proc["matrix"], dtype=np.complex128))
    if abs(sites[1] - sites[0]) > 1 and "factors" in proc:
        f0, f1 = proc["factors"]
        return _is_unit_phase_pauli(np.asarray(f0, dtype=np.complex128)) and _is_unit_phase_pauli(
            np.asarray(f1, dtype=np.complex128)
        )
    return False


def validate_noise_model_for_run(
    noise_model: NoiseModel,
    *,
    length: int,
    physical_dimensions: list[int] | int | None = None,
    representation: str | None = None,
    is_digital: bool = False,
    is_ensemble: bool = False,
    sim_params: AnalogSimParams | None = None,
) -> None:
    """Validate a sampled noise model against run context.

    Args:
        noise_model: Noise model after :meth:`NoiseModel.sample`.
        length: Number of sites / qubits.
        physical_dimensions: Per-site Hilbert-space dimensions.
        representation: Active state representation for analog runs.
        is_digital: Whether this is a circuit / digital TJM run.
        is_ensemble: Whether this is the ``list[State]`` unitary ensemble path.
        sim_params: Analog simulation parameters (required for scheduled-jump
            grid checks on the supported MPS path).

    Raises:
        ValueError: If the model is incompatible with the run context.
    """
    dims = resolve_physical_dimensions(length, physical_dimensions)

    def _check_sites(sites: list[int], kind: str) -> None:
        for site in sites:
            if site >= length:
                msg = f"{kind} site index {site} is out of range for length {length}."
                raise ValueError(msg)

    def _check_operator_dims(entry: dict[str, Any], kind: str) -> None:
        sites = entry["sites"]
        _check_sites(sites, kind)
        if "matrix" in entry:
            mat = np.asarray(entry["matrix"])
            expected = dims[sites[0]] if len(sites) == 1 else dims[sites[0]] * dims[sites[1]]
            if mat.shape != (expected, expected):
                msg = (
                    f"{kind} matrix shape {mat.shape} does not match expected "
                    f"({expected}, {expected}) for sites {sites}."
                )
                raise ValueError(msg)
        if "factors" in entry:
            factors = entry["factors"]
            for site, factor in zip(sites, factors, strict=True):
                expected = dims[site]
                arr = np.asarray(factor)
                if arr.shape != (expected, expected):
                    msg = f"{kind} factor on site {site} has shape {arr.shape}, expected ({expected}, {expected})."
                    raise ValueError(msg)

    for proc in noise_model.processes:
        _check_operator_dims(proc, "Process")
        if is_digital and len(proc["sites"]) == 2 and abs(proc["sites"][1] - proc["sites"][0]) != 1:
            msg = (
                "Digital TJM does not support non-adjacent / factorized two-site noise "
                f"(process '{proc['name']}' on sites {proc['sites']}). "
                "Gate-local digital noise scoping remains nearest-neighbor only."
            )
            raise ValueError(msg)
        if (
            representation == "mps"
            and not is_digital
            and not is_ensemble
            and len(proc["sites"]) == 2
            and abs(proc["sites"][1] - proc["sites"][0]) > 1
            and not is_pauli(proc)
        ):
            msg = (
                "Analog MPS TJM does not support non-Pauli long-range noise "
                f"(process '{proc['name']}' on sites {proc['sites']})."
            )
            raise ValueError(msg)

    if not noise_model.scheduled_jumps:
        return

    supported = representation == "mps" and not is_digital and not is_ensemble
    if not supported:
        msg = (
            "scheduled_jumps are only supported for single-State analog MPS TJM; "
            "they are not supported for MCWF, Lindblad, digital, or list[State] ensemble runs."
        )
        raise ValueError(msg)
    if sim_params is None:
        msg = "AnalogSimParams are required to validate scheduled_jumps against the time grid."
        raise ValueError(msg)
    if sim_params.order != 1:
        msg = (
            "scheduled_jumps are only supported for AnalogSimParams(order=1); "
            f"got order={sim_params.order}. Order-2 TJM applies deterministic jumps "
            "inconsistently on the sampling versus trajectory MPS."
        )
        raise ValueError(msg)

    times = np.asarray(sim_params.times, dtype=float)
    durations = np.diff(times)
    atol = (float(np.min(durations)) if durations.size else sim_params.dt) * 1e-3
    for jump in noise_model.scheduled_jumps:
        _check_operator_dims(jump, "Scheduled jump")
        jump_time = float(jump["time"])
        if not np.any(np.isclose(times, jump_time, atol=atol, rtol=0.0)):
            msg = f"Scheduled jump time {jump_time} is not on the simulation time grid (atol={atol}, rtol=0)."
            raise ValueError(msg)
