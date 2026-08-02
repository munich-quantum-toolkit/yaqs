# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Versioned, blinded target populations for Phase II state preparation."""

from __future__ import annotations

import hashlib
import hmac
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np

from benchmarks.state_preparation.constants import TARGET_IDS as PHASE_I_TARGET_IDS

from .canonical import (
    canonical_checksum,
    canonical_json,
    freeze_json_mapping,
    load_canonical_json_object,
    thaw_json_mapping,
    verify_sealed_mapping,
)
from .protocol import (
    PRIMARY_FAMILY_STRATA,
    PRIMARY_TARGET_FAMILIES,
    TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM,
    ConfirmationAuthorization,
    InitialPreregistration,
    ScreeningManifest,
)
from .validation import (
    require_checksum,
    require_exact_keys,
    require_float,
    require_int,
    require_mapping,
    require_slug,
    require_string,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

TARGET_GENERATOR_SCHEMA_VERSION = "yaqs.state_preparation.phase2.targets.v2"
TARGET_POPULATION_CONFIG_SCHEMA_VERSION = "yaqs.state_preparation.phase2.target_population_config.v1"
TARGET_INSTANCE_SPEC_SCHEMA_VERSION = "yaqs.state_preparation.phase2.target_instance_spec.v1"
TARGET_POPULATION_MANIFEST_SCHEMA_VERSION = "yaqs.state_preparation.phase2.target_population_manifest.v1"
TARGET_POPULATION_COMMITMENT_SCHEMA_VERSION = "yaqs.state_preparation.phase2.target_population_commitment.v1"
TARGET_MATERIALIZATION_SCHEMA_VERSION = "yaqs.state_preparation.phase2.target_materialization.v1"

PHASE_II_TARGET_NAMESPACE = "phase2"
PHASE_II_TARGET_ID_PREFIX = "phase2_target_"
PHASE_II_POPULATION_ID_PREFIX = "phase2_target_population_"
TARGET_INSTANCE_ROLES = ("development", "screening_selection", "confirmatory")
TARGET_POPULATION_SCOPES = ("primary_q6", "secondary_q12")

TRUSTED_TARGET_POPULATION_POLICY_CHECKSUM = "sha256:67720a4ab54ed515e8affe543ed35b199cea98fe977fe3da60541640477c5d7e"
TRUSTED_TARGET_RNG_POLICY_CHECKSUM = "sha256:5442e5f03276e2a6397c3c4aa941ed3cdc4f37a0a4f93e40463488327b9af911"
TRUSTED_TARGET_NUMERIC_POLICY_CHECKSUM = "sha256:34ef8cc1de502c1cb9a0699a1bad5cba1ec28e243124b1ea2a336a00c21c670a"
TRUSTED_TARGET_ALLOCATION_POLICY_CHECKSUM = "sha256:be051dd1ce71bbec273bc262f589166e6021d45d73f5fd643b12132679045ff0"

_HEX_128_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_HEX_256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TARGET_ID_PATTERN = re.compile(rf"^{PHASE_II_TARGET_ID_PREFIX}[0-9a-f]{{64}}$")
_POPULATION_ID_PATTERN = re.compile(rf"^{PHASE_II_POPULATION_ID_PREFIX}[0-9a-f]{{64}}$")
_MATERIALIZATION_SENTINEL = object()

_CONFIG_KEYS = frozenset({
    "schema_version",
    "generator_schema_version",
    "namespace",
    "population_id",
    "preregistration_checksum",
    "target_population_policy_checksum",
    "rng_policy_checksum",
    "numeric_policy_checksum",
    "allocation_policy_checksum",
    "role_master_entropy_commitment",
    "data_role",
    "population_scope",
    "allocations",
    "content_checksum",
})
_ALLOCATION_KEYS = frozenset({"family_id", "stratum_id", "qubit_count", "instance_count"})
_SPEC_KEYS = frozenset({
    "schema_version",
    "namespace",
    "data_role",
    "population_config_checksum",
    "family_id",
    "stratum_id",
    "qubit_count",
    "instance_seed",
    "target_instance_id",
    "parameters",
    "content_checksum",
})
_MANIFEST_KEYS = frozenset({
    "schema_version",
    "generator_schema_version",
    "namespace",
    "manifest_id",
    "preregistration_checksum",
    "population_config_checksum",
    "data_role",
    "population_scope",
    "role_master_entropy_commitment",
    "allocations",
    "instances",
    "content_checksum",
})
_COMMITMENT_KEYS = frozenset({
    "schema_version",
    "target_manifest_checksum",
    "target_count_by_family",
    "content_checksum",
})

_GAUSSIAN_PARAMETER_KEYS = frozenset({"mean", "width"})
_TFIM_PARAMETER_KEYS = frozenset({
    "attempt_index",
    "couplings",
    "fields",
    "ground_energy",
    "ground_state_gap",
    "gap_threshold",
    "spectral_norm",
})
_HAAR_PARAMETER_KEYS = frozenset({"dimension"})
_MPS_PARAMETER_KEYS = frozenset({"bond_dimension", "bond_dimensions", "tensor_shapes"})

_FAMILY_ORDER = {family_id: index for index, family_id in enumerate(PRIMARY_TARGET_FAMILIES)}
_STRATUM_ORDER = {
    (family_id, stratum_id): index
    for family_id, strata in PRIMARY_FAMILY_STRATA.items()
    for index, stratum_id in enumerate(strata)
}
_REGIME_RATIOS = {"ferromagnetic": 0.5, "critical": 1.0, "paramagnetic": 1.5}
_MPS_BOND_DIMENSIONS = {"bond2": 2, "bond3": 3}


def _require_trusted_preregistration(preregistration: InitialPreregistration) -> None:
    """Require the checked-in WP15 protocol and target policy.

    Args:
        preregistration: Protocol object to validate.

    Raises:
        TypeError: If the object is not an initial preregistration.
        ValueError: If either trusted checksum differs.
    """
    if not isinstance(preregistration, InitialPreregistration):
        msg = "preregistration must be an InitialPreregistration."
        raise TypeError(msg)
    if preregistration.content_checksum != TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM:
        msg = "Target populations require the trusted checked-in Phase II preregistration."
        raise ValueError(msg)
    if preregistration.target_population_configuration_checksum != TRUSTED_TARGET_POPULATION_POLICY_CHECKSUM:
        msg = "The preregistered target-population policy differs from the trusted WP16 policy."
        raise ValueError(msg)
    target_policy = preregistration.target_population_policy
    if target_policy["generator_schema_version"] != TARGET_GENERATOR_SCHEMA_VERSION:
        msg = f"The target policy must use generator schema {TARGET_GENERATOR_SCHEMA_VERSION!r}."
        raise ValueError(msg)
    expected_subpolicy_checksums = (
        ("rng_policy", TRUSTED_TARGET_RNG_POLICY_CHECKSUM),
        ("numeric_policy", TRUSTED_TARGET_NUMERIC_POLICY_CHECKSUM),
        ("role_allocation_policy", TRUSTED_TARGET_ALLOCATION_POLICY_CHECKSUM),
    )
    for field_name, expected_checksum in expected_subpolicy_checksums:
        if canonical_checksum(target_policy[field_name]) != expected_checksum:
            msg = f"The preregistered {field_name} differs from the trusted WP16 policy."
            raise ValueError(msg)


def _require_runtime(preregistration: InitialPreregistration) -> None:
    """Require the NumPy version frozen for target generation.

    Args:
        preregistration: Trusted protocol carrying the numeric policy.

    Raises:
        RuntimeError: If NumPy differs from the preregistered generation version.
    """
    numeric_policy = cast("Mapping[str, object]", preregistration.target_population_policy["numeric_policy"])
    expected = cast("str", numeric_policy["generation_numpy_version"])
    if np.__version__ != expected:
        msg = f"Target generation requires NumPy {expected}, found {np.__version__}."
        raise RuntimeError(msg)


def _require_hex(value: object, name: str, pattern: re.Pattern[str]) -> str:
    """Require one fixed-width lowercase hexadecimal string.

    Args:
        value: Candidate value.
        name: Human-readable field name.
        pattern: Exact accepted spelling.

    Returns:
        The validated hexadecimal string.

    Raises:
        ValueError: If the spelling or width differs.
    """
    text = require_string(value, name)
    if pattern.fullmatch(text) is None:
        msg = f"{name} must be lowercase hexadecimal with the required fixed width."
        raise ValueError(msg)
    return text


def _require_family_stratum(family_id: object, stratum_id: object, name: str) -> tuple[str, str]:
    """Validate one preregistered family/stratum pair.

    Args:
        family_id: Target-family identifier.
        stratum_id: Within-family stratum.
        name: Human-readable record name.

    Returns:
        The validated pair.

    Raises:
        ValueError: If the pair is outside the sealed family universe.
    """
    family = require_slug(family_id, f"{name}.family_id")
    stratum = require_slug(stratum_id, f"{name}.stratum_id")
    if family not in PRIMARY_FAMILY_STRATA or stratum not in PRIMARY_FAMILY_STRATA[family]:
        msg = f"{name} uses unsupported family/stratum pair {(family, stratum)!r}."
        raise ValueError(msg)
    return family, stratum


def _allocation_sort_key(allocation: TargetAllocation) -> tuple[int, int, int]:
    """Return the frozen allocation ordering key."""
    return (
        _FAMILY_ORDER[allocation.family_id],
        _STRATUM_ORDER[allocation.family_id, allocation.stratum_id],
        allocation.qubit_count,
    )


def _spec_sort_key(spec: TargetInstanceSpec) -> tuple[int, int, int, str]:
    """Return the frozen manifest ordering key."""
    return (
        _FAMILY_ORDER[spec.family_id],
        _STRATUM_ORDER[spec.family_id, spec.stratum_id],
        spec.qubit_count,
        spec.target_instance_id,
    )


@dataclass(frozen=True, slots=True)
class TargetAllocation:
    """One family/stratum/qubit allocation inside a population config."""

    family_id: str
    stratum_id: str
    qubit_count: int
    instance_count: int

    def __post_init__(self) -> None:
        """Validate and normalize the allocation."""
        family, stratum = _require_family_stratum(self.family_id, self.stratum_id, "target allocation")
        object.__setattr__(self, "family_id", family)
        object.__setattr__(self, "stratum_id", stratum)
        object.__setattr__(self, "qubit_count", require_int(self.qubit_count, "qubit_count", minimum=2))
        object.__setattr__(self, "instance_count", require_int(self.instance_count, "instance_count", minimum=1))

    def to_dict(self) -> dict[str, object]:
        """Return the strict JSON-native allocation."""
        return {
            "family_id": self.family_id,
            "stratum_id": self.stratum_id,
            "qubit_count": self.qubit_count,
            "instance_count": self.instance_count,
        }

    @classmethod
    def from_dict(cls, data: object) -> TargetAllocation:
        """Construct an allocation from a strict mapping.

        Args:
            data: Candidate allocation mapping.

        Returns:
            The validated allocation.
        """
        mapping = require_mapping(data, "target allocation")
        require_exact_keys(mapping, _ALLOCATION_KEYS, "target allocation")
        return cls(
            family_id=cast("str", mapping["family_id"]),
            stratum_id=cast("str", mapping["stratum_id"]),
            qubit_count=cast("int", mapping["qubit_count"]),
            instance_count=cast("int", mapping["instance_count"]),
        )


def _validate_allocation_policy(
    data_role: str,
    population_scope: str,
    allocations: tuple[TargetAllocation, ...],
) -> None:
    """Validate exact role-specific family, stratum, and qubit allocations.

    Args:
        data_role: Target population role.
        population_scope: Primary-q6 or secondary-q12 scope.
        allocations: Ordered target allocations.

    Raises:
        ValueError: If the allocations depart from the preregistered policy.
    """
    if not allocations:
        msg = "allocations must not be empty."
        raise ValueError(msg)
    keys = [(item.family_id, item.stratum_id, item.qubit_count) for item in allocations]
    if len(keys) != len(set(keys)):
        msg = "allocations must not repeat a family/stratum/qubit cell."
        raise ValueError(msg)
    if tuple(sorted(allocations, key=_allocation_sort_key)) != allocations:
        msg = "allocations must use the frozen family, stratum, and qubit ordering."
        raise ValueError(msg)

    counts: dict[tuple[str, int], list[int]] = {}
    for family_id, strata in PRIMARY_FAMILY_STRATA.items():
        for qubit_count in {allocation.qubit_count for allocation in allocations}:
            matching = [
                allocation.instance_count
                for allocation in allocations
                if allocation.family_id == family_id and allocation.qubit_count == qubit_count
            ]
            if matching:
                if len(matching) != len(strata) or len(set(matching)) != 1:
                    msg = f"Family {family_id!r} must be equally allocated across every stratum at q={qubit_count}."
                    raise ValueError(msg)
                counts[family_id, qubit_count] = matching

    totals = {key: sum(values) for key, values in counts.items()}
    if data_role == "development":
        if population_scope != "primary_q6":
            msg = "Development target populations must use the primary_q6 scope."
            raise ValueError(msg)
        expected = {(family_id, 6): 12 for family_id in PRIMARY_TARGET_FAMILIES}
        if totals != expected:
            msg = "Development populations require exactly 12 q=6 targets per family."
            raise ValueError(msg)
    elif data_role == "screening_selection":
        if population_scope == "primary_q6":
            expected = {(family_id, 6): 12 for family_id in PRIMARY_TARGET_FAMILIES}
        else:
            expected = {(family_id, 12): 6 for family_id in PRIMARY_TARGET_FAMILIES}
        if totals != expected:
            msg = "Screening population allocation does not match its isolated primary-q6 or secondary-q12 scope."
            raise ValueError(msg)
    else:
        if population_scope != "primary_q6":
            msg = "Confirmatory target populations must use the primary_q6 scope."
            raise ValueError(msg)
        if set(totals) != {(family_id, 6) for family_id in PRIMARY_TARGET_FAMILIES}:
            msg = "Confirmatory populations may contain only q=6 targets from every primary family."
            raise ValueError(msg)
        family_totals = tuple(totals[family_id, 6] for family_id in PRIMARY_TARGET_FAMILIES)
        if len(set(family_totals)) != 1:
            msg = "Confirmatory target counts must be equal across families."
            raise ValueError(msg)
        count = family_totals[0]
        if not 24 <= count <= 96 or count % 6:
            msg = "Confirmatory target counts must be 24 through 96 per family in increments of six."
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TargetPopulationConfig:
    """Strict scientific identity of one independent Phase II population."""

    preregistration_checksum: str
    target_population_policy_checksum: str
    rng_policy_checksum: str
    numeric_policy_checksum: str
    allocation_policy_checksum: str
    role_master_entropy_commitment: str
    data_role: str
    population_scope: str
    allocations: tuple[TargetAllocation, ...]
    schema_version: str = field(default=TARGET_POPULATION_CONFIG_SCHEMA_VERSION, init=False)
    generator_schema_version: str = field(default=TARGET_GENERATOR_SCHEMA_VERSION, init=False)
    namespace: str = field(default=PHASE_II_TARGET_NAMESPACE, init=False)

    def __post_init__(self) -> None:
        """Validate the trusted policy link and role allocation.

        Raises:
            TypeError: If allocations are not immutable typed records.
            ValueError: If a checksum, role, scope, or allocation differs.
        """
        object.__setattr__(
            self,
            "preregistration_checksum",
            require_checksum(self.preregistration_checksum, "preregistration_checksum"),
        )
        object.__setattr__(
            self,
            "target_population_policy_checksum",
            require_checksum(self.target_population_policy_checksum, "target_population_policy_checksum"),
        )
        if self.preregistration_checksum != TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM:
            msg = "TargetPopulationConfig must reference the trusted Phase II preregistration."
            raise ValueError(msg)
        if self.target_population_policy_checksum != TRUSTED_TARGET_POPULATION_POLICY_CHECKSUM:
            msg = "TargetPopulationConfig must reference the trusted target-population policy."
            raise ValueError(msg)
        expected_subpolicy_checksums = {
            "rng_policy_checksum": TRUSTED_TARGET_RNG_POLICY_CHECKSUM,
            "numeric_policy_checksum": TRUSTED_TARGET_NUMERIC_POLICY_CHECKSUM,
            "allocation_policy_checksum": TRUSTED_TARGET_ALLOCATION_POLICY_CHECKSUM,
        }
        for field_name, expected_checksum in expected_subpolicy_checksums.items():
            checksum = require_checksum(getattr(self, field_name), field_name)
            if checksum != expected_checksum:
                msg = f"TargetPopulationConfig.{field_name} differs from the trusted policy."
                raise ValueError(msg)
            object.__setattr__(self, field_name, checksum)
        object.__setattr__(
            self,
            "role_master_entropy_commitment",
            require_checksum(self.role_master_entropy_commitment, "role_master_entropy_commitment"),
        )
        role = require_slug(self.data_role, "data_role")
        if role not in TARGET_INSTANCE_ROLES:
            msg = f"data_role must be one of {TARGET_INSTANCE_ROLES!r}."
            raise ValueError(msg)
        object.__setattr__(self, "data_role", role)
        scope = require_slug(self.population_scope, "population_scope")
        if scope not in TARGET_POPULATION_SCOPES:
            msg = f"population_scope must be one of {TARGET_POPULATION_SCOPES!r}."
            raise ValueError(msg)
        object.__setattr__(self, "population_scope", scope)
        if not isinstance(self.allocations, tuple) or any(
            not isinstance(allocation, TargetAllocation) for allocation in self.allocations
        ):
            msg = "allocations must be a tuple of TargetAllocation records."
            raise TypeError(msg)
        _validate_allocation_policy(role, scope, self.allocations)

    @property
    def population_id(self) -> str:
        """Stable full-digest population identifier."""
        return f"{PHASE_II_POPULATION_ID_PREFIX}{canonical_checksum(self._identity_dict()).removeprefix('sha256:')}"

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete resolved population configuration."""
        return canonical_checksum(self._content_dict())

    def _identity_dict(self) -> dict[str, object]:
        """Return fields from which the population identifier is derived."""
        return {
            "schema_version": self.schema_version,
            "generator_schema_version": self.generator_schema_version,
            "namespace": self.namespace,
            "preregistration_checksum": self.preregistration_checksum,
            "target_population_policy_checksum": self.target_population_policy_checksum,
            "rng_policy_checksum": self.rng_policy_checksum,
            "numeric_policy_checksum": self.numeric_policy_checksum,
            "allocation_policy_checksum": self.allocation_policy_checksum,
            "role_master_entropy_commitment": self.role_master_entropy_commitment,
            "data_role": self.data_role,
            "population_scope": self.population_scope,
            "allocations": [allocation.to_dict() for allocation in self.allocations],
        }

    def _content_dict(self) -> dict[str, object]:
        """Return the checksum-covered configuration payload."""
        return {**self._identity_dict(), "population_id": self.population_id}

    def to_dict(self) -> dict[str, object]:
        """Return canonicalizable sealed configuration data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> TargetPopulationConfig:
        """Construct and checksum-verify a target population config.

        Args:
            data: Sealed configuration mapping.

        Returns:
            The validated immutable configuration.

        Raises:
            ValueError: If the schema, identifier, or checksum changes.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_CONFIG_KEYS, name="target population config")
        if mapping["schema_version"] != TARGET_POPULATION_CONFIG_SCHEMA_VERSION:
            msg = f"schema_version must be {TARGET_POPULATION_CONFIG_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        if mapping["generator_schema_version"] != TARGET_GENERATOR_SCHEMA_VERSION:
            msg = f"generator_schema_version must be {TARGET_GENERATOR_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        if mapping["namespace"] != PHASE_II_TARGET_NAMESPACE:
            msg = f"namespace must be {PHASE_II_TARGET_NAMESPACE!r}."
            raise ValueError(msg)
        raw_allocations = cast("Sequence[object]", mapping["allocations"])
        config = cls(
            preregistration_checksum=cast("str", mapping["preregistration_checksum"]),
            target_population_policy_checksum=cast("str", mapping["target_population_policy_checksum"]),
            rng_policy_checksum=cast("str", mapping["rng_policy_checksum"]),
            numeric_policy_checksum=cast("str", mapping["numeric_policy_checksum"]),
            allocation_policy_checksum=cast("str", mapping["allocation_policy_checksum"]),
            role_master_entropy_commitment=cast("str", mapping["role_master_entropy_commitment"]),
            data_role=cast("str", mapping["data_role"]),
            population_scope=cast("str", mapping["population_scope"]),
            allocations=tuple(TargetAllocation.from_dict(item) for item in raw_allocations),
        )
        supplied_population_id = require_string(mapping["population_id"], "population_id")
        if _POPULATION_ID_PATTERN.fullmatch(supplied_population_id) is None:
            msg = "population_id is not a Phase II target-population identifier."
            raise ValueError(msg)
        if supplied_population_id != config.population_id:
            msg = "population_id does not match the resolved population identity."
            raise ValueError(msg)
        if mapping["content_checksum"] != config.content_checksum:
            msg = "Target population config checksum changed during normalization."
            raise ValueError(msg)
        return config

    @classmethod
    def from_json(cls, payload: str) -> TargetPopulationConfig:
        """Construct a target population config from canonical JSON.

        Returns:
            The validated immutable configuration.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def _balanced_allocations(family_id: str, qubit_count: int, total: int) -> tuple[TargetAllocation, ...]:
    """Return one exactly balanced within-family allocation.

    Args:
        family_id: Primary target family.
        qubit_count: Target qubit count.
        total: Total instances for the family/qubit cell.

    Returns:
        Ordered stratum allocations.

    Raises:
        ValueError: If the total cannot be balanced exactly.
    """
    strata = PRIMARY_FAMILY_STRATA[family_id]
    quotient, remainder = divmod(total, len(strata))
    if remainder:
        msg = f"Target count {total} cannot be balanced across {len(strata)} strata for {family_id!r}."
        raise ValueError(msg)
    return tuple(
        TargetAllocation(
            family_id=family_id,
            stratum_id=stratum_id,
            qubit_count=qubit_count,
            instance_count=quotient,
        )
        for stratum_id in strata
    )


def role_master_entropy_commitment(role_master_entropy: bytes | str) -> str:
    """Return the non-revealing SHA-256 commitment to one role master key.

    Args:
        role_master_entropy: Exact 32-byte key or its lowercase hex encoding.

    Returns:
        A prefixed SHA-256 digest of the raw key bytes.
    """
    master = _master_bytes(role_master_entropy)
    return f"sha256:{hashlib.sha256(master).hexdigest()}"


def build_target_population_config(
    preregistration: InitialPreregistration,
    data_role: str,
    *,
    role_master_entropy_commitment: str,
    population_scope: str = "primary_q6",
    confirmatory_target_count_by_family: Mapping[str, object] | None = None,
) -> TargetPopulationConfig:
    """Build the exact preregistered population allocation for one data role.

    Args:
        preregistration: Trusted initial Phase II protocol.
        data_role: Development, screening-selection, or confirmatory role.
        role_master_entropy_commitment: SHA-256 commitment to the external
            32-byte role master key.
        population_scope: Isolated primary-q6 or secondary-q12 population.
        confirmatory_target_count_by_family: Sealed final counts, required only
            for a confirmatory population.

    Returns:
        The immutable, checksum-addressed population configuration.

    Raises:
        ValueError: If role-specific counts are absent, unexpected, or invalid.
    """
    _require_trusted_preregistration(preregistration)
    role = require_slug(data_role, "data_role")
    if role not in TARGET_INSTANCE_ROLES:
        msg = f"data_role must be one of {TARGET_INSTANCE_ROLES!r}."
        raise ValueError(msg)
    scope = require_slug(population_scope, "population_scope")
    if scope not in TARGET_POPULATION_SCOPES:
        msg = f"population_scope must be one of {TARGET_POPULATION_SCOPES!r}."
        raise ValueError(msg)
    master_commitment = require_checksum(role_master_entropy_commitment, "role_master_entropy_commitment")
    allocations: list[TargetAllocation] = []
    if role == "development":
        if confirmatory_target_count_by_family is not None:
            msg = "Development populations cannot receive confirmatory target counts."
            raise ValueError(msg)
        for family_id in PRIMARY_TARGET_FAMILIES:
            allocations.extend(_balanced_allocations(family_id, 6, 12))
    elif role == "screening_selection":
        if confirmatory_target_count_by_family is not None:
            msg = "Screening populations cannot receive confirmatory target counts."
            raise ValueError(msg)
        qubit_count = 6 if scope == "primary_q6" else 12
        target_count = 12 if scope == "primary_q6" else 6
        for family_id in PRIMARY_TARGET_FAMILIES:
            allocations.extend(_balanced_allocations(family_id, qubit_count, target_count))
    else:
        if confirmatory_target_count_by_family is None:
            msg = "Confirmatory populations require target counts from the final sample-size design."
            raise ValueError(msg)
        counts = require_mapping(confirmatory_target_count_by_family, "confirmatory_target_count_by_family")
        if frozenset(counts) != frozenset(PRIMARY_TARGET_FAMILIES):
            msg = "Confirmatory target-count keys must exactly match the four primary target families."
            raise ValueError(msg)
        for family_id in PRIMARY_TARGET_FAMILIES:
            count = require_int(counts[family_id], f"confirmatory_target_count_by_family.{family_id}", minimum=1)
            if not 24 <= count <= 96 or count % 6:
                msg = "Confirmatory target counts must be 24 through 96 per family in increments of six."
                raise ValueError(msg)
            allocations.extend(_balanced_allocations(family_id, 6, count))
    return TargetPopulationConfig(
        preregistration_checksum=preregistration.content_checksum,
        target_population_policy_checksum=preregistration.target_population_configuration_checksum,
        rng_policy_checksum=TRUSTED_TARGET_RNG_POLICY_CHECKSUM,
        numeric_policy_checksum=TRUSTED_TARGET_NUMERIC_POLICY_CHECKSUM,
        allocation_policy_checksum=TRUSTED_TARGET_ALLOCATION_POLICY_CHECKSUM,
        role_master_entropy_commitment=master_commitment,
        data_role=role,
        population_scope=scope,
        allocations=tuple(allocations),
    )


def _target_identity(
    population_config_checksum: str,
    family_id: str,
    stratum_id: str,
    qubit_count: int,
    instance_seed: str,
) -> dict[str, object]:
    """Return the exact stable instance-identity payload."""
    return {
        "generator_schema_version": TARGET_GENERATOR_SCHEMA_VERSION,
        "population_config_checksum": population_config_checksum,
        "family_id": family_id,
        "stratum_id": stratum_id,
        "qubit_count": qubit_count,
        "instance_seed": instance_seed,
    }


def _target_instance_id(
    population_config_checksum: str,
    family_id: str,
    stratum_id: str,
    qubit_count: int,
    instance_seed: str,
) -> str:
    """Return a full-digest Phase II target-instance identifier."""
    identity = _target_identity(
        population_config_checksum,
        family_id,
        stratum_id,
        qubit_count,
        instance_seed,
    )
    return f"{PHASE_II_TARGET_ID_PREFIX}{canonical_checksum(identity).removeprefix('sha256:')}"


def _require_float_sequence(
    value: object,
    name: str,
    *,
    length: int,
) -> tuple[float, ...]:
    """Require an immutable sequence of exact finite floats.

    Args:
        value: Candidate sequence.
        name: Human-readable field name.
        length: Required sequence length.

    Returns:
        The validated tuple of floats.

    Raises:
        TypeError: If the input is not a sequence of exact floats.
        ValueError: If the sequence length differs.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = f"{name} must be a sequence of floats."
        raise TypeError(msg)
    result = tuple(require_float(item, f"{name}[{index}]") for index, item in enumerate(value))
    if len(result) != length:
        msg = f"{name} must contain exactly {length} values."
        raise ValueError(msg)
    return result


def _require_int_sequence(
    value: object,
    name: str,
    *,
    length: int,
    minimum: int,
) -> tuple[int, ...]:
    """Require an immutable sequence of exact bounded integers.

    Args:
        value: Candidate sequence.
        name: Human-readable field name.
        length: Required sequence length.
        minimum: Inclusive lower bound.

    Returns:
        The validated tuple of integers.

    Raises:
        TypeError: If the input is not a sequence of exact integers.
        ValueError: If a value or sequence length differs.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = f"{name} must be a sequence of integers."
        raise TypeError(msg)
    result = tuple(require_int(item, f"{name}[{index}]", minimum=minimum) for index, item in enumerate(value))
    if len(result) != length:
        msg = f"{name} must contain exactly {length} values."
        raise ValueError(msg)
    return result


def _validate_parameters(
    family_id: str,
    stratum_id: str,
    qubit_count: int,
    parameters: object,
) -> Mapping[str, object]:
    """Strictly validate one sampled parameter record.

    Returns:
        Deeply frozen target parameters.

    Raises:
        TypeError: If a nested parameter has the wrong scalar or container type.
        ValueError: If fields or physical constraints differ from the v1 rules.
    """
    frozen = freeze_json_mapping(parameters, "target parameters")
    if family_id == "gaussian_amplitude":
        require_exact_keys(frozen, _GAUSSIAN_PARAMETER_KEYS, "Gaussian target parameters")
        mean = require_float(frozen["mean"], "Gaussian mean", minimum=0.3, maximum=0.7)
        width = require_float(frozen["width"], "Gaussian width", minimum=0.05, maximum=0.1)
        if mean >= 0.7 or width >= 0.1:
            msg = "Gaussian mean and width must lie in their preregistered half-open intervals."
            raise ValueError(msg)
        if mean - 3.0 * width < 0.0 or mean + 3.0 * width > 1.0:
            msg = "Gaussian parameters must keep three standard deviations inside [0, 1]."
            raise ValueError(msg)
    elif family_id == "tfim_ground_state":
        require_exact_keys(frozen, _TFIM_PARAMETER_KEYS, "TFIM target parameters")
        attempt_index = require_int(frozen["attempt_index"], "TFIM attempt_index")
        if attempt_index >= 100:
            msg = "TFIM attempt_index must be below the preregistered 100-attempt limit."
            raise ValueError(msg)
        _require_float_sequence(frozen["couplings"], "TFIM couplings", length=qubit_count - 1)
        _require_float_sequence(frozen["fields"], "TFIM fields", length=qubit_count)
        require_float(frozen["ground_energy"], "TFIM ground_energy")
        gap = require_float(frozen["ground_state_gap"], "TFIM ground_state_gap", minimum=0.0)
        threshold = require_float(frozen["gap_threshold"], "TFIM gap_threshold", minimum=0.0)
        spectral_norm = require_float(frozen["spectral_norm"], "TFIM spectral_norm", minimum=0.0)
        if gap <= threshold or not math.isclose(
            threshold,
            1e-10 * max(1.0, spectral_norm),
            rel_tol=1e-15,
            abs_tol=0.0,
        ):
            msg = "TFIM gap metadata does not pass the preregistered degeneracy rule."
            raise ValueError(msg)
        ratio = _REGIME_RATIOS[stratum_id]
        fields = cast("Sequence[float]", frozen["fields"])
        if any(not 0.8 * ratio <= field < 1.2 * ratio for field in fields):
            msg = "TFIM fields lie outside their preregistered regime distribution."
            raise ValueError(msg)
        couplings = cast("Sequence[float]", frozen["couplings"])
        if any(not 0.8 <= coupling < 1.2 for coupling in couplings):
            msg = "TFIM couplings lie outside their preregistered distribution."
            raise ValueError(msg)
    elif family_id == "haar_random":
        require_exact_keys(frozen, _HAAR_PARAMETER_KEYS, "Haar target parameters")
        if require_int(frozen["dimension"], "Haar dimension", minimum=1) != 2**qubit_count:
            msg = "Haar dimension must equal 2**qubit_count."
            raise ValueError(msg)
    else:
        require_exact_keys(frozen, _MPS_PARAMETER_KEYS, "random-MPS target parameters")
        bond_dimension = require_int(frozen["bond_dimension"], "MPS bond_dimension", minimum=1)
        if bond_dimension != _MPS_BOND_DIMENSIONS[stratum_id]:
            msg = "MPS bond dimension does not match its preregistered stratum."
            raise ValueError(msg)
        expected_bonds = tuple(
            min(bond_dimension, 2**site, 2 ** (qubit_count - site)) for site in range(qubit_count + 1)
        )
        bonds = _require_int_sequence(
            frozen["bond_dimensions"],
            "MPS bond_dimensions",
            length=qubit_count + 1,
            minimum=1,
        )
        if bonds != expected_bonds:
            msg = "MPS bond dimensions do not follow the preregistered finite-chain formula."
            raise ValueError(msg)
        raw_shapes = frozen["tensor_shapes"]
        if isinstance(raw_shapes, (str, bytes)) or not isinstance(raw_shapes, Sequence):
            msg = "MPS tensor_shapes must be a sequence."
            raise TypeError(msg)
        shapes = tuple(
            _require_int_sequence(item, f"MPS tensor_shapes[{index}]", length=3, minimum=1)
            for index, item in enumerate(raw_shapes)
        )
        expected_shapes = tuple((bonds[site], 2, bonds[site + 1]) for site in range(qubit_count))
        if shapes != expected_shapes:
            msg = "MPS tensor shapes do not match the preregistered bond dimensions."
            raise ValueError(msg)
    return frozen


@dataclass(frozen=True, slots=True)
class TargetInstanceSpec:
    """One seed-bearing Phase II target specification without amplitudes."""

    data_role: str
    population_config_checksum: str
    family_id: str
    stratum_id: str
    qubit_count: int
    instance_seed: str
    target_instance_id: str
    parameters: Mapping[str, object]
    _marker: object = field(repr=False, compare=False)
    schema_version: str = field(default=TARGET_INSTANCE_SPEC_SCHEMA_VERSION, init=False)
    namespace: str = field(default=PHASE_II_TARGET_NAMESPACE, init=False)

    def __post_init__(self) -> None:
        """Validate identity, namespace isolation, and sampled parameters.

        Raises:
            ValueError: If construction bypasses a factory or identity differs.
        """
        if self._marker is not _MATERIALIZATION_SENTINEL:
            msg = "TargetInstanceSpec records may only be created by the deterministic WP16 factories."
            raise ValueError(msg)
        role = require_slug(self.data_role, "data_role")
        if role not in TARGET_INSTANCE_ROLES:
            msg = f"data_role must be one of {TARGET_INSTANCE_ROLES!r}."
            raise ValueError(msg)
        object.__setattr__(self, "data_role", role)
        population_checksum = require_checksum(self.population_config_checksum, "population_config_checksum")
        object.__setattr__(self, "population_config_checksum", population_checksum)
        family, stratum = _require_family_stratum(self.family_id, self.stratum_id, "target instance spec")
        object.__setattr__(self, "family_id", family)
        object.__setattr__(self, "stratum_id", stratum)
        qubit_count = require_int(self.qubit_count, "qubit_count", minimum=2)
        object.__setattr__(self, "qubit_count", qubit_count)
        instance_seed = _require_hex(self.instance_seed, "instance_seed", _HEX_128_PATTERN)
        object.__setattr__(self, "instance_seed", instance_seed)
        target_id = require_string(self.target_instance_id, "target_instance_id")
        if (
            _TARGET_ID_PATTERN.fullmatch(target_id) is None
            or target_id in PHASE_I_TARGET_IDS
            or target_id.startswith(("legacy", "phase1", "phase_i"))
        ):
            msg = "target_instance_id must use the isolated full-digest Phase II namespace."
            raise ValueError(msg)
        expected_id = _target_instance_id(
            population_checksum,
            family,
            stratum,
            qubit_count,
            instance_seed,
        )
        if target_id != expected_id:
            msg = "target_instance_id does not match its population, family, stratum, qubits, and seed."
            raise ValueError(msg)
        object.__setattr__(self, "target_instance_id", target_id)
        object.__setattr__(
            self,
            "parameters",
            _validate_parameters(family, stratum, qubit_count, self.parameters),
        )

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete seed-bearing target specification."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return the checksum-covered target-specification payload."""
        return {
            "schema_version": self.schema_version,
            "namespace": self.namespace,
            "data_role": self.data_role,
            "population_config_checksum": self.population_config_checksum,
            "family_id": self.family_id,
            "stratum_id": self.stratum_id,
            "qubit_count": self.qubit_count,
            "instance_seed": self.instance_seed,
            "target_instance_id": self.target_instance_id,
            "parameters": thaw_json_mapping(self.parameters),
        }

    def to_dict(self) -> dict[str, object]:
        """Return canonicalizable sealed target-specification data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> TargetInstanceSpec:
        """Construct and checksum-verify a target specification.

        Returns:
            The validated immutable target specification.

        Raises:
            ValueError: If its schema, namespace, identity, or checksum differs.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_SPEC_KEYS, name="target instance spec")
        if mapping["schema_version"] != TARGET_INSTANCE_SPEC_SCHEMA_VERSION:
            msg = f"schema_version must be {TARGET_INSTANCE_SPEC_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        if mapping["namespace"] != PHASE_II_TARGET_NAMESPACE:
            msg = f"namespace must be {PHASE_II_TARGET_NAMESPACE!r}."
            raise ValueError(msg)
        spec = cls(
            data_role=cast("str", mapping["data_role"]),
            population_config_checksum=cast("str", mapping["population_config_checksum"]),
            family_id=cast("str", mapping["family_id"]),
            stratum_id=cast("str", mapping["stratum_id"]),
            qubit_count=cast("int", mapping["qubit_count"]),
            instance_seed=cast("str", mapping["instance_seed"]),
            target_instance_id=cast("str", mapping["target_instance_id"]),
            parameters=cast("Mapping[str, object]", mapping["parameters"]),
            _marker=_MATERIALIZATION_SENTINEL,
        )
        if mapping["content_checksum"] != spec.content_checksum:
            msg = "Target instance spec checksum changed during normalization."
            raise ValueError(msg)
        return spec

    @classmethod
    def from_json(cls, payload: str) -> TargetInstanceSpec:
        """Construct a target specification from canonical JSON.

        Returns:
            The validated immutable target specification.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class TargetPopulationCommitment:
    """Public checksum-only commitment to a custodied target manifest."""

    target_manifest_checksum: str
    target_count_by_family: Mapping[str, object]
    schema_version: str = field(default=TARGET_POPULATION_COMMITMENT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Validate the intentionally public manifest information.

        Raises:
            ValueError: If the checksum or family counts differ from the schema.
        """
        object.__setattr__(
            self,
            "target_manifest_checksum",
            require_checksum(self.target_manifest_checksum, "target_manifest_checksum"),
        )
        counts = freeze_json_mapping(self.target_count_by_family, "target_count_by_family")
        if frozenset(counts) != frozenset(PRIMARY_TARGET_FAMILIES):
            msg = "target_count_by_family must contain exactly the four primary families."
            raise ValueError(msg)
        for family_id in PRIMARY_TARGET_FAMILIES:
            require_int(counts[family_id], f"target_count_by_family.{family_id}", minimum=1)
        object.__setattr__(self, "target_count_by_family", counts)

    @property
    def content_checksum(self) -> str:
        """Checksum of the public commitment."""
        return canonical_checksum(self._content_dict())

    def _content_dict(self) -> dict[str, object]:
        """Return the commitment payload."""
        return {
            "schema_version": self.schema_version,
            "target_manifest_checksum": self.target_manifest_checksum,
            "target_count_by_family": thaw_json_mapping(self.target_count_by_family),
        }

    def to_dict(self) -> dict[str, object]:
        """Return canonicalizable sealed commitment data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> TargetPopulationCommitment:
        """Construct and checksum-verify a public commitment.

        Returns:
            The validated checksum-only commitment.

        Raises:
            ValueError: If its schema or checksum differs.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_COMMITMENT_KEYS, name="target population commitment")
        if mapping["schema_version"] != TARGET_POPULATION_COMMITMENT_SCHEMA_VERSION:
            msg = f"schema_version must be {TARGET_POPULATION_COMMITMENT_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        commitment = cls(
            target_manifest_checksum=cast("str", mapping["target_manifest_checksum"]),
            target_count_by_family=cast("Mapping[str, object]", mapping["target_count_by_family"]),
        )
        if mapping["content_checksum"] != commitment.content_checksum:
            msg = "Target population commitment checksum changed during normalization."
            raise ValueError(msg)
        return commitment

    @classmethod
    def from_json(cls, payload: str) -> TargetPopulationCommitment:
        """Construct a public commitment from canonical JSON.

        Returns:
            The validated checksum-only commitment.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class TargetPopulationManifest:
    """Externally custodied seed and parameter manifest without target vectors."""

    preregistration_checksum: str
    population_config_checksum: str
    data_role: str
    population_scope: str
    role_master_entropy_commitment: str
    allocations: tuple[TargetAllocation, ...]
    instances: tuple[TargetInstanceSpec, ...]
    schema_version: str = field(default=TARGET_POPULATION_MANIFEST_SCHEMA_VERSION, init=False)
    generator_schema_version: str = field(default=TARGET_GENERATOR_SCHEMA_VERSION, init=False)
    namespace: str = field(default=PHASE_II_TARGET_NAMESPACE, init=False)

    def __post_init__(self) -> None:
        """Validate custody metadata, ordering, uniqueness, and role separation.

        Raises:
            TypeError: If allocations or specifications are not immutable typed records.
            ValueError: If identity, ordering, uniqueness, or allocation differs.
        """
        preregistration_checksum = require_checksum(self.preregistration_checksum, "preregistration_checksum")
        if preregistration_checksum != TRUSTED_INITIAL_PREREGISTRATION_CHECKSUM:
            msg = "TargetPopulationManifest must reference the trusted Phase II preregistration."
            raise ValueError(msg)
        object.__setattr__(self, "preregistration_checksum", preregistration_checksum)
        population_checksum = require_checksum(self.population_config_checksum, "population_config_checksum")
        object.__setattr__(self, "population_config_checksum", population_checksum)
        role = require_slug(self.data_role, "data_role")
        if role not in TARGET_INSTANCE_ROLES:
            msg = f"data_role must be one of {TARGET_INSTANCE_ROLES!r}."
            raise ValueError(msg)
        object.__setattr__(self, "data_role", role)
        scope = require_slug(self.population_scope, "population_scope")
        if scope not in TARGET_POPULATION_SCOPES:
            msg = f"population_scope must be one of {TARGET_POPULATION_SCOPES!r}."
            raise ValueError(msg)
        object.__setattr__(self, "population_scope", scope)
        object.__setattr__(
            self,
            "role_master_entropy_commitment",
            require_checksum(self.role_master_entropy_commitment, "role_master_entropy_commitment"),
        )
        if not isinstance(self.allocations, tuple) or any(
            not isinstance(allocation, TargetAllocation) for allocation in self.allocations
        ):
            msg = "allocations must be a tuple of TargetAllocation records."
            raise TypeError(msg)
        _validate_allocation_policy(role, scope, self.allocations)
        if not isinstance(self.instances, tuple) or not self.instances:
            msg = "instances must be a nonempty tuple of TargetInstanceSpec records."
            raise TypeError(msg)
        if any(not isinstance(spec, TargetInstanceSpec) for spec in self.instances):
            msg = "instances must contain only TargetInstanceSpec records."
            raise TypeError(msg)
        if tuple(sorted(self.instances, key=_spec_sort_key)) != self.instances:
            msg = "instances must use the frozen family, stratum, qubit, and target-ID ordering."
            raise ValueError(msg)
        ids = [spec.target_instance_id for spec in self.instances]
        seeds = [spec.instance_seed for spec in self.instances]
        if len(ids) != len(set(ids)) or len(seeds) != len(set(seeds)):
            msg = "A target manifest must not contain duplicate instance identifiers or instance seeds."
            raise ValueError(msg)
        expected_counts = {
            (allocation.family_id, allocation.stratum_id, allocation.qubit_count): allocation.instance_count
            for allocation in self.allocations
        }
        actual_counts: dict[tuple[str, str, int], int] = {}
        for spec in self.instances:
            if spec.population_config_checksum != population_checksum or spec.data_role != role:
                msg = "Every target specification must belong to the manifest's exact population and data role."
                raise ValueError(msg)
            key = (spec.family_id, spec.stratum_id, spec.qubit_count)
            actual_counts[key] = actual_counts.get(key, 0) + 1
        if actual_counts != expected_counts:
            msg = "Target instances do not form the exact Cartesian allocation declared by the manifest."
            raise ValueError(msg)

    @property
    def manifest_id(self) -> str:
        """Stable full-digest identifier for the complete custodied payload."""
        digest = canonical_checksum(self._identity_dict()).removeprefix("sha256:")
        return f"{PHASE_II_POPULATION_ID_PREFIX}{digest}"

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete seed-bearing manifest."""
        return canonical_checksum(self._content_dict())

    def _identity_dict(self) -> dict[str, object]:
        """Return fields from which the manifest identifier is derived."""
        return {
            "schema_version": self.schema_version,
            "generator_schema_version": self.generator_schema_version,
            "namespace": self.namespace,
            "preregistration_checksum": self.preregistration_checksum,
            "population_config_checksum": self.population_config_checksum,
            "data_role": self.data_role,
            "population_scope": self.population_scope,
            "role_master_entropy_commitment": self.role_master_entropy_commitment,
            "allocations": [allocation.to_dict() for allocation in self.allocations],
            "instances": [spec.to_dict() for spec in self.instances],
        }

    def _content_dict(self) -> dict[str, object]:
        """Return the checksum-covered manifest payload."""
        return {**self._identity_dict(), "manifest_id": self.manifest_id}

    def to_dict(self) -> dict[str, object]:
        """Return canonicalizable sealed seed-bearing manifest data."""
        return {**self._content_dict(), "content_checksum": self.content_checksum}

    def to_json(self) -> str:
        """Return canonical sealed JSON text."""
        return canonical_json(self.to_dict())

    def public_commitment(self) -> TargetPopulationCommitment:
        """Return the checksum-only artifact safe to publish before reveal."""
        counts = {
            family_id: sum(
                allocation.instance_count for allocation in self.allocations if allocation.family_id == family_id
            )
            for family_id in PRIMARY_TARGET_FAMILIES
        }
        return TargetPopulationCommitment(
            target_manifest_checksum=self.content_checksum,
            target_count_by_family=counts,
        )

    @classmethod
    def from_dict(cls, data: object) -> TargetPopulationManifest:
        """Construct and checksum-verify a seed-bearing target manifest.

        Returns:
            The validated immutable target manifest.

        Raises:
            ValueError: If its schema, namespace, identifier, or checksum differs.
        """
        mapping = verify_sealed_mapping(data, expected_keys=_MANIFEST_KEYS, name="target population manifest")
        if mapping["schema_version"] != TARGET_POPULATION_MANIFEST_SCHEMA_VERSION:
            msg = f"schema_version must be {TARGET_POPULATION_MANIFEST_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        if mapping["generator_schema_version"] != TARGET_GENERATOR_SCHEMA_VERSION:
            msg = f"generator_schema_version must be {TARGET_GENERATOR_SCHEMA_VERSION!r}."
            raise ValueError(msg)
        if mapping["namespace"] != PHASE_II_TARGET_NAMESPACE:
            msg = f"namespace must be {PHASE_II_TARGET_NAMESPACE!r}."
            raise ValueError(msg)
        raw_instances = cast("Sequence[object]", mapping["instances"])
        manifest = cls(
            preregistration_checksum=cast("str", mapping["preregistration_checksum"]),
            population_config_checksum=cast("str", mapping["population_config_checksum"]),
            data_role=cast("str", mapping["data_role"]),
            population_scope=cast("str", mapping["population_scope"]),
            role_master_entropy_commitment=cast("str", mapping["role_master_entropy_commitment"]),
            allocations=tuple(
                TargetAllocation.from_dict(item) for item in cast("Sequence[object]", mapping["allocations"])
            ),
            instances=tuple(TargetInstanceSpec.from_dict(item) for item in raw_instances),
        )
        supplied_id = require_string(mapping["manifest_id"], "manifest_id")
        if _POPULATION_ID_PATTERN.fullmatch(supplied_id) is None or supplied_id != manifest.manifest_id:
            msg = "manifest_id does not match the complete seed-bearing manifest identity."
            raise ValueError(msg)
        if mapping["content_checksum"] != manifest.content_checksum:
            msg = "Target population manifest checksum changed during normalization."
            raise ValueError(msg)
        return manifest

    @classmethod
    def from_json(cls, payload: str) -> TargetPopulationManifest:
        """Construct a seed-bearing manifest from canonical JSON.

        Returns:
            The validated immutable target manifest.
        """
        return cls.from_dict(load_canonical_json_object(payload))


def _master_bytes(role_master_entropy: bytes | str) -> bytes:
    """Normalize an exact 256-bit role master entropy value.

    Returns:
        The exact 32 raw bytes.

    Raises:
        TypeError: If the key is neither bytes nor text.
        ValueError: If its width or hexadecimal spelling differs.
    """
    if isinstance(role_master_entropy, bytes):
        if len(role_master_entropy) != 32:
            msg = "role_master_entropy must contain exactly 32 bytes."
            raise ValueError(msg)
        return role_master_entropy
    if type(role_master_entropy) is str:
        text = _require_hex(role_master_entropy, "role_master_entropy", _HEX_256_PATTERN)
        return bytes.fromhex(text)
    msg = "role_master_entropy must be 32 bytes or its 64-character lowercase hexadecimal encoding."
    raise TypeError(msg)


def _hmac_128(master: bytes, identity: Mapping[str, object]) -> bytes:
    """Return the preregistered first 128 HMAC-SHA256 bits."""
    return hmac.new(master, canonical_json(identity).encode("utf-8"), hashlib.sha256).digest()[:16]


def _derive_instance_seed(
    master: bytes,
    config: TargetPopulationConfig,
    allocation: TargetAllocation,
    instance_index: int,
) -> str:
    """Derive one 128-bit instance seed from its canonical allocation identity.

    Returns:
        The 32-character lowercase hexadecimal seed.
    """
    identity = {
        "random_stream_domain": "target_generation",
        "substream": "instance_seed",
        "generator_schema_version": TARGET_GENERATOR_SCHEMA_VERSION,
        "population_config_checksum": config.content_checksum,
        "data_role": config.data_role,
        "family_id": allocation.family_id,
        "stratum_id": allocation.stratum_id,
        "qubit_count": allocation.qubit_count,
        "instance_index": instance_index,
    }
    return _hmac_128(master, identity).hex()


def _component_rng(
    master: bytes,
    target_instance_id: str,
    component: str,
    *,
    attempt_index: int | None = None,
) -> np.random.Generator:
    """Construct one PCG64/SeedSequence component substream.

    Returns:
        A generator initialized from the big-endian 128-bit HMAC prefix.
    """
    identity: dict[str, object] = {
        "random_stream_domain": "target_generation",
        "generator_schema_version": TARGET_GENERATOR_SCHEMA_VERSION,
        "target_instance_id": target_instance_id,
        "component_substream": component,
    }
    if attempt_index is not None:
        identity["rejection_attempt_index"] = attempt_index
    entropy_bytes = _hmac_128(master, identity)
    entropy = int.from_bytes(entropy_bytes, byteorder="big", signed=False)
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def _tfim_hamiltonian(couplings: NDArray[np.float64], fields: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the exact dense open-chain disordered TFIM Hamiltonian."""
    qubit_count = fields.size
    dimension = 2**qubit_count
    basis = np.arange(dimension, dtype=np.int64)
    hamiltonian = np.zeros((dimension, dimension), dtype=np.float64)
    diagonal = np.zeros(dimension, dtype=np.float64)
    for site, coupling in enumerate(couplings):
        left = (basis >> site) & 1
        right = (basis >> (site + 1)) & 1
        diagonal -= coupling * np.where(left == right, 1.0, -1.0)
    hamiltonian[basis, basis] = diagonal
    for site, field_value in enumerate(fields):
        hamiltonian[basis, basis ^ (1 << site)] -= field_value
    return hamiltonian


def _tfim_parameter_record(
    master: bytes,
    target_instance_id: str,
    stratum_id: str,
    qubit_count: int,
) -> dict[str, object]:
    """Sample a nondegenerate TFIM instance without constructing its eigenvectors.

    Manifest construction needs the frozen physical parameters, ground energy,
    and nondegeneracy evidence, but it must not expose or even transiently
    construct the held-out state vector.  ``eigvalsh`` is the spectrum-only
    counterpart of the preregistered dense Hermitian ``eigh`` policy; the
    authorized materializer later uses ``eigh`` on this exact Hamiltonian.

    Returns:
        The sampled JSON-native parameter and spectral record.

    Raises:
        RuntimeError: If all 100 sampled attempts violate the strict gap threshold.
    """
    ratio = _REGIME_RATIOS[stratum_id]
    for attempt_index in range(100):
        coupling_rng = _component_rng(
            master,
            target_instance_id,
            "couplings",
            attempt_index=attempt_index,
        )
        field_rng = _component_rng(
            master,
            target_instance_id,
            "fields",
            attempt_index=attempt_index,
        )
        couplings = np.asarray(0.8 + 0.4 * coupling_rng.random(qubit_count - 1), dtype=np.float64)
        fields = np.asarray(ratio * (0.8 + 0.4 * field_rng.random(qubit_count)), dtype=np.float64)
        eigenvalues = np.linalg.eigvalsh(_tfim_hamiltonian(couplings, fields))
        spectral_norm = float(np.max(np.abs(eigenvalues)))
        threshold = 1e-10 * max(1.0, spectral_norm)
        gap = float(eigenvalues[1] - eigenvalues[0])
        if gap > threshold:
            return {
                "attempt_index": attempt_index,
                "couplings": [float(value) for value in couplings],
                "fields": [float(value) for value in fields],
                "ground_energy": float(eigenvalues[0]),
                "ground_state_gap": gap,
                "gap_threshold": threshold,
                "spectral_norm": spectral_norm,
            }
    msg = f"TFIM target {target_instance_id!r} exhausted the preregistered 100 attempts."
    raise RuntimeError(msg)


def _tfim_ground_state_vector(
    parameters: Mapping[str, object],
    *,
    spectrum_agreement_rtol: float,
    spectrum_agreement_atol: float,
) -> NDArray[np.complex128]:
    """Construct an authorized TFIM ground-state vector from sealed parameters.

    Returns:
        The normalized, phase-fixed complex128 ground-state vector.

    Raises:
        ValueError: If the dense eigensystem disagrees with the sealed spectrum.
    """
    couplings = np.asarray(parameters["couplings"], dtype=np.float64)
    fields = np.asarray(parameters["fields"], dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(_tfim_hamiltonian(couplings, fields))
    spectral_norm = float(np.max(np.abs(eigenvalues)))
    gap = float(eigenvalues[1] - eigenvalues[0])
    comparisons = (
        (float(eigenvalues[0]), cast("float", parameters["ground_energy"])),
        (gap, cast("float", parameters["ground_state_gap"])),
        (spectral_norm, cast("float", parameters["spectral_norm"])),
    )
    if any(
        not math.isclose(
            actual,
            expected,
            rel_tol=spectrum_agreement_rtol,
            abs_tol=spectrum_agreement_atol,
        )
        for actual, expected in comparisons
    ):
        msg = "Authorized TFIM eigensystem does not agree with the sealed spectrum."
        raise ValueError(msg)
    threshold = cast("float", parameters["gap_threshold"])
    if gap <= threshold:
        msg = "Authorized TFIM eigensystem violates the sealed nondegeneracy threshold."
        raise ValueError(msg)
    vector = np.asarray(eigenvectors[:, 0], dtype=np.complex128)
    return _normalize_and_fix_phase(vector)


def _mps_dimensions(qubit_count: int, bond_dimension: int) -> tuple[int, ...]:
    """Return the exact finite-chain bond dimensions."""
    return tuple(min(bond_dimension, 2**site, 2 ** (qubit_count - site)) for site in range(qubit_count + 1))


def _parameter_record(
    master: bytes,
    target_instance_id: str,
    family_id: str,
    stratum_id: str,
    qubit_count: int,
) -> dict[str, object]:
    """Generate the exact sampled physical parameter record without returning amplitudes.

    Returns:
        JSON-native sampled parameters for the requested family.
    """
    if family_id == "gaussian_amplitude":
        mean_rng = _component_rng(master, target_instance_id, "parameters_mean")
        width_rng = _component_rng(master, target_instance_id, "parameters_width")
        return {
            "mean": float(mean_rng.uniform(0.3, 0.7)),
            "width": float(width_rng.uniform(0.05, 0.1)),
        }
    if family_id == "tfim_ground_state":
        return _tfim_parameter_record(
            master,
            target_instance_id,
            stratum_id,
            qubit_count,
        )
    if family_id == "haar_random":
        return {"dimension": 2**qubit_count}
    bond_dimension = _MPS_BOND_DIMENSIONS[stratum_id]
    bonds = _mps_dimensions(qubit_count, bond_dimension)
    return {
        "bond_dimension": bond_dimension,
        "bond_dimensions": list(bonds),
        "tensor_shapes": [[bonds[site], 2, bonds[site + 1]] for site in range(qubit_count)],
    }


def _normalize_and_fix_phase(vector: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Normalize and make the largest-magnitude lowest-index entry nonnegative real.

    Returns:
        The normalized complex128 vector with fixed global phase.

    Raises:
        ValueError: If the vector has zero or nonfinite norm.
    """
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        msg = "Cannot normalize a zero or nonfinite target vector."
        raise ValueError(msg)
    normalized = np.asarray(vector / norm, dtype=np.complex128)
    pivot_index = int(np.argmax(np.abs(normalized)))
    pivot = normalized[pivot_index]
    normalized *= np.conjugate(pivot) / abs(pivot)
    return np.asarray(normalized / np.linalg.norm(normalized), dtype=np.complex128)


def _materialize_vector(
    master: bytes,
    spec: TargetInstanceSpec,
    *,
    spectrum_agreement_rtol: float,
    spectrum_agreement_atol: float,
) -> NDArray[np.complex128]:
    """Materialize one target vector using only its verified seed-bearing spec.

    Returns:
        The normalized phase-fixed complex128 target vector.

    Raises:
        ValueError: If parameters change or MPS canonicalization fails.
    """
    dimension = 2**spec.qubit_count
    if spec.family_id == "gaussian_amplitude":
        indices = np.arange(dimension, dtype=np.uint64)
        grid = np.zeros(dimension, dtype=np.float64)
        for site in range(spec.qubit_count):
            bits = ((indices >> np.uint64(site)) & np.uint64(1)).astype(np.float64)
            grid += bits * (2.0 ** (-(site + 1)))
        mean = cast("float", spec.parameters["mean"])
        width = cast("float", spec.parameters["width"])
        amplitudes = np.exp(-((grid - mean) ** 2) / (4.0 * width**2))
        return _normalize_and_fix_phase(np.asarray(amplitudes, dtype=np.complex128))
    if spec.family_id == "tfim_ground_state":
        parameters = _tfim_parameter_record(
            master,
            spec.target_instance_id,
            spec.stratum_id,
            spec.qubit_count,
        )
        if canonical_checksum(parameters) != canonical_checksum(spec.parameters):
            msg = f"TFIM parameters changed before materialization for {spec.target_instance_id!r}."
            raise ValueError(msg)
        return _tfim_ground_state_vector(
            spec.parameters,
            spectrum_agreement_rtol=spectrum_agreement_rtol,
            spectrum_agreement_atol=spectrum_agreement_atol,
        )
    if spec.family_id == "haar_random":
        real_rng = _component_rng(master, spec.target_instance_id, "real")
        imag_rng = _component_rng(master, spec.target_instance_id, "imag")
        real = real_rng.standard_normal(dimension, dtype=np.float64)
        imag = imag_rng.standard_normal(dimension, dtype=np.float64)
        return _normalize_and_fix_phase(np.asarray(real + 1j * imag, dtype=np.complex128))

    bonds = cast("Sequence[int]", spec.parameters["bond_dimensions"])
    tensors = [
        np.asarray(
            _component_rng(master, spec.target_instance_id, f"tensor_site_{site}").standard_normal(
                (bonds[site], 2, bonds[site + 1]),
                dtype=np.float64,
            ),
            dtype=np.float64,
        )
        for site in range(spec.qubit_count)
    ]
    for site in range(spec.qubit_count - 1):
        left, physical, right = tensors[site].shape
        q_factor, r_factor = np.linalg.qr(tensors[site].reshape(left * physical, right), mode="reduced")
        diagonal = np.diag(r_factor)
        signs = np.where(diagonal < 0.0, -1.0, 1.0)
        q_factor *= signs[np.newaxis, :]
        r_factor *= signs[:, np.newaxis]
        tensors[site] = np.asarray(q_factor.reshape(left, physical, right), dtype=np.float64)
        tensors[site + 1] = np.asarray(np.tensordot(r_factor, tensors[site + 1], axes=([1], [0])))
    last_norm = float(np.linalg.norm(tensors[-1]))
    if last_norm <= 0.0 or not math.isfinite(last_norm):
        msg = f"Random-MPS target {spec.target_instance_id!r} produced a zero or nonfinite canonical form."
        raise ValueError(msg)
    tensors[-1] = np.asarray(tensors[-1] / last_norm, dtype=np.float64)
    dense = tensors[0][0, :, :]
    for tensor in tensors[1:]:
        dense = np.tensordot(dense, tensor, axes=([-1], [0]))
    dense = np.squeeze(dense, axis=-1)
    state = np.transpose(dense, axes=tuple(reversed(range(spec.qubit_count)))).reshape(-1)
    return _normalize_and_fix_phase(np.asarray(state, dtype=np.complex128))


def create_target_population_manifest(
    config: TargetPopulationConfig,
    preregistration: InitialPreregistration,
    role_master_entropy: bytes | str,
) -> TargetPopulationManifest:
    """Create a deterministic custodied seed manifest without state vectors.

    Args:
        config: Exact role-specific population configuration.
        preregistration: Trusted WP15 protocol.
        role_master_entropy: Externally custodied 256-bit role master seed.

    Returns:
        The checksum-sealed seed and sampled-parameter manifest.

    Raises:
        TypeError: If a config, preregistration, or key has the wrong type.
        ValueError: If the config, policy, or entropy is inconsistent.
        RuntimeError: If the runtime differs or TFIM rejection is exhausted.
    """
    _require_trusted_preregistration(preregistration)
    _require_runtime(preregistration)
    if not isinstance(config, TargetPopulationConfig):
        msg = "config must be a TargetPopulationConfig."
        raise TypeError(msg)
    if (
        config.preregistration_checksum != preregistration.content_checksum
        or config.target_population_policy_checksum != preregistration.target_population_configuration_checksum
    ):
        msg = "Target population config does not reference the supplied preregistration and target policy."
        raise ValueError(msg)
    master = _master_bytes(role_master_entropy)
    if role_master_entropy_commitment(master) != config.role_master_entropy_commitment:
        msg = "role_master_entropy does not match the commitment sealed in the population config."
        raise ValueError(msg)
    specs: list[TargetInstanceSpec] = []
    seen_seeds: set[str] = set()
    for allocation in config.allocations:
        for instance_index in range(allocation.instance_count):
            seed = _derive_instance_seed(master, config, allocation, instance_index)
            if seed in seen_seeds:
                msg = "Derived duplicate 128-bit target instance seed."
                raise RuntimeError(msg)
            seen_seeds.add(seed)
            target_id = _target_instance_id(
                config.content_checksum,
                allocation.family_id,
                allocation.stratum_id,
                allocation.qubit_count,
                seed,
            )
            parameters = _parameter_record(
                master,
                target_id,
                allocation.family_id,
                allocation.stratum_id,
                allocation.qubit_count,
            )
            specs.append(
                TargetInstanceSpec(
                    data_role=config.data_role,
                    population_config_checksum=config.content_checksum,
                    family_id=allocation.family_id,
                    stratum_id=allocation.stratum_id,
                    qubit_count=allocation.qubit_count,
                    instance_seed=seed,
                    target_instance_id=target_id,
                    parameters=parameters,
                    _marker=_MATERIALIZATION_SENTINEL,
                )
            )
    return TargetPopulationManifest(
        preregistration_checksum=preregistration.content_checksum,
        population_config_checksum=config.content_checksum,
        data_role=config.data_role,
        population_scope=config.population_scope,
        role_master_entropy_commitment=config.role_master_entropy_commitment,
        allocations=config.allocations,
        instances=tuple(sorted(specs, key=_spec_sort_key)),
    )


@dataclass(frozen=True, slots=True)
class TargetMaterializationAuthorization:
    """Opaque accidental-access guard for one exact target manifest."""

    preregistration_checksum: str
    population_config_checksum: str
    target_manifest_checksum: str
    data_role: str
    _marker: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        """Reject direct construction outside the authorization function.

        Raises:
            ValueError: If construction did not use the private factory marker.
        """
        if self._marker is not _MATERIALIZATION_SENTINEL:
            msg = "TargetMaterializationAuthorization may only be issued by authorize_target_materialization."
            raise ValueError(msg)


def authorize_target_materialization(
    preregistration: InitialPreregistration,
    config: TargetPopulationConfig,
    manifest: TargetPopulationManifest,
    role_master_entropy: bytes | str,
    confirmation_authorization: ConfirmationAuthorization | None = None,
) -> TargetMaterializationAuthorization:
    """Authorize materialization after verifying the complete seed manifest.

    Confirmatory populations additionally require the WP15 final-seal
    authorization bound to this exact manifest checksum. As specified in the
    preregistration, this token prevents accidental in-process access and is not
    a cryptographic substitute for external custody.

    Args:
        preregistration: Trusted WP15 protocol.
        config: Exact target-population configuration.
        manifest: Revealed externally custodied seed manifest.
        role_master_entropy: Revealed exact external 32-byte role master key.
        confirmation_authorization: Required WP15 token for confirmatory data.

    Returns:
        An opaque token bound to the exact config and manifest.

    Raises:
        TypeError: If an input or required confirmation token has the wrong type.
        ValueError: If a seal, role, seed, parameter, or authorization differs.
    """
    _require_trusted_preregistration(preregistration)
    _require_runtime(preregistration)
    if not isinstance(config, TargetPopulationConfig):
        msg = "config must be a TargetPopulationConfig."
        raise TypeError(msg)
    if not isinstance(manifest, TargetPopulationManifest):
        msg = "manifest must be a TargetPopulationManifest."
        raise TypeError(msg)
    if (
        config.preregistration_checksum != preregistration.content_checksum
        or manifest.preregistration_checksum != preregistration.content_checksum
        or manifest.population_config_checksum != config.content_checksum
        or manifest.data_role != config.data_role
        or manifest.population_scope != config.population_scope
        or manifest.role_master_entropy_commitment != config.role_master_entropy_commitment
        or manifest.allocations != config.allocations
    ):
        msg = "Target config, manifest, preregistration, or data role does not agree."
        raise ValueError(msg)
    if config.data_role == "confirmatory":
        if not isinstance(confirmation_authorization, ConfirmationAuthorization):
            msg = "Confirmatory target materialization requires a WP15 ConfirmationAuthorization."
            raise TypeError(msg)
        if (
            confirmation_authorization.preregistration_checksum != preregistration.content_checksum
            or confirmation_authorization.target_manifest_checksum != manifest.content_checksum
        ):
            msg = "Confirmation authorization is not bound to this preregistration and target manifest."
            raise ValueError(msg)
    elif confirmation_authorization is not None:
        msg = "A confirmation authorization cannot be reused for a nonconfirmatory target population."
        raise ValueError(msg)
    regenerated = create_target_population_manifest(config, preregistration, role_master_entropy)
    if regenerated.content_checksum != manifest.content_checksum:
        msg = "Target manifest is not the exact deterministic result of its config and role master entropy."
        raise ValueError(msg)
    return TargetMaterializationAuthorization(
        preregistration_checksum=preregistration.content_checksum,
        population_config_checksum=config.content_checksum,
        target_manifest_checksum=manifest.content_checksum,
        data_role=config.data_role,
        _marker=_MATERIALIZATION_SENTINEL,
    )


def _vector_checksum(vector: NDArray[np.complex128]) -> str:
    """Return the SHA-256 checksum of canonical little-endian complex128 bytes."""
    canonical = np.ascontiguousarray(vector, dtype=np.dtype("<c16"))
    return f"sha256:{hashlib.sha256(canonical.tobytes(order='C')).hexdigest()}"


@dataclass(frozen=True, slots=True, init=False)
class MaterializedTarget:
    """One immutable authorized target vector and its complete provenance."""

    target_instance_id: str
    target_instance_spec_checksum: str
    population_config_checksum: str
    target_manifest_checksum: str
    parameter_checksum: str
    family_id: str
    stratum_id: str
    qubit_count: int
    norm: float
    vector_checksum: str
    _state_vector_bytes: bytes = field(repr=False)

    def __init__(
        self,
        spec: TargetInstanceSpec,
        target_manifest_checksum: str,
        vector: NDArray[np.complex128],
        *,
        _marker: object,
    ) -> None:
        """Validate and freeze one authorized materialized vector.

        Raises:
            TypeError: If the spec or vector has the wrong type.
            ValueError: If authorization, shape, norm, or phase differs.
        """
        if _marker is not _MATERIALIZATION_SENTINEL:
            msg = "MaterializedTarget records may only be created by the authorized materializer."
            raise ValueError(msg)
        if not isinstance(spec, TargetInstanceSpec):
            msg = "spec must be a TargetInstanceSpec."
            raise TypeError(msg)
        manifest_checksum = require_checksum(target_manifest_checksum, "target_manifest_checksum")
        if not isinstance(vector, np.ndarray) or vector.dtype != np.dtype(np.complex128):
            msg = "vector must be an exact complex128 NumPy array."
            raise TypeError(msg)
        if vector.shape != (2**spec.qubit_count,) or not np.all(np.isfinite(vector)):
            msg = "Materialized target vector has an invalid shape or nonfinite amplitude."
            raise ValueError(msg)
        normalized = cast(
            "NDArray[np.complex128]",
            np.ascontiguousarray(vector, dtype=np.dtype("<c16")),
        )
        norm = float(np.linalg.norm(normalized))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-12):
            msg = f"Materialized target norm {norm} differs from one."
            raise ValueError(msg)
        pivot = normalized[int(np.argmax(np.abs(normalized)))]
        if pivot.real <= 0.0 or abs(pivot.imag) > 1e-12:
            msg = "Materialized target violates the frozen global-phase convention."
            raise ValueError(msg)
        object.__setattr__(self, "target_instance_id", spec.target_instance_id)
        object.__setattr__(self, "target_instance_spec_checksum", spec.content_checksum)
        object.__setattr__(self, "population_config_checksum", spec.population_config_checksum)
        object.__setattr__(self, "target_manifest_checksum", manifest_checksum)
        object.__setattr__(self, "parameter_checksum", canonical_checksum(spec.parameters))
        object.__setattr__(self, "family_id", spec.family_id)
        object.__setattr__(self, "stratum_id", spec.stratum_id)
        object.__setattr__(self, "qubit_count", spec.qubit_count)
        object.__setattr__(self, "norm", norm)
        object.__setattr__(self, "vector_checksum", _vector_checksum(normalized))
        object.__setattr__(self, "_state_vector_bytes", normalized.tobytes(order="C"))

    def state_vector_copy(self) -> NDArray[np.complex128]:
        """Return a detached writable copy of the target vector."""
        return np.frombuffer(self._state_vector_bytes, dtype=np.dtype("<c16")).astype(np.complex128, copy=True)

    def identity_dict(self) -> dict[str, object]:
        """Return the vector/checksum agreement ledger entry."""
        return {
            "target_instance_id": self.target_instance_id,
            "target_instance_spec_checksum": self.target_instance_spec_checksum,
            "population_config_checksum": self.population_config_checksum,
            "target_manifest_checksum": self.target_manifest_checksum,
            "parameter_checksum": self.parameter_checksum,
            "family_id": self.family_id,
            "stratum_id": self.stratum_id,
            "qubit_count": self.qubit_count,
            "norm": self.norm,
            "vector_checksum": self.vector_checksum,
        }


@dataclass(frozen=True, slots=True, init=False)
class TargetPopulationMaterialization:
    """Checksum-sealed collection produced by one authorized materialization."""

    target_manifest_checksum: str
    targets: tuple[MaterializedTarget, ...]
    schema_version: str = field(default=TARGET_MATERIALIZATION_SCHEMA_VERSION, init=False)

    def __init__(
        self,
        manifest: TargetPopulationManifest,
        targets: tuple[MaterializedTarget, ...],
        *,
        _marker: object,
    ) -> None:
        """Validate and freeze complete ordered manifest/vector agreement.

        Raises:
            TypeError: If the manifest or targets have the wrong record type.
            ValueError: If construction is unauthorized or any ordered manifest link differs.
        """
        if _marker is not _MATERIALIZATION_SENTINEL:
            msg = "TargetPopulationMaterialization records may only be created by the authorized materializer."
            raise ValueError(msg)
        if not isinstance(manifest, TargetPopulationManifest):
            msg = "manifest must be a TargetPopulationManifest."
            raise TypeError(msg)
        manifest_checksum = manifest.content_checksum
        if not isinstance(targets, tuple) or not targets:
            msg = "targets must be a nonempty tuple of MaterializedTarget records."
            raise TypeError(msg)
        if any(not isinstance(target, MaterializedTarget) for target in targets):
            msg = "targets must contain only MaterializedTarget records."
            raise TypeError(msg)
        expected_bindings = tuple(
            (
                spec.target_instance_id,
                spec.content_checksum,
                spec.population_config_checksum,
                manifest_checksum,
                canonical_checksum(spec.parameters),
                spec.family_id,
                spec.stratum_id,
                spec.qubit_count,
            )
            for spec in manifest.instances
        )
        actual_bindings = tuple(
            (
                target.target_instance_id,
                target.target_instance_spec_checksum,
                target.population_config_checksum,
                target.target_manifest_checksum,
                target.parameter_checksum,
                target.family_id,
                target.stratum_id,
                target.qubit_count,
            )
            for target in targets
        )
        if actual_bindings != expected_bindings:
            msg = "Materialized targets must match the manifest's exact ordered instance specifications."
            raise ValueError(msg)
        object.__setattr__(self, "schema_version", TARGET_MATERIALIZATION_SCHEMA_VERSION)
        object.__setattr__(self, "target_manifest_checksum", manifest_checksum)
        object.__setattr__(self, "targets", targets)

    @property
    def content_checksum(self) -> str:
        """Checksum of the complete manifest-to-vector ledger."""
        return canonical_checksum({
            "schema_version": self.schema_version,
            "target_manifest_checksum": self.target_manifest_checksum,
            "targets": [target.identity_dict() for target in self.targets],
        })

    def target(self, target_instance_id: str) -> MaterializedTarget:
        """Return one target by exact Phase II identifier.

        Args:
            target_instance_id: Full target instance identifier.

        Returns:
            The corresponding materialized target.

        Raises:
            KeyError: If the identifier is absent.
        """
        target_id = require_string(target_instance_id, "target_instance_id")
        for target in self.targets:
            if target.target_instance_id == target_id:
                return target
        raise KeyError(target_id)


def materialize_target_population(
    config: TargetPopulationConfig,
    preregistration: InitialPreregistration,
    manifest: TargetPopulationManifest,
    role_master_entropy: bytes | str,
    authorization: TargetMaterializationAuthorization,
) -> TargetPopulationMaterialization:
    """Materialize every vector after exact authorization and manifest checks.

    Args:
        config: Exact role-specific target configuration.
        preregistration: Trusted WP15 protocol.
        manifest: Revealed seed-bearing manifest.
        role_master_entropy: Revealed external 32-byte role master key.
        authorization: Token bound to this exact manifest.

    Returns:
        Immutable materialized vectors with a checksum agreement ledger.

    Raises:
        TypeError: If a config, manifest, or authorization has the wrong type.
        ValueError: If any authorization, seed, parameter, or checksum differs.
    """
    _require_trusted_preregistration(preregistration)
    _require_runtime(preregistration)
    if not isinstance(config, TargetPopulationConfig):
        msg = "config must be a TargetPopulationConfig."
        raise TypeError(msg)
    if not isinstance(manifest, TargetPopulationManifest):
        msg = "manifest must be a TargetPopulationManifest."
        raise TypeError(msg)
    if not isinstance(authorization, TargetMaterializationAuthorization):
        msg = "authorization must be a TargetMaterializationAuthorization."
        raise TypeError(msg)
    if (
        authorization.preregistration_checksum != preregistration.content_checksum
        or authorization.population_config_checksum != config.content_checksum
        or authorization.target_manifest_checksum != manifest.content_checksum
        or authorization.data_role != config.data_role
    ):
        msg = "Target materialization authorization is not bound to the supplied inputs."
        raise ValueError(msg)
    regenerated = create_target_population_manifest(config, preregistration, role_master_entropy)
    if regenerated.content_checksum != manifest.content_checksum:
        msg = "The target manifest changed after materialization authorization."
        raise ValueError(msg)
    master = _master_bytes(role_master_entropy)
    numeric_policy = cast("Mapping[str, object]", preregistration.target_population_policy["numeric_policy"])
    spectrum_agreement_rtol = cast("float", numeric_policy["spectrum_agreement_rtol"])
    spectrum_agreement_atol = cast("float", numeric_policy["spectrum_agreement_atol"])
    targets = tuple(
        MaterializedTarget(
            spec,
            manifest.content_checksum,
            _materialize_vector(
                master,
                spec,
                spectrum_agreement_rtol=spectrum_agreement_rtol,
                spectrum_agreement_atol=spectrum_agreement_atol,
            ),
            _marker=_MATERIALIZATION_SENTINEL,
        )
        for spec in manifest.instances
    )
    return TargetPopulationMaterialization(manifest, targets, _marker=_MATERIALIZATION_SENTINEL)


def verify_screening_target_population(
    screening_manifest: ScreeningManifest,
    target_manifest: TargetPopulationManifest,
) -> None:
    """Bind the WP15 primary screening universe to one WP16 target manifest.

    Args:
        screening_manifest: WP15 candidate/cell screening universe.
        target_manifest: WP16 primary-q6 screening target population.

    Raises:
        TypeError: If either object has the wrong versioned record type.
        ValueError: If checksums, roles, targets, metadata, or repetitions differ.
    """
    if not isinstance(screening_manifest, ScreeningManifest):
        msg = "screening_manifest must be a ScreeningManifest."
        raise TypeError(msg)
    if not isinstance(target_manifest, TargetPopulationManifest):
        msg = "target_manifest must be a TargetPopulationManifest."
        raise TypeError(msg)
    if screening_manifest.preregistration_checksum != target_manifest.preregistration_checksum:
        msg = "WP15 and WP16 screening manifests reference different preregistrations."
        raise ValueError(msg)
    if screening_manifest.screening_target_manifest_checksum != target_manifest.content_checksum:
        msg = "WP15 screening cells are not sealed to this exact WP16 target manifest."
        raise ValueError(msg)
    if target_manifest.data_role != "screening_selection" or target_manifest.population_scope != "primary_q6":
        msg = "WP15 promotion screening must use a screening_selection primary_q6 target manifest."
        raise ValueError(msg)
    expected = {
        spec.target_instance_id: (spec.family_id, spec.stratum_id, spec.qubit_count)
        for spec in target_manifest.instances
    }
    actual: dict[str, tuple[str, str, int]] = {}
    optimization_seeds: dict[str, set[int]] = {}
    target_seed_pairs: set[tuple[str, int]] = set()
    for cell in screening_manifest.cells:
        identity = (cell.family_id, cell.stratum_id, cell.qubit_count)
        previous = actual.setdefault(cell.target_instance_id, identity)
        if previous != identity:
            msg = f"WP15 target {cell.target_instance_id!r} has inconsistent family/stratum/qubit metadata."
            raise ValueError(msg)
        pair = (cell.target_instance_id, cell.optimization_seed)
        if pair in target_seed_pairs:
            msg = f"WP15 target/optimization-seed pair {pair!r} is duplicated."
            raise ValueError(msg)
        target_seed_pairs.add(pair)
        optimization_seeds.setdefault(cell.target_instance_id, set()).add(cell.optimization_seed)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        msg = f"WP15/WP16 screening target sets differ: missing={missing!r}, extra={extra!r}."
        raise ValueError(msg)
    if any(len(seeds) != 3 for seeds in optimization_seeds.values()):
        msg = "Every WP16 screening target must resolve to exactly three WP15 optimization-seed cells."
        raise ValueError(msg)


__all__ = [
    "PHASE_II_POPULATION_ID_PREFIX",
    "PHASE_II_TARGET_ID_PREFIX",
    "PHASE_II_TARGET_NAMESPACE",
    "TARGET_GENERATOR_SCHEMA_VERSION",
    "TARGET_INSTANCE_ROLES",
    "TARGET_INSTANCE_SPEC_SCHEMA_VERSION",
    "TARGET_MATERIALIZATION_SCHEMA_VERSION",
    "TARGET_POPULATION_COMMITMENT_SCHEMA_VERSION",
    "TARGET_POPULATION_CONFIG_SCHEMA_VERSION",
    "TARGET_POPULATION_MANIFEST_SCHEMA_VERSION",
    "TARGET_POPULATION_SCOPES",
    "TRUSTED_TARGET_ALLOCATION_POLICY_CHECKSUM",
    "TRUSTED_TARGET_NUMERIC_POLICY_CHECKSUM",
    "TRUSTED_TARGET_POPULATION_POLICY_CHECKSUM",
    "TRUSTED_TARGET_RNG_POLICY_CHECKSUM",
    "MaterializedTarget",
    "TargetAllocation",
    "TargetInstanceSpec",
    "TargetMaterializationAuthorization",
    "TargetPopulationCommitment",
    "TargetPopulationConfig",
    "TargetPopulationManifest",
    "TargetPopulationMaterialization",
    "authorize_target_materialization",
    "build_target_population_config",
    "create_target_population_manifest",
    "materialize_target_population",
    "role_master_entropy_commitment",
    "verify_screening_target_population",
]
