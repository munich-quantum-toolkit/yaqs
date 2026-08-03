# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Profile-complete executable binding catalogs for WP22B.

WP22A profiles bind publication candidates to typed implementation artifacts.
The WP22B implementation catalog independently identifies the repository code
that can execute those artifacts.  This module closes both registries without
weakening their separate identities: every profile member is linked to the one
matching implementation entry and to an adapter re-derived from its own sealed
artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from .canonical import (
    canonical_checksum,
    canonical_json,
    load_canonical_json_object,
    verify_sealed_mapping,
)
from .execution_bindings import (
    Preset,
    ScopedImplementationBinding,
    TargetScope,
    TrainingExecutionProfile,
)
from .implementation_catalog import (
    ExecutableImplementationEntry,
    RepositoryImplementationCatalog,
    RepositoryRunnerAdapter,
    SmokeRuntimeProgram,
)
from .validation import require_checksum, require_slug

if TYPE_CHECKING:
    from collections.abc import Callable

EXECUTABLE_SCOPED_BINDING_SCHEMA_VERSION = "yaqs.state_preparation.phase2.executable_scoped_binding.v1"
REPOSITORY_BINDING_CATALOG_SCHEMA_VERSION = "yaqs.state_preparation.phase2.repository_binding_catalog.v1"

_LINK_KEYS = frozenset({
    "schema_version",
    "binding",
    "binding_checksum",
    "implementation_entry",
    "implementation_entry_checksum",
    "runner_adapter",
    "runner_adapter_checksum",
    "content_checksum",
})
_CATALOG_KEYS = frozenset({
    "schema_version",
    "catalog_id",
    "profile",
    "profile_checksum",
    "implementation_catalog",
    "implementation_catalog_checksum",
    "bindings",
    "binding_count",
    "paper_confirm_execution_authorized",
    "content_checksum",
})


def _sealed(payload: dict[str, object]) -> dict[str, object]:
    """Return a detached checksum-sealed mapping."""
    return {**payload, "content_checksum": canonical_checksum(payload)}


@dataclass(frozen=True, slots=True)
class ExecutableScopedBinding:
    """One exact profile binding closed to its repository runner route."""

    binding: ScopedImplementationBinding
    implementation_entry: ExecutableImplementationEntry
    runner_adapter: RepositoryRunnerAdapter
    schema_version: str = field(default=EXECUTABLE_SCOPED_BINDING_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require exact schedule, payload, entry, and adapter closure.

        Raises:
            TypeError: If a nested object has the wrong record type.
            ValueError: If any profile-to-repository identity link differs.
        """
        if not isinstance(self.binding, ScopedImplementationBinding):
            msg = "binding must be a ScopedImplementationBinding."
            raise TypeError(msg)
        if not isinstance(self.implementation_entry, ExecutableImplementationEntry):
            msg = "implementation_entry must be an ExecutableImplementationEntry."
            raise TypeError(msg)
        if not isinstance(self.runner_adapter, RepositoryRunnerAdapter):
            msg = "runner_adapter must be a RepositoryRunnerAdapter."
            raise TypeError(msg)

        binding = self.binding
        entry = self.implementation_entry
        if (
            binding.preset != entry.preset
            or binding.publication_method_id != entry.publication_method_id
            or binding.target_scope_id != entry.target_scope_id
        ):
            msg = "Profile binding and implementation entry preset, method, or target scope differ."
            raise ValueError(msg)
        if (
            binding.strategy_schedule != entry.strategy_schedule
            or binding.strategy_schedule.content_checksum != entry.strategy_schedule.content_checksum
        ):
            msg = "Profile binding and implementation entry require the exact same strategy schedule."
            raise ValueError(msg)

        binding_artifact = binding.implementation_artifact
        entry_artifact = entry.implementation_artifact
        if (
            binding_artifact.implementation_kind != entry_artifact.implementation_kind
            or binding_artifact.implementation_method_id != entry_artifact.implementation_method_id
        ):
            msg = "Profile binding and implementation entry payload kind or implementation method differ."
            raise ValueError(msg)
        if (
            binding_artifact.implementation_payload_checksum != entry_artifact.implementation_payload_checksum
            or binding_artifact.implementation_payload != entry_artifact.implementation_payload
        ):
            msg = "Profile binding and implementation entry require the exact same typed payload."
            raise ValueError(msg)

        expected_adapter = RepositoryRunnerAdapter.for_artifact(binding_artifact)
        if self.runner_adapter != expected_adapter or self.runner_adapter != entry.runner_adapter:
            msg = "Runner adapter was not re-derived from the exact binding and implementation entry."
            raise ValueError(msg)
        binding_runner = self.runner_adapter.resolve_callable()
        entry_runner = entry.resolve_callable()
        if binding_runner is not entry_runner:
            msg = "Profile binding and implementation entry resolve different repository callables."
            raise ValueError(msg)

    @classmethod
    def close(
        cls,
        binding: ScopedImplementationBinding,
        implementation_entry: ExecutableImplementationEntry,
    ) -> ExecutableScopedBinding:
        """Close one profile binding to a matching implementation entry.

        Returns:
            The immutable binding-to-runner link.

        Raises:
            TypeError: If ``binding`` is not a scoped implementation binding.
        """
        if not isinstance(binding, ScopedImplementationBinding):
            msg = "binding must be a ScopedImplementationBinding."
            raise TypeError(msg)
        return cls(
            binding=binding,
            implementation_entry=implementation_entry,
            runner_adapter=RepositoryRunnerAdapter.for_artifact(binding.implementation_artifact),
        )

    @property
    def key(self) -> tuple[Preset, str, TargetScope]:
        """Unique profile lookup key retained by this executable link."""
        return self.binding.key

    def resolve_callable(self) -> Callable[..., object]:
        """Resolve the exact repository runner after rechecking entry closure.

        Returns:
            The imported repository runner class or function.

        Raises:
            RuntimeError: If the independently resolved entry route differs.
        """
        runner = self.runner_adapter.resolve_callable()
        if runner is not self.implementation_entry.resolve_callable():
            msg = "Executable binding no longer resolves the implementation entry's repository callable."
            raise RuntimeError(msg)
        return runner

    def smoke_runtime_program(self) -> SmokeRuntimeProgram:
        """Return the bounded runtime program for a training-smoke link.

        Returns:
            The checksum-bound one-update or one-growth smoke program.

        Raises:
            ValueError: If the link is not smoke or the derived programs differ.
        """
        if self.binding.preset != "training-smoke":
            msg = "Only training-smoke executable bindings have a smoke runtime program."
            raise ValueError(msg)
        binding_runtime = self.runner_adapter.materialize_smoke_runtime(self.binding.implementation_artifact)
        entry_runtime = self.implementation_entry.smoke_runtime_program()
        if binding_runtime != entry_runtime:
            msg = "Binding and implementation entry derive different smoke runtime programs."
            raise ValueError(msg)
        return binding_runtime

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered executable binding field."""
        return {
            "schema_version": self.schema_version,
            "binding": self.binding.to_dict(),
            "binding_checksum": self.binding.content_checksum,
            "implementation_entry": self.implementation_entry.to_dict(),
            "implementation_entry_checksum": self.implementation_entry.content_checksum,
            "runner_adapter": self.runner_adapter.to_dict(),
            "runner_adapter_checksum": self.runner_adapter.content_checksum,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the complete profile-to-runner link."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> ExecutableScopedBinding:
        """Decode and verify one executable scoped binding.

        Returns:
            The normalized executable profile-to-runner link.

        Raises:
            ValueError: If a nested checksum alias or normalized link differs.
        """
        mapping = verify_sealed_mapping(value, expected_keys=_LINK_KEYS, name="executable scoped binding")
        if mapping["schema_version"] != EXECUTABLE_SCOPED_BINDING_SCHEMA_VERSION:
            msg = "Executable scoped binding uses an unsupported schema version."
            raise ValueError(msg)
        linked = cls(
            binding=ScopedImplementationBinding.from_dict(mapping["binding"]),
            implementation_entry=ExecutableImplementationEntry.from_dict(mapping["implementation_entry"]),
            runner_adapter=RepositoryRunnerAdapter.from_dict(mapping["runner_adapter"]),
        )
        aliases = {
            "binding_checksum": linked.binding.content_checksum,
            "implementation_entry_checksum": linked.implementation_entry.content_checksum,
            "runner_adapter_checksum": linked.runner_adapter.content_checksum,
            "content_checksum": linked.content_checksum,
        }
        if any(mapping[name] != expected for name, expected in aliases.items()):
            msg = "Executable scoped binding checksum aliases changed during normalization."
            raise ValueError(msg)
        return linked

    @classmethod
    def from_json(cls, payload: str) -> ExecutableScopedBinding:
        """Decode canonical JSON into a verified executable scoped binding.

        Returns:
            The normalized executable profile-to-runner link.
        """
        return cls.from_dict(load_canonical_json_object(payload))


@dataclass(frozen=True, slots=True)
class RepositoryBindingCatalog:
    """Complete executable closure for exactly one WP22A execution profile."""

    catalog_id: str
    profile: TrainingExecutionProfile
    implementation_catalog: RepositoryImplementationCatalog
    bindings: tuple[ExecutableScopedBinding, ...]
    schema_version: str = field(default=REPOSITORY_BINDING_CATALOG_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        """Require complete, ordered, unique profile and implementation closure.

        Raises:
            TypeError: If a nested registry or binding has the wrong type.
            ValueError: If a member is missing, duplicated, foreign, or forged.
        """
        object.__setattr__(self, "catalog_id", require_slug(self.catalog_id, "catalog_id"))
        if not isinstance(self.profile, TrainingExecutionProfile):
            msg = "profile must be a TrainingExecutionProfile."
            raise TypeError(msg)
        if not isinstance(self.implementation_catalog, RepositoryImplementationCatalog):
            msg = "implementation_catalog must be a RepositoryImplementationCatalog."
            raise TypeError(msg)
        links = tuple(self.bindings)
        if not links or any(not isinstance(link, ExecutableScopedBinding) for link in links):
            msg = "bindings must contain ExecutableScopedBinding records."
            raise TypeError(msg)
        keys = tuple(link.key for link in links)
        if len(keys) != len(set(keys)):
            msg = "Executable binding keys must be unique within one profile catalog."
            raise ValueError(msg)
        expected = tuple(
            ExecutableScopedBinding.close(
                binding,
                self.implementation_catalog.resolve(
                    binding.preset,
                    binding.publication_method_id,
                    binding.target_scope_id,
                ),
            )
            for binding in self.profile.bindings
        )
        if links != expected:
            msg = "Executable bindings must exactly and in order close every member of their single profile."
            raise ValueError(msg)
        object.__setattr__(self, "bindings", links)

    @classmethod
    def from_profile(
        cls,
        profile: TrainingExecutionProfile,
        implementation_catalog: RepositoryImplementationCatalog,
    ) -> RepositoryBindingCatalog:
        """Close a complete WP22A profile to the WP22B repository catalog.

        Returns:
            A deterministic executable catalog containing every profile member.

        Raises:
            TypeError: If either source registry has the wrong type.
        """
        if not isinstance(profile, TrainingExecutionProfile):
            msg = "profile must be a TrainingExecutionProfile."
            raise TypeError(msg)
        if not isinstance(implementation_catalog, RepositoryImplementationCatalog):
            msg = "implementation_catalog must be a RepositoryImplementationCatalog."
            raise TypeError(msg)
        links = tuple(
            ExecutableScopedBinding.close(
                binding,
                implementation_catalog.resolve(
                    binding.preset,
                    binding.publication_method_id,
                    binding.target_scope_id,
                ),
            )
            for binding in profile.bindings
        )
        return cls(
            catalog_id=f"wp22b_{profile.profile_id}_binding_catalog",
            profile=profile,
            implementation_catalog=implementation_catalog,
            bindings=links,
        )

    def resolve(
        self,
        preset: Preset,
        publication_candidate_checksum: str,
        target_scope_id: TargetScope,
    ) -> ExecutableScopedBinding:
        """Resolve one complete executable binding by the WP22A profile key.

        Returns:
            The unique profile member and its verified repository route.

        Raises:
            KeyError: If the key is outside this exact profile, including confirmation.
            ValueError: If the checksum or target scope has invalid syntax.
        """
        requested_preset = require_slug(preset, "preset")
        candidate_checksum = require_checksum(
            publication_candidate_checksum,
            "publication_candidate_checksum",
        )
        if target_scope_id not in {"primary_q6", "secondary_q12"}:
            msg = "target_scope_id must be primary_q6 or secondary_q12."
            raise ValueError(msg)
        key = (requested_preset, candidate_checksum, target_scope_id)
        matches = tuple(link for link in self.bindings if link.key == key)
        if len(matches) != 1:
            msg = f"No executable scoped binding exists for profile key {key!r}."
            raise KeyError(msg)
        return matches[0]

    def _payload(self) -> dict[str, object]:
        """Return every checksum-covered repository binding catalog field."""
        return {
            "schema_version": self.schema_version,
            "catalog_id": self.catalog_id,
            "profile": self.profile.to_dict(),
            "profile_checksum": self.profile.content_checksum,
            "implementation_catalog": self.implementation_catalog.to_dict(),
            "implementation_catalog_checksum": self.implementation_catalog.content_checksum,
            "bindings": [binding.to_dict() for binding in self.bindings],
            "binding_count": len(self.bindings),
            "paper_confirm_execution_authorized": False,
        }

    @property
    def content_checksum(self) -> str:
        """Checksum covering the profile, catalog, and all executable links."""
        return canonical_checksum(self._payload())

    def to_dict(self) -> dict[str, object]:
        """Return strict checksum-sealed JSON-native data."""
        return _sealed(self._payload())

    def to_json(self) -> str:
        """Return canonical checksum-sealed JSON."""
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> RepositoryBindingCatalog:
        """Decode and verify one complete repository binding catalog.

        Returns:
            The normalized executable profile catalog.

        Raises:
            TypeError: If the serialized binding collection is not an array.
            ValueError: If nested aliases, membership, or authorization differ.
        """
        mapping = verify_sealed_mapping(value, expected_keys=_CATALOG_KEYS, name="repository binding catalog")
        if mapping["schema_version"] != REPOSITORY_BINDING_CATALOG_SCHEMA_VERSION:
            msg = "Repository binding catalog uses an unsupported schema version."
            raise ValueError(msg)
        raw_bindings = mapping["bindings"]
        if type(raw_bindings) is not tuple:
            msg = "bindings must be a JSON array."
            raise TypeError(msg)
        catalog = cls(
            catalog_id=cast("str", mapping["catalog_id"]),
            profile=TrainingExecutionProfile.from_dict(mapping["profile"]),
            implementation_catalog=RepositoryImplementationCatalog.from_dict(mapping["implementation_catalog"]),
            bindings=tuple(ExecutableScopedBinding.from_dict(binding) for binding in raw_bindings),
        )
        aliases = {
            "profile_checksum": catalog.profile.content_checksum,
            "implementation_catalog_checksum": catalog.implementation_catalog.content_checksum,
            "binding_count": len(catalog.bindings),
            "paper_confirm_execution_authorized": False,
            "content_checksum": catalog.content_checksum,
        }
        if any(mapping[name] != expected for name, expected in aliases.items()):
            msg = "Repository binding catalog aliases or confirmation authorization changed."
            raise ValueError(msg)
        return catalog

    @classmethod
    def from_json(cls, payload: str) -> RepositoryBindingCatalog:
        """Decode canonical JSON into a verified repository binding catalog.

        Returns:
            The normalized executable profile catalog.
        """
        return cls.from_dict(load_canonical_json_object(payload))


__all__ = [
    "EXECUTABLE_SCOPED_BINDING_SCHEMA_VERSION",
    "REPOSITORY_BINDING_CATALOG_SCHEMA_VERSION",
    "ExecutableScopedBinding",
    "RepositoryBindingCatalog",
]
