# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Bounded end-to-end custody acceptance tests for the WP22H ceremony."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import pytest

from benchmarks.state_preparation.phase2 import operational_ceremony_runner as runner
from benchmarks.state_preparation.phase2.canonical import canonical_checksum
from benchmarks.state_preparation.phase2.ceremony_store import (
    CeremonyBundleMember,
    ReopenedCeremonyBundle,
    read_ceremony_bundle_member,
    reopen_ceremony_bundle,
)
from benchmarks.state_preparation.phase2.operational_ceremony import WP22HReadinessReceipt

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class _BoundedCeremony:
    """Authenticated outputs from one generic, non-numerical four-stage chain."""

    ceremony_root: Path
    close_screen_paths: runner.WP22HOperationalPaths
    readiness: WP22HReadinessReceipt
    ready_receipt: runner.WP22HStageRunReceipt
    chain: tuple[ReopenedCeremonyBundle, ...]


def _checksum(label: str) -> str:
    """Return one stable valid checksum."""
    return canonical_checksum({"wp22h_chain_acceptance": label})


def _members_for(
    stage: runner.WP22HCeremonyStage,
    replacements: dict[str, bytes],
) -> tuple[CeremonyBundleMember, ...]:
    """Return the fixed stage shape with selected typed artifact bytes."""
    shape = runner._STAGE_SPEC[stage][2]  # noqa: SLF001 -- fixed ceremony contract
    return tuple(
        CeremonyBundleMember(
            relative_path,
            role,
            replacements.get(relative_path, f"{relative_path}:{role}\n".encode()),
        )
        for relative_path, role in sorted(shape)
    )


def _artifact_bytes(artifact: runner.WP22HOperationalPaths | WP22HReadinessReceipt) -> bytes:
    """Return the ceremony's canonical newline-terminated artifact encoding."""
    return f"{artifact.to_json()}\n".encode()


def _publish(
    ceremony_root: Path,
    stage: runner.WP22HCeremonyStage,
    members: tuple[CeremonyBundleMember, ...],
    predecessor: ReopenedCeremonyBundle | None,
) -> tuple[runner.WP22HStageRunReceipt, ReopenedCeremonyBundle]:
    """Publish and externally reopen one bounded fixed-shape stage.

    Returns:
        The externally retained run receipt and authenticated bundle.
    """
    receipt = runner._publish_stage(  # noqa: SLF001 -- focused custody acceptance
        ceremony_root,
        stage,
        members,
        None if predecessor is None else predecessor.manifest,
    )
    reopened = reopen_ceremony_bundle(
        receipt.bundle_directory,
        expected_index_checksum=receipt.bundle_index_checksum,
        expected_stage_manifest_checksum=receipt.stage_manifest_checksum,
    )
    return receipt, reopened


def _readiness_receipt(
    pre_seal_head_checksum: str,
    close_screen_paths_checksum: str,
) -> WP22HReadinessReceipt:
    """Return a strict dormant receipt bound to both ceremony custody roots."""
    return WP22HReadinessReceipt(
        source_commit="1" * 40,
        preregistration_checksum=_checksum("preregistration"),
        execution_source_manifest_checksum=_checksum("execution source"),
        analysis_source_manifest_checksum=_checksum("analysis source"),
        pilot_plan_checksum=_checksum("pilot plan"),
        pilot_primary_target_manifest_checksum=_checksum("pilot primary targets"),
        pilot_secondary_target_manifest_checksum=_checksum("pilot secondary targets"),
        pilot_custody_checksum=_checksum("pilot custody"),
        pilot_secondary_archive_checksum=_checksum("pilot secondary archive"),
        pilot_nuisance_summary_checksum=_checksum("pilot nuisance summary"),
        sample_size_design_checksum=_checksum("sample size design"),
        pilot_calibration_checksum=_checksum("pilot calibration"),
        screening_plan_checksum=_checksum("screening plan"),
        screening_target_manifest_checksum=_checksum("screening targets"),
        screening_manifest_checksum=_checksum("screening manifest"),
        screening_custody_checksum=_checksum("screening custody"),
        screening_evidence_checksum=_checksum("screening evidence"),
        promotion_decision_checksum=_checksum("promotion decision"),
        resource_calibration_checksum=_checksum("resource calibration"),
        configuration_execution_manifest_checksum=_checksum("configuration execution manifest"),
        paper_screen_binding_catalog_checksum=_checksum("screen execution catalog"),
        confirmatory_target_configuration_checksum=_checksum("confirmatory target config"),
        confirmatory_target_commitment_checksum=_checksum("confirmatory target commitment"),
        final_confirmation_seal_checksum=_checksum("final confirmation seal"),
        prior_target_exposure_inventory_checksum=_checksum("prior target exposure"),
        pre_seal_chain_head_stage_manifest_checksum=pre_seal_head_checksum,
        close_screen_operational_paths_checksum=close_screen_paths_checksum,
        confirmatory_configuration_count=2,
        confirmatory_target_count=96,
        confirmatory_optimization_seed_count=3,
        confirmatory_job_count=576,
    )


def _publish_bounded_ceremony(tmp_path: Path) -> _BoundedCeremony:
    """Publish a four-stage generic store chain without scientific execution.

    Returns:
        The bounded typed artifacts, terminal receipt, and reopened chain.
    """
    repository_root = (tmp_path / "repository").absolute()
    ceremony_root = (tmp_path / "ceremony").absolute()
    pilot_output_root = (tmp_path / "pilot-output").absolute()
    screen_output_root = (tmp_path / "screen-output").absolute()
    repository_root.mkdir()
    ceremony_root.mkdir()

    prepare_paths = runner.WP22HOperationalPaths(repository_root, ceremony_root)
    _, prepare = _publish(
        ceremony_root,
        runner.WP22HCeremonyStage.PREPARE_PILOT,
        _members_for(
            runner.WP22HCeremonyStage.PREPARE_PILOT,
            {"operational/paths.json": _artifact_bytes(prepare_paths)},
        ),
        None,
    )

    pilot_paths = runner.WP22HOperationalPaths(repository_root, ceremony_root, pilot_output_root)
    _, pilot = _publish(
        ceremony_root,
        runner.WP22HCeremonyStage.CLOSE_PILOT_PREPARE_SCREEN,
        _members_for(
            runner.WP22HCeremonyStage.CLOSE_PILOT_PREPARE_SCREEN,
            {"operational/paths.json": _artifact_bytes(pilot_paths)},
        ),
        prepare,
    )

    close_screen_paths = runner.WP22HOperationalPaths(
        repository_root,
        ceremony_root,
        pilot_output_root,
        screen_output_root,
    )
    _, screen = _publish(
        ceremony_root,
        runner.WP22HCeremonyStage.CLOSE_SCREEN_SEAL,
        _members_for(
            runner.WP22HCeremonyStage.CLOSE_SCREEN_SEAL,
            {"operational/paths.json": _artifact_bytes(close_screen_paths)},
        ),
        pilot,
    )

    readiness = _readiness_receipt(
        pilot.manifest.content_checksum,
        close_screen_paths.content_checksum,
    )
    ready_receipt, _ = _publish(
        ceremony_root,
        runner.WP22HCeremonyStage.VERIFY_READY,
        _members_for(
            runner.WP22HCeremonyStage.VERIFY_READY,
            {"readiness/receipt.json": _artifact_bytes(readiness)},
        ),
        screen,
    )
    chain = runner._reopen_chain(  # noqa: SLF001 -- bounded chain acceptance
        ceremony_root,
        runner.WP22HCeremonyStage.VERIFY_READY,
        ready_receipt.bundle_index_checksum,
    )
    return _BoundedCeremony(ceremony_root, close_screen_paths, readiness, ready_receipt, chain)


def test_bounded_chain_embeds_two_part_readiness_custody(tmp_path: Path) -> None:
    """Stage three closes exact stage-one and stage-two custody without execution."""
    bounded = _publish_bounded_ceremony(tmp_path)
    prepare, pilot, screen, ready = bounded.chain

    assert tuple(bundle.manifest.stage_ordinal for bundle in bounded.chain) == (0, 1, 2, 3)
    assert pilot.manifest.predecessor_stage_manifest_checksum == prepare.manifest.content_checksum
    assert screen.manifest.predecessor_stage_manifest_checksum == pilot.manifest.content_checksum
    assert ready.manifest.predecessor_stage_manifest_checksum == screen.manifest.content_checksum

    stored_bytes = read_ceremony_bundle_member(ready, "readiness/receipt.json")
    assert stored_bytes == _artifact_bytes(bounded.readiness)
    stored_readiness = WP22HReadinessReceipt.from_json(stored_bytes.decode())
    assert stored_readiness.pre_seal_chain_head_stage_manifest_checksum == pilot.manifest.content_checksum
    assert stored_readiness.close_screen_operational_paths_checksum == bounded.close_screen_paths.content_checksum
    assert stored_readiness == bounded.readiness

    receipt = bounded.ready_receipt
    assert receipt.stage is runner.WP22HCeremonyStage.VERIFY_READY
    assert receipt.bundle_directory == bounded.ceremony_root / "03-verify-ready"
    assert receipt.stage_manifest_checksum == ready.manifest.content_checksum
    assert receipt.bundle_index_checksum == ready.index.content_checksum
    assert receipt.predecessor_stage_manifest_checksum == screen.manifest.content_checksum


def test_bounded_chain_rejects_readiness_and_external_head_tamper(tmp_path: Path) -> None:
    """A resealed custody-field edit or a foreign retained index cannot reopen."""
    bounded = _publish_bounded_ceremony(tmp_path)
    foreign_head = replace(
        bounded.ready_receipt,
        bundle_index_checksum=_checksum("foreign stage-three bundle index"),
    )
    with pytest.raises(ValueError, match="expected terminal custody"):
        runner._reopen_chain(  # noqa: SLF001 -- retained-head rejection
            bounded.ceremony_root,
            runner.WP22HCeremonyStage.VERIFY_READY,
            foreign_head.bundle_index_checksum,
        )

    tampered_readiness = replace(
        bounded.readiness,
        pre_seal_chain_head_stage_manifest_checksum=_checksum("foreign pre-seal chain head"),
    )
    readiness_path = bounded.ready_receipt.bundle_directory / "readiness/receipt.json"
    readiness_path.write_bytes(_artifact_bytes(tampered_readiness))
    with pytest.raises(ValueError, match="differs from its immutable receipt"):
        runner._reopen_chain(  # noqa: SLF001 -- immutable-member rejection
            bounded.ceremony_root,
            runner.WP22HCeremonyStage.VERIFY_READY,
            bounded.ready_receipt.bundle_index_checksum,
        )
