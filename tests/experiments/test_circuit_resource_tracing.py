# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Focused tests for the circuit-benchmark resource tracer."""

from __future__ import annotations

import copy
import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import DigitalSimParams
from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.digital import digital_tjm

EXPERIMENT = Path(__file__).resolve().parents[2] / "experiments" / "circuit_benchmarks"
SPEC = importlib.util.spec_from_file_location("circuit_benchmark_tracing", EXPERIMENT / "tracing.py")
assert SPEC is not None
assert SPEC.loader is not None
tracing = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tracing
SPEC.loader.exec_module(tracing)

ResourceTracer = tracing.ResourceTracer
retained_mps_resources = tracing.retained_mps_resources


def _params(*, cap: int, gate_mode: str = "full-tdvp") -> DigitalSimParams:
    """Return tight deterministic parameters for small tracing tests."""
    return DigitalSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=cap,
        gate_mode=gate_mode,
        tdvp_sweeps=1,
        tdvp_mode="2site",
        trunc_mode="discarded_weight",
        svd_threshold=1e-14,
        krylov_tol=1e-12,
    )


def _assert_resource_identity(row: dict[str, object]) -> None:
    """Check the qubit-MPS parameter formula encoded by one trace row."""
    bonds = [int(value) for value in row["bond_dimensions"]]
    expected = 2 * sum(left * right for left, right in itertools.pairwise(bonds))
    assert row["parameter_count"] == expected
    assert row["peak_bond_dim"] == max(bonds)


def test_explicit_checkpoint_measures_full_mps() -> None:
    """Explicit checkpoints report the sum of all retained tensor sizes."""
    state = MPS(6, state="zeros", pad=3)
    expected = retained_mps_resources(state)
    tracer = ResourceTracer()
    row = tracer.checkpoint(state, "initial", model="test", method="none")
    assert (row["parameter_count"], row["peak_bond_dim"], row["bond_dimensions"]) == expected
    assert row["n_sites"] == state.length
    _assert_resource_identity(row)


def test_tebd_records_every_routed_split() -> None:
    """A separated TEBD gate records every forward, physical, and reverse split."""
    state = MPS(5, state="zeros")
    gate = GateLibrary.rxx([0.31])
    gate.set_sites(0, 4)

    with ResourceTracer() as tracer:
        with tracer.gate_scope(method="tebd_swap", step=1, gate_index=2, gate_name="rxx", sites=[0, 4]):
            digital_tjm.apply_two_qubit_gate_tebd(state, gate, _params(cap=4, gate_mode="swaps"))

    assert len(tracer.rows) == 7
    assert all(row["checkpoint"] == "tebd_split" for row in tracer.rows)
    assert [row["checkpoint_in_gate"] for row in tracer.rows] == list(range(7))
    assert all(row["n_sites"] == state.length for row in tracer.rows)
    assert all(row["method"] == "tebd_swap" for row in tracer.rows)
    assert sum(row["local_gate_name"] == "swap" for row in tracer.rows) == 6
    assert sum(row["local_gate_name"] == "rxx" for row in tracer.rows) == 1
    for row in tracer.rows:
        _assert_resource_identity(row)


def test_tdvp_splits_are_reconstructed_on_the_full_parent_chain() -> None:
    """Window-local TDVP checkpoints include the untouched parent exterior."""
    state = MPS(8, state="zeros", pad=2)
    gate = GateLibrary.rxx([0.17])
    gate.set_sites(2, 5)

    with (
        ResourceTracer() as tracer,
        tracer.gate_scope(method="gate_local_2tdvp", step=0, gate_index=0, gate_name="rxx", sites=[2, 5]),
    ):
        digital_tjm.apply_two_qubit_gate_tdvp(state, gate, _params(cap=4))

    # The gate-support window [2, 5] has four sites, so its symmetric 2TDVP
    # sweep performs 2*4-3 state-changing two-site SVDs.
    assert len(tracer.rows) == 5
    assert all(row["checkpoint"] == "tdvp_split" for row in tracer.rows)
    assert all(row["n_sites"] == state.length for row in tracer.rows)
    assert all(min(row["updated_sites"]) >= 2 and max(row["updated_sites"]) <= 5 for row in tracer.rows)
    assert all(len(row["bond_dimensions"]) == state.length + 1 for row in tracer.rows)
    for row in tracer.rows:
        _assert_resource_identity(row)


def test_mpo_records_contraction_and_each_prospective_compression_split() -> None:
    """MPO tracing includes the uncapped product and all compression outputs."""
    state = MPS(5, state="zeros")
    gate = GateLibrary.rxx([0.23])
    gate.set_sites(0, 4)

    with ResourceTracer() as tracer, tracer.gate_scope(
        method="mpo_contract_compress",
        step=0,
        gate_index=0,
        gate_name="rxx",
        sites=[0, 4],
    ):
        digital_tjm.apply_long_range_gate_mpo(state, gate, _params(cap=1, gate_mode="mpo"))

    assert [row["checkpoint"] for row in tracer.rows] == [
        "mpo_post_contraction",
        "mpo_compress_svd",
        "mpo_compress_svd",
        "mpo_compress_svd",
        "mpo_compress_svd",
    ]
    assert [row.get("updated_sites") for row in tracer.rows[1:]] == [[0, 1], [1, 2], [2, 3], [3, 4]]
    final_parameters, final_bond, final_profile = retained_mps_resources(state)
    assert tracer.rows[-1]["parameter_count"] == final_parameters
    assert tracer.rows[-1]["peak_bond_dim"] == final_bond
    assert tracer.rows[-1]["bond_dimensions"] == final_profile
    assert tracer.peak_parameter_count == tracer.rows[0]["parameter_count"]
    for row in tracer.rows:
        _assert_resource_identity(row)


def test_all_monkeypatches_are_restored_after_an_exception() -> None:
    """An exception inside a gate scope cannot leak process-global wrappers."""
    import mqt.yaqs.core.data_structures.mps as mps_module

    originals = {
        "apply_window": digital_tjm.apply_window,
        "tebd": digital_tjm.apply_two_qubit_gate_tebd,
        "update_center": MPS.update_center_after_split,
        "multiply_mps": MPO._multiply_mps,
        "compress": MPS.compress,
        "split": mps_module.split_two_site,
    }

    with pytest.raises(RuntimeError, match="sentinel"), ResourceTracer() as tracer:
        assert digital_tjm.apply_window is not originals["apply_window"]
        assert digital_tjm.apply_two_qubit_gate_tebd is not originals["tebd"]
        assert MPS.update_center_after_split is not originals["update_center"]
        assert MPO._multiply_mps is not originals["multiply_mps"]
        assert MPS.compress is not originals["compress"]
        assert mps_module.split_two_site is not originals["split"]
        with tracer.gate_scope(method="test"):
            msg = "sentinel"
            raise RuntimeError(msg)

    assert digital_tjm.apply_window is originals["apply_window"]
    assert digital_tjm.apply_two_qubit_gate_tebd is originals["tebd"]
    assert MPS.update_center_after_split is originals["update_center"]
    assert MPO._multiply_mps is originals["multiply_mps"]
    assert MPS.compress is originals["compress"]
    assert mps_module.split_two_site is originals["split"]


@pytest.mark.parametrize("mode", ["full-tdvp", "mpo", "swaps"])
def test_tracing_does_not_change_the_updated_state(mode: str) -> None:
    """Instrumentation leaves tensors and physical states unchanged."""
    reference = MPS(6, state="basis", basis_string="010100")
    traced = copy.deepcopy(reference)
    gate = GateLibrary.ryy([0.19])
    gate.set_sites(1, 4)
    params = _params(cap=4, gate_mode=mode)

    if mode == "full-tdvp":
        digital_tjm.apply_two_qubit_gate_tdvp(reference, gate, params)
        with ResourceTracer() as tracer, tracer.gate_scope(method=mode):
            digital_tjm.apply_two_qubit_gate_tdvp(traced, gate, params)
    elif mode == "mpo":
        digital_tjm.apply_long_range_gate_mpo(reference, gate, params)
        with ResourceTracer() as tracer, tracer.gate_scope(method=mode):
            digital_tjm.apply_long_range_gate_mpo(traced, gate, params)
    else:
        digital_tjm.apply_two_qubit_gate_tebd(reference, gate, params)
        with ResourceTracer() as tracer, tracer.gate_scope(method=mode):
            digital_tjm.apply_two_qubit_gate_tebd(traced, gate, params)

    assert [tensor.shape for tensor in traced.tensors] == [tensor.shape for tensor in reference.tensors]
    assert np.linalg.norm(traced.to_vec() - reference.to_vec()) < 1e-13
