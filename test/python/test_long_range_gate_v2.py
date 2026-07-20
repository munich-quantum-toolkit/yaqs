# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Regression tests for long-range gate benchmark v2 reference logic."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "long_range_gate_substeps"
sys.path.insert(0, str(EXPERIMENTS_DIR))

from benchmark_lib import (  # noqa: E402
    L_DEFAULT,
    TARGET_BOND_PROFILE,
    _params,
    apply_two_qubit_dense,
    fidelity,
    make_dag_node,
    make_gate,
    random_mps,
)
from plots_report import fit_angle_scaling  # noqa: E402
from sequence_gates import (  # noqa: E402
    build_exact_prefix_states,
    build_sequence_gate_specs,
    embed_dense_two_qubit_operator,
)

from mqt.yaqs.core.data_structures.mpo_utils import resolve_lr_tensor  # noqa: E402
from mqt.yaqs.digital.digital_tjm import apply_two_qubit_gate  # noqa: E402


@pytest.fixture
def seed11_state():
    rng = np.random.default_rng(11)
    mps = random_mps(L_DEFAULT, list(TARGET_BOND_PROFILE), rng)
    return mps, mps.to_vec()


@pytest.mark.parametrize("pair", [(2, 9), (9, 2), (2, 8), (3, 9)])
@pytest.mark.parametrize("gate_type", ["rxx", "ryy", "rzz"])
def test_dense_apply_matches_embedded_operator(seed11_state, pair, gate_type) -> None:
    _mps, psi = seed11_state
    gate = make_gate(gate_type, 0.05, pair[0], pair[1])
    left, right = min(pair), max(pair)
    u = resolve_lr_tensor(gate, left, right).reshape(4, 4)
    direct = apply_two_qubit_dense(psi, L_DEFAULT, pair[0], pair[1], gate)
    embedded = embed_dense_two_qubit_operator(u, pair[0], pair[1], L_DEFAULT) @ psi
    assert 1.0 - fidelity(direct, embedded) < 1e-12


@pytest.mark.parametrize("sequence", ["commuting", "mixed"])
def test_cumulative_exact_prefix(seed11_state, sequence) -> None:
    _mps, psi = seed11_state
    specs = build_sequence_gate_specs(sequence)
    prefix = build_exact_prefix_states(psi, specs, L_DEFAULT)
    manual = psi.copy()
    for idx, spec in enumerate(specs):
        gate = make_gate(spec.gate_type, spec.theta, spec.q0, spec.q1)
        manual = apply_two_qubit_dense(manual, L_DEFAULT, spec.q0, spec.q1, gate)
        assert 1.0 - fidelity(prefix[idx], manual) < 1e-12


def test_no_layer_reset(seed11_state) -> None:
    _mps, psi = seed11_state
    specs = build_sequence_gate_specs("commuting")
    prefix = build_exact_prefix_states(psi, specs, L_DEFAULT)
    manual = psi.copy()
    for spec in specs[:4]:
        gate = make_gate(spec.gate_type, spec.theta, spec.q0, spec.q1)
        manual = apply_two_qubit_dense(manual, L_DEFAULT, spec.q0, spec.q1, gate)
    assert 1.0 - fidelity(prefix[3], manual) < 1e-12
    wrong = apply_two_qubit_dense(psi, L_DEFAULT, specs[3].q0, specs[3].q1, make_gate(specs[3].gate_type, specs[3].theta, specs[3].q0, specs[3].q1))
    assert 1.0 - fidelity(prefix[3], wrong) > 1e-6


@pytest.mark.parametrize("sequence", ["commuting", "mixed"])
def test_chi64_one_layer_baselines(seed11_state, sequence) -> None:
    mps, psi = seed11_state
    specs = build_sequence_gate_specs(sequence)[:3]
    prefix = build_exact_prefix_states(psi, specs, L_DEFAULT)
    for mode in ("swaps", "mpo"):
        state = copy.deepcopy(mps)
        for spec in specs:
            apply_two_qubit_gate(state, spec.dag_node(L_DEFAULT), _params(64, gate_mode=mode))
        approx = state.to_vec()
        assert 1.0 - fidelity(prefix[-1], approx) < 1e-10
    tebd = copy.deepcopy(mps)
    mpo = copy.deepcopy(mps)
    for spec in specs:
        apply_two_qubit_gate(tebd, spec.dag_node(L_DEFAULT), _params(64, gate_mode="swaps"))
        apply_two_qubit_gate(mpo, spec.dag_node(L_DEFAULT), _params(64, gate_mode="mpo"))
    assert 1.0 - fidelity(tebd.to_vec(), mpo.to_vec()) < 1e-10


def test_resume_equivalence(seed11_state) -> None:
    mps, psi = seed11_state
    specs = build_sequence_gate_specs("mixed")[:6]
    prefix = build_exact_prefix_states(psi, specs, L_DEFAULT)

    def run_through(count: int) -> np.ndarray:
        state = copy.deepcopy(mps)
        for spec in specs[:count]:
            apply_two_qubit_gate(state, spec.dag_node(L_DEFAULT), _params(64, gate_mode="mpo"))
        return state.to_vec()

    full = run_through(6)
    interrupted = run_through(3)
    resumed = copy.deepcopy(mps)
    for spec in specs[:3]:
        apply_two_qubit_gate(resumed, spec.dag_node(L_DEFAULT), _params(64, gate_mode="mpo"))
    assert 1.0 - fidelity(interrupted, resumed.to_vec()) < 1e-12
    for spec in specs[3:6]:
        apply_two_qubit_gate(resumed, spec.dag_node(L_DEFAULT), _params(64, gate_mode="mpo"))
    assert 1.0 - fidelity(full, resumed.to_vec()) < 1e-12
    for k in range(6):
        assert 1.0 - fidelity(prefix[k], run_through(k + 1)) < 1e-12


def test_identity_controls(seed11_state) -> None:
    mps, psi = seed11_state
    for gate_type in ("rxx", "ryy", "rzz"):
        gate = make_gate(gate_type, 0.0, 2, 9)
        exact = apply_two_qubit_dense(psi, L_DEFAULT, 2, 9, gate)
        assert 1.0 - fidelity(exact, psi) < 1e-12
        tdvp = copy.deepcopy(mps)
        apply_two_qubit_gate(tdvp, make_dag_node(gate_type, 0.0, 2, 9, L_DEFAULT), _params(8, gate_mode="tdvp"))
        assert 1.0 - fidelity(exact, tdvp.to_vec()) < 1e-12
        exact64 = apply_two_qubit_dense(psi, L_DEFAULT, 2, 9, gate)
        for mode in ("swaps", "mpo", "tdvp"):
            st = copy.deepcopy(mps)
            apply_two_qubit_gate(st, make_dag_node(gate_type, 0.0, 2, 9, L_DEFAULT), _params(64, gate_mode=mode))
            assert 1.0 - fidelity(exact64, st.to_vec()) < 1e-10


def test_fit_validation_rules() -> None:
    two_point_rows = [
        {
            "experiment": "A",
            "method": "hybrid_tdvp",
            "gate_type": "rxx",
            "theta": 0.05,
            "chi_max": "8",
            "tdvp_substeps": "1",
            "infidelity": 1e-6,
        },
        {
            "experiment": "A",
            "method": "hybrid_tdvp",
            "gate_type": "rxx",
            "theta": 0.1,
            "chi_max": "8",
            "tdvp_substeps": "1",
            "infidelity": 4e-6,
        },
    ]
    slopes = fit_angle_scaling(two_point_rows)
    assert slopes["rxx"] == "not resolved: numerical floor"

    floor_rows = [
        {
            "experiment": "A",
            "method": "hybrid_tdvp",
            "gate_type": "rxx",
            "theta": t,
            "chi_max": "8",
            "tdvp_substeps": "1",
            "infidelity": 1e-15,
        }
        for t in (0.025, 0.05, 0.1)
    ]
    slopes = fit_angle_scaling(floor_rows)
    assert slopes["rxx"] == "not resolved: numerical floor"

    synthetic = [{
                "experiment": "A",
                "method": "hybrid_tdvp",
                "gate_type": "rxx",
                "theta": theta,
                "chi_max": "8",
                "tdvp_substeps": "1",
                "infidelity": float(theta**2),
            } for theta in (0.05, 0.1, 0.2)]
    slopes = fit_angle_scaling(synthetic)
    assert isinstance(slopes["rxx"], float)
    assert abs(float(slopes["rxx"]) - 2.0) < 0.05
