# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License
"""Focused tests for the circuit bond-profile figure."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import QuadMesh
from matplotlib.colors import Normalize

from experiments.circuit_benchmarks.config import CHI_MAIN, N_STEPS, N
from experiments.circuit_benchmarks.figures import bond_profiles
from experiments.circuit_benchmarks.figures.bond_profiles import (
    FIGURE_HEIGHT_MM,
    FIGURE_WIDTH_MM,
    PROFILE_COLORMAP,
    PROFILE_METHOD_LABELS,
    PROFILE_METHOD_ORDER,
    PROFILE_ROW_LABELS,
    _bond_edges,
    _internal_bond_profile,
    _stack_step_profiles,
    create_figure,
    load_profile_matrices,
)
from experiments.circuit_benchmarks.plotting import CASE_ORDER


def test_internal_bond_profile_omits_fixed_boundaries() -> None:
    """The heatmap should contain only the physical MPS cuts."""
    full_profile = [1, 2, 3, 4, 1]
    assert np.array_equal(
        _internal_bond_profile(full_profile, n_sites=4),
        np.asarray([2, 3, 4]),
    )


def test_internal_bond_profile_rejects_nonunit_boundaries() -> None:
    """Open-boundary profile corruption should fail before plotting."""
    with pytest.raises(ValueError, match="unit boundary"):
        _internal_bond_profile([2, 2, 2, 2, 1], n_sites=4)


def test_step_profile_stack_requires_every_endpoint() -> None:
    """A missing Trotter-step endpoint must not be hidden in the heatmap."""
    profiles = {0: np.ones(3), 2: np.ones(3)}
    with pytest.raises(RuntimeError, match="missing=\\[1\\]"):
        _stack_step_profiles(profiles, n_steps=2)


def test_refresh_replaces_the_portable_profile_table(tmp_path, monkeypatch) -> None:
    """A fresh campaign must be able to replace the retained compact table."""
    source = tmp_path / bond_profiles.DATA_FILENAME
    source.write_text("stale\n", encoding="utf-8")
    expected = {
        (case, method): np.full((N_STEPS + 1, N - 1), 2, dtype=np.int64)
        for case in CASE_ORDER
        for method in PROFILE_METHOD_ORDER
    }
    tasks = {key: {"task_id": f"{key[0]}-{key[1]}"} for key in expected}
    monkeypatch.setattr(bond_profiles, "_current_trajectory_tasks", lambda _output_dir: tasks)
    monkeypatch.setattr(
        bond_profiles,
        "_task_profile_matrix",
        lambda task, _output_dir: expected[
            tuple(str(task["task_id"]).split("-", maxsplit=1))
        ],
    )

    refreshed = load_profile_matrices(tmp_path, refresh_data=True)

    assert set(refreshed) == set(expected)
    assert source.read_text(encoding="utf-8").startswith("case,method,step,bond,bond_dimension")


def test_bond_edges_bound_all_internal_cuts() -> None:
    """Fifteen internal cuts require sixteen heatmap cell edges."""
    edges = _bond_edges()
    assert len(edges) == 16
    assert edges[[0, -1]] == pytest.approx([0.5, 15.5])


def test_publication_figure_uses_single_column_layout() -> None:
    """The publication heatmap should use the requested single-column layout."""
    expected_methods = ("gate_local_2tdvp", "mpo_contract_compress", "tebd_swap")
    expected_method_labels = ("TDVP", "MPO", "TEBD+SWAP")
    expected_row_labels = (
        "(a)\n1D\nIsing",
        "(b)\n1D\nHeisenberg",
        "(c)\n" + "$4\\times4$\nIsing",
        "(d)\n" + "$4\\times4$\nHeisenberg",
    )
    capped_profile = np.ones((N_STEPS + 1, N - 1), dtype=int)
    capped_profile[:, N // 2 :] = CHI_MAIN
    profiles = {(case, method): capped_profile.copy() for case in CASE_ORDER for method in PROFILE_METHOD_ORDER}

    figure = create_figure(profiles)
    axis_text = [text.get_text() for axis in figure.axes for text in axis.texts]
    titles = [axis.get_title() for axis in figure.axes if axis.get_title()]
    figure_text = [text.get_text() for text in figure.texts]
    size_mm = tuple(value * 25.4 for value in figure.get_size_inches())
    colorbar_bounds = figure.axes[-1].get_position().bounds
    data_axes = [
        axis for axis in figure.axes[:-1] if any(isinstance(collection, QuadMesh) for collection in axis.collections)
    ]
    meshes = [collection for axis in data_axes for collection in axis.collections if isinstance(collection, QuadMesh)]
    bond_axis = next(axis for axis in figure.axes if any(text.get_text() == r"Bond $b$" for text in axis.texts))
    row_label_axis = next(
        axis for axis in figure.axes if any(text.get_text() == expected_row_labels[0] for text in axis.texts)
    )
    xlabel = next(text for text in figure.texts if text.get_text() == r"Trotter steps $n$")
    plot_center = (data_axes[0].get_position().x0 + data_axes[2].get_position().x1) / 2
    plt.close(figure)

    assert expected_methods == PROFILE_METHOD_ORDER
    assert tuple(PROFILE_METHOD_LABELS[method] for method in PROFILE_METHOD_ORDER) == expected_method_labels
    assert tuple(PROFILE_ROW_LABELS[case] for case in CASE_ORDER) == expected_row_labels
    assert [text for text in axis_text if text in expected_row_labels] == list(expected_row_labels)
    assert titles[:-1] == list(expected_method_labels)
    assert titles[-1] == r"Bond dimension $\chi_b$"
    assert r"Trotter steps $n$" in figure_text
    assert size_mm == pytest.approx((FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM))
    assert colorbar_bounds[2] > colorbar_bounds[3]
    assert row_label_axis.get_position().x1 < bond_axis.get_position().x0
    assert bond_axis.get_position().x1 < data_axes[0].get_position().x0
    assert xlabel.get_position()[0] == pytest.approx(plot_center)
    assert len(meshes) == len(CASE_ORDER) * len(PROFILE_METHOD_ORDER)
    assert all(type(mesh.norm) is Normalize for mesh in meshes)
    assert all(len(axis.collections) == 1 for axis in data_axes)
    assert all(mesh.cmap.name == PROFILE_COLORMAP for mesh in meshes)
