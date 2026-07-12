# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for PRX single-column figure layout presets."""

from __future__ import annotations

from experiments.pt_entropy_figure.layouts import (
    PRX_SINGLE_COLUMN_LAYOUT,
    PRX_SINGLE_COLUMN_WIDTH_IN,
    STANDARD_LAYOUT,
)


def test_prx_single_column_width_matches_aps_spec() -> None:
    assert PRX_SINGLE_COLUMN_LAYOUT.figsize[0] == PRX_SINGLE_COLUMN_WIDTH_IN
    assert abs(PRX_SINGLE_COLUMN_WIDTH_IN - 3.375) < 1e-9


def test_prx_layout_uses_cut_vs_j_side_by_side_topology() -> None:
    assert PRX_SINGLE_COLUMN_LAYOUT.topology == "side_by_side"
    assert PRX_SINGLE_COLUMN_LAYOUT.width_ratios == (2.0, 1.0)
    assert PRX_SINGLE_COLUMN_LAYOUT.wspace == 0.06
    assert PRX_SINGLE_COLUMN_LAYOUT.hspace == 0.10


def test_standard_layout_unchanged() -> None:
    assert STANDARD_LAYOUT.output_stem == "sv_ptcb_entropy_spectra"
    assert STANDARD_LAYOUT.figsize == (7.1, 2.95)
    assert STANDARD_LAYOUT.width_ratios == (1.08, 1.0)
