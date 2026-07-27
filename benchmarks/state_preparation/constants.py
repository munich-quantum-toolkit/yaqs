# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared identifiers for the state-preparation benchmark suite."""

from __future__ import annotations

from types import MappingProxyType

SUPPORTED_QUBIT_COUNTS = (6, 12)
TARGET_FIXTURE_FORMAT = "yaqs.state_preparation_targets.v1"

TARGET_IDS = (
    "gaussian_mu0p5_sigma0p1",
    "tfim_ferro",
    "tfim_critical",
    "tfim_para",
    "haar_random_1",
    "haar_random_2",
    "haar_random_3",
    "random_mps_bond2",
    "random_mps_bond3",
)

TARGET_GENERATION_SEEDS = MappingProxyType({
    "gaussian_mu0p5_sigma0p1": None,
    "tfim_ferro": None,
    "tfim_critical": None,
    "tfim_para": None,
    "haar_random_1": 4001,
    "haar_random_2": 4002,
    "haar_random_3": 4003,
    "random_mps_bond2": 5002,
    "random_mps_bond3": 5003,
})

NOISELESS_NOISE_ID = "noiseless"
BALLARIN_NOISE_ID = "ballarin_coupled"

DEPHASING_NOISE_IDS = (
    "dephasing_1s_1q",
    "dephasing_1s_2q",
    "dephasing_1s_all",
    "dephasing_2s_2q",
    "dephasing_1s2s_all",
)

DEPOLARIZING_NOISE_IDS = (
    "depolarizing_1s_1q",
    "depolarizing_1s_2q",
    "depolarizing_1s_all",
    "depolarizing_2s_2q",
    "depolarizing_1s2s_all",
)

STANDARD_NOISE_IDS = DEPHASING_NOISE_IDS + DEPOLARIZING_NOISE_IDS
NOISE_IDS = (NOISELESS_NOISE_ID, BALLARIN_NOISE_ID, *STANDARD_NOISE_IDS)

__all__ = [
    "BALLARIN_NOISE_ID",
    "DEPHASING_NOISE_IDS",
    "DEPOLARIZING_NOISE_IDS",
    "NOISELESS_NOISE_ID",
    "NOISE_IDS",
    "STANDARD_NOISE_IDS",
    "SUPPORTED_QUBIT_COUNTS",
    "TARGET_FIXTURE_FORMAT",
    "TARGET_GENERATION_SEEDS",
    "TARGET_IDS",
]
