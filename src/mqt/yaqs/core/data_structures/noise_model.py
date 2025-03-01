# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

from ..libraries.noise_library import NoiseLibrary


class NoiseModel:
    # TODO: Currently assumes processes affect all sites equally
    def __init__(self, processes: list[str] | None = None, strengths: list[float] | None = None) -> None:
        if strengths is None:
            strengths = []
        if processes is None:
            processes = []
        assert len(processes) == len(strengths)
        self.processes = processes
        self.strengths = strengths
        self.jump_operators = []
        for process in processes:
            self.jump_operators.append(getattr(NoiseLibrary, process)().matrix)
