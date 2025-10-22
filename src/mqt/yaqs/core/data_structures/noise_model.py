# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
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
from typing import TYPE_CHECKING, Any

import numpy as np

from ..libraries.gate_library import GateLibrary
from ..libraries.noise_library import NoiseLibrary

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.core.libraries.gate_library import BaseGate


CROSSTALK_PREFIX = "longrange_crosstalk_"
PAULI_MAP = {
    "x": NoiseLibrary.pauli_x().matrix,
    "y": NoiseLibrary.pauli_y().matrix,
    "z": NoiseLibrary.pauli_z().matrix,
}


class NoiseModel:
    """A class to represent a noise model with arbitrary-site jump operators.

     Attributes.
    ----------
    processes : list of str
        A list of noise processes affecting the system.
        Each process is a dict with:
            - name: process name or identifier
            - sites: indices of sites this process acts on
            - strength: noise strength
            - matrix: matrix representing the operator on those sites (for 1-site and adjacent 2-site processes)
            - factors: tuple of two 1-site operator matrices (for long-range 2-site processes)

    Methods:
    -------
    __init__:
        Initializes the NoiseModel with given processes.
    get_operator:
        Static method to retrieve the operator matrix for a given noise process name.
    """

    def __init__(self, processes: list[dict[str, Any]] | None = None) -> None:
        """Initialize the NoiseModel.

        Parameters
        ----------
        processes :
            A list of noise process dictionaries affecting the quantum system. Default is None.

        Note:
            Input validation is performed and assertion errors may be raised by
            internal helpers if inputs are malformed.
        """
        self.processes: list[dict[str, Any]] = []
        if processes is None:
            return

        filled_processes: list[dict[str, Any]] = []
        for original in processes:
            assert "name" in original, "Each process must have a 'name' key"
            assert "sites" in original, "Each process must have a 'sites' key"
            assert "strength" in original, "Each process must have a 'strength' key"
            assert len(original["sites"]) <= 2, "Each noise process must have at most 2 sites"

            proc = dict(original)
            name = proc["name"]
            sites = proc["sites"]

            # Normalize two-site ordering
            if isinstance(sites, list) and len(sites) == 2:
                sorted_sites = sorted(sites)
                if sorted_sites != sites:
                    proc["sites"] = sorted_sites
                i, j = proc["sites"]
                is_adjacent = abs(j - i) == 1

                # Adjacent two-site: use full matrix
                if is_adjacent:
                    if str(name).startswith("crosstalk_"):
                        # infer matrix from suffix ab
                        suffix = str(name).rsplit("_", 1)[-1]
                        assert len(suffix) == 2, "Invalid crosstalk label. Expected 'crosstalk_ab' with a,b in {x,y,z}."
                        assert all(c in "xyz" for c in suffix), (
                            "Invalid crosstalk label. Expected 'crosstalk_ab' with a,b in {x,y,z}."
                        )
                        a, b = suffix[0], suffix[1]
                        proc["matrix"] = np.kron(PAULI_MAP[a], PAULI_MAP[b])
                    elif "matrix" not in proc:
                        proc["matrix"] = NoiseModel.get_operator(name)
                    filled_processes.append(proc)
                    continue

                # Non-adjacent two-site: attach per-site factors for crosstalk labels
                if str(name).startswith("crosstalk_"):
                    if "factors" not in proc:
                        suffix = str(name).rsplit("_", 1)[-1]
                        assert len(suffix) == 2, "Invalid crosstalk label. Expected 'crosstalk_ab' with a,b in {x,y,z}."
                        assert all(c in "xyz" for c in suffix), (
                            "Invalid crosstalk label. Expected 'crosstalk_ab' with a,b in {x,y,z}."
                        )
                        a, b = suffix[0], suffix[1]
                        proc["factors"] = (PAULI_MAP[a], PAULI_MAP[b])
                    filled_processes.append(proc)
                    continue

                # Long-range two-site with canonical legacy label
                if str(name).startswith(CROSSTALK_PREFIX):
                    if "factors" not in proc:
                        suffix = str(name).rsplit("_", 1)[-1]
                        assert len(suffix) == 2, (
                            f"Invalid crosstalk label. Expected '{CROSSTALK_PREFIX}ab' with a,b in {{x,y,z}}."
                        )
                        assert all(c in "xyz" for c in suffix), (
                            f"Invalid crosstalk label. Expected '{CROSSTALK_PREFIX}ab' with a,b in {{x,y,z}}."
                        )
                        a, b = suffix[0], suffix[1]
                        proc["factors"] = (PAULI_MAP[a], PAULI_MAP[b])
                    filled_processes.append(proc)
                    continue

                # Other long-range two-site: require explicit factors
                assert "factors" in proc, (
                    "Non-adjacent 2-site processes must specify 'factors' unless named 'crosstalk_{ab}'."
                )
                filled_processes.append(proc)
                continue

            # One-site: ensure matrix
            if "matrix" not in proc:
                proc["matrix"] = NoiseModel.get_operator(name)
            filled_processes.append(proc)

        self.processes = filled_processes

    @staticmethod
    def get_operator(name: str) -> NDArray[np.complex128]:
        """Retrieve the operator from NoiseLibrary, possibly as a tensor product if needed.

        Args:
            name: Name of the noise process (e.g., 'xx', 'zz').

        Returns:
            np.ndarray: The matrix representation of the operator.
        """
        operator_class = getattr(NoiseLibrary, name)
        operator: BaseGate = operator_class()
        return operator.matrix


class CompactNoiseModel:
    """A class to represent a compact noise model with multi-site noise processes."""

    def __init__(self, compact_processes: list[dict[str, Any]] | None = None) -> None:
        """Initialize the compact noise model.

        Parameters.
        ----------
        compact_processes : list[dict[str, Any]] | None, optional
            A list of compact noise process specifications or None. Each process dict
            must contain the following keys:
              - "name" (str): name of a noise process available from GateLibrary.
              - "sites" (iterable[int]): indices of sites the compact process acts on.
              - "strength" (numeric): strength/amplitude of the noise process.
        Behavior
        --------
        - If compact_processes is provided, a deep copy is stored in
          self.compact_processes to avoid external mutation; otherwise an empty list
          is used.
        - Each process dict is validated:
            - Must contain the keys "name", "sites", and "strength".
            - The GateLibrary entry for the given "name" must exist and represent a
              1-site interaction (GateLibrary.<name>().interaction == 1).
          Violations raise AssertionError or ValueError.
        - The compact processes are expanded into single-site process entries:
            - For each site in a compact process, an expanded dict
              {"name": name, "sites": [site], "strength": strength} is appended to
              self.expanded_processes.
            - For each expanded entry, the index of the originating compact process
              is appended to self.index_list.
        - A NoiseModel is created from the expanded processes and stored as
          self.expanded_noise_model.
        Attributes set
        --------------
        self.compact_processes : list[dict[str, Any]]
            Deep-copied list of compact process definitions (or empty list).
        self.expanded_processes : list[dict[str, Any]]
            List of per-site expanded process dictionaries.
        self.index_list : list[int]
            Mapping from each expanded process to the index of its original compact
            process in self.compact_processes.
        self.expanded_noise_model : NoiseModel
            NoiseModel constructed from self.expanded_processes.
        self.strength_list : np.ndarray
            Array of strengths corresponding to each compact process.

        Raises:
        ------
        ValueError: If the named gate is not found in GateLibrary. The exception message is:
                    "Gate '<name>' not found in GateLibrary".

        Notes:
        -----
        - This initializer assumes GateLibrary and NoiseModel are available in the
          surrounding module scope.
        - The method performs input validation and expansion to ensure downstream
          code can work with single-site noise process descriptions.
        """
        self.compact_processes: list[dict[str, Any]] = (
            copy.deepcopy(compact_processes) if compact_processes is not None else []
        )

        self.expanded_processes: list[dict[str, Any]] = []

        self.index_list: list[int] = []

        for i, proc in enumerate(self.compact_processes):
            assert "name" in proc, "Each process must have a 'name' key"
            name = proc["name"]
            if not hasattr(GateLibrary, name):
                msg = f"Gate '{name}' not found in GateLibrary"
                raise ValueError(msg)
            msg = "Only 1-site noise processes are supported in CompactNoiseModel"
            assert getattr(GateLibrary, name)().interaction == 1, msg
            assert "sites" in proc, "Each process must have a 'sites' key"
            assert "strength" in proc, "Each process must have a 'strength' key"

            for site in proc["sites"]:
                self.expanded_processes.append({"name": proc["name"], "sites": [site], "strength": proc["strength"]})
                self.index_list.append(i)

        self.strength_list: np.ndarray = np.array([proc["strength"] for proc in self.compact_processes])

        self.expanded_noise_model: NoiseModel = NoiseModel(self.expanded_processes)
