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

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.linalg import expm

from ..libraries.noise_library import NoiseLibrary

if TYPE_CHECKING:
    from numpy.typing import NDArray


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

            # Sort sites for consistency in any unraveling expansion
            if isinstance(sites, list) and len(sites) == 2:
                sorted_sites = sorted(sites)
                if sorted_sites != sites:
                    proc["sites"] = sorted_sites
                sites = proc["sites"]

            # Optional unraveling expansion into concrete jump operators
            if "unraveling" in proc:
                unravel = str(proc["unraveling"]).lower()
                gamma = float(proc["strength"])

                def get_pauli_string_matrix(p: dict[str, Any]) -> np.ndarray:
                    # one-site Pauli
                    if len(p["sites"]) == 1:
                        assert str(p["name"]).startswith("pauli_"), (
                            "Unraveling currently supported only for Pauli one-site processes."
                        )
                        return NoiseModel.get_operator(str(p["name"]))
                    # two-site adjacent Pauli string via crosstalk_ab naming or explicit matrix
                    ii, jj = p["sites"]
                    assert abs(jj - ii) == 1, (
                        "Unraveling currently supports only adjacent 2-site processes."
                    )
                    if str(p["name"]).startswith("crosstalk_"):
                        suffix = str(p["name"]).rsplit("_", 1)[-1]
                        assert len(suffix) == 2 and all(c in "xyz" for c in suffix), (
                            "For 2-site unraveling, use crosstalk_ab with a,b in {x,y,z}, or provide 'matrix'."
                        )
                        a, b = suffix[0], suffix[1]
                        return np.kron(PAULI_MAP[a], PAULI_MAP[b])
                    assert "matrix" in p, (
                        "For 2-site unraveling without crosstalk_ab name, an explicit 'matrix' must be provided."
                    )
                    return p["matrix"]

                P = get_pauli_string_matrix(proc)
                dim = P.shape[0]
                I = np.eye(dim, dtype=complex)

                if unravel == "projector":
                    # two jumps: J± = I ± P, each with strength gamma/2
                    for comp, sign in (("plus", +1.0), ("minus", -1.0)):
                        sub = {
                            "name": f"projector_{comp}_" + str(name),
                            "sites": list(proc["sites"]),
                            "strength": gamma / 2.0,
                            "matrix": (I + sign * P),
                        }
                        filled_processes.append(sub)
                    continue

                if unravel == "unitary_2pt":
                    theta0 = float(proc.get("theta0", 0.0))
                    s_val = np.sin(theta0) ** 2
                    assert s_val > 0.0, "theta0 too small; sin^2(theta0) must be > 0."
                    lam = gamma / s_val
                    for comp, sign in (("plus", +1.0), ("minus", -1.0)):
                        U = expm(1j * sign * theta0 * P)
                        sub = {
                            "name": f"unitary2pt_{comp}_" + str(name),
                            "sites": list(proc["sites"]),
                            "strength": lam / 2.0,
                            "matrix": U,
                        }
                        filled_processes.append(sub)
                    continue

                if unravel == "unitary_gauss":
                    sigma = float(proc["sigma"])  # required
                    M = int(proc.get("M", 31))
                    theta_max = float(proc.get("theta_max", 4.0 * sigma))
                    thetas_pos = np.linspace(0.0, theta_max, (M + 1) // 2)
                    thetas = np.concatenate([-thetas_pos[:0:-1], thetas_pos])
                    w = np.exp(-0.5 * (thetas / sigma) ** 2)
                    w = w / np.sum(w)
                    w = 0.5 * (w + w[::-1])  # exact symmetrization
                    s_weight = float(np.sum(w * (np.sin(thetas) ** 2)))
                    assert s_weight > 1e-12, (
                        "E[sin^2(theta)] too small; increase sigma or theta_max/M for unitary_gauss."
                    )
                    lam = gamma / s_weight
                    for idx, (wk, th) in enumerate(zip(w, thetas)):
                        if wk <= 0.0:
                            continue
                        U = expm(1j * th * P)
                        sub = {
                            "name": f"unitary_gauss_{idx}_" + str(name),
                            "sites": list(proc["sites"]),
                            "strength": lam * float(wk),
                            "matrix": U,
                        }
                        filled_processes.append(sub)
                    continue

                raise ValueError(f"Unknown unraveling scheme: {unravel!r}")

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
    def get_operator(name: str) -> NDArray[Any]:
        """Retrieve the operator from NoiseLibrary, possibly as a tensor product if needed.

        Args:
            name: Name of the noise process (e.g., 'xx', 'zz').

        Returns:
            np.ndarray: The matrix representation of the operator.
        """
        operator_class = getattr(NoiseLibrary, name)
        return operator_class().matrix
