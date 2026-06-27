# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Matrix Product State (MPS) for YAQS tensor-network simulations."""

from __future__ import annotations

import concurrent.futures
import copy
import multiprocessing
from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe
from tqdm import tqdm

from .. import linalg
from ..methods.decompositions import merge_two_site, right_qr, split_two_site

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..methods.decompositions import TruncMode
    from .simulation_parameters import AnalogSimParams, Observable, StrongSimParams


class MPS:
    """Matrix Product State (MPS) class for representing quantum states.

    This class forms the basis of the MPS used in YAQS simulations.
    The index order is (sigma, chi_l-1, chi_l).

    Attributes:
        length: The number of sites in the MPS.
        tensors: List of rank-3 tensors representing the MPS.
        physical_dimensions: List of physical dimensions for each site.
        flipped: Indicates if the network has been flipped.
    """

    def __init__(
        self,
        length: int,
        tensors: list[NDArray[np.complex128]] | None = None,
        physical_dimensions: list[int] | int | None = None,
        state: str = "zeros",
        pad: int | None = None,
        basis_string: str | None = None,
    ) -> None:
        """Initializes a Matrix Product State (MPS).

        Args:
            length: Number of sites (qubits) in the MPS.
            tensors: Predefined tensors representing the MPS. Must match `length` if provided.
                If None, tensors are initialized according to `state`.
            physical_dimensions: Physical dimension for each site. Defaults to qubit systems (dimension 2) if None.
            state: Initial state configuration. Valid options include:
                - "zeros": Initializes all qubits to |0⟩.
                - "ones": Initializes all qubits to |1⟩.
                - "x+": Initializes each qubit to (|0⟩ + |1⟩)/√2.
                - "x-": Initializes each qubit to (|0⟩ - |1⟩)/√2.
                - "y+": Initializes each qubit to (|0⟩ + i|1⟩)/√2.
                - "y-": Initializes each qubit to (|0⟩ - i|1⟩)/√2.
                - "Neel": Alternating pattern |0101...⟩.
                - "wall": Domain wall at given site |000111>
                - "random": Initializes each qubit randomly.
                - "haar-random": Initializes an entangled MPS via Haar-random isometries.
                - "basis": Initializes a qubit in an input computational basis.
                Default is "zeros".
            pad: Pads the state with extra zeros to increase bond dimension. Can increase numerical stability.
                For ``state="haar-random"``, this value is interpreted as the target maximum internal
                bond dimension χ_max. If omitted, χ_max defaults to 1.
            basis_string: String used to initialize the state in a specific computational basis.
                This should generally be in the form of 0s and 1s, e.g., "0101" for a 4-qubit state.
                For mixed-dimensional systems, this can be increased to 2, 3, ... etc.

        Raises:
            ValueError: If the provided `state` parameter does not match any valid initialization string.
        """
        self.flipped = False
        if tensors is not None:
            assert len(tensors) == length
            self.tensors = tensors
        else:
            self.tensors = []
        self.length = length
        if physical_dimensions is None:
            # Default case is the qubit (2-level) case
            self.physical_dimensions = []
            for _ in range(self.length):
                self.physical_dimensions.append(2)
        elif isinstance(physical_dimensions, int):
            self.physical_dimensions = []
            for _ in range(self.length):
                self.physical_dimensions.append(physical_dimensions)
        else:
            self.physical_dimensions = physical_dimensions
        assert len(self.physical_dimensions) == length

        def _bond_caps(target_dim: int) -> list[int]:
            """Compute feasible MPS bond dimensions for a target maximum.

            Args:
                target_dim: Target maximum internal bond dimension.

            Returns:
                List of length ``self.length + 1`` with bond dimensions
                ``[chi_0, ..., chi_L]`` where boundaries satisfy
                ``chi_0 = chi_L = 1``.

            Raises:
                ValueError: If ``target_dim < 1``.
            """
            if target_dim < 1:
                msg = "Target bond dimension must be at least 1."
                raise ValueError(msg)
            caps = [0] * (self.length + 1)
            caps[0] = 1
            caps[self.length] = 1

            # Left-to-right representability cap
            left_cap = 1
            for i in range(1, self.length):
                left_cap *= self.physical_dimensions[i - 1]
                caps[i] = left_cap

            # Right-to-left representability cap
            right_cap = 1
            for i in range(self.length - 1, 0, -1):
                right_cap *= self.physical_dimensions[i]
                caps[i] = min(caps[i], right_cap)

            # Apply target cap on internal bonds
            for i in range(1, self.length):
                caps[i] = min(caps[i], target_dim)

            return caps

        def _haar_random_tensor_core(
            site: int,
            local_dim: int,
            target_dim: int,
            *,
            _bond_cache: dict[str, list[int] | None] | None = None,
            _rng_cache: dict[str, np.random.Generator | None] | None = None,
        ) -> NDArray[np.complex128]:
            """Construct one Haar-random isometric MPS tensor core lazily.

            Args:
                site: Site index of the tensor core.
                local_dim: Physical dimension at the site.
                target_dim: Target maximum internal bond dimension.
                _bond_cache: Optional cache for lazily computed bond dimensions.
                _rng_cache: Optional cache for lazily initialized RNG.

            Returns:
                Tensor core with shape ``(local_dim, chi_l, chi_r)``.
            """
            if _rng_cache is None:
                _rng_cache = {"rng": None}
            if _bond_cache is None:
                _bond_cache = {"dims": None}
            if _bond_cache["dims"] is None:
                _bond_cache["dims"] = _bond_caps(target_dim)
            if _rng_cache["rng"] is None:
                _rng_cache["rng"] = np.random.default_rng()

            bond_dims = _bond_cache["dims"]
            rng = _rng_cache["rng"]
            assert bond_dims is not None
            assert rng is not None

            chi_l = bond_dims[site]
            chi_r = bond_dims[site + 1]
            assert chi_r <= local_dim * chi_l, "Invalid bond schedule for Haar-random initialization."

            x_mat = rng.standard_normal((local_dim * chi_l, chi_r)) + 1j * rng.standard_normal((
                local_dim * chi_l,
                chi_r,
            ))
            q_mat, r_mat = np.linalg.qr(x_mat, mode="reduced")

            # Fix arbitrary QR phases for a well-defined Haar isometry sample.
            diag = np.diag(r_mat)
            phases = np.ones_like(diag, dtype=np.complex128)
            non_zero = np.abs(diag) > 0
            phases[non_zero] = diag[non_zero] / np.abs(diag[non_zero])
            q_mat /= phases[np.newaxis, :]

            return q_mat.reshape(local_dim, chi_l, chi_r).astype(np.complex128)

        # Create d-level |0> state
        if not tensors:
            haar_bond_cache: dict[str, list[int] | None] | None = None
            haar_rng_cache: dict[str, np.random.Generator | None] | None = None
            if state == "haar-random":
                haar_bond_cache = {"dims": None}
                haar_rng_cache = {"rng": None}
            for i, d in enumerate(self.physical_dimensions):
                vector = np.zeros(d, dtype=complex)
                if state == "zeros":
                    # |0>
                    vector[0] = 1
                elif state == "ones":
                    # |1>
                    vector[1] = 1
                elif state == "x+":
                    # |+> = (|0> + |1>)/sqrt(2)
                    vector[0] = 1 / np.sqrt(2)
                    vector[1] = 1 / np.sqrt(2)
                elif state == "x-":
                    # |-> = (|0> - |1>)/sqrt(2)
                    vector[0] = 1 / np.sqrt(2)
                    vector[1] = -1 / np.sqrt(2)
                elif state == "y+":
                    # |+i> = (|0> + i|1>)/sqrt(2)
                    vector[0] = 1 / np.sqrt(2)
                    vector[1] = 1j / np.sqrt(2)
                elif state == "y-":
                    # |-i> = (|0> - i|1>)/sqrt(2)
                    vector[0] = 1 / np.sqrt(2)
                    vector[1] = -1j / np.sqrt(2)
                elif state == "Neel":
                    # |010101...>
                    if i % 2:
                        vector[0] = 1
                    else:
                        vector[1] = 1
                elif state == "wall":
                    # |000111>
                    if i < length // 2:
                        vector[0] = 1
                    else:
                        vector[1] = 1
                elif state == "random":
                    rng = np.random.default_rng()
                    vector[0] = rng.random()
                    vector[1] = 1 - vector[0]
                elif state == "haar-random":
                    target_dim = 1 if pad is None else pad
                    tensor = _haar_random_tensor_core(
                        i,
                        d,
                        target_dim,
                        _bond_cache=haar_bond_cache,
                        _rng_cache=haar_rng_cache,
                    )
                    self.tensors.append(tensor)
                    continue
                elif state == "basis":
                    assert basis_string is not None, "basis_string must be provided for 'basis' state initialization."
                    self.init_mps_from_basis(basis_string, self.physical_dimensions)
                    break
                else:
                    msg = "Invalid state string"
                    raise ValueError(msg)

                tensor = np.expand_dims(vector, axis=(0, 1))

                tensor = np.transpose(tensor, (2, 0, 1))
                self.tensors.append(tensor)

            if state == "random":
                self.normalize()
        if pad is not None and state != "haar-random":
            self.pad_bond_dimension(pad)

    def init_mps_from_basis(self, basis_string: str, physical_dimensions: list[int]) -> None:
        """Initialize a list of MPS tensors representing a product state from a basis string.

        Args:
            basis_string: A string like "0101" indicating the computational basis state.
            physical_dimensions: The physical dimension of each site (e.g. 2 for qubits, 3+ for qudits).
        """
        assert len(basis_string) == len(physical_dimensions)
        for site, char in enumerate(basis_string):
            idx = int(char)
            tensor = np.zeros((physical_dimensions[site], 1, 1), dtype=complex)
            tensor[idx, 0, 0] = 1.0
            self.tensors.append(tensor)

    def pad_bond_dimension(self, target_dim: int) -> None:
        """Pad MPS with extra zeros to increase bond dims.

        Enlarge every internal bond up to ``min(target_dim, 2**exp)``
        where ``exp = min(bond_index+1, L-1-bond_index)``.
        The first tensor keeps a left bond of 1, the last tensor a right bond of 1.
        After padding the state is renormalised (canonicalised).

        Args:
            target_dim: The desired bond dimension for the internal bonds.

        Raises:
            ValueError: target_dim must be at least current bond dim.
        """
        length = self.length

        # enlarge tensors
        for i, tensor in enumerate(self.tensors):
            phys, chi_l, chi_r = tensor.shape

            # compute the desired dimension for the bond left of site i
            if i == 0:
                left_target = 1
            else:
                exp_left = min(i, length - i)  # bond index = i - 1
                left_target = min(target_dim, 2**exp_left)

            if i == length - 1:
                right_target = 1
            else:
                exp_right = min(i + 1, length - 1 - i)  # bond index = i
                right_target = min(target_dim, 2**exp_right)

            # sanity-check — we must never shrink an existing bond
            if chi_l > left_target or chi_r > right_target:
                msg = "Target bond dim must be at least current bond dim."
                raise ValueError(msg)

            # allocate new tensor and copy original data
            new_tensor = np.zeros((phys, left_target, right_target), dtype=tensor.dtype)
            new_tensor[:, :chi_l, :chi_r] = tensor
            self.tensors[i] = new_tensor
        # renormalise the state
        self.normalize()

    def ensure_internal_bond_dims(
        self,
        bond_indices: list[int] | tuple[int, ...],
        min_dim: int,
        *,
        max_dim: int | None = None,
    ) -> None:
        """Zero-pad selected internal bonds to at least ``min_dim``.

        Library-internal padding helper for fixed-χ TDVP bond alignment. Bond ``b``
        connects sites ``b`` and ``b+1``. Only the listed bonds are modified; tensors
        are zero-padded on the shared index when needed. Shrinking a bond requires
        SVD truncation via :func:`mqt.yaqs.core.methods.tdvp.sweep_utils._sync_bond_dim`.

        Args:
            bond_indices: Internal bond indices ``0 <= b < length - 1``.
            min_dim: Minimum bond dimension to enforce on each listed bond.
            max_dim: Optional hard cap; when set, bonds are never padded above this
                value and no-op if ``min_dim`` exceeds ``max_dim``.

        Raises:
            ValueError: If ``min_dim`` is less than 1, a bond index is invalid, or a
                listed bond must be truncated below its current dimension.
        """
        if min_dim < 1:
            msg = "min_dim must be at least 1."
            raise ValueError(msg)
        if max_dim is not None and min_dim > max_dim:
            return
        target_dim = min_dim if max_dim is None else min(min_dim, max_dim)
        for bond in bond_indices:
            if bond < 0 or bond >= self.length - 1:
                msg = f"Bond index {bond} out of range for length {self.length}."
                raise ValueError(msg)
            left = self.tensors[bond]
            right = self.tensors[bond + 1]
            chi_out = int(left.shape[2])
            chi_in = int(right.shape[1])
            if chi_out == target_dim and chi_in == target_dim:
                continue
            if chi_out > target_dim or chi_in > target_dim:
                msg = (
                    f"Bond {bond} cannot be truncated from (chi_out={chi_out}, chi_in={chi_in}) "
                    f"to target_dim={target_dim}; use "
                    f"mqt.yaqs.core.methods.tdvp.sweep_utils._sync_bond_dim for SVD truncation."
                )
                raise ValueError(msg)
            chi_out = int(left.shape[2])
            chi_in = int(right.shape[1])
            if chi_out >= target_dim and chi_in >= target_dim:
                continue
            phys_l, chi_l, _ = left.shape
            phys_r, _, chi_r = right.shape
            new_left = np.zeros((phys_l, chi_l, target_dim), dtype=left.dtype)
            new_left[:, :, :chi_out] = left
            new_right = np.zeros((phys_r, target_dim, chi_r), dtype=right.dtype)
            new_right[:, :chi_in, :] = right
            self.tensors[bond] = new_left
            self.tensors[bond + 1] = new_right

    def bond_dimensions(self) -> list[int]:
        """Return outgoing bond dimension at each internal bond ``b``.

        Returns:
            List of bond dimensions ``[chi_0, ..., chi_{L-2}]``.
        """
        return [int(tensor.shape[2]) for tensor in self.tensors[:-1]]

    def assert_bond_shapes_consistent(self, *, max_bond_dim: int | None = None) -> None:
        """Validate adjacent tensor virtual dimensions and an optional bond cap.

        Library-internal invariant check used by fixed-χ TDVP.

        Args:
            max_bond_dim: When set, each internal bond must not exceed this value.

        Raises:
            ValueError: If outgoing/incoming bond dimensions disagree or exceed the cap.
        """
        for bond in range(self.length - 1):
            left = self.tensors[bond]
            right = self.tensors[bond + 1]
            chi_out = int(left.shape[2])
            chi_in = int(right.shape[1])
            if chi_out != chi_in:
                msg = (
                    f"MPS bond mismatch at bond {bond}: left outgoing {chi_out} "
                    f"!= right incoming {chi_in}; left shape {left.shape}, "
                    f"right shape {right.shape}"
                )
                raise ValueError(msg)
            if max_bond_dim is not None and chi_out > max_bond_dim:
                msg = f"MPS bond cap violated at bond {bond}: chi={chi_out} > max_bond_dim={max_bond_dim}"
                raise ValueError(msg)

    def get_max_bond(self) -> int:
        """Write max bond dim.

        Calculate and return the maximum bond dimension of the tensors in the network.
        This method iterates over all tensors in the network and determines the maximum
        bond dimension by comparing the first and third dimensions of each tensor's shape.
        The global maximum bond dimension is then returned.

        Returns:
            int: The maximum bond dimension found among all tensors in the network.
        """
        global_max = 0
        for tensor in self.tensors:
            local_max = max(tensor.shape[0], tensor.shape[2])
            global_max = max(global_max, local_max)

        return global_max

    def get_total_bond(self) -> int:
        """Compute total bond dimension.

        Calculates the sum of all internal bond dimensions of the network.
        Specifically, this sums the second index (left bond dimension)
        of each tensor except for the first tensor.

        Returns:
            int: The total bond dimension across all internal bonds.
        """
        bonds = [tensor.shape[1] for tensor in self.tensors[1:]]
        return sum(bonds)

    def get_cost(self) -> int:
        """Estimate contraction cost.

        Approximates the computational cost of simulating the network
        by summing the cube of each internal bond dimension. This is a
        heuristic metric for the cost of tensor contractions.

        Returns:
            int: The estimated contraction cost of the network.
        """
        cost = [tensor.shape[1] ** 3 for tensor in self.tensors[1:]]
        return sum(cost)

    def record_diagnostics(self, diagnostics: NDArray[np.float64], column_index: int) -> None:
        """Write runtime cost, max bond, and total bond into a diagnostics row buffer.

        Args:
            diagnostics: Array shaped ``(3, T)``; rows are cost, max bond, total bond.
            column_index: Column (time or layer index) to fill.
        """
        diagnostics[0, column_index] = self.get_cost()
        diagnostics[1, column_index] = self.get_max_bond()
        diagnostics[2, column_index] = self.get_total_bond()

    def get_entropy(self, sites: list[int]) -> np.float64:
        """Compute bipartite entanglement entropy.

        Calculates the von Neumann entropy of the reduced density matrix
        across the bond between two adjacent sites. The entropy is obtained
        from the Schmidt spectrum of the two-site state.

        Args:
            sites (list[int]): A list of exactly two adjacent site indices (i, i+1).

        Returns:
            np.float64: The entanglement entropy across the specified bond.

        """
        assert len(sites) == 2, "Entropy is defined on a bond (two adjacent sites)."
        i, j = sites
        assert i + 1 == j, "Entropy is only defined for nearest-neighbor cut."

        a, b = self.tensors[i], self.tensors[j]

        if a.shape[2] == 1:
            return np.float64(0.0)

        theta = np.tensordot(a, b, axes=(2, 1))
        phys_i, left = a.shape[0], a.shape[1]
        phys_j, right = b.shape[0], b.shape[2]
        theta_mat = theta.reshape(left * phys_i, phys_j * right)

        s = linalg.svd(theta_mat, full_matrices=False, compute_uv=False)
        s2 = (s.astype(np.float64)) ** 2
        norm: np.float64 = np.sum(s2, dtype=np.float64)
        if norm == np.float64(0.0):
            return np.float64(0.0)

        p = s2 / norm
        eps = np.finfo(np.float64).tiny
        ent = -1 * np.sum(p * np.log(p + eps), dtype=np.float64)

        return np.float64(ent)

    def get_schmidt_spectrum(self, sites: list[int]) -> NDArray[np.float64]:
        """Compute Schmidt spectrum.

        Calculates the singular values of the bipartition between two
        adjacent sites (the Schmidt coefficients). The spectrum is padded
        or truncated to length 500 for consistent output size.

        Args:
            sites (list[int]): A list of exactly two adjacent site indices (i, i+1).

        Returns:
            NDArray[np.float64]: The Schmidt spectrum (length 500),
            with unused entries filled with NaN.
        """
        assert len(sites) == 2, "Schmidt spectrum is defined on a bond (two adjacent sites)."
        assert sites[0] + 1 == sites[1], "Schmidt spectrum only defined for nearest-neighbor cut."
        top_schmidt_vals = 500
        i, j = sites
        a, b = self.tensors[i], self.tensors[j]

        if a.shape[2] == 1:
            padded = np.full(top_schmidt_vals, np.nan)
            padded[0] = 1.0
            return padded

        theta = np.tensordot(a, b, axes=(2, 1))
        phys_i, left = a.shape[0], a.shape[1]
        phys_j, right = b.shape[0], b.shape[2]
        theta_mat = theta.reshape(left * phys_i, phys_j * right)

        _, s_vec, _ = linalg.svd(theta_mat, full_matrices=False)

        padded = np.full(top_schmidt_vals, np.nan)
        padded[: min(top_schmidt_vals, len(s_vec))] = s_vec[:top_schmidt_vals]
        return padded

    def flip_network(self) -> None:
        """Flip MPS.

        Flips the bond dimensions in the network so that we can do operations
        from right to left rather than coding it twice.

        """
        new_tensors = []
        for tensor in self.tensors:
            new_tensor = np.transpose(tensor, (0, 2, 1))
            new_tensors.append(new_tensor)

        new_tensors.reverse()
        self.tensors = new_tensors
        self.flipped = not self.flipped

    def almost_equal(self, other: MPS) -> bool:
        """Checks if the tensors of this MPS are almost equal to the other MPS.

        Args:
            other (MPS): The other MPS to compare with.

        Returns:
            bool: True if all tensors of this tensor are almost equal to the
                other MPS, False otherwise.
        """
        if self.length != other.length:
            return False
        for i in range(self.length):
            if self.tensors[i].shape != other.tensors[i].shape:
                return False
            if not np.allclose(self.tensors[i], other.tensors[i]):
                return False
        return True

    def shift_orthogonality_center_right(self, current_orthogonality_center: int, decomposition: str = "QR") -> None:
        """Shifts orthogonality center right.

        This function performs a QR decomposition to shift the known current center to the right and move
        the canonical form. This is essential for maintaining efficient tensor network algorithms.

        Args:
            current_orthogonality_center (int): current center
            decomposition: Decides between QR or SVD decomposition. QR is faster, SVD allows bond dimension to reduce
                           Default is QR.
        """
        tensor = self.tensors[current_orthogonality_center]
        if decomposition == "QR" or current_orthogonality_center == self.length - 1:
            site_tensor, bond_tensor = right_qr(tensor)
            self.tensors[current_orthogonality_center] = site_tensor

            # If normalizing, we just throw away the R
            if current_orthogonality_center + 1 < self.length:
                self.tensors[current_orthogonality_center + 1] = oe.contract(
                    "ij, ajc->aic",
                    bond_tensor,
                    self.tensors[current_orthogonality_center + 1],
                )
        elif decomposition == "SVD":
            a, b = (
                self.tensors[current_orthogonality_center],
                self.tensors[current_orthogonality_center + 1],
            )
            merged = merge_two_site(a, b)
            a_new, b_new = split_two_site(
                merged,
                [a.shape[0], b.shape[0]],
                svd_distribution="right",
                trunc_mode="discarded_weight",
                threshold=1e-12,
                max_bond_dim=None,
            )
            (
                self.tensors[current_orthogonality_center],
                self.tensors[current_orthogonality_center + 1],
            ) = (a_new, b_new)

    def shift_orthogonality_center_left(self, current_orthogonality_center: int, decomposition: str = "QR") -> None:
        """Shifts orthogonality center left.

        This function flips the network, performs a right shift, then flips the network again.

        Args:
            current_orthogonality_center (int): current center
            decomposition: Decides between QR or SVD decomposition. QR is faster, SVD allows bond dimension to reduce
                Default is QR.
        """
        self.flip_network()
        self.shift_orthogonality_center_right(self.length - current_orthogonality_center - 1, decomposition)
        self.flip_network()

    def set_canonical_form(self, orthogonality_center: int, decomposition: str = "QR") -> None:
        """Sets canonical form of MPS.

        Left and right normalizes an MPS around a selected site.
        NOTE: Slow method compared to shifting based on known form and should be avoided.

        Args:
            orthogonality_center (int): site of matrix MPS around which we normalize
            decomposition: Type of decomposition. Default QR.
        """

        def sweep_decomposition(orthogonality_center: int, decomposition: str = "QR") -> None:
            for site, _ in enumerate(self.tensors):
                if site == orthogonality_center:
                    break
                self.shift_orthogonality_center_right(site, decomposition)

        sweep_decomposition(orthogonality_center, decomposition)
        self.flip_network()
        flipped_orthogonality_center = self.length - 1 - orthogonality_center
        sweep_decomposition(flipped_orthogonality_center, decomposition)
        self.flip_network()

    def normalize(self, form: str = "B", decomposition: str = "QR") -> None:
        """Normalize MPS.

        Normalize the network to a specified form.
        This method normalizes the network to the specified form. By default, it normalizes
        to form "B" (right canonical).
        The normalization process involves flipping the network, setting the canonical form with the
        orthogonality center at the last position, and shifting the orthogonality center to the rightmost position.

        NOTE: Slow method compared to shifting based on known form and should be avoided.

        Args:
            form (str): The form to normalize the network to. Default is "B".
            decomposition: Decides between QR or SVD decomposition. QR is faster, SVD allows bond dimension to reduce
                           Default is QR.
        """
        if form == "B":
            self.flip_network()

        self.set_canonical_form(orthogonality_center=self.length - 1, decomposition=decomposition)
        self.shift_orthogonality_center_right(self.length - 1, decomposition)

        if form == "B":
            self.flip_network()

    def compress(
        self,
        threshold: float,
        *,
        max_bond_dim: int | None = None,
        trunc_mode: TruncMode = "discarded_weight",
    ) -> None:
        """Compress in place via left-to-center and right-to-left two-site SVD sweeps.

        Args:
            threshold: SVD truncation threshold (e.g. ``sim_params.svd_threshold``).
            max_bond_dim: Optional cap on bond dimension.
            trunc_mode: ``"discarded_weight"`` or ``"relative"``.
        """
        if self.length == 1:
            return

        canonical = self.check_canonical_form()
        orth_center = canonical[0] if canonical and canonical[0] >= 0 else self.length - 1

        for site in range(orth_center):
            left_tensor = self.tensors[site]
            right_tensor = self.tensors[site + 1]
            merged = merge_two_site(left_tensor, right_tensor)
            left_new, right_new = split_two_site(
                merged,
                [left_tensor.shape[0], right_tensor.shape[0]],
                svd_distribution="right",
                trunc_mode=trunc_mode,
                threshold=threshold,
                max_bond_dim=max_bond_dim,
            )
            self.tensors[site] = left_new
            self.tensors[site + 1] = right_new

        self.flip_network()
        orth_flipped = self.length - 1 - orth_center
        for site in range(orth_flipped):
            left_tensor = self.tensors[site]
            right_tensor = self.tensors[site + 1]
            merged = merge_two_site(left_tensor, right_tensor)
            left_new, right_new = split_two_site(
                merged,
                [left_tensor.shape[0], right_tensor.shape[0]],
                svd_distribution="right",
                trunc_mode=trunc_mode,
                threshold=threshold,
                max_bond_dim=max_bond_dim,
            )
            self.tensors[site] = left_new
            self.tensors[site + 1] = right_new
        self.flip_network()

    def scalar_product(self, other: MPS, sites: int | list[int] | None = None) -> np.complex128:
        """Compute the scalar (inner) product between two Matrix Product States (MPS).

        The function contracts the corresponding tensors of two MPS objects. If no specific site is
        provided, the contraction is performed sequentially over all sites to yield the overall inner
        product. When a site is specified, only the tensors at that site are contracted.

        Args:
            other (MPS): The second Matrix Product State.
            sites: Optional site indices at which to compute the contraction. If None, the
                contraction is performed over all sites.

        Returns:
            np.complex128: The resulting scalar product as a complex number.

        Raises:
            ValueError: Invalid sites input
        """
        a_copy = copy.deepcopy(self)
        b_copy = copy.deepcopy(other)
        for i, tensor in enumerate(a_copy.tensors):
            a_copy.tensors[i] = np.conj(tensor)

        if sites is None:
            result = None
            for idx in range(self.length):
                # contract at each site into a 4-leg tensor
                theta = oe.contract("abc,ade->bdce", a_copy.tensors[idx], b_copy.tensors[idx])
                result = theta if idx == 0 else oe.contract("abcd,cdef->abef", result, theta)
            # squeeze down to scalar
            assert result is not None
            return np.complex128(np.squeeze(result))

        if isinstance(sites, int) or len(sites) == 1:
            if isinstance(sites, int):
                i = sites
            elif len(sites) == 1:
                i = sites[0]
            a = a_copy.tensors[i]
            b = b_copy.tensors[i]
            # sum over all three legs (p,l,r):
            val = oe.contract("ijk,ijk", a, b)
            return np.complex128(val)

        if len(sites) == 2:
            i, j = sites
            assert j == i + 1, "Only nearest-neighbor two-site overlaps supported."

            a_1 = a_copy.tensors[i]  # (p_i, l_i, r_i)
            b_1 = b_copy.tensors[i]  # (p_i, l_i, r'_i)
            a_2 = a_copy.tensors[j]  # (p_j, l_j=r_i, r_j)
            b_2 = b_copy.tensors[j]  # (p_j, l'_j=r'_i, r_j)

            # Contraction: a_1(a,b,c), a_2(d,c,e), b_1(a,b,f), b_2(d,f,e)
            val = oe.contract("abc,dce,abf,dfe->", a_1, a_2, b_1, b_2)
            return np.complex128(val)

        msg = f"Invalid `sites` argument: {sites!r}"
        raise ValueError(msg)

    def local_expect(self, operator: Observable, sites: int | list[int]) -> np.complex128:
        """Compute the local expectation value of an operator on an MPS.

        The function applies the given operator to the tensor at the specified site of a deep copy of the
        input MPS, then computes the scalar product between the original and the modified state at that site.
        This effectively calculates the expectation value of the operator at the specified site.

        Args:
            operator: The local operator to be applied.
            sites: The indices of the sites at which to evaluate the expectation value.

        Returns:
            np.complex128: The computed expectation value (typically, its real part is of interest).

        Notes:
            A deep copy of the state is used to prevent modifications to the original MPS.
        """
        temp_state = copy.deepcopy(self)
        num_sites = len(operator.sites) if isinstance(operator.sites, list) else 1
        if num_sites == 1:  # Local observable
            i = None
            if isinstance(sites, list):
                i = sites[0]
            elif isinstance(sites, int):
                i = sites

            if isinstance(operator.sites, list):
                assert operator.sites[0] == i, f"Operator sites mismatch {operator.sites[0]}, {i}"
            elif isinstance(operator.sites, int):
                assert operator.sites == i, f"Operator sites mismatch {operator.sites}, {i}"

            assert i is not None, f"Invalid type for 'sites': expected int or list[int], got {type(sites).__name__}"
            a = temp_state.tensors[i]
            temp_state.tensors[i] = oe.contract("ab, bcd->acd", operator.gate.matrix, a)

        elif num_sites == 2:  # Two-site correlator
            assert isinstance(sites, list)
            assert isinstance(operator.sites, list)
            i, j = sites

            assert operator.sites[0] == i, "Observable sites mismatch"
            assert operator.sites[1] == j, "Observable sites mismatch"
            assert operator.sites[0] < operator.sites[1], "Observable sites must be in ascending order."
            assert operator.sites[1] - operator.sites[0] == 1, (
                "Only nearest-neighbor observables are currently implemented."
            )
            a = temp_state.tensors[i]
            b = temp_state.tensors[j]
            d_i, left, _ = a.shape
            d_j, _, right = b.shape

            # 1) merge A,B into theta of shape (l, d_i*d_j, r)
            theta = np.tensordot(a, b, axes=(2, 1))  # (d_i, l, d_j, r)
            theta = theta.transpose(1, 0, 2, 3)  # (l, d_i, d_j, r)
            theta = theta.reshape(left, d_i * d_j, right)  # (l, d_i*d_j, r)

            # 2) apply operator on the combined phys index
            theta = oe.contract("ab, cbd->cad", operator.gate.matrix, theta)  # (l, d_i*d_j, r)
            theta = theta.reshape(left, d_i, d_j, right)  # back to (l, d_i, d_j, r)

            # 3) split via SVD
            theta_mat = theta.reshape(left * d_i, d_j * right)
            u_mat, s_vec, v_mat = linalg.svd(theta_mat, full_matrices=False)

            chi_new = len(s_vec)  # keep all singular values

            # build new A, B in (p, l, r) order
            u_tensor = u_mat.reshape(left, d_i, chi_new)  # (l, d_i, r_new)
            a_new = u_tensor.transpose(1, 0, 2)  # → (d_i, l, r_new)

            v_tensor = (np.diag(s_vec) @ v_mat).reshape(chi_new, d_j, right)  # (l_new, d_j, r)
            b_new = v_tensor.transpose(1, 0, 2)  # → (d_j, l_new, r)

            temp_state.tensors[i] = a_new
            temp_state.tensors[j] = b_new

        return self.scalar_product(temp_state, sites)

    def apply_local(self, observable: Observable) -> None:
        r"""Apply a one- or two-site local observable to this MPS in-place.

        Supports nearest-neighbor two-site gates and periodic-wrap gates on
        ``(L-1, 0)``. For ``L == 2`` with wrap ordering ``[1, 0]``, the gate is
        interpreted in ``|q_{L-1}, q_0>`` ordering and permuted to the merged
        nearest-neighbor basis on ``(0, 1)``.

        Args:
            observable: One-site (``2 x 2``) or two-site (``4 x 4``) observable.

        Raises:
            ValueError: If the observable is not one- or two-site local under the
                supported adjacency conventions.
        """

        def permuted_periodic_wrap(gate4: NDArray[np.complex128]) -> NDArray[np.complex128]:
            """Permute wrap gate from |q_{L-1}, q_0> to merged |q_0, q_{L-1}> ordering.

            Returns:
                Permuted 4x4 gate matrix.
            """
            p_perm = np.zeros((4, 4), dtype=np.complex128)
            for a in range(2):
                for b in range(2):
                    p_perm[2 * b + a, 2 * a + b] = 1.0
            return p_perm.conj().T @ gate4 @ p_perm

        def apply_two_site_nn_inplace(state: MPS, site_left: int, mat4: NDArray[np.complex128]) -> None:
            """Apply 4x4 gate to adjacent sites (site_left, site_left+1) in-place via SVD."""
            i, j = site_left, site_left + 1
            a = state.tensors[i]
            b = state.tensors[j]
            d_i, left, _ = a.shape
            d_j, _, right = b.shape

            theta = np.tensordot(a, b, axes=(2, 1)).transpose(1, 0, 2, 3)
            theta = theta.reshape(left, d_i * d_j, right)
            theta = oe.contract("ab, cbd->cad", mat4, theta).reshape(left, d_i, d_j, right)

            theta_mat = theta.reshape(left * d_i, d_j * right)
            u_mat, s_vec, v_mat = linalg.svd(theta_mat, full_matrices=False)

            u_tensor = u_mat.reshape(left, d_i, len(s_vec)).transpose(1, 0, 2)
            v_tensor = (np.diag(s_vec) @ v_mat).reshape(len(s_vec), d_j, right).transpose(1, 0, 2)

            state.tensors[i] = u_tensor
            state.tensors[j] = v_tensor

        def bubble_swaps_forward(state: MPS) -> None:
            """Move logical q_0 next to q_{L-1} via adjacent SWAPs."""
            sw = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128)
            for i in range(state.length - 2):
                apply_two_site_nn_inplace(state, i, sw)

        def bubble_swaps_backward(state: MPS) -> None:
            """Undo bubble_swaps_forward."""
            sw = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128)
            for i in reversed(range(state.length - 2)):
                apply_two_site_nn_inplace(state, i, sw)

        sites = [observable.sites] if isinstance(observable.sites, int) else observable.sites

        if observable.gate.matrix.shape[0] == 2:
            site = sites[0]
            self.tensors[site] = oe.contract("ab, bcd->acd", observable.gate.matrix, self.tensors[site])
            return

        if observable.gate.matrix.shape[0] == 4:
            i, j = int(sites[0]), int(sites[1])
            length = self.length

            if length == 2:
                if i == length - 1 and j == 0:
                    mat = np.asarray(observable.gate.matrix, dtype=np.complex128)
                    g_merged = permuted_periodic_wrap(mat)
                    apply_two_site_nn_inplace(self, 0, g_merged)
                    return
                i, j = min(i, j), max(i, j)
            elif (i == length - 1 and j == 0) or (i == 0 and j == length - 1):
                mat = np.asarray(observable.gate.matrix, dtype=np.complex128)
                bubble_swaps_forward(self)
                g_merged = permuted_periodic_wrap(mat)
                apply_two_site_nn_inplace(self, length - 2, g_merged)
                bubble_swaps_backward(self)
                return

            if j != i + 1:
                msg = "Only nearest-neighbor two-site observables are currently implemented."
                raise ValueError(msg)

            apply_two_site_nn_inplace(self, i, np.asarray(observable.gate.matrix, dtype=np.complex128))
            return

        msg = "Local observable must be one-site or nearest-neighbor two-site."
        raise ValueError(msg)

    def mixed_expectation(self, bra: MPS, observable: Observable) -> np.complex128:
        r"""Compute the mixed matrix element :math:`\langle\mathrm{bra}|O|\mathrm{ket}\rangle`.

        This applies ``observable`` to a deep copy of ``self`` (the ket) and contracts
        with ``bra`` using :meth:`scalar_product`.

        Args:
            bra: Bra MPS (left vector).
            observable: One-site or two-site local observable, same conventions as :meth:`apply_local`.

        Returns:
            The scalar contraction :math:`\langle\mathrm{bra}|O|\mathrm{ket}\rangle`.
        """
        ket_with_op = copy.deepcopy(self)
        ket_with_op.apply_local(observable)
        return bra.scalar_product(ket_with_op)

    def evaluate_observables(
        self,
        sim_params: AnalogSimParams | StrongSimParams,
        results: NDArray[np.float64],
        column_index: int = 0,
    ) -> None:
        """Evaluate and record expectation values of observables for a given MPS state.

        This method performs a deep copy of the current MPS (`self`) and iterates over
        the observables defined in the `sim_params` object. For each observable, it ensures
        the orthogonality center of the MPS is correctly positioned before computing the
        expectation value, which is then stored in the corresponding column of the `results` array.

        Parameters:
            sim_params: Simulation parameters containing a list of sorted observables.
            results: 2D array where results[observable_index, column_index] stores expectation values.
            column_index: The time or trajectory index indicating which column of the result array to fill.
        """
        temp_state = copy.deepcopy(self)
        last_site = 0
        for obs_index, observable in enumerate(sim_params.sorted_observables):
            if observable.gate.name in {"entropy", "schmidt_spectrum"}:
                assert isinstance(observable.sites, list), "Given metric requires a list of sites"
                assert len(observable.sites) == 2, "Given metric requires 2 sites to act on."
                max_site = max(observable.sites)
                min_site = min(observable.sites)
                assert max_site - min_site == 1, "Entropy and Schmidt cuts must be nearest neighbor."
                for s in observable.sites:
                    assert s in range(self.length), f"Observable acting on non-existing site: {s}"
                if observable.gate.name == "entropy":
                    results[obs_index, column_index] = self.get_entropy(observable.sites)
                elif observable.gate.name == "schmidt_spectrum":
                    results[obs_index, column_index] = self.get_schmidt_spectrum(observable.sites)

            elif observable.gate.name == "pvm":
                assert hasattr(observable.gate, "bitstring"), "Gate does not have attribute bitstring."
                bitstring = observable.gate.bitstring
                assert isinstance(bitstring, str)
                results[obs_index, column_index] = self.project_onto_bitstring(bitstring)

            else:
                idx = observable.sites[0] if isinstance(observable.sites, list) else observable.sites
                if idx > last_site:
                    for site in range(last_site, idx):
                        temp_state.shift_orthogonality_center_right(site)
                    last_site = idx
                results[obs_index, column_index] = temp_state.expect(observable)

    def expect(self, observable: Observable) -> np.float64:
        """Measurement of expectation value.

        Measure the expectation value of a given observable.

        Parameters:
            observable (Observable): The observable to measure. It must have a 'site' attribute indicating
            the site to measure and a 'name' attribute corresponding to a gate in the GateLibrary.

        Returns:
            np.float64: The real part of the expectation value of the observable.
        """
        sites_list = None
        if isinstance(observable.sites, int):
            sites_list = [observable.sites]
        elif isinstance(observable.sites, list):
            sites_list = observable.sites

        assert sites_list is not None, f"Invalid type in expect {type(observable.sites).__name__}"

        assert len(sites_list) < 3, "Only one- and two-site observables are currently implemented."

        for s in sites_list:
            assert s in range(self.length), f"Observable acting on non-existing site: {s}"

        exp = self.local_expect(observable, sites_list)

        assert exp.imag < 1e-13, f"Measurement should be real, '{exp.real:16f}+{exp.imag:16f}i'."
        return exp.real

    def measure_single_shot(self, basis: str = "Z", rng: np.random.Generator | None = None) -> int:
        """Perform a single-shot measurement on a Matrix Product State (MPS).

        This function simulates a projective measurement on an MPS. For each site, it computes the
        local reduced density matrix from the site's tensor, derives the probability distribution over
        basis states, and randomly selects an outcome. The overall measurement result is encoded as an
        integer corresponding to the measured bitstring.

        Args:
            basis: The basis to measure in. Options are "X", "Y", or "Z" (default).
            rng: Optional random number generator for outcome sampling.

        Returns:
            int: The measurement outcome represented as an integer.

        Raises:
            ValueError: If an invalid basis is provided.
        """
        temp_state = copy.deepcopy(self)
        bitstring = []

        basis = basis.upper()
        if basis == "Z":
            rotation = np.eye(2, dtype=complex)
        elif basis == "X":
            # H gate to rotate X to Z
            rotation = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        elif basis == "Y":
            # Rotate Y to Z: H Sdag (or equivalent)
            rotation = np.array([[1, -1j], [1, 1j]], dtype=complex) / np.sqrt(2)
        else:
            msg = f"Invalid basis: {basis}. Expected 'X', 'Y', or 'Z'."
            raise ValueError(msg)

        if rng is None:
            rng = np.random.default_rng()

        for site, tensor in enumerate(temp_state.tensors):
            # Rotate the tensor to the measurement basis
            # tensor shape is (p, l, r)
            rotated_tensor = oe.contract("ab, bcd->acd", rotation, tensor)

            reduced_density_matrix = oe.contract("abc, dbc->ad", rotated_tensor, np.conj(rotated_tensor))
            probabilities = np.diag(reduced_density_matrix).real
            chosen_index = rng.choice(len(probabilities), p=probabilities / np.sum(probabilities))
            bitstring.append(chosen_index)
            selected_state = np.zeros(len(probabilities))
            selected_state[chosen_index] = 1

            # Propagate the measurement to the next site.
            if site != self.length - 1:
                projected_tensor = oe.contract("a, acd->cd", selected_state, rotated_tensor)

                temp_state.tensors[site + 1] = (  # noqa: B909
                    1
                    / np.sqrt(probabilities[chosen_index])
                    * oe.contract("ab, cbd->cad", projected_tensor, temp_state.tensors[site + 1])
                )
        return sum(c << i for i, c in enumerate(bitstring))

    def measure_shots(self, shots: int, basis: str = "Z") -> dict[int, int]:
        """Perform multiple single-shot measurements on an MPS and aggregate the results.

        This function executes a specified number of measurement shots on the given MPS. For each shot,
        a single-shot measurement is performed, and the outcomes are aggregated into a histogram (dictionary)
        mapping basis states (represented as integers) to the number of times they were observed.

        Args:
            shots: The number of measurement shots to perform.
            basis: The basis to measure in. Options are "X", "Y", or "Z" (default).

        Returns:
            A dictionary where keys are measured basis states (as integers) and values are the corresponding counts.

        Notes:
            - When more than one shot is requested, measurements are parallelized using a ProcessPoolExecutor.
            - A progress bar (via tqdm) displays the progress of the measurement process.
        """
        results: dict[int, int] = {}
        if shots > 1:
            max_workers = max(1, multiprocessing.cpu_count() - 1)
            with (
                concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor,
                tqdm(total=shots, desc="Measuring shots", ncols=80) as pbar,
            ):
                futures = [executor.submit(self.measure_single_shot, basis) for _ in range(shots)]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results[result] = results.get(result, 0) + 1
                    pbar.update(1)
            return results
        basis_state = self.measure_single_shot(basis)
        results[basis_state] = results.get(basis_state, 0) + 1
        return results

    def measure(self, site: int, basis: str = "Z", rng: np.random.Generator | None = None) -> int:
        """Perform an in-place projective measurement on a single site of the MPS.

        This method modifies the MPS tensors to reflect the measurement outcome. It assumes the MPS
        is initially in a right-canonical form (orthogonality center at site 0) and shifts the center
        to the target site before measuring.

        Args:
            site: The index of the site to measure.
            basis: The basis to measure in. Options are "X", "Y", or "Z" (default).
            rng: Optional random number generator for outcome sampling.

        Returns:
            int: The measurement outcome (0 or 1 for qubits).

        Raises:
            ValueError: If an invalid site or basis is provided.
        """
        if site < 0 or site >= self.length:
            msg = f"Invalid site {site} for MPS of length {self.length}."
            raise ValueError(msg)

        # Shift orthogonality center to target site (assuming starts at 0)
        for i in range(site):
            self.shift_orthogonality_center_right(i)

        basis = basis.upper()
        if basis == "Z":
            rotation = np.eye(2, dtype=complex)
        elif basis == "X":
            rotation = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        elif basis == "Y":
            rotation = np.array([[1, -1j], [1, 1j]], dtype=complex) / np.sqrt(2)
        else:
            msg = f"Invalid basis: {basis}. Expected 'X', 'Y', or 'Z'."
            raise ValueError(msg)

        tensor = self.tensors[site]
        # Rotate the tensor to the measurement basis
        rotated_tensor = oe.contract("ab, bcd->acd", rotation, tensor)

        # Compute reduced density matrix at the orthogonality center
        reduced_density_matrix = oe.contract("abc, dbc->ad", rotated_tensor, np.conj(rotated_tensor))
        probabilities = np.diag(reduced_density_matrix).real.copy()

        # Ensure probabilities are normalized (site is center)
        norm_factor = np.sum(probabilities)
        probabilities /= norm_factor

        if rng is None:
            rng = np.random.default_rng()

        chosen_index = rng.choice(len(probabilities), p=probabilities)

        selected_state = np.zeros(len(probabilities), dtype=complex)
        selected_state[chosen_index] = 1.0

        # Project the rotated tensor onto the selected outcome
        projected_rotated_tensor = oe.contract("a, acd->cd", selected_state, rotated_tensor)

        # Rotate back to original basis for the new tensor
        original_basis_selection = oe.contract("ab, a->b", np.conj(rotation), selected_state)

        # Normalize and update the site tensor
        self.tensors[site] = (1.0 / np.sqrt(probabilities[chosen_index])) * oe.contract(
            "a, cd->acd", original_basis_selection, projected_rotated_tensor
        )

        return int(chosen_index)

    def project_onto_bitstring(self, bitstring: str) -> np.complex128:
        """Projection-valued measurement.

        Project the MPS onto a given bitstring in the computational basis
        and return the squared norm (i.e., probability of that outcome).

        This is equivalent to computing ⟨bitstring|ψ⟩⟨ψ|bitstring⟩.

        Args:
            bitstring (str): Bitstring to project onto (little-endian: site 0 is first char).

        Returns:
            float: Probability of obtaining the given bitstring under projective measurement.
        """
        assert len(bitstring) == self.length, "Bitstring length must match number of sites"
        temp_state = copy.deepcopy(self)
        total_norm = 1.0

        for site, char in enumerate(bitstring):
            state_index = int(char)
            tensor = temp_state.tensors[site]
            local_dim = self.physical_dimensions[site]
            assert 0 <= state_index < local_dim, f"Invalid state index {state_index} at site {site}"

            selected_state = np.zeros(local_dim)
            selected_state[state_index] = 1

            # Project tensor
            projected_tensor = oe.contract("a, acd->cd", selected_state, tensor)

            # Compute norm of projected tensor
            norm = float(np.linalg.norm(projected_tensor))
            if norm == 0:
                return np.complex128(0.0)
            total_norm *= norm

            # Normalize and propagate
            if site != self.length - 1:
                temp_state.tensors[site + 1] = (
                    1 / norm * oe.contract("ab, cbd->cad", projected_tensor, temp_state.tensors[site + 1])
                )

        return np.complex128(total_norm**2)

    def norm(self, site: int | None = None) -> np.float64:
        """Norm calculation.

        Calculate the norm of the state.

        Args:
            site: The specific site to calculate the norm from. If ``None``, the
                norm is calculated for the entire network.

        Returns:
            The norm of the state or the specified site.
        """
        if site is not None:
            return self.scalar_product(self, site).real
        return self.scalar_product(self).real

    def check_if_valid_mps(self) -> None:
        """MPS validity check.

        Check if the current tensor network is a valid Matrix Product State (MPS).

        This method verifies that the bond dimensions between consecutive tensors
        in the network are consistent. Specifically, it checks that the second
        dimension of each tensor matches the third dimension of the previous tensor.
        """
        right_bond = self.tensors[0].shape[2]
        for tensor in self.tensors[1::]:
            assert tensor.shape[1] == right_bond
            right_bond = tensor.shape[2]

    def check_canonical_form(self) -> list[int]:
        """Checks canonical form of MPS.

        Checks what canonical form a Matrix Product State (MPS) is in, if any.
        This method verifies if the MPS is in left-canonical form, right-canonical
        form, or mixed-canonical form. It returns a list indicating the canonical
        form status:

        - ``[0]`` if the MPS is in left-canonical form.
        - ``[self.length - 1]`` if the MPS is in right-canonical form.
        - ``[index]`` if the MPS is in mixed-canonical form, where ``index`` is the
          position where the form changes.
        - ``[-1]`` if the MPS is not in any canonical form.

        Returns:
            A list indicating the canonical form status of the MPS.
        """
        a = copy.deepcopy(self.tensors)
        for i, tensor in enumerate(self.tensors):
            a[i] = np.conj(tensor)
        b = self.tensors
        a_truth = [False for _ in range(self.length)]
        b_truth = [False for _ in range(self.length)]

        # Find the first index where the left canonical form is not satisfied.
        # We choose the rightmost index in case even that one fulfills the condition
        for i in range(self.length):
            mat = oe.contract("ijk, ijl->kl", a[i], b[i])
            test_identity = np.eye(mat.shape[0], dtype=complex)
            if np.allclose(mat, test_identity):
                a_truth[i] = True

        # Find the last index where the right canonical form is not satisfied.
        # We choose the leftmost index in case even that one fulfills the condition
        for i in reversed(range(self.length)):
            mat = oe.contract("ijk, ilk->jl", b[i], a[i])
            test_identity = np.eye(mat.shape[0], dtype=complex)
            if np.allclose(mat, test_identity):
                b_truth[i] = True

        mixed_truth = [False for _ in range(self.length)]
        for i in range(self.length):
            if all(a_truth[:i]) and all(b_truth[i + 1 :]):
                mixed_truth[i] = True

        sites = []
        for i, val in enumerate(mixed_truth):
            if val:
                sites.append(i)

        return sites

    def to_vec(self) -> NDArray[np.complex128]:
        r"""Converts the MPS to a full state vector representation.

        Returns:
            A one-dimensional NumPy array of length :math:`\prod_{\ell=1}^L d_\ell`
            representing the state vector.
        """
        # Start with the first tensor.
        # Assume each tensor has shape (d, chi_left, chi_right) with chi_left=1 for the first tensor.
        self.flip_network()
        vec = self.tensors[0]  # shape: (d_1, 1, chi_1)

        # Contract sequentially with the remaining tensors.
        for i in range(1, self.length):
            # Contract the last bond of vec with the middle index (left bond) of the next tensor.
            vec = np.tensordot(vec, self.tensors[i], axes=([-1], [1]))
            # After tensordot, if vec had shape (..., chi_i) and the new tensor has shape (d_{i+1}, chi_i, chi_{i+1}),
            # then vec now has shape (..., d_{i+1}, chi_{i+1}).
            # Reshape to merge all physical indices into one index.
            new_shape = (-1, vec.shape[-1])
            vec = np.reshape(vec, new_shape)
        self.flip_network()
        # At the end, the final bond dimension should be 1.
        vec = np.squeeze(vec, axis=-1)
        # Flatten the resulting multi-index into a one-dimensional state vector.
        return vec.flatten()
