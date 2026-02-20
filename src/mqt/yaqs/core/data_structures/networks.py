# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tensor Network Data Structures.

This module implements classes for representing quantum states and operators using tensor networks.
It defines the Matrix Product State (MPS) and Matrix Product Operator (MPO) classes, along with various
methods for network normalization, canonicalization, measurement, and validity checks. These classes and
utilities are essential for simulating quantum many-body systems using tensor network techniques.
"""

from __future__ import annotations

import concurrent.futures
import copy
import multiprocessing
import re
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import opt_einsum as oe
import scipy.sparse
from numpy.typing import NDArray
from tqdm import tqdm

from ..libraries.gate_library import Destroy
from ..methods.decompositions import right_qr, two_site_svd

if TYPE_CHECKING:
    from .simulation_parameters import AnalogSimParams, Observable, StrongSimParams


class MPS:
    """Matrix Product State (MPS) class for representing quantum states.

    This class forms the basis of the MPS used in YAQS simulations.
    The index order is (sigma, chi_l-1, chi_l).

    Attributes:
    length (int): The number of sites in the MPS.
    tensors (list[NDArray[np.complex128]]): List of rank-3 tensors representing the MPS.
    physical_dimensions (list[int]): List of physical dimensions for each site.
    flipped (bool): Indicates if the network has been flipped.

    Methods:
    __init__(length: int, tensors: list[NDArray[np.complex128]] | None = None,
                physical_dimensions: list[int] | None = None, state: str = "zeros") -> None:
        Initializes the MPS with given length, tensors, physical dimensions, and initial state.
    pad_bond_dimension():
        Pads bond dimension with zeros
    get_max_bond() -> int:
        Returns the maximum bond dimension in the MPS.
    flip_network() -> None:
        Flips the bond dimensions in the network to allow operations from right to left.
    shift_orthogonality_center_right(current_orthogonality_center: int) -> None:
        Left and right normalizes the MPS around a selected site, shifting the orthogonality center to the right.
    shift_orthogonality_center_left(current_orthogonality_center: int) -> None:
        Left and right normalizes the MPS around a selected site, shifting the orthogonality center to the left.
    set_canonical_form(orthogonality_center: int) -> None:
        Left and right normalizes the MPS around a selected site to set it in canonical form.
    normalize(form: str = "B") -> None:
        Normalizes the MPS in the specified form.
    measure(observable: Observable) -> np.float64:
        Measures the expectation value of an observable at a specified site.
    norm(site: int | None = None) -> np.float64:
        Computes the norm of the MPS, optionally at a specified site.
    write_tensor_shapes() -> None:
        Writes the shapes of the tensors in the MPS.
    check_if_valid_mps() -> None:
        Checks if the MPS is valid by verifying bond dimensions.
    check_canonical_form() -> list[int]:
        Checks the canonical form of the MPS and returns the orthogonality center(s).
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
                - "basis": Initializes a qubit in an input computational basis.
                Default is "zeros".
            pad: Pads the state with extra zeros to increase bond dimension. Can increase numerical stability.
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

        # Create d-level |0> state
        if not tensors:
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
        if pad is not None:
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

        Enlarge every internal bond up to
            min(target_dim, 2**exp)
        where exp = min(bond_index+1, L-1-bond_index).
        The first tensor keeps a left bond of 1, the last tensor a right bond of 1.
        After padding the state is renormalised (canonicalised).

        Args:
        target_dim : int
            The desired bond dimension for the internal bonds.

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

        s = np.linalg.svd(theta_mat, full_matrices=False, compute_uv=False)
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

        _, s_vec, _ = np.linalg.svd(theta_mat, full_matrices=False)

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
            a_new, b_new = two_site_svd(a, b, threshold=1e-12, max_bond_dim=None)
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

    def truncate(self, threshold: float = 1e-12, max_bond_dim: int | None = None) -> None:
        """In-place MPS truncation via repeated two-site SVDs."""
        orth_center = self.check_canonical_form()[0]
        if self.length == 1:
            return

        # ——— left­-to-­center sweep ———
        for i in range(orth_center):
            a, b = self.tensors[i], self.tensors[i + 1]
            a_new, b_new = two_site_svd(a, b, threshold, max_bond_dim)
            self.tensors[i], self.tensors[i + 1] = a_new, b_new

        # flip the network and sweep back
        self.flip_network()
        orth_flipped = self.length - 1 - orth_center
        for i in range(orth_flipped):
            a, b = self.tensors[i], self.tensors[i + 1]
            a_new, b_new = two_site_svd(a, b, threshold, max_bond_dim)
            self.tensors[i], self.tensors[i + 1] = a_new, b_new

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
        if operator.gate.matrix.shape[0] == 2:  # Local observable
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

        elif operator.gate.matrix.shape[0] == 4:  # Two-site correlator
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
            u_mat, s_vec, v_mat = np.linalg.svd(theta_mat, full_matrices=False)

            chi_new = len(s_vec)  # keep all singular values

            # build new A, B in (p, l, r) order
            u_tensor = u_mat.reshape(left, d_i, chi_new)  # (l, d_i, r_new)
            a_new = u_tensor.transpose(1, 0, 2)  # → (d_i, l, r_new)

            v_tensor = (np.diag(s_vec) @ v_mat).reshape(chi_new, d_j, right)  # (l_new, d_j, r)
            b_new = v_tensor.transpose(1, 0, 2)  # → (d_j, l_new, r)

            temp_state.tensors[i] = a_new
            temp_state.tensors[j] = b_new

        return self.scalar_product(temp_state, sites)

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
            if observable.gate.name == "runtime_cost":
                results[obs_index, column_index] = self.get_cost()
            elif observable.gate.name == "max_bond":
                results[obs_index, column_index] = self.get_max_bond()
            elif observable.gate.name == "total_bond":
                results[obs_index, column_index] = self.get_total_bond()
            elif observable.gate.name in {"entropy", "schmidt_spectrum"}:
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
                results[obs_index, column_index] = self.project_onto_bitstring(observable.gate.bitstring)

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

    def measure_single_shot(self) -> int:
        """Perform a single-shot measurement on a Matrix Product State (MPS).

        This function simulates a projective measurement on an MPS. For each site, it computes the
        local reduced density matrix from the site's tensor, derives the probability distribution over
        basis states, and randomly selects an outcome. The overall measurement result is encoded as an
        integer corresponding to the measured bitstring.

        Returns:
            int: The measurement outcome represented as an integer.
        """
        temp_state = copy.deepcopy(self)
        bitstring = []
        for site, tensor in enumerate(temp_state.tensors):
            reduced_density_matrix = oe.contract("abc, dbc->ad", tensor, np.conj(tensor))
            probabilities = np.diag(reduced_density_matrix).real
            rng = np.random.default_rng()
            chosen_index = rng.choice(len(probabilities), p=probabilities)
            bitstring.append(chosen_index)
            selected_state = np.zeros(len(probabilities))
            selected_state[chosen_index] = 1
            # Multiply state: project the tensor onto the selected state.
            projected_tensor = oe.contract("a, acd->cd", selected_state, tensor)
            # Propagate the measurement to the next site.
            if site != self.length - 1:
                temp_state.tensors[site + 1] = (  # noqa: B909
                    1
                    / np.sqrt(probabilities[chosen_index])
                    * oe.contract("ab, cbd->cad", projected_tensor, temp_state.tensors[site + 1])
                )
        return sum(c << i for i, c in enumerate(bitstring))

    def measure_shots(self, shots: int) -> dict[int, int]:
        """Perform multiple single-shot measurements on an MPS and aggregate the results.

        This function executes a specified number of measurement shots on the given MPS. For each shot,
        a single-shot measurement is performed, and the outcomes are aggregated into a histogram (dictionary)
        mapping basis states (represented as integers) to the number of times they were observed.

        Args:
            shots: The number of measurement shots to perform.

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
                futures = [executor.submit(self.measure_single_shot) for _ in range(shots)]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results[result] = results.get(result, 0) + 1
                    pbar.update(1)
            return results
        basis_state = self.measure_single_shot()
        results[basis_state] = results.get(basis_state, 0) + 1
        return results

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

        Parameters:
        site (int | None): The specific site to calculate the norm from. If None, the norm is calculated for
                           the entire network.

        Returns:
        np.float64: The norm of the state or the specified site.
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
        This method verifies if the MPS is in left-canonical form, right-canonical form, or mixed-canonical form.
        It returns a list indicating the canonical form status:
        - [0] if the MPS is in left-canonical form.
        - [self.length - 1] if the MPS is in right-canonical form.
        - [index] if the MPS is in mixed-canonical form, where `index` is the position where the form changes.
        - [-1] if the MPS is not in any canonical form.

        Parameters:
        epsilon (float): Tolerance for numerical comparisons. Default is 1e-12.

        Returns:
            list[int]: A list indicating the canonical form status of the MPS.
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
                A one-dimensional NumPy array of length \(\prod_{\ell=1}^L d_\ell\)
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


ComplexTensor = NDArray[np.complex128]


class MPO:
    """Matrix Product Operator (MPO) for YAQS tensor-network simulations.

    An MPO represents a linear operator on a 1D lattice as a chain of local tensors.
    YAQS stores each site tensor with index order::

        (phys_out, phys_in, chi_left, chi_right)

    where ``phys_out``/``phys_in`` are the physical operator legs and
    ``chi_left``/``chi_right`` are the virtual (bond) dimensions.

    Construction
    -----------
    Use classmethod factories to build common Hamiltonians or custom operators:

    - ``MPO.ising(...)`` / ``MPO.heisenberg(...)``: qubit Pauli Hamiltonians.
    - ``MPO.hamiltonian(...)``: generic one-/two-body Pauli interactions.
    - ``MPO.coupled_transmon(...)``: alternating qubit/resonator chain MPO.
    - ``from_pauli_sum(...)``: in-place build from a sum of Pauli-string terms.
    - ``identity(...)``, ``custom(...)``, ``finite_state_machine(...)``: in-place builders.

    Operations
    ----------
    - ``compress(...)``: SVD-based bond compression sweeps.
    - ``rotate(...)``: swap physical legs (optionally conjugating).

    Conversion / checks
    -------------------
    - ``to_mps()`` / ``to_matrix()``: convert to an MPS or dense matrix.
    - ``check_if_valid_mpo()``: structural bond-dimension consistency check.
    - ``check_if_identity(...)``: heuristic identity check (qubit systems).

    Notes:
    -----
    Some constructors (e.g. Pauli-string builders) currently require
    ``physical_dimension == 2``.
    """

    _PAULI_2: ClassVar[dict[str, np.ndarray]] = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }

    _VALID: ClassVar[frozenset[str]] = frozenset(_PAULI_2.keys())
    _PAULI_TOKEN_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"\b([IXYZ])\s*(\d+)\b",
        flags=re.IGNORECASE,
    )

    tensors: list[ComplexTensor]
    length: int
    physical_dimension: int

    @classmethod
    def hamiltonian(
        cls,
        *,
        length: int,
        two_body: list[tuple[complex | float, str, str]] | None = None,
        one_body: list[tuple[complex | float, str]] | None = None,
        bc: str = "open",
        physical_dimension: int = 2,
        tol: float = 1e-12,
        max_bond_dim: int | None = None,
        n_sweeps: int = 2,
    ) -> MPO:
        """Construct an MPO from specified one- and two-body Pauli interactions.

        Builds a Hamiltonian MPO by expanding the provided interaction lists into
        a sum of Pauli strings and delegating construction to ``from_pauli_sum``.
        Nearest-neighbor two-body terms are generated according to the chosen
        boundary condition.

        Args:
            length: Number of sites (L).
            two_body: List of ``(coeff, op_i, op_j)`` nearest-neighbor interactions,
                where operators are given as Pauli labels (e.g. ``"X"``, ``"Z"``).
            one_body: List of ``(coeff, op)`` on-site terms.
            bc: Boundary condition, either ``"open"`` or ``"periodic"``.
            physical_dimension: Local Hilbert-space dimension (only ``2`` supported).
            tol: SVD truncation threshold used during compression.
            max_bond_dim: Optional hard cap on the MPO bond dimension.
            n_sweeps: Number of compression sweeps (>= 0).

        Returns:
            MPO representing the specified Hamiltonian.

        Raises:
            ValueError: If ``length <= 0``, an invalid boundary condition is given,
                or an operator label is not a valid Pauli operator.
        """
        if length <= 0:
            msg = "L must be positive."
            raise ValueError(msg)
        if bc not in {"open", "periodic"}:
            msg = "bc must be 'open' or 'periodic'."
            raise ValueError(msg)

        two_body = two_body or []
        one_body = one_body or []

        def op(x: str) -> str:
            x = str(x).upper()
            if x not in cls._VALID:
                msg = f"Invalid operator {x!r}; expected one of {sorted(cls._VALID)}."
                raise ValueError(msg)
            return x

        terms: list[tuple[complex | float, str]] = []

        bonds = range(length) if bc == "periodic" else range(length - 1)
        for c, a, b in two_body:
            a_op, b_op = op(a), op(b)
            for i in bonds:
                j = (i + 1) % length
                terms.append((c, f"{a_op}{i} {b_op}{j}"))

        for c, a in one_body:
            a_op = op(a)
            terms.extend((c, f"{a_op}{i}") for i in range(length))

        mpo = cls()
        mpo.from_pauli_sum(
            terms=terms,
            length=length,
            physical_dimension=physical_dimension,
            tol=tol,
            max_bond_dim=max_bond_dim,
            n_sweeps=n_sweeps,
        )
        return mpo

    @classmethod
    def ising(
        cls,
        length: int,
        J: float,  # noqa: N803
        g: float,
        *,
        bc: str = "open",
        physical_dimension: int = 2,
        tol: float = 1e-12,
        max_bond_dim: int | None = None,
        n_sweeps: int = 2,
    ) -> MPO:
        """Construct an Ising Hamiltonian MPO.

        Args:
            length: Number of sites.
            J: ZZ coupling strength (Hamiltonian includes -J Σ Z_i Z_{i+1}).
            g: X field strength (Hamiltonian includes -g Σ X_i).
            bc: "open" or "periodic".
            physical_dimension: Local dimension (Ising Pauli builder requires 2).
            tol: SVD truncation threshold used during compression.
            max_bond_dim: Optional hard cap for MPO bond dimension during compression.
            n_sweeps: Number of compression sweeps.

        Returns:
            An MPO representing the Ising Hamiltonian.
        """
        return cls.hamiltonian(
            length=length,
            two_body=[(-J, "Z", "Z")],
            one_body=[(-g, "X")],
            bc=bc,
            physical_dimension=physical_dimension,
            tol=tol,
            max_bond_dim=max_bond_dim,
            n_sweeps=n_sweeps,
        )

    @classmethod
    def heisenberg(
        cls,
        length: int,
        Jx: float,  # noqa: N803
        Jy: float,  # noqa: N803
        Jz: float,  # noqa: N803
        h: float = 0.0,
        *,
        bc: str = "open",
        physical_dimension: int = 2,
        tol: float = 1e-12,
        max_bond_dim: int | None = None,
        n_sweeps: int = 2,
    ) -> MPO:
        """Construct a Heisenberg (XYZ) Hamiltonian MPO.

        Args:
            length: Number of sites.
            Jx: XX coupling strength (Hamiltonian includes -Jx Σ X_i X_{i+1}).
            Jy: YY coupling strength (Hamiltonian includes -Jy Σ Y_i Y_{i+1}).
            Jz: ZZ coupling strength (Hamiltonian includes -Jz Σ Z_i Z_{i+1}).
            h: Z field strength (Hamiltonian includes -h Σ Z_i).
            bc: "open" or "periodic".
            physical_dimension: Local dimension (Pauli builder requires 2).
            tol: SVD truncation threshold used during compression.
            max_bond_dim: Optional hard cap for MPO bond dimension during compression.
            n_sweeps: Number of compression sweeps.

        Returns:
            An MPO representing the Heisenberg Hamiltonian.
        """
        return cls.hamiltonian(
            length=length,
            two_body=[(-Jx, "X", "X"), (-Jy, "Y", "Y"), (-Jz, "Z", "Z")],
            one_body=[(-h, "Z")] if h != 0 else [],
            bc=bc,
            physical_dimension=physical_dimension,
            tol=tol,
            max_bond_dim=max_bond_dim,
            n_sweeps=n_sweeps,
        )

    @classmethod
    def coupled_transmon(
        cls,
        length: int,
        qubit_dim: int,
        resonator_dim: int,
        qubit_freq: float,
        resonator_freq: float,
        anharmonicity: float,
        coupling: float,
    ) -> MPO:
        """Coupled Transmon MPO.

        Initializes an MPO representation of a 1D chain of coupled transmon qubits
        and resonators.

        The chain alternates between transmon qubits (even indices) and resonators
        (odd indices), with each qubit coupled to its neighboring resonators via
        dipole-like interaction terms.

        Parameters:
            length: Total number of sites in the chain (should be even).
                        Qubit sites are placed at even indices, resonators at odd.
            qubit_dim: Local Hilbert space dimension of each transmon qubit.
            resonator_dim: Local Hilbert space dimension of each resonator.
            qubit_freq: Bare frequency of the transmon qubits.
            resonator_freq: Bare frequency of the resonators.
            anharmonicity: Strength of the anharmonic (nonlinear) term
                                for each transmon, typically negative.
            coupling : Strength of the qubit-resonator coupling term.

        Returns:
            An MPO instance representing the coupled transmon-resonator chain.

        Notes:
            - The Hamiltonian for each qubit is modeled as a Duffing oscillator:
                H_q = ω_q * n_q + (alpha/2) * n_q (n_q - 1)
            - Each resonator is a harmonic oscillator:
                H_r = ω_r * n_r
            - The interaction is implemented via dipole coupling:
                H_int = g * (b + b†)(a + a†)
            - The MPO bond dimension is 4.
        """
        b = Destroy(qubit_dim)
        b_dag = b.dag()
        a = Destroy(resonator_dim)
        a_dag = a.dag()

        id_q = np.eye(qubit_dim, dtype=complex)
        id_r = np.eye(resonator_dim, dtype=complex)
        zero_q = np.zeros_like(id_q)
        zero_r = np.zeros_like(id_r)

        n_q = b_dag.matrix @ b.matrix
        n_r = a_dag.matrix @ a.matrix
        h_q = qubit_freq * n_q + (anharmonicity / 2) * n_q @ (n_q - id_q)
        h_r = resonator_freq * n_r

        x_q = b_dag.matrix + b.matrix
        x_r = a_dag.matrix + a.matrix

        tensors: list[np.ndarray] = []

        for i in range(length):
            if i % 2 == 0:
                # Qubit site
                if i == 0:
                    tensor = np.array(
                        [
                            [
                                h_q,
                                id_q,
                                coupling * x_q,
                                id_q,
                            ]
                        ],
                        dtype=object,
                    )  # (1, 4, dq, dq)

                elif i == length - 1:
                    tensor = np.array(
                        [
                            [id_q],
                            [coupling * x_q],
                            [id_q],
                            [h_q],
                        ],
                        dtype=object,
                    )  # (4, 1, dq, dq)

                else:
                    tensor = np.empty((4, 4, qubit_dim, qubit_dim), dtype=object)
                    tensor[:, :] = [[zero_q for _ in range(4)] for _ in range(4)]
                    tensor[0, 0] = h_q
                    tensor[0, 1] = id_q
                    tensor[0, 2] = coupling * x_q  # right resonator
                    tensor[1, 3] = coupling * x_q  # left resonator
                    tensor[0, 3] = id_q
                    tensor[3, 3] = id_q
            else:
                # Resonator site
                tensor = np.empty((4, 4, resonator_dim, resonator_dim), dtype=object)
                tensor[:, :] = [[zero_r for _ in range(4)] for _ in range(4)]
                tensor[0, 0] = id_r
                tensor[1, 2] = h_r
                tensor[2, 0] = x_r
                tensor[3, 1] = x_r
                tensor[3, 3] = id_r

            # (left, right, phys_out, phys_in) -> (phys_out, phys_in, left, right)
            tensors.append(np.transpose(tensor, (2, 3, 0, 1)))

        mpo = cls()
        mpo.tensors = tensors
        mpo.length = length

        # Backward-compat: single attribute even though dims alternate.
        mpo.physical_dimension = qubit_dim

        assert mpo.check_if_valid_mpo(), "MPO initialized wrong"
        return mpo

    @classmethod
    def bose_hubbard(
        cls,
        length: int,
        local_dim: int,
        omega: float,
        hopping_j: float,
        hubbard_u: float,
    ) -> MPO:
        """Bose-Hubbard Hamiltonian.

        Initializes an MPO representation of a Bose-Hubbard Hamiltonian.

        Parameters:
            length: Total number of sites in the chain.
            local_dim: Local Hilbert space dimension of each site. Maximally
                                local_dim - 1 particles per site.
            omega: Frequency of a site.
            hopping_j: Hopping constant between sites.
            hubbard_u: Repulsive onsite Hubbard interaction on each site.

        Returns:
            An MPO instance representing the Hamiltonian.

        Raises:
            ValueError: If ``length <= 0``.

        Notes:
            - The Hamiltonian for each site is modeled as a Duffing oscillator:
                H = sum_i ω * n_i + U/2 * n_i (n_i - 1) + J * (adag_i a_{i+1} + h.c.)
            - The MPO bond dimension is D=4.
        """
        if length <= 0:
            msg = "length must be positive."
            raise ValueError(msg)

        a = Destroy(local_dim).matrix
        a_dag = Destroy(local_dim).dag().matrix

        id_boson = np.eye(local_dim, dtype=complex)
        zero = np.zeros_like(id_boson, dtype=complex)

        n = a_dag @ a
        h_loc = 0.5 * hubbard_u * (n @ (n - id_boson)) + omega * n

        tensors: list[np.ndarray] = []

        # channels: 0 = start/identity, 1 = carries adag, 2 = carries a, 3 = end/accumulator
        tensor = np.empty((4, 4, local_dim, local_dim), dtype=object)
        tensor[:, :] = [[zero for _ in range(4)] for _ in range(4)]
        tensor[0, 0] = id_boson
        tensor[0, 1] = a_dag
        tensor[0, 2] = a

        tensor[0, 3] = h_loc

        tensor[1, 3] = -hopping_j * a  # completes adag_i * a_{i+1}
        tensor[2, 3] = -hopping_j * a_dag
        tensor[3, 3] = id_boson

        # build the full tensor list
        tensors = [np.transpose(tensor.copy(), (2, 3, 0, 1)).astype(np.complex128) for _ in range(length)]

        # Left boundary: take only row 0
        tensors[0] = np.transpose(tensor.copy(), (2, 3, 0, 1))[:, :, 0:1, :].astype(np.complex128)

        # Right boundary: take only col 3
        tensors[-1] = np.transpose(tensor.copy(), (2, 3, 0, 1))[:, :, :, 3:4].astype(np.complex128)

        mpo = cls()
        mpo.tensors = tensors
        mpo.length = length

        # Backward-compat: single attribute even though dims alternate.
        mpo.physical_dimension = local_dim

        assert mpo.check_if_valid_mpo(), "MPO initialized wrong"
        return mpo

    def identity(self, length: int, physical_dimension: int = 2) -> None:
        """Initialize identity MPO.

        Initializes the network with identity matrices.

        Parameters:
            length (int): The number of identity matrices to initialize.
            physical_dimension (int, optional): The physical dimension of the identity matrices. Default is 2.

        """
        mat = np.eye(2, dtype=np.complex128)
        mat = np.expand_dims(mat, (2, 3))
        self.length = length
        self.physical_dimension = physical_dimension

        self.tensors = []
        for _ in range(length):
            self.tensors.append(mat)

    def finite_state_machine(
        self,
        length: int,
        left_bound: NDArray[np.complex128],
        inner: NDArray[np.complex128],
        right_bound: NDArray[np.complex128],
    ) -> None:
        """Custom Hamiltonian from finite state machine MPO.

        Initialize a custom Hamiltonian as a Matrix Product Operator (MPO).
        This method sets up the Hamiltonian using the provided boundary and inner tensors.
        The tensors are transposed to match the expected shape for MPOs.

        Args:
            length (int): The number of tensors in the MPO.
            left_bound (NDArray[np.complex128]): The tensor at the left boundary.
            inner (NDArray[np.complex128]): The tensor for the inner sites.
            right_bound (NDArray[np.complex128]): The tensor at the right boundary.
        """
        self.tensors = [left_bound] + [inner] * (length - 2) + [right_bound]
        for i, tensor in enumerate(self.tensors):
            # left, right, sigma, sigma'
            self.tensors[i] = np.transpose(tensor, (2, 3, 0, 1))
        assert self.check_if_valid_mpo(), "MPO initialized wrong"
        self.length = len(self.tensors)
        self.physical_dimension = self.tensors[0].shape[0]

    def custom(self, tensors: list[NDArray[np.complex128]], *, transpose: bool = True) -> None:
        """Custom MPO from tensors.

        Initialize the custom MPO (Matrix Product Operator) with the given tensors.

        Args:
            tensors: A list of tensors to initialize the MPO.
            transpose: If True, transpose each tensor to the order (2, 3, 0, 1). Default is True.

        Notes:
            This method sets the tensors, optionally transposes them, checks if the MPO is valid,
            and initializes the length and physical dimension of the MPO.
        """
        self.tensors = tensors
        if transpose:
            for i, tensor in enumerate(self.tensors):
                # left, right, sigma, sigma'
                self.tensors[i] = np.transpose(tensor, (2, 3, 0, 1))
        assert self.check_if_valid_mpo(), "MPO initialized wrong"
        self.length = len(self.tensors)
        self.physical_dimension = tensors[0].shape[0]

    def from_pauli_sum(
        self,
        *,
        terms: list[tuple[complex | float, str]],
        length: int,
        physical_dimension: int = 2,
        tol: float = 1e-12,
        max_bond_dim: int | None = None,
        n_sweeps: int = 2,
    ) -> None:
        """Build this MPO from a sum of Pauli-string terms.

        Each term is given as ``(coeff, spec)`` where ``spec`` is a string like
        ``"Z0 Z1"``, ``"X7"``, or ``""`` for the identity. Terms are assembled by
        constructing a finite state machine (FSM) that represents the sum of terms
        directly, resulting in an optimal or near-optimal bond dimension without
        intermediate compression steps.

        Args:
            terms: List of ``(coefficient, spec)`` Pauli terms.
            length: Number of sites (L).
            physical_dimension: Local dimension (only ``2`` is supported).
            tol: SVD truncation threshold used during final compression.
            max_bond_dim: Optional hard cap on the kept MPO bond dimension.
            n_sweeps: Number of compression sweeps (>= 0).

        Raises:
            ValueError: If ``length <= 0``, ``physical_dimension != 2``, a site index is
                out of bounds, an operator label is invalid, or a term spec is malformed.

        Notes:
            The resulting MPO represents the sum of all provided terms (including
            coefficients). The construction uses an FSM approach which is significantly
            faster than summing individual MPOs for large numbers of terms.
        """
        if physical_dimension != 2:
            msg = "Only physical_dimension=2 is supported by this Pauli MPO builder."
            raise ValueError(msg)
        if length <= 0:
            msg = "length must be positive."
            raise ValueError(msg)

        self.length = length
        self.physical_dimension = physical_dimension

        if not terms:
            self.tensors = [np.zeros((2, 2, 1, 1), dtype=complex) for _ in range(length)]
            return

        # 1. Parse terms into dense lists of operator names.
        #    Structure: terms list of (coeff, [op_at_site_0, op_at_site_1, ...])
        parsed_terms: list[tuple[complex | float, list[str]]] = []
        for coeff, spec in terms:
            ops_map = self._parse_pauli_string(spec)
            # Validate sites
            for site, lab in ops_map.items():
                if not (0 <= site < length):
                    msg = f"Site index {site} outside [0, {length - 1}]."
                    raise ValueError(msg)
                if lab not in self._VALID:
                    msg = f"Invalid local op {lab!r}; expected one of {sorted(self._VALID)}."
                    raise ValueError(msg)

            # Fill missing sites with Identity "I"
            op_list = [ops_map.get(i, "I") for i in range(length)]
            parsed_terms.append((coeff, op_list))

        # 2. Assign State IDs (Right-to-Left)
        #    We identify unique "suffix states" needed at each bond.
        #    A state at bond i is uniquely defined by the pair (Operator at site i, State at bond i+1).

        # `term_trajectories[term_idx][i]` stores the State ID at bond `i` for `term_idx`.
        # Bond indices range from 0 (left of site 0) to L (right of site L-1).
        term_trajectories = [[0] * (length + 1) for _ in range(len(parsed_terms))]

        # Initialize right boundary (Bond L): All terms end at the "sink" state (ID 0).
        for t_idx in range(len(parsed_terms)):
            term_trajectories[t_idx][length] = 0

        # bond_state_maps[i] stores the mapping: (Op_str, Next_State_ID) -> Current_State_ID
        bond_state_maps: list[dict[tuple[str, int], int]] = [{} for _ in range(length + 1)]

        # Sweep Right-to-Left (sites L-1 down to 1) to build the FSM transitions.
        # We stop at bond 1. Bond 0 is always the single "Start" state.
        for i in range(length - 1, 0, -1):
            next_bond = i + 1
            current_bond = i

            unique_states_map = bond_state_maps[current_bond]
            next_id = 0

            for t_idx, (_, ops) in enumerate(parsed_terms):
                op = ops[i]
                next_state = term_trajectories[t_idx][next_bond]
                signature = (op, next_state)

                if signature not in unique_states_map:
                    unique_states_map[signature] = next_id
                    next_id += 1

                term_trajectories[t_idx][current_bond] = unique_states_map[signature]

        # 3. Build Tensors (Left-to-Right)
        self.tensors = []
        paulis = self._PAULI_2

        for i in range(length):
            # Determine bond dimensions based on number of unique states at boundaries
            if i == 0:
                d_left = 1
                d_right = 1 if length == 1 else len(bond_state_maps[1])
                # Handle edge case where d_right is 0 (should not happen if terms exist)
                if length > 1 and d_right == 0:
                    d_right = 1
            else:
                d_left = len(bond_state_maps[i])
                d_right = 1 if i == length - 1 else len(bond_state_maps[i + 1])

            # Allocate tensor: (phys_out, phys_in, left, right)
            tensor = np.zeros((2, 2, d_left, d_right), dtype=complex)

            if i == 0:
                # First site: Accumulate coefficients and split into initial branches.
                for t_idx, (coeff, ops) in enumerate(parsed_terms):
                    op_name = ops[i]
                    op_mat = paulis[op_name]
                    target_state = term_trajectories[t_idx][1]

                    # Accumulate contribution. Multiple terms may map to the same target state.
                    tensor[:, :, 0, target_state] += coeff * op_mat
            else:
                # Internal sites: deterministic transitions.
                # Each row (current_id) in the tensor corresponds to a unique state from Step 2.
                # This state maps to exactly one (op, next_id) pair.
                map_i = bond_state_maps[i]

                for (op_name, next_id), current_id in map_i.items():
                    op_mat = paulis[op_name]
                    tensor[:, :, current_id, next_id] = op_mat

            self.tensors.append(tensor)

        # 4. Final Compression
        #    The FSM construction is optimal for one-sided (suffix) uniqueness.
        #    A standard two-sweep compression ("lr_rl") puts the MPO in canonical form
        #    and removes any remaining redundancies (e.g., common prefixes).
        self.compress(tol=tol, max_bond_dim=max_bond_dim, n_sweeps=n_sweeps, directions="lr_rl")
        assert self.check_if_valid_mpo(), "MPO initialized wrong"

    def compress(
        self,
        *,
        tol: float = 1e-12,
        max_bond_dim: int | None = None,
        n_sweeps: int = 1,
        directions: str = "lr_rl",
    ) -> None:
        """Compress this MPO using local SVD sweeps.

        This is a *public* convenience API that can run one or more sweeps in a chosen order.
        Each sweep applies local two-site SVD factorization along the chain, truncates singular
        values <= tol (and optionally caps the rank), and writes the factors back into the MPO.

        Args:
            tol: Truncation threshold. Singular values S_i with S_i <= tol are discarded.
            max_bond_dim: Optional hard cap on the kept rank after SVD.
            n_sweeps: Number of repetitions of the sweep schedule (must be >= 0).
            directions: Sweep schedule:
                - "lr": left-to-right only
                - "rl": right-to-left only
                - "lr_rl": do lr then rl (default)
                - "rl_lr": do rl then lr

        Raises:
            ValueError: If n_sweeps < 0 or directions is invalid.
        """
        if n_sweeps < 0:
            msg = "n_sweeps must be >= 0."
            raise ValueError(msg)
        if directions not in {"lr", "rl", "lr_rl", "rl_lr"}:
            msg = "directions must be one of {'lr', 'rl', 'lr_rl', 'rl_lr'}."
            raise ValueError(msg)

        if n_sweeps == 0:
            return

        schedule = {
            "lr": ("lr",),
            "rl": ("rl",),
            "lr_rl": ("lr", "rl"),
            "rl_lr": ("rl", "lr"),
        }[directions]

        for _ in range(n_sweeps):
            for direction in schedule:
                self._compress_one_sweep(direction=direction, tol=tol, max_bond_dim=max_bond_dim)

    def _compress_one_sweep(self, *, direction: str, tol: float, max_bond_dim: int | None) -> None:
        """Run one in-place MPO SVD compression sweep in the given direction.

        Args:
            direction: Sweep direction ("lr" or "rl").
            tol: Discard singular values <= tol.
            max_bond_dim: Optional hard cap on the kept rank.

        Raises:
            ValueError: If the direction is not 'lr' or 'rl'.
        """
        if direction not in {"lr", "rl"}:
            msg = "direction must be 'lr' or 'rl'."
            raise ValueError(msg)

        length = len(self.tensors)
        if length <= 1:
            return

        rng = range(length - 1) if direction == "lr" else range(length - 2, -1, -1)

        for k in rng:
            a = self.tensors[k]  # (d, d, Dl, Dm)
            b = self.tensors[k + 1]  # (d, d, Dm, Dr)

            phys_dim = a.shape[0]
            bond_dim_left = a.shape[2]
            bond_dim_right = b.shape[3]

            # Contract shared virtual bond (a.r with b.l): (s,t,l,r)x(u,v,r,w)->(s,t,u,v,l,w)
            theta = oe.contract("stlr,uvrw->stuvlw", a, b)

            # Group left legs (l,s,t) and right legs (u,v,w)
            theta = np.transpose(theta, (4, 0, 1, 2, 3, 5))
            matrix = theta.reshape(
                bond_dim_left * phys_dim * phys_dim,
                phys_dim * phys_dim * bond_dim_right,
            )

            u, s, vh = np.linalg.svd(matrix, full_matrices=False)
            keep = int(np.sum(tol < s))
            if max_bond_dim is not None:
                keep = min(keep, max_bond_dim)
            keep = max(1, keep)

            u = u[:, :keep]
            s = s[:keep]
            vh = vh[:keep, :]

            # Left tensor: (bond_dim_left, d, d, keep) -> (d, d, bond_dim_left, keep)
            left = u.reshape(bond_dim_left, phys_dim, phys_dim, keep).transpose(1, 2, 0, 3)

            # Right tensor: (keep, d, d, bond_dim_right) -> (d, d, keep, bond_dim_right)
            svh = (s[:, None] * vh).reshape(keep, phys_dim, phys_dim, bond_dim_right)
            right = svh.transpose(1, 2, 0, 3)

            self.tensors[k] = left
            self.tensors[k + 1] = right

    def rotate(self, *, conjugate: bool = False) -> None:
        """Rotates MPO.

        Rotates the tensors in the network by flipping the physical dimensions.
        This method transposes each tensor in the network along specified axes.
        If the `conjugate` parameter is set to True, it also takes the complex
        conjugate of each tensor before transposing.

        Args:
            conjugate (bool): If True, take the complex conjugate of each tensor
                              before transposing. Default is False.
        """
        for i, tensor in enumerate(self.tensors):
            if conjugate:
                self.tensors[i] = np.transpose(np.conj(tensor), (1, 0, 2, 3))
            else:
                self.tensors[i] = np.transpose(tensor, (1, 0, 2, 3))

    def to_mps(self) -> MPS:
        """MPO to MPS conversion.

        Converts the current tensor network to a Matrix Product State (MPS) representation.
        This method reshapes each tensor in the network from shape
        (dim1, dim2, dim3, dim4) to (dim1 * dim2, dim3, dim4) and
        returns a new MPS object with the converted tensors.

        Returns:
            MPS: An MPS object containing the reshaped tensors.
        """
        converted_tensors: list[NDArray[np.complex128]] = [
            np.reshape(
                tensor,
                (tensor.shape[0] * tensor.shape[1], tensor.shape[2], tensor.shape[3]),
            )
            for tensor in self.tensors
        ]

        return MPS(self.length, converted_tensors)

    def to_matrix(self) -> NDArray[np.complex128]:
        """MPO to matrix conversion.

        Converts a list of tensors into a matrix using Einstein summation convention.
        This method iterates over the list of tensors and performs tensor contractions
        using the Einstein summation convention (`oe.constrain`). The resulting tensor is
        then reshaped accordingly. The final matrix is squeezed to ensure the left and
        right bonds are 1.

        Returns:
            The resulting matrix after tensor contractions and reshaping.
        """
        mat = self.tensors[0]
        for tensor in self.tensors[1:]:
            mat = oe.contract("abcd, efdg->aebfcg", mat, tensor)
            mat = np.reshape(
                mat,
                (
                    mat.shape[0] * mat.shape[1],
                    mat.shape[2] * mat.shape[3],
                    mat.shape[4],
                    mat.shape[5],
                ),
            )

        # Final left and right bonds should be 1
        return np.squeeze(mat, axis=(2, 3))

    def to_sparse_matrix(self) -> scipy.sparse.csr_matrix:
        """MPO to sparse matrix conversion.

        Efficiently constructs a sparse matrix from the MPO tensors by iterating
        over the terms in the MPO sum. This avoids creating the full dense matrix
        intermediate.

        Returns:
            The sparse matrix representation of the MPO in CSR format.
        """
        d = self.physical_dimension

        current_operators = {0: scipy.sparse.csr_matrix(np.eye(1, dtype=complex))}

        for tensor in self.tensors:
            d_out, d_in, D_left, D_right = tensor.shape  # noqa: N806

            next_operators = {}

            for beta in range(D_right):
                accumulated = None

                for alpha in range(D_left):
                    if alpha not in current_operators:
                        continue

                    # Extract local operator for this bond transition (alpha -> beta)
                    op_local_dense = tensor[:, :, alpha, beta]

                    # Optimization: Skip if local op is zero
                    if np.all(op_local_dense == 0):
                        continue

                    # Convert to sparse
                    op_local = scipy.sparse.csr_matrix(op_local_dense)
                    op_left = current_operators[alpha]

                    # Kronecker product: Left (X) Local
                    term = scipy.sparse.kron(op_left, op_local, format="csr")

                    if accumulated is None:
                        accumulated = term
                    else:
                        accumulated = accumulated + term

                if accumulated is not None:
                    next_operators[beta] = accumulated

            current_operators = next_operators

        # Final result should be in current_operators[0] because the last bond dim is 1
        if 0 not in current_operators:
             # Should practically not happen for valid MPOs unless it's a zero operator
             dim = d**self.length
             return scipy.sparse.csr_matrix((dim, dim), dtype=complex)

        return current_operators[0]

    @classmethod
    def from_matrix(
        cls,
        mat: np.ndarray,
        d: int,
        max_bond: int | None = None,
        cutoff: float = 1e-12,
    ) -> MPO:
        """Factorize a dense matrix into an MPO with uniform local dimension ``d``.

        Each site has local shape ``(d, d)``.
        The number of sites ``n`` is inferred from the relation:

            mat.shape = (d**n, d**n)

        Args:
            mat (np.ndarray):
                Square matrix of shape ``(d**n, d**n)``.
            d (int):
                Physical dimension per site. Must satisfy ``d > 0``.
            max_bond (int | None):
                Maximum allowed bond dimension (before truncation).
            cutoff (float):
                Singular values ``<= cutoff`` are discarded. By default cutoff=1e-12: all numerically non-zero
                singular values are included.

        Returns:
            MPO:
                An MPO with ``n`` sites, uniform physical dimension ``d`` per site,
                and bond dimensions determined by SVD truncation.

        Raises:
            ValueError:
                If ``d <= 0``;
                If ``d == 1`` but the matrix is not ``1 x 1``;
                If the matrix is not square;
                If ``rows`` is not a power of ``d``;
                If the inferred number of sites ``n < 1``.
        """
        if d <= 0:
            msg = f"Physical dimension d must be > 0, got d={d}."
            raise ValueError(msg)

        rows, cols = mat.shape

        if rows != cols:
            msg = "Matrix must be square for uniform MPO factorization."
            raise ValueError(msg)

        if d == 1:
            if rows != 1:
                msg = "For d == 1 the matrix must be 1x1 since 1**n = 1 for any n."
                raise ValueError(msg)
            n = 1
        else:
            n_float = np.log(rows) / np.log(d)
            n = round(n_float)

            if n < 1:
                msg = f"Inferred chain length n={n} is invalid; matrix dimension {rows} too small for base d={d}."
                raise ValueError(msg)

            if not np.isclose(n_float, n):
                msg = f"Matrix dimension {rows} is not a power of d={d}."
                raise ValueError(msg)

        mat = np.asarray(mat, dtype=np.complex128)

        left_rank = 1
        rem = mat.reshape(1, rows, cols)

        tensors: list[np.ndarray] = []

        def _truncate(s: np.ndarray) -> int:
            r = s.size
            if cutoff > 0.0:
                r = max(int(np.sum(s > cutoff)), 1)
            if max_bond is not None:
                r = min(r, max_bond)
            return r

        for k in range(n - 1):
            rest = d ** (n - k - 1)

            rem = rem.reshape(left_rank, d, rest, d, rest)
            rem_perm = np.transpose(rem, (1, 3, 0, 2, 4))
            x = rem_perm.reshape(d * d * left_rank, rest * rest)

            u, s, vh = np.linalg.svd(x, full_matrices=False)

            r_keep = _truncate(s)

            u = u[:, :r_keep]
            s = s[:r_keep]
            vh = vh[:r_keep, :]

            t_k = u.reshape(d, d, left_rank, r_keep)
            tensors.append(t_k)

            rem = (s[:, None] * vh).reshape(r_keep, rest, rest)
            left_rank = r_keep

        rem = rem.reshape(left_rank, d, d)
        t_last = np.transpose(rem, (1, 2, 0)).reshape(d, d, left_rank, 1)
        tensors.append(t_last)

        mpo = cls()
        mpo.tensors = tensors
        mpo.length = n
        mpo.physical_dimension = d

        assert mpo.check_if_valid_mpo(), "MPO initialized wrong"

        return mpo

    def check_if_valid_mpo(self) -> bool:
        """MPO validity check.

        Check if the current tensor network is a valid Matrix Product Operator (MPO).
        This method verifies the consistency of the bond dimensions between adjacent tensors
        in the network. Specifically, it checks that the right bond dimension of each tensor
        matches the left bond dimension of the subsequent tensor.

        Returns:
            bool: True if the tensor network is a valid MPO, False otherwise.
        """
        right_bond = self.tensors[0].shape[3]
        for tensor in self.tensors[1::]:
            assert tensor.shape[2] == right_bond
            right_bond = tensor.shape[3]
        return True

    def check_if_identity(self, fidelity: float) -> bool:
        """MPO Identity check.

        Check if the current MPO (Matrix Product Operator) represents an identity operation
        within a given fidelity threshold.

        Args:
            fidelity (float): The fidelity threshold to determine if the MPO is an identity.

        Returns:
            bool: True if the MPO is considered an identity within the given fidelity, False otherwise.
        """
        identity_mpo = MPO()
        identity_mpo.identity(self.length)

        identity_mps = identity_mpo.to_mps()
        mps = self.to_mps()
        trace = mps.scalar_product(identity_mps)

        # Checks if trace is not a singular values for partial trace
        return not np.round(np.abs(trace), 1) / 2**self.length < fidelity

    @classmethod
    def _parse_pauli_string(cls, spec: str) -> dict[int, str]:
        """Parse a Pauli-string specification into a site-to-operator mapping.

        Converts a compact string representation of a Pauli operator product
        into a dictionary mapping site indices to Pauli labels.

        The expected format is a whitespace- or comma-separated list of tokens:
            "X0 Y2 Z5"

        Args:
            spec: Pauli-string specification.

        Returns:
            dict[int, str]: Mapping from site index to Pauli label
            ('I', 'X', 'Y', or 'Z'). An empty dictionary corresponds to the
            identity operator.

        Raises:
            ValueError: If:
                - a site index appears more than once,
                - an invalid token is encountered,
                - or the specification contains malformed entries.

        """
        s = spec.replace(",", " ").strip()
        if not s:
            return {}
        out: dict[int, str] = {}
        for op, idx in cls._PAULI_TOKEN_RE.findall(s):
            site = int(idx)
            op_up = op.upper()
            if site in out:
                msg = f"Duplicate site {site} in spec '{spec}'."
                raise ValueError(msg)
            out[site] = op_up
        cleaned = cls._PAULI_TOKEN_RE.sub("", s)
        if cleaned.split():
            msg = f"Invalid token(s) in spec '{spec}'. Use forms like 'X0 Y2 Z5'."
            raise ValueError(msg)
        return out
