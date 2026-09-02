# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Matrix Product State (MPS) for YAQS tensor-network simulations."""

from __future__ import annotations

import copy
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from numbers import Integral
from typing import TYPE_CHECKING, Any

import numpy as np
import opt_einsum as oe
import scipy.linalg
from tqdm import tqdm

from .. import linalg
from ..methods.decompositions import left_qr, merge_two_site, right_qr, split_two_site
from ..parallel_utils import available_cpus, get_parallel_context, limit_worker_threads

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..methods.decompositions import TruncMode
    from .simulation_parameters import AnalogSimParams, DigitalSimParams, Observable

# Worker-global state for parallel ``measure_shots`` (initialized once per process).
_MEASURE_SHOTS_CTX: dict[str, Any] = {}


def _measure_shots_worker_init(mps: MPS, basis: str) -> None:
    """Initialize a measure-shots worker and cap numerical thread pools."""
    limit_worker_threads(1)
    _MEASURE_SHOTS_CTX.clear()
    _MEASURE_SHOTS_CTX["mps"] = mps
    _MEASURE_SHOTS_CTX["basis"] = basis


def _measure_shots_worker(_shot_idx: int) -> int:
    """Run a single measurement shot using the worker-global MPS context.

    Args:
        _shot_idx: Unused shot index (required for executor compatibility).

    Returns:
        The measured basis state encoded as an integer.
    """
    return _MEASURE_SHOTS_CTX["mps"].measure_single_shot(_MEASURE_SHOTS_CTX["basis"])


class MPS:
    """Matrix Product State (MPS) class for representing quantum states.

    This class forms the basis of the MPS used in YAQS simulations.
    The index order is (sigma, chi_l-1, chi_l).

    Attributes:
        length: The number of sites in the MPS.
        tensors: List of rank-3 tensors representing the MPS.
        physical_dimensions: List of physical dimensions for each site.
        flipped: Indicates if the network has been flipped.
        orthogonality_center: Site index of the mixed-canonical center, or ``None`` if unknown.
            Gauge helpers use ``center`` as shorthand for this field (``set_center``,
            ``shift_center_to``, etc.). Direct ``tensors[i] = ...`` assignment bypasses
            tracking; call ``set_center(None)`` or use MPS mutators.
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
            ValueError: If ``length`` is not positive or the provided ``state`` does not match a valid
                initialization string.
        """
        if not isinstance(length, Integral) or isinstance(length, bool) or length < 1:
            msg = "length must be a positive integer."
            raise ValueError(msg)
        length = int(length)

        self.flipped = False
        self._orthogonality_center: int | None = None
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
            if state == "haar-random":
                self._orthogonality_center = None
            else:
                self._orthogonality_center = 0
        if pad is not None and state != "haar-random":
            self.pad_bond_dimension(pad)

    @property
    def orthogonality_center(self) -> int | None:
        """Site index of the mixed-canonical center, or ``None`` if the gauge is unknown."""
        return self._orthogonality_center

    def _validate_center(self, center: int, *, name: str = "center") -> int:
        """Validate a site used as an orthogonality center.

        Args:
            center: Site index to validate.
            name: Argument name used in the error message.

        Returns:
            The validated site as a Python integer.

        Raises:
            ValueError: If ``center`` is outside this MPS.
        """
        if not isinstance(center, Integral) or isinstance(center, bool) or not 0 <= center < self.length:
            msg = f"{name} must be in [0, {self.length - 1}], got {center!r}."
            raise ValueError(msg)
        return int(center)

    @staticmethod
    def _validate_decomposition(decomposition: str) -> None:
        """Validate a canonicalization decomposition name.

        Args:
            decomposition: Decomposition name to validate.

        Raises:
            ValueError: If ``decomposition`` is neither ``"QR"`` nor ``"SVD"``.
        """
        if decomposition not in {"QR", "SVD"}:
            msg = f"decomposition must be 'QR' or 'SVD', got {decomposition!r}."
            raise ValueError(msg)

    def set_center(self, center: int | None) -> None:
        """Set the tracked orthogonality center without re-canonicalizing.

        Args:
            center: Mixed-canonical center site index, or ``None`` if the gauge is unknown.
        """
        if center is not None:
            center = self._validate_center(center)
        self._orthogonality_center = center

    def update_center_after_split(self, left_site: int, right_site: int, svd_distribution: str) -> None:
        """Update the tracked center after a two-site SVD split.

        Call immediately after ``split_two_site`` or ``split_tdvp`` assigns new bond
        tensors.

        Args:
            left_site: Left site index of the split pair.
            right_site: Right site index of the split pair.
            svd_distribution: ``"left"``, ``"right"``, or ``"sqrt"``.

        Raises:
            ValueError: If the sites are not an in-range adjacent pair or the
                distribution is not ``"left"``, ``"right"``, or ``"sqrt"``.

        Notes:
            A left or right distribution moves a center that already covered the
            split pair. If the prior center did not cover the pair, or the singular
            values were split with ``"sqrt"``, the gauge remains unknown.
        """
        left_site = self._validate_center(left_site, name="left_site")
        right_site = self._validate_center(right_site, name="right_site")
        if right_site != left_site + 1:
            msg = f"Split sites must be adjacent and ordered, got ({left_site}, {right_site})."
            raise ValueError(msg)
        if svd_distribution not in {"left", "right", "sqrt"}:
            msg = f"svd_distribution must be 'left', 'right', or 'sqrt', got {svd_distribution!r}."
            raise ValueError(msg)

        if self._orthogonality_center not in {left_site, right_site} or svd_distribution == "sqrt":
            self._orthogonality_center = None
        elif svd_distribution == "right":
            self._orthogonality_center = right_site
        else:
            self._orthogonality_center = left_site

    def assert_center(self, expected: int, *, context: str) -> None:
        """Raise if the tracked center is unknown or not ``expected``.

        Args:
            expected: Required center site index.
            context: Description of the calling algorithm for error messages.

        Raises:
            ValueError: If the gauge is unknown or the center does not match.
        """
        if self._orthogonality_center is None:
            msg = f"{context}: MPS gauge unknown (orthogonality_center is None), expected site {expected}."
            raise ValueError(msg)
        if self._orthogonality_center != expected:
            msg = f"{context}: orthogonality center at site {self._orthogonality_center}, expected site {expected}."
            raise ValueError(msg)

    def check_covers_sites(self, sites: int | list[int]) -> bool:
        """Check whether the tracked center supports local contraction at ``sites``.

        Args:
            sites: One site index or a nearest-neighbor two-site pair.

        Returns:
            True if the tracked center covers the observable site(s).
        """
        if self._orthogonality_center is None:
            return False
        sites_list = [sites] if isinstance(sites, int) else list(sites)
        if len(sites_list) == 1:
            return self._orthogonality_center == sites_list[0]
        if len(sites_list) == 2:
            i, j = sites_list
            return j == i + 1 and self._orthogonality_center in {i, j}
        return False

    def shift_center_to(self, target: int, decomposition: str = "QR") -> None:
        """Shift the orthogonality center to ``target`` via incremental moves.

        Args:
            target: Desired orthogonality center site index.
            decomposition: QR or SVD decomposition for each shift step.

        Raises:
            ValueError: If the target or decomposition is invalid, or the gauge is unknown.
        """
        target = self._validate_center(target, name="target")
        self._validate_decomposition(decomposition)
        if self._orthogonality_center is None:
            msg = "Cannot shift orthogonality center when gauge is unknown."
            raise ValueError(msg)
        current = self._orthogonality_center
        while current < target:
            self.shift_orthogonality_center_right(current, decomposition)
            current += 1
        while current > target:
            self.shift_orthogonality_center_left(current, decomposition)
            current -= 1

    def init_mps_from_basis(self, basis_string: str, physical_dimensions: list[int]) -> None:
        """Initialize this MPS in place as a product state from a basis string.

        Any existing tensors are replaced, so the method is safe to call on an
        already-populated MPS.

        Args:
            basis_string: A string like "0101" indicating the computational basis state.
            physical_dimensions: The physical dimension of each site (e.g. 2 for qubits, 3+ for qudits).

        Raises:
            ValueError: If ``basis_string`` and ``physical_dimensions`` differ in length,
                or if they do not match :attr:`length`.
        """
        if len(basis_string) != len(physical_dimensions):
            msg = (
                f"basis_string has {len(basis_string)} characters but "
                f"{len(physical_dimensions)} physical dimensions were given."
            )
            raise ValueError(msg)
        if len(basis_string) != self.length:
            msg = f"basis_string has {len(basis_string)} characters but the MPS has length {self.length}."
            raise ValueError(msg)

        tensors = []
        for site, char in enumerate(basis_string):
            idx = int(char)
            tensor = np.zeros((physical_dimensions[site], 1, 1), dtype=complex)
            tensor[idx, 0, 0] = 1.0
            tensors.append(tensor)
        self.tensors = tensors

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

        target_shapes = []
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

            target_shapes.append((phys, left_target, right_target))

        padded_tensors = []
        for tensor, target_shape in zip(self.tensors, target_shapes, strict=True):
            _, chi_l, chi_r = tensor.shape
            new_tensor = np.zeros(target_shape, dtype=tensor.dtype)
            new_tensor[:, :chi_l, :chi_r] = tensor
            padded_tensors.append(new_tensor)

        padded_state = MPS(
            self.length,
            tensors=padded_tensors,
            physical_dimensions=list(self.physical_dimensions),
        )
        padded_state.flipped = self.flipped
        padded_state.normalize()

        self.tensors = padded_state.tensors
        self.physical_dimensions = padded_state.physical_dimensions
        self._orthogonality_center = padded_state.orthogonality_center
        self.flipped = padded_state.flipped

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
        bonds = list(bond_indices)

        # Validate every requested bond before changing any tensor. This keeps the
        # original state and its center intact when a later request is invalid.
        for bond in bonds:
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

        for bond in bonds:
            left = self.tensors[bond]
            right = self.tensors[bond + 1]
            chi_out = int(left.shape[2])
            chi_in = int(right.shape[1])
            if chi_out >= target_dim and chi_in >= target_dim:
                continue

            new_left = left
            if chi_out < target_dim:
                phys_l, chi_l, _ = left.shape
                new_left = np.zeros((phys_l, chi_l, target_dim), dtype=left.dtype)
                new_left[:, :, :chi_out] = left

            new_right = right
            if chi_in < target_dim:
                phys_r, _, chi_r = right.shape
                new_right = np.zeros((phys_r, target_dim, chi_r), dtype=right.dtype)
                new_right[:, :chi_in, :] = right

            self.tensors[bond] = new_left
            self.tensors[bond + 1] = new_right
            self._orthogonality_center = None

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

    def _bond_matrix(self, i: int, j: int) -> NDArray[np.complex128]:
        """Two-site matrix at the cut ``(i, j)``, gauged so its singular values are Schmidt values.

        The singular values of ``tensors[i] tensors[j]`` are the Schmidt coefficients of the
        cut only in a mixed-canonical gauge with the center on the pair; the center is moved
        onto the pair on a copy when it is not already there.

        Args:
            i: Left site of the nearest-neighbor cut.
            j: Right site of the cut, ``i + 1``.

        Returns:
            The two-site matrix reshaped to ``(d_i * chi_left, d_j * chi_right)``.
        """
        state = self
        if not self.check_covers_sites([i, j]):
            state = copy.deepcopy(self)
            if state.orthogonality_center is None:
                state.set_canonical_form(i)
            else:
                state.shift_center_to(i)

        a, b = state.tensors[i], state.tensors[j]
        theta = np.tensordot(a, b, axes=(2, 1))
        return theta.reshape(a.shape[0] * a.shape[1], b.shape[0] * b.shape[2]).astype(np.complex128)

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

        if self.tensors[i].shape[2] == 1:
            return np.float64(0.0)

        theta_mat = self._bond_matrix(i, j)

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

        if self.tensors[i].shape[2] == 1:
            padded = np.full(top_schmidt_vals, np.nan)
            padded[0] = 1.0
            return padded

        _, s_vec, _ = linalg.svd(self._bond_matrix(i, j), full_matrices=False)

        padded = np.full(top_schmidt_vals, np.nan)
        padded[: min(top_schmidt_vals, len(s_vec))] = s_vec[:top_schmidt_vals]
        return padded

    def flip_network(self) -> None:
        """Flip MPS.

        Flips the bond dimensions in the network so that we can do operations
        from right to left rather than coding it twice. Also reverses
        ``physical_dimensions`` so mixed-dimensional states remain consistent.

        """
        new_tensors = []
        for tensor in self.tensors:
            new_tensor = np.transpose(tensor, (0, 2, 1))
            new_tensors.append(new_tensor)

        new_tensors.reverse()
        self.tensors = new_tensors
        self.physical_dimensions = list(reversed(self.physical_dimensions))
        self.flipped = not self.flipped
        if self._orthogonality_center is not None:
            self._orthogonality_center = self.length - 1 - self._orthogonality_center

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
        current_orthogonality_center = self._validate_center(
            current_orthogonality_center, name="current_orthogonality_center"
        )
        self._validate_decomposition(decomposition)
        if self._orthogonality_center is not None:
            assert self._orthogonality_center == current_orthogonality_center, (
                f"shift_orthogonality_center_right: tracked center is {self._orthogonality_center}, "
                f"but shift requested from site {current_orthogonality_center}."
            )
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
        if self._orthogonality_center is not None:
            if current_orthogonality_center + 1 < self.length:
                self._orthogonality_center = current_orthogonality_center + 1
            else:
                self._orthogonality_center = current_orthogonality_center

    def shift_orthogonality_center_left(self, current_orthogonality_center: int, decomposition: str = "QR") -> None:
        """Shift the orthogonality center one site to the left.

        The method updates only the center tensor and its left neighbor without
        flipping the network. At site zero, the method normalizes the center
        tensor and leaves the center at zero.

        Args:
            current_orthogonality_center: Site that currently holds the orthogonality center.
            decomposition: Decomposition used for the shift. ``"QR"`` is faster;
                ``"SVD"`` can reduce the bond dimension.
        """
        current_orthogonality_center = self._validate_center(
            current_orthogonality_center, name="current_orthogonality_center"
        )
        self._validate_decomposition(decomposition)
        if self._orthogonality_center is not None:
            assert self._orthogonality_center == current_orthogonality_center, (
                f"shift_orthogonality_center_left: tracked center is {self._orthogonality_center}, "
                f"but shift requested from site {current_orthogonality_center}."
            )
        tensor = self.tensors[current_orthogonality_center]
        if decomposition == "QR" or current_orthogonality_center == 0:
            site_tensor, bond_tensor = left_qr(tensor)
            self.tensors[current_orthogonality_center] = site_tensor

            if current_orthogonality_center - 1 >= 0:
                self.tensors[current_orthogonality_center - 1] = oe.contract(
                    "aij, jk->aik",
                    self.tensors[current_orthogonality_center - 1],
                    bond_tensor,
                )
        elif decomposition == "SVD":
            a, b = (
                self.tensors[current_orthogonality_center - 1],
                self.tensors[current_orthogonality_center],
            )
            merged = merge_two_site(a, b)
            a_new, b_new = split_two_site(
                merged,
                [a.shape[0], b.shape[0]],
                svd_distribution="left",
                trunc_mode="discarded_weight",
                threshold=1e-12,
                max_bond_dim=None,
            )
            (
                self.tensors[current_orthogonality_center - 1],
                self.tensors[current_orthogonality_center],
            ) = (a_new, b_new)
        if self._orthogonality_center is not None:
            if current_orthogonality_center - 1 >= 0:
                self._orthogonality_center = current_orthogonality_center - 1
            else:
                self._orthogonality_center = current_orthogonality_center

    def set_canonical_form(self, orthogonality_center: int, decomposition: str = "QR") -> None:
        """Sets canonical form of MPS.

        Left and right normalizes an MPS around a selected site.
        NOTE: Slow method compared to shifting based on known form and should be avoided.

        Args:
            orthogonality_center (int): site of matrix MPS around which we normalize
            decomposition: Type of decomposition. Default QR.

        Raises:
            ValueError: If an argument is invalid or a tensor contains a non-finite value.

        """
        orthogonality_center = self._validate_center(orthogonality_center, name="orthogonality_center")
        self._validate_decomposition(decomposition)
        if not all(np.isfinite(tensor).all() for tensor in self.tensors):
            msg = "Cannot canonicalize an MPS with non-finite tensor values."
            raise ValueError(msg)

        def sweep_decomposition(orthogonality_center: int, decomposition: str = "QR") -> None:
            for site, _ in enumerate(self.tensors):
                if site == orthogonality_center:
                    break
                self.shift_orthogonality_center_right(site, decomposition)

        self._orthogonality_center = None
        sweep_decomposition(orthogonality_center, decomposition)
        self.flip_network()
        try:
            flipped_orthogonality_center = self.length - 1 - orthogonality_center
            sweep_decomposition(flipped_orthogonality_center, decomposition)
        finally:
            self.flip_network()
        if not all(np.isfinite(tensor).all() for tensor in self.tensors):
            msg = "Canonicalization produced non-finite tensor values."
            raise ValueError(msg)
        self._orthogonality_center = orthogonality_center

    def _scaled_center_tensor(self) -> tuple[NDArray[np.complex128], float, float]:
        """Return stable components for the tracked center's Frobenius norm.

        Returns:
            The center tensor divided by its largest real or imaginary component,
            that component scale, and the scaled tensor's Frobenius norm.

        Raises:
            ValueError: If the orthogonality center is unknown.
        """
        center = self._orthogonality_center
        if center is None:
            msg = "Cannot inspect MPS center: orthogonality center is unknown."
            raise ValueError(msg)

        center_tensor = self.tensors[center]
        real_scale = float(np.max(np.abs(center_tensor.real)))
        imag_scale = float(np.max(np.abs(center_tensor.imag)))
        scale = float(np.maximum(real_scale, imag_scale))
        if scale <= 0.0 or not np.isfinite(scale):
            return center_tensor, scale, scale
        scaled_tensor = center_tensor / scale
        return scaled_tensor, scale, float(np.linalg.norm(scaled_tensor))

    def normalize_center(self) -> None:
        """Normalize the MPS by rescaling its tracked center tensor.

        This operation does not move or establish the orthogonality center. It trusts
        the tracked mixed-canonical gauge and therefore touches only the center tensor.
        A stable direct norm handles ordinary tensors; scaled arithmetic handles norms
        larger than the floating-point range.

        Raises:
            ValueError: If the center is unknown or its norm is zero or non-finite.
        """
        center = self._orthogonality_center
        if center is None:
            msg = "Cannot normalize MPS: orthogonality center is unknown."
            raise ValueError(msg)

        msg = "Cannot normalize MPS: norm is zero or non-finite."
        center_tensor = self.tensors[center]
        norm = float(scipy.linalg.norm(center_tensor.ravel(order="K"), check_finite=False))
        if norm > 0.0 and np.isfinite(norm):
            self.tensors[center] = center_tensor / norm
            return

        scaled_tensor, scale, scaled_norm = self._scaled_center_tensor()
        if scale <= 0.0 or not np.isfinite(scale):
            raise ValueError(msg)
        if scaled_norm <= 0.0 or not np.isfinite(scaled_norm):
            raise ValueError(msg)
        scaled_tensor /= scaled_norm
        self.tensors[center] = scaled_tensor

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

        Raises:
            ValueError: If the decomposition is invalid or the state norm is zero or non-finite.

        """
        self._validate_decomposition(decomposition)
        msg = "Cannot normalize MPS: norm is zero or non-finite."
        if not all(np.isfinite(tensor).all() for tensor in self.tensors):
            raise ValueError(msg)

        flipped = form == "B"
        if flipped:
            self.flip_network()
        try:
            self.set_canonical_form(orthogonality_center=self.length - 1, decomposition=decomposition)
            self.normalize_center()
        finally:
            if flipped:
                self.flip_network()

    def compress(
        self,
        threshold: float,
        *,
        max_bond_dim: int | None = None,
        trunc_mode: TruncMode = "discarded_weight",
        _restore_center: int | None = None,
    ) -> None:
        """Compress in place by right-canonicalizing, then truncating left-to-right.

        Truncating while establishing a mixed gauge from a non-canonical product
        state (e.g. after uncapped MPO-MPS multiplication) is not equivalent to the optimal
        sequential SVD of the state and can leave large, θ-independent residuals.
        The procedure here is:

        1. Restore a right-canonical gauge with QR (no bond-dimension cap).
        2. Sweep left-to-right with truncated SVD at each bond.
        3. Restore the original orthogonality center without further truncation, or
           use the private target supplied by an internal caller.

        Args:
            threshold: SVD truncation threshold (e.g. ``sim_params.svd_threshold``).
            max_bond_dim: Optional cap on bond dimension.
            trunc_mode: Truncation mode forwarded to the two-site SVD split
                (``"discarded_weight"``, ``"relative"``, ``"hard_cutoff"``, or
                ``"relative_discarded_weight"``).
            _restore_center: Internal final-center target. This lets a caller keep
                gauge metadata unknown after tensor mutation while selecting the
                center left by compression.

        """
        if _restore_center is not None:
            _restore_center = self._validate_center(_restore_center, name="_restore_center")
        if self.length == 1:
            if _restore_center is not None:
                self._orthogonality_center = _restore_center
            return

        if _restore_center is not None:
            orth_center = _restore_center
        elif self._orthogonality_center is not None:
            orth_center = self._orthogonality_center
        else:
            canonical = self.check_canonical_form()
            orth_center = canonical[0] if canonical and canonical[0] >= 0 else self.length // 2

        # Right-canonical form without χ truncation (center at site 0).
        self.set_canonical_form(0, decomposition="QR")

        for site in range(self.length - 1):
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
            self._orthogonality_center = site + 1

        # Restore the caller's center without additional truncation.
        assert self._orthogonality_center is not None
        while self._orthogonality_center < orth_center:
            self.shift_orthogonality_center_right(self._orthogonality_center, "QR")
        while self._orthogonality_center > orth_center:
            self.shift_orthogonality_center_left(self._orthogonality_center, "QR")

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

        Notes:
            When ``sites`` is set, this method contracts only the stored tensors
            at those sites. Use :meth:`norm`, :meth:`local_expect`, or
            :meth:`expect` for gauge-safe physical quantities.
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

        The function contracts the operator directly with the one-site tensor or
        merged two-site tensor in a gauge centered on the requested site or pair.

        Args:
            operator: The local operator to be applied.
            sites: The indices of the sites at which to evaluate the expectation value.

        Returns:
            np.complex128: The computed expectation value (typically, its real part is of interest).

        Raises:
            ValueError: If the observable is not supported or its matrix shape does not match the target site.

        Notes:
            If the tracked center does not cover ``sites``, one copy is shifted or
            canonicalized. The original MPS is not modified.
        """
        sites_list = [sites] if isinstance(sites, int) else list(sites)
        operator_sites = [operator.sites] if isinstance(operator.sites, int) else list(operator.sites)

        if operator.gate.interaction == 1:
            assert len(sites_list) == 1, f"One-site observable requires one site, got {sites_list}."
            assert operator_sites == sites_list, f"Operator sites mismatch {operator_sites}, {sites_list}"
        elif operator.gate.interaction == 2:
            assert isinstance(sites, list)
            assert isinstance(operator.sites, list)
            i, j = sites_list
            assert operator.sites[0] == i, "Observable sites mismatch"
            assert operator.sites[1] == j, "Observable sites mismatch"
            assert operator.sites[0] < operator.sites[1], "Observable sites must be in ascending order."
            assert operator.sites[1] - operator.sites[0] == 1, (
                "Only nearest-neighbor observables are currently implemented."
            )
        else:
            msg = "Local observable must be one-site or nearest-neighbor two-site."
            raise ValueError(msg)

        for site in sites_list:
            self._validate_center(site, name="observable site")
        base_state = self
        if not self.check_covers_sites(sites_list):
            base_state = copy.deepcopy(self)
            if base_state.orthogonality_center is None:
                target = sites_list[0]
                base_state.set_canonical_form(target)
            else:
                center = base_state.orthogonality_center
                target = min(sites_list, key=lambda site: abs(center - site))
                base_state.shift_center_to(target)

        if operator.gate.interaction == 1:
            i = sites_list[0]
            a = base_state.tensors[i]
            local_dim = a.shape[0]
            matrix = np.asarray(operator.gate.matrix, dtype=np.complex128)
            if matrix.shape != (local_dim, local_dim):
                msg = f"Local observable matrix shape {matrix.shape} does not match site {i} dimension {local_dim}."
                raise ValueError(msg)
            return np.complex128(oe.contract("abc,ad,dbc->", np.conj(a), matrix, a))

        i, j = sites_list
        a = base_state.tensors[i]
        b = base_state.tensors[j]
        d_i, left, _ = a.shape
        d_j, _, right = b.shape
        matrix = np.asarray(operator.gate.matrix, dtype=np.complex128)
        if matrix.shape != (d_i * d_j, d_i * d_j):
            msg = f"Two-site observable matrix shape {matrix.shape} does not match site dimensions {d_i} and {d_j}."
            raise ValueError(msg)

        theta = np.tensordot(a, b, axes=(2, 1)).transpose(1, 0, 2, 3)
        theta = theta.reshape(left, d_i * d_j, right)
        return np.complex128(oe.contract("abc,bd,adc->", np.conj(theta), matrix, theta))

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

        Notes:
            The observable can be non-unitary. Applying it can therefore invalidate
            the canonical form, so this method marks the orthogonality center as
            unknown after changing the tensors.
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
            state.set_center(None)

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

        sites = [observable.sites] if isinstance(observable.sites, int) else list(observable.sites)

        if observable.gate.interaction == 1:
            if len(sites) != 1:
                msg = f"One-site local observable requires one site, got {sites}."
                raise ValueError(msg)
            site = sites[0]
            local_dim = self.tensors[site].shape[0]
            matrix = np.asarray(observable.gate.matrix, dtype=np.complex128)
            if matrix.shape != (local_dim, local_dim):
                msg = f"Local observable matrix shape {matrix.shape} does not match site {site} dimension {local_dim}."
                raise ValueError(msg)
            self.tensors[site] = oe.contract("ab, bcd->acd", matrix, self.tensors[site])
            self.set_center(None)
            return

        if observable.gate.interaction == 2:
            if len(sites) != 2:
                msg = f"Two-site local observable requires two sites, got {sites}."
                raise ValueError(msg)
            i, j = int(sites[0]), int(sites[1])
            length = self.length
            mat = np.asarray(observable.gate.matrix, dtype=np.complex128)
            d_i = self.tensors[i].shape[0]
            d_j = self.tensors[j].shape[0]
            if mat.shape != (d_i * d_j, d_i * d_j):
                msg = f"Two-site observable matrix shape {mat.shape} does not match site dimensions {d_i} and {d_j}."
                raise ValueError(msg)

            if length == 2:
                if i == length - 1 and j == 0:
                    g_merged = permuted_periodic_wrap(mat)
                    apply_two_site_nn_inplace(self, 0, g_merged)
                    return
                i, j = min(i, j), max(i, j)
            elif (i == length - 1 and j == 0) or (i == 0 and j == length - 1):
                bubble_swaps_forward(self)
                g_merged = permuted_periodic_wrap(mat)
                apply_two_site_nn_inplace(self, length - 2, g_merged)
                bubble_swaps_backward(self)
                return

            if j != i + 1:
                msg = "Only nearest-neighbor two-site observables are currently implemented."
                raise ValueError(msg)

            apply_two_site_nn_inplace(self, i, mat)
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
        sim_params: AnalogSimParams | DigitalSimParams,
        results: NDArray[np.float64],
        column_index: int = 0,
    ) -> None:
        """Evaluate and record expectation values of observables for a given MPS state.

        Args:
            sim_params: Simulation parameters containing sorted observables.
            results: 2D array where ``results[observable_index, column_index]`` stores
                expectation values.
            column_index: Time or trajectory index for the column to fill.

        Notes:
            Deep-copies ``self`` once and reuses that working state for all observables.
            When :attr:`orthogonality_center` covers the observable site(s), uses fast
            local contraction; otherwise shifts the center on the copy or falls back to
            full contraction when the gauge is unknown (``None``).
        """
        temp_state = copy.deepcopy(self)
        for obs_index, observable in enumerate(sim_params.sorted_observables):
            if observable.gate.name in {"entropy", "schmidt_spectrum"}:
                assert isinstance(observable.sites, list), "Given metric requires a list of sites"
                assert len(observable.sites) == 2, "Given metric requires 2 sites to act on."
                max_site = max(observable.sites)
                min_site = min(observable.sites)
                assert max_site - min_site == 1, "Entropy and Schmidt cuts must be nearest neighbor."
                for s in observable.sites:
                    assert s in range(self.length), f"Observable acting on non-existing site: {s}"
                if not temp_state.check_covers_sites(observable.sites):
                    if temp_state.orthogonality_center is None:
                        temp_state.set_canonical_form(min_site)
                    else:
                        center = temp_state.orthogonality_center
                        target = min_site if abs(center - min_site) <= abs(center - max_site) else max_site
                        temp_state.shift_center_to(target)
                if observable.gate.name == "entropy":
                    results[obs_index, column_index] = temp_state.get_entropy(observable.sites)
                elif observable.gate.name == "schmidt_spectrum":
                    results[obs_index, column_index] = temp_state.get_schmidt_spectrum(observable.sites)

            elif observable.gate.name == "pvm":
                assert hasattr(observable.gate, "bitstring"), "Gate does not have attribute bitstring."
                bitstring = observable.gate.bitstring
                assert isinstance(bitstring, str)
                results[obs_index, column_index] = self.project_onto_bitstring(bitstring)

            else:
                sites_list = [observable.sites] if isinstance(observable.sites, int) else list(observable.sites)
                if temp_state.orthogonality_center is not None and not temp_state.check_covers_sites(sites_list):
                    if len(sites_list) == 1:
                        target = sites_list[0]
                    else:
                        i, j = sites_list
                        center = temp_state.orthogonality_center
                        target = i if abs(center - i) <= abs(center - j) else j
                    temp_state.shift_center_to(target)
                if temp_state.orthogonality_center is None:
                    exp = temp_state.mixed_expectation(temp_state, observable)
                else:
                    exp = temp_state.local_expect(observable, sites_list)
                assert exp.imag < 1e-13, f"Measurement should be real, '{exp.real:16f}+{exp.imag:16f}i'."
                results[obs_index, column_index] = exp.real

    def expect(self, observable: Observable) -> np.float64:
        """Measure the expectation value of a given observable.

        Args:
            observable: One-site or two-site observable to evaluate.

        Returns:
            The real part of the expectation value.

        Notes:
            Uses fast local contraction when :attr:`orthogonality_center` covers the
            observable site(s); shifts incrementally on a copy when the center is
            known but misaligned; falls back to full contraction when the gauge is
            unknown (``None``).
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

        if self._orthogonality_center is None:
            exp = self.mixed_expectation(self, observable)
        elif self.check_covers_sites(sites_list):
            exp = self.local_expect(observable, sites_list)
        else:
            if len(sites_list) == 1:
                target = sites_list[0]
            else:
                i, j = sites_list
                center = self._orthogonality_center
                target = i if abs(center - i) <= abs(center - j) else j
            shifted = copy.deepcopy(self)
            shifted.shift_center_to(target)
            exp = shifted.local_expect(observable, sites_list)

        assert exp.imag < 1e-13, f"Measurement should be real, '{exp.real:16f}+{exp.imag:16f}i'."
        return exp.real

    def measure_single_shot(self, basis: str = "Z", rng: np.random.Generator | None = None) -> int:
        """Perform a single-shot measurement on a Matrix Product State (MPS).

        Simulates sequential projective measurement on every site. The state is first
        gauged at site 0. Each outcome is then absorbed into a local tensor for the
        unmeasured suffix without changing orthogonality-center metadata.

        Args:
            basis: The basis to measure in. Options are "X", "Y", or "Z" (default).
            rng: Optional random number generator for outcome sampling.

        Returns:
            The measurement outcome encoded as an integer bitstring.

        Raises:
            ValueError: If an invalid basis is provided.

        Notes:
            Prefer :meth:`measure` for a single-site sample when the center is already
            positioned. This method copies the MPS only when the center must move to
            site 0, then walks all sites through an active suffix tensor.
        """
        bitstring = []

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

        if rng is None:
            rng = np.random.default_rng()

        temp_state = self
        if temp_state.orthogonality_center is None:
            temp_state = copy.deepcopy(self)
            temp_state.set_canonical_form(0)
        elif temp_state.orthogonality_center != 0:
            temp_state = copy.deepcopy(self)
            temp_state.shift_center_to(0)
        active_tensor = temp_state.tensors[0]

        for site in range(temp_state.length):
            rotated_tensor = oe.contract("ab, bcd->acd", rotation, active_tensor)

            reduced_density_matrix = oe.contract("abc, dbc->ad", rotated_tensor, np.conj(rotated_tensor))
            probabilities = np.diag(reduced_density_matrix).real.copy()
            norm_factor = np.sum(probabilities)
            probabilities /= norm_factor
            chosen_index = rng.choice(len(probabilities), p=probabilities)
            bitstring.append(chosen_index)
            selected_state = np.zeros(len(probabilities))
            selected_state[chosen_index] = 1

            if site != temp_state.length - 1:
                projected_tensor = oe.contract("a, acd->cd", selected_state, rotated_tensor)
                active_tensor = (
                    1.0
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
        if shots <= 1:
            basis_state = self.measure_single_shot(basis)
            results[basis_state] = results.get(basis_state, 0) + 1
            return results

        max_workers = max(1, min(max(1, available_cpus() - 1), shots))
        if max_workers == 1:
            with tqdm(total=shots, desc="Measuring shots", ncols=80) as pbar:
                for _ in range(shots):
                    outcome = self.measure_single_shot(basis)
                    results[outcome] = results.get(outcome, 0) + 1
                    pbar.update(1)
            return results

        ctx = get_parallel_context("auto")
        inflight_factor = 2
        max_inflight = max_workers * inflight_factor

        with (
            ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=ctx,
                initializer=_measure_shots_worker_init,
                initargs=(self, basis),
            ) as executor,
            tqdm(total=shots, desc="Measuring shots", ncols=80) as pbar,
        ):
            futures: dict[Future[int], None] = {}
            next_shot = 0

            def submit_shot(idx: int) -> None:
                futures[executor.submit(_measure_shots_worker, idx)] = None

            while next_shot < shots and len(futures) < max_inflight:
                submit_shot(next_shot)
                next_shot += 1

            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    futures.pop(fut)
                    outcome = fut.result()
                    results[outcome] = results.get(outcome, 0) + 1
                    pbar.update(1)
                    if next_shot < shots:
                        submit_shot(next_shot)
                        next_shot += 1
        return results

    def measure(self, site: int, basis: str = "Z", rng: np.random.Generator | None = None) -> int:
        """Perform an in-place projective measurement on a single site of the MPS.

        This method modifies the MPS tensors to reflect the measurement outcome. When the
        orthogonality center is tracked, it is shifted incrementally to the target site before
        measuring; otherwise the state is re-canonicalized at ``site``.

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

        # Shift orthogonality center to target site.
        if self.orthogonality_center is not None:
            if self.orthogonality_center != site:
                self.shift_center_to(site)
        else:
            self.set_canonical_form(site)

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
            "a, cd->acd",
            original_basis_selection,
            projected_rotated_tensor,
        )
        self._orthogonality_center = site

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

        Calculate the Euclidean norm ``sqrt(<psi|psi>)`` of the state.

        Args:
            site: The specific site to calculate the norm from. If ``None``, the
                norm is calculated for the entire network.

        Returns:
            The norm of the state or the specified site.

        Notes:
            For jump probabilities and other quantities proportional to ``<psi|psi>``,
            use ``norm(...) ** 2``.

            For a site-specific norm, uses stable center-tensor scaling when
            :attr:`orthogonality_center` covers that site; shifts on a copy when the
            center is known but misaligned; falls back to the global norm when the
            gauge is unknown (``None``).
        """
        if site is not None and self.orthogonality_center is not None:
            if not self.check_covers_sites(site):
                temp = copy.deepcopy(self)
                temp.shift_center_to(site)
            else:
                temp = self
            _, scale, scaled_norm = temp._scaled_center_tensor()
            return np.float64(scale * scaled_norm)

        squared = float(self.scalar_product(self).real)
        return np.float64(np.sqrt(max(squared, 0.0)))

    def check_if_valid_mps(self) -> None:
        """MPS validity check.

        Check if the current tensor network is a valid Matrix Product State (MPS).

        This method verifies that the tensor count matches :attr:`length` and that the
        bond dimensions between consecutive tensors in the network are consistent.
        Specifically, it checks that the second dimension of each tensor matches the
        third dimension of the previous tensor.
        """
        assert len(self.tensors) == self.length, f"MPS has {len(self.tensors)} tensors but length {self.length}."
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
