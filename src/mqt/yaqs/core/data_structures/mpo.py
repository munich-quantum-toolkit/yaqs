# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Matrix Product Operator (MPO) for YAQS tensor-network simulations."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, ClassVar, cast, overload

import numpy as np
import opt_einsum as oe
import scipy.sparse
from numpy.typing import NDArray

from .. import linalg
from ..libraries.gate_library import Destroy
from .mpo_utils import (
    contract_mpo_site_with_mpo_site,
    contract_mpo_site_with_mps_site,
    get_support_mpo,
    make_identity_site,
)
from .mps import MPS

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..libraries.gate_library import BaseGate
    from ..methods.decompositions import TruncMode
    from .simulation_parameters import StrongSimParams, WeakSimParams

ComplexTensor = NDArray[np.complex128]


class MPO:
    """Matrix Product Operator (MPO) for YAQS tensor-network simulations.

    An MPO represents a linear operator on a 1D lattice as a chain of local tensors.
    YAQS stores each site tensor with index order::

        (phys_out, phys_in, chi_left, chi_right)

    where ``phys_out``/``phys_in`` are the physical operator legs and
    ``chi_left``/``chi_right`` are the virtual (bond) dimensions.

    **Construction**

    Use classmethod factories to build common Hamiltonians or custom operators:

    - ``MPO.ising(...)`` / ``MPO.heisenberg(...)``: qubit Pauli Hamiltonians.
    - ``MPO.pauli(...)``: generic one-/two-body Pauli interactions.
    - ``MPO.fermi_hubbard_1d(...)``: 1D Fermi-Hubbard (fermionic or Jordan-Wigner Pauli).
    - ``MPO.trapped_ions_position_grid(...)``: one or two trapped ions on a position grid.
    - ``MPO.coupled_transmon(...)``: alternating qubit/resonator chain MPO.
    - ``from_pauli_sum(...)``: in-place build from a sum of Pauli-string terms.
    - ``MPO.identity(...)``: identity operator.
    - ``custom(...)``, ``finite_state_machine(...)``: in-place builders.

    **Operations**

    - ``from_gate(...)``: build an MPO from a two-qubit gate on a chain (optionally identity-padded).
    - ``multiply(MPS)`` / ``multiply(MPO)``: apply this MPO to an MPS or left-multiply into another MPO.
    - ``compress(...)``: SVD-based bond compression sweeps.
    - ``rotate(...)``: swap physical legs (optionally conjugating).

    **Conversion / checks**

    - ``to_mps()`` / ``to_matrix()``: convert to an MPS or dense matrix.
    - ``compute_schmidt_spectrum()`` / ``compute_entanglement_entropy()``: operator bond diagnostics.
    - ``compute_identity_fidelity()``: normalized overlap with the identity.
    - ``check_if_valid_mpo()``: structural bond-dimension consistency check.
    - ``check_if_identity(...)``: heuristic identity check (qubit systems).

    **Notes**

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
    def pauli(
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
        return cls.pauli(
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
        return cls.pauli(
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
    def fermi_hubbard_1d(
        cls,
        length: int,
        t: float,
        u: float,
        *,
        jordan_wigner: bool = False,
    ) -> MPO:
        r"""Construct a 1D Fermi-Hubbard Hamiltonian MPO.

        Without ``jordan_wigner``, builds the standard fermionic MPO on sites with
        local dimension 4. The single-site basis is
        :math:`|0\\rangle, |\\!\\downarrow\\rangle, |\\!\\uparrow\\rangle, |\\!\\uparrow\\downarrow\\rangle`
        (NumPy ``kron`` ordering for :math:`|\\!\\uparrow\\rangle \\otimes |\\!\\downarrow\\rangle`).
        The Hamiltonian is
        :math:`H = -t \\sum_{i,\\sigma} (c^\\dagger_{i,\\sigma} c_{i+1,\\sigma} + \\mathrm{h.c.})
        + U \\sum_i n_{i,\\uparrow} n_{i,\\downarrow}`.

        With ``jordan_wigner=True``, builds the Jordan-Wigner Pauli-string MPO on an
        interleaved spin chain 1↑, 1↓, 2↑, 2↓, ... (local dimension 2):

        .. math::

            U n_{i,\\uparrow} n_{i,\\downarrow}
            = \\frac{U}{4} \\left(I - Z_{i,\\uparrow} - Z_{i,\\downarrow}
            + Z_{i,\\uparrow} Z_{i,\\downarrow}\\right)

            H = \\sum_i \\frac{U}{4} \\left(I - Z_{i,\\uparrow} - Z_{i,\\downarrow}
            + Z_{i,\\uparrow} Z_{i,\\downarrow}\\right)
            - \\frac{t}{2} \\sum_i \\left( X_{\\uparrow,i} Z_{\\downarrow,i} X_{\\uparrow,i+1}
            + Y_{\\uparrow,i} Z_{\\downarrow,i} Y_{\\uparrow,i+1} \\right)
            - \\frac{t}{2} \\sum_i \\left( X_{\\downarrow,i} Z_{\\uparrow,i+1} X_{\\downarrow,i+1}
            + Y_{\\downarrow,i} Z_{\\uparrow,i+1} Y_{\\downarrow,i+1} \\right)

        Without ``jordan_wigner``, the MPO uses fermionic ladder operators on composite
        dimension-4 sites (hard-core constraint per site). Inter-site algebra matches
        that embedding; use ``jordan_wigner=True`` for a Pauli-chain representation
        with full Jordan-Wigner signs between spin orbitals.

        In JW mode ``length`` is the number of **spin orbitals** and must be even and
        at least 2.

        Args:
            length: Chain length. Number of fermionic sites if ``jordan_wigner`` is
                False; number of spin orbitals (even) if True.
            t: Hopping strength.
            u: On-site interaction strength.
            jordan_wigner: If True, use the JW-transformed Pauli MPO; otherwise use
                the fermionic operator MPO.

        Returns:
            An MPO representing the 1D Fermi-Hubbard Hamiltonian.

        Raises:
            ValueError: If ``length`` is invalid for the chosen representation.
        """
        if jordan_wigner:
            if length % 2 != 0 or length < 2:
                msg = "length must be an even integer ≥ 2 (ordering: 1↑,1↓,2↑,2↓,...)."
                raise ValueError(msg)
            return cls._fermi_hubbard_1d_jordan_wigner(length=length, t=t, u=u)
        return cls._fermi_hubbard_1d_fermionic(length=length, t=t, u=u)

    @classmethod
    def _fermi_hubbard_1d_fermionic(cls, length: int, t: float, u: float) -> MPO:
        if length <= 0:
            msg = "length must be positive."
            raise ValueError(msg)

        physical_dimension = 4
        identity = np.eye(physical_dimension, dtype=complex)
        zero = np.zeros_like(identity, dtype=complex)
        c = np.array([[0, 1], [0, 0]], dtype=complex)
        c_dag = np.array([[0, 0], [1, 0]], dtype=complex)
        c_up = np.kron(c, np.eye(2, dtype=complex))
        c_down = np.kron(np.eye(2, dtype=complex), c)
        c_up_dag = np.kron(c_dag, np.eye(2, dtype=complex))
        c_down_dag = np.kron(np.eye(2, dtype=complex), c_dag)
        n_up = np.kron(c_dag @ c, np.eye(2, dtype=complex))
        n_down = np.kron(np.eye(2, dtype=complex), c_dag @ c)
        onsite = u * n_up @ n_down

        # Bond layout matches ``bose_hubbard``: channels
        # 0=identity, 1=c↑†, 2=c↓†, 3=c↑, 4=c↓, 5=accumulator.
        tensor = np.empty((6, 6, physical_dimension, physical_dimension), dtype=object)
        tensor[:, :] = [[zero for _ in range(6)] for _ in range(6)]
        tensor[0, 0] = identity
        tensor[0, 1] = c_up_dag
        tensor[0, 2] = c_down_dag
        tensor[0, 3] = c_up
        tensor[0, 4] = c_down
        tensor[0, 5] = onsite
        tensor[1, 5] = -t * c_up
        tensor[2, 5] = -t * c_down
        tensor[3, 5] = -t * c_up_dag
        tensor[4, 5] = -t * c_down_dag
        tensor[5, 5] = identity

        tensors = [np.transpose(tensor.copy(), (2, 3, 0, 1)).astype(np.complex128) for _ in range(length)]
        tensors[0] = tensors[0][:, :, 0:1, :]
        if length == 1:
            tensors[0] = tensors[0][:, :, :, 5:6]
        else:
            tensors[-1] = tensors[-1][:, :, :, 5:6]

        mpo = cls()
        mpo.tensors = tensors
        mpo.length = length
        mpo.physical_dimension = physical_dimension
        assert mpo.check_if_valid_mpo(), "MPO initialized wrong"
        return mpo

    @classmethod
    def _fermi_hubbard_1d_jordan_wigner(cls, length: int, t: float, u: float) -> MPO:
        num_sites = length // 2
        terms: list[tuple[complex | float, str]] = []
        for site in range(num_sites):
            up, down = 2 * site, 2 * site + 1
            terms.extend([
                (u / 4, ""),
                (-u / 4, f"Z{up}"),
                (-u / 4, f"Z{down}"),
                (u / 4, f"Z{up} Z{down}"),
            ])
        for site in range(num_sites - 1):
            up, down = 2 * site, 2 * site + 1
            up_next = 2 * (site + 1)
            down_next = 2 * (site + 1) + 1
            terms.extend([
                (-t / 2, f"X{up} Z{down} X{up_next}"),
                (-t / 2, f"Y{up} Z{down} Y{up_next}"),
                (-t / 2, f"X{down} Z{up_next} X{down_next}"),
                (-t / 2, f"Y{down} Z{up_next} Y{down_next}"),
            ])

        mpo = cls()
        mpo.from_pauli_sum(terms=terms, length=length, n_sweeps=0)
        return mpo

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
        tensors[0] = tensors[0][:, :, 0:1, :]
        if length == 1:
            tensors[0] = tensors[0][:, :, :, 3:4]
        else:
            tensors[-1] = tensors[-1][:, :, :, 3:4]

        mpo = cls()
        mpo.tensors = tensors
        mpo.length = length

        # Backward-compat: single attribute even though dims alternate.
        mpo.physical_dimension = local_dim

        assert mpo.check_if_valid_mpo(), "MPO initialized wrong"
        return mpo

    @classmethod
    def trapped_ions_position_grid(
        cls,
        positions: NDArray[np.float64],
        masses: Sequence[float],
        omega: float,
        *,
        trap_center: float = 0.0,
        hbar: float = 1.0,
        coulomb_strength: float = 0.0,
        softening_length: float | None = None,
        coulomb_cutoff: float = 1e-12,
        max_bond_dim: int | None = None,
    ) -> MPO:
        r"""Construct a static one- or two-ion Hamiltonian on a uniform position grid.

        Each ion is one MPO site whose local basis consists of the supplied position-grid
        points. The Hamiltonian is:

        H = sum_i[-hbar^2/(2*m_i) * d^2/dx_i^2 + (1/2)*m_i*omega^2*(x_i - q)^2]
            + g / sqrt((x_1 - x_2)^2 + a^2)

        The kinetic energy uses a centered second-order finite difference. For two ions,
        an SVD of the diagonal Coulomb coefficient matrix produces the MPO interaction
        channels. Discarding singular values according to ``coulomb_cutoff`` or
        ``max_bond_dim`` approximates only the Coulomb term.

        Args:
            positions: Uniformly spaced one-dimensional position grid.
            masses: One or two positive ion masses. Each ion becomes one MPO site.
            omega: Non-negative harmonic trap angular frequency.
            trap_center: Center ``q`` of the static harmonic trap.
            hbar: Positive reduced Planck constant.
            coulomb_strength: Coulomb prefactor ``g``. Must be zero for one ion.
            softening_length: Positive short-distance regularizer ``a`` to avoid Coulomb
                singularity. Defaults to the grid spacing for two ions.
            coulomb_cutoff: Relative SVD cutoff. Singular values no larger than this
                fraction of the largest one are discarded. Set to zero for the exact
                grid interaction.
            max_bond_dim: Optional cap on the total MPO bond dimension. For two ions,
                two channels are reserved for the local Hamiltonians.

        Returns:
            MPO for the static trapped-ion Hamiltonian in energy units.

        Raises:
            ValueError: If the grid or physical parameters are invalid, if the number of
                masses is not one or two, or if ``max_bond_dim`` is too small.

        Notes:
            YAQS time evolution applies ``exp(-1j * dt * H)``. When using dimensional
            quantities, rescale the returned MPO to represent ``H / hbar`` or measure
            time and energy in compatible units.
        """
        grid = np.asarray(positions, dtype=np.float64)
        if grid.ndim != 1 or grid.size < 3:
            msg = "positions must be a one-dimensional grid with at least three points."
            raise ValueError(msg)
        if not np.all(np.isfinite(grid)):
            msg = "positions must contain only finite values."
            raise ValueError(msg)
        spacings = np.diff(grid)
        if np.any(spacings <= 0.0) or not np.allclose(spacings, spacings[0], rtol=1e-12, atol=1e-15):
            msg = "positions must be strictly increasing and uniformly spaced."
            raise ValueError(msg)

        ion_masses = np.asarray(masses, dtype=np.float64)
        if ion_masses.ndim != 1 or ion_masses.size not in {1, 2}:
            msg = "masses must contain exactly one or two ion masses."
            raise ValueError(msg)
        if not np.all(np.isfinite(ion_masses)) or np.any(ion_masses <= 0.0):
            msg = "masses must contain only finite positive values."
            raise ValueError(msg)
        if not np.isfinite(omega) or omega < 0.0:
            msg = "omega must be finite and non-negative."
            raise ValueError(msg)
        if not np.isfinite(trap_center):
            msg = "trap_center must be finite."
            raise ValueError(msg)
        if not np.isfinite(hbar) or hbar <= 0.0:
            msg = "hbar must be finite and positive."
            raise ValueError(msg)
        if not np.isfinite(coulomb_strength):
            msg = "coulomb_strength must be finite."
            raise ValueError(msg)
        if not np.isfinite(coulomb_cutoff) or not 0.0 <= coulomb_cutoff < 1.0:
            msg = "coulomb_cutoff must be finite and satisfy 0 <= coulomb_cutoff < 1."
            raise ValueError(msg)
        if max_bond_dim is not None and max_bond_dim < 2:
            msg = "max_bond_dim must be at least 2."
            raise ValueError(msg)
        if ion_masses.size == 1 and coulomb_strength:
            msg = "coulomb_strength must be zero for a one-ion Hamiltonian."
            raise ValueError(msg)

        d = grid.size
        dx = float(spacings[0])

        def local_hamiltonian(mass: float) -> ComplexTensor:
            kinetic_diagonal = np.full(d, hbar**2 / (mass * dx**2), dtype=np.float64)
            kinetic_off_diagonal = np.full(d - 1, -(hbar**2 / (2.0 * mass * dx**2)), dtype=np.float64)
            kinetic = (
                np.diag(kinetic_diagonal) + np.diag(kinetic_off_diagonal, k=-1) + np.diag(kinetic_off_diagonal, k=1)
            )
            potential = 0.5 * mass * omega**2 * (grid - trap_center) ** 2
            return np.asarray(kinetic + np.diag(potential), dtype=np.complex128)

        local_terms = [local_hamiltonian(float(mass)) for mass in ion_masses]
        mpo = cls()
        mpo.length = int(ion_masses.size)
        mpo.physical_dimension = d

        if ion_masses.size == 1:
            mpo.tensors = [local_terms[0][:, :, None, None]]
            assert mpo.check_if_valid_mpo(), "MPO initialized wrong"
            return mpo

        if softening_length is None:
            softening_length = dx
        if not np.isfinite(softening_length) or softening_length <= 0.0:
            msg = "softening_length must be finite and positive."
            raise ValueError(msg)

        distance = grid[:, None] - grid[None, :]
        coulomb = coulomb_strength / np.sqrt(distance**2 + softening_length**2)
        u, singular_values, vh = linalg.svd(coulomb, full_matrices=False)
        if not singular_values[0]:
            rank = 0
        else:
            rank = int(np.count_nonzero(singular_values > coulomb_cutoff * singular_values[0]))
        if max_bond_dim is not None:
            rank = min(rank, max_bond_dim - 2)

        bond_dimension = rank + 2
        identity = np.eye(d, dtype=np.complex128)
        left = np.zeros((d, d, 1, bond_dimension), dtype=np.complex128)
        right = np.zeros((d, d, bond_dimension, 1), dtype=np.complex128)
        left[:, :, 0, 0] = local_terms[0]
        right[:, :, 0, 0] = identity
        left[:, :, 0, 1] = identity
        right[:, :, 1, 0] = local_terms[1]

        for alpha in range(rank):
            scale = math.sqrt(float(singular_values[alpha]))
            left[:, :, 0, alpha + 2] = np.diag(scale * u[:, alpha])
            right[:, :, alpha + 2, 0] = np.diag(scale * vh[alpha, :])

        mpo.tensors = [left, right]
        assert mpo.check_if_valid_mpo(), "MPO initialized wrong"
        return mpo

    @classmethod
    def identity(cls, length: int, physical_dimension: int = 2) -> MPO:
        """Construct an identity MPO.

        Args:
            length: Number of sites.
            physical_dimension: Local Hilbert-space dimension per site (default 2 for qubits).

        Returns:
            An MPO representing the identity operator on ``length`` sites.
        """
        mpo = cls()
        mpo.init_identity(length, physical_dimension=physical_dimension)
        return mpo

    @classmethod
    def from_gate(cls, gate: BaseGate, chain_length: int) -> MPO:
        """Build an MPO for a two-qubit gate on a chain.

        When ``chain_length`` equals the gate support size, the MPO contains only the
        extended gate tensors. When ``chain_length`` is larger, identity MPO sites are
        placed outside the support interval ``[min(sites), max(sites)]``.

        Reuses :attr:`~mqt.yaqs.core.libraries.gate_library.BaseGate.mpo_tensors` when
        already populated for the gate support.

        Args:
            gate: Two-qubit gate with ``sites`` and ``tensor`` (or ``mpo_tensors``) set.
            chain_length: Total number of MPO sites (support length or full MPS length).

        Returns:
            MPO ready for :meth:`multiply` on an MPS or another MPO.

        Raises:
            ValueError: If the gate is not two-qubit or ``chain_length`` is too small.
        """
        if gate.interaction != 2:
            msg = f"from_gate requires a two-qubit gate, got interaction {gate.interaction}."
            raise ValueError(msg)
        if len(gate.sites) != 2:
            msg = f"from_gate requires exactly two sites, got {len(gate.sites)}."
            raise ValueError(msg)

        first_site = min(gate.sites[0], gate.sites[1])
        last_site = max(gate.sites[0], gate.sites[1])
        support_len = last_site - first_site + 1
        if chain_length < support_len:
            msg = f"chain_length {chain_length} is smaller than gate support length {support_len}."
            raise ValueError(msg)

        support = get_support_mpo(gate, first_site=first_site, last_site=last_site)
        if chain_length == support_len:
            tensors = support
        else:
            phys_dim = support[0].shape[0]
            identity_site = make_identity_site(phys_dim)
            tensors = []
            for site in range(chain_length):
                if site < first_site or site > last_site:
                    tensors.append(np.array(identity_site, copy=True))
                else:
                    tensors.append(support[site - first_site])

        mpo = cls()
        mpo.custom(tensors, transpose=False)
        return mpo

    def init_identity(self, length: int, physical_dimension: int = 2) -> None:
        """Initialize this MPO in place as the identity operator.

        Prefer :meth:`identity` when constructing a new MPO.

        Args:
            length: Number of sites.
            physical_dimension: Local dimension per site (default 2).
        """
        site = make_identity_site(physical_dimension)
        self.length = length
        self.physical_dimension = physical_dimension

        self.tensors = []
        for _ in range(length):
            self.tensors.append(np.array(site, copy=True))

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
        if transpose:
            self.physical_dimension = self.tensors[0].shape[0]
        else:
            self.physical_dimension = self.tensors[0].shape[2]

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

            u, s, vh = linalg.svd(matrix, full_matrices=False)
            keep = linalg.truncate(s, mode="hard_cutoff", threshold=tol, max_bond_dim=max_bond_dim, min_keep=1)

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

    @overload
    def multiply(
        self,
        other: MPS,
        *,
        sim_params: StrongSimParams | WeakSimParams | None = None,
        compress: bool = True,
    ) -> None: ...

    @overload
    def multiply(
        self,
        other: MPO,
        *,
        start_site: int = 0,
        conjugate: bool = False,
        compress: bool = True,
        tol: float = 1e-12,
        max_bond_dim: int | None = None,
        n_sweeps: int = 1,
        directions: str = "lr_rl",
    ) -> None: ...

    def multiply(
        self,
        other: MPS | MPO,
        *,
        sim_params: StrongSimParams | WeakSimParams | None = None,
        compress: bool = True,
        start_site: int = 0,
        conjugate: bool = False,
        tol: float = 1e-12,
        max_bond_dim: int | None = None,
        n_sweeps: int = 1,
        directions: str = "lr_rl",
    ) -> None:
        """Left-multiply this MPO into ``other`` (MPS or MPO), updating ``other`` in place.

        For an :class:`~mqt.yaqs.core.data_structures.mps.MPS`, each site is updated by
        :func:`~mqt.yaqs.core.data_structures.mpo_utils.contract_mpo_site_with_mps_site`,
        optionally followed by a two-site SVD compression sweep driven by ``sim_params``.

        For another :class:`MPO`, each site uses the equivalence-checking
        :func:`~mqt.yaqs.core.data_structures.mpo_utils.contract_mpo_site_with_mpo_site`
        contraction (``abcd,cefg``), optionally
        followed by :meth:`compress` on ``other``.

        Args:
            other: Target MPS or MPO to update in place.
            sim_params: Truncation settings for MPS compression (required if ``compress``
                is True and ``other`` is an MPS).
            compress: Whether to run a compression sweep after contraction.
            start_site: When ``len(self) != len(other)``, index on ``other`` where this
                MPO is embedded (only for MPO targets).
            conjugate: Use the conjugated MPO--MPO contraction (MPO targets only).
            tol: MPO compression threshold (MPO targets only).
            max_bond_dim: Optional bond-dimension cap for MPO compression.
            n_sweeps: Number of MPO compression sweeps.
            directions: MPO compression sweep schedule (see :meth:`compress`).

        Raises:
            TypeError: If ``other`` is neither an MPS nor an MPO.
        """
        if isinstance(other, MPS):
            self._multiply_mps(
                other,
                sim_params=sim_params,
                compress=compress,
            )
            return

        if not isinstance(other, MPO):
            msg = f"multiply expects MPS or MPO, got {type(other).__name__}."
            raise TypeError(msg)

        self._multiply_mpo(
            other,
            start_site=start_site,
            conjugate=conjugate,
            compress=compress,
            tol=tol,
            max_bond_dim=max_bond_dim,
            n_sweeps=n_sweeps,
            directions=directions,
        )

    def _multiply_mps(
        self,
        state: MPS,
        *,
        sim_params: StrongSimParams | WeakSimParams | None,
        compress: bool,
    ) -> None:
        """Apply this MPO to ``state`` with optional compression.

        Raises:
            ValueError: On length mismatch or missing ``sim_params`` when compressing.

        Notes:
            Applies the MPO at every site and invalidates the tracked orthogonality
            center (``set_center(None)``). With ``compress=False`` the center remains
            ``None`` until canonicalization or compression is performed elsewhere.
            Compression re-establishes a center via
            :meth:`~mqt.yaqs.core.data_structures.mps.MPS.compress`.
        """
        if len(self.tensors) != state.length:
            msg = f"MPO length {len(self.tensors)} does not match MPS length {state.length}."
            raise ValueError(msg)

        for site, operator in enumerate(self.tensors):
            state.tensors[site] = contract_mpo_site_with_mps_site(operator, state.tensors[site])

        state.set_center(None)

        if not compress:
            return
        if sim_params is None:
            msg = "sim_params is required when compress=True for MPO.multiply(MPS)."
            raise ValueError(msg)

        state.compress(
            sim_params.svd_threshold,
            max_bond_dim=sim_params.max_bond_dim,
            trunc_mode=cast("TruncMode", sim_params.trunc_mode),
        )

    def _multiply_mpo(
        self,
        other: MPO,
        *,
        start_site: int,
        conjugate: bool,
        compress: bool,
        tol: float,
        max_bond_dim: int | None,
        n_sweeps: int,
        directions: str,
    ) -> None:
        """Left-multiply this MPO into ``other``.

        Raises:
            ValueError: If this MPO cannot be embedded at ``start_site``.
        """
        gate_len = len(self.tensors)
        target_len = len(other.tensors)

        if gate_len == target_len:
            sites = range(target_len)
        elif start_site >= 0 and start_site + gate_len <= target_len:
            sites = range(start_site, start_site + gate_len)
        else:
            msg = f"Cannot embed MPO of length {gate_len} at start_site={start_site} into MPO of length {target_len}."
            raise ValueError(msg)

        for gate_site, target_site in enumerate(sites):
            other.tensors[target_site] = contract_mpo_site_with_mpo_site(
                self.tensors[gate_site],
                other.tensors[target_site],
                conjugate=conjugate,
            )

        if compress:
            other.compress(
                tol=tol,
                max_bond_dim=max_bond_dim,
                n_sweeps=n_sweeps,
                directions=directions,
            )

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

    def _compute_bond_schmidt_spectrum(self, sites: list[int]) -> NDArray[np.float64]:
        """Return operator Schmidt singular values across a nearest-neighbor bond."""
        i, j = sites
        mps = self.to_mps()
        mps.set_canonical_form(orthogonality_center=j, decomposition="QR")

        a, b = mps.tensors[i], mps.tensors[j]
        theta = np.tensordot(a, b, axes=(2, 1))
        theta_matrix = np.reshape(theta, (a.shape[0] * a.shape[1], b.shape[0] * b.shape[2]))
        if theta_matrix.size == 0:
            return np.array([], dtype=np.float64)

        singular_values = np.linalg.svd(
            np.asarray(theta_matrix, dtype=np.complex128),
            compute_uv=False,
            full_matrices=False,
        )
        return np.asarray(singular_values, dtype=np.float64)

    def compute_schmidt_spectrum(self, cut: int) -> NDArray[np.float64]:
        """Compute operator Schmidt singular values across an integer bond cut.

        Args:
            cut: Bond cut index in ``[0, length]``. Internal cuts use bond ``(cut - 1, cut)``;
                boundary cuts ``0`` and ``length`` return the operator Frobenius norm.

        Returns:
            One-dimensional array of singular values.

        Raises:
            TypeError: If ``cut`` is not an ``int``.
            ValueError: If ``cut`` is out of range.
        """
        if isinstance(cut, bool) or not isinstance(cut, int):
            msg = f"cut must be int, got {cut!r}"
            raise TypeError(msg)
        if cut < 0 or cut > self.length:
            msg = f"cut out of range: {cut} for length={self.length}"
            raise ValueError(msg)
        if cut in {0, self.length}:
            fro_norm = float(np.linalg.norm(np.asarray(self.to_matrix(), dtype=np.complex128), ord="fro"))
            return np.array([fro_norm], dtype=np.float64)

        return self._compute_bond_schmidt_spectrum([cut - 1, cut])

    def compute_entanglement_entropy(self, cut: int, *, base: float = math.e) -> float:
        """Compute operator entanglement entropy across an integer bond cut.

        Args:
            cut: Bond cut index passed to :meth:`compute_schmidt_spectrum`.
            base: Logarithm base for the entropy (default natural log).

        Returns:
            Von Neumann entropy of the normalized Schmidt spectrum.

        Raises:
            ValueError: If ``cut`` or ``base`` is invalid.
        """
        base_float = float(base)
        if not np.isfinite(base_float) or base_float <= 0.0 or math.isclose(base_float, 1.0):
            msg = f"Entropy base must be finite, >0, and !=1; got {base!r}"
            raise ValueError(msg)

        schmidt_values = self.compute_schmidt_spectrum(cut)
        if schmidt_values.size == 0:
            return 0.0

        max_schmidt = float(np.max(np.abs(schmidt_values)))
        if not np.isfinite(max_schmidt) or max_schmidt <= 0.0:
            return 0.0

        probabilities = np.square(schmidt_values / max_schmidt)
        normalization = float(np.sum(probabilities, dtype=np.float64))
        if normalization <= 0.0:
            return 0.0
        probabilities /= normalization

        eps = np.finfo(np.float64).tiny
        nonzero = probabilities > eps
        entropy = -np.sum(probabilities[nonzero] * np.log(probabilities[nonzero]), dtype=np.float64) / math.log(
            base_float
        )
        return float(max(entropy, 0.0))

    def compute_identity_fidelity(self) -> float:
        """Compute normalized overlap of this MPO with the identity operator.

        Returns:
            ``|Tr(O)| / d`` where ``d`` is the Hilbert-space dimension.
        """
        local_dims = [int(tensor.shape[0]) for tensor in self.tensors]
        identity_mpo = MPO()
        identity_mpo.custom(
            [make_identity_site(d) for d in local_dims],
            transpose=False,
        )
        identity_mps = identity_mpo.to_mps()
        mps = self.to_mps()
        trace = mps.scalar_product(identity_mps)
        hilbert_dim = int(np.prod(local_dims, dtype=np.int64))
        return float(np.abs(trace) / hilbert_dim)

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
            _d_out, _d_in, d_left, d_right = tensor.shape

            next_operators = {}

            for beta in range(d_right):
                accumulated = None

                for alpha in range(d_left):
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

                    accumulated = term if accumulated is None else accumulated + term

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

        if np.ndim(mat) != 2:
            msg = "Matrix must be a 2-D array for uniform MPO factorization."
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
            if cutoff <= 0.0:
                r = int(s.size)
                if max_bond is not None:
                    r = min(r, max_bond)
                return r
            return linalg.truncate(
                s,
                mode="hard_cutoff",
                threshold=cutoff,
                max_bond_dim=max_bond,
                min_keep=1,
            )

        for k in range(n - 1):
            rest = d ** (n - k - 1)

            rem = rem.reshape(left_rank, d, rest, d, rest)
            rem_perm = np.transpose(rem, (1, 3, 0, 2, 4))
            x = rem_perm.reshape(d * d * left_rank, rest * rest)

            u, s, vh = linalg.svd(x, full_matrices=False)

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
            if tensor.shape[2] != right_bond:
                return False
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
        return self.compute_identity_fidelity() >= fidelity

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
