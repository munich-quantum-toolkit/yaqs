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
from scipy import linalg as scipy_linalg

from .. import linalg
from ..libraries.gate_library import BaseGate, Destroy
from .mpo_utils import (
    contract_mpo_site_with_mpo_site,
    contract_mpo_site_with_mps_site,
    get_support_mpo,
    make_identity_site,
)
from .mps import MPS

if TYPE_CHECKING:
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
    - ``schmidt_values()`` / ``operator_entanglement_entropy()``: bond-spectrum diagnostics.
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
        """
        if len(self.tensors) != state.length:
            msg = f"MPO length {len(self.tensors)} does not match MPS length {state.length}."
            raise ValueError(msg)

        for site, operator in enumerate(self.tensors):
            state.tensors[site] = contract_mpo_site_with_mps_site(operator, state.tensors[site])

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

    def _full_schmidt_values_for_bond(
        self,
        sites: list[int],
        decomposition: str = "QR",
    ) -> NDArray[np.float64]:
        """Return the complete operator Schmidt values across a nearest-neighbor bond.

        Returns:
            Singular values for the requested bond.

        Raises:
            ValueError: If ``decomposition`` is unsupported.
        """
        assert len(sites) == 2, "Schmidt spectrum is defined on a bond (two adjacent sites)."
        i, j = sites
        assert i + 1 == j, "Schmidt spectrum is only defined for nearest-neighbor cut."
        if decomposition not in {"QR", "SVD"}:
            msg = f"Unsupported decomposition: {decomposition!r}"
            raise ValueError(msg)

        mps = self.to_mps()
        mps.set_canonical_form(orthogonality_center=j, decomposition=decomposition)

        a, b = mps.tensors[i], mps.tensors[j]
        theta = np.tensordot(a, b, axes=(2, 1))
        theta_matrix = np.reshape(theta, (a.shape[0] * a.shape[1], b.shape[0] * b.shape[2]))
        if theta_matrix.size == 0:
            return np.array([], dtype=np.float64)

        return self._svd_values_with_fallback(
            np.asarray(theta_matrix, dtype=np.complex128),
            stage=f"MPO._full_schmidt_values_for_bond sites={sites}",
        )

    def _full_schmidt_values_for_cut(
        self,
        cut: str | int = "center",
        decomposition: str = "QR",
    ) -> NDArray[np.float64]:
        """Return the complete operator Schmidt values across a spatial cut."""
        cut_index = self._resolve_cut_index(cut=cut, length=len(self.tensors))
        if cut_index in {0, len(self.tensors)}:
            fro_norm = float(np.linalg.norm(np.asarray(self.to_matrix(), dtype=np.complex128), ord="fro"))
            return np.array([fro_norm], dtype=np.float64)

        singular_values = self._full_schmidt_values_for_bond(
            [cut_index - 1, cut_index],
            decomposition=decomposition,
        )
        return np.asarray(singular_values, dtype=np.float64)

    def get_entropy(self, sites: list[int], decomposition: str = "QR") -> np.float64:
        """Return the operator entanglement entropy across a nearest-neighbor MPO bond."""
        assert len(sites) == 2, "Entropy is defined on a bond (two adjacent sites)."
        i, j = sites
        assert i + 1 == j, "Entropy is only defined for nearest-neighbor cut."

        singular_values = self._full_schmidt_values_for_bond(sites, decomposition=decomposition)
        if singular_values.size == 0:
            return np.float64(0.0)

        return np.float64(self.entropy_from_schmidt_values(singular_values))

    def get_schmidt_spectrum(self, sites: list[int], decomposition: str = "QR") -> NDArray[np.float64]:
        """Return the operator Schmidt spectrum across a nearest-neighbor MPO bond."""
        assert len(sites) == 2, "Schmidt spectrum is defined on a bond (two adjacent sites)."
        i, j = sites
        assert i + 1 == j, "Schmidt spectrum is only defined for nearest-neighbor cut."

        singular_values = self._full_schmidt_values_for_bond(sites, decomposition=decomposition)

        top_schmidt_vals = 500
        padded = np.full(top_schmidt_vals, np.nan, dtype=np.float64)
        padded[: min(top_schmidt_vals, singular_values.size)] = singular_values[:top_schmidt_vals]
        return padded

    @staticmethod
    def _resolve_cut_index(cut: str | int, length: int) -> int:
        """Resolve a cut specifier to a valid integer cut index.

        Returns:
            Integer cut index in ``[0, length]``.

        Raises:
            ValueError: If ``cut`` is invalid or out of range.
        """
        if cut == "center":
            cut_index = length // 2
        elif isinstance(cut, int) and not isinstance(cut, bool):
            cut_index = cut
        else:
            msg = f"cut must be 'center' or int, got {cut!r}"
            raise ValueError(msg)

        if cut_index < 0 or cut_index > length:
            msg = f"cut out of range: {cut_index} for length={length}"
            raise ValueError(msg)
        return cut_index

    @staticmethod
    def _array_norm(array: NDArray[np.complex128] | NDArray[np.float64]) -> float:
        """Return a stable norm summary for diagnostics and validation."""
        arr = np.asarray(array)
        if arr.ndim == 0:
            return float(abs(arr))
        if arr.ndim == 2:
            return float(np.linalg.norm(arr, ord="fro"))
        return float(np.linalg.norm(arr.reshape(-1)))

    @classmethod
    def _validate_numeric_array(
        cls,
        array: NDArray[np.complex128] | NDArray[np.float64],
        *,
        stage: str,
        ndim: int | None = None,
        expected_shape: tuple[int, ...] | None = None,
        dtype: type[np.complex128 | np.float64] = np.complex128,
    ) -> NDArray[np.complex128] | NDArray[np.float64]:
        """Validate numerical arrays before dense reshapes or decompositions.

        Returns:
            Validated array cast to ``dtype``.

        Raises:
            ValueError: If shape, dimensionality, finiteness, or norm validation fails.
        """
        arr = np.asarray(array, dtype=dtype)
        if ndim is not None and arr.ndim != int(ndim):
            msg = f"Expected {stage} to have ndim={int(ndim)}, got shape={arr.shape}, dtype={arr.dtype}"
            raise ValueError(msg)
        if expected_shape is not None and arr.shape != expected_shape:
            msg = f"Expected {stage} to have shape={expected_shape}, got shape={arr.shape}, dtype={arr.dtype}"
            raise ValueError(msg)
        finite_mask = np.isfinite(arr)
        if not np.all(finite_mask):
            nonfinite_count = int(arr.size - np.count_nonzero(finite_mask))
            msg = (
                f"Non-finite values detected at {stage}: shape={arr.shape}, dtype={arr.dtype}, "
                f"nonfinite_count={nonfinite_count}"
            )
            raise ValueError(msg)
        norm_value = cls._array_norm(arr)
        if not np.isfinite(norm_value):
            msg = f"Invalid norm detected at {stage}: shape={arr.shape}, dtype={arr.dtype}, norm={norm_value!r}"
            raise ValueError(msg)
        return arr

    @classmethod
    def _validated_dense_channel_matrix(
        cls,
        channel_dense: NDArray[np.complex128],
        *,
        n_sites: int,
        local_dim: int,
        stage: str,
    ) -> NDArray[np.complex128]:
        expected_dim = int(local_dim) ** int(n_sites)
        channel = cls._validate_numeric_array(
            channel_dense,
            stage=stage,
            ndim=2,
            expected_shape=(expected_dim, expected_dim),
            dtype=np.complex128,
        )
        return np.asarray(channel, dtype=np.complex128)

    @classmethod
    def _svd_values_with_fallback(
        cls,
        matrix: NDArray[np.complex128],
        *,
        stage: str,
    ) -> NDArray[np.float64]:
        """Compute singular values with a SciPy fallback on convergence failure.

        Returns:
            One-dimensional singular-value array.

        Raises:
            RuntimeError: If both SVD backends fail.
        """
        validated = np.asarray(
            cls._validate_numeric_array(matrix, stage=stage, ndim=2, dtype=np.complex128),
            dtype=np.complex128,
        )
        try:
            singular_values = np.linalg.svd(validated, compute_uv=False, full_matrices=False)
        except np.linalg.LinAlgError:
            try:
                singular_values = scipy_linalg.svd(
                    validated,
                    compute_uv=False,
                    full_matrices=False,
                    check_finite=True,
                    lapack_driver="gesvd",
                )
            except Exception as fallback_exc:
                norm_value = cls._array_norm(validated)
                msg = f"SVD failed at {stage}: shape={validated.shape}, dtype={validated.dtype}, norm={norm_value!r}"
                raise RuntimeError(msg) from fallback_exc
        svals = np.asarray(singular_values, dtype=np.float64)
        cls._validate_numeric_array(svals, stage=f"{stage} singular_values", ndim=1, dtype=np.float64)
        return svals

    @classmethod
    def _svd_with_fallback(
        cls,
        matrix: NDArray[np.complex128],
        *,
        stage: str,
    ) -> tuple[NDArray[np.complex128], NDArray[np.float64], NDArray[np.complex128]]:
        """Compute a full SVD with a SciPy fallback on convergence failure.

        Returns:
            Tuple ``(U, S, Vh)`` from the decomposition.

        Raises:
            RuntimeError: If both SVD backends fail.
        """
        validated = np.asarray(
            cls._validate_numeric_array(matrix, stage=stage, ndim=2, dtype=np.complex128),
            dtype=np.complex128,
        )
        try:
            u_mat, s_vals, vh_mat = np.linalg.svd(validated, full_matrices=False)
        except np.linalg.LinAlgError:
            try:
                u_mat, s_vals, vh_mat = scipy_linalg.svd(
                    validated,
                    full_matrices=False,
                    check_finite=True,
                    lapack_driver="gesvd",
                )
            except Exception as fallback_exc:
                norm_value = cls._array_norm(validated)
                msg = f"SVD failed at {stage}: shape={validated.shape}, dtype={validated.dtype}, norm={norm_value!r}"
                raise RuntimeError(msg) from fallback_exc
        u_valid = np.asarray(
            cls._validate_numeric_array(u_mat, stage=f"{stage} left_vectors", ndim=2, dtype=np.complex128),
            dtype=np.complex128,
        )
        s_valid = np.asarray(
            cls._validate_numeric_array(s_vals, stage=f"{stage} singular_values", ndim=1, dtype=np.float64),
            dtype=np.float64,
        )
        vh_valid = np.asarray(
            cls._validate_numeric_array(vh_mat, stage=f"{stage} right_vectors", ndim=2, dtype=np.complex128),
            dtype=np.complex128,
        )
        return u_valid, s_valid, vh_valid

    @staticmethod
    def entropy_from_probabilities(probabilities: NDArray[np.float64], *, base: float = math.e) -> float:
        """Compute entropy from a normalized probability vector.

        Returns:
            Entropy in the requested logarithm base.

        Raises:
            ValueError: If ``base`` or probabilities are invalid.
            RuntimeError: If probability normalization or the entropy is invalid.
        """
        base_float = float(base)
        if not np.isfinite(base_float) or base_float <= 0.0 or math.isclose(base_float, 1.0):
            msg = f"Entropy base must be finite, >0, and !=1; got {base!r}"
            raise ValueError(msg)

        probs = np.array(probabilities, dtype=np.float64, copy=True).reshape(-1)
        if probs.size == 0:
            return 0.0
        if not np.all(np.isfinite(probs)):
            msg = f"Non-finite probabilities encountered while computing entropy: shape={probs.shape}"
            raise ValueError(msg)
        if np.any(probs < 0.0):
            min_probability = float(np.min(probs))
            msg = f"Negative probabilities encountered while computing entropy: min={min_probability!r}"
            raise ValueError(msg)

        normalization = float(np.sum(probs, dtype=np.float64))
        if not np.isfinite(normalization) or normalization <= 0.0:
            msg = f"Invalid probability normalization while computing entropy: sum={normalization!r}"
            raise RuntimeError(msg)

        probs = np.divide(probs, normalization)
        nonzero = probs > np.finfo(np.float64).tiny
        entropy = -np.sum(probs[nonzero] * np.log(probs[nonzero]), dtype=np.float64) / math.log(base_float)
        if not np.isfinite(entropy):
            msg = f"Invalid entropy computed from probabilities: entropy={entropy!r}"
            raise RuntimeError(msg)
        return float(max(entropy, 0.0))

    @classmethod
    def entropy_from_schmidt_values(
        cls,
        schmidt_values: NDArray[np.float64],
        *,
        base: float = math.e,
    ) -> float:
        """Compute entropy directly from Schmidt values.

        Returns:
            Entropy of the normalized Schmidt spectrum.
        """
        probabilities = cls.normalized_schmidt_probabilities(schmidt_values)
        return cls.entropy_from_probabilities(probabilities, base=base)

    def _dense_fused_site_schmidt_matrix(self, cut: str | int = "center") -> NDArray[np.complex128]:
        """Build the exact dense Schmidt matrix across a spatial cut.

        Returns:
            Dense Schmidt matrix for the requested cut.

        Raises:
            ValueError: If the MPO has no tensors or ``cut`` is invalid.
            RuntimeError: If the dense operator shape is inconsistent.
        """
        tensors_raw = [np.asarray(tensor, dtype=np.complex128) for tensor in self.tensors]
        if not tensors_raw:
            msg = "MPO has no tensors."
            raise ValueError(msg)

        cut_index = self._resolve_cut_index(cut=cut, length=len(tensors_raw))
        out_dims = [int(tensor.shape[0]) for tensor in tensors_raw]
        in_dims = [int(tensor.shape[1]) for tensor in tensors_raw]
        dense_operator = np.asarray(self.to_matrix(), dtype=np.complex128)
        expected_shape = (math.prod(out_dims), math.prod(in_dims))
        if dense_operator.shape != expected_shape:
            msg = (
                "Dense MPO matrix shape does not match the product of local legs: "
                f"{dense_operator.shape} vs {expected_shape}"
            )
            raise RuntimeError(msg)

        dense_tensor = np.reshape(dense_operator, (*out_dims, *in_dims))
        interleaved_axes: list[int] = []
        num_sites = len(tensors_raw)
        for site_index in range(num_sites):
            interleaved_axes.extend([site_index, num_sites + site_index])
        fused_tensor = np.transpose(dense_tensor, axes=interleaved_axes)

        left_dim = math.prod(out_dims[:cut_index]) * math.prod(in_dims[:cut_index])
        right_dim = math.prod(out_dims[cut_index:]) * math.prod(in_dims[cut_index:])
        return np.reshape(fused_tensor, (left_dim, right_dim))

    @staticmethod
    def normalized_schmidt_probabilities(schmidt_values: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalize Schmidt values into probabilities.

        Returns:
            Normalized probability vector.

        Raises:
            ValueError: If Schmidt values or probabilities are non-finite.
        """
        svals = np.asarray(schmidt_values, dtype=np.float64).reshape(-1)
        if svals.size == 0:
            return np.array([1.0], dtype=np.float64)
        if not np.all(np.isfinite(svals)):
            msg = f"Non-finite Schmidt values encountered: shape={svals.shape}"
            raise ValueError(msg)

        max_schmidt = float(np.max(np.abs(svals)))
        if not np.isfinite(max_schmidt) or max_schmidt <= 0.0:
            return np.array([1.0], dtype=np.float64)

        probabilities = np.square(svals / max_schmidt)
        if not np.all(np.isfinite(probabilities)):
            msg = f"Non-finite Schmidt probabilities encountered: shape={probabilities.shape}"
            raise ValueError(msg)
        normalization = float(np.sum(probabilities, dtype=np.float64))
        if not np.isfinite(normalization) or normalization <= 0.0:
            return np.array([1.0], dtype=np.float64)
        return np.asarray(probabilities / normalization, dtype=np.float64)

    @staticmethod
    def weighted_spectrum_distance(
        probabilities: NDArray[np.float64],
        reference: NDArray[np.float64],
    ) -> float:
        """Return the normalized weighted L1 Schmidt-spectrum distance.

        Raises:
            ValueError: If either spectrum contains non-finite values.
        """
        lhs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        rhs = np.asarray(reference, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(lhs)) or not np.all(np.isfinite(rhs)):
            msg = (
                "Weighted spectrum distance requires finite probabilities: "
                f"lhs_finite={bool(np.all(np.isfinite(lhs)))}, rhs_finite={bool(np.all(np.isfinite(rhs)))}"
            )
            raise ValueError(msg)
        size = max(int(lhs.size), int(rhs.size), 1)
        lhs_pad = np.zeros(size, dtype=np.float64)
        rhs_pad = np.zeros(size, dtype=np.float64)
        lhs_pad[: lhs.size] = lhs
        rhs_pad[: rhs.size] = rhs
        weights = 1.0 / (1.0 + np.arange(size, dtype=np.float64))
        return float(np.sum(weights * np.abs(lhs_pad - rhs_pad), dtype=np.float64) / np.sum(weights, dtype=np.float64))

    @classmethod
    def dense_channel_to_tensors(
        cls,
        channel_dense: NDArray[np.complex128],
        *,
        n_sites: int,
        local_dim: int = 4,
        svd_cutoff: float = 1e-12,
    ) -> list[NDArray[np.complex128]]:
        """Decompose a dense channel matrix into MPO tensors.

        Returns:
            MPO tensors in YAQS storage order.
        """
        channel = cls._validated_dense_channel_matrix(
            channel_dense,
            n_sites=n_sites,
            local_dim=local_dim,
            stage="dense_channel_to_tensors input",
        )

        tensor = channel.reshape([int(local_dim)] * (2 * int(n_sites)))
        interleaved_axes: list[int] = []
        for site_index in range(int(n_sites)):
            interleaved_axes.extend((site_index, site_index + int(n_sites)))
        remainder = np.transpose(tensor, interleaved_axes)

        tensors: list[NDArray[np.complex128]] = []
        chi_left = 1
        cutoff = float(svd_cutoff)
        for site_index in range(int(n_sites) - 1):
            remainder = np.reshape(remainder, (chi_left * int(local_dim) * int(local_dim), -1))
            remainder = np.asarray(
                cls._validate_numeric_array(
                    remainder,
                    stage=f"dense_channel_to_tensors remainder site={site_index}",
                    ndim=2,
                    dtype=np.complex128,
                ),
                dtype=np.complex128,
            )
            u_mat, s_vals, vh_mat = cls._svd_with_fallback(
                remainder,
                stage=f"dense_channel_to_tensors SVD site={site_index}",
            )
            keep = max(1, int(np.count_nonzero(np.asarray(s_vals) > cutoff)))
            u_keep = np.asarray(u_mat[:, :keep], dtype=np.complex128)
            s_keep = np.asarray(s_vals[:keep], dtype=np.float64)
            vh_keep = np.asarray(vh_mat[:keep, :], dtype=np.complex128)
            site_tensor = u_keep.reshape(chi_left, int(local_dim), int(local_dim), keep)
            tensors.append(np.transpose(site_tensor, (1, 2, 0, 3)).copy())
            remainder = np.diag(np.asarray(s_keep, dtype=np.complex128)) @ vh_keep
            remainder = np.asarray(
                cls._validate_numeric_array(
                    remainder,
                    stage=f"dense_channel_to_tensors propagated_remainder site={site_index}",
                    ndim=2,
                    dtype=np.complex128,
                ),
                dtype=np.complex128,
            )
            chi_left = keep

        last_tensor = remainder.reshape(chi_left, int(local_dim), int(local_dim), 1)
        tensors.append(np.transpose(last_tensor, (1, 2, 0, 3)).copy())
        return tensors

    @classmethod
    def from_dense_channel(
        cls,
        channel_dense: NDArray[np.complex128],
        *,
        n_sites: int,
        local_dim: int = 4,
        svd_cutoff: float = 1e-12,
    ) -> MPO:
        """Construct an MPO from a dense channel matrix.

        Returns:
            MPO whose dense matrix reconstructs ``channel_dense``.
        """
        tensors = cls.dense_channel_to_tensors(
            channel_dense,
            n_sites=n_sites,
            local_dim=local_dim,
            svd_cutoff=svd_cutoff,
        )
        mpo = cls()
        mpo.custom([np.asarray(tensor, dtype=np.complex128).copy() for tensor in tensors], transpose=False)
        return mpo

    @classmethod
    def dense_center_cut_schmidt_values(
        cls,
        channel_dense: NDArray[np.complex128],
        *,
        n_sites: int,
        cut: str | int = "center",
        local_dim: int = 4,
    ) -> NDArray[np.float64]:
        """Compute dense fused-site Schmidt values directly from a channel matrix.

        Returns:
            Singular values across the requested fused-site cut.
        """
        cut_index = cls._resolve_cut_index(cut=cut, length=int(n_sites))
        if cut_index in {0, int(n_sites)}:
            return np.array([1.0], dtype=np.float64)

        channel = cls._validated_dense_channel_matrix(
            channel_dense,
            n_sites=n_sites,
            local_dim=local_dim,
            stage="dense_center_cut_schmidt_values input",
        )

        tensor = channel.reshape([int(local_dim)] * (2 * int(n_sites)))
        interleaved_axes: list[int] = []
        for site_index in range(int(n_sites)):
            interleaved_axes.extend((site_index, site_index + int(n_sites)))
        fused_tensor = np.transpose(tensor, interleaved_axes)
        left_dim = (int(local_dim) * int(local_dim)) ** int(cut_index)
        right_dim = (int(local_dim) * int(local_dim)) ** (int(n_sites) - int(cut_index))
        schmidt_matrix = fused_tensor.reshape(left_dim, right_dim)
        svals = cls._svd_values_with_fallback(schmidt_matrix, stage="dense_center_cut_schmidt_values")
        return np.asarray(svals, dtype=np.float64)

    @classmethod
    def dense_channel_diagnostics(
        cls,
        channel_dense: NDArray[np.complex128],
        *,
        n_sites: int,
        cut: str | int = "center",
        local_dim: int = 4,
        svd_cutoff: float = 1e-12,
        rank_tol: float = 1e-12,
        target_dense: NDArray[np.complex128] | None = None,
        target_probabilities: NDArray[np.float64] | None = None,
        target_entropy: float | None = None,
        dense_cross_check: bool = False,
    ) -> dict[str, float | int | NDArray[np.float64] | None]:
        """Return canonical operator-entanglement diagnostics for a dense channel."""
        channel = cls._validated_dense_channel_matrix(
            channel_dense,
            n_sites=n_sites,
            local_dim=local_dim,
            stage="dense_channel_diagnostics input",
        )
        mpo = cls.from_dense_channel(
            channel,
            n_sites=n_sites,
            local_dim=local_dim,
            svd_cutoff=svd_cutoff,
        )
        reconstructed = np.asarray(mpo.to_matrix(), dtype=np.complex128)
        cls._validate_numeric_array(
            reconstructed, stage="dense_channel_diagnostics reconstructed", ndim=2, dtype=np.complex128
        )
        denom = max(1.0, float(np.linalg.norm(channel, ord="fro")))
        rel_error = float(np.linalg.norm(reconstructed - channel, ord="fro") / denom)

        schmidt_values = np.asarray(mpo.schmidt_values(cut=cut), dtype=np.float64)
        probabilities = cls.normalized_schmidt_probabilities(schmidt_values)
        entropy = float(cls.entropy_from_probabilities(probabilities))

        dense_entropy_diff = float("nan")
        dense_spec_diff = float("nan")
        if dense_cross_check:
            dense_svals = cls.dense_center_cut_schmidt_values(
                channel,
                n_sites=n_sites,
                cut=cut,
                local_dim=local_dim,
            )
            dense_probs = cls.normalized_schmidt_probabilities(dense_svals)
            dense_entropy = float(cls.entropy_from_probabilities(dense_probs))
            dense_entropy_diff = float(abs(entropy - dense_entropy))
            dense_spec_diff = float(cls.weighted_spectrum_distance(probabilities, dense_probs))

        entropy_error = None if target_entropy is None else float(abs(entropy - float(target_entropy)))
        weighted_spectrum_distance = None
        if target_probabilities is not None:
            cls._validate_numeric_array(
                np.asarray(target_probabilities, dtype=np.float64),
                stage="dense_channel_diagnostics target_probabilities",
                ndim=1,
                dtype=np.float64,
            )
            weighted_spectrum_distance = float(
                cls.weighted_spectrum_distance(
                    probabilities,
                    np.asarray(target_probabilities, dtype=np.float64),
                )
            )

        hs_to_target = None
        if target_dense is not None:
            reference = cls._validated_dense_channel_matrix(
                target_dense,
                n_sites=n_sites,
                local_dim=local_dim,
                stage="dense_channel_diagnostics target_dense",
            )
            hs_to_target = float(np.linalg.norm(channel - reference, ord="fro"))

        return {
            "entropy": float(entropy),
            "schmidt_values": schmidt_values,
            "probabilities": probabilities,
            "rank_tol": int(np.count_nonzero(probabilities > float(rank_tol))),
            "p1": float(probabilities[0]) if probabilities.size > 0 else 1.0,
            "largest_sv": float(schmidt_values[0]) if schmidt_values.size > 0 else 0.0,
            "mpo_rel_reconstruction_error": float(rel_error),
            "entropy_error": entropy_error,
            "weighted_spectrum_distance": weighted_spectrum_distance,
            "hs_to_target": hs_to_target,
            "dense_entropy_diff": float(dense_entropy_diff),
            "dense_spec_diff": float(dense_spec_diff),
        }

    def schmidt_values(self, cut: str | int = "center") -> NDArray[np.float64]:
        """Compute Schmidt singular values across a bond cut.

        Returns:
            Singular values at the selected cut.
        """
        return self._full_schmidt_values_for_cut(cut=cut)

    def operator_entanglement_entropy(self, cut: str | int = "center", base: float = math.e) -> float:
        """Compute operator entanglement entropy for an MPO cut.

        Returns:
            Entropy computed from normalized Schmidt weights.
        """
        schmidt_values = np.asarray(self.schmidt_values(cut=cut), dtype=np.float64)
        return self.entropy_from_schmidt_values(schmidt_values, base=base)

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
        identity_mpo = MPO.identity(self.length, physical_dimension=self.physical_dimension)

        identity_mps = identity_mpo.to_mps()
        mps = self.to_mps()
        trace = mps.scalar_product(identity_mps)

        hilbert_dim = self.physical_dimension**self.length
        # Checks if trace is not a singular values for partial trace
        return not np.round(np.abs(trace), 1) / hilbert_dim < fidelity

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
