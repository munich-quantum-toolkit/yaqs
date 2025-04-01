# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Reference implementation of the Fermi-Hubbard model.

This module provides functions for creating the Fermi-Hubbard model in QuTiP
which can be used as a reference to compare the circuit approximations.
"""

# ignore non-lowercase argument names for physics notation
# ruff: noqa: N803 N806

from __future__ import annotations

import functools
import operator

import qutip as qt


def site_index(x: int, y: int, spin: str, Lx: int) -> int:
    """Convert 2D (x, y) coordinates to a 1D index.

    Args:
        x (int): Horizontal index of the site in the lattice grid.
        y (int): Vertical index of the site in the lattice grid.
        spin (str): Spin of the particle at the given site (can be '↑' or '↓').
        Lx (int): Horizontal size of the lattice grid.

    Returns:
        int: Index of the given site on a 1D lattice.
    """
    if spin == "↑":
        spin_val = 0
    elif spin == "↓":
        spin_val = 1
    return 2 * (x + Lx * y) + spin_val


def create_fermi_hubbard_model_qutip(Lx: int, Ly: int, u: float, t: float, mu: float) -> qt.Qobj:
    """Create a Fermi-Hubbard model in QuTiP that can be used for reference.

    Args:
        Lx (int): Number of columns in the grid lattice.
        Ly (int): Number of rows in the grid lattice.
        u (float): On-site interaction parameter.
        t (float): Transfer energy parameter.
        mu (float): Chemical potential parameter.

    Returns:
        qutip.Qobj: A quantum circuit representing the Fermi-Hubbard model on a rectangular grid.
    """
    num_sites = Lx * Ly

    H_onsite = 0
    for x in range(Lx):
        for y in range(Ly):
            i_up = site_index(x, y, "↑", Lx)
            i_down = site_index(x, y, "↓", Lx)
            n_up = qt.fcreate(n_sites=2 * num_sites, site=i_up) * qt.fdestroy(n_sites=2 * num_sites, site=i_up)
            n_down = qt.fcreate(n_sites=2 * num_sites, site=i_down) * qt.fdestroy(n_sites=2 * num_sites, site=i_down)
            H_onsite += u * n_up * n_down

    H_hop = 0
    for x in range(Lx):
        for y in range(Ly):
            # Right neighbor (x+1, y)
            if x < Lx - 1:
                for spin in ["↑", "↓"]:
                    i = site_index(x, y, spin, Lx)
                    j = site_index(x + 1, y, spin, Lx)
                    create_i = qt.fcreate(n_sites=2 * num_sites, site=i)
                    create_j = qt.fcreate(n_sites=2 * num_sites, site=j)
                    destroy_i = qt.fdestroy(n_sites=2 * num_sites, site=i)
                    destroy_j = qt.fdestroy(n_sites=2 * num_sites, site=j)
                    H_hop += -t * destroy_i * create_j + destroy_j * create_i
            # Down neighbor (x, y+1)
            if y < Ly - 1:
                for spin in ["↑", "↓"]:
                    i = site_index(x, y, spin, Lx)
                    j = site_index(x, y + 1, spin, Lx)
                    create_i = qt.fcreate(n_sites=2 * num_sites, site=i)
                    create_j = qt.fcreate(n_sites=2 * num_sites, site=j)
                    destroy_i = qt.fdestroy(n_sites=2 * num_sites, site=i)
                    destroy_j = qt.fdestroy(n_sites=2 * num_sites, site=j)
                    H_hop += -t * destroy_i * create_j + destroy_j * create_i

    H_chem = 0
    for x in range(Lx):
        for y in range(Ly):
            create_up = qt.fcreate(n_sites=2 * num_sites, site=site_index(x, y, "↑", Lx))
            destroy_up = qt.fdestroy(n_sites=2 * num_sites, site=site_index(x, y, "↑", Lx))
            create_down = qt.fcreate(n_sites=2 * num_sites, site=site_index(x, y, "↓", Lx))
            destroy_down = qt.fdestroy(n_sites=2 * num_sites, site=site_index(x, y, "↓", Lx))
            H_chem += -mu * create_up * destroy_up
            H_chem += -mu * create_down * destroy_down

    hamiltonian = H_onsite + H_hop + H_chem
    return qt.Qobj(hamiltonian)


def create_alternating_init_state_qutip(num_sites: int) -> qt.Qobj:
    """Create an initial state with alternating occupation.

    Create a quantum state in QuTiP where the occupation alternates between
    |00⟩ and |11⟩ for even and odd indexed sites, respectively.

    Args:
        num_sites (int): Number of sites in the model and length of the state.

    Returns:
        qutip.Qobj: A quantum state represented as a tensor product of individual site states.
    """
    state_list = functools.reduce(operator.iadd, ([qt.basis(2, 0), qt.basis(2, x % 2)] for x in range(num_sites)), [])
    return qt.tensor(state_list).unit()
