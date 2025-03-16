import numpy as np
import qutip as qt

def site_index(x, y, spin, Lx):
        """ Convert 2D (x, y) coordinates to a 1D index. """
        if spin == '↑':
            spin = 0
        elif spin == '↓':
            spin = 1
        if spin not in (0, 1):
            raise ValueError("spin must be 0 or 1")
        return 2 * (x + Lx * y) + spin


def create_Fermi_Hubbard_model_qutip(Lx, Ly, u, t, mu):
    """
    Create a Fermi-Hubbard model in QuTiP that can be used for reference.
    """
    L = Lx * Ly
    
    H_onsite = 0
    for x in range(Lx):
        for y in range(Ly):
            i_up = site_index(x, y, '↑', Lx)
            i_down = site_index(x, y, '↓', Lx)
            n_up = qt.fcreate(n_sites=2*L, site=i_up) * qt.fdestroy(n_sites=2*L, site=i_up)
            n_down = qt.fcreate(n_sites=2*L, site=i_down) * qt.fdestroy(n_sites=2*L, site=i_down)
            H_onsite += u * n_up * n_down

    H_hop = 0
    for x in range(Lx):
        for y in range(Ly):
            # Right neighbor (x+1, y)
            if x < Lx - 1:
                for spin in ['↑', '↓']:
                    i = site_index(x, y, spin, Lx)
                    j = site_index(x+1, y, spin, Lx)
                    #H_hop += -t * qt.fcreate(n_sites=2*L, site=i) * qt.fdestroy(n_sites=2*L, site=j) + qt.fcreate(n_sites=2*L, site=j) * qt.fdestroy(n_sites=2*L, site=i)
                    H_hop += -t * qt.fdestroy(n_sites=2*L, site=i) * qt.fcreate(n_sites=2*L, site=j) + qt.fdestroy(n_sites=2*L, site=j) * qt.fcreate(n_sites=2*L, site=i)
            # Down neighbor (x, y+1)
            if y < Ly -1:
                for spin in ['↑', '↓']:
                    i = site_index(x, y, spin, Lx)
                    j = site_index(x, y+1, spin, Lx)
                    #H_hop += -t * qt.fcreate(n_sites=2*L, site=i) * qt.fdestroy(n_sites=2*L, site=j) + qt.fcreate(n_sites=2*L, site=j) * qt.fdestroy(n_sites=2*L, site=i)
                    H_hop += -t * qt.fdestroy(n_sites=2*L, site=i) * qt.fcreate(n_sites=2*L, site=j) + qt.fdestroy(n_sites=2*L, site=j) * qt.fcreate(n_sites=2*L, site=i)

    H_chem = 0
    for x in range(Lx):
        for y in range(Ly):
            H_chem += -mu * qt.fcreate(n_sites=2*L, site=site_index(x, y, '↑', Lx)) * qt.fdestroy(n_sites=2*L, site=site_index(x, y, '↑', Lx))
            H_chem += -mu * qt.fcreate(n_sites=2*L, site=site_index(x, y, '↓', Lx)) * qt.fdestroy(n_sites=2*L, site=site_index(x, y, '↓', Lx))
    
    H = H_onsite + H_hop + H_chem
    H = qt.Qobj(H)
    return H


def create_alternating_init_state_qutip(L):
    """
    Create an initial state with alternating occupation
    """
    state_list = sum(([qt.basis(2, 0), qt.basis(2, x % 2)] for x in range(L)), [])
    initial_state = qt.tensor(state_list).unit()
    return initial_state