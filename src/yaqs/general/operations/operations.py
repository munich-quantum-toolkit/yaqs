import copy
import numpy as np
import opt_einsum as oe

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from yaqs.general.data_structures.MPS import MPS


def scalar_product(A: 'MPS', B: 'MPS', site: int=-1):
    """ Calculates the scalar product of two Matrix Product States
        by contracting all positions vertically then horizontally.

    Args:
        A: first MPS
        B: second MPS

    Returns:
        result: Frobenius norm of A and B <A|B>
    """
    A_copy = copy.deepcopy(A)
    B_copy = copy.deepcopy(B)
    for i, tensor in enumerate(A_copy.tensors):
        A_copy.tensors[i] = np.conj(tensor)

    # Contract all sites
    if site == -1:
        for site in range(A.length):
            tensor = oe.contract('abc, ade->bdce', A_copy.tensors[site], B_copy.tensors[site])
            if site == 0:
                result = tensor
            else:
                result = oe.contract('abcd, cdef->abef', result, tensor)
        result = np.squeeze(result)

    # Used for ignoring other tensors if MPS is in canonical form
    else:
        # Single site operators
        result = oe.contract('ijk, ijk', A_copy.tensors[site], B_copy.tensors[site])

    return result


def local_expval(state: 'MPS', operator: np.ndarray, site: int):
    """ Expectation value for a given MPS-MPO-MPS network

    Args:
        A: MPS
        B: MPO
        C: MPS

    Returns:
        E.real: real portion of calculated expectation value
    """
    # TODO: Could be more memory-efficient by not copying state

    # This loop assumes a B form MPS
    for i in range(site):
        state.shift_orthogonality_center_right(i)

    temp_state = copy.deepcopy(state)
    temp_state.tensors[site] = oe.contract('ab, bcd->acd', operator, temp_state.tensors[site])
    E = scalar_product(state, temp_state, site)

    # Expectation values must be real

    assert E.imag < 1e-13, "Expectation Value should be real, '%16f+%16fi'." % (E.real, E.imag)

    return E.real
