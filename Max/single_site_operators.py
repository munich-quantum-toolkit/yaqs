from collections import Counter
import numpy as np


def create_pauli_x(d):
    """ Generalizes the x-gate operation to a d-level system

    Args:
        d: physical dimension of system
    Returns:
        sigma_plus: (d x d) excitation operator
    """
    x = np.zeros((d, d))
    for row, _ in enumerate(x):
        x[row][row-1] = 1

    return x


def create_pauli_y(d):
    """ Generalizes the y-gate operation to a d-level system

    Args:
        d: physical dimension of system
    Returns:
        sigma_plus: (d x d) excitation operator
    """
    y = np.array([[0, -1j], [1j, 0]])

    return y

def create_deexcitation_operator(d):
    """ Creates an excitation operator for a d-level system

    Args:
        d: int
            Physical dimension of system
    Returns:
        sigma_plus: numpy array
            (d x d) excitation operator
    """
    sigma_plus = np.zeros((d, d))
    for row, array in enumerate(sigma_plus):
        for col, _ in enumerate(array):
            if col - row == 1:
                sigma_plus[row][col] = 1

    return sigma_plus


def create_excitation_operator(d):
    """ Creates a de-excitation operator for a d-level system

    Args:
        d: int
            Physical dimension of system
    Returns:
        sigma_minus: numpy array
            (d x d) de-excitation operator
    """
    sigma_minus = np.zeros((d, d))
    for row, array in enumerate(sigma_minus):
        for col, _ in enumerate(array):
            if row - col == 1:
                sigma_minus[row][col] = 1

    return sigma_minus


# TODO: Generalize to d-levels
def create_pauli_z(d):
    """ Creates a sigma-z operator for a d-level system

    Args:
        d: int
            Physical dimension of system
    Returns:
        sigma_z: numpy array
            (d x d) sigma_z operator
    """
    max_number = d-1
    diag = []
    for elem in range(d-1, -d, -2):
        if d % 2  != 0:
            elem = elem // 2
            if elem > 0:
                while elem % 2 == 0:
                    elem += 1
            elif elem < 0:
                while elem % 2 == 0:
                    elem -= 1

        diag.append(elem)
    count = Counter(diag)
    for key in count.keys():
        if count[key] > 1 and int(key) > 0:
            diag[diag.index(int(key))] += 2
        elif count[key] > 1 and int(key) < 0:
            diag.reverse()
            diag[diag.index(int(key))] -= 2
            diag.reverse()

    pauli_z = np.diag(diag)
    return pauli_z
