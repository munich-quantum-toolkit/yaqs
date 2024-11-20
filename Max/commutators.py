def commutator(A, B):
    """ Commutator of two matrices [A, B] = AB - BA

    Args:
        A: numpy array
            Matrix 1
        B: numpy array
            Matrix 2

    Returns:
        AB - BA: numpy array
             Commutator
    """
    return A @ B - B @ A


def anticommutator(A, B):
    """ Anticommutator of two matrices {A, B} = AB + BA

    Args:
        A: numpy array
            Matrix 1
        B: numpy array
            Matrix 2

    Returns:
        AB + BA: numpy array
             Anticommutator
    """
    return A @ B + B @ A