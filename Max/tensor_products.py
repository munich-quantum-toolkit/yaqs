import numpy as np
import scipy.linalg
import scipy.sparse


def multi_site_tensor_product(operator, L):
    """ Tensor products some single site operator L times
        Ex. Input A --> A x A x A (x: Kronecker product)

    Args:
        operator: numpy array
            Single site operator
        L: int
            System size

    Returns:
        total_operator: numpy array
            Operator acting on full system
    """
    if L == 1:
        return operator
    for i in range(L-1):
        if i == 0:
            total_operator = scipy.sparse.kron(operator, operator)
        else:
            total_operator = scipy.sparse.kron(total_operator, operator)

    return total_operator


def sum_local_operators(operator, L):
    """ Outputs an operator acting on the full system
        as a summation of local terms
        Ex. A --> A x 1 x 1 + 1 x A x 1 + 1 x 1 x A

    Args:
        operator: numpy array
            Single site operator
        L: int
            System size

    Returns:
        total_operator: numpy array
            Operator acting on full system as a matrix
    """
    op_list = create_local_operators_list(operator, L)
    total_operator = np.zeros((operator.shape[0]**L, operator.shape[0]**L))

    for i in range(L):
        total_operator += op_list[i]

    total_operator = scipy.sparse.csr_matrix(total_operator)
    return total_operator


def create_local_operators_list(operator, L, phys_dims_list=None):
    """ Outputs the operators acting on the full system
        Ex. A --> [A x 1 x 1, 1 x A x 1, 1 x 1 x A]

    Args:
        operator: numpy array
            Single site operator
        L: int
            System size

    Returns:
        operators: list of numpy arrays
            List of larger operators acting on total system
    """
    operators = []
    if phys_dims_list:
        for site in range(L):
            op = []
            identity = scipy.sparse.identity(phys_dims_list[site])
            if site == 0:
                # Start with the H_s
                site_matrix = operator
                op.append("Op")
            elif site == L-1:
                site_matrix = identity
                op.append("1")
            else:
                site_matrix = identity
                op.append("1")

            # L-1 since we have already added a single site
            for i in range(L-1):
                identity = scipy.sparse.identity(phys_dims_list[i+1]) # np.eye(operator.shape[0])
                if site == 0:
                    # H_s already added, continue doing identity for other sites
                    site_matrix = scipy.sparse.kron(site_matrix, identity)
                    op.append("1")
                elif site == L-1:
                    # Final site will get H_s, others get identity
                    if i != L-2:
                        site_matrix = scipy.sparse.kron(site_matrix, identity)
                        op.append("1")
                    else:
                        site_matrix = scipy.sparse.kron(site_matrix, operator)
                        op.append("Op")
                else:
                    # Specific site gets H_s, others get identity
                    if i != site-1:
                        site_matrix = scipy.sparse.kron(site_matrix, identity)
                        op.append("1")
                    else:
                        site_matrix = scipy.sparse.kron(site_matrix, operator)
                        op.append("Op")

            # For Debugging
            # print(op)
            # print(site_matrix.shape)
            operators.append(site_matrix)
    else:
        identity = scipy.sparse.identity(operator.shape[0]) # np.eye(operator.shape[0])
        for site in range(L):
            op = []
            if site == 0:
                # Start with the H_s
                site_matrix = operator
                op.append("Op")
            elif site == L-1:
                site_matrix = identity
                op.append("1")
            else:
                site_matrix = identity
                op.append("1")

            # L-1 since we have already added a single site
            for i in range(L-1):
                if site == 0:
                    # H_s already added, continue doing identity for other sites
                    site_matrix = scipy.sparse.kron(site_matrix, identity)
                    op.append("1")
                elif site == L-1:
                    # Final site will get H_s, others get identity
                    if i != L-2:
                        site_matrix = scipy.sparse.kron(site_matrix, identity)
                        op.append("1")
                    else:
                        site_matrix = scipy.sparse.kron(site_matrix, operator)
                        op.append("Op")
                else:
                    # Specific site gets H_s, others get identity
                    if i != site-1:
                        site_matrix = scipy.sparse.kron(site_matrix, identity)
                        op.append("1")
                    else:
                        site_matrix = scipy.sparse.kron(site_matrix, operator)
                        op.append("Op")

            # For Debugging
            # print(op)
            operators.append(site_matrix)
    return operators

