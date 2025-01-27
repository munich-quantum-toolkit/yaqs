import concurrent.futures
import copy
import multiprocessing
import numpy as np
import opt_einsum as oe
from tqdm import tqdm


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from yaqs.general.data_structures.networks import MPS


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
            # tensor = oe.contract('ijk, abk->iajb', A[site], B[site])
            if site == 0:
                result = tensor
            else:
                result = oe.contract('abcd, cdef->abef', result, tensor)
                # result = oe.contract('ijkl, klcd->ijcd', result, tensor)
        result = np.squeeze(result)

    # Used for ignoring other tensors if MPS is in canonical form
    else:
        # Single site operators
        # print(site, A._check_canonical_form())
        # assert A._check_canonical_form() == site or A._check_canonical_form == 0 or A._check_canonical_form == A.length-1
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

    # state.set_canonical_form(site)
    temp_state = copy.deepcopy(state)
    # temp_state.set_canonical_form(site)
    temp_state.tensors[site] = oe.contract('ab, bcd->acd', operator, temp_state.tensors[site])
    E = scalar_product(temp_state, state, site)

    return E.real


def measure_single_shot(state):
    """
    Performs a single-shot measurement of the MPS state.
    Args:
        state ('MPS'): The MPS state to measure.
    Returns:
        int: The observed basis state as an integer.
    """
    temp_state = copy.deepcopy(state)
    bitstring = []
    for site, tensor in enumerate(temp_state.tensors):
        reduced_density_matrix = oe.contract('abc, dbc->ad', tensor, np.conj(tensor))
        probabilities = np.diag(reduced_density_matrix).real
        chosen_index = np.random.choice(len(probabilities), p=probabilities)
        bitstring.append(chosen_index)
        selected_state = np.zeros(len(probabilities))
        selected_state[chosen_index] = 1
        # Multiply state
        tensor = oe.contract('a, acd->cd', selected_state, tensor)
        # Multiply site into next site
        if site != state.length - 1:
            temp_state.tensors[site + 1] = 1 / np.sqrt(probabilities[chosen_index]) * oe.contract(
                'ab, cbd->cad', tensor, temp_state.tensors[site + 1])
    return sum(c << i for i, c in enumerate(bitstring))


def measure(state: 'MPS', shots: int):
    """
    Measures an MPS state for a given number of shots.
    
    Args:
        state ('MPS'): The MPS state to measure.
        shots (int): Number of measurements (shots) to perform.
        
    Returns:
        dict: A dictionary mapping basis states to their observed counts.
    """

    if shots > 1:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=shots, desc="Measuring shots", ncols=80) as pbar:
                results = {}
                futures = [executor.submit(measure_single_shot, copy.deepcopy(state)) for _ in range(shots)]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result()
                        results[result] = results.get(result, 0) + 1
                    except Exception as e:
                        print(f"Shot measurement failed with exception: {e}.")
                    finally:
                        pbar.update(1)
        return results
    else:
        results = {}
        basis_state = measure_single_shot(state)
        results[basis_state] = results.get(basis_state, 0) + 1
        return results


# def fidelity(A, B):
#     """ Calculates the fidelity between matrix A and B

#     Args:
#         A: numpy array
#         B: numpy array

#     Returns:
#         fidelity: float
#     """
#     epsilon = 1e-12
#     eig_A, right_A = np.linalg.eigh(A)
#     eig_A[eig_A < epsilon] = 0
#     eig_A = np.real(np.diag(eig_A))

#     eig_B, right_B = np.linalg.eigh(B)
#     eig_B[eig_B < epsilon] = 0
#     eig_B = np.real(np.diag(eig_B))

#     interior = right_A @ np.sqrt(eig_A) @ np.conj(right_A.T) @ right_B @ np.sqrt(eig_B) @ np.conj(right_B.T)
#     fidelity = np.trace(np.abs(interior))**2

#     ### Original Method
#     # start = time.time()
#     # sqrtA = scipy.linalg.sqrtm(A)
#     # interior = sqrtA @ B @ sqrtA
#     # interior = scipy.linalg.sqrtm(interior)
#     # fidelity_2 = np.trace(interior)**2
#     # end = time.time()
#     # print("Time (No eigs): ", end-start)

#     return fidelity


# def sanity_checks(psi_length, d, psi, MPS):
#     # Check if decompose-able (psi_length = d^num_sites)
#     if not float(np.log(psi_length)/np.log(d)).is_integer():
#         print('Psi is NOT decompose-able into qudits for given d')
#     else:
#         print('Psi is decomposable into an MPS with the given d and psi_length')

#     # Check if normalized
#     epsilon = 1e-12
#     if np.linalg.norm(psi) < 1-epsilon or np.linalg.norm(psi) > 1+epsilon:
#         print('Psi is NOT normalized')
#     else:
#         print('Psi is normalized')

#     E = scalar_product(MPS, MPS)
#     if E[0] < 1-epsilon or E[0] > 1+epsilon:
#         print('The MPS is not normalized')
#     else:
#         print('The MPS is normalized')