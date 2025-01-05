import numpy as np
import opt_einsum as oe
from qiskit._accelerate.circuit import DAGCircuit
from qiskit.converters import dag_to_circuit

from yaqs.general.data_structures.MPO import MPO
from yaqs.circuits.equivalence_checking.dag_utils import apply_causal_cone, check_longest_gate, select_starting_point, convert_dag_to_tensor_algorithm


def decompose_theta(theta: np.ndarray, threshold: float):
    """
    Performs an SVD-based decomposition of the tensor `theta`, truncating
    singular values below a specified threshold, and then reshapes the result
    into two rank-4 tensors.

    Args:
        theta: The high-rank tensor to decompose.
        threshold: The cutoff threshold for singular values.

    Returns:
        A tuple (U, M) of two reshaped tensors derived from the SVD.
    """
    dims = theta.shape
    # Reorder indices before flattening
    theta = np.transpose(theta, (0, 3, 2, 1, 4, 5))
    theta_matrix = np.reshape(theta, (dims[0]*dims[1]*dims[2], dims[3]*dims[4]*dims[5]))

    U, S_list, V = np.linalg.svd(theta_matrix, full_matrices=False)
    S_list = S_list[S_list > threshold]
    U = U[:, :len(S_list)]
    V = V[:len(S_list), :]

    # Reshape U
    U = np.reshape(U, (dims[0], dims[1], dims[2], len(S_list)))

    # Create site tensor
    M = np.diag(S_list) @ V
    M = np.reshape(M, (len(S_list), dims[3], dims[4], dims[5]))
    M = np.transpose(M, (1, 2, 0, 3))

    return U, M


def update_MPO(mpo: MPO, dag1: DAGCircuit, dag2: DAGCircuit, qubits: list[int], threshold: float):
    """
    Applies the gates from `dag1` and `dag2` on the specified `qubits` in `mpo`,
    first with gates from `dag1`, then gates from `dag2`.

    Args:
        mpo: The MPO object whose tensors will be updated.
        dag1: A DAGCircuit containing gates to apply from G.
        dag2: A DAGCircuit containing gates to apply from G'.
        qubits: The list of qubit indices to apply the gates on (e.g. [n, n+1]).
        threshold: SVD threshold for truncation.
    """
    n = qubits[0]
    # Contract two neighboring MPO tensors
    theta = oe.contract('abcd, efdg->aecbfg', mpo.tensors[n], mpo.tensors[n + 1])

    # Apply G gates
    theta = apply_causal_cone(theta, dag1, qubits, conjugate=False)
    # Apply G' gates
    theta = apply_causal_cone(theta, dag2, qubits, conjugate=True)

    # Decompose back
    mpo.tensors[n], mpo.tensors[n + 1] = decompose_theta(theta, threshold)


def apply_layer(mpo: MPO, circuit1_dag, circuit2_dag, first_iterator, second_iterator, threshold: float):
    """
    Applies all gates for the current layer in two sweeps:
    one using `first_iterator` and another using `second_iterator`.

    Args:
        mpo: The MPO object to update.
        circuit1_dag: First circuit's DAGCircuit representation.
        circuit2_dag: Second circuit's DAGCircuit representation.
        first_iterator: Range of qubits to apply in the first sweep.
        second_iterator: Range of qubits to apply in the second sweep.
        threshold: SVD threshold for truncation.
    """
    for n in first_iterator:
        update_MPO(mpo, circuit1_dag, circuit2_dag, [n, n+1], threshold)

    for n in second_iterator:
        update_MPO(mpo, circuit1_dag, circuit2_dag, [n, n+1], threshold)


def apply_long_range_layer(mpo: 'MPO', dag1: 'DAGCircuit', dag2: 'DAGCircuit', conjugate: bool, threshold: float):
    """
    Detects and applies a 'long-range' gate in the first layer of `dag1` (or `dag2`).
    The logic here is partial/placeholder; you must fill in or adjust for your use case.

    Args:
        mpo: The MPO object being updated.
        dag1: First circuit's DAGCircuit.
        dag2: Second circuit's DAGCircuit.
        conjugate: Whether we apply the gate from `dag2` (if True) or from `dag1` (if False).
        threshold: SVD threshold for truncation.
    """
    # TODO: MPO on both sides
    # Identify gate and its position
    if not conjugate:
        dag_to_search = dag1
    else:
        dag_to_search = dag2

    for layer in dag_to_search.layers():
        first_layer = layer
        break

    if 'first_layer' in locals():
        layer_circuit = dag_to_circuit(first_layer['graph'])
        for gate in layer_circuit.data:
            if gate.operation.num_qubits > 1:
                distance = np.abs(gate.qubits[0]._index - gate.qubits[-1]._index)+1
                if distance > 2:
                    # Save location
                    location = min(gate.qubits[0]._index, gate.qubits[-1]._index)
                    # Create gate MPO
                    if not conjugate:
                        for node in dag1.op_nodes():
                            # Guarantees the correct node is removed
                            if node.name == gate.operation.name and len(node.qargs) >= 2 and node.qargs[0]._index == gate.qubits[0]._index and node.qargs[1]._index == gate.qubits[1]._index:
                                gate_MPO = convert_dag_to_tensor_algorithm(node)[0].mpo
                                # Remove from dag
                                dag1.remove_op_node(node)
                                break
                    else:
                        for node in dag2.op_nodes():
                            # Guarantees the correct node is removed
                            if node.name == gate.operation.name and len(node.qargs) >= 2 and node.qargs[0]._index == gate.qubits[0]._index and node.qargs[1]._index == gate.qubits[1]._index:
                                gate_MPO = convert_dag_to_tensor_algorithm(node)[0].mpo
                                gate_MPO.rotate(conjugate=True)
                                # Remove from dag
                                dag2.remove_op_node(node) 
                                break
                    break

    assert 'gate_MPO' in locals()

    assert gate_MPO.length <= mpo.length
    # MPO_2 must be the larger MPO
    if gate_MPO.length == mpo.length:
        sites = range(0, mpo.length)
    else:
        sites = range(location, location+distance)

    # TODO: Only contract first site with gate, then just SVD across chain to second gate tensor
    for site_gate_MPO, overall_site in enumerate(sites):
        # Even sites of gate MPO
        if site_gate_MPO != len(sites)-1 and site_gate_MPO % 2 == 0:
            # TODO: Could be sped up without putting all tensors together
            # TODO: Remove all transposes from new index order
            # sigma i, sigma i+1, upper bond gate, upper bond MPO, sigma' i, sigma' i+1, lower bond gate, lower bond MPO
            if not conjugate:
                tensor1 = np.transpose(gate_MPO.tensors[site_gate_MPO], (0, 2, 1, 3))
                tensor2 = np.transpose(gate_MPO.tensors[site_gate_MPO+1], (0, 2, 1, 3))
                tensor3 = np.transpose(mpo.tensors[overall_site], (0, 2, 1, 3))
                tensor4 = np.transpose(mpo.tensors[overall_site+1], (0, 2, 1, 3))
                theta = oe.contract('abcd,edfg,chij,fjkl->aebhikgl', tensor1, tensor2, tensor3, tensor4)
            else:
                mpo.rotate()
                tensor1 = np.transpose(gate_MPO.tensors[site_gate_MPO], (0, 2, 1, 3))
                tensor2 = np.transpose(gate_MPO.tensors[site_gate_MPO+1], (0, 2, 1, 3))
                tensor3 = np.transpose(mpo.tensors[overall_site], (0, 2, 1, 3))
                tensor4 = np.transpose(mpo.tensors[overall_site+1], (0, 2, 1, 3))
                theta = oe.contract('abcd,edfg,chij,fjkl->ikhbaelg', tensor1, tensor2, tensor3, tensor4)
                mpo.rotate()

            # Apply causal cone on each side
            theta = apply_causal_cone(theta, dag1, [overall_site, overall_site+1], conjugate=False)
            theta = apply_causal_cone(theta, dag2, [overall_site, overall_site+1], conjugate=True)

            dims = theta.shape
            theta = np.reshape(theta, (dims[0], dims[1], dims[2]*dims[3], dims[4], dims[5], dims[6]*dims[7]))
            mpo.tensors[overall_site], mpo.tensors[overall_site+1] = decompose_theta(theta, threshold)

            # Used to track tensors already applied
            gate_MPO.tensors[site_gate_MPO] = None
            gate_MPO.tensors[site_gate_MPO+1] = None

        # Odd length gate MPO, single hanging tensor
        if site_gate_MPO == len(sites)-1 and any(type(tensor) == np.ndarray for tensor in gate_MPO.tensors):
            if not conjugate:
                tensor1 = np.transpose(gate_MPO.tensors[site_gate_MPO], (0, 2, 1, 3))
                tensor2 = np.transpose(mpo.tensors[overall_site], (0, 2, 1, 3))
                theta = oe.contract('abcd,cefg->abefdg', tensor1, tensor2)
            else:
                mpo.rotate()
                tensor1 = np.transpose(gate_MPO.tensors[site_gate_MPO], (0, 2, 1, 3))
                tensor2 = np.transpose(mpo.tensors[overall_site], (0, 2, 1, 3))
                theta = oe.contract('abcd,cefg->febagd', tensor1, tensor2)
                mpo.rotate()

            dims = theta.shape
            theta = np.reshape(theta, (dims[0], dims[1]*dims[2], dims[3], dims[4]*dims[5]))

            tensor1 = np.transpose(mpo.tensors[overall_site-1], (0, 2, 1, 3))
            theta = oe.contract('abcd, edfg->aebcfg', tensor1, theta)
            
            theta = apply_causal_cone(theta, dag1, [overall_site-1, overall_site], conjugate=False)
            theta = apply_causal_cone(theta, dag2, [overall_site-1, overall_site], conjugate=True)

            mpo.tensors[overall_site-1], mpo.tensors[overall_site] = decompose_theta(theta, threshold)
            gate_MPO.tensors[site_gate_MPO] = None

    assert not any(type(tensor) == np.ndarray for tensor in gate_MPO.tensors)


def iterate(mpo: MPO, dag1, dag2, threshold: float):
    """
    Iteratively applies gates from `dag1` and `dag2` layer by layer
    until no gates remain in either DAGCircuit.

    Args:
        mpo: The MPO object to be updated.
        dag1: First circuit's DAGCircuit.
        dag2: Second circuit's DAGCircuit.
        threshold: SVD threshold for truncation.
    """
    N = mpo.length

    if dag1.op_nodes():
        first_iterator, second_iterator = select_starting_point(N, dag1)
    else:
        first_iterator, second_iterator = select_starting_point(N, dag2)

    while dag1.op_nodes() or dag2.op_nodes():
        largest_distance1 = check_longest_gate(dag1)
        largest_distance2 = check_longest_gate(dag2)
        # If all gates are nearest-neighbor (distance <= 2), apply standard layer
        if largest_distance1 in [1, 2] and largest_distance2 in [1, 2]:
            apply_layer(mpo, dag1, dag2, first_iterator, second_iterator, threshold)
        else:
            # Handle a gate that is longer than 2 sites
            conjugate = (largest_distance2 > largest_distance1)
            apply_long_range_layer(mpo, dag1, dag2, conjugate, threshold)
