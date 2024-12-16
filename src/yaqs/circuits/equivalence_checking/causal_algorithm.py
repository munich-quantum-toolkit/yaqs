import copy
import numpy as np
import opt_einsum as oe
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.dagcircuit.dagnode import DAGOpNode
from qiskit.dagcircuit import dagcircuit
from qiskit.circuit import QuantumRegister
import qiskit

from src.initialization import initialize_identity_MPO
from src.operations import check_if_identity, flip_network
from yaqs.general.libraries.tensor_library import TensorLibrary

########### TODO: Delete
def convert_MPS_MPO(network):
    new_network = []

    if network[0].ndim == 4:
        for tensor in network:
            # Left phys, left bond, right phys, right bond (CW)
            new_tensor = np.transpose(tensor, (1, 0, 2, 3))
            # Left bond, left phys, right phys, right bond
            new_tensor = np.reshape(new_tensor, (new_tensor.shape[0], new_tensor.shape[1]*new_tensor.shape[2], new_tensor.shape[3]))

            new_network.append(new_tensor)
    elif network[0].ndim == 3:
        for tensor in network:
            # Split physical dimension
            # Left bond, left phys, right phys, right bond
            # TODO: Not generalized for d-level
            new_tensor = np.reshape(tensor, (tensor.shape[0], 2, 2, tensor.shape[2]))

            # Left phys, left bond, right phys, right bond
            new_tensor = np.transpose(new_tensor, (1, 0, 2, 3))

            new_network.append(new_tensor)

    return new_network


########### TODO: Delete
def SVD_compression_sweep(MPO, max_bond_dimension=None, threshold=None):
    assert max_bond_dimension != None or threshold != None

    MPS = convert_MPS_MPO(MPO)

    for site, tensor in enumerate(MPS):
        if site == len(MPS)-1:
            break

        dims = tensor.shape

        # Combine left bond and phys dim
        tensor_matrix = np.reshape(tensor, (dims[0]*dims[1], dims[2]))
        U, S_list, V = np.linalg.svd(tensor_matrix, full_matrices=False)
        if threshold:
            S_list = S_list[S_list > threshold]
        if max_bond_dimension:
            S_list = S_list[0:max_bond_dimension]

        U = U[:, 0:len(S_list)]
        V = V[0:len(S_list), :]

        # Create site tensors
        U = np.reshape(U, (dims[0], dims[1], len(S_list)))

        M = np.diag(S_list) @ V

        # M = np.reshape(M, (len(S_list), dims[0], dims[2], dims[3]))
        # # chi, s, t, k -> k, chi, s, t
        # M = np.transpose(M, (3, 0, 1, 2))
        MPS[site] = U
        MPS[site+1] = oe.contract('ij, jbc->ibc', M, MPS[site+1])

    MPO = convert_MPS_MPO(MPS)
    return MPO


def get_gate_count(circuit):
    ops = circuit.count_ops()
    total_gates = 0
    for key, value in ops.items():
        if key != 'measure' and key != 'barrier':
            total_gates += value

    return total_gates


def decompose_theta(theta, threshold):
    dims = theta.shape
    theta = np.transpose(theta, (0, 3, 2, 1, 4, 5))
    theta_matrix = np.reshape(theta, (dims[0]*dims[1]*dims[2], dims[3]*dims[4]*dims[5]))

    # SVD with truncation
    U, S_list, V = np.linalg.svd(theta_matrix, full_matrices=False)
    S_list = S_list[S_list > threshold]
    U = U[:, 0:len(S_list)]
    V = V[0:len(S_list), :]

    # Create site tensors
    U = np.reshape(U, (dims[0], dims[1], dims[2], len(S_list)))
    U = np.transpose(U, (0, 2, 1, 3))
    M = np.diag(S_list) @ V
    M = np.reshape(M, (len(S_list), dims[3], dims[4], dims[5]))
    M = np.transpose(M, (1, 0, 2, 3))

    return U, M


def apply_gate(gate, theta, site0, site1, conjugate=False):
    assert (gate.interaction == 1 or gate.interaction == 2)

    if gate.interaction == 1:
        assert gate.sites[0] in [site0, site1]
    elif gate.interaction == 2:
        assert gate.sites[0] in [site0, site1] and gate.sites[1] in [site0, site1]

    # Application from nearest-neighbor gates
    if theta.ndim == 6:
        if conjugate:
            theta = np.transpose(theta, (3, 4, 2, 0, 1, 5))

        if gate.name == "I":
            theta = theta
        elif gate.interaction == 1:
            if gate.sites[0] == site0:
                if conjugate:
                    theta = oe.contract('ij, jklmno->iklmno', np.conj(gate.tensor), theta)
                else:
                    theta = oe.contract('ij, jklmno->iklmno', gate.tensor, theta)
            elif gate.sites[0] == site1:
                if conjugate:
                    theta = oe.contract('ij, kjlmno->kilmno', np.conj(gate.tensor), theta)
                else:
                    theta = oe.contract('ij, kjlmno->kilmno', gate.tensor, theta)
        elif gate.interaction == 2:
            if conjugate:
                theta = oe.contract('ijkl, klmnop->ijmnop', np.conj(gate.tensor), theta)
            else:
                theta = oe.contract('ijkl, klmnop->ijmnop', gate.tensor, theta)
        
        if conjugate:
            theta = np.transpose(theta, (3, 4, 2, 0, 1, 5))

    # Application from long-range gates
    elif theta.ndim == 8:
        if conjugate:
            theta = np.transpose(theta, (4, 5, 3, 2, 0, 1, 6, 7))
        if gate.name == "I":
            theta = theta
        elif gate.interaction == 1:
            if gate.sites[0] == site0:
                if conjugate:
                    theta = oe.contract('ab, bcdefghi->acdefghi', np.conj(gate.tensor), theta)
                else:
                    theta = oe.contract('ab, bcdefghi->acdefghi', gate.tensor, theta)
            elif gate.sites[0] == site1:
                if conjugate:
                    theta = oe.contract('ab, cbdefghi->cadefghi', np.conj(gate.tensor), theta)
                else:
                    theta = oe.contract('ab, cbdefghi->cadefghi', gate.tensor, theta)

        elif gate.interaction == 2:
            if conjugate:
                theta = oe.contract('abcd, cdefghij->abefghij', np.conj(gate.tensor), theta)
            else:
                theta = oe.contract('abcd, cdefghij->abefghij', gate.tensor, theta)

        if conjugate:
            theta = np.transpose(theta, (4, 5, 3, 2, 0, 1, 6, 7))

    return theta


def convert_dag_to_tensor_algorithm(dag):
    algorithm = []

    # Check if single node
    if type(dag) == DAGOpNode:
        gate = dag
        name = gate.op.name

        attr = getattr(TensorLibrary, name)
        gate_object = attr()
        if gate.op.params:
            gate_object.set_params(gate.op.params)

        sites = [gate.qargs[0]._index]
        if len(gate.qargs) == 2:
            sites.append(gate.qargs[1]._index)
        if len(gate.qargs) == 3:
            sites.append(gate.qargs[1]._index)
            sites.append(gate.qargs[2]._index)

        gate_object.set_sites(*sites)

        algorithm.append(gate_object)
    else:
        for gate in dag.op_nodes():
            name = gate.op.name

            if name == 'measure' or name == 'barrier':
                continue
            attr = getattr(TensorLibrary, name)
            gate_object = attr()
            if gate.op.params:
                gate_object.set_params(gate.op.params)

            # TODO: Cleanup ifs
            sites = [gate.qargs[0]._index]
            if len(gate.qargs) == 2:
                sites.append(gate.qargs[1]._index)
            if len(gate.qargs) == 3:
                sites.append(gate.qargs[1]._index)
                sites.append(gate.qargs[2]._index)

            gate_object.set_sites(*sites)

            algorithm.append(gate_object)

    return algorithm


def get_causal_cone(dag, qubits):
    new_dag = dag.copy_empty_like()
    layers = list(dag.multigraph_layers())
    qubits_to_check = set()
    for qubit in range(min(qubits), max(qubits)+1):
        qubits_to_check.add(dag.qubits[qubit])

    for layer in layers:
        for node in layer:
            if isinstance(node, DAGOpNode):
                qubit_set = set(node.qargs)

                # Gate is entirely within cone
                if qubit_set <= qubits_to_check:
                    if node.op.name == 'measure' or node.op.name == 'barrier':
                        dag.remove_op_node(node)
                        continue
                    new_dag.apply_operation_back(node.op, node.qargs)
                    dag.remove_op_node(node)
                else:
                    if node.op.name == 'measure' or node.op.name == 'barrier':
                        dag.remove_op_node(node)
                        continue
                    # If not a subset, but there is overlap
                    for item in qubit_set & qubits_to_check:
                        qubits_to_check.remove(item)

        # Cone has ended for all
        if len(qubits_to_check) == 0:
            break
    # print(dag_to_circuit(new_dag))
    return new_dag


def apply_causal_cone(theta, dag, qubits, conjugate=False):
    n = qubits[0]
    if dag.op_nodes():
        causal_circuit = get_causal_cone(dag, [n, n+1])
        tensor_circuit = convert_dag_to_tensor_algorithm(causal_circuit)
        for gate in tensor_circuit:
            theta = apply_gate(gate, theta, n, n+1, conjugate)
    return theta


def update_MPO(MPO, dag1, dag2, qubits, threshold):
    n = qubits[0]

    # print("UPDATE", n, n+1)
    theta = oe.contract('ijkl, mlno->imjkno', MPO[n], MPO[n+1])
    theta = apply_causal_cone(theta, dag1, qubits)
    theta = apply_causal_cone(theta, dag2, qubits, conjugate=True)
    MPO[n], MPO[n+1] = decompose_theta(theta, threshold)
    return MPO



def apply_layer(MPO, circuit1_dag, circuit2_dag, first_iterator, second_iterator, threshold):
    for n in first_iterator:
        # print(n, n+1)
        MPO = update_MPO(MPO, circuit1_dag, circuit2_dag, [n, n+1], threshold)

    for n in second_iterator:
        # print(n, n+1)
        MPO = update_MPO(MPO, circuit1_dag, circuit2_dag, [n, n+1], threshold)

    return MPO


def multiply_MPOs(MPO_1, MPO_2, max_bond_dimension=None, sites=None, conjugate=False):
    # TODO: Generalize and rename function
    assert len(MPO_1) <= len(MPO_2)
    # MPO_1 must be the larger MPO
    if len(MPO_1) == len(MPO_2):
        sites = [0, len(MPO_1)-1]
    if len(sites) == 1:
        sites.append(sites[0])

    # if len(MPO_1) > len(MPO_2):
    #     sites_larger_MPO = range(len(MPO_1))
    # else:
    sites_larger_MPO = range(len(MPO_2))

    direct_result_MPO = []
    site_smaller_MPO = 0
    for site_larger_MPO in sites_larger_MPO:
        if site_larger_MPO in range(sites[0], sites[1]+1):
            # Always multiplies in order MPO1-MPO2 --> MPO2 is G'
            updated_site = oe.contract('ijkl, kbcd->ijbcld',  MPO_1[site_smaller_MPO], MPO_2[site_larger_MPO])

            dims = updated_site.shape
            updated_site = np.reshape(updated_site, (dims[0], dims[1]*dims[2], dims[3], dims[4]*dims[5]))
            direct_result_MPO.append(updated_site)
            site_smaller_MPO += 1
        else:
            direct_result_MPO.append(MPO_2[site_larger_MPO])

    direct_result_MPO = SVD_compression_sweep(direct_result_MPO, max_bond_dimension, threshold=1e-13)

    return direct_result_MPO


def apply_long_range_layer(MPO, dag1, dag2, conjugate, threshold):
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
                                gate_MPO = convert_dag_to_tensor_algorithm(node)[0].tensor
                                # Remove from dag
                                dag1.remove_op_node(node)
                                break
                    else:
                        for node in dag2.op_nodes():
                            # Guarantees the correct node is removed
                            if node.name == gate.operation.name and len(node.qargs) >= 2 and node.qargs[0]._index == gate.qubits[0]._index and node.qargs[1]._index == gate.qubits[1]._index:
                                gate_MPO = convert_dag_to_tensor_algorithm(node)[0].tensor
                                flip_network(gate_MPO, conjugate=True)
                                # Remove from dag
                                dag2.remove_op_node(node) 
                                break
                    break

    assert 'gate_MPO' in locals()

    assert len(gate_MPO) <= len(MPO)
    # MPO_2 must be the larger MPO
    if len(gate_MPO) == len(MPO):
        sites = range(0, len(MPO))
    else:
        sites = range(location, location+distance)

    # TODO: Only contract first site with gate, then just SVD across chain to second gate tensor
    for site_gate_MPO, overall_site in enumerate(sites):
        # Even sites of gate MPO
        if site_gate_MPO != len(sites)-1 and site_gate_MPO % 2 == 0:
            # sigma i, sigma i+1, upper bond gate, upper bond MPO, sigma' i, sigma' i+1, lower bond gate, lower bond MPO
            if not conjugate:
                theta = oe.contract('abcd,edfg,chij,fjkl->aebhikgl', gate_MPO[site_gate_MPO], gate_MPO[site_gate_MPO+1], MPO[overall_site], MPO[overall_site+1])
            else:
                flip_network(MPO)
                theta = oe.contract('abcd,edfg,chij,fjkl->ikhbaelg', gate_MPO[site_gate_MPO], gate_MPO[site_gate_MPO+1], MPO[overall_site], MPO[overall_site+1])
                flip_network(MPO)

            # Apply causal cone on each side
            theta = apply_causal_cone(theta, dag1, [overall_site, overall_site+1], conjugate=False)
            theta = apply_causal_cone(theta, dag2, [overall_site, overall_site+1], conjugate=True)

            dims = theta.shape
            theta = np.reshape(theta, (dims[0], dims[1], dims[2]*dims[3], dims[4], dims[5], dims[6]*dims[7]))
            MPO[overall_site], MPO[overall_site+1] = decompose_theta(theta, threshold)

            # Used to track tensors already applied
            gate_MPO[site_gate_MPO] = None
            gate_MPO[site_gate_MPO+1] = None

        # Odd length gate MPO, single hanging tensor
        if site_gate_MPO == len(sites)-1 and any(type(tensor) == np.ndarray for tensor in gate_MPO):
            if not conjugate:
                theta = oe.contract('abcd,cefg->abefdg', gate_MPO[site_gate_MPO], MPO[overall_site])
            else:
                flip_network(MPO)
                theta = oe.contract('abcd,cefg->febagd', gate_MPO[site_gate_MPO], MPO[overall_site])
                flip_network(MPO)

            dims = theta.shape
            theta = np.reshape(theta, (dims[0], dims[1]*dims[2], dims[3], dims[4]*dims[5]))

            theta = oe.contract('abcd, edfg->aebcfg', MPO[overall_site-1], theta)
            
            theta = apply_causal_cone(theta, dag1, [overall_site-1, overall_site], conjugate=False)
            theta = apply_causal_cone(theta, dag2, [overall_site-1, overall_site], conjugate=True)

            MPO[overall_site-1], MPO[overall_site] = decompose_theta(theta, threshold)
            gate_MPO[site_gate_MPO] = None

    assert not any(type(tensor) == np.ndarray for tensor in gate_MPO)

    return MPO


def select_starting_point(N, DAG_circuit):
    assert N > 0

    for layer in DAG_circuit.layers():
        first_layer = layer
        break

    first_iterator = range(0, N-1, 2)
    second_iterator = range(1, N-1, 2)
    odd = False
    if 'first_layer' in locals():
        layer_circuit = dag_to_circuit(first_layer['graph'])
        for i, gate in enumerate(layer_circuit.data):
            if gate.operation.num_qubits == 2:
                if i % 2 != 0:
                    odd = True
                break

        if odd:
            first_iterator = range(1, N-1, 2)
            second_iterator = range(0, N-1, 2)

    return first_iterator, second_iterator


def check_longest_gate(DAG_circuit):
    largest_distance = 1
    for layer in DAG_circuit.layers():
        first_layer = layer
        break

    if 'first_layer' in locals():
        layer_circuit = dag_to_circuit(first_layer['graph'])

        for gate in layer_circuit.data:
            if gate.operation.num_qubits > 1:
                distance = np.abs(gate.qubits[0]._index - gate.qubits[-1]._index)+1
                if distance > largest_distance:
                    largest_distance = distance

    return largest_distance


def iterate(MPO, dag1, dag2, threshold):
    # Ensures we start at a two-qubit gate (Adds performance boost)
    # Loop while nodes are removed from DAG
    N = len(MPO)
    if dag1.op_nodes():
        first_iterator, second_iterator = select_starting_point(N, dag1)
    else:
        first_iterator, second_iterator = select_starting_point(N, dag2)

    while dag1.op_nodes() or dag2.op_nodes():
        # TODO: Generalize to dag1 or dag2
        largest_distance1  = check_longest_gate(dag1)
        largest_distance2  = check_longest_gate(dag2)

        if largest_distance1 in [1, 2] and largest_distance2 in [1, 2]:
            MPO = apply_layer(MPO, dag1, dag2, first_iterator, second_iterator, threshold)
        else:
            if largest_distance1 >= largest_distance2:
                conjugate = False
            else:
                conjugate = True
            MPO = apply_long_range_layer(MPO, dag1, dag2, conjugate, threshold)
    return MPO


def run(circuit1, circuit2, threshold=1e-13, fidelity=1-1e-13):
    # Initialization
    N = circuit1.num_qubits
    MPO = initialize_identity_MPO(N)
    circuit1_dag = circuit_to_dag(circuit1)
    circuit2_dag = circuit_to_dag(circuit2)
    MPO = iterate(MPO, circuit1_dag, circuit2_dag, threshold)

    return check_if_identity(MPO, fidelity)