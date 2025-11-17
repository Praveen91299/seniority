import numpy as np
from math import log
from openfermion import (
    QubitOperator,
    get_sparse_operator
)
import networkx as nx
from seniority.src.measurement_new.utils_csf_categorize import categorize_csf_state

#
#    Functions to convert factorized format ("factorization_dict") to length 2^N vector format for quantum states
#

def expand_tensor_product(factorization_dict, Nqubits):
    labels          = 'abcdefghijklmnopqrstuvwxyz'
    tensor_dict     = dict([(''.join([labels[x] for x in k]), np.reshape(v, [2] * len(k))) for (k,v) in factorization_dict.items()])
    einsum_equation = ','.join([x for x in tensor_dict.keys()]) + '->' + labels[:Nqubits]
    psi_tensor      = np.einsum(einsum_equation, *tensor_dict.values())
    psi             = np.reshape(psi_tensor, 2 ** Nqubits)

    return psi

def expand_tensor_product_for_incomplete_qubit_set(factorization_dict):
    labels          = 'abcdefghijklmnopqrstuvwxyz'
    tensor_dict     = dict([(''.join([labels[x] for x in k]), np.reshape(v, [2] * len(k))) for (k,v) in factorization_dict.items()])
    einsum_equation = ','.join([x for x in tensor_dict.keys()]) + '->' + ''.join(sorted(''.join([x for x in tensor_dict.keys()])))
    psi_tensor      = np.einsum(einsum_equation, *tensor_dict.values())
    psi             = np.reshape(psi_tensor, 2 ** len(''.join(sorted(''.join([x for x in tensor_dict.keys()])))))

    return psi

#
#    obtain factorization of input Q-SENSE basis state
#

def partial_trace_einsum(psi, qubit_indices, n_total_qubits):
    """
    Compute reduced density matrix by tracing out some qubits using np.einsum.
    
    psi: full state vector (2**n_qubits,)
    qubit_indices: list of qubit indices to keep
    n_total_qubits: total number of qubits in psi

    Function may change the sign of the state
    """
    psi = psi / np.linalg.norm(psi)
    trace_out = [i for i in range(n_total_qubits) if i not in qubit_indices]

    # Labels for einsum - use single characters only
    # Available characters for einsum indices
    available_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

    if n_total_qubits > len(available_chars) // 2:
        raise ValueError(f"Too many qubits ({n_total_qubits}) for einsum method. Maximum supported: {len(available_chars) // 2}")

    # Assign single character labels
    ket_labels = list(available_chars[:n_total_qubits])
    bra_labels = list(available_chars[n_total_qubits:2*n_total_qubits])

    # For traced-out qubits, we identify a_i = b_i (trace)
    for i in trace_out:
        bra_labels[i] = ket_labels[i]

    # Build the einsum string
    einsum_str = f"{''.join(ket_labels)}," \
                 f"{''.join(bra_labels)}->" \
                 f"{''.join(ket_labels[i] for i in qubit_indices)}" \
                 f"{''.join(bra_labels[i] for i in qubit_indices)}"

    # Perform contraction
    psi_tensor = psi.reshape([2] * n_total_qubits)
    rho = np.einsum(einsum_str, psi_tensor, np.conjugate(psi_tensor))

    # Reshape to matrix form
    dim = 2 ** len(qubit_indices)
    rho = rho.reshape((dim, dim))

    eigvals, eigvecs = np.linalg.eigh(rho)
    # Find index of the largest eigenvalue
    idx = np.argmax(np.real(eigvals))
    phi = eigvecs[:, idx]

    purity = np.trace(rho @ rho)
    is_pure = np.abs(purity - 1.0) < 1e-10

    # Normalize (just to be safe)
    phi /= np.linalg.norm(phi)

    return phi, is_pure

def get_indices_mapping_2_wvn(basis_state, mp2_amplitude, Norb, verbose=False):
    """
    Args:
        basis_state: list of occupied spin-orbital indices in the CSF basis state (list_list_refCSF[i][j])
        mp2_amplitude: list of MP2 amplitude data [[[i, a]], amplitude_value (list_list_Uext_mp2_ampld[i])
        Norb: number of spatial orbitals
    Return: {index: 'W' or 'V' or 'N'}

    """
    S_W, S_V, S_N = [], [], []
    
    if not mp2_amplitude == []:
        if verbose:
            print(f'MP2 amplitude data provided: {mp2_amplitude}')
        
        # Extract the spatial indices that have to be quantumly treated
        for amplitude_data in mp2_amplitude:
            # amplitude_data format: [[[0, 5]], amplitude_value]
            # Extract the indices from the first element
            indices = amplitude_data[0][0]  # Gets [0, 5] from [[[0, 5]], amplitude_value]
            if verbose:
                print(f'Indices from MP2 amplitude data: {indices}')
            S_W.extend(indices)  # Add both indices to S_W
        S_W = list(set(S_W))
    
    # Extract indices that are affected by Vu rotations
    (state_type, k, l, a, b) = categorize_csf_state(basis_state)
    if verbose:
        print(f'Categorized CSF state indices: k={k}, l={l}, a={a}, b={b}')
    # Convert spin-orbitals to spatial orbitals (integer division by 2), skip None values
    spatial_orbitals = [idx for idx in [k, l, a, b] if idx is not None]
    S_V = list(set([idx for idx in spatial_orbitals]))
    # Get the invariant indices under Vu and Wu rotations
    S_N = [k for k in range(Norb) if k not in S_W and k not in S_V]
    return {index: 'W' for index in S_W} | {index: 'V' for index in S_V} | {index: 'N' for index in S_N}, state_type

def get_indices_mapping_2_wvn_vo(basis_state, mp2_amplitude, Norb, verbose=False):
    """
    Length of list_CSF, list_list_ia_CSF, tz_states are all the same.

    Args: 
        basis_state: list of occupied spin-orbital indices in the CSF basis state (list_CSF[k])
        mp2_amplitude: list of MP2 amplitude data [[[i, a]], amplitude_value (list_list_ia_CSF[k]: The ia list for k-th CSF)
        Norb: number of spatial orbitals
    Return: {index: 'W' or 'V' or 'N'}
    """
    S_W, S_V, S_N = [], [], []
    if not mp2_amplitude == []:

        if verbose:
            print(f'MP2 amplitude data provided: {mp2_amplitude}')
        
        # Extract the spatial indices that have to be quantumly treated
        for amplitude_data in mp2_amplitude:
            # amplitude_data format: [[[0, 5]], amplitude_value]
            # Extract the indices from the first element
            indices = amplitude_data[0]  # Gets [0, 5] from [[[0, 5]], amplitude_value]
            if verbose:
                print(f'Indices from MP2 amplitude data: {indices}')
            S_W.extend(indices)  # Add both indices to S_W
        S_W = list(set(S_W))
    (state_type, k, l, a, b) = categorize_csf_state(basis_state)
    if verbose:
        print(f'Categorized CSF state indices: k={k}, l={l}, a={a}, b={b}')
    # Convert spin-orbitals to spatial orbitals (integer division by 2), skip None values
    spatial_orbitals = [idx for idx in [k, l, a, b] if idx is not None]
    S_V = list(set([idx for idx in spatial_orbitals]))
    # Get the invariant indices under Vu and Wu rotations
    S_N = [k for k in range(Norb) if k not in S_W and k not in S_V]
    return {index: 'W' for index in S_W} | {index: 'V' for index in S_V} | {index: 'N' for index in S_N}, state_type

def factorize_state(basis_csr_state, S_w, S_v, S_n, state_type, correct_sign=True):
    """
    Factorize a quantum state into tensor product components based on orbital sets.

    Input:
        basis_csr_state: Sparse CSR matrix representing the quantum state (scipy.sparse.csr_matrix)
                        State is in spatial orbital basis with dimension 2^n_spatial_orbitals
        S_w: list of spatial orbital indices for W block (quantum treatment needed)
        S_v: list of spatial orbital indices for V block (affected by rotations)
        S_n: list of spatial orbital indices for N block (invariant)

    Return:
        dict mapping spatial orbital tuples to state vectors:
        {(orbitals_w...): psi_w, (orbitals_v_1...): psi_v_1, (orbitals_v_2...): psi_v_2, (i,): psi_i, ...}

    The keys are tuples of spatial orbital indices, values are normalized state vectors.

    Factorization strategy:
    - S_W: All W orbitals kept as one block
    - S_V with 2 spatial orbitals: kept as one block (S_v_1)
    - S_V with 4 spatial orbitals: split into two blocks of 2 spatial orbitals each (S_v_1, S_v_2) if it's State type 3. 
    - S_N: Each spatial orbital factorized separately as (i,)
    - correct_sign: the factorization may change the sign (global phase) of the state. Set correct_sign to True to ensure the sign is correct
    """
    # Convert sparse matrix to dense array
    if basis_csr_state.shape[0] == 1:
        full_state = np.array(basis_csr_state.todense()).flatten()
    else:
        full_state = np.array(basis_csr_state.todense()).flatten()

    # Determine number of spatial orbitals from state dimension
    state_dim = len(full_state)
    n_spatial_orbitals = int(np.log2(state_dim))

    factorization_dict = {}

    # Handle S_W block (if non-empty)
    if len(S_w) > 0:
        S_w_sorted = tuple(sorted(S_w))
        factorization_dict[S_w_sorted], _ = partial_trace_einsum(
            full_state, S_w_sorted, n_spatial_orbitals
        )

    # Handle S_V block with different factorization strategies based on number of spatial orbitals
    n_spatial_V = len(S_v)

    if n_spatial_V == 0:
        # S_V is empty - nothing to factorize
        pass
    elif n_spatial_V == 2:
        # S_V has 2 spatial orbitals - cannot be factorized further, keep as one block
        S_v_sorted = tuple(sorted(S_v))
        factorization_dict[S_v_sorted], _ = partial_trace_einsum(
            full_state, S_v_sorted, n_spatial_orbitals
        )
    elif n_spatial_V == 4:
        if state_type == 3:
            # S_V has 4 spatial orbitals - factorize into two blocks of 2 spatial orbitals each
            # Sort S_v to ensure consistent ordering
            S_v_sorted = sorted(S_v)

            # Split into two pairs of spatial orbitals
            S_v_1 = tuple([S_v_sorted[0], S_v_sorted[1]])
            S_v_2 = tuple([S_v_sorted[2], S_v_sorted[3]])

            # Extract factorized states for each block
            factorization_dict[S_v_1], is_pure = partial_trace_einsum(
                full_state, S_v_1, n_spatial_orbitals
            )
            factorization_dict[S_v_2], _ = partial_trace_einsum(
                full_state, S_v_2, n_spatial_orbitals
            )
            if not is_pure:
                factorization_dict.pop(S_v_1)
                factorization_dict.pop(S_v_2)
                S_v_1 = tuple([S_v_sorted[0], S_v_sorted[2]])
                S_v_2 = tuple([S_v_sorted[1], S_v_sorted[3]])

                # Extract factorized states for each block
                factorization_dict[S_v_1], is_pure2 = partial_trace_einsum(
                    full_state, S_v_1, n_spatial_orbitals
                )
                factorization_dict[S_v_2], _ = partial_trace_einsum(
                    full_state, S_v_2, n_spatial_orbitals
                )
                if not is_pure2:
                    factorization_dict.pop(S_v_1)
                    factorization_dict.pop(S_v_2)
                    S_v_1 = tuple([S_v_sorted[0], S_v_sorted[3]])
                    S_v_2 = tuple([S_v_sorted[2], S_v_sorted[3]])

                    # Extract factorized states for each block
                    factorization_dict[S_v_1], _ = partial_trace_einsum(
                        full_state, S_v_1, n_spatial_orbitals
                    )
                    factorization_dict[S_v_2], _ = partial_trace_einsum(
                        full_state, S_v_2, n_spatial_orbitals
                    )
        elif state_type == 4:
            # State type 4 has i,j,a,b entangled so cannot be factorized further
            S_v_sorted = tuple(sorted(S_v))
            factorization_dict[S_v_sorted], _ = partial_trace_einsum(
                full_state, S_v_sorted, n_spatial_orbitals
            )
    else:
        # For other cases (1, 3, >4 spatial orbitals), keep as one block, though this is not expected
        if len(S_v) > 0:
            S_v_sorted = tuple(sorted(S_v))
            factorization_dict[S_v_sorted], _ = partial_trace_einsum(
                full_state, S_v_sorted, n_spatial_orbitals
            )

    # Handle S_N block - factorize into individual spatial orbitals
    for spatial_idx in S_n:
        # Each spatial orbital is treated separately
        factorization_dict[(spatial_idx,)], _ = partial_trace_einsum(
            full_state, (spatial_idx,), n_spatial_orbitals
        )

    # ensure correct sign if desired
    if correct_sign:
        state_input = basis_csr_state.toarray()[0]
        state_recov = expand_tensor_product_for_incomplete_qubit_set(factorization_dict)

        if np.allclose(state_input, -state_recov):
            arbitrary_key                     = list(factorization_dict.keys())[0]
            factorization_dict[arbitrary_key] = -factorization_dict[arbitrary_key]

        assert np.allclose(expand_tensor_product_for_incomplete_qubit_set(factorization_dict), state_input)


    return factorization_dict

#
#    Functions to obtain unified factorization of bra and ket given maximal factorizations
#

def partition_from_dict(factorization_dict):
    return list(factorization_dict.keys())

def obtain_join_graph(partA, partB):
    all_vals = {val for block in partA + partB for val in block}

    G = nx.Graph()
    G.add_nodes_from(all_vals)

    for block in partA + partB:

        for i in block:
            for j in block:
                if j > i:
                    G.add_edge(i, j)

    return G

def obtain_join_partition(partA, partB):
    G   = obtain_join_graph(partA, partB)
    ccs = nx.connected_components(G)
    
    return [tuple(sorted(cc)) for cc in ccs]

def obtain_coarse_partitioning(join_partition, factorization_dict):
    coarse_dict = {}

    for block in join_partition:
        sub_dict           = {k : v for (k, v) in factorization_dict.items() if set(k).issubset(set(block))}
        psi                = expand_tensor_product_for_incomplete_qubit_set(sub_dict)
        coarse_dict[block] = psi

    return coarse_dict

def obtain_coarse_dicts(factorization_dict1, factorization_dict2):
    partition1 = partition_from_dict(factorization_dict1)
    partition2 = partition_from_dict(factorization_dict2)

    join_partition = obtain_join_partition(partition1, partition2)

    coarse1 = obtain_coarse_partitioning(join_partition, factorization_dict1)
    coarse2 = obtain_coarse_partitioning(join_partition, factorization_dict2)

    return join_partition, coarse1, coarse2

#
#    Function to assign qubits as quantum or classical in matrix element estimation
#

def QC_assignment_from_qubit_labels(bra_labels, ket_labels, join_partition):
    QC_dictionary = {}

    for block in join_partition:
        assignment = 'C'
        for i in block:
            if bra_labels[i] == 'W' or ket_labels[i] == 'W':
                assignment = 'Q'
                break
        QC_dictionary[block] = assignment
    
    return QC_dictionary

#
#    Functions to obtain partially evaluated Hamiltonian matrix elements based on quantum/classical block assignments and unified-factorizations
#

def split_pauli_operator(T, C):
    
    if isinstance(T, QubitOperator):
        T = list(T.terms.keys())[0]

    TC = tuple([term for term in T if term[0] in C])
    TR = tuple([term for term in T if not term[0] in C])

    return TC, TR

def relabel_qubits_in_pauli_term(TC, C):
    label_reassignment = {x : i for i, x in enumerate(C)}
    new_terms_tuple    = []
    for term in TC:
        new_term = (label_reassignment[term[0]], term[1])
        new_terms_tuple.append(new_term)
    return tuple(new_terms_tuple)

def evaluate_matrix_element_given_term(psi, phi, TCrelabelled):
    Nqubits      = int(log(len(psi), 2))
    Pauli_sparse = get_sparse_operator(QubitOperator(TCrelabelled), Nqubits)
    return psi @ Pauli_sparse @ phi.T

def partially_evaluate_pauli_term(psi, phi, T, C):
    TC, TR       = split_pauli_operator(T, C)
    TCrelabelled = relabel_qubits_in_pauli_term(TC, C)
    return evaluate_matrix_element_given_term(psi, phi, TCrelabelled) * QubitOperator(TR)

def partially_evaluate_hamiltonian_matrix_element(psi, phi, H, C):
    H_evaluated = QubitOperator()
    for term, coef in H.terms.items():
        H_evaluated += coef * partially_evaluate_pauli_term(psi, phi, term, C)
    return H_evaluated

def evaluate_fully_classical_factors(factorization_dict_bra, factorization_dict_ket, bra_labels, ket_labels, H):

    join_partition, coarse_dict_bra, coarse_dict_ket = obtain_coarse_dicts(factorization_dict_bra, factorization_dict_ket)
    QC_assignment_dict                               = QC_assignment_from_qubit_labels(bra_labels, ket_labels, join_partition)

    full_Q_block = set()
    for block, assignment in QC_assignment_dict.items():
        if assignment == 'C':
            H = partially_evaluate_hamiltonian_matrix_element(coarse_dict_bra[block], coarse_dict_ket[block], H, block)
            del coarse_dict_bra[block]
            del coarse_dict_ket[block]

        else:
            full_Q_block.update(block)

    if coarse_dict_bra == dict():
        assert coarse_dict_ket == dict()
        return H, np.array([]), np.array([]), 0

    full_Q_block = sorted(tuple(full_Q_block))            
    Heff         = QubitOperator()
    for term, coef in H.terms.items():
        Heff += coef * QubitOperator(relabel_qubits_in_pauli_term(term, full_Q_block))

    Qstate_bra = expand_tensor_product_for_incomplete_qubit_set(coarse_dict_bra)
    Qstate_ket = expand_tensor_product_for_incomplete_qubit_set(coarse_dict_ket)
    NqubitsQ   = int(log(len(Qstate_bra), 2))

    return Heff, Qstate_bra, Qstate_ket, NqubitsQ


