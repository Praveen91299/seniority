from openfermion import (
    QubitOperator
)
from seniority.src.measurement_new.utils_basic import (
    clifford,
    copy_hamiltonian,
    apply_unitary_product,
    sigma_dict, 
)

#
#    Functions for Implementing Clifford Transformation
#

def CNOT_clifford_operator(i, j):
    """
    returns Clifford operator implementation of exp(-i * pi / 4) * CNOT(i,j)

    the data structure for a Clifford operator is a list of operators of the form (A + B) / sqrt(2), where A and B anticommute
    note that the adjoint of a clifford U represented in this data structure is U[::-1]
    """

    CNOT_clifford = []
    CNOT_clifford.append(clifford(QubitOperator(f'Y{j}'), QubitOperator(f'Z{j}')))
    CNOT_clifford.append(clifford(QubitOperator(f'X{i} Y{j}'), QubitOperator(f'Y{i} Y{j}')))
    CNOT_clifford.append(clifford(QubitOperator(f'X{i}'), QubitOperator(f'Y{i} X{j}')))

    return CNOT_clifford

def seniority_solving_clifford_operator(Nqubits):
    """
    returns Clifford operator U such that U z_2i z_2i+1 U^T = z_2i for all 0 <= i <= Nqubits-1

    the data structure for a Clifford operator is a list of operators of the form (A + B) / sqrt(2), where A and B anticommute
    """
    assert Nqubits % 2 == 0
    solving_clifford = []
    for i in range(Nqubits // 2):
        solving_clifford += CNOT_clifford_operator(2*i + 1, 2*i)
    return solving_clifford

def group_odds_and_evens(H, Nqubits):
    """
    returns new H with reordered qubits 01234567... -> 0246...1357...
    """
    evens        = [n for n in range(Nqubits) if n % 2 == 0]
    odds         = [n for n in range(Nqubits) if n % 2 == 1]
    neworder     = evens + odds
    reorder_dict = {v : k for k,v in enumerate(neworder)}
    
    Hreordered = QubitOperator()
    for term, coef in H.terms.items():
        newterm = []
        for op in term:
            newterm.append((reorder_dict[op[0]], op[1]))
        newterm = tuple(newterm)
        Hreordered += coef * QubitOperator(newterm)
    
    return Hreordered

def ungroup_odds_and_evens(Hreordered, Nqubits):
    """
    go back to original order from all-evens then all-odds order
    """
    evens           = [n for n in range(Nqubits) if n % 2 == 0]
    odds            = [n for n in range(Nqubits) if n % 2 == 1]
    neworder        = evens + odds
    un_reorder_dict = {k : v for k,v in enumerate(neworder)}
    
    H = QubitOperator()
    for term, coef in Hreordered.terms.items():
        newterm = []
        for op in term:
            newterm.append((un_reorder_dict[op[0]], op[1]))
        H += coef * QubitOperator(newterm)

    return H

#
#    Functions for Implementing Qubit Tapering based on seniorities
#

def split_pauli_operator_seniority(T, k):
    """
    inputs
        T : QubitOperator with one term or terms-tuple of a Pauli acting on any number of qubits
        k : qubit index

    return
        T0 and T1: T0 is the part of T that acts on qubits 0, ..., k-1
                   T1 is the part of T that acts on qubits k, ...
    """
    if isinstance(T, QubitOperator):
        T = list(T.terms.keys())[0]

    T0 = tuple([term for term in T if term[0] < k])
    T1 = tuple([term for term in T if term[0] >= k])
    
    return T0, T1

def pauli_matrix_element_with_basis_state(T, v, w):
    """
    inputs
        T       : QubitOperator with one term or terms-tuple of a Pauli acting on any number of qubits
        v and w : length N lists of 0 and 1

    return
        Pauli matrix element <v|T[:N]|w>; T[:N] denotes the part of T that acts on first N qubits
    """
    if isinstance(T, QubitOperator):
        T = list(T.terms.keys())[0]

    assert len(v) == len(w)
    N = len(v)
    
    val = 1
    for n in range(N):

        if v[n] == w[n]:
            if (n, 'X') in T:
                return 0
            elif (n, 'Y') in T:
                return 0
            elif (n, 'Z') in T:
                val *= sigma_dict['Z'][v[n],w[n]]
            else:
                val *= 1.0

        else:
            if (n, 'X') in T:
                val *= sigma_dict['X'][v[n],w[n]]
            elif (n, 'Y') in T:
                val *= sigma_dict['Y'][v[n],w[n]]
            elif (n, 'Z') in T:
                return 0.0
            else:
                return 0.0
    
    return val

def taper_pauli_term(T, v, w):
    """
    inputs
        T       : QubitOperator with one term or terms-tuple of a Pauli acting on any number of qubits
        v and w : length N lists of 0 and 1

    return
        Pauli operator <v|T[:N]|w> * T[N:]; T[:N] denotes the part of T that acts on first N qubits; T[N:] acts on the rest of the qubits
    """
    assert len(v) == len(w)
    N = len(v)

    Trel, Tirrel = split_pauli_operator_seniority(T, N)

    return pauli_matrix_element_with_basis_state(Trel, v, w) * QubitOperator(Tirrel)

def shift_hamiltonian_qubits_uniformly(H, K):
    """
    H is an N qubit Hamiltonian which acts as I on the first K qubits

    return is an N - K qubit Hamiltonian Hc which is the same as H but acts on the first N - K qubits
    """
    Hc = QubitOperator()

    for term, coef in H.terms.items():
        shifted_term = []
        for op in term:
            shifted_term.append( (op[0] - K, op[1]) )
        shifted_term = tuple(shifted_term)
        Hc += coef * QubitOperator(shifted_term)

    return Hc

def taper_hamiltonian(H, v, w, shift_to_zero=True):
    """
    inputs
        H       : a QubitOperator on M >= N qubits (2N in this application)
        v and w : length N lists of 0 and 1

    return
        a new Hamiltonian H which acts on M-N qubits, sigmas on first N qubits are replaced by scalars <v|sigma|w>
    """
    assert len(v) == len(w)
    N = len(v)

    H_tapered = QubitOperator()
    for term, coef in H.terms.items():
        H_tapered += coef * taper_pauli_term(term, v, w)
        
    if shift_to_zero:
        return shift_hamiltonian_qubits_uniformly(H_tapered, N)
    return H_tapered

def project_out_seniority_symmetries(H, Nqubits, v, w):
    """
    implements one of two ways to process the Hamiltonian in qubit space to reduce the cost of measurements
    this method uses the seniority configurations v and w to taper qubits in a way to produce an N < 2N qubit
    Hamiltonian representation of the block of H that couples v-states to w-states (i.e., qubit tapering).
    It does change the space, so symmetry verification cannot be used

    the algorithm is as follows:
        1. apply Clifford unitary that maps z_{2i} z_{2i+1} -> z_{2i}
        2. reorder qubits 012345 -> 024...135...
        3. replace all Pauli terms P = P[:N] otimes P[N:] with <v|P[:N]|w> P[N:]
        4. rename the qubits N,N+1,... -> 0,1,... and return 
    """
    Ucliff              = seniority_solving_clifford_operator(Nqubits)
    H_rotated           = apply_unitary_product(H, Ucliff)
    H_rotated_reordered = group_odds_and_evens(H_rotated, Nqubits)
    H_tapered           = taper_hamiltonian(H_rotated_reordered, v, w, shift_to_zero=True)

    return H_tapered

#
#    Functions for removal of Hamiltonian terms that don't mix bra-seniority-subspace with ket-seniority-subspace 
#    Currently this is not used in Q-SENSE
#

def remove_irrelevant_pauli_terms(H, v, w):
    """
    inputs
        H       : a QubitOperator on M >= N qubits (2N in this application)
        v and w : length N lists of 0 and 1

    return
        a new Hamiltonian Hvw which acts on M qubits, where all terms that don't couple |v> and |w> are removed
    """
    assert len(v) == len(w)
    N = len(v)

    Hvw = copy_hamiltonian(H)
    for n in range(N):
        terms_to_delete = []
        
        if v[n] == w[n]:
            for term, _ in Hvw.terms.items():
                if ((n, 'X') in term) or ((n, 'Y') in term):
                    terms_to_delete.append(term)

        else:
            for term, _ in Hvw.terms.items():
                if ((n, 'X') not in term) and ((n, 'Y') not in term):
                    terms_to_delete.append(term)

        for term in terms_to_delete:
            del Hvw.terms[term]

    return Hvw

def process_qubit_hamiltonian_to_remove_irrelevant_terms(H, Nqubits, v, w):
    """
    implements one of two ways to process the Hamiltonian in qubit space to reduce the cost of measurements
    this method removes all Pauli terms in H that do not couple states with seniority configuration v to 
    states with seniority configuration w. It does not change the space, so symmetry verification can still 
    be done

    the algorithm is as follows:
        1. apply Clifford unitary that maps z_{2i} z_{2i+1} -> z_{2i}
        2. reorder qubits 012345... -> 024...135...
        3. delete Pauli terms based on v[i] oplus w[i] values corresponding to Xi/Yi or Ii/Zi
        4. reorder qubits 024...135... -> 012345...
        5. apply adjoint of Clifford unitary and return
    """
    Ucliff                        = seniority_solving_clifford_operator(Nqubits)
    H_rotated                     = apply_unitary_product(H, Ucliff)
    H_rotated_reordered           = group_odds_and_evens(H_rotated, Nqubits)
    H_rotated_reordered_processed = remove_irrelevant_pauli_terms(H_rotated_reordered, v, w)
    H_rotated_processed           = ungroup_odds_and_evens(H_rotated_reordered_processed, Nqubits)
    H_processed                   = apply_unitary_product(H_rotated_processed, Ucliff[::-1])

    return H_processed 


