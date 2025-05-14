import numpy as np
from openfermion import (
    QubitOperator,
)
from utils_basic import copy_hamiltonian, shift_hamiltonian_qubits


sigmax = np.array([
    [0.0, 1.0],
    [1.0, 0.0]
])

sigmay = np.array([
    [0.0, -1.0j],
    [1.0j, 0.0]
])

sigmaz = np.array([
    [1.0, 0.0],
    [0.0, -1.0]
])

sigma0 = np.array([
    [1.0, 0.0],
    [0.0, 1.0]
])

sigma_dict = {
    'X' : sigmax,
    'Y' : sigmay,
    'Z' : sigmaz,
    '0' : sigma0
}

def split_pauli_operator(T, k):
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
        Pauli operator <v|T[:N]|w> * T[N:]; T[:N] denotes the part of T that acts on first N qubits; T[N:] acts on the rest of the qubits
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

    Trel, Tirrel = split_pauli_operator(T, N)

    return pauli_matrix_element_with_basis_state(Trel, v, w) * QubitOperator(Tirrel)

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
        return shift_hamiltonian_qubits(H_tapered, N)
    return H_tapered

