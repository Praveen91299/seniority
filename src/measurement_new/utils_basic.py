# this module contains basic functions 

import numpy as np
import networkx as nx
from openfermion import QubitOperator as Q
from openfermion import MajoranaOperator as M
from openfermion import hermitian_conjugated as dagger
from openfermion import commutator, anticommutator, get_fermion_operator
from openfermion import get_sparse_operator as gso
import random
from numpy.random import uniform
from math import ceil, log

#
#    Pauli Matrices
#

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

#
#    Miscellaneous Simple Functions
#

def print_state(psi, threshold=1e-12):
    """
    gives a sort-of nice print out of the statevector psi
    """
    Nqubits = int(log(len(psi), 2))
    for i in range(len(psi)):
        if np.abs(psi[i]) > 1e-12:
            bin_string = bin(i)[2:]
            bin_string = '0' * (Nqubits - len(bin_string)) + bin_string
            print(bin_string, np.round(psi[i], 6))

    return None

def copy_hamiltonian(H):
    """
    returns an exact copy of the QubitOperator H
    """
    H_copy = Q().zero()

    for t, s in H.terms.items():
        H_copy += s * Q(t)

    assert (H - H_copy) == Q().zero()
    return H_copy

def random_pauli_term(Nqubits):
    """
    returns the term-tuple and QubitOperator for a random Pauli operator on Nqubits
    """
    
    letters = ['X', 'Y', 'Z', 'I']

    term_tuple = []

    for i in range(Nqubits):
        current_letter = random.sample(letters, 1)[0]
        if current_letter != 'I':
            term_tuple.append( (i, current_letter) )

    return tuple(term_tuple), Q(tuple(term_tuple))

def random_pauli_hamiltonian(Nqubits, Nterms):
    """
    returns a random qubit Hamiltonian on Nqubits, with <= Nterms

    Note that Nterms is basically how many random Pauli operators are generated; it can generate the same Pauli more than once in general
    """
    H = Q()

    for _ in range(Nterms):
        H += uniform(-2, 2) * random_pauli_term(Nqubits)[1]
    
    return H

def random_bin_list(N):
    """
    returns a random binary list (like [0,1,1,0,0,1,1,1,1,0]) of length N
    """
    return [random.randint(0, 1) for _ in range(N)]

def state_from_bin_list(bin, N=None):
    if N is None:
        N = len(bin)
    
    psi                = np.zeros(2 ** N)
    b_str              = ''.join('1' if b else '0' for b in bin)
    psi[int(b_str, 2)] = 1.0
    return psi

def random_hamiltonian_with_specified_terms(op_list):
    """
    op_list is a list of QubitOperators with a single term

    return is a random linear combination of the terms as a QubitOperator
    """
    H = Q().zero()
    for op in op_list:
        H += uniform(-2, 2) * op
    return H

#
#    Pauli Product Algebraic Relations
#

def is_commuting(A, B):
    """
    checks if A and B commute. They can be QubitOperators or terms-tuples
    """
    if isinstance(A, tuple):
        A = Q(A)

    if isinstance(B, tuple):
        B = Q(B)

    return commutator(A, B) == Q().zero()

def is_qubit_wise_commuting(A, B):
    """
    checks if A and B qubit-wise-commute. They can be QubitOperators or terms-tuples
    """
    if isinstance(A, tuple):
        A = Q(A)

    if isinstance(B, tuple):
        B = Q(B)

    ps_dict = {}

    pw, _ = A.terms.copy().popitem()

    for ps in pw:
        ps_dict[ps[0]] = ps[1]

    pw, _ = B.terms.copy().popitem()
    for ps in pw:
        if ps[0] in ps_dict:
            if ps[1] != ps_dict[ps[0]]:
                return False

    return True

def is_anticommuting(A, B):
    """
    checks if A and B anti-commute. They can be terms-tuples or QubitOperators
    """
    if isinstance(A, tuple):
        A = Q(A)

    if isinstance(B, tuple):
        B = Q(B)

    return anticommutator(A, B) == Q().zero()

#
#    Basic Clifford Unitary Functions (note: Binary-Symplectic formalism for Cliffords is implemented in utils_fc.py)
#

def apply_unitary_product(H, U_list):
    """
    H      : a QubitOperator 
    U_list : a list of QubitOperators which are assumed to be unitary
             this function will work for any list of QubitOperators though

    result : H will be conjugated by all elements of U_list
             so the output is U_list[-1]*...*U_list[0]*H*U_list[0]^t*...*U_list[-1]^t
    """
    for U in U_list:
        H = U * H * dagger(U)
        H.compress()
    return H

def compute_product_of_unitaries(U_list):
    """
    U_list : a list of QubitOperators which are assumed to be unitary
             this function will work for any list of QubitOperators though

    result : the product U_list[-1] * U_list[-2] * ... * U_list[1] * U_list[0]
    """
    U_prod = Q("")
    for U in U_list:
        U_prod = U * U_prod
    return U_prod

def is_unitary(U):
    """
    checks if the QubitOperator U is unitary
    """
    return U * dagger(U) == Q("")

def clifford(A, B):
    """
    A and B are QubitOperators

    returns the Clifford unitary 1 / sqrt{2} * (A + B)
    """
    assert is_anticommuting(A, B)
    return (A + B) / np.sqrt(2)

#
#    Brute Force Linear Algebra Operations for Many-Body Operators
#

def goto_matrix(X, N):
    """
    X is a FermionOperator, MajoranaOperator, or QubitOperator
    N can be two things:
        a. the number of qubits if X is a FermionOperator or QubitOperator
        b. the number of Majorana modes if X is a MajoranaOperator
    
    return is repsentation of X in the computational basis as a numpy array
    """
    if isinstance(X, M):
        return gso(get_fermion_operator(X), ceil(N/2)).toarray()
    else:
        return gso(X, N).toarray()

def obtain_spectrum(X, N):
    """
    X is a FermionOperator, MajoranaOperator, or QubitOperator
    N can be two things:
        a. the number of qubits if X is a FermionOperator or QubitOperator
        b. the number of Majorana modes if X is a MajoranaOperator

    return is a sorted list of eigenvalues of X, rounded to 8 decimal places, with repetitions
    """
    Xmat = goto_matrix(X, N)
    return sorted(np.round(np.linalg.eig(Xmat)[0], 8))

def obtain_spectrum_no_degeneracies(X, N):
    """
    X is a FermionOperator, MajoranaOperator, or QubitOperator
    N can be two things:
        a. the number of qubits if X is a FermionOperator or QubitOperator
        b. the number of Majorana modes if X is a MajoranaOperator

    return is a sorted list of eigenvalues of X, rounded to 8 decimal places, without repetitions
    """
    spectrum = obtain_spectrum(X, N)
    return sorted(list(set(spectrum)))

