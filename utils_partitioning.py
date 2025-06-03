# Purpose: functions for calculating QWC and FC decompositions of a qubit Hamiltonian via sorted insertion decomposition

import numpy as np
from openfermion import (
    QubitOperator as Q,
    expectation,
    get_sparse_operator
)

from utils_basic import (
    is_commuting,
    is_qubit_wise_commuting
)
import tequila as tq

from utils_si_ics import (
    OverlappingAuxiliary,
    OverlappingGroups
)
import scipy.sparse._csr

def is_qwc_hamiltonian(H):
    """
    checks if H is a QWC Hamiltonian
    """
    for i, A in enumerate(H.terms.keys()):
        for j, B in enumerate(H.terms.keys()):
            if i > j:
                if not is_qubit_wise_commuting(A, B):
                    return False
    return True

def is_fc_hamiltonian(H):
    """
    checks if H is an FC Hamiltonian
    """
    for i, A in enumerate(H.terms.keys()):
        for j, B in enumerate(H.terms.keys()):
            if i > j:
                if not is_commuting(A, B):
                    return False
    return True

def abs_of_dict_value(x):
    """
    sub-routine used to sort Hamiltonian terms by absolute value of coefficients
    """
    return np.abs(x[1])


def inclusion_criterion(fragment, term, methodtag):
    """
    checks if term can be included in fragment 
    while preserving solvability characteristic methodtag which is in {'fc', 'qwc'}
    """

    if methodtag == 'fc':
        for fragment_term, _ in fragment.terms.items():
            if not is_commuting(Q(fragment_term), Q(term)):
                return False
        return True
    
    elif methodtag == 'qwc':
        for fragment_term, _ in fragment.terms.items():
            if not is_qubit_wise_commuting(Q(fragment_term), Q(term)):
                return False
        return True
    
    else:
        print("not implemented")
        return None

def sorted_insertion_decomposition(H, methodtag):
    """
    implements sorted insertion decomposition of H
    methodtag denotes solvability characteristic for fragments {'fc', 'qwc'}
    
    return is a list of QubitOperator
    returns None if H has a constant term --> it must be removed first 
    """
    
    if H.constant != 0.0:
        print("Constant term in H must be removed before sorted insertion decomposition")
        return None

    H.terms  = dict(sorted(H.terms.items(), key=abs_of_dict_value, reverse=True))
    
    decomp = [Q().zero()]
    for term, coef in H.terms.items():
        success = False
        for fragment in decomp:
            if fragment == Q().zero():
                fragment += coef * Q(term)
                success   = True
                break
            
            elif inclusion_criterion(fragment, term, methodtag):
                fragment += coef * Q(term)
                success   = True
                break
        
        if not success:
            decomp = decomp + [coef * Q(term)]
    
    return decomp

def augment_decomp_with_pauli_x(decomp, N):
    """
    puts a sigma_x on qubit N for all fragments. this is needed for extended swap test
    """
    x = Q(f'X{N}')
    return [Op * x for Op in decomp]

def augment_decomp_with_pauli_x_plus_i_pauli_y(decomp, N):
    """
    puts an x, and an i * sigma_y on qubit N for all fragments. this is needed for extended swap test
    this doubles the number of fragments
    it is needed for the off diagonal matrix elements
    """
    x  = Q(f'X{N}')
    iy = 1j * Q(f'Y{N}')
    return [Op * x for Op in decomp] + [Op * iy for Op in decomp]

# SI-ICS code
# potential flaw: Hamiltonians now have complex coefficients, so this code may not work

def convert_QubitOperator_to_BinaryHamiltonian(H):
    Htequila = tq.QubitHamiltonian.from_openfermion(H)
    Hbinary  = tq.grouping.binary_rep.BinaryHamiltonian.init_from_qubit_hamiltonian(Htequila, ignore_const=True)

    return Hbinary

def calculate_cov_dict(state, Hbin, Nqubits):
    """
    state is a 2**N vector or a csr_matrix
    Hbin is a BinaryHamiltonian (Tequila)

    return: dictionary of (bin1, bin2) : Cov_{Op[bin1], Op[bin2]}

    note: if state is a csr_matrix, this function will convert it to a vector
    """
    if isinstance(state, scipy.sparse._csr.csr_matrix):
        state = state.toarray()[0]

    cov_dict = dict()

    for i, term1 in enumerate(Hbin.binary_terms):
        Op1 = get_sparse_operator(Q(term1.to_pauli_strings().key_openfermion()), Nqubits)
        for j, term2 in enumerate(Hbin.binary_terms):
            if i >= j and term1.commute(term2):
                Op2 = get_sparse_operator(Q(term2.to_pauli_strings().key_openfermion()), Nqubits)
                cov = expectation(Op1 @ Op2, state) - expectation(Op1, state) * expectation(Op2, state)
                cov_dict[(term1.binary_tuple(), term2.binary_tuple())] = cov

    return cov_dict

