# Purpose: functions for calculating QWC and FC decompositions of a qubit Hamiltonian via sorted insertion decomposition

import numpy as np
from openfermion import (
    QubitOperator as Q
)
from utils_basic import (
    is_commuting,
    is_qubit_wise_commuting
)

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
