"""
TODO: the process_fermionic_hamiltonian_to_remove_irrelevant_terms function uses the 2 ** N representation of the bra and ket to remove irrelevant terms
      the brute force numerical nature of this implies that it could remove terms which are "accidentally" zero
      it also suffers from exponential complexity to run

      this should be fixed by re-writing the code to use the generators and seniority configurations directly to remove states
"""

#
#    Purpose: functions for processing and/or tapering fermionic Hamiltonians using seniority symmetries 
#
#    Note: the bulk of the fermionic processing code is in utils_CSF_and_UCSF.py, so incorporating that is to be done later
#          for now, this file implements fermionic Hamiltonian processing in a brute force way
#          it should end up being the same as what is in utils_CSF_and_UCSF.py
#

import numpy as np
import scipy.sparse
from openfermion import FermionOperator
from utils_ferm import op_action_tz
from utils_states import convert_TZ_format_to_sparse_format

def process_fermionic_hamiltonian_to_remove_irrelevant_terms(H, Nqubits, bra, ket):
    """
    H   : FermionOperator
    bra : and scipy.sparse.csr_matrix implementing
    ket : a TZ_state

    return : Heff, containing only the fermionic terms for which <bra|term|ket> is non-zero

    note: TZ_state is not a formal Class, but it is essentially a data structure used in some of the functions to model the states
          it is a list with three entries
              1. a list of np.array of 0 and 1, denoting the Slater determinants present in the state 
              2. a list of integers, corresponding to decimal representations of the Slater determinant occupations
              3. a list of coefficients
          We use a TZ_state for the ket since the function which applys H to a state takes this as input
    """
    dim  = 2 ** Nqubits
    Heff = FermionOperator()
    
    for term, coef in H.terms.items():
        term_on_ket = op_action_tz(FermionOperator(term), ket[0], ket[1], ket[2])
        matrix_element = bra.dot(convert_TZ_format_to_sparse_format(dim, term_on_ket).T)
        if np.abs(matrix_element) > 1e-10:
            Heff += coef * FermionOperator(term)
    
    return Heff

def process_fermionic_hamiltonian_to_remove_irrelevant_and_classically_efficient_terms(H, Nqubits, bra, ket, bra_somos, ket_somos):
    """
    generalized version of previous function which also takes into account terms which do not need the quantum computer to evaluate

    "conditions" in the implementation refer to things that must be true for the term to be included in the final modified Hamiltonian
    """
    dim       = 2 ** Nqubits
    Heff      = FermionOperator()
    somos_set = set(bra_somos + ket_somos)
    
    for term, coef in H.terms.items():
        p, q, r, s = term[0][0], term[0][1], term[0][2], term[0][3]
        condition1 = {p,q,r,s}.intersection(somos_set) != {p,q,r,s}                         # p,q,r,s leaves the classically efficient subset of orbitals

        term_on_ket    = op_action_tz(FermionOperator(term), ket[0], ket[1], ket[2])
        matrix_element = bra.dot(convert_TZ_format_to_sparse_format(dim, term_on_ket).T)    
        condition2     = np.abs(matrix_element) > 1e-10                                     # matrix element is non-zero
        if condition1 and condition2:
            Heff += coef * FermionOperator(term)
    
    return Heff
 
