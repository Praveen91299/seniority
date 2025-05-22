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