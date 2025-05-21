import numpy as np
import scipy.sparse

def convert_TZ_format_to_sparse_format(dim, tz_state):
    """
    converts quantum state expressed in TZ format to scipy.sparse.csr_matrix format

    TZ format is a list with three entries:
        1. list of np.arrays of 0 and 1, indicating a Slater determinant/computational basis state
        2. list of integers, corresponding to decimal representation of Slater determinant occupations
        3. list of coefficients, corresponding to the Slater determinants in the previous two strings

    """
    indices    = tz_state[1]
    coefs      = tz_state[2]
    num_values = len(indices)

    assert num_values == len(coefs)

    non_zero_v_entries = ([0] * num_values, indices)

    return scipy.sparse.csr_matrix((coefs, non_zero_v_entries), shape=(1, dim))

def variance_of_operator(Op, State):
    """
    computes the variance of a Hermitian operator <psi|H^2|psi> - <psi|H|psi>^2
    """
    first  = (State @ Op) @ (Op @ State.T)
    second = (State @ Op @ State.T) ** 2
    return first - second

def variance_of_general_operator(Op, State):
    """
    computes the variance of a general non-Hermitian operator<psi|H^t H|psi> - <psi|H^t|psi><psi|H|psi>

    note that qubit Hamiltonians with complex coefficients are not Hermitian. But a QWC or FC Hamiltonian can be measured independent of
    what the coefficients are
    """
    first  = (State @ Op.conjugate().transpose()) @ (Op @ State.T)
    second = (State @ Op @ State.T)
    third  = (State @ Op.conjugate().transpose() @ State.T) 
    return first - (second * third)
