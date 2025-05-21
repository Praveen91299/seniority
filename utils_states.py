import numpy as np
import scipy.sparse

# functions for creating statevectors from raw data

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

def create_composite_state(v, w, N):
    """
    creates \frac{1}{\sqrt{2}}(|v>|0> + |w>|1>)

    note that the corresponding swap test Hamiltonian is H \otimes x, not x \otimes H
    """
    composite_column_indices = []
    composite_coefficients   = []

    v_column_indices = v.nonzero()[-1]
    for column_index in v_column_indices:
        coefficient = v[0,column_index] / np.sqrt(2)
        binary_column_index = bin(column_index)[2:]
        larger_column_index = int(binary_column_index + '0', 2)
        composite_column_indices.append(larger_column_index)
        composite_coefficients.append(coefficient)

    w_column_indices = w.nonzero()[-1]
    for column_index in w_column_indices:
        coefficient = w[0,column_index] / np.sqrt(2)
        binary_column_index = bin(column_index)[2:]
        larger_column_index = int(binary_column_index + '1', 2)
        composite_column_indices.append(larger_column_index)
        composite_coefficients.append(coefficient)
    
    non_zero_composite_entries = ([0]*len(composite_column_indices), composite_column_indices)

    return scipy.sparse.csr_matrix((composite_coefficients, non_zero_composite_entries), shape=(1, 2 ** (N + 1)))



# functions for evaluating linear algebraic quantities

def expectation(Op, State):
    return (State @ Op @ State.T)[0,0]

def matrix_element(Op, Bra, Ket):
    return (Bra @ Op @ Ket.T)[0,0]

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
