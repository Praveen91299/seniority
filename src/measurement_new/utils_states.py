import numpy as np
import scipy.sparse
from .utils_fc import decimal_to_binary_string
from math import log
from openfermion import get_sparse_operator

#
#    functions for obtaining statevectors from raw data
#

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

def convert_dense_format_to_sparse_format(dense_state):
    return scipy.sparse.csr_matrix(dense_state.reshape(1,-1))

def somos_to_seniority_config(somos, Norb):
    """
    takes a list of singly occupied molecular orbitals and returns the list of seniority eigenvalues
    """

    config = []

    for i in range(Norb):
        if i in somos:
            config.append(1)
        else:
            config.append(0)

    return config

def basis_state_seniority_config(basis_state):
    Nqubits = len(basis_state)
    Norb    = Nqubits // 2

    config = []
    for i in range(Norb):
        if basis_state[2*i] == basis_state[2*i+1]:
            config.append(0)
        else:
            config.append(1)
    return config

def tz_state_seniority_config(tz_state):
    basis_states = tz_state[0]
    config_trial = basis_state_seniority_config(basis_states[0])
    for basis_state in basis_states:
        assert config_trial == basis_state_seniority_config(basis_state)
    return config_trial

#
#    functions for handling effect of qubit tapering on statevectors
#

def compress_state(psi):
    """
    psi is a 2N qubit state

    return is "tapered" version of psi on N qubits using MO parity operators.

    It is assumed that psi is an eigenstate of the MO parity operators
    """
    Nqubits = int(log(len(psi), 2))
    psi_t   = np.zeros(2 ** (Nqubits // 2))

    for i, coef in enumerate(psi):
        if np.abs(coef) > 1e-12:
            bin_i_2N               = decimal_to_binary_string(i, length=Nqubits)
            bin_i_N                = bin_i_2N[1::2]
            psi_t[int(bin_i_N, 2)] = coef

    return psi_t

def enlarge_binary_index_using_config(bin_i_N, config):
    """
    subroutine of decompress_state
    """
    N        = len(bin_i_N)
    bin_i_2N = ''

    for idx, z2 in enumerate(bin_i_N):
        Omega     = config[idx]
        z1        = str((Omega + int(z2)) % 2)
        bin_i_2N += z1 + z2

    return bin_i_2N

def decompress_state(psi_t, config):
    """
    psi_t is an N qubit "tapered" state. No assumptions are made about its structure a priori
    config is a length N list of 0 and 1, encoding the eigenvalues of the MO parity operators

    return is 2N qubit state psi whose tapered version is psi_t
    """
    Nqubits = 2 * int(log(len(psi_t), 2))
    psi     = np.zeros(2 ** Nqubits)
    
    for i, coef in enumerate(psi_t):
        if np.abs(coef) > 1e-12:
            bin_i_N               = decimal_to_binary_string(i, length=Nqubits//2)
            bin_i_2N              = enlarge_binary_index_using_config(bin_i_N, config)
            psi[int(bin_i_2N, 2)] = coef

    return psi

#
#    functions for preparing SWAP test states
#

def create_composite_state(v, w, N):
    """
    creates (1 / sqrt(2)) * (|v>|0> + |w>|1>)

    note that the corresponding swap test Hamiltonian is H \otimes x, not x \otimes H
    """
    composite_column_indices = []
    composite_coefficients   = []

    v_column_indices = v.nonzero()[-1]
    for column_index in v_column_indices:
        coefficient         = v[0,column_index] / np.sqrt(2)
        binary_column_index = bin(column_index)[2:]
        larger_column_index = int(binary_column_index + '0', 2)
        composite_column_indices.append(larger_column_index)
        composite_coefficients.append(coefficient)

    w_column_indices = w.nonzero()[-1]
    for column_index in w_column_indices:
        coefficient         = w[0,column_index] / np.sqrt(2)
        binary_column_index = bin(column_index)[2:]
        larger_column_index = int(binary_column_index + '1', 2)
        composite_column_indices.append(larger_column_index)
        composite_coefficients.append(coefficient)
    
    non_zero_composite_entries = ([0]*len(composite_column_indices), composite_column_indices)

    return scipy.sparse.csr_matrix((composite_coefficients, non_zero_composite_entries), shape=(1, 2 ** (N + 1)))

def create_composite_state_prepended(v, w, N):
    """
    creates (1 / sqrt(2)) * (|0>|v> + |1>|w>)

    note that the corresponding swap test Hamiltonian is x \otimes H not x \otimes H

    currently, this version is not used 
    """
    composite_column_indices = []
    composite_coefficients   = []

    v_column_indices = v.nonzero()[-1]
    for column_index in v_column_indices:
        coefficient         = v[0,column_index] / np.sqrt(2)
        binary_column_index = decimal_to_binary_string(column_index, length=N)
        larger_column_index = int('0' + binary_column_index, 2)
        composite_column_indices.append(larger_column_index)
        composite_coefficients.append(coefficient)

    w_column_indices = w.nonzero()[-1]
    for column_index in w_column_indices:
        coefficient         = w[0,column_index] / np.sqrt(2)
        binary_column_index = decimal_to_binary_string(column_index, length=N)
        larger_column_index = int('1' + binary_column_index, 2)
        composite_column_indices.append(larger_column_index)
        composite_coefficients.append(coefficient)
    
    non_zero_composite_entries = ([0]*len(composite_column_indices), composite_column_indices)

    return scipy.sparse.csr_matrix((composite_coefficients, non_zero_composite_entries), shape=(1, 2 ** (N + 1)))

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

def frag_SD_of_decomp(decomp, State, N, general=False):
    """
    Standard deviations of fragmentation decomp
    """
    if not general:
        var_list = [variance_of_operator(get_sparse_operator(Op, N), State) for Op in decomp]
    else:
        var_list = [variance_of_general_operator(get_sparse_operator(Op, N), State) for Op in decomp]
    var_list = np.array([x.toarray()[0,0] for x in var_list])
    return var_list**(1/2)
