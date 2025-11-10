from __future__ import annotations
from scipy.sparse import csr_matrix
import numpy as np
from openfermion import FermionOperator, count_qubits
from copy import deepcopy

def SD_to_int(SD: tuple):
    """
    occupations to integer.  

    Follows OpenFermion index convention: Big endian, i.e, (1, 0, 0) == 4
    
    """
    val = 0
    for i, occ in enumerate(SD):
        val += occ * 1<<(len(SD) - 1 - i)
    return val

class SDState:
    """
    Stores a wavefunction as a dictionary of slater determinants (tuples) and their coefficients
    
    """

    def __init__(self, state: dict[tuple, complex], n_modes = None):
        self.state = state
        if n_modes is None:
            n_modes = len(list(self.state.keys())[0])
        self.n_modes = n_modes


    def __str__(self):
        print_str = "SD state over {} modes consisting of {} determinants.".format(self.n_modes, len(self.state.keys()))

        count = 0
        for k, v in self.state.items():
            print_str += "\n{}: {}".format(k, v)
            count +=1
            if count >50:
                print_str+= "\n..."
                break
        
        return print_str
    
    @classmethod
    def init_from_sparse_array(cls, state_array):
        #TODO

        return
    
    @classmethod
    def init_from_SD_list(cls, bases: list[tuple], coeffs: list[complex], n_modes: int = None):
        state = {b: c for b, c in zip(bases, coeffs)}
        return cls(state=state, n_modes=n_modes)
    
    def scalar_prod(self, fac):
        """
        In-place scalar multiplication of state by fac

        """
        for k in self.state.keys():
            self.state[k] *= fac
    
    def tensor_prod(self, other: SDState, modes_self = None, modes_other= None):
        """
        Tensor product two states, 
        modes_self, modes_other: list[int] specifies qubit indices of self and other in the joint register. other added to end if not specified.


        """
        def append_keys(k1, k2, idx1, idx2):
            new_key = [0]*(len(k1) + len(k2))
            for i, k in zip(idx1, k1):
                new_key[i] = k
            for i, k in zip(idx2, k2):
                new_key[i] = k
            return tuple(new_key)

        N = self.n_modes + other.n_modes
        if modes_self is None and modes_other is None:
            modes_self = list(range(self.n_modes))
            modes_other = list(range(self.n_modes, self.n_modes + other.n_modes))
        else:
            if modes_self is None:
                assert len(modes_other) == other.n_modes, "Incorrect number of modes specified"
                modes_self = set(range(N))
                _ = [modes_self.remove(i) for i in modes_other]
            elif modes_other is None:
                assert len(modes_self) == self.n_modes, "Incorrect number of modes specified"
                modes_other = set(range(N))
                _ = [modes_other.remove(i) for i in modes_self]
            else:
                assert set(modes_self).union(set(modes_other)) == set(range(N)), "Incorrect modes specified"
        
        new_state = {}
        for k1, v1 in self.state.items():
            for k2, v2 in other.state.items():
                k_new = append_keys(k1, k2, modes_self, modes_other)
                new_state[k_new] = v1*v2
        
        return SDState(new_state, self.n_modes + other.n_modes)
    
    def __rmul__(self, scalar: complex):
        """
        Returns scalar * psi

        """

        new_state = {}
        for k, v in self.state.items():
            new_state[k] = scalar * v
        
        return SDState(new_state, self.n_modes)
    
    def __add__(self, other: SDState):
        """
        Returns new SDState object with added SDStates. Adds coefficients of shared SD
        
        """
        assert self.n_modes == other.n_modes, "Incompatible SDStates of modes {} and {} being added.".format(self.n_modes, other.n_modes)

        new_state = deepcopy(self.state)
        for k, v in other.state.items():
            if k in new_state:
                new_state[k] += v
            else:
                new_state[k] = v
        
        return SDState(new_state, self.n_modes)
    
    def normalize(self):
        N = self.inner_prod(self)
        self.scalar_prod(1/np.sqrt(N))
    
    def apply_ferm_op(self, ferm_op: FermionOperator, tol=1e-5):
        """
        Applies FermionOperator onto state, returns new SDState object

        """
        assert self.n_modes >= count_qubits(ferm_op)
        state_new = {}

        #iterate SD
        for SD, SD_coeff in zip(self.state.keys(), self.state.values()):
            
            #iterate ops
            for op_idx, op_coeff in ferm_op.terms.items():
                SD_new = list(SD)
                new_coeff = SD_coeff*op_coeff

                #CR/AN ops in fermion string
                for pos, dag in reversed(op_idx):
                    if SD_new[pos] != dag:
                        SD_new[pos] = dag
                        new_coeff *= (-1)**(sum(SD_new[:pos]))
                    else:
                        new_coeff = 0
                        break

                if abs(new_coeff) <= tol:
                    continue
                else:
                    SD_new = tuple(SD_new)
                    if SD_new in state_new.keys():
                        state_new[SD_new] += new_coeff
                    else:
                        state_new[SD_new] = new_coeff

        return SDState(state_new, self.n_modes)
    
    def inner_prod(self, other):
        """
        Calculate < other | self >
        """
        assert self.n_modes == other.n_modes, "Incompatible states!"
        prod = 0

        for k in self.state:
            if k in other.state:
                prod += np.conj(other.state[k])*self.state[k]
        return prod
    
    def expectation_ferm_op(self, ferm_op: FermionOperator):
        
        return (self.apply_ferm_op(ferm_op=ferm_op)).inner_prod(other=self)
    
    def mat_element_obt_tbt(self, other, obt, tbt, e_nuc):
        """
        Determines matrix element <other | H | self> with H described by obt, tbt, e_nuc using slater condon rules
        TODO
        """

        return
    
    def get_sparse_state(self):
        """
        Returns state as a CSR object

        """
        data = list(self.state.values())
        
        k = len(data)
        cols = [0]*k
        rows = [SD_to_int(SD) for SD in self.state.keys()]
        return csr_matrix((data, (rows, cols)), shape=(1<<self.n_modes, 1))

    @classmethod
    def construct_subspace_H(H: FermionOperator, SDState_list: list[SDState], hermitian = True):
        """
        Construct Hamiltonian matrix in provided subspace

        Evaluates only upper triangular part of the matrix if Hermitian set True

        """
        m = len(SDState_list)
        H_mat = np.zeros(shape=(m, m), dtype=complex)

        if hermitian:
            for i, SD1 in enumerate(SDState_list):
                H_SD1 = SD1.apply_ferm_op(H)

                for j, SD2 in enumerate(SDState_list[i:]):
                    SD2_H_SD1 = H_SD1.inner_prod(SD1)

                    H_mat[i, j] = SD2_H_SD1
                    H_mat[j, i] = np.conj(SD2_H_SD1)
        else:
            for i, SD1 in enumerate(SDState_list):
                H_SD1 = SD1.apply_ferm_op(H)

                for j, SD2 in enumerate(SDState_list):
                    SD2_H_SD1 = H_SD1.inner_prod(SD1)
                    H_mat[i, j] = SD2_H_SD1
        return H_mat