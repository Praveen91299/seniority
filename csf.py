import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from utils_circuit import *

class CSF:
    """
    Class to store CSF object, their corresponding excitation rotations, and retrieve normal and tapered circuits
    
    """
    def __init__(self, kind, orbitals, n_orb, ne, exc_list = [], thetas = None):
        self.n_orb = n_orb
        self.ne = ne
        self.kind = kind

        assert self.get_num_targ_orb() == len(orbitals), 'Incorrect number of orbitals in {} for kind: {}'.format(orbitals, self.kind)
        self.orbitals = orbitals

        self.exc_list = exc_list
        self.thetas = None
        self.initialize_thetas(thetas)
    
    def initialize_thetas(self, thetas):

        if thetas is None:

            name = self.kind
            for i in self.orbitals:
                name += '_{}'.format(i)
            
            self.thetas = [Parameter(name=name+'_{}'.format(i)) for i in range(len(self.exc_list))]
        else:
            assert len(thetas) == len(self.exc_list), "Incorrect number of thetas passed! Excitation count :{}, theta count: {}".format(len(self.exc_list), len(thetas))
            self.thetas = thetas
    
    def get_thetas(self):
        return self.thetas
    
    def get_excitations(self):
        return self.exc_list
    
    def get_num_targ_orb(self):
        if self.kind == "hf":
            return 0
        if self.kind == "Tia":
            return 2
        if self.kind == "Stt":
            return 4
        if self.kind == "Sss":
            return 4
    
    def get_tapered_circuit(self, add_hf = True):
        qc = QuantumCircuit(self.n_orb) # 1 for control

        if add_hf:
            qc.append(get_tapered_hf_circuit(self.n_orb, self.ne).to_gate(), qc.qubits)

        if self.kind == "hf":
            return qc
        
        if self.kind == "Tia":
            i, a = self.orbitals

            qc.append(get_tapered_Tia_circuit(i, a, self.n_orb).to_gate(), qc.qubits)
        if self.kind == "Stt":
            i, j, a, b = self.orbitals

            qc.append(get_tapered_Stt_circuit(i, j, a, b, self.n_orb).to_gate(), qc.qubits)
        if self.kind == "Sss":
            i, j, a, b = self.orbitals
            assert i < j and a < b, 'i: {} j: {} a: {} b: {}'.format(i, j, a, b)

            qc.append(get_tapered_Sss_circuit(i, j, a, b, self.n_orb).to_gate(), qc.qubits)

        return qc
    
    def get_parity_string(self):

        parity = np.zeros(self.n_orb)

        for i in self.orbitals:
            parity[i] = 1

        return parity