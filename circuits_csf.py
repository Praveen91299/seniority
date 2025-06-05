import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from seniority.circuits_pair_excitation import PairedExcitationRotation


def get_hf_circuit(n_orb, ne):
    assert ne <= n_orb*2, "Provided number of electrons {} incompatible with {} orbitals.".format(ne, n_orb)

    qc = QuantumCircuit(2*n_orb)
    for i in range(ne):
        qc.x(i)
    return qc

def get_tapered_hf_circuit(n_orb, ne):
    assert ne <= n_orb*2, "Provided number of electrons {} incompatible with {} orbitals.".format(ne, n_orb)

    qc = QuantumCircuit(n_orb)
    for i in range(ne//2):
        qc.x(i)
    if ne%2:
        qc.x(ne//2)
    return qc

def get_tapered_Tia_circuit(i, a, n_orb):
    qc = QuantumCircuit(n_orb)

    qc.x(a)
    qc.h(a)
    qc.cx(control_qubit=a, target_qubit=i)

    return qc

def get_tapered_Stt_circuit(i, j, a, b, n_orb):
    qc = QuantumCircuit(n_orb)

    # takes care of hf state
    qc.x(i)
    qc.x(j)

    qc.h(i)
    qc.ry(theta=(-2)*np.arctan2(1, np.sqrt(2)), qubit=j)
    qc.x(a)
    qc.x(j)
    qc.cx(j, a)
    qc.cx(i, j)
    qc.ch(a, b)
    qc.cx(b, a)
    qc.cx(i, a)
    qc.cx(i, b)
    qc.x(i)

    return qc
 
def get_tapered_Sss_circuit(i, j, a, b, n_orb):
    qc = QuantumCircuit(n_orb)

    qc.append(get_tapered_Tia_circuit(i, a, n_orb).to_gate(), qc.qubits)
    qc.append(get_tapered_Tia_circuit(j, b, n_orb).to_gate(), qc.qubits)

    return qc

def CG(S, M, t, sigma):
    """
    Returns Clebsch Gordon coefficient for
    S - total spin
    M - projected spin
    t - change in total spin, t_n - t_{n-1}, provides information about the Genealogical path.
    sigma - added electron projected/total spin
    
    """
    assert abs(M) <= abs(S)
    
    if t == (1/2):
        assert S != 0, "S cannot be 0 if t is +ve"
            
        return np.sqrt((S + 2*sigma*M)/(2*S))
    if t == (-1/2):
        return -2*sigma*np.sqrt((S + 1 - 2*sigma*M)/(2*(S+1)))

def validate_t_vec(t_vec, N):
    """
    Check if t vector is valid and of sufficient length

    """

    assert len(t_vec) >= N #ensures the number of spins N+1 does not exceed available information
    assert t_vec[0] == 0.5, 'Invalid first total spin S1: {}'.format(t_vec[0])

    for i in range(len(t_vec)):
        Si = np.sum(t_vec[:i+1])

        assert abs(t_vec[i]) == 0.5, 'Invalid change in total spin, Si - Si-1: '.format(t_vec[i])
        assert Si >= 0, 'Total spin Si: {} should be positive'.format(Si)
        assert 0.5*np.floor(Si/0.5) == Si, 'Total spin Si: {} should be a multiple of 1/2'.format(Si)
    return

def build_recursive_CSF_circuit(t_vec, N, S, M):
    """
    Build recursive circuit to obtain CSF state on an empty state

    S : total spin
    M : projected spin 
    t_vec : Genealogy 
    N: current number of spins, recursive step number. N=1 is base case 
    
    """
    #checks
    assert S >= 0, "Total spin S: {} is not positive".format(S)
    assert S >= abs(M), "Projected spin: {} is larger than total spin S: {}".format(M, S)
    validate_t_vec(t_vec, N)   

    qc = QuantumCircuit(2*N)

    tN = t_vec[N-1]
    alpha = 1/2
    beta = -1/2

    ### edge cases
    if N == 1:
        # single spin
        if M == 1/2:
            qc.x(0)
        elif M == -1/2:
            qc.x(1)
        return qc
    
    if S == 0: assert tN == -1/2, "Total spin change tN: {}, but needs to be negative to reach S: {}".format(tN, S)

    coeff_alpha = CG(S, M, tN, alpha)
    coeff_beta = CG(S, M, tN, beta)
    qubits_N_1 = qc.qubits[:-2]
    qubit_alpha = qc.qubits[-2]
    qubit_beta = qc.qubits[-1]

    theta = 2*np.arctan2(coeff_alpha, coeff_beta)

    qc.ry(theta, qubit=qubit_alpha)
    qc.cx(qubit_alpha, qubit_beta)
    qc.x(qubit_beta)

    if coeff_alpha != 0:
        Ua = build_recursive_CSF_circuit(t_vec, N-1, S - tN, M - alpha)
        qc.append(Ua.to_gate().control(num_ctrl_qubits=1),  [qubit_alpha]+qubits_N_1)
    
    if coeff_beta != 0:
        Ub = build_recursive_CSF_circuit(t_vec, N-1, S - tN, M - beta)
        qc.append(Ub.to_gate().control(num_ctrl_qubits=1, ctrl_state=0),  [qubit_alpha]+qubits_N_1)
        
    return qc


def build_tapered_recursive_CSF_circuit(t_vec, N, S, M):
    """
    Build recursive circuit to obtain tapered CSF state on an empty 0 state

    S : total spin
    M : projected spin 
    t_vec : Genealogy 
    N: current number of spins, recursive step number. N=1 is base case 
    
    """
    #checks
    assert S >= 0, "Total spin S: {} is not positive".format(S)
    assert S >= abs(M), "Projected spin: {} is larger than total spin S: {}".format(M, S)
    validate_t_vec(t_vec, N)   

    qc = QuantumCircuit(N)

    tN = t_vec[N-1]
    alpha = 1/2
    beta = -1/2

    ### edge cases
    if N == 1:
        # single spin
        if M == 1/2:
            qc.x(0)
        return qc
    
    if S == 0: assert tN == -1/2, "Total spin change tN: {}, but needs to be negative to reach S: {}".format(tN, S)

    coeff_alpha = CG(S, M, tN, alpha)
    coeff_beta = CG(S, M, tN, beta)
    qubits_N_1 = qc.qubits[:-1]
    qubit_N = qc.qubits[-1]

    theta = 2*np.arctan2(coeff_alpha, coeff_beta)

    qc.ry(theta, qubit=qubit_N)

    if coeff_alpha != 0:
        Ua = build_tapered_recursive_CSF_circuit(t_vec, N-1, S - tN, M - alpha)
        qc.append(Ua.to_gate().control(num_ctrl_qubits=1),  [qubit_N]+qubits_N_1)
    
    if coeff_beta != 0:
        Ub = build_tapered_recursive_CSF_circuit(t_vec, N-1, S - tN, M - beta)
        qc.append(Ub.to_gate().control(num_ctrl_qubits=1, ctrl_state=0),  [qubit_N]+qubits_N_1)
        
    return qc

class CSF:
    """
    Class to store CSF object, their corresponding excitation rotations, and retrieve normal and tapered circuits

    t_vec : list[float] - genealogy path, consisting of \pm 1/2 of length N=len(orbitals) (Note S, M = 0 always (siglet))
    orbitals : list[int] - orbitals involved in singlet state
    n_orb : Total number of spatial orbitals
    ne : number of electrons
    excitations: list[PairedExcitationRotation] - 


    
    """
    def __init__(self, t_vec, orbitals, n_orb, ne, excitations : list[PairedExcitationRotation] = []):
        self.n_orb = n_orb
        self.t_vec = t_vec
        self.ne = ne
        self.t_vec = t_vec

        assert self.get_num_targ_orb() == len(orbitals), 'Incorrect number of orbitals in {} for genealogy vector: {}'.format(orbitals, self.t_vec)
        self.orbitals = orbitals
        self.excitations = excitations
    
    def get_excitations(self) ->list[PairedExcitationRotation] :
        return self.excitations
    
    def get_num_targ_orb(self):
        return len(self.t_vec)
    
    def get_tapered_csf_circuit(self, add_hf = True):
        qc = QuantumCircuit(self.n_orb) # 1 for control

        if add_hf:
            qc.append(get_tapered_hf_circuit(self.n_orb, self.ne).to_gate(), qc.qubits)
        
        if self.t_vec == []:
            return qc
        elif self.t_vec == [1/2, -1/2]:
            i, a = self.orbitals

            qc.append(get_tapered_Tia_circuit(i, a, self.n_orb).to_gate(), qc.qubits)
        elif self.t_vec == [1/2, 1/2, -1/2, -1/2]:
            i, j, a, b = self.orbitals

            qc.append(get_tapered_Stt_circuit(i, j, a, b, self.n_orb).to_gate(), qc.qubits)
        elif self.t_vec == [1/2, -1/2, 1/2, -1/2]:
            i, j, a, b = self.orbitals
            assert i < j and a < b, 'i: {} j: {} a: {} b: {}'.format(i, j, a, b)

            qc.append(get_tapered_Sss_circuit(i, j, a, b, self.n_orb).to_gate(), qc.qubits)
        else:
            ### general case
            # redo hf

            N = self.get_num_targ_orb()
            S = 0
            M = 0
            qc.append(build_tapered_recursive_CSF_circuit(self.t_vec, N, S, M).to_gate(), self.orbitals)
        
        return qc
    
    def get_parity_string(self):

        parity = np.zeros(self.n_orb)

        for i in self.orbitals:
            parity[i] = 1

        return parity