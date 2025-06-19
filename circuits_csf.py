import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from seniority.circuits_pair_excitation import PairedExcitationRotation, SymmetricPairedExcitationRotation
from scipy import sparse as sp
from openfermion import s_squared_operator, get_sparse_operator
import pickle


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

def append_tapered_Tia_circuit(qc, i, a):
    """
    Append circuit to prepare 0.707*[|01> - |10>]

    """

    qc.x(i)
    qc.x(a)
    qc.h(a)
    qc.cx(control_qubit=a, target_qubit=i)

    return qc

def append_tapered_Sss_circuit(qc, i, j, a, b):

    append_tapered_Tia_circuit(qc, i, a)
    append_tapered_Tia_circuit(qc, j, b)

def append_tapered_Stt_circuit(qc, i, j, a, b):

    # takes care of hf state
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

def append_x_occ(qc, occ):
    assert len(qc.qubits) >= len(occ), "Insufficient number of qubits."

    for idx, i in enumerate(occ):
        if i == 1:
            qc.x(idx)

def get_state_from_csf_data(csf_state, orbitals):
    spin_orbs = []
    for o in orbitals:
        spin_orbs.append(2*o)
        spin_orbs.append(2*o + 1)
    
    rows = []

    get_int = lambda sd: np.sum([val * (2**(len(sd) - 1 - i)) for i, val in enumerate(sd)])

    for SD in csf_state[0]:
        trunc_SD = SD[spin_orbs]
        rows.append(get_int(trunc_SD))
    
    arr = sp.csr_matrix((csf_state[2], (rows, [0]*len(rows))), shape=(2**len(spin_orbs), 1), dtype=complex)

    return arr

def get_t_vec(state, n_orb):
    """
    Get t vector from state by taking difference of successive S values

    """
    get_s = lambda ss: -0.5 + 0.5*np.sqrt(1 + 4*ss)

    s_list = [get_s((state.T @ (get_sparse_operator(s_squared_operator(i), 2*n_orb)) @ state).toarray()[0, 0]) for i in range(n_orb+1)]
    t_vec = np.array([s_list[i+1] - s_list[i] for i in range(n_orb)])
    t_vec = np.array(np.rint(2*t_vec), int)
    t_vec = t_vec/2
    return list(t_vec)

def determine_t_vec(csf_state, orbitals):
    """
    Determine the genealogy vector from the csf state [SD vectors, int of SD, coeff], over specified orbitals

    csf_state: list[list[array], list[int], array] - list of SDs and coefficient array for the CSF
    orbitals: list[int] - singly occupied orbitals involved in singlet creation (SOMOs)
    
    """

    arr = get_state_from_csf_data(csf_state=csf_state, orbitals=orbitals)
    return get_t_vec(arr, len(orbitals))

def get_csfs_from_dump(input_file):

    with open(input_file,'rb') as f:
        list_CSF,list_list_ia_CSF,list_list_theta_CSF,list_sym_CSF_vec,list_UCSF_tz,list_UCSF_smik,\
        list_list_SOMO_UCSF_smik,psi_GS_UCSF_smik,list_orb_rot,x_orbrot,Enuc,obt_spatial,tbt_spatial = pickle.load(f)
    
    n_orb=len(tbt_spatial)
    print(n_orb)
    ne = sum(list_CSF[0][0][0])
    csfs = []

    for iCSF in range(len(list_CSF)):
        list_ia = list_list_ia_CSF[iCSF]
        orbitals = list_list_SOMO_UCSF_smik[iCSF]

        if len(list_ia) == 0:
            
            excitations = []
        else:
            list_theta = list_list_theta_CSF[iCSF]
            n_U_train = len(list_theta) // len(list_ia)

            excitations = []

            for i_train in range(n_U_train):

                for ipair in range(len(list_ia)):

                    exc = list_ia[ipair]
                    theta = list_theta[ipair+i_train*len(list_ia)]
                    
                    if len(exc) == 1:
                        excitations.append(PairedExcitationRotation(exc, -theta, n_orb)) ### note the -ve
                    elif len(exc) == 2:
                        if len(set.intersection(set(exc[0]), set(exc[1]))) == 1:
                            excitations.append(SymmetricPairedExcitationRotation(exc, -theta, n_orb))
                        else:
                            excitations.append(PairedExcitationRotation([exc[0]], -theta, n_orb))
                            excitations.append(PairedExcitationRotation([exc[1]], -theta, n_orb))
        
        csf = CSF(determine_t_vec(list_CSF[iCSF], orbitals), orbitals=orbitals, n_orb=n_orb, ne = ne, excitations=excitations)
        csfs.append(csf)
    
    return csfs

class CSF:
    """
    Class to store CSF object, their corresponding excitation rotations, and retrieve normal and tapered circuits

    t_vec : list[float] - genealogy path, consisting of \pm 1/2 of length N=len(orbitals) (Note S, M = 0 always (siglet))
    orbitals : list[int] - orbitals involved in singlet state (unpaired orbitals)
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
    
    def get_tapered_csf_circuit(self, add_double_occ = True)->QuantumCircuit:
        qc = QuantumCircuit(self.n_orb) # 1 for control

        if add_double_occ:
            occ = self.get_doubly_occ_orbitals()
            append_x_occ(qc, occ)
        
        if self.t_vec == []:
            return qc
        elif self.t_vec == [1/2, -1/2]:
            i, a = self.orbitals

            append_tapered_Tia_circuit(qc, i, a)
        elif self.t_vec == [1/2, 1/2, -1/2, -1/2]:
            i, j, a, b = self.orbitals

            append_tapered_Stt_circuit(qc, i, j, a, b)
        elif self.t_vec == [1/2, -1/2, 1/2, -1/2]:
            i, j, a, b = self.orbitals
            assert i < j and a < b, 'i: {} j: {} a: {} b: {}'.format(i, j, a, b)

            append_tapered_Sss_circuit(qc, i, j, a, b)
        else:
            ### general case
            # redo hf

            N = self.get_num_targ_orb()
            S = 0
            M = 0
            qc.append(build_tapered_recursive_CSF_circuit(self.t_vec, N, S, M).to_gate(), self.orbitals)
        
        return qc
    
    def get_doubly_occ_orbitals(self):
        """
        Return idx of doubly occupied spatial orbitals

        """
        excess_elec = self.ne - len(self.orbitals)
        to_add = np.ceil(excess_elec/2)

        d_occ = np.array([0]*self.n_orb)

        if excess_elec == 0:
            return d_occ

        added = 0
        for i in range(self.n_orb):

            if i not in self.orbitals:
                d_occ[i] = 1
                added += 1
            
            if added == to_add:
                return d_occ
        
        raise ValueError('Insufficient orbitals {} for {} electrons and {} singly occupied orbitals'.format(self.n_orb, self.ne, len(self.orbitals)))

    def get_tapered_full_circuit(self):
        qc = self.get_tapered_csf_circuit(True)
        for exc in self.excitations:
            exc.append_tapered_circuit(qc, qc.qubits)
        
        return qc
    
    def get_parity_string(self):

        parity = np.zeros(self.n_orb)

        for i in self.orbitals:
            parity[i] = 1

        return parity