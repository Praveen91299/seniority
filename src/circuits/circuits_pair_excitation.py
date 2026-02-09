from __future__ import annotations
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator
import numpy as np
from openfermion import QubitOperator, FermionOperator, jordan_wigner, get_sparse_operator, taper_off_qubits
from .utils_circuit import qubit_op_to_sparse_pauli_op, show_operator_matrix, check_equivalent_operators
import scipy.sparse as sp

Tia = lambda i, a: FermionOperator('{}^ {}^ {} {}'.format(2*a, 2*a + 1, 2*i + 1, 2*i), 1.0) - FermionOperator('{}^ {}^ {} {}'.format(2*i, 2*i + 1, 2*a + 1, 2*a), 1.0)
Tia_tap = lambda i, a: 1.j/2 * (QubitOperator(f'Y{i} X{a}', 1.0) - QubitOperator(f'X{i} Y{a}', 1.0)) # TAP[a_{2a}^a_{2a+1}^a_{2i + 1}a_{2i} - h.c]

def append_tapered_exc_rot(qc: QuantumCircuit, i, a, theta):
    """
    Append tapered rotation, exp(0.5*i*theta*(XaYi - YaXi))

    """

    qc.rz(np.pi/2, i)
    qc.rx(np.pi/2, i)
    qc.rx(np.pi/2, a)
    qc.cx(i, a)
    qc.rx(theta, i)
    qc.rz(theta, a)
    qc.cx(i, a)
    qc.rx(-np.pi/2, i)
    qc.rx(-np.pi/2, a)
    qc.rz(-np.pi/2, i)

def append_tapered_ctrl_exc_rot(qc: QuantumCircuit, c, i, a, theta):
    """
    Append tapered rotation of theta, controlled on qubit 0

    """

    qc.rz(np.pi/2, a)
    qc.ry(-np.pi/2, i)
    qc.rz(-np.pi/2, i)
    qc.cx(a, i)
    qc.rz(np.pi/2, i)
    qc.ry(np.pi/2, i)

    qc.ry(-theta/2, a)
    qc.cz(c, a)
    qc.ry(theta/2, a)
    qc.cz(a, i)
    qc.ry(-theta/2, a)
    qc.cz(c, a)
    qc.ry(theta/2, a)

    qc.cx(a, i)

def append_tapered_ctrl_exc_rot_comb(qc, c, i, a, theta0, theta1):
    """
    Appends tapered combined rotation for theta0 and theta1 
    conditioned on the control on being |0> and |1> respectively, where qubit 0 is control

    """

    delta = theta1 - theta0
    sigma = theta1 + theta0

    qc.rz(np.pi/2, a)
    qc.ry(-np.pi/2, i)
    qc.rz(-np.pi/2, i)
    qc.cx(a, i)
    qc.rz(np.pi/2, i)
    qc.ry(np.pi/2, i)

    qc.ry(-sigma/2, a)
    qc.cz(c, a)
    qc.ry(delta/2, a)
    qc.cz(a, i)
    qc.ry(-delta/2, a)
    qc.cz(c, a)
    qc.ry(sigma/2, a)

    qc.cx(a, i)

### symmetric pair excitations

def append_tapered_sym_exc_rot(qc: QuantumCircuit, i, a, b, theta):
    """
    Appends tapered symmetrized pair excitation onto i, a, b
    
    """

    qc.cx(b, a)
    qc.cx(i, b)
    qc.ry(theta = np.pi/4, qubit=b)
    qc.cx(i, b)
    qc.cx(b, a)

    qc.cx(b, i)
    qc.ry(theta= np.sqrt(2)*theta, qubit=b)
    qc.cx(b, i)
    qc.cx(a, b)

    qc.cx(i, a)
    qc.cx(b, a)
    qc.ry(theta= -np.pi/4, qubit=a)
    qc.cx(b, a)
    qc.ry(theta= +np.pi/4, qubit=a)
    qc.cx(a, i)
    qc.cx(i, b)
    qc.ry(theta= np.sqrt(2)*theta, qubit=i)
    qc.cx(i, b)
    qc.cx(a, i)
    qc.ry(theta= -np.pi/4, qubit=a)
    qc.cx(i, a)
    qc.cx(a, b)

def append_tapered_ctrl_sym_exc_rot(qc: QuantumCircuit, c, i, a, b, theta):
    """
    Append controlled tapered symmetrized excitation rotations

    """
    
    qc.cx(b, a)
    qc.cx(i, b)
    qc.ry(theta = np.pi/4, qubit=b)
    qc.cx(i, b)
    qc.cx(b, a)

    qc.cx(b, i)

    qc.cx(c, b)
    qc.ry(theta= -theta/np.sqrt(2), qubit=b)
    qc.cx(c, b)
    qc.ry(theta= theta/np.sqrt(2), qubit=b)
    
    qc.cx(b, i)
    qc.cx(a, b)

    qc.cx(i, a)

    qc.cx(b, a)
    qc.ry(theta= -np.pi/4, qubit=a)
    qc.cx(b, a)
    qc.ry(theta= +np.pi/4, qubit=a)

    qc.cx(a, i)
    qc.cx(i, b)
    
    qc.cx(c, i)
    qc.ry(theta= -theta/np.sqrt(2), qubit=i)
    qc.cx(c, i)
    qc.ry(theta= theta/np.sqrt(2), qubit=i)
    
    qc.cx(i, b)
    qc.cx(a, i)
    qc.ry(theta= -np.pi/4, qubit=a)
    qc.cx(i, a)
    qc.cx(a, b)

def append_tapered_ctrl_sym_exc_rot_comb(qc, c, i, a, b, theta0, theta1):
    """
    Appends combined tapered symmetrized pair excitation rotations
    theta0, theta1 when control (0) is 0, 1 respectively
    
    """

    a0 = np.sqrt(2)*theta0
    a1 = np.sqrt(2)*theta1
    delta = (a1 - a0)
    sigma = (a1 + a0)
    
    qc.cx(b, a)
    qc.cx(i, b)
    qc.ry(theta = np.pi/4, qubit=b)
    qc.cx(i, b)
    qc.cx(b, a)
    qc.cx(b, i)

    qc.ry(theta= sigma/2, qubit=b)
    qc.cx(c, b)
    qc.ry(theta= -delta/2, qubit=b)
    qc.cx(c, b)
    
    
    qc.cx(b, i)
    qc.cx(a, b)

    qc.cx(i, a)

    qc.cx(b, a)
    qc.ry(theta= -np.pi/4, qubit=a)
    qc.cx(b, a)
    qc.ry(theta= +np.pi/4, qubit=a)

    qc.cx(a, i)
    qc.cx(i, b)
    
    qc.ry(theta= sigma/2, qubit=i)
    qc.cx(c, i)
    qc.ry(theta= -delta/2, qubit=i)
    qc.cx(c, i)
    
    qc.cx(i, b)
    qc.cx(a, i)
    qc.ry(theta= -np.pi/4, qubit=a)
    qc.cx(i, a)
    qc.cx(a, b)

def check_pair_excitations():
    """
    Check construction of all pair excitation circuits
    
    """
    # tapered excitation generator check
    tapered_exact = taper_off_qubits(jordan_wigner(Tia(0, 1)), [QubitOperator('Z0 Z1'), QubitOperator('Z2 Z3')], manual_input=True, fixed_positions=[1, 3], output_tapered_positions=True)
    ctrl = lambda c, state: 0.5*(1 + QubitOperator(f'Z{c}', (-1)**state))

    orb_i, orb_a, orb_b = 0, 1, 2

    assert Tia_tap(0, 1) == tapered_exact[0]

    # tapered pair excitation
    theta = np.random.rand()
    
    i, a = orb_i, orb_a
    U_exact = sp.linalg.expm(theta * get_sparse_operator(Tia_tap(i, a), 2))
    qc = QuantumCircuit(2)
    append_tapered_exc_rot(qc, i, a, theta)

    assert check_equivalent_operators(U_exact, show_operator_matrix(qc))

    PER = PairedExcitationRotation([[orb_i, orb_a]], theta, 2)
    qc= QuantumCircuit(2)
    PER.append_tapered_circuit(qc, qc.qubits)

    assert check_equivalent_operators(U_exact, show_operator_matrix(qc))

    m = Operator(PER.get_PauliEvolutionGate(True)).reverse_qargs().to_matrix()
    qc = QuantumCircuit(2)

    PER.append_tapered_circuit(qc, [i, a])

    assert check_equivalent_operators(m, show_operator_matrix(qc))
    
    # controlled tapered pair excitation
    theta = np.random.rand()
    
    c, i, a = 0, orb_i + 1, orb_a + 1
    U_exact = sp.linalg.expm(theta * get_sparse_operator(ctrl(c, 1) * Tia_tap(i, a), 3))
    qc = QuantumCircuit(3)
    append_tapered_ctrl_exc_rot(qc, c, i, a, theta)
    U = show_operator_matrix(qc)

    assert check_equivalent_operators(U_exact, U)

    PER = PairedExcitationRotation([[orb_i, orb_a]], theta, 2) # -1 due to c = 0
    qc= QuantumCircuit(3)
    PER.append_controlled_tapered_circuit(qc, c, [i, a])

    assert check_equivalent_operators(U_exact, show_operator_matrix(qc))

    # combined controlled tapered pair excitations
    theta0, theta1 = np.random.rand(), np.random.rand()

    c, i, a = 0, orb_i + 1, orb_a + 1
    U_exact = sp.linalg.expm(theta0 * get_sparse_operator(ctrl(c, 0) * Tia_tap(i, a), 3)) @ sp.linalg.expm(theta1 * get_sparse_operator(ctrl(c, 1) * Tia_tap(i, a), 3))
    qc = QuantumCircuit(3)
    append_tapered_ctrl_exc_rot_comb(qc, c, i, a, theta0, theta1)
    U = show_operator_matrix(qc)

    assert check_equivalent_operators(U_exact, U)

    PER0 = PairedExcitationRotation([[orb_i, orb_a]], theta0, 2) # -1 due to c = 0
    PER1 = PairedExcitationRotation([[orb_i, orb_a]], theta1, 2) # -1 due to c = 0
    qc= QuantumCircuit(3)
    PER0.append_combined_controlled_tapered_circuit(PER1, qc, c, [i, a], 0)

    assert check_equivalent_operators(U_exact, show_operator_matrix(qc))

    # tapered symmetrized pair excitation
    theta = np.random.rand()

    i, a, b = orb_i, orb_a, orb_b
    U_exact = sp.linalg.expm(theta*get_sparse_operator(Tia_tap(i, a) + Tia_tap(i, b), 3))
    qc= QuantumCircuit(3)
    append_tapered_sym_exc_rot(qc, i, a, b, theta)
    U = show_operator_matrix(qc)
    
    assert check_equivalent_operators(U_exact, U)

    SPER = SymmetricPairedExcitationRotation([[orb_i, orb_a], [orb_i, orb_b]], theta, 3)
    qc = QuantumCircuit(3)
    SPER.append_tapered_circuit(qc, [i, a, b])

    assert check_equivalent_operators(U_exact, show_operator_matrix(qc))
    
    # controlled tapered symmetrized pair excitation
    theta = np.random.rand()

    c, i, a, b = 0, 1, 2, 3
    U_exact = sp.linalg.expm(theta*get_sparse_operator(ctrl(c, 1)*(Tia_tap(i, a) + Tia_tap(i, b)), 4))
    qc = QuantumCircuit(4)
    append_tapered_ctrl_sym_exc_rot(qc, c, i, a, b, theta)
    U = show_operator_matrix(qc)

    assert check_equivalent_operators(U_exact, U)

    SPER = SymmetricPairedExcitationRotation([[orb_i, orb_a], [orb_i, orb_b]], theta, 3)
    qc = QuantumCircuit(4)
    SPER.append_controlled_tapered_circuit(qc, c, [i, a, b])

    assert check_equivalent_operators(U_exact, show_operator_matrix(qc))

    # combined controlled tapered symmetrized pair excitation
    theta0, theta1 = np.random.rand(), np.random.rand()

    c, i, a, b = 0, 1, 2, 3
    U_exact = sp.linalg.expm(theta0 * get_sparse_operator(ctrl(c, 0) * (Tia_tap(i, a) + Tia_tap(i, b)), 4)) @ sp.linalg.expm(theta1 * get_sparse_operator(ctrl(c, 1) * (Tia_tap(i, a) + Tia_tap(i, b)), 4))
    qc = QuantumCircuit(4)
    append_tapered_ctrl_sym_exc_rot_comb(qc, c, i, a, b, theta0, theta1)
    U = show_operator_matrix(qc)

    assert check_equivalent_operators(U_exact, U)

    SPER0 = SymmetricPairedExcitationRotation([[orb_i, orb_a], [orb_i, orb_b]], theta0, 3)
    SPER1 = SymmetricPairedExcitationRotation([[orb_i, orb_a], [orb_i, orb_b]], theta1, 3)
    qc = QuantumCircuit(4)
    SPER0.append_combined_controlled_tapered_circuit(SPER1, qc, c, [i, a, b], 0)

    assert check_equivalent_operators(U_exact, show_operator_matrix(qc))

    return True

class PairedExcitationRotation:
    """
    Class to store paired excitations

    excitation list[list[int]] : eq: [[0, 1]]
    
    """

    def __init__(self, excitations, theta, n_orb):
        self.n_orb = n_orb
        self.theta = theta

        self.check_excitation(excitations=excitations, n_orb=n_orb)
        self.excitations = excitations
    
    @classmethod
    def init_from_excitations_list(cls, excitations, n_orb, theta_init = None):

        if theta_init is None:
            theta_init = np.zeros(len(excitations))
        
        assert len(theta_init) == len(excitations)
        return [cls(exc, theta, n_orb) for exc, theta in zip(excitations, theta_init)]

    @classmethod
    def check_excitation(cls, excitations, n_orb):
        """
        Check the passed excitations are valid - contain single paired excitations, or symmetrized only

        """
        
        assert len(excitations) == 1, "Wrong number of excitations for single Paired Excitations"
        assert len(excitations[0]) == 2, "Wrong number of indices for excitations"
        assert excitations[0][0] < n_orb and excitations[0][1] < n_orb, "Orbital index exceeds available orbitals"
    
    def get_generators(self, taper = True):
        """
        Returns SparsePauli generator list of the unitary, with entries of the list containing commuting Pauli products

        """

        if taper:
            i, a = self.get_indices()
            generators_of = [-1.j * Tia_tap(i, a)] # commuting
            generators = [qubit_op_to_sparse_pauli_op(gen, self.n_orb) for gen in generators_of]
        else:
            i, a = self.get_indices()
            generators_of = [-1.j * Tia(i, a)]
            generators = [qubit_op_to_sparse_pauli_op(gen, 2*self.n_orb) for gen in generators_of]

        return generators
    
    def get_PauliEvolutionGate(self, taper = True):
        """
        Returns qiskit.circuit.library.PauliEvolutionGate object 
        
        exp(i*0.5*theta*G)
        
        """
        generators = self.get_generators(taper)
        time = - self.get_theta()

        return PauliEvolutionGate(operator=generators, time=time)
    
    def get_excitations(self):
        return self.excitations
    
    def get_indices(self):
        return self.get_excitations()[0]
    
    def get_theta(self):
        return self.theta
    
    def append_tapered_circuit(self, qc: QuantumCircuit, target_qubits):
        """
        Append tapered circuit

        """
        assert len(target_qubits) == self.n_orb

        i, a = self.get_indices()
        append_tapered_exc_rot(qc, target_qubits[i], target_qubits[a], self.get_theta())

    
    def append_controlled_tapered_circuit(self, qc : QuantumCircuit, control_qubit, target_qubits):
        """
        Return controlled tapered circuit of excitation

        """
        assert len(target_qubits) == self.n_orb

        i, a = self.get_indices()
        append_tapered_ctrl_exc_rot(qc, control_qubit, target_qubits[i], target_qubits[a], self.get_theta())
    
    def append_combined_controlled_tapered_circuit(self, other: PairedExcitationRotation, qc : QuantumCircuit, control_qubit, target_qubits, control_self_on = 1):
        """
        Append 
        
        """
        assert len(target_qubits) == self.n_orb

        #check compatibility
        assert (self.get_excitations() == other.get_excitations()) and self.n_orb == other.n_orb, "Incompatible excitations"

        i, a = self.get_indices()

        if control_self_on:
            theta1 = self.get_theta()
            theta0 = other.get_theta()
        else:
            theta1 = other.get_theta()
            theta0 = self.get_theta()

        append_tapered_ctrl_exc_rot_comb(qc, control_qubit, target_qubits[i], target_qubits[a], theta0, theta1)

class SymmetricPairedExcitationRotation(PairedExcitationRotation):
    def __init__(self, excitations, theta, n_orb):
        self.n_orb = n_orb
        self.check_excitation(excitations, self.n_orb)
        self.excitations = excitations

        self.check_theta_negation()
        self.theta = theta

    def get_common_index(self):
        return set.intersection(set(self.excitations[0]), set(self.excitations[1])).pop()
    
    def get_disjoint_indices(self):
        exc = self.get_excitations()
        return list(set.symmetric_difference(set(exc[0]), set(exc[1])))
    
    def get_indices(self):
        """
        Returns excitation indices in order (i, a, b) where i is the common index

        """
        return [self.get_common_index()] + self.get_disjoint_indices()
    
    def get_generators(self, taper=True):
        """
        Returns generators of the symmetrized pair excitation, grouped into commuting terms
        """
        i, a, b = self.get_indices()

        if taper:    
            generators_of = [-1.j * Tia_tap(i, a),
                             -1.j * Tia_tap(i, b)] # commuting
            generators = [qubit_op_to_sparse_pauli_op(gen, self.n_orb) for gen in generators_of]
        else:
            generators_of = [-1.j * Tia(i, a),
                             -1.j * Tia(i, b)] # commuting
            generators = [qubit_op_to_sparse_pauli_op(gen, 2*self.n_orb) for gen in generators_of]

        return generators
    
    def get_theta(self):
        if self._negate_theta:
            return -self.theta
        else:
            return self.theta

    @classmethod
    def check_excitation(cls, excitations, n_orb):

        assert len(excitations) == 2, "Not a symmetric excitation description"
        assert len(excitations[0]) == 2 and len(excitations[1]) == 2, "Wrong number of indices for excitations"
        assert excitations[0][0] < n_orb and excitations[0][1] < n_orb and excitations[1][0] < n_orb and excitations[1][1] < n_orb, "Orbital index exceeds available orbitals"
        assert len(set.union(set(excitations[0]), set(excitations[1]))) == 3, "Symmetric excitation invalid"

    def check_theta_negation(self):
        if self.excitations[0][1] == self.excitations[1][1]:
            self._negate_theta = True
        else:
            self._negate_theta = False
    
    def append_tapered_circuit(self, qc: QuantumCircuit, target_qubits):
        """
        Append tapered circuit

        """
        assert len(target_qubits) == self.n_orb

        i = self.get_common_index()
        a, b = self.get_disjoint_indices()
        
        append_tapered_sym_exc_rot(qc, target_qubits[i], target_qubits[a], target_qubits[b], self.get_theta())

    
    def append_controlled_tapered_circuit(self, qc : QuantumCircuit, control_qubit, target_qubits):
        """
        Return controlled tapered circuit of excitation

        """
        assert len(target_qubits) == self.n_orb

        i = self.get_common_index()
        a, b = self.get_disjoint_indices()

        append_tapered_ctrl_sym_exc_rot(qc, control_qubit, target_qubits[i], target_qubits[a], target_qubits[b], self.get_theta())
    
    def append_combined_controlled_tapered_circuit(self, other: PairedExcitationRotation, qc : QuantumCircuit, control_qubit, target_qubits, control_self_on = 1):
        """
        Append 
        
        """
        assert len(target_qubits) == self.n_orb

        #check compatibility
        assert (self.get_excitations() == other.get_excitations()) and self.n_orb == other.n_orb, "Incompatible excitations"

        i = self.get_common_index()
        a, b = self.get_disjoint_indices()

        if control_self_on:
            theta1 = self.get_theta()
            theta0 = other.get_theta()
        else:
            theta1 = other.get_theta()
            theta0 = self.get_theta()

        append_tapered_ctrl_sym_exc_rot_comb(qc, control_qubit, target_qubits[i], target_qubits[a], target_qubits[b], theta0, theta1)

def init_exc_list(excitations, n_orb, thetas = None):
    """
    Initialize a list of excitations with the corresponding classes

    
    """
    if thetas is None:
        thetas == np.zeros(len(excitations))

    assert len(excitations) == len(thetas)

    U = []
    for exc, theta in zip(excitations, thetas):
        if len(exc) == 1:
            U.append(PairedExcitationRotation(exc, theta, n_orb=n_orb))
        if len(exc) == 2:
            U.append(SymmetricPairedExcitationRotation(exc, theta, n_orb=n_orb))
    
    return U

def filter_parity_U(U: PairedExcitationRotation, orb_parities: list):
    """
    Retain paired excitation rotation in U that have non-zero support in the even parity orbitals in atleast one of orb_parities
    
    """
    
    exc_list = U.get_excitations()
    exc_list_new = []

    for exc in exc_list:
        include = False
        for parity in orb_parities:
            if parity[exc[0]] == 0 and parity[exc[1]] == 0:
                include=True
        
        if include: exc_list_new.append(exc)

    return U.__init__(exc_list_new, U.get_theta(), U.n_orb)

def filter_parity_U_list(U_list: list[PairedExcitationRotation], orb_parities: list):
    """
    
    """
    U_fil_list = []
    for U in U_list:
        U_fil = filter_parity_U(U, orb_parities)
        if len(U_fil.excitations) > 0:
            U_fil_list.append(U_fil)

    return  U_fil_list