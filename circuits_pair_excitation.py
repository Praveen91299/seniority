from __future__ import annotations
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
import numpy as np
from openfermion import QubitOperator
from seniority.utils_circuit import qubit_op_to_sparse_pauli_op

def append_tapered_exc_rot(qc: QuantumCircuit, i, a, theta):
    """
    Append tapered rotation, exp(0.5*i*theta*(XY - YX))

    """

    qc.cx(i, a)

    qc.ry(theta, i)
    qc.cx(a, i)
    qc.ry(-theta, i)
    qc.cx(a, i)

    qc.cx(i, a)

def append_tapered_ctrl_exc_rot(qc: QuantumCircuit, c, i, a, theta):
    """
    Append tapered rotation of theta, controlled on qubit 0

    """

    qc.cx(i, a)

    qc.ry(theta/4, i)
    qc.cx(a, i)
    qc.ry(-theta/4, i)
    qc.cx(c, i)
    qc.ry(theta/4, i)
    qc.cx(a, i)
    qc.ry(-theta/4, i)
    qc.cx(c, i)

    qc.cx(i, a)

def append_tapered_ctrl_exc_rot_comb(qc, c, i, a, theta0, theta1):
    """
    Appends tapered combined rotation for theta0 and theta1 
    conditioned on the control on being |0> and |1> respectively, where qubit 0 is control

    """

    delta = theta1 - theta0
    sigma = theta1 + theta0

    qc.cx(i, a)

    qc.cx(c, i)
    qc.ry(-sigma/4, i)
    qc.cx(a, i)
    qc.ry(delta/4, i)
    qc.cx(c, i)
    qc.ry(-delta/4, i)
    qc.cx(a, i)
    qc.ry(sigma/4, i)
    
    qc.cx(i, a)

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
    qc.ry(theta= -np.sqrt(2)*theta, qubit=b)
    qc.cx(b, i)
    qc.cx(a, b)

    qc.cx(i, a)
    qc.cx(b, a)
    qc.ry(theta= -np.pi/4, qubit=a)
    qc.cx(b, a)
    qc.ry(theta= +np.pi/4, qubit=a)
    qc.cx(a, i)
    qc.cx(i, b)
    qc.ry(theta= -np.sqrt(2)*theta, qubit=i)
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
    qc.ry(theta= theta/np.sqrt(2), qubit=b)
    qc.cx(c, b)
    qc.ry(theta= -theta/np.sqrt(2), qubit=b)
    
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
    qc.ry(theta= theta/np.sqrt(2), qubit=i)
    qc.cx(c, i)
    qc.ry(theta= -theta/np.sqrt(2), qubit=i)
    
    qc.cx(i, b)
    qc.cx(a, i)
    qc.ry(theta= -np.pi/4, qubit=a)
    qc.cx(i, a)
    qc.cx(a, b)

def append_tapered_ctrl_sym_exc_rot_comb(qc, c, i, a, b, n_orb, theta0, theta1):
    """
    Appends combined tapered symmetrized pair excitation rotations
    theta0, theta1 when control (0) is 0, 1 respectively
    
    """

    a0 = -np.sqrt(2)*theta0
    a1 = -np.sqrt(2)*theta1
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
            generators_of = [QubitOperator('X{} Y{}'.format(i, a), 1.0) - QubitOperator('Y{} X{}'.format(i, a), 1.0)] # commuting
            generators = [qubit_op_to_sparse_pauli_op(gen, self.n_orb) for gen in generators_of]

        return generators
    
    def get_PauliEvolutionGate(self):
        """
        Returns qiskit.circuit.library.PauliEvolutionGate object 
        
        exp(i*0.5*theta*G)
        
        """
        generators = self.get_generators()
        time = -0.5*self.theta

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
        self.theta = theta

        self.check_excitation(excitations, self.n_orb)
        self.excitations = excitations

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
        
        if taper:
            i, a, b = self.get_indices()
            generators_of = [QubitOperator('X{} Y{}'.format(i, a), 1.0) - QubitOperator('Y{} X{}'.format(i, a), 1.0),
                             QubitOperator('X{} Y{}'.format(i, b), 1.0) - QubitOperator('Y{} X{}'.format(i, b), 1.0)] # commuting
            generators = [qubit_op_to_sparse_pauli_op(gen, self.n_orb) for gen in generators_of]

        return generators

    @classmethod
    def check_excitation(cls, excitations, n_orb):

        assert len(excitations) == 2, "Not a symmetric excitation description"
        assert len(excitations[0]) == 2 and len(excitations[1]) == 2, "Wrong number of indices for excitations"
        assert excitations[0][0] < n_orb and excitations[0][1] < n_orb and excitations[1][0] < n_orb and excitations[1][1] < n_orb, "Orbital index exceeds available orbitals"
        assert len(set.union(set(excitations[0]), set(excitations[1]))) == 3, "Symmetric excitation invalid"
    
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