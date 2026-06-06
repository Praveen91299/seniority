import numpy as np
from qiskit import QuantumCircuit
from .circuits.circuits_csf import CSF
from .circuits.utils_circuit import show_state, get_sparse_state
from openfermion import QubitOperator, count_qubits
import networkx as nx

#utils

def split_measurements(circuit: QuantumCircuit):
    """
    Split the measurements in the circuit into a list (to add back later)

    returns new circuit and list of measurements

    """
    new_circuit = QuantumCircuit(circuit.qubits, circuit.clbits)
    measurements = []

    for instr, qargs, cargs in circuit.data:
        if instr.name == "measure":
            measurements.append((instr, qargs, cargs))
        else:
            new_circuit.append(instr, qargs, cargs)

    return new_circuit, measurements

def append_measurements(circuit, measurements):
    """
    Appends measurements to the circuit
    """

    for instr, qargs, cargs in measurements:
        circuit.append(instr, qargs, cargs)

#SV for parity 

class ParityMitigator:
    """
    Class for symmetry verification of parity for CSFs normal

    Adds parity counting circuit consisting of CNOTs to an ancilla
    """

    def __init__(self, parity):
        self.parity = parity
    
    def append_parity_circuit(self, qc, parity_qubit, target_qubits):
        
        for target in target_qubits:
            qc.cx(target, parity_qubit)
    
    def mitigate(self, result_dict: dict, parity_qubit, silent=False, remove_parity_qubit=False):
        def remove_bit(bit_str, idx):
            char_list = list(bit_str)
            char_list.pop(idx)
            return "".join(char_list)

        if not silent: print('\nMitigation: Symmetry verification.')
        
        filtered_dict = {}
        total_shots = 0
        filtered_shots = 0

        for bit_str, v in zip(result_dict.keys(), result_dict.values()):
            total_shots += v

            if int(bit_str[parity_qubit]) == self.parity:
                if remove_parity_qubit:
                    filtered_dict[remove_bit(bit_str, parity_qubit)] = v
                else:
                    filtered_dict[bit_str] = v

                filtered_shots += v

        if not silent: print(f'SV: {filtered_shots} retained out of {total_shots} shots')

        if total_shots == 0:
            return filtered_dict, 1
        return filtered_dict, filtered_shots/total_shots
    
    def estimate_overhead(self):
        #TODO
        return
    

class ExtParityMitigator(ParityMitigator):
    """
    Class for parity mitigator for extended swap test circuits

    Adds parity counting circuit consisting of CNOTs to an ancilla, and an extra CNOT for the control qubit when parity_0 ^ parity_1 == 1

    Defaults to checking for parity_0 (converts parity_1 to parity_0 in circuit with a CNOT from ctrl qubit)

    """

    def __init__(self, parity_0, parity_1):
        self.parity_0, self.parity_1 = parity_0, parity_1
    
    def append_parity_circuit(self, qc, parity_qubit, target_qubits, control_qubit):
        
        for target in target_qubits:
            qc.cx(target,  parity_qubit)
        
        if self.parity_0 != self.parity_1:
            qc.cx(control_qubit, parity_qubit)
    
    def mitigate(self, result_dict: dict, parity_qubit, silent=False):
        
        if not silent: print('\nMitigation: Symmetry verification.')

        filtered_dict = {}
        total_shots = 0
        filtered_shots = 0

        for bit_str, v in zip(result_dict.keys(), result_dict.values()):
            total_shots += v

            if int(bit_str[parity_qubit]) == self.parity_0:
                filtered_dict[bit_str] = v

                filtered_shots += v

        if not silent: print(f'SV: {filtered_shots} retained out of {total_shots} shots')

        return filtered_dict, filtered_shots/total_shots


def determine_tapered_parity(csf: CSF, quantum_qubits):
    """
    Determines parity of CSF over quantum_qubits
    """

    so_overlap = []

    for i in csf.orbitals:
        if i in quantum_qubits:
            so_overlap.append(i)
    
    n = len(so_overlap)//2
    do =  csf.get_doubly_occ_orbitals()
    for i in quantum_qubits:
        if do[i] == 1:
            n+=1

    return n % 2

def append_tapered_sep_parity_circuit_offdiag(qc: QuantumCircuit, csf0: CSF, csf1: CSF, control_qubit, state_register, parity_qubit0, parity_qubit1, quantum_qubits):
    """
    Appends CNOT network to check parity, individually for both csfs
    
    """
    assert qc.num_qubits >= len(state_register) + 3, "Insufficient number of qubits in circuit"
    assert 2*len(quantum_qubits) == len(state_register), "Incompatible quantum index set {} and state register".format(quantum_qubits)
    
    p0 = determine_tapered_parity(csf0, quantum_qubits=quantum_qubits)
    p1 = determine_tapered_parity(csf1, quantum_qubits=quantum_qubits)

    #controlled
    qc.x(control_qubit)
    for qubit in state_register:
        qc.ccx(qubit, control_qubit, parity_qubit0)
    qc.x(control_qubit)

    for qubit in state_register:
        qc.ccx(qubit, control_qubit, parity_qubit1)

    #makes target parity qubits to 0 (chosen convention)
    if p0 == 1:
        qc.x(parity_qubit0)
    
    if p1 == 1:
        qc.x(parity_qubit1)

def append_tapered_parity_circuit_offdiag(qc: QuantumCircuit, csf0: CSF, csf1: CSF, control_qubit, state_register, parity_qubit, quantum_qubits):
    """
    Appends CNOT network to check parity of bitstrings, such that correct state results in parity qubit in 0
    
    """

    assert qc.num_qubits >= len(state_register) + 2, "Insufficient number of qubits in circuit"
    assert 2*len(quantum_qubits) == len(state_register), "Incompatible quantum index set {} and state register".format(quantum_qubits)

    #get parities
    p0 = determine_tapered_parity(csf0, quantum_qubits=quantum_qubits)
    p1 = determine_tapered_parity(csf1, quantum_qubits=quantum_qubits)

    #if diff, add CNOT to control qubit
    for qubit in state_register:
        qc.cx(qubit, parity_qubit)

    #makes parity to be same
    if p0 != p1:
        qc.cx(control_qubit, parity_qubit)

    #makes target parity qubit to 0
    if p0 == 1:
        qc.x(parity_qubit)

def append_tapered_parity_circuit_diag(qc: QuantumCircuit, csf, state_register, parity_qubit, quantum_qubits):
    """
    Appends CNOT network to check parity of bitstrings, such that correct state results in parity qubit in 0
    
    """

    assert qc.num_qubits >= len(state_register) + 1, "Insufficient number of qubits in circuit"
    assert len(quantum_qubits) == len(state_register), "Incompatible quantum index set {} and state register".format(quantum_qubits)

    #get parities
    p = determine_tapered_parity(csf, quantum_qubits=quantum_qubits)

    #if diff, add CNOT to control qubit
    for qubit in state_register:
        qc.cx(qubit, parity_qubit)
    
    #makes target parity qubit to 0
    if p == 1:
        qc.x(parity_qubit)

#constructing symmetry operators and projectors
def construct_z2_projector(sym_list: list[QubitOperator], eig_vals: list[int] = None):
    """
    Construct projectors to z2 symmetry subspace defined by sym_list, eig_vals

    """
    if eig_vals is None:
        eig_vals = np.array([1]*len(sym_list))

    assert len(sym_list) == len(eig_vals), "Incorrect number of symmetries and eigenvalues specified!"
    if len(sym_list) == 0:
        return QubitOperator('', coefficient=1.0)

    sym, e = sym_list[0], eig_vals[0]
    projector = construct_z2_projector(sym_list=sym_list[1:], eig_vals=eig_vals[1:])
    
    return projector * (0.5 + 0.5 * e * sym)

def get_extended_symmetry(sym: QubitOperator, eig_val, ctrl_state, ctrl_qubit=0) -> QubitOperator:
    """
    Symmetry induced by sym on the extended state, of eigenvalue eig_val 
    ctrl_state \in \{0, 1\} determines which of the defining states the sym is a symmetry of

    """
    def insert_qubit_op_at_pos(op, op_insert, pos:int, inserted_nqubits=None):
        """
        Inserts op_insert at pos in op (as in tensor product)
        
        """
        

        if inserted_nqubits is None:
            inserted_nqubits = count_qubits(op_insert)
        assert inserted_nqubits >= count_qubits(op_insert), "Invalid number of qubits to be inserted!"

        shift_first_entry = lambda l, k: [(l0+k, l1) for l0, l1 in l]

        op_new = QubitOperator()
        for term, coeff in op.terms.items():
            terms_before = [t for t in term if t[0] < pos]
            terms_after = [t for t in term if t[0] >= pos]

            for term_insert, coeff_insert in op_insert.terms.items():
                op_new += QubitOperator(terms_before 
                                        + shift_first_entry(term_insert, pos) 
                                        + shift_first_entry(terms_after, pos+inserted_nqubits), 
                                        coeff*coeff_insert)
                
        return op_new

    p0i= lambda i : 0.5*(1 + QubitOperator('Z{}'.format(i)))
    p1i= lambda i : 0.5*(1 - QubitOperator('Z{}'.format(i)))

    if ctrl_state == 1:
        ext_op = insert_qubit_op_at_pos(sym, p1i(0), ctrl_qubit, 1)
        ext_op += eig_val*p0i(ctrl_qubit)
    else:
        ext_op = insert_qubit_op_at_pos(sym, p0i(0), ctrl_qubit, 1)
        ext_op += eig_val*p1i(ctrl_qubit)
    return ext_op

def find_csf_connected_qubits(csf: CSF, quantum_qubits: list):
    """
    Finds connected subsets of qubits in the quantum_qubits of csf

    """

    #build graph
    G = nx.Graph()
    G.add_nodes_from(quantum_qubits)

    for pairexc in csf.get_excitations():
        edges = []
        for exc in pairexc.get_excitations():
            #raise assertion error when exc cuts across quantum_qubits
            assert (exc[0] in quantum_qubits and exc[1] in quantum_qubits) or (exc[0] not in quantum_qubits and exc[1] not in quantum_qubits), "Pair excitation cuts to outside quantum_qubits!"
            edges.append(exc)
        G.add_edges_from(edges)
    
    components = list(nx.connected_components(G))

    return components

def list_csf_z2_symmetries(csf: CSF, quantum_qubits, verify=False):
    """
    Returns symmetries with corresponding eigenvalues for csf, quantum qubits is a subset of unoccupied/doubly occupied orbitals
    Finds quantum qubits connected by pair excitations
    
    Returned symmetries over reduced number of qubits!

    """
    def get_parity_op(qubits):
        return QubitOperator(''.join(['Z{} '.format(i) for i in qubits]), 1.0)
    
    def get_X_op(qubits):
        return QubitOperator(''.join(['X{} '.format(i) for i in qubits]), 1.0)
    
    list_sym = []
    list_eig = []
    #ensure SOMOs not in CSF (currently not handled!) TODO
    #assert all([q not in csf.orbitals for q in quantum_qubits]), "Quantum qubits consists of SOMO!"

    #SOMO
    sen1_quantum_qubits = [q for q in quantum_qubits if q in csf.orbitals]
    #treat case by case TODO for off-diagonal entries
    n_sen1 = len(sen1_quantum_qubits)

    if n_sen1 > 0:

        if np.all(csf.t_vec == [1/2, -1/2]):
            # [1/2, -1/2] case, bell state with syms XX (-1) ZZ (-1)
            assert n_sen1 == 2, "Invalid SOMO quantum qubits."
            i, a = csf.orbitals
            qub_pos = [quantum_qubits.index(i), quantum_qubits.index(a)]
            list_sym.extend([get_parity_op(qub_pos), get_X_op(qub_pos)])
            list_eig.extend([-1, -1])
        elif np.all(csf.t_vec == [1/2, -1/2, 1/2, -1/2]):
            #double bell state
            assert n_sen1 == 2 or n_sen1 == 4, "Invalid SOMO quantum qubits."

            i, a, j, b = csf.orbitals

            assert (i in sen1_quantum_qubits and a in sen1_quantum_qubits) or (i not in sen1_quantum_qubits and a not in sen1_quantum_qubits), "CSF qubits split up."
            
            if i in sen1_quantum_qubits:
                qub_pos = [quantum_qubits.index(i), quantum_qubits.index(a)]
                list_sym.extend([get_parity_op(qub_pos), get_X_op(qub_pos)])
                list_eig.extend([-1, -1])
            
            assert (j in sen1_quantum_qubits and b in sen1_quantum_qubits) or (j not in sen1_quantum_qubits and b not in sen1_quantum_qubits), "CSF qubits split up."
            
            if j in sen1_quantum_qubits:
                qub_pos = [quantum_qubits.index(j), quantum_qubits.index(b)]
                list_sym.extend([get_parity_op(qub_pos), get_X_op(qub_pos)])
                list_eig.extend([-1, -1])
        elif np.all(csf.t_vec == [1/2, 1/2, -1/2, -1/2]):
            assert n_sen1 == 4, "Invalid SOMO quantum qubits."

            i, j, a, b = csf.orbitals
            qi, qj, qa, qb = quantum_qubits.index(i), quantum_qubits.index(j), quantum_qubits.index(a), quantum_qubits.index(b)

            sym1 = get_X_op([qi, qj, qa, qb])
            sym2 = QubitOperator('Z{} Z{}'.format(qi, qj), 1/3) + QubitOperator('X{} X{} Z{} Z{}'.format(qj, qa, qi, qb), 2/3) - QubitOperator('X{} X{}'.format(qj, qb), 2/3)
            sym3 = get_parity_op([qi, qj, qa, qb])
            sym4 = QubitOperator('X{} X{}'.format(qa, qb), 0.5) - QubitOperator('Z{} Z{}'.format(qi, qb), 0.5) - QubitOperator('Z{} Z{}'.format(qi, qa), 0.5) - QubitOperator('Z{} Z{} X{} X{}'.format(qa, qb, qa, qb), 0.5)
            list_sym.extend([sym1, sym2, sym3, sym4])
            list_eig.extend([1.0, 1.0, 1.0, 1.0])
            
        else:
            assert False, "t_vec = {} not handled yet!".format(csf.t_vec)
    
    #DOMO
    sen0_quantum_qubits = [q for q in quantum_qubits if q not in csf.orbitals]
    
    connected_qubits = find_csf_connected_qubits(csf, sen0_quantum_qubits) # doubly occupied and unoccupied orbitals
    for comp in connected_qubits:
        qub_pos = [quantum_qubits.index(c) for c in comp]
        list_sym.append(get_parity_op(qub_pos))
        parity = np.sum([csf.get_doubly_occ_orbitals()[i] for i in comp]) % 2
        list_eig.append((-1)**(parity))
    
    if verify:
        #check all symmetries explicitly
        qc = csf.get_tapered_full_circuit(quantum_qubits)
        n_qubits = len(quantum_qubits)

        state = show_state(qc)
        sparse_vec = get_sparse_state(state)

        for e, sym in zip(list_eig, list_sym):
            sym_sparse = get_sparse_operator(sym, n_qubits)
            e_val = (sparse_vec.conjugate() @ sym_sparse @ sparse_vec.T)[0, 0]
            print(e_val)
            assert np.isclose(e_val, e)
    
    return list_sym, list_eig

def construct_csf_sym_projector(csf, quantum_qubits):

    list_sym, list_eig_vals = list_csf_z2_symmetries(csf, quantum_qubits)

    return construct_z2_projector(list_sym, list_eig_vals)

def list_ext_symmetries(csf0, csf1, quantum_qubits, ctrl_qubit_pos):
    """
    List/construct extended state symmetries

    """

    #parity over D, parity over S, any product or CSF states TODO symmetries of S
    list_syms = []
    list_eig_vals = []

    list_syms0, list_eig_vals0 = list_csf_z2_symmetries(csf0, quantum_qubits)
    list_syms1, list_eig_vals1 = list_csf_z2_symmetries(csf1, quantum_qubits)

    #get_extended_symmetry()
    for sym, eig in zip(list_syms0, list_eig_vals0):
        list_syms.append(get_extended_symmetry(sym, eig, 0, ctrl_qubit_pos))
        list_eig_vals.append(eig)
    
    for sym, eig in zip(list_syms1, list_eig_vals1):
        list_syms.append(get_extended_symmetry(sym, eig, 1, ctrl_qubit_pos))
        list_eig_vals.append(eig)

    return list_syms, list_eig_vals

def construct_ext_sym_projector(csf0, csf1, quantum_qubits, ctrl_qubit_pos=None):
    if ctrl_qubit_pos is None:
        ctrl_qubit_pos = -1 # end as default
    
    list_syms, list_eig_vals = list_ext_symmetries(csf0, csf1, quantum_qubits, ctrl_qubit_pos)

    return construct_z2_projector(list_syms, list_eig_vals)

#ZNE
class NoiseAmplifier:
    """
    Base class to fold/increase qiskit circuit noise

    DO NOT CALL DIRECTLY!

    """
    def __init__(self):
        pass

    def get_amplified_circuit(self, circuit: QuantumCircuit, l):
        return circuit

class FullLocalFoldNoiseAmplifier(NoiseAmplifier):
    """
    Class to locally fold every gate in the circuit to amplify noise
    
    """
    def __init__(self):
        super().__init__()
    
    def get_amplified_circuit(self, circuit: QuantumCircuit, l: int):

        assert l%2 == 1, "Amplification factor {} is not valid, need odd integers".format(l)
        n = l//2

        circuit.decompose()
        circuit_no_meas, measurements = split_measurements(circuit)
        
        qc = QuantumCircuit(circuit_no_meas.qubits, circuit_no_meas.clbits)

        for instr, qargs, cargs in circuit_no_meas.data:
            qc.append(instr, qargs, cargs)

            for i in range(n):
                qc.append(instr.inverse(), qargs, cargs)
                qc.append(instr, qargs, cargs)

        append_measurements(qc, measurements)
        return qc

class ProbabilisticLocalFoldNoiseAmplifier(NoiseAmplifier):
    """
    Class to locally fold a fraction of gates

    Use when circuits are deep and full folding is not possible/useful

    """
    def __init__(self):
        super().__init__()
    
    def get_amplified_circuit(self, circuit: QuantumCircuit, l: float):
        assert l >= 1, "Invalid extrapolation factor {}".format(l)
        assert l <= 3, "Extrapolation factor too big, keep 1 <= l <= 3"
        n = (l-1)/2 #fold factor
        
        circuit.decompose()
        circuit_no_meas, measurements = split_measurements(circuit)
        
        qc = QuantumCircuit(circuit_no_meas.qubits, circuit_no_meas.clbits)

        for instr, qargs, cargs in circuit_no_meas.data:
            qc.append(instr, qargs, cargs)

            #add with probability
            r_float = np.random.rand()
            if r_float <= n:
                qc.append(instr.inverse(), qargs, cargs)
                qc.append(instr, qargs, cargs)
        
        append_measurements(qc, measurements)
        return qc

class Extrapolator:
    """
    Base class for estimate noise extrapolator

    DO NOT CALL
    
    """
    def __init__(self, noise_levels, circuit_amplifier):
        self.circuit_modifer = circuit_amplifier
        self.noise_levels = noise_levels
    
    def extrapolate(self, estimates):
        
        raise Exception("Base Extrapolator undefined!")
    
    def get_circuits(self, circuit):
        return [self.circuit_amplifier.get_amplified_circuit(circuit, l) for l in self.noise_levels]
    
    def estimate_overhead(self):
        """
        Estimate expected sampling overhead

        """
        #TODO
        return

class LinearExtrapolator(Extrapolator):
    """
    Linear extrapolator
    """

    def __init__(self, noise_levels, circuit_amplifier):
        super().__init__(noise_levels, circuit_amplifier)
    
    def extrapolate(self, estimates):
        coeffs = self.get_fit_coeff()

        return np.sum([c*e for c, e in zip(coeffs, estimates)])
    
    def get_fit_coeff(self):
        """
        Coefficients for multiplying estimates, to minimize the l2 norm
        
        """
        lmean = np.mean(self.noise_levels)
        coeffs = []
        for lk in self.noise_levels:
            coeffs.append(sum([l*(l - lk) for l in self.noise_levels])/(len(self.noise_levels) * sum([(ll - lmean)**2 for ll in self.noise_levels]) ))
        return coeffs


class ReferenceStateShift:
    """
    Determine closest classically simulable estimate and the corresponding circuit to determine shift

    Modifies experiment and estimate
    
    TODO
    """

    def __init__(self):
        return
    

#RM
class ReadoutMitigator:
    """
    Base class for readout mitigation routines, to perform experiments and mitigate readout errors
    
    DO NOT USE
    """
    def __init__(self, n_qubits, A = None):
        self.n_qubits = n_qubits
        
        if A is None:
            A = np.identity(2**self.n_qubits)
        
        self.A = A #inversion matrix

    def get_calibration_circuits(self):
        return []
    
    def process_calibration_results(self, results):
        self.A = np.identity(1<<self.n_qubits)
    
    def set_inversion_matrix(self, A):
        assert np.shape(A) == (1<<self.n_qubits, 1<<self.n_qubits), "Inversion matrix of incorrect dimensions!"
        self.A = A
    
    def mitigate(self, samples):
        #form distribution
        p_vec = form_probability_vector(samples)
        p_vec_mitig = self.A @ p_vec
        
        return p_vec_mitig

class LocalReadoutMitigator(ReadoutMitigator):
    """
    Local, 1 qubit readout mitigation/inversion

    TODO
    """
    def __init__(self, A = None):
        return