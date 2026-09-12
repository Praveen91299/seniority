import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
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


def _append_native_inverse(circuit, operation, qubits, clbits):
    """Append a minimal inverse, keeping SX inversion in the IBM native basis."""
    if operation.name == "sx":
        qubit = qubits[0]
        circuit.rz(np.pi, qubit)
        circuit.sx(qubit)
        circuit.rz(-np.pi, qubit)
    else:
        circuit.append(operation.inverse(), qubits, clbits)


class FullLocalFoldNoiseAmplifier(NoiseAmplifier):
    """
    Class to locally fold every gate in the circuit to amplify noise
    
    """
    def __init__(self):
        super().__init__()
    
    def get_amplified_circuit(self, circuit: QuantumCircuit, l: int):
        if not isinstance(l, (int, np.integer)) or l < 1 or l % 2 != 1:
            raise ValueError(
                f"Amplification factor {l} is invalid; use a positive odd integer."
            )
        repetitions = (int(l) - 1) // 2
        qc = circuit.copy_empty_like(
            name=f"{circuit.name}_local_fold_{int(l)}"
        )
        excluded = {"measure", "reset", "barrier", "delay"}
        for instruction in circuit.data:
            operation = instruction.operation
            qc.append(operation, instruction.qubits, instruction.clbits)
            if operation.name in excluded:
                continue
            for _ in range(repetitions):
                _append_native_inverse(
                    qc,
                    operation,
                    instruction.qubits,
                    instruction.clbits,
                )
                qc.append(
                    operation,
                    instruction.qubits,
                    instruction.clbits,
                )
        return qc

class ProbabilisticLocalFoldNoiseAmplifier(NoiseAmplifier):
    """
    Class to locally fold a fraction of gates

    Use when circuits are deep and full folding is not possible/useful

    """
    def __init__(self, seed=None, fold_gate_names=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.fold_gate_names = (
            None if fold_gate_names is None else set(fold_gate_names)
        )
    
    def get_amplified_circuit(self, circuit: QuantumCircuit, l: float):
        if not 1 <= l <= 3:
            raise ValueError(
                f"Invalid probabilistic amplification factor {l}; "
                "use 1 <= factor <= 3."
            )
        fold_probability = (float(l) - 1.0) / 2.0
        excluded = {"measure", "reset", "barrier", "delay"}
        qc = circuit.copy_empty_like(
            name=f"{circuit.name}_probabilistic_fold_{float(l):g}"
        )
        folded_gate_count = 0
        eligible_gate_count = 0
        for instruction in circuit.data:
            operation = instruction.operation
            qc.append(
                operation,
                instruction.qubits,
                instruction.clbits,
            )
            if operation.name in excluded or (
                self.fold_gate_names is not None
                and operation.name not in self.fold_gate_names
            ):
                continue
            eligible_gate_count += 1
            if self.rng.random() >= fold_probability:
                continue
            _append_native_inverse(
                qc,
                operation,
                instruction.qubits,
                instruction.clbits,
            )
            qc.append(
                operation,
                instruction.qubits,
                instruction.clbits,
            )
            folded_gate_count += 1
        qc.metadata = {
            **(qc.metadata or {}),
            "zne_noise_factor": float(l),
            "zne_fold_probability": fold_probability,
            "zne_eligible_gate_count": eligible_gate_count,
            "zne_folded_gate_count": folded_gate_count,
        }
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

    def extrapolate_variance(self, variances):
        """Propagate independent estimator variances to the zero-noise intercept."""
        coeffs = np.asarray(self.get_fit_coeff(), dtype=float)
        variances = np.asarray(variances, dtype=float)
        if variances.shape != coeffs.shape:
            raise ValueError("One variance is required for each noise level.")
        return float(np.sum(coeffs**2 * variances))


def local_fold_circuit(circuit, scale_factor, backend=None):
    """Return a fully local-folded circuit, optionally in a backend-native basis."""
    folded = FullLocalFoldNoiseAmplifier().get_amplified_circuit(
        circuit,
        scale_factor,
    )
    if backend is None:
        return folded
    unsupported = {
        instruction.operation.name
        for instruction in folded.data
        if instruction.operation.name not in backend.operation_names
    }
    if not unsupported:
        return folded
    return transpile(
        folded,
        backend=backend,
        initial_layout=list(range(circuit.num_qubits)),
        optimization_level=0,
    )


def probabilistic_local_fold_circuit(
    circuit,
    scale_factor,
    seed=None,
    backend=None,
):
    """Randomly fold gates to realize an expected factor in the interval [1, 3]."""
    fold_gate_names = None
    if backend is not None:
        fold_gate_names = set()
        for name in backend.operation_names:
            try:
                properties = backend.target[name].values()
            except (KeyError, AttributeError):
                continue
            if any(
                item is not None
                and getattr(item, "error", None) is not None
                and item.error > 0
                for item in properties
            ):
                fold_gate_names.add(name)
    folded = ProbabilisticLocalFoldNoiseAmplifier(
        seed=seed,
        fold_gate_names=fold_gate_names,
    ).get_amplified_circuit(circuit, scale_factor)
    if backend is None:
        return folded
    unsupported = {
        instruction.operation.name
        for instruction in folded.data
        if instruction.operation.name not in backend.operation_names
    }
    if not unsupported:
        return folded
    return transpile(
        folded,
        backend=backend,
        initial_layout=list(range(circuit.num_qubits)),
        optimization_level=0,
    )


def linear_zne(noise_factors, estimates, variances=None):
    """Return the least-squares linear zero-noise intercept and its variance."""
    extrapolator = LinearExtrapolator(
        list(noise_factors),
        FullLocalFoldNoiseAmplifier(),
    )
    estimate = extrapolator.extrapolate(estimates)
    variance = (
        None
        if variances is None
        else extrapolator.extrapolate_variance(variances)
    )
    return estimate, variance, np.asarray(
        extrapolator.get_fit_coeff(),
        dtype=float,
    )


def variance_weighted_linear_zne(noise_factors, estimates, variances):
    """Return a variance-weighted linear zero-noise intercept.

    The factor estimates are assumed independent.  Their inverse variances
    define the generalized least-squares weights, and the returned
    coefficients map the supplied estimates directly to the fitted
    intercept.
    """
    noise_factors = np.asarray(noise_factors, dtype=float)
    estimates = np.asarray(estimates)
    variances = np.asarray(variances, dtype=float)
    if not (
        noise_factors.ndim == estimates.ndim == variances.ndim == 1
        and noise_factors.size == estimates.size == variances.size
    ):
        raise ValueError(
            "Noise factors, estimates, and variances must be equal-length "
            "one-dimensional arrays."
        )
    if noise_factors.size < 2:
        raise ValueError("At least two noise factors are required.")
    if np.any(~np.isfinite(variances)) or np.any(variances <= 0):
        raise ValueError("All variances must be finite and strictly positive.")

    design = np.column_stack(
        (np.ones(noise_factors.size), noise_factors)
    )
    inverse_variances = 1.0 / variances
    normal_inverse = np.linalg.inv(
        design.T @ (inverse_variances[:, None] * design)
    )
    coefficients = (
        normal_inverse[0]
        @ (design.T * inverse_variances)
    )
    estimate = coefficients @ estimates
    variance = float(np.sum(coefficients**2 * variances))
    return estimate, variance, coefficients


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

class LocalReadoutMitigator:
    """Independent per-qubit assignment-matrix readout mitigation."""

    def __init__(self, assignment_matrices):
        self.assignment_matrices = {
            int(qubit): np.asarray(matrix, dtype=float)
            for qubit, matrix in assignment_matrices.items()
        }
        self.inverse_assignment_matrices = {
            qubit: np.linalg.inv(matrix)
            for qubit, matrix in self.assignment_matrices.items()
        }

    def expectation_values(self, counts, paulis, physical_qubits):
        """Return mitigated means and single-shot variances for Pauli products."""
        physical_qubits = list(map(int, physical_qubits))
        if paulis.num_qubits != len(physical_qubits):
            raise ValueError(
                "Measured Pauli width and physical-qubit mapping differ: "
                f"{paulis.num_qubits} versus {len(physical_qubits)}."
            )
        missing = [
            qubit
            for qubit in physical_qubits
            if qubit not in self.inverse_assignment_matrices
        ]
        if missing:
            raise ValueError(
                f"Missing readout calibrations for physical qubits {missing}."
            )

        supports = np.logical_or(paulis.z, paulis.x)
        z_weights = {
            qubit: np.array([1.0, -1.0])
            @ self.inverse_assignment_matrices[qubit]
            for qubit in physical_qubits
        }
        means = np.zeros(len(paulis), dtype=float)
        second_moments = np.zeros(len(paulis), dtype=float)
        total_shots = int(sum(counts.values()))
        if total_shots <= 0:
            raise ValueError("Cannot mitigate an empty count histogram.")

        for bitstring, frequency in counts.items():
            measured_register = bitstring.split(" ", 1)[0]
            outcome = int(measured_register, 2)
            for pauli_index, support in enumerate(supports):
                value = 1.0
                for local_qubit in np.flatnonzero(support):
                    measured_bit = (outcome >> int(local_qubit)) & 1
                    physical_qubit = physical_qubits[int(local_qubit)]
                    value *= z_weights[physical_qubit][measured_bit]
                means[pauli_index] += frequency * value
                second_moments[pauli_index] += frequency * value**2

        means /= total_shots
        second_moments /= total_shots
        variances = np.maximum(second_moments - means**2, 0.0)
        return means, variances


def _bit_marginal(counts, classical_position, value):
    selected = 0
    total = int(sum(counts.values()))
    for bitstring, frequency in counts.items():
        bits_little_endian = bitstring.replace(" ", "")[::-1]
        if int(bits_little_endian[classical_position]) == value:
            selected += int(frequency)
    return selected / total


def run_local_readout_calibration(
    noise_backend,
    physical_qubits,
    shots,
    seed_simulator,
    simulator_method="statevector",
):
    """Calibrate independent assignment matrices with parallel 0/1 circuits."""
    physical_qubits = list(map(int, physical_qubits))
    if not physical_qubits:
        raise ValueError("No measured qubits were found for readout calibration.")
    num_qubits = int(noise_backend.num_qubits)
    calibration_0 = QuantumCircuit(
        num_qubits,
        len(physical_qubits),
        name="readout_calibration_0",
    )
    calibration_1 = QuantumCircuit(
        num_qubits,
        len(physical_qubits),
        name="readout_calibration_1",
    )
    for classical_position, physical_qubit in enumerate(physical_qubits):
        calibration_0.measure(physical_qubit, classical_position)
        calibration_1.x(physical_qubit)
        calibration_1.measure(physical_qubit, classical_position)

    calibration_simulator = AerSimulator.from_backend(
        noise_backend,
        method=simulator_method,
        enable_truncation=True,
    )
    result = calibration_simulator.run(
        [calibration_0, calibration_1],
        shots=shots,
        seed_simulator=seed_simulator,
    ).result()
    counts_0 = {
        str(bitstring): int(frequency)
        for bitstring, frequency in result.get_counts(0).items()
    }
    counts_1 = {
        str(bitstring): int(frequency)
        for bitstring, frequency in result.get_counts(1).items()
    }
    assignment_matrices = {}
    for position, qubit in enumerate(physical_qubits):
        matrix = np.array(
            [
                [
                    _bit_marginal(counts_0, position, 0),
                    _bit_marginal(counts_1, position, 0),
                ],
                [
                    _bit_marginal(counts_0, position, 1),
                    _bit_marginal(counts_1, position, 1),
                ],
            ]
        )
        if abs(np.linalg.det(matrix)) < 1e-8:
            raise ValueError(
                f"Singular readout assignment matrix for qubit {qubit}: "
                f"{matrix}."
            )
        assignment_matrices[qubit] = matrix

    mitigator = LocalReadoutMitigator(assignment_matrices)
    calibration_data = {
        "enabled": True,
        "model": "independent_local_assignment_matrices",
        "physical_qubits": physical_qubits,
        "shots_per_calibration_circuit": int(shots),
        "seed_simulator": int(seed_simulator),
        "assignment_matrices": assignment_matrices,
        "inverse_assignment_matrices": mitigator.inverse_assignment_matrices,
        "prepared_0_counts": counts_0,
        "prepared_1_counts": counts_1,
    }
    return mitigator, calibration_data
