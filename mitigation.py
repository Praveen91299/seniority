import numpy as np
from qiskit import QuantumCircuit

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
    
    def mitigate(self, result_dict: dict, parity_qubit, silent=False):

        if not silent: print('\nMitigation: Symmetry verification.')
        
        filtered_dict = {}
        total_shots = 0
        filtered_shots = 0

        for bit_str, v in zip(result_dict.keys(), result_dict.values()):
            total_shots += v

            if int(bit_str[parity_qubit]) == self.parity:
                filtered_dict[bit_str] = v

                filtered_shots += v

        if not silent: print(f'SV: {filtered_shots} retained out of {total_shots} shots')

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