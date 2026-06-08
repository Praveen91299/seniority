### functions required to implement qsense on device

from openfermion import FermionOperator, QubitOperator
from .circuits.circuits_csf import CSF
from qiskit.result import Result
import numpy as np

from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import EstimatorV2

def get_shot_allocation(sig_frag_mat: dict, c, N, tol=1e-10):
    """
    Returns measurment allocations for QSENSE

    sigma_frag_mat: dict[uv tuple: list[float] ] - dictionary with entries consisting of list of fragment standard deviations
    c: solution state coefficient vector
    N: total shots

    dict[tup: list[float] ] returns dictionary of measurement shots for some matrix elements

    """

    c_abs = np.abs(c)
    sig_matrix = {}#np.zeros((n, n)) # SD of matrix entries
    Muv = {} #entry variance

    M = 0 #total variance
    for uv in sig_frag_mat:
        sig_matrix[uv] = sum(sig_frag_mat[uv])

        u, v = uv
        if u == v:
            #diagonal
            Muv[uv] = c_abs[u]**2 * sig_matrix[uv]
        else:
            #off diagonal, extra factor of two
            Muv[uv] = 2 * c_abs[u] * c_abs[v] * sig_matrix[uv]
        
        M += Muv[uv]

    F = {} #shots per fragment
    for uv in sig_frag_mat:
        u, v = uv

        if abs(sig_matrix[uv]) > tol:
            F[uv] = np.ceil(np.abs(N * (Muv[uv] / M) * (np.array(sig_frag_mat[uv]) / sig_matrix[u, v]))) # sets a minimum of 1 shot
        else:
            F[uv] = [0]*len(sig_frag_mat[uv])
    
    return F


def openfermion_to_sparse_pauli_op(qubit_op: QubitOperator, num_qubits: int = None) -> SparsePauliOp:
    """
    Converts an OpenFermion QubitOperator into a Qiskit SparsePauliOp.
    
    Parameters:
        qubit_op (QubitOperator): The OpenFermion operator to convert.
        num_qubits (int): Total number of qubits. If None, it is inferred 
                          from the highest qubit index present in qubit_op.
    """
    if not isinstance(qubit_op, QubitOperator):
        raise TypeError("Input must be an openfermion.QubitOperator instance.")

    # 1. Determine total qubits if not provided
    if num_qubits is None:
        if not qubit_op.terms:
            return SparsePauliOp.from_list([("I", 0.0)])
        # Find the maximum qubit index across all terms
        num_qubits = max([index for term in qubit_op.terms for index, _ in term] + [0]) + 1

    pauli_list = []
    
    # 2. Loop through OpenFermion's dictionary representation
    # Example format: { ((0, 'X'), (2, 'Z')): 0.5 }
    for term, coefficient in qubit_op.terms.items():
        # Initialize an identity string array of the correct length
        pauli_string_list = ["I"] * num_qubits
        
        # Populate specific Pauli operators
        for qubit_index, operator in term:
            if qubit_index >= num_qubits:
                raise ValueError(f"Qubit index {qubit_index} exceeds total num_qubits ({num_qubits}).")
            pauli_string_list[qubit_index] = operator
            
        # 3. Handle Qiskit little-endian ordering (reverse the string list)
        # Qiskit layout maps qubit 0 to the rightmost character: "ZXI" -> I on q0, X on q1, Z on q2
        pauli_string_list.reverse()
        pauli_string = "".join(pauli_string_list)
        
        pauli_list.append((pauli_string, complex(coefficient)))
        
    return SparsePauliOp.from_list(pauli_list)

def get_precision_for_shots(observable: SparsePauliOp, desired_shots):
    # Sum of absolute values of coefficients
    sum_abs_coeffs = np.sum(np.abs(observable.coeffs))
    # Solve for epsilon: shots = (sum / eps)**2  -->  eps = sum / sqrt(shots)
    return sum_abs_coeffs / np.sqrt(desired_shots)

def get_precision_for_shots_std(std, desired_shots):
    # Solve for eps: shots = (quantum std.dev/ eps)**2
    return np.real(std / np.sqrt(desired_shots))

def calculate_matrix_std(c, std_dict):
    """
    Return final estimate standard deviation from element standard deviation

    c : solution coefficient vector
    std_dict : estimate standard deviations
    """
    var = 0
    for uv in std_dict.keys():
        i, j = uv

        if i == j:
            var += (c[i] * c[j] * std_dict[uv])**2
        else:
            var += (2 * c[i] * c[j] * std_dict[uv])**2
    
    return var ** 0.5

def run_estimator(circuits: list[QuantumCircuit], observables: list[SparsePauliOp], shots: list[int], estimator: EstimatorV2, compiled=False):
    """
    Runs estimator for provided circuits and observables
    circuit should be compiled with pass manager 
        pm = generate_pass_manager(optimization_level=3, backend=backend)
        pm.run(circuit)
    Returns estimates, standard deviations and result objects

    """
    # assuming no parameters right now

    isa_observables = []
    if not compiled:
        for circuit, observable in zip(circuits, observables):
            isa_observable = observable.apply_layout(circuit.layout)
            isa_observables.append(isa_observable)
    
    else:
        isa_observables = observables

    pub = [(ansatz, isa_hamiltonian, None, get_precision_for_shots(isa_hamiltonian, shot)) for ansatz, isa_hamiltonian, shot in zip(circuits, isa_observables, shots)]
    job, _ = run_pub(pub, estimator)
    return retrieve_results(job)

def run_pub(pub, estimator):
    job = estimator.run(pub)
    job_id = job.job_id() if hasattr(job, "job_id") else None
    return job, job_id

def retrieve_results(job):
    results = job.result()
    estimates = [result.data.evs for result in results]
    stds = [result.data.stds for result in results]

    return estimates, stds, results


class Expt:
    """
    Class that holds a state prep U, observable, and mitigation related options - provides helper functions for experiment building and running.


    mitigation currently supported:
    post-selectors

    """
    def __init__(self, circuit, observable):
        self.circuit = circuit
        self.shots = None
        return
    
    @classmethod
    def ObservableEstimator(cls, observable): # fragment and store circuits
        return

class SubspaceExpt:
    """
    Class to configure subspace experiment, obtain circuits to run, process and obtain final estimates

    TODO
    
    """
    def __init__(self, csf_states: list[CSF], H: FermionOperator, mitigation_options = None, device_options = None):

        self.csf_states =  csf_states
        self.H = H

        self.do_zne = False
        self.do_readout = False
        self.do_sv = False

        if mitigation_options is not None:
            self.configure_mitigation_options(mitigation_options)
        
        self.device_options = None
        if device_options is not None:
            self.configure_device(device_options=device_options)
        
        #build fragments
    
    def configure_mitigation_options(self, options: dict):
        if "extrapolation" in options:
            self.do_zne = True
            self.zne_extrapolator = options["extrapolation"]
        if "readout" in options:
            self.do_readout = True
            self.readout_mitigator = options["readout"]
        if "symmetry" in options:
            self.do_sv = True
            self.symmetry_mitigator = options["symmetry"]
        
        return
    
    def configure_device(self, device_options):
        """
        Set device and access options
        """

        return
    
    def make_fragments(self, ):
        return
    
    def get_experiment(self, shots):
        """
        Main function to generate instances of circuits and corresponding shots
        
        """

        #get states and circuits

        ### fragments, return measurement circuits and estimated shots required from self.H (without mitigation) SMIK TODO
        #get reduced H
        #fragmenting reduced H

        #preprocess for mitigation routines (SV)

        #add measurement circuits

        #preprocess for mitigation routines (zne)

        return

    def get_estimate_from_results(self, results: list[Result]):
        """
        Returns final estimate and estimated inaccuracy from list[Qiskit.Result]
        
        """

        #process results for mitigation routines (SV, readout)

        #build estimates from Dict{bitstring: probability} SMIK TODO

        #process for mitigation routines (zne, RefShift)

        #process for final estimate
        #diag stuff

        #determine inaccuracy

        return
