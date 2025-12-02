### functions required to implement qsense on device

from openfermion import FermionOperator
from seniority.src.circuits.circuits_csf import CSF
import qiskit_aer
from qiskit.result import Result
import numpy as np

def get_shot_allocation(sig_frag_mat: dict, c, N):
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
        F[uv] = np.ceil(np.abs(N * (Muv[uv] / M) * (np.array(sig_frag_mat[uv]) / sig_matrix[u, v]))) # sets a minimum of 1 shot
    
    return F

def get_willow_noise_model(noise_type = None):
    noise_model = qiskit_aer.noise.NoiseModel()

    if noise_type is None:
        return noise_model
    
    print("Retrieving Willow's {} noise...".format(noise_type))
    # noise data from Google Willow
    p_1qubit = 0.035/100
    p_2qubit = 0.33/100
    p_meas = 0.77/100
    T_1 = 68*1e3 # measurement unit is nanoseconds
    T_2 = 89*1e3
    exec_1qubit = 25
    exec_2qubit = 43
    exec_measurement = 409 # this number was calculated by copilot based on figure S7 of https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-024-08449-y/MediaObjects/41586_2024_8449_MOESM1_ESM.pdf readout pulse time + ringdown time

    error1 = qiskit_aer.noise.depolarizing_error(p_1qubit, 1)
    error2 = qiskit_aer.noise.depolarizing_error(p_2qubit, 2)
    error_therm_1qubit = qiskit_aer.noise.thermal_relaxation_error(T_1, T_2, exec_1qubit)
    error_therm_2qubit = qiskit_aer.noise.thermal_relaxation_error(T_1, T_2, exec_2qubit)
    error_therm_measure = qiskit_aer.noise.thermal_relaxation_error(T_1, T_2, exec_measurement)
    error_meas = qiskit_aer.noise.ReadoutError([[1-p_meas, p_meas], [p_meas, 1-p_meas]])

    if noise_type == 'decoherence':
        noise_model.add_all_qubit_quantum_error(error1, ['u1', 'u2', 'u3', 'X', 'Y', 'Z', 'H'])
        noise_model.add_all_qubit_quantum_error(error2, ['cx'])
    elif noise_model == 'measurement':
        noise_model.add_all_qubit_readout_error(error_meas, ['measure'])
    elif noise_type == 'thermal':
        noise_model.add_all_qubit_quantum_error(error_therm_1qubit, ['u1', 'u2', 'u3', 'X', 'Y', 'Z', 'H'])
        noise_model.add_all_qubit_quantum_error(error_therm_2qubit.expand(error_therm_2qubit), ['cx'])
        noise_model.add_all_qubit_quantum_error(error_therm_measure, ['measure'])
    elif noise_type == 'all':
        noise_model.add_all_qubit_quantum_error(error1, ['u1', 'u2', 'u3', 'X', 'Y', 'Z', 'H'])
        noise_model.add_all_qubit_quantum_error(error2, ['cx'])
        noise_model.add_all_qubit_quantum_error(error_therm_1qubit, ['u1', 'u2', 'u3', 'X', 'Y', 'Z', 'H'])
        noise_model.add_all_qubit_quantum_error(error_therm_2qubit.expand(error_therm_2qubit), ['cx'])
        noise_model.add_all_qubit_quantum_error(error_therm_measure, ['measure'])
        noise_model.add_all_qubit_readout_error(error_meas, ['measure'])
    
    return noise_model

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