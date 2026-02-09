### functions required to implement qsense on device

from openfermion import FermionOperator
from seniority.src.circuits.circuits_csf import CSF
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