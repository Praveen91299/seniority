from openfermion import FermionOperator
from qiskit.result import Result

class SubspaceExpt:
    """
    Class to configure subspace experiment, obtain circuits to run, process and obtain final estimates

    TODO
    
    """
    def __init__(self, csf_states, H: FermionOperator, mitigation_options = None, devic_options = None):

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
            continue
        
        return
    
    def configure_device(self, device_options):
        """
        Set device and access options
        """

        return
    
    def get_experiment(self, shots):
        """
        Main function to generate instances of circuits and corresponding shots
        
        """

        #get states and circuits

        #fragments, return measurement circuits and estimated shots required from self.H (without mitigation) SMIK TODO

        #preprocess for mitigation routines (SV)

        #add measurement circuits

        #preprocess for mitigation routines (zne)

        return

    def get_estimate_from_results(results: list[Result]):
        """
        Returns final estimate and estimated inaccuracy from list[Qiskit.Result]
        
        """

        #process results for mitigation routines (SV, readout)

        #build estimates from Dict{bitstring: count} SMIK TODO

        #process for mitigation routines (zne, RefShift)

        #process for final estimate

        #determine inaccuracy

        return