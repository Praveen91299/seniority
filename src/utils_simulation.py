import numpy as np
import qiskit_aer

def resample_counts(counts: dict, N: int, reps : int = 1):
    """
    Resamples counts by bootstrap sampling, with N samples, repeated rep times.

    counts: dict
    N: number of samples in a bootstrap replicate
    rep: repetitions of sampling

    returns replicates : list[dict]

    """
    M = sum(list(counts.values()))
    objects = list(counts.keys())
    probabilities = [v/M for v in counts.values()]

    replicates = []

    for rep in range(reps):
        samples  = np.random.choice(objects, N, replace= True, p = probabilities)

        #convert to dict
        sample_dict = {}
        for sample in samples:
            if sample not in sample_dict:
                sample_dict[sample] = 1
            else:
                sample_dict[sample] += 1
        replicates.append(sample_dict)
    return replicates

def get_depol_noise(p1, p2, noise_model=None):
    if noise_model is None:
        noise_model = qiskit_aer.noise.NoiseModel()

    error1 = qiskit_aer.noise.depolarizing_error(p1, 1)
    error2 = qiskit_aer.noise.depolarizing_error(p2, 2)
    
    noise_model.add_all_qubit_quantum_error(error1, ['u1', 'u2', 'u3', 'X', 'Y', 'Z', 'H'])
    noise_model.add_all_qubit_quantum_error(error2, ['cx'])
    return noise_model

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