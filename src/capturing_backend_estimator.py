"""BackendEstimatorV2 adapter that preserves portable raw count histograms."""

import threading

import numpy as np
from qiskit.primitives import BackendEstimatorV2
from qiskit.primitives.backend_estimator_v2 import (
    _prepare_counts,
    _run_circuits,
)


class RawCountCollector:
    """Thread-safe storage for BackendEstimatorV2 measurement histograms."""

    def __init__(self):
        self._lock = threading.Lock()
        self._records = []

    def reset(self):
        with self._lock:
            self._records = []

    def add(self, record):
        with self._lock:
            self._records.append(record)

    def snapshot(self):
        with self._lock:
            return list(self._records)


RAW_SIMULATOR_COUNTS = RawCountCollector()


class CapturingBackendEstimatorV2(BackendEstimatorV2):
    """BackendEstimatorV2 that retains the exact histograms it used."""

    def __init__(self, *args, readout_mitigator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.readout_mitigator = readout_mitigator

    def set_readout_mitigator(self, mitigator):
        self.readout_mitigator = mitigator

    def _preprocess_pub(self, pub):
        data = super()._preprocess_pub(pub)
        source_metadata = pub.circuit.metadata or {}
        qsense_metadata = {
            key: source_metadata[key]
            for key in (
                "qsense_pub_index",
                "qsense_uv",
                "qsense_fragment_index",
            )
            if key in source_metadata
        }
        for circuit in data.circuits:
            circuit.metadata = {
                **(circuit.metadata or {}),
                **qsense_metadata,
            }
        return data

    def _run_pubs(self, pubs, shots):
        preprocessed_data = []
        flat_circuits = []
        for pub in pubs:
            data = self._preprocess_pub(pub)
            preprocessed_data.append(data)
            flat_circuits.extend(data.circuits)

        run_result, metadata = _run_circuits(
            flat_circuits,
            self._backend,
            shots=shots,
            seed_simulator=self._options.seed_simulator,
        )
        counts = _prepare_counts(run_result)

        results = []
        start = 0
        for pub, data in zip(pubs, preprocessed_data):
            end = start + len(data.circuits)
            pub_counts = counts[start:end]
            pub_metadata = metadata[start:end]
            for measurement_group, (count, item) in enumerate(
                zip(pub_counts, pub_metadata)
            ):
                RAW_SIMULATOR_COUNTS.add(
                    {
                        "pub_index": item.get("qsense_pub_index"),
                        "uv": tuple(item["qsense_uv"])
                        if "qsense_uv" in item
                        else None,
                        "fragment_index": item.get(
                            "qsense_fragment_index"
                        ),
                        "measurement_group": measurement_group,
                        "shots": int(sum(count.values())),
                        "counts": {
                            str(bitstring): int(frequency)
                            for bitstring, frequency in count.items()
                        },
                        "original_pauli_labels": list(
                            item["orig_paulis"].to_labels()
                        ),
                        "measured_pauli_labels": list(
                            item["meas_paulis"].to_labels()
                        ),
                        "parameter_index": tuple(item["param_index"]),
                    }
                )
            expval_map = self._calc_expval_map(
                pub_counts,
                pub_metadata,
            )
            results.append(
                self._postprocess_pub(pub, expval_map, data, shots)
            )
            start = end
        return results

    def _calc_expval_map(self, counts, metadata):
        if self.readout_mitigator is None:
            return super()._calc_expval_map(counts, metadata)

        expval_map = {}
        for count, item in zip(counts, metadata):
            orig_paulis = item["orig_paulis"]
            meas_paulis = item["meas_paulis"]
            param_index = item["param_index"]
            support = np.logical_or.reduce(
                np.logical_or(orig_paulis.z, orig_paulis.x),
                axis=0,
            )
            physical_qubits = np.flatnonzero(support).tolist()
            if not physical_qubits:
                physical_qubits = [0]
            expvals, variances = self.readout_mitigator.expectation_values(
                count,
                meas_paulis,
                physical_qubits,
            )
            for pauli, expval, variance in zip(
                orig_paulis,
                expvals,
                variances,
            ):
                expval_map[param_index, pauli.to_label()] = (
                    expval,
                    variance,
                )
        return expval_map
