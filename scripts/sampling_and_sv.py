### script that does sampling currently
# SV can be added

### 18/11/2025 VO create fragments and job, job_desc

from _bootstrap import REPO_ROOT

from src.measurement_new.utils_ferm import (
    orthogonal_transform_obt_tbt,
    obt_phys_spatial_to_spin,
    tbt_phys_spatial_to_spin,
    make_short_H_ferm_op
)
from src.measurement_new.utils_states import (
    convert_TZ_format_to_sparse_format,
    convert_dense_format_to_sparse_format,
    tz_state_seniority_config,
    compress_state,
    create_composite_state
)
from src.measurement_new.utils_m1_seniority import (
    project_out_seniority_symmetries
)
from src.measurement_new.utils_m2_factorize import (
    get_indices_mapping_2_wvn,
    get_indices_mapping_2_wvn_vo,
    factorize_state,
    expand_tensor_product_for_incomplete_qubit_set,
    evaluate_fully_classical_factors,
    obtain_coarse_dicts,
    QC_assignment_from_qubit_labels
)
from src.measurement_new.utils_m3_swap import (
    XorY_augment
)
from src.measurement_new.utils_results import (
    sampling_cost
)
from openfermion import (
    get_sparse_operator,
    jordan_wigner
)

import csv
import gzip
import json
import os
import pickle
import re
import threading
import time

import numpy as np
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.circuits.circuits_csf import (
    CSF,
    get_csfs_from_dump,
    get_Uext_csfs_from_dump,
)
from src.circuits.circuits_pair_excitation import PairedExcitationRotation
from src.circuits.utils_circuit import get_decomp_circuits_estimators
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Statevector
from src.capturing_backend_estimator import (
    CapturingBackendEstimatorV2,
    RAW_SIMULATOR_COUNTS,
)
from src.circuits.circuits_swap import get_parallelswap_subcircuit

from src.expt import get_shot_allocation
from src.measurement_new.utils_states import frag_SD_of_decomp

from src.mitigation import (
    determine_tapered_parity,
    run_local_readout_calibration,
)
from openfermion import QubitOperator

RUNTIME_CACHE_VERSION = 8
USE_RUNTIME_CACHE = True
REBUILD_RUNTIME_CACHE = os.environ.get(
    "QSENSE_REBUILD_RUNTIME_CACHE", "1"
).strip().lower() in {"1", "true", "yes"}
RUNTIME_CACHE_DIR = Path("saved/runtime_cache")
RUN_OUTPUT_DIR = Path("saved/runtime_runs")
SAVE_RUN_OUTPUTS = True
# Portable JSON/QPY/NPZ files are always written. Enable only when an older
# analysis script still requires the legacy, version-sensitive pickle bundle.
SAVE_LEGACY_PICKLE_OUTPUTS = False
MIN_FRAGMENT_SHOTS = 1
MIN_FRAGMENT_STD = 1e-12
# Local noisy simulation: bypass Runtime Batch and submit all PUBs in one job.
# Set LOCAL_SIMULATOR_SINGLE_JOB=False before enabling a hardware configuration.
RUNTIME_EXECUTION_MODE = "job" # "job", "batch", or "session"
LOCAL_SIMULATOR_SINGLE_JOB = True
NOISY_SIMULATOR_METHOD = "statevector"
DISABLE_RUNTIME_MITIGATION = True
RUNTIME_JOB_TAGS = ["qsense_quebec_estimator"]
CONFIRM_BEFORE_SUBMIT = True
MAX_PAYLOAD_BYTES_PER_ESTIMATOR_JOB = 20 * 1024 * 1024
MAX_ESTIMATED_QPU_SECONDS_PER_ESTIMATOR_JOB = 600
PER_JOB_QPU_OVERHEAD_SECONDS = 2
SECONDS_PER_EXECUTION_ESTIMATE = 0.00035
MAX_PUBS_PER_ESTIMATOR_JOB = None
TARGET_ESTIMATE_ACCURACY = 1.6e-3
SPLIT_BASIS_DUMP = os.environ.get("QSENSE_SPLIT_BASIS_DUMP")
BASIS_VARIANT = os.environ.get("QSENSE_BASIS_VARIANT", "original")
SCIENTIFIC_METHOD = os.environ.get("QSENSE_SCIENTIFIC_METHOD", "vo").lower()
PREPARE_CACHE_ONLY = os.environ.get(
    "QSENSE_PREPARE_CACHE_ONLY", "0"
).strip().lower() in {"1", "true", "yes"}
if SCIENTIFIC_METHOD not in {"vo", "pt"}:
    raise ValueError(
        "QSENSE_SCIENTIFIC_METHOD must be either 'vo' or 'pt'."
    )
# Set this to True to add explicit local assignment-matrix mitigation.
enable_readout_mitigation = False
if "QSENSE_ENABLE_READOUT_MITIGATION" in os.environ:
    enable_readout_mitigation = os.environ[
        "QSENSE_ENABLE_READOUT_MITIGATION"
    ].strip().lower() in {"1", "true", "yes"}
READOUT_CALIBRATION_SHOTS = 20_000
READOUT_CALIBRATION_SEED = 8675309


class _SimulationProgress:
    """Thread-safe textual progress for BackendEstimatorV2 simulator groups."""

    def __init__(self):
        self._lock = threading.Lock()
        self.configure([])

    def configure(self, pub_metadata):
        with self._lock:
            self.total_groups = len({item["shots"] for item in pub_metadata})
            self.total_fragments = len(pub_metadata)
            self.started_groups = 0
            self.completed_groups = 0
            self.completed_measurement_circuits = 0

    def start_group(self, circuits, shots):
        active_qubits = [
            len(
                {
                    circuit.find_bit(qubit).index
                    for instruction in circuit.data
                    if instruction.operation.name != "barrier"
                    for qubit in instruction.qubits
                }
            )
            for circuit in circuits
        ]
        with self._lock:
            self.started_groups += 1
            group_index = self.started_groups
        active_range = _format_int_range(active_qubits)
        print(
            f"Starting simulator group {group_index}/{self.total_groups}: "
            f"{len(circuits)} measurement circuits, shots={shots}, "
            f"active qubits before Aer truncation={active_range}",
            flush=True,
        )
        return group_index

    def finish_group(self, group_index, result):
        experiment_results = getattr(result, "results", [])
        metadata = [
            getattr(experiment, "metadata", {}) or {}
            for experiment in experiment_results
        ]
        simulated_qubits = []
        for item in metadata:
            if "active_input_qubits" in item:
                simulated_qubits.append(len(item["active_input_qubits"]))
            elif "num_qubits" in item:
                simulated_qubits.append(item["num_qubits"])
        methods = sorted(
            {str(item["method"]) for item in metadata if "method" in item}
        )
        with self._lock:
            self.completed_groups += 1
            self.completed_measurement_circuits += len(experiment_results)
            completed = self.completed_groups
            completed_circuits = self.completed_measurement_circuits
            total = max(self.total_groups, completed, 1)
        width = 30
        filled = int(width * completed / total)
        bar = "#" * filled + "-" * (width - filled)
        method_text = ",".join(methods) if methods else "unreported"
        qubit_text = _format_int_range(simulated_qubits)
        print(
            f"[{bar}] {completed}/{total} simulator groups complete; "
            f"group {group_index} finished {len(experiment_results)} measurement circuits; "
            f"cumulative circuits={completed_circuits}, "
            f"submitted fragments={self.total_fragments}; "
            f"Aer method={method_text}; simulated qubits={qubit_text}",
            flush=True,
        )

    def fail_group(self, group_index, exc):
        print(f"Simulator group {group_index} failed: {exc}", flush=True)


class _ProgressJob:
    """Delegate to an Aer job and report progress when its result is ready."""

    def __init__(self, job, progress, group_index):
        self._job = job
        self._progress = progress
        self._group_index = group_index

    def result(self, *args, **kwargs):
        try:
            result = self._job.result(*args, **kwargs)
        except Exception as exc:
            self._progress.fail_group(self._group_index, exc)
            raise
        self._progress.finish_group(self._group_index, result)
        return result

    def __getattr__(self, name):
        return getattr(self._job, name)



def _format_int_range(values):
    if not values:
        return "unreported"
    minimum = min(values)
    maximum = max(values)
    return str(minimum) if minimum == maximum else f"{minimum}-{maximum}"


SIMULATION_PROGRESS = _SimulationProgress()


def _safe_cache_token(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _backend_name(backend):
    name = getattr(backend, "name", backend.__class__.__name__)
    return name() if callable(name) else name


def _runtime_cache_path(molecule, bond_length, methodtag, backend):
    method_token = _safe_cache_token(methodtag)
    backend_token = _safe_cache_token(_backend_name(backend))
    basis_token = _safe_cache_token(BASIS_VARIANT)
    return RUNTIME_CACHE_DIR / (
        f"sampling_and_sv_{molecule}_{bond_length}_{basis_token}_"
        f"{method_token}_{backend_token}.pkl"
    )


def _create_run_output_dir(molecule, bond_length, methodtag, backend):
    if not SAVE_RUN_OUTPUTS:
        return None

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    method_token = _safe_cache_token(methodtag)
    backend_token = _safe_cache_token(_backend_name(backend))
    basis_token = _safe_cache_token(BASIS_VARIANT)
    mitigation_token = (
        "_readout_mitigated" if enable_readout_mitigation else ""
    )
    output_dir = RUN_OUTPUT_DIR / (
        f"{run_id}_{molecule}_{bond_length}_{basis_token}_"
        f"{method_token}_{backend_token}{mitigation_token}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_pickle(path, payload):
    if path is None:
        return True

    try:
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        return True
    except Exception as exc:
        _write_text(path.with_suffix(path.suffix + ".error.txt"), f"pickle save failed: {exc}")
        return False


def _jsonable(value):
    """Convert scientific Python values to a reversible JSON representation."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, complex):
        return {"__complex__": [value.real, value.imag]}
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, tuple):
        return {"__tuple__": [_jsonable(item) for item in value]}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: _jsonable(item) for key, item in value.items()}
        return {
            "__mapping__": [
                [_jsonable(key), _jsonable(item)]
                for key, item in value.items()
            ]
        }
    raise TypeError(
        f"Value of type {type(value).__name__} is not JSON serializable."
    )


def _save_json(path, payload):
    """Save reversible JSON, using gzip when the filename ends in .gz."""
    if path is None:
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as file:
        json.dump(
            _jsonable(payload),
            file,
            indent=None if path.suffix == ".gz" else 2,
            separators=(",", ":") if path.suffix == ".gz" else None,
        )
        file.write("\n")


def _portable_observable(observable):
    return [
        {
            "label": label,
            "coefficient": complex(coefficient),
        }
        for label, coefficient in observable.to_list()
    ]


def _save_fragment_bundle(output_dir, fragment_artifacts):
    """Save fragment metadata/observables as JSON and circuits as QPY."""
    circuit_path = output_dir / "fragment_circuits.qpy"
    with circuit_path.open("wb") as file:
        qpy.dump(
            [record["circuit"] for record in fragment_artifacts],
            file,
        )

    manifest = []
    for circuit_index, record in enumerate(fragment_artifacts):
        portable = {
            key: value
            for key, value in record.items()
            if key not in {"circuit", "observable"}
        }
        portable["circuit_index"] = circuit_index
        portable["observable_terms"] = _portable_observable(
            record["observable"]
        )
        manifest.append(portable)
    _save_json(output_dir / "fragment_manifest.json.gz", manifest)


def _save_runtime_pub_bundle(output_dir, flat_pubs, flat_pub_metadata):
    """Save the submitted circuits and observables before execution starts."""
    with (output_dir / "runtime_circuits.qpy").open("wb") as file:
        qpy.dump([pub[0] for pub in flat_pubs], file)
    records = []
    for pub_index, (pub, metadata) in enumerate(
        zip(flat_pubs, flat_pub_metadata)
    ):
        _, observable, parameter_values, precision = pub
        records.append(
            {
                **metadata,
                "pub_index": pub_index,
                "circuit_index": pub_index,
                "observable_terms": _portable_observable(observable),
                "parameter_values": parameter_values,
                "target_precision": precision,
            }
        )
    _save_json(output_dir / "runtime_pub_manifest.json.gz", records)


def _append_text(path, text):
    if path is None:
        return

    with open(path, "a") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def _write_text(path, text):
    if path is None:
        return

    with open(path, "w") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def _job_records(submitted_blocks):
    return [
        {
            "block_index": index,
            "job_id": block["job_id"],
            "num_pubs": block["num_pubs"],
            "payload_bytes": block["payload_bytes"],
            "estimated_qpu_seconds": block["estimated_qpu_seconds"],
            "pub_indices": block["pub_indices"],
        }
        for index, block in enumerate(submitted_blocks, start=1)
    ]


def _write_job_ids(output_dir, submitted_blocks):
    if output_dir is None:
        return

    lines = [
        "Estimator job IDs",
        f"created_at_utc: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    for record in _job_records(submitted_blocks):
        lines.append(
            f"block {record['block_index']}: job_id={record['job_id']}, "
            f"PUBs={record['num_pubs']}, "
            f"payload={record['payload_bytes'] / (1024 ** 2):.2f} MB, "
            f"estimated_qpu_seconds={record['estimated_qpu_seconds']:.3f}"
        )

    _write_text(output_dir / "job_ids.txt", "\n".join(lines))
    _save_json(output_dir / "job_ids.json", _job_records(submitted_blocks))
    if SAVE_LEGACY_PICKLE_OUTPUTS:
        _save_pickle(output_dir / "job_ids.pkl", _job_records(submitted_blocks))


def _set_option_path(options, path, value):
    target = options
    try:
        for attr in path[:-1]:
            target = getattr(target, attr)
        setattr(target, path[-1], value)
        return True
    except Exception:
        return False


def _configure_estimator_options(
    estimator,
    disable_mitigation=DISABLE_RUNTIME_MITIGATION,
    job_tags=RUNTIME_JOB_TAGS,
):
    options = getattr(estimator, "options", None)
    if options is None:
        return estimator

    if job_tags:
        _set_option_path(options, ("environment", "job_tags"), job_tags)

    if disable_mitigation:
        _set_option_path(options, ("resilience", "measure_mitigation"), False)
        _set_option_path(options, ("resilience", "zne_mitigation"), False)
        _set_option_path(options, ("resilience", "pec_mitigation"), False)
        _set_option_path(options, ("twirling", "enable_gates"), False)
        _set_option_path(options, ("twirling", "enable_measure"), False)
        _set_option_path(options, ("dynamical_decoupling", "enable"), False)

    return estimator


@contextmanager
def _estimator_execution_context(
    backend,
    estimator,
    disable_mitigation=DISABLE_RUNTIME_MITIGATION,
):
    if RUNTIME_EXECUTION_MODE == "job":
        yield _configure_estimator_options(
            estimator,
            disable_mitigation=disable_mitigation,
        )
        return

    try:
        from qiskit_ibm_runtime import Batch, Session, EstimatorV2
    except ImportError as exc:
        raise RuntimeError(
            f"RUNTIME_EXECUTION_MODE={RUNTIME_EXECUTION_MODE!r} requires qiskit-ibm-runtime."
        ) from exc

    if RUNTIME_EXECUTION_MODE == "batch":
        with Batch(backend=backend) as batch:
            yield _configure_estimator_options(
                EstimatorV2(mode=batch),
                disable_mitigation=disable_mitigation,
            )
    elif RUNTIME_EXECUTION_MODE == "session":
        with Session(backend=backend) as session:
            yield _configure_estimator_options(
                EstimatorV2(mode=session),
                disable_mitigation=disable_mitigation,
            )
    else:
        raise ValueError(
            "RUNTIME_EXECUTION_MODE must be one of 'job', 'batch', or 'session'."
        )


def _runtime_inputs_summary(
    PUBs,
    quantum_indices,
    shot_alloc,
    circuits_transpiled,
    pub_metadata,
    pruning_summary,
    pub_blocks=None,
):
    all_circuits = [
        circuit
        for circuit_list in circuits_transpiled.values()
        for circuit in circuit_list
    ]
    two_qubit_ops = {"cx", "cz", "ecr"}
    two_qubit_counts = [
        sum(count for gate, count in circuit.count_ops().items() if gate in two_qubit_ops)
        for circuit in all_circuits
    ]
    nominal_qubit_counts = [circuit.num_qubits for circuit in all_circuits]
    active_qubit_counts = [
        len(
            {
                circuit.find_bit(qubit).index
                for instruction in circuit.data
                if instruction.operation.name != "barrier"
                for qubit in instruction.qubits
            }
        )
        for circuit in all_circuits
    ]

    num_estimator_jobs = len(pub_blocks) if pub_blocks is not None else 0
    estimated_qpu_seconds = (
        sum(block["estimated_qpu_seconds"] for block in pub_blocks)
        if pub_blocks is not None
        else 0
    )
    payload_bytes = (
        sum(block["payload_bytes"] for block in pub_blocks)
        if pub_blocks is not None
        else sum(item.get("payload_bytes", 0) for item in pub_metadata)
    )

    return {
        "target_estimate_accuracy": TARGET_ESTIMATE_ACCURACY,
        "simulator_method": NOISY_SIMULATOR_METHOD,
        "local_simulator_single_job": LOCAL_SIMULATOR_SINGLE_JOB,
        "readout_mitigation_enabled": enable_readout_mitigation,
        "readout_calibration_shots": (
            2 * READOUT_CALIBRATION_SHOTS
            if enable_readout_mitigation
            else 0
        ),
        "num_quantum_matrix_elements": len(quantum_indices),
        "num_pub_groups": len(PUBs),
        "num_pubs": sum(len(pubs) for pubs in PUBs.values()),
        "num_estimator_jobs": num_estimator_jobs,
        "max_payload_mb_per_estimator_job": MAX_PAYLOAD_BYTES_PER_ESTIMATOR_JOB / (1024 ** 2),
        "max_estimated_qpu_seconds_per_estimator_job": MAX_ESTIMATED_QPU_SECONDS_PER_ESTIMATOR_JOB,
        "allocated_shots_before_pruning": pruning_summary["allocated_shots_before_pruning"],
        "num_pruned_fragments": pruning_summary["num_pruned_fragments"],
        "num_pruned_matrix_elements": pruning_summary["num_pruned_matrix_elements"],
        "kept_allocated_shots": int(sum(item["shots"] for item in pub_metadata)),
        "payload_mb": payload_bytes / (1024 ** 2),
        "estimated_qpu_seconds": estimated_qpu_seconds,
        "nominal_circuit_qubits": _format_int_range(nominal_qubit_counts),
        "active_circuit_qubits_before_truncation": _format_int_range(active_qubit_counts),
        "max_circuit_depth": max((circuit.depth() for circuit in all_circuits), default=0),
        "max_two_qubit_gates": max(two_qubit_counts, default=0),
        "total_two_qubit_gates": int(sum(two_qubit_counts)),
    }


def _print_runtime_summary(summary):
    print(
        "Runtime input summary: "
        f"{summary['num_quantum_matrix_elements']} quantum matrix elements, "
        f"{summary['num_pubs']} PUBs, "
        f"{summary['num_estimator_jobs']} estimator jobs, "
        f"{summary['kept_allocated_shots']} kept shots "
        f"({summary['allocated_shots_before_pruning']} before pruning), "
        f"{summary['payload_mb']:.2f} MB payload, "
        f"~{summary['estimated_qpu_seconds']:.1f}s estimated QPU time, "
        f"{summary['num_pruned_fragments']} pruned fragments, "
        f"active qubits {summary['active_circuit_qubits_before_truncation']} "
        f"of nominal {summary['nominal_circuit_qubits']}, "
        f"max depth {summary['max_circuit_depth']}, "
        f"max two-qubit gates {summary['max_two_qubit_gates']}"
    )


def _flatten_pubs(PUBs, pub_metadata):
    flat_pubs = [
        pub
        for uv in PUBs
        for pub in PUBs[uv]
    ]

    if len(flat_pubs) != len(pub_metadata):
        raise ValueError(
            f"PUB metadata length mismatch: {len(flat_pubs)} PUBs, "
            f"{len(pub_metadata)} metadata entries."
        )

    return flat_pubs, list(pub_metadata)


def _estimate_pub_payload_bytes(pub):
    return len(pickle.dumps(pub, protocol=pickle.HIGHEST_PROTOCOL))


def _estimate_pub_qpu_seconds(metadata):
    return SECONDS_PER_EXECUTION_ESTIMATE * metadata["shots"]


def _backend_estimator_precision_for_shots(shots):
    """Choose a precision whose ceil(1 / precision**2) is exactly ``shots``."""
    if shots < 1:
        raise ValueError("BackendEstimatorV2 requires at least one shot.")
    # The small upward offset avoids floating-point roundoff producing shots + 1.
    return (1.0 + 1e-12) / np.sqrt(shots)


def _readout_support_qubits(flat_pubs):
    support = set()
    for _, observable, _, _ in flat_pubs:
        active = np.any(
            np.logical_or(observable.paulis.z, observable.paulis.x),
            axis=0,
        )
        support.update(map(int, np.flatnonzero(active)))
    return sorted(support)


def _benchmark_flat_pubs(flat_pubs, flat_pub_metadata):
    for pub_index, (pub, metadata) in enumerate(zip(flat_pubs, flat_pub_metadata)):
        metadata["pub_index"] = pub_index
        metadata["payload_bytes"] = _estimate_pub_payload_bytes(pub)
        metadata["estimated_qpu_seconds"] = _estimate_pub_qpu_seconds(metadata)
        circuit = pub[0]
        circuit.metadata = {
            **(circuit.metadata or {}),
            "qsense_pub_index": pub_index,
            "qsense_uv": tuple(metadata["uv"]),
            "qsense_fragment_index": int(metadata["fragment_index"]),
        }
    return flat_pub_metadata


def _build_pub_blocks(flat_pubs, flat_pub_metadata):
    if LOCAL_SIMULATOR_SINGLE_JOB:
        if not flat_pubs:
            return []
        pub_indices = list(range(len(flat_pubs)))
        return [
            {
                "start_index": 0,
                "pub_indices": pub_indices,
                "num_pubs": len(flat_pubs),
                "payload_bytes": sum(
                    metadata["payload_bytes"] for metadata in flat_pub_metadata
                ),
                "estimated_qpu_seconds": PER_JOB_QPU_OVERHEAD_SECONDS + sum(
                    metadata["estimated_qpu_seconds"]
                    for metadata in flat_pub_metadata
                ),
                "pubs": list(flat_pubs),
                "metadata": list(flat_pub_metadata),
            }
        ]

    pub_blocks = []
    current_indices = []
    current_payload_bytes = 0
    current_qpu_seconds = PER_JOB_QPU_OVERHEAD_SECONDS

    def flush_current():
        nonlocal current_indices, current_payload_bytes, current_qpu_seconds
        if not current_indices:
            return
        pub_blocks.append(
            {
                "start_index": current_indices[0],
                "pub_indices": current_indices,
                "num_pubs": len(current_indices),
                "payload_bytes": current_payload_bytes,
                "estimated_qpu_seconds": current_qpu_seconds,
            }
        )
        current_indices = []
        current_payload_bytes = 0
        current_qpu_seconds = PER_JOB_QPU_OVERHEAD_SECONDS

    for pub_index, metadata in enumerate(flat_pub_metadata):
        pub_payload_bytes = metadata["payload_bytes"]
        pub_qpu_seconds = metadata["estimated_qpu_seconds"]
        max_pubs_reached = (
            MAX_PUBS_PER_ESTIMATOR_JOB is not None
            and len(current_indices) >= MAX_PUBS_PER_ESTIMATOR_JOB
        )
        payload_limit_reached = (
            current_indices
            and current_payload_bytes + pub_payload_bytes > MAX_PAYLOAD_BYTES_PER_ESTIMATOR_JOB
        )
        runtime_limit_reached = (
            current_indices
            and current_qpu_seconds + pub_qpu_seconds > MAX_ESTIMATED_QPU_SECONDS_PER_ESTIMATOR_JOB
        )

        if max_pubs_reached or payload_limit_reached or runtime_limit_reached:
            flush_current()

        current_indices.append(pub_index)
        current_payload_bytes += pub_payload_bytes
        current_qpu_seconds += pub_qpu_seconds

    flush_current()
    return [
        {
            **block,
            "pubs": [flat_pubs[index] for index in block["pub_indices"]],
            "metadata": [flat_pub_metadata[index] for index in block["pub_indices"]],
        }
        for block in pub_blocks
    ]


def _as_scalar(value):
    array_value = np.asarray(value)
    if array_value.size == 1:
        return array_value.item()
    return np.sum(array_value)


def _job_id(job):
    if hasattr(job, "job_id"):
        return job.job_id()
    return None


def _job_metrics(job):
    if hasattr(job, "metrics"):
        try:
            return job.metrics()
        except Exception as exc:
            return {"metrics_error": str(exc)}
    return {}


def _job_usage_seconds(job):
    if hasattr(job, "usage"):
        try:
            return job.usage()
        except Exception:
            pass

    metrics = _job_metrics(job)
    usage = metrics.get("usage", {}) if isinstance(metrics, dict) else {}
    return usage.get("quantum_seconds")


def _submit_estimator_pub_blocks(pub_blocks, estimator, output_dir=None):
    RAW_SIMULATOR_COUNTS.reset()
    SIMULATION_PROGRESS.configure(
        [metadata for block in pub_blocks for metadata in block["metadata"]]
    )
    submitted_blocks = []
    for block_index, block in enumerate(pub_blocks):
        print(
            f"Submitting estimator block {block_index + 1}/{len(pub_blocks)} "
            f"with {block['num_pubs']} PUBs, "
            f"{block['payload_bytes'] / (1024 ** 2):.2f} MB payload, "
            f"~{block['estimated_qpu_seconds']:.1f}s estimated QPU time"
        )
        submitted_at = time.perf_counter()
        job, job_id = run_pub(block["pubs"], estimator)
        submitted_blocks.append(
            {
                **block,
                "job": job,
                "job_id": job_id,
                "submitted_at": submitted_at,
            }
        )
        print(f"Submitted block {block_index + 1} job_id={job_id}")
        _write_job_ids(output_dir, submitted_blocks)
    return submitted_blocks


def _retrieve_estimator_pub_block_results(submitted_blocks):
    estimates_by_uv = {}
    variance_by_uv = {}
    block_results = []
    job_benchmarks = []
    fragment_results = []

    if not submitted_blocks:
        return (
            estimates_by_uv,
            variance_by_uv,
            block_results,
            job_benchmarks,
            fragment_results,
        )

    for block_index, block in enumerate(submitted_blocks):
        print(
            f"Retrieving estimator block {block_index + 1}/{len(submitted_blocks)} "
            f"job_id={block['job_id']}"
        )

        estimates, sds, results = retrieve_results(block["job"])
        completed_at = time.perf_counter()
        block_results.append(results)

        metadata_block = block["metadata"]
        if len(estimates) != len(metadata_block) or len(sds) != len(metadata_block):
            raise RuntimeError(
                "Estimator result length mismatch for block "
                f"{block_index + 1}: submitted {len(metadata_block)} PUBs, "
                f"received {len(estimates)} estimates and {len(sds)} stds."
            )

        for metadata, estimate, sd in zip(metadata_block, estimates, sds):
            uv = metadata["uv"]
            estimate = _as_scalar(estimate)
            sd = _as_scalar(sd)

            estimates_by_uv[uv] = estimates_by_uv.get(uv, 0) + estimate
            variance_by_uv[uv] = variance_by_uv.get(uv, 0) + sd ** 2
            fragment_results.append(
                {
                    **metadata,
                    "estimate": estimate,
                    "std": sd,
                }
            )

        usage_seconds = _job_usage_seconds(block["job"])
        job_benchmarks.append(
            {
                "job_id": block["job_id"],
                "num_pubs": block["num_pubs"],
                "payload_bytes": block["payload_bytes"],
                "estimated_qpu_seconds": block["estimated_qpu_seconds"],
                "actual_qpu_seconds": usage_seconds,
                "wall_seconds": completed_at - block["submitted_at"],
                "metrics": _job_metrics(block["job"]),
            }
        )

    return (
        estimates_by_uv,
        variance_by_uv,
        block_results,
        job_benchmarks,
        fragment_results,
    )


def _print_job_benchmarks(job_benchmarks):
    if not job_benchmarks:
        print("No estimator jobs were submitted.")
        return

    estimated_qpu_seconds = sum(job["estimated_qpu_seconds"] for job in job_benchmarks)
    actual_qpu_seconds = [
        job["actual_qpu_seconds"]
        for job in job_benchmarks
        if job["actual_qpu_seconds"] is not None
    ]
    total_actual_qpu_seconds = (
        sum(actual_qpu_seconds)
        if actual_qpu_seconds
        else None
    )
    payload_mb = sum(job["payload_bytes"] for job in job_benchmarks) / (1024 ** 2)
    wall_seconds = sum(job["wall_seconds"] for job in job_benchmarks)

    print(
        "Estimator job benchmark: "
        f"{len(job_benchmarks)} jobs, "
        f"{payload_mb:.2f} MB total payload, "
        f"~{estimated_qpu_seconds:.1f}s estimated QPU time, "
        f"{wall_seconds:.1f}s local elapsed wait time"
    )
    if total_actual_qpu_seconds is not None:
        print(f"Actual reported QPU time: {total_actual_qpu_seconds:.3f}s")

    for index, job in enumerate(job_benchmarks, start=1):
        actual = job["actual_qpu_seconds"]
        actual_text = f"{actual:.3f}s" if actual is not None else "unavailable"
        print(
            f"  job {index}: id={job['job_id']}, "
            f"PUBs={job['num_pubs']}, "
            f"payload={job['payload_bytes'] / (1024 ** 2):.2f} MB, "
            f"estimated={job['estimated_qpu_seconds']:.1f}s, "
            f"actual={actual_text}, "
            f"wall={job['wall_seconds']:.1f}s"
        )


def _format_matrix(matrix):
    return np.array2string(matrix, precision=12, suppress_small=False)


def _format_dict(dictionary):
    return "\n".join(f"  {key}: {value}" for key, value in dictionary.items())


def _entry_standard_errors(sig_frag_mat, shot_alloc):
    """Return ideal sampling standard errors for the retained fragments."""
    standard_errors = {}
    for uv, sigmas in sig_frag_mat.items():
        shots = shot_alloc[uv]
        if len(sigmas) != len(shots):
            raise ValueError(
                f"Fragment count mismatch for matrix entry {uv}: "
                f"{len(sigmas)} standard deviations and {len(shots)} shot allocations."
            )
        variance = sum(
            float(np.real(sigma)) ** 2 / int(fragment_shots)
            for sigma, fragment_shots in zip(sigmas, shots)
            if int(fragment_shots) > 0
        )
        standard_errors[uv] = np.sqrt(variance)
    return standard_errors


def _ideal_fragment_estimates(decomp, state, num_qubits):
    """Return the exact mean of every measured Hamiltonian fragment."""
    estimates = []
    for operator in decomp:
        sparse_operator = get_sparse_operator(operator, num_qubits)
        value = (
            state.conjugate() @ sparse_operator @ state.T
        ).toarray()[0, 0]
        estimates.append(np.real_if_close(value).item())
    return estimates


def _matrix_entry_records(
    ideal_matrix,
    noisy_matrix,
    ideal_std_dev_dict,
    noisy_std_dev_dict,
):
    """Create one reusable record for every matrix position."""
    records = []
    for i in range(ideal_matrix.shape[0]):
        for j in range(ideal_matrix.shape[1]):
            uv = (i, j)
            measurement_uv = (max(i, j), min(i, j))
            ideal = ideal_matrix[i, j]
            noisy = noisy_matrix[i, j]
            records.append(
                {
                    "uv": uv,
                    "measurement_uv": measurement_uv,
                    "row": i,
                    "column": j,
                    "ideal_estimate": ideal,
                    "noisy_estimate": noisy,
                    "ideal_bias": 0.0,
                    "noisy_bias": noisy - ideal,
                    "ideal_std": float(ideal_std_dev_dict.get(measurement_uv, 0.0)),
                    "noisy_std": float(noisy_std_dev_dict.get(measurement_uv, 0.0)),
                    "is_quantum": measurement_uv in ideal_std_dev_dict,
                }
            )
    return records


def _print_coefficients_and_diagonal_biases(
    ideal_matrix,
    noisy_matrix,
    ideal_eigenvector,
    noisy_eigenvector,
    noisy_std_dev_dict,
    labels,
):
    """Print basis weights and diagonal-entry errors after diagonalization."""
    print("\nBasis weights and diagonal-entry biases")
    print(
        " i  basis state                         "
        "ideal |c_i|^2   noisy |c_i|^2   diagonal bias      diag. std"
    )
    for i, label in enumerate(labels):
        ideal_weight = float(abs(ideal_eigenvector[i]) ** 2)
        noisy_weight = float(abs(noisy_eigenvector[i]) ** 2)
        diagonal_bias = float(
            np.real(noisy_matrix[i, i] - ideal_matrix[i, i])
        )
        diagonal_std = float(np.real(noisy_std_dev_dict.get((i, i), 0.0)))
        print(
            f"{i:2d}  {label:<34}"
            f"{ideal_weight:14.8g}  {noisy_weight:14.8g}  "
            f"{diagonal_bias:16.8g}  {diagonal_std:12.8g}"
        )
    print(
        "    sums"
        f"{np.sum(np.abs(ideal_eigenvector) ** 2):43.12g}"
        f"{np.sum(np.abs(noisy_eigenvector) ** 2):17.12g}"
    )


def _fragment_artifact_records(flat_pubs, flat_pub_metadata, fragment_results):
    """Join every submitted fragment's circuit, observable, shots, and result."""
    results_by_pub_index = {
        result["pub_index"]: result for result in fragment_results
    }
    raw_by_pub_index = {}
    for raw_record in RAW_SIMULATOR_COUNTS.snapshot():
        raw_by_pub_index.setdefault(raw_record["pub_index"], []).append(
            raw_record
        )
    records = []
    for pub_index, (pub, metadata) in enumerate(zip(flat_pubs, flat_pub_metadata)):
        result = results_by_pub_index.get(pub_index, {})
        circuit, observable, parameter_values, precision = pub
        records.append(
            {
                **metadata,
                "pub_index": pub_index,
                "circuit": circuit,
                "observable": observable,
                "parameter_values": parameter_values,
                "target_precision": precision,
                "ideal_fragment_estimate": metadata[
                    "ideal_fragment_estimate"
                ],
                "ideal_fragment_std": metadata["sig"],
                "allocated_shots": metadata["shots"],
                "noisy_fragment_estimate": result.get("estimate"),
                "noisy_fragment_std": result.get("std"),
                "readout_mitigation_enabled": enable_readout_mitigation,
                "raw_measurement_histograms": raw_by_pub_index.get(
                    pub_index,
                    [],
                ),
            }
        )
    return records


def _portable_estimator_block_results(estimator_block_results):
    """Convert Qiskit PubResult/DataBin objects into reliably pickleable data."""
    return [
        [
            {
                "evs": np.asarray(result.data.evs),
                "stds": np.asarray(result.data.stds),
                "metadata": dict(result.metadata or {}),
            }
            for result in block_results
        ]
        for block_results in estimator_block_results
    ]


def _save_matrix_entry_csv(path, matrix_entry_records):
    fieldnames = [
        "row",
        "column",
        "measurement_row",
        "measurement_column",
        "ideal_estimate_real",
        "ideal_estimate_imag",
        "noisy_estimate_real",
        "noisy_estimate_imag",
        "ideal_bias_real",
        "ideal_bias_imag",
        "noisy_bias_real",
        "noisy_bias_imag",
        "ideal_std",
        "noisy_std",
        "is_quantum",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in matrix_entry_records:
            writer.writerow(
                {
                    "row": record["row"],
                    "column": record["column"],
                    "measurement_row": record["measurement_uv"][0],
                    "measurement_column": record["measurement_uv"][1],
                    "ideal_estimate_real": np.real(record["ideal_estimate"]),
                    "ideal_estimate_imag": np.imag(record["ideal_estimate"]),
                    "noisy_estimate_real": np.real(record["noisy_estimate"]),
                    "noisy_estimate_imag": np.imag(record["noisy_estimate"]),
                    "ideal_bias_real": np.real(record["ideal_bias"]),
                    "ideal_bias_imag": np.imag(record["ideal_bias"]),
                    "noisy_bias_real": np.real(record["noisy_bias"]),
                    "noisy_bias_imag": np.imag(record["noisy_bias"]),
                    "ideal_std": record["ideal_std"],
                    "noisy_std": record["noisy_std"],
                    "is_quantum": record["is_quantum"],
                }
            )


def _plot_matrix_entry_bias(output_path, matrix_entry_records):
    """Plot ideal/noisy entry bias over the full matrix grid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    quantum_mask = np.array(
        [record["is_quantum"] for record in matrix_entry_records],
        dtype=bool,
    )
    classical_mask = ~quantum_mask
    rows = np.array([record["row"] for record in matrix_entry_records])
    columns = np.array([record["column"] for record in matrix_entry_records])
    ideal_bias = np.zeros(np.count_nonzero(quantum_mask))
    noisy_bias = np.real(
        np.array([record["noisy_bias"] for record in matrix_entry_records])
    )[quantum_mask]
    ideal_std = np.array(
        [record["ideal_std"] for record in matrix_entry_records]
    )[quantum_mask]
    noisy_std = np.array(
        [record["noisy_std"] for record in matrix_entry_records]
    )[quantum_mask]
    quantum_rows = rows[quantum_mask]
    quantum_columns = columns[quantum_mask]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    offset = 0.10
    ax.errorbar(
        quantum_rows - offset,
        quantum_columns,
        ideal_bias,
        zerr=ideal_std,
        fmt="o",
        markersize=2.5,
        markeredgewidth=0.4,
        elinewidth=0.75,
        color="tab:blue",
        ecolor="tab:blue",
        alpha=0.85,
        capsize=1.5,
        label="Ideal (bias = 0)",
    )
    ax.errorbar(
        quantum_rows + offset,
        quantum_columns,
        noisy_bias,
        zerr=noisy_std,
        fmt="o",
        markersize=2.5,
        markeredgewidth=0.4,
        elinewidth=0.75,
        color="tab:orange",
        ecolor="tab:orange",
        alpha=0.85,
        capsize=1.5,
        label="Noisy",
    )
    if np.any(classical_mask):
        ax.scatter(
            rows[classical_mask],
            columns[classical_mask],
            np.zeros(np.count_nonzero(classical_mask)),
            s=7,
            marker="x",
            linewidths=0.5,
            color="0.35",
            alpha=0.20,
            label="Fully classical (exact)",
        )
    ax.set_xlabel("Matrix row $i$", labelpad=10)
    ax.set_ylabel("Matrix column $j$", labelpad=10)
    ax.set_zlabel("Entry bias (real part)", labelpad=12)
    ax.set_title("Ideal and noisy matrix-entry bias with standard errors")
    ax.view_init(elev=24, azim=-55)
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99))
    fig.subplots_adjust(left=0.03, right=0.86, bottom=0.08, top=0.92)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def _write_run_summary(
    output_dir,
    runtime_summary,
    job_benchmarks,
    estimator_matrix,
    std_dev_dict,
    final_summary,
):
    if output_dir is None:
        return

    lines = [
        "Q-SENSE estimator run summary",
        f"created_at_utc: {datetime.now(timezone.utc).isoformat()}",
        f"molecule: {molecule}",
        f"bond_length: {bond_length}",
        f"basis_variant: {BASIS_VARIANT}",
        f"methodtag: {methodtag}",
        f"backend: {_backend_name(backend)}",
        f"runtime_execution_mode: {RUNTIME_EXECUTION_MODE}",
        f"disable_runtime_mitigation: {DISABLE_RUNTIME_MITIGATION}",
        "",
        "Runtime inputs:",
        _format_dict(runtime_summary),
        "",
        "Estimator jobs:",
    ]

    for index, job in enumerate(job_benchmarks, start=1):
        actual = job["actual_qpu_seconds"]
        actual_text = f"{actual:.6f}" if actual is not None else "unavailable"
        lines.append(
            f"  {index}: job_id={job['job_id']}, PUBs={job['num_pubs']}, "
            f"payload_mb={job['payload_bytes'] / (1024 ** 2):.6f}, "
            f"estimated_qpu_seconds={job['estimated_qpu_seconds']:.6f}, "
            f"actual_qpu_seconds={actual_text}, "
            f"wall_seconds={job['wall_seconds']:.6f}"
        )

    lines.extend(
        [
            "",
            "Estimator matrix:",
            _format_matrix(estimator_matrix),
            "",
            "Matrix element standard deviations:",
            _format_dict(std_dev_dict),
            "",
            "Final summary:",
            _format_dict(final_summary),
        ]
    )

    _write_text(output_dir / "run_summary.txt", "\n".join(lines))


def _save_run_result_files(
    output_dir,
    runtime_summary,
    submitted_blocks,
    job_benchmarks,
    estimator_block_results,
    estimates_by_uv,
    variance_by_uv,
    ideal_matrix,
    classical_matrix,
    estimator_matrix,
    ideal_std_dev_dict,
    std_dev_dict,
    ideal_eigenvector,
    noisy_eigenvector,
    fragment_artifacts,
    final_summary,
):
    if output_dir is None:
        return

    _write_job_ids(output_dir, submitted_blocks)
    _write_run_summary(
        output_dir,
        runtime_summary,
        job_benchmarks,
        estimator_matrix,
        std_dev_dict,
        final_summary,
    )
    matrix_entry_records = _matrix_entry_records(
        ideal_matrix,
        estimator_matrix,
        ideal_std_dev_dict,
        std_dev_dict,
    )
    portable_estimator_results = _portable_estimator_block_results(
        estimator_block_results
    )
    _save_json(output_dir / "matrix_entries.json.gz", matrix_entry_records)
    _save_matrix_entry_csv(output_dir / "matrix_entries.csv", matrix_entry_records)
    np.savez(
        output_dir / "matrices_and_eigenvectors.npz",
        ideal_matrix=ideal_matrix,
        classical_matrix=classical_matrix,
        noisy_matrix=estimator_matrix,
        ideal_eigenvector=ideal_eigenvector,
        noisy_eigenvector=noisy_eigenvector,
    )
    _save_fragment_bundle(output_dir, fragment_artifacts)
    _save_json(
        output_dir / "experiment_inputs.json.gz",
        {
            "target_estimate_accuracy": TARGET_ESTIMATE_ACCURACY,
            "basis_metadata": basis_metadata,
            "readout_mitigation": readout_mitigation_data,
            "ideal_energy": final_summary["reference_energy"],
            "ideal_std_dev_dict": ideal_std_dev_dict,
            "numeric_arrays_file": "matrices_and_eigenvectors.npz",
            "fragment_manifest_file": "fragment_manifest.json.gz",
            "fragment_circuits_file": "fragment_circuits.qpy",
        },
    )
    raw_histogram_records = [
        {
            "uv": record["uv"],
            "fragment_index": record["fragment_index"],
            "pub_index": record["pub_index"],
            "allocated_shots": record["allocated_shots"],
            "measurement_histograms": record[
                "raw_measurement_histograms"
            ],
        }
        for record in fragment_artifacts
    ]
    _save_json(
        output_dir / "raw_measurement_histograms.json.gz",
        raw_histogram_records,
    )
    _plot_matrix_entry_bias(
        output_dir / "matrix_entry_bias.png",
        matrix_entry_records,
    )
    portable_run_results = {
        "runtime_summary": runtime_summary,
        "job_records": _job_records(submitted_blocks),
        "job_benchmarks": job_benchmarks,
        "estimator_block_results": portable_estimator_results,
        "estimates_by_uv": estimates_by_uv,
        "variance_by_uv": variance_by_uv,
        "ideal_std_dev_dict": ideal_std_dev_dict,
        "std_dev_dict": std_dev_dict,
        "matrix_entry_records": matrix_entry_records,
        "numeric_arrays_file": "matrices_and_eigenvectors.npz",
        "fragment_manifest_file": "fragment_manifest.json.gz",
        "fragment_circuits_file": "fragment_circuits.qpy",
        "raw_measurement_histograms_file": (
            "raw_measurement_histograms.json.gz"
        ),
        "target_estimate_accuracy": TARGET_ESTIMATE_ACCURACY,
        "basis_metadata": basis_metadata,
        "readout_mitigation": readout_mitigation_data,
        "final_summary": final_summary,
    }
    _save_json(output_dir / "run_results.json.gz", portable_run_results)

    if SAVE_LEGACY_PICKLE_OUTPUTS:
        _save_pickle(output_dir / "matrix_entries.pkl", matrix_entry_records)
        _save_pickle(
            output_dir / "experiment_inputs.pkl",
            {
                "target_estimate_accuracy": TARGET_ESTIMATE_ACCURACY,
                "basis_metadata": basis_metadata,
                "readout_mitigation": readout_mitigation_data,
                "ideal_matrix": ideal_matrix,
                "classical_matrix": classical_matrix,
                "ideal_energy": final_summary["reference_energy"],
                "ideal_eigenvector": ideal_eigenvector,
                "ideal_std_dev_dict": ideal_std_dev_dict,
                "fragments": fragment_artifacts,
            },
        )
        _save_pickle(output_dir / "fragment_artifacts.pkl", fragment_artifacts)
        _save_pickle(
            output_dir / "raw_measurement_histograms.pkl",
            raw_histogram_records,
        )
        _save_pickle(
            output_dir / "run_results.pkl",
            {
                **portable_run_results,
                "ideal_matrix": ideal_matrix,
                "classical_matrix": classical_matrix,
                "estimator_matrix": estimator_matrix,
                "ideal_eigenvector": ideal_eigenvector,
                "noisy_eigenvector": noisy_eigenvector,
                "fragment_artifacts_file": "fragment_artifacts.pkl",
            },
        )
    print(f"Saved run outputs to {output_dir}")


def _validate_quantum_estimates(quantum_indices, estimates_by_uv, variance_by_uv):
    missing_estimates = [uv for uv in quantum_indices if uv not in estimates_by_uv]
    missing_variances = [uv for uv in quantum_indices if uv not in variance_by_uv]

    if missing_estimates or missing_variances:
        raise RuntimeError(
            "Missing estimator results for quantum matrix elements: "
            f"estimates={missing_estimates}, variances={missing_variances}"
        )


def _load_runtime_cache(path, methodtag, backend):
    if not USE_RUNTIME_CACHE or REBUILD_RUNTIME_CACHE or not path.exists():
        return None

    with open(path, "rb") as f:
        payload = pickle.load(f)

    if payload.get("cache_version") != RUNTIME_CACHE_VERSION:
        print(f"Ignoring stale runtime cache version at {path}")
        return None

    if payload.get("backend_name") != _backend_name(backend):
        print(f"Ignoring runtime cache for different backend at {path}")
        return None

    if payload.get("methodtag") != methodtag:
        print(f"Ignoring runtime cache for different methodtag at {path}")
        return None

    if payload.get("basis_variant") != BASIS_VARIANT:
        print(f"Ignoring runtime cache for different basis at {path}")
        return None

    if payload.get("target_estimate_accuracy") != TARGET_ESTIMATE_ACCURACY:
        print(f"Ignoring runtime cache for different target accuracy at {path}")
        return None

    print(f"Loaded runtime cache from {path}")
    return payload


def _save_runtime_cache(path, payload):
    if not USE_RUNTIME_CACHE:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["cache_version"] = RUNTIME_CACHE_VERSION
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()

    with open(path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved runtime cache to {path}")


def get_parity_op(csf, quantum_qubits):
    return QubitOperator(''.join(['Z{} '.format(i) for i in range(len(quantum_qubits))]), (-1)**determine_tapered_parity(csf, quantum_qubits))


class _PhaseAdjustedCSF(CSF):
    """CSF adapter that preserves a fitted state's relative global phase."""

    def __init__(self, *args, state_phase=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_phase = float(state_phase)

    def get_tapered_full_circuit(self, quantum_indices=None):
        circuit = super().get_tapered_full_circuit(quantum_indices)
        circuit.global_phase += self.state_phase
        return circuit

    @staticmethod
    def _append_branch_phase(qc, ctrl_qubit, phase, ctrl_state):
        if abs(phase) <= 1e-14:
            return
        if ctrl_state == 0:
            qc.x(ctrl_qubit)
            qc.p(phase, ctrl_qubit)
            qc.x(ctrl_qubit)
        else:
            qc.p(phase, ctrl_qubit)

    def append_ctrl_tapered_init0_full_circuit(
        self, qc, target_qubits, ctrl_qubit, ctrl_state=1
    ):
        super().append_ctrl_tapered_init0_full_circuit(
            qc, target_qubits, ctrl_qubit, ctrl_state
        )
        self._append_branch_phase(
            qc, ctrl_qubit, self.state_phase, ctrl_state
        )

    def append_ctrl_tapered_init0_traced_full_circuit(
        self,
        qc,
        target_qubits,
        ctrl_qubit,
        quantum_indices,
        ctrl_state=1,
    ):
        super().append_ctrl_tapered_init0_traced_full_circuit(
            qc,
            target_qubits,
            ctrl_qubit,
            quantum_indices,
            ctrl_state,
        )
        self._append_branch_phase(
            qc, ctrl_qubit, self.state_phase, ctrl_state
        )


def _phase_safe_factorize(sparse_state, S_w, S_v, S_n, state_type):
    """Factorize and align an arbitrary global phase, rather than only a sign."""
    factors = factorize_state(
        sparse_state,
        S_w,
        S_v,
        S_n,
        state_type,
        correct_sign=False,
    )
    target = np.asarray(sparse_state.toarray()[0])
    reconstructed = expand_tensor_product_for_incomplete_qubit_set(factors)
    phase = np.vdot(reconstructed, target)
    if abs(phase) == 0:
        raise ValueError("Cannot phase-align a zero reconstructed state.")
    preferred_key = tuple(sorted(S_w))
    key = preferred_key if preferred_key in factors else next(iter(factors))
    factors[key] = factors[key] * phase / abs(phase)
    reconstructed = expand_tensor_product_for_incomplete_qubit_set(factors)
    if not np.allclose(reconstructed, target, atol=1e-9):
        raise ValueError("State factorization failed reconstruction.")
    return factors


def _make_split_csfs(split_payload, source_csfs, n_orb):
    retained = split_payload["retained_generators"]
    fixed_orbital = int(split_payload["fixed_orbital"])
    if any(fixed_orbital in record["support"] for record in retained):
        raise ValueError("A retained split-state generator acts on the fixed orbital.")

    split_csfs = []
    verification = []
    for fixed_value in (0, 1):
        sector = split_payload["sectors"][fixed_value]
        angles = np.asarray(sector["optimized_angles"], dtype=float)
        if len(angles) != len(retained):
            raise ValueError("Retained generators and optimized angles differ in length.")
        excitations = [
            PairedExcitationRotation(
                record["excitations"],
                float(angle),
                n_orb,
            )
            for record, angle in zip(retained, angles)
        ]
        csf = _PhaseAdjustedCSF(
            t_vec=[],
            orbitals=[],
            n_orb=n_orb,
            ne=source_csfs[0].ne,
            excitations=excitations,
            doub_occ=np.asarray(sector["reference_occupation"], dtype=int),
            state_phase=float(sector.get("phase_correction", 0.0)),
        )
        prepared = np.asarray(
            Statevector.from_instruction(csf.get_tapered_full_circuit()).data
        )
        saved = np.asarray(sector["prepared_full_statevector"])
        fidelity = float(abs(np.vdot(saved, prepared)) ** 2)
        if not np.allclose(prepared, saved, atol=1e-9):
            raise ValueError(
                f"Reconstructed split state Z{fixed_orbital}={fixed_value} "
                f"does not match the saved state (fidelity={fidelity:.16g})."
            )
        verification.append(
            {
                "fixed_value": fixed_value,
                "circuit_fidelity": fidelity,
                "reference_occupation": list(sector["reference_occupation"]),
                "num_generators": len(excitations),
            }
        )
        split_csfs.append(csf)
    return split_csfs, verification


def _load_split_payload(path):
    with open(path, "rb") as file:
        payload = pickle.load(file)
    if payload.get("format") != "qsense_fixed_orbital_split_v1":
        raise ValueError(f"Unsupported split-basis dump format in {path}.")
    if int(payload["state_index"]) != 0:
        raise ValueError("This benchmark expects the split of original state 0.")
    return payload


# load Q-SENSE basis states

molecule        = 'h2o'
bond_length     = '1.0'
methodtag       = 'qwc'
if SCIENTIFIC_METHOD == "pt":
    filename = (
        f"data/{molecule}_data/"
        f"Uext_CSF_for_Praveen_Smik_{bond_length}.dump"
    )
else:
    filename = (
        f"data/{molecule}_data/"
        f"UCSF_sym_comp_for_Praveen_Smik_{bond_length}.dump"
    )

# compile circuits and observables
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from src.expt import openfermion_to_sparse_pauli_op, run_pub, retrieve_results, calculate_matrix_std

##### fast noise free simulation
# from qiskit_aer.primitives import EstimatorV2 as AerEstimator
# from qiskit_aer import AerSimulator
# backend = AerSimulator()#FakeQuebec()
# estimator = AerEstimator()#.from_backend(backend)


##### for fast estimator with fake noise model
from qiskit.primitives import BackendEstimatorV2
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeQuebec


class ProgressAerSimulator(AerSimulator):
    """AerSimulator that reports each BackendEstimatorV2 shot group."""

    def run(self, run_input, **run_options):
        circuits = (
            [run_input]
            if isinstance(run_input, QuantumCircuit)
            else list(run_input)
        )
        group_index = SIMULATION_PROGRESS.start_group(
            circuits,
            run_options.get("shots", "default"),
        )
        job = super().run(circuits, **run_options)
        return _ProgressJob(job, SIMULATION_PROGRESS, group_index)


backend = FakeQuebec()
simulation_backend = ProgressAerSimulator.from_backend(
    backend,
    method=NOISY_SIMULATOR_METHOD,
    enable_truncation=True,
)
estimator = CapturingBackendEstimatorV2(
    backend=simulation_backend,
    options={"abelian_grouping": True},
)

##### for (slow) estimator with noise model from real device
# from qiskit_aer import AerSimulator
# backend=AerSimulator.from_backend(real_backend)
# print(f"We are using {backend.name}")
# estimator = EstimatorV2(mode=backend)

##### for real device, enable or disable mitigation methods accordingly - NOTE shots will automatically increase for mitigation methods
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2

### retrieve backend 
#service = QiskitRuntimeService() #for saved details
#real_backend = service.backend("ibm_quebec")
# from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
# service = QiskitRuntimeService()
# real_backend = service.least_busy(
#     simulator=False, operational=True, min_num_qubits=21
# )
#backend = real_backend
#estimator = EstimatorV2(mode=real_backend)
_configure_estimator_options(estimator)
#estimator.options.dynamical_decoupling.sequence_type = "XY4"

runtime_cache_path = _runtime_cache_path(molecule, bond_length, methodtag, backend)
runtime_cache = _load_runtime_cache(runtime_cache_path, methodtag, backend)

if runtime_cache is None:
    split_payload = (
        _load_split_payload(SPLIT_BASIS_DUMP)
        if SPLIT_BASIS_DUMP
        else None
    )
    with open(filename, 'rb') as f:
        source_dump_payload = pickle.load(f)
    if split_payload is not None:
        source_dump_payload = split_payload["source_dump_payload"]
    if SCIENTIFIC_METHOD == "pt":
        (
            list_list_refCSF,
            list_list_Uext_mp2_CSF,
            list_list_Uext_mp2_ampld,
            _list_list_Uext_opt_ampld,
            list_orb_rot,
            x_orbrot,
            Enuc,
            obt_spatial,
            tbt_spatial,
        ) = source_dump_payload
        CSF_tz_states = [
            state for group in list_list_refCSF for state in group
        ]
        UCSF_tz_states = [
            state for group in list_list_Uext_mp2_CSF for state in group
        ]
        W_amplitudes = [
            list_list_Uext_mp2_ampld[group_index]
            for group_index, group in enumerate(list_list_refCSF)
            for _ in group
        ]
        source_CSFs = get_Uext_csfs_from_dump(
            filename,
            verify_states=True,
            use_opt_amplitudes=False,
            verbose=True,
        )
    else:
        (
            CSF_tz_states,
            W_amplitudes,
            list_list_theta_CSF,
            list_sym_CSF_vec,
            list_UCSF_tz,
            UCSF_tz_states,
            somos_list,
            psi_GS_UCSF_smik,
            list_orb_rot,
            x_orbrot,
            Enuc,
            obt_spatial,
            tbt_spatial
        ) = source_dump_payload
        source_CSFs = get_csfs_from_dump(
            filename,
            verify_states=True,
            verbose=True,
        )
    
    # rotate orbitals and obtain Hamiltonian operator
    
    if len(list_orb_rot) != 0:
        obt, tbt = orthogonal_transform_obt_tbt(x_orbrot,list_orb_rot,obt_spatial,tbt_spatial)
    else:
        obt = obt_phys_spatial_to_spin(obt_spatial)
        tbt = tbt_phys_spatial_to_spin(tbt_spatial)
    
    Hfer    = make_short_H_ferm_op(Enuc, obt, tbt)
    Hqub    = jordan_wigner(Hfer)
    
    Nqubits = obt.shape[0]
    Norb    = Nqubits // 2
    dim     = 2 ** Nqubits
    
    # process information so that we can taper and factorize the Q-SENSE states
    
    source_configs = [
        tz_state_seniority_config(tz_state)
        for tz_state in UCSF_tz_states
    ]
    source_statevectors = [
        convert_TZ_format_to_sparse_format(dim, tz_state)
        for tz_state in UCSF_tz_states
    ]
    source_tapered_statevectors = [
        convert_dense_format_to_sparse_format(
            compress_state(psi.toarray()[0])
        )
        for psi in source_statevectors
    ]

    if split_payload is None:
        CSFs = source_CSFs
        configs = source_configs
        basis_CSF_tz_states = CSF_tz_states
        basis_W_amplitudes = W_amplitudes
        tapered_statevectors = source_tapered_statevectors
        label_prefix = "pt_state" if SCIENTIFIC_METHOD == "pt" else "original_state"
        basis_labels = [f"{label_prefix}_{i}" for i in range(len(CSFs))]
        basis_metadata = {
            "variant": BASIS_VARIANT,
            "scientific_method": SCIENTIFIC_METHOD,
            "basis_labels": basis_labels,
            "source_dump": filename,
        }
    else:
        split_CSFs, split_verification = _make_split_csfs(
            split_payload,
            source_CSFs,
            Norb,
        )
        split_W_amplitudes = [
            record["excitations"]
            for record in split_payload["retained_generators"]
        ]
        split_tapered_statevectors = [
            convert_dense_format_to_sparse_format(
                np.asarray(
                    Statevector.from_instruction(
                        split_payload["sectors"][fixed_value][
                            "fixed_sector_circuit"
                        ].reverse_bits()
                    ).data
                )
            )
            for fixed_value in (0, 1)
        ]
        CSFs = split_CSFs + source_CSFs[1:]
        configs = [
            source_configs[0],
            source_configs[0],
            *source_configs[1:],
        ]
        basis_CSF_tz_states = [
            CSF_tz_states[0],
            CSF_tz_states[0],
            *CSF_tz_states[1:],
        ]
        basis_W_amplitudes = [
            split_W_amplitudes,
            split_W_amplitudes,
            *W_amplitudes[1:],
        ]
        tapered_statevectors = [
            *split_tapered_statevectors,
            *source_tapered_statevectors[1:],
        ]
        fixed_orbital = int(split_payload["fixed_orbital"])
        basis_labels = [
            f"original_state_0_Z{fixed_orbital}_0",
            f"original_state_0_Z{fixed_orbital}_1",
            *[
                f"original_state_{i}"
                for i in range(1, len(source_CSFs))
            ],
        ]
        basis_metadata = {
            "variant": BASIS_VARIANT,
            "basis_labels": basis_labels,
            "source_dump": filename,
            "split_dump": str(SPLIT_BASIS_DUMP),
            "fixed_orbital": fixed_orbital,
            "split_state_index": 0,
            "split_state_verification": split_verification,
            "reconstruction_fidelity": float(
                split_payload["reconstruction_fidelity"]
            ),
            "orthogonality_reason": (
                "Different seniority configurations are orthogonal; the two "
                "states replacing original state 0 are separated by fixed Z."
            ),
        }

    Nstates = len(CSFs)
    mapping_function = (
        get_indices_mapping_2_wvn
        if SCIENTIFIC_METHOD == "pt"
        else get_indices_mapping_2_wvn_vo
    )
    UCSF_information = [
        mapping_function(
            basis_CSF_tz_states[i],
            basis_W_amplitudes[i],
            Norb,
        )
        for i in range(Nstates)
    ]
    
    SW_list          = [tuple([k for k, v in UCSF_information[i][0].items() if v == 'W']) for i in range(Nstates)]
    SV_list          = [tuple([k for k, v in UCSF_information[i][0].items() if v == 'V']) for i in range(Nstates)]
    SN_list          = [tuple([k for k, v in UCSF_information[i][0].items() if v == 'N']) for i in range(Nstates)]
    state_type_list  = [UCSF_information[i][1] for i in range(Nstates)]
    
    # taper and factorize the Q-SENSE basis states
    
    factorized_tapered_statevectors = [
        _phase_safe_factorize(
            tapered_statevectors[i],
            SW_list[i],
            SV_list[i],
            SN_list[i],
            state_type_list[i],
        )
        for i in range(Nstates)
    ]

    print(f"Basis variant: {BASIS_VARIANT}")
    print(f"Basis states ({Nstates}): {basis_labels}")
    if split_payload is not None:
        for fixed_value in (0, 1):
            print(
                f"Split state Z{split_payload['fixed_orbital']}="
                f"{fixed_value}: labels={UCSF_information[fixed_value][0]}, "
                f"factors={list(factorized_tapered_statevectors[fixed_value])}"
            )
    
    
    Hsub       = np.zeros([Nstates, Nstates], dtype=np.complex128)
    sig_matrix = np.zeros([Nstates, Nstates], dtype=np.complex128)
    
    def get_quantum_qubits(bra_f, ket_f, bra_labels, ket_labels):
        quantum_qubits = []
    
        join_partition, coarse_dict_bra, coarse_dict_ket = obtain_coarse_dicts(bra_f, ket_f)
        QC_assignment_dict                               = QC_assignment_from_qubit_labels(bra_labels, ket_labels, join_partition)
    
        for k, v in QC_assignment_dict.items():
            if v == 'Q':
                for idx in k:
                    quantum_qubits.append(idx)
        return quantum_qubits
    
    circuit_cx_counts = []
    circuit_depth = []
    
    ### job
    quantum_indices = [] # list of tuples indicating quantum calculated entries
    jobs = {} #i: circuit, shots 
    circuits = {} #only circuits
    jobs_desc = {} #
    sig_frag_mat = {}
    ideal_frag_estimates = {}
    frag_zops = {}
    H_classical = np.zeros([Nstates, Nstates], dtype=np.complex128) # save only classical parts
    tol=1e-5
    
    ##mitigation options TODO
    
    for i in range(Nstates):
        
        print(f'{i, i}')
        uv = (i, i)
    
        ket_f      = factorized_tapered_statevectors[i]
        ket_labels = UCSF_information[i][0]
        ket_config = configs[i]
    
        Htapered        = project_out_seniority_symmetries(Hqub, Nqubits, ket_config, ket_config)
        HQ, ketQ, _, NQ = evaluate_fully_classical_factors(ket_f, ket_f, ket_labels, ket_labels, Htapered) 
    
        if NQ == 0 or np.sum(np.abs(list(HQ.terms.values()))) <=tol:
            Hsub[i,i] = HQ.constant
            H_classical[i,i] = HQ.constant
    
        else:
            quantum_indices.append(uv)
            quantum_qubits = get_quantum_qubits(ket_f, ket_f, ket_labels, ket_labels)
    
            HQsparse   = get_sparse_operator(HQ)
            ketQ       = convert_dense_format_to_sparse_format(ketQ)
            Hsub[i,i] = (ketQ @ HQsparse @ ketQ.T)[0,0]
    
            H_classical[i, i] = HQ.constant
            HQ              -= HQ.constant
    
            HQ.compress()
    
            ### build circuits!
            decomp, meas_circuits, z_ops = get_decomp_circuits_estimators(HQ, NQ, methodtag=methodtag)
            csf_circ = CSFs[i].get_tapered_full_circuit(quantum_qubits)
    
            #assert verify_circuit_state(csf_circ, ketQ)
            #csf_circ_t = transpile(csf_circ, basis_gates=['u3', 'cx'], optimization_level=3)
    
            csf_circ_frags = [QuantumCircuit.compose(csf_circ, c) for c in meas_circuits]
            
            circuits[uv] = csf_circ_frags #make estimators for jobs?
            sig_frag_mat[uv] = frag_SD_of_decomp(decomp, ketQ, NQ, general=True)
            ideal_frag_estimates[uv] = _ideal_fragment_estimates(
                decomp,
                ketQ,
                NQ,
            )
            sig_matrix[i, i] = np.sum(sig_frag_mat[uv])
            frag_zops[uv] = z_ops
    
    for i in range(Nstates):
        for j in range(Nstates):
            if i > j:
                
                print(f'{i, j}')
                uv = (i, j)
    
                ij_shift           = 0.5 * (Hsub[i,i] + Hsub[j,j])
    
                bra_f              = factorized_tapered_statevectors[i]
                bra_labels         = UCSF_information[i][0]
                bra_config         = configs[i]
    
                ket_f              = factorized_tapered_statevectors[j]
                ket_labels         = UCSF_information[j][0]
                ket_config         = configs[j]
    
                Htapered           = project_out_seniority_symmetries(Hqub - ij_shift, Nqubits, bra_config, ket_config)
                HQ, braQ, ketQ, NQ = evaluate_fully_classical_factors(bra_f, ket_f, bra_labels, ket_labels, Htapered)
    
                if NQ == 0 or np.sum(np.abs(list(HQ.terms.values()))) <=tol:
                    Hsub[i,j] = HQ.constant
                    Hsub[j,i] = HQ.constant
    
                    #off-diagonal used twice
                    circuit_cx_counts.append(0)
                    circuit_depth.append(0)
                    circuit_cx_counts.append(0)
                    circuit_depth.append(0)
    
                else:
                    quantum_indices.append(uv)
                    quantum_qubits = get_quantum_qubits(bra_f, ket_f, bra_labels, ket_labels)
    
                    HQsparse         = get_sparse_operator(HQ, NQ)
                    braQ             = convert_dense_format_to_sparse_format(braQ)
                    ketQ             = convert_dense_format_to_sparse_format(ketQ)
                    Hsub[i,j]       = (braQ @ HQsparse @ ketQ.T)[0,0]
                    Hsub[j,i]       = (braQ @ HQsparse @ ketQ.T)[0,0]
    
                    comp             = create_composite_state(braQ, ketQ, NQ)
                    HQ_aug           = XorY_augment(HQ, NQ)
    
                    ### build circuits!
                    decomp, meas_circuits, z_ops = get_decomp_circuits_estimators(HQ_aug, NQ + 1, methodtag=methodtag)
                    csf_circ = get_parallelswap_subcircuit(CSFs[i], CSFs[j], quantum_qubits=quantum_qubits, control_qubit_pos=NQ)
                    #assert verify_circuit_state(csf_circ, comp, truncate_bitstrings=list(range(NQ+1)))
                    #csf_circ_t = transpile(csf_circ, basis_gates=['u3', 'cx'], optimization_level=3)
                    
                    csf_circ_frags = [QuantumCircuit.compose(csf_circ, c) for c in meas_circuits]
                    
                    circuits[uv] = csf_circ_frags #make estimators for jobs?
                    sig_frag_mat[uv] = frag_SD_of_decomp(decomp, comp, NQ + 1, general=True)
                    ideal_frag_estimates[uv] = _ideal_fragment_estimates(
                        decomp,
                        comp,
                        NQ + 1,
                    )
                    sig_matrix[i, j] = np.sum(sig_frag_mat[uv])
                    sig_matrix[j, i] = np.sum(sig_frag_mat[uv])
                    frag_zops[uv] = z_ops
                    
    
    vals, vecs = np.linalg.eigh(Hsub)
    Egs        = vals[0]
    c_ideal    = vecs[:,0]
    cost       = sampling_cost(c_ideal, sig_matrix)
    
    
    print(f'''
        Estimate Results:
            Method              : {SCIENTIFIC_METHOD.upper()}
            Molecule            : {molecule}
            Bond Length         : {bond_length}
            Ground State Energy : {Egs}
            Sampling Cost       : {cost}
    ''')
    
    print('''
    Beginning quantum experiments...
          ''')
    
    N = int(np.ceil(np.abs(cost / TARGET_ESTIMATE_ACCURACY ** 2)))
    shot_alloc = get_shot_allocation(sig_frag_mat, c_ideal, N)
    
    #run stuff
    
    print("Allocated {} shots, beginning quantum expts...".format(N))
    
    # Create pass manager for transpilation
    pm = generate_preset_pass_manager(optimization_level=3,
                                        backend=backend,
                                        seed_transpiler=0)
    
    circuits_transpiled = {}
    fragment_zops_transpiled = {}
    PUBs = {} # combine circuits, observables and precision required (shots)
    pub_metadata = []
    pruning_summary = {
        "allocated_shots_before_pruning": int(sum(np.sum(shots) for shots in shot_alloc.values())),
        "num_pruned_fragments": 0,
        "num_pruned_matrix_elements": 0,
    }
    executable_quantum_indices = []

    for uv in circuits.keys():
        transpiled_circuits = pm.run(circuits[uv])

        observable_transpiled = [openfermion_to_sparse_pauli_op(observable, circuit.num_qubits).apply_layout(transpiled_circuit.layout) for observable, circuit, transpiled_circuit in zip(frag_zops[uv], circuits[uv], transpiled_circuits)]

        kept_circuits = []
        kept_observables = []
        kept_sigmas = []
        kept_ideal_estimates = []
        kept_shots = []
        kept_pubs = []

        for fragment_index, (
            circuit,
            observable,
            sig,
            ideal_fragment_estimate,
            shots,
        ) in enumerate(
            zip(
                transpiled_circuits,
                observable_transpiled,
                sig_frag_mat[uv],
                ideal_frag_estimates[uv],
                shot_alloc[uv],
            )
        ):
            shots = int(np.ceil(np.real(shots)))
            sig = np.real(sig)

            if shots < MIN_FRAGMENT_SHOTS or abs(sig) <= MIN_FRAGMENT_STD:
                pruning_summary["num_pruned_fragments"] += 1
                continue

            # BackendEstimatorV2 executes ceil(1 / precision**2) shots.
            # This value therefore preserves the explicit fragment allocation.
            precision = _backend_estimator_precision_for_shots(shots)
            if not np.isfinite(precision) or precision <= 0:
                pruning_summary["num_pruned_fragments"] += 1
                continue

            kept_circuits.append(circuit)
            kept_observables.append(observable)
            kept_sigmas.append(sig)
            kept_ideal_estimates.append(ideal_fragment_estimate)
            kept_shots.append(shots)
            kept_pubs.append((circuit, observable, None, precision))
            pub_metadata.append(
                {
                    "uv": uv,
                    "fragment_index": fragment_index,
                    "sig": sig,
                    "ideal_fragment_estimate": ideal_fragment_estimate,
                    "shots": shots,
                    "precision": precision,
                }
            )

        if not kept_pubs:
            pruning_summary["num_pruned_matrix_elements"] += 1
            continue

        executable_quantum_indices.append(uv)
        circuits_transpiled[uv] = kept_circuits
        fragment_zops_transpiled[uv] = kept_observables
        sig_frag_mat[uv] = kept_sigmas
        ideal_frag_estimates[uv] = kept_ideal_estimates
        shot_alloc[uv] = kept_shots
        PUBs[uv] = kept_pubs

    quantum_indices = executable_quantum_indices
    flat_pubs, flat_pub_metadata = _flatten_pubs(PUBs, pub_metadata)
    flat_pub_metadata = _benchmark_flat_pubs(flat_pubs, flat_pub_metadata)
    pub_blocks = _build_pub_blocks(flat_pubs, flat_pub_metadata)
    pub_metadata = flat_pub_metadata
    runtime_summary = _runtime_inputs_summary(
        PUBs,
        quantum_indices,
        shot_alloc,
        circuits_transpiled,
        pub_metadata,
        pruning_summary,
        pub_blocks,
    )

    _save_runtime_cache(
        runtime_cache_path,
        {
            "molecule": molecule,
            "bond_length": bond_length,
            "methodtag": methodtag,
            "backend_name": _backend_name(backend),
            "basis_variant": BASIS_VARIANT,
            "scientific_method": SCIENTIFIC_METHOD,
            "basis_metadata": basis_metadata,
            "Nstates": Nstates,
            "Nqubits": Nqubits,
            "quantum_indices": quantum_indices,
            "Hsub": Hsub,
            "H_classical": H_classical,
            "sig_matrix": sig_matrix,
            "sig_frag_mat": sig_frag_mat,
            "ideal_frag_estimates": ideal_frag_estimates,
            "Egs": Egs,
            "c_ideal": c_ideal,
            "cost": cost,
            "N": N,
            "target_estimate_accuracy": TARGET_ESTIMATE_ACCURACY,
            "shot_alloc": shot_alloc,
            "circuits_transpiled": circuits_transpiled,
            "fragment_zops_transpiled": fragment_zops_transpiled,
            "PUBs": PUBs,
            "pub_metadata": pub_metadata,
            "pruning_summary": pruning_summary,
            "runtime_summary": runtime_summary,
        },
    )

else:
    basis_metadata = runtime_cache["basis_metadata"]
    Nstates = runtime_cache["Nstates"]
    quantum_indices = runtime_cache["quantum_indices"]
    Hsub = runtime_cache["Hsub"]
    H_classical = runtime_cache["H_classical"]
    sig_matrix = runtime_cache["sig_matrix"]
    sig_frag_mat = runtime_cache["sig_frag_mat"]
    ideal_frag_estimates = runtime_cache["ideal_frag_estimates"]
    Egs = runtime_cache["Egs"]
    c_ideal = runtime_cache["c_ideal"]
    cost = runtime_cache["cost"]
    N = runtime_cache["N"]
    shot_alloc = runtime_cache["shot_alloc"]
    circuits_transpiled = runtime_cache["circuits_transpiled"]
    fragment_zops_transpiled = runtime_cache["fragment_zops_transpiled"]
    PUBs = runtime_cache["PUBs"]
    pub_metadata = runtime_cache["pub_metadata"]
    pruning_summary = runtime_cache["pruning_summary"]
    print(f"Restored {sum(len(pubs) for pubs in PUBs.values())} PUBs for {_backend_name(backend)}")

if PREPARE_CACHE_ONLY:
    print(f"PT sampling cache ready: {runtime_cache_path}", flush=True)
    raise SystemExit(0)


flat_pubs, flat_pub_metadata = _flatten_pubs(PUBs, pub_metadata)
flat_pub_metadata = _benchmark_flat_pubs(flat_pubs, flat_pub_metadata)
pub_blocks = _build_pub_blocks(flat_pubs, flat_pub_metadata)
runtime_summary = _runtime_inputs_summary(
    PUBs,
    quantum_indices,
    shot_alloc,
    circuits_transpiled,
    flat_pub_metadata,
    pruning_summary,
    pub_blocks,
)
_print_runtime_summary(runtime_summary)
if CONFIRM_BEFORE_SUBMIT:
    target = "local noisy simulator" if LOCAL_SIMULATOR_SINGLE_JOB else "hardware"
    response = input(f"Submit this estimator job to the {target}? Type 'yes' to continue: ")
    if response.strip().lower() != "yes":
        raise SystemExit("Aborted before hardware submission.")

run_output_dir = _create_run_output_dir(molecule, bond_length, methodtag, backend)
readout_mitigation_data = {
    "enabled": False,
    "model": None,
    "reason": "enable_readout_mitigation is False",
}
if enable_readout_mitigation:
    if not isinstance(estimator, CapturingBackendEstimatorV2):
        raise RuntimeError(
            "The explicit readout stack currently requires the local "
            "CapturingBackendEstimatorV2 path."
        )
    readout_qubits = _readout_support_qubits(flat_pubs)
    print(
        "Calibrating local readout mitigation on physical qubits "
        f"{readout_qubits} with {READOUT_CALIBRATION_SHOTS} shots per "
        "calibration circuit...",
        flush=True,
    )
    readout_mitigator, readout_mitigation_data = (
        run_local_readout_calibration(
            backend,
            readout_qubits,
            shots=READOUT_CALIBRATION_SHOTS,
            seed_simulator=READOUT_CALIBRATION_SEED,
            simulator_method=NOISY_SIMULATOR_METHOD,
        )
    )
    estimator.set_readout_mitigator(readout_mitigator)
    print("Local assignment-matrix readout mitigation enabled.", flush=True)

if run_output_dir is not None:
    _save_json(
        run_output_dir / "readout_mitigation.json.gz",
        readout_mitigation_data,
    )
    np.savez(
        run_output_dir / "runtime_reference_arrays.npz",
        ideal_matrix=Hsub,
        classical_matrix=H_classical,
        ideal_eigenvector=c_ideal,
    )
    _save_runtime_pub_bundle(
        run_output_dir,
        flat_pubs,
        flat_pub_metadata,
    )
    portable_runtime_inputs = {
        "runtime_summary": runtime_summary,
        "pub_metadata": flat_pub_metadata,
        "pub_blocks": [
            {
                key: value
                for key, value in block.items()
                if key not in {"pubs", "metadata"}
            }
            for block in pub_blocks
        ],
        "quantum_indices": quantum_indices,
        "basis_metadata": basis_metadata,
        "readout_mitigation": readout_mitigation_data,
        "target_estimate_accuracy": TARGET_ESTIMATE_ACCURACY,
        "ideal_energy": Egs,
        "shot_alloc": shot_alloc,
        "sig_frag_mat": sig_frag_mat,
        "ideal_frag_estimates": ideal_frag_estimates,
        "numeric_arrays_file": "runtime_reference_arrays.npz",
        "pub_manifest_file": "runtime_pub_manifest.json.gz",
        "circuits_file": "runtime_circuits.qpy",
    }
    _save_json(
        run_output_dir / "runtime_inputs.json.gz",
        portable_runtime_inputs,
    )
    if SAVE_LEGACY_PICKLE_OUTPUTS:
        _save_pickle(
            run_output_dir / "readout_mitigation.pkl",
            readout_mitigation_data,
        )
        _save_pickle(
            run_output_dir / "runtime_inputs.pkl",
            {
                "runtime_summary": runtime_summary,
                "pub_metadata": flat_pub_metadata,
                "pub_blocks": [
                    {
                        key: value
                        for key, value in block.items()
                        if key not in {"pubs", "metadata"}
                    }
                    for block in pub_blocks
                ],
                "quantum_indices": quantum_indices,
                "basis_metadata": basis_metadata,
                "readout_mitigation": readout_mitigation_data,
                "target_estimate_accuracy": TARGET_ESTIMATE_ACCURACY,
                "ideal_matrix": Hsub,
                "classical_matrix": H_classical,
                "ideal_energy": Egs,
                "ideal_eigenvector": c_ideal,
                "shot_alloc": shot_alloc,
                "sig_frag_mat": sig_frag_mat,
                "ideal_frag_estimates": ideal_frag_estimates,
                "circuits_transpiled": circuits_transpiled,
                "fragment_zops_transpiled": fragment_zops_transpiled,
                "PUBs": PUBs,
            },
        )
    _write_text(
        run_output_dir / "runtime_inputs.txt",
        "\n".join(
            [
                "Runtime input benchmark",
                f"created_at_utc: {datetime.now(timezone.utc).isoformat()}",
                f"molecule: {molecule}",
                f"bond_length: {bond_length}",
                f"basis_variant: {BASIS_VARIANT}",
                f"methodtag: {methodtag}",
                f"backend: {_backend_name(backend)}",
                f"runtime_execution_mode: {RUNTIME_EXECUTION_MODE}",
                f"disable_runtime_mitigation: {DISABLE_RUNTIME_MITIGATION}",
                f"enable_readout_mitigation: "
                f"{enable_readout_mitigation}",
                "",
                _format_dict(runtime_summary),
            ]
        ),
    )
    print(f"Saving run outputs under {run_output_dir}")

with _estimator_execution_context(backend, estimator) as active_estimator:
    submitted_blocks = _submit_estimator_pub_blocks(
        pub_blocks,
        active_estimator,
        output_dir=run_output_dir,
    )

(
    estimates_by_uv,
    variance_by_uv,
    estimator_block_results,
    job_benchmarks,
    fragment_results,
) = _retrieve_estimator_pub_block_results(submitted_blocks)
_print_job_benchmarks(job_benchmarks)
_validate_quantum_estimates(quantum_indices, estimates_by_uv, variance_by_uv)

estimator_matrix = np.zeros([Nstates, Nstates], dtype=np.complex128)
std_dev_dict = {} # uv : dev
for i in range(Nstates):
    #diagonal
    j = i
    uv = (i, j)
    if uv in quantum_indices:
        estimator_matrix[i, j] = estimates_by_uv[uv] + H_classical[i, j]
        std_dev_dict[uv] = np.sqrt(variance_by_uv[uv]) #sqrt of variance
        print("Estimator estimate: {} +- {}".format(estimator_matrix[i, j], std_dev_dict[uv]))
    else:
        estimator_matrix[i, i] = Hsub[i, i]
    
    #off-diagonal
    for j in range(i):
        uv = (i, j)
        print(uv)
        if uv in quantum_indices:
            estimator_matrix[i, j] = estimates_by_uv[uv] + H_classical[i, j]
            estimator_matrix[j, i] = estimates_by_uv[uv] + H_classical[j, i]
            std_dev_dict[uv] = np.sqrt(variance_by_uv[uv]) #sqrt of variance
            print("Estimator estimate: {} +- {}".format(estimator_matrix[i, j], std_dev_dict[uv]))

        else:
            estimator_matrix[i, j] = Hsub[i, j]
            estimator_matrix[j, i] = Hsub[j, i]

vals, vecs = np.linalg.eigh(estimator_matrix)
c_noisy    = vecs[:,0]
std = calculate_matrix_std(c_noisy, std_dev_dict)
mean = vals[0]
bias = mean - Egs
rmse = np.sqrt(bias **2 + std ** 2)
ideal_std_dev_dict = _entry_standard_errors(
    {uv: sig_frag_mat[uv] for uv in quantum_indices},
    {uv: shot_alloc[uv] for uv in quantum_indices},
)
fragment_artifacts = _fragment_artifact_records(
    flat_pubs,
    flat_pub_metadata,
    fragment_results,
)
final_summary = {
    "mean_energy_estimate": mean,
    "std": std,
    "bias": bias,
    "rmse": rmse,
    "reference_energy": Egs,
    "target_estimate_accuracy": TARGET_ESTIMATE_ACCURACY,
    "allocated_shots": N,
    "readout_mitigation_enabled": enable_readout_mitigation,
    "readout_calibration_shots": (
        2 * READOUT_CALIBRATION_SHOTS
        if enable_readout_mitigation
        else 0
    ),
}
print("""mean energy estimate from samples: {}\n
        std: {}      
        bias: {}
        RMSE: {}
      """.format(mean, std, bias, rmse))
_print_coefficients_and_diagonal_biases(
    Hsub,
    estimator_matrix,
    c_ideal,
    c_noisy,
    std_dev_dict,
    basis_metadata["basis_labels"],
)
_save_run_result_files(
    run_output_dir,
    runtime_summary,
    submitted_blocks,
    job_benchmarks,
    estimator_block_results,
    estimates_by_uv,
    variance_by_uv,
    Hsub,
    H_classical,
    estimator_matrix,
    ideal_std_dev_dict,
    std_dev_dict,
    c_ideal,
    c_noisy,
    fragment_artifacts,
    final_summary,
)
