### script that does sampling currently
# SV can be added

### 18/11/2025 VO create fragments and job, job_desc

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
    get_indices_mapping_2_wvn_vo,
    factorize_state,
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

import numpy as np
import pickle
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.circuits.circuits_csf import get_csfs_from_dump
from src.circuits.utils_circuit import get_decomp_circuits_estimators
from qiskit import QuantumCircuit
from src.circuits.circuits_swap import get_parallelswap_subcircuit

from src.expt import get_shot_allocation
from src.measurement_new.utils_states import frag_SD_of_decomp

from src.mitigation import determine_tapered_parity
from openfermion import QubitOperator

RUNTIME_CACHE_VERSION = 4
USE_RUNTIME_CACHE = True
REBUILD_RUNTIME_CACHE = True
RUNTIME_CACHE_DIR = Path("saved/runtime_cache")
RUN_OUTPUT_DIR = Path("saved/runtime_runs")
SAVE_RUN_OUTPUTS = True
MIN_FRAGMENT_SHOTS = 1
MIN_FRAGMENT_STD = 1e-12
RUNTIME_EXECUTION_MODE = "batch" # "job", "batch", or "session"
DISABLE_RUNTIME_MITIGATION = True
RUNTIME_JOB_TAGS = ["qsense_quebec_estimator"]
CONFIRM_BEFORE_SUBMIT = True
MAX_PAYLOAD_BYTES_PER_ESTIMATOR_JOB = 20 * 1024 * 1024
MAX_ESTIMATED_QPU_SECONDS_PER_ESTIMATOR_JOB = 600
PER_JOB_QPU_OVERHEAD_SECONDS = 2
SECONDS_PER_EXECUTION_ESTIMATE = 0.00035
MAX_PUBS_PER_ESTIMATOR_JOB = None


def _safe_cache_token(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def _backend_name(backend):
    name = getattr(backend, "name", backend.__class__.__name__)
    return name() if callable(name) else name


def _runtime_cache_path(molecule, bond_length, methodtag, backend):
    method_token = _safe_cache_token(methodtag)
    backend_token = _safe_cache_token(_backend_name(backend))
    return RUNTIME_CACHE_DIR / f"sampling_and_sv_{molecule}_{bond_length}_{method_token}_{backend_token}.pkl"


def _create_run_output_dir(molecule, bond_length, methodtag, backend):
    if not SAVE_RUN_OUTPUTS:
        return None

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    method_token = _safe_cache_token(methodtag)
    backend_token = _safe_cache_token(_backend_name(backend))
    output_dir = RUN_OUTPUT_DIR / f"{run_id}_{molecule}_{bond_length}_{method_token}_{backend_token}"
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
        _set_option_path(options, ("resilience_level",), 0)
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


def _benchmark_flat_pubs(flat_pubs, flat_pub_metadata):
    for pub_index, (pub, metadata) in enumerate(zip(flat_pubs, flat_pub_metadata)):
        metadata["pub_index"] = pub_index
        metadata["payload_bytes"] = _estimate_pub_payload_bytes(pub)
        metadata["estimated_qpu_seconds"] = _estimate_pub_qpu_seconds(metadata)
    return flat_pub_metadata


def _build_pub_blocks(flat_pubs, flat_pub_metadata):
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

    if not submitted_blocks:
        return estimates_by_uv, variance_by_uv, block_results, job_benchmarks

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

    return estimates_by_uv, variance_by_uv, block_results, job_benchmarks


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
    estimator_matrix,
    std_dev_dict,
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
    saved_full_results = _save_pickle(
        output_dir / "run_results.pkl",
        {
            "runtime_summary": runtime_summary,
            "job_records": _job_records(submitted_blocks),
            "job_benchmarks": job_benchmarks,
            "estimator_block_results": estimator_block_results,
            "estimates_by_uv": estimates_by_uv,
            "variance_by_uv": variance_by_uv,
            "estimator_matrix": estimator_matrix,
            "std_dev_dict": std_dev_dict,
            "final_summary": final_summary,
        },
    )
    if not saved_full_results:
        _save_pickle(
            output_dir / "run_results_without_raw_estimator_results.pkl",
            {
                "runtime_summary": runtime_summary,
                "job_records": _job_records(submitted_blocks),
                "job_benchmarks": job_benchmarks,
                "estimates_by_uv": estimates_by_uv,
                "variance_by_uv": variance_by_uv,
                "estimator_matrix": estimator_matrix,
                "std_dev_dict": std_dev_dict,
                "final_summary": final_summary,
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

# load Q-SENSE basis states

molecule        = 'h2o'
bond_length     = '1.0'
methodtag       = 'qwc'
filename        = f'data/{molecule}_data/UCSF_sym_comp_for_Praveen_Smik_{bond_length}.dump'

# compile circuits and observables
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from src.expt import openfermion_to_sparse_pauli_op, run_pub, retrieve_results, calculate_matrix_std, get_precision_for_shots_std

##### fast noise free simulation
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_aer import AerSimulator
backend = AerSimulator()#FakeQuebec()
estimator = AerEstimator()#.from_backend(backend)


##### for fast estimator with fake noise model
# from qiskit_aer.primitives import EstimatorV2 as AerEstimator
# from qiskit_ibm_runtime.fake_provider import FakeQuebec
# backend = FakeQuebec()
# estimator = AerEstimator().from_backend(backend)

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
    with open(filename, 'rb') as f:
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
        ) = pickle.load(f)
    
    CSFs = get_csfs_from_dump(filename, verify_states=True, verbose=True)
    
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
    
    Nstates          = len(UCSF_tz_states)
    configs          = [tz_state_seniority_config(tz_state) for tz_state in UCSF_tz_states]
    UCSF_information = [get_indices_mapping_2_wvn_vo(CSF_tz_states[i], W_amplitudes[i], Norb) for i in range(Nstates)]
    
    SW_list          = [tuple([k for k, v in UCSF_information[i][0].items() if v == 'W']) for i in range(Nstates)]
    SV_list          = [tuple([k for k, v in UCSF_information[i][0].items() if v == 'V']) for i in range(Nstates)]
    SN_list          = [tuple([k for k, v in UCSF_information[i][0].items() if v == 'N']) for i in range(Nstates)]
    state_type_list  = [UCSF_information[i][1] for i in range(Nstates)]
    
    # taper and factorize the Q-SENSE basis states
    
    statevectors                    = [convert_TZ_format_to_sparse_format(dim, tz_state) for tz_state in UCSF_tz_states]
    tapered_statevectors            = [convert_dense_format_to_sparse_format(compress_state(psi.toarray()[0])) for psi in statevectors]
    factorized_tapered_statevectors = [factorize_state(tapered_statevectors[i], SW_list[i], SV_list[i], SN_list[i], state_type_list[i]) 
                                       for i in range(Nstates)]
    
    
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
                    sig_matrix[i, j] = np.sum(sig_frag_mat[uv])
                    sig_matrix[j, i] = np.sum(sig_frag_mat[uv])
                    frag_zops[uv] = z_ops
                    
    
    vals, vecs = np.linalg.eigh(Hsub)
    Egs        = vals[0]
    c          = vecs[:,0]
    cost       = sampling_cost(c, sig_matrix)
    
    
    print(f'''
        Estimate Results:
            Method              : {'VO'}
            Molecule            : {molecule}
            Bond Length         : {bond_length}
            Ground State Energy : {Egs}
            Sampling Cost       : {cost}
    ''')
    
    print('''
    Beginning quantum experiments...
          ''')
    
    N = int(cost * 1e6)
    shot_alloc = get_shot_allocation(sig_frag_mat, c, N)
    
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
        kept_shots = []
        kept_pubs = []

        for fragment_index, (circuit, observable, sig, shots) in enumerate(
            zip(transpiled_circuits, observable_transpiled, sig_frag_mat[uv], shot_alloc[uv])
        ):
            shots = int(np.ceil(np.real(shots)))
            sig = np.real(sig)

            if shots < MIN_FRAGMENT_SHOTS or abs(sig) <= MIN_FRAGMENT_STD:
                pruning_summary["num_pruned_fragments"] += 1
                continue

            precision = get_precision_for_shots_std(sig, shots)
            if not np.isfinite(precision) or precision <= 0:
                pruning_summary["num_pruned_fragments"] += 1
                continue

            kept_circuits.append(circuit)
            kept_observables.append(observable)
            kept_sigmas.append(sig)
            kept_shots.append(shots)
            kept_pubs.append((circuit, observable, None, precision))
            pub_metadata.append(
                {
                    "uv": uv,
                    "fragment_index": fragment_index,
                    "sig": sig,
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
            "Nstates": Nstates,
            "Nqubits": Nqubits,
            "quantum_indices": quantum_indices,
            "Hsub": Hsub,
            "H_classical": H_classical,
            "sig_matrix": sig_matrix,
            "sig_frag_mat": sig_frag_mat,
            "Egs": Egs,
            "c": c,
            "cost": cost,
            "N": N,
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
    Nstates = runtime_cache["Nstates"]
    quantum_indices = runtime_cache["quantum_indices"]
    Hsub = runtime_cache["Hsub"]
    H_classical = runtime_cache["H_classical"]
    sig_matrix = runtime_cache["sig_matrix"]
    sig_frag_mat = runtime_cache["sig_frag_mat"]
    Egs = runtime_cache["Egs"]
    c = runtime_cache["c"]
    cost = runtime_cache["cost"]
    N = runtime_cache["N"]
    shot_alloc = runtime_cache["shot_alloc"]
    circuits_transpiled = runtime_cache["circuits_transpiled"]
    fragment_zops_transpiled = runtime_cache["fragment_zops_transpiled"]
    PUBs = runtime_cache["PUBs"]
    pub_metadata = runtime_cache["pub_metadata"]
    pruning_summary = runtime_cache["pruning_summary"]
    print(f"Restored {sum(len(pubs) for pubs in PUBs.values())} PUBs for {_backend_name(backend)}")


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
    response = input("Submit these Runtime jobs to hardware? Type 'yes' to continue: ")
    if response.strip().lower() != "yes":
        raise SystemExit("Aborted before hardware submission.")

run_output_dir = _create_run_output_dir(molecule, bond_length, methodtag, backend)
if run_output_dir is not None:
    _save_pickle(
        run_output_dir / "runtime_inputs.pkl",
        {
            "runtime_summary": runtime_summary,
            "pub_metadata": flat_pub_metadata,
            "pub_blocks": [
                {key: value for key, value in block.items() if key not in {"pubs", "metadata"}}
                for block in pub_blocks
            ],
            "quantum_indices": quantum_indices,
            "shot_alloc": shot_alloc,
            "sig_frag_mat": sig_frag_mat,
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
                f"methodtag: {methodtag}",
                f"backend: {_backend_name(backend)}",
                f"runtime_execution_mode: {RUNTIME_EXECUTION_MODE}",
                f"disable_runtime_mitigation: {DISABLE_RUNTIME_MITIGATION}",
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

estimates_by_uv, variance_by_uv, estimator_block_results, job_benchmarks = (
    _retrieve_estimator_pub_block_results(submitted_blocks)
)
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
c          = vecs[:,0]
std = calculate_matrix_std(c, std_dev_dict)
mean = vals[0]
bias = mean - Egs
rmse = np.sqrt(bias **2 + std ** 2)
final_summary = {
    "mean_energy_estimate": mean,
    "std": std,
    "bias": bias,
    "rmse": rmse,
    "reference_energy": Egs,
}
print("""mean energy estimate from samples: {}\n
        std: {}      
        bias: {}
        RMSE: {}
      """.format(mean, std, bias, rmse))
_save_run_result_files(
    run_output_dir,
    runtime_summary,
    submitted_blocks,
    job_benchmarks,
    estimator_block_results,
    estimates_by_uv,
    variance_by_uv,
    estimator_matrix,
    std_dev_dict,
    final_summary,
)
