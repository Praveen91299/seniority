"""Readout-mitigated probabilistic linear ZNE for a cached Q-SENSE matrix.

The (0, 0) entry is never sampled: it is set to its ideal value before the
matrix is diagonalized.  Every other quantumly evaluated independent entry is
run at the requested noise factors.  Fractional local folds are randomized,
and a variance-weighted straight line is fitted entry by entry at zero noise.
"""

from __future__ import annotations

from _bootstrap import REPO_ROOT

import argparse
import csv
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit import qpy
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeQuebec

from src.capturing_backend_estimator import (
    CapturingBackendEstimatorV2,
    RAW_SIMULATOR_COUNTS,
)
from src.mitigation import (
    linear_zne,
    probabilistic_local_fold_circuit,
    run_local_readout_calibration,
    variance_weighted_linear_zne,
)
from src.saved_run_io import save_json
from src.saved_run_io import load_json


DEFAULT_SOURCE_RUN = Path(os.environ.get(
    "QSENSE_ZNE_SOURCE_RUN",
    "saved/runtime_runs/20260722T164209Z_h2o_1.0_qwc_fake_quebec",
))
DEFAULT_CACHE = Path(os.environ.get(
    "QSENSE_ZNE_CACHE",
    "saved/runtime_cache/sampling_and_sv_h2o_1.0_qwc_fake_quebec.pkl",
))
OUTPUT_LABEL = os.environ.get(
    "QSENSE_ZNE_OUTPUT_LABEL", "full_matrix_readout_linear_zne"
)
FIXED_ENTRY = (0, 0)


def _support_qubits(pubs):
    """Return physical qubits appearing in the mapped PUB observables."""
    support = set()
    for _, observable, _, _ in pubs:
        active = np.any(
            np.logical_or(observable.paulis.z, observable.paulis.x),
            axis=0,
        )
        support.update(map(int, np.flatnonzero(active)))
    return sorted(support)


def _split_integer(total, parts):
    quotient, remainder = divmod(int(total), int(parts))
    return [quotient + (index < remainder) for index in range(parts)]


def _noisy_gate_count(circuit):
    counts = circuit.count_ops()
    return int(sum(counts.get(name, 0) for name in ("sx", "x", "ecr")))


def _real_scalar(value, label, tolerance=1e-10):
    value = complex(value)
    if abs(value.imag) > tolerance:
        raise ValueError(f"{label} has an unexpected imaginary part: {value}.")
    return float(value.real)


def _entry_energy_variance(eigenvector, entry_variances):
    """First-order variance of an eigenvalue of a real symmetric matrix."""
    variance = 0.0
    size = eigenvector.size
    for row in range(size):
        variance += (
            abs(eigenvector[row]) ** 4 * entry_variances[row, row]
        )
        for column in range(row):
            sensitivity = 2.0 * np.real(
                np.conjugate(eigenvector[row]) * eigenvector[column]
            )
            variance += (
                sensitivity**2 * entry_variances[row, column]
            )
    return float(variance)


def _eigenvalue_summary(matrix, ideal_energy, ideal_eigenvector, variances):
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    overlaps = np.abs(eigenvectors.conj().T @ ideal_eigenvector) ** 2
    target_index = int(np.argmax(overlaps))

    ground_variance = _entry_energy_variance(
        eigenvectors[:, 0], variances
    )
    target_variance = _entry_energy_variance(
        eigenvectors[:, target_index], variances
    )
    return {
        "ground_energy": eigenvalues[0],
        "ground_bias": eigenvalues[0] - ideal_energy,
        "ground_std_first_order": np.sqrt(ground_variance),
        "ground_overlap_with_ideal": overlaps[0],
        "target_index_by_overlap": target_index,
        "target_energy": eigenvalues[target_index],
        "target_bias": eigenvalues[target_index] - ideal_energy,
        "target_std_first_order": np.sqrt(target_variance),
        "target_overlap_with_ideal": overlaps[target_index],
        "ground_eigenvector": eigenvectors[:, 0],
        "target_eigenvector": eigenvectors[:, target_index],
    }


def _source_fragment_records(cache, quantum_entries):
    metadata_by_entry = {}
    for item in cache["pub_metadata"]:
        entry = tuple(item["uv"])
        if entry in quantum_entries:
            metadata_by_entry.setdefault(entry, []).append(item)

    records = []
    circuits = []
    for entry in quantum_entries:
        metadata = sorted(
            metadata_by_entry[entry],
            key=lambda item: item["fragment_index"],
        )
        pubs = cache["PUBs"][entry]
        if len(metadata) != len(pubs):
            raise ValueError(
                f"PUB/metadata mismatch for entry {entry}: "
                f"{len(pubs)} versus {len(metadata)}."
            )
        allocated_shots = cache["shot_alloc"][entry]
        if len(allocated_shots) != len(pubs):
            raise ValueError(
                f"PUB/shot-allocation mismatch for entry {entry}: "
                f"{len(pubs)} versus {len(allocated_shots)}."
            )
        for kept_position, (source_item, pub) in enumerate(
            zip(metadata, pubs)
        ):
            circuit, observable, parameter_values, _ = pub
            shots = int(allocated_shots[kept_position])
            metadata_shots = int(source_item["shots"])
            if shots != metadata_shots:
                raise ValueError(
                    f"Shot mismatch for entry {entry}, surviving fragment "
                    f"position {kept_position}: cache={shots}, "
                    f"metadata={metadata_shots}."
                )
            circuit_index = len(circuits)
            circuits.append(circuit)
            records.append(
                {
                    "entry": entry,
                    "fragment_index": source_item["fragment_index"],
                    "circuit_index": circuit_index,
                    "observable_terms": [
                        {"label": label, "coefficient": coefficient}
                        for label, coefficient in observable.to_list()
                    ],
                    "parameter_values": parameter_values,
                    # fragment_index is the original decomposition index and
                    # may be non-contiguous after pruning.  Shots are aligned
                    # with the compact surviving-PUB position instead.
                    "cached_allocated_shots": shots,
                }
            )
    return circuits, records


def _write_entry_csv(path, records):
    fields = [
        "row",
        "column",
        "ideal_estimate",
        "readout_only_estimate",
        "readout_only_bias",
        "readout_only_std",
        "zne_estimate",
        "zne_bias",
        "zne_std",
        "linear_intercept_coefficients",
        "linear_chi_square",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "row": record["entry"][0],
                    "column": record["entry"][1],
                    "ideal_estimate": record["ideal_estimate"],
                    "readout_only_estimate": record[
                        "factor_estimates"
                    ][0],
                    "readout_only_bias": record["factor_biases"][0],
                    "readout_only_std": np.sqrt(
                        record["factor_variances"][0]
                    ),
                    "zne_estimate": record["zne_estimate"],
                    "zne_bias": record["zne_bias"],
                    "zne_std": np.sqrt(record["zne_variance"]),
                    "linear_intercept_coefficients": " ".join(
                        f"{coefficient:.16g}"
                        for coefficient in record[
                            "linear_intercept_coefficients"
                        ]
                    ),
                    "linear_chi_square": record["linear_chi_square"],
                }
            )


def _load_completed_batches(
    output_dir, cache, quantum_entries, factors, randomizations
):
    """Reconstruct entry replicas from portable completed-batch files."""
    replica_estimates = {}
    replica_variances = {}
    batch_dir = output_dir / "batches"
    expected_batches = 1 + (len(factors) - 1) * randomizations
    loaded_batches = 0
    for factor_index, factor in enumerate(factors):
        repetitions = 1 if factor == 1.0 else randomizations
        for randomization in range(repetitions):
            batch_name = (
                f"factor_{factor_index:02d}_lambda_{factor:g}_"
                f"randomization_{randomization:02d}.json.gz"
            )
            path = batch_dir / batch_name
            if not path.exists():
                raise FileNotFoundError(
                    f"Cannot resume: missing completed batch {path}."
                )
            payload = load_json(path)
            if (
                int(payload["factor_index"]) != factor_index
                or float(payload["noise_factor"]) != factor
                or int(payload["randomization"]) != randomization
            ):
                raise ValueError(f"Batch metadata mismatch in {path}.")
            entry_estimates = {entry: 0.0 for entry in quantum_entries}
            entry_variances = {entry: 0.0 for entry in quantum_entries}
            fragment_counts = {entry: 0 for entry in quantum_entries}
            for item in payload["fragment_results"]:
                entry = tuple(item["entry"])
                if entry not in entry_estimates:
                    raise ValueError(f"Unexpected entry {entry} in {path}.")
                entry_estimates[entry] += float(
                    item["mitigated_fragment_estimate"]
                )
                entry_variances[entry] += float(
                    item["mitigated_fragment_std"]
                ) ** 2
                fragment_counts[entry] += 1
            for entry in quantum_entries:
                expected_fragments = len(cache["PUBs"][entry])
                if fragment_counts[entry] != expected_fragments:
                    raise ValueError(
                        f"Fragment count mismatch for {entry} in {path}: "
                        f"{fragment_counts[entry]} versus {expected_fragments}."
                    )
                key = (factor_index, randomization, entry)
                replica_estimates[key] = (
                    entry_estimates[entry]
                    + _real_scalar(
                        cache["H_classical"][entry],
                        f"classical contribution for {entry}",
                    )
                )
                replica_variances[key] = entry_variances[entry]
            loaded_batches += 1
    print(
        f"Loaded {loaded_batches}/{expected_batches} completed batches from "
        f"{output_dir}; no simulator jobs will be run.",
        flush=True,
    )
    return replica_estimates, replica_variances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--noise-factors",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0],
    )
    parser.add_argument("--randomizations", type=int, default=8)
    parser.add_argument(
        "--per-factor-shot-multiplier",
        type=float,
        default=1.0,
    )
    parser.add_argument("--fold-seed", type=int, default=104_729)
    parser.add_argument("--calibration-shots", type=int, default=20_000)
    parser.add_argument("--calibration-seed", type=int, default=8_675_309)
    parser.add_argument("--simulator-seed", type=int, default=16_180)
    parser.add_argument(
        "--skip-folded-circuit-save",
        action="store_true",
        help=(
            "Do not save every folded circuit. Source circuits, fold seeds, "
            "and manifests are always saved and reproduce them exactly."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print workload totals without simulation.",
    )
    parser.add_argument(
        "--resume-run",
        type=Path,
        help=(
            "Reuse all completed portable batch JSON files in this output "
            "directory and perform only aggregation/ZNE post-processing."
        ),
    )
    args = parser.parse_args()

    factors = tuple(map(float, args.noise_factors))
    if len(factors) < 2 or factors[0] != 1.0 or any(
        right <= left for left, right in zip(factors, factors[1:])
    ):
        raise ValueError(
            "At least two noise factors must increase strictly from 1."
        )
    if any(factor > 3.0 for factor in factors):
        raise ValueError("Probabilistic factors must not exceed 3.")
    if args.randomizations < 1:
        raise ValueError("randomizations must be positive.")
    if args.per_factor_shot_multiplier <= 0:
        raise ValueError("per-factor-shot-multiplier must be positive.")

    # This trusted local cache contains Qiskit circuit objects. New outputs
    # produced by this script use JSON, QPY, NPZ, and CSV.
    with args.cache.open("rb") as file:
        cache = pickle.load(file)
    matrix_size = int(cache["Nstates"])
    if matrix_size < 1:
        raise ValueError(
            f"Invalid empty matrix cache: {args.cache}."
        )

    ideal_matrix = np.asarray(cache["Hsub"], dtype=complex)
    if np.max(np.abs(ideal_matrix.imag)) > 1e-10:
        raise ValueError("The current implementation expects a real matrix.")
    ideal_matrix = ideal_matrix.real
    ideal_eigenvalues, ideal_eigenvectors = np.linalg.eigh(ideal_matrix)
    ideal_energy = float(ideal_eigenvalues[0])
    ideal_eigenvector = ideal_eigenvectors[:, 0]

    quantum_entries = sorted(
        tuple(entry)
        for entry in cache["quantum_indices"]
        if tuple(entry) != FIXED_ENTRY
    )
    source_pubs = [
        pub
        for entry in quantum_entries
        for pub in cache["PUBs"][entry]
    ]
    support_qubits = _support_qubits(source_pubs)

    factor_fragment_shots = {}
    total_pubs = 0
    factor_shot_totals = []
    for factor_index, factor in enumerate(factors):
        repetitions = 1 if factor == 1.0 else args.randomizations
        factor_total = 0
        for entry in quantum_entries:
            scaled = [
                max(
                    repetitions,
                    int(round(args.per_factor_shot_multiplier * shots)),
                )
                for shots in cache["shot_alloc"][entry]
            ]
            factor_fragment_shots[factor_index, entry] = scaled
            factor_total += sum(scaled)
            total_pubs += repetitions * len(scaled)
        factor_shot_totals.append(factor_total)

    print(
        f"Matrix size: {ideal_matrix.shape[0]}x"
        f"{ideal_matrix.shape[1]}; fixed ideal entry: {FIXED_ENTRY}.",
        flush=True,
    )
    print(
        f"Quantum entries simulated: {len(quantum_entries)}; "
        f"source fragments: {len(source_pubs)}; physical readout support: "
        f"{support_qubits}.",
        flush=True,
    )
    print(
        "Per-factor allocated shots: "
        + ", ".join(
            f"lambda={factor:g}: {shots}"
            for factor, shots in zip(factors, factor_shot_totals)
        ),
        flush=True,
    )
    print(
        f"Planned folded PUBs: {total_pubs}; simulator batches: "
        f"{1 + (len(factors) - 1) * args.randomizations}.",
        flush=True,
    )
    if args.dry_run:
        return

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = (
        args.resume_run
        if args.resume_run is not None
        else args.source_run / f"{OUTPUT_LABEL}_{timestamp}"
    )
    batch_dir = output_dir / "batches"
    batch_dir.mkdir(parents=True, exist_ok=args.resume_run is not None)

    if args.resume_run is not None:
        replica_estimates, replica_variances = _load_completed_batches(
            output_dir,
            cache,
            quantum_entries,
            factors,
            args.randomizations,
        )
    else:
        source_circuits, source_records = _source_fragment_records(
            cache, quantum_entries
        )
        with (output_dir / "source_fragment_circuits.qpy").open("wb") as file:
            qpy.dump(source_circuits, file)
        save_json(
            output_dir / "source_fragment_manifest.json.gz", source_records
        )

        backend = FakeQuebec()
        readout_mitigator, calibration = run_local_readout_calibration(
            backend,
            support_qubits,
            shots=args.calibration_shots,
            seed_simulator=args.calibration_seed,
            # These are product-state circuits on the union of all measured
            # physical qubits. MPS avoids exponential statevector allocation.
            simulator_method="matrix_product_state",
        )
        save_json(output_dir / "readout_calibration.json.gz", calibration)
        print("Local readout calibration complete.", flush=True)

        original_gate_counts = {
            (entry, fragment_index): _noisy_gate_count(pub[0])
            for entry in quantum_entries
            for fragment_index, pub in enumerate(cache["PUBs"][entry])
        }
        replica_estimates = {}
        replica_variances = {}
        batch_index = 0

    for factor_index, factor in enumerate(factors) if args.resume_run is None else []:
        repetitions = 1 if factor == 1.0 else args.randomizations
        for randomization in range(repetitions):
            batch_index += 1
            batch_pubs = []
            manifest = []
            for entry in quantum_entries:
                source_entry_pubs = cache["PUBs"][entry]
                total_fragment_shots = factor_fragment_shots[
                    factor_index, entry
                ]
                shots_by_fragment = [
                    _split_integer(total, repetitions)
                    for total in total_fragment_shots
                ]
                for fragment_index, source_pub in enumerate(
                    source_entry_pubs
                ):
                    circuit, observable, parameter_values, _ = source_pub
                    shots = shots_by_fragment[fragment_index][randomization]
                    fold_seed = (
                        args.fold_seed
                        + factor_index * 10_000_000
                        + randomization * 100_000
                        + entry[0] * 1_000
                        + entry[1] * 100
                        + fragment_index
                    )
                    folded = probabilistic_local_fold_circuit(
                        circuit,
                        factor,
                        seed=fold_seed,
                        backend=backend,
                    )
                    pub_index = len(batch_pubs)
                    folded.metadata = {
                        **(folded.metadata or {}),
                        "qsense_pub_index": pub_index,
                        "qsense_uv": entry,
                        "qsense_fragment_index": fragment_index,
                    }
                    batch_pubs.append(
                        (
                            folded,
                            observable,
                            parameter_values,
                            (1.0 + 1e-12) / np.sqrt(shots),
                        )
                    )
                    original_count = original_gate_counts[
                        entry, fragment_index
                    ]
                    folded_count = _noisy_gate_count(folded)
                    manifest.append(
                        {
                            "pub_index": pub_index,
                            "entry": entry,
                            "fragment_index": fragment_index,
                            "noise_factor": factor,
                            "randomization": randomization,
                            "shots": shots,
                            "fold_seed": fold_seed,
                            "original_noisy_gate_count": original_count,
                            "folded_noisy_gate_count": folded_count,
                            "realized_noisy_gate_factor": (
                                folded_count / original_count
                                if original_count
                                else 1.0
                            ),
                            "original_depth": circuit.depth(),
                            "folded_depth": folded.depth(),
                        }
                    )

            print(
                f"Starting batch {batch_index}/"
                f"{1 + (len(factors) - 1) * args.randomizations}: "
                f"lambda={factor:g}, randomization={randomization}, "
                f"PUBs={len(batch_pubs)}, "
                f"shots={sum(item['shots'] for item in manifest)}.",
                flush=True,
            )
            simulator = AerSimulator.from_backend(
                backend,
                method="statevector",
                enable_truncation=True,
            )
            estimator = CapturingBackendEstimatorV2(
                backend=simulator,
                readout_mitigator=readout_mitigator,
                options={
                    "abelian_grouping": True,
                    "seed_simulator": args.simulator_seed + batch_index - 1,
                },
            )
            RAW_SIMULATOR_COUNTS.reset()
            batch_result = estimator.run(batch_pubs).result()
            histograms_by_pub = {}
            for histogram in RAW_SIMULATOR_COUNTS.snapshot():
                histograms_by_pub.setdefault(
                    histogram["pub_index"], []
                ).append(histogram)

            entry_estimates = {entry: 0.0 for entry in quantum_entries}
            entry_variances = {entry: 0.0 for entry in quantum_entries}
            fragment_results = []
            for item, pub_result in zip(manifest, batch_result):
                entry = tuple(item["entry"])
                estimate = _real_scalar(
                    np.asarray(pub_result.data.evs).item(),
                    f"fragment estimate for {entry}",
                )
                std = float(np.asarray(pub_result.data.stds).item())
                histograms = histograms_by_pub.get(item["pub_index"], [])
                if not histograms:
                    raise ValueError(
                        f"No histogram captured for PUB {item['pub_index']} "
                        f"in batch {batch_index}."
                    )
                if any(
                    sum(histogram["counts"].values()) != item["shots"]
                    for histogram in histograms
                ):
                    raise ValueError(
                        f"Histogram shot mismatch for PUB {item['pub_index']} "
                        f"in batch {batch_index}."
                    )
                entry_estimates[entry] += estimate
                entry_variances[entry] += std**2
                fragment_results.append(
                    {
                        **item,
                        "mitigated_fragment_estimate": estimate,
                        "mitigated_fragment_std": std,
                        "raw_measurement_histograms": histograms,
                    }
                )

            for entry in quantum_entries:
                key = (factor_index, randomization, entry)
                replica_estimates[key] = (
                    entry_estimates[entry]
                    + _real_scalar(
                        cache["H_classical"][entry],
                        f"classical contribution for {entry}",
                    )
                )
                replica_variances[key] = entry_variances[entry]

            batch_name = (
                f"factor_{factor_index:02d}_lambda_{factor:g}_"
                f"randomization_{randomization:02d}"
            )
            save_json(
                batch_dir / f"{batch_name}.json.gz",
                {
                    "factor_index": factor_index,
                    "noise_factor": factor,
                    "randomization": randomization,
                    "simulator_seed": args.simulator_seed + batch_index - 1,
                    "fragment_results": fragment_results,
                },
            )
            if not args.skip_folded_circuit_save:
                with (batch_dir / f"{batch_name}.qpy").open("wb") as file:
                    qpy.dump([pub[0] for pub in batch_pubs], file)
            print(
                f"Completed batch {batch_index}; portable counts saved.",
                flush=True,
            )

    size = ideal_matrix.shape[0]
    factor_estimate_matrices = np.repeat(
        ideal_matrix[None, :, :], len(factors), axis=0
    )
    factor_variance_matrices = np.zeros(
        (len(factors), size, size), dtype=float
    )
    factor_summaries_by_entry = {}
    variance_floor_applications = 0

    for entry in quantum_entries:
        summaries = []
        for factor_index, factor in enumerate(factors):
            repetitions = 1 if factor == 1.0 else args.randomizations
            estimates = np.asarray(
                [
                    replica_estimates[
                        factor_index, randomization, entry
                    ]
                    for randomization in range(repetitions)
                ]
            )
            variances = np.asarray(
                [
                    replica_variances[
                        factor_index, randomization, entry
                    ]
                    for randomization in range(repetitions)
                ]
            )
            estimate = float(np.mean(estimates))
            shot_variance_of_mean = float(
                np.sum(variances) / repetitions**2
            )
            if repetitions > 1:
                observed_replica_variance = float(
                    np.var(estimates, ddof=1)
                )
                pattern_variance = max(
                    observed_replica_variance - float(np.mean(variances)),
                    0.0,
                )
                pattern_variance_of_mean = (
                    pattern_variance / repetitions
                )
            else:
                observed_replica_variance = 0.0
                pattern_variance_of_mean = 0.0
            empirical_variance = (
                shot_variance_of_mean + pattern_variance_of_mean
            )
            ideal_fragment_stds = np.asarray(
                cache["sig_frag_mat"][entry], dtype=float
            )
            allocated_fragment_shots = np.asarray(
                factor_fragment_shots[factor_index, entry], dtype=float
            )
            if ideal_fragment_stds.shape != allocated_fragment_shots.shape:
                raise ValueError(
                    f"Ideal-std/shot mismatch for entry {entry}: "
                    f"{ideal_fragment_stds.shape} versus "
                    f"{allocated_fragment_shots.shape}."
                )
            ideal_sampling_variance_floor = float(
                np.sum(
                    ideal_fragment_stds**2 / allocated_fragment_shots
                )
            )
            variance = max(
                empirical_variance, ideal_sampling_variance_floor
            )
            floor_applied = variance > empirical_variance
            variance_floor_applications += int(floor_applied)
            row, column = entry
            factor_estimate_matrices[factor_index, row, column] = estimate
            factor_estimate_matrices[factor_index, column, row] = estimate
            factor_variance_matrices[factor_index, row, column] = variance
            factor_variance_matrices[factor_index, column, row] = variance
            summaries.append(
                {
                    "noise_factor": factor,
                    "estimate": estimate,
                    "variance": variance,
                    "std": np.sqrt(variance),
                    "empirical_variance": empirical_variance,
                    "ideal_sampling_variance_floor": (
                        ideal_sampling_variance_floor
                    ),
                    "variance_floor_applied": floor_applied,
                    "shot_variance_of_mean": shot_variance_of_mean,
                    "pattern_variance_of_mean": pattern_variance_of_mean,
                    "observed_replica_variance": observed_replica_variance,
                    "replica_estimates": estimates,
                    "replica_variances": variances,
                }
            )
        factor_summaries_by_entry[entry] = summaries

    print(
        "Applied the analytical ideal finite-shot variance floor to "
        f"{variance_floor_applications}/"
        f"{len(quantum_entries) * len(factors)} entry-factor estimates.",
        flush=True,
    )

    # Enforce the premise of this experiment exactly at every factor.
    fixed_row, fixed_column = FIXED_ENTRY
    factor_estimate_matrices[:, fixed_row, fixed_column] = ideal_matrix[
        fixed_row, fixed_column
    ]
    factor_variance_matrices[:, 0, 0] = 0.0

    zne_matrix = ideal_matrix.copy()
    zne_entry_variances = np.zeros_like(ideal_matrix)
    entry_fit_records = []
    unweighted_fit_fallbacks = 0
    for entry in quantum_entries:
        summaries = factor_summaries_by_entry[entry]
        estimates = np.asarray([item["estimate"] for item in summaries])
        variances = np.asarray([item["variance"] for item in summaries])
        design = np.column_stack(
            (np.ones(len(factors)), np.asarray(factors))
        )
        inverse_variances = 1.0 / variances
        weighted_normal = design.T @ (
            inverse_variances[:, None] * design
        )
        weighted_condition_number = float(np.linalg.cond(weighted_normal))
        if weighted_condition_number <= 1e12:
            estimate, variance, coefficients = (
                variance_weighted_linear_zne(
                    factors, estimates, variances
                )
            )
            parameters = np.linalg.solve(
                weighted_normal,
                design.T @ (inverse_variances * estimates),
            )
            fit_weighting = "inverse_variance"
        else:
            # A near-zero variance at one factor can make the slope
            # unidentified under inverse-variance weighting. Keep the same
            # linear model but use its fixed ordinary-least-squares
            # coefficients rather than assigning effectively infinite weight.
            estimate, variance, coefficients = linear_zne(
                factors, estimates, variances
            )
            parameters = np.linalg.lstsq(
                design, estimates, rcond=None
            )[0]
            fit_weighting = "unweighted_ill_conditioned_fallback"
            unweighted_fit_fallbacks += 1
        fitted_values = design @ parameters
        chi_square = float(
            np.sum((estimates - fitted_values) ** 2 / variances)
        )

        row, column = entry
        zne_matrix[row, column] = estimate
        zne_matrix[column, row] = estimate
        zne_entry_variances[row, column] = variance
        zne_entry_variances[column, row] = variance
        ideal_estimate = ideal_matrix[row, column]
        entry_fit_records.append(
            {
                "entry": entry,
                "ideal_estimate": ideal_estimate,
                "factor_estimates": estimates,
                "factor_biases": estimates - ideal_estimate,
                "factor_variances": variances,
                "factor_summaries": summaries,
                "zne_estimate": estimate,
                "zne_bias": estimate - ideal_estimate,
                "zne_variance": variance,
                "linear_parameters": parameters,
                "linear_fit_weighting": fit_weighting,
                "weighted_normal_condition_number": (
                    weighted_condition_number
                ),
                "linear_intercept_coefficients": coefficients,
                "linear_fitted_values": fitted_values,
                "linear_chi_square": chi_square,
                "linear_degrees_of_freedom": len(factors) - 2,
            }
        )

    zne_matrix[fixed_row, fixed_column] = ideal_matrix[
        fixed_row, fixed_column
    ]
    zne_entry_variances[fixed_row, fixed_column] = 0.0
    readout_only_summary = _eigenvalue_summary(
        factor_estimate_matrices[0],
        ideal_energy,
        ideal_eigenvector,
        factor_variance_matrices[0],
    )
    zne_summary = _eigenvalue_summary(
        zne_matrix,
        ideal_energy,
        ideal_eigenvector,
        zne_entry_variances,
    )

    source_unmitigated_summary = None
    source_arrays = args.source_run / "matrices_and_eigenvectors.npz"
    if source_arrays.exists():
        with np.load(source_arrays, allow_pickle=False) as archive:
            source_unmitigated = np.asarray(
                archive["noisy_matrix"], dtype=complex
            )
        if np.max(np.abs(source_unmitigated.imag)) <= 1e-10:
            source_unmitigated = source_unmitigated.real
            source_unmitigated[fixed_row, fixed_column] = ideal_matrix[
                fixed_row, fixed_column
            ]
            source_unmitigated_summary = _eigenvalue_summary(
                source_unmitigated,
                ideal_energy,
                ideal_eigenvector,
                np.zeros_like(ideal_matrix),
            )

    results = {
        "source_run": args.source_run,
        "source_cache": args.cache,
        "fixed_ideal_entry": FIXED_ENTRY,
        "noise_factors": factors,
        "randomizations_per_nonunit_factor": args.randomizations,
        "per_factor_shot_multiplier": args.per_factor_shot_multiplier,
        "factor_shot_totals": factor_shot_totals,
        "calibration_shots_per_prepared_state": args.calibration_shots,
        "calibration_seed": args.calibration_seed,
        "fold_seed": args.fold_seed,
        "simulator_seed_base": args.simulator_seed,
        "physical_readout_qubits": support_qubits,
        "ideal_energy": ideal_energy,
        "source_unmitigated_with_ideal_00": source_unmitigated_summary,
        "readout_only_with_ideal_00": readout_only_summary,
        "readout_plus_linear_zne_with_ideal_00": zne_summary,
        "entry_fits": entry_fit_records,
        "variance_floor_applications": variance_floor_applications,
        "unweighted_fit_fallbacks": unweighted_fit_fallbacks,
        "weighted_fit_condition_threshold": 1e12,
        "variance_floor_definition": (
            "max(empirical variance, sum_f ideal_fragment_std_f^2 / "
            "allocated_factor_shots_f); this prevents one-shot fragments "
            "from being interpreted as exact zero-variance observations"
        ),
        "uncertainty_scope": (
            "Entry variances include finite experiment shots and resolved "
            "folding-pattern variance, with the cached ideal finite-shot "
            "variance as a conservative lower bound. Eigenvalue standard "
            "deviations use "
            "first-order propagation. Finite readout-calibration uncertainty "
            "is not yet propagated; calibration counts are saved for a "
            "hierarchical bootstrap."
        ),
        "folded_circuits_saved": not args.skip_folded_circuit_save,
    }
    save_json(output_dir / "results.json.gz", results)
    _write_entry_csv(output_dir / "entry_comparison.csv", entry_fit_records)
    np.savez(
        output_dir / "matrices_and_zne.npz",
        ideal_matrix=ideal_matrix,
        ideal_eigenvector=ideal_eigenvector,
        noise_factors=np.asarray(factors),
        factor_estimate_matrices=factor_estimate_matrices,
        factor_variance_matrices=factor_variance_matrices,
        zne_matrix=zne_matrix,
        zne_entry_variances=zne_entry_variances,
        zne_ground_eigenvector=zne_summary["ground_eigenvector"],
        zne_target_eigenvector=zne_summary["target_eigenvector"],
    )

    print("\nFinal matrix-level comparison")
    if source_unmitigated_summary is not None:
        print(
            "  Source unmitigated, ideal H00: "
            f"ground bias={source_unmitigated_summary['ground_bias']}, "
            f"target bias={source_unmitigated_summary['target_bias']}"
        )
    print(
        "  Readout mitigation only, ideal H00: "
        f"ground bias={readout_only_summary['ground_bias']}, "
        f"ground std={readout_only_summary['ground_std_first_order']}, "
        f"target bias={readout_only_summary['target_bias']}"
    )
    print(
        "  Readout + probabilistic linear ZNE, ideal H00: "
        f"ground bias={zne_summary['ground_bias']}, "
        f"ground std={zne_summary['ground_std_first_order']}, "
        f"target bias={zne_summary['target_bias']}, "
        f"target overlap={zne_summary['target_overlap_with_ideal']}"
    )
    print(f"Saved results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
