"""Resample the dominant PT (23, 0) fragment at larger shot counts.

Only fragment 0 is rerun.  Every other matrix entry and fragment remains at
its saved lambda=1 value, allowing a direct test of how this fragment affects
the entry estimate and the overlap-tracked PT ground-state energy.
"""

from _bootstrap import REPO_ROOT

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from qiskit.primitives.backend_estimator_v2 import (
    _pauli_expval_with_variance,
)
from qiskit.quantum_info import PauliList
from qiskit.result import Counts
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeQuebec

from src.capturing_backend_estimator import (
    CapturingBackendEstimatorV2,
    RAW_SIMULATOR_COUNTS,
)
from src.mitigation import LocalReadoutMitigator
from src.saved_run_io import load_json, save_json


ENTRY = (23, 0)
FRAGMENT_POSITION = 0
DEFAULT_CACHE = REPO_ROOT / (
    "saved/runtime_cache/"
    "sampling_and_sv_h2o_1.0_pt_mp2_qwc_fake_quebec.pkl"
)
DEFAULT_RUN = REPO_ROOT / (
    "saved/runtime_runs/pt_h2o_1.0/"
    "pt_readout_linear_zne_20260813T134200Z"
)


def _precision_for_shots(shots: int) -> float:
    return (1.0 + 1e-12) / np.sqrt(shots)


def _raw_fragment_estimate(histograms: list[dict], observable) -> float:
    """Evaluate a PUB observable without readout inversion."""
    pauli_expectations = {}
    for histogram in histograms:
        means, _ = _pauli_expval_with_variance(
            Counts(histogram["counts"]),
            PauliList(histogram["measured_pauli_labels"]),
        )
        pauli_expectations.update(
            zip(histogram["original_pauli_labels"], means)
        )
    estimate = sum(
        complex(coefficient) * pauli_expectations[label]
        for label, coefficient in observable.to_list()
    )
    if abs(estimate.imag) > 1e-10:
        raise ValueError(f"Unexpected complex fragment estimate: {estimate}.")
    return float(estimate.real)


def _tracked_eigensolution(
    matrix: np.ndarray,
    ideal_energy: float,
    ideal_eigenvector: np.ndarray,
) -> dict:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    overlaps = np.abs(eigenvectors.T.conj() @ ideal_eigenvector) ** 2
    target_index = int(np.argmax(overlaps))
    return {
        "target_energy": float(eigenvalues[target_index]),
        "target_bias": float(eigenvalues[target_index] - ideal_energy),
        "target_overlap": float(overlaps[target_index]),
        "target_index": target_index,
        "lowest_energy": float(eigenvalues[0]),
        "lowest_bias": float(eigenvalues[0] - ideal_energy),
    }


def _summary(records: list[dict], prefix: str) -> dict:
    fields = (
        "fragment_estimate",
        "entry_estimate",
        "entry_bias",
        "target_energy",
        "target_bias",
        "target_overlap",
    )
    result = {}
    for field in fields:
        values = np.asarray([record[f"{prefix}_{field}"] for record in records])
        result[field] = {
            "mean": float(np.mean(values)),
            "sample_std_across_replicas": float(np.std(values, ddof=1)),
        }
    target_biases = np.asarray(
        [record[f"{prefix}_target_bias"] for record in records]
    )
    result["target_bias"]["rmse"] = float(
        np.sqrt(np.mean(target_biases**2))
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--shots", type=int, nargs="+", default=[100, 500, 2000, 10000]
    )
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23_000_813)
    args = parser.parse_args()
    if any(shots < 1 for shots in args.shots) or args.replicas < 2:
        raise ValueError("Shots must be positive and replicas must be at least 2.")

    # Trusted local cache: it contains Qiskit circuit and observable objects.
    with args.cache.open("rb") as file:
        cache = pickle.load(file)
    ideal_matrix = np.asarray(cache["Hsub"], dtype=complex).real
    ideal_eigenvalues, ideal_eigenvectors = np.linalg.eigh(ideal_matrix)
    ideal_energy = float(ideal_eigenvalues[0])
    ideal_eigenvector = ideal_eigenvectors[:, 0]

    arrays = np.load(args.run / "matrices_and_zne.npz")
    mitigated_matrix = np.asarray(arrays["factor_estimate_matrices"][0]).real
    raw_matrix = np.asarray(
        np.load(args.run / "unmitigated_same_shots_matrix.npz")[
            "unmitigated_matrix"
        ]
    ).real

    batch = load_json(
        args.run
        / "batches/factor_00_lambda_1_randomization_00.json.gz"
    )
    saved_fragment = next(
        item
        for item in batch["fragment_results"]
        if tuple(item["entry"]) == ENTRY
        and int(item["fragment_index"]) == FRAGMENT_POSITION
    )
    circuit, observable, parameter_values, _ = cache["PUBs"][ENTRY][
        FRAGMENT_POSITION
    ]
    old_raw_fragment = _raw_fragment_estimate(
        saved_fragment["raw_measurement_histograms"], observable
    )
    old_mitigated_fragment = float(
        saved_fragment["mitigated_fragment_estimate"]
    )

    calibration = load_json(args.run / "readout_calibration.json.gz")
    mitigator = LocalReadoutMitigator(calibration["assignment_matrices"])
    backend = FakeQuebec()
    seed_sequences = np.random.SeedSequence(args.seed).spawn(
        len(args.shots) * args.replicas
    )
    simulator_seeds = iter(
        int(sequence.generate_state(1, dtype=np.uint32)[0])
        for sequence in seed_sequences
    )

    records = []
    for shots in args.shots:
        print(f"Starting {args.replicas} replicas at {shots} shots.", flush=True)
        for replica in range(args.replicas):
            # Spawned streams avoid the strong relationship that adjacent
            # integer seeds can have in repeated simulator experiments.
            seed = next(simulator_seeds)
            simulator = AerSimulator.from_backend(
                backend, method="statevector", enable_truncation=True
            )
            estimator = CapturingBackendEstimatorV2(
                backend=simulator,
                readout_mitigator=mitigator,
                options={"abelian_grouping": True, "seed_simulator": seed},
            )
            rerun_circuit = circuit.copy()
            rerun_circuit.metadata = {
                **(rerun_circuit.metadata or {}),
                "qsense_pub_index": 0,
                "qsense_uv": ENTRY,
                "qsense_fragment_index": FRAGMENT_POSITION,
            }
            RAW_SIMULATOR_COUNTS.reset()
            pub_result = estimator.run(
                [
                    (
                        rerun_circuit,
                        observable,
                        parameter_values,
                        _precision_for_shots(shots),
                    )
                ]
            ).result()[0]
            histograms = RAW_SIMULATOR_COUNTS.snapshot()
            if not histograms:
                raise ValueError("No raw histograms were captured.")
            raw_fragment = _raw_fragment_estimate(histograms, observable)
            mitigated_fragment = float(np.asarray(pub_result.data.evs).item())

            record = {
                "shots": shots,
                "replica": replica,
                "seed": seed,
                "raw_measurement_histograms": histograms,
            }
            for prefix, fragment, old_fragment, base_matrix in (
                ("raw", raw_fragment, old_raw_fragment, raw_matrix),
                (
                    "mitigated",
                    mitigated_fragment,
                    old_mitigated_fragment,
                    mitigated_matrix,
                ),
            ):
                updated_matrix = base_matrix.copy()
                updated_entry = float(
                    base_matrix[ENTRY] + fragment - old_fragment
                )
                updated_matrix[ENTRY] = updated_entry
                updated_matrix[ENTRY[::-1]] = updated_entry
                eigensolution = _tracked_eigensolution(
                    updated_matrix, ideal_energy, ideal_eigenvector
                )
                record.update(
                    {
                        f"{prefix}_fragment_estimate": fragment,
                        f"{prefix}_entry_estimate": updated_entry,
                        f"{prefix}_entry_bias": (
                            updated_entry - ideal_matrix[ENTRY]
                        ),
                        **{
                            f"{prefix}_{key}": value
                            for key, value in eigensolution.items()
                        },
                    }
                )
            records.append(record)
        print(f"Completed {shots}-shot replicas.", flush=True)

    summaries = []
    for shots in args.shots:
        selected = [record for record in records if record["shots"] == shots]
        summaries.append(
            {
                "shots": shots,
                "raw": _summary(selected, "raw"),
                "readout_mitigated": _summary(selected, "mitigated"),
            }
        )

    output = {
        "entry": ENTRY,
        "fragment_position": FRAGMENT_POSITION,
        "ideal_entry": ideal_matrix[ENTRY],
        "ideal_energy": ideal_energy,
        "old_fragment_shots": int(saved_fragment["shots"]),
        "old_raw_fragment_estimate": old_raw_fragment,
        "old_mitigated_fragment_estimate": old_mitigated_fragment,
        "baseline_raw": _tracked_eigensolution(
            raw_matrix, ideal_energy, ideal_eigenvector
        ),
        "baseline_readout_mitigated": _tracked_eigensolution(
            mitigated_matrix, ideal_energy, ideal_eigenvector
        ),
        "summaries": summaries,
        "replicas": records,
    }
    save_json(args.run / "resample_entry_23_0_fragment_0.json.gz", output)
    with (args.run / "resample_entry_23_0_fragment_0.csv").open(
        "w", newline=""
    ) as file:
        csv_records = [
            {
                key: value
                for key, value in record.items()
                if key != "raw_measurement_histograms"
            }
            for record in records
        ]
        writer = csv.DictWriter(file, fieldnames=list(csv_records[0]))
        writer.writeheader()
        writer.writerows(csv_records)

    print("\nMean results across replicas")
    for item in summaries:
        raw = item["raw"]
        mitigated = item["readout_mitigated"]
        print(
            f"  shots={item['shots']}: "
            f"raw entry bias={raw['entry_bias']['mean']:+.6g}, "
            f"raw target bias={raw['target_bias']['mean']:+.6g}, "
            f"raw overlap={raw['target_overlap']['mean']:.6g}; "
            f"mitigated entry bias={mitigated['entry_bias']['mean']:+.6g}, "
            f"mitigated target bias={mitigated['target_bias']['mean']:+.6g}, "
            f"mitigated overlap={mitigated['target_overlap']['mean']:.6g}"
        )


if __name__ == "__main__":
    main()
