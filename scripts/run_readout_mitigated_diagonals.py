"""Run only the split-basis diagonal PUBs with local readout mitigation."""

from __future__ import annotations

from _bootstrap import REPO_ROOT

import argparse
import csv
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeQuebec

from src.capturing_backend_estimator import (
    CapturingBackendEstimatorV2,
    RAW_SIMULATOR_COUNTS,
)
from src.mitigation import run_local_readout_calibration
from src.saved_run_io import save_json


DEFAULT_RUN_DIR = Path(
    "saved/runtime_runs/"
    "20260724T024719Z_h2o_1.0_"
    "state0_orbital0_split_v1_qwc_fake_quebec"
)
DEFAULT_CACHE = Path(
    "saved/runtime_cache/"
    "sampling_and_sv_h2o_1.0_"
    "state0_orbital0_split_v1_qwc_fake_quebec.pkl"
)


class ProgressAerSimulator(AerSimulator):
    """Report every shot group submitted by BackendEstimatorV2."""

    def run(self, run_input, **run_options):
        circuits = (
            [run_input]
            if isinstance(run_input, QuantumCircuit)
            else list(run_input)
        )
        active_counts = []
        for circuit in circuits:
            active_counts.append(
                len(
                    {
                        circuit.find_bit(qubit).index
                        for instruction in circuit.data
                        if instruction.operation.name != "barrier"
                        for qubit in instruction.qubits
                    }
                )
            )
        active_text = (
            str(active_counts[0])
            if len(set(active_counts)) == 1
            else f"{min(active_counts)}-{max(active_counts)}"
        )
        print(
            f"Starting noisy group: {len(circuits)} measurement circuits, "
            f"shots={run_options.get('shots')}, "
            f"active qubits before Aer truncation={active_text}",
            flush=True,
        )
        return super().run(circuits, **run_options)


def _load_previous_diagonals(path):
    entries = {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            i = int(row["row"])
            j = int(row["column"])
            if i == j:
                entries[i] = {
                    "previous_noisy_estimate": complex(
                        float(row["noisy_estimate_real"]),
                        float(row["noisy_estimate_imag"]),
                    ),
                    "previous_bias": complex(
                        float(row["noisy_bias_real"]),
                        float(row["noisy_bias_imag"]),
                    ),
                    "previous_std": float(row["noisy_std"]),
                    "is_quantum": row["is_quantum"].lower() == "true",
                }
    return entries


def _support_qubits(pubs):
    support = set()
    for _, observable, _, _ in pubs:
        active = np.any(
            np.logical_or(observable.paulis.z, observable.paulis.x),
            axis=0,
        )
        support.update(map(int, np.flatnonzero(active)))
    return sorted(support)


def _portable_observable(observable):
    return [
        {"label": label, "coefficient": complex(coefficient)}
        for label, coefficient in observable.to_list()
    ]


def _raw_fragment_estimate(histograms, observable_terms, allocated_shots):
    """Reconstruct the unmitigated estimate from the same raw histograms."""
    coefficients = {}
    for term in observable_terms:
        label = term["label"]
        coefficients[label] = (
            coefficients.get(label, 0.0) + term["coefficient"]
        )
    estimate = 0.0j
    for histogram in histograms:
        shots = int(sum(histogram["counts"].values()))
        if shots != int(allocated_shots):
            raise ValueError(
                f"Histogram has {shots} shots; expected {allocated_shots}."
            )
        group_value = 0.0j
        for bitstring, frequency in histogram["counts"].items():
            outcome = int(bitstring.split(" ", 1)[0], 2)
            outcome_value = 0.0j
            for original, measured in zip(
                histogram["original_pauli_labels"],
                histogram["measured_pauli_labels"],
            ):
                support_mask = 0
                for qubit, pauli in enumerate(reversed(measured)):
                    if pauli != "I":
                        support_mask |= 1 << qubit
                sign = (-1) ** ((support_mask & outcome).bit_count() % 2)
                outcome_value += coefficients[original] * sign
            group_value += int(frequency) * outcome_value
        estimate += group_value / shots
    return np.real_if_close(estimate).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--calibration-shots", type=int, default=20_000)
    parser.add_argument("--calibration-seed", type=int, default=8_675_309)
    parser.add_argument("--simulator-seed", type=int, default=31_415)
    args = parser.parse_args()

    # This is a trusted, locally generated compilation cache. New results below
    # are saved only as JSON/QPY/NPZ/CSV.
    with args.cache.open("rb") as file:
        cache = pickle.load(file)

    diagonal_uvs = sorted(
        uv for uv in cache["quantum_indices"] if uv[0] == uv[1]
    )
    flat_pubs = []
    fragment_manifest = []
    for uv in diagonal_uvs:
        metadata = sorted(
            (
                item
                for item in cache["pub_metadata"]
                if tuple(item["uv"]) == uv
            ),
            key=lambda item: item["fragment_index"],
        )
        pubs = cache["PUBs"][uv]
        if len(pubs) != len(metadata):
            raise ValueError(f"PUB/metadata mismatch for {uv}.")
        for pub, item in zip(pubs, metadata):
            expected_shots = int(cache["shot_alloc"][uv][item["fragment_index"]])
            if int(item["shots"]) != expected_shots:
                raise ValueError(f"Saved shot allocation mismatch for {uv}.")
            circuit, observable, parameter_values, precision = pub
            circuit.metadata = {
                **(circuit.metadata or {}),
                "qsense_pub_index": int(item["pub_index"]),
                "qsense_uv": tuple(uv),
                "qsense_fragment_index": int(item["fragment_index"]),
            }
            flat_pubs.append(pub)
            fragment_manifest.append(
                {
                    **item,
                    "uv": tuple(uv),
                    "circuit_index": len(flat_pubs) - 1,
                    "observable_terms": _portable_observable(observable),
                    "parameter_values": parameter_values,
                    "target_precision": precision,
                }
            )

    support_qubits = _support_qubits(flat_pubs)
    print(
        f"Loaded {len(flat_pubs)} diagonal fragments for "
        f"{len(diagonal_uvs)} quantum entries; "
        f"allocated shots={sum(item['shots'] for item in fragment_manifest)}.",
        flush=True,
    )
    print(
        f"Readout calibration support: {len(support_qubits)} physical qubits "
        f"{support_qubits}.",
        flush=True,
    )

    noise_backend = FakeQuebec()
    mitigator, calibration = run_local_readout_calibration(
        noise_backend,
        support_qubits,
        shots=args.calibration_shots,
        seed_simulator=args.calibration_seed,
        simulator_method="statevector",
    )
    print("Readout calibration complete.", flush=True)

    simulator = ProgressAerSimulator.from_backend(
        noise_backend,
        method="statevector",
        enable_truncation=True,
    )
    estimator = CapturingBackendEstimatorV2(
        backend=simulator,
        readout_mitigator=mitigator,
        options={
            "abelian_grouping": True,
            "seed_simulator": args.simulator_seed,
        },
    )
    RAW_SIMULATOR_COUNTS.reset()
    result = estimator.run(flat_pubs).result()
    print("All diagonal noisy groups complete.", flush=True)

    raw_histograms = RAW_SIMULATOR_COUNTS.snapshot()
    raw_by_pub = {}
    for histogram in raw_histograms:
        raw_by_pub.setdefault(histogram["pub_index"], []).append(histogram)

    fragment_results = []
    entry_quantum_estimates = {uv: 0.0 for uv in diagonal_uvs}
    entry_paired_raw_estimates = {uv: 0.0 for uv in diagonal_uvs}
    entry_variances = {uv: 0.0 for uv in diagonal_uvs}
    for item, pub_result in zip(fragment_manifest, result):
        uv = tuple(item["uv"])
        estimate = np.asarray(pub_result.data.evs).item()
        std = np.asarray(pub_result.data.stds).item()
        histograms = raw_by_pub.get(item["pub_index"], [])
        paired_raw_estimate = _raw_fragment_estimate(
            histograms,
            item["observable_terms"],
            item["shots"],
        )
        entry_quantum_estimates[uv] += estimate
        entry_paired_raw_estimates[uv] += paired_raw_estimate
        entry_variances[uv] += std**2
        fragment_results.append(
            {
                **item,
                "paired_raw_fragment_estimate": paired_raw_estimate,
                "mitigated_fragment_estimate": estimate,
                "mitigated_fragment_std": std,
                "raw_measurement_histograms": histograms,
            }
        )

    previous = _load_previous_diagonals(
        args.run_dir / "matrix_entries.csv"
    )
    entries = []
    nstates = int(cache["Nstates"])
    for i in range(nstates):
        uv = (i, i)
        ideal = complex(cache["Hsub"][uv])
        is_quantum = uv in entry_quantum_estimates
        if is_quantum:
            estimate = (
                entry_quantum_estimates[uv]
                + complex(cache["H_classical"][uv])
            )
            paired_raw_estimate = (
                entry_paired_raw_estimates[uv]
                + complex(cache["H_classical"][uv])
            )
            std = float(np.sqrt(entry_variances[uv]))
            allocated_shots = int(sum(cache["shot_alloc"][uv]))
        else:
            estimate = ideal
            paired_raw_estimate = ideal
            std = 0.0
            allocated_shots = 0
        old = previous[i]
        entries.append(
            {
                "uv": uv,
                "is_quantum": is_quantum,
                "ideal_estimate": ideal,
                "previous_noisy_estimate": old["previous_noisy_estimate"],
                "previous_bias": old["previous_bias"],
                "previous_std": old["previous_std"],
                "paired_raw_estimate": paired_raw_estimate,
                "paired_raw_bias": paired_raw_estimate - ideal,
                "mitigated_estimate": estimate,
                "mitigated_bias": estimate - ideal,
                "mitigated_std": std,
                "absolute_bias_change": (
                    abs(estimate - ideal) - abs(old["previous_bias"])
                ),
                "allocated_shots": allocated_shots,
            }
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.run_dir / f"readout_mitigated_diagonals_{timestamp}"
    output_dir.mkdir(parents=True)
    save_json(output_dir / "calibration.json.gz", calibration)
    save_json(output_dir / "fragment_results.json.gz", fragment_results)
    save_json(output_dir / "raw_measurement_histograms.json.gz", raw_histograms)
    save_json(
        output_dir / "results.json.gz",
        {
            "source_run_dir": args.run_dir,
            "source_runtime_cache": args.cache,
            "calibration_shots_per_circuit": args.calibration_shots,
            "calibration_seed": args.calibration_seed,
            "simulator_seed": args.simulator_seed,
            "physical_readout_qubits": support_qubits,
            "entries": entries,
        },
    )
    with (output_dir / "diagonal_circuits.qpy").open("wb") as file:
        qpy.dump([pub[0] for pub in flat_pubs], file)
    np.savez(
        output_dir / "diagonal_arrays.npz",
        ideal=np.asarray([item["ideal_estimate"] for item in entries]),
        previous=np.asarray(
            [item["previous_noisy_estimate"] for item in entries]
        ),
        paired_raw=np.asarray(
            [item["paired_raw_estimate"] for item in entries]
        ),
        mitigated=np.asarray(
            [item["mitigated_estimate"] for item in entries]
        ),
        previous_std=np.asarray([item["previous_std"] for item in entries]),
        mitigated_std=np.asarray([item["mitigated_std"] for item in entries]),
    )
    csv_fields = [
        "index",
        "is_quantum",
        "allocated_shots",
        "ideal_estimate",
        "previous_noisy_estimate",
        "previous_bias",
        "previous_std",
        "paired_raw_estimate",
        "paired_raw_bias",
        "mitigated_estimate",
        "mitigated_bias",
        "mitigated_std",
        "absolute_bias_change",
    ]
    with (output_dir / "diagonal_comparison.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        for i, item in enumerate(entries):
            writer.writerow(
                {
                    "index": i,
                    **{key: item[key] for key in csv_fields if key != "index"},
                }
            )

    print("\nDiagonal comparison")
    print(
        " i   shots      ideal       previous bias  paired raw bias  "
        "mitigated bias  mitigated std"
    )
    for i, item in enumerate(entries):
        print(
            f"{i:2d} {item['allocated_shots']:7d} "
            f"{item['ideal_estimate'].real:12.6f} "
            f"{item['previous_bias'].real:15.8f} "
            f"{item['paired_raw_bias'].real:15.8f} "
            f"{item['mitigated_bias'].real:15.8f} "
            f"{item['mitigated_std']:14.8f}"
        )
    print(f"\nSaved portable results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
