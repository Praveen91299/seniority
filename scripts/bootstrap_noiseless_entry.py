"""Count-level bootstrap verification for a saved matrix entry."""

from _bootstrap import REPO_ROOT

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
from qiskit import ClassicalRegister
from qiskit_aer import AerSimulator


def _latest_completed_run(runs_dir):
    candidates = [
        path.parent
        for path in runs_dir.glob("*/experiment_inputs.pkl")
        if (path.parent / "matrix_entries.pkl").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed runs found under {runs_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _observable_support(observable):
    if np.any(observable.paulis.x):
        raise ValueError("Bootstrap expects the saved observable to contain only I/Z terms.")
    return np.flatnonzero(np.any(observable.paulis.z, axis=0)).tolist()


def _observable_values(observable, measured_qubits):
    z_support = np.asarray(observable.paulis.z[:, measured_qubits], dtype=np.int8)
    coefficients = np.asarray(observable.coeffs)
    values = []
    for packed_bits in range(2 ** len(measured_qubits)):
        bits = np.array(
            [(packed_bits >> index) & 1 for index in range(len(measured_qubits))],
            dtype=np.int8,
        )
        signs = 1 - 2 * ((z_support @ bits) % 2)
        values.append(float(np.real_if_close(coefficients @ signs)))
    return np.asarray(values)


def _observed_outcomes(counts, all_outcome_values):
    outcomes = []
    frequencies = []
    values = []
    for bitstring, frequency in counts.items():
        # Qiskit prints classical registers newest-first. The dedicated
        # bootstrap register added below is therefore the left-most group;
        # saved fragment circuits can also contain an older unused register.
        bootstrap_register_bits = bitstring.split()[0]
        packed_bits = int(bootstrap_register_bits, 2)
        outcomes.append(bitstring)
        frequencies.append(int(frequency))
        values.append(all_outcome_values[packed_bits])
    return outcomes, np.asarray(frequencies), np.asarray(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--column", type=int, default=0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=314159)
    args = parser.parse_args()

    run_dir = args.run_dir or _latest_completed_run(Path("saved/runtime_runs"))
    uv = (args.row, args.column)
    with (run_dir / "experiment_inputs.pkl").open("rb") as file:
        inputs = pickle.load(file)
    fragments = [
        item for item in inputs["fragments"] if tuple(item["uv"]) == uv
    ]
    if not fragments:
        raise ValueError(f"No quantum fragments found for entry {uv}")

    simulator = AerSimulator(method="statevector", enable_truncation=True)
    rng = np.random.default_rng(args.seed)
    nonparametric_entry_estimates = np.zeros(args.bootstrap_repetitions)
    parametric_entry_estimates = np.zeros(args.bootstrap_repetitions)
    fragment_records = []
    started = time.perf_counter()

    for position, fragment in enumerate(fragments, start=1):
        observable = fragment["observable"]
        measured_qubits = _observable_support(observable)
        all_outcome_values = _observable_values(observable, measured_qubits)
        probability_circuit = fragment["circuit"].copy()
        probability_circuit.save_probabilities(
            measured_qubits,
            label="exact_probabilities",
        )
        exact_probabilities = np.asarray(
            simulator.run(probability_circuit, shots=1).result().data(0)[
                "exact_probabilities"
            ]
        )
        exact_mean = float(exact_probabilities @ all_outcome_values)
        exact_single_shot_std = float(
            np.sqrt(
                exact_probabilities
                @ (all_outcome_values - exact_mean) ** 2
            )
        )
        circuit = fragment["circuit"].copy()
        classical_register = ClassicalRegister(
            len(measured_qubits),
            f"bootstrap_{fragment['fragment_index']}",
        )
        circuit.add_register(classical_register)
        circuit.measure(measured_qubits, classical_register)
        shots = int(fragment["allocated_shots"])
        simulator_seed = args.seed + position
        print(
            f"Sampling fragment {position}/{len(fragments)}: "
            f"index={fragment['fragment_index']}, shots={shots}, "
            f"measured_qubits={len(measured_qubits)}",
            flush=True,
        )
        result = simulator.run(
            circuit,
            shots=shots,
            seed_simulator=simulator_seed,
        ).result()
        counts = result.get_counts(0)
        outcomes, frequencies, outcome_values = _observed_outcomes(
            counts,
            all_outcome_values,
        )
        probabilities = frequencies / frequencies.sum()
        histogram_mean = float(probabilities @ outcome_values)
        histogram_single_shot_std = float(
            np.sqrt(probabilities @ (outcome_values - histogram_mean) ** 2)
        )
        histogram_standard_error = histogram_single_shot_std / np.sqrt(shots)
        nonparametric_counts = rng.multinomial(
            shots,
            probabilities,
            size=args.bootstrap_repetitions,
        )
        nonparametric_estimates = (
            nonparametric_counts @ outcome_values / shots
        )
        parametric_counts = rng.multinomial(
            shots,
            exact_probabilities,
            size=args.bootstrap_repetitions,
        )
        parametric_estimates = (
            parametric_counts @ all_outcome_values / shots
        )
        nonparametric_entry_estimates += nonparametric_estimates
        parametric_entry_estimates += parametric_estimates
        fragment_records.append(
            {
                "fragment_index": fragment["fragment_index"],
                "shots": shots,
                "seed_simulator": simulator_seed,
                "measured_qubits": measured_qubits,
                "counts": counts,
                "outcomes": outcomes,
                "outcome_values": outcome_values,
                "exact_probabilities": exact_probabilities,
                "all_outcome_values": all_outcome_values,
                "exact_mean": exact_mean,
                "exact_single_shot_std": exact_single_shot_std,
                "histogram_mean": histogram_mean,
                "histogram_single_shot_std": histogram_single_shot_std,
                "histogram_standard_error": histogram_standard_error,
                "saved_ideal_fragment_std": float(fragment["ideal_fragment_std"]),
                "nonparametric_bootstrap_mean": float(
                    np.mean(nonparametric_estimates)
                ),
                "nonparametric_bootstrap_std": float(
                    np.std(nonparametric_estimates, ddof=1)
                ),
                "parametric_bootstrap_mean": float(
                    np.mean(parametric_estimates)
                ),
                "parametric_bootstrap_std": float(
                    np.std(parametric_estimates, ddof=1)
                ),
            }
        )

    predicted_entry_std = float(inputs["ideal_std_dev_dict"][uv])
    nonparametric_entry_std = float(
        np.std(nonparametric_entry_estimates, ddof=1)
    )
    parametric_entry_std = float(np.std(parametric_entry_estimates, ddof=1))
    histogram_entry_mean = sum(
        record["histogram_mean"] for record in fragment_records
    )
    payload = {
        "uv": uv,
        "seed": args.seed,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "elapsed_seconds": time.perf_counter() - started,
        "fragments": fragment_records,
        "histogram_quantum_entry_mean": histogram_entry_mean,
        "nonparametric_entry_mean": float(np.mean(nonparametric_entry_estimates)),
        "nonparametric_entry_std": nonparametric_entry_std,
        "parametric_entry_mean": float(np.mean(parametric_entry_estimates)),
        "parametric_entry_std": parametric_entry_std,
        "predicted_ideal_entry_std": predicted_entry_std,
        "nonparametric_to_predicted_ratio": (
            nonparametric_entry_std / predicted_entry_std
        ),
        "parametric_to_predicted_ratio": parametric_entry_std / predicted_entry_std,
        "nonparametric_entry_estimates": nonparametric_entry_estimates,
        "parametric_entry_estimates": parametric_entry_estimates,
    }
    stem = f"bootstrap_noiseless_entry_{uv[0]}_{uv[1]}"
    with (run_dir / f"{stem}.pkl").open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    with (run_dir / f"{stem}.txt").open("w") as file:
        file.write(f"uv: {uv}\n")
        file.write(f"bootstrap_repetitions: {args.bootstrap_repetitions}\n")
        file.write(f"predicted_ideal_entry_std: {predicted_entry_std}\n")
        file.write(f"nonparametric_entry_std: {nonparametric_entry_std}\n")
        file.write(f"parametric_entry_std: {parametric_entry_std}\n")
        file.write(
            f"nonparametric_to_predicted_ratio: "
            f"{nonparametric_entry_std / predicted_entry_std}\n"
        )
        file.write(
            f"parametric_to_predicted_ratio: "
            f"{parametric_entry_std / predicted_entry_std}\n"
        )
        for record in fragment_records:
            file.write(f"fragment: {record}\n")

    print("\nBootstrap verification")
    for record in fragment_records:
        print(
            f"  fragment {record['fragment_index']}: "
            f"saved single-shot std={record['saved_ideal_fragment_std']:.12g}, "
            f"exact={record['exact_single_shot_std']:.12g}, "
            f"count-derived={record['histogram_single_shot_std']:.12g}"
        )
    print(f"  predicted entry std: {predicted_entry_std:.12g}")
    print(f"  count-resampling bootstrap std: {nonparametric_entry_std:.12g}")
    print(f"  exact-probability bootstrap std: {parametric_entry_std:.12g}")
    print(
        f"  exact-probability bootstrap/predicted: "
        f"{parametric_entry_std / predicted_entry_std:.6f}"
    )
    print(f"Saved bootstrap counts and estimates under {run_dir}")


if __name__ == "__main__":
    main()
