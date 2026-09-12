"""Bootstrap a matrix entry directly from saved noisy count histograms."""

from _bootstrap import REPO_ROOT

import argparse
from pathlib import Path

import numpy as np
from src.saved_run_io import load_experiment_inputs, save_json


def _latest_completed_run(runs_dir, allow_legacy_pickle=False):
    candidates = [
        path.parent
        for path in runs_dir.glob("*/fragment_manifest.json.gz")
        if (path.parent / "experiment_inputs.json.gz").exists()
    ]
    if allow_legacy_pickle:
        candidates.extend(
            path.parent
            for path in runs_dir.glob("*/fragment_artifacts.pkl")
            if (path.parent / "experiment_inputs.pkl").exists()
        )
    if not candidates:
        raise FileNotFoundError(f"No completed runs found under {runs_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _coefficient_map(observable):
    coefficients = {}
    for label, coefficient in observable.to_list():
        coefficients[label] = coefficients.get(label, 0.0) + coefficient
    return coefficients


def _outcome_values(histogram, coefficient_map):
    original_labels = histogram["original_pauli_labels"]
    measured_labels = histogram["measured_pauli_labels"]
    coefficients = np.asarray(
        [coefficient_map[label] for label in original_labels],
        dtype=complex,
    )
    bitstrings = list(histogram["counts"])
    values = np.empty(len(bitstrings), dtype=complex)
    for outcome_index, bitstring in enumerate(bitstrings):
        measured = bitstring.split(" ", 1)[0]
        outcome = int(measured, 2)
        signs = []
        for label in measured_labels:
            support_mask = 0
            for qubit, pauli in enumerate(reversed(label)):
                if pauli != "I":
                    support_mask |= 1 << qubit
            signs.append((-1) ** ((support_mask & outcome).bit_count() % 2))
        values[outcome_index] = coefficients @ np.asarray(signs)
    frequencies = np.asarray(
        [histogram["counts"][bitstring] for bitstring in bitstrings],
        dtype=int,
    )
    return bitstrings, frequencies, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--column", type=int, default=0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument(
        "--allow-legacy-pickle",
        action="store_true",
        help="Allow trusted older runs that predate portable JSON/QPY output.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or _latest_completed_run(
        Path("saved/runtime_runs"),
        allow_legacy_pickle=args.allow_legacy_pickle,
    )
    uv = (max(args.row, args.column), min(args.row, args.column))
    inputs = load_experiment_inputs(
        run_dir,
        allow_legacy_pickle=args.allow_legacy_pickle,
    )
    all_fragments = inputs["fragments"]

    fragments = [
        record for record in all_fragments
        if tuple(record["uv"]) == uv
    ]
    if not fragments:
        raise ValueError(f"No quantum fragments found for entry {uv}.")
    if any(not record["raw_measurement_histograms"] for record in fragments):
        raise ValueError(
            "This run predates raw histogram capture or was not produced by "
            "the local capturing estimator."
        )

    rng = np.random.default_rng(args.seed)
    entry_samples = np.zeros(args.bootstrap_repetitions, dtype=complex)
    fragment_summaries = []

    for fragment in fragments:
        fragment_samples = np.zeros(
            args.bootstrap_repetitions,
            dtype=complex,
        )
        coefficient_map = _coefficient_map(fragment["observable"])
        group_summaries = []
        for histogram in fragment["raw_measurement_histograms"]:
            bitstrings, frequencies, values = _outcome_values(
                histogram,
                coefficient_map,
            )
            shots = int(frequencies.sum())
            probabilities = frequencies / shots
            resampled = rng.multinomial(
                shots,
                probabilities,
                size=args.bootstrap_repetitions,
            )
            group_samples = resampled @ values / shots
            fragment_samples += group_samples
            group_summaries.append(
                {
                    "measurement_group": histogram["measurement_group"],
                    "shots": shots,
                    "bitstrings": bitstrings,
                    "frequencies": frequencies,
                    "outcome_values": values,
                    "histogram_mean": probabilities @ values,
                }
            )
        entry_samples += fragment_samples
        fragment_summaries.append(
            {
                "fragment_index": fragment["fragment_index"],
                "ideal_fragment_estimate": fragment[
                    "ideal_fragment_estimate"
                ],
                "saved_noisy_fragment_estimate": fragment[
                    "noisy_fragment_estimate"
                ],
                "histogram_fragment_estimate": sum(
                    group["histogram_mean"] for group in group_summaries
                ),
                "bootstrap_mean": np.mean(fragment_samples),
                "bootstrap_std": np.std(fragment_samples, ddof=1),
                "measurement_groups": group_summaries,
            }
        )

    classical = complex(inputs["classical_matrix"][uv])
    ideal = complex(inputs["ideal_matrix"][uv])
    entry_samples += classical
    mean = np.mean(entry_samples)
    std = np.std(entry_samples, ddof=1)
    payload = {
        "uv": uv,
        "seed": args.seed,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "classical_contribution": classical,
        "ideal_entry": ideal,
        "bootstrap_mean": mean,
        "bootstrap_std": std,
        "bootstrap_bias": mean - ideal,
        "fragments": fragment_summaries,
    }
    stem = f"bootstrap_saved_histograms_{uv[0]}_{uv[1]}"
    output = run_dir / f"{stem}.json.gz"
    save_json(output, payload)
    np.savez(
        run_dir / f"{stem}_samples.npz",
        bootstrap_samples=entry_samples,
    )

    print(f"Entry: {uv}")
    print(f"Ideal: {ideal}")
    print(f"Bootstrap mean: {mean}")
    print(f"Bootstrap standard deviation: {std}")
    print(f"Bootstrap bias: {mean - ideal}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
