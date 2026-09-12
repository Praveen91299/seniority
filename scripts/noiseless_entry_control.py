"""Run a noiseless control for one matrix entry from a saved noisy run."""

from _bootstrap import REPO_ROOT

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
from qiskit.primitives import BackendEstimatorV2
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--column", type=int, default=0)
    parser.add_argument("--seed", type=int, default=271828)
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

    pubs = [
        (
            item["circuit"],
            item["observable"],
            item["parameter_values"],
            item["target_precision"],
        )
        for item in fragments
    ]
    estimator = BackendEstimatorV2(
        backend=AerSimulator(method="statevector", enable_truncation=True),
        options={"abelian_grouping": True, "seed_simulator": args.seed},
    )

    print(
        f"Running {len(pubs)} noiseless PUBs for entry {uv} with shots "
        f"{[item['allocated_shots'] for item in fragments]}...",
        flush=True,
    )
    started = time.perf_counter()
    results = estimator.run(pubs).result()
    elapsed = time.perf_counter() - started

    fragment_estimates = [
        float(np.asarray(result.data.evs).item()) for result in results
    ]
    fragment_stds = [
        float(np.asarray(result.data.stds).item()) for result in results
    ]
    quantum_estimate = sum(fragment_estimates)
    quantum_std = float(np.sqrt(np.sum(np.square(fragment_stds))))
    classical = complex(inputs["classical_matrix"][uv])
    ideal = complex(inputs["ideal_matrix"][uv])
    full_estimate = classical + quantum_estimate
    bias = full_estimate - ideal
    payload = {
        "uv": uv,
        "seed_simulator": args.seed,
        "elapsed_seconds": elapsed,
        "allocated_shots": [item["allocated_shots"] for item in fragments],
        "fragment_estimates": fragment_estimates,
        "fragment_stds": fragment_stds,
        "quantum_estimate": quantum_estimate,
        "quantum_std": quantum_std,
        "classical_contribution": classical,
        "full_estimate": full_estimate,
        "ideal_estimate": ideal,
        "bias": bias,
    }

    stem = f"noiseless_control_entry_{uv[0]}_{uv[1]}"
    with (run_dir / f"{stem}.pkl").open("wb") as file:
        pickle.dump(payload, file)
    with (run_dir / f"{stem}.txt").open("w") as file:
        for key, value in payload.items():
            file.write(f"{key}: {value}\n")

    print(f"Fragment estimates: {fragment_estimates}")
    print(f"Fragment standard deviations: {fragment_stds}")
    print(f"Full noiseless estimate: {full_estimate}")
    print(f"Ideal entry: {ideal}")
    print(f"Bias: {bias}")
    print(f"Elapsed seconds: {elapsed:.3f}")
    print(f"Saved control results under {run_dir}")


if __name__ == "__main__":
    main()
