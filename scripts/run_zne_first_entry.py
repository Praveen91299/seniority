"""Readout-mitigated local-folding ZNE for split-basis entry (0, 0)."""

from __future__ import annotations

from _bootstrap import REPO_ROOT

import argparse
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
    local_fold_circuit,
    run_local_readout_calibration,
)
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
NOISE_FACTORS = (1, 3, 5)
ENTRY = (0, 0)


def _support_qubits(pubs):
    support = set()
    for _, observable, _, _ in pubs:
        active = np.any(
            np.logical_or(observable.paulis.z, observable.paulis.x),
            axis=0,
        )
        support.update(map(int, np.flatnonzero(active)))
    return sorted(support)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--calibration-shots", type=int, default=20_000)
    parser.add_argument("--calibration-seed", type=int, default=8_675_309)
    parser.add_argument("--simulator-seed", type=int, default=27_182)
    parser.add_argument(
        "--shots-per-fragment",
        type=int,
        help=(
            "Override the cached allocation for a statistically meaningful "
            "diagnostic; the default preserves the cached allocation."
        ),
    )
    args = parser.parse_args()

    # Trusted, locally generated compilation cache. New output is portable.
    with args.cache.open("rb") as file:
        cache = pickle.load(file)

    source_pubs = cache["PUBs"][ENTRY]
    metadata = sorted(
        (
            item
            for item in cache["pub_metadata"]
            if tuple(item["uv"]) == ENTRY
        ),
        key=lambda item: item["fragment_index"],
    )
    if len(source_pubs) != len(metadata):
        raise ValueError("PUB/metadata mismatch for entry (0, 0).")

    noise_backend = FakeQuebec()
    folded_pubs = []
    folded_manifest = []
    for scale_factor in NOISE_FACTORS:
        for source_pub, item in zip(source_pubs, metadata):
            circuit, observable, parameter_values, precision = source_pub
            expected_shots = int(
                cache["shot_alloc"][ENTRY][item["fragment_index"]]
            )
            if int(item["shots"]) != expected_shots:
                raise ValueError("Saved fragment allocation is inconsistent.")
            effective_shots = (
                expected_shots
                if args.shots_per_fragment is None
                else int(args.shots_per_fragment)
            )
            precision = 1.0 / np.sqrt(effective_shots)
            folded = local_fold_circuit(
                circuit,
                scale_factor,
                backend=noise_backend,
            )
            pub_index = len(folded_pubs)
            folded.metadata = {
                **(folded.metadata or {}),
                "qsense_pub_index": pub_index,
                "qsense_uv": ENTRY,
                "qsense_fragment_index": int(item["fragment_index"]),
            }
            folded_pubs.append(
                (folded, observable, parameter_values, precision)
            )
            folded_manifest.append(
                {
                    "pub_index": pub_index,
                    "uv": ENTRY,
                    "fragment_index": int(item["fragment_index"]),
                    "noise_factor": scale_factor,
                    "cached_allocated_shots": expected_shots,
                    "allocated_shots": effective_shots,
                    "target_precision": precision,
                    "original_depth": circuit.depth(),
                    "folded_depth": folded.depth(),
                    "original_instruction_count": len(circuit.data),
                    "folded_instruction_count": len(folded.data),
                }
            )

    support = _support_qubits(folded_pubs)
    print(
        f"Entry {ENTRY}: {len(source_pubs)} fragments, "
        f"shots per scale={sum(item['allocated_shots'] for item in folded_manifest if item['noise_factor'] == 1)}, "
        f"noise factors={NOISE_FACTORS}.",
        flush=True,
    )
    print(f"Physical readout support: {support}.", flush=True)
    for scale_factor in NOISE_FACTORS:
        depths = [
            item["folded_depth"]
            for item in folded_manifest
            if item["noise_factor"] == scale_factor
        ]
        print(
            f"Scale {scale_factor}: folded circuit depths "
            f"{min(depths)}-{max(depths)}.",
            flush=True,
        )

    mitigator, calibration = run_local_readout_calibration(
        noise_backend,
        support,
        shots=args.calibration_shots,
        seed_simulator=args.calibration_seed,
        simulator_method="statevector",
    )
    print("Readout calibration complete.", flush=True)

    simulator = AerSimulator.from_backend(
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
    result = estimator.run(folded_pubs).result()
    print("Folded noisy simulations complete.", flush=True)

    estimates = {factor: 0.0 for factor in NOISE_FACTORS}
    variances = {factor: 0.0 for factor in NOISE_FACTORS}
    fragment_results = []
    raw_by_pub = {}
    for histogram in RAW_SIMULATOR_COUNTS.snapshot():
        raw_by_pub.setdefault(histogram["pub_index"], []).append(histogram)
    for item, pub_result in zip(folded_manifest, result):
        factor = item["noise_factor"]
        estimate = np.asarray(pub_result.data.evs).item()
        std = np.asarray(pub_result.data.stds).item()
        estimates[factor] += estimate
        variances[factor] += std**2
        fragment_results.append(
            {
                **item,
                "mitigated_fragment_estimate": estimate,
                "mitigated_fragment_std": std,
                "raw_measurement_histograms": raw_by_pub.get(
                    item["pub_index"], []
                ),
            }
        )

    classical = complex(cache["H_classical"][ENTRY])
    scale_estimates = np.asarray(
        [estimates[factor] + classical for factor in NOISE_FACTORS]
    )
    scale_variances = np.asarray(
        [variances[factor] for factor in NOISE_FACTORS],
        dtype=float,
    )
    zne_estimate, zne_variance, coefficients = linear_zne(
        NOISE_FACTORS,
        scale_estimates,
        scale_variances,
    )
    ideal = complex(cache["Hsub"][ENTRY])
    bias = zne_estimate - ideal
    empirical_std = float(np.sqrt(max(zne_variance, 0.0)))
    empirical_rmse = float(np.sqrt(abs(bias) ** 2 + zne_variance))

    # With one shot per fragment, sample variances can be identically zero.
    # Also report variance propagated from the exact noiseless fragment SDs.
    scale_one_manifest = [
        item for item in folded_manifest if item["noise_factor"] == 1
    ]
    ideal_scale_variance = sum(
        float(np.real(sigma)) ** 2 / int(item["allocated_shots"])
        for sigma, item in zip(
            cache["sig_frag_mat"][ENTRY],
            scale_one_manifest,
        )
    )
    ideal_variance_proxy = float(
        np.sum(coefficients**2) * ideal_scale_variance
    )
    ideal_std_proxy = float(np.sqrt(ideal_variance_proxy))
    rmse_proxy = float(np.sqrt(abs(bias) ** 2 + ideal_variance_proxy))

    output_dir = args.run_dir / (
        "zne_first_entry_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output_dir.mkdir(parents=True)
    payload = {
        "entry": ENTRY,
        "noise_factors": NOISE_FACTORS,
        "linear_extrapolation_coefficients": coefficients,
        "scale_estimates": scale_estimates,
        "scale_stds": np.sqrt(scale_variances),
        "ideal_estimate": ideal,
        "zne_estimate": zne_estimate,
        "bias": bias,
        "empirical_std": empirical_std,
        "empirical_rmse": empirical_rmse,
        "ideal_sampling_std_proxy": ideal_std_proxy,
        "rmse_with_ideal_sampling_proxy": rmse_proxy,
        "shots_per_scale": int(
            sum(item["allocated_shots"] for item in scale_one_manifest)
        ),
        "total_fragment_shots": int(
            len(NOISE_FACTORS)
            * sum(item["allocated_shots"] for item in scale_one_manifest)
        ),
        "calibration_shots": 2 * args.calibration_shots,
        "physical_readout_qubits": support,
        "simulator_seed": args.simulator_seed,
        "calibration_seed": args.calibration_seed,
    }
    save_json(output_dir / "results.json.gz", payload)
    save_json(output_dir / "calibration.json.gz", calibration)
    save_json(output_dir / "fragment_results.json.gz", fragment_results)
    with (output_dir / "folded_circuits.qpy").open("wb") as file:
        qpy.dump([pub[0] for pub in folded_pubs], file)
    np.savez(
        output_dir / "zne_arrays.npz",
        noise_factors=np.asarray(NOISE_FACTORS),
        scale_estimates=scale_estimates,
        scale_variances=scale_variances,
        coefficients=coefficients,
    )

    print("\nReadout-mitigated scale estimates")
    for factor, estimate, variance in zip(
        NOISE_FACTORS,
        scale_estimates,
        scale_variances,
    ):
        print(
            f"  lambda={factor}: {estimate} +- {np.sqrt(variance)}",
            flush=True,
        )
    print(f"Ideal entry: {ideal}")
    print(f"Linear ZNE estimate: {zne_estimate}")
    print(f"Observed bias: {bias}")
    print(f"Empirical ZNE standard deviation: {empirical_std}")
    print(f"Empirical RMSE: {empirical_rmse}")
    print(f"Ideal-sampling standard-deviation proxy: {ideal_std_proxy}")
    print(f"RMSE with ideal-sampling proxy: {rmse_proxy}")
    print(f"Saved portable results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
