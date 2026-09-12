"""Portable loaders for sampling_and_sv run artifacts."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
from qiskit import qpy
from qiskit.quantum_info import SparsePauliOp


def _decode_json(value):
    if isinstance(value, list):
        return [_decode_json(item) for item in value]
    if not isinstance(value, dict):
        return value
    if set(value) == {"__complex__"}:
        real, imag = value["__complex__"]
        return complex(real, imag)
    if set(value) == {"__tuple__"}:
        return tuple(_decode_json(item) for item in value["__tuple__"])
    if set(value) == {"__mapping__"}:
        return {
            _decode_json(key): _decode_json(item)
            for key, item in value["__mapping__"]
        }
    return {key: _decode_json(item) for key, item in value.items()}


def _encode_json(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return _encode_json(value.item())
    if isinstance(value, complex):
        return {"__complex__": [value.real, value.imag]}
    if isinstance(value, np.ndarray):
        return [_encode_json(item) for item in value.tolist()]
    if isinstance(value, tuple):
        return {"__tuple__": [_encode_json(item) for item in value]}
    if isinstance(value, list):
        return [_encode_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        if all(isinstance(key, str) for key in value):
            return {key: _encode_json(item) for key, item in value.items()}
        return {
            "__mapping__": [
                [_encode_json(key), _encode_json(item)]
                for key, item in value.items()
            ]
        }
    raise TypeError(f"Cannot encode {type(value).__name__} as JSON.")


def save_json(path, payload):
    """Save a plain or gzip-compressed reversible JSON artifact."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as file:
        json.dump(_encode_json(payload), file, separators=(",", ":"))
        file.write("\n")


def load_json(path):
    """Load a plain or gzip-compressed JSON artifact."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as file:
        return _decode_json(json.load(file))


def load_numeric_arrays(run_dir, filename="matrices_and_eigenvectors.npz"):
    """Load numerical run arrays without enabling NumPy object unpickling."""
    with np.load(Path(run_dir) / filename, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def load_fragment_artifacts(run_dir, allow_legacy_pickle=False):
    """Reconstruct fragment dictionaries, including circuits and observables."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / "fragment_manifest.json.gz"
    circuits_path = run_dir / "fragment_circuits.qpy"
    if manifest_path.exists() and circuits_path.exists():
        records = load_json(manifest_path)
        with circuits_path.open("rb") as file:
            circuits = list(qpy.load(file))
        if len(records) != len(circuits):
            raise ValueError(
                "Fragment manifest/circuit count mismatch: "
                f"{len(records)} records versus {len(circuits)} circuits."
            )
        for record in records:
            record["circuit"] = circuits[record["circuit_index"]]
            record["observable"] = SparsePauliOp.from_list(
                [
                    (term["label"], term["coefficient"])
                    for term in record.pop("observable_terms")
                ]
            )
        return records

    legacy_path = run_dir / "fragment_artifacts.pkl"
    if allow_legacy_pickle and legacy_path.exists():
        with legacy_path.open("rb") as file:
            return pickle.load(file)
    raise FileNotFoundError(
        f"No portable fragment bundle found under {run_dir}. "
        "Pass allow_legacy_pickle=True only for a trusted older run."
    )


def load_experiment_inputs(run_dir, allow_legacy_pickle=False):
    """Load experiment metadata and attach numerical arrays and fragments."""
    run_dir = Path(run_dir)
    json_path = run_dir / "experiment_inputs.json.gz"
    if json_path.exists():
        inputs = load_json(json_path)
        inputs.update(
            load_numeric_arrays(run_dir, inputs["numeric_arrays_file"])
        )
        inputs["fragments"] = load_fragment_artifacts(run_dir)
        return inputs

    legacy_path = run_dir / "experiment_inputs.pkl"
    if allow_legacy_pickle and legacy_path.exists():
        with legacy_path.open("rb") as file:
            return pickle.load(file)
    raise FileNotFoundError(
        f"No portable experiment inputs found under {run_dir}. "
        "Pass allow_legacy_pickle=True only for a trusted older run."
    )


def load_saved_run(run_dir, allow_legacy_pickle=False):
    """Load the convenient top-level dictionary for a completed run."""
    run_dir = Path(run_dir)
    results_path = run_dir / "run_results.json.gz"
    if results_path.exists():
        result = load_json(results_path)
        result.update(
            load_numeric_arrays(run_dir, result["numeric_arrays_file"])
        )
        result["fragment_artifacts"] = load_fragment_artifacts(run_dir)
        return result

    legacy_path = run_dir / "run_results.pkl"
    if allow_legacy_pickle and legacy_path.exists():
        with legacy_path.open("rb") as file:
            return pickle.load(file)
    raise FileNotFoundError(
        f"No portable completed-run bundle found under {run_dir}. "
        "Pass allow_legacy_pickle=True only for a trusted older run."
    )
