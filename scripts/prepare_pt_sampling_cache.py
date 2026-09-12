"""Construct the H2O PT sampling cache without running noisy simulations.

The PT basis uses the MP2 Uext states from the trusted local dump.  The shared
sampling workflow constructs the ideal matrix, classifies classical factors,
builds and level-3 transpiles measurement circuits for FakeQuebec, and assigns
shots for the target matrix-estimation accuracy of 1.6e-3.
"""

from _bootstrap import REPO_ROOT

import os
import runpy
from pathlib import Path


HERE = Path(__file__).resolve().parent
PT_DUMP = REPO_ROOT / "data/h2o_data/Uext_CSF_for_Praveen_Smik_1.0.dump"

if not PT_DUMP.exists():
    raise FileNotFoundError(f"Missing PT dump: {PT_DUMP}")

os.environ["QSENSE_SCIENTIFIC_METHOD"] = "pt"
os.environ["QSENSE_BASIS_VARIANT"] = "pt_mp2"
os.environ["QSENSE_PREPARE_CACHE_ONLY"] = "1"
os.environ.setdefault("QSENSE_REBUILD_RUNTIME_CACHE", "0")

runpy.run_path(str(HERE / "sampling_and_sv.py"), run_name="__main__")
