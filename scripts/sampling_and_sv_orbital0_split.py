"""Run the sampling benchmark with original state 0 split by fixed orbital 0.

Basis order:
    state_0_Z0_0, state_0_Z0_1, original_state_1, ..., original_state_10.

The sampling-cost analysis, shot allocation, level-3 backend transpilation,
saved artifacts, and local FakeQuebec noisy simulation are provided by
``sampling_and_sv.py``.  Its explicit confirmation prompt is retained before
the noisy estimator job is submitted.
"""

from _bootstrap import REPO_ROOT

import os
import runpy
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPLIT_DUMP = (
    REPO_ROOT
    / "data/h2o_data"
    / "UCSF_sym_comp_for_Praveen_Smik_1.0_state0_orbital0_split.dump"
)

if not SPLIT_DUMP.exists():
    raise FileNotFoundError(
        f"Missing split-state dump: {SPLIT_DUMP}\n"
        "Run scripts/build_orbital0_split_states.py first."
    )

os.environ["QSENSE_SPLIT_BASIS_DUMP"] = str(SPLIT_DUMP)
os.environ["QSENSE_BASIS_VARIANT"] = "state0_orbital0_split_v1"
os.environ.setdefault("QSENSE_REBUILD_RUNTIME_CACHE", "0")

runpy.run_path(str(HERE / "sampling_and_sv.py"), run_name="__main__")
