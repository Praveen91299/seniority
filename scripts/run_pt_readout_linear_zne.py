"""Run readout mitigation and probabilistic linear ZNE for the H2O PT matrix.

Prepare the PT cache first with ``prepare_pt_sampling_cache.py``.  Entry
(0, 0) is fixed to its ideal value; all other quantumly sampled independent
entries use local readout mitigation and factors 1, 1.5, and 2 with randomized
probabilistic local folding.
"""

from _bootstrap import REPO_ROOT

import os
import runpy
from pathlib import Path


HERE = Path(__file__).resolve().parent
PT_CACHE = (
    REPO_ROOT
    / "saved/runtime_cache"
    / "sampling_and_sv_h2o_1.0_pt_mp2_qwc_fake_quebec.pkl"
)
PT_OUTPUT_ROOT = REPO_ROOT / "saved/runtime_runs/pt_h2o_1.0"

if not PT_CACHE.exists():
    raise FileNotFoundError(
        f"Missing PT sampling cache: {PT_CACHE}\n"
        "Run: python scripts/prepare_pt_sampling_cache.py"
    )

os.environ["QSENSE_ZNE_CACHE"] = str(PT_CACHE)
os.environ["QSENSE_ZNE_SOURCE_RUN"] = str(PT_OUTPUT_ROOT)
os.environ["QSENSE_ZNE_OUTPUT_LABEL"] = "pt_readout_linear_zne"

runpy.run_path(
    str(HERE / "run_full_matrix_readout_linear_zne.py"),
    run_name="__main__",
)
