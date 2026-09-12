# Scripts

Run current scripts from the repository root, for example:

```bash
python scripts/sampling_and_sv_orbital0_split.py
python scripts/run_probabilistic_zne_entry.py --entry 1
```

`_bootstrap.py` adds the repository root to `sys.path`; it is support code, not
a standalone program. Relative `data/` and `saved/` paths continue to resolve
from the repository root. Moving the scripts did not move or modify any
existing results.

## Current sampling and analysis scripts

| Script | Created | Purpose | Outputs and existing results |
|---|---|---|---|
| `sampling_and_sv.py` | 2026-06-06 | Main Q-SENSE sampling-cost analysis, shot allocation, level-3 backend transpilation, local FakeQuebec estimation, progress reporting, and portable artifact saving. Supports the original or split basis through environment variables. | Runtime caches under `saved/runtime_cache/`; runs under `saved/runtime_runs/<timestamp>_*`. The latest completed split run is `saved/runtime_runs/20260724T024719Z_h2o_1.0_state0_orbital0_split_v1_qwc_fake_quebec/`. |
| `sampling_and_sv_orbital0_split.py` | 2026-07-23 | Wrapper that selects the state-0 fixed-orbital split dump and runs `sampling_and_sv.py`. Basis order is the two split sectors followed by original states 1--10. | Same locations as `sampling_and_sv.py`; the completed split run is listed above. |
| `build_orbital0_split_states.py` | 2026-07-23 | Removes generators acting on a chosen orbital, refits the remaining angles independently in fixed 0/1 sectors, and verifies reconstruction of the original state. | Default dump and summary beside the source dump. Existing outputs: `data/h2o_data/UCSF_sym_comp_for_Praveen_Smik_1.0_state0_orbital0_split.dump` and `.txt`. |
| `run_readout_mitigated_diagonals.py` | 2026-07-24 | Reuses the split-run diagonal PUBs and allocations, calibrates local assignment matrices, and compares paired raw and readout-mitigated diagonal estimates. | Subdirectories named `readout_mitigated_diagonals_<timestamp>` inside the source run. Preferred completed result: `.../readout_mitigated_diagonals_20260724T211017Z/`. |
| `run_zne_first_entry.py` | 2026-07-25 | Diagnostic full local-folding ZNE for split entry `(0,0)` using factors 1, 3, and 5, with optional fixed shots per fragment. This entry has only one cached shot per fragment and is not the dominant split sector. | `zne_first_entry_<timestamp>/` inside the split run. Latest validated diagnostic: `.../zne_first_entry_20260725T185806Z/`. |
| `run_probabilistic_zne_entry.py` | 2026-07-25 | Readout-mitigated probabilistic local-folding ZNE for a selected diagonal entry. Defaults to entry `(1,1)`, factors 1, 1.5, 2, and eight randomized folds. With `--pilot-results`, it preserves a fixed total budget while allocating factor shots in proportion to the pilot \(\lvert c_i\rvert\sigma_i\), and saves both ordinary and variance-weighted linear fits. | `probabilistic_zne_entry_<i>_<i>_<timestamp>/` inside the split run. Initial equal-allocation `(1,1)` result: `.../probabilistic_zne_entry_1_1_20260725T200028Z/`; optimized-allocation result: `.../probabilistic_zne_entry_1_1_20260725T203218Z/`. |
| `fit_zne_models.py` | 2026-07-27 | Compares variance-weighted linear, quadratic, and offset-exponential models on one or more saved probabilistic-ZNE runs. Independently calibrated estimates are inverse-variance pooled; an optional factor can be held out for prediction. Performs factor-level parametric bootstrap resampling and reports intercept bias, variance, RMSE, residual \(\chi^2\), AIC, and BIC. | Writes `zne_model_comparison.json.gz` and `zne_model_bootstrap_samples.npz` beside the final source result. Four-factor result: `.../probabilistic_zne_entry_1_1_20260727T155621Z/`; final independent-seed pooled five-factor/held-out test: `.../probabilistic_zne_entry_1_1_20260727T162129Z/`. |
| `run_full_matrix_readout_linear_zne.py` | 2026-08-12 | Runs the original 11-state matrix with local readout mitigation and probabilistic local-folding linear ZNE on every quantum entry except `(0,0)`, which is fixed exactly to its ideal value. Defaults to factors 1, 1.5, 2 and eight randomized partial folds. Readout calibration uses MPS for its wide but unentangled product-state circuits; folded problem circuits use truncated statevectors. Runs one factor/randomization batch at a time, saves raw histograms and circuits, constructs the mitigated matrix, and reports both the lowest and ideal-overlap-tracked eigenvalue biases. | Creates `full_matrix_readout_linear_zne_<timestamp>/` inside the original `20260722T164209Z_h2o_1.0_qwc_fake_quebec` run. Use `--dry-run` to inspect the workload first. No complete run exists yet. |
| `prepare_pt_sampling_cache.py` | 2026-08-13 | Constructs the 37-state H2O PT basis from the MP2 `Uext_CSF` dump, verifies its preparation circuits, evaluates the ideal matrix and fragment variances, allocates shots for accuracy `1.6e-3`, and creates optimization-level-3 FakeQuebec PUBs. It exits before noisy simulation. | Writes `saved/runtime_cache/sampling_and_sv_h2o_1.0_pt_mp2_qwc_fake_quebec.pkl`. |
| `run_pt_readout_linear_zne.py` | 2026-08-13 | PT counterpart of the full-matrix mitigation experiment. It fixes `(0,0)` to ideal and applies local readout mitigation plus probabilistic linear ZNE at factors 1, 1.5, and 2 to every other quantum entry. It uses MPS for wide product-state readout calibration and truncated noisy statevectors for folded problem circuits. One-shot zero empirical variances receive the cached analytical ideal finite-shot variance floor; ill-conditioned inverse-variance fits retain the same linear model with ordinary least-squares weights. `--resume-run PATH` performs post-processing from completed portable batches without new simulations. | Creates `pt_readout_linear_zne_<timestamp>/` under `saved/runtime_runs/pt_h2o_1.0/`. Completed result: `pt_readout_linear_zne_20260813T134200Z/`. |
| `resample_pt_entry_23_0_fragment_0.py` | 2026-08-13 | Reruns only the leading PT eigenvector-error fragment, `(23,0)` fragment 0, with several larger shot counts and independent seeds. It reports raw and readout-mitigated entry errors and rediagonalized overlap-tracked energy errors while holding every other saved estimate fixed. | Writes `resample_entry_23_0_fragment_0.json.gz` and `.csv` in the completed PT ZNE run. |
| `resample_noiseless_pt_entry_23_0_fragment_0.py` | 2026-08-13 | Verifies finite-shot convergence of the same PT fragment without a noise model. Other fragments are fixed at their ideal values, so the reported entry error isolates this fragment. | Writes `resample_noiseless_entry_23_0_fragment_0.json.gz` in the completed PT ZNE run. |
| `compare_pt_csf_only_all_bonds.py` | 2026-08-13 | Projects the exact H2O Hamiltonian into both the reference-CSF and saved MP2-rotated PT bases at every available `Uext_CSF` bond length. It verifies orthonormality/rank and reports energies, energy changes, physical ground-state fidelities, and MP2-angle magnitudes. | Writes CSV/JSON/NPZ summaries and per-geometry matrices under `saved/runtime_runs/pt_csf_only_all_bonds/`. |
| `compare_n2_pt_csf_only_all_bonds.py` | 2026-08-13 | N2 counterpart of the all-bond PT/CSF-only comparison. It evaluates the 20-spin-orbital fermionic Hamiltonian directly in the union of saved sparse determinants, avoiding a full `2^20` Jordan--Wigner matrix, and verifies the direct convention against the H2O reference implementation. | Writes CSV/JSON/NPZ summaries and per-geometry matrices under `saved/runtime_runs/n2_pt_csf_only_all_bonds/`. |
| `compare_vo_csf_only_all_bonds.py` | 2026-08-17 | Projects the same orbital-rotated H2O Hamiltonian into the unrotated CSFs and optimized VO/UCSF states at every saved geometry. It verifies orthonormality/rank and compares energies, physical ground-state fidelities, and optimized pair-rotation magnitudes. | Writes CSV/JSON/NPZ summaries and per-geometry matrices under `saved/runtime_runs/vo_csf_only_all_bonds/`. |
| `bootstrap_saved_histograms.py` | 2026-07-23 | Bootstraps a matrix entry directly from portable saved noisy histograms and reconstructs fragment/entry sampling distributions. Trusted legacy pickle loading is opt-in. | Writes `bootstrap_saved_histograms_<i>_<j>.json.gz` and `_samples.npz` into the selected run. No portable output has yet been generated. |
| `bootstrap_noiseless_entry.py` | 2026-07-22 | Runs a noiseless count-level experiment for an I/Z matrix entry and bootstraps the resulting counts as a sampling-standard-deviation check. Legacy-run utility. | Existing `bootstrap_noiseless_entry_0_0.pkl` and `.txt` in `saved/runtime_runs/20260722T164209Z_h2o_1.0_qwc_fake_quebec/`. |
| `noiseless_entry_control.py` | 2026-07-22 | Re-executes saved PUBs without a noise model to separate estimator/sampling behavior from device-model bias. Legacy-run utility. | Existing `noiseless_control_entry_0_0.pkl` and `.txt` in the 20260722 run above. |
| `symmetry_shift_noisy_fragment.py` | 2026-07-22 | Tests an exact parity control-variate shift for one diagonal saved fragment and bootstraps raw and shifted estimates. | Existing `symmetry_shift_entry_0_0_fragment_0.pkl` and `.txt` in the 20260722 run. |
| `number_shift_noisy_fragment.py` | 2026-07-22 | Tests shifts based on the exact pair-number constraint, including user-selected coefficients, using saved fragment counts. | Existing `number_shift_entry_0_0_fragment_0*.pkl` and `.txt` variants in the 20260722 run. |
| `z0_weighted_number_shift.py` | 2026-07-22 | Tests the exact control variate `(N_pair-5)Z_0` and reports diagnostic/variance-optimal coefficients from saved counts. | Existing `z0_weighted_number_shift_entry_0_0_fragment_0.pkl` and `.txt` in the 20260722 run. |

In the table, paths beginning with `.../` are relative to
`saved/runtime_runs/20260724T024719Z_h2o_1.0_state0_orbital0_split_v1_qwc_fake_quebec/`.

## Legacy research scripts

These predate the current `src.measurement_new` workflow. Several contain
historical hard-coded paths or package imports and should be reviewed before
rerunning.

| Script | Created | Purpose | Expected output |
|---|---|---|---|
| `benchmark_circuits.py` | 2025-07-05 | Finds deep CSF-pair circuits and benchmarks CNOT/transpiled-gate counts. | Console output; QASM saving is currently commented out. |
| `main_h2o_method1.py` | 2025-08-17 | Original H2O method-1 sampling study. | Appends to `results/h2o_method1_<bond_length>`. |
| `main_h2o_method2.py` | 2025-08-19 | Original H2O method-2/Uext sampling study. | Appends to `results/h2o_method2_<bond_length>`. |
| `main_h2o_method2_parallel.py` | 2025-08-25 | Joblib-parallel H2O method-2 study. | Appends to `results/h2o_method2_parallel_<bond_length>`. |
| `main_n2_method1.py` | 2025-08-17 | Original N2 method-1 sampling study. | Appends to `results/n2_method1_<bond_length>`. |
| `main_n2_method2.py` | 2025-08-19 | Original N2 method-2/Uext sampling study. | Appends to `results/n2_method2_<bond_length>`. |
| `main_n2_method2_parallel.py` | 2025-08-25 | Joblib-parallel N2 method-2 study. | Appends to `results/n2_method2_parallel_<bond_length>`. |
| `full_h2o_method1.py` | 2025-10-07 | Extended/full H2O method-1 sweep. | Appends to `results/h2o_method1_<bond_length>`. |
| `full_h2o_method2.py` | 2025-10-07 | Extended/full H2O method-2 sweep. | Appends to `results/h2o_method2_<bond_length>`. |
| `dry_run_old.py` | 2025-11-11 | Dry run of the older measurement implementation. | Primarily console diagnostics; result-file writes are commented out. |
| `dry_run_new.py` | 2025-12-02 | Dry run of the newer fragment-generation implementation as it existed in late 2025. | Console diagnostics; no explicit result artifact. |

## Portable output conventions

Current scripts prefer:

- compressed JSON for metadata, calibration, estimates, and histograms;
- QPY for Qiskit circuits;
- NPZ with `allow_pickle=False` for numerical arrays;
- CSV for human-readable comparisons.

The runtime compilation cache and historical source dumps remain trusted local
pickle files because they contain live Qiskit or legacy Python objects.
