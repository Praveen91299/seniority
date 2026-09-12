# H2O Q-SENSE dump files

This directory contains the original H2O Q-SENSE data and derived data such
as the fixed-orbital split of state 0. The files are Python pickles and should
only be loaded from a trusted source.

## Original `UCSF_sym_comp` dumps

Files named

```text
UCSF_sym_comp_for_Praveen_Smik_<bond_length>.dump
```

contain a 13-element list. The standard unpacking order is:

```python
from pathlib import Path
import pickle

path = Path(
    "data/h2o_data/UCSF_sym_comp_for_Praveen_Smik_1.0.dump"
)

with path.open("rb") as file:
    (
        CSF_tz_states,
        W_amplitudes,
        list_list_theta_CSF,
        list_sym_CSF_vec,
        list_UCSF_tz,
        UCSF_tz_states,
        somos_list,
        psi_GS_UCSF_smik,
        list_orb_rot,
        x_orbrot,
        Enuc,
        obt_spatial,
        tbt_spatial,
    ) = pickle.load(file)
```

The entries have the following meanings:

1. `CSF_tz_states`: reference CSFs in TZ sparse-state format.
2. `W_amplitudes`: pair-excitation supports used to define each W circuit.
3. `list_list_theta_CSF`: optimized pair-rotation angles. Repeated blocks of
   angles represent repeated W layers.
4. `list_sym_CSF_vec`: symmetry-component metadata from the CSF construction.
5. `list_UCSF_tz`: optimized UCSF states before selecting their symmetry
   components, in TZ format.
6. `UCSF_tz_states`: symmetry-resolved UCSF states used as the Q-SENSE basis,
   in TZ format.
7. `somos_list`: singly occupied molecular orbitals for each basis state.
8. `psi_GS_UCSF_smik`: classical ground-state coefficient vector in the
   Q-SENSE basis.
9. `list_orb_rot`: orbital-rotation pairs.
10. `x_orbrot`: orbital-rotation parameters.
11. `Enuc`: nuclear-repulsion energy.
12. `obt_spatial`: spatial-orbital one-body tensor.
13. `tbt_spatial`: spatial-orbital two-body tensor.

### TZ state format and qubit count

Each TZ state is represented as:

```python
[determinant_bit_arrays, decimal_indices, coefficients]
```

For this seven-spatial-orbital H2O problem, every determinant bit array has
length 14 and describes the full spin-orbital state. The state is sparse; the
dump does **not** store a dense array of length `1 << 14`.

The following converts one basis state to the full sparse 14-qubit state and
then compresses/tapers the seven orbital-seniority qubits:

```python
from src.measurement_new.utils_states import (
    compress_state,
    convert_TZ_format_to_sparse_format,
)

n_orbitals = len(obt_spatial)       # 7
n_spin_qubits = 2 * n_orbitals      # 14

state_14q_sparse = convert_TZ_format_to_sparse_format(
    1 << n_spin_qubits,
    UCSF_tz_states[0],
)
state_7q_dense = compress_state(state_14q_sparse.toarray()[0])

assert state_14q_sparse.shape == (1, 1 << 14)
assert state_7q_dense.shape == (1 << 7,)
```

The corresponding logical seven-qubit state-preparation objects can be
reconstructed and optionally verified using:

```python
from src.circuits.circuits_csf import get_csfs_from_dump

csfs = get_csfs_from_dump(
    path,
    verify_states=True,
    verbose=False,
)

state0_circuit = csfs[0].get_tapered_full_circuit()
```

`get_csfs_from_dump` expects the original 13-element schema. It does not
directly parse the derived split-state dictionary described below.

## State-0 fixed-orbital split dump

The file

```text
UCSF_sym_comp_for_Praveen_Smik_1.0_state0_orbital0_split.dump
```

is a dictionary with format tag:

```python
"qsense_fixed_orbital_split_v1"
```

Load it with:

```python
from pathlib import Path
import pickle

split_path = Path(
    "data/h2o_data/"
    "UCSF_sym_comp_for_Praveen_Smik_1.0_state0_orbital0_split.dump"
)

with split_path.open("rb") as file:
    split = pickle.load(file)

assert split["format"] == "qsense_fixed_orbital_split_v1"

sector_0 = split["sectors"][0]  # tapered orbital/qubit 0 fixed to |0>
sector_1 = split["sectors"][1]  # tapered orbital/qubit 0 fixed to |1>

circuit_0 = sector_0["fixed_sector_circuit"]
circuit_1 = sector_1["fixed_sector_circuit"]
state_0 = sector_0["prepared_full_statevector"]
state_1 = sector_1["prepared_full_statevector"]
angles_0 = sector_0["optimized_angles"]
angles_1 = sector_1["optimized_angles"]
```

Important top-level fields are:

- `source_dump_file`: path recorded when the split was constructed.
- `source_dump_payload`: complete original 13-element dump payload.
- `state_index`: original state that was split; currently `0`.
- `fixed_orbital`: tapered orbital/qubit that was fixed; currently `0`.
- `source_circuit` and `source_statevector`: original seven-qubit state 0.
- `removed_generators`: generators acting on fixed orbital 0.
- `retained_generators`: generators retained and reoptimized in each sector.
- `sectors[0]` and `sectors[1]`: normalized fixed-sector states and circuits.
- `reconstructed_statevector`, `reconstruction_overlap`,
  `reconstruction_fidelity`, and `reconstruction_l2_error`: verification that
  the two weighted sectors recover the original state.

Each sector contains:

- its fixed value and sector probability;
- `reconstruction_weight = sqrt(sector_probability)`;
- the reference occupation and active-orbital mapping;
- warm-start and optimized angles;
- exact projected target and prepared statevectors;
- active six-qubit and fixed-sector seven-qubit circuits;
- circuit metrics, overlaps, phase correction, and optimizer diagnostics.

### Meaning of `full` and `active` in the split dump

In this derived dump, `target_full_statevector` and
`prepared_full_statevector` are dense vectors on the **full seven-qubit
compressed register**, so their length is `1 << 7 = 128`. They are not dense
14-qubit spin-orbital states.

`target_active_statevector` and `prepared_active_statevector` omit fixed
orbital 0 and therefore have length `1 << 6 = 64`.

The occupation lists are ordered as:

```text
[q0, q1, q2, q3, q4, q5, q6]
```

The two optimized reference occupations are:

```python
sector_0["reference_occupation"] == [0, 1, 1, 1, 1, 1, 0]
sector_1["reference_occupation"] == [1, 1, 1, 1, 1, 0, 0]
```

The normalized `q0=0` state contains 6 determinants and the normalized
`q0=1` state contains 15 determinants. Together they partition the original
21 determinants.

The original state is reconstructed as:

```python
reconstructed = (
    sector_0["reconstruction_weight"]
    * sector_0["prepared_full_statevector"]
    + sector_1["reconstruction_weight"]
    * sector_1["prepared_full_statevector"]
)
```

The reconstruction weights are only for recovering the old state. The two
normalized sector states are used independently, without these weights, when
they replace state 0 in the expanded Q-SENSE basis.

The saved circuits are logical state-preparation circuits. They do not include
measurement circuits and are not the optimization-level-3, backend-native
transpiled circuits. Those are constructed and saved by the sampling scripts.

## Running the split-basis sampling benchmark

The split benchmark uses the basis ordering:

```text
state_0_Z0_0, state_0_Z0_1, original_state_1, ..., original_state_10
```

Run it from the `seniority` source directory:

```bash
python scripts/sampling_and_sv_orbital0_split.py
```

The benchmark retains a confirmation prompt before starting the local noisy
simulation. Its completed run directories under `saved/runtime_runs` contain
the ideal/noisy matrices, eigenvectors, shot allocations, observables,
transpiled circuits, fragment estimates, and raw measurement histograms.
