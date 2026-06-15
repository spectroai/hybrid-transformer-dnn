# Data Schema

The raw training data are not committed by default. Add the following files locally before training:

## `gnps_clean.csv`

Required columns:

- `mz_values`: list-like m/z values for each spectrum, stored as a Python-style list string or parsed list.
- `intensities`: list-like peak intensities aligned with `mz_values`.
- `precursor_type`: precursor/adduct type used as an embedding input.

Recommended molecule identifier columns for leakage-safe splitting:

- `inchikey`
- `molecule_id`
- `smiles`

The script uses this priority order: `inchikey`, then `molecule_id`, then `smiles`. If RDKit is installed, SMILES are normalized before grouping.

## `gnps_clean_PubChemFP.csv`

Required content:

- One row per spectrum, aligned with `gnps_clean.csv`.
- Binary fingerprint columns for the target multilabel PubChem fingerprint.

Optional columns automatically ignored:

- `Original_Index`
- `SMILES`
