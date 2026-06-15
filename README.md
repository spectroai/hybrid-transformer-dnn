# Hybrid Transformer-DNN for Mass-Spectrum Fingerprint Prediction

This repository contains the training code and model artifact for a hybrid neural network that predicts molecular fingerprints from tandem mass spectra. The model combines a binned-spectrum multilayer perceptron with a Transformer encoder over the top intensity-ranked peaks.

## Repository Contents

- `train_hybrid_fp.py` - reproducible training pipeline with molecule-grouped train/validation/test splitting, early stopping, checkpointing, and final test metrics.
- `train.ipynb` - original notebook version of the training workflow.
- `best_model_complete.pth` - saved PyTorch checkpoint from the completed training run.
- `MODEL_CARD.md` - model artifact description, intended use, and limitations.
- `data/README.md` - expected input data schema.
- `results/README.md` - recommended location for exported tables, figures, and metric summaries.

## Method Summary

The pipeline builds two spectrum representations:

1. A normalized binned intensity vector from m/z 20 to 1200 with 1 Da bins.
2. A fixed-length peak sequence containing up to 300 peaks, normalized per spectrum and encoded by a Transformer.

The two representations are fused and passed through a multilabel DNN to predict an 881-dimensional PubChem fingerprint. Splitting is performed at molecule level using `inchikey`, `molecule_id`, or normalized `smiles` to reduce train/test leakage.

## Installation

Create a clean Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

RDKit is optional but recommended for SMILES normalization during molecule-grouped splitting:

```bash
conda install -c conda-forge rdkit
```

## Data

The training script expects two row-aligned CSV files:

```text
gnps_clean.csv
gnps_clean_PubChemFP.csv
```

Place them in the repository root or pass explicit paths:

```bash
python train_hybrid_fp.py --spectra-csv path/to/gnps_clean.csv --fingerprint-csv path/to/gnps_clean_PubChemFP.csv
```

See `data/README.md` for required columns.

## Training

Run the full training and evaluation pipeline:

```bash
python train_hybrid_fp.py
```

Outputs are written to `checkpoints/` by default:

- `best_model_complete.pth`
- `best_metrics.json`
- `tmp_best.pth`

Use a custom output directory with:

```bash
python train_hybrid_fp.py --save-dir checkpoints/run_001
```

## Reproducibility Notes

- Random seed: `42`
- Split strategy: molecule-grouped `GroupShuffleSplit`
- Train/validation/test target split: approximately 72% / 8% / 20%
- Early stopping metric: validation sample-F1
- Final metrics are computed once on the held-out test split

## Citation

If you use this code or model artifact, please cite the associated paper. Update `CITATION.cff` with the final paper metadata before public release.

## Before Public Release

Replace these placeholders:

- `CITATION.cff`: author names and repository URL.
- `LICENSE`: copyright holder.
- `README.md`: paper title, DOI, and dataset access instructions if applicable.

## License

This project is prepared with an MIT license placeholder. Replace the copyright holder in `LICENSE` before publishing.
