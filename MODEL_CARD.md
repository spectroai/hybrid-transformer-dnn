# Model Card

## Model

Hybrid binned-spectrum MLP plus peak-sequence Transformer for multilabel PubChem fingerprint prediction from tandem mass spectra.

## Intended Use

This model is intended for research use as part of a reproducible mass-spectrometry machine-learning workflow. It can be used to benchmark molecular fingerprint prediction from MS/MS spectra or as a baseline for follow-up experiments.

## Inputs

The model consumes:

- normalized binned spectrum vectors covering m/z 20-1200 with 1 Da bins
- fixed-length peak sequences with up to 300 peaks per spectrum
- precursor/adduct type indices

## Outputs

The model outputs logits for an 881-dimensional PubChem fingerprint. Apply a sigmoid activation and the selected threshold from the checkpoint metadata before converting to binary predictions.

## Training and Evaluation

The training pipeline uses molecule-grouped splitting to reduce leakage between train, validation, and test sets. Validation sample-F1 is used for early stopping and model selection; final metrics are computed once on the held-out test set.

## Limitations

- Performance depends on the preprocessing and quality of the source spectra.
- Applicability outside the training data distribution has not been established.
- The checkpoint should be evaluated against independent external data before use in high-confidence annotation workflows.

## Ethical and Scientific Use

This artifact is provided for transparent research and reproducibility. Users should report dataset provenance, preprocessing choices, train/test split strategy, and all thresholding decisions when reusing or extending the model.
