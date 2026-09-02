# FMN and MR-FMN benchmark code

Minimal scripts used to run the three benchmark analyses. Benchmark data and the trained Transformer-DNN checkpoint are distributed separately and are not duplicated in this repository.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -r requirements.txt
```

## 1. FMN held-out benchmark

```bash
python 01_FMN_heldout_664/run_fmn_network.py \
  --spectra /path/to/01_FMN_heldout_664/gnps_clean_testdata_originindex.csv \
  --fingerprints /path/to/01_FMN_heldout_664/Transformer.csv \
  --output-dir output/fmn
```

The default thresholds are cosine similarity 0.70 and Transformer-DNN fingerprint similarity 0.90, with at most three edges per method and reciprocal neighbor filtering enabled.

## 2. Curated 453-pair MR-FMN benchmark

```bash
python 02_MRFMN_curated_453/run_curated_benchmark.py \
  --data-dir /path/to/02_MRFMN_curated_453 \
  --output-dir output/curated_453
```

The data directory must contain `reactions_verified_true.csv`, `products_enriched.csv`, and `reactants_enriched.csv`.

## 3. External Rhea-GNPS benchmark

```bash
python 03_Rhea_GNPS_external_1031/run_external_benchmark.py \
  --input /path/to/03_Rhea_GNPS_external_1031/positive_pairs_clean.csv \
  --model /path/to/best_model_complete_transformer_DNN.pth
```

The external evaluation writes its results to `03_Rhea_GNPS_external_1031/output/`. The model checkpoint is required for Transformer-DNN scoring but is intentionally excluded from GitHub.

## Repository scope

Only benchmark execution code is included. Data-cleaning utilities, GUI review tools, intermediate plotting scripts, generated results, caches, and model weights are excluded.
