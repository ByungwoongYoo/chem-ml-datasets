# chem-ood-eval: scoring script and baselines

`score.py` scores property-prediction models on frozen random, scaffold and cluster test splits and prints your R², RMSE and MAE next to our random-forest baselines (RDKit descriptors and Morgan fingerprints). Pure Python 3, no dependencies.

- `BASELINES_5seed.csv`: all 120 baselines across 14 chemistry property sets, as mean ± sd over 5 RF seeds (`n_seeds`, `R2_mean`, `R2_sd`, `RMSE_mean`, `RMSE_sd`, `MAE_mean`, `MAE_sd`, `license`).
- `BASELINES.csv`: the original single-seed table.
- `manifest.json`: maps each set to its CSV and targets.

Splits are frozen: the train/valid/test assignment never changes between seeds. The multi-seed numbers vary only the random-forest seed, so they apply to the RF baselines only.

Free 150–200 row previews are on Hugging Face: https://huggingface.co/woongworks

## Usage

Put the pack folders under `packs/` next to the script.

    python3 score.py PACK [SPLIT] predictions.csv [--target COL]
    python3 score.py PACK [SPLIT] --template out.csv [--target COL]
    python3 score.py PACK [SPLIT] --diagnose [--target COL]

`SPLIT` is optional. It defaults to `scaffold_split`. You can also write `random`, `scaffold` or `cluster`.

## 1. Multiple prediction columns (e.g. one per training seed)

Every column whose name starts with `pred` gets its own score. With two or more columns, the script also prints mean ± sd across them. Each line is compared with the 5-seed RF baseline mean ± sd:

    $ python3 score.py melting_point test_preds/melting_point_scaffold_split_mp_C_5seed.csv
    melting_point scaffold_split target=mp_C (1953 labelled test rows)
      yours [pred_seed0] n=1953: R2=0.425 RMSE=56.76 MAE=42.49
      ...
      yours [pred_seed4] n=1953: R2=0.433 RMSE=56.37 MAE=42.23
      yours mean +- sd over 5 columns: R2=0.429 +- 0.004  RMSE=56.568 +- 0.202  MAE=42.384 +- 0.134
      baseline RF_Morgan2048 (5 RF seeds): R2=0.071 +- 0.002 RMSE=72.16 +- 0.085 (n_test=1953)
      baseline RF_RDKit_descriptors (5 RF seeds): R2=0.476 +- 0.001 RMSE=54.17 +- 0.075 (n_test=1953)

## 2. Single column, no row_index

A file with only a `pred` column works when it lists the test rows in order:

    $ python3 score.py melting_point cluster_split test_preds/melting_point_cluster_split_mp_C_single_noindex.csv
    melting_point cluster_split target=mp_C (1849 labelled test rows)
      yours [pred] n=1849: R2=0.689 RMSE=48.28 MAE=36.98
      baseline RF_Morgan2048 (5 RF seeds): R2=0.603 +- 0.003 RMSE=54.51 +- 0.2 (n_test=1849)
      baseline RF_RDKit_descriptors (5 RF seeds): R2=0.749 +- 0.001 RMSE=43.39 +- 0.071 (n_test=1849)

## 3. --diagnose

This shows label stats for each fold, the test-vs-train shift in train-sd units, a predict-the-train-mean RMSE, exact-SMILES test/train overlap, and flag counts:

    $ python3 score.py henry scaffold --diagnose
    henry scaffold_split target=log10_Hscp_298K  (labels from henry_ml_ready_v0.csv)
      fold      rows  labelled       mean         sd        min        max
      train     3382      3382     -2.055      3.074     -9.097      15.69
      valid      422       422    -0.8286      3.407     -5.456      12.91
      test       424       424      1.843       3.28     -5.921      12.95
      test-vs-train label mean shift: +1.268 train-sd units (test mean 1.843 vs train mean -2.055, train sd 3.074)
      predict-the-train-mean on test (n=424): RMSE=5.092 MAE=4.249
      labelled test rows whose smiles string exactly equals a train smiles: 0 of 424
      labelled test rows with disagreement_flag=True: 43 of 424
      labelled test rows with n_values>1 (label is a median of several reports): 198 of 424

Byungwoong Yoo (BigWonder)

## Seed-0 reproduction note

Re-running seed 0 (`run_baselines_multiseed.py`) matches the original `BASELINES.csv` within 1e-3 R2 for 112 of 120 rows. The other 8 rows are all `chromophore_optical` solvent models (`RF_Morgan2048+solv256` / `RF_RDKit_desc+solv256`). They differ by at most 0.005 R2 (largest: emi_max_nm scaffold RDKit, 0.557 vs 0.562). Do not claim bit-exact reproduction for those rows.

Cause not confirmed. What was checked: cached and freshly computed features are identical, and the features match the original feature code in `packs/chromophore_optical/build.py`. `n_jobs` has no effect, and the seed-0 result is reproducible locally. These models are very sensitive to tiny target changes. Adding noise below 1e-6 to quantum_yield moved the scaffold Morgan R2 from 0.1216 to 0.1182. The original numbers were trained on unrounded in-memory targets, but the released CSV rounds targets to 6 decimals. The raw source file and the original RDKit/scikit-learn versions are not available, so this could not be tested.
