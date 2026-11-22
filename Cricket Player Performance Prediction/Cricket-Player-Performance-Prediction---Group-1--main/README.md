# Cricket Player Performance Prediction

This project explores batsman performance analysis and next-match run prediction using IPL ball-by-ball data, feature engineering, and regression models.

## Project structure

The repository contains the project under `Cricket Player Performance Prediction/Cricket-Player-Performance-Prediction---Group-1--main/`. Within that project directory, the main workflow includes:

- `milestone 3/scripts/` — baseline, training, feature-analysis, and evaluation scripts.
- `data/` — source and processed datasets.
- `models/` — serialized preprocessing/model artifacts.
- `notebooks/` — exploratory analysis and experiments.
- `.github/workflows/python-checks.yml` — automated Python checks.

Some datasets and generated artifacts may be tracked using Git LFS. Make sure LFS objects are available before running data-dependent scripts.

## Environment setup

Run commands from the project directory:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use the requirements file in the project directory. Some experimental scripts may require additional optional libraries; install those only for the corresponding experiment.

## Running experiments

The milestone 3 scripts are standalone experiments and expect their input files under `data/processed/`. For example, the rolling-average baseline can be run from the project directory with:

```bash
python "milestone 3/scripts/baseline.py"
```

The baseline sorts each batsman's records by date and uses the previous ten matches (with at least three prior observations) to calculate a prediction. It then evaluates the later portion of the data and writes `baseline_predictions.csv` to the processed-data directory.

## Evaluation guidance

- Compare learned models against a simple baseline before claiming an improvement.
- Preserve chronological ordering when evaluating future-match predictions.
- Check player and match identifiers to avoid leakage between training and test sets.
- Fit preprocessing steps on training data only, then apply the fitted transformations to validation/test data.
- Report the dataset version, split strategy, MAE, RMSE, and R² together.
- Treat predictions as exploratory estimates, not guarantees of future player performance.
