# Cricket Player Performance Prediction

This project explores batsman performance analysis and next-match run prediction using IPL ball-by-ball data, feature engineering, a Random Forest regressor, and a Streamlit interface.

## Project files

- `src/data_cleaning.py` — cleans delivery-level data and creates derived fields.
- `split_data.py` — creates train and test CSV files from the engineered batsman dataset.
- `train_data.py` — fits the preprocessing pipeline and Random Forest model, then evaluates on the held-out test split.
- `streamlit_app.py` — interactive next-match run prediction interface.
- `models/featurepipeline.pkl` — preprocessing pipeline used by training and inference.
- `data/` — source and derived datasets (large files may be stored using Git LFS).

## Environment setup

Run commands from this project directory:

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
```

## Rebuild the model

Ensure the Git LFS data files are available locally, then run:

```bash
python split_data.py
python train_data.py
```

The split script currently uses `matches_played <= 50` for training and `matches_played > 50` for testing. This is a simple career-stage split; it is not a substitute for a carefully designed player- and time-aware validation strategy. Review the player and match identifiers before using the results in a report.

Training fits the saved preprocessing pipeline on the training rows only, transforms the test rows with that fitted pipeline, and writes `models/batsman_runs_model.pkl`.

## Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

The app expects both the preprocessing pipeline and trained model in `models/`. Its displayed regression metrics are computed from `data/test.csv`.

## Important evaluation notes

- Do not train and evaluate on the same rows.
- Keep the preprocessing fit restricted to training data to avoid leakage.
- Report MAE, RMSE, MSE, and R² alongside the exact split strategy and dataset version.
- Predictions are exploratory estimates, not guarantees of an individual player's future performance.
