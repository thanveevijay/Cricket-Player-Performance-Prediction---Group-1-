"""Train and evaluate the next-match batsman runs regressor.

Run from this file's directory or any working directory:
    python train_data.py

The script consumes train.csv and test.csv produced by split_data.py.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
TARGET_COLUMN = "runs_next_match"


def load_split(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} was not found. Run split_data.py first to create the train/test files."
        )
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path.name} is empty; check the data split.")
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"{TARGET_COLUMN!r} is missing from {path.name}.")
    return frame


def main() -> None:
    train_df = load_split(DATA_DIR / "train.csv")
    test_df = load_split(DATA_DIR / "test.csv")

    feature_columns = [column for column in train_df.columns if column != TARGET_COLUMN]
    if not feature_columns:
        raise ValueError("The dataset contains no feature columns.")
    missing_features = sorted(set(feature_columns) - set(test_df.columns))
    if missing_features:
        raise ValueError(f"Test data is missing training features: {missing_features}")

    X_train = train_df[feature_columns]
    y_train = pd.to_numeric(train_df[TARGET_COLUMN], errors="raise")
    X_test = test_df[feature_columns]
    y_test = pd.to_numeric(test_df[TARGET_COLUMN], errors="raise")

    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Feature data contains missing values; clean or impute them before training.")

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions) if len(y_test) > 1 else float("nan")

    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "batsman_runs_model.pkl"
    joblib.dump(model, model_path)
    print(f"Trained model saved to: {model_path}")


if __name__ == "__main__":
    main()
