"""Evaluate a leakage-aware 10-match rolling-average run baseline."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
INPUT_FILE = PROC / "batsman_match_model_ready.csv"
OUTPUT_FILE = PROC / "baseline_predictions.csv"
REQUIRED_COLUMNS = {"match_id", "batsman", "date_parsed", "runs"}


def build_rolling_baseline(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a prediction based only on each batsman's previous ten matches."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")

    result = frame.copy()
    result["date_parsed"] = pd.to_datetime(result["date_parsed"], errors="coerce")
    result["runs"] = pd.to_numeric(result["runs"], errors="coerce")
    result = result.dropna(subset=["batsman", "date_parsed", "runs"])
    result = result.sort_values(["batsman", "date_parsed", "match_id"]).reset_index(drop=True)

    result["baseline_rolling_10"] = (
        result.groupby("batsman")["runs"]
        .transform(lambda values: values.shift(1).rolling(10, min_periods=3).mean())
    )
    return result


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Expected processed input at {INPUT_FILE}. "
            "Run the preprocessing workflow before evaluating the baseline."
        )

    df = build_rolling_baseline(pd.read_csv(INPUT_FILE))
    baseline_df = df.dropna(subset=["baseline_rolling_10"])

    if baseline_df.empty:
        raise ValueError("Not enough historical matches to calculate the rolling baseline.")

    split_date = baseline_df["date_parsed"].quantile(0.80)
    test = baseline_df.loc[baseline_df["date_parsed"] > split_date].copy()
    if test.empty:
        raise ValueError("The chronological test split is empty; check the input dates.")

    y_true = test["runs"]
    y_pred = test["baseline_rolling_10"]

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(test) > 1 else float("nan")

    print("BASELINE (10-match rolling average)")
    print(f"Test rows: {len(test)}")
    print(f"Split date: {split_date.date()}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")
    print(f"R²  : {r2:.3f}")

    PROC.mkdir(parents=True, exist_ok=True)
    columns = ["match_id", "batsman", "date_parsed", "runs", "baseline_rolling_10"]
    test[columns].to_csv(OUTPUT_FILE, index=False)
    print(f"Saved predictions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
