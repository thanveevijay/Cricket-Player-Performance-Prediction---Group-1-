"""Create a chronological proxy split for batsman performance records.

The current project uses matches_played <= 50 for training and later career
records for testing. For a rigorous time-based evaluation, keep all rows from
the same player ordered by match chronology and avoid mixing a player's future
records into training.
"""

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SOURCE_FILE = DATA_DIR / "final_batsman_features.csv"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
REQUIRED_COLUMNS = {"matches_played", "runs_next_match"}


def main() -> None:
    if not SOURCE_FILE.is_file():
        raise FileNotFoundError(f"Dataset not found: {SOURCE_FILE}")

    data = pd.read_csv(SOURCE_FILE)
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    data["matches_played"] = pd.to_numeric(data["matches_played"], errors="coerce")
    data["runs_next_match"] = pd.to_numeric(data["runs_next_match"], errors="coerce")
    data = data.dropna(subset=["matches_played", "runs_next_match"])

    train_df = data.loc[data["matches_played"] <= 50].copy()
    test_df = data.loc[data["matches_played"] > 50].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "The matches_played threshold produced an empty split. "
            "Review the dataset distribution before training."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print(f"Training rows: {len(train_df)} -> {TRAIN_FILE}")
    print(f"Test rows: {len(test_df)} -> {TEST_FILE}")
    print("Split rule: matches_played <= 50 for train; > 50 for test.")


if __name__ == "__main__":
    main()
