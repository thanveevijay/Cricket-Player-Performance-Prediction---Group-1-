"""Unit tests for the rolling-average baseline.

Run from this directory with: python -m pytest -q
"""

import pandas as pd
import pytest

from baseline import build_rolling_baseline


def sample_matches():
    return pd.DataFrame(
        {
            "match_id": [1, 2, 3, 4, 5],
            "batsman": ["A", "A", "A", "A", "A"],
            "date_parsed": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ],
            "runs": [10, 20, 30, 40, 50],
        }
    )


def test_rolling_prediction_uses_only_previous_matches():
    result = build_rolling_baseline(sample_matches())

    # The first three rows do not have the required three prior observations.
    assert result["baseline_rolling_10"].iloc[:3].isna().all()
    # For match four, the prediction is the mean of matches one through three.
    assert result["baseline_rolling_10"].iloc[3] == pytest.approx(20.0)
    # The current match's 40 runs must not leak into its own prediction.
    assert result["baseline_rolling_10"].iloc[3] != 25.0


def test_rolling_prediction_is_grouped_by_batsman():
    frame = sample_matches()
    second_player = frame.iloc[:3].copy()
    second_player["batsman"] = "B"
    second_player["runs"] = [3, 6, 9]

    result = build_rolling_baseline(pd.concat([frame, second_player], ignore_index=True))
    player_b = result[result["batsman"] == "B"].reset_index(drop=True)

    assert player_b["baseline_rolling_10"].isna().all()


def test_missing_required_columns_raise_clear_error():
    with pytest.raises(ValueError, match="missing required columns"):
        build_rolling_baseline(pd.DataFrame({"runs": [1, 2, 3]}))
