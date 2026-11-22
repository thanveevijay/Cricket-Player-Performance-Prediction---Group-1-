# Leakage Review

Check that target-match runs, wickets, overs, and post-match aggregates are not included in pre-match features. Fit encoders and scalers on training data only. Verify rolling averages are shifted so they use only completed prior matches.