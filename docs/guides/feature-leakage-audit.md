# Feature Leakage Audit

For every feature, ask whether it would be known at the exact time the prediction is made.

Check post-match aggregates, target-derived columns, future season statistics, and duplicate match records. Remove or recompute any feature that uses future information.