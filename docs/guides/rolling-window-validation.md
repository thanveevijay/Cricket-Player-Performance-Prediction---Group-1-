# Rolling Window Validation

Evaluate the model across multiple chronological cutoffs. For each cutoff, train only on matches available before that point and test on the subsequent period.

Report metrics for each window and summarize variation over time. This reveals whether performance depends heavily on one season or competition period.