# Cricket Feature Engineering Notes

## Candidate feature groups
- Recent form: rolling summaries of a player's recent match performance.
- Career context: historical averages and participation counts available before the match.
- Match context: innings number, venue, opposition, and match format when available.
- Opportunity: batting position, overs bowled, or other role-related context when known before prediction.

## Temporal correctness
Every feature must be computable using information available at prediction time. Sort records chronologically before calculating rolling statistics and shift historical aggregates so the current match result is never included.

## Missing values
- Distinguish a genuine zero from an unavailable statistic.
- Consider a missingness indicator when absence itself carries information.
- Fit imputers on training data only.
- Apply the same fitted preprocessing to validation and test sets.

## Validation
Use a chronological split where the prediction task is future-match forecasting. Compare against a simple baseline before evaluating more complex feature sets.
