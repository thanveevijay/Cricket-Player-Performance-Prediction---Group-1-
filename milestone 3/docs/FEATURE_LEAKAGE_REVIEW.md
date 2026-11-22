# Feature leakage review guide

Review every engineered feature before model training.

## Questions to ask
- Was this value available at the exact time the prediction would be made?
- Does it contain information from the target match or a future match?
- Was it aggregated across the full dataset before splitting?
- Does it encode the target directly or indirectly?
- Does it include post-match statistics when the intended use is pre-match prediction?

## Rolling features
Sort by player and match date, compute statistics from prior observations only, and shift the result so the current match is excluded. Define a minimum history requirement and document how players with limited history are handled.

## Validation
- Inspect feature-generation code for unshifted rolling windows.
- Compare feature timestamps with prediction timestamps.
- Recompute a sample feature manually for a few players.
- Keep preprocessing fit operations inside the training pipeline.
- Record the feature availability assumptions in the experiment report.

Remove or redesign features that cannot be justified as available at prediction time.
