# Model comparison protocol

Use a common evaluation protocol when comparing player-performance models.

## Fair comparison
- Use identical chronological train, validation, and test boundaries.
- Keep the target definition and prediction horizon fixed.
- Apply the same eligible observations to each model where possible.
- Tune hyperparameters only on training/validation data.
- Preserve a simple baseline so added complexity can be evaluated.

## Metrics
Choose metrics that match the target type. For regression, report MAE and RMSE with the target units. For classification, report per-class precision, recall, F1, and a confusion matrix. Include sample counts and avoid comparing scores computed on different test sets.

## Analysis
Review errors by player, season, role, and match context when sample sizes support it. Report uncertainty or variation across repeated runs where appropriate. Record model version, feature set, random seed, and execution environment.
