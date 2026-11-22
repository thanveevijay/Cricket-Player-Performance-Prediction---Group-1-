# Player History Features

Build historical aggregates using only matches completed before the prediction timestamp.

Use a minimum-history rule for new players and document fallback behavior. Verify that rolling windows are shifted so the current match cannot contribute to its own prediction.