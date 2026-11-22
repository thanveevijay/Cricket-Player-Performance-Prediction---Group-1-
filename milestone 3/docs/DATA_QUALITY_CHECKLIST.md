# Cricket dataset quality checklist

Run these checks before training or evaluating a player-performance model.

## Schema and validity
- Confirm each row has a documented match identifier, player identifier, date, and target definition.
- Parse dates consistently and sort observations chronologically.
- Check numeric columns for impossible values, unexpected units, and sentinel values.
- Inspect missingness by column, player, season, and match.
- Identify duplicate player-match records and resolve them using a documented rule.

## Leakage prevention
- Build time-aware train, validation, and test partitions.
- Compute rolling statistics using past matches only; shift features before calculating windows.
- Fit imputers, scalers, encoders, and feature selectors on training data only.
- Keep all rows from a match on the same side of a split when match-level information could leak.

## Reporting
Record dataset version, row counts after cleaning, split date boundaries, feature list, target definition, and exclusions. Report baseline performance alongside the proposed model and include error analysis across players and match contexts.
