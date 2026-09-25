# Airline delay data: imbalanced 1:100 sample

An imbalanced binary classification dataset built from 2005 US flight data: predict `dep_delayed_15min`
(departure delayed 15 min or more), with Y:N ≈ 1:100.

## Source

`data/2005.csv`: the 2005 file of the US airline on-time data (ASA Data Expo 2009), 7,140,596 flights, 29 columns.
It is not in git (670 MB); put it in `data/` before running `prepare_data.py`.

## Preparation

`prepare_data.py` does the following:

1. Label: `dep_delayed_15min` is `Y` if `DepDelay` ≥ 15, else `N`.
2. Keep 8 features:
   - `Month`, `DayofMonth`, `DayOfWeek`: strings with a `c-` prefix (e.g. `c-9`), so they load as categorical
   - `DepTime`: departure time as hhmm (e.g. `1328`)
   - `UniqueCarrier`, `Origin`, `Dest`: carrier and airport codes
   - `Distance`: miles
3. Drop rows with missing values: 133,730 flights, all of them cancelled (no `DepTime` or `DepDelay`).
   That leaves 7,006,866 flights: 5,727,462 N and 1,279,404 Y (18.3% delayed, Y:N ≈ 1:4.5).
4. Sample 5,000,000 N and 50,000 Y (seed 123). Y is downsampled to create the 1:100 imbalance.
5. Split each class 3:1:1 into train / eval / holdout, then shuffle each file.

Run with `.venv/bin/python prepare_data.py`. It prints the class counts and writes:

| file | rows | N | Y |
|---|---|---|---|
| `data/train.csv` | 3,030,000 | 3,000,000 | 30,000 |
| `data/eval.csv` | 1,010,000 | 1,000,000 | 10,000 |
| `data/holdout.csv` | 1,010,000 | 1,000,000 | 10,000 |

Notes:

- `DepTime` is the actual departure time; the scheduled time (`CRSDepTime`) is not kept. Delayed flights have later `DepTime` by construction.
- The split is random within 2005, not by date, so train, eval and holdout cover the same days.

## Experiments

[`GBMs-unbalanced_data/`](GBMs-unbalanced_data/README.md): XGBoost and H2O GBM on this data, and whether GBMs still struggle with the 1:100 imbalance.
