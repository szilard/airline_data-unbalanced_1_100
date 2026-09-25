# Airline delay: GBMs on imbalanced data

Predict `dep_delayed_15min` (departure delayed 15 min or more) from 2005 US flight data,
and test whether GBMs still struggle with class imbalance.

## Data

- Source: `data/2005.csv`. Features: `Month`, `DayofMonth`, `DayOfWeek`, `DepTime`, `UniqueCarrier`, `Origin`, `Dest`, `Distance`.
- Sample: 5,000,000 N + 50,000 Y (Y:N ≈ 1:100), split 3:1:1 per class into train / eval / holdout.
- Holdout is not used yet. All metrics below are on eval (10,000 Y).

## Scripts

| script | what it does |
|---|---|
| `prepare_data.py` | sample and split, writes `data/{train,eval,holdout}.csv` |
| `train_xgb.py` | XGBoost on train at Y:N 1:100 / 1:10 / 1:1 (N downsampled), ROC and PR plots |
| `train_h2o.py` | same with H2O GBM |
| `exp1_h2o_leaf_limits.py` | experiment 1 (below) |
| `exp2_xgb_old_gbm.py` | experiment 2 and its ablation (below) |

All models: 30 trees, depth 6, learning rate 0.1. Run with `.venv/bin/python <script>` (4-core box).

## Baseline results

| run | XGB time | XGB AUC | XGB prec@rec0.2 | H2O time | H2O AUC | H2O prec@rec0.2 |
|---|---|---|---|---|---|---|
| Y:N 1:100 (full) | 28s | **0.7490** | **0.0642** | 86s | 0.7474 | 0.0659 |
| Y:N 1:10 | 2s | 0.7486 | 0.0623 | 12s | **0.7496** | **0.0674** |
| Y:N 1:1 | 0.3s | 0.7384 | 0.0495 | 6s | 0.7388 | 0.0482 |

- **At 30 trees, downsampling N to Y:N 1:10 is nearly free** (at most 0.0004 AUC lost) and trains 7–14× faster. Going to 1:1 costs about 0.01 AUC.
- **But 30 trees is undertrained.** At 100 trees XGBoost gains 0.0125 AUC, and full data beats 1:10 on every metric (AUC 0.7615 vs 0.7597, AP 0.095 vs 0.080, prec@rec0.2 0.081 vs 0.075). Once the model has capacity, the extra N rows pay off.
- **XGBoost handles 1:100 fine.** H2O does slightly better at 1:10 on AUC and prec@rec0.2 (+0.002), but full data wins on AP (0.084 vs 0.077). The gap is at noise level.

## Why did old GBMs struggle with imbalance?

Candidate causes: (1) the Newton leaf step Σ(y−p) / Σp(1−p) grows like 1/p, so small leaves full of positives get huge values; (2) XGBoost started every row at p = 0.5 (`base_score`) until version 2.0; (3) people judged models on accuracy at a 0.5 threshold.

**Experiment 1: H2O, limit leaf values or require larger leaves** (AUC, full vs 1:10)

| params | max leaf (full / 1:10) | full | 1:10 |
|---|---|---|---|
| default | 10.1 / 1.1 | 0.7474 | 0.7496 |
| `max_abs_leafnode_pred=0.5` | 0.5 / 0.5 | 0.7455 | 0.7494 |
| `min_rows=100` | 10.1 / 1.1 | 0.7484 | 0.7490 |

Leaf values do hit the theoretical limit of learning rate ÷ p (0.1/0.0099 = 10.1), but capping them *hurts*. Larger leaves narrow the gap to 0.0006. **The large leaf steps are not what costs H2O here.**

**Experiment 2: XGBoost with the guards removed** (`reg_lambda=0`, `min_child_weight=0`, `base_score=0.5`)

An ablation changes one setting at a time to find which one causes the effect.

| params | trees | max leaf (full) | full AUC | 1:10 AUC | full − 1:10 |
|---|---|---|---|---|---|
| default | 30 | 5.0 | 0.7490 | 0.7486 | +0.0004 |
| old GBM (all three) | 30 | 3.0 | 0.7306 | 0.7443 | −0.0137 |
| `base_score=0.5` only | 30 | 0.5 | 0.7299 | 0.7439 | −0.0140 |
| `reg_lambda=0` + `min_child_weight=0` only | 30 | 119,837 | 0.7488 | 0.7492 | −0.0004 |
| default | 100 | 5.0 | 0.7615 | 0.7597 | +0.0018 |
| old GBM (all three) | 100 | 3,745 | 0.7600 | 0.7599 | +0.0001 |

- **Removing the guards reproduces the old issue:** full data trails 1:10 by 0.014 AUC.
- **`base_score=0.5` alone causes the whole drop.** Starting at p = 0.5, 30 trees at learning rate 0.1 barely reach the 1% base rate (log-odds −4.6). 1:10 only needs −2.3, so it is hurt about 4× less.
- **Without the leaf guards, leaf values explode (up to about 120,000 log-odds) but AUC is unchanged.** Such leaves push their few rows to p = 0 or 1. That would matter for log-loss and calibrated probabilities (not measured here), not for ranking.
- **With 100 trees the gap closes**, and old GBM ends up within 0.0015 AUC of the modern default.

**Bottom line:** in XGBoost, the old imbalance problem came from starting every row at p = 0.5, combined with too few trees to recover. The large Newton leaf steps are real but did not hurt AUC in either library.

## Caveats and next steps

- Single seed and a single downsample per run, so differences below about 0.002 AUC are likely noise.
- Next: train to convergence (several hundred trees, early stopping on eval) and recheck full vs 1:10; measure log-loss and calibration; repeat over several seeds; score the final models on holdout.
