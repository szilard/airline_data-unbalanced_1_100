# Experiment 1: in H2O GBM, full data (Y:N 1:100) is slightly worse than Y:N 1:10.
# Does limiting the leaf values (max_abs_leafnode_pred) or requiring larger leaves (min_rows)
# close that gap? The 1:10 runs with the same params are the control: a fix for the imbalance
# should help 1:100 more than 1:10.

import time
from pathlib import Path

import h2o
import pandas as pd
from h2o.estimators import H2OGradientBoostingEstimator
from h2o.tree import H2OTree
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

data_dir = Path(__file__).parent.parent / "data"

cat_cols = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
num_cols = ["DepTime", "Distance"]
target   = "dep_delayed_15min"

train = pd.read_csv(data_dir / "train.csv")
eval_df = pd.read_csv(data_dir / "eval.csv")

h2o.init()
h2o.no_progress()


def to_h2o(df):
    hf = h2o.H2OFrame(df[num_cols + cat_cols + [target]])
    for col in cat_cols + [target]:
        hf[col] = hf[col].asfactor()
    return hf

eval_hf = to_h2o(eval_df)
y_eval = (eval_df[target] == "Y").astype(int).to_numpy()


train_neg = train[train[target] == "N"]
train_pos = train[train[target] == "Y"]

train_sets = {
    "Y:N 1:100": train,
    "Y:N 1:10": pd.concat([train_neg.sample(n=len(train_neg) // 10, random_state=42), train_pos]),
}
train_hfs = {name: to_h2o(df) for name, df in train_sets.items()}

# run name -> (train set, extra GBM params)
runs = {
    "Y:N 1:100": ("Y:N 1:100", {}),
    "Y:N 1:10": ("Y:N 1:10", {}),
    "Y:N 1:100 max_abs_leaf=0.5": ("Y:N 1:100", {"max_abs_leafnode_pred": 0.5}),
    "Y:N 1:10 max_abs_leaf=0.5": ("Y:N 1:10", {"max_abs_leafnode_pred": 0.5}),
    "Y:N 1:100 min_rows=100": ("Y:N 1:100", {"min_rows": 100}),
    "Y:N 1:10 min_rows=100": ("Y:N 1:10", {"min_rows": 100}),
}


def max_abs_leaf(model):
    # leaf values are each tree's contribution to the log-odds (learn_rate already applied)
    vals = []
    for i in range(model.actual_params["ntrees"]):
        tree = H2OTree(model=model, tree_number=i)
        vals += [abs(v) for v, left in zip(tree.predictions, tree.left_children) if left == -1]
    return max(vals)


results = []

for name, (train_set, params) in runs.items():
    df = train_sets[train_set]

    model = H2OGradientBoostingEstimator(
        ntrees=30,
        max_depth=6,
        learn_rate=0.1,
        seed=42,
        **params,
    )

    print(f"\n{name}: {(df[target] == 'N').sum()} N, {(df[target] == 'Y').sum()} Y")

    t0 = time.time()
    model.train(x=num_cols + cat_cols, y=target, training_frame=train_hfs[train_set])
    train_time = time.time() - t0

    y_pred = model.predict(eval_hf)["Y"].as_data_frame()["Y"].to_numpy()
    precision, recall, _ = precision_recall_curve(y_eval, y_pred)

    results.append({
        "run": name,
        "train_time_s": round(train_time, 1),
        "max_leaf": round(max_abs_leaf(model), 3),
        "AUC": round(roc_auc_score(y_eval, y_pred), 4),
        "AP": round(average_precision_score(y_eval, y_pred), 4),
        "prec@rec0.2": round(precision[recall >= 0.2].max(), 4),
    })
    print(results[-1])

print()
print(pd.DataFrame(results).to_string(index=False))

h2o.cluster().shutdown()
