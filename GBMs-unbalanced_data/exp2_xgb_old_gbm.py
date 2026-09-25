# Experiment 2: recreate "old GBM" behavior in XGBoost by removing the modern guards:
# no L2 on leaf values, no hessian-based min leaf size, and starting at p = 0.5 instead of
# the base rate. With them removed, does full data (Y:N 1:100) start losing to Y:N 1:10?
#
# Ablation: which of the removed guards causes it?
#   A: base_score=0.5 only
#   B: reg_lambda=0 + min_child_weight=0 only
#   C: all three with 100 trees instead of 30 (plus a default 100-tree control)

import time
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

data_dir = Path(__file__).parent.parent / "data"

cat_cols = ["Month", "DayofMonth", "DayOfWeek", "UniqueCarrier", "Origin", "Dest"]
num_cols = ["DepTime", "Distance"]
target   = "dep_delayed_15min"

train = pd.read_csv(data_dir / "train.csv")
eval_df = pd.read_csv(data_dir / "eval.csv")


cat_levels = {col: sorted(train[col].unique()) for col in cat_cols}

def prepare(df):
    X = df[num_cols + cat_cols].copy()
    for col in cat_cols:
        X[col] = pd.Categorical(
            X[col].where(X[col].isin(cat_levels[col])),
            categories=cat_levels[col],
        )
    y = (df[target] == "Y").astype(int).to_numpy()
    return X, y

X_eval, y_eval = prepare(eval_df)


train_neg = train[train[target] == "N"]
train_pos = train[train[target] == "Y"]

train_sets = {
    "Y:N 1:100": train,
    "Y:N 1:10": pd.concat([train_neg.sample(n=len(train_neg) // 10, random_state=42), train_pos]),
}

model_params = dict(
    n_estimators=30,
    max_depth=6,
    learning_rate=0.1,
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
)

no_leaf_guards = {"reg_lambda": 0, "min_child_weight": 0}
start_at_half = {"base_score": 0.5}
old_gbm = no_leaf_guards | start_at_half

# run name -> (train set, XGBoost params overriding model_params)
runs = {
    "Y:N 1:100": ("Y:N 1:100", {}),
    "Y:N 1:10": ("Y:N 1:10", {}),
    "Y:N 1:100 old GBM": ("Y:N 1:100", old_gbm),
    "Y:N 1:10 old GBM": ("Y:N 1:10", old_gbm),
    # A
    "Y:N 1:100 base_score=0.5": ("Y:N 1:100", start_at_half),
    "Y:N 1:10 base_score=0.5": ("Y:N 1:10", start_at_half),
    # B
    "Y:N 1:100 no leaf guards": ("Y:N 1:100", no_leaf_guards),
    "Y:N 1:10 no leaf guards": ("Y:N 1:10", no_leaf_guards),
    # C
    "Y:N 1:100 100 trees": ("Y:N 1:100", {"n_estimators": 100}),
    "Y:N 1:10 100 trees": ("Y:N 1:10", {"n_estimators": 100}),
    "Y:N 1:100 old GBM 100 trees": ("Y:N 1:100", old_gbm | {"n_estimators": 100}),
    "Y:N 1:10 old GBM 100 trees": ("Y:N 1:10", old_gbm | {"n_estimators": 100}),
}


def max_abs_leaf(model):
    # leaf values are each tree's contribution to the log-odds (learning_rate already applied)
    trees = model.get_booster().trees_to_dataframe()
    return trees.loc[trees["Feature"] == "Leaf", "Gain"].abs().max()


results = []

for name, (train_set, params) in runs.items():
    X_train, y_train = prepare(train_sets[train_set])

    model = xgb.XGBClassifier(**(model_params | params))

    print(f"\n{name}: {(y_train == 0).sum()} N, {(y_train == 1).sum()} Y")

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict_proba(X_eval)[:, 1]
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
