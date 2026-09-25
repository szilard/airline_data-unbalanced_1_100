import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

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
    "Y:N 1:100": train,  # full train, slow on this box
    "Y:N 1:10": pd.concat([train_neg.sample(n=len(train_neg) // 10, random_state=42), train_pos]),
    "Y:N 1:1": pd.concat([train_neg.sample(n=len(train_neg) // 100, random_state=42), train_pos]),
}

preds = {}
results = []

for name, df in train_sets.items():
    X_train, y_train = prepare(df)

    model = xgb.XGBClassifier(
        n_estimators=30,
        max_depth=6,
        learning_rate=0.1,
        enable_categorical=True,
        random_state=42,
        n_jobs=-1,
    )

    print(f"\n{name}: {(y_train == 0).sum()} N, {(y_train == 1).sum()} Y")

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f}s")

    y_pred = model.predict_proba(X_eval)[:, 1]
    preds[name] = y_pred
    auc = roc_auc_score(y_eval, y_pred)
    print(f"Eval AUC: {auc:.4f}")
    print(f"Eval AP:  {average_precision_score(y_eval, y_pred):.4f}")

    precision, recall, _ = precision_recall_curve(y_eval, y_pred)
    prec_at_rec02 = precision[recall >= 0.2].max()
    print(f"Eval precision @ recall 0.2: {prec_at_rec02:.4f}")

    results.append({"run": name, "train_time_s": round(train_time, 1), "AUC": round(auc, 4),
                    "prec@rec0.2": round(prec_at_rec02, 4)})

print()
print(pd.DataFrame(results).to_string(index=False))


colors = {"Y:N 1:100": "#2a78d6", "Y:N 1:10": "#eb6834", "Y:N 1:1": "#1baf7a"}
ink, muted, grid = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": ink, "axes.titlecolor": ink,
    "xtick.color": muted, "ytick.color": muted, "text.color": ink,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.8,
})

fig_roc, ax_roc = plt.subplots(figsize=(6.5, 5.5))
fig_prc, ax_prc = plt.subplots(figsize=(6.5, 5.5))

for name, y_pred in preds.items():
    color = colors[name]
    fpr, tpr, _ = roc_curve(y_eval, y_pred)
    ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{name}  (AUC {roc_auc_score(y_eval, y_pred):.4f})")

    precision, recall, _ = precision_recall_curve(y_eval, y_pred)
    ax_prc.plot(recall, precision, color=color, lw=2, label=f"{name}  (AP {average_precision_score(y_eval, y_pred):.4f})")

ax_roc.plot([0, 1], [0, 1], color=muted, lw=1, label="random")
ax_roc.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (eval)")
ax_roc.legend(frameon=False, loc="lower right")

ax_prc.axhline(y_eval.mean(), color=muted, lw=1, label=f"random  (AP {y_eval.mean():.4f})")
ax_prc.set(xlim=(0, 1), ylim=(0, None), xlabel="Recall", ylabel="Precision", title="Precision-recall curve (eval)")
ax_prc.legend(frameon=False, loc="upper right")

for fig, ax, filename in [(fig_roc, ax_roc, "roc_xgb.png"), (fig_prc, ax_prc, "prc_xgb.png")]:
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / filename, dpi=150)
    print(f"Saved {filename}")
