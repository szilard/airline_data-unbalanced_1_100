from pathlib import Path

import polars as pl

data_dir = Path(__file__).parent / "data"

keep_cols = ["Month", "DayofMonth", "DayOfWeek", "DepTime", "UniqueCarrier",
             "Origin", "Dest", "Distance", "dep_delayed_15min"]

df = pl.read_csv(data_dir / "2005.csv", null_values="NA")

df = df.with_columns(
    pl.when(pl.col("DepDelay").cast(pl.Int32, strict=False) >= 15).then(pl.lit("Y")).otherwise(pl.lit("N"))
      .alias("dep_delayed_15min"),
    *[("c-" + pl.col(col).cast(pl.Utf8)).alias(col)
      for col in ["Month", "DayofMonth", "DayOfWeek"]],
)

df = df.select(keep_cols).drop_nulls()

print(df.shape)
print(df.head())
print(df["dep_delayed_15min"].value_counts().sort("dep_delayed_15min"))


def split_3_1_1(d):
    n_train = d.height * 3 // 5
    n_eval = d.height // 5
    return d.slice(0, n_train), d.slice(n_train, n_eval), d.slice(n_train + n_eval)


df_neg = df.filter(pl.col("dep_delayed_15min") == "N").sample(n=5_000_000, shuffle=True, seed=123)
df_pos = df.filter(pl.col("dep_delayed_15min") == "Y").sample(n=50_000, shuffle=True, seed=123)

df_train, df_eval, df_holdout = [
    pl.concat([neg, pos]).sample(fraction=1.0, shuffle=True, seed=123)
    for neg, pos in zip(split_3_1_1(df_neg), split_3_1_1(df_pos))
]

for name, d in [("train", df_train), ("eval", df_eval), ("holdout", df_holdout)]:
    print(f"\n{name}: {d.shape}")
    print(d["dep_delayed_15min"].value_counts().sort("dep_delayed_15min"))
    d.write_csv(data_dir / f"{name}.csv")
