"""
EDA + formal statistical tests for the stage-1 preprocessing output
(out/trips_clean_sample.parquet, produced by 01_prep_clean.ipynb). For every
ML-feature step (missing/outliers/skew/encoding/class-imbalance), produces
both a visual diagnostic (saved to out/figs/) and a numeric statistical test,
so claims like "missing not at random" or "log transform reduces skew" are
backed by a p-value/statistic, not just a plot.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pyspark.sql import SparkSession

FIG_DIR = "out/figs"
os.makedirs(FIG_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

spark = (SparkSession.builder.appName("porto-taxi-eda").master("local[*]")
         .config("spark.driver.memory", "4g").getOrCreate())
spark.sparkContext.setLogLevel("WARN")

sdf = spark.read.parquet("out/trips_clean_sample.parquet")
cols = [
    "TRIP_ID", "CALL_TYPE", "ORIGIN_CALL", "ORIGIN_STAND", "DAY_TYPE",
    "distance_km", "distance_km_wins", "distance_km_log", "distance_km_sqrt",
    "duration_sec", "duration_sec_wins", "duration_sec_log", "duration_sec_sqrt",
    "avg_kmh", "avg_kmh_wins",
    "n_points", "TAXI_ID_freq", "ORIGIN_STAND_grouped", "ORIGIN_STAND_freq",
    "ORIGIN_CALL_filled", "class_weight",
]
df = sdf.select(*cols).toPandas()
print("rows:", len(df))


# ===========================================================================
# Step 1 - Missing data: statistically confirm MNAR-on-CALL_TYPE (chi-square)
# ===========================================================================

def missing_chisq(df, target_col, label):
    is_missing = df[target_col].isnull()
    ct = pd.crosstab(df["CALL_TYPE"], is_missing)
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    print(f"\n[{label}] missingness vs CALL_TYPE  chi2={chi2:.1f}  dof={dof}  p={p:.3e}")
    print(ct)
    return ct, chi2, p


ct_call, chi2_call, p_call = missing_chisq(df, "ORIGIN_CALL", "ORIGIN_CALL")
ct_stand, chi2_stand, p_stand = missing_chisq(df, "ORIGIN_STAND", "ORIGIN_STAND")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
(ct_call.div(ct_call.sum(1), axis=0) * 100).plot.bar(stacked=True, ax=ax[0],
    color=["#4daf4a", "#e41a1c"])
ax[0].set_title(f"ORIGIN_CALL missing % by CALL_TYPE\nchi2={chi2_call:.0f}, p={p_call:.2e}")
ax[0].set_ylabel("%"); ax[0].legend(["present", "missing"])
(ct_stand.div(ct_stand.sum(1), axis=0) * 100).plot.bar(stacked=True, ax=ax[1],
    color=["#4daf4a", "#e41a1c"])
ax[1].set_title(f"ORIGIN_STAND missing % by CALL_TYPE\nchi2={chi2_stand:.0f}, p={p_stand:.2e}")
ax[1].set_ylabel("%"); ax[1].legend(["present", "missing"])
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/01_missing_chisq.png", dpi=120); plt.close()


# ===========================================================================
# Step 2 - Outliers: box plots raw vs winsorized + IQR outlier counts
# ===========================================================================

def iqr_outlier_stats(s, label):
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = ((s < lo) | (s > hi)).sum()
    print(f"  {label}: IQR=[{q1:.2f},{q3:.2f}] fence=({lo:.2f},{hi:.2f}) "
          f"outliers={n_out} ({100*n_out/len(s):.2f}%)")
    return n_out


pairs = [("distance_km", "distance_km_wins"), ("duration_sec", "duration_sec_wins"),
         ("avg_kmh", "avg_kmh_wins")]

print("\n[Step 2] IQR-based outlier counts, raw vs winsorized:")
outlier_rows = []
for raw, wins in pairs:
    print(f" {raw}:")
    n_raw = iqr_outlier_stats(df[raw], "raw")
    n_wins = iqr_outlier_stats(df[wins], "winsorized")
    outlier_rows.append({"variable": raw, "outliers_raw": n_raw, "outliers_winsorized": n_wins})

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for i, (raw, wins) in enumerate(pairs):
    sns.boxplot(y=df[raw], ax=axes[0, i], color="#e41a1c")
    axes[0, i].set_title(f"{raw} (raw)")
    sns.boxplot(y=df[wins], ax=axes[1, i], color="#4daf4a")
    axes[1, i].set_title(f"{wins} (winsorized p95)")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/02_boxplots_winsorize.png", dpi=120); plt.close()


# ===========================================================================
# Step 3 - Skew transforms: histograms + skew/kurtosis + normality tests
# ===========================================================================

variants = {
    "distance_km": ["distance_km", "distance_km_wins", "distance_km_log", "distance_km_sqrt"],
    "duration_sec": ["duration_sec", "duration_sec_wins", "duration_sec_log", "duration_sec_sqrt"],
}

stat_rows = []
for base, variant_cols in variants.items():
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    for ax, c in zip(axes, variant_cols):
        s = df[c].dropna()
        sns.histplot(s, kde=True, ax=ax, color="#377eb8")
        ax.set_title(c)
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/03_hist_{base}.png", dpi=120); plt.close()

    print(f"\n[Step 3] {base}: skewness / kurtosis / normality tests per variant")
    for c in variant_cols:
        s = df[c].dropna()
        skew, kurt = stats.skew(s), stats.kurtosis(s)
        skew_z, skew_p = stats.skewtest(s)
        k2, normal_p = stats.normaltest(s)              # D'Agostino-Pearson, any n
        shapiro_sample = s.sample(min(5000, len(s)), random_state=42)
        shapiro_w, shapiro_p = stats.shapiro(shapiro_sample)  # Shapiro valid for n<=5000
        print(f"  {c:<22} skew={skew:7.3f} kurt={kurt:7.3f}  "
              f"skewtest p={skew_p:.2e}  normaltest p={normal_p:.2e}  shapiro(n=5000) p={shapiro_p:.2e}")
        stat_rows.append({"variable": c, "skew": round(skew, 3), "kurtosis": round(kurt, 3),
                           "skewtest_p": skew_p, "normaltest_p": normal_p, "shapiro_p": shapiro_p})


# ===========================================================================
# Step 4 - Encoding diagnostics
# ===========================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

df["CALL_TYPE"].value_counts().plot.bar(ax=axes[0], color="#984ea3")
axes[0].set_title("CALL_TYPE counts"); axes[0].set_ylabel("trips")

top_stands = df["ORIGIN_STAND_grouped"].value_counts().head(15)
top_stands.plot.bar(ax=axes[1], color="#ff7f00")
axes[1].set_title("Top 15 ORIGIN_STAND_grouped\n(incl. Other_Stand)")
axes[1].tick_params(axis="x", rotation=75)

sns.histplot(df["TAXI_ID_freq"], bins=40, ax=axes[2], color="#a65628")
axes[2].set_yscale("log")
axes[2].set_title("TAXI_ID_freq distribution (log y)")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/04_encoding.png", dpi=120); plt.close()

# Pareto check: what % of trips do the top 20% of taxis account for?
freq_per_taxi = df["TAXI_ID_freq"].drop_duplicates().sort_values(ascending=False)
n_taxis = len(freq_per_taxi)
top20_n = max(1, int(0.2 * n_taxis))
pareto_share = freq_per_taxi.head(top20_n).sum() / freq_per_taxi.sum()
print(f"\n[Step 4] top 20% of taxis ({top20_n}/{n_taxis}) account for "
      f"{100*pareto_share:.1f}% of trips (Pareto check)")


# ===========================================================================
# Step 5 - DAY_TYPE imbalance
# ===========================================================================

fig, ax = plt.subplots(figsize=(5, 4))
df["DAY_TYPE"].value_counts().plot.bar(ax=ax, color="#e41a1c")
ax.set_title("DAY_TYPE counts (class imbalance)")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/05_day_type_imbalance.png", dpi=120); plt.close()
print("\n[Step 5] DAY_TYPE counts:\n", df["DAY_TYPE"].value_counts())
print("class_weight per DAY_TYPE:\n", df.groupby("DAY_TYPE")["class_weight"].first())


# ===========================================================================
# Bonus - correlation heatmap among numeric trip features
# ===========================================================================

num_cols = ["distance_km", "duration_sec", "avg_kmh", "n_points"]
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation - trip features")
plt.tight_layout(); plt.savefig(f"{FIG_DIR}/06_correlation.png", dpi=120); plt.close()


# ===========================================================================
# Summary tables
# ===========================================================================

outlier_df = pd.DataFrame(outlier_rows)
stat_df = pd.DataFrame(stat_rows)
print("\n=== Outlier summary ===\n", outlier_df.to_string(index=False))
print("\n=== Skew / normality summary ===\n", stat_df.to_string(index=False))

outlier_df.to_csv(f"{FIG_DIR}/../outlier_summary.csv", index=False)
stat_df.to_csv(f"{FIG_DIR}/../skew_normality_summary.csv", index=False)

print("\nfigures saved under:", FIG_DIR)
spark.stop()
