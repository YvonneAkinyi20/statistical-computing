"""
Topic 15: Data transformation and normalization
Libraries: pandas, numpy, sklearn.preprocessing (scipy, matplotlib for support)

Outputs: original versus transformed data, standardized values, normalized
values and comparison plots, saved to the output folder.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, RobustScaler, StandardScaler

# ---------- Load data ----------
HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE.parent / "data" / "Cleaned_Students_Performance.csv")
df.columns = [c.strip().lower().replace(" ", "_").replace("/", "_") for c in df.columns]

OUT = HERE / "output"
OUT.mkdir(exist_ok=True)


def find_col(keyword):
    for c in df.columns:
        if keyword in c:
            return c
    raise KeyError(f"No column containing '{keyword}'. Columns: {list(df.columns)}")


COLS = {"Math": find_col("math"), "Reading": find_col("reading"), "Writing": find_col("writing")}
GENDER = find_col("gender")
orig = df[list(COLS.values())].dropna().copy()
orig.columns = list(COLS)
names = list(COLS)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", None)

print("\nORIGINAL DATA (first 5 rows)\n")
print(orig.head())

# =====================================================================
# 1. Standardization (z-scores): z = (x - mean) / sd
# =====================================================================
z_scaler = StandardScaler()
z = pd.DataFrame(z_scaler.fit_transform(orig), columns=[f"{n}_z" for n in names])
z_manual = (orig - orig.mean()) / orig.std(ddof=0)
print("\n1. STANDARDIZATION (z-score)")
print("Formula: z = (x - mean) / sd")
print("Manual calculation matches StandardScaler:",
      bool(np.allclose(z.to_numpy(), z_manual.to_numpy())))
print("Mean of z-scores:", z.mean().round(6).to_dict())
print("SD of z-scores:  ", z.std(ddof=0).round(6).to_dict())

# =====================================================================
# 2. Normalization (min-max): x' = (x - min) / (max - min)
# =====================================================================
mm_scaler = MinMaxScaler()
mm = pd.DataFrame(mm_scaler.fit_transform(orig), columns=[f"{n}_minmax" for n in names])
mm_manual = (orig - orig.min()) / (orig.max() - orig.min())
print("\n2. NORMALIZATION (min-max to the range 0 to 1)")
print("Formula: x' = (x - min) / (max - min)")
print("Manual calculation matches MinMaxScaler:",
      bool(np.allclose(mm.to_numpy(), mm_manual.to_numpy())))
print("Minimum:", mm.min().round(4).to_dict(), "| Maximum:", mm.max().round(4).to_dict())

# =====================================================================
# 3. Robust scaling: (x - median) / IQR  (less affected by outliers)
# =====================================================================
rb = pd.DataFrame(RobustScaler().fit_transform(orig), columns=[f"{n}_robust" for n in names])
print("\n3. ROBUST SCALING: (x - median) / IQR")
print("Median of robust-scaled values:", rb.median().round(6).to_dict())

# =====================================================================
# 4. Transformations to reduce skewness
# =====================================================================
reflected_log = np.log1p(orig.max() - orig)                # for left-skewed data: log(max + 1 - x)
sqrt_t = np.sqrt(orig.max() - orig)                        # reflected square root
log_t = np.log1p(orig)                                     # log(1 + x), safe when scores can be 0
yj = pd.DataFrame(PowerTransformer(method="yeo-johnson", standardize=False).fit_transform(orig),
                  columns=names)
print("\n4. TRANSFORMATIONS FOR SKEWNESS")
print("Scores are bounded between 0 and 100 and are already close to symmetric, so log or "
      "square-root transforms may not help and can even add skewness. Compare the "
      "table below to see which transformation works best for each score.")

skew = pd.DataFrame({
    "Original": orig.apply(lambda s: stats.skew(s)),
    "log(1 + x)": log_t.apply(lambda s: stats.skew(s)),
    "Reflected sqrt": sqrt_t.apply(lambda s: -stats.skew(s)),     # sign restored to compare direction
    "Reflected log": reflected_log.apply(lambda s: -stats.skew(s)),
    "Yeo-Johnson": yj.apply(lambda s: stats.skew(s)),
    "Standardized (z)": z.set_axis(names, axis=1).apply(lambda s: stats.skew(s)),
}).round(3)
print("\nSkewness before and after (closer to 0 is more symmetric)\n")
print(skew)
skew.to_csv(OUT / "skewness_comparison.csv")

# Normality check (Shapiro-Wilk) on Math, original vs Yeo-Johnson
for label, series in [("Original Math", orig["Math"]), ("Yeo-Johnson Math", yj["Math"])]:
    w, p = stats.shapiro(series)
    print(f"Shapiro-Wilk, {label}: W = {w:.4f}, p = {p:.4g}")

# =====================================================================
# 5. Encoding a categorical variable
# =====================================================================
encoded = pd.get_dummies(df[GENDER], prefix=GENDER, dtype=int)
labels = df[GENDER].astype("category")
print("\n5. ENCODING A CATEGORICAL VARIABLE (gender)")
print("Label encoding:", dict(enumerate(labels.cat.categories)))
print("One-hot encoding (first 5 rows)\n")
print(encoded.head())

# =====================================================================
# Original versus transformed table and summary statistics
# =====================================================================
comparison = pd.concat([orig.reset_index(drop=True), z, mm, rb], axis=1)
comparison.round(4).to_csv(OUT / "original_vs_transformed.csv", index=False)
print("\nORIGINAL VERSUS TRANSFORMED (first 8 rows, Math only)\n")
print(comparison[["Math", "Math_z", "Math_minmax", "Math_robust"]].head(8).round(3).to_string(index=False))

stat_rows = {}
for label, frame in [("Original", orig), ("Standardized (z)", z), ("Normalized (min-max)", mm),
                     ("Robust", rb)]:
    for col in frame.columns:
        stat_rows[f"{label}: {col}"] = {
            "mean": frame[col].mean(), "sd": frame[col].std(ddof=0),
            "min": frame[col].min(), "max": frame[col].max(),
            "skewness": stats.skew(frame[col])}
stats_table = pd.DataFrame(stat_rows).T.round(4)
stats_table.to_csv(OUT / "summary_statistics_by_method.csv")
print("\nSummary statistics by method\n")
print(stats_table.to_string())

# Full transformed dataset for later use
final = pd.concat([df.reset_index(drop=True)[[GENDER]], comparison, encoded.reset_index(drop=True)], axis=1)
final.round(4).to_csv(OUT / "transformed_dataset.csv", index=False)

# ---------- Interpretation ----------
print("\nInterpretation")
print("- Standardization puts every score on a common scale with mean 0 and sd 1, so "
      "values show how many standard deviations a student is from the average.")
print("- Min-max normalization rescales to 0-1 and keeps the shape of the distribution; "
      "it is sensitive to outliers because they set the minimum and maximum.")
print("- Robust scaling uses the median and IQR, so extreme scores affect it less.")
print("- The skewness table shows which transformation helps: values closest to 0 are best, and "
      "a transform that increases skewness should not be used.")
print("- Scaling changes the units but not the shape: skewness is unchanged by "
      "standardization or min-max scaling. Only non-linear transformations reduce skewness.")
print("- Here all three scores already share the 0-100 scale, so scaling matters most "
      "when you combine them with variables on different scales (for example in PCA, "
      "clustering or regression with penalties).")
print("- Fit scalers on the training data only when building predictive models, then apply "
      "them to the test data to avoid information leakage.")

# ---------- Charts ----------
# Figure 1: Math score under different transformations
fig1, axes = plt.subplots(2, 2, figsize=(12, 8))
panels = [("Original Math score", orig["Math"], "steelblue"),
          ("Standardized (z-score)", z["Math_z"], "seagreen"),
          ("Normalized (min-max, 0 to 1)", mm["Math_minmax"], "darkorange"),
          ("Yeo-Johnson transformed", yj["Math"], "purple")]
for ax, (title, values, colour) in zip(axes.ravel(), panels):
    ax.hist(values, bins=25, color=colour, edgecolor="white")
    ax.axvline(values.mean(), color="black", linestyle="--", label=f"mean = {values.mean():.2f}")
    ax.set_title(f"{title}\nskewness = {stats.skew(values):.2f}")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
fig1.suptitle("Math score: original versus transformed")
fig1.tight_layout()
fig1.savefig(OUT / "math_original_vs_transformed.png", dpi=150)

# Figure 2: boxplots of all three scores on each scale
fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (title, frame) in zip(axes, [("Original", orig),
                                      ("Standardized (z)", z.set_axis(names, axis=1)),
                                      ("Normalized (min-max)", mm.set_axis(names, axis=1))]):
    ax.boxplot([frame[n] for n in names], patch_artist=True)
    ax.set_xticklabels(names)
    ax.set_title(title)
axes[0].set_ylabel("Value")
fig2.suptitle("All three scores on each scale")
fig2.tight_layout()
fig2.savefig(OUT / "boxplots_by_scale.png", dpi=150)

# Figure 3: skewness comparison and Q-Q plots
fig3, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
skew[["Original", "log(1 + x)", "Reflected sqrt", "Yeo-Johnson"]].plot.bar(ax=ax1, rot=0)
ax1.axhline(0, color="black", lw=0.8)
ax1.set_title("Skewness by transformation")
ax1.set_ylabel("Skewness")
ax1.legend(fontsize=8)
stats.probplot(orig["Math"], dist="norm", plot=ax2)
ax2.set_title("Q-Q plot: original Math")
stats.probplot(yj["Math"], dist="norm", plot=ax3)
ax3.set_title("Q-Q plot: Yeo-Johnson Math")
fig3.tight_layout()
fig3.savefig(OUT / "skewness_and_qq_plots.png", dpi=150)

print(f"\nSaved results and charts to: {OUT}")
plt.show()