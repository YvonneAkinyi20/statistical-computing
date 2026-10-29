from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

# ---------- Load data ----------
HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE.parent / "data" / "Cleaned_Students_Performance.csv")
df.columns = [c.replace(" ", "_").replace("/", "_") for c in df.columns]

out = HERE / "output"
out.mkdir(exist_ok=True)

scores = ["math_score", "reading_score", "writing_score"]
original = df[scores].copy()          # the complete "true" data
n = len(df)
print(f"Missing values in the original file: {int(original.isna().sum().sum())}\n")

# lunch: 1 = standard, 0 = free/reduced (inferred from the original data)
if pd.api.types.is_numeric_dtype(df["lunch"]):
    standard_lunch = df["lunch"] == 1
else:
    standard_lunch = df["lunch"] == "standard"

# ---------- Step 1: create artificial missingness ----------
rng = np.random.default_rng(42)
data = original.copy()
data.loc[rng.random(n) < 0.10, "math_score"] = np.nan               # MCAR
p_read = np.where(standard_lunch, 0.05, 0.20)                       # MAR
data.loc[rng.random(n) < p_read, "reading_score"] = np.nan
data.loc[rng.random(n) < 0.05, "writing_score"] = np.nan            # MCAR

# ---------- Step 2: missing-data summary ----------
miss = data.isna()
summary = pd.DataFrame({
    "missing_n": miss.sum(),
    "missing_pct": (miss.mean() * 100).round(2),
    "observed_n": data.notna().sum(),
})
print("=== Missing-data summary ===")
print(summary, "\n")
print(f"Rows with at least one missing value : {int(miss.any(axis=1).sum())} "
      f"({miss.any(axis=1).mean() * 100:.1f}%)")
print(f"Complete rows                        : {int((~miss.any(axis=1)).sum())}\n")
summary.to_csv(out / "py_missing_summary.csv")

patterns = miss.astype(int).astype(str).agg("".join, axis=1).value_counts()
print("Missingness patterns (1 = missing; order: math, reading, writing)")
print(patterns, "\n")

# Is reading-score missingness related to lunch type? (MAR check)
a = miss.loc[standard_lunch, "reading_score"].astype(int)
b = miss.loc[~standard_lunch, "reading_score"].astype(int)
t, p = stats.ttest_ind(a, b, equal_var=False)
print("=== Does reading missingness depend on lunch type? ===")
print(f"Proportion missing, standard lunch    = {a.mean():.3f}")
print(f"Proportion missing, free/reduced lunch = {b.mean():.3f}")
print(f"Welch t-test p-value                  = {p:.4g}\n")

# ---------- Step 3: imputation ----------
imputers = {
    "Mean": SimpleImputer(strategy="mean"),
    "Median": SimpleImputer(strategy="median"),
    "KNN (k=5)": KNNImputer(n_neighbors=5),
    "Iterative (regression)": IterativeImputer(random_state=42, max_iter=20),
}
imputed = {}
for name, imp in imputers.items():
    imputed[name] = pd.DataFrame(imp.fit_transform(data), columns=scores)
complete_cases = data.dropna()

# ---------- Step 4: compare before and after ----------
rows = []
mask = miss.values
for name, imp_df in imputed.items():
    rmse = np.sqrt(np.mean((imp_df.values[mask] - original.values[mask]) ** 2))
    rows.append({
        "method": name,
        "RMSE_on_missing_cells": rmse,
        "mean_math": imp_df["math_score"].mean(),
        "sd_math": imp_df["math_score"].std(),
        "corr_math_reading": imp_df["math_score"].corr(imp_df["reading_score"]),
    })
rows.append({
    "method": "Complete-case deletion",
    "RMSE_on_missing_cells": np.nan,
    "mean_math": complete_cases["math_score"].mean(),
    "sd_math": complete_cases["math_score"].std(),
    "corr_math_reading": complete_cases["math_score"].corr(
        complete_cases["reading_score"]),
})
rows.append({
    "method": "TRUE (original data)",
    "RMSE_on_missing_cells": 0.0,
    "mean_math": original["math_score"].mean(),
    "sd_math": original["math_score"].std(),
    "corr_math_reading": original["math_score"].corr(original["reading_score"]),
})
comparison = pd.DataFrame(rows).round(3)
print("=== Comparison of methods (smaller RMSE is better) ===")
print(comparison, f"\n\nRows kept by complete-case deletion: "
      f"{len(complete_cases)} of {n}\n")
comparison.to_csv(out / "py_imputation_comparison.csv", index=False)

best = comparison.dropna(subset=["RMSE_on_missing_cells"]).iloc[:-1] \
    .sort_values("RMSE_on_missing_cells").iloc[0]["method"]
print(f"Best method by RMSE: {best}")

best_df = imputed[best]
best_df.round(2).to_csv(out / "py_imputed_dataset.csv", index=False)

# ---------- Charts ----------
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

axes[0].imshow(miss.values, aspect="auto", cmap="gray_r", interpolation="nearest")
axes[0].set_xticks(range(3))
axes[0].set_xticklabels(["math", "reading", "writing"])
axes[0].set_ylabel("Student (row)")
axes[0].set_title("Missingness map (black = missing)")

for name, imp_df in imputed.items():
    imp_df["math_score"].plot(kind="kde", ax=axes[1], label=name)
original["math_score"].plot(kind="kde", ax=axes[1], color="black",
                            linewidth=2, linestyle="--", label="True")
axes[1].set_xlabel("Math score")
axes[1].set_title("Math score distribution after imputation")
axes[1].legend(fontsize=8)

comp = comparison.dropna(subset=["RMSE_on_missing_cells"]).iloc[:-1]
axes[2].bar(comp["method"], comp["RMSE_on_missing_cells"],
            edgecolor="black", alpha=0.75)
axes[2].set_ylabel("RMSE on the deleted cells")
axes[2].set_title("Imputation accuracy")
axes[2].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig(out / "py_missing_data_plots.png", dpi=150)
plt.show()