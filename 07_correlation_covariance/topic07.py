"""
Topic 07: Correlation and covariance
Libraries: pandas, numpy, scipy (matplotlib for the charts)

Outputs: covariance matrix, correlation matrix, scatter plots and an
interpretation, saved to the output folder.
"""
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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
scores = df[list(COLS.values())].dropna()
scores.columns = list(COLS.keys())


def strength(r):
    """Describe the strength and direction of a correlation."""
    a = abs(r)
    word = ("very strong" if a >= 0.8 else "strong" if a >= 0.6 else
            "moderate" if a >= 0.4 else "weak" if a >= 0.2 else "very weak")
    return f"{word} {'positive' if r > 0 else 'negative'}"


# ---------- Covariance matrix ----------
cov = scores.cov()                       # sample covariance (n - 1)
cov.round(3).to_csv(OUT / "covariance_matrix.csv")
print("\nCovariance matrix\n")
print(cov.round(2))

# ---------- Correlation matrices ----------
pearson = scores.corr(method="pearson")
spearman = scores.corr(method="spearman")
pearson.round(3).to_csv(OUT / "correlation_matrix_pearson.csv")
spearman.round(3).to_csv(OUT / "correlation_matrix_spearman.csv")
print("\nPearson correlation matrix\n")
print(pearson.round(3))
print("\nSpearman (rank) correlation matrix\n")
print(spearman.round(3))

# Check: correlation = covariance / (sd_x * sd_y)
check = cov.loc["Math", "Reading"] / (scores["Math"].std() * scores["Reading"].std())
print(f"\nCheck: cov(Math, Reading) / (sd_Math * sd_Reading) = {check:.4f} "
      f"(matches Pearson r = {pearson.loc['Math', 'Reading']:.4f})")

# ---------- Pairwise tests ----------
rows = []
for a, b in combinations(scores.columns, 2):
    r, p = stats.pearsonr(scores[a], scores[b])
    rho, p_s = stats.spearmanr(scores[a], scores[b])
    rows.append({"pair": f"{a} vs {b}",
                 "covariance": round(cov.loc[a, b], 3),
                 "pearson_r": round(r, 4), "pearson_p": p,
                 "r_squared": round(r**2, 4),
                 "spearman_rho": round(rho, 4), "spearman_p": p_s,
                 "strength": strength(r)})
pairs = pd.DataFrame(rows)
pairs.to_csv(OUT / "pairwise_correlations.csv", index=False)
print("\nPairwise correlations\n")
print(pairs[["pair", "covariance", "pearson_r", "r_squared",
             "spearman_rho", "strength"]].to_string(index=False))

# ---------- Interpretation ----------
print("\nInterpretation")
for _, row in pairs.iterrows():
    print(f"- {row['pair']}: r = {row['pearson_r']:.3f} ({row['strength']}). "
          f"About {row['r_squared'] * 100:.0f}% of the variation in one score "
          "is shared with the other.")
if (pairs["covariance"] > 0).all():
    print("- Covariance is positive for every pair: students who score above average in "
          "one subject tend to score above average in the others.")
else:
    print("- Covariance signs differ across pairs: positive means the scores move "
          "together, negative means they move in opposite directions.")
print("- Covariance depends on the units of measurement, so correlation is used to "
      "judge the strength of the relationship.")
print("- Correlation does not prove causation: a strong link between scores does not "
      "mean one subject causes the other.")

# ---------- Charts ----------
# Figure 1: correlation heatmap
fig1, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(pearson.values, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(pearson)))
ax.set_yticks(range(len(pearson)))
ax.set_xticklabels(pearson.columns)
ax.set_yticklabels(pearson.index)
for i in range(len(pearson)):
    for j in range(len(pearson)):
        ax.text(j, i, f"{pearson.values[i, j]:.2f}", ha="center", va="center")
fig1.colorbar(im, ax=ax, label="Pearson r")
ax.set_title("Correlation matrix of test scores")
fig1.tight_layout()
fig1.savefig(OUT / "correlation_heatmap.png", dpi=150)

# Figure 2: scatter plots with a fitted line
fig2, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, (a, b) in zip(axes, combinations(scores.columns, 2)):
    x, y = scores[a].to_numpy(float), scores[b].to_numpy(float)
    ax.scatter(x, y, alpha=0.4, s=14, color="steelblue")
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="red", lw=2)
    r = pearson.loc[a, b]
    ax.set_title(f"{a} vs {b} (r = {r:.2f})")
    ax.set_xlabel(f"{a} score")
    ax.set_ylabel(f"{b} score")
fig2.tight_layout()
fig2.savefig(OUT / "scatter_plots.png", dpi=150)

print(f"\nSaved results and charts to: {OUT}")
plt.show()