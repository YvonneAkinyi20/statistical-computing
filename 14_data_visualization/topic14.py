"""
Topic 14: Statistical data visualization
Libraries: matplotlib, plotly (numpy, pandas, scipy for support)

Outputs: histogram, boxplot, scatter plot and line graph, plus interactive
plotly charts, saved to the output folder.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# ---------- Load data ----------
HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE.parent / "data" / "Cleaned_Students_Performance.csv")
df.columns = [c.strip().lower().replace(" ", "_").replace("/", "_") for c in df.columns]

OUT = HERE / "output"
OUT.mkdir(exist_ok=True)


def find_col(keyword, required=True):
    for c in df.columns:
        if keyword in c:
            return c
    if required:
        raise KeyError(f"No column containing '{keyword}'. Columns: {list(df.columns)}")
    return None


MATH, READING, WRITING = find_col("math"), find_col("reading"), find_col("writing")
GENDER = find_col("gender")
PREP = find_col("prep", required=False)            # test preparation course (if present)
EDU = find_col("parental", required=False)         # parental level of education (if present)

SCORES = {"Math": MATH, "Reading": READING, "Writing": WRITING}
groups = sorted(df[GENDER].dropna().unique())
palette = dict(zip(groups, ["tab:red", "tab:blue", "tab:green", "tab:orange"]))

# ---------- Quick numerical summary to support the charts ----------
print("\nNumerical summary behind the charts\n")
summary = pd.DataFrame({
    name: {"mean": df[c].mean(), "median": df[c].median(), "sd": df[c].std(),
           "skewness": stats.skew(df[c].dropna())}
    for name, c in SCORES.items()}).T.round(3)
print(summary)
summary.to_csv(OUT / "visualization_summary.csv")
print("\nMedian score by gender\n")
print(df.groupby(GENDER)[list(SCORES.values())].median().rename(columns={v: k for k, v in SCORES.items()}))

# =====================================================================
# Chart 1: Histograms (shape of each distribution)
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, (name, col) in zip(axes, SCORES.items()):
    values = df[col].dropna()
    ax.hist(values, bins=20, density=True, color="steelblue", edgecolor="white", alpha=0.85)
    grid = np.linspace(values.min(), values.max(), 200)
    ax.plot(grid, stats.gaussian_kde(values)(grid), color="navy", lw=2, label="Density curve")
    ax.axvline(values.mean(), color="red", linestyle="--", label=f"Mean = {values.mean():.1f}")
    ax.axvline(values.median(), color="green", linestyle=":", lw=2, label=f"Median = {values.median():.1f}")
    ax.set_title(f"{name} scores")
    ax.set_xlabel("Score")
    ax.legend(fontsize=8)
axes[0].set_ylabel("Density")
fig.suptitle("Histograms: distribution of test scores")
fig.tight_layout()
fig.savefig(OUT / "01_histograms.png", dpi=150)

# =====================================================================
# Chart 2: Boxplots (spread, median and outliers by group)
# =====================================================================
n_panels = 2 if PREP else 1
fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5.5), squeeze=False)
ax = axes[0, 0]
width, positions = 0.8 / len(groups), np.arange(len(SCORES))
for i, g in enumerate(groups):
    data = [df.loc[df[GENDER] == g, c].dropna() for c in SCORES.values()]
    bp = ax.boxplot(data, positions=positions + (i - (len(groups) - 1) / 2) * width,
                    widths=width * 0.9, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(palette[g])
        patch.set_alpha(0.6)
    ax.plot([], [], color=palette[g], lw=8, alpha=0.6, label=str(g))
ax.set_xticks(positions)
ax.set_xticklabels(list(SCORES))
ax.set_title("Scores by gender")
ax.set_ylabel("Score")
ax.legend(title=GENDER, loc="upper left", bbox_to_anchor=(1.01, 1))
if PREP:
    ax2 = axes[0, 1]
    levels = sorted(df[PREP].dropna().unique())
    ax2.boxplot([df.loc[df[PREP] == lv, MATH].dropna() for lv in levels], patch_artist=True)
    ax2.set_xticklabels([str(lv) for lv in levels])
    ax2.set_title("Math score by test preparation course")
    ax2.set_ylabel("Math score")
fig.suptitle("Boxplots: median, quartiles and outliers")
fig.tight_layout()
fig.savefig(OUT / "02_boxplots.png", dpi=150, bbox_inches="tight")

# =====================================================================
# Chart 3: Scatter plots (relationship between two scores)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, (xc, yc, xn, yn) in zip(axes, [(READING, MATH, "Reading", "Math"),
                                        (WRITING, READING, "Writing", "Reading")]):
    for g in groups:
        sub = df[df[GENDER] == g]
        ax.scatter(sub[xc], sub[yc], s=14, alpha=0.45, color=palette[g], label=str(g))
    slope, intercept = np.polyfit(df[xc], df[yc], 1)
    xs = np.linspace(df[xc].min(), df[xc].max(), 100)
    r = df[xc].corr(df[yc])
    ax.plot(xs, intercept + slope * xs, color="black", lw=2, label=f"Fit (r = {r:.2f})")
    ax.set_title(f"{yn} vs {xn}")
    ax.set_xlabel(f"{xn} score")
    ax.set_ylabel(f"{yn} score")
    ax.legend(fontsize=8)
fig.suptitle("Scatter plots with fitted lines")
fig.tight_layout()
fig.savefig(OUT / "03_scatter_plots.png", dpi=150)

# =====================================================================
# Chart 4: Line graphs
#   (a) average Math score by reading-score band (trend across ordered groups)
#   (b) empirical cumulative distribution of Math scores by gender
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
bands = pd.cut(df[READING], bins=range(0, 101, 10), include_lowest=True)
band_stats = df.groupby(bands, observed=True)[MATH].agg(["mean", "sem", "count"])
band_stats = band_stats[band_stats["count"] >= 5]
mid = [interval.mid for interval in band_stats.index]
ax1.errorbar(mid, band_stats["mean"], yerr=1.96 * band_stats["sem"], marker="o",
             color="navy", capsize=4, lw=2)
ax1.set_title("Average Math score by Reading score band (95% CI)")
ax1.set_xlabel("Reading score (band midpoint)")
ax1.set_ylabel("Average Math score")
ax1.grid(alpha=0.3)

for g in groups:
    v = np.sort(df.loc[df[GENDER] == g, MATH].dropna().to_numpy())
    ax2.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=palette[g], lw=2, label=str(g))
ax2.set_title("Cumulative distribution of Math scores")
ax2.set_xlabel("Math score")
ax2.set_ylabel("Proportion of students at or below")
ax2.legend(title=GENDER)
ax2.grid(alpha=0.3)
fig.suptitle("Line graphs")
fig.tight_layout()
fig.savefig(OUT / "04_line_graphs.png", dpi=150)

# =====================================================================
# Chart 5: Bar charts (counts and group means)
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
counts = df[GENDER].value_counts().reindex(groups)
ax1.bar([str(g) for g in groups], counts, color=[palette[g] for g in groups])
for i, v in enumerate(counts):
    ax1.text(i, v + counts.max() * 0.01, str(v), ha="center")
ax1.set_title("Number of students by gender")
ax1.set_ylabel("Count")

means = df.groupby(GENDER)[list(SCORES.values())].mean()
for i, g in enumerate(groups):
    ax2.bar(np.arange(3) + (i - (len(groups) - 1) / 2) * 0.8 / len(groups), means.loc[g],
            width=0.8 / len(groups), color=palette[g], label=str(g))
ax2.set_xticks(range(3))
ax2.set_xticklabels(list(SCORES))
ax2.set_ylim(means.values.min() - 10, means.values.max() + 5)
ax2.set_title("Average score by gender")
ax2.set_ylabel("Average score (axis starts above zero)")
ax2.legend(title=GENDER)
fig.tight_layout()
fig.savefig(OUT / "05_bar_charts.png", dpi=150)

# =====================================================================
# Interactive plotly charts (open the HTML files in a browser)
# =====================================================================
if HAS_PLOTLY:
    fig_p = px.scatter(df, x=READING, y=MATH, color=GENDER, trendline="ols", opacity=0.6,
                       title="Math vs Reading (interactive: hover, zoom, click the legend)")
    fig_p.write_html(OUT / "interactive_scatter.html")
    fig_b = px.box(df.melt(id_vars=GENDER, value_vars=list(SCORES.values()),
                           var_name="subject", value_name="score"),
                   x="subject", y="score", color=GENDER, points="outliers",
                   title="Scores by subject and gender (interactive)")
    fig_b.write_html(OUT / "interactive_boxplot.html")
    fig_h = px.histogram(df, x=MATH, color=GENDER, barmode="overlay", opacity=0.6, nbins=25,
                         title="Math score histogram (interactive)")
    fig_h.write_html(OUT / "interactive_histogram.html")
    print("\nSaved interactive plotly charts (.html) to the output folder.")
else:
    print("\nplotly is not installed, so the interactive charts were skipped. "
          "Run: pip install plotly   then run this script again.")

# ---------- Interpretation ----------
print("\nHow to read the charts")
print("- Histogram: shows the shape of each score distribution. Compare the mean and "
      "median: when they differ, the distribution is skewed.")
print("- Boxplot: the box is the middle 50% of students, the line is the median, and "
      "points beyond the whiskers are potential outliers.")
print("- Scatter plot: each dot is one student. An upward cloud and a positive r "
      "mean that students who do well in one subject tend to do well in the other.")
print("- Line graph: the first line shows how the average math score rises with the "
      "reading band; the second shows the share of students below each math score.")
print("- Bar chart: good for comparing counts and averages between categories.")

print(f"\nSaved results and charts to: {OUT}")
plt.show()