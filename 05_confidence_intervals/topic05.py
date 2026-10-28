"""
Topic 05: Confidence intervals
Libraries: scipy.stats, statsmodels (numpy, pandas, matplotlib for support)

Outputs: point estimates, confidence intervals and an interpretation
of the uncertainty, saved to the output folder.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.weightstats import DescrStatsW

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


SUBJECTS = {
    "Math": find_col("math"),
    "Reading": find_col("reading"),
    "Writing": find_col("writing"),
}
GENDER = find_col("gender")
PASS_MARK = 50
rng = np.random.default_rng(42)


def mean_ci(x, conf=0.95):
    """t-based confidence interval for a mean (scipy.stats)."""
    x = np.asarray(x, dtype=float)
    se = stats.sem(x)
    lo, hi = stats.t.interval(conf, df=len(x) - 1, loc=x.mean(), scale=se)
    return x.mean(), se, lo, hi


# ---------- Q5a: Point estimate and CI for each mean ----------
rows = []
for subject, col in SUBJECTS.items():
    x = df[col].dropna().to_numpy(dtype=float)
    for conf in (0.90, 0.95, 0.99):
        m, se, lo, hi = mean_ci(x, conf)
        rows.append({"subject": subject, "n": len(x), "confidence": conf,
                     "point_estimate": m, "std_error": se,
                     "ci_lower": lo, "ci_upper": hi, "margin_of_error": (hi - lo) / 2})
ci_table = pd.DataFrame(rows).round(3)
ci_table.to_csv(OUT / "ci_for_means.csv", index=False)
print("\nConfidence intervals for the mean score\n")
print(ci_table.to_string(index=False))

# Cross-check with statsmodels (should match the 95% scipy result)
x_math = df[SUBJECTS["Math"]].dropna().to_numpy(dtype=float)
sm_lo, sm_hi = DescrStatsW(x_math).tconfint_mean(alpha=0.05)
print(f"\nstatsmodels check (Math, 95%): ({sm_lo:.3f}, {sm_hi:.3f})")

# ---------- Q5b: CI by gender and for the difference ----------
g_rows = []
groups = {}
for g, sub in df.groupby(GENDER):
    x = sub[SUBJECTS["Math"]].dropna().to_numpy(dtype=float)
    groups[g] = x
    m, se, lo, hi = mean_ci(x, 0.95)
    g_rows.append({"group": g, "n": len(x), "point_estimate": m,
                   "ci_lower": lo, "ci_upper": hi})
gender_table = pd.DataFrame(g_rows).round(3)
gender_table.to_csv(OUT / "ci_math_by_gender.csv", index=False)
print("\n95% CI for the mean math score by gender\n")
print(gender_table.to_string(index=False))

names = list(groups)
if len(names) == 2:
    a, b = groups[names[0]], groups[names[1]]
    diff = a.mean() - b.mean()
    se_d = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    dof = se_d**4 / ((a.var(ddof=1) / len(a))**2 / (len(a) - 1)
                     + (b.var(ddof=1) / len(b))**2 / (len(b) - 1))   # Welch
    d_lo, d_hi = stats.t.interval(0.95, dof, loc=diff, scale=se_d)
    print(f"\nDifference in mean math score ({names[0]} - {names[1]}): "
          f"{diff:.2f}, 95% CI ({d_lo:.2f}, {d_hi:.2f})")

# ---------- Q5c: CI for a proportion ----------
pass_count = int((x_math >= PASS_MARK).sum())
p_hat = pass_count / len(x_math)
p_lo, p_hi = proportion_confint(pass_count, len(x_math), alpha=0.05, method="wilson")
print(f"\nProportion with math score >= {PASS_MARK}: {p_hat:.3f}, "
      f"95% CI ({p_lo:.3f}, {p_hi:.3f})")

# ---------- Interpretation of uncertainty ----------
m, se, lo, hi = mean_ci(x_math, 0.95)
m99, _, lo99, hi99 = mean_ci(x_math, 0.99)
print("\nInterpretation")
print(f"- The sample mean math score is {m:.2f}. We are 95% confident that the "
      f"true mean lies between {lo:.2f} and {hi:.2f}.")
print("  This means: if we repeated the sampling many times, about 95% of the "
      "intervals built this way would contain the true mean.")
print(f"- Going to 99% confidence widens the interval to ({lo99:.2f}, {hi99:.2f}): "
      "more confidence costs precision.")
print("- A larger sample gives a smaller standard error and a narrower interval.")

# ---------- Charts ----------
# Figure 1: 95% CI for each subject and for math by gender
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
labels, means, errs = [], [], []
for subject, col in SUBJECTS.items():
    mm, _, l, h = mean_ci(df[col].dropna(), 0.95)
    labels.append(subject); means.append(mm); errs.append((h - l) / 2)
ax1.errorbar(labels, means, yerr=errs, fmt="o", capsize=8, color="navy")
ax1.set_title("Mean score with 95% confidence interval")
ax1.set_ylabel("Score")

ax2.errorbar(gender_table["group"].astype(str), gender_table["point_estimate"],
             yerr=(gender_table["ci_upper"] - gender_table["ci_lower"]) / 2,
             fmt="s", capsize=8, color="darkgreen")
ax2.set_title("Math score by gender (95% CI)")
ax2.set_ylabel("Score")
fig1.tight_layout()
fig1.savefig(OUT / "ci_for_means.png", dpi=150)

# Figure 2: CI width versus confidence level
levels = np.linspace(0.80, 0.99, 20)
widths = [mean_ci(x_math, c)[3] - mean_ci(x_math, c)[2] for c in levels]
fig2, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(levels * 100, widths, "o-")
ax.set_title("Higher confidence gives a wider interval")
ax.set_xlabel("Confidence level (%)")
ax.set_ylabel("Interval width (Math score)")
fig2.tight_layout()
fig2.savefig(OUT / "ci_width_vs_confidence.png", dpi=150)

# Figure 3: coverage simulation - how often does a 95% CI capture the true mean?
true_mean = x_math.mean()
n_sim, n = 50, 30
fig3, ax = plt.subplots(figsize=(8, 8))
hits = 0
for i in range(n_sim):
    s = rng.choice(x_math, size=n, replace=False)
    sm, _, sl, sh = mean_ci(s, 0.95)
    covered = sl <= true_mean <= sh
    hits += covered
    ax.plot([sl, sh], [i, i], color="steelblue" if covered else "red", lw=2)
    ax.plot(sm, i, "k.", ms=4)
ax.axvline(true_mean, color="black", linestyle="--", label=f"True mean = {true_mean:.1f}")
ax.set_title(f"{n_sim} confidence intervals (n = {n}): {hits} contain the true mean\n"
             "(red = missed)")
ax.set_xlabel("Math score")
ax.set_ylabel("Sample number")
ax.legend()
fig3.tight_layout()
fig3.savefig(OUT / "ci_coverage_simulation.png", dpi=150)

print(f"\nCoverage simulation: {hits} of {n_sim} intervals contained the true mean.")
print(f"Saved results and charts to: {OUT}")
plt.show()