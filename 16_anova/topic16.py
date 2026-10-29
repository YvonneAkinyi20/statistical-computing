"""
Topic 16: ANOVA and experimental design
Libraries: statsmodels, scipy.stats (pandas, numpy, matplotlib for support)

Outputs: ANOVA table, F-statistic, p-value, group means and post-hoc
comparisons, saved to the output folder.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# ---------- Load data ----------
HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE.parent / "data" / "Cleaned_Students_Performance.csv")
df.columns = [c.strip().lower().replace(" ", "_").replace("/", "_") for c in df.columns]

OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", None)


def find_col(keyword, required=True):
    for c in df.columns:
        if keyword in c:
            return c
    if required:
        raise KeyError(f"No column containing '{keyword}'. Columns: {list(df.columns)}")
    return None


RESPONSE = find_col("math")
GENDER = find_col("gender")

# One-way factor: the first categorical column with at least 3 groups
FACTOR = None
for key in ("race", "ethnic", "parental", "education"):
    col = find_col(key, required=False)
    if col and df[col].nunique() >= 3:
        FACTOR = col
        break
if FACTOR is None:
    raise KeyError("No categorical column with 3 or more groups was found. "
                   f"Set FACTOR by hand to one of: {list(df.columns)}")

data = df[[RESPONSE, FACTOR, GENDER]].dropna().copy()
data[FACTOR] = data[FACTOR].astype(str)
data[GENDER] = data[GENDER].astype(str)
data = data.rename(columns={RESPONSE: "score", FACTOR: "group", GENDER: "gender"})
levels = sorted(data["group"].unique())
ALPHA = 0.05

print("\nEXPERIMENTAL DESIGN")
print(f"- Response variable: {RESPONSE} (continuous)")
print(f"- Factor: {FACTOR} with {len(levels)} levels: {levels}")
print(f"- Second factor for the two-way design: {GENDER}")
print(f"- Sample size: {len(data)} students; group sizes are unequal (unbalanced design).")
print("- This is observational data: students were not randomly assigned to the groups, so "
      "differences cannot be interpreted as causal effects.")

# =====================================================================
# 1. Group means
# =====================================================================
summary = data.groupby("group")["score"].agg(["count", "mean", "std", "min", "max"])
summary["sem"] = summary["std"] / np.sqrt(summary["count"])
summary["ci_lower"] = summary["mean"] - stats.t.ppf(0.975, summary["count"] - 1) * summary["sem"]
summary["ci_upper"] = summary["mean"] + stats.t.ppf(0.975, summary["count"] - 1) * summary["sem"]
summary = summary.round(3)
summary.to_csv(OUT / "group_means.csv")
print("\n1. GROUP MEANS\n")
print(summary)
print(f"Grand mean = {data['score'].mean():.3f}")

# =====================================================================
# 2. One-way ANOVA
# =====================================================================
samples = [g["score"].to_numpy() for _, g in data.groupby("group")]
f_stat, p_val = stats.f_oneway(*samples)

model1 = smf.ols("score ~ C(group)", data=data).fit()
anova1 = sm.stats.anova_lm(model1, typ=2)
ss_between, ss_resid = anova1.loc["C(group)", "sum_sq"], anova1.loc["Residual", "sum_sq"]
eta_sq = ss_between / (ss_between + ss_resid)
anova1["mean_sq"] = anova1["sum_sq"] / anova1["df"]
anova1.round(4).to_csv(OUT / "oneway_anova_table.csv")

print("\n2. ONE-WAY ANOVA")
print("H0: all group means are equal   |   H1: at least one group mean is different\n")
print(anova1.round(4))
print(f"\nF({int(anova1.loc['C(group)', 'df'])}, {int(anova1.loc['Residual', 'df'])}) = {f_stat:.3f}, "
      f"p = {p_val:.4g} (scipy.stats.f_oneway agrees with statsmodels)")
print(f"Effect size: eta-squared = {eta_sq:.4f} "
      f"({eta_sq * 100:.1f}% of the variation in scores is explained by {FACTOR})")
print(f"Decision at alpha = {ALPHA}: "
      f"{'Reject H0' if p_val < ALPHA else 'Fail to reject H0'}")

# =====================================================================
# 3. Assumption checks
# =====================================================================
lev_stat, lev_p = stats.levene(*samples)
sh_stat, sh_p = stats.shapiro(model1.resid)
kw_stat, kw_p = stats.kruskal(*samples)
print("\n3. ASSUMPTION CHECKS")
print(f"- Equal variances (Levene): statistic = {lev_stat:.3f}, p = {lev_p:.4f} "
      f"-> {'variances look equal' if lev_p > ALPHA else 'variances differ; interpret with care'}")
print(f"- Normal residuals (Shapiro-Wilk): W = {sh_stat:.4f}, p = {sh_p:.4f} "
      f"-> {'consistent with normality' if sh_p > ALPHA else 'some departure from normality (ANOVA is fairly robust with large samples)'}")
print(f"- Non-parametric check (Kruskal-Wallis): H = {kw_stat:.3f}, p = {kw_p:.4g} "
      f"-> {'agrees with ANOVA' if (kw_p < ALPHA) == (p_val < ALPHA) else 'disagrees with ANOVA'}")

# =====================================================================
# 4. Post-hoc comparisons: Tukey HSD
# =====================================================================
tukey = pairwise_tukeyhsd(endog=data["score"], groups=data["group"], alpha=ALPHA)
table = tukey._results_table.data
tukey_df = pd.DataFrame(table[1:], columns=table[0])
tukey_df.to_csv(OUT / "tukey_hsd_posthoc.csv", index=False)
print("\n4. POST-HOC COMPARISONS (Tukey HSD)\n")
print(tukey_df.to_string(index=False))
sig = tukey_df[tukey_df["reject"].astype(bool)]
print(f"\n{len(sig)} of {len(tukey_df)} pairs differ significantly at alpha = {ALPHA}.")
for _, row in sig.iterrows():
    print(f"  - {row['group1']} vs {row['group2']}: mean difference = {row['meandiff']:.2f} "
          f"(adjusted p = {row['p-adj']:.4f})")

# =====================================================================
# 5. Two-way ANOVA with interaction
# =====================================================================
model2 = smf.ols("score ~ C(group) * C(gender)", data=data).fit()
anova2 = sm.stats.anova_lm(model2, typ=2)
anova2["mean_sq"] = anova2["sum_sq"] / anova2["df"]
anova2.round(4).to_csv(OUT / "twoway_anova_table.csv")
print(f"\n5. TWO-WAY ANOVA: score ~ {FACTOR} x {GENDER}\n")
print(anova2.round(4))
for term, label in [("C(group)", FACTOR), ("C(gender)", GENDER), ("C(group):C(gender)", f"{FACTOR} x {GENDER} interaction")]:
    p = anova2.loc[term, "PR(>F)"]
    print(f"- {label}: F = {anova2.loc[term, 'F']:.3f}, p = {p:.4g} -> "
          f"{'significant' if p < ALPHA else 'not significant'}")

# ---------- Interpretation ----------
print("\nInterpretation")
if p_val < ALPHA:
    print(f"- The one-way ANOVA is significant (p = {p_val:.4g}): mean math scores are not the "
          f"same across all {FACTOR} groups.")
else:
    print(f"- The one-way ANOVA is not significant (p = {p_val:.4g}): there is no evidence that "
          f"mean math scores differ between {FACTOR} groups.")
print("- ANOVA only says that some difference exists. Tukey HSD shows which pairs differ while "
      "controlling the overall error rate across all comparisons.")
print(f"- Eta-squared = {eta_sq:.3f}: a statistically significant result can still be a small "
      "effect, so always report the effect size as well as the p-value.")
print("- A significant interaction would mean the effect of one factor depends on the level "
      "of the other.")
print("- Good experimental design uses randomization, replication and blocking; because this "
      "data is observational, the results show association, not causation.")

# ---------- Charts ----------
order = summary.sort_values("mean").index.tolist()

# Figure 1: boxplots and group means with 95% CI
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
ax1.boxplot([data.loc[data["group"] == g, "score"] for g in order], patch_artist=True)
ax1.set_xticklabels(order, rotation=20)
ax1.axhline(data["score"].mean(), color="red", linestyle="--", label="Grand mean")
ax1.set_title(f"Score by {FACTOR}")
ax1.set_ylabel("Score")
ax1.legend()

means = summary.loc[order]
ax2.errorbar(range(len(order)), means["mean"], yerr=[means["mean"] - means["ci_lower"],
                                                     means["ci_upper"] - means["mean"]],
             fmt="o", capsize=6, color="navy")
ax2.axhline(data["score"].mean(), color="red", linestyle="--", label="Grand mean")
ax2.set_xticks(range(len(order)))
ax2.set_xticklabels(order, rotation=20)
ax2.set_title(f"Group means with 95% CI (F = {f_stat:.2f}, p = {p_val:.3g})")
ax2.set_ylabel("Mean score")
ax2.legend()
fig1.tight_layout()
fig1.savefig(OUT / "group_comparison.png", dpi=150)

# Figure 2: Tukey simultaneous confidence intervals
fig2, ax = plt.subplots(figsize=(8, 5.5))
tukey.plot_simultaneous(ax=ax, xlabel="Mean score", ylabel=FACTOR)
ax.set_title("Tukey HSD: simultaneous 95% confidence intervals")
fig2.tight_layout()
fig2.savefig(OUT / "tukey_hsd_plot.png", dpi=150)

# Figure 3: interaction plot and residual diagnostics
fig3, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
cell = data.groupby(["group", "gender"])["score"].mean().unstack()
for g in cell.columns:
    ax1.plot(range(len(order)), cell.loc[order, g], "o-", lw=2, label=g)
ax1.set_xticks(range(len(order)))
ax1.set_xticklabels(order, rotation=20)
ax1.set_title(f"Interaction plot: {FACTOR} x {GENDER}")
ax1.set_ylabel("Mean score")
ax1.legend(title=GENDER)

ax2.scatter(model1.fittedvalues, model1.resid, alpha=0.4, s=12)
ax2.axhline(0, color="red", linestyle="--")
ax2.set_title("Residuals vs fitted (one-way model)")
ax2.set_xlabel("Fitted values")
ax2.set_ylabel("Residual")
stats.probplot(model1.resid, dist="norm", plot=ax3)
ax3.set_title("Normal Q-Q plot of residuals")
fig3.tight_layout()
fig3.savefig(OUT / "interaction_and_residuals.png", dpi=150)

print(f"\nSaved results and charts to: {OUT}")
plt.show()