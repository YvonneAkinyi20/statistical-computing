"""
Topic 17: Chi-square and categorical data analysis
Libraries: scipy.stats, statsmodels (pandas, numpy, matplotlib for support)

Outputs: contingency tables, expected frequencies, chi-square statistic,
p-value and conclusion, saved to the output folder.
"""
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import Table2x2

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


MATH = find_col("math")
ALPHA = 0.05
PASS_MARK = 50

# ---------- Categorical variables ----------
cat = pd.DataFrame({"gender": df[find_col("gender")].astype(str)})
for label, key in [("lunch", "lunch"), ("test_prep", "prep"),
                   ("ethnicity", "race"), ("parent_education", "parental")]:
    col = find_col(key, required=False)
    if col:
        cat[label] = df[col].astype(str)

# Categories made from the math score, so a test is always possible
math = df[MATH]
q1, q2 = np.quantile(math.dropna(), [1 / 3, 2 / 3])
cat["math_level"] = pd.cut(math, bins=[-np.inf, q1, q2, np.inf],
                           labels=["Low", "Medium", "High"], include_lowest=True).astype(str)
cat["math_result"] = np.where(math >= PASS_MARK, "Pass", "Fail")
cat = cat[math.notna()]
n_total = len(cat)

print("\nCATEGORICAL VARIABLES USED")
for c in cat.columns:
    print(f"- {c}: {sorted(cat[c].unique())}")
print(f"- math_level: terciles of the math score (Low < {q1:.0f}, Medium up to {q2:.0f}, High above)")
print(f"- math_result: Pass if math score >= {PASS_MARK}")


def cramers_v(chi2, table):
    r, c = table.shape
    return float(np.sqrt(chi2 / (table.to_numpy().sum() * (min(r, c) - 1))))


def effect_label(v):
    return "negligible" if v < 0.1 else "small" if v < 0.3 else "medium" if v < 0.5 else "large"


# =====================================================================
# 1. Chi-square goodness of fit
# =====================================================================
print("\n1. CHI-SQUARE GOODNESS OF FIT")
obs = cat["gender"].value_counts().sort_index()
exp = np.full(len(obs), obs.sum() / len(obs))
gof_stat, gof_p = stats.chisquare(obs, exp)
print(f"H0: the gender groups are equally likely (expected {exp[0]:.1f} each)")
print(pd.DataFrame({"observed": obs, "expected": exp}).round(1))
print(f"chi-square = {gof_stat:.3f}, df = {len(obs) - 1}, p = {gof_p:.4f} -> "
      f"{'Reject H0' if gof_p < ALPHA else 'Fail to reject H0'}")


# =====================================================================
# 2. Chi-square test of independence (detailed)
# =====================================================================
def analyse(a, b):
    """Detailed chi-square test of independence between two categorical variables."""
    observed = pd.crosstab(cat[a], cat[b])
    chi2, p, dof, expected = stats.chi2_contingency(observed, correction=False)
    expected = pd.DataFrame(expected, index=observed.index, columns=observed.columns)
    row_p = observed.sum(axis=1) / n_total
    col_p = observed.sum(axis=0) / n_total
    denom = np.sqrt(expected * np.outer(1 - row_p, 1 - col_p))
    adj_resid = (observed - expected) / denom
    v = cramers_v(chi2, observed)
    ok = (expected >= 5).all().all()

    print(f"\n{'=' * 70}\nTEST OF INDEPENDENCE: {a} x {b}")
    print("H0: the two variables are independent   |   H1: they are associated\n")
    print("Observed frequencies\n")
    print(observed)
    print("\nExpected frequencies (if independent)\n")
    print(expected.round(2))
    print("\nRow percentages\n")
    print((observed.div(observed.sum(axis=1), axis=0) * 100).round(1))
    print(f"\nchi-square = {chi2:.3f}, df = {dof}, p = {p:.4g}")
    print(f"Cramer's V = {v:.3f} ({effect_label(v)} association)")
    print(f"Assumption (all expected counts >= 5): {'met' if ok else 'NOT met, interpret with care'} "
          f"(smallest expected count = {expected.min().min():.2f})")
    print(f"Decision at alpha = {ALPHA}: "
          f"{'Reject H0, the variables are associated' if p < ALPHA else 'Fail to reject H0, no evidence of association'}")
    if p < ALPHA:
        big = [(r, c, adj_resid.loc[r, c]) for r in adj_resid.index for c in adj_resid.columns
               if abs(adj_resid.loc[r, c]) > 1.96]
        for r, c, val in sorted(big, key=lambda t: -abs(t[2])):
            direction = "more" if val > 0 else "fewer"
            print(f"  - {a} = {r}, {b} = {c}: {direction} students than expected (adjusted residual {val:+.2f})")

    name = f"{a}_x_{b}"
    observed.to_csv(OUT / f"{name}_observed.csv")
    expected.round(3).to_csv(OUT / f"{name}_expected.csv")
    adj_resid.round(3).to_csv(OUT / f"{name}_adjusted_residuals.csv")

    # Chart: counts, row proportions, adjusted residuals
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    observed.plot.bar(ax=axes[0], rot=0)
    axes[0].set_title(f"Observed counts: {a} by {b}")
    axes[0].set_ylabel("Students")
    props = observed.div(observed.sum(axis=1), axis=0)
    props.plot.bar(stacked=True, ax=axes[1], rot=0)
    axes[1].set_title("Row proportions")
    axes[1].set_ylabel("Proportion")
    axes[1].legend(title=b, fontsize=8)
    im = axes[2].imshow(adj_resid.to_numpy(), cmap="coolwarm", vmin=-4, vmax=4)
    axes[2].set_xticks(range(adj_resid.shape[1]))
    axes[2].set_xticklabels(adj_resid.columns)
    axes[2].set_yticks(range(adj_resid.shape[0]))
    axes[2].set_yticklabels(adj_resid.index)
    for i in range(adj_resid.shape[0]):
        for j in range(adj_resid.shape[1]):
            axes[2].text(j, i, f"{adj_resid.iloc[i, j]:.1f}", ha="center", va="center")
    fig.colorbar(im, ax=axes[2], label="Adjusted residual")
    axes[2].set_title(f"Adjusted residuals\nchi2 = {chi2:.2f}, p = {p:.3g}, V = {v:.2f}")
    fig.tight_layout()
    fig.savefig(OUT / f"chi_square_{name}.png", dpi=150)
    return chi2, p, dof, v, observed


print("\n2. CHI-SQUARE TESTS OF INDEPENDENCE")
pairs = [("gender", "math_level")]
for extra in ("lunch", "test_prep"):
    if extra in cat.columns:
        pairs.append((extra, "math_level"))
for a, b in pairs:
    analyse(a, b)

# =====================================================================
# 3. A 2 x 2 table: chi-square with correction, Fisher's exact test, odds ratio
# =====================================================================
two = pd.crosstab(cat["gender"], cat["math_result"])
if two.shape == (2, 2):
    print(f"\n{'=' * 70}\n3. 2 x 2 TABLE: gender x math_result (Pass / Fail)\n")
    print(two)
    chi_c, p_c, _, _ = stats.chi2_contingency(two, correction=True)
    odds, p_f = stats.fisher_exact(two)
    t22 = Table2x2(two.to_numpy())
    lo, hi = t22.oddsratio_confint()
    print(f"\nChi-square with Yates continuity correction: {chi_c:.3f}, p = {p_c:.4g}")
    print(f"Fisher's exact test: p = {p_f:.4g}")
    print(f"Odds ratio ({two.index[0]} vs {two.index[1]}, odds of '{two.columns[0]}'): "
          f"{t22.oddsratio:.3f}, 95% CI ({lo:.3f}, {hi:.3f})")
    print("An odds ratio of 1 means no association; the interval contains 1 -> "
          f"{'not significant' if lo <= 1 <= hi else 'significant'}.")
    two.to_csv(OUT / "gender_x_math_result_2x2.csv")

# =====================================================================
# 4. Screening all pairs of categorical variables
# =====================================================================
rows = []
cols = [c for c in cat.columns if c not in ("math_result",)]
for a, b in combinations(cols, 2):
    t = pd.crosstab(cat[a], cat[b])
    chi2, p, dof, exp_ = stats.chi2_contingency(t, correction=False)
    rows.append({"variable_1": a, "variable_2": b, "chi_square": chi2, "df": dof,
                 "p_value": p, "cramers_v": cramers_v(chi2, t),
                 "min_expected": exp_.min(), "significant": p < ALPHA})
screen = pd.DataFrame(rows).sort_values("p_value")
screen["strength"] = screen["cramers_v"].apply(effect_label)
screen = screen.round(5)
screen.to_csv(OUT / "all_pairs_chi_square.csv", index=False)
print(f"\n{'=' * 70}\n4. SCREENING OF ALL PAIRS (sorted by p-value)\n")
print(screen.to_string(index=False))
print(f"\nBecause {len(screen)} tests were run, about {len(screen) * ALPHA:.1f} could be 'significant' "
      "by chance alone; a Bonferroni threshold would be "
      f"{ALPHA / len(screen):.4f}.")

# ---------- Interpretation ----------
print("\nInterpretation")
print("- The chi-square test compares observed counts with the counts expected if the two "
      "variables were independent. A small p-value means the pattern is unlikely under independence.")
print("- Cramer's V measures how strong the association is (0 = none, 1 = perfect). A tiny "
      "p-value with a small V means a real but weak association, which is common with large samples.")
print("- Adjusted residuals beyond +/-1.96 show which cells drive the association.")
print("- The test needs expected counts of at least 5 in the cells; otherwise use Fisher's "
      "exact test or combine categories.")
print("- Association does not imply causation.")

print(f"\nSaved results and charts to: {OUT}")
plt.show()