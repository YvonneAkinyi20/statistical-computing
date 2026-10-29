"""
Topic 19: Non-parametric statistical methods
Libraries: scipy.stats (statsmodels, pandas, numpy, matplotlib for support)

Outputs: rank-based test statistics, p-values and a comparison with the
parametric methods, saved to the output folder.
"""
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM

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


MATH, READING, WRITING = find_col("math"), find_col("reading"), find_col("writing")
GENDER = find_col("gender")
ALPHA = 0.05
data = df.dropna(subset=[MATH, READING, WRITING, GENDER]).reset_index(drop=True)
math, reading, writing = (data[c].to_numpy(float) for c in (MATH, READING, WRITING))

comparison = []        # rows of the final comparison table


def decide(p):
    return "Reject H0" if p < ALPHA else "Fail to reject H0"


def record(situation, np_name, np_stat, np_p, par_name, par_stat, par_p):
    comparison.append({"situation": situation,
                       "non_parametric_test": np_name, "np_statistic": np_stat, "np_p_value": np_p,
                       "parametric_test": par_name, "par_statistic": par_stat, "par_p_value": par_p,
                       "np_decision": decide(np_p), "parametric_decision": decide(par_p),
                       "same_conclusion": (np_p < ALPHA) == (par_p < ALPHA)})


# =====================================================================
# 0. Why non-parametric? Check the normality assumption
# =====================================================================
print("\n0. NORMALITY CHECK (why a rank-based method may be preferred)")
for name, x in [("Math", math), ("Reading", reading), ("Writing", writing)]:
    w, p = stats.shapiro(x)
    print(f"- {name}: Shapiro-Wilk W = {w:.4f}, p = {p:.4g}, skewness = {stats.skew(x):.3f} "
          f"-> {'normality rejected' if p < ALPHA else 'consistent with normality'}")
print("Rank-based tests do not assume normality and are not distorted by outliers. "
      "They compare ranks (or medians) instead of means.")

# =====================================================================
# 1. Two independent groups: Mann-Whitney U vs Welch t-test
# =====================================================================
groups = sorted(data[GENDER].astype(str).unique())
g1 = data.loc[data[GENDER].astype(str) == groups[0], MATH].to_numpy(float)
g2 = data.loc[data[GENDER].astype(str) == groups[1], MATH].to_numpy(float)
u, p_u = stats.mannwhitneyu(g1, g2, alternative="two-sided")
t, p_t = stats.ttest_ind(g1, g2, equal_var=False)
cles = u / (len(g1) * len(g2))
rank_biserial = 2 * cles - 1
d = (g1.mean() - g2.mean()) / np.sqrt((g1.var(ddof=1) + g2.var(ddof=1)) / 2)
record(f"Math: {groups[0]} vs {groups[1]}", "Mann-Whitney U", u, p_u, "Welch t-test", t, p_t)
print(f"\n1. TWO INDEPENDENT GROUPS: Math score, {groups[0]} vs {groups[1]}")
print("H0: the two distributions are the same (no tendency for one group to score higher)")
print(f"   medians: {groups[0]} = {np.median(g1):.1f}, {groups[1]} = {np.median(g2):.1f}  |  "
      f"means: {g1.mean():.2f} vs {g2.mean():.2f}")
print(f"   Mann-Whitney U = {u:.1f}, p = {p_u:.4g} -> {decide(p_u)}")
print(f"   Welch t-test   t = {t:.3f}, p = {p_t:.4g} -> {decide(p_t)}")
print(f"   Effect sizes: rank-biserial r = {rank_biserial:.3f}, "
      f"P({groups[0]} scores higher) = {cles:.3f}, Cohen's d = {d:.3f}")

# =====================================================================
# 2. Paired samples: Wilcoxon signed-rank vs paired t-test
# =====================================================================
w_stat, p_w = stats.wilcoxon(reading, writing)
t_p, p_tp = stats.ttest_rel(reading, writing)
diff = reading - writing
nz = diff[diff != 0]
ranks = stats.rankdata(np.abs(nz))
rb_paired = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()
record("Reading vs Writing (same students)", "Wilcoxon signed-rank", w_stat, p_w,
       "Paired t-test", t_p, p_tp)
print("\n2. PAIRED SAMPLES: Reading vs Writing score for the same students")
print("H0: the median of the paired differences is zero")
print(f"   median difference = {np.median(diff):.1f}, mean difference = {diff.mean():.2f}")
print(f"   Wilcoxon signed-rank W = {w_stat:.1f}, p = {p_w:.4g} -> {decide(p_w)}")
print(f"   Paired t-test        t = {t_p:.3f}, p = {p_tp:.4g} -> {decide(p_tp)}")
print(f"   Rank-biserial effect size = {rb_paired:.3f}")

# =====================================================================
# 3. Three or more groups: Kruskal-Wallis vs one-way ANOVA
# =====================================================================
FACTOR, factor_note = None, ""
for key in ("race", "ethnic", "parental", "education"):
    col = find_col(key, required=False)
    if col and data[col].nunique() >= 3:
        FACTOR = col
        break
if FACTOR is None:
    data["reading_level"] = pd.qcut(data[READING], 3, labels=["Low", "Medium", "High"], duplicates="drop")
    FACTOR = "reading_level"
    factor_note = " (no 3-group categorical column found, so reading-score terciles are used)"
labels = sorted(data[FACTOR].astype(str).unique())
samples = [data.loc[data[FACTOR].astype(str) == lv, MATH].to_numpy(float) for lv in labels]
h, p_h = stats.kruskal(*samples)
f_val, p_f = stats.f_oneway(*samples)
n = sum(len(s) for s in samples)
eps_sq = h * (n + 1) / (n**2 - 1)
record(f"Math across {FACTOR} groups", "Kruskal-Wallis H", h, p_h, "One-way ANOVA F", f_val, p_f)
print(f"\n3. THREE OR MORE GROUPS: Math score across {FACTOR}{factor_note}")
print(f"   groups: {labels}")
print(f"   medians: {[float(np.median(s)) for s in samples]}")
print(f"   Kruskal-Wallis H = {h:.3f}, df = {len(samples) - 1}, p = {p_h:.4g} -> {decide(p_h)}")
print(f"   One-way ANOVA  F = {f_val:.3f}, p = {p_f:.4g} -> {decide(p_f)}")
print(f"   Epsilon-squared (rank effect size) = {eps_sq:.4f}")

if p_h < ALPHA:
    pairs = list(combinations(range(len(labels)), 2))
    post = []
    for i, j in pairs:
        u_ij, p_ij = stats.mannwhitneyu(samples[i], samples[j], alternative="two-sided")
        post.append({"group1": labels[i], "group2": labels[j], "U": u_ij, "p_value": p_ij,
                     "p_bonferroni": min(1.0, p_ij * len(pairs))})
    post = pd.DataFrame(post)
    post["significant"] = post["p_bonferroni"] < ALPHA
    post.round(5).to_csv(OUT / "kruskal_posthoc_mannwhitney.csv", index=False)
    print("\n   Post-hoc pairwise Mann-Whitney tests with Bonferroni correction\n")
    print(post.round(4).to_string(index=False))

# =====================================================================
# 4. Several related samples: Friedman test vs repeated-measures ANOVA
# =====================================================================
chi_f, p_fr = stats.friedmanchisquare(math, reading, writing)
long = pd.DataFrame({"student": np.tile(np.arange(len(math)), 3),
                     "subject": np.repeat(["Math", "Reading", "Writing"], len(math)),
                     "score": np.concatenate([math, reading, writing])})
rm = AnovaRM(long, "score", "student", within=["subject"]).fit()
f_rm, p_rm = float(rm.anova_table["F Value"].iloc[0]), float(rm.anova_table["Pr > F"].iloc[0])
record("Math, Reading, Writing (same students)", "Friedman chi-square", chi_f, p_fr,
       "Repeated-measures ANOVA F", f_rm, p_rm)
print("\n4. SEVERAL RELATED SAMPLES: Math, Reading and Writing for the same students")
print(f"   Friedman chi-square = {chi_f:.3f}, p = {p_fr:.4g} -> {decide(p_fr)}")
print(f"   Repeated-measures ANOVA F = {f_rm:.3f}, p = {p_rm:.4g} -> {decide(p_rm)}")

# =====================================================================
# 5. Association: Spearman and Kendall vs Pearson
# =====================================================================
rho, p_rho = stats.spearmanr(math, reading)
tau, p_tau = stats.kendalltau(math, reading)
r, p_r = stats.pearsonr(math, reading)
record("Association between Math and Reading", "Spearman rank correlation", rho, p_rho,
       "Pearson correlation", r, p_r)
print("\n5. ASSOCIATION: Math vs Reading")
print(f"   Spearman rho = {rho:.4f} (p = {p_rho:.4g})")
print(f"   Kendall tau  = {tau:.4f} (p = {p_tau:.4g})")
print(f"   Pearson r    = {r:.4f} (p = {p_r:.4g})")
print("   Spearman and Kendall measure any monotonic relationship and ignore outliers' size.")

# =====================================================================
# 6. Distribution and spread comparisons
# =====================================================================
ks, p_ks = stats.ks_2samp(g1, g2)
fl, p_fl = stats.fligner(g1, g2)
lev, p_lev = stats.levene(g1, g2)
record(f"Equality of spread, Math ({groups[0]} vs {groups[1]})", "Fligner-Killeen", fl, p_fl,
       "Levene", lev, p_lev)
print("\n6. DISTRIBUTION AND SPREAD")
print(f"   Kolmogorov-Smirnov (same distribution?): D = {ks:.4f}, p = {p_ks:.4g} -> {decide(p_ks)}")
print(f"   Fligner-Killeen (equal spread, rank-based): statistic = {fl:.3f}, p = {p_fl:.4g}")
print(f"   Levene (equal spread, parametric):          statistic = {lev:.3f}, p = {p_lev:.4g}")

# =====================================================================
# 7. Sensitivity to outliers (demonstration)
# =====================================================================
rng = np.random.default_rng(42)
contaminated = g1.copy()
k = max(5, int(0.02 * len(contaminated)))
idx = rng.choice(len(contaminated), size=k, replace=False)
contaminated[idx] = 0                                 # a few extreme (wrong) scores
u_c, p_uc = stats.mannwhitneyu(contaminated, g2, alternative="two-sided")
t_c, p_tc = stats.ttest_ind(contaminated, g2, equal_var=False)
print(f"\n7. SENSITIVITY TO OUTLIERS: {k} scores in the {groups[0]} group are changed to 0")
print(f"   Mean of the group: {g1.mean():.2f} -> {contaminated.mean():.2f}; "
      f"median: {np.median(g1):.1f} -> {np.median(contaminated):.1f}")
print(f"   Welch t-test p-value:   {p_t:.4g} -> {p_tc:.4g}")
print(f"   Mann-Whitney p-value:   {p_u:.4g} -> {p_uc:.4g}")
print("   The mean and the t-test react strongly to a few extreme values; the median and the "
      "rank-based test change much less.")

# ---------- Comparison table ----------
table = pd.DataFrame(comparison)
table[["np_statistic", "par_statistic"]] = table[["np_statistic", "par_statistic"]].round(4)
table.to_csv(OUT / "parametric_vs_nonparametric_comparison.csv", index=False)
print("\nCOMPARISON OF NON-PARAMETRIC AND PARAMETRIC RESULTS\n")
print(table[["situation", "non_parametric_test", "np_p_value", "parametric_test",
             "par_p_value", "same_conclusion"]].to_string(index=False,
      formatters={"np_p_value": "{:.4g}".format, "par_p_value": "{:.4g}".format}))

# ---------- Interpretation ----------
agree = int(table["same_conclusion"].sum())
print("\nInterpretation")
print(f"- The non-parametric and parametric tests reached the same conclusion in {agree} of "
      f"{len(table)} comparisons at alpha = {ALPHA}.")
print("- When the data are roughly symmetric and the sample is large, the two families agree. "
      "Differences appear with skewed data, outliers or small samples.")
print("- Non-parametric tests compare ranks or medians, so they answer a slightly different "
      "question from a test of means.")
print("- With normal data the parametric test is a little more powerful; with outliers or skew "
      "the rank-based test is safer.")
print("- Report an effect size (rank-biserial, epsilon-squared, Cohen's d) along with the p-value.")

# ---------- Charts ----------
# Figure 1: p-values, parametric vs non-parametric
fig1, ax = plt.subplots(figsize=(11, 5.5))
ypos = np.arange(len(table))
floor = 1e-300
ax.barh(ypos + 0.2, -np.log10(table["np_p_value"].clip(lower=floor)), height=0.4,
        color="seagreen", label="Non-parametric")
ax.barh(ypos - 0.2, -np.log10(table["par_p_value"].clip(lower=floor)), height=0.4,
        color="steelblue", label="Parametric")
ax.axvline(-np.log10(ALPHA), color="red", linestyle="--", label=f"alpha = {ALPHA}")
ax.set_yticks(ypos)
ax.set_yticklabels([f"{s}\n({a} / {b})" for s, a, b in zip(table["situation"],
                    table["non_parametric_test"], table["parametric_test"])], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("-log10(p-value)   (longer bar = stronger evidence)")
ax.set_title("Non-parametric versus parametric tests")
ax.legend()
fig1.tight_layout()
fig1.savefig(OUT / "pvalue_comparison.png", dpi=150)

# Figure 2: distributions by gender, and ranks
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
ax1.boxplot([g1, g2], patch_artist=True)
ax1.set_xticklabels(groups)
ax1.set_title(f"Math by gender\nMann-Whitney p = {p_u:.3g}, Welch p = {p_t:.3g}")
ax1.set_ylabel("Math score")

for g, colour in zip(groups, ["tab:red", "tab:blue"]):
    v = np.sort(data.loc[data[GENDER].astype(str) == g, MATH].to_numpy(float))
    ax2.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=colour, label=g)
ax2.set_title(f"Empirical CDFs (KS D = {ks:.3f}, p = {p_ks:.3g})")
ax2.set_xlabel("Math score")
ax2.set_ylabel("Cumulative proportion")
ax2.legend()

rk_m, rk_r = stats.rankdata(math), stats.rankdata(reading)
ax3.scatter(rk_r, rk_m, s=8, alpha=0.4, color="purple")
ax3.set_title(f"Ranks of Math vs Reading (Spearman rho = {rho:.2f})")
ax3.set_xlabel("Rank of Reading score")
ax3.set_ylabel("Rank of Math score")
fig2.tight_layout()
fig2.savefig(OUT / "distributions_and_ranks.png", dpi=150)

# Figure 3: outlier sensitivity
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.boxplot([g1, contaminated, g2], patch_artist=True)
ax1.set_xticklabels([f"{groups[0]}\noriginal", f"{groups[0]}\nwith {k} zeros", groups[1]])
ax1.set_title("Effect of a few extreme scores")
ax1.set_ylabel("Math score")
vals = [p_t, p_tc, p_u, p_uc]
ax2.bar(["Welch t\noriginal", "Welch t\noutliers", "Mann-Whitney\noriginal", "Mann-Whitney\noutliers"],
        vals, color=["steelblue", "navy", "seagreen", "darkgreen"])
ax2.axhline(ALPHA, color="red", linestyle="--", label=f"alpha = {ALPHA}")
ax2.set_yscale("log")
ax2.set_title("p-values before and after adding outliers")
ax2.set_ylabel("p-value (log scale)")
ax2.legend()
fig3.tight_layout()
fig3.savefig(OUT / "outlier_sensitivity.png", dpi=150)

print(f"\nSaved results and charts to: {OUT}")
plt.show()