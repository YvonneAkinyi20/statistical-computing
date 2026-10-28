"""
Topic 06: Hypothesis testing
Libraries: scipy.stats, statsmodels (numpy, pandas, matplotlib for support)

Outputs: test statistic, p-value, decision on the hypothesis and an
interpretation for each test, saved to the output folder.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

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


MATH, READING, WRITING = find_col("math"), find_col("reading"), find_col("writing")
GENDER = find_col("gender")

ALPHA = 0.05        # significance level
MU0 = 70            # hypothesised population mean math score (Test 1)
PASS_MARK = 50      # pass mark used for the proportion test (Test 4)
P0 = 0.80           # hypothesised pass rate (Test 4)


def decision(p):
    return "Reject H0" if p < ALPHA else "Fail to reject H0"


results = []


def record(name, h0, h1, stat, p):
    results.append({"test": name, "H0": h0, "H1": h1,
                    "test_statistic": round(float(stat), 4),
                    "p_value": round(float(p), 4),
                    "decision": decision(p)})


# ---------- Test 1: one-sample t-test ----------
x = df[MATH].dropna().to_numpy(dtype=float)
t1, p1 = stats.ttest_1samp(x, MU0)
record("One-sample t-test (Math)", f"mean = {MU0}", f"mean != {MU0}", t1, p1)
print(f"\nTest 1: One-sample t-test, H0: mean math score = {MU0}")
print(f"  sample mean = {x.mean():.2f}, t = {t1:.3f}, p = {p1:.4g} -> {decision(p1)}")

# Normality check (supports the use of t-tests)
sh_stat, sh_p = stats.shapiro(x)
print(f"  Shapiro-Wilk normality check: W = {sh_stat:.3f}, p = {sh_p:.4f} "
      "(with a large sample the t-test is robust by the CLT)")

# ---------- Test 2: two-sample (Welch) t-test by gender ----------
groups = {g: s[MATH].dropna().to_numpy(dtype=float) for g, s in df.groupby(GENDER)}
names = list(groups)
if len(names) == 2:
    a, b = groups[names[0]], groups[names[1]]
    t2, p2 = stats.ttest_ind(a, b, equal_var=False)
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    cohen_d = (a.mean() - b.mean()) / pooled_sd
    u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
    record("Two-sample Welch t-test (Math by gender)",
           "mean(group 1) = mean(group 2)", "means differ", t2, p2)
    record("Mann-Whitney U (Math by gender)",
           "distributions are equal", "distributions differ", u_stat, u_p)
    print(f"\nTest 2: Welch t-test, H0: mean math score is equal for "
          f"{names[0]} and {names[1]}")
    print(f"  means: {names[0]} = {a.mean():.2f}, {names[1]} = {b.mean():.2f}")
    print(f"  t = {t2:.3f}, p = {p2:.4g} -> {decision(p2)}, Cohen's d = {cohen_d:.2f}")
    print(f"  Mann-Whitney check: U = {u_stat:.1f}, p = {u_p:.4g} -> {decision(u_p)}")

# ---------- Test 3: paired t-test (reading vs writing) ----------
pair = df[[READING, WRITING]].dropna().to_numpy(dtype=float)
t3, p3 = stats.ttest_rel(pair[:, 0], pair[:, 1])
diff = pair[:, 0] - pair[:, 1]
record("Paired t-test (Reading vs Writing)", "mean difference = 0",
       "mean difference != 0", t3, p3)
print("\nTest 3: Paired t-test, H0: mean reading score = mean writing score")
print(f"  mean difference (reading - writing) = {diff.mean():.2f}, "
      f"t = {t3:.3f}, p = {p3:.4g} -> {decision(p3)}")

# ---------- Test 4: one-proportion z-test ----------
count = int((x >= PASS_MARK).sum())
z4, p4 = proportions_ztest(count, len(x), value=P0)
record(f"One-proportion z-test (Math >= {PASS_MARK})", f"p = {P0}", f"p != {P0}", z4, p4)
print(f"\nTest 4: One-proportion z-test, H0: pass rate (math >= {PASS_MARK}) = {P0}")
print(f"  sample proportion = {count / len(x):.3f}, z = {z4:.3f}, p = {p4:.4f} "
      f"-> {decision(p4)}")

# ---------- Summary table ----------
summary = pd.DataFrame(results)
summary.to_csv(OUT / "hypothesis_test_summary.csv", index=False)
print(f"\nSummary of results (alpha = {ALPHA})\n")
print(summary[["test", "test_statistic", "p_value", "decision"]].to_string(index=False))

# ---------- Interpretation ----------
print("\nInterpretation")
print(f"- Decision rule: if p < {ALPHA} we reject H0, otherwise we fail to reject it.")
print("- 'Fail to reject' does not prove H0 true; it means the data do not give "
      "enough evidence against it.")
print("- A very small p-value means the observed result would be unlikely if H0 "
      "were true. It does not say how large or important the effect is, so "
      "effect sizes such as Cohen's d should be read alongside it.")

# ---------- Charts ----------
# Figure 1: t-distribution with rejection regions for Test 1
dof = len(x) - 1
t_crit = stats.t.ppf(1 - ALPHA / 2, dof)
lim = max(4.5, abs(t1) + 1)
grid = np.linspace(-lim, lim, 600)
pdf = stats.t.pdf(grid, dof)
fig1, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(grid, pdf, color="black")
ax.fill_between(grid, pdf, where=grid <= -t_crit, color="red", alpha=0.4,
                label=f"Rejection region (alpha = {ALPHA})")
ax.fill_between(grid, pdf, where=grid >= t_crit, color="red", alpha=0.4)
ax.axvline(t1, color="blue", lw=2, label=f"Observed t = {t1:.2f} (p = {p1:.4g})")
ax.set_title(f"One-sample t-test: H0 mean math score = {MU0}")
ax.set_xlabel("t value")
ax.set_ylabel("Density")
ax.legend()
fig1.tight_layout()
fig1.savefig(OUT / "t_test_rejection_region.png", dpi=150)

# Figure 2: group comparison and paired differences
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
ax1.boxplot([groups[g] for g in names])
ax1.set_xticklabels([str(g) for g in names])
ax1.set_title("Math score by gender")
ax1.set_ylabel("Score")
ax2.hist(diff, bins=25, color="steelblue", edgecolor="white")
ax2.axvline(0, color="red", linestyle="--", label="No difference")
ax2.axvline(diff.mean(), color="black", label=f"Mean = {diff.mean():.2f}")
ax2.set_title("Reading minus writing score (paired differences)")
ax2.set_xlabel("Difference")
ax2.set_ylabel("Frequency")
ax2.legend()
fig2.tight_layout()
fig2.savefig(OUT / "group_comparison_and_paired_differences.png", dpi=150)

print(f"\nSaved results and charts to: {OUT}")
plt.show()