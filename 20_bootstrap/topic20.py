from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils import resample

# ---------- Load data ----------
HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE.parent / "data" / "Cleaned_Students_Performance.csv")
df.columns = [c.replace(" ", "_").replace("/", "_") for c in df.columns]

# 0/1 codes inferred from the original data
if pd.api.types.is_numeric_dtype(df["gender"]):
    df["gender"] = df["gender"].map({0: "female", 1: "male"})

out = HERE / "output"
out.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
B = 5000                      # number of bootstrap samples
math = df["math_score"].to_numpy()
n = len(math)

# ---------- Q1: bootstrap the mean ----------
boot_means = np.array([rng.choice(math, size=n, replace=True).mean()
                       for _ in range(B)])

se_boot = boot_means.std(ddof=1)
se_formula = math.std(ddof=1) / np.sqrt(n)
ci_pct = np.percentile(boot_means, [2.5, 97.5])

# classical t-interval for comparison
ci_t = stats.t.interval(0.95, n - 1, loc=math.mean(), scale=se_formula)

print("=== Bootstrap of the mean math score ===")
print(f"Sample mean                  = {math.mean():.3f}")
print(f"Bootstrap mean of means      = {boot_means.mean():.3f}")
print(f"Bootstrap standard error     = {se_boot:.4f}")
print(f"Formula standard error s/√n  = {se_formula:.4f}")
print(f"95% bootstrap percentile CI  = ({ci_pct[0]:.3f}, {ci_pct[1]:.3f})")
print(f"95% classical t CI           = ({ci_t[0]:.3f}, {ci_t[1]:.3f})\n")

# ---------- Q2: bootstrap other statistics ----------
def boot_stat(data, func, B=B):
    return np.array([func(resample(data, replace=True, n_samples=len(data),
                                   random_state=int(i)))
                     for i in range(B)])

rows = []
for name, func in [("mean", np.mean), ("median", np.median),
                   ("std_dev", lambda x: np.std(x, ddof=1))]:
    dist = boot_stat(math, func)
    lo, hi = np.percentile(dist, [2.5, 97.5])
    rows.append({"statistic": name, "estimate": func(math),
                 "boot_SE": dist.std(ddof=1), "CI_lower": lo, "CI_upper": hi})
summary = pd.DataFrame(rows).round(3)
print("=== Bootstrap summary (math score, 95% percentile CI) ===")
print(summary, "\n")
summary.to_csv(out / "py_bootstrap_summary.csv", index=False)

# ---------- Q3: bootstrap the difference in means (male - female) ----------
male = df.loc[df["gender"] == "male", "math_score"].to_numpy()
female = df.loc[df["gender"] == "female", "math_score"].to_numpy()
diffs = np.array([
    rng.choice(male, len(male)).mean() - rng.choice(female, len(female)).mean()
    for _ in range(B)])
d_lo, d_hi = np.percentile(diffs, [2.5, 97.5])
print("=== Difference in mean math score (male - female) ===")
print(f"Observed difference     = {male.mean() - female.mean():.3f}")
print(f"Bootstrap SE            = {diffs.std(ddof=1):.4f}")
print(f"95% CI                  = ({d_lo:.3f}, {d_hi:.3f})")
print("CI excludes 0: the difference is significant at 5%"
      if d_lo > 0 or d_hi < 0 else "CI includes 0: no significant difference")

# ---------- Charts ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(boot_means, bins=40, edgecolor="black", alpha=0.7)
axes[0].axvline(math.mean(), color="red", linestyle="--", label="Sample mean")
axes[0].axvline(ci_pct[0], color="green", linestyle=":", label="95% CI")
axes[0].axvline(ci_pct[1], color="green", linestyle=":")
axes[0].set_title("Bootstrap sampling distribution of the mean")
axes[0].set_xlabel("Mean math score")
axes[0].set_ylabel("Frequency")
axes[0].legend()

axes[1].hist(diffs, bins=40, edgecolor="black", alpha=0.7, color="orange")
axes[1].axvline(0, color="red", linestyle="--", label="No difference")
axes[1].set_title("Bootstrap distribution: male - female mean")
axes[1].set_xlabel("Difference in mean math score")
axes[1].legend()
plt.tight_layout()
plt.savefig(out / "py_bootstrap_distributions.png", dpi=150)
plt.show()