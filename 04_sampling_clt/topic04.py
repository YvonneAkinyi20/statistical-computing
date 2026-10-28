"""
Topic 04: Sampling distributions and the Central Limit Theorem
Libraries: numpy, scipy, matplotlib (pandas only to load the CSV)

The students in the dataset are treated as the POPULATION. We repeatedly
draw random samples from it and study the distribution of the sample means.
"""
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
    """Find the column whose name contains the keyword (e.g. 'math')."""
    for c in df.columns:
        if keyword in c:
            return c
    raise KeyError(f"No column containing '{keyword}'. Columns: {list(df.columns)}")


SUBJECTS = {
    "Math": find_col("math"),
    "Reading": find_col("reading"),
    "Writing": find_col("writing"),
}
MAIN_SUBJECT = "Math"          # subject used for the charts
SAMPLE_SIZES = [5, 30, 100]    # sample sizes compared
N_SAMPLES = 1000               # number of samples drawn for each size
rng = np.random.default_rng(42)


def sample_means(values, n, reps=N_SAMPLES):
    """Draw `reps` random samples of size n and return their means."""
    return rng.choice(values, size=(reps, n), replace=True).mean(axis=1)


# ---------- Q4a: Sampling distribution of the mean ----------
rows = []
for subject, col in SUBJECTS.items():
    x = df[col].dropna().to_numpy(dtype=float)
    mu, sigma = x.mean(), x.std(ddof=0)
    for n in SAMPLE_SIZES:
        means = sample_means(x, n)
        rows.append({
            "subject": subject,
            "sample_size": n,
            "population_mean": mu,
            "mean_of_sample_means": means.mean(),
            "population_sd": sigma,
            "theoretical_SE": sigma / np.sqrt(n),
            "empirical_SE": means.std(ddof=1),
            "skewness_of_means": stats.skew(means),
            "shapiro_p_value": stats.shapiro(means).pvalue,
        })

results = pd.DataFrame(rows).round(4)
results.to_csv(OUT / "sampling_distribution_summary.csv", index=False)
print("\nSampling distribution of the mean "
      f"({N_SAMPLES} samples per size)\n")
print(results.to_string(index=False))

# ---------- Q4b: Standard error versus sample size ----------
x = df[SUBJECTS[MAIN_SUBJECT]].dropna().to_numpy(dtype=float)
mu, sigma = x.mean(), x.std(ddof=0)
skew_pop = stats.skew(x)
print(f"\n{MAIN_SUBJECT}: population mean = {mu:.2f}, sd = {sigma:.2f}, "
      f"skewness = {skew_pop:.3f}")

ns = [2, 5, 10, 20, 30, 50, 100, 200]
empirical_se = [sample_means(x, n).std(ddof=1) for n in ns]
theoretical_se = [sigma / np.sqrt(n) for n in ns]

# Figure 1: population distribution and SE curve
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
ax1.hist(x, bins=20, color="steelblue", edgecolor="white")
ax1.axvline(mu, color="red", linestyle="--", label=f"Mean = {mu:.1f}")
ax1.set_title(f"Population distribution: {MAIN_SUBJECT} score")
ax1.set_xlabel("Score")
ax1.set_ylabel("Frequency")
ax1.legend()

ax2.plot(ns, theoretical_se, "o-", label=r"Theoretical  $\sigma/\sqrt{n}$")
ax2.plot(ns, empirical_se, "s--", label="Empirical (simulated)")
ax2.set_title("Standard error decreases as n grows")
ax2.set_xlabel("Sample size (n)")
ax2.set_ylabel("Standard error of the mean")
ax2.legend()
fig1.tight_layout()
fig1.savefig(OUT / "population_and_standard_error.png", dpi=150)

# ---------- Q4c: CLT visualization ----------
fig2, axes = plt.subplots(1, len(SAMPLE_SIZES), figsize=(14, 4.5), sharex=True)
for ax, n in zip(axes, SAMPLE_SIZES):
    means = sample_means(x, n)
    se = sigma / np.sqrt(n)
    ax.hist(means, bins=30, density=True, color="seagreen",
            edgecolor="white", alpha=0.8, label="Sample means")
    grid = np.linspace(means.min(), means.max(), 200)
    ax.plot(grid, stats.norm.pdf(grid, mu, se), "r-", lw=2,
            label=f"Normal(mean, SE = {se:.2f})")
    ax.axvline(mu, color="black", linestyle="--", lw=1)
    ax.set_title(f"n = {n}")
    ax.set_xlabel("Sample mean")
    ax.legend(fontsize=8)
axes[0].set_ylabel("Density")
fig2.suptitle(f"Central Limit Theorem: sampling distribution of the mean "
              f"({MAIN_SUBJECT} score, {N_SAMPLES} samples each)")
fig2.tight_layout()
fig2.savefig(OUT / "clt_sampling_distributions.png", dpi=150)

print(f"\nSaved results and charts to: {OUT}")
plt.show()