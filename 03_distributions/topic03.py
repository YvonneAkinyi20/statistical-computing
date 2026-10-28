from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------- Load data ----------
HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE.parent / "data" / "Cleaned_Students_Performance.csv")
df.columns = [c.replace(" ", "_").replace("/", "_") for c in df.columns]

# 0/1 codes inferred from the original data (1 = completed prep)
if pd.api.types.is_numeric_dtype(df["test_preparation_course"]):
    df["test_preparation_course"] = df["test_preparation_course"].map(
        {1: "completed", 0: "none"})

out = HERE / "output"
out.mkdir(exist_ok=True)

scores = ["math_score", "reading_score", "writing_score"]

# ---------- Q1: fit a normal distribution to each score ----------
rows = []
for col in scores:
    x = df[col]
    mu, sigma = stats.norm.fit(x)            # maximum likelihood estimates
    ks_stat, ks_p = stats.kstest(x, "norm", args=(mu, sigma))
    rows.append({"score": col, "mean_mu": mu, "std_sigma": sigma,
                 "skewness": x.skew(), "KS_statistic": ks_stat,
                 "KS_p_value": ks_p})
params = pd.DataFrame(rows).round(4)
print("=== Fitted normal parameters and goodness of fit ===")
print(params, "\n")
params.to_csv(out / "py_distribution_parameters.csv", index=False)

# ---------- Q2: probability values from the fitted normal (math) ----------
mu, sigma = stats.norm.fit(df["math_score"])
dist = stats.norm(mu, sigma)

p_ge_70 = 1 - dist.cdf(70)                   # P(X >= 70)
p_lt_50 = dist.cdf(50)                       # P(X < 50)
p_between = dist.cdf(70) - dist.cdf(50)      # P(50 <= X <= 70)
q90 = dist.ppf(0.90)                         # score that 90% fall below
z_80 = (80 - mu) / sigma                     # z-score of an 80

emp_ge_70 = (df["math_score"] >= 70).mean()
emp_lt_50 = (df["math_score"] < 50).mean()

print("=== Normal probabilities (math score) ===")
print(f"P(X >= 70)        theoretical = {p_ge_70:.4f}   observed = {emp_ge_70:.4f}")
print(f"P(X < 50)         theoretical = {p_lt_50:.4f}   observed = {emp_lt_50:.4f}")
print(f"P(50 <= X <= 70)  theoretical = {p_between:.4f}")
print(f"90th percentile   = {q90:.2f}")
print(f"z-score of 80     = {z_80:.2f}\n")

# ---------- Q3: histogram with fitted curve ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, col in zip(axes, scores):
    x = df[col]
    m, s = stats.norm.fit(x)
    ax.hist(x, bins=20, density=True, alpha=0.6, edgecolor="black")
    grid = np.linspace(x.min(), x.max(), 200)
    ax.plot(grid, stats.norm.pdf(grid, m, s), "r-", linewidth=2,
            label=f"Normal({m:.1f}, {s:.1f})")
    ax.set_title(col)
    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.legend()
plt.tight_layout()
plt.savefig(out / "py_histograms_fitted_normal.png", dpi=150)

# ---------- Q4: binomial distribution ----------
# If 10 students are chosen at random, how many completed the prep course?
p_prep = (df["test_preparation_course"] == "completed").mean()
n = 10
binom = stats.binom(n, p_prep)

print("=== Binomial: completed prep among 10 random students ===")
print(f"p (completed prep) = {p_prep:.4f}")
print(f"Expected number    = {binom.mean():.2f}")
print(f"Std deviation      = {binom.std():.2f}")
print(f"P(X = 4)           = {binom.pmf(4):.4f}")
print(f"P(X >= 5)          = {1 - binom.cdf(4):.4f}\n")

k = np.arange(0, n + 1)
plt.figure(figsize=(7, 4.5))
plt.bar(k, binom.pmf(k), edgecolor="black")
plt.xlabel("Number who completed prep (out of 10)")
plt.ylabel("Probability")
plt.title(f"Binomial(n={n}, p={p_prep:.2f})")
plt.tight_layout()
plt.savefig(out / "py_binomial_pmf.png", dpi=150)

plt.show()