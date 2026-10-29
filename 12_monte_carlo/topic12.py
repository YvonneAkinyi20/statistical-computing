"""
Topic 12: Random numbers and Monte Carlo methods
Libraries: numpy.random, scipy.stats (pandas, matplotlib for support)

Outputs: random samples, simulated estimates, Monte Carlo approximations and
convergence plots, saved to the output folder.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special, stats

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


math = df[find_col("math")].dropna().to_numpy(dtype=float)
THRESHOLD = 80          # we estimate P(Math score >= THRESHOLD)
N = 100_000             # Monte Carlo sample size
SEED = 42
rng = np.random.default_rng(SEED)
summary = []            # rows for the results table


def add(name, estimate, exact=None):
    row = {"quantity": name, "monte_carlo_estimate": estimate}
    if exact is not None:
        row["exact_or_reference"] = exact
        row["absolute_error"] = abs(estimate - exact)
    summary.append(row)


# =====================================================================
# 1. Random numbers: generation and reproducibility
# =====================================================================
print("\n1. RANDOM NUMBERS")
a = np.random.default_rng(SEED).random(3)
b = np.random.default_rng(SEED).random(3)
print(f"Same seed gives the same numbers: {a.round(4)} and {b.round(4)} -> {bool(np.allclose(a, b))}")

samples = {
    "Uniform(0, 1)": rng.random(N),
    "Normal(65, 15)": rng.normal(65, 15, N),
    "Exponential(mean 10)": rng.exponential(10, N),
}
sample_table = pd.DataFrame({
    name: {"mean": s.mean(), "sd": s.std(ddof=1), "min": s.min(), "max": s.max()}
    for name, s in samples.items()}).T.round(3)
sample_table.to_csv(OUT / "random_samples_summary.csv")
print(f"\nSummary of {N:,} random samples\n")
print(sample_table)
print("Theory: Uniform mean 0.5, sd 0.289 | Normal mean 65, sd 15 | Exponential mean 10, sd 10")

# =====================================================================
# 2. Monte Carlo approximation of pi
# =====================================================================
print("\n2. MONTE CARLO APPROXIMATION OF PI")
px, py = rng.random(N), rng.random(N)
inside = (px**2 + py**2) <= 1
pi_running = 4 * np.cumsum(inside) / np.arange(1, N + 1)
pi_hat = pi_running[-1]
print(f"Estimate with {N:,} points: {pi_hat:.5f} (true value {np.pi:.5f}, "
      f"error {abs(pi_hat - np.pi):.5f})")
add("pi", pi_hat, np.pi)

# =====================================================================
# 3. Monte Carlo integration: integral of exp(-x^2) from 0 to 1
# =====================================================================
print("\n3. MONTE CARLO INTEGRATION of exp(-x^2) on [0, 1]")
u = rng.random(N)
f = np.exp(-u**2)
int_running = np.cumsum(f) / np.arange(1, N + 1)
int_hat = int_running[-1]
int_exact = np.sqrt(np.pi) / 2 * special.erf(1)
int_se = f.std(ddof=1) / np.sqrt(N)
print(f"Estimate: {int_hat:.5f} +/- {1.96 * int_se:.5f} (95%), exact value {int_exact:.5f}")
add("integral of exp(-x^2) on [0,1]", int_hat, int_exact)

# =====================================================================
# 4. Simulated estimate of a probability from the student data
# =====================================================================
mu, sigma = math.mean(), math.std(ddof=1)
print(f"\n4. SIMULATING STUDENT SCORES FROM Normal(mean = {mu:.2f}, sd = {sigma:.2f})")
sim_scores = rng.normal(mu, sigma, N)
hits = sim_scores >= THRESHOLD
prob_running = np.cumsum(hits) / np.arange(1, N + 1)
prob_hat = prob_running[-1]
prob_exact = stats.norm.sf(THRESHOLD, mu, sigma)
prob_empirical = np.mean(math >= THRESHOLD)
prob_se = np.sqrt(prob_hat * (1 - prob_hat) / N)
print(f"P(score >= {THRESHOLD}): Monte Carlo = {prob_hat:.5f} +/- {1.96 * prob_se:.5f}, "
      f"exact normal = {prob_exact:.5f}, observed proportion in the data = {prob_empirical:.5f}")
add(f"P(Math >= {THRESHOLD}) under the fitted normal", prob_hat, prob_exact)
add(f"P(Math >= {THRESHOLD}) observed in the data", prob_empirical)

# =====================================================================
# 5. Bootstrap simulation of the sampling distribution of the mean
# =====================================================================
print("\n5. BOOTSTRAP (resampling the real scores)")
B = 10_000
boot_means = rng.choice(math, size=(B, len(math)), replace=True).mean(axis=1)
boot_se = boot_means.std(ddof=1)
boot_ci = np.percentile(boot_means, [2.5, 97.5])
t_se = stats.sem(math)
t_ci = stats.t.interval(0.95, len(math) - 1, loc=mu, scale=t_se)
print(f"Sample mean = {mu:.3f}")
print(f"Bootstrap SE = {boot_se:.4f} | theoretical SE = {t_se:.4f}")
print(f"Bootstrap 95% CI = ({boot_ci[0]:.3f}, {boot_ci[1]:.3f}) | t-interval = "
      f"({t_ci[0]:.3f}, {t_ci[1]:.3f})")
add("Mean Math score (bootstrap mean of means)", boot_means.mean(), mu)

results = pd.DataFrame(summary).round(5)
results.to_csv(OUT / "monte_carlo_estimates.csv", index=False)
print("\nSummary of simulated estimates\n")
print(results.to_string(index=False))

# ---------- Interpretation ----------
print("\nInterpretation")
print("- Monte Carlo estimates get closer to the true value as the number of "
      "simulations grows; the error shrinks in proportion to 1/sqrt(n).")
print("- A fixed seed makes a simulation reproducible; changing it gives a slightly "
      "different estimate within the expected Monte Carlo error.")
print("- The bootstrap needs no formula: resampling the data shows how much the mean "
      "varies, and agrees with the theoretical standard error.")

# ---------- Charts ----------
# Figure 1: random samples
fig1, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, (name, s), colour in zip(axes, samples.items(), ["tab:blue", "tab:green", "tab:orange"]):
    ax.hist(s, bins=50, density=True, color=colour, edgecolor="white")
    ax.set_title(f"{name} ({N:,} draws)")
    ax.set_xlabel("Value")
axes[0].set_ylabel("Density")
fig1.tight_layout()
fig1.savefig(OUT / "random_samples.png", dpi=150)

# Figure 2: pi points and the bootstrap distribution
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
m = 3000
ax1.scatter(px[:m][inside[:m]], py[:m][inside[:m]], s=4, color="tab:blue", label="Inside circle")
ax1.scatter(px[:m][~inside[:m]], py[:m][~inside[:m]], s=4, color="tab:red", label="Outside circle")
theta = np.linspace(0, np.pi / 2, 200)
ax1.plot(np.cos(theta), np.sin(theta), "k-", lw=1)
ax1.set_aspect("equal")
ax1.set_title(f"Monte Carlo estimate of pi ({m:,} of {N:,} points shown)")
ax1.legend(loc="upper right", fontsize=8)

ax2.hist(boot_means, bins=50, color="seagreen", edgecolor="white")
ax2.axvline(mu, color="black", lw=2, label=f"Sample mean = {mu:.2f}")
ax2.axvline(boot_ci[0], color="red", linestyle="--", label="Bootstrap 95% CI")
ax2.axvline(boot_ci[1], color="red", linestyle="--")
ax2.set_title(f"Bootstrap distribution of the mean ({B:,} resamples)")
ax2.set_xlabel("Mean Math score")
ax2.set_ylabel("Frequency")
ax2.legend()
fig2.tight_layout()
fig2.savefig(OUT / "monte_carlo_pi_and_bootstrap.png", dpi=150)

# Figure 3: convergence plots
n_axis = np.arange(1, N + 1)
fig3, axes = plt.subplots(2, 2, figsize=(13, 9))
for ax, running, truth, title in [
    (axes[0, 0], pi_running, np.pi, "Convergence: estimate of pi"),
    (axes[0, 1], int_running, int_exact, "Convergence: integral of exp(-x^2)"),
    (axes[1, 0], prob_running, prob_exact, f"Convergence: P(Math >= {THRESHOLD})"),
]:
    ax.semilogx(n_axis, running, color="navy", lw=1)
    ax.axhline(truth, color="red", linestyle="--", label=f"True value = {truth:.4f}")
    ax.set_title(title)
    ax.set_xlabel("Number of simulations (log scale)")
    ax.set_ylabel("Running estimate")
    ax.legend()

ax = axes[1, 1]
for running, truth, name in [(pi_running, np.pi, "pi"),
                             (int_running, int_exact, "integral"),
                             (prob_running, prob_exact, "probability")]:
    ax.loglog(n_axis, np.abs(running - truth) + 1e-12, lw=0.6, alpha=0.8, label=name)
ref = 1 / np.sqrt(n_axis)
ax.loglog(n_axis, ref, "k--", lw=1.5, label=r"$1/\sqrt{n}$ reference")
ax.set_title("Absolute error versus number of simulations")
ax.set_xlabel("Number of simulations")
ax.set_ylabel("Absolute error")
ax.legend()
fig3.tight_layout()
fig3.savefig(OUT / "convergence_plots.png", dpi=150)

print(f"\nSaved results and charts to: {OUT}")
plt.show()