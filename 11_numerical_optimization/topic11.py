"""
Topic 11: Numerical optimization
Libraries: scipy.optimize (numpy, pandas, matplotlib for support)

Two problems are solved numerically with the student scores:
  A. Least squares regression (Math ~ Reading) by minimising the mean squared error
  B. Maximum likelihood estimation of a normal distribution for the Math scores

Outputs: objective function, estimated optimum, convergence results and
optimization plots, saved to the output folder.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats

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


MATH, READING = find_col("math"), find_col("reading")
data = df[[MATH, READING]].dropna()
x = data[READING].to_numpy(dtype=float)
y = data[MATH].to_numpy(dtype=float)

# =====================================================================
# PROBLEM A: least squares by numerical optimization
# =====================================================================
# Objective function: f(b0, b1) = mean( (y - b0 - b1 * x)^2 )
def mse(params):
    b0, b1 = params
    return np.mean((y - b0 - b1 * x) ** 2)


def mse_grad(params):
    b0, b1 = params
    r = y - b0 - b1 * x
    return np.array([-2 * np.mean(r), -2 * np.mean(r * x)])


start = np.array([0.0, 0.0])
print("\nPROBLEM A: minimise the mean squared error of Math = b0 + b1 x Reading")
print(f"Objective at the starting point {start}: {mse(start):.2f}")

# Exact answer from the closed-form least squares formulas (for comparison)
b1_exact = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
b0_exact = y.mean() - b1_exact * x.mean()
f_exact = mse([b0_exact, b1_exact])
print(f"Closed-form solution: b0 = {b0_exact:.4f}, b1 = {b1_exact:.4f}, MSE = {f_exact:.4f}")

methods = {
    "BFGS": dict(method="BFGS", jac=mse_grad),
    "CG": dict(method="CG", jac=mse_grad),
    "L-BFGS-B": dict(method="L-BFGS-B", jac=mse_grad),
    "Nelder-Mead": dict(method="Nelder-Mead",
                        options=dict(maxiter=5000, xatol=1e-8, fatol=1e-10)),
}

paths, rows = {}, []
for name, kwargs in methods.items():
    path = [start.copy()]

    def callback(xk, path=path):
        path.append(np.array(xk, dtype=float))

    res = optimize.minimize(mse, start, callback=callback, **kwargs)
    paths[name] = np.array(path)
    rows.append({
        "method": name,
        "intercept_b0": res.x[0],
        "slope_b1": res.x[1],
        "final_MSE": res.fun,
        "iterations": res.nit,
        "function_evaluations": res.nfev,
        "converged": bool(res.success),
    })

# Gradient descent written by hand (on a standardised predictor so it behaves well)
mx, sx = x.mean(), x.std(ddof=1)
xz = (x - mx) / sx


def gd_objective(p):
    return np.mean((y - p[0] - p[1] * xz) ** 2)


p = np.array([0.0, 0.0])
lr, gd_values = 0.1, [mse(start)]
gd_path = [start.copy()]
for it in range(1, 501):
    r = y - p[0] - p[1] * xz
    grad = np.array([-2 * np.mean(r), -2 * np.mean(r * xz)])
    p = p - lr * grad
    b1_gd = p[1] / sx
    b0_gd = p[0] - b1_gd * mx
    gd_values.append(mse([b0_gd, b1_gd]))
    gd_path.append(np.array([b0_gd, b1_gd]))
    if np.linalg.norm(grad) < 1e-8:
        break
paths["Gradient descent"] = np.array(gd_path)
rows.append({"method": "Gradient descent (by hand)", "intercept_b0": b0_gd,
             "slope_b1": b1_gd, "final_MSE": gd_values[-1], "iterations": it,
             "function_evaluations": it, "converged": bool(it < 500)})

results_a = pd.DataFrame(rows)
results_a["error_vs_closed_form"] = (results_a["slope_b1"] - b1_exact).abs()
results_a = results_a.round(6)
results_a.to_csv(OUT / "least_squares_optimization.csv", index=False)
print("\nEstimated optimum and convergence for each method\n")
print(results_a.to_string(index=False))
best = results_a.loc[results_a["final_MSE"].idxmin()]
print(f"\nEstimated optimum: Math = {best['intercept_b0']:.3f} + {best['slope_b1']:.3f} x Reading "
      f"(MSE = {best['final_MSE']:.4f})")

# =====================================================================
# PROBLEM B: maximum likelihood estimation for a normal distribution
# =====================================================================
scores = data[MATH].to_numpy(dtype=float)


def neg_log_likelihood(params):
    mu, sigma = params
    return -np.sum(stats.norm.logpdf(scores, loc=mu, scale=sigma))


print("\nPROBLEM B: maximum likelihood estimation of Math ~ Normal(mu, sigma)")
mle_start = np.array([40.0, 5.0])
mle = optimize.minimize(neg_log_likelihood, mle_start, method="L-BFGS-B",
                        bounds=[(None, None), (1e-6, None)])
mle_nm = optimize.minimize(neg_log_likelihood, mle_start, method="Nelder-Mead")
mu_hat, sigma_hat = mle.x
print(f"Objective (negative log-likelihood) at the start: {neg_log_likelihood(mle_start):.2f}")
print(f"L-BFGS-B   : mu = {mu_hat:.4f}, sigma = {sigma_hat:.4f}, "
      f"-logL = {mle.fun:.2f}, converged = {mle.success}")
print(f"Nelder-Mead: mu = {mle_nm.x[0]:.4f}, sigma = {mle_nm.x[1]:.4f}, "
      f"-logL = {mle_nm.fun:.2f}, converged = {mle_nm.success}")
print(f"Analytic MLE: mu = sample mean = {scores.mean():.4f}, "
      f"sigma = sd with n in the denominator = {scores.std(ddof=0):.4f}")

# One-dimensional search: profile of the likelihood over mu with sigma fixed
sigma_fixed = scores.std(ddof=0)
scalar = optimize.minimize_scalar(lambda m: neg_log_likelihood([m, sigma_fixed]),
                                  bounds=(scores.min(), scores.max()), method="bounded")
print(f"minimize_scalar (sigma fixed): mu = {scalar.x:.4f}")

pd.DataFrame({
    "method": ["L-BFGS-B", "Nelder-Mead", "Analytic"],
    "mu": [mu_hat, mle_nm.x[0], scores.mean()],
    "sigma": [sigma_hat, mle_nm.x[1], scores.std(ddof=0)],
    "neg_log_likelihood": [mle.fun, mle_nm.fun, neg_log_likelihood([scores.mean(), scores.std(ddof=0)])],
    "iterations": [mle.nit, mle_nm.nit, np.nan],
}).round(4).to_csv(OUT / "mle_optimization.csv", index=False)

# ---------- Interpretation ----------
print("\nInterpretation")
print("- All gradient-based methods and Nelder-Mead reach the same minimum as the "
      "closed-form least squares solution, so the numerical optimum is correct.")
print("- Methods that use the gradient (BFGS, L-BFGS-B, CG) need far fewer function "
      "evaluations than the derivative-free Nelder-Mead.")
print("- Gradient descent converges steadily but more slowly, and depends on the "
      "learning rate and on the scaling of the predictor.")
print("- Maximum likelihood by optimization recovers the sample mean and the "
      "standard deviation with n in the denominator.")

# ---------- Charts ----------
# Figure 1: objective surface with optimizer paths
all_pts = np.vstack(list(paths.values()) + [[b0_exact, b1_exact]])
b0_lo, b0_hi = all_pts[:, 0].min(), all_pts[:, 0].max()
b1_lo, b1_hi = all_pts[:, 1].min(), all_pts[:, 1].max()
pad0, pad1 = 0.1 * (b0_hi - b0_lo) + 1, 0.1 * (b1_hi - b1_lo) + 0.05
B0, B1 = np.meshgrid(np.linspace(b0_lo - pad0, b0_hi + pad0, 120),
                     np.linspace(b1_lo - pad1, b1_hi + pad1, 120))
Z = np.array([[mse([a, b]) for a, b in zip(r0, r1)] for r0, r1 in zip(B0, B1)])

fig1, ax = plt.subplots(figsize=(9, 6))
cs = ax.contour(B0, B1, Z, levels=np.geomspace(max(f_exact, 1e-3) * 1.05, Z.max(), 25),
                cmap="viridis")
ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
for name, pts in paths.items():
    ax.plot(pts[:, 0], pts[:, 1], "o-", ms=3, lw=1.3, label=f"{name} ({len(pts) - 1} steps)")
ax.plot(b0_exact, b1_exact, "r*", ms=16, label="Optimum")
ax.set_title("Objective function (MSE) contours and optimizer paths")
ax.set_xlabel("Intercept b0")
ax.set_ylabel("Slope b1")
ax.legend(fontsize=8)
fig1.tight_layout()
fig1.savefig(OUT / "optimization_paths.png", dpi=150)

# Figure 2: convergence plot and likelihood curve
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for name, pts in paths.items():
    vals = np.array([mse(pt) for pt in pts]) - f_exact
    ax1.semilogy(np.maximum(vals, 1e-12), label=name)
ax1.set_title("Convergence: distance from the minimum MSE")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("MSE - minimum MSE (log scale)")
ax1.legend()

mus = np.linspace(scores.mean() - 15, scores.mean() + 15, 300)
nll = [neg_log_likelihood([m, sigma_fixed]) for m in mus]
ax2.plot(mus, nll, color="navy")
ax2.axvline(scalar.x, color="red", linestyle="--", label=f"Optimum mu = {scalar.x:.2f}")
ax2.set_title("Negative log-likelihood for mu (sigma fixed)")
ax2.set_xlabel("mu")
ax2.set_ylabel("Negative log-likelihood")
ax2.legend()
fig2.tight_layout()
fig2.savefig(OUT / "convergence_and_likelihood.png", dpi=150)

print(f"\nSaved results and charts to: {OUT}")
plt.show()