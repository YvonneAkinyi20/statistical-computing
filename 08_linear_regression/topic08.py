"""
Topic 08: Linear regression and least squares
Libraries: statsmodels, scikit-learn, numpy (pandas, scipy, matplotlib for support)

Outputs: regression equation, coefficients, p-values, R-squared, residuals
and a fitted regression plot, saved to the output folder.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

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
data = df[[MATH, READING, WRITING, GENDER]].dropna().copy()

# ---------- Model 1: simple linear regression (Math ~ Reading) ----------
x = data[READING].to_numpy(float)
y = data[MATH].to_numpy(float)

# Least squares by hand: slope = Sxy / Sxx, intercept = ybar - slope * xbar
slope = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
intercept = y.mean() - slope * x.mean()
print("\nMODEL 1: simple linear regression, Math ~ Reading")
print(f"Least squares by hand: intercept = {intercept:.4f}, slope = {slope:.4f}")

# Same model with statsmodels (gives p-values, R-squared, confidence intervals)
X1 = sm.add_constant(pd.Series(x, name="Reading"))
model1 = sm.OLS(y, X1).fit()
print(model1.summary())

print("\nRegression equation:")
print(f"  Math = {model1.params['const']:.3f} + {model1.params['Reading']:.3f} x Reading")

coef1 = pd.DataFrame({
    "coefficient": model1.params,
    "std_error": model1.bse,
    "t_value": model1.tvalues,
    "p_value": model1.pvalues,
    "ci_lower": model1.conf_int()[0],
    "ci_upper": model1.conf_int()[1],
}).round(4)
coef1.to_csv(OUT / "simple_regression_coefficients.csv")
(OUT / "simple_regression_summary.txt").write_text(model1.summary().as_text())

print(f"\nR-squared = {model1.rsquared:.4f}, adjusted R-squared = {model1.rsquared_adj:.4f}")
print(f"F-statistic = {model1.fvalue:.2f}, p-value = {model1.f_pvalue:.4g}")

# ---------- Cross-check with scikit-learn ----------
sk = LinearRegression().fit(x.reshape(-1, 1), y)
print(f"\nscikit-learn check: intercept = {sk.intercept_:.4f}, slope = {sk.coef_[0]:.4f}, "
      f"R-squared = {sk.score(x.reshape(-1, 1), y):.4f}")

# ---------- Model 2: multiple regression (Math ~ Reading + Writing + gender) ----------
dummies = pd.get_dummies(data[GENDER], drop_first=True, dtype=float)
dummies.columns = [f"gender_{c}" for c in dummies.columns]
X2 = sm.add_constant(pd.concat([data[[READING, WRITING]].rename(
    columns={READING: "Reading", WRITING: "Writing"}), dummies], axis=1))
model2 = sm.OLS(y, X2).fit()
print("\nMODEL 2: multiple regression, Math ~ Reading + Writing + gender")
print(model2.summary())

terms = " ".join(f"{'+' if v >= 0 else '-'} {abs(v):.3f} x {k}"
                 for k, v in model2.params.items() if k != "const")
print(f"\nRegression equation:\n  Math = {model2.params['const']:.3f} {terms}")

coef2 = pd.DataFrame({
    "coefficient": model2.params, "std_error": model2.bse,
    "t_value": model2.tvalues, "p_value": model2.pvalues,
    "ci_lower": model2.conf_int()[0], "ci_upper": model2.conf_int()[1],
}).round(4)
coef2.to_csv(OUT / "multiple_regression_coefficients.csv")
(OUT / "multiple_regression_summary.txt").write_text(model2.summary().as_text())

# ---------- Predictive check: train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X2.drop(columns="const"), y, test_size=0.2, random_state=42)
lr = LinearRegression().fit(X_train, y_train)
pred = lr.predict(X_test)
rmse = float(np.sqrt(np.mean((y_test - pred) ** 2)))
print(f"\nHold-out test (20%): R-squared = {r2_score(y_test, pred):.4f}, RMSE = {rmse:.2f} points")

# ---------- Model comparison ----------
comparison = pd.DataFrame({
    "model": ["Simple: Math ~ Reading", "Multiple: Math ~ Reading + Writing + gender"],
    "R_squared": [model1.rsquared, model2.rsquared],
    "adj_R_squared": [model1.rsquared_adj, model2.rsquared_adj],
    "AIC": [model1.aic, model2.aic],
}).round(4)
comparison.to_csv(OUT / "model_comparison.csv", index=False)
print("\nModel comparison\n")
print(comparison.to_string(index=False))

# ---------- Interpretation ----------
sig = "statistically significant" if model1.pvalues["Reading"] < 0.05 else "not significant"
print("\nInterpretation")
print(f"- Each extra point in reading is associated with {slope:.2f} extra points in math "
      f"(p = {model1.pvalues['Reading']:.4g}, {sig}).")
print(f"- R-squared = {model1.rsquared:.2f}: reading explains about "
      f"{model1.rsquared * 100:.0f}% of the variation in math scores.")
print(f"- Adding writing and gender changes R-squared to {model2.rsquared:.2f}.")
print("- Coefficients in the multiple model are 'holding the other variables constant'.")
print("- Regression shows association, not causation.")

# ---------- Residual diagnostics ----------
resid = model1.resid
fitted = model1.fittedvalues
sh_stat, sh_p = stats.shapiro(resid)
print(f"\nResiduals: mean = {resid.mean():.4f}, sd = {resid.std(ddof=2):.2f}, "
      f"Shapiro-Wilk p = {sh_p:.4f}")

# ---------- Charts ----------
# Figure 1: fitted regression plot with confidence band
xs = np.linspace(x.min(), x.max(), 100)
pred_df = model1.get_prediction(sm.add_constant(pd.Series(xs, name="Reading"))).summary_frame(alpha=0.05)
fig1, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(x, y, alpha=0.4, s=14, color="steelblue", label="Students")
ax.plot(xs, pred_df["mean"], color="red", lw=2,
        label=f"Math = {intercept:.1f} + {slope:.2f} x Reading")
ax.fill_between(xs, pred_df["mean_ci_lower"], pred_df["mean_ci_upper"],
                color="red", alpha=0.2, label="95% CI of the mean")
ax.set_title(f"Fitted regression line (R-squared = {model1.rsquared:.2f})")
ax.set_xlabel("Reading score")
ax.set_ylabel("Math score")
ax.legend()
fig1.tight_layout()
fig1.savefig(OUT / "fitted_regression_plot.png", dpi=150)

# Figure 2: residual diagnostics
fig2, axes = plt.subplots(1, 3, figsize=(15, 4.8))
axes[0].scatter(fitted, resid, alpha=0.4, s=14, color="steelblue")
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_title("Residuals vs fitted values")
axes[0].set_xlabel("Fitted values")
axes[0].set_ylabel("Residual")
axes[1].hist(resid, bins=25, color="seagreen", edgecolor="white")
axes[1].set_title("Distribution of residuals")
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Frequency")
stats.probplot(resid, dist="norm", plot=axes[2])
axes[2].set_title("Normal Q-Q plot of residuals")
fig2.tight_layout()
fig2.savefig(OUT / "residual_plots.png", dpi=150)

print(f"\nSaved results and charts to: {OUT}")
plt.show()
