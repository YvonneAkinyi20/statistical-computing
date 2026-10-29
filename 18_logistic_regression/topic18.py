"""
Topic 18: Logistic regression
Libraries: statsmodels, scikit-learn (pandas, numpy, matplotlib for support)

Outputs: logistic coefficients, odds ratios, predicted probabilities,
confusion matrix and classification performance, saved to the output folder.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
PASS_MARK = 50          # outcome: 1 = passed math (score >= PASS_MARK), 0 = failed

# ---------- Outcome and predictors ----------
data = df.dropna(subset=[MATH, READING, WRITING, GENDER]).copy()
y = (data[MATH] >= PASS_MARK).astype(int)

X = data[[READING, WRITING]].rename(columns={READING: "reading", WRITING: "writing"}).astype(float)
categorical = [GENDER]
for key in ("lunch", "prep"):
    col = find_col(key, required=False)
    if col:
        categorical.append(col)
dummies = pd.get_dummies(data[categorical].astype(str), drop_first=True, dtype=float)
X = pd.concat([X, dummies], axis=1)

print("\nMODEL SETUP")
print(f"Outcome: math score >= {PASS_MARK} (1 = pass, 0 = fail)")
print(f"Passed: {y.sum()} ({y.mean() * 100:.1f}%), failed: {(1 - y).sum()} ({(1 - y.mean()) * 100:.1f}%)")
print(f"Predictors: {list(X.columns)}")
print(f"A model that always predicts the majority class would be right "
      f"{max(y.mean(), 1 - y.mean()) * 100:.1f}% of the time (baseline accuracy).")

# =====================================================================
# 1. Logistic regression with statsmodels (coefficients and odds ratios)
# =====================================================================
Xc = sm.add_constant(X)
logit = sm.Logit(y, Xc).fit(disp=False)
print("\n1. LOGISTIC REGRESSION (statsmodels, all students)\n")
print(logit.summary())

ci = logit.conf_int()
coef_table = pd.DataFrame({
    "coefficient": logit.params,
    "std_error": logit.bse,
    "z_value": logit.tvalues,
    "p_value": logit.pvalues,
    "odds_ratio": np.exp(logit.params),
    "OR_ci_lower": np.exp(ci[0]),
    "OR_ci_upper": np.exp(ci[1]),
}).round(4)
coef_table.to_csv(OUT / "logistic_coefficients_odds_ratios.csv")
(OUT / "logistic_summary.txt").write_text(logit.summary().as_text())
print("\nCoefficients and odds ratios\n")
print(coef_table)

print(f"\nModel fit: McFadden pseudo R-squared = {logit.prsquared:.4f}, "
      f"likelihood-ratio test p = {logit.llr_pvalue:.4g}, AIC = {logit.aic:.2f}")

print("\nInterpretation of the odds ratios")
for name in X.columns:
    orr, p = float(np.exp(logit.params[name])), float(logit.pvalues[name])
    sig = "significant" if p < 0.05 else "not significant"
    if name in ("reading", "writing"):
        print(f"- {name}: each extra point multiplies the odds of passing by {orr:.3f} "
              f"(about {orr ** 10:.2f} times per 10 points), p = {p:.4g}, {sig}.")
    else:
        direction = "higher" if orr > 1 else "lower"
        print(f"- {name}: odds of passing are {abs(orr - 1) * 100:.0f}% {direction} than the "
              f"reference group (OR = {orr:.3f}), p = {p:.4g}, {sig}.")

# Multicollinearity check
vif = pd.DataFrame({"variable": Xc.columns,
                    "VIF": [variance_inflation_factor(Xc.to_numpy(), i) for i in range(Xc.shape[1])]})
vif = vif[vif["variable"] != "const"].round(3)
vif.to_csv(OUT / "vif.csv", index=False)
print("\nVariance inflation factors (above 5 to 10 suggests multicollinearity)\n")
print(vif.to_string(index=False))

# =====================================================================
# 2. Predicted probabilities
# =====================================================================
prob_all = logit.predict(Xc)
print("\n2. PREDICTED PROBABILITIES OF PASSING")
example = pd.DataFrame({"reading": [30, 40, 50, 60, 80], "writing": [30, 40, 50, 60, 80]})
for c in X.columns[2:]:
    example[c] = 0.0
example_prob = logit.predict(sm.add_constant(example[X.columns], has_constant="add"))
print("Reference student (all dummy variables = 0) with equal reading and writing scores:\n")
print(pd.DataFrame({"reading": example["reading"], "writing": example["writing"],
                    "P(pass)": example_prob.round(3)}).to_string(index=False))

# =====================================================================
# 3. Prediction with scikit-learn: train / test split
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
clf = LogisticRegression(C=1e6, max_iter=5000)          # almost no regularisation
clf.fit(X_train, y_train)
prob_test = clf.predict_proba(X_test)[:, 1]
pred_test = (prob_test >= 0.5).astype(int)

print("\n3. CLASSIFICATION PERFORMANCE (20% hold-out test set, threshold 0.5)")
cm = confusion_matrix(y_test, pred_test)
tn, fp, fn, tp = cm.ravel()
print("\nConfusion matrix (rows = actual, columns = predicted)\n")
print(pd.DataFrame(cm, index=["Actual fail", "Actual pass"], columns=["Predicted fail", "Predicted pass"]))
print(f"\nTrue negatives = {tn}, false positives = {fp}, false negatives = {fn}, true positives = {tp}")

metrics = {
    "accuracy": accuracy_score(y_test, pred_test),
    "precision": precision_score(y_test, pred_test, zero_division=0),
    "recall (sensitivity)": recall_score(y_test, pred_test, zero_division=0),
    "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
    "F1 score": f1_score(y_test, pred_test, zero_division=0),
    "ROC AUC": roc_auc_score(y_test, prob_test),
    "baseline accuracy (majority class)": max(y_test.mean(), 1 - y_test.mean()),
}
metrics_df = pd.DataFrame({"value": metrics}).round(4)
metrics_df.to_csv(OUT / "classification_metrics.csv")
print("\nMetrics\n")
print(metrics_df)
print("\nClassification report\n")
print(classification_report(y_test, pred_test, target_names=["fail", "pass"], zero_division=0))

cv_auc = cross_val_score(LogisticRegression(C=1e6, max_iter=5000), X, y, cv=5, scoring="roc_auc")
print(f"5-fold cross-validated ROC AUC: {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})")

# Effect of the decision threshold
thr_rows = []
for thr in (0.3, 0.5, 0.7, 0.85):
    pred = (prob_test >= thr).astype(int)
    thr_rows.append({"threshold": thr,
                     "accuracy": accuracy_score(y_test, pred),
                     "precision": precision_score(y_test, pred, zero_division=0),
                     "recall": recall_score(y_test, pred, zero_division=0),
                     "F1": f1_score(y_test, pred, zero_division=0)})
thr_df = pd.DataFrame(thr_rows).round(4)
thr_df.to_csv(OUT / "threshold_comparison.csv", index=False)
print("\nEffect of the decision threshold on the test set\n")
print(thr_df.to_string(index=False))

pred_table = X_test.copy()
pred_table["actual_pass"] = y_test.to_numpy()
pred_table["predicted_probability"] = prob_test.round(4)
pred_table["predicted_pass"] = pred_test
pred_table.to_csv(OUT / "test_set_predictions.csv", index=False)

# ---------- Interpretation ----------
print("\nInterpretation")
print("- Coefficients are on the log-odds scale; exponentiating them gives odds ratios. "
      "An odds ratio above 1 raises the chance of passing, below 1 lowers it.")
print(f"- Accuracy {metrics['accuracy'] * 100:.1f}% should be compared with the baseline of "
      f"{metrics['baseline accuracy (majority class)'] * 100:.1f}%. When one class is much larger, "
      "recall, precision and AUC give a fuller picture than accuracy.")
print(f"- Recall = {metrics['recall (sensitivity)']:.2f}: share of actual passers the model finds. "
      f"Specificity = {metrics['specificity']:.2f}: share of actual failers it identifies.")
print(f"- ROC AUC = {metrics['ROC AUC']:.2f}: the probability that a random passer gets a higher "
      "predicted probability than a random failer (0.5 = guessing, 1 = perfect).")
print("- Lowering the threshold finds more passers but lets more failers through; raising it does "
      "the opposite. The right choice depends on which error is costlier.")
print("- Reading and writing scores are correlated, so check the VIF before comparing their "
      "individual coefficients.")

# ---------- Charts ----------
# Figure 1: odds ratios (forest plot) and predicted probability curve
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
ors = coef_table.drop(index="const")
ypos = np.arange(len(ors))
ax1.errorbar(ors["odds_ratio"], ypos,
             xerr=[ors["odds_ratio"] - ors["OR_ci_lower"], ors["OR_ci_upper"] - ors["odds_ratio"]],
             fmt="o", color="navy", capsize=5)
ax1.axvline(1, color="red", linestyle="--", label="OR = 1 (no effect)")
ax1.set_yticks(ypos)
ax1.set_yticklabels(ors.index)
ax1.set_xscale("log")
ax1.set_xlabel("Odds ratio (log scale) with 95% CI")
ax1.set_title("Odds ratios for passing math")
ax1.legend()

grid = np.linspace(data[READING].min(), data[READING].max(), 200)
curve = pd.DataFrame({"reading": grid, "writing": X["writing"].mean()})
for c in X.columns[2:]:
    curve[c] = 0.0
ax2.plot(grid, logit.predict(sm.add_constant(curve[X.columns], has_constant="add")),
         color="darkgreen", lw=2)
ax2.axhline(0.5, color="grey", linestyle=":")
ax2.set_title("Predicted probability of passing vs reading score\n"
              "(writing held at its mean, other variables at the reference level)")
ax2.set_xlabel("Reading score")
ax2.set_ylabel("P(pass)")
ax2.set_ylim(0, 1.02)
fig1.tight_layout()
fig1.savefig(OUT / "odds_ratios_and_probability_curve.png", dpi=150)

# Figure 2: confusion matrix, ROC curve and probability distributions
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))
ConfusionMatrixDisplay(cm, display_labels=["Fail", "Pass"]).plot(ax=ax1, cmap="Blues", colorbar=False)
ax1.set_title("Confusion matrix (test set, threshold 0.5)")

fpr, tpr, _ = roc_curve(y_test, prob_test)
ax2.plot(fpr, tpr, color="navy", lw=2, label=f"AUC = {metrics['ROC AUC']:.3f}")
ax2.plot([0, 1], [0, 1], "r--", label="Random guessing")
ax2.set_xlabel("False positive rate (1 - specificity)")
ax2.set_ylabel("True positive rate (recall)")
ax2.set_title("ROC curve")
ax2.legend()

ax3.hist(prob_test[y_test.to_numpy() == 1], bins=20, alpha=0.6, color="seagreen", label="Actually passed")
ax3.hist(prob_test[y_test.to_numpy() == 0], bins=20, alpha=0.6, color="tomato", label="Actually failed")
ax3.axvline(0.5, color="black", linestyle="--", label="Threshold 0.5")
ax3.set_xlabel("Predicted probability of passing")
ax3.set_ylabel("Number of students")
ax3.set_title("Predicted probabilities by actual outcome")
ax3.legend()
fig2.tight_layout()
fig2.savefig(OUT / "classification_performance.png", dpi=150)

print(f"\nSaved results and charts to: {OUT}")
plt.show()