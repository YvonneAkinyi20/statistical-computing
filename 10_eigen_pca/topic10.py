"""
Topic 10: Eigenvalues, eigenvectors and PCA
Libraries: numpy.linalg, sklearn.decomposition (pandas, matplotlib for support)

Outputs: eigenvalues, explained variance, scree plot and PCA component plots,
saved to the output folder.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

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
data = df[[MATH, READING, WRITING, GENDER]].dropna().reset_index(drop=True)
names = ["Math", "Reading", "Writing"]
X = data[[MATH, READING, WRITING]].to_numpy(dtype=float)

np.set_printoptions(precision=4, suppress=True, linewidth=120)

# ---------- Standardise the variables ----------
# PCA on standardised scores is the same as PCA on the correlation matrix.
Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

# ---------- Eigen-decomposition with numpy.linalg ----------
R = np.corrcoef(X, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(R)            # eigh: for symmetric matrices
order = np.argsort(eigvals)[::-1]               # largest eigenvalue first
eigvals, eigvecs = eigvals[order], eigvecs[:, order]

print("\nCorrelation matrix R\n")
print(pd.DataFrame(R, index=names, columns=names).round(3))

print("\nEigenvalues of R (variance carried by each component):", eigvals)
print("\nEigenvectors of R (columns = principal directions)\n")
pcs = [f"PC{i + 1}" for i in range(len(names))]
print(pd.DataFrame(eigvecs, index=names, columns=pcs).round(4))

# Verify the definition: R v = lambda v
v = eigvecs[:, 0]
print("\nCheck R @ v1 equals lambda1 * v1:", bool(np.allclose(R @ v, eigvals[0] * v)))
print("Sum of eigenvalues = trace of R =", round(eigvals.sum(), 4),
      "| product of eigenvalues = det R =", round(np.prod(eigvals), 4),
      "=", round(np.linalg.det(R), 4))

# ---------- Explained variance ----------
ratio = eigvals / eigvals.sum()
cumulative = np.cumsum(ratio)
variance_table = pd.DataFrame({
    "component": pcs,
    "eigenvalue": eigvals,
    "explained_variance_ratio": ratio,
    "cumulative_ratio": cumulative,
}).round(4)
variance_table.to_csv(OUT / "explained_variance.csv", index=False)
print("\nExplained variance\n")
print(variance_table.to_string(index=False))

# ---------- PCA with scikit-learn ----------
pca = PCA(n_components=3)
scores = pca.fit_transform(Z)
print("\nscikit-learn PCA explained variance ratio:", pca.explained_variance_ratio_)
print("Matches the eigenvalue results:",
      bool(np.allclose(pca.explained_variance_, eigvals)))

loadings = pd.DataFrame(pca.components_.T, index=names, columns=pcs).round(4)
loadings.to_csv(OUT / "pca_loadings.csv")
print("\nLoadings (weights of each score in each component)\n")
print(loadings)

scores_df = pd.DataFrame(scores, columns=pcs).round(4)
scores_df[GENDER] = data[GENDER]
scores_df.to_csv(OUT / "pca_scores.csv", index=False)

# ---------- Interpretation ----------
n_keep = int((eigvals > 1).sum())
print("\nInterpretation")
print(f"- PC1 explains {ratio[0] * 100:.1f}% of the total variance. All three loadings "
      f"have {'the same sign' if (np.sign(loadings['PC1']) == np.sign(loadings['PC1'].iloc[0])).all() else 'mixed signs'}, "
      "so PC1 acts as an overall academic performance index.")
print(f"- PC2 and PC3 add {ratio[1] * 100:.1f}% and {ratio[2] * 100:.1f}%. They describe "
      "contrasts between subjects: look at which loadings have opposite signs.")
print(f"- Kaiser rule (keep components with eigenvalue > 1): keep {n_keep} component(s).")
print(f"- The first two components together explain {cumulative[1] * 100:.1f}% of the variance.")
print("- The sign of a component is arbitrary; only the pattern of loadings matters.")

# ---------- Charts ----------
# Figure 1: scree plot and cumulative explained variance
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
ax1.plot(range(1, 4), eigvals, "o-", color="navy", lw=2)
ax1.axhline(1, color="red", linestyle="--", label="Kaiser line (eigenvalue = 1)")
ax1.set_xticks([1, 2, 3])
ax1.set_title("Scree plot")
ax1.set_xlabel("Principal component")
ax1.set_ylabel("Eigenvalue")
ax1.legend()

ax2.bar(pcs, ratio * 100, color="steelblue", label="Individual")
ax2.plot(pcs, cumulative * 100, "o-", color="red", label="Cumulative")
for i, c in enumerate(cumulative):
    ax2.text(i, c * 100 + 1.5, f"{c * 100:.1f}%", ha="center")
ax2.set_ylim(0, 110)
ax2.set_title("Explained variance")
ax2.set_ylabel("Percent of total variance")
ax2.legend()
fig1.tight_layout()
fig1.savefig(OUT / "scree_and_explained_variance.png", dpi=150)

# Figure 2: PCA component plot (scores) and biplot with loading arrows
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
for g, colour in zip(sorted(data[GENDER].unique()), ["tab:red", "tab:blue", "tab:green"]):
    mask = (data[GENDER] == g).to_numpy()
    ax1.scatter(scores[mask, 0], scores[mask, 1], alpha=0.5, s=14, color=colour, label=str(g))
ax1.axhline(0, color="grey", lw=0.8)
ax1.axvline(0, color="grey", lw=0.8)
ax1.set_title("Students in PC1 / PC2 space")
ax1.set_xlabel(f"PC1 ({ratio[0] * 100:.1f}%)")
ax1.set_ylabel(f"PC2 ({ratio[1] * 100:.1f}%)")
ax1.legend(title=GENDER)

ax2.scatter(scores[:, 0], scores[:, 1], alpha=0.15, s=10, color="grey")
scale = np.abs(scores[:, :2]).max() * 0.8
for name in names:
    lx, ly = loadings.loc[name, "PC1"] * scale, loadings.loc[name, "PC2"] * scale
    ax2.arrow(0, 0, lx, ly, color="red", head_width=0.12, length_includes_head=True)
    ax2.text(lx * 1.08, ly * 1.08, name, color="red", ha="center", va="center", fontsize=11)
ax2.axhline(0, color="grey", lw=0.8)
ax2.axvline(0, color="grey", lw=0.8)
ax2.set_title("Biplot: loading vectors")
ax2.set_xlabel(f"PC1 ({ratio[0] * 100:.1f}%)")
ax2.set_ylabel(f"PC2 ({ratio[1] * 100:.1f}%)")
fig2.tight_layout()
fig2.savefig(OUT / "pca_component_plot.png", dpi=150)

print(f"\nSaved results and charts to: {OUT}")
plt.show()