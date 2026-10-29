from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.multivariate.manova import MANOVA

# ---------- Load data ----------
HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE.parent / "data" / "Cleaned_Students_Performance.csv")
df.columns = [c.replace(" ", "_").replace("/", "_") for c in df.columns]

# 0/1 codes inferred from the original data
if pd.api.types.is_numeric_dtype(df["gender"]):
    df["gender"] = df["gender"].map({0: "female", 1: "male"})

out = HERE / "output"
out.mkdir(exist_ok=True)

# total_score and average_score are sums/means of these three, so we leave
# them out (they would be perfectly collinear with the three scores)
scores = ["math_score", "reading_score", "writing_score"]
X = df[scores]

# ---------- Q1: multivariate summaries ----------
mean_vec = X.mean()
cov = X.cov()
corr = X.corr()
print("=== Mean vector ===")
print(mean_vec.round(2), "\n")
print("=== Covariance matrix ===")
print(cov.round(2), "\n")
print("=== Correlation matrix ===")
print(corr.round(3), "\n")
print(f"Generalised variance (det of covariance) = {np.linalg.det(cov):.1f}")
print(f"Total variance (trace of covariance)     = {np.trace(cov):.1f}\n")
cov.round(3).to_csv(out / "py_covariance_matrix.csv")
corr.round(3).to_csv(out / "py_correlation_matrix.csv")

# ---------- Q2: multivariate outliers (Mahalanobis distance) ----------
diff = X.values - mean_vec.values
inv_cov = np.linalg.inv(cov.values)
d2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)       # squared distances
cutoff = stats.chi2.ppf(0.975, df=len(scores))
n_out = int((d2 > cutoff).sum())
print("=== Mahalanobis outliers ===")
print(f"Cut-off (chi-square 97.5%, df=3) = {cutoff:.2f}")
print(f"Multivariate outliers            = {n_out} of {len(df)}\n")

# ---------- Q3: PCA ----------
Z = StandardScaler().fit_transform(X)
pca = PCA()
scores_pc = pca.fit_transform(Z)

expl = pca.explained_variance_ratio_
print("=== PCA (standardised variables) ===")
print("Eigenvalues         :", np.round(pca.explained_variance_, 3))
print("Explained variance  :", np.round(expl, 3))
print("Cumulative variance :", np.round(np.cumsum(expl), 3))
loadings = pd.DataFrame(pca.components_.T, index=scores,
                        columns=[f"PC{i+1}" for i in range(len(scores))])
print("\nLoadings:")
print(loadings.round(3), "\n")
loadings.round(3).to_csv(out / "py_pca_loadings.csv")

# ---------- Q4: clustering (K-means) ----------
print("=== K-means: choosing k with the silhouette score ===")
sil = {}
for k in range(2, 7):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Z)
    sil[k] = silhouette_score(Z, labels)
    print(f"k = {k}: silhouette = {sil[k]:.3f}")
best_k = max(sil, key=sil.get)
print(f"Best k = {best_k}\n")

km = KMeans(n_clusters=best_k, n_init=10, random_state=42)
df["cluster"] = km.fit_predict(Z)
profile = df.groupby("cluster")[scores].mean().round(2)
profile["size"] = df["cluster"].value_counts().sort_index()
print("=== Cluster profiles (mean scores) ===")
print(profile, "\n")
profile.to_csv(out / "py_cluster_profiles.csv")

# ---------- Q5: MANOVA - do the score profiles differ by gender? ----------
manova = MANOVA.from_formula(
    "math_score + reading_score + writing_score ~ gender", data=df)
print("=== MANOVA: scores ~ gender ===")
print(manova.mv_test())

# ---------- Charts ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

axes[0].bar(range(1, len(expl) + 1), expl, edgecolor="black", alpha=0.7)
axes[0].plot(range(1, len(expl) + 1), np.cumsum(expl), "ro-",
             label="Cumulative")
axes[0].set_xticks(range(1, len(expl) + 1))
axes[0].set_xlabel("Principal component")
axes[0].set_ylabel("Proportion of variance")
axes[0].set_title("Scree plot")
axes[0].legend()

sc = axes[1].scatter(scores_pc[:, 0], scores_pc[:, 1], c=df["cluster"],
                     cmap="viridis", alpha=0.6, s=15)
axes[1].set_xlabel(f"PC1 ({expl[0]*100:.1f}%)")
axes[1].set_ylabel(f"PC2 ({expl[1]*100:.1f}%)")
axes[1].set_title("Component scores coloured by cluster")
plt.colorbar(sc, ax=axes[1], label="Cluster")

axes[2].plot(list(sil.keys()), list(sil.values()), "o-")
axes[2].set_xlabel("Number of clusters k")
axes[2].set_ylabel("Silhouette score")
axes[2].set_title("Choosing k")

plt.tight_layout()
plt.savefig(out / "py_multivariate_plots.png", dpi=150)
plt.show()