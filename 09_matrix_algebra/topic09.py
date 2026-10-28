"""
Topic 09: Matrix algebra in statistics
Libraries: numpy, scipy.linalg (pandas to load the CSV, matplotlib for a chart)

Outputs: matrix addition, multiplication, transpose, determinant, inverse and
the solution of linear equations, saved to the output folder.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg

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
scores = df[[MATH, READING, WRITING]].dropna().to_numpy(dtype=float)
labels = ["Math", "Reading", "Writing"]

np.set_printoptions(precision=4, suppress=True, linewidth=120)
report = []


def show(title, value):
    """Print a titled matrix/value and keep it for the text report."""
    text = f"{title}\n{value}\n"
    print(text)
    report.append(text)


# ---------- Matrices built from the data ----------
X5 = scores[:5]                                  # 5 x 3 data matrix (first 5 students)
S = np.cov(scores, rowvar=False)                 # 3 x 3 covariance matrix
R = np.corrcoef(scores, rowvar=False)            # 3 x 3 correlation matrix

show("Data matrix X (first 5 students: Math, Reading, Writing), shape "
     f"{X5.shape}", X5)
show("Covariance matrix S (all students)", S)
show("Correlation matrix R (all students)", R)

# ---------- 1. Matrix addition ----------
show("1a. Addition: S + R", S + R)
show("1b. Subtraction: S - R", S - R)
show("1c. Scalar multiple: 0.5 * S", 0.5 * S)

# ---------- 2. Transpose ----------
show(f"2. Transpose of X, X' (shape {X5.T.shape})", X5.T)

# ---------- 3. Multiplication ----------
XtX = X5.T @ X5
show("3a. Multiplication: X'X (3 x 3 sums of squares and cross-products)", XtX)
show("3b. Multiplication: S @ R", S @ R)
show("3c. Is matrix multiplication commutative? max |S@R - R@S|",
     round(float(np.abs(S @ R - R @ S).max()), 4))

# ---------- 4. Determinant ----------
det_S = np.linalg.det(S)
show("4a. Determinant of S (the generalised variance)", f"{det_S:,.2f}")
show("4b. Determinant of R", f"{np.linalg.det(R):.4f}")
show("4c. Trace of S (total variance) and rank of S",
     f"trace = {np.trace(S):.2f}, rank = {np.linalg.matrix_rank(S)}")

# ---------- 5. Inverse ----------
S_inv = np.linalg.inv(S)
show("5a. Inverse of S (the precision matrix)", S_inv)
show("5b. Check: S @ S^-1 should be the identity matrix", np.round(S @ S_inv, 6))
show("5c. Condition number of S (large values mean an unstable inverse)",
     f"{np.linalg.cond(S):.2f}")

# ---------- 6. Solving linear equations ----------
# (a) A small system  A x = b  with a known solution
A = np.array([[2.0, 1.0, -1.0],
              [-3.0, -1.0, 2.0],
              [-2.0, 1.0, 2.0]])
b = np.array([8.0, -11.0, -3.0])
x_solve = np.linalg.solve(A, b)
show("6a. Solve A x = b with A = [[2,1,-1],[-3,-1,2],[-2,1,2]], b = [8,-11,-3]\n"
     "    (exact answer is x = [2, 3, -1])", x_solve)

# (b) Normal equations of regression: (X'X) beta = X'y
y = scores[:, 0]                                         # Math
Xd = np.column_stack([np.ones(len(scores)), scores[:, 1], scores[:, 2]])  # 1, Reading, Writing
XtX_full = Xd.T @ Xd
Xty = Xd.T @ y
beta_solve = np.linalg.solve(XtX_full, Xty)              # numpy.linalg.solve
beta_inv = np.linalg.inv(XtX_full) @ Xty                 # using the inverse
beta_scipy = linalg.solve(XtX_full, Xty)                 # scipy.linalg.solve
beta_lstsq = np.linalg.lstsq(Xd, y, rcond=None)[0]       # least squares routine

show("6b. Normal equations (X'X) beta = X'y for Math ~ Reading + Writing", "")
coef = pd.DataFrame({
    "np.linalg.solve": beta_solve,
    "inverse method": beta_inv,
    "scipy.linalg.solve": beta_scipy,
    "np.linalg.lstsq": beta_lstsq,
}, index=["Intercept", "Reading", "Writing"]).round(4)
print(coef, "\n")
report.append(coef.to_string() + "\n")
print(f"Regression equation: Math = {beta_solve[0]:.3f} + {beta_solve[1]:.3f} x Reading "
      f"+ {beta_solve[2]:.3f} x Writing\n")
print("All four methods give the same coefficients:",
      bool(np.allclose(beta_solve, beta_inv) and np.allclose(beta_solve, beta_lstsq)), "\n")

# ---------- Eigen check linking to later topics ----------
eigvals = np.linalg.eigvalsh(S)
show("Bonus: eigenvalues of S (all positive, so S is positive definite)", eigvals[::-1])

# ---------- Save reports ----------
(OUT / "matrix_results.txt").write_text("\n".join(report), encoding="utf-8")
pd.DataFrame(S, index=labels, columns=labels).round(4).to_csv(OUT / "covariance_matrix.csv")
pd.DataFrame(S_inv, index=labels, columns=labels).round(6).to_csv(OUT / "inverse_covariance_matrix.csv")
coef.to_csv(OUT / "normal_equations_solution.csv")

# ---------- Chart: S and its inverse ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, M, title in zip(axes, (S, S_inv), ("Covariance matrix S", "Inverse of S")):
    im = ax.imshow(M, cmap="coolwarm")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.3g}", ha="center", va="center")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig(OUT / "covariance_and_inverse.png", dpi=150)

print(f"Saved results and charts to: {OUT}")
plt.show()