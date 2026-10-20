from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE.parent / "data" / "Cleaned_Students_Performance.csv")
df.columns = [c.replace(" ", "_").replace("/", "_") for c in df.columns]

# In this file gender is coded 0/1 (0 = female, 1 = male); convert to labels
if pd.api.types.is_numeric_dtype(df["gender"]):
    df["gender"] = df["gender"].map({0: "female", 1: "male"})

out = HERE / "output"
out.mkdir(exist_ok=True)

scores = ["math_score", "reading_score", "writing_score"]


def summarise(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return pd.Series({
        "n": s.count(),
        "mean": s.mean(),
        "median": s.median(),
        "mode": s.mode().iloc[0],
        "variance": s.var(),
        "std_dev": s.std(),
        "range": s.max() - s.min(),
        "Q1": q1,
        "Q3": q3,
        "IQR": q3 - q1,
        "skewness": s.skew(),
    })


# Q1: summary statistics
summary = df[scores].apply(summarise).round(2)
print(summary, "\n")
summary.to_csv(out / "py_summary_stats.csv")

# Q2: dispersion by gender
grp = df.groupby("gender")[scores]
print(grp.agg(["mean", "median", "std"]).round(2), "\n")
print((grp.std() / grp.mean() * 100).round(2), "\n")

# Q3: outliers and boxplots
for col in scores:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = df.loc[(df[col] < low) | (df[col] > high), col]
    print(f"{col}: fences=({low:.1f}, {high:.1f}), outliers={len(outliers)}")

df[scores].plot(kind="box", figsize=(7, 5), title="Exam scores")
plt.ylabel("Score")
plt.tight_layout()
plt.savefig(out / "py_boxplot_scores.png", dpi=150)

df.boxplot(column="math_score", by="gender", figsize=(6, 5))
plt.suptitle("")
plt.title("Math score by gender")
plt.ylabel("Math score")
plt.tight_layout()
plt.savefig(out / "py_boxplot_math_by_gender.png", dpi=150)

plt.show()