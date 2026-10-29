from pathlib import Path

import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

HERE = Path(__file__).resolve().parent
(HERE / "output").mkdir(exist_ok=True)

cells = []
def md(text):   cells.append(nbf.v4.new_markdown_cell(text.strip()))
def code(text): cells.append(nbf.v4.new_code_cell(text.strip()))

md('''
# Statistical Report: Student Exam Performance
**Course:** BIT 4131 Statistical Computing and Reporting
**Author:** Yvonne Odhiambo

## 1. Introduction
This report analyses exam scores of 1,000 students in mathematics, reading
and writing. It asks three questions:
1. How are the scores distributed?
2. Do male and female students differ in their scores?
3. Is completing the test-preparation course associated with higher scores?

Every number, table and chart below is produced by the code in this notebook,
so the whole report can be re-run to give identical results.
''')

code('''
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from IPython.display import Markdown, display

df = pd.read_csv(Path.cwd().parent / "data" / "Cleaned_Students_Performance.csv")
df.columns = [c.replace(" ", "_").replace("/", "_") for c in df.columns]

# 0/1 codes inferred from the original Kaggle data
if pd.api.types.is_numeric_dtype(df["gender"]):
    df["gender"] = df["gender"].map({0: "female", 1: "male"})
if pd.api.types.is_numeric_dtype(df["test_preparation_course"]):
    df["test_preparation_course"] = df["test_preparation_course"].map(
        {1: "completed", 0: "none"})

scores = ["math_score", "reading_score", "writing_score"]
print("Rows, columns:", df.shape)
print("Missing values:", int(df[scores].isna().sum().sum()))
''')

md('''
## 2. Data
The data are the *Students Performance in Exams* dataset (Kaggle), in a cleaned
version. Each row is one student. The scores range from 0 to 100.
Gender and test-preparation status are stored as 0/1 codes and are converted
to labels in the code above.
''')

md('## 3. Descriptive statistics')
code('''
summary = df[scores].describe().T
summary["median"] = df[scores].median()
summary["skewness"] = df[scores].skew()
summary = summary[["count", "mean", "median", "std", "min", "max", "skewness"]].round(2)
summary.to_csv("output/summary_table.csv")
summary
''')

md('## 4. Graphs')
code('''
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, scores):
    ax.hist(df[col], bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(df[col].mean(), color="red", linestyle="--", label="Mean")
    ax.set_title(col)
    ax.set_xlabel("Score")
    ax.set_ylabel("Number of students")
    ax.legend()
plt.tight_layout()
plt.savefig("output/histograms.png", dpi=150)
plt.show()

df.boxplot(column=scores, by="gender", layout=(1, 3), figsize=(15, 4))
plt.suptitle("Scores by gender")
plt.tight_layout()
plt.savefig("output/boxplots_by_gender.png", dpi=150)
plt.show()
''')

md('## 5. Statistical analysis')
code('''
# Question 2: gender differences (Welch t-test and Cohen's d)
rows = []
for col in scores:
    f = df.loc[df["gender"] == "female", col]
    m = df.loc[df["gender"] == "male", col]
    t, p = stats.ttest_ind(m, f, equal_var=False)
    pooled = np.sqrt((f.var() + m.var()) / 2)
    rows.append({"score": col, "female_mean": f.mean(), "male_mean": m.mean(),
                 "difference": m.mean() - f.mean(), "t": t, "p_value": p,
                 "cohens_d": (m.mean() - f.mean()) / pooled})
gender_tbl = pd.DataFrame(rows).round(4)
gender_tbl.to_csv("output/gender_tests.csv", index=False)
gender_tbl
''')
code('''
# Question 3: test preparation (Welch t-test and Cohen's d)
rows = []
for col in scores:
    c = df.loc[df["test_preparation_course"] == "completed", col]
    n_ = df.loc[df["test_preparation_course"] == "none", col]
    t, p = stats.ttest_ind(c, n_, equal_var=False)
    pooled = np.sqrt((c.var() + n_.var()) / 2)
    rows.append({"score": col, "completed_mean": c.mean(), "none_mean": n_.mean(),
                 "difference": c.mean() - n_.mean(), "t": t, "p_value": p,
                 "cohens_d": (c.mean() - n_.mean()) / pooled})
prep_tbl = pd.DataFrame(rows).round(4)
prep_tbl.to_csv("output/prep_tests.csv", index=False)
prep_tbl
''')
code('''
corr = df[scores].corr().round(3)
corr.to_csv("output/correlations.csv")
corr
''')

md('## 6. Interpretation')
code('''
g = gender_tbl.set_index("score")
p_ = prep_tbl.set_index("score")

def verdict(p):
    return "statistically significant" if p < 0.05 else "not statistically significant"

lines = []
lines.append(f"- **Distribution.** Mean scores are {summary.loc['math_score','mean']:.1f} "
             f"(math), {summary.loc['reading_score','mean']:.1f} (reading) and "
             f"{summary.loc['writing_score','mean']:.1f} (writing). "
             f"Skewness is slightly negative for all three, so each has a longer tail "
             f"of low scores.")
for col in scores:
    d = g.loc[col, "difference"]
    who = "males" if d > 0 else "females"
    lines.append(f"- **Gender, {col}.** {who.capitalize()} scored {abs(d):.1f} points higher "
                 f"on average (p = {g.loc[col,'p_value']:.4g}, d = {g.loc[col,'cohens_d']:.2f}); "
                 f"the difference is {verdict(g.loc[col,'p_value'])}.")
for col in scores:
    lines.append(f"- **Test preparation, {col}.** Students who completed the course scored "
                 f"{p_.loc[col,'difference']:.1f} points {'higher' if p_.loc[col,'difference']>0 else 'lower'} "
                 f"(p = {p_.loc[col,'p_value']:.4g}, d = {p_.loc[col,'cohens_d']:.2f}); "
                 f"{verdict(p_.loc[col,'p_value'])}.")
lines.append(f"- **Correlation.** The scores are strongly related: math and reading "
             f"r = {corr.loc['math_score','reading_score']:.2f}, reading and writing "
             f"r = {corr.loc['reading_score','writing_score']:.2f}.")
display(Markdown("\\n".join(lines)))
''')

md('''
## 7. Conclusions
- The three scores are roughly bell-shaped and strongly positively correlated, so
  students who do well in one subject tend to do well in the others.
- The gender comparison and the test-preparation comparison are summarised in
  Section 6 using Welch t-tests (which do not assume equal variances) and
  Cohen's d as the effect size.
- **Limitations.** The data are observational, so the results show association,
  not cause. For example, students who complete the course may differ in other
  ways. The 0/1 coding of the variables was inferred from the original dataset.

## 8. Reproducibility
Run the notebook from top to bottom to regenerate every table and graph.
The software environment used is shown below.
''')
code('''
import matplotlib, scipy
print("Python    :", sys.version.split()[0])
print("pandas    :", pd.__version__)
print("numpy     :", np.__version__)
print("scipy     :", scipy.__version__)
print("matplotlib:", matplotlib.__version__)
''')

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                "language": "python"}

print("Running the notebook (this takes a few seconds)...")
ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(
    nb, {"metadata": {"path": str(HERE)}})

nbf.write(nb, HERE / "report.ipynb")
body, _ = HTMLExporter().from_notebook_node(nb)
(HERE / "report.html").write_text(body, encoding="utf-8")
print("Created report.ipynb and report.html in", HERE)