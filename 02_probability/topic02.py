from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- Load data ----------
HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE.parent / "data" / "Cleaned_Students_Performance.csv")
df.columns = [c.replace(" ", "_").replace("/", "_") for c in df.columns]

# Convert the 0/1 codes to labels (0/1 meanings inferred from the original data)
if pd.api.types.is_numeric_dtype(df["gender"]):
    df["gender"] = df["gender"].map({0: "female", 1: "male"})
if pd.api.types.is_numeric_dtype(df["lunch"]):
    df["lunch"] = df["lunch"].map({1: "standard", 0: "free/reduced"})
if pd.api.types.is_numeric_dtype(df["test_preparation_course"]):
    df["test_preparation_course"] = df["test_preparation_course"].map(
        {1: "completed", 0: "none"})

out = HERE / "output"
out.mkdir(exist_ok=True)

PASS_MARK = 50   # change this if your question defines "pass" differently
HIGH_MARK = 70

# ---------- Define events (True/False for every student) ----------
passed_math = df["math_score"] >= PASS_MARK
high_math = df["math_score"] >= HIGH_MARK
prep_done = df["test_preparation_course"] == "completed"
standard_lunch = df["lunch"] == "standard"
female = df["gender"] == "female"


def cond(a, b):
    """P(A | B) = P(A and B) / P(B)"""
    return (a & b).mean() / b.mean()


# ---------- Q1: basic and joint probabilities ----------
print("=== Basic probabilities ===")
print(f"P(female)                    = {female.mean():.4f}")
print(f"P(passed math)               = {passed_math.mean():.4f}")
print(f"P(high math >= {HIGH_MARK})         = {high_math.mean():.4f}")
print(f"P(completed prep)            = {prep_done.mean():.4f}")
print(f"P(passed AND completed prep) = {(passed_math & prep_done).mean():.4f}")
print(f"P(passed OR completed prep)  = {(passed_math | prep_done).mean():.4f}\n")

# ---------- Q2: conditional probabilities ----------
p_pass_prep = cond(passed_math, prep_done)
p_pass_noprep = cond(passed_math, ~prep_done)
p_high_std = cond(high_math, standard_lunch)
p_high_free = cond(high_math, ~standard_lunch)
p_prep_high = cond(prep_done, high_math)   # Bayes-type: P(prep | high math)

print("=== Conditional probabilities ===")
print(f"P(passed | completed prep)   = {p_pass_prep:.4f}")
print(f"P(passed | no prep)          = {p_pass_noprep:.4f}")
print(f"P(high math | standard lunch)= {p_high_std:.4f}")
print(f"P(high math | free lunch)    = {p_high_free:.4f}")
print(f"P(completed prep | high math)= {p_prep_high:.4f}\n")

# Independence check: if independent, P(A and B) = P(A) * P(B)
joint = (passed_math & prep_done).mean()
product = passed_math.mean() * prep_done.mean()
print("=== Independence check (passed vs completed prep) ===")
print(f"P(A and B)   = {joint:.4f}")
print(f"P(A) * P(B)  = {product:.4f}")
print("Independent" if np.isclose(joint, product, atol=0.01) else "Not independent", "\n")

results = pd.DataFrame({
    "probability": [
        "P(passed math)", "P(completed prep)", "P(passed and completed prep)",
        "P(passed | completed prep)", "P(passed | no prep)",
        "P(high math | standard lunch)", "P(high math | free lunch)",
        "P(completed prep | high math)"],
    "value": [
        passed_math.mean(), prep_done.mean(), joint,
        p_pass_prep, p_pass_noprep, p_high_std, p_high_free, p_prep_high],
}).round(4)
results.to_csv(out / "py_probabilities.csv", index=False)

# ---------- Q3: Monte Carlo simulation ----------
# Pick random students (with replacement) and watch the estimate of
# P(passed | completed prep) settle on the exact value.
rng = np.random.default_rng(42)
N = 10_000
idx = rng.integers(0, len(df), N)
s_prep = prep_done.values[idx]
s_pass = passed_math.values[idx]

running = np.cumsum(s_prep & s_pass) / np.maximum(np.cumsum(s_prep), 1)
print("=== Simulation ===")
print(f"Simulated P(passed | completed prep) after {N} draws = {running[-1]:.4f}")
print(f"Exact value from the data                          = {p_pass_prep:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(running, label="Simulated estimate")
plt.axhline(p_pass_prep, color="red", linestyle="--", label="Exact value")
plt.xlabel("Number of random draws")
plt.ylabel("P(passed | completed prep)")
plt.title("Simulation converging to the exact conditional probability")
plt.legend()
plt.tight_layout()
plt.savefig(out / "py_simulation_convergence.png", dpi=150)
plt.show()