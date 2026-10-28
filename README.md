# Statistical Computing

Python scripts for descriptive statistics, built as coursework for a statistical computing course.

## Topic 01: Central Tendency and Dispersion

`01_central_tendency/topic01.py` analyses exam scores for 1,000 students in math, reading and writing. It covers:

1. **Summary statistics:** count, mean, median, mode, variance, standard deviation, range, quartiles (Q1, Q3), IQR and skewness.
2. **Dispersion by gender:** mean, median and standard deviation for each gender, plus the coefficient of variation.
3. **Outlier detection:** the 1.5 x IQR rule, with boxplots of all scores and of math score by gender.

## Dataset

Students Performance in Exams (Kaggle), a dataset of student exam scores with background information such as gender, parental education and test preparation. The file used here is `data/Cleaned_Students_Performance.csv`, a cleaned version in which gender is coded 0 = female and 1 = male. The script converts these codes to labels.

## Project Structure
```
statistical-computing/
├── 01_central_tendency/
│   ├── topic01.py
│   └── output/
├── 02_probability/
│   ├── topic02.py
│   └── output/
├── data/
│   └── Cleaned_Students_Performance.csv
└── README.md
```

## Topic 02: Probability and Conditional Probability

`02_probability/topic02.py` uses the same dataset to calculate:

1. **Basic and joint probabilities:** for example P(passed math), P(completed prep course) and P(passed and completed prep).
2. **Conditional probabilities:** for example P(passed | completed prep) compared with P(passed | no prep), plus an independence check.
3. **Monte Carlo simulation:** 10,000 random draws showing the estimate converging to the exact conditional probability.

Results are saved in `02_probability/output/`.

## How to Run

1. Install the libraries:
   ```
   pip install pandas matplotlib numpy scipy
   ```
2. From the project root, run a script, for example:
   ```
   python 02_probability/topic02.py
   ```

## Author

Yvonne Odhiambo
