# Statistical Computing

Python scripts for a statistical computing course, analysing the Students Performance dataset.

## Topics

| Folder | Topic | What it covers |
|--------|-------|----------------|
| `01_central_tendency` | Descriptive statistics | Mean, median, mode, variance, std dev, IQR, skewness, dispersion by gender, outliers (1.5 × IQR), boxplots |
| `02_probability` | Probability | Probability calculations |
| `03_distributions` | Distributions | Probability distributions, e.g. P(X = 4) and P(X >= 5) |

## Dataset

`data/Cleaned_Students_Performance.csv`

## How to run

```
pip install pandas numpy matplotlib scipy
python 01_central_tendency/topic01.py
python 02_probability/topic02.py
python 03_distributions/topic03.py
```

Results (CSV files and charts) are saved in each topic's `output` folder.