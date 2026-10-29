# Statistical Computing and Reporting

Python solutions to the 23 topics of **BIT 4131 Statistical Computing and Reporting** (group work), analysing the Students Performance dataset. Each topic has its own numbered folder containing a script (`topicNN.py`) and an `output/` folder with the tables and charts it produces.

## Topics

| # | Folder | Topic | Main libraries | What it covers |
|---|--------|-------|----------------|----------------|
| 1 | `01_central_tendency` | Measures of central tendency and dispersion | pandas, numpy | Mean, median, mode, variance, std dev, quartiles, IQR, skewness, dispersion by gender, outliers (1.5 × IQR), boxplots |
| 2 | `02_probability` | Probability and conditional probability | numpy, scipy | Joint and conditional probabilities, independence check, Monte Carlo simulation |
| 3 | `03_distributions` | Probability distributions | scipy.stats, numpy | Fitted normal parameters, probability values, histograms with fitted curves, binomial probabilities |
| 4 | `04_sampling_clt` | Sampling distributions and Central Limit Theorem | numpy, scipy, matplotlib | Sampling distribution, sample means, standard errors, CLT visualisation |
| 5 | `05_confidence_intervals` | Confidence intervals | scipy.stats, statsmodels | Point estimates, confidence intervals, interpretation of uncertainty |
| 6 | `06_hypothesis_testing` | Hypothesis testing | scipy.stats, statsmodels | Test statistics, p-values, decisions and interpretation |
| 7 | `07_correlation` | Correlation and covariance | pandas, numpy, scipy | Covariance matrix, correlation matrix, scatter plots |
| 8 | `08_regression` | Linear regression and least squares | statsmodels, scikit-learn | Regression equation, coefficients, p-values, R-squared, residual and fitted plots |
| 9 | `09_matrix_algebra` | Matrix algebra in statistics | numpy, scipy.linalg | Matrix addition, multiplication, transpose, determinant, inverse, linear systems |
| 10 | `10_pca` | Eigenvalues, eigenvectors and PCA | numpy.linalg, scikit-learn | Eigenvalues, eigenvectors, explained variance, scree plot, component plot |
| 11 | `11_optimization` | Numerical optimization | scipy.optimize | Objective function, estimated optimum, convergence result and plot |
| 12 | `12_monte_carlo` | Random numbers and Monte Carlo methods | numpy.random, scipy.stats | Random samples, simulated estimates, Monte Carlo approximation, convergence plot |
| 13 | `13_time_series` | Time-series analysis | pandas, statsmodels | Time-series plot, trend, seasonality, autocorrelation, basic forecast |
| 14 | `14_visualization` | Statistical data visualization | matplotlib, plotly | Histogram, boxplot, scatter plot, line graph |
| 15 | `15_transformation` | Data transformation and normalization | pandas, numpy, scikit-learn | Original versus transformed, standardised and normalised values, comparison plots |
| 16 | `16_anova` | ANOVA and experimental design | statsmodels, scipy.stats | ANOVA table, F-statistic, p-value, group means, post-hoc comparisons |
| 17 | `17_chi_square` | Chi-square and categorical data analysis | scipy.stats, statsmodels | Contingency table, expected frequencies, chi-square statistic, p-value, conclusion |
| 18 | `18_logistic_regression` | Logistic regression | statsmodels, scikit-learn | Coefficients, odds ratios, probabilities, confusion matrix, classification performance |
| 19 | `19_nonparametric` | Non-parametric statistical methods | scipy.stats | Rank-based test statistics and p-values, comparison with parametric methods |
| 20 | `20_bootstrap` | Bootstrap and resampling methods | numpy, scipy, scikit-learn | 5,000 bootstrap samples, sampling distribution, standard error, 95% confidence intervals, male vs female difference |
| 21 | `21_multivariate` | Multivariate statistical analysis | numpy, scikit-learn, statsmodels | Covariance structure, Mahalanobis outliers, PCA, K-means clustering, MANOVA |
| 22 | `22_missing_data` | Missing-data analysis and imputation | pandas, scikit-learn | Missingness summary and map, mean/median/KNN/iterative imputation, before-and-after comparison |
| 23 | `23_reproducible_report` | Statistical reporting and reproducible analysis | pandas, matplotlib, Jupyter | Executed notebook and HTML report with code, tables, graphs, interpretation and conclusions |

## Dataset

*Students Performance in Exams* (Kaggle): exam scores of 1,000 students in math, reading and writing, with background variables. The file used is `data/Cleaned_Students_Performance.csv`, a cleaned version in which gender, lunch and test preparation are stored as 0/1 codes. The scripts convert them to labels (gender 0 = female, 1 = male; lunch 1 = standard; preparation 1 = completed, inferred from the original dataset).

## Project structure

```
statistical-computing/
├── 01_central_tendency/
├── 02_probability/
├── 03_distributions/
├── 04_sampling_clt/ ... 15_transformation/
├── 16_anova/
├── 17_chi_square/
├── 18_logistic_regression/
├── 19_nonparametric/
├── 20_bootstrap/
├── 21_multivariate/
├── 22_missing_data/
├── 23_reproducible_report/
├── data/
│   └── Cleaned_Students_Performance.csv
└── README.md
```

## How to run

1. Install Python 3 and the libraries:
   ```
   pip install pandas numpy scipy matplotlib statsmodels scikit-learn plotly jupyter nbformat nbconvert ipykernel
   ```
2. From the project root, run any topic, for example:
   ```
   python 20_bootstrap/topic20.py
   ```
3. Tables and charts are saved in that topic's `output/` folder. Close each chart window for the script to finish.
4. For the final report:
   ```
   python 23_reproducible_report/make_report.py
   ```
   Then open `23_reproducible_report/report.html` in a browser.

## Notes

- **Topic 22:** the dataset has no missing values, so values are deleted on purpose (random and lunch-dependent) to compare imputation methods against the known true values.
- The data are observational, so results show association, not causation.
- Random seeds are fixed in the simulation and resampling topics so results can be reproduced.

## Author

Yvonne Odhiambo