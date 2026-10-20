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
