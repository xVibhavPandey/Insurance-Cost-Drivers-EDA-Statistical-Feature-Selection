# Insurance-Cost-Drivers-EDA-Statistical-Feature-Selection

Exploratory data analysis, cleaning, and feature engineering pipeline on the Kaggle Medical Cost Personal Dataset, with statistical feature selection using Pearson correlation and Chi-Square tests to identify the strongest predictors of insurance charges.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
  - [1. Exploratory Data Analysis](#1-exploratory-data-analysis)
  - [2. Data Cleaning](#2-data-cleaning)
  - [3. Feature Engineering](#3-feature-engineering)
  - [4. Feature Selection](#4-feature-selection)
- [Key Findings](#key-findings)
- [Final Feature Set](#final-feature-set)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)

## Overview

This project analyzes the relationship between personal attributes (age, BMI, smoking status, region, etc.) and medical insurance charges. The notebook walks through a complete data preparation pipeline — from raw data exploration to a statistically validated, model-ready feature set — with the goal of understanding *which* factors most strongly influence insurance costs and *why*.

## Dataset

The dataset (`insurance.csv`) is sourced from Kaggle's [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance) and contains **1,338 records** across **7 columns**:

| Column | Description | Type |
|---|---|---|
| `age` | Age of the primary beneficiary | int |
| `sex` | Gender of the beneficiary | categorical |
| `bmi` | Body Mass Index | float |
| `children` | Number of dependents covered | int |
| `smoker` | Smoking status | categorical |
| `region` | Residential region in the US | categorical |
| `charges` | Individual medical costs billed (target variable) | float |

The dataset had no missing values, but did contain **1 duplicate record**, which was removed during cleaning.

## Project Workflow

### 1. Exploratory Data Analysis

Initial exploration to understand structure, distributions, and relationships in the raw data:

- Inspected shape, dtypes, summary statistics, and null counts
- Visualized distributions of `age`, `bmi`, `children`, and `charges` using histograms with KDE overlays
- Examined categorical balance for `sex`, `smoker`, and `children` via count plots
- Generated a correlation heatmap across numeric features to spot early linear relationships

### 2. Data Cleaning

- Removed 1 duplicate row (1,338 → 1,337 records)
- Verified no missing values remained post-cleaning
- Encoded binary categoricals numerically:
  - `sex` → `is_female` (male: 0, female: 1)
  - `smoker` → `is_smoker` (no: 0, yes: 1)
- One-hot encoded the multi-class `region` column
- Cast all engineered columns to integer type for consistency

### 3. Feature Engineering

- Binned continuous `bmi` into clinically meaningful categories using standard BMI thresholds:
  - **Underweight**: < 18.5
  - **Normal**: 18.5 – 24.9
  - **Overweight**: 25 – 29.9
  - **Obese**: 30+
- One-hot encoded the resulting `bmi_category` feature
- Applied `StandardScaler` to normalize continuous features (`age`, `bmi`, `children`) for scale-sensitive downstream modeling

### 4. Feature Selection

Two complementary statistical tests were used to validate which features actually matter for predicting `charges`:

**Pearson Correlation** — measured the linear relationship between each numeric/binary feature and `charges`.

**Chi-Square Test of Independence** — since Pearson correlation only captures linear relationships, `charges` was binned into quartiles (`charges_bin`) and each categorical feature was tested for independence against this binned target at a significance level of **α = 0.05**. Features with p-values below 0.05 were retained; the rest were dropped as statistically uninformative.

## Key Findings

**Pearson correlation with `charges`** (top features):

| Feature | Pearson r |
|---|---|
| `is_smoker` | 0.787 |
| `age` | 0.298 |
| `bmi_category_obese` | 0.200 |
| `bmi` | 0.196 |
| `region_southeast` | 0.074 |
| `children` | 0.067 |

Smoking status is, by a wide margin, the strongest predictor of insurance charges, followed distantly by age and obesity.

**Chi-Square test results** (features vs. charges quartile):

| Feature | χ² statistic | p-value | Decision |
|---|---|---|---|
| `is_smoker` | 848.22 | < 0.001 | Keep |
| `region_southeast` | 16.00 | 0.001 | Keep |
| `is_female` | 10.26 | 0.016 | Keep |
| `bmi_category_obese` | 8.52 | 0.036 | Keep |
| `region_northeast` | 6.44 | 0.092 | Drop |
| `region_southwest` | 5.09 | 0.165 | Drop |
| `bmi_category_overweight` | 4.25 | 0.236 | Drop |
| `bmi_category_normal` | 3.71 | 0.295 | Drop |
| `region_northwest` | 1.13 | 0.769 | Drop |

Only 4 of the 9 categorical features tested showed a statistically significant association with the charges distribution.

## Final Feature Set

Based on the combined results of both tests, the following features were retained for the final cleaned dataset (`cleaned_data.csv`):

```
age, bmi, children, is_smoker, is_female, region_southeast, bmi_category_obese, charges
```

## Tech Stack

- **Python 3**
- **pandas** / **numpy** — data manipulation
- **matplotlib** / **seaborn** — visualization
- **scikit-learn** — feature scaling (`StandardScaler`)
- **scipy** — statistical testing (`pearsonr`, `chi2_contingency`)

## Project Structure

```
├── insurance.csv          # Raw dataset (Kaggle source)
├── notebook.ipynb          # Full analysis notebook (EDA → cleaning → feature engineering → selection)
├── cleaned_data.csv        # Final, model-ready dataset after feature selection
└── README.md
```

## How to Run

1. Clone the repository and ensure `insurance.csv` is in the project root.
2. Install dependencies:
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn scipy
   ```
3. Open and run the notebook top to bottom:
   ```bash
   jupyter notebook notebook.ipynb
   ```
4. The final feature-selected dataset will be exported as `cleaned_data.csv`.

---

*This notebook focuses on data preparation and statistical feature selection as groundwork for future predictive modeling of insurance charges.*
