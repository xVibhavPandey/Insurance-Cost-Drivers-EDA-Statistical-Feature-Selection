# Insurance-Cost-Drivers-EDA-Statistical-Feature-Selection

Exploratory data analysis, cleaning, and feature engineering pipeline on the Kaggle Medical Cost Personal Dataset, with statistical feature selection using Pearson correlation and Chi-Square tests to identify key cost drivers, culminating in a trained Linear Regression model deployed as an interactive Streamlit web application.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
  - [1. Exploratory Data Analysis](#1-exploratory-data-analysis)
  - [2. Data Cleaning](#2-data-cleaning)
  - [3. Feature Engineering](#3-feature-engineering)
  - [4. Feature Selection](#4-feature-selection)
  - [5. Model Training & Evaluation](#5-model-training--evaluation)
- [Key Findings](#key-findings)
- [Final Feature Set](#final-feature-set)
- [Interactive Web App](#interactive-web-app)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)

## Overview

This project analyzes the relationship between personal attributes (age, BMI, smoking status, region, etc.) and medical insurance charges. The pipeline spans from raw data exploration to a statistically validated feature set, culminating in a predictive **Linear Regression** model serialized with Joblib and served via an interactive **Streamlit** user interface.

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
- Normalized continuous variables where appropriate for linear modeling

### 4. Feature Selection

Two complementary statistical tests were used to validate which features actually matter for predicting `charges`:

- **Pearson Correlation** — measured the linear relationship between each numeric/binary feature and `charges`.
- **Chi-Square Test of Independence** — since Pearson correlation only captures linear relationships, `charges` was binned into quartiles (`charges_bin`) and each categorical feature was tested for independence against this binned target at a significance level of **α = 0.05**. Features with p-values below 0.05 were retained; the rest were dropped as statistically uninformative.

### 5. Model Training & Evaluation

- Split the feature-selected dataset using an **80/20 train-test split** (`test_size=0.20`, `random_state=42`).
- Trained a **Linear Regression** baseline model on the 7 selected predictors.
- Serialized the final fitted estimator to `Insurance_model.pkl` using Joblib for real-time inference.

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

Based on the combined statistical tests, the following 7 features are used as model inputs:
