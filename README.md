# Insurance Cost Drivers: EDA, Statistical Feature Selection & Multi-Model Evaluation

Exploratory data analysis, cleaning, and feature engineering pipeline on the Kaggle Medical Cost Personal Dataset, featuring statistical feature selection (Pearson correlation & Chi-Square tests) and a rigorous benchmark across Linear Regression, K-Nearest Neighbors, and Random Forest models. The top-performing Random Forest Regressor is serialized with Joblib and deployed as an interactive Streamlit web application.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
  - [1. Exploratory Data Analysis](#1-exploratory-data-analysis)
  - [2. Data Cleaning](#2-data-cleaning)
  - [3. Feature Engineering](#3-feature-engineering)
  - [4. Statistical Feature Selection](#4-statistical-feature-selection)
  - [5. Model Training & Benchmarking](#5-model-training--benchmarking)
- [Key Findings & Statistical Tests](#key-findings--statistical-tests)
- [Final Feature Set](#final-feature-set)
- [Model Evaluation & Comparative Analysis](#model-evaluation--comparative-analysis)
  - [Benchmark Results](#benchmark-results)
  - [Metric Interpretation & Why Random Forest Won](#metric-interpretation--why-random-forest-won)
- [Interactive Web App](#interactive-web-app)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)

---

## Overview

This project investigates the multi-factorial relationships between individual demographic/health factors (age, BMI, smoking habits, region) and medical insurance charges. The pipeline bridges rigorous exploratory analysis and statistical hypothesis testing with multi-model benchmarking. While Linear Regression and K-Nearest Neighbors served as valuable baselines, an ensemble Random Forest model was selected for production deployment to properly capture sharp non-linear interactions (e.g., high BMI compounded by smoking status).

---

## Dataset

The dataset (`insurance.csv`) is sourced from Kaggle's Medical Cost Personal Dataset and contains 1,338 records across 7 columns:

| Column | Description | Type |
| :--- | :--- | :--- |
| `age` | Age of the primary beneficiary | Integer |
| `sex` | Gender of the beneficiary (`male`, `female`) | Categorical |
| `bmi` | Body Mass Index ($kg/m^2$) | Float |
| `children` | Number of dependents covered by insurance | Integer |
| `smoker` | Smoking status (`yes`, `no`) | Categorical |
| `region` | Residential area in the US (`northeast`, `southeast`, `southwest`, `northwest`) | Categorical |
| `charges` | Individual medical costs billed per year (**Target**) | Float |

*Data Hygiene Note:* The dataset contained zero missing values. One duplicate row was detected and purged during data cleaning.

---

## Project Workflow

### 1. Exploratory Data Analysis
- Inspected distributions, column data types, and summary statistics across all features.
- Visualized heavy right-skewed behavior in the target `charges` variable via distribution plots and KDE curves.
- Examined categorical splits across `sex`, `smoker`, `children`, and `region`.
- Analyzed bivariate relationships indicating clear discrete cluster shifts in charges between smokers and non-smokers.

### 2. Data Cleaning
- Purged 1 duplicate record, leaving 1,337 unique records.
- Formally mapped binary categoricals to numerical formats:
  - `sex` $\rightarrow$ `is_female` (Male: `0`, Female: `1`)
  - `smoker` $\rightarrow$ `is_smoker` (No: `0`, Yes: `1`)
- One-hot encoded categorical residential regions.
- Ensured uniform numerical data types across all pipeline transformations.

### 3. Feature Engineering
- Clinical binning of continuous `bmi` into established medical classes:
  - **Underweight:** $< 18.5$
  - **Normal:** $18.5 - 24.9$
  - **Overweight:** $25.0 - 29.9$
  - **Obese:** $\ge 30.0$
- Derived the binary indicator `bmi_category_obese` to isolate clinical threshold effects where charges scale non-linearly.

### 4. Statistical Feature Selection
Features were filtered using two formal hypothesis checks:
1. **Pearson Correlation ($r$):** Filtered numerical/binary predictors with direct linear association to `charges`.
2. **Chi-Square ($\chi^2$) Test of Independence:** Discretized continuous `charges` into quartiles (`charges_bin`) to assess non-linear categorical dependencies at $\alpha = 0.05$. Categoricals failing this significance threshold were dropped.

### 5. Model Training & Benchmarking
- Partitioned the cleaned, feature-selected data using an 80/20 train-test split (`test_size=0.20`, `random_state=42`).
- Trained and evaluated three candidate model families:
  1. **Linear Regression (Baseline OLS)**
  2. **K-Nearest Neighbors Regressor (Instance-based learning)**
  3. **Random Forest Regressor (Ensemble Bagging)**
- Evaluated performance across both scale-dependent (MAE, RMSE) and percentage-based (MAPE, Explained Variance) metrics.
- Exported the winning Random Forest model to `Insurance_model_ensemble_rf.pkl` using Joblib.

---

## Key Findings & Statistical Tests

### Pearson Correlation with Charges
| Feature | Pearson $r$ | Assessment |
| :--- | :--- | :--- |
| `is_smoker` | **0.787** | Dominant primary driver of high charges |
| `age` | **0.298** | Strong positive linear baseline trend |
| `bmi_category_obese` | **0.200** | Statistically significant threshold indicator |
| `bmi` | **0.196** | Steady continuous upward cost pressure |
| `region_southeast` | **0.074** | Weak but statistically valid correlation |
| `children` | **0.067** | Minor positive correlation |

### Chi-Square Test Results (vs. Charges Quartiles)
$$\alpha = 0.05$$

| Feature | $\chi^2$ Statistic | $p$-value | Decision |
| :--- | :--- | :--- | :--- |
| `is_smoker` | 848.22 | $< 0.001$ | **Keep** |
| `region_southeast` | 16.00 | $0.001$ | **Keep** |
| `is_female` | 10.26 | $0.016$ | **Keep** |
| `bmi_category_obese` | 8.52 | $0.036$ | **Keep** |
| `region_northeast` | 6.44 | $0.092$ | **Drop** |
| `region_southwest` | 5.09 | $0.165$ | **Drop** |
| `bmi_category_overweight` | 4.25 | $0.236$ | **Drop** |
| `bmi_category_normal` | 3.71 | $0.295$ | **Drop** |
| `region_northwest` | 1.13 | $0.769$ | **Drop** |

---

## Final Feature Set

The final supervised learning models were trained on the following 7 predictors:
1. `age` *(int)*
2. `bmi` *(float)*
3. `children` *(int)*
4. `is_smoker` *(int: 0 or 1)*
5. `region_southeast` *(int: 0 or 1)*
6. `is_female` *(int: 0 or 1)*
7. `bmi_category_obese` *(int: 0 or 1)*

---

## Model Evaluation & Comparative Analysis

Continuous target evaluation requires tracking both typical linear deviations and sensitivity to severe claim spikes. The candidate models were evaluated on the test set using:
- **Mean Absolute Error (MAE):** Average magnitude of absolute residuals (linear weighting).
- **Root Mean Squared Error (RMSE):** Quadratic residual penalty highlighting large, catastrophic claim misses.
- **RMSE / MAE Ratio:** Diagnostic index of error distribution normality vs. heavy-tailed outlier penalties.
- **Mean Absolute Percentage Error (MAPE):** Relative percentage deviation across samples.

### Benchmark Results

| Model Architecture | MAE | RMSE | RMSE / MAE Ratio | Performance Summary |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest Regressor** | **$2,638.26** | **$4,573.59** | 1.73 | **Selected Winner:** Superior non-linear fit across all error tiers |
| **Linear Regression** | $4,295.20 | $6,000.26 | 1.40 | Good global baseline; underfits compound interactions |
| **KNN Regressor ($k \approx 5$)** | $3,545.12 | $6,781.85 | 1.91 | Tighter local cluster fit, but high error variance on edge cases |
| *KNN Regressor ($k = 15$)* | *$4,851.52* | *$8,091.98* | 1.67 | Underfitting caused by neighborhood over-smoothing |

### Metric Interpretation & Why Random Forest Won

1. **Why Linear Regression Hit a Performance Ceiling:**
   Ordinary Least Squares assumes strictly additive predictor effects. In healthcare costs, however, risk compounds multiplicatively—a high BMI in a non-smoker causes modest cost increases, but high BMI combined with smoking triggers exponential cost surges. Without manual polynomial interaction terms, Linear Regression leaves significant residual variance unexplained ($\text{RMSE} \approx \$6,000.26$).

2. **The Trade-Offs in K-Nearest Neighbors:**
   - At lower neighbor counts ($k \le 7$), KNN achieved a stronger typical fit ($\text{MAE} = \$3,545.12$) than Linear Regression by interpolating homogenous low-cost clusters.
   - However, when calculating distance across mixed categorical boundaries, high-cost smoker claims get smoothed together with nearby non-smokers. This leads to massive squared penalties on outliers, driving RMSE up to $\$6,781.85$.
   - Increasing neighborhood size to $k=15$ caused severe underfitting, degrading MAE to $\$4,851.52$ and RMSE to $\$8,091.98$.

3. **Why Random Forest Was Selected:**
   - **Error Reduction:** Random Forest achieved an MAE of **$2,638.26** (a **38.6% error reduction** over Linear Regression) and lowered RMSE to **$4,573.59** (a **23.8% reduction**).
   - **Threshold Splitting:** Decision tree ensembles naturally handle step-function discontinuities without manual transformation, cleanly isolating subsets like `is_smoker == 1` $\land$ `bmi >= 30`.
   - **Percentage Accuracy Dynamics:** The model achieves ~63.37% raw percentage accuracy ($100\% - \text{MAPE}$). This metric is heavily depressed by low-denominator distortion (e.g., an acceptable $1,000 error on a baseline $2,000 annual claim registers as a 50% penalty). Across operational tolerance bands, the model captures **~85% of total variance ($R^2 \ge 0.85$)** with high reliability.

---

## Interactive Web App

The production model is integrated into a lightweight, local Streamlit dashboard allowing users to input patient metrics and generate instant, real-time cost forecasts:

- **Resource Caching:** Employs `@st.cache_resource` for low-latency model loading via Joblib.
- **Input Guardrails:** Form layout with numeric and categorical constraints matching training validation rules.
- **Dynamic Feature Formatting:** Encodes raw form values directly into the model's exact 7-feature schema before generating predictions.

---

## Tech Stack

- **Language:** Python 3.10+
- **Machine Learning & Preprocessing:** scikit-learn, NumPy, Pandas, SciPy
- **Model Serialization:** Joblib
- **Web Dashboard:** Streamlit
- **Visualization:** Matplotlib, Seaborn

---

## Project Structure

```text
├── app.py                               # Streamlit web application
├── Insurance_model_ensemble_rf.pkl      # Production Random Forest estimator
├── notebooks/
│   └── insurance_eda_modeling.ipynb     # EDA, hypothesis testing & model benchmarking
├── data/
│   └── insurance.csv                    # Kaggle Medical Cost Personal dataset
├── requirements.txt                     # Environment dependencies
└── README.md                            # Project documentation
