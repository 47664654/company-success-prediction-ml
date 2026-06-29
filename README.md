# Company Success Prediction Using Machine Learning

A dual machine learning project — regression and classification — 
analysing what drives company success across financial, innovation, 
and organizational culture dimensions.

Built as part of the Master in Business Analytics program at 
Mediterranean School of Business (MSB), Tunis — October 2025.

**Authors:** Amine Bouraoui · Emna Labben

---

## Project Overview

This project investigates how **financial performance**, 
**innovation**, and **organizational culture** shape a company's 
overall success. Using a synthetic dataset of 18,000+ company 
records, we applied machine learning to answer two questions:

- **How successful will a company be?** → Regression task  
- **Is a company High or Low Success?** → Classification task

The project covers the full pipeline: exploratory analysis, 
data preparation, feature selection, model building, balancing 
techniques, hyperparameter tuning, and actionable business insights.

---

## Dataset

- **Type:** Synthetic dataset (simulated business records)
- **Size:** 18,000+ companies
- **Features (48 initial, 21 selected):**
  - Financial: Profit Margin, Annual Revenue, Operating Expenses, 
    Expense Ratio
  - Innovation: Innovation Index, R&D Investment
  - Culture: Work-Life Balance, Employee Turnover Rate, 
    Customer Satisfaction, Diversity Score, Environmental Score
- **Targets:**
  - `Success_Score` — continuous score (Regression)
  - `Success_Class` — binary: High / Low Success (Classification)
- **Class distribution:** ~67% High Success, ~33% Low Success 
  (moderate imbalance)

---

## Methodology

### 1. Exploratory Data Analysis
- Descriptive statistics: Innovation Index (mean 60.87), 
  Profit Margin (mean 15.97%), Employee Turnover (mean 10.36%)
- Correlation heatmap: strong Employee Count ↔ Operating Expenses 
  link (r = 0.77); moderate Profit Margin ↔ Expense Ratio 
  negative correlation (r = −0.41)
- Distribution analysis of both target variables

### 2. Data Preparation
- Missing value imputation using median strategy
- One-hot encoding for categorical variables
- MinMaxScaler normalization
- Outlier detection and treatment

### 3. Feature Selection
- Started with 48 features after encoding
- Applied **ANOVA F-test** to measure statistical relationship 
  with target
- Applied **Recursive Feature Elimination (RFE)** to remove 
  weak predictors iteratively
- Retained **top 21 most predictive features**

### 4. Model Building

**Regression (predicting Success Score):**

| Model | R² | MAPE |
|---|---|---|
| Linear Regression | **0.426** | **10.19%** |
| Random Forest | 0.390 | 11.8% |
| Gradient Boosting (Tuned) | 0.415 | 10.31% |

**Classification (predicting Success Class):**

| Method | Best Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Original (Unbalanced) | Logistic Regression | **0.769** | **0.667** | 0.482 | 0.559 |
| SMOTE (Balanced) | Random Forest | 0.758 | 0.613 | **0.554** | **0.582** |
| Random Oversampling | Random Forest | 0.759 | 0.648 | 0.455 | 0.535 |

All models evaluated with **80/20 train-test split** and 
**5-fold cross-validation**.

### 5. Class Balancing
- Compared three approaches: Original, SMOTE, Random Oversampling
- **SMOTE** produced the most balanced F1-score (0.582) with 
  better recall for the minority "Low Success" class
- Logistic Regression on original data gave highest raw accuracy 
  (76.9%) with AUC = 0.82

### 6. Hyperparameter Tuning
- **Gradient Boosting Regressor** tuned via GridSearchCV 
  (learning rate, max depth, n_estimators)
- **Random Forest (SMOTE)** tuned for best depth, split 
  criteria, estimator count
- Result: tuning confirmed models were already well-calibrated; 
  marginal precision gain, no significant accuracy change

---

## Key Findings

**Top predictors of company success (consistent across both tasks):**

| Driver | Direction | Signal Strength |
|---|---|---|
| Profit Margin | ↑ Positive | Strongest — #1 predictor in both tasks |
| Innovation Index | ↑ Positive | Strong — fuels growth and market resilience |
| Customer Satisfaction | ↑ Positive | Strong — sustains long-term performance |
| Work-Life Balance | ↑ Positive | Moderate — supports employee retention |
| Environmental Score | ↑ Positive | Moderate — CSR signals long-term stability |
| Expense Ratio | ↓ Negative | High costs erode success |
| Employee Turnover Rate | ↓ Negative | Instability limits performance |

**The single most decisive question:**  
*"What is your Profit Margin?"*  
— Profit Margin is the strongest and most consistent predictor 
across both regression and classification tasks.
