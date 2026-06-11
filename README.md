# Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange?style=flat&logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat&logo=jupyter)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)

---

## Objective

Build a machine learning model to predict customer churn using a small,
imbalanced dataset. Apply **SMOTE data augmentation** and **K-Fold Cross
Validation** to improve model performance and ensure reliable evaluation.

---

## Repository Structure

    customer-churn-prediction/
    │
    ├── churn_analysis.ipynb      
    ├── churn_dataset.csv         
    └── README.md                 

---

## Dataset Overview

| Property | Details |
|---|---|
| Total Samples | 200 |
| Features | 5 (age, tenure_months, monthly_spend, num_support_calls, avg_login_per_week) |
| Target | churn (0 = No Churn, 1 = Churn) |
| Class Distribution | 190 No Churn (95%) vs 10 Churn (5%) |
| Missing Values | None |

> Severe class imbalance (19:1 ratio) — addressed using SMOTE augmentation.

---

## Tech Stack

| Library | Purpose |
|---|---|
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| matplotlib and seaborn | Visualizations |
| scikit-learn | Model training, K-Fold CV, metrics |
| imbalanced-learn | SMOTE oversampling |

---

## Approach

**1. Exploratory Data Analysis**
- Class distribution analysis
- Feature correlation heatmap
- Boxplots comparing churned vs non-churned customers

**2. Preprocessing**
- Separated features (X) and target (y)
- Applied StandardScaler for feature normalization

**3. Baseline Model — Without SMOTE**
- Algorithm: Random Forest Classifier (100 estimators)
- Validation: Stratified K-Fold (K=5)
- Result: High accuracy but near-zero Recall for churn class

**4. Augmented Model — With SMOTE**
- SMOTE applied only inside each training fold — prevents data leakage
- Test fold always remains original and untouched
- Result: Significant improvement in Recall and F1-Score

---

## Results

| Metric | Baseline (No SMOTE) | With SMOTE |
|---|---|---|
| Accuracy | ~0.95 | ~0.90 |
| Precision | ~0.00 | improved |
| Recall | ~0.00 | significantly improved |
| F1-Score | ~0.00 | significantly improved |
| ROC-AUC | ~0.50 | improved |

> Exact values available in the notebook output.

---

## Key Insights

- Baseline model was misleading — 95% accuracy achieved by predicting No Churn for everything
- SMOTE balanced each training fold from 152:8 to 152:152
- No data leakage — SMOTE applied strictly inside training folds only
- Recall is the most important metric for churn prediction
- K=5 chosen over K=10 — with only 10 churn samples, K=10 gives unstable metrics

---

## How to Run

**1. Clone the repository**

    git clone https://github.com/your-username/customer-churn-prediction.git
    cd customer-churn-prediction

**2. Install dependencies**

    pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter

**3. Launch Jupyter Notebook**

    jupyter notebook

**4. Open and run**

    Open churn_analysis.ipynb → Kernel → Restart & Run All

---

## Deliverables

- [x] Jupyter Notebook with complete code and outputs
- [x] EDA visualizations
- [x] Baseline model with K-Fold cross validation
- [x] SMOTE-augmented model with leakage-safe implementation
- [x] Before vs After comparison table and charts
- [x] Confusion matrices for both models
- [x] Final report summary with insights

---

## Author

**Pragya R K**
B.E. — Artificial Intelligence and Machine Learning
Bapuji Institute of Engineering and Technology (BIET), Davanagere

