---

## 📊 Dataset Overview

| Property | Details |
|---|---|
| Total Samples | 200 |
| Features | 5 (age, tenure_months, monthly_spend, num_support_calls, avg_login_per_week) |
| Target | churn (0 = No Churn, 1 = Churn) |
| Class Distribution | 190 No Churn (95%) vs 10 Churn (5%) |
| Missing Values | None |

> ⚠️ **Severe class imbalance (19:1 ratio)** — addressed using SMOTE augmentation.

---

## 🔧 Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` & `seaborn` | Visualizations |
| `scikit-learn` | Model training, K-Fold CV, metrics |
| `imbalanced-learn` | SMOTE oversampling |

---

## 🚀 Approach

### 1. Exploratory Data Analysis (EDA)
- Class distribution analysis
- Feature correlation heatmap
- Boxplots comparing churned vs non-churned customers
- Identified key patterns in `monthly_spend` and `num_support_calls`

### 2. Preprocessing
- Separated features (X) and target (y)
- Applied `StandardScaler` for feature normalization

### 3. Baseline Model — Without SMOTE
- Algorithm: **Random Forest Classifier** (100 estimators)
- Validation: **Stratified K-Fold (K=5)**
- Evaluated on 5 evaluation metrics
- Result: High accuracy but near-zero Recall for churn class

### 4. Augmented Model — With SMOTE
- **SMOTE applied only inside each training fold** — prevents data leakage
- Test fold always remains original and untouched
- Same Random Forest model retrained on balanced training data
- Result: Significant improvement in Recall and F1-Score

---

## 📈 Results

| Metric | Baseline (No SMOTE) | With SMOTE | Change |
|---|---|---|---|
| Accuracy | ~0.95 | ~0.90 | ▼ slight drop |
| Precision | ~0.00 | ~improved | ▲ |
| Recall | ~0.00 | ~improved | ▲ significant |
| F1-Score | ~0.00 | ~improved | ▲ significant |
| ROC-AUC | ~0.50 | ~improved | ▲ |

> 📝 Exact values available in the notebook output (Cell 8 comparison table).

---

## 💡 Key Insights

- **Baseline model was misleading** — 95% accuracy was achieved by simply
  predicting "No Churn" for all samples due to class imbalance
- **SMOTE fixed the imbalance** by generating synthetic minority class
  samples, balancing each training fold from ~152:8 to ~152:152
- **No data leakage** — SMOTE was applied strictly inside training folds.
  The test fold was never seen or modified during augmentation
- **Recall is the most important metric** for churn prediction —
  missing a churner (False Negative) costs more than a false alarm
- **K=5 was chosen over K=10** intentionally — with only 10 churn
  samples, K=10 would result in ~1 churn sample per test fold,
  making metrics unreliable

---

## ▶️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

**3. Launch Jupyter Notebook**
```bash
jupyter notebook
```

**4. Open and run**
