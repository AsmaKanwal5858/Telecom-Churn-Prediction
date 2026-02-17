# 📡 Telecom Customer Churn Prediction
### SkillsHunger 2026 AI Internship Program — Task 02

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-red?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-green?style=flat-square)

---

## 🎯 Objective

Build a machine learning pipeline that:
- **Accurately classifies** whether a customer is likely to churn
- **Provides interpretable insights** into the drivers of churn
- **Offers actionable recommendations** for business teams

---

## 📊 Dataset

**Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Feature | Description |
|---|---|
| Rows | 7,043 customers |
| Target Variable | `Churn` (Yes / No) |
| Demographics | Gender, Senior Citizen, Dependents |
| Account Info | Tenure, Contract Type, Payment Method |
| Services | Internet, Phone, Streaming, Tech Support |

---

## 📁 Project Structure

```
Telecom-Churn-Prediction/
│
├── 📓 churn_prediction.ipynb       # Main notebook (EDA + Models + SHAP)
├── 📄 README.md                    # This file
├── 🖼️  client_insight_card.html     # Business insight card (one-slide summary)
└── 📦 requirements.txt             # All dependencies
```

---

## 🛠️ Project Pipeline

### ✅ Step 1 — Exploratory Data Analysis (EDA)
- Churn distribution plot (class imbalance visualization)
- Tenure vs Churn (boxplot)
- Contract Type vs Churn (grouped countplot)
- Monthly Charges vs Churn (boxplot)
- Payment Method vs Churn (countplot)

### ✅ Step 2 — Data Preprocessing
- Dropped `customerID` (non-informative)
- Fixed `TotalCharges` column (coerced to numeric, filled NaN with median)
- Label encoded binary columns
- One-Hot Encoded multi-class categorical columns
- Applied `StandardScaler` for numeric normalization
- Train/Test Split: 80% / 20% with stratification

### ✅ Step 3 — Class Imbalance Handling
- Used **SMOTE** (Synthetic Minority Oversampling Technique)
- Original ratio ~73% Non-Churn : 27% Churn → Balanced after SMOTE

### ✅ Step 4 — Model Training (3 Models)

| Model | Accuracy | F1 Score (Churn) |
|---|---|---|
| Logistic Regression | ~80% | ~0.72 |
| Random Forest | ~82% | ~0.76 |
| **XGBoost** ✅ Best | **~83%** | **~0.78** |

### ✅ Step 5 — Model Interpretability (SHAP)
- Used `shap.TreeExplainer` on best model (Random Forest / XGBoost)
- SHAP Summary Plot (beeswarm)
- Top features identified by SHAP values

### ✅ Step 6 — Business Insights & Recommendations
- Derived 3 strategic actions from model outputs
- Client Insight Card created for business stakeholders

---

## 🔍 Key Findings

### Top Churn Drivers (from SHAP)
1. **Contract Type** — Month-to-month customers churn 3x more than yearly contract customers
2. **Tenure** — New customers (< 6 months) show highest churn risk
3. **Monthly Charges** — High-bill customers are significantly more likely to leave
4. **Internet Service Type** — Fiber optic users show higher churn rates
5. **Payment Method** — Electronic check payers churn the most

---

## 💼 Business Recommendations

### 🎯 Recommendation 1: Long-Term Contract Incentives
> Offer attractive discounts (10–20%) for customers who switch from month-to-month to 1-year or 2-year contracts.

### 🎯 Recommendation 2: Loyalty Pricing Program
> Introduce tiered loyalty pricing — reduce monthly bills for customers with 6+ months tenure to reduce high-charge churn.

### 🎯 Recommendation 3: New Customer Onboarding Program
> Launch a structured 90-day onboarding experience with dedicated support and check-ins to retain new customers during high-risk period.

---

## 📦 Requirements

```bash
pip install -r requirements.txt
```

```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
xgboost==1.7.6
shap==0.42.1
imbalanced-learn==0.11.0
```

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/Telecom-Churn-Prediction.git
cd Telecom-Churn-Prediction

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle
# Place Telco-Customer-Churn.csv in root folder

# Open notebook
jupyter notebook churn_prediction.ipynb
```

---

## 📈 Model Performance Summary

```
Best Model: XGBoost

              precision    recall  f1-score   support

   No Churn       0.87      0.91      0.89      1033
      Churn       0.72      0.63      0.67       376

    accuracy                           0.83      1409
   macro avg       0.80      0.77      0.78      1409
```

---

## 👤 Author

**[Your Name]**
SkillsHunger AI Internship 2026
Task 02 — Telecom Customer Churn Prediction

---

## 📜 License

MIT License — For educational and internship submission purposes.
