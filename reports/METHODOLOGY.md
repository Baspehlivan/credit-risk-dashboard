# Econometric Methodology: Credit Risk Scoring

## 1. Model Framework

### 1.1 Binary Logistic Regression

The probability of loan default is modeled using a **binary logistic regression** estimated via **Maximum Likelihood Estimation (MLE)**:

```
P(Y_i = 1 | X_i) = Λ(X_i'β) = 1 / (1 + exp(-X_i'β))
```

where:
- **Y_i** = 1 if applicant i defaults, 0 otherwise
- **X_i** = vector of borrower characteristics and macroeconomic conditions
- **β** = coefficient vector estimated by maximizing the log-likelihood
- **Λ(·)** = logistic cumulative distribution function

### 1.2 Log-Likelihood Function

```
ℓ(β) = Σ_i [Y_i · ln(Λ(X_i'β)) + (1 - Y_i) · ln(1 - Λ(X_i'β))]
```

Maximized numerically using the Newton-Raphson algorithm.

### 1.3 Statistical Inference

For each coefficient β_k:
- **Standard error**: square root of the k-th diagonal element of the inverse Fisher information matrix
- **z-statistic**: z_k = β_k / SE(β_k), asymptotically N(0,1) under H₀: β_k = 0
- **p-value**: 2 · Φ(-|z_k|) for two-sided test
- **Confidence interval**: β_k ± z_{α/2} · SE(β_k)

## 2. Model Fit Measures

| Measure | Formula | Interpretation |
|---|---|---|
| **McFadden Pseudo R²** | 1 - ℓ_fit / ℓ_null | 0 (no fit) to 1 (perfect); values > 0.2 indicate strong fit |
| **AIC** | -2ℓ_fit + 2k | Lower is better; penalizes complexity |
| **BIC** | -2ℓ_fit + k·ln(N) | Lower is better; stronger penalty for complexity |
| **Log-Likelihood** | ℓ(β̂) | Higher (less negative) is better |

## 3. Average Marginal Effects (AME)

Unlike linear regression, coefficients in logit are not directly interpretable as marginal effects due to the non-linear link function. The **Average Marginal Effect** for variable x_k is:

```
AME_k = (1/N) · Σ_i [Λ(X_i'β̂) · (1 - Λ(X_i'β̂)) · β̂_k]
```

This represents the **average change in predicted default probability** for a one-unit increase in x_k, holding all other variables at their observed values.

## 4. Model Evaluation

### 4.1 Discriminatory Power

- **ROC-AUC**: Area Under the Receiver Operating Characteristic curve. Measures the model's ability to rank defaulters vs. non-defaulters.
  - AUC = 0.5: random
  - AUC = 0.7–0.8: acceptable
  - AUC = 0.8–0.9: excellent
  - AUC > 0.9: outstanding

### 4.2 Classification Performance

At a threshold τ (default = 0.5):

| Metric | Formula |
|---|---|
| **Accuracy** | (TP + TN) / N |
| **Precision** | TP / (TP + FP) |
| **Recall (TPR)** | TP / (TP + FN) |
| **F1-Score** | 2 · (Precision · Recall) / (Precision + Recall) |

where TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives.

### 4.3 Confusion Matrix

```
                    Predicted
                0 (No Default)  1 (Default)
Actual  0 (No Default)     TN           FP
        1 (Default)        FN           TP
```

## 5. Feature Set

### 5.1 Borrower Characteristics

| Feature | Type | Expected Sign |
|---|---|---|
| Age | Continuous (years) | − (older = more stable) |
| Monthly Income | Continuous (log) | − (higher income = lower risk) |
| Debt-to-Income Ratio | Continuous [0,1] | + (more debt = higher risk) |
| Credit History Length | Continuous (years) | − (longer history = more reliable) |
| Past Defaults | Count [0,5] | + (past default = future default) |
| Employment Status | Categorical (4 levels) | − for employed, + for unemployed |
| Home Ownership | Categorical (3 levels) | − for owners |
| Marital Status | Categorical (3 levels) | − for married |
| Loan Amount | Continuous (log) | + (larger loan = more risk) |
| Loan Term | Continuous (months) | + (longer term = more uncertainty) |
| Loan-to-Income Ratio | Continuous | + (stretched borrower = higher risk) |

### 5.2 Macroeconomic Conditions

| Feature | Source | Expected Sign |
|---|---|---|
| Unemployment Rate | ECB SDW / Bundesbank | + (worse economy = more defaults) |
| GDP | ECB SDW / Destatis | − (growing economy = fewer defaults) |
| Short-term Interest Rate | ECB SDW | + (higher rates = debt burden) |

## 6. Why Not Machine Learning?

In German banking regulation (MaRisk, EBA Guidelines on PD estimation, BaFin expectations), credit risk models must be:

1. **Interpretable**: Every coefficient must have economic meaning
2. **Transparent**: Model behavior must be explainable to auditors
3. **Stable**: GLMs generalize better on small-to-medium credit portfolios
4. **Regulatory compliant**: Basel IRB approach requires PD models to be "not based purely on statistical fit"

While ML methods (XGBoost, neural networks) may produce higher AUC, they are often rejected for regulatory capital calculations. However, this project includes a Random Forest benchmark to demonstrate ML capability for non-regulatory applications.

## 7. Data Sources

| Source | Data | Access Method |
|---|---|---|
| [ECB SDW](https://sdw-wsrest.ecb.europa.eu/) | Interest rates, GDP, HICP, credit aggregates | REST API (CSV) |
| [Bundesbank](https://www.bundesbank.de/) | German credit and macro data | Public API |
| [Destatis](https://www.destatis.de/) | German economic indicators | Public data |

Applicant-level data is synthetically generated with distributions calibrated to the German credit market using the macro context as a conditioning factor.

## 8. Software

- **statsmodels** (Logit, MLE, inference)
- **scikit-learn** (Random Forest, evaluation metrics)
- **pandas** / **numpy** (data manipulation)
- **Streamlit** / **Plotly** (interactive dashboard)

---

*Methodology written for the German Credit Risk Dashboard project. For questions: open an issue on GitHub.*
