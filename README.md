# German Credit Risk Scoring Dashboard

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**Live demo:** _coming soon after deployment_

An interactive credit risk scoring dashboard built with **econometric methods** and **German macroeconomic data** from the ECB and Bundesbank. Designed to demonstrate applied econometrics, statistical modeling, and data pipeline skills for roles in **Risk Analytics, Quantitative Analysis, and Data Science** in the German finance/banking sector.

---

## What This Project Shows

| Skill | Demonstrated By |
|---|---|
| **Econometric modeling** | Logistic regression via MLE with full inference (p-values, CIs, AME, Pseudo R²) |
| **Python data engineering** | Automated data pipeline: fetch -> validate -> transform -> model -> visualize |
| **Statistical software** | statsmodels, scikit-learn, pandas, numpy |
| **Data visualization** | Interactive Plotly charts, Streamlit dashboard |
| **Finance domain knowledge** | Credit risk modeling, Schufa-style features, macro-financial linkages |
| **Reproducible research** | Full pipeline script, version-controlled, deployable |
| **API integration** | ECB Statistical Data Warehouse (SDW), Bundesbank public data |

---

## How It Works

```
ECB SDW API ─┐
Bundesbank   ─┤
               ├──> Macro Data ──> Feature Engineering ──> Logistic Regression ──> Streamlit Dashboard
Destatis     ─┘
                              \
                Synthetic Credit Applicant Data (calibrated to German market)
```

1. **Data pipeline** fetches German macroeconomic indicators (GDP, unemployment, inflation, interest rates, credit volumes) from public ECB/Bundesbank APIs.
2. **Synthetic applicant generation** creates 5,000 realistic loan applicants with features calibrated to the German credit market (Schufa-style).
3. **Logistic regression** estimates the probability of default with full statistical inference — coefficients, p-values, standard errors, and average marginal effects.
4. **Random Forest** benchmark for comparison.
5. **Streamlit dashboard** provides an interactive credit scoring tool with model diagnostics, visualizations, and methodology documentation.

---

## Quick Start

```bash
# Clone
git clone https://github.com/pehlivan-dagli/credit-risk-dashboard
cd credit-risk-dashboard

# Install (using uv for speed)
pip install -r requirements.txt

# Full pipeline
python scripts/run_pipeline.py

# Launch dashboard
streamlit run dashboard/app.py
```

Or skip model training if artifacts already exist:

```bash
python scripts/run_pipeline.py --quick
```

---

## Project Structure

```
credit-risk-dashboard/
├── data/
│   ├── fetch_german_data.py    # ECB/Bundesbank API + applicant generation
│   ├── raw/                    # Cached API responses
│   └── processed/              # Parquet datasets
├── model/
│   └── credit_scoring_model.py # Logit + Random Forest training, evaluation
├── dashboard/
│   └── app.py                  # Streamlit interactive dashboard
├── scripts/
│   └── run_pipeline.py         # End-to-end pipeline orchestrator
├── tests/
│   └── test_model.py           # Model validation tests
├── reports/
│   └── METHODOLOGY.md          # Detailed econometric write-up
├── requirements.txt
└── README.md
```

---

## Interactive Dashboard Features

- **Credit Score Simulator**: Adjust borrower characteristics and see real-time probability of default
- **Model Performance**: ROC curves, confusion matrix, metrics comparison
- **Statistical Inference**: Full regression output with p-values and significance levels
- **Marginal Effects**: Understand what drives default risk
- **Macroeconomic Context**: German economic indicators visualized
- **Methodology**: Explainable econometric approach vs. black-box ML

---

## Methodology

The core model is a **binary logistic regression** estimated via Maximum Likelihood:

\[
P(\text{default} = 1 \mid X) = \frac{1}{1 + e^{-X\beta}}
\]

where \(X\) includes borrower characteristics (age, income, DTI, employment status, housing, past defaults, loan characteristics) and macroeconomic conditions (unemployment rate, interest rates).

**Why logistic regression?** In German banking regulation (BaFin, EBA, MaRisk), credit risk models must be **interpretable, transparent, and auditable**. Every coefficient needs economic meaning. While ML methods (XGBoost, neural networks) may offer marginally higher AUC, regulators require explainability — making logit the industry standard for regulatory credit scoring. A Random Forest is included as a benchmark to show capability with non-linear methods.

See [METHODOLOGY.md](reports/METHODOLOGY.md) for the full econometric write-up.

---

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.10+ |
| Data | pandas, numpy, requests |
| Modeling | statsmodels, scikit-learn |
| Visualization | Plotly, matplotlib, seaborn |
| Dashboard | Streamlit |
| Testing | pytest |
| Deployment | Streamlit Cloud |

---

## Author

**Pehlivan Dağlı** — Master's student in Econometrics (NRW, Germany)

Working towards roles in Risk Analytics, Quantitative Analysis, and Data Science.

---

## License

MIT
