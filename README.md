---
title: German Credit Risk Dashboard
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# German Credit Risk Scoring Dashboard

A production-grade credit risk model with econometric methodology — built with Python, statsmodels, and Streamlit.

## Features

- Logistic Regression PD model with full statistical inference (coeffs, p-values, marginal effects)
- Random Forest benchmark (ROC-AUC 0.855)
- Interactive credit score simulator with German economic context
- ECB / Bundesbank macro data integration
- Regulatory-ready methodology (BaFIN / MaRisk aligned)

## How to run locally

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```
