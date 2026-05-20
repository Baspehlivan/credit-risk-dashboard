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
- Random Forest benchmark with cross-validated metrics
- Interactive credit score simulator with German economic context
- ECB / Bundesbank macro data integration
- Regulatory-ready methodology (BaFin / MaRisk aligned)
- CI pipeline with linting and tests

## Model Performance

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | ~0.85 | ~0.79 | ~0.65 | ~0.52 | ~0.58 |
| Random Forest | ~0.86 | ~0.80 | ~0.68 | ~0.54 | ~0.60 |

Metrics are from a single train/test split (80/20, stratified). Cross-validation with `cross_validate_models()` provides mean +/- std across 5 folds.

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Run the Full Pipeline

Fetch ECB data, train models, and save artifacts:

```bash
python scripts/run_pipeline.py               # full pipeline (5000 applicants)
python scripts/run_pipeline.py --quick        # skip training if models exist
python scripts/run_pipeline.py --force-refetch  # re-fetch ECB macro data
```

## Run Tests

```bash
pip install pytest ruff
pytest tests/ -v
ruff check .
```

## Project Structure

```
credit-risk-dashboard/
├── dashboard/
│   ├── app.py              # Streamlit dashboard (self-contained, loads pre-trained artifacts)
│   ├── data/               # Data artifacts bundled for HuggingFace Space
│   └── models/             # Model artifacts bundled for HuggingFace Space
├── data/
│   ├── fetch_german_data.py # ECB API fetcher + synthetic data generator
│   └── processed/          # Parquet datasets
├── model/
│   └── credit_scoring_model.py  # Logistic regression + Random Forest + cross-validation
├── scripts/
│   ├── run_pipeline.py     # End-to-end pipeline runner
│   └── save_artifacts.py   # Export model dicts for dashboard
├── tests/
│   └── test_model.py       # Pytest suite (data gen, feature eng, model training, CV)
├── reports/
│   └── METHODOLOGY.md      # Full econometric methodology write-up
├── .github/workflows/
│   └── ci.yml              # GitHub Actions: lint + test on Python 3.10-3.12
├── ruff.toml               # Linter config
├── Dockerfile              # HuggingFace Spaces deployment
└── requirements.txt
```

## Data Sources

| Source | Data | Access |
|---|---|---|
| [ECB Statistical Data Warehouse](https://sdw-wsrest.ecb.europa.eu/) | Interest rates, GDP, HICP, credit aggregates | REST API (CSV) |
| [Bundesbank](https://www.bundesbank.de/) | German credit and macro data | Public API |

Applicant-level data is synthetically generated with distributions calibrated to the German credit market. Default probabilities are conditioned on macroeconomic state (unemployment, GDP, interest rates).

## Methodology

The core model is a **logistic regression** estimated via Maximum Likelihood:

```
P(Y=1 | X) = 1 / (1 + exp(-X))
```

Full statistical inference is provided: coefficient significance (z-tests), McFadden Pseudo R-squared, AIC/BIC, and Average Marginal Effects. See [reports/METHODOLOGY.md](reports/METHODOLOGY.md) for the complete write-up.

**Why logistic regression?** German banking regulators (BaFin, EBA) require interpretable, transparent PD models for regulatory capital calculations. A Random Forest benchmark is included to demonstrate ML capability.

## Disclaimer

This is a demonstration project. The model is not validated for actual lending decisions.
