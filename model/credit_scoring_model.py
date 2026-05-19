"""Credit scoring model using logistic regression with full statistical inference.

Uses statsmodels.Logit for:
  - Coefficient estimates with p-values, standard errors, confidence intervals
  - Pseudo R-squared, log-likelihood, AIC/BIC
  - Marginal effects (AME)
  - Confusion matrix, ROC-AUC, precision-recall

Also includes a Random Forest as a non-linear benchmark.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---- Modelling ----
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
import joblib

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Feature engineering ----
FEATURES = [
    "age",
    "employment_status",
    "monthly_income",
    "dti_ratio",
    "credit_history_years",
    "past_defaults",
    "marital_status",
    "home_ownership",
    "dependents",
    "loan_purpose",
    "loan_amount",
    "loan_term_months",
]

CATEGORICAL_FEATURES = {
    "employment_status": "Employment (0=Employed, 1=Self-Emp, 2=Student, 3=Unemp)",
    "marital_status": "Marital (0=Single, 1=Married, 2=Divorced)",
    "home_ownership": "Housing (0=Rent, 1=Own, 2=Parents)",
    "loan_purpose": "Loan Purpose (0=Car, 1=Educ, 2=Home, 3=Personal, 4=Business)",
}


def _prepare_data(df: pd.DataFrame, with_macro: bool = False) -> pd.DataFrame:
    """Create feature matrix with dummy variables for categorical features."""
    X = df[FEATURES].copy()

    # Convert categoricals to dummies
    for col in [
        "employment_status",
        "marital_status",
        "home_ownership",
        "loan_purpose",
    ]:
        dummies = pd.get_dummies(X[col], prefix=col, drop_first=True).astype(float)
        X = pd.concat([X.drop(columns=[col]), dummies], axis=1)

    # Log transform skewed variables
    X["log_income"] = np.log(X["monthly_income"].clip(lower=1))
    X["log_loan_amount"] = np.log(X["loan_amount"].clip(lower=1))
    X["loan_to_income"] = X["loan_amount"] / X["monthly_income"].clip(lower=1)
    X["loan_to_income"] = X["loan_to_income"].clip(upper=10)

    # Drop raw versions of transformed vars
    X = X.drop(columns=["monthly_income", "loan_amount"])

    # Optional macro features
    if with_macro and "_macro_unemp" in df.columns:
        X["macro_unemp"] = df["_macro_unemp"].fillna(df["_macro_unemp"].median())
        X["macro_short_rate"] = df["_macro_short_rate"].fillna(
            df["_macro_short_rate"].median()
        )

    return X


def train_logistic(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> dict:
    """Train logistic regression via statsmodels. Returns a dict with model + metrics."""
    print("=" * 60)
    print("Logistic Regression — Credit Scoring Model")
    print("=" * 60)

    # Prepare
    X = _prepare_data(df)
    y = df["default"].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Statsmodels Logit — add constant
    X_train_sm = sm.add_constant(X_train.astype(float))
    X_test_sm = sm.add_constant(X_test.astype(float))

    # Fit model
    print(f"\nTraining on {len(X_train)} observations...")
    logit_model = sm.Logit(y_train, X_train_sm)
    logit_result = logit_model.fit(disp=False, maxiter=1000)

    # ---- Statistical output ----
    print("\n--- Model Summary ---")
    print(f"  Pseudo R-squared (McFadden): {logit_result.prsquared:.4f}")
    print(f"  Log-Likelihood:              {logit_result.llf:.2f}")
    print(f"  AIC:                         {logit_result.aic:.2f}")
    print(f"  BIC:                         {logit_result.bic:.2f}")
    print(f"  N (train):                   {int(logit_result.nobs)}")
    print(f"  N (test):                    {len(X_test)}")

    print("\n--- Coefficients ---")
    summary = logit_result.summary2().tables[1]
    for idx, row in summary.iterrows():
        sig = ""
        if row["P>|z|"] < 0.01:
            sig = "***"
        elif row["P>|z|"] < 0.05:
            sig = "**"
        elif row["P>|z|"] < 0.1:
            sig = "*"
        print(f"  {idx:30s} {row['Coef.']:8.4f}  p={row['P>|z|']:.4f} {sig}")

    # ---- Predictions ----
    y_prob = logit_result.predict(X_test_sm)
    y_pred = (y_prob >= 0.5).astype(int)

    # ---- Metrics ----
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "prsquared": logit_result.prsquared,
        "aic": logit_result.aic,
        "bic": logit_result.bic,
    }

    print("\n--- Test Set Metrics ---")
    for k, v in metrics.items():
        if k in ("prsquared", "aic", "bic"):
            continue
        print(f"  {k:15s}: {v:.4f}")

    # ---- Confusion matrix ----
    cm = pd.crosstab(
        y_test,
        y_pred,
        rownames=["Actual"],
        colnames=["Predicted"],
    )
    print(f"\n--- Confusion Matrix ---\n{cm}")

    # ---- Marginal Effects (Average Marginal Effects) ----
    print("\n--- Average Marginal Effects (AME) ---")
    # AME: average of dF/dX over all observations
    # For Logit: dF/dX = f(X*beta) * beta, where f() is the logistic PDF
    X_all_sm = sm.add_constant(X.astype(float))
    fitted = logit_result.predict(X_all_sm)
    ame_factor = (fitted * (1 - fitted)).mean()
    for var in logit_result.params.index:
        if var == "const":
            continue
        ame = logit_result.params[var] * ame_factor
        print(f"  {var:30s} AME = {ame:8.6f}")

    # ---- Store results ----
    result = {
        "model": logit_result,
        "X_train": X_train_sm,
        "X_test": X_test_sm,
        "y_train": y_train,
        "y_test": y_test,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "metrics": metrics,
        "feature_names": list(X.columns),
        "type": "logistic",
    }
    return result


def train_random_forest(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> dict:
    """Train a Random Forest as a non-linear benchmark model."""
    print("\n" + "=" * 60)
    print("Random Forest — Benchmark Model")
    print("=" * 60)

    X = _prepare_data(df)
    y = df["default"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    print(f"\nTraining on {len(X_train)} observations...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=20,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    y_prob = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    print("\n--- Test Set Metrics ---")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}")

    # Feature importance
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)
    print(f"\n--- Top 10 Features ---\n{importance.head(10).to_string(index=False)}")

    return {
        "model": rf,
        "X_test": X_test,
        "y_test": y_test,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "metrics": metrics,
        "feature_importance": importance,
        "type": "random_forest",
    }


def save_models(logit_result: dict, rf_result: dict | None = None):
    """Save model artifacts to disk."""
    # Save logistic regression (pickle the result object)
    logit_path = MODELS_DIR / "logistic_model.joblib"
    joblib.dump(logit_result["model"], logit_path)
    print(f"\nLogistic model saved to {logit_path}")

    # Save feature names
    feat_path = MODELS_DIR / "feature_names.joblib"
    joblib.dump(logit_result["feature_names"], feat_path)

    if rf_result:
        rf_path = MODELS_DIR / "random_forest.joblib"
        joblib.dump(rf_result["model"], rf_path)
        print(f"Random forest saved to {rf_path}")


def run_pipeline(n_applicants: int = 5000) -> dict:
    """Run the full model pipeline.

    1. Load dataset (builds it if not found)
    2. Train logistic regression
    3. Train random forest benchmark
    4. Save models
    """
    data_path = PROCESSED_DIR / "credit_applicants.parquet"
    if not data_path.exists():
        print("Dataset not found. Building it first...")
        from data.fetch_german_data import build_dataset

        build_dataset(n_applicants=n_applicants)
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df)} applicants from {data_path}")

    logit = train_logistic(df)
    rf = train_random_forest(df)
    save_models(logit, rf)

    return {"logistic": logit, "random_forest": rf, "data": df}


if __name__ == "__main__":
    results = run_pipeline()
