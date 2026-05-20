"""Save full training artifacts for instant dashboard loading."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from model.credit_scoring_model import MODELS_DIR, PROCESSED_DIR, _prepare_data

print("Loading dataset...")
df = pd.read_parquet(PROCESSED_DIR / "credit_applicants.parquet")

print("Computing features...")
X = _prepare_data(df)
y = df["default"].values

# Split (same seed as training)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Load models
logit_model = joblib.load(MODELS_DIR / "logistic_model.joblib")
rf_model = joblib.load(MODELS_DIR / "random_forest.joblib")

# Logit predictions
X_train_sm = sm.add_constant(X_train.astype(float), has_constant="add")
X_test_sm = sm.add_constant(X_test.astype(float), has_constant="add")

y_train_prob = logit_model.predict(X_train_sm)
y_test_prob = logit_model.predict(X_test_sm)
y_test_pred = (y_test_prob >= 0.5).astype(int)

# RF predictions
y_test_prob_rf = rf_model.predict_proba(X_test)[:, 1]
y_test_pred_rf = (y_test_prob_rf >= 0.5).astype(int)


def compute_metrics(y_true, y_pred, y_prob, model=None):
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }
    if model is not None:
        m["prsquared"] = model.prsquared
        m["llf"] = model.llf
        m["aic"] = model.aic
        m["bic"] = model.bic
    return m


logit_dict = {
    "model": logit_model,
    "X_train": X_train_sm,
    "y_test": y_test,
    "y_pred": y_test_pred,
    "y_prob": y_test_prob,
    "metrics": compute_metrics(y_test, y_test_pred, y_test_prob, logit_model),
    "feature_names": joblib.load(MODELS_DIR / "feature_names.joblib"),
}

rf_dict = {
    "model": rf_model,
    "X_train": X_train,
    "y_test": y_test,
    "y_pred": y_test_pred_rf,
    "y_prob": y_test_prob_rf,
    "metrics": compute_metrics(y_test, y_test_pred_rf, y_test_prob_rf),
    "feature_importance": pd.DataFrame(
        {
            "feature": joblib.load(MODELS_DIR / "feature_names.joblib"),
            "importance": rf_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False),
}

# Save full dicts
joblib.dump(logit_dict, MODELS_DIR / "logit_dict.joblib")
joblib.dump(rf_dict, MODELS_DIR / "rf_dict.joblib")
print("Artifacts saved.")
print(f"Logit AUC: {logit_dict['metrics']['roc_auc']:.3f}")
print(f"RF AUC: {rf_dict['metrics']['roc_auc']:.3f}")
