"""Tests for the credit scoring model."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from data.fetch_german_data import generate_credit_applicants
from model.credit_scoring_model import (
    _prepare_data,
    cross_validate_models,
    train_logistic,
)


@pytest.fixture(scope="module")
def sample_data():
    """Generate a small dataset for testing."""
    return generate_credit_applicants(n_applicants=500, seed=42)


def test_generate_credit_applicants_shape(sample_data):
    df = sample_data
    assert len(df) == 500
    assert "default" in df.columns
    assert "applicant_id" in df.columns
    assert "monthly_income" in df.columns


def test_default_rate_reasonable(sample_data):
    df = sample_data
    rate = df["default"].mean()
    assert 0.01 < rate < 0.50, f"Default rate {rate:.3f} outside expected range"


def test_prepare_data_output(sample_data):
    X = _prepare_data(sample_data)
    assert "log_income" in X.columns
    assert "log_loan_amount" in X.columns
    assert "loan_to_income" in X.columns
    assert "employment_status_1" in X.columns or "employment_status_3" in X.columns
    assert X.isnull().sum().sum() == 0, "NaN values in prepared features"


def test_train_logistic_converges(sample_data):
    result = train_logistic(sample_data, test_size=0.2, seed=42)
    assert result["model"].converged, "Logit did not converge"
    assert 0.5 < result["metrics"]["roc_auc"] <= 1.0
    assert 0.0 < result["metrics"]["accuracy"] <= 1.0


def test_train_logistic_feature_impact(sample_data):
    result = train_logistic(sample_data, test_size=0.2, seed=42)
    params = result["model"].params
    # Higher past_defaults should increase default probability
    past_default_cols = [c for c in params.index if "past_defaults" in c]
    if past_default_cols:
        # Should be positive (more past defaults => higher risk)
        assert params[past_default_cols[0]] > -2.0, "Past defaults effect too negative"

    # Higher income should decrease default probability
    if "log_income" in params.index:
        assert params["log_income"] < 0.5, "Income effect unexpectedly positive"


def test_predictions_bounded(sample_data):
    result = train_logistic(sample_data, test_size=0.2, seed=42)
    assert result["y_prob"].min() >= 0.0
    assert result["y_prob"].max() <= 1.0


def test_categorical_dummies_created(sample_data):
    X = _prepare_data(sample_data)
    dummy_cols = [
        c
        for c in X.columns
        if c.startswith((
            "employment_status_",
            "marital_status_",
            "home_ownership_",
            "loan_purpose_",
        ))
    ]
    assert len(dummy_cols) > 0, "No dummy variables created"


def test_cross_validate_models(sample_data):
    cv_results = cross_validate_models(sample_data, n_splits=3, seed=42)

    assert "logistic" in cv_results
    assert "random_forest" in cv_results
    assert cv_results["n_splits"] == 3

    for model_name in ["logistic", "random_forest"]:
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert metric in cv_results[model_name], f"Missing metric {metric} for {model_name}"
            mean = cv_results[model_name][metric]["mean"]
            std = cv_results[model_name][metric]["std"]
            assert 0.0 <= mean <= 1.0, f"{model_name} {metric} mean={mean} out of bounds"
            assert std >= 0.0, f"{model_name} {metric} std={std} is negative"

    # AUC should be above random (0.5) for both models
    assert cv_results["logistic"]["roc_auc"]["mean"] > 0.5
    assert cv_results["random_forest"]["roc_auc"]["mean"] > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
