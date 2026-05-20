"""Streamlit dashboard: German Credit Risk Scoring with Econometric Methodology.
Self-contained — loads pre-trained artifacts only. No model training at startup.
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import roc_curve

# ---- Paths (relative to this file) ----
HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
DATA_DIR = HERE / "data"

LOGIT_DICT_PATH = MODELS_DIR / "logit_dict.joblib"
RF_DICT_PATH = MODELS_DIR / "rf_dict.joblib"
DATA_PATH = DATA_DIR / "credit_applicants.parquet"
MACRO_PATH = DATA_DIR / "macro_data.parquet"


# ---- Page config ----
st.set_page_config(
    page_title="German Credit Risk Dashboard",
    page_icon=":bank:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS ----
st.markdown(
    """
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.4rem; font-weight: 500; color: #555; }
    .stat-box { background: #f0f2f6; border-radius: 8px; padding: 1rem; text-align: center; }
    .stat-value { font-size: 1.8rem; font-weight: 700; }
    .stat-label { font-size: 0.85rem; color: #666; }
    .methodology { background: #f8f9fa; border-left: 4px solid #1f77b4; padding: 1rem 1.5rem; margin: 1rem 0; border-radius: 0 8px 8px 0; }
    .risk-low { color: #28a745; font-weight: 700; }
    .risk-moderate { color: #ffc107; font-weight: 700; }
    .risk-high { color: #dc3545; font-weight: 700; }
    .header-container { text-align: center; padding: 1rem 0; border-bottom: 2px solid #e0e0e0; margin-bottom: 2rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ---- Load pre-trained artifacts (instant startup) ----


@st.cache_resource
def load_models():
    logit = joblib.load(LOGIT_DICT_PATH)
    rf = joblib.load(RF_DICT_PATH)
    return logit, rf


@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    macro = None
    if MACRO_PATH.exists():
        macro = pd.read_parquet(MACRO_PATH)
    return df, macro


df, macro = load_data()
logit, rf = load_models()

# ---- Feature engineering (replicates _prepare_data inline) ----

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


def prepare_data(df_input, with_macro=False):
    """Create feature matrix matching training format."""
    X = df_input[FEATURES].copy()
    for col in [
        "employment_status",
        "marital_status",
        "home_ownership",
        "loan_purpose",
    ]:
        dummies = pd.get_dummies(X[col], prefix=col, drop_first=True).astype(float)
        X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
    X["log_income"] = np.log(X["monthly_income"].clip(lower=1))
    X["log_loan_amount"] = np.log(X["loan_amount"].clip(lower=1))
    X["loan_to_income"] = X["loan_amount"] / X["monthly_income"].clip(lower=1)
    X["loan_to_income"] = X["loan_to_income"].clip(upper=10)
    X = X.drop(columns=["monthly_income", "loan_amount"])
    if with_macro and "_macro_unemp" in df_input.columns:
        X["macro_unemp"] = df_input["_macro_unemp"].fillna(
            df_input["_macro_unemp"].median()
        )
        X["macro_short_rate"] = df_input["_macro_short_rate"].fillna(
            df_input["_macro_short_rate"].median()
        )
    return X


# =====================================================================
# HEADER
# =====================================================================
st.markdown(
    f'<div class="header-container">'
    f'<div class="main-header">German Credit Risk Scoring</div>'
    f'<div class="sub-header">Applied Econometrics for Consumer Lending</div>'
    f'<p style="margin-top:0.5rem;color:#888;">Dataset: {len(df):,} loan applicants | '
    f"Default rate: {df['default'].mean():.1%} | "
    f"Period: {df['application_date'].min().year}–{df['application_date'].max().year}</p>"
    f"</div>",
    unsafe_allow_html=True,
)

# =====================================================================
# INTERACTIVE CREDIT SCORE SIMULATOR
# =====================================================================
st.markdown("## Assess Your Credit Score")
st.markdown(
    "Adjust the sliders and observe how borrower characteristics affect the estimated "
    "**Probability of Default (PD)** based on the econometric model."
)

col1, col2 = st.columns(2)

with col1:
    monthly_income = st.slider("Monthly Income (EUR)", 500, 15000, 3200, step=100)
    age = st.slider("Age", 18, 75, 32)
    credit_history = st.slider("Credit History (years)", 0, 40, 5)
    dti = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.25, step=0.01)
    past_defaults = st.slider("Past Defaults", 0, 5, 0)

with col2:
    employment = st.selectbox(
        "Employment Status",
        options=["Employed", "Self-Employed", "Student", "Unemployed"],
    )
    emp_map = {"Employed": 0, "Self-Employed": 1, "Student": 2, "Unemployed": 3}
    housing = st.selectbox("Housing", options=["Rent", "Own", "With Parents"])
    housing_map = {"Rent": 0, "Own": 1, "With Parents": 2}
    marital = st.selectbox("Marital Status", options=["Single", "Married", "Divorced"])
    marital_map = {"Single": 0, "Married": 1, "Divorced": 2}
    purpose = st.selectbox(
        "Loan Purpose", options=["Car", "Education", "Home", "Personal", "Business"]
    )
    purpose_map = {"Car": 0, "Education": 1, "Home": 2, "Personal": 3, "Business": 4}
    loan_amount = st.number_input(
        "Loan Amount (EUR)", min_value=500, max_value=500000, value=15000, step=500
    )
    loan_term = st.selectbox(
        "Loan Term (months)",
        options=[12, 24, 36, 48, 60, 84, 120, 180, 240, 360],
        index=4,
    )

# Build applicant vector
applicant = pd.DataFrame(
    [
        {
            "age": age,
            "employment_status": emp_map[employment],
            "monthly_income": monthly_income,
            "dti_ratio": dti,
            "credit_history_years": credit_history,
            "past_defaults": past_defaults,
            "marital_status": marital_map[marital],
            "home_ownership": housing_map[housing],
            "dependents": 0,
            "loan_purpose": purpose_map[purpose],
            "loan_amount": loan_amount,
            "loan_term_months": loan_term,
        }
    ]
)

X_app = prepare_data(applicant)
# Add constant column
X_app_sm = X_app.copy()
X_app_sm.insert(0, "const", 1.0)

# Align columns with training set
for col in logit["X_train"].columns:
    if col not in X_app_sm.columns:
        X_app_sm[col] = 0.0
X_app_sm = X_app_sm[logit["X_train"].columns]

prob = logit["model"].predict(X_app_sm)[0]
score = int((1 - prob) * 1000)

# Result cards
st.markdown("---")
res_col1, res_col2, res_col3, res_col4 = st.columns(4)

if prob < 0.10:
    risk_class, risk_color = "Low Risk", "risk-low"
elif prob < 0.25:
    risk_class, risk_color = "Moderate Risk", "risk-moderate"
else:
    risk_class, risk_color = "High Risk", "risk-high"

with res_col1:
    st.markdown(
        f'<div class="stat-box"><div class="stat-value">{score}</div><div class="stat-label">Credit Score (0–1000)</div></div>',
        unsafe_allow_html=True,
    )
with res_col2:
    st.markdown(
        f'<div class="stat-box"><div class="stat-value">{prob:.1%}</div><div class="stat-label">Probability of Default</div></div>',
        unsafe_allow_html=True,
    )
with res_col3:
    st.markdown(
        f'<div class="stat-box"><div class="{risk_color}" style="font-size:1.8rem;font-weight:700;">{risk_class}</div><div class="stat-label">Risk Classification</div></div>',
        unsafe_allow_html=True,
    )
with res_col4:
    contrib = logit["model"].params.drop("const") * X_app_sm.values[0][1:]
    top_feat = contrib.abs().idxmax()
    st.markdown(
        f'<div class="stat-box"><div class="stat-value" style="font-size:1.0rem;">{top_feat}</div><div class="stat-label">Top Driver</div></div>',
        unsafe_allow_html=True,
    )

# =====================================================================
# MODEL PERFORMANCE
# =====================================================================
st.markdown("## Model Performance")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Metrics Comparison",
        "ROC Curve",
        "Coefficients",
        "Feature Importance",
        "Confusion Matrix",
    ]
)

with tab1:
    col_m1, col_m2, col_m3 = st.columns(3)
    for col, (name, res) in zip(
        [col_m1, col_m2], [("Logistic Regression", logit), ("Random Forest", rf)]
    ):
        with col:
            st.markdown(f"**{name}**")
            m = res["metrics"]
            for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
                st.metric(k.replace("_", " ").title(), f"{m[k]:.3f}")
    with col_m3:
        st.markdown("**Logit Model Fit**")
        st.metric("Pseudo R", f"{logit['metrics']['prsquared']:.4f}")
        st.metric("AIC", f"{logit['metrics']['aic']:.0f}")
        st.metric("BIC", f"{logit['metrics']['bic']:.0f}")

with tab2:
    fig_roc = go.Figure()
    for name, res in [("Logistic", logit), ("Random Forest", rf)]:
        fpr, tpr, _ = roc_curve(res["y_test"], res["y_prob"])
        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{name} (AUC={res['metrics']['roc_auc']:.3f})",
            )
        )
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Random",
        )
    )
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=700,
        height=500,
        title="ROC Curve",
    )
    st.plotly_chart(fig_roc, use_container_width=True)

with tab3:
    st.markdown("**Logistic Regression Coefficients (statsmodels)**")
    summary = logit["model"].summary2().tables[1]
    summary["Sig."] = summary["P>|z|"].apply(
        lambda p: (
            "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        )
    )
    styled = summary.style.background_gradient(
        subset=["Coef."], cmap="RdBu_r", vmin=-1, vmax=1
    )
    st.dataframe(styled, use_container_width=True)
    st.caption(
        "*** p<0.01, ** p<0.05, * p<0.1. Positive coefficients increase default probability."
    )

with tab4:
    imp = rf["feature_importance"].head(15)
    fig_imp = px.bar(
        imp,
        x="importance",
        y="feature",
        orientation="h",
        title="Random Forest - Top 15 Feature Importances",
        color="importance",
        color_continuous_scale="Blues",
    )
    fig_imp.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    st.plotly_chart(fig_imp, use_container_width=True)

with tab5:
    cm = pd.crosstab(
        logit["y_test"], logit["y_pred"], rownames=["Actual"], colnames=["Predicted"]
    )
    fig_cm = px.imshow(
        cm.values,
        x=["Predicted: No Default", "Predicted: Default"],
        y=["Actual: No Default", "Actual: Default"],
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix - Logistic Regression",
    )
    fig_cm.update_layout(width=500, height=400)
    st.plotly_chart(fig_cm, use_container_width=True)

# =====================================================================
# DATA EXPLORATION
# =====================================================================
st.markdown("## Data Exploration")

tab_d1, tab_d2, tab_d3 = st.tabs(["Default Drivers", "Macro Context", "Distribution"])

with tab_d1:
    st.markdown(
        "**What drives default?** Average Marginal Effects (AME) from the logit model."
    )
    fitted = logit["model"].predict(logit["X_train"])
    ame_factor = (fitted * (1 - fitted)).mean()
    ame_data = []
    for var in logit["model"].params.index:
        if var == "const":
            continue
        ame = logit["model"].params[var] * ame_factor
        ame_data.append({"Variable": var, "AME": ame})
    ame_df = (
        pd.DataFrame(ame_data).sort_values("AME", key=abs, ascending=False).head(12)
    )
    fig_ame = px.bar(
        ame_df,
        x="AME",
        y="Variable",
        orientation="h",
        title="Average Marginal Effects - Top 12",
        color="AME",
        color_continuous_scale="RdBu_r",
    )
    fig_ame.update_layout(yaxis={"categoryorder": "total ascending"}, height=450)
    st.plotly_chart(fig_ame, use_container_width=True)

with tab_d2:
    if macro is not None and len(macro) > 0:
        st.markdown("**German macroeconomic indicators used as model context.**")
        macro_long = macro.reset_index().melt(
            id_vars="index", var_name="Indicator", value_name="Value"
        )
        macro_long.columns = ["Date", "Indicator", "Value"]
        fig_macro = px.line(
            macro_long,
            x="Date",
            y="Value",
            color="Indicator",
            title="German Macroeconomic Indicators",
        )
        fig_macro.update_layout(height=450)
        st.plotly_chart(fig_macro, use_container_width=True)
    else:
        st.info(
            "Macro data not available. Run locally with internet access to fetch ECB/Bundesbank data."
        )

with tab_d3:
    feat_to_plot = st.selectbox(
        "Feature",
        ["monthly_income", "age", "dti_ratio", "credit_history_years", "loan_amount"],
    )
    fig_hist = px.histogram(
        df,
        x=feat_to_plot,
        color="default",
        barmode="overlay",
        opacity=0.6,
        title=f"{feat_to_plot} by Default Status",
        labels={
            "default": "Defaulted",
            feat_to_plot: feat_to_plot.replace("_", " ").title(),
        },
        color_discrete_map={0: "#1f77b4", 1: "#d62728"},
    )
    fig_hist.update_layout(height=400)
    st.plotly_chart(fig_hist, use_container_width=True)

# =====================================================================
# METHODOLOGY
# =====================================================================
st.markdown("## Methodology & Econometric Approach")
st.markdown(
    """
<div class="methodology">
<h4>Model: Binary Logistic Regression</h4>
<p>The core model is a <strong>logistic regression</strong> estimated via maximum likelihood (MLE):</p>
<p style="text-align:center;font-style:italic;">P(Y=1 | X) = 1 / (1 + exp(-Xβ))</p>
<p>where Y=1 indicates loan default and X includes borrower characteristics (age, income, DTI, past defaults, employment status, housing, loan purpose) and macroeconomic conditions.</p>

<h4>Inference & Diagnostics</h4>
<ul>
<li><strong>Statistical significance:</strong> Coefficients tested with z-tests; p-values, standard errors reported.</li>
<li><strong>Model fit:</strong> McFadden Pseudo R, log-likelihood, AIC, BIC for model selection.</li>
<li><strong>Average Marginal Effects (AME):</strong> Change in default probability per unit change in each predictor.</li>
<li><strong>ROC-AUC:</strong> Measures discriminatory power (0.5 = random, 1.0 = perfect).</li>
</ul>

<h4>Why logistic regression over ML?</h4>
<p>In credit risk, regulators (BaFin, EBA) require <strong>interpretable, transparent models</strong>. A Random Forest is also included as a non-linear benchmark to demonstrate ML capability.</p>

<h4>Data</h4>
<p>Applicant data calibrated to the German credit market (Schufa-style). Macroeconomic context from <strong>ECB Statistical Data Warehouse</strong> and Bundesbank public APIs. Fully reproducible.</p>
</div>
""",
    unsafe_allow_html=True,
)

# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("## About")
    st.markdown(
        "Built with Python and econometric methods for a Master's in Econometrics "
        "project at a German university."
    )
    st.markdown("**Contact**")
    st.markdown("[GitHub Profile](https://github.com/Baspehlivan)")
    st.markdown("---")
    st.markdown(f"**Dataset stats**")
    st.markdown(f"- Applicants: {len(df):,}")
    st.markdown(f"- Default rate: {df['default'].mean():.1%}")
    st.markdown(f"- Features: {len(df.columns)}")
    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.8rem;color:#999;">'
        "Disclaimer: This is a demonstration project. "
        "The model is not validated for actual lending decisions."
        "</p>",
        unsafe_allow_html=True,
    )
