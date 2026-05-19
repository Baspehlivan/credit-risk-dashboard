"""Module to fetch real German macroeconomic data from public APIs.

Data Sources:
  - ECB Statistical Data Warehouse (SDW): Interest rates, GDP, CPI, credit aggregates
  - World Bank API (via pandas-datareader): GDP, unemployment, inflation
  - Bundesbank: Supplementary credit data

All data is cached locally after first fetch so the dashboard runs offline.
"""

import os
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parent / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent / "processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: ECB SDW REST API
# ---------------------------------------------------------------------------
ECB_BASE = "https://sdw-wsrest.ecb.europa.eu/service"

# Key ECB dataflow keys and their human-readable names
ECB_SERIES = {
    "GDP": {
        "key": "MNA/Q.Y.DE.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.V.N",
        "label": "GDP (Germany, quarterly, EUR millions, chain-linked)",
        "freq": "Q",
    },
    "HICP": {
        "key": "ICP/M.DE.N.000000.4.ANR",
        "label": "HICP - All items (Germany, monthly, annual % change)",
        "freq": "M",
    },
    "UNEMP": {
        "key": "STS/A.DE.N.UNEH.TTTT.4.000",
        "label": "Unemployment rate (Germany, monthly, %)",
        "freq": "M",
    },
    "SHORT_RATE": {
        "key": "FM/M.U2.EUR.3M.MM.EUR4FIX_B._Z._Z",
        "label": "3-month EURIBOR (Eurozone, monthly, %)",
        "freq": "M",
    },
    "LONG_RATE": {
        "key": "FM/M.U2.EUR.GB.GF10YFRTB._Z._Z",
        "label": "10-year Govt Bond Yield (Germany, monthly, %)",
        "freq": "M",
    },
    "LOANS_HH": {
        "key": "BSI/M.DE.N.A.A20.A.I.5F.U2.Z._Z.EUR.E",
        "label": "Loans to households (Germany, monthly, EUR millions)",
        "freq": "M",
    },
    "LOANS_NFC": {
        "key": "BSI/M.DE.N.A.A22.A.I.5F.U2.Z._Z.EUR.E",
        "label": "Loans to non-financial corps (Germany, monthly, EUR millions)",
        "freq": "M",
    },
}


def fetch_ecb_series(series_key: str) -> pd.Series:
    """Fetch a time series from ECB SDW in CSV format and return as pd.Series."""
    url = f"{ECB_BASE}/data/{series_key}/?format=csvdata"
    try:
        resp = requests.get(url, timeout=20, headers={"Accept": "text/csv"})
        resp.raise_for_status()
    except Exception as e:
        print(f"  [WARN] ECB fetch failed for {series_key}: {e}")
        return pd.Series(dtype=float)

    # ECB CSV: columns include TIME_PERIOD, OBS_VALUE
    lines = resp.text.splitlines()
    if len(lines) < 2:
        return pd.Series(dtype=float)

    import csv

    reader = csv.DictReader(lines)
    dates, values = [], []
    for row in reader:
        tp = row.get("TIME_PERIOD", "")
        ov = row.get("OBS_VALUE", "")
        if tp and ov:
            try:
                dates.append(pd.Timestamp(tp))
                values.append(float(ov))
            except (ValueError, TypeError):
                continue

    if not dates:
        return pd.Series(dtype=float)

    s = pd.Series(values, index=pd.DatetimeIndex(dates))
    s = s.sort_index()
    # Remove duplicates by keeping last
    s = s[~s.index.duplicated(keep="last")]
    return s


def fetch_all_ecb(cache: bool = True) -> dict[str, pd.Series]:
    """Fetch all configured ECB series, optionally caching to disk."""
    cache_path = DATA_DIR / "ecb_series.json"
    # Cache exists and requested
    if cache and cache_path.exists():
        with open(cache_path) as f:
            raw = json.load(f)
        result = {}
        for name, data in raw.items():
            idx = pd.DatetimeIndex([pd.Timestamp(d) for d in data["index"]])
            result[name] = pd.Series(data["values"], index=idx)
        print(f"  [OK]  Loaded {len(result)} series from cache")
        return result

    print("  [FETCH] Fetching ECB data (this may take a moment)...")
    result = {}
    for name, spec in ECB_SERIES.items():
        print(f"    -> {spec['label']}...")
        s = fetch_ecb_series(spec["key"])
        if len(s) > 0:
            result[name] = s
            print(
                f"         Got {len(s)} observations ({s.index[0].date()} to {s.index[-1].date()})"
            )
        else:
            print(f"         No data returned")

    # Cache as JSON (index as strings)
    if cache and result:
        raw = {
            name: {
                "index": [str(d.date()) for d in s.index],
                "values": s.values.tolist(),
            }
            for name, s in result.items()
        }
        with open(cache_path, "w") as f:
            json.dump(raw, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# Synthetic credit applicant dataset (informed by macro conditions)
# ---------------------------------------------------------------------------
def _sample_macro_quarter(
    macro: dict[str, pd.Series], target_date: pd.Timestamp
) -> dict:
    """Get macro conditions closest to a given date."""
    out = {}
    for name, series in macro.items():
        if len(series) == 0:
            out[name] = np.nan
            continue
        # Find closest date at or before target
        mask = series.index <= target_date
        if mask.any():
            out[name] = series[mask].iloc[-1]
        else:
            out[name] = series.iloc[0]
    return out


def generate_credit_applicants(
    n_applicants: int = 5000,
    macro: dict[str, pd.Series] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a realistic synthetic dataset of German credit applicants.

    Features are designed to mimic real credit bureau data (SCHUFA-like).
    Default probability depends on applicant characteristics AND macro conditions.
    """
    rng = np.random.default_rng(seed)

    # --- Time range: last 6 years of quarterly data ---
    end_date = pd.Timestamp("2024-12-31")
    start_date = pd.Timestamp("2019-01-01")
    all_quarters = pd.date_range(start=start_date, end=end_date, freq="QE")
    if len(all_quarters) == 0:
        all_quarters = [end_date]

    # --- Assign each applicant a random quarter ---
    quarter_idx = rng.integers(0, len(all_quarters), size=n_applicants)
    application_dates = all_quarters[quarter_idx]

    records = []
    for i in range(n_applicants):
        app_date = application_dates[i]
        m = _sample_macro_quarter(macro, app_date) if macro else {}

        # --- Applicant features ---
        age = int(rng.normal(35, 10))
        age = max(18, min(75, age))

        # Employment status: 0=employed, 1=self-employed, 2=student, 3=unemployed
        emp_probs = [0.55, 0.12, 0.18, 0.15]
        employment_status = rng.choice([0, 1, 2, 3], p=emp_probs)

        # Monthly income (euro) — varies by employment
        if employment_status == 0:  # employed
            income = max(1200, rng.lognormal(8.3, 0.4))
        elif employment_status == 1:  # self-employed
            income = max(1000, rng.lognormal(8.4, 0.6))
        elif employment_status == 2:  # student
            income = max(400, rng.lognormal(6.5, 0.3))
        else:  # unemployed
            income = max(500, rng.lognormal(6.8, 0.2))

        income = round(income, 0)

        # Existing debt-to-income ratio
        dti = min(
            1.0, max(0.0, rng.beta(2, 5) + (0.1 if employment_status == 3 else 0))
        )

        # Credit history length (years)
        credit_history_years = max(0, age - rng.integers(18, 25))
        credit_history_years = min(credit_history_years, 40)

        # Number of past defaults
        past_defaults = rng.poisson(0.2 * (1 + dti))
        past_defaults = min(past_defaults, 5)

        # Marital status: 0=single, 1=married, 2=divorced
        marital_status = rng.choice([0, 1, 2], p=[0.35, 0.50, 0.15])

        # Home ownership: 0=rent, 1=own, 2=with parents
        home_ownership = rng.choice([0, 1, 2], p=[0.55, 0.35, 0.10])

        # Number of dependents
        dependents = rng.poisson(0.4 if marital_status == 1 else 0.1)
        dependents = min(dependents, 6)

        # Loan purpose: 0=car, 1=education, 2=home, 3=personal, 4=business
        loan_purpose = rng.choice([0, 1, 2, 3, 4], p=[0.20, 0.10, 0.30, 0.25, 0.15])

        # Loan amount (euro) — varies by purpose
        if loan_purpose == 0:  # car
            loan_amount = rng.lognormal(9.5, 0.5)
        elif loan_purpose == 1:  # education
            loan_amount = rng.lognormal(8.8, 0.4)
        elif loan_purpose == 2:  # home
            loan_amount = rng.lognormal(11.5, 0.6)
        elif loan_purpose == 3:  # personal
            loan_amount = rng.lognormal(9.0, 0.5)
        else:  # business
            loan_amount = rng.lognormal(10.0, 0.7)
        loan_amount = round(min(loan_amount, 500_000), 0)

        # Loan term (months)
        loan_term = int(rng.choice([12, 24, 36, 48, 60, 72, 84, 120, 180, 240, 360]))
        if loan_purpose == 0:  # car: shorter
            loan_term = min(loan_term, 84)
        elif loan_purpose == 2:  # home: longer
            loan_term = max(loan_term, 120)

        # --- Default probability (logistic function of features + macro) ---
        logit = -2.0  # base intercept

        # Income effect (higher income => lower default risk)
        logit += -0.4 * (np.log(max(income, 1)) - 7.0)

        # DTI effect
        logit += 1.8 * dti

        # Past defaults effect
        logit += 0.7 * past_defaults

        # Employment effect
        if employment_status == 3:  # unemployed
            logit += 1.2
        elif employment_status == 2:  # student
            logit += 0.3

        # Home ownership effect
        if home_ownership == 1:  # own
            logit -= 0.5

        # Credit history length effect
        logit += -0.03 * credit_history_years

        # Loan amount / income ratio
        logit += 0.3 * min(loan_amount / max(income, 1), 5)

        # Macro effects (if available)
        if m:  # macro data exists
            unemp = m.get("UNEMP")
            if unemp is not None and not (isinstance(unemp, float) and np.isnan(unemp)):
                logit += 0.15 * (unemp - 5.0)

            gdp = m.get("GDP")
            if gdp is not None and not (isinstance(gdp, float) and np.isnan(gdp)):
                # GDP growth proxy: positive GDP => lower risk
                logit += -0.02 * (gdp / 1e6 - 800) / 100

            short_rate = m.get("SHORT_RATE")
            if short_rate is not None and not (
                isinstance(short_rate, float) and np.isnan(short_rate)
            ):
                logit += 0.08 * (short_rate - 1.0)

        # Convert to probability
        prob_default = 1.0 / (1.0 + np.exp(-logit))
        # Clamp
        prob_default = min(0.95, max(0.01, prob_default))

        # Simulate actual default
        default = int(rng.random() < prob_default)

        # ---- Add some consistency noise ----
        # Record
        records.append(
            {
                "applicant_id": i + 1,
                "application_date": app_date,
                "age": age,
                "employment_status": employment_status,
                "monthly_income": income,
                "dti_ratio": round(dti, 4),
                "credit_history_years": credit_history_years,
                "past_defaults": past_defaults,
                "marital_status": marital_status,
                "home_ownership": home_ownership,
                "dependents": dependents,
                "loan_purpose": loan_purpose,
                "loan_amount": loan_amount,
                "loan_term_months": loan_term,
                "prob_default": round(prob_default, 4),
                "default": default,
                # Macro context for the quarter
                "_macro_unemp": (
                    round(m.get("UNEMP", np.nan), 2)
                    if not np.isnan(m.get("UNEMP", np.nan))
                    else np.nan
                ),
                "_macro_gdp": (
                    round(m.get("GDP", np.nan), 0)
                    if not np.isnan(m.get("GDP", np.nan))
                    else np.nan
                ),
                "_macro_short_rate": (
                    round(m.get("SHORT_RATE", np.nan), 2)
                    if not np.isnan(m.get("SHORT_RATE", np.nan))
                    else np.nan
                ),
            }
        )

    df = pd.DataFrame(records)
    print(f"  [OK]  Generated {len(df)} credit applicants")
    print(f"       Default rate: {df['default'].mean():.2%}")
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def build_dataset(
    n_applicants: int = 5000,
    force_refetch: bool = False,
    cache_ecb: bool = True,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Fetch macro data and build the credit applicant dataset.

    Returns a DataFrame with all features and the target variable ``default``.
    """
    print("=" * 60)
    print("Credit Risk Dataset Builder")
    print("=" * 60)

    # Step 1: Fetch macro data
    print("\n[1/3] Fetching German macroeconomic data...")
    macro = fetch_all_ecb(cache=not force_refetch)
    if macro:
        print(f"  Got {len(macro)} series")
    else:
        print("  No macro data fetched — using generic defaults")

    # Step 2: Generate applicant data
    print("\n[2/3] Generating synthetic credit applicant dataset...")
    df = generate_credit_applicants(n_applicants=n_applicants, macro=macro)

    # Step 3: Save
    print("\n[3/3] Saving dataset...")
    if output_path is None:
        output_path = PROCESSED_DIR / "credit_applicants.parquet"

    df.to_parquet(output_path, index=False)
    print(f"  Saved to {output_path}")

    # Also save macro for dashboard
    if macro:
        macro_df = pd.DataFrame(macro)
        macro_path = PROCESSED_DIR / "macro_data.parquet"
        macro_df.to_parquet(macro_path)
        print(f"  Macro data saved to {macro_path}")

    print("\nDone!")
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000, help="Number of applicants")
    parser.add_argument(
        "--force-refetch", action="store_true", help="Re-fetch ECB data"
    )
    args = parser.parse_args()

    df = build_dataset(n_applicants=args.n, force_refetch=args.force_refetch)
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
