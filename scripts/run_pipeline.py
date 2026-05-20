#!/usr/bin/env python3
"""End-to-end pipeline: fetch data -> train models -> evaluate -> launch info.

Usage:
    python scripts/run_pipeline.py            # full pipeline
    python scripts/run_pipeline.py --quick    # skip model training if saved
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from data.fetch_german_data import build_dataset
from model.credit_scoring_model import MODELS_DIR, PROCESSED_DIR, run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000, help="Number of applicants")
    parser.add_argument(
        "--quick", action="store_true", help="Skip training if models exist"
    )
    parser.add_argument(
        "--force-refetch", action="store_true", help="Re-fetch ECB data"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Credit Risk Dashboard — Pipeline")
    print("=" * 60)

    # Step 1: Build dataset
    data_path = PROCESSED_DIR / "credit_applicants.parquet"
    if not data_path.exists() or args.force_refetch:
        print("\n[1/3] Building dataset...")
        build_dataset(n_applicants=args.n, force_refetch=args.force_refetch)
    else:
        print(f"\n[1/3] Dataset exists at {data_path} (use --force-refetch to rebuild)")

    # Step 2: Train models (skip if quick + models exist)
    logit_path = MODELS_DIR / "logistic_model.joblib"
    if args.quick and logit_path.exists():
        print(f"\n[2/3] Models exist at {MODELS_DIR} (use --quick to skip)")
    else:
        print("\n[2/3] Training models...")
        run_pipeline(n_applicants=args.n)

    # Step 3: Verify and launch info
    print("\n[3/3] Verification...")
    print(f"  Dataset:   {data_path}")
    print(f"  Logit:     {MODELS_DIR / 'logistic_model.joblib'}")
    print(f"  RF:        {MODELS_DIR / 'random_forest.joblib'}")
    print(f"  Dashboard: dashboard/app.py")

    print("\n" + "=" * 60)
    print("Ready! Launch the dashboard with:")
    print("  streamlit run dashboard/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
