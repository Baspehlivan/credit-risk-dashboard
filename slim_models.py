#!/usr/bin/env python3
"""Re-save logit_dict without statsmodels objects — pure Python/numpy only."""
import joblib
import numpy as np

old = joblib.load("dashboard/models/logit_dict.joblib")

print("Keys in logit_dict:")
for k, v in old.items():
    print(f"  {k}: {type(v).__name__}")

# Build a new dict with only the things the dashboard uses
# Look for: model coefficients, pvalues, predictions, auc, etc.
new = {}
for k, v in old.items():
    if hasattr(v, "__class__"):
        cls_name = type(v).__name__
        # Skip full statsmodels result objects
        if (
            "LogitResults" in cls_name
            or "DiscreteResults" in cls_name
            or "BinaryResults" in cls_name
        ):
            print(f"  SKIPPING {k} ({cls_name}) — statsmodels object")
            continue
    new[k] = v

joblib.dump(new, "dashboard/models/logit_dict_slim.joblib")
print(f"\nSaved slim version: {len(new)} keys")

# Verify it loads without statsmodels
import subprocess

result = subprocess.run(
    [
        "python3",
        "-c",
        "import sys; [sys.modules.__setitem__(m,None) for m in ['statsmodels','statsmodels.api','statsmodels.discrete','statsmodels.discrete.discrete_model']]; import joblib; d=joblib.load('dashboard/models/logit_dict_slim.joblib'); print('Loaded OK:', len(d), 'keys')",
    ],
    capture_output=True,
    text=True,
    timeout=10,
)
print(result.stdout.strip())
if result.stderr:
    print("ERR:", result.stderr[:500])
