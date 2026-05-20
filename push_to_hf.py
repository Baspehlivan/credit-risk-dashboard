#!/usr/bin/env python3
"""Push repo to HF Space with proper exclude patterns."""
import subprocess
import sys

result = subprocess.run(
    [
        "hf",
        "upload",
        "wiebuch/credit-risk-dashboard",
        ".",
        ".",
        "--repo-type",
        "space",
        "--exclude",
        ".venv/**",
        "--exclude",
        "__pycache__/**",
        "--exclude",
        ".pytest_cache/**",
        "--exclude",
        "*.pyc",
        "--exclude",
        "logs/**",
        "--exclude",
        "create_space.py",
        "--exclude",
        "push_to_hf.py",
        "--commit-message",
        "Deploy credit risk dashboard with Docker SDK",
        "--commit-description",
        "Self-contained Streamlit app with pre-trained models, data files, and Dockerfile for HF Spaces.",
    ],
    capture_output=True,
    text=True,
    timeout=180,
)

out = result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout
print("STDOUT:", out)
if result.stderr:
    err = result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr
    print("STDERR:", err, file=sys.stderr)
print(f"Exit code: {result.returncode}")
