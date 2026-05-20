#!/usr/bin/env python3
"""Check HF Space build status."""
import json
import subprocess
import time

result = subprocess.run(
    ["hf", "spaces", "info", "wiebuch/credit-risk-dashboard"],
    capture_output=True,
    text=True,
    timeout=30,
)
d = json.loads(result.stdout)
r = d.get("runtime", {})
stage = r.get("stage", "unknown")
print(f"Stage: {stage}")
print(f"URL: https://wiebuch-credit-risk-dashboard.hf.space")

if stage in ("BUILD_ERROR", "RUNNING", "APP_STARTING"):
    if "errorMessage" in r.get("raw", {}):
        print(f"Error: {r['raw']['errorMessage']}")

if stage in ("BUILDING", "QUEUED", "APP_STARTING"):
    print("Build in progress, waiting...")
    for i in range(15):
        time.sleep(30)
        result = subprocess.run(
            ["hf", "spaces", "info", "wiebuch/credit-risk-dashboard"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        d = json.loads(result.stdout)
        r = d.get("runtime", {})
        stage = r.get("stage", "unknown")
        print(f"Poll {i+1}/15: {stage}")
        if stage in ("RUNNING", "BUILD_ERROR", "SLEEPING", "STOPPED"):
            break

print(f"\nFinal: {stage}")
print(f"Open: https://wiebuch-credit-risk-dashboard.hf.space")
