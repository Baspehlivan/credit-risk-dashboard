#!/usr/bin/env python3
"""Verify the HF Space app is accessible."""
import urllib.request

# First, let's check the actual domain
domains = [
    "https://wiebuch-credit-risk-dashboard.hf.space",
]

for url in domains:
    try:
        req = urllib.request.Request(url, method="GET")
        req.add_header("User-Agent", "Mozilla/5.0")
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
            print(f"URL: {url}")
            print(f"Status: {resp.status}")
            print(f"Content length: {len(body)}")
            # Streamlit returns HTML - check for key strings
            if "credit" in body.lower() or "risk" in body.lower():
                print("SUCCESS: Dashboard is live!")
            else:
                print("App responded but content might not be dashboard")
    except Exception as e:
        print(f"URL: {url} -> Error: {type(e).__name__}: {e}")
