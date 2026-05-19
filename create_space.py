#!/usr/bin/env python3
"""Create HF Space for Baspehlivan namespace."""
import urllib.request, urllib.error
import json
import os

token = os.environ.get("HF_TOKEN", "")
url = "https://huggingface.co/api/repos/create"
payload = json.dumps(
    {
        "name": "credit-risk-dashboard",
        "type": "space",
        "sdk": "docker",
        "hardware": "cpu-basic",
    }
).encode()

req = urllib.request.Request(url, data=payload, method="POST")
req.add_header("Authorization", f"Bearer {token}")
req.add_header("Content-Type", "application/json")

try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
        print(f"Created: https://huggingface.co/spaces/{result.get('id', 'unknown')}")
        print(f"Full response: {json.dumps(result, indent=2)[:500]}")
except urllib.error.HTTPError as e:
    body = e.read().decode()
    print(f"HTTP {e.code}: {body}")
