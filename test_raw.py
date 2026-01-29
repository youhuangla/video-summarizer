#!/usr/bin/env python3
"""Test raw HTTP API"""
import requests
import json
import time

URL = "http://127.0.0.1:1234/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer 123"
}
DATA = {
    "model": "Qwen3-VL-30B-A3B-Instruct-XH-IQ4_NL_16.gguf",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 20
}

print(f"POST {URL}")
print(f"Headers: {HEADERS}")
print(f"Data: {json.dumps(DATA, indent=2)}")
print("\nWaiting for response...")

start = time.time()
try:
    resp = requests.post(URL, headers=HEADERS, json=DATA, timeout=120)
    print(f"\nStatus: {resp.status_code}")
    print(f"Time: {time.time()-start:.1f}s")
    print(f"Response: {resp.text[:500]}")
except Exception as e:
    print(f"Error: {e}")
