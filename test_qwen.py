#!/usr/bin/env python3
import time
from openai import OpenAI

client = OpenAI(api_key="dummy", base_url="http://127.0.0.1:1234/v1")

print("Testing Qwen3-VL-30B...")
try:
    start = time.time()
    resp = client.chat.completions.create(
        model="Qwen3-VL-30B-A3B-Instruct-XH-IQ4_NL_16.gguf",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=20,
        timeout=60
    )
    print(f"OK! Time: {time.time()-start:.1f}s")
    print(f"Response: {resp.choices[0].message.content}")
except Exception as e:
    print(f"Failed: {e}")
