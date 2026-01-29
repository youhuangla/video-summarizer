#!/usr/bin/env python3
"""Debug API connection"""
import openai
import json

API_KEY = "123"  # 尝试密码
BASE_URL = "http://127.0.0.1:1234/v1"

print(f"Testing API at {BASE_URL}")
print(f"API Key: {API_KEY}")

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 1. 测试模型列表
print("\n1. Testing /v1/models...")
try:
    models = client.models.list()
    model_id = models.data[0].id if models.data else None
    print(f"   Models: {[m.id for m in models.data]}")
    print(f"   Using model: {model_id}")
except Exception as e:
    print(f"   Error: {e}")
    model_id = None

# 2. 测试简单文本请求
if model_id:
    print(f"\n2. Testing chat.completions with model: {model_id}")
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=20
        )
        print(f"   Success! Response: {resp.choices[0].message.content}")
    except Exception as e:
        print(f"   Error: {type(e).__name__}: {e}")

# 3. 检查请求格式
print("\n3. Request format check:")
print(f"   Content-Type: application/json")
print(f"   Authorization: Bearer {API_KEY[:10]}...")
