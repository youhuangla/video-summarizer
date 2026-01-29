#!/usr/bin/env python3
"""测试 VLM 是否正常工作"""
import time
from openai import OpenAI

client = OpenAI(
    api_key="dummy",
    base_url="http://127.0.0.1:1234/v1"
)

print("测试 VLM 服务...")
print("发送简单文本请求（无图像）...")
start = time.time()

try:
    response = client.chat.completions.create(
        model="GLM46V-Flash",
        messages=[
            {"role": "user", "content": "Hello, are you working?"}
        ],
        max_tokens=50,
        timeout=60
    )
    elapsed = time.time() - start
    print(f"✅ 成功! 耗时: {elapsed:.1f}s")
    print(f"回复: {response.choices[0].message.content[:100]}...")
except Exception as e:
    print(f"❌ 失败: {e}")
