#!/usr/bin/env python3
import requests

resp = requests.post(
    'http://127.0.0.1:1234/v1/chat/completions',
    headers={'Authorization': 'Bearer 123'},
    json={
        'model': 'Qwen3-VL-30B-A3B-Instruct-XH-IQ4_NL_16.gguf',
        'messages': [{'role': 'user', 'content': 'Hello'}],
        'max_tokens': 20
    },
    timeout=60
)
print(f'Status: {resp.status_code}')
data = resp.json()
content = data['choices'][0]['message']['content']
print(f'Response: {content}')
