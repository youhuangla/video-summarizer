#!/usr/bin/env python3
"""Debug Whisper transcription."""

import requests
from pathlib import Path
import json

# Test first video
video_path = Path(r'C:\Users\Admin\Documents\你被Clawdbot刷屏了吗,FOMO了吗_.你被Clwadbot刷屏了吗,FOMO了吗_.35625174287.mp4')
url = 'http://127.0.0.1:18181/v1/audio/transcriptions'

print('Testing Whisper with first video...')
print(f'File: {video_path}')
print(f'File size: {video_path.stat().st_size / 1024 / 1024:.1f} MB')
print()

with open(video_path, 'rb') as f:
    files = {'file': (video_path.name, f, 'video/mp4')}
    data = {'model': 'base', 'language': 'zh', 'response_format': 'verbose_json'}
    print('Sending request to Whisper...')
    r = requests.post(url, files=files, data=data, timeout=120)
    result = r.json()

print(f'Status Code: {r.status_code}')
print(f'Response Keys: {list(result.keys())}')
print()

if 'segments' in result:
    print(f'Segments Count: {len(result["segments"])}')
    if result['segments']:
        print('\nFirst 3 segments:')
        for seg in result['segments'][:3]:
            print(f'  [{seg["start"]:.1f}s - {seg["end"]:.1f}s] {seg["text"][:50]}...')
    else:
        print('WARNING: segments array is empty!')
else:
    print('WARNING: no segments in response!')
    print('Full response:')
    print(json.dumps(result, indent=2, ensure_ascii=False))

# Check if there's text
if 'text' in result:
    print(f'\nFull text preview: {result["text"][:200]}...')
