#!/usr/bin/env python3
"""Test NovaAI Whisper API."""

import requests
from pathlib import Path
import json

BASE_URL = "http://127.0.0.1:8281/v1"
API_KEY = "novaai"

def test_api():
    """Test Whisper API connection."""
    print("=" * 60)
    print("Testing NovaAI Whisper API")
    print("=" * 60)
    print(f"Base URL: {BASE_URL}")
    print()
    
    # Test with video file
    video_path = Path(r'C:\Users\Admin\Documents\你被Clawdbot刷屏了吗,FOMO了吗_.你被Clwadbot刷屏了吗,FOMO了吗_.35625174287.mp4')
    
    print(f"File: {video_path.name}")
    print(f"Size: {video_path.stat().st_size / 1024 / 1024:.1f} MB")
    print()
    
    url = f"{BASE_URL}/audio/transcriptions"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    
    print("Sending request...")
    with open(video_path, "rb") as f:
        files = {"file": (video_path.name, f, "video/mp4")}
        data = {
            "model": "whisper-1",
            "language": "zh",
            "response_format": "verbose_json"
        }
        
        try:
            response = requests.post(url, headers=headers, files=files, data=data, timeout=120)
            print(f"Status Code: {response.status_code}")
            print()
            
            if response.status_code == 200:
                result = response.json()
                print(f"Response Keys: {list(result.keys())}")
                
                if "segments" in result:
                    print(f"Segments Count: {len(result['segments'])}")
                    if result['segments']:
                        print("\nFirst 3 segments:")
                        for seg in result['segments'][:3]:
                            print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text'][:50]}...")
                elif "text" in result:
                    print(f"Text: {result['text'][:200]}...")
                else:
                    print(f"Full response: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}")
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Exception: {e}")

if __name__ == "__main__":
    test_api()
