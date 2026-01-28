#!/usr/bin/env python3
"""Diagnose Whisper.cpp Server HTTP API issues."""

import requests
import json
from pathlib import Path

BASE_URL = "http://127.0.0.1:18181"

def test_health():
    """Test if server is running."""
    print("=" * 60)
    print("1. Testing server availability...")
    print("=" * 60)
    
    try:
        # Try root endpoint
        r = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"GET / - Status: {r.status_code}")
        print(f"Response: {r.text[:500]}")
    except Exception as e:
        print(f"ERROR: {e}")
    
    print()

def test_endpoints():
    """Test available endpoints."""
    print("=" * 60)
    print("2. Testing available endpoints...")
    print("=" * 60)
    
    endpoints = [
        "/",
        "/v1/audio/transcriptions",
        "/inference",
        "/load",
    ]
    
    for endpoint in endpoints:
        try:
            r = requests.get(f"{BASE_URL}{endpoint}", timeout=5)
            print(f"GET {endpoint} - Status: {r.status_code}")
        except Exception as e:
            print(f"GET {endpoint} - ERROR: {e}")
    
    print()

def test_transcription_simple():
    """Test transcription with minimal parameters."""
    print("=" * 60)
    print("3. Testing /inference endpoint...")
    print("=" * 60)
    
    video_path = Path(r'C:\Users\Admin\Documents\你被Clawdbot刷屏了吗,FOMO了吗_.你被Clwadbot刷屏了吗,FOMO了吗_.35625174287.mp4')
    
    # Test with minimal params
    print(f"Testing with file: {video_path.name}")
    print(f"File exists: {video_path.exists()}")
    print(f"File size: {video_path.stat().st_size / 1024 / 1024:.1f} MB")
    print()
    
    # Test /inference
    print("Testing POST /inference...")
    with open(video_path, "rb") as f:
        files = {"file": (video_path.name, f, "video/mp4")}
        data = {"response_format": "json"}
        
        try:
            r = requests.post(f"{BASE_URL}/inference", files=files, data=data, timeout=120)
            print(f"Status: {r.status_code}")
            print(f"Response: {json.dumps(r.json(), indent=2, ensure_ascii=False)[:1000]}")
        except Exception as e:
            print(f"ERROR: {e}")
    
    print()

def test_transcription_verbose():
    """Test transcription with verbose_json format."""
    print("=" * 60)
    print("4. Testing with verbose_json format...")
    print("=" * 60)
    
    video_path = Path(r'C:\Users\Admin\Documents\你被Clawdbot刷屏了吗,FOMO了吗_.你被Clwadbot刷屏了吗,FOMO了吗_.35625174287.mp4')
    
    with open(video_path, "rb") as f:
        files = {"file": (video_path.name, f, "video/mp4")}
        data = {"response_format": "verbose_json"}
        
        try:
            r = requests.post(f"{BASE_URL}/inference", files=files, data=data, timeout=120)
            print(f"Status: {r.status_code}")
            result = r.json()
            print(f"Keys in response: {list(result.keys())}")
            if "error" in result:
                print(f"ERROR: {result['error']}")
            else:
                print(f"Response preview: {json.dumps(result, indent=2, ensure_ascii=False)[:1000]}")
        except Exception as e:
            print(f"ERROR: {e}")
    
    print()

def test_load_model():
    """Test /load endpoint."""
    print("=" * 60)
    print("5. Testing /load endpoint...")
    print("=" * 60)
    
    # Try to check model status
    try:
        r = requests.get(f"{BASE_URL}/load", timeout=5)
        print(f"GET /load - Status: {r.status_code}")
        print(f"Response: {r.text[:500]}")
    except Exception as e:
        print(f"ERROR: {e}")
    
    print()

if __name__ == "__main__":
    test_health()
    test_endpoints()
    test_transcription_simple()
    test_transcription_verbose()
    test_load_model()
    
    print("=" * 60)
    print("Diagnosis complete")
    print("=" * 60)
